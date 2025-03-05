import copy
import wandb
import random
import time
import numpy as np
import torch
from collections import defaultdict

import torch.multiprocessing as mp
from config import device
from predict import validate
from data_utils.dataloader import get_client_dataloader
from train import execute_epoch

class Base:
    def __init__(self, global_model, args, client_groups=[]):
        """Initialize Federator with global model and arguments."""
        self.global_model = global_model
        self.num_rounds = args.num_rounds
        self.num_clients = args.num_clients
        self.sample_rate = args.sample_rate
        self.alpha = args.alpha
        self.horizontal_scale_ratios = args.horizontal_scale_ratios
        self.client_split_ratios = args.client_split_ratios
        self.num_levels = len(self.client_split_ratios)
        self.client_groups = client_groups
        self.use_gpu = args.use_gpu
        self.record_table = defaultdict(int)


    def train(self, train_set, test_loader, user_group, criterion, args, batch_size):
        """Perform federated training over multiple rounds."""
        best_acc = [0, 0, 0] # 0

        # Assign clients to level groups based on split ratios. (computing)
        if not self.client_groups:
            client_idxs = np.arange(self.num_clients)
            np.random.seed(args.seed)
            shuffled_clients = np.random.permutation(client_idxs)
            s = 0
            for ratio in self.client_split_ratios:
                e = s + int(len(shuffled_clients) * ratio)
                self.client_groups.append(shuffled_clients[s:e])
                s = e

        for round_idx in range(self.num_rounds):
            print(f'\n | Global Training Round : {round_idx} |\n')
            train_loss, test_loss, test_acc = self.execute_round(train_set, test_loader, user_group, criterion, args, batch_size, round_idx)

            if args.client_split_ratios[0] == 1:
                if best_acc[0] < test_acc[0].avg:
                    best_acc[0] = test_acc[0].avg
            elif args.client_split_ratios[1] == 1:
                if best_acc[1] < test_acc[1].avg:
                    for i in range(2):
                        best_acc[i] = test_acc[i].avg
            else:
                if best_acc[2] < test_acc[2].avg:
                    for i in range(3):
                        best_acc[i] = test_acc[i].avg

            wandb.log({
                f"train_loss": train_loss,
                f"test_acc_ee2": test_acc[2].avg,
                f"test_acc_ee1": test_acc[1].avg,
                f"test_acc_ee0": test_acc[0].avg,
                f"test_loss": test_loss.avg,
                f"best_acc_ee2": best_acc[2],
                f"best_acc_ee1": best_acc[1],
                f"best_acc_ee0": best_acc[0],
            }, commit=True)

        return best_acc

    def execute_round(self, train_set, test_loader, user_group, criterion, args, batch_size, round_idx):
        """Execute a single round of federated training."""
        self.global_model.train()

        # sample client
        m = max(int(self.sample_rate * self.num_clients), 1)
        selected_clients = np.random.choice(range(self.num_clients), m, replace=False)
        
        # dataloader level h_level
        client_train_loaders = [get_client_dataloader(train_set, user_group[client_idxs], args, batch_size) for client_idxs in selected_clients]
        levels = [self.get_level(idx) for idx in selected_clients] # 0 1 2
        h_scale_ratios = [self.horizontal_scale_ratios[level] for level in levels] # 1 2 3
        print(f"Client = {selected_clients}")
        print(f"Initial levels: {levels}")
        print(f"h_scale_ratios: {h_scale_ratios}")

        # local model setting
        local_models = [copy.deepcopy(self.global_model) for _ in range(len(selected_clients))]

        local_weights = []
        local_grad_flags = []
        local_losses = []

        # selected clients local training
        for i, client_idx in enumerate(selected_clients):
            result = self.execute_client_round(criterion, args, round_idx, local_models[i], client_train_loaders[i], h_scale_ratios[i], client_idx)
            local_weights.append(result[0])
            local_grad_flags.append(result[1])
            local_losses.append(result[2])
            print(f'Client {i+1}/{len(selected_clients)} completely finished')
            
        train_loss = sum(local_losses) / len(selected_clients)

        # Update the global model
        global_weights = self.average_weights(local_weights, local_grad_flags, self.global_model)
        self.global_model.load_state_dict(global_weights)
        # Validate
        test_loss, test_acc = validate(self.global_model, test_loader, criterion, args)
        return train_loss, test_loss, test_acc
    
    
    def execute_client_round(self, criterion, args, round_idx, local_model, client_train_loader, h_scale_ratio, client_idx):
        """Execute a single round of training on a client."""

        if args.use_gpu:
            local_model = local_model.to(device)

        optimizer = torch.optim.SGD(local_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        loss = 0.0
        start = time.time()
        print(f'Start epochs for Client {client_idx}')
        for epoch in range(args.num_epoch):
            print(f'Client{client_idx} - epoch {epoch+1}/5') 
            loss = execute_epoch(local_model, client_train_loader, criterion, optimizer, round_idx, args, h_scale_ratio)
        
        end = time.time()
        print(f"Time = {end-start}s")

        print(f'Finished epochs for Client {client_idx}')
        local_weights = {k: v.cpu() for k, v in local_model.state_dict(keep_vars=True).items()}
        local_grad_flags = {k: v.grad is not None for k, v in local_model.state_dict(keep_vars=True).items()}

        del local_model
        torch.cuda.empty_cache()

        return local_weights, local_grad_flags, loss
    

    def get_level(self, idx):
        for level, group in enumerate(self.client_groups):
            if idx in group:
                return level
        return -1
    

    def average_weights(self, local_weights, local_grad_flags, model):
        """聚合僅考慮深度縮放的模型權重。"""
        w_avg = copy.deepcopy(model.state_dict())
        
        for key in w_avg.keys():
            # # 如果是統計參數，直接選取第一個客戶端的值
            # if 'num_batches_tracked' in key or 'running' in key:
            #     w_avg[key] = local_weights[0][key]
            #     continue
            valid_weights = [w[key] for i, w in enumerate(local_weights) if local_grad_flags[i][key]]
            if valid_weights:
                w_avg[key] = torch.stack(valid_weights).mean(dim=0)
        return w_avg