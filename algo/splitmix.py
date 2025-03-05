import copy
import wandb
import random
import time
import numpy as np
import torch
from collections import defaultdict

import torch.multiprocessing as mp
from config import device
from predict import validate_splitmix
from data_utils.dataloader import get_client_dataloader
from train import execute_epoch_splitmix

class Splitmix:
    def __init__(self, global_model_pool, model_pool, args, client_groups=[]):
        """Initialize Federator with global model and arguments."""
        self.global_model_pool = global_model_pool
        self.model_pool = model_pool
        self.num_rounds = args.num_rounds
        self.num_clients = args.num_clients
        self.sample_rate = args.sample_rate
        self.alpha = args.alpha
        self.horizontal_scale_ratios = args.horizontal_scale_ratios
        self.client_split_ratios = args.client_split_ratios
        self.num_levels = len(self.client_split_ratios)
        self.client_groups = client_groups
        self.use_gpu = args.use_gpu
        # 記錄每個客戶端訓練的進度（模型索引）
        self.record_table = defaultdict(int)


    def train(self, train_set, test_loader, user_group, criterion, args, batch_size):
        """Perform federated training over multiple rounds."""
        best_acc = 0 # 0

        # Assign clients to level groups based on split ratios.
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
            # , test_acc0, test_acc1, test_acc2, test_acc3

            if best_acc < test_acc.avg:
                best_acc = test_acc.avg

            wandb.log({
                #f"output0": test_acc0.avg,
                #f"output1": test_acc1.avg,
                #f"output2": test_acc2.avg,
                #f"output3": test_acc3.avg,
                f"train_loss": train_loss,
                f"test_acc_ee2": test_acc.avg,
                f"test_loss": test_loss.avg,
                f"best_acc_ee2": best_acc,
            }, commit=True)

        return best_acc

    def execute_round(self, train_set, test_loader, user_group, criterion, args, batch_size, round_idx):
        """Execute a single round of federated training."""

        # sample client
        m = max(int(self.sample_rate * self.num_clients), 1)
        selected_clients = np.random.choice(range(self.num_clients), m, replace=False)
        
        # dataloader, level
        client_train_loaders = [get_client_dataloader(train_set, user_group[client_idx], args, batch_size) for client_idx in selected_clients]
        levels = [self.get_level(client_idx) for client_idx in selected_clients] # 0 1 2
        print(f"Client = {selected_clients}")
        print(f"Initial levels: {levels}")

        '''Algo'''
        # local model setting 這邊要根據 1.計算能力(level) 2.table紀錄算到哪裡接著算level個
        local_models = []
        for client_idx, level in zip(selected_clients, levels):
            # 根據記錄表格計算應分配的模型
            start_idx = self.record_table[client_idx]
            models_to_train = []
            for i in range(level):
                model_idx = (start_idx + i) % len(self.model_pool)  # 循環選擇模型
                models_to_train.append(copy.deepcopy(self.model_pool[model_idx]))
            local_models.append(models_to_train)
            # update record_table
            self.record_table[client_idx] = (start_idx + level) % len(self.model_pool)

        print(f"更新後的記錄表: {dict(sorted(self.record_table.items()))}")


        local_weights = [[] for _ in range(4)]
        local_losses = []

        # selected clients local training
        for i, client_idx in enumerate(selected_clients):
            result = self.execute_client_round(criterion, args, round_idx, local_models[i], client_train_loaders[i], client_idx)
            # local_weights.append(result[0])
            for model_idx, weight in enumerate(result[0]):
                local_weights[model_idx].append(weight)
            local_losses.append(result[1])
            print(f'Client {i+1}/{len(selected_clients)} completely finished')
            
        train_loss = sum(local_losses) / len(selected_clients)

        # Update the global model
        global_weights = self.average_weights(local_weights, self.global_model_pool)
        for i in range(len(self.global_model_pool)):
            self.global_model_pool[i].load_state_dict(global_weights[i])
        #update model_pool
        self.assign_weights_to_model_pool(self.global_model_pool, self.model_pool)

        # , test_acc0, test_acc1, test_acc2, test_acc3
        test_loss, test_acc = validate_splitmix(self.global_model_pool, test_loader, criterion, args)
        #, test_acc0, test_acc1, test_acc2, test_acc3
        return train_loss, test_loss, test_acc
    
    
    def execute_client_round(self, criterion, args, round_idx, local_model, client_train_loader, client_idx):
        """Execute a single round of training on a client."""


        if args.use_gpu:
            local_model = [model.to(device) for model in local_model]

        optimizer = [torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay) for model in local_model]
        

        loss = 0.0
        start = time.time()
        print(f'Start epochs for Client {client_idx}')
        for epoch in range(args.num_epoch):
            print(f'Client{client_idx} - epoch {epoch+1}/{args.num_epoch}')
            loss = execute_epoch_splitmix(local_model, client_train_loader, criterion, optimizer, round_idx, args)
        
        end = time.time()
        print(f"Time = {end-start}s")

        print(f'Finished epochs for Client {client_idx}')

        local_weight = [
            {k: v.cpu() for k, v in model.state_dict(keep_vars=True).items()}
            for model in local_model
        ]

        del local_model
        torch.cuda.empty_cache()

        return local_weight, loss
    

    def get_level(self, client_idx):
        for level, group in enumerate(self.client_groups):
            if client_idx in group:
                return 2 ** level
        return -1
    

    def average_weights(self, local_weights, global_model_pool):
        """average four different model"""

        aggregated_weights_pool = []
    
        for i in range(len(global_model_pool)):
            aggregated_weights = copy.deepcopy(global_model_pool[i].state_dict())
            for key in aggregated_weights.keys():
                key_params = [local_weight[key] for local_weight in local_weights[i] if key in local_weight]
                
                if 'num_batches_tracked' in key and len(key_params) > 0:
                    aggregated_weights[key] = key_params[0] 
                    continue
                if 'running' in key and len(key_params) > 0:
                    aggregated_weights[key] = sum(key_params) / len(key_params)  
                    continue
                
                if len(key_params) > 0:
                    aggregated_weights[key] = torch.mean(torch.stack(key_params), dim=0)
                else:
                    print(f"Key {key} is not present in any local weights. Keeping global_model's default value.")
                
            aggregated_weights_pool.append(aggregated_weights)
        
        return aggregated_weights_pool
    
    
    def assign_weights_to_model_pool(self, global_model_pool, model_pool):
        """
        將聚合後的權重分配給每個模型
        """
        for i in range(len(model_pool)):
            model_pool[i].load_state_dict(global_model_pool[i].state_dict(), strict=False)