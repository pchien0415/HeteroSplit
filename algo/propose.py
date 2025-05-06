import copy
import wandb
import random
import time
import numpy as np
import torch
from collections import defaultdict

import torch.multiprocessing as mp
from config import device
from predict import validate_split
from data_utils.dataloader import get_client_dataloader
from train import execute_epoch_split

class Propose:
    def __init__(self, global_model, model_pool, args, client_groups=[]):
        """Initialize Federator with global model and arguments."""
        self.global_model = global_model
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
        self.record_table = defaultdict(int)
        self.record_table = {client_idx: {"level1": 0, "level2": 0} for client_idx in range(self.num_clients)}


    def train(self, train_set, test_loader, user_group, criterion, args, batch_size):
        """Perform federated training over multiple rounds."""
        best_acc = [0, 0, 0] # 0

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
        levels = [self.get_level(client_idx) for client_idx in selected_clients] # 0 1 2
        print(f"Client = {selected_clients}")
        print(f"初始 levels: {levels}")
        

        '''==========================Algo========================='''
        times_level2 = [self.record_table[client_idx]["level2"] for client_idx in selected_clients]
        times_level1 = [self.record_table[client_idx]["level1"] for client_idx in selected_clients]
        print(f"更新前的記錄表: {dict(sorted(self.record_table.items()))}")

        final_levels = [0] * len(selected_clients)

        # level=2
        sorted_indices_level2 = []
        level2_count = levels.count(2)
        if level2_count > 0:
            # 按 level2 次數排序，取出次數最少的客戶端索引
            sorted_indices_level2 = sorted(range(len(times_level2)), key=lambda i: times_level2[i])
            for i in range(level2_count):
                final_levels[sorted_indices_level2[i]] = 2

            # record assigned client
            assigned_indices = set(sorted_indices_level2[:level2_count])
        else:
            assigned_indices = set()  # 如果没有 level2，初始化为空集
            
        # level=1
        level1_count = levels.count(1)
        if level1_count > 0:
            remaining_indices = [i for i in range(len(selected_clients)) if i not in assigned_indices]
            # 按 level1 次數排序，取出次數最少的客戶端索引
            sorted_indices_level1 = sorted(remaining_indices, key=lambda i: times_level1[i])
            for i in range(level1_count):
                final_levels[sorted_indices_level1[i]] = 1
                assigned_indices.add(sorted_indices_level1[i])

        # level=0
        remaining_indices = [i for i in range(len(selected_clients)) if i not in assigned_indices]
        for idx in remaining_indices:
            final_levels[idx] = 0

        # update
        print("最終分配的 final_levels:", final_levels)

        '''
            levels       : 原本計算能力
            final_levels : 需要計算多少(誰幫你算)
        '''

        # update record_table
        for i, final_level in enumerate(final_levels):
            if final_level == 2:
                self.record_table[selected_clients[i]]["level2"] += 1
            elif final_level == 1:
                self.record_table[selected_clients[i]]["level1"] += 1

        # local model setting
        local_models = []
        for level, final_level in zip(levels, final_levels):
            if level == 0 and final_level == 0:
                local_models.append([copy.deepcopy(self.model_pool[0])])
            elif level == 0 and final_level == 1:
                local_models.append([copy.deepcopy(self.model_pool[6]), copy.deepcopy(self.model_pool[7])])
            elif level == 0 and final_level == 2:
                local_models.append([copy.deepcopy(self.model_pool[9]), copy.deepcopy(self.model_pool[11])])
            elif level == 1 and final_level == 0:
                local_models.append([copy.deepcopy(self.model_pool[0])])
            elif level == 1 and final_level == 1:
                local_models.append([copy.deepcopy(self.model_pool[1])])
            elif level == 1 and final_level == 2:
                local_models.append([copy.deepcopy(self.model_pool[6]), copy.deepcopy(self.model_pool[8])])
            elif level == 2 and final_level == 0:
                local_models.append([copy.deepcopy(self.model_pool[0])])
            elif level == 2 and final_level == 1:
                local_models.append([copy.deepcopy(self.model_pool[0]), copy.deepcopy(self.model_pool[3])])
            elif level == 2 and final_level == 2:
                local_models.append([copy.deepcopy(self.model_pool[2])])

        print(f"更新後的記錄表: {dict(sorted(self.record_table.items()))}")


        local_weights = []
        local_losses = []
        # selected clients local training
        for i, client_idx in enumerate(selected_clients):
            result = self.execute_client_round(criterion, args, round_idx, local_models[i], client_train_loaders[i], client_idx)
            # local_weights.append(result[0])
            for weight in result[0]:
                local_weights.append(weight)
            local_losses.append(result[1])
            print(f'Client {i+1}/{len(selected_clients)} completely finished')
            
        train_loss = sum(local_losses) / len(selected_clients)

        # Update the global model
        global_weights = self.average_weights(local_weights, self.global_model)
        self.global_model.load_state_dict(global_weights)
        #update model_pool
        self.assign_weights_to_model_pool(self.global_model, self.model_pool)

        test_loss, test_acc = validate_split(self.global_model, test_loader, criterion, args)
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
            loss = execute_epoch_split(local_model, client_train_loader, criterion, optimizer, round_idx, args)
        
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
                return level
        return -1
    

    def average_weights(self, local_weights, global_model):
        """聚合僅考慮深度縮放的模型權重。"""
        aggregated_weights = copy.deepcopy(global_model.state_dict())

        for key in aggregated_weights.keys():
            key_params = [local_weight[key] for local_weight in local_weights if key in local_weight]
            
            if 'num_batches_tracked' in key and len(key_params) > 0:
                aggregated_weights[key] = key_params[0] 
                continue
            if 'running' in key and len(key_params) > 0:
                aggregated_weights[key] = sum(key_params) / len(key_params)  
                continue

            if len(key_params) > 0:  # 至少有一個模型包含該鍵
                # 計算等權平均
                aggregated_weights[key] = torch.mean(torch.stack(key_params), dim=0)
            else:
                # 如果沒有模型包含該鍵，保留全局模型的默認值
                print(f"Key {key} is not present in any local weights. Keeping global_model's default value.")
        
        return aggregated_weights
    
    
    def assign_weights_to_model_pool(self, global_model, model_pool):
        """
        將聚合後的權重分配給每個模型
        """
        for model in model_pool:
            model.load_state_dict(global_model.state_dict(), strict=False)