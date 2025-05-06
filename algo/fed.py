import copy
import wandb
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.multiprocessing as mp

from predict import validate
from data_utils.dataloader import get_client_dataloader
from train import execute_epoch
from utils.grad_traceback import get_downscale_index

mp.set_start_method('spawn', force=True)


class Federator:
    def __init__(self, global_model, args, client_groups=[]):
        """Initialize Federator with global model and arguments."""
        self.global_model = global_model
        self.num_rounds = args.num_rounds
        self.num_clients = args.num_clients
        self.sample_rate = args.sample_rate
        self.alpha = args.alpha
        self.vertical_scale_ratios = args.vertical_scale_ratios
        self.horizontal_scale_ratios = args.horizontal_scale_ratios
        self.client_split_ratios = args.client_split_ratios
        self.num_levels = len(self.client_split_ratios)
        self.idx_dicts = [get_downscale_index(global_model, args, s) for s in self.vertical_scale_ratios]
        self.client_groups = client_groups
        self.use_gpu = args.use_gpu
        self.record_table = defaultdict(int)


    def fed_train(self, train_set, test_loader, user_groups, criterion, args, batch_size):
        """Perform federated training over multiple rounds."""
        best_acc = [0, 0, 0] # 0

        # Assign clients to groups based on split ratios.
        if not self.client_groups:
            idxs = np.arange(self.num_clients)
            np.random.seed(args.seed)
            shuffled_clients = np.random.permutation(idxs)
            s = 0
            for ratio in self.client_split_ratios:
                e = s + int(len(shuffled_clients) * ratio)
                self.client_groups.append(shuffled_clients[s:e])
                s = e

        for round_idx in range(self.num_rounds):
            print(f'\n | Global Training Round : {round_idx} |\n')
            train_loss, test_loss, test_acc = self.execute_round(
                train_set, test_loader, user_groups, criterion, args, batch_size, round_idx
            )

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

    def execute_round(self, train_set, test_loader, user_groups, criterion, args, batch_size, round_idx):
        """Execute a single round of federated training."""
        self.global_model.train()

        # random sample client
        m = max(int(self.sample_rate * self.num_clients), 1)
        selected_clients = np.random.choice(range(self.num_clients), m, replace=False)
        client_train_loaders = [
            get_client_dataloader(train_set, user_groups[0][idx], args, batch_size)
            for idx in selected_clients
        ]
        levels = [self.get_level(idx) for idx in selected_clients] 
        print(f"Client = {selected_clients}")
        print(f"Initial levels: {levels}")

        # =======================================Algo=============================================
        """Balance levels based on client training times."""
        if args.algo == 'average_time':
            # 取得選擇的客戶端對應的訓練次數
            times = [self.record_table[idx] for idx in selected_clients]
            # 印出每個選擇客戶端的當前訓練次數
            for idx in selected_clients:
                print(f"當前選擇的 client {idx} : 訓練次數 {self.record_table[idx]}")
            # 根據訓練次數進行排序，取得排序後的索引index(訓練越少次的索引排越後面 次數 大->小)
            sorted_indices = sorted(range(len(times)), key=lambda i: times[i], reverse=True)
            # 根據 levels 的值進行排序（由小到大）
            sorted_levels = sorted(levels)
            # 初始化一個新的 levels 列表
            final_levels = [0] * len(levels)
            # 根據排序後的索引重新分配 levels # 根據訓練深層次數排序 並根據索引依序填上levels
            for i, idx in enumerate(sorted_indices):
                final_levels[idx] = sorted_levels[i]
            # 印出排序後的 levels
            print("排序後的 levels:", final_levels)
            # 更新 levels
            levels = final_levels
            # 更新訓練次數（只有訓練 level 2 的客戶端會增加次數）
            for i in range(m):  # 假設 m 是選擇的客戶端數量
                if levels[i] == 2:
                    self.record_table[selected_clients[i]] += 1
            # 再次印出當前選擇的客戶端訓練次數
            for idx in selected_clients:
                print(f"當前選擇的 client {idx} : 訓練次數 {self.record_table[idx]}")
            # 最後印出更新後的整個表
            print(f"更新後的記錄表: {dict(self.record_table)}")
        else:
            print("DepthFL")

        h_scale_ratios = [self.horizontal_scale_ratios[level] for level in levels]
        #(width scale)
        scales = [self.vertical_scale_ratios[level] for level in levels]
        local_models = [self.get_local_split(levels[i], scales[i]) for i in range(len(selected_clients))]

        # model = copy.deepcopy(self.global_model)
        # local_models = [model for i in range(len(selected_clients))]

        local_weights = []
        local_losses = []
        local_grad_flags = []

        for i, idx in enumerate(selected_clients):
            #client_args = [criterion, args, round_idx, local_models[i], client_train_loaders[i], h_scale_ratios[i], idx]
            result = self.execute_client_round(criterion, args, round_idx, local_models[i], client_train_loaders[i], h_scale_ratios[i], idx)
            if args.use_gpu:
                for k, v in result[0].items():
                    result[0][k] = v.cuda(0)

            local_weights.append(result[0])
            local_grad_flags.append(result[1])
            local_losses.append(result[2])
            print(f'Client {i+1}/{len(selected_clients)} completely finished')
            
        train_loss = sum(local_losses) / len(selected_clients)

        # Update the global model
        print("----------average weight-----------")
        global_weights = self.average_weights(local_weights, local_grad_flags, levels, self.global_model)
        print("----------Done-----------")
        self.global_model.load_state_dict(global_weights)

        print("----------validate-----------")
        test_loss, test_acc = validate(self.global_model, test_loader, criterion, args)
        print("----------Done-----------")
        return train_loss, test_loss, test_acc
    
    def get_level(self, idx):
        try:
            level = np.where([idx in c for c in self.client_groups])[0][0]
        except:
            level = -1
        return level
    
    def get_loss(self, model, dataloader):
        # 使用全局模型對每個client的數據進行前向傳播，計算損失值
        model.cuda()
        model.eval()
        loss_fn = torch.nn.CrossEntropyLoss()
        total_loss = 0
        total_samples = 0
        with torch.no_grad():
            for i, (inp, target) in enumerate(dataloader):
                target = target.cuda()
                inp = inp.cuda()
                outputs = model(inp)
                loss = loss_fn(outputs[3], target)
                total_loss += loss.item() * inp.size(0)
                total_samples += inp.size(0)
                break
        return total_loss / total_samples
    
    # width scale
    def get_local_split(self, level, scale):
        """Create a local model scaled according to the level and scale."""
        model = copy.deepcopy(self.global_model)

        if scale == 1:
            return model
        print("---------------------width scale---------------------")
        model_kwargs = model.stored_inp_kwargs
        if 'scale' in model_kwargs.keys():
            model_kwargs['scale'] = scale
        else:
            model_kwargs['params']['scale'] = scale

        # 根據 model_kwargs 重新創建一個新模型實例
        local_model = type(self.global_model)(**model_kwargs) 

        local_state_dict = local_model.state_dict()

        for n, p in self.global_model.state_dict().items():

            if 'num_batches_tracked' in n:
                local_state_dict[n] = p
                continue

            global_shape = p.shape
            local_shape = local_state_dict[n].shape

            if len(global_shape) != len(local_shape):
                print('Models are not alignable!')
                raise RuntimeError

            idx_array = self.fix_idx_array(self.idx_dicts[level][n], local_shape)
            local_state_dict[n] = p[idx_array].reshape(local_shape)

        local_model.load_state_dict(local_state_dict)

        return local_model
    
    def fix_idx_array(self, idx_array, local_shape):
        """Fix the index array to match the shape of the local model."""
        idx_shape = self.get_idx_shape(idx_array, local_shape)
        if all([idx_shape[i] >= local_shape[i] for i in range(len(local_shape))]):
            pass
        else:
            idx_array = idx_array[idx_array.sum(dim=1).argmax()].repeat((idx_array.shape[0], 1))
            idx_shape = self.get_idx_shape(idx_array, local_shape)

        ind_list = [slice(None)] * len(idx_array.shape)
        for i in range(len(local_shape)):

            lim = idx_array.shape[i]
            while idx_shape[i] != local_shape[i]:
                lim -= 1
                ind_list[i] = slice(0, lim)
                idx_shape = self.get_idx_shape(idx_array[tuple(ind_list)], local_shape)

        tmp = torch.zeros_like(idx_array, dtype=bool)
        tmp[tuple(ind_list)] = idx_array[tuple(ind_list)]
        idx_array = tmp

        if len(idx_array.shape) == 4:
            dim_1 = idx_array.shape[2] // 2
            dim_2 = idx_array.shape[3] // 2
            if idx_array.sum(dim=0).sum(dim=0)[0, 0] != idx_array.sum(dim=0).sum(dim=0)[dim_1, dim_2]:
                idx_array = idx_array[:, :, dim_1, dim_2].repeat(idx_array.shape[2], idx_array.shape[3], 1, 1).permute(
                    2, 3, 0, 1)
        return idx_array
    
    def get_idx_shape(self, inp, local_shape):
        """Compute the shape of the index array based on input and local shape."""
        # Return the output shape for binary mask input
        # [[1, 1, 0], [1, 1, 0], [0, 0, 0,]] -> [2, 2]
        if any([s == 0 for s in inp.shape]):
            print('Indexing error')
            raise RuntimeError

        if len(local_shape) == 4:
            dim_1 = inp.shape[2] // 2
            dim_2 = inp.shape[3] // 2
            idx_shape = (inp[:, 0, dim_1, dim_2].sum().item(),
                         inp[0, :, dim_1, dim_2].sum().item(), *local_shape[2:])
        elif len(local_shape) == 2:
            idx_shape = (inp[:, 0].sum().item(),
                         inp[0, :].sum().item())
        else:
            idx_shape = (inp.sum(),)

        return idx_shape

    
    def average_weights(self, w, grad_flags, levels, model):
        """Average the weights of the local models to update the global model."""
        w_avg = copy.deepcopy(model.state_dict())
        for key in w_avg.keys():

            if 'num_batches_tracked' in key:
                w_avg[key] = w[0][key]
                continue

            if 'running' in key:
                w_avg[key] = sum([w_[key] for w_ in w]) / len(w)
                continue
            
            # 儲存各個客戶端的權重加總值與計數次數
            tmp = torch.zeros_like(w_avg[key])
            count = torch.zeros_like(tmp)
            for i in range(len(w)):
                if grad_flags[i][key]:
                    idx = self.idx_dicts[levels[i]][key]
                    idx = self.fix_idx_array(idx, w[i][key].shape)
                    tmp[idx] += w[i][key].flatten()
                    count[idx] += 1
            w_avg[key][count != 0] = tmp[count != 0]
            count[count == 0] = 1
            w_avg[key] = w_avg[key] / count
        return w_avg
    
    def average_weights_depth_only(self, w, grad_flags, levels, model):
        """聚合僅考慮深度縮放的模型權重。"""
        w_avg = copy.deepcopy(model.state_dict())
        
        for key in w_avg.keys():
            # 如果是統計參數，直接選取第一個客戶端的值
            if 'num_batches_tracked' in key or 'running' in key:
                w_avg[key] = w[0][key]
                continue
            
            # 初始化加權和與計數
            tmp = torch.zeros_like(w_avg[key])
            count = 0
            
            for i in range(len(w)):
                # 如果該層參數在客戶端中被訓練，則累加權重
                if grad_flags[i][key]:
                    tmp += w[i][key]
                    count += 1
            
            # 聚合結果
            if count > 0:
                w_avg[key] = tmp / count
            else:
                # 如果沒有客戶端訓練該層，保持全局參數
                w_avg[key] = model.state_dict()[key]

        return w_avg
    
    
    def execute_client_round(self, criterion, args, round_idx, local_model, client_train_loader, h_scale_ratio, idx):
        """Execute a single round of training on a client."""

        if args.use_gpu:
            local_model = local_model.cuda()

        # base_params = [v for k, v in local_model.named_parameters() if 'ee_' not in k]
        # exit_params = [v for k, v in local_model.named_parameters() if 'ee_' in k]

        optimizer = torch.optim.SGD(local_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        loss = 0.0
        start = time.time()
        print(f'Start epochs for Client {idx}')
        for epoch in range(args.num_epoch):
            print(f'Client{idx} - epoch {epoch+1}/5')
            iter_idx = round_idx
            
            loss = execute_epoch(local_model, client_train_loader, criterion, optimizer, iter_idx,
                                args, h_scale_ratio)
        end = time.time()
        print(f"Time = {end-start}s")

        print(f'Finished epochs for Client {idx}')
        local_weights = {k: v.cpu() for k, v in local_model.state_dict(keep_vars=True).items()}
        local_grad_flags = {k: v.grad is not None for k, v in local_model.state_dict(keep_vars=True).items()}

        del local_model
        torch.cuda.empty_cache()

        return local_weights, local_grad_flags, loss