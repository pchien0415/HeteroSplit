import os
import wandb
from utils.store_path import CHECKPOINT_ROOT

import torch.nn as nn
import numpy as np
import random
import torch
# func
from args import arg_parser, modify_args
from data_utils.dataloader import get_datasets, get_dataloaders, get_user_groups
from utils.utils import measure_flops
from thop import profile
# model type
# from models.resnet import ResNet20
from models.resnet20_split import (
    ResNet20_split,
    ResNet20_split_1,
    ResNet20_split_1to2,
    ResNet20_split_1to4,
    ResNet20_split_2,
    ResNet20_split_2to4,
    ResNet20_split_3to4,
    ResNet20_split_05,
    ResNet20_split_06to4,
    ResNet20_split_025,
    ResNet20_split_026to4,
    ResNet20_split_06to2,
    ResNet20_split_026to2,
)
from models.resnet32_split import (
    ResNet32_split,
    ResNet32_split_1,
    ResNet32_split_1to2,
    ResNet32_split_1to4,
    ResNet32_split_2,
    ResNet32_split_2to4,
    ResNet32_split_3to4,
    ResNet32_split_05,
    ResNet32_split_06to4,
    ResNet32_split_025,
    ResNet32_split_026to4,
    ResNet32_split_06to2,
    ResNet32_split_026to2,
)
# algo
from algo.fed import Federator
from algo.base import Base
from algo.depthfl import DepthFL
from algo.splitmix import Splitmix
from algo.propose import Propose
from algo.random import Random
# loss
from models.model_utils import KDLoss
# gpu
from config import device

np.set_printoptions(precision=2)

args = arg_parser.parse_args()
args = modify_args(args)

# Reproducibility settings
torch.manual_seed(args.seed)                
torch.cuda.manual_seed(args.seed)           
torch.cuda.manual_seed_all(args.seed)       
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True 
torch.backends.cudnn.benchmark = False      



def main():
    # Initialize wandb
    wandb.init(project='New_clientratio_cifar10', mode='online', config=vars(args))

    # Initialize model
    global_model = ResNet20_split(num_classes=args.num_classes).to(device)
    if args.algo == 'depthfl' or args.algo == 'random' or args.algo == 'propose':
        model_pool = [
            ResNet20_split_1(num_classes=args.num_classes).to(device),      # 0
            ResNet20_split_1to2(num_classes=args.num_classes).to(device),   # 1
            ResNet20_split_1to4(num_classes=args.num_classes).to(device),   # 2
            ResNet20_split_2(num_classes=args.num_classes).to(device),      # 3
            ResNet20_split_2to4(num_classes=args.num_classes).to(device),   # 4
            ResNet20_split_3to4(num_classes=args.num_classes).to(device),   # 5
            ResNet20_split_05(num_classes=args.num_classes).to(device),     # 6  FLOPs: 7.52M
            ResNet20_split_06to2(num_classes=args.num_classes).to(device),  # 7
            ResNet20_split_06to4(num_classes=args.num_classes).to(device),  # 8
            ResNet20_split_025(num_classes=args.num_classes).to(device),    # 9　FLOPs: 2.80M
            ResNet20_split_026to2(num_classes=args.num_classes).to(device), # 10
            ResNet20_split_026to4(num_classes=args.num_classes).to(device)  # 11
        ]
        # model_pool = [
        #     ResNet32_split_1(num_classes=args.num_classes).to(device),      # 0
        #     ResNet32_split_1to2(num_classes=args.num_classes).to(device),   # 1
        #     ResNet32_split_1to4(num_classes=args.num_classes).to(device),   # 2
        #     ResNet32_split_2(num_classes=args.num_classes).to(device),      # 3
        #     ResNet32_split_2to4(num_classes=args.num_classes).to(device),   # 4
        #     ResNet32_split_3to4(num_classes=args.num_classes).to(device),   # 5
        #     ResNet32_split_05(num_classes=args.num_classes).to(device),     # 6  FLOPs: 7.52M
        #     ResNet32_split_06to2(num_classes=args.num_classes).to(device),  # 7
        #     ResNet32_split_06to4(num_classes=args.num_classes).to(device),  # 8
        #     ResNet32_split_025(num_classes=args.num_classes).to(device),    # 9　FLOPs: 2.80M
        #     ResNet32_split_026to2(num_classes=args.num_classes).to(device), # 10
        #     ResNet32_split_026to4(num_classes=args.num_classes).to(device)  # 11
        # ]
        for model in model_pool:
            model.load_state_dict(global_model.state_dict(), strict=False)
    elif args.algo == 'splitmix':
        global_model_pool = [
            ResNet20_split_1(num_classes=args.num_classes).to(device),
            ResNet20_split_1(num_classes=args.num_classes).to(device),
            ResNet20_split_1(num_classes=args.num_classes).to(device),
            ResNet20_split_1(num_classes=args.num_classes).to(device)
        ]
        model_pool = [
            ResNet20_split_1(num_classes=args.num_classes).to(device),
            ResNet20_split_1(num_classes=args.num_classes).to(device),
            ResNet20_split_1(num_classes=args.num_classes).to(device),
            ResNet20_split_1(num_classes=args.num_classes).to(device)
        ]
        for i in range(len(model_pool)):
            model_pool[i].load_state_dict(global_model_pool[i].state_dict(), strict=False)  

    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Load datasets and dataloaders
    train_set, val_set, test_set = get_datasets(args)
    _, val_loader, test_loader = get_dataloaders(args, args.batch_size, (train_set, val_set, test_set))
    val_set = val_set or val_loader.dataset  # Use test set as validation set if val_set is None 

    # Get non-IID user groups
    train_user_groups = get_user_groups(train_set, args)

    # algo
    if args.algo == 'depthfl':
        print("Algo: depthfl")
        propose = DepthFL(global_model, model_pool, args)
        best_acc1 = propose.train(train_set, test_loader, train_user_groups, criterion, args, args.batch_size)
    elif args.algo == 'random':
        print("Algo: random")
        propose = Random(global_model, model_pool, args)
        best_acc1 = propose.train(train_set, test_loader, train_user_groups, criterion, args, args.batch_size)
    elif args.algo == 'propose':
        print("Algo: propose")
        propose2 = Propose(global_model, model_pool, args)
        best_acc1 = propose2.train(train_set, test_loader, train_user_groups, criterion, args, args.batch_size)
    elif args.algo == 'splitmix':
        print("Algo: splitmix")
        splitmix = Splitmix(global_model_pool, model_pool, args)
        best_acc1 = splitmix.train(train_set, test_loader, train_user_groups, criterion, args, args.batch_size)

    # Print best validation accuracies
    if args.algo == 'splitmix':
        print(f'Best acc: {best_acc1:.4f}')
    else:
        for i, acc in enumerate(best_acc1):
            print(f'Best val_acc1 [{i}]: {acc:.4f}')
    
    print('End of Training')

    return





if __name__ == '__main__':
    main()