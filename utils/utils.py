import numpy as np
import torch
from utils.op_counter import measure_model
from models.resnet20_split import ResNet20_split
from models.resnet32_split import ResNet32_split


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, round, args):
    if args.lr_type == 'multistep':
        lr, decay_rate = args.lr, args.decay_rate
        if round >= args.decay_rounds[1]:
            lr *= decay_rate ** 2
        elif round >= args.decay_rounds[0]:
            lr *= decay_rate
    else:
        lr = args.lr
    # 更新所有參數組的學習率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def measure_flops(args):
    model = ResNet32_split()
    model.eval()
    n_flops, n_params = measure_model(model, args.image_size[0], args.image_size[1])
    print(f"-------FLOPS: {n_flops}-----PARAMETERS: {n_params}----------")
    del (model)