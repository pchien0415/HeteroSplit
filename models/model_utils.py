import torch.nn as nn
import torch
import copy
import numpy as np


class KDLoss(nn.Module):
    def __init__(self, args):
        super(KDLoss, self).__init__()

        self.kld_loss = nn.KLDivLoss(reduction='batchmean')
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

        self.T = args.KD_T
        # self.gamma = args.KD_gamma

    def loss_fn_kd(self, pred, soft_target):
        T = self.T
        loss = self.kld_loss(self.log_softmax(pred / T), self.softmax(soft_target / T))  * T * T
        return loss
