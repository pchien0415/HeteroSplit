#!/usr/bin/env python3
import os
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.optim

from utils.utils import AverageMeter
from config import device

def validate(model, val_loader, criterion, args, client_idx=0, exit_idx=0, save=False):

    exit_idx = 0 # 沒指定early exit 因此會全部跑

    losses = AverageMeter()
    top1 = [AverageMeter() for _ in range(3)]

    print(f'Validation results for Global')
    model.eval()
    with torch.no_grad():
        for i, (inp, target) in enumerate(val_loader):
            inp, target = inp.to(device), target.to(device)

            output = model(inp, manual_early_exit_index=exit_idx)
            loss = 0.0
            
            # loss
            for j in range(len(output)):
                loss += criterion(output[j], target)
            
            # accuracy
            for j in range(len(output)):
                _, predicted = torch.max(output[j], dim=1) # values, indices 
                correct = (predicted == target).sum().item()
                total = target.size(0)
                prec1 = correct * 100 / total
                top1[j].update(prec1, total)


            losses.update(loss.item(), inp.size(0))

    for i in range(3):
        print(f'top1 accuracy exit{i+1}: {top1[i].avg}')

    return losses, top1


def validate_split(model, val_loader, criterion, args, client_idx=0, exit_idx=0, save=False):

    losses = AverageMeter()
    top1 = [AverageMeter() for _ in range(3)]

    print(f'Validation results for Global')

    model.eval()
    with torch.no_grad():
        for i, (inp, target) in enumerate(val_loader):
            inp, target = inp.to(device), target.to(device)

            # 多了一個中間值的輸出
            _, output = model(inp)
            loss = 0.0
            
            # loss
            for j in range(len(output)):
                loss += criterion(output[j], target)
            
            # accuracy
            for j in range(len(output)):
                _, predicted = torch.max(output[j], dim=1)
                correct = (predicted == target).sum().item()
                total = target.size(0)
                prec1 = correct * 100 / total
                top1[j].update(prec1, total)


            losses.update(loss.item(), inp.size(0))

    for i in range(3):
        print(f'top1 accuracy exit{i+1}: {top1[i].avg}')

    return losses, top1



def validate_splitmix(model_pool, val_loader, criterion, args, save=False):

    losses = AverageMeter()
    top = AverageMeter()

    # top0 = AverageMeter()
    # top1 = AverageMeter()
    # top2 = AverageMeter()
    # top3 = AverageMeter()

    print(f'Validation results for Global')

    with torch.no_grad():
        for i, (inp, target) in enumerate(val_loader):
            inp, target = inp.to(device), target.to(device)

            outputs = []
            for model in model_pool:
                model.eval()  
                _, output = model(inp)
                outputs.append(output[0])

            
            # **Ensemble Learning**: 平均预测概率
            ensemble_output = sum(outputs) / len(model_pool)

            # loss
            loss = criterion(ensemble_output, target)

            # accuracy
            _, predicted = torch.max(ensemble_output, dim=1)
            # _, predicted0 = torch.max(outputs[0], dim=1)
            # _, predicted1 = torch.max(outputs[1], dim=1)
            # _, predicted2 = torch.max(outputs[2], dim=1)
            # _, predicted3 = torch.max(outputs[3], dim=1)
            correct = (predicted == target).sum().item()
            # correct0 = (predicted0 == target).sum().item()
            # correct1 = (predicted1 == target).sum().item()
            # correct2 = (predicted2 == target).sum().item()
            # correct3 = (predicted3 == target).sum().item()

            total = target.size(0)
            prec = correct * 100 / total
            top.update(prec, total)

            # prec0 = correct0 * 100 / total
            # top0.update(prec0, total)

            # prec1 = correct1 * 100 / total
            # top1.update(prec1, total)

            # prec2 = correct2 * 100 / total
            # top2.update(prec2, total)

            # prec3 = correct3 * 100 / total
            # top3.update(prec3, total)


            losses.update(loss.item(), inp.size(0))

    print(f'top1 accuracy : {top.avg}')
    # , top0, top1, top2, top3
    return losses, top