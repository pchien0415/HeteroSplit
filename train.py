#!/usr/bin/env python3

import time
import numpy as np
import torch

from utils.utils import adjust_learning_rate, AverageMeter
from config import device
from models.model_utils import KDLoss


def execute_epoch(model, train_loader, criterion, optimizer, round, args, h_level):
    losses = AverageMeter()

    model.train()

    for _, (inp, target) in enumerate(train_loader):
        
        adjust_learning_rate(optimizer, round, args)

        inp = inp.to(device)
        target = target.to(device)

        output = model(inp, manual_early_exit_index=h_level) # model的forward()

        loss = 0.0
        # 如果要用distillation記得回去看code
        for j in range(len(output)):
            loss += criterion(output[j], target)

        losses.update(loss.item(), inp.size(0)) # inp.size(0) = batchsize

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg


# def execute_epoch_split(models, train_loader, criterion, optimizers, round, args):
#     losses = AverageMeter()

#     for model in models:
#         model.train()

#     for _, (inp, target) in enumerate(train_loader):
        
#         for optimizer in optimizers:
#             adjust_learning_rate(optimizer, round, args)

#         inp = inp.to(device)
#         target = target.to(device)

#         intermediate, output = models[0](inp) # model的forward()

#         total_loss = 0.0
#         # 如果要用distillation記得回去看code
#         for j in range(len(output)):
#             total_loss += criterion(output[j], target)
            
#         optimizers[0].zero_grad()

#         if len(models) == 1:
#             losses.update(total_loss.item(), inp.size(0))
#             total_loss.backward()
#             optimizers[0].step()
#         else:
#             _, output = models[1](intermediate)
            
#             #loss_second = 0
#             for j in range(len(output)): 
#                 total_loss += criterion(output[j], target)
#             losses.update(total_loss.item(), inp.size(0))
#             optimizers[1].zero_grad()
#             total_loss.backward()

#             optimizers[0].step()
#             optimizers[1].step()

#     return losses.avg

def execute_epoch_split(models, train_loader, criterion, optimizers, round, args):
    losses = AverageMeter()

    for model in models:
        model.train()

    criterion_kl = KDLoss(args)

    for _, (inp, target) in enumerate(train_loader):
        
        for optimizer in optimizers:
            adjust_learning_rate(optimizer, round, args)
        
        total_output = []
        total_loss = 0.0

        if len(models) == 1:
            inp = inp.to(device)
            target = target.to(device)

            intermediate, output = models[0](inp) # model的forward()

            # 如果要用distillation記得回去看code
            for j in range(len(output)):
                total_loss += criterion(output[j], target)
                if args.KD:
                    for i in range(len(output)):
                        if j == i:
                            continue
                        else:
                            total_loss += args.KD_gamma*criterion_kl.loss_fn_kd(output[j], output[i])/(len(output) - 1) # 設計每個老師權重 以及 整個kd權重
                
            optimizers[0].zero_grad()
            losses.update(total_loss.item(), inp.size(0))
            total_loss.backward()
            optimizers[0].step()

        else:
            inp = inp.to(device)
            target = target.to(device)

            intermediate, output1 = models[0](inp) # model的forward()
            total_output += output1
            
            _, output2 = models[1](intermediate)
            total_output += output2
            # 如果要用distillation記得回去看code
            for j in range(len(total_output)):
                total_loss += criterion(total_output[j], target)
                if args.KD:
                    for i in range(len(total_output)):
                        if j == i:
                            continue
                        else:
                            total_loss += args.KD_gamma*criterion_kl.loss_fn_kd(total_output[j], total_output[i])/(len(total_output) - 1) # 設計每個老師權重 以及 整個kd權重
            losses.update(total_loss.item(), inp.size(0))
            optimizers[0].zero_grad()
            optimizers[1].zero_grad()
            total_loss.backward()

            optimizers[0].step()
            optimizers[1].step()

    return losses.avg


def execute_epoch_splitmix(models, train_loader, criterion, optimizers, round, args):
    losses = AverageMeter()
    for model, optimizer in zip(models, optimizers):
        model.train()
        for _, (inp, target) in enumerate(train_loader):
            
            adjust_learning_rate(optimizer, round, args)

            inp = inp.to(device)
            target = target.to(device)

            _, output = model(inp) # model的forward()

            loss = 0.0
            # 如果要用distillation記得回去看code
            loss += criterion(output[0], target)

            losses.update(loss.item(), inp.size(0)) # inp.size(0) = batchsize

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return losses.avg






# def execute_epoch_split_real_split(models, train_loader, criterion, optimizers, round, args):
#     losses = AverageMeter()


#     for model in models:
#         model.train()

#     for _, (inp, target) in enumerate(train_loader):
        
#         for optimizer in optimizers:
#             adjust_learning_rate(optimizer, round, args)

#         inp = inp.to(device)
#         target = target.to(device)

#         intermediate, output = models[0](inp) # model的forward()

#         loss_first = 0.0
#         # 如果要用distillation記得回去看code
#         for j in range(len(output)):
#             loss_first += criterion(output[j], target)

#         if len(output) > 0:
#             losses.update(loss_first.item(), inp.size(0))
#         optimizers[0].zero_grad()

#         if len(models) == 1:
#             loss_first.backward()
#             optimizers[0].step()
#         else:
#             #print(f"Intermediate shape: {intermediate.size()}")
#             if len(output) > 0:
#                 loss_first.backward(retain_graph=True)
        
#             intermediate_copy = intermediate.clone().detach().requires_grad_(True)
            
#             _, output = models[1](intermediate_copy)
            
#             loss_second = 0
#             for j in range(len(output)): 
#                 loss_second += criterion(output[j], target)
#             losses.update(loss_second.item(), inp.size(0))
#             optimizers[1].zero_grad()
#             loss_second.backward()

#             grad_from_second = intermediate_copy.grad.clone().detach()
#             intermediate.backward(grad_from_second)

#             optimizers[0].step()
#             optimizers[1].step()

#     return losses.avg