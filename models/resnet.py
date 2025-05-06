import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = F.relu(out)

        return out
    
class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet20, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        # 手动实现 layer1
        self.block1 = BasicBlock(16, 16)
        self.block2 = BasicBlock(16, 16)
        self.block3 = BasicBlock(16, 16)
        # Early Exit 層
        self.early_exit_conv = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.early_exit_bn = nn.BatchNorm2d(16)
        self.early_exit_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.early_exit_fc = nn.Linear(16, num_classes)
        # 手动实现 layer2
        downsample = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
        )
        self.block4 = BasicBlock(16, 32, stride=2, downsample=downsample)
        self.block5 = BasicBlock(32, 32)
        self.block6 = BasicBlock(32, 32)
        # Early Exit 層
        self.early_exit_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.early_exit_bn2 = nn.BatchNorm2d(32)
        self.early_exit_pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.early_exit_fc2 = nn.Linear(32, num_classes)
        # 手动实现 layer3
        downsample = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
        )
        self.block7 = BasicBlock(32, 64, stride=2, downsample=downsample)
        self.block8 = BasicBlock(64, 64)
        self.block9 = BasicBlock(64, 64)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x, manual_early_exit_index=0):
        ee_outs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        # Early Exit
        ee_out = self.early_exit_conv(x)
        ee_out = self.early_exit_bn(ee_out)
        ee_out = F.relu(ee_out)
        ee_out = self.early_exit_pool(ee_out)
        ee_out = torch.flatten(ee_out, 1)
        ee_out = self.early_exit_fc(ee_out)
        ee_outs.append(ee_out)
        if manual_early_exit_index == 1:
            return ee_outs
        x = self.block3(x)
        x = self.block4(x)
        # Early Exit2
        ee_out2 = self.early_exit_conv2(x)
        ee_out2 = self.early_exit_bn2(ee_out2)
        ee_out2 = F.relu(ee_out2)
        ee_out2 = self.early_exit_pool2(ee_out2)
        ee_out2 = torch.flatten(ee_out2, 1)
        ee_out2 = self.early_exit_fc2(ee_out2)
        ee_outs.append(ee_out2)
        if manual_early_exit_index == 2:
            return ee_outs
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        ee_outs.append(x)

        return ee_outs