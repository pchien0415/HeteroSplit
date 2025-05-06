import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from thop import profile

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

class ResNet32_split(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet32_split, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # 手动实现 layer1
        self.block1 = BasicBlock(16, 16)
        self.block2 = BasicBlock(16, 16)
        self.block3 = BasicBlock(16, 16)
        self.block4 = BasicBlock(16, 16)
        self.block5 = BasicBlock(16, 16)

        # Early Exit1
        self.early_exit_conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.early_exit_bn1 = nn.BatchNorm2d(32)
        self.early_exit_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.early_exit_fc = nn.Linear(32, num_classes)

        # 手动实现 layer2
        downsample = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
        )
        self.block6 = BasicBlock(16, 32, stride=2, downsample=downsample)
        self.block7 = BasicBlock(32, 32)
        self.block8 = BasicBlock(32, 32)
        self.block9 = BasicBlock(32, 32)
        self.block10 = BasicBlock(32, 32)

        # Early Exit2
        self.early_exit_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.early_exit_bn2 = nn.BatchNorm2d(64)
        self.early_exit_pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.early_exit_fc2 = nn.Linear(64, num_classes)

        # 手动实现 layer3
        downsample = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
        )
        self.block11 = BasicBlock(32, 64, stride=2, downsample=downsample)
        self.block12 = BasicBlock(64, 64)
        self.block13 = BasicBlock(64, 64)
        self.block14 = BasicBlock(64, 64)
        self.block15 = BasicBlock(64, 64)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        ee_outs = []

        # Initial convolution and BatchNorm
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Early Exit
        ee_out = self.early_exit_conv1(x)
        ee_out = self.early_exit_bn1(ee_out)
        ee_out = F.relu(ee_out)
        ee_out = self.early_exit_pool(ee_out)
        ee_out = torch.flatten(ee_out, 1)
        ee_out = self.early_exit_fc(ee_out)
        ee_outs.append(ee_out)

        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        # Early Exit2
        ee_out2 = self.early_exit_conv2(x)
        ee_out2 = self.early_exit_bn2(ee_out2)
        ee_out2 = F.relu(ee_out2)
        ee_out2 = self.early_exit_pool2(ee_out2)
        ee_out2 = torch.flatten(ee_out2, 1)
        ee_out2 = self.early_exit_fc2(ee_out2)
        ee_outs.append(ee_out2)

        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)


        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        ee_outs.append(x)

        return 0, ee_outs
    
    
class ResNet32_split_1(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet32_split_1, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # 手动实现 layer1
        self.block1 = BasicBlock(16, 16)
        self.block2 = BasicBlock(16, 16)
        self.block3 = BasicBlock(16, 16)

        # Early Exit1
        self.early_exit_conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.early_exit_bn1 = nn.BatchNorm2d(32)
        self.early_exit_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.early_exit_fc = nn.Linear(32, num_classes)

    def forward(self, x):
        ee_outs = []

        # Initial convolution and BatchNorm
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Early Exit
        ee_out = self.early_exit_conv1(x)
        ee_out = self.early_exit_bn1(ee_out)
        ee_out = F.relu(ee_out)
        ee_out = self.early_exit_pool(ee_out)
        ee_out = torch.flatten(ee_out, 1)
        ee_out = self.early_exit_fc(ee_out)
        ee_outs.append(ee_out)

        return x, ee_outs
    
class ResNet32_split_1to2(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet32_split_1to2, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # 手动实现 layer1
        self.block1 = BasicBlock(16, 16)
        self.block2 = BasicBlock(16, 16)
        self.block3 = BasicBlock(16, 16)
        self.block4 = BasicBlock(16, 16)
        self.block5 = BasicBlock(16, 16)

        # Early Exit1
        self.early_exit_conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.early_exit_bn1 = nn.BatchNorm2d(32)
        self.early_exit_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.early_exit_fc = nn.Linear(32, num_classes)

        # 手动实现 layer2
        downsample = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
        )
        self.block6 = BasicBlock(16, 32, stride=2, downsample=downsample)

        # Early Exit2
        self.early_exit_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.early_exit_bn2 = nn.BatchNorm2d(64)
        self.early_exit_pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.early_exit_fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        ee_outs = []

        # Initial convolution and BatchNorm
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Early Exit
        ee_out = self.early_exit_conv1(x)
        ee_out = self.early_exit_bn1(ee_out)
        ee_out = F.relu(ee_out)
        ee_out = self.early_exit_pool(ee_out)
        ee_out = torch.flatten(ee_out, 1)
        ee_out = self.early_exit_fc(ee_out)
        ee_outs.append(ee_out)

        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        # Early Exit2
        ee_out2 = self.early_exit_conv2(x)
        ee_out2 = self.early_exit_bn2(ee_out2)
        ee_out2 = F.relu(ee_out2)
        ee_out2 = self.early_exit_pool2(ee_out2)
        ee_out2 = torch.flatten(ee_out2, 1)
        ee_out2 = self.early_exit_fc2(ee_out2)
        ee_outs.append(ee_out2)

        return x, ee_outs
    
class ResNet32_split_1to4(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet32_split_1to4, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # 手动实现 layer1
        self.block1 = BasicBlock(16, 16)
        self.block2 = BasicBlock(16, 16)
        self.block3 = BasicBlock(16, 16)
        self.block4 = BasicBlock(16, 16)
        self.block5 = BasicBlock(16, 16)

        # Early Exit1
        self.early_exit_conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.early_exit_bn1 = nn.BatchNorm2d(32)
        self.early_exit_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.early_exit_fc = nn.Linear(32, num_classes)

        # 手动实现 layer2
        downsample = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
        )
        self.block6 = BasicBlock(16, 32, stride=2, downsample=downsample)
        self.block7 = BasicBlock(32, 32)
        self.block8 = BasicBlock(32, 32)
        self.block9 = BasicBlock(32, 32)
        self.block10 = BasicBlock(32, 32)

        # Early Exit2
        self.early_exit_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.early_exit_bn2 = nn.BatchNorm2d(64)
        self.early_exit_pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.early_exit_fc2 = nn.Linear(64, num_classes)

        # 手动实现 layer3
        downsample = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
        )
        self.block11 = BasicBlock(32, 64, stride=2, downsample=downsample)
        self.block12 = BasicBlock(64, 64)
        self.block13 = BasicBlock(64, 64)
        self.block14 = BasicBlock(64, 64)
        self.block15 = BasicBlock(64, 64)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        ee_outs = []

        # Initial convolution and BatchNorm
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Early Exit
        ee_out = self.early_exit_conv1(x)
        ee_out = self.early_exit_bn1(ee_out)
        ee_out = F.relu(ee_out)
        ee_out = self.early_exit_pool(ee_out)
        ee_out = torch.flatten(ee_out, 1)
        ee_out = self.early_exit_fc(ee_out)
        ee_outs.append(ee_out)

        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        # Early Exit2
        ee_out2 = self.early_exit_conv2(x)
        ee_out2 = self.early_exit_bn2(ee_out2)
        ee_out2 = F.relu(ee_out2)
        ee_out2 = self.early_exit_pool2(ee_out2)
        ee_out2 = torch.flatten(ee_out2, 1)
        ee_out2 = self.early_exit_fc2(ee_out2)
        ee_outs.append(ee_out2)

        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)


        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        ee_outs.append(x)

        return 0, ee_outs
    
class ResNet32_split_2(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet32_split_2, self).__init__()

        self.block4 = BasicBlock(16, 16)
        self.block5 = BasicBlock(16, 16)

        # 手动实现 layer2
        downsample = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
        )
        self.block6 = BasicBlock(16, 32, stride=2, downsample=downsample)

        # Early Exit2
        self.early_exit_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.early_exit_bn2 = nn.BatchNorm2d(64)
        self.early_exit_pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.early_exit_fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        ee_outs = []

        # Initial convolution and BatchNorm
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        # Early Exit2
        ee_out2 = self.early_exit_conv2(x)
        ee_out2 = self.early_exit_bn2(ee_out2)
        ee_out2 = F.relu(ee_out2)
        ee_out2 = self.early_exit_pool2(ee_out2)
        ee_out2 = torch.flatten(ee_out2, 1)
        ee_out2 = self.early_exit_fc2(ee_out2)
        ee_outs.append(ee_out2)

        return x, ee_outs
    

class ResNet32_split_2to4(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet32_split_2to4, self).__init__()

        self.block4 = BasicBlock(16, 16)
        self.block5 = BasicBlock(16, 16)

        # 手动实现 layer2
        downsample = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
        )
        self.block6 = BasicBlock(16, 32, stride=2, downsample=downsample)
        self.block7 = BasicBlock(32, 32)
        self.block8 = BasicBlock(32, 32)
        self.block9 = BasicBlock(32, 32)
        self.block10 = BasicBlock(32, 32)

        # Early Exit2
        self.early_exit_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.early_exit_bn2 = nn.BatchNorm2d(64)
        self.early_exit_pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.early_exit_fc2 = nn.Linear(64, num_classes)

        # 手动实现 layer3
        downsample = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
        )
        self.block11 = BasicBlock(32, 64, stride=2, downsample=downsample)
        self.block12 = BasicBlock(64, 64)
        self.block13 = BasicBlock(64, 64)
        self.block14 = BasicBlock(64, 64)
        self.block15 = BasicBlock(64, 64)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        ee_outs = []

        # Initial convolution and BatchNorm

        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        # Early Exit2
        ee_out2 = self.early_exit_conv2(x)
        ee_out2 = self.early_exit_bn2(ee_out2)
        ee_out2 = F.relu(ee_out2)
        ee_out2 = self.early_exit_pool2(ee_out2)
        ee_out2 = torch.flatten(ee_out2, 1)
        ee_out2 = self.early_exit_fc2(ee_out2)
        ee_outs.append(ee_out2)

        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)


        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        ee_outs.append(x)

        return 0, ee_outs
    

class ResNet32_split_3to4(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet32_split_3to4, self).__init__()

        self.block7 = BasicBlock(32, 32)
        self.block8 = BasicBlock(32, 32)
        self.block9 = BasicBlock(32, 32)
        self.block10 = BasicBlock(32, 32)

        # 手动实现 layer3
        downsample = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
        )
        self.block11 = BasicBlock(32, 64, stride=2, downsample=downsample)
        self.block12 = BasicBlock(64, 64)
        self.block13 = BasicBlock(64, 64)
        self.block14 = BasicBlock(64, 64)
        self.block15 = BasicBlock(64, 64)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        ee_outs = []

        # Initial convolution and BatchNorm

        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)


        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        ee_outs.append(x)

        return 0, ee_outs
    
#================early transmission===================

class ResNet32_split_05(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet32_split_05, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # 手动实现 layer1
        self.block1 = BasicBlock(16, 16)
        self.block2 = BasicBlock(16, 16)

    def forward(self, x):
        ee_outs = []

        # Initial convolution and BatchNorm
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.block1(x)
        x = self.block2(x)

        return x, ee_outs
    
class ResNet32_split_06to2(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet32_split_06to2, self).__init__()

        self.block3 = BasicBlock(16, 16)
        self.block4 = BasicBlock(16, 16)
        self.block5 = BasicBlock(16, 16)

        # Early Exit1
        self.early_exit_conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.early_exit_bn1 = nn.BatchNorm2d(32)
        self.early_exit_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.early_exit_fc = nn.Linear(32, num_classes)

        # 手动实现 layer2
        downsample = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
        )
        self.block6 = BasicBlock(16, 32, stride=2, downsample=downsample)

        # Early Exit2
        self.early_exit_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.early_exit_bn2 = nn.BatchNorm2d(64)
        self.early_exit_pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.early_exit_fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        ee_outs = []

        # Initial convolution and BatchNorm
        x = self.block3(x)

        # Early Exit
        ee_out = self.early_exit_conv1(x)
        ee_out = self.early_exit_bn1(ee_out)
        ee_out = F.relu(ee_out)
        ee_out = self.early_exit_pool(ee_out)
        ee_out = torch.flatten(ee_out, 1)
        ee_out = self.early_exit_fc(ee_out)
        ee_outs.append(ee_out)

        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        # Early Exit2
        ee_out2 = self.early_exit_conv2(x)
        ee_out2 = self.early_exit_bn2(ee_out2)
        ee_out2 = F.relu(ee_out2)
        ee_out2 = self.early_exit_pool2(ee_out2)
        ee_out2 = torch.flatten(ee_out2, 1)
        ee_out2 = self.early_exit_fc2(ee_out2)
        ee_outs.append(ee_out2)

        return 0, ee_outs
    
class ResNet32_split_06to4(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet32_split_06to4, self).__init__()
        
        self.block3 = BasicBlock(16, 16)
        self.block4 = BasicBlock(16, 16)
        self.block5 = BasicBlock(16, 16)

        # Early Exit1
        self.early_exit_conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.early_exit_bn1 = nn.BatchNorm2d(32)
        self.early_exit_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.early_exit_fc = nn.Linear(32, num_classes)

        # 手动实现 layer2
        downsample = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
        )
        self.block6 = BasicBlock(16, 32, stride=2, downsample=downsample)
        self.block7 = BasicBlock(32, 32)
        self.block8 = BasicBlock(32, 32)
        self.block9 = BasicBlock(32, 32)
        self.block10 = BasicBlock(32, 32)

        # Early Exit2
        self.early_exit_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.early_exit_bn2 = nn.BatchNorm2d(64)
        self.early_exit_pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.early_exit_fc2 = nn.Linear(64, num_classes)

        # 手动实现 layer3
        downsample = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
        )
        self.block11 = BasicBlock(32, 64, stride=2, downsample=downsample)
        self.block12 = BasicBlock(64, 64)
        self.block13 = BasicBlock(64, 64)
        self.block14 = BasicBlock(64, 64)
        self.block15 = BasicBlock(64, 64)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        ee_outs = []

        # Initial convolution and BatchNorm
        x = self.block3(x)

        # Early Exit
        ee_out = self.early_exit_conv1(x)
        ee_out = self.early_exit_bn1(ee_out)
        ee_out = F.relu(ee_out)
        ee_out = self.early_exit_pool(ee_out)
        ee_out = torch.flatten(ee_out, 1)
        ee_out = self.early_exit_fc(ee_out)
        ee_outs.append(ee_out)

        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        # Early Exit2
        ee_out2 = self.early_exit_conv2(x)
        ee_out2 = self.early_exit_bn2(ee_out2)
        ee_out2 = F.relu(ee_out2)
        ee_out2 = self.early_exit_pool2(ee_out2)
        ee_out2 = torch.flatten(ee_out2, 1)
        ee_out2 = self.early_exit_fc2(ee_out2)
        ee_outs.append(ee_out2)

        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)


        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        ee_outs.append(x)

        return 0, ee_outs
    
class ResNet32_split_025(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet32_split_025, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # 手动实现 layer1
        self.block1 = BasicBlock(16, 16)

    def forward(self, x):
        ee_outs = []

        # Initial convolution and BatchNorm
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.block1(x)

        return x, ee_outs
    
class ResNet32_split_026to2(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet32_split_026to2, self).__init__()

        self.block2 = BasicBlock(16, 16)
        self.block3 = BasicBlock(16, 16)
        self.block4 = BasicBlock(16, 16)
        self.block5 = BasicBlock(16, 16)

        # Early Exit1
        self.early_exit_conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.early_exit_bn1 = nn.BatchNorm2d(32)
        self.early_exit_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.early_exit_fc = nn.Linear(32, num_classes)

        # 手动实现 layer2
        downsample = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
        )
        self.block6 = BasicBlock(16, 32, stride=2, downsample=downsample)

        # Early Exit2
        self.early_exit_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.early_exit_bn2 = nn.BatchNorm2d(64)
        self.early_exit_pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.early_exit_fc2 = nn.Linear(64, num_classes)


    def forward(self, x):
        ee_outs = []

        # Initial convolution and BatchNorm
        x = self.block2(x)
        x = self.block3(x)

        # Early Exit
        ee_out = self.early_exit_conv1(x)
        ee_out = self.early_exit_bn1(ee_out)
        ee_out = F.relu(ee_out)
        ee_out = self.early_exit_pool(ee_out)
        ee_out = torch.flatten(ee_out, 1)
        ee_out = self.early_exit_fc(ee_out)
        ee_outs.append(ee_out)

        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        # Early Exit2
        ee_out2 = self.early_exit_conv2(x)
        ee_out2 = self.early_exit_bn2(ee_out2)
        ee_out2 = F.relu(ee_out2)
        ee_out2 = self.early_exit_pool2(ee_out2)
        ee_out2 = torch.flatten(ee_out2, 1)
        ee_out2 = self.early_exit_fc2(ee_out2)
        ee_outs.append(ee_out2)

        return 0, ee_outs
    
class ResNet32_split_026to4(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet32_split_026to4, self).__init__()
        
        self.block2 = BasicBlock(16, 16)
        self.block3 = BasicBlock(16, 16)
        self.block4 = BasicBlock(16, 16)
        self.block5 = BasicBlock(16, 16)

        # Early Exit1
        self.early_exit_conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.early_exit_bn1 = nn.BatchNorm2d(32)
        self.early_exit_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.early_exit_fc = nn.Linear(32, num_classes)

        # 手动实现 layer2
        downsample = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
        )
        self.block6 = BasicBlock(16, 32, stride=2, downsample=downsample)
        self.block7 = BasicBlock(32, 32)
        self.block8 = BasicBlock(32, 32)
        self.block9 = BasicBlock(32, 32)
        self.block10 = BasicBlock(32, 32)

        # Early Exit2
        self.early_exit_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.early_exit_bn2 = nn.BatchNorm2d(64)
        self.early_exit_pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.early_exit_fc2 = nn.Linear(64, num_classes)

        # 手动实现 layer3
        downsample = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
        )
        self.block11 = BasicBlock(32, 64, stride=2, downsample=downsample)
        self.block12 = BasicBlock(64, 64)
        self.block13 = BasicBlock(64, 64)
        self.block14 = BasicBlock(64, 64)
        self.block15 = BasicBlock(64, 64)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        ee_outs = []

        # Initial convolution and BatchNorm
        
        x = self.block2(x)
        x = self.block3(x)

        # Early Exit
        ee_out = self.early_exit_conv1(x)
        ee_out = self.early_exit_bn1(ee_out)
        ee_out = F.relu(ee_out)
        ee_out = self.early_exit_pool(ee_out)
        ee_out = torch.flatten(ee_out, 1)
        ee_out = self.early_exit_fc(ee_out)
        ee_outs.append(ee_out)

        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        # Early Exit2
        ee_out2 = self.early_exit_conv2(x)
        ee_out2 = self.early_exit_bn2(ee_out2)
        ee_out2 = F.relu(ee_out2)
        ee_out2 = self.early_exit_pool2(ee_out2)
        ee_out2 = torch.flatten(ee_out2, 1)
        ee_out2 = self.early_exit_fc2(ee_out2)
        ee_outs.append(ee_out2)

        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)


        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        ee_outs.append(x)

        return 0, ee_outs