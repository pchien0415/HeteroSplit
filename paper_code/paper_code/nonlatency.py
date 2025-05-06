from torchvision.models import resnet50, resnet18
from thop import profile
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
class ResNet20_split(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet20_split, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        
        # 手动实现 layer1
        self.block1 = BasicBlock(16, 16)
        self.block2 = BasicBlock(16, 16)
        self.block3 = BasicBlock(16, 16)

        # Early Exit Layer 1
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

    def forward(self, x):
        ee_outs = []

        # Initial convolution and BatchNorm
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.block1(x)
        x = self.block2(x)

        # Early  Exit
        ee_out = self.early_exit_pool(x)
        ee_out = torch.flatten(ee_out, 1)
        ee_out = self.early_exit_fc(ee_out)
        ee_outs.append(ee_out)

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

        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        ee_outs.append(x)

        return 0, ee_outs

def calculate_round_latency(model_flops, device_flops):
    # 計算延遲
    latency = model_flops / device_flops  # 將 G FLOPS 換算為 FLOPS
    return latency

# 模型的計算量 (M FLOPs)
model = ResNet20_split()
input = torch.randn(500, 3, 32, 32)
macs, params = profile(model, inputs=(input, ))
flops = macs / 1e6 * 2
params = params / 1e6
print(f"FLOPs: {flops:.5f} M")
print(f"Params: {params:.5f} M")
model_flops = flops * 1e6  # 將 M FLOPs 換算為 FLOPS

# 設備的計算能力 (FLOPs)
devices = {
    "Weak": 1 * 1e10,  
    "Medium": 2 * 1e10,  
    "Strong": 4 * 1e10,
    "Transsimion": 2.4 * 1e9
}

# FLOPs & 5.86547 M &  15.56480 & 25.29725 M &  47.33443 M & 92.92432 M \\
# Parameters & 0.00280 M & 0.00747 M & 0.01231 M & 0.04112 M  &  0.28459 M  \\

model_level4 = 92.92432 * 1e6 * 500
model_level2 = 47.33443 * 1e6 * 500
model_level1 = 25.29725 * 1e6 * 500
model_level05 = 15.56480 * 1e6 * 500
model_level025 = 5.86547 * 1e6 * 500

model_level05_2 = 31.76963 * 1e6 * 500
model_level05_4 = 77.35952 * 1e6 * 500
model_level025_2 = 41.46896 * 1e6 * 500
model_level025_4 = 87.05885 * 1e6 * 500

model_level1_2 = 22.03718 * 1e6 * 500
model_level1_4 = 67.62707 * 1e6 * 500

# 計算各設備的延遲
epoch = 5

# FL(large)
latency = calculate_round_latency(model_level4, devices["Weak"])
latency = latency * epoch
#print(f"FL(large) 設備的延遲: {latency:.6f} 秒")

# Propose 加上傳輸時間
latency = 0
latency += calculate_round_latency(model_level1, devices["Weak"])
latency += calculate_round_latency(model_level1_2, devices["Medium"])
latency = latency * epoch
latency_try = 0
latency_try += calculate_round_latency(model_level1, devices["Weak"])
latency_try += calculate_round_latency(model_level1_4, devices["Strong"])
latency_try = latency_try * epoch
latency = max(latency, latency_try)
#latency = latency_try # only strong and weak
latency_with_wifi4  = latency + 2.08
latency_with_6g = latency + 1.25 # (0.0625MB * 500 * 5 * 8) / 1200 Mbps (1.2Gbps)
latency_with_wifi5 = latency + 0.69
latency_with_wifi6 = latency + 0.52
print(f"Propose 設備的延遲: {latency:.6f} 秒")
print(f"Propose 設備的延遲: {latency_with_wifi4:.6f} 秒")
print(f"Propose 設備的延遲: {latency_with_6g:.6f} 秒")
print(f"Propose 設備的延遲: {latency_with_wifi5:.6f} 秒")
print(f"Propose 設備的延遲: {latency_with_wifi6:.6f} 秒")

# Depthfl
# SplitMix
# FL(small)
latency = 0
latency_weak = calculate_round_latency(model_level1, devices["Weak"])
latency_middle = calculate_round_latency(model_level2, devices["Medium"])
latency_strong = calculate_round_latency(model_level4, devices["Strong"])
latency = max(latency_weak, latency_middle, latency_strong)
latency = latency * epoch
#print(f"Others method 設備的延遲: {latency:.6f} 秒")
