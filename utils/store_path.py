"""Configuration file for defining paths to data."""
import os
import platform

def make_if_not_exist(p):
    if not os.path.exists(p):
        os.makedirs(p)

if platform.system() == "Windows":
    hostname = os.getenv('COMPUTERNAME')  # Windows 系統
else:
    hostname = os.uname()[1]  # Unix-like 系統

# Update your paths here.
CHECKPOINT_ROOT = './checkpoint'
data_root = './data'
make_if_not_exist(data_root)
make_if_not_exist(CHECKPOINT_ROOT)
DATA_PATHS = {
    "Digits": data_root + "/Digits",
    "DomainNet": data_root + "/DomainNet",
    "DomainNetPathList": data_root + "/DomainNet/domainnet10/",  # store the path list file from FedBN
    "Cifar10": data_root,
}
