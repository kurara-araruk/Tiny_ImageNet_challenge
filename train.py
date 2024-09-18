import numpy as np
import os
import re

from sklearn.model_selection import train_test_split

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from dataset import CustomImageDataset
from utils import fit, evaluate_history, torch_seed
from models.base_models import CNN_v1, CNN_v2, CNN_v3, CNN_v4
from models.Densenet import DenseNet
from models.rdn import RdnDenseNet


########################################################################################################################


# デバイスの確認
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


########################################################################################################################


# データ読み込み

# パスを指定
train_data_root = 'data/tiny-imagenet-200/train'
test_data_root = 'data/tiny-imagenet-200/val'
submission_data = 'data/tiny-imagenet-200/test'

# transformの定義：train用
transform_train = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
])
# transformの定義：val, test, submission用
transform_test = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# クラスインデックスを作成
pattern = re.compile(r'n[0-9]{8}')
classes = sorted(filter(pattern.fullmatch, os.listdir(train_data_root)))
class_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

# データの読み込み
full_dataset = CustomImageDataset(root_dir=train_data_root, mode='full', class_idx=class_idx, transform=transform_train)
train_data, val_data = train_test_split(full_dataset.data, test_size=0.2, random_state=42)

train_set = CustomImageDataset(root_dir=train_data_root, mode='train', class_idx=class_idx, transform=transform_train, load_data=train_data)
val_set = CustomImageDataset(root_dir=train_data_root, mode='val', class_idx=class_idx, transform=transform_test, load_data=val_data)
test_set = CustomImageDataset(root_dir=test_data_root, mode='test', class_idx=class_idx, transform=transform_test)

# ミニバッチのサイズ指定
batch_size = 8

# 訓練データローダー
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


########################################################################################################################

"""
# CNN_v4 学習
torch_seed()
net = CNN_v4(n_output=200, n_hidden=128).to(device)
criterion = nn.CrossEntropyLoss()
lr=0.01
optimizer = optim.Adam(net.parameters(), lr=lr)
history = np.zeros((0, 5))
num_epochs = 10
history = fit(net, optimizer, criterion, num_epochs, train_loader, val_loader, device, history)
evaluate_history(history, "CNN_v4", "history_image")
#"""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
# Densenet の学習
torch_seed()
net = DenseNet(growth_rate=32, block_config=(6, 12, 64, 48), num_init_features=32, drop_rate=0.01).to(device)
criterion = nn.CrossEntropyLoss()
lr=0.001
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30], gamma=0.1)
history = np.zeros((0, 5))
num_epochs = 50
history = fit(net, optimizer, criterion, num_epochs, train_loader, val_loader, device, history, scheduler)
evaluate_history(history, "DenseNet", "history_image")
#"""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
# RdnDenseNet の学習
torch.cuda.empty_cache()
torch_seed()
net = RdnDenseNet(block_config=(6, 12, 24, 16), num_init_features=64, drop_rate=0.2, rdn_block_config=(8, 8, 8, 8), rdn_drop_rate=0.1).to(device)
criterion = nn.CrossEntropyLoss()
lr=0.0001
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=0.000001)
history = np.zeros((0, 5))
num_epochs = 25
history = fit(net, optimizer, criterion, num_epochs, train_loader, val_loader, device, history, scheduler)
evaluate_history(history, "RdnDenseNet", "history_image")
#"""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""     ->f.log / non transforms
torch.cuda.empty_cache()
torch_seed()
net = RdnDenseNet(
    block_config=(6, 12, 24, 32, 12, 4),
    num_init_features=64,
    drop_rate=0.2,
    rdn_scale_factor=4,
    rdn_num_init_features=64,
    rdn_block_config=(24, 4),
    rdn_drop_rate=0.2
    ).to(device)
criterion = nn.CrossEntropyLoss()
lr=0.0001
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=0.000001)
history = np.zeros((0, 5))
num_epochs = 25
history = fit(net, optimizer, criterion, num_epochs, train_loader, val_loader, device, history, scheduler)
evaluate_history(history, "RdnDenseNet", "history_image")
#"""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

""" ->g.log
torch.cuda.empty_cache()
torch_seed()
net = RdnDenseNet(
    growth_rate = 48,
    block_config=(6, 12, 24, 32, 12, 4),
    num_init_features=64,
    drop_rate=0.2,
    rdn_scale_factor=4,
    rdn_num_init_features=64,
    rdn_block_config=(24, 4),
    rdn_drop_rate=0.2
    ).to(device)
criterion = nn.CrossEntropyLoss()
lr=0.0001
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=0.000001)
history = np.zeros((0, 5))
num_epochs = 25
history = fit(net, optimizer, criterion, num_epochs, train_loader, val_loader, device, history, scheduler)
evaluate_history(history, "RdnDenseNet", "history_image")
#"""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""     ->h.log
torch.cuda.empty_cache()
torch_seed()
net = RdnDenseNet(
    growth_rate = 32,
    block_config=(2, 6, 12, 24, 32, 12),
    num_init_features=64,
    bn_size = 4,
    drop_rate=0.2,
    rdn_scale_factor=4,
    rdn_num_init_features=64,
    rdn_growth_rate = 32,
    rdn_block_config=(2, 2),
    rdn_bn_size = 4,
    rdn_drop_rate=0.2
    ).to(device)
criterion = nn.CrossEntropyLoss()
lr=0.0001
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=0.000001)
history = np.zeros((0, 5))
num_epochs = 25
history = fit(net, optimizer, criterion, num_epochs, train_loader, val_loader, device, history, scheduler)
evaluate_history(history, "RdnDenseNet", "history_image")
#"""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

""" ->i.log
torch.cuda.empty_cache()
torch_seed()
net = RdnDenseNet(
    growth_rate = 32,
    block_config=(6, 12, 24, 32, 12, 8),
    num_init_features=64,
    bn_size = 4,
    drop_rate=0.2,
    rdn_scale_factor=4,
    rdn_num_init_features=64,
    rdn_growth_rate = 32,
    rdn_block_config=(18, 18),
    rdn_bn_size = 4,
    rdn_drop_rate=0.2
    ).to(device)
criterion = nn.CrossEntropyLoss()
lr=0.0001
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=0.000001)
history = np.zeros((0, 5))
num_epochs = 15
history = fit(net, optimizer, criterion, num_epochs, train_loader, val_loader, device, history, scheduler)
evaluate_history(history, "RdnDenseNet", "history_image")
#"""

########################################################################################################################

"""
# モデルの保存
model_save_path = 'saved_models/RdnDenseNet.pth'  # 保存先のパスを指定
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
torch.save(net.state_dict(), model_save_path)
print(f"モデルを {model_save_path} に保存しました。")
#"""

########################################################################################################################

""" 追加学習  ->j.log
torch.cuda.empty_cache()
torch_seed()
net = RdnDenseNet(
    growth_rate = 32,
    block_config=(6, 12, 24, 32, 12, 8),
    num_init_features=64,
    bn_size = 4,
    drop_rate=0.2,
    rdn_scale_factor=4,
    rdn_num_init_features=64,
    rdn_growth_rate = 32,
    rdn_block_config=(18, 18),
    rdn_bn_size = 4,
    rdn_drop_rate=0.2
    ).to(device)
net.load_state_dict(torch.load('saved_models/RdnDenseNet.pth'))
criterion = nn.CrossEntropyLoss()
lr = 0.00001
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0000001)
history = np.zeros((0, 5))
num_epochs = 10
history = fit(net, optimizer, criterion, num_epochs, train_loader, val_loader, device, history, scheduler)
evaluate_history(history, "RdnDenseNet_1", "history_image")

model_save_path = 'saved_models/RdnDenseNet_1.pth'
torch.save(net.state_dict(), model_save_path)
print(f"追加学習後のモデルを {model_save_path} に保存しました。")
#"""

#""" ->transforms.Resize((512, 512))
torch.cuda.empty_cache()
torch_seed()
net = DenseNet(
    growth_rate = 32,
    block_config=(6, 12, 24, 32, 12, 8),
    num_init_features=64,
    bn_size = 4,
    drop_rate=0.2
    ).to(device)
criterion = nn.CrossEntropyLoss()
lr=0.001
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0.000001)
history = np.zeros((0, 5))
num_epochs = 30
history = fit(net, optimizer, criterion, num_epochs, train_loader, val_loader, device, history, scheduler)
evaluate_history(history, "DenseNet", "history_image")
#"""