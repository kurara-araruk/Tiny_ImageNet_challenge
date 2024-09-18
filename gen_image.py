import re
import numpy as np
import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import CustomImageDataset
from models.rdn import RdnDenseNet  # モデルのインポート


# デバイスの確認
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# パスを指定
test_data_root = 'data/tiny-imagenet-200/val'

# transformの定義
transform_test = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# クラスインデックスの作成
pattern = re.compile(r'n[0-9]{8}')
classes = sorted(filter(pattern.fullmatch, os.listdir('data/tiny-imagenet-200/train')))
class_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

# データセットの読み込み
test_set = CustomImageDataset(root_dir=test_data_root, mode='test', class_idx=class_idx, transform=transform_test)
test_set = torch.utils.data.Subset(test_set, list(range(5)))


# 学習済みのRdnDenseNetモデルのロード
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

# 学習済みのモデルの重みをロード
net.load_state_dict(torch.load('saved_models/RdnDenseNet.pth'))

# 推論の際にself.rdnsrのみを使う
def infer_with_rdnsr(input_data):
    net.eval()  # 推論モード
    with torch.no_grad():  # 勾配計算を無効化してメモリ効率化
        rdn_output = net.rdnsr(input_data)  # rdnsr部分のみの推論
    return rdn_output


# 推論を実行
for i in range(len(test_set)):
    img, _ = test_set[i]

    # 元の画像を逆正規化して保存
    original_img = img.squeeze(0).cpu()  # 元の画像をそのままCPUに転送
    original_img = (original_img * 0.5 + 0.5)  # [-1, 1] から [0, 1] に変換
    original_image_np = original_img.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
    original_image_np = (original_image_np * 255).astype(np.uint8)  # [0, 1] を [0, 255] にスケーリング

    # PILで元の画像を保存
    original_image_pil = Image.fromarray(original_image_np)
    original_image_pil.save(f"gen_images/original_image_{i}.png")
    print(f"元の画像を 'original_image_{i}.png' に保存しました。")

    # 推論処理
    img = img.unsqueeze(0).to(device)  # バッチ次元を追加
    rdnsr_output = infer_with_rdnsr(img)

    # 出力された画像を[0, 1]の範囲に戻すために逆正規化
    rdnsr_output = rdnsr_output.squeeze(0).cpu()  # バッチ次元を削除してCPUに転送
    rdnsr_output = (rdnsr_output * 0.5 + 0.5)  # [-1, 1] から [0, 1] に変換

    # NumPy配列に変換してチャンネルごとに分ける
    rdnsr_image = rdnsr_output.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
    rdnsr_image = (rdnsr_image * 255).astype(np.uint8)  # [0, 1] の範囲を [0, 255] にスケーリング

    # 各チャンネル（R, G, B）の画像を保存
    r_channel = rdnsr_image[:, :, 0]  # Rチャンネル
    g_channel = rdnsr_image[:, :, 1]  # Gチャンネル
    b_channel = rdnsr_image[:, :, 2]  # Bチャンネル

    # R, G, B チャンネルごとに画像を保存
    Image.fromarray(r_channel).save(f"gen_images/rdnsr_output_{i}_R.png")
    Image.fromarray(g_channel).save(f"gen_images/rdnsr_output_{i}_G.png")
    Image.fromarray(b_channel).save(f"gen_images/rdnsr_output_{i}_B.png")

    print(f"推論結果のチャンネルごとの画像を 'rdnsr_output_{i}_R.png', 'rdnsr_output_{i}_G.png', 'rdnsr_output_{i}_B.png' に保存しました。")