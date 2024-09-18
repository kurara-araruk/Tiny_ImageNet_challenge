import re
import os
import torch
import pandas as pd
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
test_loader = DataLoader(test_set, batch_size=100, shuffle=False)

# モデルのロード
model_path = 'saved_models/RdnDenseNet.pth'  # 保存したモデルのパスを指定
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
net.load_state_dict(torch.load(model_path))
net.eval()  # 推論モードに設定

# 推論の実行
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 元のラベル名に変換
inv_class_idx = {v: k for k, v in class_idx.items()}  # インデックスからクラス名を逆引きする辞書
all_preds_labels = [inv_class_idx[pred] for pred in all_preds]

# CSVファイルに保存
output_df = pd.DataFrame({
    'Image': [os.path.basename(test_set.data[i][0]) for i in range(len(test_set))],
    'Predicted_Label': all_preds_labels
})
output_csv_path = 'predictions/RdnDenseNet_pred.csv'
output_df.to_csv(output_csv_path, index=False, header=False)
print(f"予測結果を {output_csv_path} に保存しました。")

# 精度の計算と表示
correct = sum(p == l for p, l in zip(all_preds, all_labels))
total = len(all_labels)
accuracy = correct / total
print(f"Accuracy: {accuracy:.5f}")