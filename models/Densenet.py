from typing import Any, List, Optional, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch.utils.checkpoint as cp

########################################################################################################################

# Denselayerの定義
class _DenseLayer(nn.Module):
    def __init__(
        self,
        num_input_features: int, # DenseLayerに入力される特徴マップのチャネル数
        growth_rate: int, # 各DenseLayerの出力後に追加されるチャネル数
        bn_size: int, # ボトルネック層のチャネル数を決定する倍率（通常は4倍）
        drop_rate: float # ドロップアウトの確率
        ):

        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = torch.cat(x, 1)
        x = self.conv1(self.relu(self.norm1(x)))
        x = self.conv2(self.relu(self.norm2(x)))
        if self.drop_rate > 0:
            x = self.dropout(x)
        return x

########################################################################################################################

# Denseblockの定義
class _DenseBlock(nn.ModuleDict):
    def __init__(
        self,
        num_layers: int, # Denselayerを重ねる数
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        ):

        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.add_module(f"denselayer{i + 1}", layer)

    def forward(self, x):
        x_in = [x]
        for name, layer in self.items():
            x_out = layer(x_in)
            x_in.append(x_out)

        return torch.cat(x_in, 1)

########################################################################################################################

# Transition層の定義
class _Transition(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

########################################################################################################################

# Densenetの定義
class DenseNet(nn.Module):
    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16), # 各Denseblock内のDenselayerを重ねる数
        num_init_features: int = 64, # 最初の畳み込みで出力するチャネル数
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 200 # DenseNetの出力数（分類クラス）
    ):

        super().__init__()
        # 最初の畳み込み処理. カーネルサイズが大きめに設定される.
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", nn.BatchNorm2d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        # （ Denseblock - Transition )の連結
        num_features = num_init_features # 最初のDenseblockへの入力チャネル数
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            num_features = num_features + num_layers * growth_rate # Denseblock からの出力チャネル数（transition への入力チャネル数）
            # 最後の Denseblock でない場合は、Transition を追加する。
            if i != len(block_config) - 1:
                trans = _Transition(input_channels=num_features, output_channels=num_features // 2)
                self.features.add_module(f"transition{i + 1}", trans)
                num_features = num_features // 2 # transitionからの出力チャネル数（次のDenseblockへの入力チャネル数）

        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        self.features.add_module("relu5", nn.ReLU(inplace=True))
        self.features.add_module("pool5", nn.AdaptiveAvgPool2d((1, 1)))

        self.classifier = nn.Linear(num_features, num_classes)

        # 初期化？よくわからん。
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x