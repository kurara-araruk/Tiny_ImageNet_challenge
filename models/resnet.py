from typing import Any, List, Optional, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch.utils.checkpoint as cp

########################################################################################################################

# res bottleneck型 layerの定義
class _ResBottlekLayer(nn.Module):
    def __init__(
        self,
        in_out_channel: int,
        mid_channel: int,
    ):

        super().__init__()
        self.conv1 = nn.Conv2d(in_out_channel, mid_channel, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channel)
        self.conv2 = nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channel)
        self.conv3 = nn.Conv2d(mid_channel, in_out_channel, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x += identity

        return x

########################################################################################################################

# res bottleneck型 Blockの定義
class _ResBottlekBlock(nn.ModuleDict):
    def __init__(
        self,
        num_layers: int, # Reslayerを重ねる数
        in_out_channel: int,
        mid_channel: int
        ):

        super().__init__()
        for i in range(num_layers):
            layer = _ResBottlekLayer(
                in_out_channel,
                mid_channel
            )
            self.add_module(f"resbottleklayer{i + 1}", layer)

    def forward(self, x):
        for name, layer in self.items():
            x = layer(x)

        return x

########################################################################################################################

# Resnetの定義
class ResNet(nn.Module):
    def __init__(
        self,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16), # 各Denseblock内のDenselayerを重ねる数

        num_init_features: int = 64, # 最初の畳み込みで出力するチャネル数
        num_classes: int = 200,

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

        num_features = num_init_features # 最初のResblockへの入力チャネル数
        for i, num_layers in enumerate(block_config):
            block = _ResBottlekBlock(
                num_layers=num_layers, # Reslayerを重ねる数
                in_out_channel=num_features,
                mid_channel=a
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            num_features = num_features + num_layers * growth_rate # Denseblock からの出力チャネル数（transition への入力チャネル数）
            # 最後の Denseblock でない場合は、Transition を追加する。
            if i != len(block_config) - 1:
                trans = _Transition(input_channels=num_features, output_channels=num_features // 2)
                self.features.add_module(f"transition{i + 1}", trans)
                num_features = num_features // 2 # transitionからの出力チャネル数（次のDenseblockへの入力チャネル数）
