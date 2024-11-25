# 模型定義模組
# 定義 CNN5d 模型，包含兩層卷積層和全連接層

import torch
import torch.nn as nn
from collections import OrderedDict

IMG_HEIGHT_SELECTED = 32
IMG_WIDTH_SELECTED = 15

class CNN5d(nn.Module):
    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)  # 修正
            m.bias.data.fill_(0.01)
    
    def __init__(self):
        super(CNN5d, self).__init__()
        self.conv1 = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(1, 64, (5, 3), padding=(2, 1), stride=(1, 1), dilation=(1, 1))),
            ('BN', nn.BatchNorm2d(64, affine=True)),
            ('ReLU', nn.ReLU()),
            ('Max-Pool', nn.MaxPool2d((2, 1)))
        ]))
        self.conv1 = self.conv1.apply(self.init_weights)
        
        self.conv2 = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(64, 128, (5, 3), padding=(2, 1), stride=(1, 1), dilation=(1, 1))),
            ('BN', nn.BatchNorm2d(128, affine=True)),
            ('ReLU', nn.ReLU()),
            ('Max-Pool', nn.MaxPool2d((2, 1)))
        ]))
        self.conv2 = self.conv2.apply(self.init_weights)
        
        # 計算攤平大小
        dummy_input = torch.zeros(1, 1, IMG_HEIGHT_SELECTED, IMG_WIDTH_SELECTED)
        flattened_size = self.conv2(self.conv1(dummy_input)).view(1, -1).shape[1]

        self.DropOut = nn.Dropout(p=0.5)
        self.FC = nn.Linear(flattened_size, 2)
        self.FC.apply(self.init_weights)

    def forward(self, x): 
        # 輸入數據應為 [N, 32, 15]
        if x.ndim == 4:  # 若有多餘維度，去掉
            x = x.squeeze(1)
        x = x.unsqueeze(1).to(torch.float32)  # 增加通道維度，變為 [N, 1, 32, 15]
        x = self.conv1(x)  # 通過卷積層
        x = self.conv2(x)  # 通過第二層卷積
        x = self.DropOut(x.view(x.shape[0], -1))  # 攤平
        x = self.FC(x)  # 全連接層
        return x