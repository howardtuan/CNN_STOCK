# 數據處理模組
# 定義自定義數據集類，負責讀取和處理圖片數據

import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

IMG_HEIGHT = {5: 32}
IMG_WIDTH = {5: 15}

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, labels_file, transform=None):
        # 初始化數據集
        self.img_dir = img_dir
        self.labels = pd.read_excel(labels_file)
        self.transform = transform or transforms.Compose([
            transforms.Resize((IMG_HEIGHT[5], IMG_WIDTH[5])),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 根據索引讀取圖片和標籤
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        try:
            image = Image.open(img_path).convert('1')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('1', (IMG_WIDTH[5], IMG_HEIGHT[5]), 0)
        image = self.transform(image)
        label = int(self.labels.iloc[idx, 1])
        return image, label
