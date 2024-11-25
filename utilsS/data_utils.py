# 數據處理模組
# 定義自定義數據集類，負責讀取和處理圖片數據

import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from utilsS.config import Config

# DATA PREPARE
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, labels_file, transform=None):
        self.img_dir = img_dir
        self.labels = pd.read_csv(labels_file,sep="\t")
        self.transform = transform or transforms.Compose([
            transforms.Resize((Config.IMG_HEIGHT[5], Config.IMG_WIDTH[5])),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self._get_valid_image_path(idx)
        image = self._load_and_transform_image(img_path)
        label = int(self.labels.iloc[idx, 1])
        return image, label

    def _get_valid_image_path(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        if not os.path.exists(img_path):
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                test_path = img_path + ext
                if os.path.exists(test_path):
                    return test_path
        return img_path

    def _load_and_transform_image(self, img_path):
        try:
            image = Image.open(img_path).convert('1')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('1', (Config.IMG_WIDTH[5], Config.IMG_HEIGHT[5]), 0)

        try:
            return self.transform(image)
        except Exception as e:
            print(f"Error transforming image {img_path}: {e}")
            blank = Image.new('1', (Config.IMG_WIDTH[5], Config.IMG_HEIGHT[5]), 0)
            return self.transform(blank)