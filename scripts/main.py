# 主程序模組
# 負責整合所有模組，完成數據加載、模型訓練與評估
import sys
sys.dont_write_bytecode = True

import os

# 將專案根目錄添加到系統路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)
# 匯入模組
import torch
from torch.utils.data import DataLoader, random_split
from models.cnn_model import CNN5d
from utils.data_utils import CustomImageDataset
from scripts.train import train_model
from scripts.evaluate import evaluate_model



def main():
    img_dir = 'C:\\Users\\680-9000\\Desktop\\finance\\datav3\\Image1'
    labels_file = 'C:\\Users\\680-9000\\Desktop\\finance\\datav3\\X_5_5.xlsx'
    dataset = CustomImageDataset(img_dir=img_dir, labels_file=labels_file)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print("初始化模型...")
    model = CNN5d()
    print("開始訓練...")
    train_model(model, train_loader, test_loader)
    print("評估模型...")
    accuracy, conf_matrix = evaluate_model(model, test_loader)
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
