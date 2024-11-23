# 訓練腳本模組
# 定義訓練函數，負責模型訓練

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.cnn_model import CNN5d
from utils.data_utils import CustomImageDataset

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_model(model, train_loader, test_loader, num_epochs=30, save_path="trained_model.pth"):
    """
    訓練模型並儲存參數
    :param model: 待訓練的模型
    :param train_loader: 訓練集的 DataLoader
    :param test_loader: 測試集的 DataLoader
    :param num_epochs: 訓練的總 epoch 數
    :param save_path: 儲存模型參數的路徑
    """
    model = model.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # 訓練完成後儲存模型
    torch.save(model.state_dict(), save_path)
    print(f"模型參數已儲存至 {save_path}")
