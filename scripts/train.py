# 訓練腳本模組
# 定義訓練函數，負責模型訓練

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.cnn_model import CNN5d
from utils.data_utils import CustomImageDataset

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_model(model, train_loader, test_loader, num_epochs=30):
    """
    訓練模型並儲存參數
    :param model: 待訓練的模型
    :param train_loader: 訓練集的 DataLoader
    :param test_loader: 測試集的 DataLoader
    :param num_epochs: 訓練的總 epoch 數
    :param save_path: 儲存模型參數的路徑
    """
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    torch.backends.cudnn.benchmark = True
    
    train_losses, test_losses = [], []
    best_test_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                test_loss += criterion(outputs, labels).item()
        
        train_loss = running_loss / len(train_loader)
        test_loss /= len(test_loader)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        print(f'GPU 記憶體使用: {torch.cuda.memory_allocated(DEVICE) / 1024**2:.1f}MB')
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        scheduler.step(test_loss)

        # 訓練完成後儲存模型(loss最小的模型)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss,
            }, 'best_model.pth')
            print("已儲存目前loss做最小的模型參數")
    return train_losses, test_losses
