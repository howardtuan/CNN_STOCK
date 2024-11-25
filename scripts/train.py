# 訓練腳本模組
# 定義訓練函數，負責模型訓練

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.cnn_model import CNN5d
from utilsS.data_utils import CustomImageDataset
from sklearn.metrics import accuracy_score
from utilsS.config import Config
from utilsS.train_epoch import _train_epoch
from utilsS.validate_epoch import _validate_epoch
from utilsS.early_stopping import _early_stopping_check
from utilsS.training_log import _print_training_log
from utilsS.model_save import save_model
import os

# Modify the train_model function to use the new save_model function
def train_model(model, train_loader, test_loader, device):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    torch.backends.cudnn.benchmark = True

    train_losses, test_losses = [], []
    # best_test_loss = float('inf')

    early_stopping_counter = 0
    best_loss = float('inf')

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        test_loss = _validate_epoch(model, test_loader, criterion, device)
        scheduler.step(test_loss)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        should_stop = _early_stopping_check(
            test_loss,
            best_loss,
            early_stopping_counter
        )

        # TRAINING LOG
        _print_training_log(
            epoch,
            train_loss,
            test_loss,
            device,
            early_stopping_counter
        )

        if should_stop:
            print(f"Early stopping：{early_stopping_counter} epochs 未改變")
            break

        if test_loss < best_loss - Config.MIN_DELTA:
            best_loss = test_loss
            early_stopping_counter = 0
            # Use the new save_model function
            save_model(model, optimizer, epoch, test_loss, 'best_model.pth')
        else:
            early_stopping_counter += 1

    return train_losses, test_losses