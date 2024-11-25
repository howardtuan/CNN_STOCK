# 主程序模組
# 負責整合所有模組，完成數據加載、模型訓練與評估
import sys
sys.dont_write_bytecode = True

import os

# 將專案根目錄添加到系統路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)
import torch
import torch.nn as nn
from utilsS.model_loader import load_model
from utilsS.config import Config
from utilsS.setup_gpu import setup_gpu

from models.cnn_model import CNN5d

from utilsS.data_utils import CustomImageDataset
from scripts.train import train_model
from scripts.evaluate import evaluate_model
from torch.utils.data import DataLoader, random_split

def main():
    try:
        device = setup_gpu()
        print(f"使用設備: {device}")
        torch.cuda.empty_cache()

        # DATA dir
        img_dir = 'C:\\Users\\user\\Desktop\\Leet\\finance\\data\\pred_5_to_5\\5_5_clear\\Image1'
        labels_file = 'C:\\Users\\user\\Desktop\\Leet\\finance\\data\\pred_5_to_5\\5_5_clear\\X_5_5_clear.txt'

        print("Data loading...")
        dataset = CustomImageDataset(img_dir=img_dir, labels_file=labels_file)

        # data split
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size]) if Config.USE_RANDOM_SPLIT else (dataset, dataset)

        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE,
                                  shuffle=True, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE,
                                 shuffle=False, num_workers=0, pin_memory=True)

        print("初始化模型...")
        model = CNN5d()

        # Check if best_model.pth exists before trying to load
        import os
        if os.path.exists('best_model.pth'):
            try:
                # best model loading with new load_model function
                model, optimizer_state, epoch, loss = load_model(model, 'best_model.pth')
                print(f"Loaded model from epoch {epoch} with loss {loss}")
            except Exception as load_error:
                print(f"Error loading model: {load_error}. Proceeding with initial model.")
        else:
            print("No saved model found. Training from scratch.")

        if Config.USE_DATAPARALLEL and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        # If no model was successfully loaded, train the model
        if not hasattr(model, 'trained'):
            print("Training model...")
            train_model(model, train_loader, test_loader, device)

        print("評估模型...")
        
        evaluate_model(model, test_loader, device)

    except Exception as e:
        print(f"錯誤: {str(e)}")
        raise
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
