# CNN 股價預測專案
本專案基於卷積神經網絡 (CNN)，用於預測股價。專案整合了數據處理、模型訓練和評估的完整流程，適合用於金融相關的機器學習研究與應用。

## 專案架構
```bash
CNN_STOCK/
│
├── models/                        # 模型定義
│   ├── __init__.py
│   └── cnn_model.py               # CNN 模型程式碼
│
├── notebooks/                     # Jupyter Notebook
│   ├── 1120.ipynb                 # 初始版本的 Notebook
│   ├── 1121_good.ipynb            # 已修改的 Notebook
│   └── predict.ipynb              # 預測 Notebook
│
├── scripts/                       # 腳本程式
│   ├── evaluate.py                # 評估腳本
│   ├── main.py                    # 主程式
│   ├── predict.py                 # 預測腳本
│   └── train.py                   # 訓練腳本
│
├── utilsS/                         # 工具函式
│   ├── __init__.py
│   ├── config.py                  # 配置檔案
│   ├── data_utils.py              # 數據處理工具
│   ├── early_stopping.py          # 提早停止的功能
│   ├── model_loader.py            # 模型載入
│   ├── model_save.py              # 模型儲存
│   ├── setup_gpu.py               # GPU 設定
│   ├── train_epoch.py             # 訓練單一 epoch
│   ├── training_log.py            # 訓練日誌記錄
│   └── validate_epoch.py          # 驗證單一 epoch
│
├── README.md                      # 專案說明文件
└── requirements.txt               # Python 依賴清單
```

## **環境需求**

- Python 版本：3.8 或以上
- CUDA：若使用 GPU 訓練，請確保 CUDA 已正確安裝

## **安裝與啟用**
### **步驟 1：複製專案到本地**
   ```bash
   git clone https://github.com/howardtuan/orangeapple.git 
   ```
### **步驟 2：建立及啟用虛擬環境（可選但建議使用）**
```bash
python -m venv venv
source venv/bin/activate   # Windows 系統執行：venv\Scripts\activate
```
### **步驟 3：安裝依賴**
首先，安裝專案所需的 Python 模組：
```bash
pip install -r requirements.txt
```
### **步驟 4：準備數據**
將您的數據文件放入 data/ 文件夾，並更新路徑參數（例如 main.py 中的 img_dir 和 labels_file）。
### **步驟 5：啟用專案**
執行以下命令以運行主程式：
```bash
python scripts/main.py
```
## **功能模組說明**
1. 模型定義 (models/cnn_model.py)： 
    * 定義 CNN 模型，包括卷積層、池化層和全連接層。
2. 數據處理 (utils/data_utils.py)：
    * 提供自定義數據集類別，用於讀取圖像和標籤。
3. 訓練模組 (scripts/train.py)：
    * 包括訓練邏輯，損失計算與模型更新。 
4. 評估模組 (scripts/evaluate.py)：
    * 提供性能評估（包括準確率、混淆矩陣和分類報告）。
5. 主程式 (scripts/main.py)：
    * 整合數據處理、模型訓練與評估，作為入口點。
