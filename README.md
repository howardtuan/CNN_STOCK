# CNN 股價預測專案
本專案基於卷積神經網絡 (CNN)，用於預測股價。專案整合了數據處理、模型訓練和評估的完整流程，適合用於金融相關的機器學習研究與應用。

## 專案架構
```bash
CNN_STOCK/
│
├── models/                        # 模型定義
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
   git clone https://github.com/howardtuan/CNN_STOCK.git 
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
### **步驟 4：準備資料集**
將您的資料集文件放入 data/ 文件夾，並更新路徑參數（例如 main.py 中的 img_dir 和 labels_file）。
### **步驟 5：啟用專案**
執行以下命令以運行主程式：
```bash
python scripts/main.py
```
## **功能模組說明**
1. models/
   * 包含所有模型相關的程式碼模組：
   * 檔案：
      * cnn_model.py：定義 CNN 模型結構。
2. notebooks/
   * 存放不同版本的 Jupyter Notebook，用於模型的開發、測試或紀錄訓練與評估結果。
   * 檔案：
      * 1120.ipynb：初始版本的 Notebook。
      * 1121_good.ipynb：已修改、優化的 Notebook。
      * predict.ipynb：專注於模型預測的 Notebook。
3. scripts/
   * 包含主要程式腳本，負責模型的訓練、評估、預測等功能。
   * 檔案：
      * train.py：模型訓練邏輯。
      * evaluate.py：模型評估邏輯，包含混淆矩陣與 ROC 曲線。
      * predict.py：用於載入模型並對新數據進行預測。
      * main.py：專案的主入口，組織訓練與測試流程。
4. utilsS/
   * 提供工具函式與輔助模組。
   * 檔案：
      * config.py：儲存專案的配置參數，例如批次大小、學習率等。
      * data_utils.py：數據處理工具，用於數據集的讀取與預處理。
      * early_stopping.py：實現提早停止訓練的功能。
      * model_loader.py：負責模型的載入與檢查點還原。
      * model_save.py：負責保存模型檢查點。
      * setup_gpu.py：自動檢測並配置 GPU。
      * train_epoch.py：執行單一訓練迭代。
      * validate_epoch.py：執行單一驗證迭代。
      * training_log.py：記錄訓練過程中的詳細日誌。
5. 根目錄檔案
   * README.md：專案的說明文件，包含專案背景、使用方法及目錄結構等資訊。
   * requirements.txt：列出專案所需的 Python 套件，用於環境的快速配置。
