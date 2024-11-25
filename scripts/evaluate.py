# 評估腳本模組
# 負責模型的評估與結果可視化

import torch

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc,accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
def _plot_probability_distribution(probs, labels):

    plt.figure(figsize=(10, 6))
    

    plt.boxplot(
        [probs[labels == 0][:, 1], probs[labels == 1][:, 1]], 
        labels=['Class 0', 'Class 1']
    )
    
    plt.title('distribute of category prob')
    plt.ylabel('prob of predict as "1"')
    plt.show()

def _plot_confusion_matrix(all_labels, all_preds):
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0', 'Class 1'], 
                yticklabels=['Class 0', 'Class 1'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

def _plot_roc_curve(labels, probs):

    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def evaluate_model(model, test_loader, device, plot_roc=True):

    model = model.to(device)
    model.eval()
    
    # 儲存結果
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())


    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)


    _plot_confusion_matrix(all_labels, all_preds)


    print("\n分類結果:")
    print(classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1']))


    print("\n機率分布統計:")
    for cls in [0, 1]:
        cls_probs = all_probs[all_labels == cls]
        print(f"\nClass {cls} 的機率分布:")
        print(f"平均機率: {cls_probs.mean(axis=0)}")
        print(f"機率標準差: {cls_probs.std(axis=0)}")

    # Visualize prob 葉主任說可以不用
    # _plot_probability_distribution(all_probs, all_labels)

    # ROC CURVE
    if plot_roc:
        _plot_roc_curve(all_labels, all_probs[:, 1])


    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\n模型ACC : {accuracy:.4f}")
    
    return accuracy
