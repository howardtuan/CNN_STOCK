import torch
def _validate_epoch(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()

    return test_loss / len(test_loader)