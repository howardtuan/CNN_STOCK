from utils.config import Config
def _train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch+1}/{Config.NUM_EPOCHS}], '
                  f'Batch [{batch_idx}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}')

    return running_loss / len(train_loader)