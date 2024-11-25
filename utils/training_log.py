import torch
def _print_training_log(
    epoch, 
    train_loss, 
    test_loss, 
    device, 
    early_stopping_counter
):
    """
    PRINT LOG
    """
    print(f'GPU mem used: {torch.cuda.memory_allocated(device) / 1024**2:.1f}MB')
    print(f'Epoch [{epoch+1}], '
          f'Train Loss: {train_loss:.4f}, '
          f'Test Loss: {test_loss:.4f}, '
          f'Early Stopping Counter: {early_stopping_counter}\n\n\n')
