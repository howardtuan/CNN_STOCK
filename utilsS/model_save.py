import torch
import torch.nn as nn
def save_model(model, optimizer, epoch, test_loss, filename='best_model.pth'):
    """
    Save the model with consistent state_dict keys
    
    Args:
        model (nn.Module): The PyTorch model to save
        optimizer (torch.optim.Optimizer): The model's optimizer
        epoch (int): Current training epoch
        test_loss (float): Current test loss
        filename (str, optional): Filename to save the model. Defaults to 'best_model.pth'.
    """
    # Ensure we save the base model if it's wrapped in DataParallel
    if isinstance(model, nn.DataParallel):
        model = model.module

    # Create a state dictionary with consistent key names
    state_dict = {
        'epoch': epoch,
        'model_state_dict': {
            'conv1.Conv.weight': model.conv1[0].weight,
            'conv1.Conv.bias': model.conv1[0].bias,
            'conv1.BN.weight': model.conv1[1].weight,
            'conv1.BN.bias': model.conv1[1].bias,
            'conv1.BN.running_mean': model.conv1[1].running_mean,
            'conv1.BN.running_var': model.conv1[1].running_var,
            
            'conv2.Conv.weight': model.conv2[0].weight,
            'conv2.Conv.bias': model.conv2[0].bias,
            'conv2.BN.weight': model.conv2[1].weight,
            'conv2.BN.bias': model.conv2[1].bias,
            'conv2.BN.running_mean': model.conv2[1].running_mean,
            'conv2.BN.running_var': model.conv2[1].running_var,
            
            'FC.weight': model.FC.weight,
            'FC.bias': model.FC.bias
        },
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': test_loss,
    }

    torch.save(state_dict, filename)