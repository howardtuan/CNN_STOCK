import torch
import torch.nn as nn

def load_model(model, filename='best_model.pth'):
    """
    Load the model with robust key mapping

    Args:
        model (nn.Module): The PyTorch model to load state dict into
        filename (str, optional): Filename to load the model from. Defaults to 'best_model.pth'.

    Returns:
        tuple: Loaded model, optimizer state dict, epoch, loss
    """
    # Ensure we load into the base model if it's wrapped in DataParallel
    base_model = model.module if isinstance(model, nn.DataParallel) else model

    # Load the checkpoint
    checkpoint = torch.load(filename)

    # Create a mapping for state dict keys
    state_dict_mapping = {
        'conv1.Conv.weight': 'conv1.0.weight',
        'conv1.Conv.bias': 'conv1.0.bias',
        'conv1.BN.weight': 'conv1.1.weight',
        'conv1.BN.bias': 'conv1.1.bias',
        'conv1.BN.running_mean': 'conv1.1.running_mean',
        'conv1.BN.running_var': 'conv1.1.running_var',
        
        'conv2.Conv.weight': 'conv2.0.weight',
        'conv2.Conv.bias': 'conv2.0.bias',
        'conv2.BN.weight': 'conv2.1.weight',
        'conv2.BN.bias': 'conv2.1.bias',
        'conv2.BN.running_mean': 'conv2.1.running_mean',
        'conv2.BN.running_var': 'conv2.1.running_var',
        
        'FC.weight': 'FC.weight',
        'FC.bias': 'FC.bias'
    }

    # Prepare the state dict for loading
    model_state_dict = checkpoint['model_state_dict']
    new_state_dict = {}

    for saved_key, model_key in state_dict_mapping.items():
        if saved_key in model_state_dict:
            new_state_dict[model_key] = model_state_dict[saved_key]

    # Load the state dict with strict=False to allow some flexibility
    base_model.load_state_dict(new_state_dict, strict=False)

    return (
        base_model, 
        checkpoint.get('optimizer_state_dict'), 
        checkpoint.get('epoch'), 
        checkpoint.get('loss')
    )
