from utils.config import Config

def _early_stopping_check(
    current_loss, 
    best_loss, 
    early_stopping_counter
):
    """   
    Args:
        current_loss (float): 目前val loss
        best_loss (float): 最佳val loss
        early_stopping_counter (int): 未改變cnt
    
    Returns:
        bool: early stopping (T/F)
    """
    return (
        early_stopping_counter >= Config.EARLY_STOPPING_PATIENCE
    )