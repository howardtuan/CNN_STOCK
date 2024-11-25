class Config:
    USE_GPU = True
    USE_RANDOM_SPLIT = False
    USE_DATAPARALLEL = True
    SEED = 42
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.00001
    BATCH_SIZE = 128
    IMG_HEIGHT = {5: 32, 20: 64, 60: 96}
    IMG_WIDTH = {5: 15, 20: 60, 60: 180}
    IMG_CHANNELS = 1
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 2  
    MIN_DELTA = 0.00001
