import torch

class Config:
    epoch              = 20
    lr                 = 5e-5
    train_size         = 0.9  # Training = 90%; Validation = 10%
    data_sample_size   = 1000 # None = full
    dataset_block_size = 512
    device             = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size         = 4