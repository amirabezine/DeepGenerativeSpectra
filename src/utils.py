# utils.py

import torch
from torch.utils.data import DataLoader, random_split
import os
import yaml

def get_dataloaders(dataset, batch_size, num_workers, split_ratios):
    lengths = [int(len(dataset) * ratio) for ratio in split_ratios]
    lengths[-1] = len(dataset) - sum(lengths[:-1])  # Adjust the last split
    train_dataset, val_dataset, test_dataset = random_split(dataset, lengths)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader


def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_config():
    config_path = os.path.join(get_project_root(), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def resolve_path(path):
    return os.path.join(get_project_root(), path)