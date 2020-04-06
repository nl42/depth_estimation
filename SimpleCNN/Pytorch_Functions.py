import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import trange, tqdm

def string_of_selected_parameters(model, acc=False, lr=False, wd=False, 
                                  opt=False, loss=False, sep='', equal='='):
    return  f"{ f'accuracy{equal}{model.accuracy:0.2f}{sep}' if acc else ''}" \
            f"{ f'learning_rate{equal}{model.learning_rate}{sep}' if lr else ''}" \
            f"{ f'weight_decay{equal}{model.weight_decay}{sep}' if wd else ''}" \
            f"{ f'optimiser{equal}{model.optimiser}{sep}' if opt else ''}" \
            f"{ f'loss method{equal}{model.loss_method}{sep}' if loss else ''}" \

def save_models(models, path, *args, **kwargs):
    for model in models:
        name = string_of_selected_parameters(model, equal='-', sep='_', *args, **kwargs)
        torch.save(model, f"{path}/{name}.pth")

def load_models(path):
    directory = Path(path)
    
    return [torch.load(file) for file in sorted(directory.iterdir()) if file.suffix == '.pth']

def get_stats(dataloader):
    count, rolling_sum, rolling_sum_squared = torch.zeros(1), torch.zeros(3), torch.zeros(3)
    
    for image,_ in tqdm(dataloader):
        count += image.shape[0]*image.shape[2]*image.shape[3]
        rolling_sum += image.sum(dim=(0,2,3))
        rolling_sum_squared += (image*image).sum(dim=(0,2,3))

    μ = rolling_sum / count
    σ = (rolling_sum_squared / count - μ**2).sqrt()
    return μ, σ