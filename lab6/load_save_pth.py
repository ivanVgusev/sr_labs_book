# Load and save pth files


# Import of modules
import os

import torch


def saveParameters(model, optimizer, scheduler, num_epoch, path, filename=None):
    # Save parameters
    
    checkpoint = {}
    checkpoint['model']     = model.state_dict()
    checkpoint['optimizer'] = optimizer.state_dict()
    checkpoint['scheduler'] = scheduler[0].state_dict()
    checkpoint['num_epoch'] = num_epoch
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    if filename is None:
        filename = ''.join(['lab6_model_', str(num_epoch).zfill(4), '.pth'])

    output_path = os.path.join(path, filename)
    temporary_path = output_path + '.tmp'
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, output_path)
    
def loadParameters(model, optimizer, scheduler, path):
    # Load parameters

    device = next(model.parameters()).device
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler[0].load_state_dict(checkpoint['scheduler'])
    
    return checkpoint['num_epoch']
