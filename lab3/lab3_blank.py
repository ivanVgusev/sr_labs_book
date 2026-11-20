# Import of modules
import os
import sys

sys.path.append(os.path.realpath('..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import download_dataset, concatenate, extract_dataset, part_extract, download_protocol, split_musan
from exercises_blank import train_dataset_loader, test_dataset_loader, ResNet, MainModel, train_network, test_network
from ResNetBlocks import BasicBlock
from LossFunction import AAMSoftmaxLoss
from Optimizer import SGDOptimizer
from Scheduler import OneCycleLRScheduler
from load_save_pth import saveParameters, loadParameters


# Use the locally extracted VoxCeleb1 dev/test sets and metadata
from common.voxceleb_paths import (
    VOX1_DEV_WAV,
    VOX1_META,
    VOX1_TEST_WAV,
    MUSAN_PATH,
    RIR_PATH,
    require_voxceleb,
)

require_voxceleb(VOX1_DEV_WAV, VOX1_TEST_WAV, VOX1_META)
print(f"VoxCeleb1 dev: {VOX1_DEV_WAV}")
print(f"VoxCeleb1 test: {VOX1_TEST_WAV}")
print(f"VoxCeleb1 metadata: {VOX1_META}")

# Select hyperparameters

# Acoustic features
n_mels            = 40                                   # number of mel filters in bank filters
log_input         = True                                 # logarithm of features by level

# Neural network archtecture
layers            = [3, 4, 6, 3]                         # number of ResNet blocks in different level of frame level
activation        = nn.ReLU                              # activation function used in ResNet blocks
num_filters       = [32, 64, 128, 256]                   # number of filters of ResNet blocks in different level of frame level
encoder_type      = 'SP'                                 # type of statistic pooling layer ('SP'  – classical statistic pooling 
                                                         # layer and 'ASP' – attentive statistic pooling)
nOut              = 1211                                  # embedding size

# Loss function for angular losses
margin            = 0.35                                 # margin parameter
scale             = 32.0                                 # scale parameter

# Train dataloader
max_frames_train  = 200                                  # number of frame to train
train_path        = VOX1_DEV_WAV
batch_size_train  = 128                                  # batch size to train
pin_memory        = False                                # pin memory
num_workers_train = 4                                    # number of workers to train
shuffle           = True                                 # shuffling of training examples

# Validation dataloader
max_frames_val    = 1000                                 # number of frame to validate
val_path          = VOX1_DEV_WAV
batch_size_val    = 128                                  # batch size to validate
num_workers_val   = 4                                    # number of workers to validate

# Test dataloader
max_frames_test   = 1000                                 # number of frame to test
test_path         = VOX1_DEV_WAV
batch_size_test   = 128                                  # batch size to test
num_workers_test  = 4                                    # number of workers to test

# Optimizer
lr                = 2.5                                  # learning rate value
weight_decay      = 0                                    # weight decay value

# Scheduler
val_interval      = 5                                    # frequency of validation step
max_epoch         = 40                                   # number of epoches

# Augmentation
musan_path        = MUSAN_PATH
rir_path          = RIR_PATH


# Generate data lists
train_list = []
val_list   = []
test_list  = []

with open(VOX1_META / 'iden_split.txt', 'r') as f:
    lines = f.readlines()
    
black_list = os.listdir(VOX1_TEST_WAV)   # exclude speaker IDs from VoxCeleb1 test set
num_train_spk = []                                      # number of train speakers

for line in lines:
    line   = line.strip().split(' ')
    spk_id = line[1].split('/')[0]
    
    if not (spk_id in black_list):
        num_train_spk.append(spk_id)
        
    else:
        continue
    
    # Train list
    if (line[0] == '1'):
        train_list.append(' '.join([spk_id, line[1]]))
    
    # Validation list
    elif (line[0] == '2'):
        val_list.append(' '.join([spk_id, line[1]]))
    
    # Test list
    elif (line[0] == '3'):
        test_list.append(' '.join([spk_id, line[1]]))
        
num_train_spk = len(set(num_train_spk))


# Initialize model
model      = ResNet(BasicBlock, layers=layers, activation=activation, num_filters=num_filters, nOut=nOut, encoder_type=encoder_type, n_mels=n_mels, log_input=log_input)
trainfunc  = AAMSoftmaxLoss(nOut=nOut, nClasses=num_train_spk, margin=margin, scale=scale)
main_model = MainModel(model, trainfunc).cuda()


# Initialize train dataloader (without augmentation)
train_dataset = train_dataset_loader(train_list=train_list, max_frames=max_frames_train, train_path=train_path)
train_loader  = DataLoader(train_dataset, batch_size=batch_size_train, pin_memory=pin_memory, num_workers=num_workers_train, shuffle=shuffle)

# Initialize validation dataloader
val_dataset = test_dataset_loader(test_list=val_list, max_frames=max_frames_val, test_path=val_path)
val_loader  = DataLoader(val_dataset, batch_size=batch_size_val, num_workers=num_workers_val)


# Initialize optimizer and scheduler
optimizer = SGDOptimizer(main_model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = OneCycleLRScheduler(optimizer, 
                                pct_start=0.30, 
                                cycle_momentum=False, 
                                max_lr=lr, 
                                div_factor=20, 
                                final_div_factor=10000, 
                                total_steps=max_epoch*len(train_loader))


# Monkey patch
import numpy as np

np.float = float    
np.int = int
np.object = object
np.bool = bool


start_epoch = 0
checkpoint_flag = False

if checkpoint_flag:
    start_epoch = loadParameters(main_model, optimizer, scheduler, path='../data/lab3_models/lab3_model_0004.pth')
    start_epoch = start_epoch + 1

# Train model
for num_epoch in tqdm(range(start_epoch, max_epoch)):
    train_loss, train_top1 = train_network(train_loader, main_model, optimizer, scheduler, num_epoch)
    
    print("Epoch {:1.0f}, Loss (train set) {:f}, Accuracy (train set) {:2.3f}%".format(num_epoch, train_loss, train_top1))

    # Валидация
    if (num_epoch + 1) % val_interval == 0:
        _, val_top1 = test_network(val_loader, main_model)
        print(f"Validation accuracy: {val_top1}")

    # Сохранение после каждой эпохи
    saveParameters(
        main_model, optimizer, scheduler,
        num_epoch, path='../data/lab3_models'
    )


# Initialize model
model      = ResNet(BasicBlock, layers=layers, activation=activation, num_filters=num_filters, nOut=nOut, encoder_type=encoder_type, n_mels=n_mels, log_input=log_input)
trainfunc  = AAMSoftmaxLoss(nOut=nOut, nClasses=num_train_spk, margin=margin, scale=scale)
main_model = MainModel(model, trainfunc).cuda()


# Initialize train dataloader (with augmentation)
train_dataset = train_dataset_loader(train_list=train_list, 
                                     max_frames=max_frames_train, 
                                     train_path=train_path, 
                                     augment=True, 
                                     musan_path=musan_path, 
                                     rir_path=rir_path)

train_loader  = DataLoader(train_dataset, batch_size=batch_size_train, pin_memory=pin_memory, num_workers=num_workers_train, shuffle=shuffle)

# Initialize validation dataloader
val_dataset = test_dataset_loader(test_list=val_list, max_frames=max_frames_val, test_path=val_path)
val_loader  = DataLoader(val_dataset, batch_size=batch_size_val, num_workers=num_workers_val)


# Initialize optimizer and scheduler
optimizer = SGDOptimizer(main_model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = OneCycleLRScheduler(optimizer, 
                                pct_start=0.30, 
                                cycle_momentum=False, 
                                max_lr=lr, 
                                div_factor=20, 
                                final_div_factor=10000, 
                                total_steps=max_epoch*len(train_loader))


start_epoch = 0
checkpoint_flag = False

if checkpoint_flag:
    start_epoch = loadParameters(main_model, optimizer, scheduler, path='../data/lab3_models_aug/lab3_model_0004.pth')
    start_epoch = start_epoch + 1

# Train model
for num_epoch in tqdm(range(start_epoch, max_epoch)):
    train_loss, train_top1 = train_network(train_loader, main_model, optimizer, scheduler, num_epoch)
    
    print("Epoch {:1.0f}, Loss (train set) {:f}, Accuracy (train set) {:2.3f}%".format(num_epoch, train_loss, train_top1))

    # Валидация
    if (num_epoch + 1) % val_interval == 0:
        _, val_top1 = test_network(val_loader, main_model)
        print(f"Validation accuracy: {val_top1}")

    # Сохранение после каждой эпохи
    saveParameters(
        main_model, optimizer, scheduler,
        num_epoch, path='../data/lab3_models'
    )


# Initialize test dataloader
test_dataset = test_dataset_loader(test_list=test_list, max_frames=max_frames_test, test_path=test_path)
test_loader = DataLoader(test_dataset, batch_size=batch_size_test, num_workers=num_workers_test)


# Load model without augmentation
num_epoch = loadParameters(main_model, optimizer, scheduler, path='../data/lab3_models/lab3_model_0039.pth')

# Test model
_, test_top1 = test_network(test_loader, main_model)

print("Epoch {:1.0f}, Accuracy (test set) {:2.3f}%".format(num_epoch, test_top1))


# Load model with augmentation
num_epoch = loadParameters(main_model, optimizer, scheduler, path='../data/lab3_models_aug/lab3_model_0039.pth')

# Test model
_, test_top1 = test_network(test_loader, main_model)

print("Epoch {:1.0f}, Accuracy (test set) {:2.3f}%".format(num_epoch, test_top1))
