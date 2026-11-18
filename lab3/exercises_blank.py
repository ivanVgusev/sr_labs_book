# Exercises in order to perform laboratory work


# Import of modules
import os
import random

import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset

from common import loadWAV, AugmentWAV
from preproc import PreEmphasis


class train_dataset_loader(Dataset):
    """VoxCeleb training set with optional online augmentation."""
    
    def __init__(self, train_list, max_frames, train_path, augment=False, musan_path=None, rir_path=None):

        self.max_frames  = max_frames
        self.augment     = augment

        if self.augment:
            self.augment_wav = AugmentWAV(musan_path=musan_path, rir_path=rir_path, max_frames=max_frames)

        # Make a dictionary of ID names and ID indices
        dictkeys = list(set([x.split()[0] for x in train_list]))
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}

        # Parse the training list into file names and ID indices
        self.data_list  = []
        self.data_label = []
        
        for line in train_list:
            data = line.strip().split()

            speaker_label = dictkeys[data[0]]
            filename = os.path.join(train_path, data[1])
            
            self.data_label.append(speaker_label)
            self.data_list.append(filename)

    def __getitem__(self, index):
        
        audio = loadWAV(self.data_list[index], self.max_frames, evalmode=False)

        if self.augment:
            augtype = random.randint(0, 5)

            if augtype == 1:
                audio = self.augment_wav.reverberate(audio)
            elif augtype == 2:
                audio = self.augment_wav.additive_noise('music', audio)
            elif augtype == 3:
                audio = self.augment_wav.additive_noise('speech', audio)
            elif augtype == 4:
                audio = self.augment_wav.additive_noise('noise', audio)
            elif augtype == 5:
                audio = self.augment_wav.additive_noise('speech', audio)
                audio = self.augment_wav.additive_noise('music', audio)
            
        return torch.FloatTensor(audio), self.data_label[index]

    def __len__(self):
        
        return len(self.data_list)
    
class test_dataset_loader(Dataset):
    """VoxCeleb validation or test set."""
    
    def __init__(self, test_list, max_frames, test_path):

        self.max_frames  = max_frames

        # Make a dictionary of ID names and ID indices
        dictkeys = list(set([x.split()[0] for x in test_list]))
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}

        # Parse the training list into file names and ID indices
        self.data_list  = []
        self.data_label = []
        
        for line in test_list:
            data = line.strip().split()

            speaker_label = dictkeys[data[0]]
            filename = os.path.join(test_path, data[1])
            
            self.data_label.append(speaker_label)
            self.data_list.append(filename)

    def __getitem__(self, index):

        audio = loadWAV(self.data_list[index], self.max_frames, evalmode=True, num_eval=1)

        return torch.FloatTensor(audio), self.data_label[index]

    def __len__(self):
        
        return len(self.data_list)

class MaxoutLinear(nn.Module):
    """Maximum of two independent linear transformations."""
    
    def __init__(self, *args, **kwargs):
        
        super(MaxoutLinear, self).__init__()

        self.linear1 = nn.Linear(*args, **kwargs)
        self.linear2 = nn.Linear(*args, **kwargs)

    def forward(self, x):
        
        return torch.max(self.linear1(x), self.linear2(x))

class ResNet(nn.Module):
    """ResNet encoder that converts a waveform into a speaker embedding."""

    def __init__(self, block, layers, activation, num_filters, nOut, encoder_type='SP', n_mels=64, log_input=True, **kwargs):
        
        super(ResNet, self).__init__()

        print('Embedding size is %d, encoder %s.' % (nOut, encoder_type))

        self.inplanes     = num_filters[0]
        self.encoder_type = encoder_type
        self.n_mels       = n_mels
        self.log_input    = log_input

        self.torchfb        = torch.nn.Sequential(PreEmphasis(), 
                                                  torchaudio.transforms.MelSpectrogram(sample_rate=16000, 
                                                                                       n_fft=512, 
                                                                                       win_length=400, 
                                                                                       hop_length=160, 
                                                                                       window_fn=torch.hamming_window, 
                                                                                       n_mels=n_mels))
        self.instancenorm   = nn.InstanceNorm1d(n_mels)

        self.conv1  = nn.Conv2d(1, num_filters[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(num_filters[0])
        self.relu   = activation(inplace=True)
        
        self.layer1 = self._make_layer(block, num_filters[0], layers[0], stride=1, activation=activation)
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=2, activation=activation)
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=2, activation=activation)
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=2, activation=activation)

        outmap_size = int(self.n_mels/8)

        self.attention = nn.Sequential(nn.Conv1d(num_filters[3]*outmap_size, 128, kernel_size=1), 
                                       nn.ReLU(), 
                                       nn.BatchNorm1d(128), 
                                       nn.Conv1d(128, num_filters[3]*outmap_size, kernel_size=1), 
                                       nn.Softmax(dim=2))
        
        if self.encoder_type in {"SP", "ASP"}:
            out_dim = num_filters[3]*outmap_size*2
        else:
            raise ValueError("encoder_type must be 'SP' or 'ASP'")

        self.fc = nn.Sequential(MaxoutLinear(out_dim, nOut), nn.BatchNorm1d(nOut, affine=False))

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, activation=nn.ReLU):

        downsample = None

        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, activation=activation))
        self.inplanes = planes*block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, activation=activation))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Acoustic features: waveform -> log Mel spectrogram.
        with torch.no_grad():
            x = self.torchfb(x.float()) + 1e-6

            if self.log_input:
                x = x.log()

            x = self.instancenorm(x).unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.reshape(x.size(0), -1, x.size(-1))

        # Statistical pooling removes the time dimension.
        if self.encoder_type == "SP":
            mean = torch.mean(x, dim=2)
            std = torch.std(x, dim=2, correction=0)
        else:
            weights = self.attention(x)
            mean = torch.sum(x * weights, dim=2)
            variance = torch.sum((x ** 2) * weights, dim=2) - mean ** 2
            std = torch.sqrt(variance.clamp(min=1e-5))

        x = torch.cat((mean, std), dim=1)

        return self.fc(x)

class MainModel(nn.Module):
    """Speaker encoder together with its classification loss."""

    def __init__(self, model, trainfunc, **kwargs):
        super(MainModel, self).__init__()

        self.__S__ = model
        self.__L__ = trainfunc

    def forward(self, data, label=None):
        data = data.reshape(-1, data.size(-1))
        embeddings = self.__S__(data)

        if label is None:
            return embeddings

        return self.__L__(embeddings, label)
        
def train_network(train_loader, main_model, optimizer, scheduler, num_epoch, verbose=False):
    """Train the model for one epoch and return mean loss and accuracy."""

    assert scheduler[1] in ["epoch", "iteration"]

    main_model.train()
    device = next(main_model.parameters()).device

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for counter, (data, data_label) in enumerate(train_loader, start=1):

        data = data.to(device)
        data_label = data_label.to(device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        batch_loss, batch_accuracy = main_model(data, data_label)
        batch_loss.backward()
        optimizer.step()

        batch_size = data_label.size(0)
        total_loss += batch_loss.item() * batch_size
        total_correct += batch_accuracy.item() * batch_size / 100
        total_samples += batch_size

        if verbose:
            print(
                "Epoch {:d}, Batch {:d}, LR {:.6f}, "
                "Loss {:.4f}, Accuracy {:.3f}%".format(
                    num_epoch,
                    counter,
                    optimizer.param_groups[0]["lr"],
                    total_loss / total_samples,
                    100 * total_correct / total_samples
                )
            )

        if scheduler[1] == "iteration":
            scheduler[0].step()

    if total_samples == 0:
        raise ValueError("train_loader не содержит данных")

    if scheduler[1] == "epoch":
        scheduler[0].step()

    return (
        total_loss / total_samples,
        100 * total_correct / total_samples
    )

def test_network(test_loader, main_model):
    """Evaluate a model trained with AAM-Softmax."""

    main_model.eval()
    device = next(main_model.parameters()).device

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, data_label in test_loader:
            data = data.to(device)
            data_label = data_label.to(device, dtype=torch.long)
            batch_loss, batch_accuracy = main_model(data, data_label)
            batch_size = data_label.size(0)
            total_loss += batch_loss.item() * batch_size
            total_correct += batch_accuracy.item() * batch_size / 100
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("test_loader не содержит данных")

    return (
        total_loss / total_samples,
        100 * total_correct / total_samples
    )


def test_legacy_classifier(test_loader, main_model):
    """Evaluate the old checkpoint whose embedding layer was used as logits."""

    main_model.eval()
    device = next(main_model.parameters()).device
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, data_label in test_loader:
            data = data.to(device)
            data_label = data_label.to(device, dtype=torch.long)
            logits = main_model(data)
            batch_loss = criterion(logits, data_label)

            batch_size = data_label.size(0)
            total_loss += batch_loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == data_label).sum().item()
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("test_loader не содержит данных")

    return total_loss / total_samples, 100 * total_correct / total_samples
