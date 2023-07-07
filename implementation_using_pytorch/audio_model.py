import torch
from leaf_audio import frontend
import sys
import os
import logging
from scipy.io import wavfile
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam
import torch.nn as nn
import random

""""
This project of emotion detection was made for cldc
@Author: Neel Shah
@Mail: neeldevenshah@gmail.com
"""

# Python has a built-in module logging which allows writing status messages to a file or any other output streams.
# Logger set
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

np.random.seed(1234)
torch.manual_seed(1234)

# CUDA devices enabled
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backend.cudnn.deterministic = True
torch.cuda.empty_cache()

class MyDataset(Dataset):
    '''
    Dataset class for audio which reads in the audio signals and prepares them for
    training. In particular it pads all of them to the maximum length of the audio
    signal that is present. All audio signals are padded/sampled upto 34s in this
    dataset.
    '''
    
    def __init__(self, list_audio, data_path):
        self.list_audio = list_audio
        self.data_path = str(data_path)
        self.duration = 34000
        self.sr = 16000
        self.channel = 1
        with open('../label_dict.json', 'r') as f:
            self.labels = json.load(f)
            
    def __len__(self):
        return len(self.list_audio)
    
    def __getitem__(self, audio_ind):
        audio_file = os.path.join(self.data_path, self.list_audio[audio_ind])
        class_id = self.labels[self.list_audio[audio_ind]]
        
        (sr, sig) = wavfile.read(audio_file)
        
        aud = (sig, sr)
        reud = (sig, self.sr)
        resig = sig
        sig_len = resig.shape[0]
        max_len = self.sr//1000 * self.duration
        
        if len(resig.shape) == 2:
            resig = np.mean(resig, axis=1)
            
        if (sig_len > max_len):
            # Truncating the signal to the given length
            final_sig = resig[:max_len]
            
        elif (sig_len < max_len):
            # Length of padding to add to the beginning and end of the signal
            padd_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - padd_begin_len
            
            # Pad with 0s
            pad_begin = np.zeros((padd_begin_len))
            pad_end = np.zeros((pad_end_len))
            
            final_sig = np.float32(np.concatenate((pad_begin, resig, pad_end), 0))
            final_aud = (final_sig, self.sr)
        return final_sig, class_id
    
class Net(nn.Module):
    '''Defines the CNN network that works on the output from LEAF for audio classification.'''
    
    def __init__(self):
        super(Net, self).__init__()
        self.start = 16
        self.final = 100
        self.conv1 = nn.Conv2d(1, self.start*2, 3, 1)
        self.bn1 = nn.BatchNorm2d(self.start*2)
        
        self.conv2 = nn.Conv2d(self.start*2, self.start*2, 3, 1)
        self.bn2 = nn.BatchNorm2d(self.start*2)
        
        self.conv3 = nn.Conv2d(self.start*2, self.start*4, 3, 1)
        self.bn3 = nn.BatchNorm2d(self.start*4)
        
        self.conv4 = nn.Conv2d(self.start*4, self.start*4, 3, 1)
        self.bn4 = nn.BatchNorm2d(self.start*4)
        
        self.conv5 = nn.Conv2d(self.start*4, self.start*8, 3, 1)
        self.bn5 = nn.BatchNorm2d(self.start*8)
        
        self.conv6 = nn.Conv2d(self.start*8, self.final, 3, 1)
        self.bn6 = nn.BatchNorm2d(self.final)
        
        self.pool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(self.final, 8)
        
    def forward(self, x):
        x = x.to(device)
        x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        
        x = self.conv2(x)
        x = self.relu(self.bn2(x))
        
        x = self.conv3(x)
        x = self.relu(self.bn3(x))
        
        x = self.conv4(x)
        x = self.relu(self.bn4(x))
        
        x = self.conv5(x)
        x = self.relu(self.bn5(x))
        
        x = self.conv6(x)
        x = self.relu(self.bn6(x))
        x = self.pool(x)
        
        x = x.squeeze(-2)
        x1 = torch.mean(x, dim=2)
        x = self.fc(x1)
        
        return x, x1
    
def compute_accuracy(output, labels):
    
    # Function for calculating accuracy
    pred = torch.argmax(output, dim=1)
    correct_pred = (pred == labels).float()
    tot_corret = correct_pred.sum()
    return tot_corret
    
def compute_loss(output, labels):
    