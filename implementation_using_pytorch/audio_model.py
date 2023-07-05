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
