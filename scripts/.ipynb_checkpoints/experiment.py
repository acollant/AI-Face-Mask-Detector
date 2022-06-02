import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

import os
#train and test data directory
data_dir = "data/no_mask_test"
test_data_dir = "no-mask-test/"


#load the train and test data
dataset = ImageFolder(data_dir,transform = transforms.Compose([
    transforms.Resize((150,150)),transforms.ToTensor()
]))
test_dataset = ImageFolder(test_data_dir,transforms.Compose([
    transforms.Resize((150,150)),transforms.ToTensor()
]))