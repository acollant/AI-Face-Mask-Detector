"""
This file will expose all variables regarding the dataset.

"""
import torch
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from params import batch_size

torch.manual_seed(0)

DATA_DIR = "./data"

dataset = ImageFolder(
    DATA_DIR,
    transform=transforms.Compose(
        [transforms.Resize((150, 150)), transforms.ToTensor()]
    ),
)

TEST_SIZE = int(len(dataset) * 0.2)
TRAIN_SIZE = len(dataset) - TEST_SIZE

train_data, test_data = random_split(dataset, [TRAIN_SIZE, TEST_SIZE])

train_dl = DataLoader(train_data, batch_size, shuffle=True)
test_dl = DataLoader(test_data, batch_size * 2, shuffle=False)

print(TRAIN_SIZE, TEST_SIZE)
