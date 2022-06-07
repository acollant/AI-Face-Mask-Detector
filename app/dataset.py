import matplotlib.pyplot as plt
import torch
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from params import batch_size

# torch.manual_seed(0)

MULTIPLE_DATA_DIR = "./data/multiple"
SINGLE_DATA_DIR = "./data/single"

dataset = ImageFolder(
    MULTIPLE_DATA_DIR,
    transform=transforms.Compose(
        [transforms.Resize((150, 150)), transforms.ToTensor()]
    ),
)

single_image = ImageFolder(
    SINGLE_DATA_DIR,
    transform=transforms.Compose(
        [transforms.Resize((150, 150)), transforms.ToTensor()]
    ),
)

TEST_SIZE = int(len(dataset) * 0.2)
TRAIN_SIZE = len(dataset) - TEST_SIZE

train_data, test_data = random_split(
    dataset, [TRAIN_SIZE, TEST_SIZE], generator=torch.Generator().manual_seed(40)
)

train_dl = DataLoader(train_data, batch_size, shuffle=True)
test_dl = DataLoader(test_data, batch_size, shuffle=False)
single_test_dl = DataLoader(single_image, batch_size, shuffle=False)
