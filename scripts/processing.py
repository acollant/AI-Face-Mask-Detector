
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder


from torchvision import models
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import lr_scheduler
from torch import optim
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader

import matplotlib.pyplot as plt
from utils import initialize_train_dataset, initialize_test_dataset, imshow, train_model, show_batch

import time

if torch.cuda.is_available():
    is_cuda = True

#load Pytorch
train = initialize_train_dataset()
test  = initialize_test_dataset()

print("Follwing classes are there train: \n",train.classes)

#print(train.class_to_idx)
#print(train.classes) 
#imshow(train[30][0]) #uncomment to show a picture

"""
The data is divided into batches using the PyTorch DataLoader class. We create two objects train_dataloader and 
test_dataloder for training and validation data respectively by giving parameters training data and batch size into the
DataLoader Class.
"""
train_dataloader = DataLoader(train, 
                            shuffle = True, 
                            batch_size = 64,
                            num_workers = 0, 
                            pin_memory = True
                            )

test_dataloder = DataLoader(test, 
                            batch_size = 64,
                            num_workers = 0, 
                            pin_memory = True)

dataset_sizes = {
                 'train':len(train_dataloader.dataset),
                 'test':len(test_dataloder.dataset)
                }

dataloaders = {
               'train':train_dataloader, 
               'test':test_dataloder
              }

#show_batch(train_dataloader)

#create the model
model_ft = models.resnet18(pretrained=True) #Deep Residual Learning for Image Recognition
num_ftrs = model_ft.fc.in_features # define the number of input for the linear layer
model_ft.fc = nn.Linear(num_ftrs, 5) # number of classes

#check for GPU availability
if torch.cuda.is_available():
    model_ft = model_ft.cuda()


print(model_ft)

# Loss and Optimizer
learning_rate = 0.001
criterion = nn.CrossEntropyLoss() # https://medium.com/swlh/cross-entropy-loss-in-pytorch-c010faf97bab
optimizer_ft = optim.SGD(model_ft.parameters(), learning_rate=0.01, momentum=0.9) #https://pytorch.org/docs/stable/optim.html
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)




# I need to improve the train_model
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,  dataloaders, dataset_sizes, num_epochs=5)