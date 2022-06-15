import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from sklearn.metrics import confusion_matrix

from dataset import dataset


def test_stats(model, data_loader, device="cpu"):
    correct = 0
    total = 0
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return predicted, labels, correct, total


def construct_confusion_matrix(labels, predicted, filename, dataset=dataset):
    cf_matrix = confusion_matrix(labels, predicted)
    df_cm = pd.DataFrame(
        cf_matrix / np.sum(cf_matrix) * 10,
        index=[i for i in dataset.classes],
        columns=[i for i in dataset.classes],
    )
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(f"{filename}.png")


def train_epoch(model, device, dataloader, loss_func, optimizer):
    train_loss, correct, total = 0.0, 0, 0
    model.train()
    for images, labels in dataloader:

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return predicted, labels, correct, total, train_loss


def validation_epoch(model, device, dataloader, loss_func, optimizer):
    validation_loss, correct, total = 0.0, 0, 0
    model.eval()
    for images, labels in dataloader:

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_func(outputs, labels)
        optimizer.step()
        validation_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return predicted, labels, correct, total, validation_loss
