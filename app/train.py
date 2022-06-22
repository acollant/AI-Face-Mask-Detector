import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, SubsetRandomSampler

from cnn import CNNV3
from dataset import dataset
from params import batch_size, device, learning_rate, num_classes, num_epochs, splits
from util import train_epoch, validation_epoch

model = CNNV3(num_classes)


criterion = nn.CrossEntropyLoss()


for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
    print(f"Fold {fold+1}")
    fold_predicted = -1
    fold_labels = -1

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    # model = CNNV3(num_classes)
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9
    )

    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(num_epochs):
        (
            train_predicted,
            train_labels,
            train_correct,
            train_total,
            train_loss,
        ) = train_epoch(model, device, train_loader, criterion, optimizer)
        (
            test_predicted,
            test_labels,
            test_correct,
            test_total,
            test_loss,
        ) = validation_epoch(model, device, test_loader, criterion, optimizer)
        if epoch == 0:
            fold_predicted = test_predicted
            fold_labels = test_labels
        else:
            fold_predicted = torch.cat((fold_predicted, test_predicted), 0)
            fold_labels = torch.cat((fold_labels, test_labels), 0)

        train_loss = train_loss / len(train_loader.sampler)
        train_acc = train_correct / len(train_loader.sampler) * 100
        test_loss = test_loss / len(test_loader.sampler)
        test_acc = test_correct / len(test_loader.sampler) * 100

        print(
            "Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(
                epoch + 1, num_epochs, train_loss, test_loss, train_acc, test_acc
            )
        )

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

torch.save(model, "./model/CNNV3_kfold_gender.pb")
