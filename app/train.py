import torch
import torch.nn as nn

from cnn import CNNV1, CNNV2, CNNV3
from dataset import train_dl
from params import device, learning_rate, num_classes, num_epochs

model = CNNV3(num_classes)


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9
)

total_step = len(train_dl)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_dl):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, loss.item()))

torch.save(model, "./model/CNNV3.pb")
