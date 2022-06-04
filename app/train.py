import torch
import torch.nn as nn

from cnn import CNN
from dataset import train_dl
from params import device, learning_rate, num_classes, num_epochs

model = CNN(num_classes)

# Set Loss function with criterion
criterion = nn.CrossEntropyLoss()

# Set optimizer with optimizer
optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9
)

total_step = len(train_dl)

for epoch in range(num_epochs):
    # Load in the data in batches using the train_dl object
    for i, (images, labels) in enumerate(train_dl):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, loss.item()))

torch.save(model, "./model/cnn.pb")
