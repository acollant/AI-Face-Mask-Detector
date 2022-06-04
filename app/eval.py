import torch

from dataset import test_dl, train_dl
from params import device

model = torch.load("./model/cnn.pb")
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_dl:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(
        "Accuracy of the network on the {} test images: {} %".format(
            total, 100 * correct / total
        )
    )
