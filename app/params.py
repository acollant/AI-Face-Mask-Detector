import torch

batch_size = 100
num_classes = 5
learning_rate = 0.005
num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
