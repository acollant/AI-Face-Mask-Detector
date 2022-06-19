import torch
from sklearn.model_selection import KFold

batch_size = 100
num_classes = 5
learning_rate = 0.005
num_epochs = 10
splits = KFold(n_splits=10, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
