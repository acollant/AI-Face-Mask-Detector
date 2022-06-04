import torchvision.transforms as transforms

base_transform = transforms.Compose(
    [transforms.Resize((32, 32)), transforms.ToTensor()]
)
