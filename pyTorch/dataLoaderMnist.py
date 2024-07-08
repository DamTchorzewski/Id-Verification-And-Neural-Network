# Ładowanie i Przetwarzanie Danych:
# Załaduj zbiór danych (np. MNIST) przy użyciu PyTorch DataLoader i
# wykonaj na nim proste przekształcenia (normalizacja, zmiana rozmiaru itp.).

import torch
from torchvision import datasets, transforms

print("Data Loader MNIST")

# Definition of a data set
class MyMNIST(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(MyMNIST, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index):
        # Obtaining an image-label pair
        img, target = self.data[index], int(self.targets[index])

        # Resize the image to 28x28
        img = transforms.ToPILImage()(img)
        img = transforms.Resize((28, 28))(img)

        # Convert img from (H, W, C) to (C, H, W)
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)

        # Adding new dimensionality to store single-frame images (C, H, W)
        img = img.unsqueeze(0)

        # Applying normalization to img
        img = transforms.Normalize((0.1307,), (0.3081,))(img)

        return img, target
    
    # Define the transformations to apply to each dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# Loading the data set
train_data = MyMNIST(root='./data', train=True, download=True, transform=transform)
test_data = MyMNIST(root='./data', train=False, download=True, transform=transform)

# Created DataLoader
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=64, shuffle=False)

# Downloading the first batch of data from the training set
data, labels = next(iter(train_loader))