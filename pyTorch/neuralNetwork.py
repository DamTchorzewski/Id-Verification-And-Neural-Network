

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
from torchvision import datasets

print("Neural Network")

# Building a Simple Neural Network Model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Loss Functions and Optimizers
def create_loss_and_optimizer(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5, weight_decay=0.0001)
    return criterion, optimizer

# Visualization of Model Results
def visualize_results(train_losses, train_accs, test_losses, test_accs):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss vs Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Training Accuracy')
    plt.plot(epochs, test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy vs Epoch')
    plt.legend()

    plt.show()

# Save and Load Model
def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f'Model saved to {filename}')

def load_model(model, filename):
    model.load_state_dict(torch.load(filename))
    model.eval()  # Set the model to evaluation mode
    print(f'Model loaded from {filename}')

# Definition of the Training Function
def train_model(model, criterion, optimizer, train_loader, test_loader, epochs, device, patience=10):
    model.train()
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    min_test_loss = float('inf')
    no_improvement_counter = 0

    for epoch in range(epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU if available
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / (i + 1)
        train_acc = 100. * correct_train / total_train
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Testing
        model.eval()
        running_test_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)  # Move data to GPU if available
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_test_loss += loss.item()
                optimizer.zero_grad() 
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_loss = running_test_loss / len(test_loader)
        test_acc = 100. * correct_test / total_test
        test_losses.append(test_loss)
        test_accs.append(test_acc)


        print(f'Epoch {epoch + 1}, Train: Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, Test: Loss:{test_loss:.4f}, Acc:{test_acc:.2f}%')

       

    train_losses = [loss for epoch, loss in enumerate(train_losses, 1)]
    train_accs = [acc for epoch, acc in enumerate(train_accs, 1)]
    test_losses = [loss for epoch, loss in enumerate(test_losses, 1)]
    test_accs = [acc for epoch, acc in enumerate(test_accs, 1)]

    return train_losses, train_accs, test_losses, test_accs

# Definition of the DataLoader Creation Function
def create_data_loader(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    # Load the dataset
    mnist_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Split the dataset into training and testing sets
    train_dataset, test_dataset = train_test_split(mnist_dataset, test_size=0.2, random_state=42)

    # Create DataLoader for training set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create DataLoader for test set
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Calling the DataLoader Creation Function
batch_size = 64
train_loader, test_loader = create_data_loader(batch_size)

# Initialization of the Model, Loss Function, and Optimizer
model = SimpleNet()

criterion, optimizer = create_loss_and_optimizer(model)

# Calling the Training Function
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move model to GPU if available
train_losses, train_accs, test_losses, test_accs = train_model(model, criterion, optimizer, train_loader, test_loader, epochs=num_epochs, patience=10, device=device)

# Calling the Results Visualization Function
visualize_results(train_losses, train_accs, test_losses, test_accs)

# Save the Model
save_model(model, './pyTorch/path/simple_net.pth')

# Load the Model
loaded_model = SimpleNet()
loaded_model.to(device)  # Move loaded model to GPU if available
load_model(loaded_model, './pyTorch/path/simple_net.pth')

# Test the Loaded Model
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)  # Move data to GPU if available
        outputs = loaded_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the loaded network on the 10000 test images: {100 * correct / total}%')
