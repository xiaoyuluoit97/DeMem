import torch
import torch.nn as nn
import torch.optim as optim
import os, json, time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import alexnet
import torch.nn.functional as F
from models.AlexNet import AlexNet
from dataset import get_loaders, root

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR100(root='/data/luo/reproduce/privacy_and_aug/cifar100', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR100(root='/data/luo/reproduce/privacy_and_aug/cifar100', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)



# model = alexnet()
# num_ftrs = model.classifier[-1].in_features
# model.classifier[-1] = nn.Linear(num_ftrs, 100)

model = AlexNet(100)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# criterion = nn.CrossEntropyLoss()
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

num_epochs = 100
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / (i + 1)}')
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct / total}%')