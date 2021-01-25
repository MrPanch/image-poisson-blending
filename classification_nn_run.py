
import numpy as np
import torch
import os

from tqdm import tqdm, tqdm_notebook
from PIL import Image

from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import pandas as pd


import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)


class XDataset(Dataset):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

        self.df = pd.read_csv(f'validation.csv')

        self.root_dir = ''
        self.len_ = 0
        self.files = []
        self.border = 0
        if self.mode == 'train':
            self.root_dir = f'GLCM_images/train_cropped/train_merged/'
            self.files = os.listdir(self.root_dir)
            self.len_ = len(self.files)
            self.border = len(os.listdir(f'GLCM_images/train_cropped/train_negative_cropped')) - 1

        elif self.mode == 'validation':
            self.root_dir = f'GLCM_images/validation_cropped/'
            self.files = os.listdir(self.root_dir)
            self.len_ = len(self.files)
        else:
            raise Exception('Wrong option! Available: train and validation')

    def __len__(self):
        return self.len_

    def load_sample(self, file):
        image = Image.open(os.path.join(self.root_dir,file))
        image.load()
        return image

    def __getitem__(self, index):
        x = self.load_sample(self.files[index])
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        x = transform(x)
        x = np.array(x / 255, dtype='float32')
        if self.mode == 'validation':
            y = self.df[self.df['image_name'] == self.files[index]].iloc[0]['class']
            return x, y
        else:
            y = -1
            if index <= self.border:
                y = 0
            if index > self.border:
                y = 1
            return x, y


# class ConvNet(nn.Module):
#     def __init__(self, num_classes=2):
#         super(ConvNet, self).__init__()
#         self.layer1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2)
#         self.layer2 = nn.Conv2d(6, 12, kernel_size=5, stride=1, padding=2)
#         self.fc = nn.Linear(12 * 100 * 100, num_classes)
#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = out.reshape(out.size(0), -1)
#         out = self.fc(out)
#         return out


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 12, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(12 * 25 * 25, 2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return out

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = XDataset(mode="train")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=16)

    val_dataset = XDataset(mode="validation")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16,
                                             shuffle=True)

    test_dataset = False
    if test_dataset:
        for i in tqdm(train_loader):
            pass
        for i in tqdm(val_loader):
            pass

    model = ConvNet(2).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    total_step = len(train_loader)

    num_epochs = 100
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))
