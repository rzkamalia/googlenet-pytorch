import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms

import os

from googlenet import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

if not os.path.exists('./models/'):
    os.makedirs('./models/')

# dataset
img_dir = '../data-group/trainval/'

def dataset(img_dir):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    data = datasets.ImageFolder(img_dir, transform = transform)

    train_rasio = 0.80
    train_size = int(len(data) * train_rasio)
    val_size = len(data) - train_size

    train_dataset, val_dataset = random_split(data, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = 4, shuffle = False)
    return train_loader, val_loader

def training(epochs, training_loader, validation_loader):
    for epoch in range(epochs):
        train_loss = 0.0
        train_correct = 0
        model.train()
        for data, target in training_loader:
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = output.argmax(dim = 1, keepdim = True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()

        train_loss /= len(training_loader.dataset)
        train_acc = 100. * train_correct / len(training_loader.dataset)

        best_val_acc = 0
        val_loss = 0.0
        val_correct = 0
        model.eval()
        with torch.no_grad():
            for data, target in validation_loader:
                data = data.to(device)
                target = target.to(device)
                
                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                pred = output.argmax(dim = 1, keepdim = True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(validation_loader.dataset)
        val_acc = 100. * val_correct / len(validation_loader.dataset)

        print('Epoch {}, Train Loss: {:.4f}, Train Acc: {:.2f}%, Val Loss: {:.4f}, Val Acc: {:.2f}%'
            .format(epoch + 1, train_loss, train_acc, val_loss, val_acc))
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print('Best Accuracy {:.2f}%, and save the model.'.format(best_val_acc))
            torch.save(model.state_dict(), './models/model_{}.pt'.format(epoch))

epochs = 2

if __name__ == '__main__':
    model = GoogleNet(num_classes = 2)
    model.to(device)

    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(model.parameters()))

    training_loader, validation_loader = dataset(img_dir)

    training(epochs, training_loader, validation_loader)    