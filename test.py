import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from googlenet import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

img_dir_test = '../data-group/testing/'
model_path = './models/model_1.pt'
classes = ['1stgen', 'JuNeeBang']

def testing(model_load, test_loader):
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model_load(inputs)
            _, predicted = torch.max(outputs.data, 1)
            true_labels += labels.tolist()
            pred_labels += predicted.tolist()

    print(classification_report(true_labels, pred_labels))

    conf_matrix = confusion_matrix(true_labels, pred_labels)
    norm_conf_matrix = conf_matrix / conf_matrix.sum(axis = 1, keepdims = True)

    plt.imshow(norm_conf_matrix, cmap = plt.cm.Blues)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, format(norm_conf_matrix[i, j], '.2f'),
                    ha = "center", va = "center", color = "white" if norm_conf_matrix[i, j] > 0.5 else "black")

    plt.savefig('confussion_matrix.jpg')
    plt.show()

if __name__ == '__main__':
    model_load = GoogleNet(num_classes = 2)
    weights = torch.load(model_path)
    model_load.load_state_dict(weights)
    model_load.eval()
    model_load.to(device)

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    data = datasets.ImageFolder(img_dir_test, transform = transform)
    test_loader = DataLoader(data, batch_size = 4, shuffle = False)

    testing(model_load, test_loader)