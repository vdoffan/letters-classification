import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import secrets


class LinearNN(nn.Module):
    def __init__(self):
        super(LinearNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, 64)
    
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

model = LinearNN()
model.load_state_dict(torch.load('./linear_model.pth'))

test_data = datasets.EMNIST('./emnist-data', split='letters', train=False, download=True, transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)

strl = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
def getChank(data):
    for i in range(10):
        random_index = secrets.randbelow(len(data))
        test_sample, sample_class = data[random_index]
        image = test_sample.squeeze(0)
        ax = axes[i // 5, i % 5]
        ax.imshow(image)
        ax.set_title(f"Label: {strl[sample_class-1]}, ({sample_class})")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

