import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score


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

        return x

train_data = datasets.EMNIST(root='./emnist-data', split='letters', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.EMNIST(root='./emnist-data', split='letters', train=False, download=True, transform=transforms.ToTensor())

train_size = int(len(train_data)*0.8)
val_size = len(train_data) - train_size
train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])


train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LinearNN().to(device)

learning_rate = 1e-3
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)


def evaluate(model, dataloader, loss_fn):
    y_pred_list = []
    y_true_list = []
    losses = []
    
    for i, batch in enumerate(dataloader):
        x_batch, y_batch = batch
        
        with torch.no_grad():
            logits = model(x_batch.to(device))
            loss = loss_fn(logits, y_batch.to(device))
            
            loss = loss.item()
            losses.append(loss)

            y_pred = torch.argmax(logits, dim=1)

        y_pred_list.extend(y_pred.cpu().numpy())
        y_true_list.extend(y_batch.numpy())
    
    accuracy = accuracy_score(y_true_list, y_pred_list)
    
    return accuracy, np.mean(losses)


def train(model, loss_fn, optimizer, n_epochs=10):
    model.train(True)
    
    for epoch in range(n_epochs):
        print(f'On epoch: {epoch+1}/{n_epochs}')

        for i, batch in enumerate(train_loader):
            x_batch, y_batch = batch
            
            logits = model(x_batch.to(device))
            
            loss = loss_fn(logits, y_batch.to(device))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        train_acc, train_loss = evaluate(model, train_loader, loss_fn)
        print('Train acc:', train_acc, 'Train loss:', train_loss)

        val_acc, val_loss = evaluate(model, val_loader, loss_fn)
        print('Val acc:', val_acc, 'Val loss:', val_loss)

    return model

model = train(model, loss_fn, optimizer, 6)

test_acc, test_loss = evaluate(model, test_loader, loss_fn)
print('Test acc:', test_acc, 'Test loss:', test_loss)

torch.save(model.state_dict(), 'linear_model.pth')