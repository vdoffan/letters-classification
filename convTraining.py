import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Улучшенная модель
class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 26)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [batch, 32, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 64, 8, 8]
        x = self.pool(F.relu(self.conv3(x)))  # [batch, 128, 4, 4]
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def evaluate(model, dataloader, loss_fn, device):
    y_pred_list = []
    y_true_list = []
    losses = []

    for i, batch in enumerate(dataloader):
        X_batch, y_batch = batch

        with torch.no_grad():
            logits = model(X_batch.to(device))

            loss = loss_fn(logits, y_batch.to(device))

            loss = loss.item()
            losses.append(loss)

            y_pred = torch.argmax(logits, dim=1)

        y_pred_list.extend(y_pred.cpu().numpy())
        y_true_list.extend(y_batch.numpy())

    accuracy = accuracy_score(y_true_list, y_pred_list)

    return accuracy, np.mean(losses)


def train(model, loss_fn, optimizer, dataloader, n_epochs=6):
    model.train(True)

    train_loader = dataloader
    for epoch in range(n_epochs):
        print(f"On epoch {epoch+1}/{n_epochs}")

        for i, batch in enumerate(train_loader):
            X_batch, y_batch = batch

            logits = model(X_batch.to(device))
            loss = loss_fn(logits, y_batch.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc, train_loss = evaluate(model, train_loader, loss_fn, device)
        print(f"Train accuracy: {train_acc}. Train loss: {train_loss}")

    return model


def main():
    model = ConvNN()
    model.to(device)

    train_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_data = datasets.ImageFolder(root="datasets/train", transform=train_transform)

    test_data = datasets.ImageFolder(root="datasets/test", transform=test_transform)

    print(train_data.classes)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    model = train(model, loss_fn, optimizer, train_loader, n_epochs=400)

    test_acc, test_loss = evaluate(model, test_loader, loss_fn, device)
    print(f"Test accuracy: {test_acc}. Test loss: {test_loss}")

    torch.save(model.state_dict(), "conv_model.pth")


if __name__ == "__main__":
    main()
