import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score

# Устройство для вычислений (GPU, если доступен)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Определение нейронной сети с несколькими сверточными слоями
class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()
        # Сверточные слои с 32 и 64 фильтрами соответственно
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Слой максимального пулинга
        self.pool = nn.MaxPool2d(2, 2)
        # Полносвязные слои
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 26)
        # Слой Dropout для регуляризации
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Применение сверток и пулинга с активацией ReLU
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Изменение формы тензора перед подачей в полносвязный слой
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))  # Активация ReLU для первого полносвязного слоя
        x = self.dropout(x)  # Применение Dropout
        x = self.fc2(x)  # Выходной слой
        return x


# Функция для оценки точности модели на данных (train или test)
def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    y_pred_list = []
    y_true_list = []
    losses = []

    # Итерация по батчам данных
    for batch in dataloader:
        X_batch, y_batch = batch

        with torch.no_grad():  # Выключаем градиенты для оценки
            logits = model(X_batch.to(device))  # Получаем прогнозы модели

            loss = loss_fn(logits, y_batch.to(device))  # Вычисление потерь
            loss = loss.item()  # Получение значения потерь
            losses.append(loss)

            y_pred = torch.argmax(
                logits, dim=1
            )  # Выбор класса с максимальной вероятностью

        # Добавление предсказаний и истинных значений для расчета точности
        y_pred_list.extend(y_pred.cpu().numpy())
        y_true_list.extend(y_batch.numpy())

    # Вычисление точности и средней потери
    accuracy = accuracy_score(y_true_list, y_pred_list)

    return accuracy, np.mean(losses)


# Функция тренировки модели
def train(model, loss_fn, optimizer, dataloader, n_epochs=100):
    model.train(True)  # Устанавливаем модель в режим тренировки
    train_loader = dataloader

    for epoch in range(n_epochs + 1):
        for batch in train_loader:
            X_batch, y_batch = batch

            logits = model(X_batch.to(device))  # Получение прогноза модели
            loss = loss_fn(logits, y_batch.to(device))  # Вычисление потерь

            loss.backward()  # Вычисление градиентов
            optimizer.step()  # Обновление параметров модели
            optimizer.zero_grad()  # Обнуляем градиенты

        if epoch % 25 == 0:
            print(f"On epoch {epoch}/{n_epochs}")

            # Оценка точности и потерь на тренировочных данных
            train_acc, train_loss = evaluate(model, train_loader, loss_fn, device)
            print(f"Train accuracy: {train_acc}. Train loss: {train_loss}")

    return model


# Основная функция
def main():
    model = ConvNN()  # Создаем модель
    model.to(device)  # Перемещаем модель на выбранное устройство (GPU или CPU)

    # Определение трансформаций для тренировочных данных
    train_transform = transforms.Compose(
        [
            transforms.Grayscale(),  # Преобразование в оттенки серого
            transforms.RandomRotation(15),  # Случайное вращение
            transforms.RandomAffine(
                degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
            ),  # Аффинные преобразования
            transforms.RandomHorizontalFlip(),  # Случайное горизонтальное отражение
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),  # Случайные изменения яркости, контраста, насыщенности и оттенка
            transforms.ToTensor(),  # Преобразование в тензор
            transforms.Normalize((0.5,), (0.5,)),  # Нормализация
        ]
    )

    # Трансформация для тестовых данных (без аугментаций)
    test_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Загрузка данных с указанием пути к папке и трансформаций
    train_data = datasets.ImageFolder(root="datasets/train", transform=train_transform)
    test_data = datasets.ImageFolder(root="datasets/test", transform=test_transform)

    # Создание DataLoader для тренировки и тестирования
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

    # Определение гиперпараметров и оптимизатора
    learning_rate = 5e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()  # Кросс-энтропия для задачи классификации

    # Тренировка модели
    model = train(model, loss_fn, optimizer, train_loader, n_epochs=200)

    # Оценка точности на тестовых данных
    test_acc, test_loss = evaluate(model, test_loader, loss_fn, device)
    print(f"Test accuracy: {test_acc}. Test loss: {test_loss}")

    # Сохранение обученной модели
    torch.save(model.state_dict(), "conv_model.pth")


if __name__ == "__main__":
    main()  # Запуск основной функции
