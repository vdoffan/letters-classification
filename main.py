import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import secrets
from convTraining import ConvNN, evaluate
from sklearn.metrics import accuracy_score

# Определение устройства для вычислений (GPU, если доступен)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Строка с алфавитом для отображения меток классов
strl = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# Выбирает случайные 10 изображений из набора данных и отображает их
def getChank(data):
    chanks = []
    # Создание сетки для отображения изображений (2 строки по 5 столбцов)
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))

    for i in range(10):
        # Выбор случайного индекса
        random_index = secrets.randbelow(len(data))
        # Получение изображения и его класса
        test_sample, sample_class = data[random_index]
        chanks.append((test_sample, sample_class))
        # Удаление канала цвета для отображения в оттенках серого
        image = test_sample.squeeze(0)
        # Выбор соответствующей оси для отображения изображения
        ax = axes[i // 5, i % 5]
        ax.imshow(image, cmap="gray")
        ax.set_title(f"Label: {strl[sample_class]}")
        ax.axis("off")

    # Установка заголовка для всей фигуры
    fig.suptitle("Картинки букв для распознавания")
    plt.tight_layout()
    plt.show()

    return chanks


# Оценивает модель на случайных изображениях и выводит предсказания
def evaluateOnPics(model, data):
    # Получение случайных изображений для оценки
    test_chanks = getChank(data)
    y_pred_list = []
    y_true_list = []

    for chank in test_chanks:
        x_test, y_true = chank
        # Прогноз модели
        logits = model(x_test.to(device))
        # Выбор класса с максимальной вероятностью
        y_pred = torch.argmax(logits, dim=1)

        # Вывод предсказания для текущего изображения
        print(f"Для буквы {strl[y_true]} предсказание {strl[y_pred]}")

        # Добавление предсказанных и истинных значений в списки
        y_pred_list.append(y_pred.cpu().item())
        y_true_list.append(y_true)

    # Вычисление точности
    accuracy = accuracy_score(y_true_list, y_pred_list)
    return accuracy


# Основная функция
def main():
    print("В процессе работы программы появится картинка с буквами для предсказания.")
    print("Чтобы продолжить работу программы, закройте эту картинку.")

    # Создание экземпляра модели
    model = ConvNN()

    # Загрузка сохраненных весов модели
    model.load_state_dict(
        torch.load("conv_model.pth", weights_only=True, map_location=device)
    )
    model.to(device)  # Перемещение модели на устройство
    model.eval()  # Установка модели в режим оценки

    # Определение трансформаций для тестовых данных
    test_transform = transforms.Compose(
        [
            transforms.Grayscale(),  # Преобразование в оттенки серого
            transforms.ToTensor(),  # Преобразование в тензор
            transforms.Normalize((0.5,), (0.5,)),  # Нормализация
        ]
    )

    # Загрузка тестовых данных
    test_data = datasets.ImageFolder(root="datasets/test", transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

    # Определение функции потерь
    loss_fn = torch.nn.CrossEntropyLoss()

    # Оценка модели на тестовых данных
    test_acc, test_loss = evaluate(model, test_loader, loss_fn, device)
    print(
        "Для распознавания букв используется нейросеть со следующими характеристиками:"
    )
    print(
        "Точность на тестовой выборке:",
        test_acc,
        "Loss на тестовой выборке:",
        test_loss,
    )

    # Оценка модели на случайных изображениях
    accuracy = evaluateOnPics(model, test_data)
    print(f"Точность предсказаний для данных букв: {int(accuracy*10)}/10")


if __name__ == "__main__":
    main()  # Запуск основной функции
