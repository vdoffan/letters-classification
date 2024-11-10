import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import secrets
from linearTraining import LinearNN, evaluate
from sklearn.metrics import accuracy_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
strl = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def getChank(data):
    chanks = []
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(10):
        random_index = secrets.randbelow(len(data))
        test_sample, sample_class = data[random_index]
        chanks.append((test_sample, sample_class))
        image = test_sample.squeeze(0)
        ax = axes[i // 5, i % 5]
        ax.imshow(image)
        ax.set_title(f"Label: {strl[sample_class-1]}")
        ax.axis("off")

    fig.suptitle("Картинки букв для распознавания")
    plt.tight_layout()
    plt.show()

    return chanks


def evaluateOnPics(model, data):
    test_chanks = getChank(data)
    y_pred_list = []
    y_true_list = []
    for i, chank in enumerate(test_chanks):
        x_test, y_true = chank
        logits = model(x_test.to(device))

        y_pred = torch.argmax(logits, dim=1)

        print(f"Для буквы {strl[y_true - 1]} предсказание {strl[y_pred - 1]}")

        y_pred_list.append(y_pred)
        y_true_list.append(y_true)

    accuracy = accuracy_score(y_true_list, y_pred_list)
    return accuracy


def main():
    print("В процессе работы программы появится картинка с буквами для предсказания.")
    print("Чтоы продолжить работу программы, закройте эту картинку.")

    model = LinearNN()
    model.load_state_dict(torch.load("linear_model.pth", weights_only=True))
    model.eval()

    test_data = datasets.EMNIST(
        "./emnist-data",
        split="letters",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss()

    test_acc, test_loss = evaluate(model, test_loader, loss_fn)
    print(
        "Для распознавания букв используется нейросеть со следующими характеристиками:"
    )
    print("Точность на тестовой выборке:", test_acc, "Loss на тестовой выборке:", test_loss)

    accuracy = evaluateOnPics(model, test_data)
    print(f"Точность предсказаний для данных букв: {int(accuracy*10)}/10")


if __name__ == "__main__":
    main()
