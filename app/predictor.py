# Функция распознавания
# app/predictor.py

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import sys


# ───────────────────────────────────────────────
# 1. Повторяем архитектуру сети (должна совпадать)
# ───────────────────────────────────────────────
class CNN(nn.Module):
    '''
    Класс для создания модели CNN
    '''
    def __init__(self):
        '''
        Инициализация модели
        '''
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 11)
        )

    def forward(self, x):
        '''
        Прямой проход модели
        '''
        return self.net(x)

# ───────────────────────────────────────────────
# 2. Функция: загрузить модель и сделать предсказание
# ───────────────────────────────────────────────
def predict_image(image_path: str) -> list:
    '''
    Функция для предсказания изображения
    '''
    # Проверка пути
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Файл не найден: {image_path}")

    # Устройство (CPU или GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка модели
    model = CNN().to(device)
    # model.load_state_dict(torch.load("model/mnist_cnn.pth", map_location=device))

    # для поддержки PyInstaller
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")

    model_path = os.path.join(base_path, "model", "mnist_cnn.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))


    model.eval()

    # Преобразование изображения
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # преобразование изображения в ч/б
        transforms.Resize((28, 28)), # изменение размера изображения
        transforms.ToTensor() # преобразование изображения в тензор
    ])

    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0).to(device)  # [1, 1, 28, 28]

    # Предсказание
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    return probabilities.tolist()

if __name__ == "__main__":
    path = "testing/10/405.png"
    probs = predict_image(path)

    for i, p in enumerate(probs):
        print(f"{i}: {p:.4f}")
    print(f"Предсказано: {probs.index(max(probs))}")
