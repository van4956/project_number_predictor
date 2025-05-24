# Обучение модели
# model/train_model.py

import torch # библиотека для работы с тензорами
import torch.nn as nn # библиотека для работы с нейронными сетями
import torch.optim as optim # библиотека для работы с оптимизацией
from torch.utils.data import DataLoader # библиотека для работы с датасетами

from dataset import MNISTDataset # импортируем класс для работы с датасетом MNIST


# 1. Класс модели: простая сверточная нейросеть
class CNN(nn.Module):
    '''
    Класс для создания модели CNN
    '''
    def __init__(self):
        '''
        Инициализация модели
        '''
        super(CNN, self).__init__()  # вызываем конструктор родительского класса
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # [B, 1, 28, 28] -> [B, 32, 28, 28] # сверточный слой
            nn.ReLU(),                                   # активационная функция
            nn.MaxPool2d(2),                             # -> [B, 32, 14, 14] # пулинговый слой

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # -> [B, 64, 14, 14] # сверточный слой
            nn.ReLU(),                                   # активационная функция
            nn.MaxPool2d(2),                             # -> [B, 64, 7, 7] # пулинговый слой

            nn.Flatten(),                                # -> [B, 64*7*7] # плоский слой
            nn.Linear(64 * 7 * 7, 128),                  # -> [B, 128] # полносвязный слой
            nn.ReLU(),                                   # активационная функция
            nn.Linear(128, 11)                           # -> [B, 11] # выходной слой
        )

    def forward(self, x):
        '''
        Прямой проход модели
        '''
        return self.net(x)

# 2. Параметры
BATCH_SIZE = 64 # размер батча
EPOCHS = 5 # количество эпох
LR = 0.001 # скорость обучения

# 3. Загрузка данных
train_dataset = MNISTDataset("training")
test_dataset = MNISTDataset("testing")
train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True) # перемешивание данных
test_loader = DataLoader(test_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False) # не перемешивание данных

# 4. Инициализация модели, функции потерь и оптимизатора
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # выбираем устройство для обучения
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 5. Обучение
for epoch in range(EPOCHS):
    model.train() # устанавливаем модель в режим обучения
    total_loss = 0 # инициализируем переменную для хранения суммы потерь
    correct = 0 # инициализируем переменную для хранения количества правильных ответов
    total = 0 # инициализируем переменную для хранения общего количества ответов

    # проходим по всем изображениям и меткам в датасете
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    # вычисляем точность модели
    acc = correct / total * 100
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Accuracy: {acc:.2f}%")

# 6. Оценка на тесте
model.eval() # устанавливаем модель в режим оценки
correct = 0 # инициализируем переменную для хранения количества правильных ответов
total = 0 # инициализируем переменную для хранения общего количества ответов

# проходим по всем изображениям и меткам в датасете
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

# вычисляем точность модели
test_acc = correct / total * 100
print(f"Test Accuracy: {test_acc:.2f}%")

# 7. Сохранение модели
torch.save(model.state_dict(), "model/mnist_cnn.pth")
print("Модель сохранена в model/mnist_cnn.pth")
