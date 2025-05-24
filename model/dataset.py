# Кастомный Dataset
# model/dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset  # это класс для работы с датасетами
import torch  # это библиотека для работы с тензорами
from torchvision import transforms  # это библиотека для работы с изображениями

class MNISTDataset(Dataset):
    '''
    Класс для работы с датасетом MNIST
    '''
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []

        # Сканируем подпапки 0-10
        for label in range(11):
            label_dir = os.path.join(root_dir, str(label))
            for filename in os.listdir(label_dir):
                if filename.endswith('.png'):
                    self.image_paths.append(os.path.join(label_dir, filename))
                    self.labels.append(label)

        # Преобразование: черно-белое изображение 28x28 → тензор [1, 28, 28], значения от 0 до 1
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

    def __len__(self):
        '''
        Возвращает количество изображений в датасете
        '''
        return len(self.image_paths)

    def __getitem__(self, idx):
        '''
        Возвращает изображение и метку по индексу
        '''
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("L")  # ЧБ-режим
        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
