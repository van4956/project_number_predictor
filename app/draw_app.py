# Рисовалка
# app/draw_app.py

import tkinter as tk  # библиотека для создания графического интерфейса
from PIL import Image, ImageDraw, ImageOps  # библиотека для работы с изображениями
from .predictor import predict_image  # функция для предсказания изображения
import os

CANVAS_SIZE = 280  # размер холста
IMG_SIZE = 28  # размер изображения

class DrawApp:
    '''
    Класс для создания графического интерфейса
    '''
    def __init__(self, master):
        '''
        Инициализация графического интерфейса
        '''
        self.master = master
        master.title("Распознавание цифр")

        # Холст для рисования
        self.canvas = tk.Canvas(master, width=CANVAS_SIZE, height=CANVAS_SIZE, bg='black')
        self.canvas.grid(row=0, column=0, rowspan=11)  # размещаем холст в верхней части интерфейса
        self.canvas.bind("<B1-Motion>", self.draw)  # привязываем рисование к левой кнопке мыши
        self.canvas.bind("<B3-Motion>", self.erase)  # привязываем стирание к правой кнопке мыши

        # Кнопка очистки
        self.clear_button = tk.Button(master, text="Очистить", command=self.clear)
        self.clear_button.grid(row=11, column=0, sticky="we")

        # Кнопка сохранения
        # self.save_counter = 0
        # self.save_button = tk.Button(master, text="Сохранить", command=self.save_image)
        # self.save_button.grid(row=1, column=1, sticky="we")

        # Изображение (внутренняя память рисования)
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw_img = ImageDraw.Draw(self.image)

        # Панель предсказаний (мини-гистограммы)
        self.bars = []
        for i in range(11):
            frame = tk.Frame(master)
            frame.grid(row=i+1, column=1, sticky="w", pady=1)
            if i == 10: i = "~"
            label = tk.Label(frame, text=f"{i}:", width=2, font=("Courier", 12))
            label.pack(side="left")

            bar = tk.Canvas(frame, width=100, height=10, bg="white", highlightthickness=0)
            bar.pack(side="left")
            self.bars.append((label, bar))

        # Периодическое обновление предсказания
        self.update_prediction()

    def draw(self, event):
        '''
        Рисование на холсте
        '''
        x, y = event.x, event.y
        r = 7  # радиус рисования
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='white', outline='white')
        self.draw_img.ellipse([x-r, y-r, x+r, y+r], fill=255)

    def erase(self, event):
        '''
        Стирание на холсте
        '''
        x, y = event.x, event.y
        r = 12  # радиус стирания
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black', outline='black')
        self.draw_img.ellipse([x-r, y-r, x+r, y+r], fill=0)

    def clear(self):
        '''
        Очистка холста
        '''
        self.canvas.delete("all")
        self.draw_img.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
        for lbl, bar in self.bars:
            lbl.config(font=("Courier", 12))
            bar.delete("bar")

    def update_prediction(self):
        '''
        Обновление предсказаний
        '''
        img_resized = self.image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)  # уменьшаем изображение
        # img_resized = ImageOps.invert(img_resized)  # инвертируем изображение
        img_resized.save("tmp_input.png")  # сохраняем изображение

        # Предсказание вероятности принадлежности к классу
        probs = predict_image("tmp_input.png")
        max_idx = probs.index(max(probs))

        for i, (lbl, bar) in enumerate(self.bars):
            bar.delete("bar")
            width = int(probs[i] * 100)
            bar.create_rectangle(0, 0, width, 10, fill="green", tags="bar")

            if i == max_idx:
                lbl.config(font=("Courier", 13, "bold"))
            else:
                lbl.config(font=("Courier", 12, "normal"))

        # Запускаем обновление через 300 мс
        self.master.after(300, self.update_prediction)

    def save_image(self):
        '''
        Сохраняет текущее изображение в папку not_digit с уникальным именем
        '''
        while True:
            filename = f"not_digit/{self.save_counter:03d}.png"
            if not os.path.exists(filename):
                break
            self.save_counter += 1
        img_resized = self.image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        img_resized.save(filename)
        self.save_counter += 1

# ───────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = DrawApp(root)
    root.mainloop()
