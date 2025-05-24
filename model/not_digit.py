from PIL import Image, ImageDraw
import numpy as np
import random
import os

os.makedirs('not_digit', exist_ok=True) # создаём папку для нецифр

for i in range(8000):
    img = Image.new('L', (28, 28), 0)
    draw = ImageDraw.Draw(img)
    t = random.choice(['cross', 'plus', 'square', 'triangle', 'noise', 'empty', 'full'])

    if t == 'cross': # крест
        draw.line((0, 0, 27, 27), fill=255, width=2)
        draw.line((0, 27, 27, 0), fill=255, width=2)
    elif t == 'plus': # плюс
        draw.line((14, 0, 14, 27), fill=255, width=2)
        draw.line((0, 14, 27, 14), fill=255, width=2)
    elif t == 'square': # квадрат
        draw.rectangle((4, 4, 23, 23), outline=255, width=2)
    elif t == 'triangle': # треугольник
        draw.polygon([(14, 3), (3, 24), (25, 24)], outline=255)
    elif t == 'noise': # шум
        arr = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
        img = Image.fromarray(arr, 'L')
    elif t == 'empty':
        pass
    elif t == 'full':
        img = Image.new('L', (28, 28), 255)

    img.save(f'not_digit/0{i}.png')
