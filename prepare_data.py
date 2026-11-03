"""
Подготовка данных.
Читает исходные файлы MNIST / Fashion-MNIST (формат idx) из папки archive,
сохраняет картинки по папкам (одна папка = один класс) и создает CSV файлы.

Пример:
    python prepare_data.py --src archive --out data/mnist
    python prepare_data.py --src archive_fashion --out data/fashion
"""
import argparse
import csv
import os

import numpy as np
from PIL import Image


def read_images(path):
    # Первые 16 байт - заголовок, дальше пиксели по 28*28 на картинку
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return data.reshape(-1, 28, 28)


def read_labels(path):
    # Первые 8 байт - заголовок, дальше по одному байту на метку
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=8)


def save_part(images, labels, folder, csv_path):
    # Сохраняем картинки в папки folder/<класс>/<номер>.png
    # и записываем строки "путь,метка" в CSV
    rows = []
    for i in range(len(images)):
        label = str(labels[i])
        class_dir = os.path.join(folder, label)
        os.makedirs(class_dir, exist_ok=True)

        path = os.path.join(class_dir, f"{i}.png")
        Image.fromarray(images[i]).save(path)
        rows.append([path, label])

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="archive", help="папка с idx файлами")
    parser.add_argument("--out", default="data/mnist", help="куда сохранить картинки")
    args = parser.parse_args()

    train_images = read_images(os.path.join(args.src, "train-images.idx3-ubyte"))
    train_labels = read_labels(os.path.join(args.src, "train-labels.idx1-ubyte"))
    test_images = read_images(os.path.join(args.src, "t10k-images.idx3-ubyte"))
    test_labels = read_labels(os.path.join(args.src, "t10k-labels.idx1-ubyte"))

    os.makedirs(args.out, exist_ok=True)
    print("Сохраняю обучающую выборку:", len(train_images), "картинок")
    save_part(train_images, train_labels, os.path.join(args.out, "train"),
              os.path.join(args.out, "ground-truth-train.csv"))

    print("Сохраняю тестовую выборку:", len(test_images), "картинок")
    test_rows = save_part(test_images, test_labels, os.path.join(args.out, "test"),
                          os.path.join(args.out, "ground-truth-test.csv"))

    # input.csv - только пути к тестовым картинкам, без меток
    with open(os.path.join(args.out, "input.csv"), "w", newline="") as f:
        csv.writer(f).writerows([[row[0]] for row in test_rows])

    print("Готово:", args.out)


if __name__ == "__main__":
    main()