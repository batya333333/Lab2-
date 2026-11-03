"""
Основной скрипт: обучение и классификация.

Обучение:
    python mnist.py --mode train --dataset data/mnist/ground-truth-train.csv --model models/mnist.joblib
Классификация:
    python mnist.py --mode inference --model models/mnist.joblib --input data/mnist/input.csv --output predictions_mnist.csv --proba proba_mnist.csv

Что делает обучение:
    1. GridSearchCV: кросс-валидация всех моделей из models.py на подвыборке (--n картинок)
    2. Подбор порога для каждого класса по кросс-валидации лучшей модели
    3. Обучение лучшей модели на всех данных и сохранение
"""
import argparse
import csv
import os

import joblib
import numpy as np
from PIL import Image
from sklearn.base import clone
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from models import make_grid, make_pipeline


#Чтение данных

def read_csv(path):
    # Каждая строка: путь к картинке и (если есть) метка
    with open(path, newline="") as f:
        return [row for row in csv.reader(f) if row]


def load_images(paths):
    # Каждая картинка 28x28 -> 784 числа от 0 до 1
    return np.array([np.array(Image.open(p).convert("L"), dtype=np.float32).flatten() / 255.0
                     for p in paths])


#Обучение

def model_name(params):
    # Короткое имя кандидата для таблицы: модель + её гиперпараметры
    name = type(params["model"]).__name__
    extra = [f"{k.split('__')[1]}={v}" for k, v in params.items() if "__" in k]
    return name + (" (" + ", ".join(extra) + ")" if extra else "")


def choose_thresholds(y, proba):
    # Для каждого класса задача "эта цифра или нет" несбалансирована (1 к 9).
    # Выбираем порог, при котором F1 максимальна.
    thresholds = []
    for c in range(proba.shape[1]):
        precision, recall, thr = precision_recall_curve(y == c, proba[:, c])
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        best = f1[:-1].argmax()
        thresholds.append(float(thr[best]))
    return thresholds


def train(dataset_csv, model_path, n):
    rows = read_csv(dataset_csv)
    X = load_images([r[0] for r in rows])
    y = np.array([int(r[1]) for r in rows])
    print("Картинок для обучения:", len(X))

    # Подвыборка для сравнения моделей (на всех 60 000 перебор слишком долгий)
    idx = np.random.default_rng(42).choice(len(X), size=min(n, len(X)), replace=False)
    X_small, y_small = X[idx], y[idx]
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # 1. Кросс-валидация всех кандидатов
    print(f"\n1. Сравнение моделей: кросс-валидация, 3 фолда, {len(X_small)} картинок")
    search = GridSearchCV(make_pipeline(), make_grid(), cv=cv,
                          scoring={"accuracy": "accuracy", "f1": "f1_macro", "logloss": "neg_log_loss"},
                          refit="f1", verbose=1)
    search.fit(X_small, y_small)

    res = search.cv_results_
    table = []
    for i, params in enumerate(res["params"]):
        table.append([model_name(params),
                      round(res["mean_test_accuracy"][i], 4),
                      round(res["mean_test_f1"][i], 4),
                      round(res["std_test_f1"][i], 4),
                      round(-res["mean_test_logloss"][i], 4),
                      round(res["mean_fit_time"][i], 1)])
    table.sort(key=lambda r: -r[2])

    print(f"{'Модель':35s} accuracy  F1      ±       log loss  время, с")
    for r in table:
        print(f"{r[0]:35s} {r[1]:.4f}    {r[2]:.4f}  {r[3]:.4f}  {r[4]:.4f}    {r[5]}")
    print("Лучшая:", table[0][0])

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    base = os.path.splitext(model_path)[0]
    with open(base + "_cv.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "accuracy", "f1", "f1_std", "logloss", "fit_time_s"])
        writer.writerows(table)

    # 2. Пороги: вероятности лучшей модели по кросс-валидации
    print("\n2. Подбор порогов")
    proba = cross_val_predict(search.best_estimator_, X_small, y_small, cv=cv, method="predict_proba")
    thresholds = choose_thresholds(y_small, proba)
    with open(base + "_thresholds.csv", "w", newline="") as f:
        csv.writer(f).writerows([[c, round(t, 4)] for c, t in enumerate(thresholds)])
    print("Пороги по классам:", [round(t, 3) for t in thresholds])

    # 3. Лучшая модель на всех данных
    print(f"\n3. Обучение лучшей модели на всех {len(X)} картинках")
    model = clone(search.best_estimator_).fit(X, y)
    joblib.dump(model, model_path)
    print("Модель сохранена:", model_path)


#Классификация

def inference(model_path, input_csv, output_csv, proba_csv):
    model = joblib.load(model_path)
    paths = [r[0] for r in read_csv(input_csv)]
    proba = model.predict_proba(load_images(paths))
    labels = model.classes_[proba.argmax(axis=1)]

    with open(output_csv, "w", newline="") as f:
        csv.writer(f).writerows(zip(paths, labels))
    print("Предсказания сохранены:", output_csv)

    if proba_csv:  # вероятности нужны для AUC, ROC и PR
        with open(proba_csv, "w", newline="") as f:
            csv.writer(f).writerows([[p] + list(np.round(pr, 6)) for p, pr in zip(paths, proba)])
        print("Вероятности сохранены:", proba_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "inference"], required=True)
    parser.add_argument("--model", required=True, help="train: куда сохранить, inference: откуда загрузить")
    parser.add_argument("--dataset", help="CSV для обучения (путь,метка)")
    parser.add_argument("--n", type=int, default=10000, help="картинок для сравнения моделей")
    parser.add_argument("--input", help="CSV с путями к картинкам")
    parser.add_argument("--output", help="CSV для предсказаний")
    parser.add_argument("--proba", help="CSV для вероятностей (необязательно)")
    args = parser.parse_args()

    if args.mode == "train":
        train(args.dataset, args.model, args.n)
    else:
        inference(args.model, args.input, args.output, args.proba)


if __name__ == "__main__":
    main()
