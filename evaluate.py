"""
Скрипт оценки качества.

Обязательно (по заданию): Accuracy и Confusion Matrix.
Дополнительно: Precision, Recall, F1, а если передать --proba:
AUC, log loss (кросс-энтропия), кривые ROC и PR, сравнение порогов.

Пример:
    python evaluate.py --ground-truth data/mnist/ground-truth-test.csv --predictions predictions_mnist.csv
    python evaluate.py --ground-truth data/mnist/ground-truth-test.csv --predictions predictions_mnist.csv \
    --proba proba_mnist.csv --thresholds models/mnist_thresholds.csv --plots plots/mnist
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")  # рисуем в файлы, без окна
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, log_loss, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)


def read_csv(path):
    # Словарь: путь к картинке -> остальные значения строки
    with open(path, newline="") as f:
        return {row[0]: row[1:] for row in csv.reader(f) if row}


def main_metrics(y_true, y_pred, plots):
    print("Accuracy: ", round(accuracy_score(y_true, y_pred), 4))
    print("Precision:", round(precision_score(y_true, y_pred, average="macro"), 4))
    print("Recall:   ", round(recall_score(y_true, y_pred, average="macro"), 4))
    print("F1:       ", round(f1_score(y_true, y_pred, average="macro"), 4))
    print("\nПо классам:")
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion Matrix (строки - правильный класс, столбцы - ответ модели):")
    print(confusion_matrix(y_true, y_pred))

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues")
    plt.savefig(os.path.join(plots, "confusion_matrix.png"), dpi=120)
    plt.close()


def proba_metrics(y_true, P, plots):
    print("\nAUC:     ", round(roc_auc_score(y_true, P, multi_class="ovr"), 4))
    print("Log loss:", round(log_loss(y_true, P / P.sum(axis=1, keepdims=True)), 4))

    # Кривые строим для каждого класса: "эта цифра или нет"
    fig_roc, roc = plt.subplots(figsize=(7, 6))
    fig_pr, pr = plt.subplots(figsize=(7, 6))
    for c in range(P.shape[1]):
        fpr, tpr, _ = roc_curve(y_true == c, P[:, c])
        roc.plot(fpr, tpr, label=str(c))
        precision, recall, _ = precision_recall_curve(y_true == c, P[:, c])
        pr.plot(recall, precision, label=str(c))
    roc.plot([0, 1], [0, 1], "k--")
    roc.set(xlabel="FPR", ylabel="TPR", title="ROC-кривые")
    pr.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall кривые")
    roc.legend()
    pr.legend()
    fig_roc.savefig(os.path.join(plots, "roc.png"), dpi=120)
    fig_pr.savefig(os.path.join(plots, "pr.png"), dpi=120)
    plt.close("all")


def compare_thresholds(y_true, P, thresholds_csv):
    # F1 каждого класса при пороге 0.5 и при подобранном пороге
    with open(thresholds_csv, newline="") as f:
        thresholds = [float(r[1]) for r in csv.reader(f) if r]
    print("\nКласс  порог   F1 (0.5)  F1 (подобранный)")
    for c, t in enumerate(thresholds):
        f1_default = f1_score(y_true == c, P[:, c] >= 0.5)
        f1_tuned = f1_score(y_true == c, P[:, c] >= t)
        print(f"{c:5d}  {t:.3f}   {f1_default:.4f}    {f1_tuned:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", required=True, help="CSV с правильными метками")
    parser.add_argument("--predictions", required=True, help="CSV с ответами модели")
    parser.add_argument("--proba", help="CSV с вероятностями")
    parser.add_argument("--thresholds", help="CSV с подобранными порогами")
    parser.add_argument("--plots", default="plots", help="папка для графиков")
    args = parser.parse_args()
    os.makedirs(args.plots, exist_ok=True)

    truth = read_csv(args.ground_truth)
    pred = read_csv(args.predictions)
    paths = [p for p in truth if p in pred]  # пикчи в обоих файлах
    y_true = np.array([int(truth[p][0]) for p in paths])
    y_pred = np.array([int(pred[p][0]) for p in paths])
    print("Картинок:", len(paths))

    main_metrics(y_true, y_pred, args.plots)

    if args.proba:
        proba = read_csv(args.proba)
        P = np.array([[float(v) for v in proba[p]] for p in paths])
        proba_metrics(y_true, P, args.plots)
        if args.thresholds:
            compare_thresholds(y_true, P, args.thresholds)

    print("\nГрафики сохранены в папку:", args.plots)


if __name__ == "__main__":
    main()
