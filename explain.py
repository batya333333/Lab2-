"""
SHAP: какие пиксели повлияли на ответ модели.
Работает для моделей на деревьях (LightGBM, XGBoost, Random Forest, Extra Trees).

Графики:
    shap_summary_bar.png      - самые важные пиксели (все классы)
    shap_summary_class8.png   - summary plot для одного класса
    shap_importance_map.png   - важность пикселей картинкой 28x28
    shap_local_1.png, _2.png  - 2 локальных примера: верный ответ и ошибка

Пример:
    python explain.py --model models/mnist.joblib --data data/mnist/ground-truth-test.csv --plots plots/mnist
"""
import argparse
import csv
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from PIL import Image


def load_images(paths):
    return np.array([np.array(Image.open(p).convert("L"), dtype=np.float32).flatten() / 255.0
                     for p in paths])


def save(path):
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close("all")


def local_example(x, label, pred, values, path):
    # Слева картинка, справа SHAP: красный - пиксель за ответ модели, синий - против
    lim = np.abs(values).max()
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(x.reshape(28, 28), cmap="gray")
    ax[0].set_title(f"Правильно: {label}, модель: {pred}")
    ax[1].imshow(x.reshape(28, 28), cmap="gray", alpha=0.3)
    im = ax[1].imshow(values.reshape(28, 28), cmap="bwr", vmin=-lim, vmax=lim, alpha=0.8)
    ax[1].set_title(f"SHAP для класса {pred}")
    for a in ax:
        a.axis("off")
    fig.colorbar(im, ax=ax[1])
    save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, help="CSV путь,метка")
    parser.add_argument("--n", type=int, default=500, help="сколько картинок объяснять")
    parser.add_argument("--class-id", type=int, default=8, help="класс для summary plot")
    parser.add_argument("--plots", default="plots")
    args = parser.parse_args()
    os.makedirs(args.plots, exist_ok=True)

    with open(args.data, newline="") as f:
        rows = [r for r in csv.reader(f) if r][:args.n]
    X = load_images([r[0] for r in rows])
    y = np.array([int(r[1]) for r in rows])

    pipeline = joblib.load(args.model)
    X = pipeline[:-1].transform(X)        # шаги до модели (scaler), если они есть
    model = pipeline.named_steps["model"]
    pred = model.predict(X)

    # sv[картинка, пиксель, класс] - вклад пикселя в вероятность класса
    sv = np.array(shap.TreeExplainer(model).shap_values(X))
    if sv.shape[0] != len(X):             # некоторые версии shap отдают (класс, картинка, пиксель)
        sv = sv.transpose(1, 2, 0)
    names = [f"px({i // 28},{i % 28})" for i in range(784)]

    shap.summary_plot([sv[:, :, c] for c in range(sv.shape[2])], X, feature_names=names,
                      plot_type="bar", max_display=15, show=False)
    save(os.path.join(args.plots, "shap_summary_bar.png"))

    c = args.class_id
    shap.summary_plot(sv[:, :, c], X, feature_names=names, max_display=15, show=False)
    save(os.path.join(args.plots, f"shap_summary_class{c}.png"))

    importance = np.abs(sv).mean(axis=(0, 2)).reshape(28, 28)
    plt.imshow(importance, cmap="hot")
    plt.colorbar()
    plt.title("Важность пикселей (средний |SHAP|)")
    save(os.path.join(args.plots, "shap_importance_map.png"))
    print("Пикселей, которые модель не использует:", (importance == 0).sum(), "из 784")

    right = np.where(pred == y)[0][0]
    wrong = np.where(pred != y)[0]
    examples = [right, wrong[0] if len(wrong) else np.where(pred == y)[0][1]]
    for n, i in enumerate(examples, start=1):
        local_example(X[i], y[i], pred[i], sv[i, :, pred[i]],
                      os.path.join(args.plots, f"shap_local_{n}.png"))
        print(f"Пример {n}: {rows[i][0]}, правильно {y[i]}, модель {pred[i]}")

    print("Графики SHAP сохранены в папку:", args.plots)


if __name__ == "__main__":
    main()
