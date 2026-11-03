# Классификатор MNIST / Fashion-MNIST

Лабораторная работа 2. Классификация изображений 28×28 на 10 классов.
Модель выбирается автоматически: несколько алгоритмов сравниваются на кросс-валидации (GridSearchCV), лучший обучается на всех данных.

## Структура

```
archive/           исходные файлы MNIST в формате idx
prepare_data.py    idx -> PNG-картинки по папкам классов + CSV-файлы
models.py          модели-кандидаты и сетка гиперпараметров
mnist.py           основной скрипт: --mode train / --mode inference
evaluate.py        оценка качества (Accuracy, Confusion Matrix и др.)
explain.py         SHAP-объяснение модели
```

Кандидаты в models.py: логистическая регрессия, MLP, Random Forest, Extra Trees, LightGBM и XGBoost (с early stopping).

## Установка

```bash
python3 -m venv venv
source venv/bin/activate
pip install scikit-learn pillow joblib lightgbm xgboost shap matplotlib
```

На macOS для LightGBM может понадобиться: `brew install libomp`

## Запуск

### 1. Подготовка данных

```bash
python prepare_data.py --src archive --out data/mnist
```

Создаёт:
- `data/mnist/train/<класс>/*.png`, `data/mnist/test/<класс>/*.png` — картинки
- `data/mnist/ground-truth-train.csv` — путь, метка (обучающая выборка)
- `data/mnist/ground-truth-test.csv` — путь, метка (тестовая выборка)
- `data/mnist/input.csv` — только пути к тестовым картинкам

### 2. Обучение

```bash
python mnist.py --mode train --dataset data/mnist/ground-truth-train.csv --model models/mnist.joblib
```

1. Сравнение моделей: кросс-валидация на 3 фолдах на подвыборке (`--n`, по умолчанию 10 000 картинок).
2. Подбор порога классификации для каждого класса.
3. Обучение лучшей модели на всей выборке.

Создаёт:
- `models/mnist.joblib` — обученная модель
- `models/mnist_cv.csv` — таблица сравнения моделей
- `models/mnist_thresholds.csv` — подобранные пороги

### 3. Классификация (inference)

```bash
python mnist.py --mode inference --model models/mnist.joblib --input data/mnist/input.csv --output predictions_mnist.csv --proba proba_mnist.csv
```

Создаёт:
- `predictions_mnist.csv` — путь, предсказанная метка
- `proba_mnist.csv` — путь, вероятности 10 классов (флаг `--proba` необязательный)

### 4. Оценка качества

```bash
python evaluate.py --ground-truth data/mnist/ground-truth-test.csv --predictions predictions_mnist.csv --proba proba_mnist.csv --thresholds models/mnist_thresholds.csv --plots plots/mnist
```

Выводит Accuracy, Precision, Recall, F1 и Confusion Matrix.
С `--proba`: AUC, log loss, ROC- и PR-кривые. С `--thresholds`: сравнение порога 0.5 и подобранных порогов.
Графики сохраняются в `plots/mnist/`.

Минимальный запуск по заданию:

```bash
python evaluate.py --ground-truth data/mnist/ground-truth-test.csv --predictions predictions_mnist.csv
```

### 5. SHAP

```bash
python explain.py --model models/mnist.joblib --data data/mnist/ground-truth-test.csv --plots plots/mnist
```

Строит summary plot, карту важности пикселей и 2 локальных примера (верный ответ и ошибка).
Работает для моделей на деревьях. Для Extra Trees расчёт долгий, ускорить можно флагом `--n 100`.

## Fashion-MNIST

Положить idx-файлы Fashion-MNIST в `archive_fashion/` (имена как у MNIST) и выполнить те же шаги, заменив `mnist` на `fashion`:

```bash
python prepare_data.py --src archive_fashion --out data/fashion
python mnist.py --mode train --dataset data/fashion/ground-truth-train.csv --model models/fashion.joblib
python mnist.py --mode inference --model models/fashion.joblib --input data/fashion/input.csv --output predictions_fashion.csv --proba proba_fashion.csv
python evaluate.py --ground-truth data/fashion/ground-truth-test.csv --predictions predictions_fashion.csv --proba proba_fashion.csv --thresholds models/fashion_thresholds.csv --plots plots/fashion
python explain.py --model models/fashion.joblib --data data/fashion/ground-truth-test.csv --plots plots/fashion
```
