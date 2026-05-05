import argparse
import os
import numpy as np
import pandas as pd
import time
import json
from data_loader import get_clean_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from config import get_paths
from data_loader import load_dataset

# Словарь для хранения времени векторизации
vectorization_time = {}


def save_vec(name, X_train, X_test, duration, paths):
    """Вспомогательная функция для сохранения векторов и времени"""
    # .npy — бинарный формат numpy.
    np.save(f"{paths['VECTORS']}/train_{name}_vectors.npy", X_train.astype(np.float32))
    np.save(f"{paths['VECTORS']}/test_{name}_vectors.npy", X_test.astype(np.float32))

    # Сохраняем время в словарь
    vectorization_time[name] = duration
    print(f"--- Метод {name} сохранен (Время: {duration:.2f} сек) ---")


def main(dataset_key):
    print("Загрузка данных...")
    paths = get_paths(dataset_key)
    # Создаем папки, если их нет
    os.makedirs(paths["VECTORS"], exist_ok=True)

    data, target = load_dataset(dataset_key)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        data, target, test_size=0.2, random_state=42
    )

    np.save(f"{paths['VECTORS']}/y_train.npy", y_train)
    np.save(f"{paths['VECTORS']}/y_test.npy", y_test)

    # --- 1 & 2: Binary и Bag of Words ---
    #     max_features=3000 → берем только 3000 самых частых слов
    # binary: "cat dog dog"→ [1,1] Просто отмечает, есть слово в тексте (1) или нет (0)
    # bow "cat dog dog" → [1,2] Считает количество вхождений слова.
    print("\nВекторизация: Binary & BoW...")
    for name, is_binary in [("binary", True), ("bow", False)]:
        start = time.time()
        vec = CountVectorizer(binary=is_binary, stop_words="english", max_features=3000)
        xt = vec.fit_transform(X_train_raw).toarray()
        xv = vec.transform(X_test_raw).toarray()
        duration = time.time() - start
        save_vec(name, xt, xv, duration, paths)

    # --- 3 & 4: TF-IDF Standard и Bigrams ---
    # Не просто считает слова, а оценивает их важность.
    # Если слово встречается во всех текстах (как "the"), его вес падает. Если оно редкое и важное — вес растет.
    # ngram_range=(1, 2): Метод смотрит не только на отдельные слова, но и на пары слов (биграммы), что позволяет уловить контекст вроде "not good".
    print("\nВекторизация: TF-IDF...")
    tfidf_params = [("tfidf_std", (1, 1)), ("tfidf_bigrams", (1, 2))]
    for name, ngram in tfidf_params:
        start = time.time()
        vec = TfidfVectorizer(
            ngram_range=ngram, stop_words="english", max_features=3000
        )
        xt = vec.fit_transform(X_train_raw).toarray()
        xv = vec.transform(X_test_raw).toarray()
        duration = time.time() - start
        save_vec(name, xt, xv, duration, paths)

    # --- 5: Word2Vec (Average) ---
    # Word2Vec обучается понимать смысл слова через его окружение.
    print("\nВекторизация: Word2Vec...")
    start = time.time()
    # Разбиваем текст на слова
    tokenized_train = [t.lower().split() for t in X_train_raw]
    tokenized_test = [t.lower().split() for t in X_test_raw]
    # Обучение модели (какие слова встречаются рядом → значит похожи по смыслу)
    w2v = Word2Vec(
        sentences=tokenized_train,  # Слова
        vector_size=100,  # Размер вектора слова
        window=5,  # Сколько слов вокруг (5 соседей слева и справа)
        min_count=2,  # Игнорируем редкие слова
        workers=4,  # Параллельные потоки (для скорости)
    )

    # Функция усреднения
    def get_avg_vec(tokens):
        # wv[w] — вектор слова w. Если его нет, то [0, 0, 0, ...] (словарь слово-вектор)
        v = [
            w2v.wv[w] for w in tokens if w in w2v.wv
        ]  # берем все слова, проверяем есть ли оно в словаре нашей обученной модели, добавляем в список
        return (
            np.mean(v, axis=0) if v else np.zeros(100)
        )  # если слов нет (v- пустой) возвращаем 0 вектор

    # Создаем вектора документов
    # for каждый текст:
    # взять слова
    # взять их вектора
    # усреднить
    # сохранить
    xt = np.array([get_avg_vec(t) for t in tokenized_train])
    xv = np.array([get_avg_vec(t) for t in tokenized_test])
    duration = time.time() - start
    save_vec("w2v", xt, xv, duration, paths)

    # --- 6: Doc2Vec ---
    # В отличие от Word2Vec, Doc2Vec сразу обучается представлять весь документ (текст) целиком.
    # Он добавляет специальный "ID документа" в процессе обучения, что теоретически позволяет лучше сохранять структуру длинных текстов.
    print("\nВекторизация: Doc2Vec...")
    start = time.time()
    # Создаем список вида (каждому тексту присваиваем id)
    # [
    #   (["i", "love", "nlp"], [0]),
    #   (["machine", "learning"], [1]),
    #   ...
    # ]
    # enumerate дает i - номер документа, d - текст
    tagged = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_train)]
    # Обучение
    # Модель учится предсказывать слова в документе исползуя сам документ как контекст
    # Модель читатет тексты и понимает их смысл (главная цель - научиться понимать тексты)
    d2v = Doc2Vec(
        tagged,
        vector_size=100,  # размер вектора документа
        epochs=20,  # сколько раз пройтись по данным (чем больше тем лучше, но дольше)
    )
    # Затем обученной модели даем новые тексты и смотрим как она его поймет
    # Используя знания модели, подбери вектор для этого текста
    # обучение:
    # модель учится понимать тексты

    # infer_vector:
    # модель смотрит на новый текст и говорит
    # "примерно такой у него вектор"
    xt = np.array([d2v.infer_vector(t) for t in tokenized_train])
    xv = np.array([d2v.infer_vector(t) for t in tokenized_test])
    duration = time.time() - start
    save_vec("d2v", xt, xv, duration, paths)

    # --- 7: BERT ---
    # Теория: Это SOTA-метод (State-of-the-Art). BERT использует механизм Attention (внимания). Он не просто смотрит на слова, он понимает их смысл в зависимости от контекста. Например, в фразах "ключ от замка" и "ключ от квартиры" слово "ключ" получит разные векторы.
    # all-MiniLM-L6-v2: Это компактная и быстрая версия BERT, которая идеально подходит для учебных работ: она дает высокую точность, но не требует суперкомпьютера для расчетов.
    print("\nВекторизация: BERT (это долго)...")
    start = time.time()
    # Загрузка уже обученной модели
    # обучена на огромных текстах
    # уже “понимает язык”
    # умеет кодировать смысл предложений
    # all-MiniLM-L6-v2 - кайф, потому что Более крупные модели дают немного лучшее качество, но значительно увеличивают время вычислений, что нецелесообразно в рамках данной работы.
    bert = SentenceTransformer("all-MiniLM-L6-v2")
    # для каждого вектора кодируем текста - получаем вектора\
    # пример
    # "I love NLP"
    # "I really like natural language processing"
    # BERT даст похожие вектора, потому что смысл одинаковый
    xt = bert.encode(X_train_raw, show_progress_bar=True)
    # encode:  1. разбивает текст на токены, 2. прогоняет через нейронку, 3. берет скрытое представление. Результат - вектор длиной ~ 384
    xv = bert.encode(X_test_raw, show_progress_bar=True)
    duration = time.time() - start
    save_vec("bert", xt, xv, duration, paths)
    # Проверяем размеры сохраненных эмбеддингов
    # Проверяем размеры сохраненных эмбеддингов

    # Сохраняем все замеры времени в файл
    with open(f"{paths['VECTORS']}/vectorization_time.json", "w") as f:
        json.dump(vectorization_time, f)
    print(
        f"\nВсе замеры времени сохранены в {paths['VECTORS']}/vectorization_time.json"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Векторизация текстовых датасетов")
    parser.add_argument(
        "dataset",
        type=str,
        choices=["20newsgroups", "ag_news", "imdb", "udmurt_media"],
        help="Выберите датасет: 20newsgroups, ag_news или imdb",
    )

    args = parser.parse_args()
    main(args.dataset)

