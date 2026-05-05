import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import re

def get_clean_20newsgroups():
    data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    # Здесь можно добавить что то
    return data
from sklearn.datasets import fetch_20newsgroups

def get_target_names(dataset_key):
    """
    Возвращает названия классов для выбранного датасета
    """

    if dataset_key == "20newsgroups":
        data = fetch_20newsgroups(
            subset='all',
            remove=('headers', 'footers', 'quotes')
        )
        return data.target_names

    elif dataset_key == "ag_news":
        return [
            "World",
            "Sports",
            "Business",
            "Sci/Tech"
        ]

    elif dataset_key == "imdb":
        return [
            "negative",
            "positive"
        ]
    elif dataset_key == "udmurt_media":
        df = pd.read_csv("../data/raw/udmurt_media/udmurt_media.csv")
        # фикс категории
        df["category"] = df["category"].replace({
            "Ужбергатон": "Экономика но коньдон"
        })
        # берём уникальные и сортируем (ВАЖНО — как в get_udmurt_media)
        categories = sorted(df["category"].dropna().unique())
        return categories
    else:
        raise ValueError(f"Неизвестный датасет: {dataset_key}")

def get_ag_news():
    train_df = pd.read_csv("../data/raw/ag_news/train.csv")
    test_df = pd.read_csv("../data/raw/ag_news/test.csv")

    df = pd.concat([train_df, test_df], ignore_index=True)

    df["text"] = df["Title"] + " " + df["Description"]
    df["label"] = df["Class Index"] - 1

    df = df.dropna()
    df = df[df["text"].str.strip() != ""]

    return df["text"].tolist(), df["label"].values
def get_imdb():
    df = pd.read_csv("../data/raw/imdb/imdb.csv")

    # очистка текста
    df["review"] = df["review"].str.replace("<br />", " ", regex=False)

    # удаляем пустые значения
    df = df.dropna()
    df = df[df["review"].str.strip() != ""]

    # метки
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

    return df["review"].tolist(), df["sentiment"].values

def get_udmurt_media():
    df = pd.read_csv("../data/raw/udmurt_media/udmurt_media.csv")

    # --- фикс кривой категории ---
    df["category"] = df["category"].replace({
        "Ужбергатон": "Экономика но коньдон"
    })

    # --- базовая очистка ---
    df = df.dropna()
    df = df[df["content"].str.strip() != ""]

    # --- текст ---
    df["text"] = df["title"].fillna("") + " " + df["content"]

    # чистка переносов и пробелов
    df["text"] = df["text"].apply(
        lambda x: re.sub(r"\s+", " ", x.replace("\n", " ")).strip()
    )

    df = df[df["text"].str.strip() != ""]

    # --- СТАБИЛЬНЫЙ порядок категорий ---
    categories = sorted(df["category"].unique())

    cat2id = {cat: i for i, cat in enumerate(categories)}
    df["label"] = df["category"].map(cat2id)

    # вывод для контроля
    print("\n📊 Categories mapping:")
    for cat, idx in cat2id.items():
        print(f"{idx}: {cat}")

    return df["text"].tolist(), df["label"].values
def load_dataset(name):
    if name == "20newsgroups":
        data = get_clean_20newsgroups()
        return data.data, data.target

    elif name == "ag_news":
        return get_ag_news()

    elif name == "imdb":
        return get_imdb()
    
    elif name == "udmurt_media":
        return get_udmurt_media()