import os

# --- пути ---
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR_PATH = os.path.join(BASE_PATH, "data")
VECTORS_PATH = os.path.join(DATA_DIR_PATH, "processed")
RESULTS_PATH = os.path.join(BASE_PATH, "results")
PLOTS_PATH = os.path.join(RESULTS_PATH, "plots")
TABLES_PATH = os.path.join(RESULTS_PATH, "tables")
PRED_PATH = os.path.join(RESULTS_PATH, "predictions")
MODELS_PATH = os.path.join(DATA_DIR_PATH, "models")

# --- методы ---
ALL_METHODS = [
    "Binary",
    "Bag of Words",
    "TF-IDF Standard",
    "TF-IDF Bigrams",
    "Word2Vec",
    "Doc2Vec",
    "BERT",
]

METHODS_DICT = {
    "Binary": "binary",
    "Bag of Words": "bow",
    "TF-IDF Standard": "tfidf_std",
    "TF-IDF Bigrams": "tfidf_bigrams",
    "Word2Vec": "w2v",
    "Doc2Vec": "d2v",
    "BERT": "bert",
}
DATASETS = {
    "20 Newsgroups": {"key": "20newsgroups", "n_classes": 20},
    "AG News": {"key": "ag_news", "n_classes": 4},
    "IMDB": {"key": "imdb", "n_classes": 2},
}


def get_paths(dataset_key):
    return {
        "VECTORS": f"{VECTORS_PATH}/{dataset_key}",
        "PLOTS": f"{PLOTS_PATH}/{dataset_key}",
        "TABLES": f"{TABLES_PATH}/{dataset_key}",
        "PRED": f"{PRED_PATH}/{dataset_key}",
        "MODELS": f"{MODELS_PATH}/{dataset_key}",
    }
