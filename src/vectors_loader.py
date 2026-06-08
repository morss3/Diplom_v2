from scipy import sparse
import numpy as np


def load_vectors(vectors_path, key, dense=False):
    """
    Загружает векторы нужного метода.

    dense=True:
        sparse-матрицы автоматически переводятся в dense format.
    """

    sparse_methods = ["binary", "bow", "tfidf_std", "tfidf_bigrams"]

    if key in sparse_methods:
        X_train = sparse.load_npz(
            f"{vectors_path}/train_{key}_vectors.npz"
        )

        X_test = sparse.load_npz(
            f"{vectors_path}/test_{key}_vectors.npz"
        )

        if dense:
            X_train = X_train.toarray()
            X_test = X_test.toarray()

    else:
        X_train = np.load(
            f"{vectors_path}/train_{key}_vectors.npy"
        )

        X_test = np.load(
            f"{vectors_path}/test_{key}_vectors.npy"
        )

    return X_train, X_test