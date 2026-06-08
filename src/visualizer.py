import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.lines as mlines
from sklearn.manifold import TSNE


def get_cmap(n_classes):
    """Надёжная дискретная палитра с фиксированным количеством цветов"""
    if n_classes <= 10:
        return plt.cm.tab10.resampled(n_classes)  # ← важно!
    elif n_classes <= 20:
        return plt.cm.tab20.resampled(n_classes)  # ← важно!
    else:
        return plt.cm.hsv.resampled(n_classes)


def get_pca_plot(X, y, target_names, method_name):
    """PCA с корректными цветами (решение проблемы розового)"""
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(15, 10))

    num_classes = len(target_names)
    cmap = get_cmap(num_classes)

    # Основная отрисовка
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1], c=y, s=8, alpha=0.65, cmap=cmap, edgecolors="none"
    )

    # Центроиды
    for i in range(num_classes):
        mask = y == i
        if np.any(mask):
            points = X_pca[mask]
            center = points.mean(axis=0)
            ax.text(
                center[0],
                center[1],
                str(i),
                fontsize=9,
                weight="bold",
                va="center",
                ha="center",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2),
            )

    exp_var = pca.explained_variance_ratio_
    ax.set_title(f"PCA Projection: {method_name}", fontsize=15)
    ax.set_xlabel(f"PC1 ({exp_var[0]:.1%})", fontsize=12)
    ax.set_ylabel(f"PC2 ({exp_var[1]:.1%})", fontsize=12)

    # Легенда — цвета берём тем же cmap
    legend_handles = []
    for i in range(num_classes):
        if "kmeans" in str(method_name).lower() or "kmeans" in str(method_name):
            label_text = f"Cluster {i}"  # Предсказанные кластеры
            legend_title = "Кластеры KMeans"
        else:
            label_text = f"{i}: {target_names[i]}"  # Реальные темы
            legend_title = "Категории (ID: Название)"
        color = cmap(i)  # теперь будет точно совпадать
        line = mlines.Line2D(
            [],
            [],
            color=color,
            marker="o",
            linestyle="None",
            markersize=9,
            label=label_text,
            alpha=0.9,
        )
        legend_handles.append(line)

    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        title=legend_title,
        fontsize=9,
        ncol=1,
    )

    plt.tight_layout(rect=[0, 0, 0.82, 0.95])
    return fig


def get_confusion_matrix_plot(y_test, y_pred, target_names, method_name):
    """Создает и возвращает объект фигуры с матрицей ошибок"""
    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=target_names,
        xticks_rotation=90,
        cmap="Blues",
        values_format="d",
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix: {method_name}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def get_classification_report_df(y_test, y_pred, target_names):
    """
    Генерирует текстовый отчет в виде DataFrame для удобного вывода в таблицу.
    """
    report_dict = classification_report(
        y_test, y_pred, output_dict=True, target_names=target_names
    )
    return pd.DataFrame(report_dict).transpose()


# Адаптивные размеры подвыборки для t-SNE в зависимости от датасета
def get_tsne_sample_size(n_samples: int) -> int:
    """
    Автоматически определяет оптимальный размер подвыборки для t-SNE
    в зависимости от общего количества точек в датасете.
    """
    if n_samples <= 20000:
        return 8000
    elif n_samples <= 60000:
        return 10000
    else:
        return 12000


def compute_tsne_sample(X, dataset=None, max_samples=None):
    """Случайная подвыборка для t-SNE (ускорение + лучшая читаемость)"""
    if max_samples is None:
        max_samples = get_tsne_sample_size(X.shape[0])

    if X.shape[0] > max_samples:
        idx = np.random.choice(X.shape[0], max_samples, replace=False)
        print(f"t-SNE: выбрано {max_samples} точек из {X.shape[0]} (подвыборка)")
        return X[idx], idx
    print(f"t-SNE: используется ВСЕ {X.shape[0]} точек (меньше лимита {max_samples})")
    return X, None


def get_tsne_embedding(X_reduced):
    """Считает t-SNE один раз на подвыборке"""
    perplexity = min(50, max(5, X_reduced.shape[0] // 80))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,  # адаптивная perplexity
        init="pca",
        learning_rate="auto",
        random_state=42,
        max_iter=1000,
    )
    print(f"Perplexity = {perplexity}")
    return tsne.fit_transform(X_reduced)


def get_tsne_plot(X_tsne, y, target_names, method_name, label_type=""):
    """Новая версия: рисует уже готовый t-SNE embedding"""
    fig, ax = plt.subplots(figsize=(15, 10))

    num_classes = len(target_names)
    cmap = get_cmap(num_classes)  # используем твою существующую функцию

    # Рисуем точки
    scatter = ax.scatter(
        X_tsne[:, 0], X_tsne[:, 1], c=y, s=10, alpha=0.7, cmap=cmap, edgecolors="none"
    )

    # Центроиды + подписи классов (как в твоей PCA функции)
    for i in range(num_classes):
        mask = y == i
        if np.any(mask):
            points = X_tsne[mask]
            center = points.mean(axis=0)
            ax.text(
                center[0],
                center[1],
                str(i),
                fontsize=9,
                weight="bold",
                va="center",
                ha="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
            )
    # Заголовок
    if "real" in str(label_type).lower():
        title_suffix = " (Real Topics)"
        legend_title = "Категории (ID: Название)"
    else:
        title_suffix = " (Predicted Clusters)"
        legend_title = "Кластеры KMeans"

    ax.set_title(f"t-SNE Projection: {method_name}{title_suffix}", fontsize=15)
    ax.set_xlabel("t-SNE feature 1", fontsize=12)
    ax.set_ylabel("t-SNE feature 2", fontsize=12)

    # Легенда (почти копия твоей логики из PCA)
    legend_handles = []
    for i in range(num_classes):
        color = cmap(i)
        if "real" in str(label_type).lower():
            label_text = f"{i}: {target_names[i]}"  # настоящие названия
        else:
            label_text = f"Cluster {i}"
        line = mlines.Line2D(
            [],
            [],
            color=color,
            marker="o",
            linestyle="None",
            markersize=9,
            label=label_text,
            alpha=0.9,
        )
        legend_handles.append(line)

    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        title=legend_title,
        fontsize=9,
        ncol=1,
    )

    plt.tight_layout(rect=[0, 0, 0.82, 0.95])  # оставляем место для легенды
    return fig
