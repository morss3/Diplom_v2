import json

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import streamlit as st
import numpy as np
import os
import pandas as pd
import plotly.express as px

from visualizer import (
    compute_tsne_sample,
    get_pca_plot,
    get_confusion_matrix_plot,
    get_classification_report_df,
    get_tsne_embedding,
    get_tsne_plot
)
from data_loader import get_target_names
from config import *
from vectors_loader import load_vectors


# --- ФУНКЦИИ ОТОБРАЖЕНИЯ (Теперь они на месте!) ---
def show_pca(name, key, dataset):
    # 0. Переопределяем пути
    plots_path = f"{PLOTS_PATH}/{dataset}"
    vectors_path = f"{VECTORS_PATH}/{dataset}"
    # 1. PCA
    st.markdown(f"### Визуализация PCA")

    pca_path = f"{plots_path}/pca_{key}.png"
    if os.path.exists(pca_path):
        st.image(pca_path, caption=f"Визуализация {name} (загружена из кеша)")
    else:
        with st.spinner(f"Генерируем PCA для {name}..."):
            try:
                # Загружаем данные для отрисовки
                # X = np.load(f"{vectors_path}/train_{key}_vectors.npy")
                X, _ = load_vectors(vectors_path, key, dense=True)
                y = np.load(f"{vectors_path}/y_train.npy")
                target_names = get_target_names(dataset)

                fig = get_pca_plot(X, y, target_names, name)
                # Сохраняем, чтобы в следующий раз было быстро
                fig.savefig(pca_path, bbox_inches="tight", dpi=300)

                # Отображаем
                st.pyplot(fig)
                st.caption(f"Визуализация {name} (загружена сейчас)")
            except FileNotFoundError:
                st.error(f"Файлы векторов для {name} не найдены в {vectors_path}")


def show_confusion_matrix(name, key, dataset):
     # 0. Переопределяем пути
    plots_path = f"{PLOTS_PATH}/{dataset}"
    vectors_path = f"{VECTORS_PATH}/{dataset}"
    pred_path = f"{PRED_PATH}/{dataset}"
    # 2. Матрица ошибок
    st.markdown("### Матрица ошибок")
    cm_path = f"{plots_path}/confusion_matrix_{key}.png"
    if os.path.exists(cm_path):
        st.image(cm_path, caption=f"Матрица ошибок {name} (загружена из кеша)")
    else:
        with st.spinner(f"Генерируем матрицу ошибок для {name}..."):
            try:
                y_test = np.load(f"{vectors_path}/y_test.npy")
                y_pred = np.load(f"{pred_path}/{key}_y_pred.npy")
                target_names = get_target_names(dataset)

                fig_cm = get_confusion_matrix_plot(y_test, y_pred, target_names, name)
                fig_cm.savefig(cm_path, bbox_inches="tight", dpi=300)

                st.pyplot(fig_cm)
                st.caption(f"Матрица ошибок {name} (загружена сейчас)")
            except Exception as e:
                st.error(f"Нет данных для матрицы ошибок. Ошибка {e}")


def show_report(key, dataset):
     # 0. Переопределяем пути
    tables_path = f"{TABLES_PATH}/{dataset}"
    vectors_path = f"{VECTORS_PATH}/{dataset}"
    pred_path = f"{PRED_PATH}/{dataset}"
    st.markdown("#### Метрики классификации")
    cr_path = f"{tables_path}/classification_report_{key}.csv"
    if os.path.exists(cr_path):
        report_df = pd.read_csv(cr_path)
        # Округляем для красоты и выводим таблицу
        numeric_cols = report_df.select_dtypes(include=["float", "int"]).columns
        st.dataframe(
            report_df.style.format({col: "{:.4f}" for col in numeric_cols}),
            use_container_width=True,
        )
    else:
        try:
            y_test = np.load(f"{vectors_path}/y_test.npy")
            y_pred = np.load(f"{pred_path}/{key}_y_pred.npy")
            report_df = get_classification_report_df(
                y_test, y_pred, target_names=get_target_names(dataset)
            )
            report_df.to_csv(cr_path)
            # Округляем для красоты и выводим таблицу
            numeric_cols = report_df.select_dtypes(include=["float"]).columns
            st.dataframe(
                report_df.style.format({col: "{:.4f}" for col in numeric_cols}),
                use_container_width=True,
            )

        except Exception as e:
            st.error(f"Отчет недоступен. Ошибка {e}")


def show_visuals(name, key, dataset):
    """Отображает PCA, Матрицу ошибок и Отчет для одного метода"""
    show_pca(name, key, dataset)
    show_confusion_matrix(name, key, dataset)
    show_report(key, dataset)


def display_method_results(name, key, dataset):
    """Детальная страница одного метода"""
    st.subheader(f"Детальный анализ: {name}")
    show_visuals(name, key, dataset)


def display_comparison(name1, name2, dataset):
    """Сравнение двух методов side-by-side"""
    key1, key2 = METHODS_DICT[name1], METHODS_DICT[name2]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### {name1}")
        show_visuals(name1, key1, dataset)

    with col2:
        st.markdown(f"### {name2}")
        show_visuals(name2, key2, dataset)


def load_or_generate_cluster_plot(method_name, key, plot_type, label_type, dataset):
    """
    plot_type: 'pca' или 'tsne'
    label_type: 'real' или 'pred'
    """
     # 0. Переопределяем пути
    plots_path = f"{PLOTS_PATH}/{dataset}"
    vectors_path = f"{VECTORS_PATH}/{dataset}"
    pred_path = f"{PRED_PATH}/{dataset}"

    filename = f"clustering_{key}_{plot_type}_{label_type}.png"
    path = os.path.join(plots_path, filename)

    if os.path.exists(path):
        st.image(path, caption=f"{plot_type.upper()} ({label_type}) загружен из кэша")
    else:
        with st.spinner(f"Генерируем {plot_type} для {method_name}..."):
            # 1. Загрузка и подготовка данных
            # X_train = np.load(f'{vectors_path}/train_{key}_vectors.npy')
            # X_test = np.load(f'{vectors_path}/test_{key}_vectors.npy')
            X_train, X_test = load_vectors(vectors_path, key, dense=True)
            X_all = np.vstack([X_train, X_test])
            X_norm = normalize(X_all)
            
            # Для PCA/t-SNE берем сжатые до 50 компонент данные (как в ноутбуке)
            pca_50 = PCA(n_components=50, random_state=42)
            X_reduced = pca_50.fit_transform(X_norm)
            
            # 2. Выбор меток
            if label_type == 'real':
                y_train = np.load(f'{vectors_path}/y_train.npy')
                y_test = np.load(f'{vectors_path}/y_test.npy')
                y = np.concatenate([y_train, y_test])
                title = f"{method_name} (Real Topics)"
            else:
                y = np.load(f"{pred_path}/{key}_clusters.npy")
                title = f"{method_name} (KMeans Clusters)"
            
            target_names = get_target_names(dataset)

            # 3. Отрисовка
            if plot_type == 'pca':
                fig = get_pca_plot(X_reduced, y, target_names, title)
            else:
                X_sample, idx = compute_tsne_sample(X_reduced)
                X_tsne = get_tsne_embedding(X_sample)
                # Синхронизация меток с подвыборкой
                if idx is not None:
                    y = y[idx]
                
                fig = get_tsne_plot(X_tsne, y, target_names,method_name, label_type)
            
            # Сохраняем и показываем
            fig.savefig(path, dpi=300, bbox_inches='tight')
            st.pyplot(fig)
            st.caption(f"{plot_type} для {method_name} загружен сейчас")

def display_clustering_results(name, key, dataset):
    """Отображение графиков для одного метода (парами в ряд)"""
    st.subheader(f"Результаты: {name}")
    
    # Первая строка: PCA
    st.markdown("### 1. Метод главных компонент (PCA)")
    col1, col2 = st.columns(2)
    with col1:
        load_or_generate_cluster_plot(name, key, 'pca', 'real', dataset)
    with col2:
        load_or_generate_cluster_plot(name, key, 'pca', 'pred', dataset)
    
    # Вторая строка: t-SNE
    st.markdown("### 2. Нелинейная визуализация (t-SNE)")
    col3, col4 = st.columns(2)
    with col3:
        load_or_generate_cluster_plot(name, key, 'tsne', 'real', dataset)
    with col4:
        load_or_generate_cluster_plot(name, key, 'tsne', 'pred', dataset)

def render_method_column(name, key, dataset):
    """Отрисовка полной колонки анализа для одного метода кластеризации"""
    st.subheader(f"📊 {name}")
    
    # Можно добавить краткую справку по метрикам из DF, если хочешь
    # Но главное — графики
    
    with st.expander("✨ Визуализация PCA (Глобальная структура)", expanded=True):
        st.write("**Реальные темы vs Кластеры**")
        # Рисуем друг за другом
        load_or_generate_cluster_plot(name, key, 'pca', 'real', dataset)
        load_or_generate_cluster_plot(name, key, 'pca', 'pred', dataset)
        
    with st.expander("🌀 Визуализация t-SNE (Локальные связи)", expanded=False):
        st.write("**Реальные темы vs Кластеры**")
        # Рисуем друг за другом
        load_or_generate_cluster_plot(name, key, 'tsne', 'real', dataset)
        load_or_generate_cluster_plot(name, key, 'tsne', 'pred', dataset)

def display_metrics_cards(name, df):
    """
    Универсальная функция вывода метрик для Классификации или Кластеризации.
    """
    if df is None or df.empty:
        st.warning("Данные для метрик недоступны.")
        return

    # Ищем строку метода
    method_data = df[df["Method"] == name]
    
    if method_data.empty:
        st.error(f"Метод '{name}' не найден в таблице результатов.")
        return

    row = method_data.iloc[0]
    
    # Создаем контейнер для метрик
    cols = st.columns(3)
    
    # Логика для КЛАССИФИКАЦИИ
    if "Accuracy" in row:
        cols[0].metric("Accuracy", f"{row['Accuracy']:.4f}")
        cols[1].metric("F1-macro", f"{row['F1-macro']:.4f}")
        
    # Логика для КЛАСТЕРИЗАЦИИ
    elif "ARI" in row:
        cols[0].metric("ARI", f"{row['ARI']:.4f}")
        cols[1].metric("Silhouette", f"{row['Silhouette']:.4f}")

    # Общая метрика времени для обоих задач
    total_t = row.get('Total Time (s)', 0)
    cols[2].metric("Время (сек)", f"{total_t:.2f}")


def display_efficiency_chart(df):
    """
    Рисует интерактивный график: Качество (ARI/Accuracy) vs Время.
    """
    st.subheader("📈 Анализ эффективности: Качество vs Время")
    
    # Определяем, какую метрику качества использовать
    y_axis = "Accuracy" if "Accuracy" in df.columns else "ARI"
    
    if y_axis not in df.columns or "Total Time (s)" not in df.columns:
        st.warning("Недостаточно данных для построения графика эффективности.")
        return

    # Создаем интерактивный Scatter-plot
    fig = px.scatter(
        df,
        x="Total Time (s)",
        y=y_axis,
        size=[10] * len(df), # одинаковый размер точек
        color="Method",
        hover_name="Method",
        labels={
            "Total Time (s)": "Общее время (сек)",
            y_axis: f"Качество ({y_axis})"
        },
        title=f"Сравнение методов: {y_axis} относительно времени работы"
    )

    # Настраиваем отображение текста (чтобы названия не перекрывали точки)
    # fig.update_traces(textposition='top center')
    
    # Делаем график аккуратным
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True,
        legend_title_text="Методы"
    )

    st.plotly_chart(fig, use_container_width=True)


# Загружаем топ-методы для значений по умолчанию
def get_top_methods(task, dataset_key):
    try:
        with open(f"{RESULTS_PATH}/{dataset_key}_top_methods_{task}.json", "r") as f:
            top_list = json.load(f)
            if len(top_list) >= 2:
                return top_list[0], top_list[1]
            elif len(top_list) == 1:
                return top_list[0], top_list[0]
    except:
        pass

    # запасной вариант
    return "TF-IDF Standard", "BERT"