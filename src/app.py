import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# Импортируем твои инструменты
from ui_helpers import (
    display_efficiency_chart, display_method_results, display_comparison, 
    display_clustering_results, display_metrics_cards, get_top_methods, render_method_column
)
from config import *

# --- КОНФИГУРАЦИЯ ---
st.set_page_config(layout="wide", page_title="Diplom ML Dashboard")

# --- ОСНОВНАЯ ЛОГИКА ПРИЛОЖЕНИЯ ---

st.sidebar.title("Навигация")
# Выбор датасета
dataset_name = st.sidebar.selectbox(
    "Выберите датасет",
    list(DATASETS.keys())
)
# Извлекаем данные конкретного датасета по выбранному названию
dataset_info = DATASETS[dataset_name]
dataset_key = dataset_info["key"]
n_classes = dataset_info["n_classes"]

paths = get_paths(dataset_key)

VECTORS_PATH = paths["VECTORS"]
PLOTS_PATH = paths["PLOTS"]
TABLES_PATH = paths["TABLES"]
PRED_PATH = paths["PRED"]
MODELS_PATH = paths["MODELS"]

task = st.sidebar.selectbox("Задача", ["Классификация", "Кластеризация"])

st.sidebar.info(f"Текущий датасет: {dataset_name}")

if task == "Классификация":
    st.title(f"Анализ методов в задаче классификации в: {dataset_name}" )
    top_1, top_2 = get_top_methods("classification", dataset_key)

    # 1. Сравнительная таблица
    st.header("1. Общий отчет")
    try:
        df = pd.read_csv(f"{TABLES_PATH}/classification_compare.csv")
        # Выбираем только числовые столбцы (float), чтобы не пытаться округлять строки
        numeric_cols = df.select_dtypes(include=['float']).columns

        # Выводим отформатированную таблицу
        st.dataframe(
            df.style.format(precision=2).format(subset=["Accuracy", "F1-macro"], formatter="{:.4f}"),
            use_container_width=True
        )

        display_efficiency_chart(df)
        
    except:
        st.warning(
            f"Файл {TABLES_PATH}/classification_compare.csv не найден. Сначала запусти расчеты в ноутбуке."
        )

    # 2. Выбор режима
    st.header("2. Визуализация и сравнение")
    mode = st.radio(
        "Режим:", ["Обзор одного метода", "Сравнение двух методов"], horizontal=True
    )

    if mode == "Обзор одного метода":
        selected = st.selectbox(
            "Выберите метод для анализа:",
            ALL_METHODS,
            index=0 if top_1 not in ALL_METHODS else ALL_METHODS.index(top_1),
        )
        display_metrics_cards(selected, df)
        display_method_results(selected, METHODS_DICT[selected], dataset_key)

    else:
        c1, c2 = st.columns(2)
        with c1:
            m1 = st.selectbox(
                "Метод слева:", ALL_METHODS, index=ALL_METHODS.index(top_1)
            )
            display_metrics_cards(m1, df)
        with c2:
            m2 = st.selectbox(
                "Метод справа:", ALL_METHODS, index=ALL_METHODS.index(top_2)
            )
            display_metrics_cards(m2, df)

        display_comparison(m1, m2, dataset_key)

else:
    st.title(f"Анализ методов в задаче кластеризации в: {dataset_name}")
    top_1, top_2 = get_top_methods("clusterization", dataset_key)
    st.header("1. Общий отчет")
    try:
        df_cl = pd.read_csv(f"{TABLES_PATH}/clusterization_compare.csv")
        numeric_cols = df_cl.select_dtypes(include=['float']).columns

        # Выводим отформатированную таблицу
        st.dataframe(
            df_cl.style.format(precision=2).format(subset=["ARI", "Silhouette"], formatter="{:.4f}"),
            use_container_width=True
        )

        display_efficiency_chart(df_cl)
    except:
        st.warning(f"Файл {TABLES_PATH}/clusterization_compare.csv не найден.")
    
    # Визуализация
    st.header("2. Визуализация кластеров")
    
    mode = st.radio("Режим работы:", ["Обзор одного метода", "Сравнение двух методов"], key="cluster_mode", horizontal=True)

    if mode == "Обзор одного метода":
        selected = st.selectbox("Метод:", ALL_METHODS, index=ALL_METHODS.index(top_1))
        # Используем новую функцию из ui_helpers
        display_metrics_cards(selected, df_cl)
        display_clustering_results(selected, METHODS_DICT[selected], dataset_key)
    else:
        col_l, col_r = st.columns(2)
        with col_l:
            m1 = st.selectbox("Метод №1:", ALL_METHODS, index=ALL_METHODS.index(top_1), key="m1_cl")
            st.divider()
            display_metrics_cards(m1, df_cl)
            render_method_column(m1, METHODS_DICT[m1], dataset_key)
            
        with col_r:
            m2 = st.selectbox("Метод №2:", ALL_METHODS, index=ALL_METHODS.index(top_2), key="m2_cl")
            st.divider()
            display_metrics_cards(m2, df_cl)
            render_method_column(m2, METHODS_DICT[m2], dataset_key)