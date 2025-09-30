import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def hist(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.histplot(df[col].dropna(), bins=30, kde=True, ax=ax)
    ax.set_title(f"Histograma de {col}")
    return fig

def box(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.boxplot(x=df[col].dropna(), ax=ax)
    ax.set_title(f"Boxplot de {col}")
    return fig

def scatter(df: pd.DataFrame, x: str, y: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=df[x], y=df[y], s=10, ax=ax)
    ax.set_title(f"Dispersão: {x} vs {y}")
    return fig

def heatmap_corr(df: pd.DataFrame):
    corr = df.select_dtypes("number").corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", ax=ax)
    ax.set_title("Mapa de Correlação")
    return fig

def timeseries(df: pd.DataFrame, tcol: str, ycol: str):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df[tcol], df[ycol])
    ax.set_title(f"Série temporal: {ycol} vs {tcol}")
    return fig