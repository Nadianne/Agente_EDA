# utils/charts.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def hist(df: pd.DataFrame, col: str, bins: int = 30, figsize=(6, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(df[col].dropna(), bins=bins, ax=ax)
    ax.set_title(f"Histograma de {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Contagem")
    fig.tight_layout()
    return fig

def box(df: pd.DataFrame, col: str, figsize=(6, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(x=df[col], ax=ax)
    ax.set_title(f"Boxplot de {col}")
    ax.set_xlabel(col)
    fig.tight_layout()
    return fig

def scatter(df_num: pd.DataFrame, x: str, y: str, figsize=(6, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=df_num, x=x, y=y, ax=ax, s=12)
    ax.set_title(f"Dispersão: {x} vs {y}")
    fig.tight_layout()
    return fig

def heatmap_corr(df: pd.DataFrame, figsize=(8, 6)):
    num = df.select_dtypes("number")
    corr = num.corr()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Mapa de correlação")
    fig.tight_layout()
    return fig

def timeseries(df: pd.DataFrame, tcol: str, ycol: str, figsize=(8, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(data=df, x=tcol, y=ycol, ax=ax)
    ax.set_title(f"Série temporal: {ycol} por {tcol}")
    ax.set_xlabel(tcol)
    ax.set_ylabel(ycol)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig

def bar_counts(df: pd.DataFrame, col: str, topn: int = 20, figsize=(6, 4)):
    """Gráfico de barras para contagens (categóricas)."""
    vc = df[col].value_counts(dropna=True).head(topn)
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=vc.values, y=vc.index, ax=ax)
    ax.set_title(f"Top {topn} valores de {col}")
    ax.set_xlabel("Contagem")
    ax.set_ylabel(col)
    fig.tight_layout()
    return fig
