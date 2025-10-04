# utils/charts.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(context="notebook")

def hist(df: pd.DataFrame, col: str, bins: int = 30, figsize=(6, 4)):
    """Histograma para coluna numérica."""
    serie = pd.to_numeric(df[col], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(serie, bins=bins, ax=ax)
    ax.set_title(f"Histograma de {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Contagem")
    fig.tight_layout()
    return fig

def box(df: pd.DataFrame, col: str, figsize=(6, 4)):
    """Boxplot para coluna numérica."""
    serie = pd.to_numeric(df[col], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(x=serie, ax=ax)
    ax.set_title(f"Boxplot de {col}")
    ax.set_xlabel(col)
    fig.tight_layout()
    return fig

def scatter(df_num: pd.DataFrame, x: str, y: str, figsize=(6, 4)):
    """Dispersão entre duas colunas numéricas."""
    # garante numérico
    x_s = pd.to_numeric(df_num[x], errors="coerce")
    y_s = pd.to_numeric(df_num[y], errors="coerce")
    tmp = pd.DataFrame({x: x_s, y: y_s}).dropna()
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=tmp, x=x, y=y, ax=ax, s=12)
    ax.set_title(f"Dispersão: {x} vs {y}")
    fig.tight_layout()
    return fig

def heatmap_corr(df: pd.DataFrame, figsize=(8, 6)):
    """Mapa de correlação para colunas numéricas."""
    num = df.select_dtypes("number")
    if num.shape[1] == 0:
        # evita erro caso não haja numéricas
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Sem colunas numéricas para correlação",
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig
    corr = num.corr()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Mapa de correlação")
    fig.tight_layout()
    return fig

def timeseries(df: pd.DataFrame, tcol: str, ycol: str, figsize=(8, 4)):
    """Série temporal (converte tcol se necessário)."""
    x = df[tcol]
    if not pd.api.types.is_datetime64_any_dtype(x):
        x = pd.to_datetime(x, errors="coerce")
    y = pd.to_numeric(df[ycol], errors="coerce")
    tmp = pd.DataFrame({tcol: x, ycol: y}).dropna()
    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(data=tmp, x=tcol, y=ycol, ax=ax)
    ax.set_title(f"Série temporal: {ycol} por {tcol}")
    ax.set_xlabel(tcol)
    ax.set_ylabel(ycol)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig

def bar_counts(df: pd.DataFrame, col: str, topn: int = 20, figsize=(6, 4)):
    """Gráfico de barras para contagens (categóricas)."""
    vc = df[col].astype(str).value_counts(dropna=True).head(topn)
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=vc.values, y=vc.index, ax=ax)
    ax.set_title(f"Top {topn} valores de {col}")
    ax.set_xlabel("Contagem")
    ax.set_ylabel(col)
    fig.tight_layout()
    return fig
