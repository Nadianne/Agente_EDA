import pandas as pd
import numpy as np
import unicodedata
from sklearn.cluster import KMeans
from utils import memory


# ---------------------- Utilitários base ----------------------
def _save_conclusion(pergunta: str, conclusion: str) -> str:
    """
    Salva a conclusão no 'memory' e retorna a mesma string.
    (Útil para manter uma linha só em cada bloco.)
    """
    memory.salvar(pergunta, conclusion)
    return conclusion

def _norm(s: str) -> str:
    """Normaliza para minúsculo, removendo acentos."""
    s = s or ""
    s = s.strip().lower()
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


# ---------------------- Tipos / Resumos ----------------------
def tipos(df: pd.DataFrame):
    """Retorna uma tabela detalhada de tipos + resumo (numérica, data/tempo, categórica)."""
    categorias = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            categorias.append("Numérica")
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            categorias.append("Data/Tempo")
        else:
            categorias.append("Categórica")

    tipos_df = pd.DataFrame({
        "Coluna": df.columns,
        "Tipo detectado": df.dtypes.astype(str),
        "Categoria": categorias,
        "Não nulos": df.notna().sum().values,
        "Nulos (%)": (df.isna().mean().values * 100).round(2)
    })

    ordem = {"Numérica": 0, "Data/Tempo": 1, "Categórica": 2}
    tipos_df["__ordem__"] = tipos_df["Categoria"].map(ordem).fillna(99)
    tipos_df = (
        tipos_df.sort_values(["__ordem__", "Coluna"])
                .drop(columns="__ordem__")
                .reset_index(drop=True)
    )
    resumo = tipos_df["Categoria"].value_counts()
    return tipos_df, resumo


def intervalo(df: pd.DataFrame):
    return df.select_dtypes("number").agg(["min", "max"]).T


def tendencia_central(df: pd.DataFrame):
    # renomeando colunas da agregação para pt-br
    out = df.select_dtypes("number").agg(["mean", "median"]).T
    return out.rename(columns={"mean": "Média", "median": "Mediana"})


def variabilidade(df: pd.DataFrame):
    return df.select_dtypes("number").agg(["std", "var"]).T


def frequencias(df: pd.DataFrame, topn=10):
    out = {}
    for c in df.columns:
        vc = df[c].value_counts(dropna=True).head(topn)
        if not vc.empty:
            out[c] = vc
    return out


# ---------------------- Outliers (IQR) ----------------------
def outliers_iqr_mask(num: pd.DataFrame):
    Q1 = num.quantile(0.25)
    Q3 = num.quantile(0.75)
    IQR = Q3 - Q1
    return (num.lt(Q1 - 1.5 * IQR)) | (num.gt(Q3 + 1.5 * IQR))


def outliers_iqr(df: pd.DataFrame):
    num = df.select_dtypes("number")
    if num.empty:
        return pd.Series(dtype=float)
    mask = outliers_iqr_mask(num)
    pct = mask.sum() / len(num) * 100
    return pct.sort_values(ascending=False).round(2)


def efeito_outliers(df: pd.DataFrame):
    """Compara média e desvio com/sem outliers (IQR) para mostrar impacto."""
    num = df.select_dtypes("number")
    if num.empty:
        return pd.DataFrame()
    mask_linha_com_out = outliers_iqr_mask(num).any(axis=1)
    sem_out = num.loc[~mask_linha_com_out]
    comp = pd.DataFrame({
        "mean_com_out": num.mean(),
        "mean_sem_out": sem_out.mean(),
        "std_com_out":  num.std(),
        "std_sem_out":  sem_out.std()
    })
    comp["delta_mean_abs"] = (comp["mean_sem_out"] - comp["mean_com_out"]).abs()
    comp["delta_std_abs"]  = (comp["std_sem_out"]  - comp["std_com_out"]).abs()
    return comp


# ---------------------- Tempo / Clusters / Influência ----------------------
def detectar_tempo(df: pd.DataFrame):
    cand = [c for c in df.columns if c.lower() in ("time", "timestamp", "date", "datetime", "year")]
    if cand:
        c = cand[0]
        if df[c].dtype == "object":
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
            except Exception:
                pass
        return c
    return None


def clusters(df: pd.DataFrame, k=3, sample=5000, random_state=42):
    num = df.select_dtypes("number")
    if num.shape[1] < 2 or len(num) < 2:
        return None, "Dados numéricos insuficientes para clusterização."
    X = num.sample(n=min(sample, len(num)), random_state=random_state).fillna(num.mean())
    Xn = (X - X.mean()) / X.std(ddof=0)
    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    labs = km.fit_predict(Xn)
    resumo = pd.Series(labs).value_counts().sort_index().rename("contagem")
    return resumo, f"Clusters (k={k}) em amostra de {len(X)} linhas."


def variaveis_mais_influentes(df: pd.DataFrame):
    """Heurística: correlação absoluta média de cada coluna numérica com as demais."""
    num = df.select_dtypes("number")
    if num.shape[1] < 2:
        return pd.Series(dtype=float)
    corr = num.corr().abs()
    score = corr.mean().sort_values(ascending=False)
    return score


# ---------------------- “Agente” por palavras-chave ----------------------
def responder(df: pd.DataFrame, pergunta: str):
    # normaliza e cria uma versão sem acentos para casar palavras-chave com/sem acento
    q_raw = (pergunta or "").strip().lower()
    q = _norm(pergunta)

    num_cols = df.select_dtypes("number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    # ---------------- PRIORIDADE ALTA ----------------
    # OUTLIERS
    if "outlier" in q or "atipic" in q:
        texto = "Outliers (IQR) e impacto:"
        pct = outliers_iqr(df)
        if pct.empty or (pct == 0).all():
            conclusion = _save_conclusion(pergunta, "Não foram detectados outliers pelo critério IQR nas colunas numéricas.")
        else:
            top = pct[pct > 0].head(5)
            pares = ", ".join([f"{c}: {v:.2f}%" for c, v in top.items()])
            conclusion = _save_conclusion(pergunta, f"Outliers identificados (IQR). Maiores incidências → {pares}.")
        return texto, "dupla_tabela", {
            "pct": pct.to_frame("pct_linhas_outlier"),
            "efeito": efeito_outliers(df),
            "conclusion": conclusion
        }

    # TENDÊNCIA CENTRAL (média / mediana)
    if "tendencia central" in q or "media" in q or "mediana" in q:
        texto = "Tendência central (média/mediana):"
        tc = tendencia_central(df)
        conclusion = _save_conclusion(pergunta, f"Cálculo de média e mediana para {tc.shape[0]} coluna(s) numérica(s).")
        return texto, "tabela", {"data": tc, "conclusion": conclusion}

    # INTERVALO (min/max)
    if any(k in q for k in ("intervalo", "min", "max", "minimo", "maximo")):
        texto = "Intervalos (min/max) por coluna numérica:"
        inter = intervalo(df)
        conclusion = _save_conclusion(pergunta, f"Gerados mínimos e máximos para {inter.shape[0]} coluna(s) numérica(s).")
        return texto, "tabela", {"data": inter, "conclusion": conclusion}

    # VARIABILIDADE
    if "desvio" in q or "varianc" in q:
        texto = "Variabilidade (desvio/variância):"
        var = variabilidade(df)
        conclusion = _save_conclusion(pergunta, f"Desvio-padrão e variância calculados para {var.shape[0]} coluna(s) numérica(s).")
        return texto, "tabela", {"data": var, "conclusion": conclusion}

    # FREQUÊNCIAS
    if "frequent" in q or "moda" in q:
        texto = "Top frequências por coluna (top 10):"
        mapa = frequencias(df)
        qtd = len(mapa)
        conclusion = _save_conclusion(pergunta, f"Listadas frequências para {qtd} coluna(s).")
        return texto, "dict_series", {"mapa": mapa, "conclusion": conclusion}

    # CORRELAÇÃO
    if "correlac" in q:
        if len(num_cols) < 2:
            return "Preciso de pelo menos duas colunas numéricas para calcular correlação.", None, {}
        texto = "Mapa de correlação (on-demand)."
        conclusion = _save_conclusion(pergunta, f"Heatmap de correlação entre {len(num_cols)} variáveis numéricas.")
        return texto, "heatmap_corr", {"conclusion": conclusion}

    # DISPERSÃO
    if "dispers" in q or "scatter" in q:
        if len(num_cols) >= 2:
            texto = f"Dispersão entre {num_cols[0]} e {num_cols[1]}:"
            conclusion = _save_conclusion(pergunta, f"Gráfico de dispersão gerado para {num_cols[0]} vs {num_cols[1]}.")
            return texto, "scatter", {"x": num_cols[0], "y": num_cols[1], "conclusion": conclusion}
        return "Colunas numéricas insuficientes para dispersão.", None, {}

    # (além do bloco de cima, aceita as variações com acento)
    if "tendência central" in q or "medidas de tendência" in q or "média" in q or "mediana" in q:
        texto = "Medidas de tendência central (média e mediana) para variáveis numéricas:"
        tc = tendencia_central(df)
        conclusion = _save_conclusion(pergunta, f"Cálculo de média e mediana para {tc.shape[0]} coluna(s) numérica(s).")
        return texto, "tabela", {"data": tc, "conclusion": conclusion}

    # TENDÊNCIA TEMPORAL
    if "tendenc" in q or "temporal" in q or "serie" in q:
        tcol = detectar_tempo(df)
        ycols = [c for c in num_cols if c != tcol]
        if tcol and ycols:
            texto = f"Série temporal de {ycols[0]} vs {tcol}:"
            conclusion = _save_conclusion(pergunta, f"Série temporal traçada: {ycols[0]} ao longo de {tcol}.")
            return texto, "timeseries", {"tcol": tcol, "ycol": ycols[0], "conclusion": conclusion}
        elif tcol:
            return (f"Identifiquei a coluna temporal '{tcol}', mas não encontrei nenhuma outra "
                    "variável numérica para comparar."), None, {}

    # CLUSTERS
    if "cluster" in q or "agrup" in q:
        resumo, msg = clusters(df)
        if resumo is None:
            return msg, None, {}
        texto = msg
        conclusion = _save_conclusion(pergunta, f"Clusterização executada ({resumo.shape[0]} grupos).")
        return texto, "tabela", {"data": resumo.to_frame(), "conclusion": conclusion}

    # VARIÁVEIS INFLUENTES
    if "influenc" in q or "importanc" in q:
        if len(num_cols) < 2:
            return "Preciso de pelo menos duas colunas numéricas para estimar influência por correlação.", None, {}
        score = variaveis_mais_influentes(df)
        texto = "Variáveis com maior correlação média (heurística de influência):"
        # pequeno destaque das 3 primeiras
        top = score.head(3)
        destaque = ", ".join([f"{c} ({v:.3f})" for c, v in top.items()])
        conclusion = _save_conclusion(pergunta, f"Maior centralidade de correlação: {destaque}.")
        return texto, "serie", {"serie": score, "conclusion": conclusion}

    # DISTRIBUIÇÃO
    if "distribuic" in q:
        resultados = []
        for c in num_cols:
            resultados.append(("hist", c))
        for c in cat_cols:
            resultados.append(("bar", c))
        texto = "Distribuição de variáveis numéricas e categóricas."
        conclusion = _save_conclusion(pergunta, f"Gerados {len(num_cols)} histogramas e {len(cat_cols)} gráficos de barras.")
        return texto, "multi_plot", {"resultados": resultados, "conclusion": conclusion}

    # HISTOGRAMA ESPECÍFICO
    if "histogram" in q:
        alvos = [c for c in df.columns if c.lower() in q_raw]
        if not alvos and num_cols:
            alvos = [num_cols[0]]
        if alvos:
            texto = f"Histograma de {alvos[0]}:"
            conclusion = _save_conclusion(pergunta, f"Histograma exibido para {alvos[0]}.")
            return texto, "hist", {"col": alvos[0], "conclusion": conclusion}
        return "Não encontrei coluna apropriada para histograma.", None, {}

    # TABELA CRUZADA
    if "tabela cruzada" in q or "crosstab" in q:
        cats = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        if len(cats) >= 2:
            ct = pd.crosstab(df[cats[0]], df[cats[1]])
            texto = f"Tabela cruzada entre {cats[0]} e {cats[1]}:"
            conclusion = _save_conclusion(pergunta, f"Tabela cruzada gerada para {cats[0]} × {cats[1]}.")
            return texto, "tabela", {"data": ct, "conclusion": conclusion}
        return "Não encontrei duas colunas categóricas para tabela cruzada.", None, {}

    # TIPOS DE DADOS (por último)
    if any(k in q for k in ("tipo", "tipos de dados", "categ", "numer", "dtype", "dtypes")):
        tipos_df, resumo = tipos(df)
        texto = (f"Detectadas {int(resumo.get('Numérica', 0))} colunas **numéricas**, "
                 f"{int(resumo.get('Data/Tempo', 0))} de **data/tempo** e "
                 f"{int(resumo.get('Categórica', 0))} **categóricas**.")
        conclusion = _save_conclusion(pergunta, f"Tipos de dados: {int(resumo.get('Numérica', 0))} num., "
                                                f"{int(resumo.get('Data/Tempo', 0))} tempo, "
                                                f"{int(resumo.get('Categórica', 0))} categ.")
        return texto, "tabela", {"data": tipos_df, "conclusion": conclusion}

    # AJUDA
    return (
        "Não entendi. Exemplos: 'tipos de dados', 'intervalo', 'média', 'variância', "
        "'frequentes', 'outliers', 'correlação', 'dispersão', 'tendência temporal', "
        "'clusters', 'variáveis mais influentes', 'distribuição de variáveis', "
        "'histograma de Price', 'tabela cruzada'."
    ), None, {}
