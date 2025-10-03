# app.py
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import eda, charts
from utils.memory import all_md, clear  # memória

# (Opcional) Token da HF se for usar LLM depois
HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", ""))

# ---------------------- Estilo customizado ----------------------
st.set_page_config(page_title="Agente de Análise EDA", layout="wide")

st.markdown("""
    <style>
        .main {background-color: #0e0e0e;}
        .stApp {background-color: #0e0e0e; color: #f5f5f5;}
        h1, h2, h3, h4, h5 {color: #ff4b4b;}
        .stButton button {background-color: #ff4b4b; color: white; border-radius: 8px;}
        .stButton button:hover {background-color: #d43c3c;}
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    </style>
""", unsafe_allow_html=True)

# ---------------------- Cabeçalho ----------------------
st.title("Agente de Análise EDA")
st.markdown("**Aluna: Nadianne Galvão**")

with st.container():
    st.markdown("""
    ### Atividade — *Institut d'Intelligence Artificielle Appliquée*
    Esta atividade tem por objetivo criar um ou mais **agentes de E.D.A** (Exploratory Data Analysis) que permitam a um usuário fazer perguntas sobre qualquer arquivo CSV.  
    A solução entrega uma **interface interativa** onde o usuário informa a pergunta e o agente gera a resposta → carregando o CSV, executando queries e/ou gerando o código Python necessário.

    ---
    #### 🛠️ Frameworks e Bibliotecas utilizadas:
    - **Streamlit** → Interface web
    - **Pandas** → Manipulação e análise de dados
    - **NumPy** → Cálculos numéricos
    - **Scikit-learn** → KMeans (clusters)
    - **Matplotlib & Seaborn** → Visualizações
    ---
    """)

# ---------------------- Manual ----------------------
with st.expander("📖 Manual (passo a passo)"):
    st.markdown("""
    1) **Upload de CSV** → Envie um arquivo `.csv`.  
    2) **Pergunte em linguagem natural** → Ex.: *"Quais são as médias?"*, *"Existem outliers?"*.  
    3) **Resultados** → Tabelas e gráficos sob demanda.  
    4) **Memória de Conclusões** → Cada resposta registra um resumo na aba **Conclusões**.  
    """)

# ---------------------- Helper: preview + expandir ----------------------
def preview_and_expand(make_fig, label: str, small=(6, 4), big=(11, 7)):
    """
    Renderiza um preview e um expander com a versão ampliada do mesmo gráfico.
    - make_fig: função que retorna figura matplotlib e aceita 'figsize' como kwarg.
      Ex.: lambda **kw: charts.hist(df, "Amount", **kw)
    - label: texto do expander
    - small: tamanho preview
    - big: tamanho expandido
    """
    fig_small = make_fig(figsize=small)
    st.pyplot(fig_small)
    with st.expander(f"🔍 {label} — ver maior"):
        fig_big = make_fig(figsize=big)
        st.pyplot(fig_big)

# ---------------------- Estado de chat ----------------------
if "chat" not in st.session_state:
    # cada item: {"pergunta": str, "texto": str, "acao": str|None, "params": dict}
    st.session_state["chat"] = []

# ---------------------- Upload ----------------------
uploaded_file = st.file_uploader("📂 Faça upload de um arquivo CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False)
    st.success(f"Arquivo carregado! {df.shape[0]:,} linhas × {df.shape[1]} colunas.")

    # ordenar por tempo se existir
    tcol = eda.detectar_tempo(df)
    if tcol:
        df = df.sort_values(tcol).reset_index(drop=True)

    # --- Visão Geral (rolável) ---
    with st.expander("📑 Visão Geral (clique para abrir)", expanded=False):
        st.caption(f"{df.shape[0]:,} linhas × {df.shape[1]} colunas")
        mostrar_tudo = st.toggle("Mostrar todas as linhas (pode ficar lento)", value=False)
        limite = st.slider("Linhas quando NÃO mostrar tudo:", 100, 10000, 2000, step=100)
        df_view = df if mostrar_tudo else df.head(limite)
        st.dataframe(df_view, use_container_width=True, height=500, hide_index=False)

    # --- Abas ---
    tabs = st.tabs(["🤖 Perguntas ", "📈 Gráficos", "🧠 Conclusões"])

    # ---- Aba 1: Perguntas (chat) ----
    with tabs[0]:
        st.info("Exemplos: 'tipos de dados', 'intervalo', 'média', 'variância', "
                "'frequentes', 'outliers', 'correlação', 'dispersão', "
                "'tendência temporal', 'clusters', 'variáveis mais influentes', "
                "'distribuição de variáveis', 'tabela cruzada'.")

        # Histórico
        st.subheader("Histórico")
        if not st.session_state["chat"]:
            st.caption("Sem interações ainda. Faça uma pergunta abaixo.")
        else:
            for turn in st.session_state["chat"]:
                with st.container():
                    st.markdown(f"**Você:** {turn['pergunta']}")
                    st.markdown(f"**Agente:** {turn['texto']}")

                    acao = turn["acao"]
                    params = turn["params"] or {}

                    if acao == "tabela":
                        st.dataframe(params["data"], use_container_width=True)

                    elif acao == "dupla_tabela":
                        st.subheader("Outliers por coluna (%)")
                        st.dataframe(params["pct"], use_container_width=True)
                        st.subheader("Efeito dos outliers (comparação mean/std)")
                        st.dataframe(params["efeito"], use_container_width=True)

                    elif acao == "dict_series":
                        for k, series in params["mapa"].items():
                            st.subheader(k)
                            st.write(series)

                    elif acao == "serie":
                        st.write(params["serie"])

                    elif acao == "heatmap_corr":
                        preview_and_expand(
                            lambda **kw: charts.heatmap_corr(df, **kw),
                            label="Mapa de correlação"
                        )

                    elif acao == "scatter":
                        df_num = df.select_dtypes("number")
                        preview_and_expand(
                            lambda **kw: charts.scatter(df_num, params["x"], params["y"], **kw),
                            label=f"Dispersão: {params['x']} vs {params['y']}"
                        )

                    elif acao == "timeseries":
                        preview_and_expand(
                            lambda **kw: charts.timeseries(df, params["tcol"], params["ycol"], **kw),
                            label=f"Série temporal: {params['ycol']} por {params['tcol']}"
                        )

                    elif acao == "hist":
                        preview_and_expand(
                            lambda **kw: charts.hist(df, params["col"], **kw),
                            label=f"Histograma: {params['col']}"
                        )

                    elif acao == "multi_plot":
                        for tipo, col in params["resultados"]:
                            st.subheader(f"{col} ({'Numérica' if tipo=='hist' else 'Categórica'})")
                            if tipo == "hist":
                                preview_and_expand(
                                    lambda **kw: charts.hist(df, col, **kw),
                                    label=f"Histograma: {col}"
                                )
                            elif tipo == "bar":
                                preview_and_expand(
                                    lambda **kw: charts.bar_counts(df, col, topn=20, **kw),
                                    label=f"Top valores: {col}"
                                )

                    # Conclusão curta (se veio)
                    if isinstance(params, dict) and params.get("conclusion"):
                        st.markdown("> **Conclusão:**")
                        st.info(params["conclusion"])

                    st.markdown("---")

        # Entrada + enviar
        pergunta = st.text_input("Digite sua pergunta ao agente:")
        if st.button("Enviar"):
            texto, acao, params = eda.responder(df, pergunta)
            st.session_state["chat"].append({
                "pergunta": pergunta,
                "texto": texto,
                "acao": acao,
                "params": params,
            })
            st.rerun()

    # ---- Aba 2: Gráficos sob demanda ----
    with tabs[1]:
        gtab = st.radio("Escolha", ["Histograma", "Boxplot", "Dispersão", "Correlação", "Série Temporal", "Tabela Cruzada"], horizontal=True)
        num_cols = df.select_dtypes("number").columns.tolist()
        cat_cols = [c for c in df.columns if c not in num_cols]

        if gtab == "Histograma" and num_cols:
            c = st.selectbox("Coluna numérica", num_cols)
            if st.button("Gerar histograma"):
                preview_and_expand(
                    lambda **kw: charts.hist(df, c, **kw),
                    label=f"Histograma: {c}"
                )

        if gtab == "Boxplot" and num_cols:
            c = st.selectbox("Coluna numérica", num_cols, key="box")
            if st.button("Gerar boxplot"):
                preview_and_expand(
                    lambda **kw: charts.box(df, c, **kw),
                    label=f"Boxplot: {c}"
                )

        if gtab == "Dispersão" and len(num_cols) >= 2:
            x = st.selectbox("Eixo X", num_cols, key="x")
            y = st.selectbox("Eixo Y", num_cols, key="y")
            if st.button("Gerar dispersão"):
                preview_and_expand(
                    lambda **kw: charts.scatter(df.select_dtypes("number"), x, y, **kw),
                    label=f"Dispersão: {x} vs {y}"
                )

        if gtab == "Correlação" and len(num_cols) >= 2:
            if st.button("Gerar correlação"):
                preview_and_expand(
                    lambda **kw: charts.heatmap_corr(df, **kw),
                    label="Mapa de correlação"
                )

        if gtab == "Série Temporal":
            tcol_here = eda.detectar_tempo(df)
            if tcol_here and num_cols:
                y = st.selectbox("Variável (Y)", num_cols, key="tsy")
                if st.button("Gerar série"):
                    preview_and_expand(
                        lambda **kw: charts.timeseries(df, tcol_here, y, **kw),
                        label=f"Série temporal: {y} por {tcol_here}"
                    )
            else:
                st.info("Não identifiquei coluna temporal + numérica.")

        if gtab == "Tabela Cruzada" and len(cat_cols) >= 2:
            a = st.selectbox("Categórica A", cat_cols, key="cta")
            b = st.selectbox("Categórica B", cat_cols, key="ctb")
            if st.button("Gerar crosstab"):
                st.dataframe(pd.crosstab(df[a], df[b]), use_container_width=True)

    # ---- Aba 3: Conclusões acumuladas ----
    with tabs[2]:
        st.markdown(all_md())
        if st.button("Limpar conclusões"):
            clear()
            st.success("Conclusões limpas. Faça novas perguntas e volte aqui 🙂")

else:
    st.info("Envie um arquivo CSV para começar.")
