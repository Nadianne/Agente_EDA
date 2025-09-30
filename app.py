import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import eda, charts
from utils.memory import all_md, clear  # memória


# ---------------------- Estilo customizado ----------------------
st.set_page_config(page_title="Agente de Análise EDA", layout="wide")

st.markdown("""
    <style>
        .main {background-color: #0e0e0e;}
        .stApp {background-color: #0e0e0e; color: #f5f5f5;}
        h1, h2, h3, h4, h5 {color: #ff4b4b;}
        .stButton button {background-color: #ff4b4b; color: white; border-radius: 8px;}
        .stButton button:hover {background-color: #d43c3c;}
    </style>
""", unsafe_allow_html=True)

# ---------------------- Cabeçalho ----------------------
st.title("Agente de Análise EDA")
st.markdown(" **Aluna: Nadianne Galvão**", unsafe_allow_html=True)

with st.container():
    st.markdown("""
    ###  Atividade — *Institut d'Intelligence Artificielle Appliquée*
    Esta atividade tem por objetivo criar um ou mais **agentes de E.D.A (Exploratory Data Analysis)** que permitam a um usuário fazer perguntas sobre qualquer arquivo CSV disponibilizado.  
    A solução entrega uma **interface interativa** onde o usuário informa a pergunta e o agente gera a resposta → carregando o CSV, executando queries e/ou gerando o código Python necessário.  

    ---

    #### 🛠️ Frameworks e Bibliotecas utilizadas:
    - 📌 **Streamlit** → Interface web interativa  
    - 🐼 **Pandas** → Manipulação e análise de dados  
    - 🔢 **NumPy** → Cálculos numéricos e estatísticos  
    - 🤖 **Scikit-learn** → Algoritmos de machine learning (KMeans para clusters)  
    - 📈 **Matplotlib & Seaborn** → Visualizações estatísticas  

    ---
    """)
    st.markdown("---")

# ---------------------- Manual ----------------------
with st.expander("📖 Manual (passo a passo)"):
    st.markdown("""
    **Como utilizar o agente de análise EDA:**

    1️⃣ **Upload de CSV** → Envie um arquivo `.csv` para análise.  
    2️⃣ **Perguntas em linguagem natural** → Digite perguntas como *"Quais são as médias?"* ou *"Existem outliers?"*.  
    3️⃣ **Métricas estatísticas** → O agente calcula automaticamente medidas como média, mediana, variância etc.  
    4️⃣ **Gráficos sob demanda** → Gere histogramas, boxplots, dispersões, correlações e séries temporais.  
    5️⃣ **Memória de conclusões** → Cada resposta gera uma conclusão que pode ser revisitada na aba **Conclusões**.  

     Pronto! Agora basta explorar os dados de forma interativa.  
    """)

# Estado para o "chat"
if "chat" not in st.session_state:
    # cada item: {"pergunta": str, "texto": str, "acao": str|None, "params": dict}
    st.session_state["chat"] = []

uploaded_file = st.file_uploader(" Faça upload de um arquivo CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False)
    st.success(f"Arquivo carregado! {df.shape[0]:,} linhas × {df.shape[1]} colunas.")

    # ordenar por tempo se existir
    tcol = eda.detectar_tempo(df)
    if tcol:
        df = df.sort_values(tcol).reset_index(drop=True)

    # --- Visão Geral ---
    with st.expander(" Visão Geral (clique para abrir)", expanded=False):
        st.caption(f"{df.shape[0]:,} linhas × {df.shape[1]} colunas")
        mostrar_tudo = st.toggle("Mostrar todas as linhas (pode ficar lento)", value=False)
        limite = st.slider("Linhas quando NÃO mostrar tudo:", 100, 10000, 2000, step=100)
        df_view = df if mostrar_tudo else df.head(limite)
        st.dataframe(df_view, use_container_width=True, height=500, hide_index=False)

    # abas p/ navegação
    tabs = st.tabs([" Perguntas (Agente)", " Gráficos", " Conclusões"])

    # ---- Aba 1: Perguntas (chat) ----
    with tabs[0]:
        st.info("Exemplos: 'tipos de dados', 'intervalo', 'média', 'variância', "
                "'frequentes', 'outliers', 'correlação', 'dispersão', "
                "'tendência temporal', 'clusters', 'variáveis mais influentes', "
                "'distribuição de variáveis', 'tabela cruzada'.")

        # Histórico (render)
        st.subheader("Histórico")
        if not st.session_state["chat"]:
            st.caption("Sem interações ainda. Faça uma pergunta abaixo")
        else:
            for turn in st.session_state["chat"]:
                with st.container():
                    st.markdown(f"**Você:** {turn['pergunta']}")
                    st.markdown(f"**Agente:** {turn['texto']}")

                    # Render da ação (se houver)
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
                            st.subheader(k); st.write(series)
                    elif acao == "serie":
                        st.write(params["serie"])
                    elif acao == "heatmap_corr":
                        st.pyplot(charts.heatmap_corr(df))
                    elif acao == "scatter":
                        st.pyplot(charts.scatter(df.select_dtypes("number"), params["x"], params["y"]))
                    elif acao == "timeseries":
                        st.pyplot(charts.timeseries(df, params["tcol"], params["ycol"]))
                    elif acao == "hist":
                        st.pyplot(charts.hist(df, params["col"]))
                    elif acao == "multi_plot":
                        for tipo, col in params["resultados"]:
                            st.subheader(f"{col} ({'Numérica' if tipo=='hist' else 'Categórica'})")
                            if tipo == "hist":
                                st.pyplot(charts.hist(df, col))
                            elif tipo == "bar":
                                st.bar_chart(df[col].value_counts().head(20))

                    # Conclusão curtinha (se veio)
                    if isinstance(params, dict) and params.get("conclusion"):
                        st.markdown("> **Conclusão:**")
                        st.info(params["conclusion"])

                    st.markdown("---")

        # Entrada + enviar
        pergunta = st.text_input("Digite sua pergunta ao agente:")
        if st.button("Responder"):
            texto, acao, params = eda.responder(df, pergunta)

            # guarda o turno no histórico do chat
            st.session_state["chat"].append({
                "pergunta": pergunta,
                "texto": texto,
                "acao": acao,
                "params": params,
            })

            # re-render para mostrar já no histórico
            st.rerun()

    # ---- Aba 2: Gráficos sob demanda ----
    with tabs[1]:
        gtab = st.radio("Escolha", ["Histograma", "Boxplot", "Dispersão", "Correlação", "Série Temporal", "Tabela Cruzada"], horizontal=True)
        num_cols = df.select_dtypes("number").columns.tolist()
        cat_cols = [c for c in df.columns if c not in num_cols]

        if gtab == "Histograma" and num_cols:
            c = st.selectbox("Coluna numérica", num_cols)
            if st.button("Gerar histograma"):
                st.pyplot(charts.hist(df, c))

        if gtab == "Boxplot" and num_cols:
            c = st.selectbox("Coluna numérica", num_cols, key="box")
            if st.button("Gerar boxplot"):
                st.pyplot(charts.box(df, c))

        if gtab == "Dispersão" and len(num_cols) >= 2:
            x = st.selectbox("Eixo X", num_cols, key="x")
            y = st.selectbox("Eixo Y", num_cols, key="y")
            if st.button("Gerar dispersão"):
                st.pyplot(charts.scatter(df, x, y))

        if gtab == "Correlação" and len(num_cols) >= 2:
            if st.button("Gerar correlação"):
                st.pyplot(charts.heatmap_corr(df))

        if gtab == "Série Temporal":
            tcol_here = eda.detectar_tempo(df)
            if tcol_here and num_cols:
                y = st.selectbox("Variável (Y)", num_cols, key="tsy")
                if st.button("Gerar série"):
                    st.pyplot(charts.timeseries(df, tcol_here, y))
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
            st.success("Conclusões limpas. Faça novas perguntas e volte aqui")

else:
    st.info("Envie um arquivo CSV para começar.")

