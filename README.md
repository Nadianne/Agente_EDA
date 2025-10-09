# Agente de Análise EDA
Este projeto consiste em um agente de **Análise Exploratória de Dados (EDA)**, desenvolvido em **Python** com interface web usando **Streamlit**.  
Ele permite que o usuário carregue um arquivo CSV e faça perguntas em **linguagem natural** para extrair estatísticas, visualizações e insights de forma automatizada.

Link para uso online:  [https://agenteeda-ke5dbmrwy2xvv2fxtsfyvd.streamlit.app/#class-numerica](https://agenteeda-ke5dbmrwy2xvv2fxtsfyvd.streamlit.app/#class-numerica)



---

##  Funcionalidades

Upload de CSV e visão geral (tipos, NAs, duplicatas)

Estatística descritiva: média, mediana, desvio-padrão, variância, (opcional) assimetria e curtose

Outliers via IQR (e Z-score robusto/MAD, opcional), com comparação de impacto

Correlação (Pearson/Spearman; opcional: Cramér’s V / correlation ratio η)

Gráficos: histogramas, boxplots, dispersões, heatmap de correlação, séries temporais

Ranking simples de variáveis mais influentes (por correlação)

Memória de conclusões da sessão (exportável)

LLM como roteadora de intenção (JSON/label), com fallback sem LLM

Cache leve e semente fixa para reprodutibilidade

---

##  Estrutura do projeto

agente_analise_eda/

├── app.py                ← Interface principal com Streamlit  
├── requirements.txt      ← Dependências do projeto  
├── utils/  
│   ├── eda.py             ← Lógica de análise exploratória  
│   ├── charts.py          ← Funções de plotagem  
│   └── memory.py          ← Armazenamento de conclusões  
|   └── nlp.py             ← Roteador de intenção (LLM ou regras); nunca responde conteúdo final
├── 

## Como rodar localmente:

# 1) (opcional) criar venv
python -m venv .venv && source .venv/bin/activate

# 2) instalar dependências
pip install -r requirements.txt

# 3) (opcional) definir token da LLM (apenas para roteamento)
# export HF_TOKEN=seu_token_aqui

# 4) executar
streamlit run app.py

## Limitações e cuidados

O agente não realiza imputações complexas; limpeza é mínima e transparente.

Correlação não implica causalidade; gráficos/estatísticas são exploratórios.

Para bases muito grandes, a UI pode aplicar amostragem em scatter plots (sem afetar cálculos).

Este projeto foi desenvolvido como atividade do curso do Institut d'Intelligence Artificielle Appliquée.

Aluna: Nadianne Galvão





