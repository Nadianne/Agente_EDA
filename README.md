# Agente de Análise EDA

Agente de **Análise Exploratória de Dados (EDA)** em **Python** com **Streamlit**.  
O usuário carrega um arquivo CSV e faz perguntas em **linguagem natural**; a **LLM atua apenas como roteadora de intenção** e o agente executa **cálculos determinísticos** (Pandas/NumPy/Scikit-learn/Matplotlib) para gerar estatísticas, visualizações e insights.

**Demo online:**  
https://agenteeda-ke5dbmrwy2xvv2fxtsfyvd.streamlit.app/#class-numerica

---

## Funcionalidades

- Upload de CSV e visão geral: tipos de variáveis, valores ausentes (NA) e duplicatas  
- Estatística descritiva: média, mediana, desvio-padrão, variância  
  - Opcionais: assimetria (skew) e curtose  
- Outliers via IQR; opção de Z-score robusto (MAD) e comparação de impacto  
- Correlação: Pearson e Spearman  
  - Opcionais: Cramér’s V e correlation ratio (η) para variáveis categóricas  
- Gráficos: histogramas, boxplots, dispersões, heatmap de correlação e séries temporais  
- Ranking simples de variáveis mais influentes por correlação  
- Memória de conclusões da sessão (com opção de exportação)  
- LLM como roteadora de intenção (label/JSON), com fallback determinístico sem LLM  
- Cache leve e semente fixa para reprodutibilidade

---

## Estrutura do projeto

agente_analise_eda/

├── app.py                ← Interface principal com Streamlit  
├── requirements.txt      ← Dependências do projeto  
├── utils/  
│   ├── eda.py             ← Lógica de análise exploratória  
│   ├── charts.py          ← Funções de plotagem  
│   └── memory.py          ← Armazenamento de conclusões  
|   └── nlp.py             ← Roteador de intenção (LLM ou regras); nunca responde conteúdo final
├── 


## Como funciona

1. O usuário carrega o CSV e faz uma pergunta em linguagem natural.  
2. O módulo `nlp.py` classifica a intenção (por exemplo: `stats`, `outliers`, `correlation`, `cluster`, `describe`).  
   - Se a LLM estiver indisponível, aplica-se um conjunto de regras locais.  
3. O `app.py` invoca as funções de `utils/eda.py` e `utils/charts.py` para produzir resultados determinísticos.  
4. As conclusões são registradas em `utils/memory.py` e podem ser visualizadas e exportadas.

---

## Como rodar localmente:

### 1) (opcional) criar venv
python -m venv .venv && source .venv/bin/activate

### 2) instalar dependências
pip install -r requirements.txt

### 3) (opcional) definir token da LLM (apenas para roteamento)
 export HF_TOKEN=seu_token_aqui

### 4) executar
streamlit run app.py

## Limitações e cuidados

O agente não realiza imputações complexas; limpeza é mínima e transparente.

Correlação não implica causalidade; gráficos/estatísticas são exploratórios.

Para bases muito grandes, a UI pode aplicar amostragem em scatter plots (sem afetar cálculos).

---

Este projeto foi desenvolvido como atividade do curso do Institut d'Intelligence Artificielle Appliquée.

Aluna: Nadianne Galvão







