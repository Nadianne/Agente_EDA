### Agente de Análise EDA

Link para uso online:  [https://agenteeda-ke5dbmrwy2xvv2fxtsfyvd.streamlit.app/#class-numerica](https://agenteeda-ke5dbmrwy2xvv2fxtsfyvd.streamlit.app/#class-numerica)

##  Descrição

Este projeto consiste em um agente de **Análise Exploratória de Dados (EDA)**, desenvolvido em **Python** com interface web usando **Streamlit**.  
Ele permite que o usuário carregue um arquivo CSV e faça perguntas em **linguagem natural** para extrair estatísticas, visualizações e insights de forma automatizada.

---

##  Funcionalidades

-  Upload de arquivo CSV e exibição de visão geral  
-  Identificação de tipos de variáveis (numéricas, categóricas, temporais)  
-  Cálculo de medidas estatísticas (média, mediana, desvio, variância)  
-  Detecção de outliers via IQR e comparação de impacto  
-  Geração de gráficos sob demanda (histogramas, boxplot, dispersões, correlações, séries temporais)  
-  Ranking de variáveis mais influentes por correlação  
-  **Memória de conclusões** → o agente armazena e exibe os insights gerados  
-  Interface web interativa e responsiva

---

##  Estrutura do projeto

agente_analise_eda/
├── app.py                ← Interface principal com Streamlit  
├── requirements.txt      ← Dependências do projeto  
├── utils/  
│   ├── eda.py             ← Lógica de análise exploratória  
│   ├── charts.py          ← Funções de plotagem  
│   └── memory.py          ← Armazenamento de conclusões  
└── data/                  ← Exemplos de CSVs (opcional)  


Este projeto foi desenvolvido como atividade do curso do Institut d'Intelligence Artificielle Appliquée.
Aluna: Nadianne Galvão

