# utils/nlp.py
import os
import re
import streamlit as st
from typing import Literal, Optional
import requests

# Categorias que o agente entende (mapeie para seu eda.responder)
CATEGORIAS = [
    "tipos",                 # tipos de dados
    "intervalo",             # min/max
    "tendencia_central",     # média/mediana
    "variabilidade",         # desvio/variância
    "frequencias",           # valores frequentes
    "outliers",
    "correlacao",
    "dispersao",
    "temporal",
    "clusters",
    "influencia",            # variáveis mais influentes
    "distribuicao",          # distribuição geral
    "tabela_cruzada"
]

HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"

def _hf_token() -> Optional[str]:
    # 1) Primeiro tenta nos secrets do Streamlit Cloud
    if "HF_TOKEN" in st.secrets:
        return st.secrets["HF_TOKEN"]
    # 2) Fallback: variável de ambiente (opcional)
    return os.getenv("HF_TOKEN")

def _chutar_regra(pergunta: str) -> str:
    """Fallback determinístico por palavras-chave caso a API falhe/sem token."""
    q = (pergunta or "").lower()

    if any(x in q for x in ["tipo", "dtype", "dtypes", "categó", "numér"]):
        return "tipos"
    if any(x in q for x in ["intervalo", "min", "máx", "minimo", "maximo"]):
        return "intervalo"
    if any(x in q for x in ["tendência", "tendencia", "média", "media", "mediana"]):
        return "tendencia_central"
    if any(x in q for x in ["variân", "varianc", "desvio"]):
        return "variabilidade"
    if "frequ" in q or "moda" in q:
        return "frequencias"
    if "outlier" in q or "atípic" in q or "atipic" in q:
        return "outliers"
    if "correla" in q:
        return "correlacao"
    if "dispers" in q or "scatter" in q:
        return "dispersao"
    if any(x in q for x in ["tendên", "tendenc", "temporal", "série", "serie"]):
        return "temporal"
    if "cluster" in q or "agrup" in q:
        return "clusters"
    if "influên" in q or "influenc" in q or "importânc" in q or "importanc" in q:
        return "influencia"
    if "distribui" in q:
        return "distribuicao"
    if "tabela cruzada" in q or "crosstab" in q:
        return "tabela_cruzada"
    return "tipos"  # fallback padrão

def interpretar_pergunta(pergunta: str) -> str:
    """
    Usa LLM (Hugging Face Inference API) para classificar a pergunta em UMA categoria.
    Se não houver token/API falhar, cai no fallback determinístico.
    """
    token = _hf_token()
    if not token:
        return _chutar_regra(pergunta)

    prompt = f"""
Você é um classificador para análise exploratória de dados (EDA).
Receba a pergunta do usuário e responda APENAS com UMA das categorias abaixo (exatamente como escrito):
{", ".join(CATEGORIAS)}

Pergunta: \"{pergunta}\"
Responda SOMENTE com a categoria.
"""

    try:
        resp = requests.post(
            HF_API_URL,
            headers={"Authorization": f"Bearer {token}"},
            json={"inputs": prompt, "parameters": {"max_new_tokens": 10}}
        )
        resp.raise_for_status()
        data = resp.json()

        # Formato da FLAN-t5-small: retorna string em 'generated_text' (várias libs encapsulam diferente)
        if isinstance(data, list) and data and "generated_text" in data[0]:
            saida = data[0]["generated_text"].strip().lower()
        elif isinstance(data, dict) and "generated_text" in data:
            saida = data["generated_text"].strip().lower()
        else:
            # alguns endpoints retornam diretamente a string
            saida = str(data).strip().lower()

        # Normaliza: tira coisas fora do alfabeto/underscore e valida
        saida = re.sub(r"[^a-zA-Z_]", "", saida)
        if saida not in CATEGORIAS:
            # tenta mapear respostas tipo "media" -> "tendencia_central"
            if saida in ("media", "mediana", "tendenciacentral"):
                saida = "tendencia_central"
            elif saida in ("correlacao", "correlacoes"):
                saida = "correlacao"
            elif saida in ("distribuicao", "distribuicoes"):
                saida = "distribuicao"
            elif saida in ("tabelacruzada",):
                saida = "tabela_cruzada"
            elif saida in ("variabilidade", "desvio", "variancia"):
                saida = "variabilidade"
            elif saida in ("frequencias", "frequentes", "moda"):
                saida = "frequencias"
            elif saida in ("temporal", "tendenciatemporal", "serie"):
                saida = "temporal"
            elif saida in ("dispersao", "scatter"):
                saida = "dispersao"

        if saida in CATEGORIAS:
            return saida
        return _chutar_regra(pergunta)
    except Exception:
        return _chutar_regra(pergunta)
