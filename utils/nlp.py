# utils/nlp.py
import os
import re
import streamlit as st
from typing import Optional
import requests

CATEGORIAS = [
    "tipos", "intervalo", "tendencia_central", "variabilidade", "frequencias",
    "outliers", "correlacao", "dispersao", "temporal", "clusters",
    "influencia", "distribuicao", "tabela_cruzada"
]

HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"

def _hf_token() -> Optional[str]:
    return st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")

def _chutar_regra(pergunta: str) -> str:
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
    if any(x in q for x in ["temporal", "série", "serie"]):
        return "temporal"
    if "cluster" in q or "agrup" in q:
        return "clusters"
    if "influenc" in q or "importanc" in q:
        return "influencia"
    if "distribui" in q:
        return "distribuicao"
    if "tabela cruzada" in q or "crosstab" in q:
        return "tabela_cruzada"
    return "tipos"

def interpretar_pergunta(pergunta: str) -> str:
    """Tenta classificar a pergunta via Hugging Face; se falhar, usa o fallback."""
    token = _hf_token()
    if not token:
        return _chutar_regra(pergunta)

    prompt = f"""
Você é um classificador para análise exploratória de dados (EDA).
Receba a pergunta do usuário e responda APENAS com UMA das categorias abaixo:
{', '.join(CATEGORIAS)}.

Pergunta: "{pergunta}"
Responda SOMENTE com a categoria.
"""
    try:
        resp = requests.post(
            HF_API_URL,
            headers={"Authorization": f"Bearer {token}"},
            json={"inputs": prompt, "parameters": {"max_new_tokens": 10}}
        )
        data = resp.json()
        saida = ""

        if isinstance(data, list) and "generated_text" in data[0]:
            saida = data[0]["generated_text"].strip().lower()
        elif isinstance(data, dict) and "generated_text" in data:
            saida = data["generated_text"].strip().lower()

        saida = re.sub(r"[^a-z_]", "", saida)
        return saida if saida in CATEGORIAS else _chutar_regra(pergunta)
    except Exception:
        return _chutar_regra(pergunta)
