from datetime import datetime
import streamlit as st

_KEY = "conclusoes_agente"

def _store():
    """Garante a lista de conclusões na sessão do Streamlit."""
    if _KEY not in st.session_state:
        st.session_state[_KEY] = []
    return st.session_state[_KEY]

def salvar(pergunta: str, resposta: str) -> None:
    """
    Registra uma conclusão do agente.
    - pergunta: texto perguntado pelo usuário
    - resposta: texto-resumo que o agente retornou
    """
    itens = _store()
    resumo = (resposta or "").strip()
    if len(resumo) > 600:
        resumo = resumo[:600] + "…"
    itens.append({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "pergunta": (pergunta or "").strip(),
        "resumo": resumo,
    })

def all_md() -> str:
    """Renderiza todas as conclusões em Markdown para a aba ' Conclusões'."""
    itens = _store()
    if not itens:
        return "Ainda não há conclusões registradas. Faça perguntas ao agente e volte aqui."
    linhas = ["### Conclusões do agente", ""]
    for i, it in enumerate(itens, 1):
        linhas.append(
            f"**{i}.** _{it['ts']}_  \n"
            f"**Pergunta:** {it['pergunta']}  \n"
            f"**Conclusão:** {it['resumo']}"
        )
    return "\n\n".join(linhas)

def clear() -> None:
    """Limpa todas as conclusões da sessão."""
    st.session_state[_KEY] = []
