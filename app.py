#!/usr/bin/env python3
"""
app.py — Paso 6 del proyecto "Chat con mi Vault (RAG)".

Interfaz de chat (Streamlit) sobre el loop RAG del Paso 5: cada pregunta
pasa por `ask()` → retrieval (Chroma) + generación con citas (Claude).
El streaming llega en vivo al navegador vía el callback `on_text` —
el mismo que usa el CLI.

Correr localmente:
    streamlit run app.py

Deploy en Streamlit Cloud:
    Añade OPENAI_API_KEY y ANTHROPIC_API_KEY en Settings → Secrets.
    En el primer arranque, la app construye automáticamente el índice
    vectorial desde demo_vault/ (~30 s, costo ~$0.002 en embeddings).
"""

import os

# Load API keys from Streamlit secrets (Streamlit Cloud) into os.environ
# BEFORE importing ask/search, which call load_dotenv() at import time.
# load_dotenv() does not overwrite existing env vars, so this takes priority.
try:
    import streamlit as _st
    for _key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        if _key not in os.environ:
            _val = _st.secrets.get(_key)
            if _val:
                os.environ[_key] = _val
except Exception:
    pass

import streamlit as st

from ask import ask
from demo_setup import CHROMA_DIR, ensure_index

st.set_page_config(page_title="Chat con mi Vault", page_icon="💬")

# ── Auto-build demo index on cold start ───────────────────────────────────────────
if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
    _status = st.empty()
    with st.spinner("Construyendo índice vectorial… (~30 s la primera vez)"):
        ensure_index(status_fn=lambda msg: _status.caption(msg))
    _status.empty()
    st.rerun()

st.title("💬 Chat con mi Vault")
st.caption(
    "RAG sobre notas de ML/AI — cada respuesta cita la nota fuente [n]  \n"
    "> **Demo pública:** corpus sintético de 12 notas. "
    "El vault real (privado) tiene 100+ notas de estudio con métricas medidas."
)

# Streamlit re-ejecuta TODO el script en cada interacción (su modelo mental
# central). El historial vive en session_state para sobrevivir esos reruns.
if "history" not in st.session_state:
    st.session_state.history = []  # [{question, answer, sources, usage}]

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Sesión")
    total = sum(t["usage"]["cost_usd"] for t in st.session_state.history)
    st.metric("Preguntas", len(st.session_state.history))
    st.metric("Costo acumulado", f"${total:.4f}")
    st.divider()
    st.markdown(
        "**Pipeline:** pregunta → embedding (OpenAI) → Chroma top-8 → "
        "Claude Opus 4.8 → respuesta con citas `[n]`."
    )
    st.markdown(
        "Si la respuesta no está en las notas, el asistente lo dice — "
        "no inventa (regla anti-alucinación)."
    )
    st.divider()
    st.markdown("**Temas en este vault:**")
    st.markdown(
        "- Embeddings, Transformers, RAG\n"
        "- Descenso de Gradiente, Backpropagation\n"
        "- Overfitting, Bias-Variance Tradeoff\n"
        "- Funciones de Pérdida, Redes Neuronales\n"
        "- Validación Cruzada, Vector DBs\n"
        "- Prompt Engineering"
    )


def render_turn(turn: dict) -> None:
    """Dibuja un turno completo del historial: pregunta, respuesta y fuentes."""
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        u = turn["usage"]
        with st.expander(f"Fuentes ({len(turn['sources'])}) · ${u['cost_usd']}"):
            for s in turn["sources"]:
                head = f" › {s['heading']}" if s["heading"] else ""
                st.markdown(f"**[{s['n']}] {s['title']}**{head}  \n`{s['source']}`")


# Redibujar el historial en cada rerun
for turn in st.session_state.history:
    render_turn(turn)

question = st.chat_input("Pregúntale algo a tus notas…")
if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_Buscando en el vault…_")
        parts: list[str] = []

        def on_text(t: str) -> None:
            parts.append(t)
            placeholder.markdown("".join(parts) + "▌")  # cursor de streaming

        result = ask(question, on_text=on_text)
        placeholder.markdown(result["answer"])

    st.session_state.history.append(
        {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "usage": result["usage"],
        }
    )
    # Rerun para que el historial y las métricas del sidebar se refresquen
    st.rerun()
