#!/usr/bin/env python3
"""
app.py — Paso 6 del proyecto "Chat con mi Vault (RAG)".

Interfaz de chat (Streamlit) sobre el loop RAG del Paso 5: cada pregunta
pasa por `ask()` → retrieval (Chroma) + generación con citas (Claude).
El streaming llega en vivo al navegador vía el callback `on_text` —
el mismo que usa el CLI.

Correr:
    streamlit run app.py
"""

import streamlit as st

from ask import ask

st.set_page_config(page_title="Chat con mi Vault", page_icon="💬")

st.title("💬 Chat con mi Vault")
st.caption("RAG sobre mis notas de Obsidian — cada respuesta cita la nota fuente [n]")

# Streamlit re-ejecuta TODO el script en cada interacción (su modelo mental
# central). El historial vive en session_state para sobrevivir esos reruns.
if "history" not in st.session_state:
    st.session_state.history = []  # [{question, answer, sources, usage}]

# --- Sidebar: estado de la sesión ---
with st.sidebar:
    st.header("Sesión")
    total = sum(t["usage"]["cost_usd"] for t in st.session_state.history)
    st.metric("Preguntas", len(st.session_state.history))
    st.metric("Costo acumulado", f"${total:.4f}")
    st.divider()
    st.markdown(
        "**Pipeline:** pregunta → embedding (OpenAI) → Chroma top-5 → "
        "Claude Opus 4.8 → respuesta con citas `[n]`."
    )
    st.markdown(
        "Si la respuesta no está en las notas, el asistente lo dice — "
        "no inventa (regla anti-alucinación)."
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
