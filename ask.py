#!/usr/bin/env python3
"""
ask.py — Paso 5 del proyecto "Chat con mi Vault (RAG)".

Generación con citas: toma una pregunta, recupera los chunks más relevantes
(Paso 4, `search.py`) y se los pasa a Claude para que responda citando la
fuente [1], [2], ... Cierra el loop RAG completo:

    pregunta → embedding → Chroma → top-k chunks → Claude → respuesta con citas

Como herramienta (CLI):
    python ask.py qué es overfitting
    python ask.py "cómo funciona el descenso de gradiente"

Como función (lo usará el Paso 6 — Streamlit):
    from ask import ask
    result = ask("qué es un embedding")
    result["answer"], result["sources"], result["usage"]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from search import search

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# .env junto a ESTE archivo (no al cwd): así funciona aunque la app
# se lance desde otro directorio (ej. el preview de Streamlit).
load_dotenv(Path(__file__).parent / ".env")

from config import GEN_MODEL as MODEL, TOP_K  # generation model + context size (override via .env)
MAX_TOKENS = 16000   # output cap, not a target — answers are typically far shorter

# Precio de claude-opus-4-8 por millón de tokens (para mostrar costo por pregunta)
PRICE_IN_PER_MTOK = 5.00
PRICE_OUT_PER_MTOK = 25.00

# El system prompt es FIJO (no interpolar nada dinámico): así Claude puede
# cachearlo como prefijo cuando haya conversación multi-turno en el Paso 6.
SYSTEM = """You are an assistant for the user's personal notes — an Obsidian-style \
Markdown vault. You answer questions using ONLY the provided context: excerpts \
retrieved from their notes.

Rules:
- Cite the source of every claim with [n], where n is the number of the excerpt.
- If the context does not contain the answer, say so plainly. Never make anything up.
- Answer in the same language as the question.
- Be direct, no filler — but complete: cover every relevant point the context offers
  (what it is, how it works, when it applies). Don't name a key concept without
  explaining it if the context provides the explanation."""


def build_context(hits: list[dict]) -> str:
    """Formatea los chunks recuperados como fragmentos numerados [1]..[k]."""
    blocks = []
    for i, hit in enumerate(hits, 1):
        head = f" › {hit['heading']}" if hit["heading"] else ""
        blocks.append(f"[{i}] {hit['title']}{head}\n{hit['text']}")
    return "\n\n---\n\n".join(blocks)


def ask(question: str, k: int = TOP_K, on_text=None) -> dict:
    """Loop RAG completo: recupera chunks, genera respuesta con citas.

    `on_text` (opcional): callback que recibe cada trozo de texto según llega
    (streaming). El CLI lo usa para imprimir en vivo; Streamlit hará lo mismo.
    """
    hits = search(question, k=k)
    context = build_context(hits)

    user_msg = (
        f"Contexto (fragmentos recuperados):\n\n{context}\n\n"
        f"Pregunta: {question}"
    )

    client = anthropic.Anthropic()
    parts: list[str] = []
    with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        thinking={"type": "adaptive"},
        system=SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        for text in stream.text_stream:
            parts.append(text)
            if on_text:
                on_text(text)
        final = stream.get_final_message()

    usage = final.usage
    cost = (
        usage.input_tokens * PRICE_IN_PER_MTOK
        + usage.output_tokens * PRICE_OUT_PER_MTOK
    ) / 1_000_000

    return {
        "answer": "".join(parts),
        "context": context,  # lo que Claude vio — lo necesitan los evals (groundedness)
        "sources": [
            {"n": i, "title": h["title"], "heading": h["heading"], "source": h["source"]}
            for i, h in enumerate(hits, 1)
        ],
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cost_usd": round(cost, 4),
        },
    }


def main():
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        sys.exit('Uso: python ask.py <tu pregunta>   (ej: python ask.py qué es overfitting)')
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("ERROR: falta OPENAI_API_KEY en .env (se usa para embeber la pregunta)")
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: falta ANTHROPIC_API_KEY en .env (se usa para generar la respuesta)")

    print(f"Pregunta: {question}\n" + "-" * 60)
    result = ask(question, on_text=lambda t: print(t, end="", flush=True))

    print("\n\nFuentes:")
    for s in result["sources"]:
        head = f" › {s['heading']}" if s["heading"] else ""
        print(f"  [{s['n']}] {s['title']}{head}")
        print(f"      {s['source']}")

    u = result["usage"]
    print(f"\n[{u['input_tokens']} tokens in · {u['output_tokens']} out · ~${u['cost_usd']}]")


if __name__ == "__main__":
    main()
