#!/usr/bin/env python3
"""
demo_setup.py — Builds the Chroma index from demo_vault/ synthetic notes.

Called automatically by app.py on first run (or after the index is cleared).
Can also be run standalone to pre-build the index locally:
    python demo_setup.py
"""
from __future__ import annotations

import re
from pathlib import Path

import chromadb
from openai import OpenAI

DEMO_VAULT = Path(__file__).parent / "demo_vault"
CHROMA_DIR = Path(__file__).parent / "chroma"
COLLECTION = "vault"
EMBED_MODEL = "text-embedding-3-small"
TARGET_CHARS = 800
OVERLAP_CHARS = 100


def _chunk_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    title = path.stem.replace("_", " ").title()
    parts = re.split(r"^(#{1,3}\s+.+)$", text, flags=re.MULTILINE)

    chunks: list[dict] = []
    heading = ""
    buf = ""
    for part in parts:
        if re.match(r"^#{1,3}\s+", part):
            if buf.strip():
                chunks.extend(_split(buf.strip(), title, heading, path))
            heading = part.strip().lstrip("#").strip()
            buf = ""
        else:
            buf += part
    if buf.strip():
        chunks.extend(_split(buf.strip(), title, heading, path))
    return chunks


def _split(text: str, title: str, heading: str, path: Path) -> list[dict]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + TARGET_CHARS, len(text))
        if end < len(text):
            brk = text.rfind("\n", start, end)
            if brk > start:
                end = brk
        snippet = text[start:end].strip()
        if snippet:
            prefix = title + (f" — {heading}" if heading else "")
            chunks.append({
                "text": f"{prefix}\n\n{snippet}",
                "source": path.name,
                "title": title,
                "heading": heading,
                "n_chars": len(snippet),
            })
        start = end - OVERLAP_CHARS if end < len(text) else end
    return chunks


def ensure_index(status_fn=None) -> None:
    """Build Chroma index from demo_vault/ if it does not already exist."""
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        return

    if status_fn:
        status_fn("Leyendo notas del vault de demo…")

    chunks: list[dict] = []
    for md in sorted(DEMO_VAULT.glob("*.md")):
        chunks.extend(_chunk_file(md))

    if status_fn:
        status_fn(f"Generando embeddings para {len(chunks)} fragmentos (OpenAI)…")

    oai = OpenAI()
    embeddings: list[list[float]] = []
    for i in range(0, len(chunks), 100):
        batch = [c["text"] for c in chunks[i : i + 100]]
        resp = oai.embeddings.create(model=EMBED_MODEL, input=batch)
        embeddings.extend(d.embedding for d in resp.data)

    if status_fn:
        status_fn("Cargando vectores en ChromaDB…")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[str(i) for i in range(len(chunks))],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[
            {
                "source": c["source"],
                "title": c["title"],
                "heading": c["heading"],
                "n_chars": c["n_chars"],
            }
            for c in chunks
        ],
    )
    if status_fn:
        status_fn(f"Índice listo: {len(chunks)} fragmentos cargados en ChromaDB.")


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("ERROR: falta OPENAI_API_KEY en .env")
    ensure_index(status_fn=print)
    print("Done. Index built in ./chroma/")
