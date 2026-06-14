#!/usr/bin/env python3
"""
search.py — Paso 4 del proyecto "Chat con mi Vault (RAG)".

Retrieval reusable: dada una pregunta, embebe la pregunta con el MISMO modelo del
Paso 2, busca en Chroma los chunks más parecidos y los devuelve con su fuente.

Como herramienta (CLI):
    python search.py qué es overfitting
    python search.py "cómo aprenden las redes neuronales"

Como función (lo usará el Paso 5 — generación):
    from search import search
    hits = search("qué es un embedding", k=5)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

load_dotenv(Path(__file__).parent / ".env")  # .env junto a este archivo, no al cwd

EMBED_MODEL = "text-embedding-3-small"   # debe coincidir con el Paso 2
CHROMA_DIR = Path(__file__).parent / "chroma"
COLLECTION = "vault"
TOP_K = 5


def search(question: str, k: int = TOP_K) -> list[dict]:
    """Devuelve los k chunks más relevantes a `question`, cada uno con su fuente."""
    oai = OpenAI()
    qvec = oai.embeddings.create(model=EMBED_MODEL, input=[question]).data[0].embedding

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION)
    res = collection.query(query_embeddings=[qvec], n_results=k)

    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({
            "text": doc,
            "source": meta["source"],
            "title": meta["title"],
            "heading": meta["heading"],
            "distance": dist,
        })
    return hits


def main():
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        sys.exit('Uso: python search.py <tu pregunta>   (ej: python search.py qué es overfitting)')
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("ERROR: falta OPENAI_API_KEY en .env")
    if not CHROMA_DIR.exists():
        sys.exit("ERROR: no hay índice. Corre antes: python build_index.py")

    print(f"Pregunta: {question!r}\n" + "-" * 60)
    for i, hit in enumerate(search(question), 1):
        snippet = hit["text"][:120].replace("\n", " ")
        head = hit["heading"] or "(sin encabezado)"
        print(f"{i}. [dist {hit['distance']:.3f}] {hit['title']} › {head}")
        print(f"   fuente: {hit['source']}")
        print(f"   {snippet}...\n")


if __name__ == "__main__":
    main()
