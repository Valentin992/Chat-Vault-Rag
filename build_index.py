#!/usr/bin/env python3
"""
build_index.py — Paso 3 del proyecto "Chat con mi Vault (RAG)".

Carga chunks_embedded.jsonl (del Paso 2) en una vector DB local (Chroma) que
persiste en ./chroma/. Aquí las notas se vuelven *buscables por significado*.

Al final hace UNA búsqueda de demostración para probar que funciona (eso es ya
un adelanto del Paso 4: retrieval).

    python build_index.py
"""

from __future__ import annotations

import json
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

load_dotenv()


# ─── CONFIG ───
EMBED_MODEL = "text-embedding-3-small"   # mismo modelo del Paso 2 (¡debe coincidir!)
INPUT_FILE = Path(__file__).parent / "chunks_embedded.jsonl"
CHROMA_DIR = Path(__file__).parent / "chroma"
COLLECTION = "vault"
ADD_BATCH = 500
DEMO_QUERY = "¿qué es el descenso de gradiente?"


def load_embedded() -> list[dict]:
    if not INPUT_FILE.exists():
        sys.exit(f"ERROR: falta {INPUT_FILE.name}. Corre antes: python embed_chunks.py")
    with INPUT_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    rows = load_embedded()
    print(f"Cargando {len(rows)} chunks en Chroma...")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Empezar limpio para que re-correr no duplique.
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},  # distancia coseno: 0 = idéntico, 2 = opuesto
    )

    # Cargar en lotes: ids, vectores, texto y metadata (para citar la fuente).
    for start in range(0, len(rows), ADD_BATCH):
        batch = rows[start:start + ADD_BATCH]
        collection.add(
            ids=[str(start + i) for i in range(len(batch))],
            embeddings=[r["embedding"] for r in batch],
            documents=[r["text"] for r in batch],
            metadatas=[{
                "source": r["source"],
                "title": r["title"],
                "heading": r["heading"],
                "n_chars": r["n_chars"],
            } for r in batch],
        )
        print(f"  cargados {min(start + ADD_BATCH, len(rows))}/{len(rows)}...")

    print(f"  colección '{COLLECTION}' = {collection.count()} vectores. Persistida en ./chroma/")

    # ── Demo del Paso 4: una búsqueda real por significado ──
    if not os.getenv("OPENAI_API_KEY"):
        print("(sin OPENAI_API_KEY → me salto la búsqueda de demo, pero el índice ya está)")
        return

    print("-" * 60)
    print(f"DEMO de búsqueda — pregunta: {DEMO_QUERY!r}")
    oai = OpenAI()
    qvec = oai.embeddings.create(model=EMBED_MODEL, input=[DEMO_QUERY]).data[0].embedding
    res = collection.query(query_embeddings=[qvec], n_results=3)
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        snippet = doc[:90].replace("\n", " ")
        head = meta["heading"] or "(sin encabezado)"
        print(f"  [dist {dist:.3f}] {meta['title']} › {head}")
        print(f"             {snippet}...")
    print("=" * 60)
    print("  Chroma listo. Siguiente → Paso 4: convertir la búsqueda en función reusable.")


if __name__ == "__main__":
    main()
