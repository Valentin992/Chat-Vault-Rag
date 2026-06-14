#!/usr/bin/env python3
"""
embed_chunks.py — Paso 2 del proyecto "Chat con mi Vault (RAG)".

Lee chunks.jsonl (del Paso 1), genera un embedding por chunk y guarda todo en
chunks_embedded.jsonl. El Paso 3 cargará ese archivo en la vector DB (Chroma).

    python embed_chunks.py

NOTA IMPORTANTE — Anthropic NO tiene API de embeddings.
  Claude (Anthropic) se usa para *generar* la respuesta (Paso 5), no para embeber.
  Para embeddings se usa otro proveedor. Aquí: OpenAI `text-embedding-3-small`
  (lo que usa el curso de DeepLearning.AI, barato y confiable). Alternativa
  recomendada por Anthropic: Voyage AI (ver README para cambiar de proveedor).

Necesita una API key de OpenAI en un archivo `.env` (ver .env.example).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

load_dotenv()  # carga las variables de .env (la API key)


# ─────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────

# Modelo de embeddings. text-embedding-3-small: 1536 dimensiones, ~$0.02 / 1M
# tokens. Para tu vault (~90K tokens) el costo total es de centavos.
# Alternativa de más calidad: "text-embedding-3-large" (3072 dims, ~6x el costo).
MODEL = "text-embedding-3-small"

INPUT_FILE = Path(__file__).parent / "chunks.jsonl"
OUTPUT_FILE = Path(__file__).parent / "chunks_embedded.jsonl"

# Cuántos chunks mandar por llamada a la API. La API acepta listas; agrupar
# reduce el número de requests (~1053 chunks / 100 ≈ 11 llamadas).
BATCH_SIZE = 100

# Reintentos cuando OpenAI devuelve rate limit (429). El tier inicial tiene un
# límite de tokens por minuto (~40K TPM); al pegarle, esperamos y reintentamos
# el mismo lote. Una request rechazada por 429 no se cobra.
RETRY_WAIT_SECONDS = 20
MAX_RETRIES = 8


# ─────────────────────────────────────────────────────────────────────────
#  Lógica
# ─────────────────────────────────────────────────────────────────────────

def load_chunks() -> list[dict]:
    """Lee chunks.jsonl → lista de dicts."""
    if not INPUT_FILE.exists():
        sys.exit(f"ERROR: no encuentro {INPUT_FILE.name}. Corre antes: python chunk_vault.py")
    with INPUT_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_embed_text(chunk: dict) -> str:
    """
    Texto que realmente se embebe. Antepone el título de la nota y el rastro de
    encabezados al cuerpo del chunk: el vector captura también *dónde vive* el
    fragmento, lo que mejora la recuperación. (Esta era la idea del Paso 1.)
    """
    parts = [chunk["title"]]
    if chunk.get("heading"):
        parts.append(chunk["heading"])
    context = " — ".join(parts)
    return f"{context}\n\n{chunk['text']}"


def embed_batches(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Embebe todos los textos, en lotes, con reintentos por rate limit."""
    vectors: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = client.embeddings.create(model=MODEL, input=batch)
                # La API devuelve los embeddings en el mismo orden que el input.
                vectors.extend(item.embedding for item in resp.data)
                break
            except RateLimitError:
                if attempt == MAX_RETRIES:
                    raise
                print(f"  rate limit (tokens/min) — esperando {RETRY_WAIT_SECONDS}s "
                      f"y reintentando (intento {attempt}/{MAX_RETRIES})...")
                time.sleep(RETRY_WAIT_SECONDS)
        print(f"  embebidos {min(start + BATCH_SIZE, total)}/{total}...")
    return vectors


def main():
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit(
            "ERROR: falta OPENAI_API_KEY.\n"
            "  1. Copia .env.example a .env\n"
            "  2. Crea una key en https://platform.openai.com/api-keys\n"
            "  3. Pégala en .env y vuelve a correr este script."
        )

    chunks = load_chunks()
    texts = [build_embed_text(c) for c in chunks]

    print(f"Embebiendo {len(chunks)} chunks con '{MODEL}'...")
    client = OpenAI()  # lee OPENAI_API_KEY del entorno
    try:
        vectors = embed_batches(client, texts)
    except Exception as e:
        sys.exit(f"\nERROR llamando a la API de OpenAI: {e}")

    # Guardar: cada chunk + su embedding → lo consume el Paso 3.
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for chunk, vector in zip(chunks, vectors):
            record = {**chunk, "embedding": vector}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── Resumen ──
    dims = len(vectors[0]) if vectors else 0
    total_chars = sum(len(t) for t in texts)
    est_tokens = total_chars / 4  # aproximación
    est_cost = est_tokens / 1_000_000 * 0.02  # $0.02 / 1M tokens (3-small)
    print("=" * 60)
    print("  EMBEDDINGS LISTOS")
    print("=" * 60)
    print(f"  Chunks embebidos : {len(vectors)}")
    print(f"  Dimensiones      : {dims} por vector")
    print(f"  Costo estimado   : ~${est_cost:.4f} USD")
    print(f"  Guardado en      : {OUTPUT_FILE.name}")
    print("=" * 60)
    print("  Siguiente → Paso 3: cargar los embeddings en Chroma (vector DB).")


if __name__ == "__main__":
    main()
