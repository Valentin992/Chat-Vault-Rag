#!/usr/bin/env python3
"""
chunk_vault.py — Paso 1 del proyecto "Chat con mi Vault (RAG)".

Lee todos los .md del vault de Obsidian, los divide en *chunks* (fragmentos)
con metadata para citar la nota fuente, y los guarda en chunks.jsonl.

El Paso 2 (embeddings) leerá ese chunks.jsonl. Por eso este script no necesita
ninguna API ni librería externa: corre con Python estándar.

    python chunk_vault.py                    # usa la variable de entorno VAULT_PATH
    python chunk_vault.py "D:\\otro\\vault"  # o le pasas otra ruta

La TAREA del Paso 1 es decidir el tamaño de chunk y documentar POR QUÉ.
Por eso los parámetros están arriba, comentados, para que experimentes.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# Que los acentos y emojis se impriman bien en la consola de Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

if load_dotenv:
    load_dotenv(Path(__file__).parent / ".env")


# ─────────────────────────────────────────────────────────────────────────
#  CONFIG — esto es lo que experimentas en el Paso 1
# ─────────────────────────────────────────────────────────────────────────

DEFAULT_VAULT_PATH = "C:/path/to/your/vault"
VAULT_PATH = Path(os.getenv("VAULT_PATH", DEFAULT_VAULT_PATH))

# Carpetas que NO son notas de conocimiento → se omiten del corpus.
#  - .obsidian / .trash : config y papelera de Obsidian
#  - copilot            : prompts de plugin y logs de chat (ruido)
#  - 05 - Templates     : plantillas con sintaxis {{title}} (ruido)
SKIP_DIRS = {".obsidian", ".trash", "copilot", "05 - Templates"}

# Tamaño objetivo de cada chunk, en caracteres. ~1000 chars ≈ ~250 tokens.
# POR QUÉ este tamaño:
#   - Pequeño → cada chunk es un pasaje enfocado → mejores embeddings y
#     citas precisas a la nota fuente.
#   - Grande  → cada chunk se sostiene solo (lleva su propio contexto).
#   Tus notas de concepto son atómicas: muchas entran en 1-2 chunks.
TARGET_CHARS = 1000

# Tope duro: si una sección pasa de esto, se parte aunque quede a la mitad.
MAX_CHARS = 1500

# Solapamiento entre chunks consecutivos (en caracteres). Arrastra un poco
# de contexto de un chunk al siguiente para no cortar una idea en la frontera.
# (Es la misma idea de "chunk overlap" que verás en el curso de DeepLearning.AI.)
OVERLAP_CHARS = 150

# Tamaño mínimo: los chunks más chicos que esto se descartan como ruido.
# Decisión (2026-06-09): a <50 chars TODO eran divisores (`---`), links sueltos
# (`[[Naive Bayes]]`), navegación y boilerplate de plantilla — nada útil para
# recuperar. ~12 tokens. Subir este número descarta más; bajarlo conserva ruido.
MIN_CHARS = 50

# Archivo de salida que consumirá el Paso 2 (embeddings).
OUTPUT_FILE = Path(__file__).parent / "chunks.jsonl"


# ─────────────────────────────────────────────────────────────────────────
#  Estructura de un chunk
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text: str       # el fragmento de texto que se va a embeber
    source: str     # ruta relativa, p.ej. "02 - Conceptos/RAG.md" (para citar)
    title: str      # nombre de la nota (= wikilink de Obsidian [[title]])
    heading: str    # rastro de encabezados, p.ej. "Embeddings > Por qué funcionan"
    index: int      # índice del chunk dentro de su nota
    n_chars: int     # tamaño, para inspeccionar la distribución


# ─────────────────────────────────────────────────────────────────────────
#  Lógica de chunking
# ─────────────────────────────────────────────────────────────────────────

FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def iter_markdown_files(vault: Path):
    """Devuelve todos los .md del vault, saltando las carpetas de SKIP_DIRS."""
    for path in vault.rglob("*.md"):
        rel_parts = path.relative_to(vault).parts
        if any(part in SKIP_DIRS for part in rel_parts):
            continue
        yield path


def strip_frontmatter(text: str) -> str:
    """Quita el bloque YAML (--- ... ---) del inicio de la nota."""
    return FRONTMATTER_RE.sub("", text, count=1)


def split_into_sections(text: str):
    """
    Parte la nota por encabezados markdown, llevando el rastro de la jerarquía.
    Devuelve una lista de (heading_trail, body), p.ej.:
        ("RAG > Embeddings", "Los embeddings convierten texto en vectores...")
    """
    sections = []
    heading_stack: list[tuple[int, str]] = []  # (nivel, título)
    buf: list[str] = []

    def trail() -> str:
        return " > ".join(title for _, title in heading_stack)

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append((trail(), body))
        buf.clear()

    for line in text.splitlines():
        m = HEADER_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
        else:
            buf.append(line)
    flush()
    return sections


def split_oversized(body: str) -> list[str]:
    """
    Parte una sección demasiado grande en piezas de ~TARGET_CHARS, primero por
    párrafos (líneas en blanco) y, como último recurso, por corte duro.
    Añade OVERLAP_CHARS de contexto del chunk anterior.
    """
    if len(body) <= MAX_CHARS:
        return [body]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    pieces: list[str] = []
    current = ""

    for para in paragraphs:
        if not current:
            current = para
        elif len(current) + 2 + len(para) <= TARGET_CHARS:
            current += "\n\n" + para
        else:
            pieces.append(current)
            overlap = current[-OVERLAP_CHARS:] if OVERLAP_CHARS else ""
            current = (overlap + "\n\n" + para).strip() if overlap else para
    if current:
        pieces.append(current)

    # Si algún párrafo solo ya supera MAX_CHARS, corte duro por caracteres.
    final: list[str] = []
    step = max(1, TARGET_CHARS - OVERLAP_CHARS)
    for piece in pieces:
        if len(piece) <= MAX_CHARS:
            final.append(piece)
        else:
            for i in range(0, len(piece), step):
                final.append(piece[i:i + TARGET_CHARS])
    return final


def chunk_file(path: Path, vault: Path) -> list[Chunk]:
    """Convierte una nota en una lista de Chunks."""
    raw = path.read_text(encoding="utf-8")
    text = strip_frontmatter(raw)

    source = path.relative_to(vault).as_posix()
    title = path.stem

    chunks: list[Chunk] = []
    for heading, body in split_into_sections(text):
        for piece in split_oversized(body):
            if len(piece) < MIN_CHARS:
                continue  # descartar ruido: divisores, links sueltos, boilerplate
            chunks.append(Chunk(
                text=piece,
                source=source,
                title=title,
                heading=heading,
                index=len(chunks),
                n_chars=len(piece),
            ))
    return chunks


# ─────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    vault = Path(sys.argv[1]) if len(sys.argv) > 1 else VAULT_PATH
    if vault.as_posix() == DEFAULT_VAULT_PATH:
        sys.exit(
            "ERROR: configura VAULT_PATH en tu entorno o pasa la ruta del vault "
            "como argumento."
        )
    if not vault.exists():
        sys.exit(f"ERROR: no encuentro el vault en {vault}")

    files = list(iter_markdown_files(vault))
    all_chunks: list[Chunk] = []
    for path in files:
        all_chunks.extend(chunk_file(path, vault))

    # Guardar a JSONL (una línea JSON por chunk) → lo lee el Paso 2.
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    # ── Resumen (el dopamine hit: ver que funcionó) ──
    sizes = [c.n_chars for c in all_chunks] or [0]
    print("=" * 60)
    print("  CHUNKING LISTO")
    print("=" * 60)
    print(f"  Vault            : {vault}")
    print(f"  Notas procesadas : {len(files)}")
    print(f"  Chunks generados : {len(all_chunks)}")
    print(f"  Tamaño chunk     : min {min(sizes)} / "
          f"prom {sum(sizes)//len(sizes)} / max {max(sizes)} chars")
    print(f"  Guardado en      : {OUTPUT_FILE.name}")
    print("-" * 60)
    if all_chunks:
        sample = all_chunks[len(all_chunks) // 2]  # un chunk del medio
        print("  EJEMPLO DE CHUNK:")
        print(f"    nota    : {sample.title}")
        print(f"    fuente  : {sample.source}")
        print(f"    heading : {sample.heading or '(sin encabezado)'}")
        print(f"    chars   : {sample.n_chars}")
        preview = sample.text[:280].replace("\n", " ")
        print(f"    texto   : {preview}...")
    print("=" * 60)
    print("  Siguiente → Paso 2: generar embeddings de cada chunk.")


if __name__ == "__main__":
    main()
