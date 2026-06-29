#!/usr/bin/env python3
"""
config.py — central configuration for VaultChat.

Everything a user might want to tune lives here, in one place. Every value can
be overridden with an environment variable (put them in a `.env` file — see
`.env.example`). Each script imports what it needs from this module, so you
never have to edit the same setting in three files.
"""

from __future__ import annotations

import os
from pathlib import Path

# Load .env (sitting next to this file) if python-dotenv is installed.
# Wrapped in try/except so the no-dependency chunking step still runs.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

ROOT = Path(__file__).parent


# ─── Your vault ──────────────────────────────────────────────────────────
# Point this at YOUR Obsidian vault (or any folder of Markdown files).
# Set VAULT_PATH in .env, or pass the path as an argument to chunk_vault.py.
DEFAULT_VAULT_PATH = "C:/path/to/your/vault"
VAULT_PATH = Path(os.getenv("VAULT_PATH", DEFAULT_VAULT_PATH))

# Folders to skip (not knowledge — config, trash, plugin logs, etc.).
# Obsidian/git internals are skipped by default. Add your own, comma-separated,
# via SKIP_DIRS in .env — e.g.  SKIP_DIRS=Templates,Archive,Daily Notes
_DEFAULT_SKIP = {".obsidian", ".trash", ".git"}
_EXTRA_SKIP = {d.strip() for d in os.getenv("SKIP_DIRS", "").split(",") if d.strip()}
SKIP_DIRS = _DEFAULT_SKIP | _EXTRA_SKIP


# ─── Models ──────────────────────────────────────────────────────────────
# Embeddings (OpenAI). Must be the SAME model at index time and query time.
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
# Answer generation (Anthropic / Claude).
GEN_MODEL = os.getenv("GEN_MODEL", "claude-opus-4-8")


# ─── Retrieval ───────────────────────────────────────────────────────────
# How many chunks to retrieve and hand to the model as context.
# (Raised from 5 → 8 after evals showed fact-bearing chunks ranking 6-7.)
TOP_K = int(os.getenv("TOP_K", "8"))


# ─── Chunking ────────────────────────────────────────────────────────────
TARGET_CHARS = int(os.getenv("TARGET_CHARS", "1000"))   # ~250 tokens
MAX_CHARS = int(os.getenv("MAX_CHARS", "1500"))         # hard cap per chunk
OVERLAP_CHARS = int(os.getenv("OVERLAP_CHARS", "150"))  # context carried between chunks
MIN_CHARS = int(os.getenv("MIN_CHARS", "50"))           # below this = noise, dropped


# ─── Storage / files ─────────────────────────────────────────────────────
CHROMA_DIR = ROOT / "chroma"
COLLECTION = os.getenv("COLLECTION", "vault")
CHUNKS_FILE = ROOT / "chunks.jsonl"
EMBEDDED_FILE = ROOT / "chunks_embedded.jsonl"
