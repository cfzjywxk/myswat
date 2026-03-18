"""Embedding support: local BGE-M3 with TiDB built-in fallback."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

_model = None
_available: bool | None = None


def is_available() -> bool:
    """Check if FlagEmbedding is installed (local BGE-M3)."""
    global _available
    if _available is None:
        try:
            import FlagEmbedding  # noqa: F401
            _available = True
        except ImportError:
            _available = False
    return _available


def _get_model():
    """Lazy-load BGE-M3 model on first use (~2GB download on first run)."""
    global _model
    if _model is None:
        import contextlib
        import io
        import logging
        import os
        import warnings

        # Suppress noisy model-loading output
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        logging.getLogger("FlagEmbedding").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", message=".*XLMRobertaTokenizerFast.*")
        warnings.filterwarnings("ignore", message=".*`__call__` method is faster.*")

        from FlagEmbedding import BGEM3FlagModel
        # Redirect stderr to suppress tqdm progress bars from huggingface_hub
        with contextlib.redirect_stderr(io.StringIO()):
            _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
    return _model


def preload_model() -> None:
    """Preload the embedding model in background. Call from a daemon thread."""
    if is_available():
        _get_model()


def embed(text: str) -> list[float] | None:
    """Generate a 1024-dim embedding for a single text. Returns None if model unavailable."""
    if not is_available():
        return None
    model = _get_model()
    result = model.encode([text])
    return result["dense_vecs"][0].tolist()


def embed_batch(texts: list[str]) -> list[list[float] | None]:
    """Generate 1024-dim embeddings for a batch of texts."""
    if not texts:
        return []
    if not is_available():
        return [None] * len(texts)
    model = _get_model()
    result = model.encode(texts)
    return [v.tolist() for v in result["dense_vecs"]]


def embedding_to_sql(vec: list[float]) -> str:
    """Convert a Python list of floats to a TiDB VEC_FROM_TEXT argument."""
    return json.dumps(vec)


# ── TiDB built-in embedding helpers ──────────────────────────────────


def tidb_embed_expr(model: str) -> str:
    """Return the SQL expression for TiDB's built-in EMBEDDING() function.

    Usage in SQL: ``f"... {tidb_embed_expr(model)} ..."`` with the text
    passed as a query parameter (%s).
    """
    # EMBEDDING('model', %s) is a TiDB built-in that returns VECTOR directly.
    return f"EMBEDDING('{model}', %s)"


def resolve_embed_sql(
    text: str,
    tidb_model: str = "",
    backend: str = "auto",
) -> tuple[str, list]:
    """Decide how to embed *text* and return (sql_fragment, params).

    Returns one of:
      - ("VEC_FROM_TEXT(%s)", [json_vec])  — local model produced a vector
      - ("EMBEDDING('model', %s)", [text]) — fall back to TiDB built-in
      - ("NULL", [])                       — no embedding available

    The caller splices the sql_fragment into the INSERT/SELECT and appends
    params to its argument list.
    """
    mode = (backend or "auto").strip().lower()
    if mode not in {"auto", "local", "tidb"}:
        mode = "auto"

    # Try local first unless TiDB-only mode was requested.
    if mode != "tidb":
        vec = embed(text)
        if vec is not None:
            return "VEC_FROM_TEXT(%s)", [embedding_to_sql(vec)]
        if mode == "local":
            return "NULL", []

    # Fall back to TiDB built-in when allowed.
    if tidb_model:
        return tidb_embed_expr(tidb_model), [text]

    # No embedding available
    return "NULL", []
