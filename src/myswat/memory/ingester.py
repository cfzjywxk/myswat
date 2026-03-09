"""DocumentIngester — chunks and distills documents into knowledge entries."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from myswat.memory.store import MemoryStore

if TYPE_CHECKING:
    from myswat.agents.base import AgentRunner

# Max chars per chunk before sending to AI for distillation
CHUNK_SIZE = 8000
CHUNK_OVERLAP = 500

INGESTION_PROMPT = """You are a knowledge extraction specialist. Given the following document chunk,
extract the key knowledge items that would be useful for a software engineering team.

Source file: {source_file}
Chunk {chunk_index} of {total_chunks}

For each item, output a JSON object with:
- "category": one of "architecture", "pattern", "decision", "api_reference", "configuration", "lesson_learned"
- "title": concise title (max 100 chars)
- "content": detailed knowledge content (1-3 paragraphs)
- "tags": list of relevant tags (max 5)
- "relevance_score": float 0.0-1.0 (how important/reusable is this)

Output ONLY a JSON array of these objects. No other text.
If the chunk has no useful technical knowledge, output [].

DOCUMENT CHUNK:
{chunk}
"""


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks at paragraph boundaries."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Try to break at a paragraph boundary
        if end < len(text):
            # Look for double newline near the end
            boundary = text.rfind("\n\n", start + chunk_size // 2, end)
            if boundary != -1:
                end = boundary + 2
            else:
                # Fall back to single newline
                boundary = text.rfind("\n", start + chunk_size // 2, end)
                if boundary != -1:
                    end = boundary + 1

        chunks.append(text[start:end])
        start = end - overlap if end < len(text) else end

    return chunks


class DocumentIngester:
    """Ingests documents into knowledge entries via chunking + AI distillation."""

    def __init__(
        self,
        store: MemoryStore,
        runner: AgentRunner | None = None,
    ) -> None:
        self._store = store
        self._runner = runner

    def ingest_file(
        self, file_path: str, project_id: int, agent_id: int | None = None,
    ) -> list[int]:
        """Ingest a single file into knowledge entries.

        Returns list of created knowledge entry IDs.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            return []

        source_name = str(path.name)
        chunks = chunk_text(text)
        created_ids = []

        for i, chunk in enumerate(chunks):
            ids = self._ingest_chunk(
                chunk=chunk,
                chunk_index=i + 1,
                total_chunks=len(chunks),
                source_file=str(path),
                source_name=source_name,
                project_id=project_id,
                agent_id=agent_id,
            )
            created_ids.extend(ids)

        return created_ids

    def _ingest_chunk(
        self,
        chunk: str,
        chunk_index: int,
        total_chunks: int,
        source_file: str,
        source_name: str,
        project_id: int,
        agent_id: int | None,
    ) -> list[int]:
        """Process a single chunk through AI distillation."""
        if self._runner is None:
            # No AI runner — store the raw chunk as a single knowledge entry
            return self._store_raw_chunk(
                chunk, chunk_index, source_file, source_name, project_id, agent_id,
            )

        prompt = INGESTION_PROMPT.format(
            source_file=source_name,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            chunk=chunk,
        )

        response = self._runner.invoke(prompt)
        if not response.success:
            # Fallback: store raw chunk
            return self._store_raw_chunk(
                chunk, chunk_index, source_file, source_name, project_id, agent_id,
            )

        from myswat.memory.compactor import parse_compaction_output
        items = parse_compaction_output(response.content)

        if not items:
            return self._store_raw_chunk(
                chunk, chunk_index, source_file, source_name, project_id, agent_id,
            )

        created_ids = []
        for item in items:
            category = item.get("category", "architecture")
            title = item.get("title", f"{source_name} chunk {chunk_index}")[:512]
            content = item.get("content", "")
            if not content:
                continue
            tags = item.get("tags", [])
            if source_name not in tags:
                tags.append(source_name)
            relevance = min(max(float(item.get("relevance_score", 0.7)), 0.0), 1.0)

            kid = self._store.store_knowledge(
                project_id=project_id,
                agent_id=agent_id,
                source_file=source_file,
                category=category,
                title=title,
                content=content,
                tags=tags[:5],
                relevance_score=relevance,
                confidence=0.9,
            )
            created_ids.append(kid)

        return created_ids

    def _store_raw_chunk(
        self, chunk: str, chunk_index: int,
        source_file: str, source_name: str,
        project_id: int, agent_id: int | None,
    ) -> list[int]:
        """Store a raw chunk without AI distillation."""
        kid = self._store.store_knowledge(
            project_id=project_id,
            agent_id=agent_id,
            source_file=source_file,
            category="architecture",
            title=f"{source_name} (chunk {chunk_index})",
            content=chunk[:4000],
            tags=[source_name],
            relevance_score=0.5,
            confidence=0.7,
        )
        return [kid]
