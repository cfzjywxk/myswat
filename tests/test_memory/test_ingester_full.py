"""Comprehensive tests for myswat.memory.ingester module."""

import pytest
from unittest.mock import MagicMock, patch, mock_open

from myswat.memory.ingester import DocumentIngester, chunk_text
from myswat.agents.base import AgentResponse


# ---------------------------------------------------------------------------
# chunk_text tests
# ---------------------------------------------------------------------------

class TestChunkText:
    """Tests for the chunk_text function."""

    def test_short_text_returns_single_chunk(self):
        """Text shorter than chunk_size should return a single-element list."""
        text = "This is a short piece of text."
        result = chunk_text(text, chunk_size=8000, overlap=500)
        assert result == [text]

    def test_exact_chunk_size_returns_single_chunk(self):
        """Text exactly equal to chunk_size should return a single-element list."""
        text = "x" * 8000
        result = chunk_text(text, chunk_size=8000, overlap=500)
        assert result == [text]

    def test_empty_text_returns_single_chunk(self):
        """Empty text should return a list with the empty string."""
        result = chunk_text("", chunk_size=8000, overlap=500)
        assert result == [""]

    def test_long_text_splits_into_multiple_chunks(self):
        """Text longer than chunk_size should be split into multiple chunks."""
        # Build text that exceeds chunk_size
        text = "word " * 2000  # ~10000 chars
        result = chunk_text(text, chunk_size=100, overlap=10)
        assert len(result) > 1
        # Reassembled text (accounting for overlap) should cover the original
        for chunk in result:
            assert len(chunk) <= 100 + 50  # some tolerance for boundary splits

    def test_splits_at_paragraph_boundaries(self):
        """Chunks should prefer splitting at paragraph boundaries (\\n\\n)."""
        paragraph_a = "A" * 60
        paragraph_b = "B" * 60
        paragraph_c = "C" * 60
        text = paragraph_a + "\n\n" + paragraph_b + "\n\n" + paragraph_c
        # chunk_size chosen so that two paragraphs don't fit but one does
        result = chunk_text(text, chunk_size=80, overlap=10)
        assert len(result) >= 2
        # First chunk should end at or near a paragraph boundary
        # i.e., it should contain paragraph_a content
        assert paragraph_a[:50] in result[0]

    def test_falls_back_to_newline_split(self):
        """When no paragraph boundary exists, should fall back to \\n split."""
        line_a = "A" * 60
        line_b = "B" * 60
        line_c = "C" * 60
        text = line_a + "\n" + line_b + "\n" + line_c
        result = chunk_text(text, chunk_size=80, overlap=10)
        assert len(result) >= 2

    def test_overlap_between_chunks(self):
        """Consecutive chunks should share overlapping content."""
        # Create predictable text with paragraph boundaries
        paragraphs = [f"Paragraph {i}: " + "x" * 50 for i in range(20)]
        text = "\n\n".join(paragraphs)
        result = chunk_text(text, chunk_size=200, overlap=50)
        assert len(result) >= 2
        # Check that there is overlapping content between consecutive chunks
        for i in range(len(result) - 1):
            tail_of_current = result[i][-50:]
            head_of_next = result[i + 1][:100]
            # Some portion of the tail of the current chunk should appear
            # in the head of the next chunk (the overlap region)
            overlap_found = any(
                tail_of_current[j:j+20] in head_of_next
                for j in range(len(tail_of_current) - 20)
                if len(tail_of_current[j:j+20]) == 20
            )
            assert overlap_found, (
                f"No overlap detected between chunk {i} and chunk {i+1}"
            )

    def test_custom_chunk_size_and_overlap(self):
        """Custom chunk_size and overlap values should be respected."""
        text = "abcdefghij" * 10  # 100 chars
        result = chunk_text(text, chunk_size=30, overlap=5)
        assert len(result) > 1
        for chunk in result:
            # Chunks should not vastly exceed chunk_size
            assert len(chunk) <= 60  # generous tolerance for boundary logic

    def test_uses_default_constants(self):
        """When called without explicit args, should use module defaults."""
        short_text = "Hello, world!"
        result = chunk_text(short_text)
        assert result == [short_text]


# ---------------------------------------------------------------------------
# DocumentIngester.ingest_file tests
# ---------------------------------------------------------------------------

class TestIngestFile:
    """Tests for DocumentIngester.ingest_file."""

    def test_file_not_found_raises(self, tmp_path):
        """Ingesting a nonexistent file should raise FileNotFoundError."""
        store = MagicMock()
        ingester = DocumentIngester(store)
        nonexistent = str(tmp_path / "does_not_exist.txt")
        with pytest.raises(FileNotFoundError):
            ingester.ingest_file(nonexistent, project_id="proj1")

    def test_empty_file_returns_empty_list(self, tmp_path):
        """Ingesting an empty file should return an empty list."""
        store = MagicMock()
        ingester = DocumentIngester(store)
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")
        result = ingester.ingest_file(str(empty_file), project_id="proj1")
        assert result == []

    def test_normal_file_returns_knowledge_ids(self, tmp_path):
        """Ingesting a normal file should return a list of knowledge IDs."""
        store = MagicMock()
        store.store_knowledge.return_value = "kid-1"
        ingester = DocumentIngester(store)
        normal_file = tmp_path / "doc.txt"
        normal_file.write_text("Some content for ingestion.")
        result = ingester.ingest_file(str(normal_file), project_id="proj1")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_normal_file_processes_all_chunks(self, tmp_path):
        """Each chunk from chunk_text should be processed via _ingest_chunk."""
        store = MagicMock()
        ingester = DocumentIngester(store)
        # Patch _ingest_chunk to track calls
        ingester._ingest_chunk = MagicMock(return_value=["kid-1"])
        normal_file = tmp_path / "doc.txt"
        # Write enough text to produce multiple chunks
        normal_file.write_text("A" * 100)
        with patch("myswat.memory.ingester.chunk_text", return_value=["chunk0", "chunk1", "chunk2"]):
            result = ingester.ingest_file(str(normal_file), project_id="proj1")
        assert ingester._ingest_chunk.call_count == 3
        # Result should be the concatenation of all returned IDs
        assert len(result) == 3

    def test_ingest_file_passes_agent_id(self, tmp_path):
        """agent_id should be forwarded to _ingest_chunk."""
        store = MagicMock()
        ingester = DocumentIngester(store)
        ingester._ingest_chunk = MagicMock(return_value=["kid-1"])
        f = tmp_path / "doc.txt"
        f.write_text("content")
        with patch("myswat.memory.ingester.chunk_text", return_value=["chunk0"]):
            ingester.ingest_file(str(f), project_id="proj1", agent_id="agent-42")
        call_kwargs = ingester._ingest_chunk.call_args
        # agent_id should appear in either args or kwargs
        all_args = list(call_kwargs.args) + list(call_kwargs.kwargs.values())
        assert "agent-42" in all_args


# ---------------------------------------------------------------------------
# DocumentIngester._ingest_chunk tests
# ---------------------------------------------------------------------------

class TestIngestChunk:
    """Tests for DocumentIngester._ingest_chunk."""

    def _make_ingester(self, store=None, runner=None):
        store = store or MagicMock()
        return DocumentIngester(store, runner=runner)

    def test_no_runner_stores_raw_chunk(self):
        """Without a runner, _ingest_chunk should delegate to _store_raw_chunk."""
        store = MagicMock()
        ingester = self._make_ingester(store=store, runner=None)
        ingester._store_raw_chunk = MagicMock(return_value=["kid-raw"])
        result = ingester._ingest_chunk(
            chunk="some text",
            chunk_index=0,
            total_chunks=1,
            source_file="/path/to/file.txt",
            source_name="file.txt",
            project_id="proj1",
            agent_id=None,
        )
        ingester._store_raw_chunk.assert_called_once()
        assert result == ["kid-raw"]

    def test_successful_runner_parses_and_stores(self):
        """A successful runner invocation should parse output and store knowledge entries."""
        store = MagicMock()
        store.store_knowledge.return_value = "kid-parsed"
        runner = MagicMock()
        # Simulate runner returning a successful AgentResponse with parseable content
        runner_response = AgentResponse(
            content='[{"title": "Concept A", "content": "Details about A", "category": "design"}]',
            exit_code=0,
        )
        runner.invoke.return_value = runner_response
        ingester = self._make_ingester(store=store, runner=runner)
        result = ingester._ingest_chunk(
            chunk="some text about architecture",
            chunk_index=0,
            total_chunks=1,
            source_file="/path/to/file.txt",
            source_name="file.txt",
            project_id="proj1",
            agent_id="agent-1",
        )
        runner.invoke.assert_called_once()
        assert isinstance(result, list)
        # Should have stored at least one knowledge entry
        assert store.store_knowledge.call_count >= 1

    def test_runner_exception_propagates(self):
        """When the runner raises an exception, it propagates (no internal catch)."""
        store = MagicMock()
        runner = MagicMock()
        runner.invoke.side_effect = Exception("Runner crashed")
        ingester = self._make_ingester(store=store, runner=runner)
        with pytest.raises(Exception, match="Runner crashed"):
            ingester._ingest_chunk(
                chunk="some text",
                chunk_index=0,
                total_chunks=1,
                source_file="/path/to/file.txt",
                source_name="file.txt",
                project_id="proj1",
                agent_id=None,
            )

    def test_runner_unsuccessful_response_falls_back(self):
        """An unsuccessful AgentResponse should trigger raw chunk fallback."""
        store = MagicMock()
        runner = MagicMock()
        runner_response = AgentResponse(content="", exit_code=1)
        runner.invoke.return_value = runner_response
        ingester = self._make_ingester(store=store, runner=runner)
        ingester._store_raw_chunk = MagicMock(return_value=["kid-fallback"])
        result = ingester._ingest_chunk(
            chunk="some text",
            chunk_index=0,
            total_chunks=1,
            source_file="/path/to/file.txt",
            source_name="file.txt",
            project_id="proj1",
            agent_id=None,
        )
        ingester._store_raw_chunk.assert_called_once()
        assert result == ["kid-fallback"]

    def test_empty_parsed_items_falls_back_to_raw(self):
        """If the runner returns parseable but empty items, should fall back to raw chunk."""
        store = MagicMock()
        runner = MagicMock()
        runner_response = AgentResponse(content="[]", exit_code=0)
        runner.invoke.return_value = runner_response
        ingester = self._make_ingester(store=store, runner=runner)
        ingester._store_raw_chunk = MagicMock(return_value=["kid-empty-fallback"])
        result = ingester._ingest_chunk(
            chunk="some text",
            chunk_index=0,
            total_chunks=1,
            source_file="/path/to/file.txt",
            source_name="file.txt",
            project_id="proj1",
            agent_id=None,
        )
        ingester._store_raw_chunk.assert_called_once()
        assert result == ["kid-empty-fallback"]

    def test_items_with_empty_content_skipped(self):
        """Parsed items whose content is empty should be skipped."""
        store = MagicMock()
        store.store_knowledge.return_value = "kid-valid"
        runner = MagicMock()
        # Two items: one with content, one with empty content
        runner_response = AgentResponse(
            content=(
                '[{"title": "Good", "content": "Has content", "category": "design"},'
                ' {"title": "Bad", "content": "", "category": "design"}]'
            ),
            exit_code=0,
        )
        runner.invoke.return_value = runner_response
        ingester = self._make_ingester(store=store, runner=runner)
        result = ingester._ingest_chunk(
            chunk="some text",
            chunk_index=0,
            total_chunks=1,
            source_file="/path/to/file.txt",
            source_name="file.txt",
            project_id="proj1",
            agent_id="agent-1",
        )
        # Only the item with non-empty content should have been stored
        assert store.store_knowledge.call_count == 1

    def test_all_items_empty_content_returns_empty(self):
        """If all parsed items have empty content, returns empty list (no store calls)."""
        store = MagicMock()
        runner = MagicMock()
        runner_response = AgentResponse(
            content='[{"title": "Empty", "content": "", "category": "design"}]',
            exit_code=0,
        )
        runner.invoke.return_value = runner_response
        ingester = self._make_ingester(store=store, runner=runner)
        result = ingester._ingest_chunk(
            chunk="some text",
            chunk_index=0,
            total_chunks=1,
            source_file="/path/to/file.txt",
            source_name="file.txt",
            project_id="proj1",
            agent_id=None,
        )
        store.store_knowledge.assert_not_called()
        assert result == []


# ---------------------------------------------------------------------------
# DocumentIngester._store_raw_chunk tests
# ---------------------------------------------------------------------------

class TestStoreRawChunk:
    """Tests for DocumentIngester._store_raw_chunk."""

    def test_stores_with_correct_params(self):
        """_store_raw_chunk should call store.store_knowledge with category='architecture'."""
        store = MagicMock()
        store.store_knowledge.return_value = "kid-raw-1"
        ingester = DocumentIngester(store)
        result = ingester._store_raw_chunk(
            chunk="raw chunk content",
            chunk_index=2,
            source_file="/path/to/source.txt",
            source_name="source.txt",
            project_id="proj1",
            agent_id="agent-5",
        )
        store.store_knowledge.assert_called_once()
        call_kwargs = store.store_knowledge.call_args
        # Flatten args and kwargs to check key values
        all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        all_args = call_kwargs.args if call_kwargs.args else ()
        # Category should be "architecture"
        assert "architecture" in list(all_kwargs.values()) + list(all_args), (
            "Expected category='architecture' in store.store_knowledge call"
        )
        # project_id should be passed
        assert "proj1" in list(all_kwargs.values()) + list(all_args), (
            "Expected project_id='proj1' in store.store_knowledge call"
        )
        # The chunk content should be passed
        assert "raw chunk content" in list(all_kwargs.values()) + list(all_args), (
            "Expected chunk content in store.store_knowledge call"
        )
        # Should return a list with the knowledge ID
        assert result == ["kid-raw-1"]

    def test_returns_list_of_single_kid(self):
        """_store_raw_chunk should return a list containing exactly one knowledge ID."""
        store = MagicMock()
        store.store_knowledge.return_value = "kid-abc"
        ingester = DocumentIngester(store)
        result = ingester._store_raw_chunk(
            chunk="text",
            chunk_index=0,
            source_file="/file.txt",
            source_name="file.txt",
            project_id="p1",
            agent_id=None,
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == "kid-abc"

    def test_passes_agent_id_to_store(self):
        """agent_id should be forwarded to the store."""
        store = MagicMock()
        store.store_knowledge.return_value = "kid-1"
        ingester = DocumentIngester(store)
        ingester._store_raw_chunk(
            chunk="text",
            chunk_index=0,
            source_file="/file.txt",
            source_name="file.txt",
            project_id="p1",
            agent_id="agent-99",
        )
        call_kwargs = store.store_knowledge.call_args
        all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        all_args = call_kwargs.args if call_kwargs.args else ()
        assert "agent-99" in list(all_kwargs.values()) + list(all_args), (
            "Expected agent_id='agent-99' in store.store_knowledge call"
        )

    def test_source_file_and_name_passed(self):
        """source_file and source_name should be forwarded to the store."""
        store = MagicMock()
        store.store_knowledge.return_value = "kid-1"
        ingester = DocumentIngester(store)
        ingester._store_raw_chunk(
            chunk="text",
            chunk_index=0,
            source_file="/some/path/readme.md",
            source_name="readme.md",
            project_id="p1",
            agent_id=None,
        )
        call_kwargs = store.store_knowledge.call_args
        all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        all_args = call_kwargs.args if call_kwargs.args else ()
        combined = list(all_kwargs.values()) + list(all_args)
        assert "/some/path/readme.md" in combined or any(
            "/some/path/readme.md" in str(v) for v in combined
        ), "Expected source_file in store.store_knowledge call"
        assert "readme.md" in combined or any(
            "readme.md" in str(v) for v in combined
        ), "Expected source_name in store.store_knowledge call"
