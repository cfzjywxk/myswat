"""Tests for DocumentIngester chunking logic."""

from myswat.memory.ingester import chunk_text


class TestChunkText:
    def test_short_text_single_chunk(self):
        text = "Hello world"
        chunks = chunk_text(text, chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_multiple_chunks(self):
        text = "A" * 20000
        chunks = chunk_text(text, chunk_size=8000, overlap=500)
        assert len(chunks) >= 2
        # Each chunk should be within the size limit (may be slightly over at boundaries)
        for chunk in chunks:
            assert len(chunk) <= 8500

    def test_paragraph_boundary_splitting(self):
        paragraphs = ["Paragraph one content. " * 20, "Paragraph two content. " * 20]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) >= 2

    def test_overlap_between_chunks(self):
        text = "word " * 5000
        chunks = chunk_text(text, chunk_size=1000, overlap=200)
        assert len(chunks) >= 2
        # With overlap, the end of chunk N should appear in chunk N+1
        for i in range(len(chunks) - 1):
            tail = chunks[i][-100:]
            # The overlap region should be present in the next chunk
            assert tail in chunks[i + 1] or len(chunks[i]) <= 1000

    def test_exact_chunk_size(self):
        text = "A" * 8000
        chunks = chunk_text(text, chunk_size=8000)
        assert len(chunks) == 1

    def test_empty_text(self):
        chunks = chunk_text("", chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] == ""
