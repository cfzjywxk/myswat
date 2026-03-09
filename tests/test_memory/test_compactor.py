"""Tests for KnowledgeCompactor and parse_compaction_output."""

from myswat.memory.compactor import parse_compaction_output


class TestParseCompactionOutput:
    def test_clean_json_array(self):
        raw = '[{"category": "decision", "title": "Use TiDB", "content": "We chose TiDB.", "tags": ["db"], "relevance_score": 0.9, "confidence": 0.8}]'
        items = parse_compaction_output(raw)
        assert len(items) == 1
        assert items[0]["category"] == "decision"
        assert items[0]["title"] == "Use TiDB"

    def test_json_in_markdown_code_block(self):
        raw = """Here are the knowledge items:

```json
[{"category": "pattern", "title": "Retry logic", "content": "Always retry on timeout.", "tags": [], "relevance_score": 0.7, "confidence": 0.9}]
```
"""
        items = parse_compaction_output(raw)
        assert len(items) == 1
        assert items[0]["category"] == "pattern"

    def test_json_in_plain_code_block(self):
        raw = """```
[{"category": "bug_fix", "title": "Fix null pointer", "content": "Check for None.", "tags": ["bug"], "relevance_score": 0.8, "confidence": 0.7}]
```"""
        items = parse_compaction_output(raw)
        assert len(items) == 1

    def test_empty_array(self):
        assert parse_compaction_output("[]") == []

    def test_no_json(self):
        assert parse_compaction_output("No useful knowledge found.") == []

    def test_json_with_surrounding_text(self):
        raw = 'Here are the results:\n[{"category": "architecture", "title": "Test", "content": "Content."}]\nEnd.'
        items = parse_compaction_output(raw)
        assert len(items) == 1

    def test_invalid_json(self):
        assert parse_compaction_output("[{broken json}]") == []

    def test_multiple_items(self):
        raw = """[
            {"category": "decision", "title": "A", "content": "First."},
            {"category": "pattern", "title": "B", "content": "Second."}
        ]"""
        items = parse_compaction_output(raw)
        assert len(items) == 2
