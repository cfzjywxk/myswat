"""Comprehensive tests for myswat.memory.embedder module.

Since FlagEmbedding may not be installed in the test environment, all tests
mock is_available, _available, and _get_model / _model as needed.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from myswat.memory import embedder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_module_globals():
    """Reset cached module-level state before every test."""
    original_model = embedder._model
    original_available = embedder._available
    yield
    embedder._model = original_model
    embedder._available = original_available


@pytest.fixture()
def mock_model():
    """Return a MagicMock that behaves like a BGEM3FlagModel instance.

    model.encode(texts) returns {"dense_vecs": <numpy-like list of lists>}
    """
    model = MagicMock()

    def _encode(texts, **kwargs):
        # Return a deterministic fake embedding per input text.
        vecs = [[float(i) + 0.1 * j for j in range(3)] for i in range(len(texts))]
        return {"dense_vecs": [FakeNdarray(v) for v in vecs]}

    model.encode.side_effect = _encode
    return model


class FakeNdarray:
    """Minimal stand-in for a numpy ndarray that supports .tolist()."""

    def __init__(self, data: list[float]):
        self._data = data

    def tolist(self) -> list[float]:
        return list(self._data)


# ---------------------------------------------------------------------------
# 1. is_available
# ---------------------------------------------------------------------------

class TestIsAvailable:
    """Tests for embedder.is_available()."""

    def test_returns_true_when_flag_embedding_importable(self):
        """is_available() should return True when FlagEmbedding can be imported."""
        embedder._available = None  # force re-check
        with patch.dict("sys.modules", {"FlagEmbedding": MagicMock()}):
            result = embedder.is_available()
        assert result is True

    def test_returns_false_when_flag_embedding_not_importable(self):
        """is_available() should return False when FlagEmbedding import fails."""
        embedder._available = None  # force re-check
        import builtins
        _real_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "FlagEmbedding":
                raise ImportError("no FlagEmbedding")
            return _real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_mock_import):
            result = embedder.is_available()
        assert result is False

    def test_caches_result_on_subsequent_calls(self):
        """Once determined, is_available() should return the cached value."""
        embedder._available = True
        assert embedder.is_available() is True

        embedder._available = False
        assert embedder.is_available() is False

    def test_does_not_recheck_when_already_cached_true(self):
        """If _available is already True, no import attempt should occur."""
        embedder._available = True
        # If it tried to import, the mock would be called; it should not be.
        with patch("builtins.__import__", side_effect=AssertionError("should not import")) as mock_imp:
            result = embedder.is_available()
        assert result is True
        mock_imp.assert_not_called()

    def test_does_not_recheck_when_already_cached_false(self):
        """If _available is already False, no import attempt should occur."""
        embedder._available = False
        with patch("builtins.__import__", side_effect=AssertionError("should not import")) as mock_imp:
            result = embedder.is_available()
        assert result is False
        mock_imp.assert_not_called()


# ---------------------------------------------------------------------------
# 2. embed
# ---------------------------------------------------------------------------

class TestEmbed:
    """Tests for embedder.embed()."""

    def test_returns_none_when_not_available(self):
        """embed() should return None when the model is not available."""
        with patch.object(embedder, "is_available", return_value=False):
            result = embedder.embed("hello world")
        assert result is None

    def test_returns_list_of_floats_when_available(self, mock_model):
        """embed() should return a list of floats from model.encode()."""
        with (
            patch.object(embedder, "is_available", return_value=True),
            patch.object(embedder, "_get_model", return_value=mock_model),
            patch.object(embedder, "_model", mock_model),
        ):
            result = embedder.embed("test sentence")

        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_calls_model_encode_with_single_element_list(self, mock_model):
        """embed() should pass [text] to model.encode()."""
        with (
            patch.object(embedder, "is_available", return_value=True),
            patch.object(embedder, "_get_model", return_value=mock_model),
            patch.object(embedder, "_model", mock_model),
        ):
            embedder.embed("hello")

        mock_model.encode.assert_called_once()
        args = mock_model.encode.call_args
        assert args[0][0] == ["hello"]

    def test_returns_first_vector_from_dense_vecs(self, mock_model):
        """embed() should return dense_vecs[0].tolist()."""
        expected = [0.0, 0.1, 0.2]
        with (
            patch.object(embedder, "is_available", return_value=True),
            patch.object(embedder, "_get_model", return_value=mock_model),
            patch.object(embedder, "_model", mock_model),
        ):
            result = embedder.embed("anything")

        assert result == expected

    def test_embed_with_empty_string(self, mock_model):
        """embed() should handle an empty string without error."""
        with (
            patch.object(embedder, "is_available", return_value=True),
            patch.object(embedder, "_get_model", return_value=mock_model),
            patch.object(embedder, "_model", mock_model),
        ):
            result = embedder.embed("")

        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 3. embed_batch
# ---------------------------------------------------------------------------

class TestEmbedBatch:
    """Tests for embedder.embed_batch()."""

    def test_empty_list_returns_empty_list(self):
        """embed_batch([]) should return [] regardless of availability."""
        result = embedder.embed_batch([])
        assert result == []

    def test_not_available_returns_list_of_nones(self):
        """When not available, embed_batch should return [None] * len(texts)."""
        texts = ["a", "b", "c"]
        with patch.object(embedder, "is_available", return_value=False):
            result = embedder.embed_batch(texts)

        assert result == [None, None, None]

    def test_not_available_single_text_returns_one_none(self):
        """Single-element list when not available returns [None]."""
        with patch.object(embedder, "is_available", return_value=False):
            result = embedder.embed_batch(["hello"])

        assert result == [None]

    def test_available_returns_list_of_vectors(self, mock_model):
        """When available, embed_batch should return a list of float lists."""
        texts = ["alpha", "beta"]
        with (
            patch.object(embedder, "is_available", return_value=True),
            patch.object(embedder, "_get_model", return_value=mock_model),
            patch.object(embedder, "_model", mock_model),
        ):
            result = embedder.embed_batch(texts)

        assert len(result) == 2
        for vec in result:
            assert isinstance(vec, list)
            assert all(isinstance(v, float) for v in vec)

    def test_available_encodes_all_texts_at_once(self, mock_model):
        """embed_batch should pass all texts to model.encode() in one call."""
        texts = ["one", "two", "three"]
        with (
            patch.object(embedder, "is_available", return_value=True),
            patch.object(embedder, "_get_model", return_value=mock_model),
            patch.object(embedder, "_model", mock_model),
        ):
            embedder.embed_batch(texts)

        mock_model.encode.assert_called_once()
        args = mock_model.encode.call_args
        assert args[0][0] == texts

    def test_available_returns_correct_number_of_vectors(self, mock_model):
        """Number of returned vectors should match number of input texts."""
        texts = ["a", "b", "c", "d", "e"]
        with (
            patch.object(embedder, "is_available", return_value=True),
            patch.object(embedder, "_get_model", return_value=mock_model),
            patch.object(embedder, "_model", mock_model),
        ):
            result = embedder.embed_batch(texts)

        assert len(result) == len(texts)


# ---------------------------------------------------------------------------
# 4. embedding_to_sql
# ---------------------------------------------------------------------------

class TestEmbeddingToSql:
    """Tests for embedder.embedding_to_sql()."""

    def test_returns_json_string(self):
        """embedding_to_sql should return a valid JSON string."""
        vec = [1.0, 2.5, 3.7]
        result = embedder.embedding_to_sql(vec)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == vec

    def test_roundtrips_correctly(self):
        """json.loads(embedding_to_sql(v)) should equal the original vector."""
        vec = [0.0, -1.23, 4.56, 7.89]
        result = embedder.embedding_to_sql(vec)
        assert json.loads(result) == vec

    def test_empty_vector(self):
        """embedding_to_sql([]) should return '[]'."""
        result = embedder.embedding_to_sql([])
        assert result == "[]"

    def test_single_element(self):
        """embedding_to_sql with a single-element list."""
        vec = [3.14]
        result = embedder.embedding_to_sql(vec)
        assert json.loads(result) == vec

    def test_negative_values(self):
        """embedding_to_sql should handle negative floats."""
        vec = [-1.0, -0.5, 0.0, 0.5, 1.0]
        result = embedder.embedding_to_sql(vec)
        assert json.loads(result) == vec

    def test_output_is_json_dumps(self):
        """Result should be identical to json.dumps(vec)."""
        vec = [1.1, 2.2, 3.3]
        assert embedder.embedding_to_sql(vec) == json.dumps(vec)


# ---------------------------------------------------------------------------
# 5. preload_model
# ---------------------------------------------------------------------------

class TestPreloadModel:
    """Tests for embedder.preload_model()."""

    def test_calls_get_model_when_available(self):
        """preload_model should call _get_model() when is_available() is True."""
        with (
            patch.object(embedder, "is_available", return_value=True),
            patch.object(embedder, "_get_model") as mock_get,
        ):
            embedder.preload_model()

        mock_get.assert_called_once()

    def test_does_not_call_get_model_when_not_available(self):
        """preload_model should NOT call _get_model() when is_available() is False."""
        with (
            patch.object(embedder, "is_available", return_value=False),
            patch.object(embedder, "_get_model") as mock_get,
        ):
            embedder.preload_model()

        mock_get.assert_not_called()

    def test_no_exception_when_not_available(self):
        """preload_model should silently do nothing when not available."""
        with patch.object(embedder, "is_available", return_value=False):
            embedder.preload_model()  # should not raise


# ---------------------------------------------------------------------------
# 6. _get_model (basic behavior through preload_model)
# ---------------------------------------------------------------------------

class TestGetModel:
    """Indirect tests for _get_model via preload_model and embed."""

    def test_preload_triggers_model_loading(self):
        """Calling preload_model when available should trigger _get_model."""
        sentinel_model = MagicMock()
        with (
            patch.object(embedder, "is_available", return_value=True),
            patch.object(embedder, "_get_model", return_value=sentinel_model) as mock_get,
        ):
            embedder.preload_model()
            mock_get.assert_called_once()
