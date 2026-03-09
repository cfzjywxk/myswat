"""v006: Remove fixed dimension from embedding column.

Allows both local BGE-M3 (1024-dim) and TiDB built-in EMBEDDING()
(variable dimension) to store vectors in the same column.
"""

VERSION = 6
DESCRIPTION = "Remove fixed vector dimension constraint from knowledge.embedding"

STATEMENTS = [
    """
    ALTER TABLE knowledge MODIFY COLUMN embedding VECTOR
    """,
]
