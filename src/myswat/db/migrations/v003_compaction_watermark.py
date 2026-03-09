"""v003: Add compaction watermark to sessions for tracking which turns have been distilled."""

VERSION = 3
DESCRIPTION = "Add compacted_through_turn_index to sessions; tracks which turns are covered by knowledge"

STATEMENTS = [
    """
    ALTER TABLE sessions ADD COLUMN compacted_through_turn_index INT DEFAULT -1
    """,
]
