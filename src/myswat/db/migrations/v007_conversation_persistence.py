"""v007: Conversation persistence schema changes."""

VERSION = 7
DESCRIPTION = "Add compacted_at timestamp and turn recency index for conversation persistence"

STATEMENTS = [
    "ALTER TABLE sessions ADD COLUMN compacted_at TIMESTAMP NULL DEFAULT NULL",
    "CREATE INDEX idx_session_turns_recency ON session_turns (created_at DESC, id DESC)",
]
