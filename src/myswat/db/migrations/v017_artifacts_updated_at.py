"""v017: Add updated_at to artifacts for correct latest-artifact ordering."""

VERSION = 17
DESCRIPTION = "Add updated_at column to artifacts so upserts are ordered correctly"

STATEMENTS = [
    "ALTER TABLE artifacts ADD COLUMN updated_at TIMESTAMP NOT NULL "
    "DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP",
]
