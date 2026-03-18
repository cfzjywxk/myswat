"""v016: Add phase_result to artifacts.artifact_type ENUM."""

VERSION = 16
DESCRIPTION = "Add phase_result artifact type for workflow resume support"

STATEMENTS = [
    "ALTER TABLE artifacts MODIFY COLUMN artifact_type "
    "ENUM('proposal', 'diff', 'patch', 'test_plan', 'design_doc', 'phase_result') "
    "NOT NULL",
]
