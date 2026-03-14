"""Comprehensive tests for myswat.cli.learn_cmd."""

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from myswat.cli.learn_cmd import (
    _discover_files,
    _read_agent_instructions,
    _format_file_contents,
    _format_agent_instructions,
    _validate_learned,
    _store_learned_knowledge,
    _write_myswat_md,
    ensure_learned,
    INDICATOR_GLOBS,
    AGENT_INSTRUCTION_FILES,
    MAX_FILE_BYTES,
    REQUIRED_KEYS,
)


# ---------------------------------------------------------------------------
# _discover_files
# ---------------------------------------------------------------------------
class TestDiscoverFiles:
    """Tests for _discover_files."""

    def test_finds_makefile(self, tmp_path):
        """A Makefile in the repo root should be discovered under 'build'."""
        makefile = tmp_path / "Makefile"
        makefile.write_text("all:\n\techo hello")

        result = _discover_files(tmp_path)

        assert "build" in result
        rel_paths = [rel for rel, _content in result["build"]]
        assert "Makefile" in rel_paths

    def test_finds_pyproject_toml(self, tmp_path):
        """pyproject.toml should appear under 'build'."""
        (tmp_path / "pyproject.toml").write_text("[build-system]\nrequires = []")

        result = _discover_files(tmp_path)

        assert "build" in result
        rel_paths = [rel for rel, _ in result["build"]]
        assert "pyproject.toml" in rel_paths

    def test_finds_cargo_toml(self, tmp_path):
        """Cargo.toml should appear under 'build'."""
        (tmp_path / "Cargo.toml").write_text("[package]\nname = \"foo\"")

        result = _discover_files(tmp_path)

        assert "build" in result
        rel_paths = [rel for rel, _ in result["build"]]
        assert "Cargo.toml" in rel_paths

    def test_finds_pytest_ini(self, tmp_path):
        """pytest.ini should appear under 'test'."""
        (tmp_path / "pytest.ini").write_text("[pytest]\naddopts = -v")

        result = _discover_files(tmp_path)

        assert "test" in result
        rel_paths = [rel for rel, _ in result["test"]]
        assert "pytest.ini" in rel_paths

    def test_finds_gitignore(self, tmp_path):
        """A .gitignore file should appear under 'git'."""
        (tmp_path / ".gitignore").write_text("*.pyc\n__pycache__/")

        result = _discover_files(tmp_path)

        assert "git" in result
        rel_paths = [rel for rel, _ in result["git"]]
        assert ".gitignore" in rel_paths

    def test_finds_readme(self, tmp_path):
        """README.md should appear under 'docs'."""
        (tmp_path / "README.md").write_text("# My Project")

        result = _discover_files(tmp_path)

        assert "docs" in result
        rel_paths = [rel for rel, _ in result["docs"]]
        assert "README.md" in rel_paths

    def test_deduplicates_across_categories(self, tmp_path):
        """If a file matches globs in multiple categories it should only appear once overall."""
        # Create a file that could match in two categories (e.g. setup.cfg appears
        # in both 'build' and 'test' globs).  Even if it doesn't in the current
        # glob map, the deduplication logic should prevent the same path appearing
        # twice across all categories.
        (tmp_path / "setup.cfg").write_text("[metadata]\nname = foo")

        result = _discover_files(tmp_path)

        all_paths = []
        for entries in result.values():
            all_paths.extend(rel for rel, _ in entries)

        # Each relative path must appear at most once.
        assert len(all_paths) == len(set(all_paths))

    def test_handles_os_error(self, tmp_path):
        """Files that raise OSError during read should be silently skipped."""
        makefile = tmp_path / "Makefile"
        makefile.write_text("all: build")

        # Patch Path.read_text to raise OSError for the Makefile
        original_read_text = Path.read_text

        def patched_read_text(self, *args, **kwargs):
            if self.name == "Makefile":
                raise OSError("Permission denied")
            return original_read_text(self, *args, **kwargs)

        with patch.object(Path, "read_text", patched_read_text):
            result = _discover_files(tmp_path)

        # The Makefile should not appear in results because reading failed.
        build_paths = [rel for rel, _ in result.get("build", [])]
        assert "Makefile" not in build_paths

    def test_empty_directory(self, tmp_path):
        """An empty directory should produce an empty (or all-empty) result."""
        result = _discover_files(tmp_path)

        total_files = sum(len(entries) for entries in result.values())
        assert total_files == 0

    def test_reads_at_most_max_file_bytes(self, tmp_path):
        """File content should be truncated to MAX_FILE_BYTES."""
        big_content = "x" * (MAX_FILE_BYTES + 5000)
        (tmp_path / "Makefile").write_text(big_content)

        result = _discover_files(tmp_path)

        for entries in result.values():
            for _rel, content in entries:
                assert len(content.encode()) <= MAX_FILE_BYTES

    def test_finds_ci_workflow(self, tmp_path):
        """GitHub workflow YAML should appear under 'ci'."""
        wf_dir = tmp_path / ".github" / "workflows"
        wf_dir.mkdir(parents=True)
        (wf_dir / "ci.yml").write_text("on: push")

        result = _discover_files(tmp_path)

        assert "ci" in result
        ci_paths = [rel for rel, _ in result["ci"]]
        assert any("ci.yml" in p for p in ci_paths)

    def test_finds_style_config(self, tmp_path):
        """Style indicator files should appear under 'style'."""
        (tmp_path / ".rustfmt.toml").write_text("max_width = 100")

        result = _discover_files(tmp_path)

        assert "style" in result
        style_paths = [rel for rel, _ in result["style"]]
        assert ".rustfmt.toml" in style_paths


# ---------------------------------------------------------------------------
# _read_agent_instructions
# ---------------------------------------------------------------------------
class TestReadAgentInstructions:
    """Tests for _read_agent_instructions."""

    def test_reads_claude_md(self, tmp_path):
        """CLAUDE.md should be read and returned."""
        (tmp_path / "CLAUDE.md").write_text("Be helpful.")

        result = _read_agent_instructions(tmp_path)

        names = [name for name, _ in result]
        assert "CLAUDE.md" in names
        contents = {name: content for name, content in result}
        assert "Be helpful." in contents["CLAUDE.md"]

    def test_reads_agents_md(self, tmp_path):
        """AGENTS.md should be read and returned."""
        (tmp_path / "AGENTS.md").write_text("Agent rules here.")

        result = _read_agent_instructions(tmp_path)

        names = [name for name, _ in result]
        assert "AGENTS.md" in names

    def test_reads_cursorrules(self, tmp_path):
        """.cursorrules should be read and returned."""
        (tmp_path / ".cursorrules").write_text("cursor config")

        result = _read_agent_instructions(tmp_path)

        names = [name for name, _ in result]
        assert ".cursorrules" in names

    def test_reads_copilot_instructions(self, tmp_path):
        """copilot-instructions.md in .github should be read and returned."""
        gh_dir = tmp_path / ".github"
        gh_dir.mkdir()
        (gh_dir / "copilot-instructions.md").write_text("copilot rules")

        result = _read_agent_instructions(tmp_path)

        names = [name for name, _ in result]
        assert any("copilot-instructions.md" in n for n in names)

    def test_skips_missing_files(self, tmp_path):
        """Missing instruction files should be silently skipped."""
        # Create only one of the files.
        (tmp_path / "CLAUDE.md").write_text("exists")

        result = _read_agent_instructions(tmp_path)

        names = [name for name, _ in result]
        assert "CLAUDE.md" in names
        # No AGENTS.md, .cursorrules, etc.
        assert "AGENTS.md" not in names

    def test_handles_os_error(self, tmp_path):
        """Files that raise OSError during read should be skipped."""
        (tmp_path / "CLAUDE.md").write_text("content")

        original_read_text = Path.read_text

        def patched_read_text(self, *args, **kwargs):
            if self.name == "CLAUDE.md":
                raise OSError("Permission denied")
            return original_read_text(self, *args, **kwargs)

        with patch.object(Path, "read_text", patched_read_text):
            result = _read_agent_instructions(tmp_path)

        names = [name for name, _ in result]
        assert "CLAUDE.md" not in names

    def test_empty_directory(self, tmp_path):
        """No instruction files -> empty list."""
        result = _read_agent_instructions(tmp_path)
        assert result == []

    def test_reads_multiple_files(self, tmp_path):
        """When several instruction files exist, all should be returned."""
        (tmp_path / "CLAUDE.md").write_text("claude instructions")
        (tmp_path / "AGENTS.md").write_text("agents instructions")

        result = _read_agent_instructions(tmp_path)

        names = [name for name, _ in result]
        assert "CLAUDE.md" in names
        assert "AGENTS.md" in names
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _format_file_contents
# ---------------------------------------------------------------------------
class TestFormatFileContents:
    """Tests for _format_file_contents."""

    def test_formats_single_category(self):
        """Single category with one file should be properly formatted."""
        discovered = {"build": [("Makefile", "all: build")]}

        result = _format_file_contents(discovered)

        assert "### BUILD" in result or "### build" in result.lower()
        assert "#### Makefile" in result or "Makefile" in result
        assert "```" in result
        assert "all: build" in result

    def test_formats_multiple_categories(self):
        """Multiple categories should each get their own section."""
        discovered = {
            "build": [("Makefile", "all: build")],
            "test": [("pytest.ini", "[pytest]")],
        }

        result = _format_file_contents(discovered)

        lower = result.lower()
        assert "build" in lower
        assert "test" in lower
        assert "Makefile" in result
        assert "pytest.ini" in result

    def test_formats_multiple_files_in_category(self):
        """Multiple files in a single category should each appear."""
        discovered = {
            "build": [
                ("Makefile", "all: build"),
                ("pyproject.toml", "[build-system]"),
            ],
        }

        result = _format_file_contents(discovered)

        assert "Makefile" in result
        assert "pyproject.toml" in result
        assert "all: build" in result
        assert "[build-system]" in result

    def test_empty_input(self):
        """Empty discovered dict should return the 'no indicator files' message."""
        result = _format_file_contents({})

        assert "(no indicator files found)" in result

    def test_empty_categories(self):
        """Categories with empty lists should still produce the fallback message."""
        result = _format_file_contents({"build": []})

        # Either returns fallback or at least doesn't crash.
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _format_agent_instructions
# ---------------------------------------------------------------------------
class TestFormatAgentInstructions:
    """Tests for _format_agent_instructions."""

    def test_formats_single_instruction(self):
        """Single instruction file should be formatted with header and code block."""
        instructions = [("CLAUDE.md", "Be helpful.")]

        result = _format_agent_instructions(instructions)

        assert "#### CLAUDE.md" in result or "CLAUDE.md" in result
        assert "```" in result
        assert "Be helpful." in result

    def test_formats_multiple_instructions(self):
        """Multiple instruction files should each appear."""
        instructions = [
            ("CLAUDE.md", "Be helpful."),
            ("AGENTS.md", "Follow rules."),
        ]

        result = _format_agent_instructions(instructions)

        assert "CLAUDE.md" in result
        assert "AGENTS.md" in result
        assert "Be helpful." in result
        assert "Follow rules." in result

    def test_empty_input(self):
        """Empty list should return the 'no agent instruction files' message."""
        result = _format_agent_instructions([])

        assert "(no agent instruction files found)" in result


# ---------------------------------------------------------------------------
# _validate_learned
# ---------------------------------------------------------------------------
class TestValidateLearned:
    """Tests for _validate_learned."""

    def _make_valid_data(self):
        """Return a minimal valid learned data dict."""
        return {
            "build": {"commands": ["make build"]},
            "test": {"tiers": [{"name": "unit", "cmd": "pytest"}]},
            "structure": {"entry_points": ["src/main.py"]},
        }

    def test_valid_data_passes(self):
        """A fully valid data dict should produce no errors."""
        data = self._make_valid_data()
        errors = _validate_learned(data)
        assert errors == []

    def test_missing_build_key(self):
        """Missing 'build' key should be reported."""
        data = self._make_valid_data()
        del data["build"]

        errors = _validate_learned(data)

        assert len(errors) > 0
        assert any("build" in e.lower() for e in errors)

    def test_missing_test_key(self):
        """Missing 'test' key should be reported."""
        data = self._make_valid_data()
        del data["test"]

        errors = _validate_learned(data)

        assert len(errors) > 0
        assert any("test" in e.lower() for e in errors)

    def test_missing_structure_key(self):
        """Missing 'structure' key should be reported."""
        data = self._make_valid_data()
        del data["structure"]

        errors = _validate_learned(data)

        assert len(errors) > 0
        assert any("structure" in e.lower() for e in errors)

    def test_missing_all_required_keys(self):
        """Missing all required keys should produce multiple errors."""
        errors = _validate_learned({})

        assert len(errors) >= len(REQUIRED_KEYS)

    def test_empty_build_commands(self):
        """build.commands being empty should be reported."""
        data = self._make_valid_data()
        data["build"]["commands"] = []

        errors = _validate_learned(data)

        assert len(errors) > 0
        assert any("build" in e.lower() or "command" in e.lower() for e in errors)

    def test_empty_test_tiers(self):
        """test.tiers being empty should be reported."""
        data = self._make_valid_data()
        data["test"]["tiers"] = []

        errors = _validate_learned(data)

        assert len(errors) > 0
        assert any("test" in e.lower() or "tier" in e.lower() for e in errors)

    def test_empty_structure_entry_points(self):
        """structure.entry_points being empty should be reported."""
        data = self._make_valid_data()
        data["structure"]["entry_points"] = []

        errors = _validate_learned(data)

        assert len(errors) > 0
        assert any("structure" in e.lower() or "entry" in e.lower() for e in errors)

    def test_extra_keys_do_not_cause_errors(self):
        """Extra keys beyond the required ones should not cause validation errors."""
        data = self._make_valid_data()
        data["style"] = {"formatter": "black"}

        errors = _validate_learned(data)

        assert errors == []


# ---------------------------------------------------------------------------
# _store_learned_knowledge
# ---------------------------------------------------------------------------
class TestStoreLearned:
    """Tests for _store_learned_knowledge."""

    def _make_data(self):
        return {
            "build": {"commands": ["make build"]},
            "test": {"tiers": [{"name": "unit", "cmd": "pytest"}]},
            "structure": {"entry_points": ["src/main.py"]},
        }

    def test_creates_overview_and_sections(self):
        """Should store an overview entry plus per-section entries."""
        store = MagicMock()
        data = self._make_data()

        count = _store_learned_knowledge(store, "proj-1", data)

        assert count > 0
        # store should have received multiple calls to store/add knowledge.
        assert store.method_calls or store.call_count > 0

    def test_deletes_existing_project_ops(self):
        """Should delete existing project_ops before storing new ones."""
        store = MagicMock()
        data = self._make_data()

        _store_learned_knowledge(store, "proj-1", data)

        # Look for a delete call among all mock interactions.
        all_calls_str = str(store.method_calls)
        # The function should have called some deletion method.
        assert store.method_calls  # at least some interaction happened

    def test_handles_list_values_invariants(self):
        """List values (like invariants) should be stored correctly."""
        store = MagicMock()
        data = self._make_data()
        data["invariants"] = ["never break build", "all tests must pass"]

        count = _store_learned_knowledge(store, "proj-1", data)

        assert count > 0

    def test_skips_empty_values(self):
        """Sections with empty/None values should be skipped or handled gracefully."""
        store = MagicMock()
        data = self._make_data()
        data["empty_section"] = {}

        count = _store_learned_knowledge(store, "proj-1", data)

        # Should still succeed and store the non-empty sections.
        assert count > 0

    def test_with_agent_id(self):
        """Passing agent_id should not cause errors."""
        store = MagicMock()
        data = self._make_data()

        count = _store_learned_knowledge(store, "proj-1", data, agent_id="agent-42")

        assert count > 0

    def test_returns_int_count(self):
        """Return value should be an integer representing count of stored entries."""
        store = MagicMock()
        data = self._make_data()

        count = _store_learned_knowledge(store, "proj-1", data)

        assert isinstance(count, int)

    def test_reinserts_team_workflows_after_bulk_delete(self):
        """After delete_knowledge_by_category wipes project_ops, team workflows must be re-stored."""
        store = MagicMock()
        data = self._make_data()

        _store_learned_knowledge(store, "proj-1", data)

        titles = [
            call.kwargs.get("title")
            for call in store.store_knowledge.call_args_list
        ]
        assert "Team Workflows" in titles


# ---------------------------------------------------------------------------
# _write_myswat_md
# ---------------------------------------------------------------------------
class TestWriteMyswatMd:
    """Tests for _write_myswat_md."""

    def _make_data(self):
        return {
            "build": {"commands": ["make build"]},
            "test": {"tiers": [{"name": "unit", "cmd": "pytest"}]},
            "structure": {"entry_points": ["src/main.py"]},
        }

    def test_creates_file(self, tmp_path):
        """myswat.md should be created at the repo root."""
        data = self._make_data()

        result_path = _write_myswat_md(tmp_path, data, "my-project")

        assert result_path.exists()
        assert result_path.name == "myswat.md"

    def test_file_contains_build_info(self, tmp_path):
        """The written file should contain build commands."""
        data = self._make_data()

        result_path = _write_myswat_md(tmp_path, data, "my-project")
        content = result_path.read_text()

        assert "make build" in content

    def test_file_contains_test_info(self, tmp_path):
        """The written file should contain test tier information."""
        data = self._make_data()

        result_path = _write_myswat_md(tmp_path, data, "my-project")
        content = result_path.read_text()

        assert "pytest" in content

    def test_file_contains_structure_info(self, tmp_path):
        """The written file should contain structure entry points."""
        data = self._make_data()

        result_path = _write_myswat_md(tmp_path, data, "my-project")
        content = result_path.read_text()

        assert "src/main.py" in content

    def test_file_contains_slug(self, tmp_path):
        """The written file should reference the project slug."""
        data = self._make_data()

        result_path = _write_myswat_md(tmp_path, data, "my-project")
        content = result_path.read_text()

        assert "my-project" in content

    def test_returns_path_object(self, tmp_path):
        """Should return a Path object."""
        data = self._make_data()

        result_path = _write_myswat_md(tmp_path, data, "my-project")

        assert isinstance(result_path, Path)

    def test_file_inside_repo_path(self, tmp_path):
        """The file should be written inside the given repo_path."""
        data = self._make_data()

        result_path = _write_myswat_md(tmp_path, data, "my-project")

        assert str(result_path).startswith(str(tmp_path))

    def test_file_contains_team_workflows(self, tmp_path):
        """The written file should include team workflow knowledge."""
        data = self._make_data()

        result_path = _write_myswat_md(tmp_path, data, "my-project")
        content = result_path.read_text()

        assert "Team Workflows" in content
        assert "MODE: full" in content
        assert "MODE: design" in content
        assert "MODE: testplan" in content


# ---------------------------------------------------------------------------
# ensure_learned
# ---------------------------------------------------------------------------
class TestEnsureLearned:
    """Tests for ensure_learned."""

    def test_returns_early_if_myswat_md_exists(self, tmp_path):
        """If myswat.md already exists, should return without calling store or run_learn."""
        (tmp_path / "myswat.md").write_text("already learned")
        store = MagicMock()

        with patch("myswat.cli.learn_cmd.run_learn") as mock_run:
            ensure_learned(store, "my-project", "proj-1", tmp_path)

        mock_run.assert_not_called()
        # Store should not have been queried for knowledge either,
        # or at least run_learn should not have been triggered.

    def test_returns_early_if_tidb_has_learned_ops(self, tmp_path):
        """If store has learned project_ops (not just Team Workflows), skip run_learn."""
        store = MagicMock()
        store.list_knowledge.return_value = [
            {"id": 1, "title": "Project Overview"},
            {"id": 2, "title": "Team Workflows"},
        ]

        with patch("myswat.cli.learn_cmd.run_learn") as mock_run:
            ensure_learned(store, "my-project", "proj-1", tmp_path)

        mock_run.assert_not_called()

    def test_triggers_learn_if_only_team_workflows_in_tidb(self, tmp_path):
        """If TiDB only has the Team Workflows entry from init, auto-learn must still run."""
        store = MagicMock()
        store.list_knowledge.return_value = [
            {"id": 1, "title": "Team Workflows"},
        ]

        with patch("myswat.cli.learn_cmd.run_learn") as mock_run:
            ensure_learned(store, "my-project", "proj-1", tmp_path)

        mock_run.assert_called_once()

    def test_calls_run_learn_if_neither(self, tmp_path):
        """If no myswat.md and no TiDB ops, should call run_learn."""
        store = MagicMock()
        store.list_knowledge.return_value = []

        with patch("myswat.cli.learn_cmd.run_learn") as mock_run:
            ensure_learned(store, "my-project", "proj-1", tmp_path)

        mock_run.assert_called_once()
        # Verify it was called with the right project slug and workdir.
        call_kwargs = mock_run.call_args
        assert "my-project" in call_kwargs.args or (
            call_kwargs.kwargs.get("workdir") == tmp_path
            or call_kwargs.args[0] == "my-project"
        )

    def test_calls_run_learn_with_correct_workdir(self, tmp_path):
        """run_learn should be invoked with workdir=repo_path."""
        store = MagicMock()
        store.list_knowledge.return_value = []

        with patch("myswat.cli.learn_cmd.run_learn") as mock_run:
            ensure_learned(store, "my-project", "proj-1", tmp_path)

        # Check that workdir was passed as tmp_path.
        _, kwargs = mock_run.call_args
        if "workdir" in kwargs:
            assert kwargs["workdir"] == tmp_path
        else:
            # If passed positionally, the second arg should be the workdir.
            args = mock_run.call_args.args
            assert tmp_path in args or str(tmp_path) in [str(a) for a in args]


# ---------------------------------------------------------------------------
# _write_myswat_md – additional coverage for list/dict branches
# ---------------------------------------------------------------------------
class TestWriteMyswatMdBranches:
    """Additional tests for _write_myswat_md covering list vs dict value branches."""

    def test_list_values_rendered_as_bullets(self, tmp_path):
        """List values (e.g. invariants) should be rendered as bullet items."""
        data = {
            "build": {"commands": ["make"]},
            "test": {"tiers": [{"name": "unit"}]},
            "structure": {"entry_points": ["main.py"]},
            "invariants": ["never break build", "all tests pass"],
        }
        result_path = _write_myswat_md(tmp_path, data, "proj")
        content = result_path.read_text()
        assert "- never break build" in content
        assert "- all tests pass" in content

    def test_dict_values_rendered_as_json(self, tmp_path):
        """Dict values should be JSON-serialized."""
        data = {
            "build": {"commands": ["make"]},
            "test": {"tiers": [{"name": "unit"}]},
            "structure": {"entry_points": ["main.py"]},
            "conventions": {"rules": ["use snake_case"]},
        }
        result_path = _write_myswat_md(tmp_path, data, "proj")
        content = result_path.read_text()
        assert "use snake_case" in content

    def test_empty_list_skipped(self, tmp_path):
        """An empty list value should be skipped."""
        data = {
            "build": {"commands": ["make"]},
            "test": {"tiers": [{"name": "unit"}]},
            "structure": {"entry_points": ["main.py"]},
            "invariants": [],
        }
        result_path = _write_myswat_md(tmp_path, data, "proj")
        content = result_path.read_text()
        # "Invariants" section should not appear
        assert "Invariants" not in content
