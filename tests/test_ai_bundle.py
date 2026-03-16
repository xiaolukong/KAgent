"""Tests for the bundled AI guidance files in kagent._ai.

Note: force-include files (CLAUDE.md, AI_MANIFEST.json, examples/*.py) are
only present after a non-editable install (``pip install .``).  In an editable
install these files live at the repo root, not inside kagent/_ai/.  Tests that
require the bundled files are skipped when they are not present.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import kagent


def _has_bundle() -> bool:
    """Check whether the _ai/ bundle is materialized on disk."""
    try:
        guide_dir = kagent.get_ai_guide_path()
        return (guide_dir / "CLAUDE.md").is_file()
    except FileNotFoundError:
        return False


requires_bundle = pytest.mark.skipif(
    not _has_bundle(),
    reason="Bundled _ai/ files not present (editable install?)",
)


class TestAiPackage:
    """Basic tests that always pass."""

    def test_ai_package_importable(self) -> None:
        """kagent._ai should be importable as a package."""
        import kagent._ai

        assert kagent._ai is not None

    def test_get_ai_guide_path_exported(self) -> None:
        """get_ai_guide_path should be in kagent's public API."""
        assert "get_ai_guide_path" in kagent.__all__
        assert callable(kagent.get_ai_guide_path)


@requires_bundle
class TestAiBundle:
    """Tests that require the full bundle (non-editable install)."""

    def test_get_ai_guide_path_returns_directory(self) -> None:
        guide_dir = kagent.get_ai_guide_path()
        assert isinstance(guide_dir, Path)
        assert guide_dir.is_dir()

    def test_claude_md_exists(self) -> None:
        guide_dir = kagent.get_ai_guide_path()
        claude_md = guide_dir / "CLAUDE.md"
        assert claude_md.is_file()
        content = claude_md.read_text(encoding="utf-8")
        assert "KAgent" in content
        assert "Quick Import" in content

    def test_ai_manifest_valid_json(self) -> None:
        guide_dir = kagent.get_ai_guide_path()
        manifest = guide_dir / "AI_MANIFEST.json"
        assert manifest.is_file()
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert data["framework"] == "KAgent"
        assert "intents" in data
        assert len(data["intents"]) == 12

    def test_examples_directory_has_12_files(self) -> None:
        guide_dir = kagent.get_ai_guide_path()
        examples_dir = guide_dir / "examples"
        assert examples_dir.is_dir()
        py_files = sorted(f for f in examples_dir.glob("*.py") if f.name != "__init__.py")
        assert len(py_files) == 12
        assert py_files[0].name == "01_basic_chat.py"
        assert py_files[-1].name == "12_self_correction.py"

    def test_manifest_reference_files_resolve(self) -> None:
        guide_dir = kagent.get_ai_guide_path()
        manifest = guide_dir / "AI_MANIFEST.json"
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for intent in data["intents"] + data.get("composite_intents", []):
            for ref_path in intent["reference_files"]:
                full_path = guide_dir / ref_path
                assert full_path.is_file(), (
                    f"Reference file {ref_path} from intent {intent['id']} not found at {full_path}"
                )

    def test_importlib_resources_traversal(self) -> None:
        import importlib.resources

        ai_files = importlib.resources.files("kagent._ai")
        claude = ai_files.joinpath("CLAUDE.md")
        content = claude.read_text(encoding="utf-8")
        assert "KAgent" in content
