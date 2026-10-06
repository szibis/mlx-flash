"""Release preparation keeps package metadata and tracked release notes together."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_release.py"


@pytest.fixture
def project(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.8.0"\n')
    package = tmp_path / "mlx_flash_compress"
    package.mkdir()
    (package / "__init__.py").write_text('__version__ = "0.8.0"\n')
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased] — v0.6.0\n\n### Added\n- Existing notes.\n\n"
        "## [0.5.1] - 2026-03-26\n\n- Historical note.\n"
    )
    (tmp_path / "notes.txt").write_text("feat: native reuse (#21)\nfix: token measurements (#19)\n")
    return tmp_path


def prepare(project, version="0.9.0", date="2026-10-06"):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(project),
            "--version",
            version,
            "--date",
            date,
            "--notes",
            str(project / "notes.txt"),
        ],
        capture_output=True,
        text=True,
    )


def test_prepare_promotes_unreleased_and_preserves_history(project):
    result = prepare(project)
    assert result.returncode == 0, result.stderr
    assert 'version = "0.9.0"' in (project / "pyproject.toml").read_text()
    assert '__version__ = "0.9.0"' in (project / "mlx_flash_compress/__init__.py").read_text()
    changelog = (project / "CHANGELOG.md").read_text()
    assert "## [Unreleased]\n\n## [0.9.0] - 2026-10-06" in changelog
    assert "Existing notes." in changelog and "Historical note." in changelog
    assert "feat: native reuse (#21)" in changelog


def test_prepare_rerun_is_idempotent(project):
    assert prepare(project).returncode == 0
    before = (project / "CHANGELOG.md").read_bytes()
    assert prepare(project).returncode == 0
    assert (project / "CHANGELOG.md").read_bytes() == before


def test_prepare_accepts_single_quoted_runtime_version(project):
    runtime = project / "mlx_flash_compress/__init__.py"
    runtime.write_text("__version__ = '0.8.0'\n")
    result = prepare(project)
    assert result.returncode == 0, result.stderr
    assert '__version__ = "0.9.0"' in runtime.read_text()


@pytest.mark.parametrize(
    "version,date", [("0.9.0\nbad=value", "2026-10-06"), ("0.9.0", "2026-02-30"), ("0.7.0", "2026-10-06")]
)
def test_invalid_preparation_leaves_files_unchanged(project, version, date):
    before = {p: p.read_bytes() for p in [project / "pyproject.toml", project / "CHANGELOG.md"]}
    assert prepare(project, version, date).returncode != 0
    assert all(p.read_bytes() == value for p, value in before.items())
