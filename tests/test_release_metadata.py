"""Publishing decisions must be pinned to the tested commit and package version."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "release_metadata.py"


def run_metadata(tmp_path, *args):
    if "--root" not in args:
        (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.8.0"\n')
        package = tmp_path / "mlx_flash_compress"
        package.mkdir(exist_ok=True)
        (package / "__init__.py").write_text('__version__ = "0.8.0"\n')
        args = (*args, "--root", str(tmp_path))
    result = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)
    return result


def test_pr_without_number_in_commit_message_selects_exact_merged_main_commit(tmp_path):
    prs = tmp_path / "prs.json"
    prs.write_text(
        json.dumps(
            [
                {
                    "number": 1,
                    "merged_at": "date",
                    "merge_commit_sha": "other",
                    "base": {"ref": "main"},
                    "labels": [{"name": "release:major"}],
                },
                {
                    "number": 2,
                    "merged_at": "date",
                    "merge_commit_sha": "tested",
                    "base": {"ref": "main"},
                    "labels": [{"name": "release:patch"}, {"name": "documentation"}],
                },
            ]
        )
    )
    result = run_metadata(tmp_path, "plan", "--prs", str(prs), "--sha", "tested", "--latest", "v0.7.1")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["number=2", "bump=patch", "tag=v0.8.0"]


@pytest.mark.parametrize(
    "changes",
    [
        {"merged_at": None},
        {"base": {"ref": "develop"}},
        {"merge_commit_sha": "other"},
        {"labels": [{"name": "not-release:patch"}]},
    ],
)
def test_unrelated_or_unlabelled_pr_cannot_publish(tmp_path, changes):
    pr = {
        "number": 2,
        "merged_at": "date",
        "merge_commit_sha": "tested",
        "base": {"ref": "main"},
        "labels": [{"name": "release:patch"}],
    }
    pr.update(changes)
    prs = tmp_path / "prs.json"
    prs.write_text(json.dumps([pr]))
    result = run_metadata(tmp_path, "plan", "--prs", str(prs), "--sha", "tested", "--latest", "v0.7.1")
    assert result.returncode == 0, result.stderr
    assert "bump=none" in result.stdout.splitlines()


def test_existing_version_cannot_be_retagged_as_new_release(tmp_path):
    prs = tmp_path / "prs.json"
    prs.write_text(
        json.dumps(
            [
                {
                    "number": 2,
                    "merged_at": "date",
                    "merge_commit_sha": "tested",
                    "base": {"ref": "main"},
                    "labels": [{"name": "release:minor"}],
                }
            ]
        )
    )
    result = run_metadata(tmp_path, "plan", "--prs", str(prs), "--sha", "tested", "--latest", "v0.8.0")
    assert result.returncode != 0
    assert "pre-bump" in result.stderr


@pytest.mark.parametrize("tag", ["v0.9.0", "refs/heads/main", "v0.8.0\ninvalid=value"])
def test_manual_release_rejects_wrong_or_unsafe_tag(tmp_path, tag):
    result = run_metadata(tmp_path, "verify", "--tag", tag, "--latest", "v0.7.1", "--repository", "Mixed/Repo")
    assert result.returncode != 0


def test_older_release_does_not_promote_latest(tmp_path):
    result = run_metadata(tmp_path, "verify", "--tag", "v0.8.0", "--latest", "v0.9.0", "--repository", "Mixed/Repo")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["version=0.8.0", "tag=v0.8.0", "promote=false", "image=ghcr.io/mixed/repo"]


def test_package_and_runtime_versions_must_agree(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.8.0"\n')
    package = tmp_path / "mlx_flash_compress"
    package.mkdir()
    (package / "__init__.py").write_text('__version__ = "0.7.1"\n')
    result = run_metadata(tmp_path, "verify", "--root", str(tmp_path), "--tag", "v0.8.0", "--repository", "owner/repo")
    assert result.returncode != 0
    assert "version mismatch" in result.stderr


def test_latest_uses_highest_stable_version_not_creation_order(tmp_path):
    releases = tmp_path / "releases.json"
    releases.write_text(
        json.dumps(
            [
                [
                    {"tag_name": "v0.7.1", "draft": False, "prerelease": False},
                    {"tag_name": "v0.10.0", "draft": False, "prerelease": False},
                ],
                [
                    {"tag_name": "v0.9.0", "draft": False, "prerelease": False},
                    {"tag_name": "v1.0.0", "draft": True, "prerelease": False},
                    {"tag_name": "v2.0.0", "draft": False, "prerelease": True},
                ],
            ]
        )
    )
    result = run_metadata(tmp_path, "latest", "--releases", str(releases))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "v0.10.0"


@pytest.mark.parametrize("label,target", [("minor", "0.9.0"), ("patch", "0.8.1"), ("major", "1.0.0")])
def test_automatic_plan_prepares_unbumped_package(tmp_path, label, target):
    prs = tmp_path / "prs.json"
    prs.write_text(
        json.dumps(
            [
                {
                    "number": 21,
                    "merged_at": "date",
                    "merge_commit_sha": "tested",
                    "base": {"ref": "main"},
                    "labels": [{"name": "release:" + label}],
                }
            ]
        )
    )
    result = run_metadata(
        tmp_path, "plan", "--prs", str(prs), "--sha", "tested", "--latest", "v0.8.0", "--prepare-missing-version"
    )
    assert result.returncode == 0, result.stderr
    assert f"prepare={target}" in result.stdout.splitlines()
    assert "tag=" in result.stdout.splitlines()


def test_conventional_feature_title_requests_automatic_preparation(tmp_path):
    prs = tmp_path / "prs.json"
    prs.write_text(
        json.dumps(
            [
                {
                    "number": 21,
                    "title": "feat: native cache",
                    "merged_at": "date",
                    "merge_commit_sha": "tested",
                    "base": {"ref": "main"},
                    "labels": [],
                }
            ]
        )
    )
    result = run_metadata(
        tmp_path, "plan", "--prs", str(prs), "--sha", "tested", "--latest", "v0.8.0", "--prepare-missing-version"
    )
    assert result.returncode == 0, result.stderr
    assert "prepare=0.9.0" in result.stdout.splitlines()


def test_automatic_tag_requires_matching_changelog_section(tmp_path):
    prs = tmp_path / "prs.json"
    prs.write_text(
        json.dumps(
            [
                {
                    "number": 22,
                    "merged_at": "date",
                    "merge_commit_sha": "tested",
                    "base": {"ref": "main"},
                    "labels": [{"name": "release:minor"}],
                }
            ]
        )
    )
    package = tmp_path / "mlx_flash_compress"
    package.mkdir()
    (package / "__init__.py").write_text('__version__ = "0.9.0"\n')
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.9.0"\n')
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## [Unreleased]\n")
    args = (
        "plan",
        "--root",
        str(tmp_path),
        "--prs",
        str(prs),
        "--sha",
        "tested",
        "--latest",
        "v0.8.0",
        "--prepare-missing-version",
    )
    result = run_metadata(tmp_path, *args)
    assert result.returncode == 0 and "prepare=0.9.0" in result.stdout.splitlines()
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## [0.9.0] - 2026-10-06\n\n- Cache reuse.\n")
    result = run_metadata(tmp_path, *args)
    assert result.returncode == 0 and "tag=v0.9.0" in result.stdout.splitlines()


def test_rerunning_already_published_commit_resumes_same_tag(tmp_path):
    prs = tmp_path / "prs.json"
    prs.write_text(
        json.dumps(
            [
                {
                    "number": 21,
                    "merged_at": "date",
                    "merge_commit_sha": "tested",
                    "base": {"ref": "main"},
                    "labels": [{"name": "release:minor"}],
                }
            ]
        )
    )
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## [0.8.0] - 2026-10-06\n\n- Changes.\n")
    result = run_metadata(
        tmp_path,
        "plan",
        "--prs",
        str(prs),
        "--sha",
        "tested",
        "--latest",
        "v0.8.0",
        "--prepare-missing-version",
        "--tag-at-head",
        "v0.8.0",
    )
    assert result.returncode == 0, result.stderr
    assert "tag=v0.8.0" in result.stdout.splitlines()
    assert "prepare=" in result.stdout.splitlines()


def test_prebumped_patch_cannot_satisfy_major_release_label(tmp_path):
    prs = tmp_path / "prs.json"
    prs.write_text(
        json.dumps(
            [
                {
                    "number": 21,
                    "merged_at": "date",
                    "merge_commit_sha": "tested",
                    "base": {"ref": "main"},
                    "labels": [{"name": "release:major"}],
                }
            ]
        )
    )
    package = tmp_path / "mlx_flash_compress"
    package.mkdir()
    (package / "__init__.py").write_text('__version__ = "0.8.1"\n')
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.8.1"\n')
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## [0.8.1] - 2026-10-06\n\n- Change.\n")
    result = run_metadata(
        tmp_path,
        "plan",
        "--root",
        str(tmp_path),
        "--prs",
        str(prs),
        "--sha",
        "tested",
        "--latest",
        "v0.8.0",
        "--prepare-missing-version",
    )
    assert result.returncode == 0, result.stderr
    assert "tag=" in result.stdout.splitlines() and "prepare=1.0.0" in result.stdout.splitlines()
