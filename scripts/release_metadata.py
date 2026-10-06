"""Validate release metadata without importing MLX or interpreting commit messages."""

import argparse
import ast
import json
import re
import sys
from pathlib import Path


def version_tuple(value):
    if not re.fullmatch(r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)", value):
        raise ValueError(f"Invalid stable version: {value!r}")
    return tuple(map(int, value.split(".")))


def package_version(root):
    project = re.search(r"(?ms)^\[project\]\s*\n(.*?)(?=^\[|\Z)", (root / "pyproject.toml").read_text())
    match = re.search(r'^version\s*=\s*"([^"]+)"\s*$', project.group(1), re.M) if project else None
    if not match:
        raise ValueError("Missing project version")
    version = match.group(1)
    version_tuple(version)
    runtime = None
    for node in ast.parse((root / "mlx_flash_compress" / "__init__.py").read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets
        ):
            runtime = ast.literal_eval(node.value)
    if version != runtime:
        raise ValueError(f"Package/runtime version mismatch: {version!r} != {runtime!r}")
    return version


def latest_version(tag):
    if not tag:
        return (-1, -1, -1)
    if not tag.startswith("v"):
        raise ValueError("Latest release must have a v-prefixed stable tag")
    return version_tuple(tag[1:])


def highest_release(pages):
    tags = []
    for page in pages:
        for release in page:
            tag = release.get("tag_name", "")
            if not release.get("draft") and not release.get("prerelease") and re.fullmatch(r"v\d+\.\d+\.\d+", tag):
                tags.append(tag)
    return max(tags, key=latest_version, default="")


def plan(prs, sha, version, latest, prepare_missing=False, changelog="", tag_at_head=""):
    candidates = [
        pr
        for pr in prs
        if pr.get("merged_at") and pr.get("merge_commit_sha") == sha and pr.get("base", {}).get("ref") == "main"
    ]
    if len(candidates) > 1:
        raise ValueError("Multiple merged PRs match the tested commit")
    if not candidates:
        return {"number": "", "bump": "none", "tag": ""}
    pr = candidates[0]
    labels = {label["name"] for label in pr.get("labels", [])}
    bumps = [b for b in ("major", "minor", "patch") if f"release:{b}" in labels]
    if any(
        label.startswith("release:")
        and label not in {"release:major", "release:minor", "release:patch", "release:automated"}
        for label in labels
    ):
        raise ValueError("Unknown release label")
    if not bumps and prepare_missing:
        title = pr.get("title", "").lower()
        if re.match(r"^[a-z]+(?:\([^)]*\))?!:", title):
            bumps = ["major"]
        elif re.match(r"^feat(?:\([^)]*\))?:", title):
            bumps = ["minor"]
        elif re.match(r"^(?:fix|perf|refactor)(?:\([^)]*\))?:", title):
            bumps = ["patch"]
    if not bumps:
        return {"number": pr["number"], "bump": "none", "tag": ""}
    if len(bumps) != 1:
        raise ValueError("Use exactly one release label")
    if prepare_missing:
        if tag_at_head:
            if tag_at_head != f"v{version}":
                raise ValueError("Existing tag at tested commit must match package version")
            if not re.search(r"^## \[" + re.escape(version) + r"\](?:\s|$)", changelog, re.M):
                raise ValueError("Existing release tag is missing tracked release notes")
            return {"number": pr["number"], "bump": bumps[0], "tag": tag_at_head, "prepare": ""}
        target = version
        if latest:
            major, minor, patch = latest_version(latest)
            requested = {
                "major": f"{major + 1}.0.0",
                "minor": f"{major}.{minor + 1}.0",
                "patch": f"{major}.{minor}.{patch + 1}",
            }[bumps[0]]
            if version_tuple(version) < version_tuple(requested):
                target = requested
        has_notes = re.search(r"^## \[" + re.escape(target) + r"\](?:\s|$)", changelog, re.M)
        if target != version or not has_notes:
            return {"number": pr["number"], "bump": bumps[0], "tag": "", "prepare": target}
        return {"number": pr["number"], "bump": bumps[0], "tag": f"v{version}", "prepare": ""}
    if version_tuple(version) <= latest_version(latest):
        raise ValueError("Release PR must pre-bump both package versions above the latest release")
    return {"number": pr["number"], "bump": bumps[0], "tag": f"v{version}"}


def verify(tag, version, latest, repository):
    if tag != f"v{version}":
        raise ValueError(f"Release tag must match checked-out package version v{version}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Invalid repository name")
    return {
        "version": version,
        "tag": tag,
        "promote": str(version_tuple(version) >= latest_version(latest)).lower(),
        "image": f"ghcr.io/{repository.lower()}",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "verify", "latest"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--latest", default="")
    parser.add_argument("--prs", type=Path)
    parser.add_argument("--sha")
    parser.add_argument("--tag")
    parser.add_argument("--repository")
    parser.add_argument("--releases", type=Path)
    parser.add_argument("--prepare-missing-version", action="store_true")
    parser.add_argument("--tag-at-head", default="")
    args = parser.parse_args()
    try:
        if args.command == "latest":
            print(highest_release(json.loads(args.releases.read_text())))
            return
        version = package_version(args.root)
        if args.command == "plan":
            changelog_path = args.root / "CHANGELOG.md"
            changelog = changelog_path.read_text() if changelog_path.exists() else ""
            outputs = plan(
                json.loads(args.prs.read_text()),
                args.sha,
                version,
                args.latest,
                args.prepare_missing_version,
                changelog,
                args.tag_at_head,
            )
        else:
            outputs = verify(args.tag, version, args.latest, args.repository)
        for key, value in outputs.items():
            print(f"{key}={value}")
    except (ValueError, TypeError, OSError) as exc:
        parser.exit(1, f"Release metadata error: {exc}\n")


if __name__ == "__main__":
    main()
