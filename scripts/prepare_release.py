"""Prepare consistent versions and release notes in a reviewable release branch."""

import argparse
import datetime
import re
from pathlib import Path

from release_metadata import package_version, version_tuple


def prepare(root, version, date, notes):
    version_tuple(version)
    datetime.date.fromisoformat(date)
    current = package_version(root)
    if version_tuple(version) < version_tuple(current):
        raise ValueError("Release preparation cannot downgrade the package")
    project_path = root / "pyproject.toml"
    runtime_path = root / "mlx_flash_compress" / "__init__.py"
    changelog_path = root / "CHANGELOG.md"
    project = project_path.read_text()
    runtime = runtime_path.read_text()
    changelog = changelog_path.read_text()
    # Restrict replacement to [project], preserving optional dependency metadata.
    project = re.sub(
        r'(?ms)(^\[project\]\s*\n(?:(?!^\[).)*?^version\s*=\s*)"[^"\n]+"',
        lambda match: match.group(1) + '"' + version + '"',
        project,
        count=1,
    )
    runtime = re.sub(
        r"^__version__\s*=\s*(['\"])[^'\"\n]+\1", '__version__ = "' + version + '"', runtime, count=1, flags=re.M
    )
    if not re.search(r"^## \[" + re.escape(version) + r"\](?:\s|$)", changelog, re.M):
        bullets = list(dict.fromkeys(line.strip()[:500] for line in notes.splitlines() if line.strip()))
        if not bullets:
            raise ValueError("Release notes must include at least one change")
        section = f"## [{version}] - {date}\n\n### Changes\n" + "".join(f"- {line}\n" for line in bullets) + "\n"
        unreleased = re.search(r"^## \[Unreleased\][^\n]*\n", changelog, re.M)
        if unreleased:
            changelog = (
                changelog[: unreleased.start()]
                + "## [Unreleased]\n\n"
                + section
                + changelog[unreleased.end() :].lstrip("\n")
            )
        else:
            first = re.search(r"^## ", changelog, re.M)
            index = first.start() if first else len(changelog)
            changelog = changelog[:index].rstrip() + "\n\n## [Unreleased]\n\n" + section + changelog[index:]
    project_path.write_text(project)
    runtime_path.write_text(runtime)
    changelog_path.write_text(changelog)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--version", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--notes", type=Path, required=True)
    args = parser.parse_args()
    try:
        prepare(args.root, args.version, args.date, args.notes.read_text())
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Release preparation error: {exc}\n")


if __name__ == "__main__":
    main()
