# Releasing MLX-Flash

Before merging a release PR, bump `version` in `pyproject.toml` and
`__version__` in `mlx_flash_compress/__init__.py` to the same stable version
(`major.minor.patch`) and add exactly one label: `release:major`,
`release:minor`, or `release:patch`. The labels opt in to publishing; they
do not modify package versions. Update the changelog in the PR.

After the complete `CI` workflow succeeds for the current `main` commit,
Auto-Tag Release selects the merged PR whose merge SHA is that tested
commit. Unlabelled PRs and stale CI runs skip publishing. It creates an
annotated tag and directly calls the reusable release workflow, because
tags pushed with `GITHUB_TOKEN` do not trigger another workflow run.
The release workflow rechecks package versions, tests the pinned tag,
builds wheel and source distributions, checks that the wheel includes Metal
shader sources, and attaches them to a GitHub release. The reusable Qwen
workflow also validates the exact tagged commit before publishing; its native
Metal gate follows the repository's runner configuration.

To retry publishing, run **Release** manually and supply an existing tag
such as `v0.8.0`. The tag must point to a commit on `main`, contain the
release helper, and match both package versions. Every publishing job uses
the commit resolved from that tag. Dispatch the workflow from main or a
trusted tag; other branches are rejected so they cannot skip native inference.
Uploads may be retried for the same version; published package
contents are immutable and must never be replaced with another commit.

Repository secrets:

- `PYPI_TOKEN`: PyPI API token for `mlx-flash`. If absent, PyPI is explicitly
  skipped; GitHub release and GHCR still run.
- `HOMEBREW_TAP_TOKEN`: token with Contents write access to
  `<repository-owner>/homebrew-mlx-flash`. If absent, Homebrew is explicitly
  skipped. Homebrew uses the GitHub source archive and can run independently
  of PyPI.
- `GITHUB_TOKEN`: provided by Actions for repository releases and GHCR;
  workflow permissions must allow Contents write and Packages write.

GHCR publishes `ghcr.io/<owner>/<repository>:<version>` in lowercase. Only
the newest stable release can update the GitHub latest marker, Docker
`latest`, and the Homebrew formula. The Docker image runs platform-independent
tests and packaging on Linux; inference runs natively on Apple Silicon.
Docker intentionally installs the local package with `--no-deps` after
installing its non-MLX test dependencies.

Check publishing changes locally with:

```sh
python -m pytest tests/test_release_metadata.py -q
actionlint .github/workflows/auto-tag.yml .github/workflows/release.yml
python -m build
python -m twine check dist/*
MLX_FLASH_DIST_DIR=dist python -m pytest tests/test_package_artifacts.py -q
```

The repository's full test suite remains required. Local packaging checks
do not substitute for the native Metal smoke gate in CI.
