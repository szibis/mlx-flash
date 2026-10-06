# Releasing MLX-Flash

After successful current-main CI, Auto-Tag Release reads the exact merged PR.
One `release:major`, `release:minor`, or `release:patch` label chooses the bump;
otherwise conventional `feat!:` / `feat:` / `fix:` / `perf:` / `refactor:` titles
choose major/minor/patch. Scoped conventional titles also work. Other changes
skip publication. Conflicting or unknown bump labels fail explicitly.

If the package version is already published, or its changelog section is missing,
the workflow opens `release/prepare-v<version>` with matching `pyproject.toml`
and runtime versions plus a dated `CHANGELOG.md` section. It includes commit
subjects since the latest stable published release and promotes existing
Unreleased notes, preserving older history. Review and merge this release PR;
the workflow never merges it automatically or writes directly to main. Existing
open preparation PRs are reused without force-pushing. Changes merged after a
preparation PR was opened may require a subsequent release; review its notes and
base before merging.

The repository must allow GitHub Actions to create pull requests (Settings →
Actions → General → Workflow permissions). The workflow requests Contents,
Pull requests and Actions write permissions. A bot-created PR does not trigger
ordinary PR checks, so CI, security and the model-free harness are dispatched
explicitly on its branch. Dispatch runs cannot initiate publishing: publication
still requires a successful push CI on current main. No PAT or branch-protection
bypass is needed. See [GitHub's token-trigger rules](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow).

After the complete `CI` workflow succeeds for the current `main` commit,
Auto-Tag Release selects the merged PR whose merge SHA is that tested
commit. Stale CI runs skip publishing. When both package versions and the
matching changelog section are ready, it creates an
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
python -m pytest tests/test_release_metadata.py tests/test_prepare_release.py -q
actionlint .github/workflows/auto-tag.yml .github/workflows/release.yml
python -m build
python -m twine check dist/*
MLX_FLASH_DIST_DIR=dist python -m pytest tests/test_package_artifacts.py -q
```

The repository's full test suite remains required. Local packaging checks
do not substitute for the native Metal smoke gate in CI.
