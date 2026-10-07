# Real Qwen Metal CI

The opt-in `Real Qwen Metal smoke` workflow runs offline native HTTP generation
on a privately managed Apple Silicon Mac. Trusted CI/release workflows call it
through `workflow_call`; it also accepts manual dispatches from main or a trusted
tag. Callers pass their validated source commit in `source_ref` (default: the
calling event's `github.sha`). Both hosted and hardware jobs check out that exact
commit. It never runs pull-request code on the Mac.
PR harness checks run on GitHub-hosted Ubuntu; existing ordinary PR CI remains
hosted. A hosted status job explicitly reports when hardware CI is disabled.
Missing models or unavailable Metal fail an enabled hardware job; they never
produce a passing inference check.

Register a repository-specific runner with labels
`self-hosted`, `macOS`, `ARM64`, `qwen-metal`. Set the repository variable
`QWEN_METAL_ENABLED=true` only after the runner and model store are ready.
Keep these machine-specific paths in the runner's environment, outside Git:

| Variable | Meaning |
| --- | --- |
| `QWEN_CI_PYTHON` | Absolute Python executable in the cached native MLX environment |
| `QWEN_SMALL_MODEL_PATH` | Absolute cached small instruction model; the current lab uses LFM2.5-8B-A1B |
| `QWEN_LARGE_MODEL_PATH` | Absolute cached large instruction model; the current lab uses Gemma 4 26B-A4B |
| `QWEN_CI_LOCK_PATH` | Optional shared lock, default `/private/tmp/qwen-metal-ci.lock` |

The job installs the **checked-out MLX-Flash source**, editable, into a temporary
venv derived from the cached runtime. Native package versions are pinned to
MLX 0.32.3, mlx-lm 0.32.0, and transformers 5.18.0. Build tools can be downloaded;
model downloading is disabled. `QWEN_MLX_FLASH_BIN` is set to this job's executable,
so an old host installation cannot silently satisfy the current-source test.

The lab and runner model store now select
[`mlx-community/Qwen3.8-27B-4bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/tree/10c35caafbb80f7dc6a7a432cdd11af10a6d4818)
at revision `10c35caafbb80f7dc6a7a432cdd11af10a6d4818`, with all three weight
shards checked against Hub SHA-256 metadata. Include `chat_template.jinja` when
preparing a cached directory. Existing `qwen3_5` support in mlx-lm 0.32.0 loads
the language model for text-only generation; vision/video are not covered by
this smoke. Small remains Qwen3.5-4B. Model paths stay host configuration rather
than hard-coded workflow downloads.

The interactive Sentinel lab passed all three exact role markers, completed
thinking, required Claude tool schema output, and OpenAI Responses formatting
with this artifact on 2026-10-06. This is local compatibility evidence, not a
claim that the new artifact has already passed the entire CI hardware matrix.
Thinking defaults to the model template's `xhigh` effort; the runtime currently
accepts only `enable_thinking` as a template override. The dense 27B model may
decode slower than the previous Qwen3.6 35B-A3B MoE. Quality and speed require
representative task evaluation; do not copy old model benchmarks onto this one.

Small and large models run sequentially to limit memory. Each receives real
Chat Completions requests with thinking disabled and enabled. Both must return
the exact final marker with normal token accounting; thinking must complete its
reasoning section. For a local small-only diagnostic, set `QWEN_RUN_LARGE=false`;
the workflow always runs both models. The harness validates cached files and
shard indexes before starting a server and never alters model weights.

The harness uses ports 19190/19191, leaving the interactive lab's 19090/19091/19092
alone. It rejects occupied CI ports, owns fresh process groups, and cleans up on
errors and SIGINT/SIGTERM. Multiple repository-specific runners on the same Mac
must share the lock path and OS account. GitHub concurrency is repository-local;
the file lock serializes inference across MLX-Flash and Sentinel for the entire
preflight/run/cleanup interval. Waiting for the lock fails after 30 minutes.

## Optional Sentinel lab memory handoff

Configure all three repository variables as absolute paths on the self-hosted
Mac: `SENTINEL_CI_TOOLS_BIN`, `SENTINEL_CI_PROJECT_ROOT`, and
`SENTINEL_CI_LAB_ROOT`. The binary must be the current Go `sentinel-tools`; also
install it at `<project>/bin/sentinel-tools` for lab restoration. With the shared
Metal lock held, the harness calls its `ci-lab pause` command, runs its own models,
then calls `ci-lab resume` with the returned private ownership token. This frees
lab model memory during CI and restores it even after a generation failure.
Restore failures fail the smoke. Partial configuration and nonexecutable or
symlink helpers are rejected. With all variables absent, behavior is unchanged.

The Go coordinator refuses busy, unknown, attached or already reserved labs.
Stopped labs stay stopped; an explicit user stop during CI cancels restoration.
Configure the same handoff in Sentinel's runner and share the OS account and
`QWEN_CI_LOCK_PATH`. SIGKILL and host power loss require inspection of the private
lab `ci-pause.json` and owned processes before manual recovery. Never remove a
lease while its hardware job is still running.

`qwen-metal-artifacts/results.json` contains health capability summaries, each
passed check's usage, and overall success/failure. Runtime logs are sanitized for
machine paths before upload. Failure propagates as a nonzero job exit. Interrupted
jobs have best-effort cleanup; SIGKILL or host power loss cannot run cleanup code.
The workflow removes its temporary venv after artifact collection.

Model-free lifecycle verification:

```sh
python3 -m unittest discover -s scripts -p test_qwen_metal_smoke.py -v
```

Direct real smoke, using an already installed current-source venv:

```sh
QWEN_MLX_FLASH_BIN=/absolute/venv/bin/mlx-flash python3 scripts/qwen_metal_smoke.py
```

This checks native generation and request profiles. It is not a coding-quality
benchmark or a claim of general tool reliability.
