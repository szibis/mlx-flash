# Native API regression proofs

Run the hardware-free validators with
`python -m unittest discover -s scripts -p 'test_*.py' -v`, then the full repository
suite with `python -m pytest tests/ -q`.

On Apple Silicon with cached models and the configured external runtime, use:

```sh
python scripts/qwen_metal_smoke.py --integration-proofs --artifacts /private/tmp/mlx-api-proofs
```

Set `QWEN_MLX_FLASH_BIN`, `QWEN_SMALL_MODEL_PATH`, `QWEN_LARGE_MODEL_PATH` and
`QWEN_CI_LOCK_PATH` explicitly. Existing Qwen environment names also support the
Gemma/LFM family profiles. This starts owned processes on isolated CI ports and
uses the host lock; it does not download weights or call commercial providers.
Use Python from the installed native environment so Metal can initialize.

The extended gate verifies cold/warm/fresh session parity, exact token-prefix
reuse, JSON/SSE text/finish/logical-usage parity, bounded cache state, completed
request/token counters, one-token budget termination, thinking-profile isolation
where supported, invalid request rejection before generation and release-time
cache clearing. Disabled thinking controls are recorded as not applicable for
families that have no toggle. Native reasoning output is checked by the separate
full-generation marker smoke; short budget/parity probes need not finish reasoning.

`results.json` retains named pass/fail checks, hashes of synthetic inputs/results,
actual usage and cache metadata. CI adds immutable revision information in
`revisions.json`; absent model revision values remain unknown. Failed probes
leave partial progress and fail the job. A skipped hardware job is not a pass.

Existing core tests separately protect divergence, profile/session namespaces,
LRU eviction, byte admission limits, disabled caches and completion accounting.
The current suite is for standard unbatched generation: optional speculative,
batching and compression modules retain their own unit suites and are not
represented as having real hardware coverage by these checks.

The Sentinel counterpart checks client Messages/Responses/Chat Completions,
framed streaming, validated tool continuation, gateway session scope, activity
and local policy. Its dependency is pinned: when updating either repository,
review the tested counterpart commit and retain both API proof artifacts.

Future native API/cache changes must add a negative regression and update the
hardware proof when relevant. Test results establish the contracts asserted;
they do not prove commercial model parity or provider-billed dollar savings.
