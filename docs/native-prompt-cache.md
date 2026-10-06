# Native prompt reuse

Standard unbatched MLX generation retains up to eight exact-token-prefix
snapshots, within a 512 MiB budget per runtime. These are native MLX-LM cache
objects, including recurrent and rotating state. They are never cropped.
Snapshots are captured at the evaluated all-but-last prefill boundary, before
decode lookahead changes model state. Oversized snapshots are rejected before
copying; least recently used snapshots are evicted before admission.

This path requires MLX-LM 0.32 or newer. Older releases have different prefill
callback boundaries and lack these model-family implementations. The measured
checks used MLX-LM 0.32.0, MLX 0.32.3 and Transformers 5.18.0.

Use `--prompt-cache-bytes 0` to disable retention, or set
`--prompt-cache-bytes BYTES --prompt-cache-entries COUNT` to change limits.
At least one input token is processed on every request. A changed prefix can
miss the cache; arbitrary overlapping prefixes are not restored.

Requests may supply `cache_scope`, a nonempty string of at most 128 characters.
Native state is partitioned by model path, thinking profile and cache scope.
Standalone callers that omit it share the runtime's `default` namespace;
multi-user servers should assign distinct scopes. Sentinel supplies hashed
session identifiers, or isolated request identifiers when no session is known.
Caches are cleared on model reload/switch, KV precision changes, explicit
release and memory pressure. Speculative decoding and continuous batching do
not use this cache. Retained state is memory-only and expires with the process.

`/status.prompt_cache` reports hits, misses, entries, bytes, limits, evictions,
reused tokens and processed tokens. Hits/misses count lookups; token counters
count completed native generations only. Full logical input counts remain in
`usage.prompt_tokens`; `last_generation` and response extension metadata report
`cached_prompt_tokens` and `processed_prompt_tokens` separately. Reuse saves
prefill work, not conversation length or a claimed amount of money. Commercial
provider billing is outside this local runtime.

`/status.optimizations` reports configured prompt reuse, speculative mode,
batching and KV quantization. Nonzero KV precision now reaches standard native
generation. Local reasoning/tool profiles are advertised in `/health`:
Qwen and Gemma 4 support `enable_thinking`; LFM2.5 does not support that toggle.
Unknown families do not advertise it. LFM2.5 native generation defaults to its
recommended repetition penalty of 1.05, which can be overridden by a positive
finite `repetition_penalty`. Cached generations retain the full token history
for repetition processing, including the cached input prefix.

## Measured check on Apple M5 Pro, 64 GB

One synthetic marker prompt, greedy sampling, cached state versus a fresh
uncached run after kernel warm-up, October 6, 2026:

| Model | Logical input | Reused input | Fresh uncached TTFT | Cached TTFT | Output tokens |
|---|---:|---:|---:|---:|---:|
| Gemma 4 26B-A4B Instruct 4-bit | 1,239 | 1,238 | 488 ms | 28 ms | 4 |
| LFM2.5-8B-A1B 4-bit | 1,235 | 1,234 | 184 ms | 12 ms | 163 |

Cold, cached and fresh uncached outputs were identical for each model and
finished at EOS. LFM's output includes reasoning; its total generation lasted
about 0.91 seconds cached versus 1.04 seconds fresh uncached. Its first greedy
run without repetition control repeated context and exhausted the 256-token
budget; the native repetition penalty allowed the same task to finish. These
checks establish cache parity for these prompts, not a general quality ranking
or an expected speedup for every workload.

Pinned artifacts:

- `mlx-community/gemma-4-26b-a4b-it-4bit` at `0d77464eeb233a2da68ebf9d7dc4edaac7db956d`.
- `LiquidAI/LFM2.5-8B-A1B-MLX-4bit` at `2e92b640a63d47ad4dcf81a19a366b902356b3bc`.

The existing `qwen_metal_smoke.py` command and `QWEN_*` environment variables
are retained for compatibility. Its preflight and real-generation checks now
recognize these families and test only their supported thinking controls.
