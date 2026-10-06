# Generation measurements and their limits

The standard unbatched chat path obtains text and accounting from `mlx_lm.stream_generate`, using its final `GenerationResponse`. It does not estimate completion tokens by re-tokenizing decoded output. Native `prompt_tokens` counts the actual formatted prompt; `generation_tokens` counts generated steps, including a terminating EOS token. `finish_reason` distinguishes EOS (`stop`) from a token budget (`length`). EOS at the budget boundary can therefore have `completion_tokens == max_tokens` and still finish normally. These fields follow the [MLX-LM generation implementation](https://github.com/ml-explore/mlx-lm/blob/v0.32.0/mlx_lm/generate.py).

Each standard chat response includes `mlx_flash_compress.native_generation_metadata: true` and `usage_source: "exact_mlx_lm_generation"`. Clients can use this marker to distinguish actual native stop metadata from an older server that reported `stop` for every result. An incomplete native iterator is an error, not a completed response. The existing package version remains 0.8.0; inspect the capability marker rather than assuming every 0.8.0 installation has these changes.

| Field | Measurement scope |
| --- | --- |
| `usage.prompt_tokens` | Model input after system text, chat template, tool descriptions and history are formatted and tokenized. |
| `usage.completion_tokens` | Native generated steps, including thinking/control/EOS tokens; not only visible answer text. |
| `usage.total_tokens` | Sum of the two known counts. Missing counts remain unknown rather than fabricated zero. |
| `mlx_flash_compress.tok_per_s` | Generated tokens divided by total generation elapsed time, including prefill and final synchronization. Kept for compatibility. |
| `mlx_flash_compress.generation_tps` | MLX-LM's reported generation throughput, distinct from the total elapsed rate. |
| `mlx_flash_compress.prompt_tps` | MLX-LM's reported prompt processing rate. |
| `mlx_flash_compress.ttft_ms` | Wall time from generation start to the first native iterator response; that response's decoded text can be empty. |
| `mlx_flash_compress.peak_memory_gb` | Native allocator peak reported by MLX-LM, potentially the high-water mark since the last allocator reset. Not machine-wide RAM or incremental memory allocated by this request. |
| `/status.stats.last_generation` | Latest completed generation's counts, stop reason, elapsed time and available rates, without prompts or answers. `null` before any completion. |

Unbatched SSE remains buffered: inference completes before SSE content is delivered. The response explicitly labels this with `X-MLX-Generation: buffered; chunks are not live model tokens`. Network time to the first content byte therefore includes the whole generation, even when native `ttft_ms` is much shorter. Setting `stream_options.include_usage` includes counts in the final SSE event. This change removes simulated word-by-word chunking; it does not claim true native token streaming.

## Sampling and other paths

Temperature, `top_p` and `top_k` reach MLX-LM's [sampler constructor](https://github.com/ml-explore/mlx-lm/blob/v0.32.0/mlx_lm/sample_utils.py). Temperature zero is greedy; temperature must be finite and nonnegative, `0 < top_p <= 1`, and `top_k` must be a nonnegative integer. The request token budget must be a positive integer. Unsupported request controls, including custom stop strings, seeds, penalties, multiple choices and native tool metadata, return HTTP 400 instead of silently doing nothing. Tools can still be encoded in prompt text by a caller such as Sentinel.

Continuous batching honors its existing temperature sampler and rejects active `top_p`/`top_k` filters. It records actual EOS versus budget stop reasons and token-list counts as `usage_source: "exact_batch_tokens"`; its native-generation marker remains false. Latest batching timing includes queue time, with unknown prefill rate and peak allocation reported as `null`. Its SSE path delivers worker tokens; cancellation has no successful stop reason. Existing per-token text decoding limitations are separate from these accounting fixes.

The custom speculative engines currently provide greedy token arrays rather than MLX-LM response metadata. Requests must explicitly use temperature zero without sampling filters; unsupported controls are rejected. Their counts come from the actual prompt and returned token arrays (`exact_speculative_tokens`), while unavailable decode rate, prefill rate, first-response timing and peak allocation remain `null`. A stop reason is inferred only from an explicit EOS token or an exhausted budget; otherwise it remains unknown. They do not receive the standard native-generation marker.

The `/status` cumulative counters represent completed chat generations handled by this server instance. Model loading/warm-up, profiling requests, failed generations and incomplete batching requests are not interchangeable with those counters. Corrections performed by an upstream adapter create separate backend generations; correlate them outside MLX-Flash before reporting successful user turns. Separate runtime processes each have their own counters, while available RAM, swap and pressure describe the same machine and must not be summed.

These measurements do not provide commercial billing, cached-token discounts, dollar costs, task quality or proof that a model changed a file. Provider billing/cache data must come from that provider. Quality needs evaluated tasks and real tool-result evidence; speed, token counts and successfully parsed tool calls alone cannot establish answer correctness.

## Model-free verification

`tests/test_generation_measurements.py` imports isolated source modules with MLX dependencies mocked before import. Its tests exercise actual server generation/HTTP logic against fake native response objects, including EOS at the budget boundary, truncation, sampling forwarding, incomplete iterators, honest buffered SSE and batching counts. No model download, Metal inference or running-server restart is required:

```sh
python -m unittest discover -s tests -p test_generation_measurements.py -v
```

The interpreter still needs the server's ordinary Python dependencies such as NumPy and PyYAML. Native performance and real model quality require a separate, explicitly scheduled hardware evaluation; this test suite does not manufacture those measurements.
