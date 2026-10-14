# Optional image understanding

The normal `mlx-lm` runtime remains text only. Install the optional `vision`
extra and choose a vision checkpoint explicitly to serve OpenAI chat content
containing `image_url` parts:

The managed launcher also supports `make up VISION=1 PORT=8082`; it selects the
small Gemma checkpoint by default, installs the optional extra even when the
text environment already exists, and passes `--vision --prompt-cache-bytes 0`.
Use `MODEL=...` to choose another supported vision checkpoint. For direct use
of `scripts/local_runtime.py`, set `MLX_FLASH_VISION=1`.

```sh
pip install 'mlx-flash[vision]'
mlx-flash --vision --model mlx-community/gemma-4-e2b-it-4bit \
  --preload --host 127.0.0.1 --port 8082 --prompt-cache-bytes 0
```

The adapter uses the upstream [mlx-vlm Python API](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma4/README.md):
`load`, `apply_chat_template`, and `generate` with decoded images. Its dependency
range is `mlx-vlm>=0.7.6,<0.8`. Model weights and processors come from the
operator-selected model. This adapter performs image understanding; it provides
no image generation, cutout, matting, or editing endpoint.

Only one image per request is accepted, as a base64 PNG or JPEG data URL on a
user turn. Remote URLs, file URLs, audio, video and animated images are rejected.
Each decoded image is bounded to 4 MiB and 4,194,304 pixels. The HTTP request
limit remains 8 MiB. The adapter preserves ordered image/text content on its
original user turn, including across a structured-output repair attempt.

`GET /v1/capabilities` advertises vision only after the selected model has loaded
successfully and exposed both a vision configuration and image processor.
`--preload` validates these before the server starts accepting traffic. The
status includes the exact `model`, `runtime: mlx_vlm`, `model_available`, and
`unload_after_request: true`. A verified idle runtime has `model_loaded: false`
and `model_available: true`; the latter describes validated availability, not
GPU residency or visual accuracy. `/v1/models` retains the client alias `local`
and includes the selected model identity and actual modalities.

The server admits one vision request at a time and rejects concurrent requests
with HTTP 503 `runtime_busy` before reading their image bodies. It releases
the admission slot after a failed or incomplete upload as well. Vision body
reads have a total deadline of `min(request_timeout, 30 seconds)`; a stalled or
trickling client receives HTTP 408 `request_timeout`, with its connection closed.
The deadline covers upload reads and does not interrupt active model inference.
The adapter releases its own model and processor references, runs collection, synchronizes Metal,
and clears the allocator cache after every generation, including failed calls.
This avoids keeping its model resident between sequential phases handled by
other processes. Reloading cached weights on each request costs latency.
Independent processes still require operator coordination; this adapter cannot
reserve system RAM or prevent another application loading a model.

Vision requests reject streaming, thinking output, prompt reuse, batching,
speculative decoding, and KV quantization. Native configuration/model switching
and profiling endpoints are disabled for this runtime. Text completions and
simple text Responses use the same vision runtime but accept text only.
Structured chat responses retain local JSON/schema validation and one bounded
repair attempt; invalid final JSON returns HTTP 502. Empty answers, incomplete
generation metadata and unresolved control tokens also fail without success.

The API reports native `mlx-vlm` token counts, finish reason, throughput and peak
Metal allocation. These do not prove factual grounding or segmentation quality.
The following explicit real-model check sends two unpredictable synthetic color
layouts and compares the answers against the pixels:

```sh
python scripts/verify_vision_api.py --base http://127.0.0.1:8082 \
  --out /tmp/vision-proof.json
```

An optional `--image /path/to/private-photo.jpg` adds a general visible-scene
schema probe. No input bytes or descriptions are saved by the script. Its
summary explicitly leaves visual accuracy unverified; a human must review
actual image-specific output separately. Do not infer broad model quality or
16 GB suitability solely from passing unit tests or synthetic color probes.

## Recorded native proof (2026-10-11)

On the development Apple Silicon Mac with 16 GB RAM, the cached
`mlx-community/gemma-4-e2b-it-4bit` checkpoint with `mlx-vlm 0.7.6` returned
HTTP 200 and exact answers for both randomly permuted color layouts. End-to-end
latencies were 4.41 s and 2.85 s; reported peak Metal allocations were 4.035 GB
and 4.049 GB. The model was nonresident with validated availability after both
requests. An additional private-image schema probe returned HTTP 200 in 3.29 s
with reported peak 4.238 GB; its visual accuracy was not evaluated by this
universal API proof. Input data and generated descriptions were not committed.

These results establish actual image-conditioned structured generation and
sequential release for this small checkpoint. They do not establish broader
photographic accuracy, concurrent coexistence with another model, or image
editing quality. Metal compiler kernel checks and distribution-artifact checks
remain skipped in the ordinary test environment and require separate jobs.
