# OpenAI-compatible API support

The server preserves the existing `POST /v1/chat/completions` contract and exposes `GET /v1/capabilities` for callers that need to check features before sending work. `GET /v1/models` continues to return the local model and now includes additive modality and capability metadata. The server also supports non-streaming text-only `POST /v1/completions` with one string `prompt`, and non-streaming text-only `POST /v1/responses` with a string input or simple role/content message array and optional `instructions`.

The current server supports text chat completions, streaming text completions, `response_format.type=json_object`, and `response_format.type=json_schema`. Structured output is validated before the server returns HTTP success. If the first generation is invalid, the server makes one bounded correction attempt; if that output is invalid too, it returns HTTP 502 with `invalid_model_output`. Schema errors are rejected before inference. Schema keywords outside the documented implementation subset are rejected instead of silently ignored. This release does not yet constrain token sampling with a grammar, so the capability inventory reports the exact enforcement mode as `validated_before_success`.

The supported schema subset is `type`, `properties`, `required`, `additionalProperties` (boolean), `items`, `enum`, `const`, `anyOf`, numeric bounds, string length and `pattern`, and array length. The validator rejects remote or local `$ref` values, unknown keywords, excessive nesting/size, and regex constructs likely to cause unbounded matching time. Request JSON rejects duplicate object keys, non-finite numeric constants, and nesting deeper than 64 levels. The HTTP request body is limited to 8 MiB and structured model output to 256 KiB.

Streaming with `response_format` and structured output through continuous batching return an explicit 400 response. `/v1/completions` and `/v1/responses` currently reject streaming and unsupported request fields explicitly. The current server is text-only: multimodal message content is rejected before inference. Image understanding, image editing, and `/v1/embeddings` remain unavailable and are reported as such by `/v1/capabilities`.

The default server binds to loopback. Browser CORS responses no longer use a wildcard: preflight is allowed only for `http://localhost:<server-port>`, `http://127.0.0.1:<server-port>`, or `[::1]` on that same port. Requests with other origins receive no cross-origin permission; preflight receives HTTP 403.

Example capability check:

```sh
curl --fail http://127.0.0.1:8080/v1/capabilities
```

For a live end-to-end check against a running server, run the dependency-free verifier from the repository root:

```sh
python3 scripts/verify_openai_api.py --base-url http://127.0.0.1:8080 > openai-api-verification.json
```

It checks health/model readiness, advertised capabilities, chat/completions/responses generation, schema-validated JSON generation, and rejection of an invalid schema and image input. The JSON report includes each HTTP status and latency; a failed check exits non-zero and includes the failure reason. It sends several small prompts to the configured model, so run it when inference is idle. A passing report is evidence for that exact local runtime/model configuration, not a claim that unsupported vision or image editing works.

The verifier also rejects leaked chat control tokens and checks source fidelity
with a small generic record, including an absent value that must remain null.
Correction receives the previous output as assistant context (at most 8192
characters) plus bounded schema diagnostics. Diagnostics describe constraints,
never generated values. The original instructions and sampling/cache options
remain in effect; the one retry uses temperature zero and is validated again.

`openai-contract.yml` runs model-free HTTP/schema/runtime regressions on Python
3.12 and 3.13 and retains JUnit evidence. These tests include 250 seeded schema
mutations. They do not prove real Metal inference; use the live verifier for
that separately. Existing native/cache hardware CI remains independent.

Example strict structured request:

```json
{
  "model": "local",
  "messages": [{"role": "user", "content": "Return a JSON confirmation."}],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "confirmation",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": false
      }
    }
  }
}
```
