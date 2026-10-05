# Request thinking profiles

The Chat Completions server accepts a tokenizer profile independently on each request:

```json
{
  "model": "local",
  "messages": [{"role": "user", "content": "Plan this change"}],
  "max_tokens": 8192,
  "stream": false,
  "chat_template_kwargs": {"enable_thinking": true}
}
```

Set `enable_thinking` to `false` for a non-thinking Qwen response. This lets a gateway map lightweight/coding/reasoning roles onto a small Qwen and a shared large Qwen without loading a separate copy of the large model for each effort profile. The tokenizer must support the selected option. This is a tokenizer instruction, not a hard reasoning-token budget or a model-quality guarantee.

Only a boolean `enable_thinking` is allowed. Unknown options, non-boolean values and explicit null are rejected with HTTP 400 before generation. Omitting the field preserves the tokenizer's existing defaults and legacy formatting fallback. When an explicit profile is requested, template failures are reported instead of silently replacing the template with joined text.

Profiles apply to both normal and buffered-SSE unbatched requests, including speculative paths that use the formatted prompt. They are scoped to the request and restored after failures. A per-model lock prevents overlapping unbatched operations on the shared native model/tokenizer. The profile is rejected in continuous-batching mode because background workers do not inherit the request's context. `/health` advertises supported profile options; this does not establish that a particular checkpoint has been loaded or tested.

The existing server still buffers its SSE generation and lacks verified request-level native cancellation. Output counters and finish reasons retain their existing limits. Tests for profile validation, scoped formatting, concurrency and recovery require no Metal device or weights; native checkpoint quality needs its own verification.
