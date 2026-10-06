"""Bounded synthetic proofs against an owned native MLX HTTP runtime."""

import hashlib
import json
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:19191"
LIMIT = 2 * 1024 * 1024


def require(condition, name):
    if not condition:
        raise RuntimeError("Native integration proof failed: " + name)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def validate_completion(data, budget):
    try:
        require(isinstance(data["id"], str) and bool(data["id"]), "nonempty completion identity")
        require(data["object"] == "chat.completion", "completion object")
        choices = data["choices"]
        require(len(choices) == 1 and choices[0]["index"] == 0, "single native choice")
        choice = choices[0]
        require(choice["message"]["role"] == "assistant", "assistant role")
        text = choice["message"]["content"]
        require(isinstance(text, str), "text content")
        finish = choice["finish_reason"]
        require(finish in ("stop", "length"), "native finish reason")
        usage = data["usage"]
        prompt, completion, total = (usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
        require(integer(prompt, 1) and integer(completion) and integer(total), "integer usage")
        require(completion <= budget and total == prompt + completion, "logical usage sum")
        require(finish != "length" or completion == budget, "truthful budget finish")
        meta = data["mlx_flash_compress"]
        cached, processed = meta["cached_prompt_tokens"], meta["processed_prompt_tokens"]
        require(
            meta["native_generation_metadata"] is True and meta["usage_source"] == "exact_mlx_lm_generation",
            "native metadata",
        )
        require(integer(cached) and integer(processed, 1) and cached + processed == prompt, "cache prompt accounting")
        return {
            "text": text,
            "finish_reason": finish,
            "usage": usage,
            "cache": {"cached_prompt_tokens": cached, "processed_prompt_tokens": processed},
        }
    except (KeyError, TypeError, IndexError) as error:
        raise RuntimeError("Native integration proof failed: malformed completion") from error


def parse_sse(body, budget):
    require(isinstance(body, str) and len(body.encode()) <= LIMIT, "bounded SSE")
    body = body.replace("\r\n", "\n")
    require(body.endswith("\n\n"), "terminated SSE frame")
    text, done, final, identity = "", False, None, None
    for block in body[:-2].split("\n\n"):
        require(not done and block.startswith("data: ") and "\n" not in block, "SSE framing")
        raw = block[6:]
        if raw == "[DONE]":
            require(final is not None, "finish before DONE")
            done = True
            continue
        try:
            event = json.loads(raw)
            require(event["object"] == "chat.completion.chunk", "chunk object")
            require(isinstance(event["id"], str) and bool(event["id"]), "nonempty stream identity")
            if identity is None:
                identity = event["id"]
            require(event["id"] == identity and final is None, "stable chunk identity and ordering")
            choices = event["choices"]
            require(len(choices) == 1 and choices[0]["index"] == 0, "single stream choice")
            choice = choices[0]
            delta = choice["delta"]
            require(isinstance(delta, dict), "stream delta")
            require(delta.get("role", "assistant") == "assistant", "stream role")
            content = delta.get("content", "")
            require(isinstance(content, str), "stream text")
            text += content
            if choice["finish_reason"] is not None:
                require(not delta, "empty terminal delta")
                final = dict(
                    event,
                    object="chat.completion",
                    choices=[
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": choice["finish_reason"],
                        }
                    ],
                )
        except (ValueError, KeyError, TypeError, IndexError) as error:
            raise RuntimeError("Native integration proof failed: malformed SSE") from error
    require(done and final is not None, "SSE DONE and final usage")
    return validate_completion(final, budget)


def validate_status(state):
    try:
        cache = state["prompt_cache"]
        for key in ("entries", "memory_bytes", "max_entries", "max_bytes", "reused_tokens", "processed_tokens"):
            require(integer(cache[key]), "cache status integers")
        require(cache["enabled"] is True, "enabled native cache")
        require(
            cache["entries"] <= cache["max_entries"] and cache["memory_bytes"] <= cache["max_bytes"],
            "bounded native cache",
        )
        for key in ("requests", "tokens_generated"):
            require(integer(state["stats"][key]), "completed counters")
        return cache
    except (KeyError, TypeError) as error:
        raise RuntimeError("Native integration proof failed: malformed status") from error


def assert_rejected_unchanged(before, after):
    validate_status(before)
    validate_status(after)
    for section, keys in (
        ("stats", ("requests", "tokens_generated")),
        ("prompt_cache", ("reused_tokens", "processed_tokens", "entries", "memory_bytes", "hits", "misses")),
    ):
        for key in keys:
            require(before[section].get(key) == after[section].get(key), "rejection before generation: " + key)


def proof_payload(thinking, scope, budget=32):
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": "Synthetic API probe. Reply exactly QWEN_ROLE_READY."}],
        "temperature": 0,
        "max_tokens": budget,
        "stream": False,
        "cache_scope": scope,
    }
    if thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": thinking}
    return payload


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def proof_record(name, payload, result):
    return {
        "check": name,
        "passed": True,
        "input_sha256": digest(payload),
        "response_sha256": digest(result),
        "usage": result["usage"],
        "cache": result["cache"],
        "finish_reason": result["finish_reason"],
    }


def exchange(opener, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with opener.open(req, timeout=600) as response:
            body = response.read(LIMIT + 1)
            require(len(body) <= LIMIT, "bounded HTTP response")
            return response.status, body.decode(), response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as error:
        # Error bodies may echo input; never retain them.
        return error.code, "", ""


def run_native_proofs(opener, request, capabilities, model_family, profiles, checks, model_size):
    require(capabilities.get("model_family") == model_family, "capabilities match cached family")
    scope = "native-proof-" + model_size
    profile = profiles[0]

    def status():
        state = request(BASE + "/status")
        validate_status(state)
        return state

    def check(name, action):
        item = {"check": name, "model_size": model_size, "passed": False}
        checks.append(item)
        detail = action()
        if detail:
            item.update(detail)
        item["passed"] = True

    def generate(name, payload, streamed=False):
        before = status()
        code, body, content_type = exchange(opener, "/v1/chat/completions", payload)
        require(code == 200, name + " HTTP success")
        require(("text/event-stream" if streamed else "application/json") in content_type, name + " content type")
        try:
            result = (
                parse_sse(body, payload["max_tokens"])
                if streamed
                else validate_completion(json.loads(body), payload["max_tokens"])
            )
        except ValueError as error:
            raise RuntimeError("Native integration proof failed: malformed JSON") from error
        after = status()
        require(after["stats"]["requests"] - before["stats"]["requests"] == 1, "one completed request")
        require(
            after["stats"]["tokens_generated"] - before["stats"]["tokens_generated"]
            == result["usage"]["completion_tokens"],
            "completed token accounting",
        )
        for counter, key in (
            ("reused_tokens", "cached_prompt_tokens"),
            ("processed_tokens", "processed_prompt_tokens"),
        ):
            require(
                after["prompt_cache"][counter] - before["prompt_cache"][counter] == result["cache"][key],
                "completed cache accounting",
            )
        result["status_cache"] = {
            key: after["prompt_cache"][key] for key in ("entries", "memory_bytes", "max_entries", "max_bytes")
        }
        return result

    values = {}

    def completion(name, payload, cached, parity=None, streamed=False):
        def action():
            result = generate(name, payload, streamed)
            require(
                result["cache"]["cached_prompt_tokens"] > 0 if cached else result["cache"]["cached_prompt_tokens"] == 0,
                name + " cache isolation",
            )
            if cached:
                require(result["cache"]["processed_prompt_tokens"] == 1, name + " full exact-prefix reuse")
            if parity:
                prior = values[parity]
                require(
                    all(result[k] == prior[k] for k in ("text", "finish_reason", "usage")),
                    name + " deterministic parity",
                )
            values[name] = result
            return dict(proof_record(name, payload, result), status_cache=result["status_cache"])

        check(name, action)

    payload = proof_payload(profile, scope)
    completion("native.cold", payload, False)
    completion("native.warm_exact_prefix", payload, True, "native.cold")
    completion("native.fresh_scope", proof_payload(profile, scope + "-fresh"), False, "native.cold")
    streamed = dict(payload, stream=True, stream_options={"include_usage": True})
    completion("native.sse_json_parity", streamed, True, "native.cold", True)
    completion("native.one_token_budget", proof_payload(profile, scope, 1), True)
    if len(profiles) > 1:
        completion("native.thinking_profile_isolation", proof_payload(profiles[1], scope, 1), False)
    else:
        checks.append(
            {
                "check": "native.thinking_profile_isolation",
                "model_size": model_size,
                "status": "not_applicable",
                "reason": "native family has no toggle",
            }
        )
    for name, invalid in (("sampler", {"temperature": -1}), ("scope", {"cache_scope": ""})):

        def rejection(invalid=invalid):
            before = status()
            bad = dict(payload, **invalid)
            code, _, _ = exchange(opener, "/v1/chat/completions", bad)
            require(code == 400, "invalid request rejected")
            assert_rejected_unchanged(before, status())
            return {"input_sha256": digest(bad), "http_status": code}

        check("native.invalid_" + name + "_before_generation", rejection)

    def release():
        request(BASE + "/release")
        cache = validate_status(request(BASE + "/status"))
        require(cache["entries"] == cache["memory_bytes"] == 0, "release clears native cache")
        return {"cache": {key: cache[key] for key in ("entries", "memory_bytes", "max_entries", "max_bytes")}}

    check("native.release_cache_clear", release)
