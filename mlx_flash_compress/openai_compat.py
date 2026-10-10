"""Bounded OpenAI API contract helpers shared by HTTP handlers and tests."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

MAX_SCHEMA_BYTES = 64 * 1024
MAX_SCHEMA_DEPTH = 32
MAX_SCHEMA_NODES = 10_000
MAX_STRUCTURED_OUTPUT_BYTES = 256 * 1024
MAX_JSON_DEPTH = 64
_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SCHEMA_KEYS = {
    "$schema",
    "$id",
    "title",
    "description",
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
    "const",
    "anyOf",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "pattern",
    "minItems",
    "maxItems",
}


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return not isinstance(value, float) or math.isfinite(value)


class APIRequestError(ValueError):
    """A safe, client-actionable error in an OpenAI-compatible request."""

    def __init__(self, message: str, *, status: int = 400, code: str = "invalid_request_error"):
        self.status = status
        self.code = code
        super().__init__(message)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _check_json_depth(raw: bytes | str) -> None:
    data = raw if isinstance(raw, bytes) else raw.encode("utf-8")
    depth = 0
    in_string = escaped = False
    for byte in data:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                in_string = False
            continue
        if byte == 34:
            in_string = True
        elif byte in (123, 91):
            depth += 1
            if depth > MAX_JSON_DEPTH:
                raise ValueError("JSON nesting limit exceeded")
        elif byte in (125, 93):
            depth = max(depth - 1, 0)


def loads_json(raw: bytes | str) -> Any:
    """Load bounded API JSON without duplicate keys, NaN/Infinity or deep-stack leaks."""
    try:
        _check_json_depth(raw)
        return json.loads(
            raw,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON number")),
            object_pairs_hook=_unique_object,
        )
    except (json.JSONDecodeError, ValueError, RecursionError):
        raise APIRequestError("Request body must be valid JSON without duplicate keys or non-finite numbers.") from None


def validate_chat_request(data: dict[str, Any]) -> None:
    supported = {
        "model",
        "messages",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "stream",
        "response_format",
        "chat_template_kwargs",
        "reasoning_effort",
        "top_p",
        "top_k",
        "stream_options",
        "cache_scope",
        "repetition_penalty",
    }
    unsupported = set(data) - supported
    if unsupported:
        raise APIRequestError("Unsupported chat completion fields: " + ", ".join(sorted(unsupported)[:5]))
    if "max_tokens" in data and "max_completion_tokens" in data:
        raise APIRequestError("Use only one of max_tokens or max_completion_tokens.")
    if "model" in data and (not isinstance(data["model"], str) or len(data["model"]) > 200):
        raise APIRequestError("model must be a string of at most 200 characters.")
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages or len(messages) > 2048:
        raise APIRequestError("messages must be an array containing 1–2048 messages.")
    allowed_roles = {"system", "developer", "user", "assistant"}
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) - {"role", "content", "name"}:
            raise APIRequestError(f"messages[{index}] has an unsupported shape.")
        if not isinstance(message.get("role"), str) or message["role"] not in allowed_roles:
            raise APIRequestError(f"messages[{index}].role is unsupported.")
        if not isinstance(message.get("content"), str):
            raise APIRequestError(
                "Image and non-text message content is not available in this server build.", code="unsupported_modality"
            )
        if "name" in message and (not isinstance(message["name"], str) or len(message["name"]) > 128):
            raise APIRequestError(f"messages[{index}].name must be a string of at most 128 characters.")
    max_tokens = data.get("max_completion_tokens", data.get("max_tokens", 256))
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 8192:
        raise APIRequestError("max_tokens must be an integer from 1 to 8192.")
    temperature = data.get("temperature", 0.7)
    if not _is_finite_number(temperature) or not 0 <= temperature <= 2:
        raise APIRequestError("temperature must be a finite number from 0 to 2.")
    if "stream" in data and not isinstance(data["stream"], bool):
        raise APIRequestError("stream must be a boolean.")
    if "reasoning_effort" in data and data["reasoning_effort"] != "none":
        raise APIRequestError("Only reasoning_effort=none is currently supported.")


@dataclass(frozen=True)
class ResponseFormat:
    kind: str
    schema: dict[str, Any] | None = None
    name: str | None = None
    strict: bool = False


def api_capabilities() -> dict[str, Any]:
    """Return a machine-readable inventory; never imply image support via chat text."""
    return {
        "object": "mlx_flash.capabilities",
        "api_version": "1",
        "endpoints": {
            "chat_completions": {"supported": True, "streaming": True},
            "models": {"supported": True},
            "responses": {"supported": True, "streaming": False, "input": ["text", "simple_messages"]},
            "completions": {"supported": True, "streaming": False, "prompt": "string"},
            "embeddings": {"supported": False},
        },
        "structured_outputs": {
            "supported": True,
            "formats": ["json_object", "json_schema"],
            "enforcement": "validated_before_success",
            "schema_dialect": "https://json-schema.org/draft/2020-12/schema",
            "schema_limit_bytes": MAX_SCHEMA_BYTES,
            "output_limit_bytes": MAX_STRUCTURED_OUTPUT_BYTES,
            "streaming": False,
        },
        "vision": {"supported": False, "reason": "text_generation_adapter_only", "modalities": ["text"]},
        "image_operations": {"supported": False, "reason": "no_native_image_operation_adapter", "operations": []},
    }


def _check_schema_bounds(value: Any, *, depth: int = 0, count: list[int] | None = None) -> None:
    if count is None:
        count = [0]
    count[0] += 1
    if depth > MAX_SCHEMA_DEPTH or count[0] > MAX_SCHEMA_NODES:
        raise APIRequestError("JSON Schema exceeds the supported size or nesting limit.")
    if isinstance(value, dict):
        if "$ref" in value or "$dynamicRef" in value:
            raise APIRequestError("JSON Schema references are not supported.")
        unknown = set(value) - _SCHEMA_KEYS
        if unknown:
            raise APIRequestError("JSON Schema uses unsupported keywords: " + ", ".join(sorted(unknown)[:5]))
        if "type" in value:
            types = value["type"] if isinstance(value["type"], list) else [value["type"]]
            if not types or any(
                not isinstance(t, str) or t not in {"object", "array", "string", "integer", "number", "boolean", "null"}
                for t in types
            ):
                raise APIRequestError("JSON Schema contains an unsupported type.")
        for key in ("title", "description"):
            if key in value and (not isinstance(value[key], str) or len(value[key]) > 4096):
                raise APIRequestError(f"JSON Schema {key} must be text no longer than 4096 characters.")
        if "properties" in value:
            if not isinstance(value["properties"], dict) or not all(isinstance(k, str) for k in value["properties"]):
                raise APIRequestError("JSON Schema properties must be an object.")
            for child in value["properties"].values():
                if not isinstance(child, dict):
                    raise APIRequestError("Property schemas must be JSON objects.")
                _check_schema_bounds(child, depth=depth + 1, count=count)
        if "required" in value:
            required = value["required"]
            if (
                not isinstance(required, list)
                or any(not isinstance(k, str) for k in required)
                or len(set(required)) != len(required)
            ):
                raise APIRequestError("JSON Schema required must be a unique array of strings.")
        if "additionalProperties" in value and not isinstance(value["additionalProperties"], bool):
            raise APIRequestError("Schema additionalProperties must be boolean.")
        if "items" in value:
            if not isinstance(value["items"], dict):
                raise APIRequestError("Array item schemas must be JSON objects.")
            _check_schema_bounds(value["items"], depth=depth + 1, count=count)
        if "anyOf" in value:
            choices = value["anyOf"]
            if not isinstance(choices, list) or not choices:
                raise APIRequestError("JSON Schema anyOf must be a non-empty array.")
            for child in choices:
                if not isinstance(child, dict):
                    raise APIRequestError("anyOf entries must be JSON schema objects.")
                _check_schema_bounds(child, depth=depth + 1, count=count)
        if "enum" in value and (not isinstance(value["enum"], list) or not value["enum"]):
            raise APIRequestError("JSON Schema enum must be a non-empty array.")
        for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
            if key in value and not _is_finite_number(value[key]):
                raise APIRequestError(f"JSON Schema {key} must be a finite number.")
        for key in ("minLength", "maxLength", "minItems", "maxItems"):
            if key in value and (isinstance(value[key], bool) or not isinstance(value[key], int) or value[key] < 0):
                raise APIRequestError(f"JSON Schema {key} must be a non-negative integer.")
        if "pattern" in value:
            if not isinstance(value["pattern"], str) or len(value["pattern"]) > 512:
                raise APIRequestError("JSON Schema pattern must be a string of at most 512 characters.")
            repetitions = re.findall(r"(?<!\\)[*+?]|\{\d+(?:,\d*)?\}", value["pattern"])
            if len(repetitions) > 8 or re.search(
                r"\\[1-9]|\(\?|\([^)]*\|[^)]*\)(?:[+*]|\{)|\([^)]*(?:[+*]|\{\d+(?:,\d*)?\})[^)]*\)(?:[+*]|\{)",
                value["pattern"],
            ):
                raise APIRequestError("JSON Schema pattern uses a construct that could cause excessive matching time.")
            try:
                re.compile(value["pattern"])
            except re.error:
                raise APIRequestError("JSON Schema pattern is invalid.") from None
        if "required" in value and "properties" in value and not set(value["required"]).issubset(value["properties"]):
            raise APIRequestError("JSON Schema required names must exist in properties.")
        if value.get("$schema") not in (None, "https://json-schema.org/draft/2020-12/schema"):
            raise APIRequestError("Only JSON Schema Draft 2020-12 is supported.")
    elif isinstance(value, list):
        for child in value:
            _check_schema_bounds(child, depth=depth + 1, count=count)


def parse_response_format(value: Any) -> ResponseFormat | None:
    """Validate OpenAI chat `response_format`; reject unsupported forms early."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise APIRequestError("response_format must be an object.")

    kind = value.get("type")
    if kind == "json_object":
        if set(value) - {"type"}:
            raise APIRequestError("json_object response_format has unsupported fields.")
        return ResponseFormat(kind="json_object")
    if kind != "json_schema":
        raise APIRequestError("Unsupported response_format.type; use json_object or json_schema.")

    if set(value) - {"type", "json_schema"}:
        raise APIRequestError("json_schema response_format has unsupported fields.")
    descriptor = value.get("json_schema")
    if not isinstance(descriptor, dict) or set(descriptor) - {"name", "description", "strict", "schema"}:
        raise APIRequestError("json_schema must contain only name, description, strict, and schema.")
    name = descriptor.get("name")
    strict = descriptor.get("strict", False)
    schema = descriptor.get("schema")
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise APIRequestError("json_schema.name must be 1–64 letters, digits, underscores, or hyphens.")
    if "description" in descriptor and (
        not isinstance(descriptor["description"], str) or len(descriptor["description"]) > 4096
    ):
        raise APIRequestError("json_schema.description must be text no longer than 4096 characters.")
    if not isinstance(strict, bool):
        raise APIRequestError("json_schema.strict must be a boolean.")
    if not isinstance(schema, dict):
        raise APIRequestError("json_schema.schema must be a JSON Schema object.")
    try:
        encoded = json.dumps(schema, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError):
        raise APIRequestError("json_schema.schema must contain finite JSON values.") from None
    if len(encoded) > MAX_SCHEMA_BYTES:
        raise APIRequestError("JSON Schema exceeds the 64 KiB request limit.")
    _check_schema_definition(schema)
    return ResponseFormat(kind=kind, schema=schema, name=name, strict=strict)


def _check_schema_definition(schema: dict[str, Any]) -> None:
    _check_schema_bounds(schema)


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return _is_finite_number(value)
    if expected == "boolean":
        return isinstance(value, bool)
    return value is None


def _validate_value(value: Any, schema: dict[str, Any]) -> bool:
    """Validate the intentionally bounded JSON Schema subset supported by this API."""
    expected = schema.get("type")
    if expected is not None and not any(
        _matches_type(value, item) for item in (expected if isinstance(expected, list) else [expected])
    ):
        return False
    if "enum" in schema and not any(type(value) is type(item) and value == item for item in schema["enum"]):
        return False
    if "const" in schema and (type(value) is not type(schema["const"]) or value != schema["const"]):
        return False
    if "anyOf" in schema and not any(_validate_value(value, choice) for choice in schema["anyOf"]):
        return False
    if isinstance(value, dict):
        if any(name not in value for name in schema.get("required", [])):
            return False
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False and any(name not in properties for name in value):
            return False
        if any(name in properties and not _validate_value(child, properties[name]) for name, child in value.items()):
            return False
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", math.inf):
            return False
        if "items" in schema and any(not _validate_value(child, schema["items"]) for child in value):
            return False
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", math.inf):
            return False
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            return False
    if _is_finite_number(value):
        if isinstance(value, float) and not math.isfinite(value):
            return False
        if "minimum" in schema and value < schema["minimum"]:
            return False
        if "maximum" in schema and value > schema["maximum"]:
            return False
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            return False
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            return False
    return True


def _contains_non_finite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_contains_non_finite(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_non_finite(child) for child in value)
    return False


def _schema_diagnostics(value: Any, schema: dict[str, Any], path: str = "$", *, limit: int = 8) -> list[str]:
    """Bounded format feedback for one retry. Never echo generated values/keys."""
    if limit <= 0 or _validate_value(value, schema):
        return []
    expected = schema.get("type")
    types = expected if isinstance(expected, list) else [expected]
    if expected is not None and not any(_matches_type(value, item) for item in types):
        return [f"{path}: wrong JSON type"]
    if "anyOf" in schema and not any(_validate_value(value, choice) for choice in schema["anyOf"]):
        for choice in schema["anyOf"]:
            kinds = choice.get("type")
            if kinds is None or any(
                _matches_type(value, kind) for kind in (kinds if isinstance(kinds, list) else [kinds])
            ):
                return _schema_diagnostics(value, choice, path, limit=limit)
        return [f"{path}: no anyOf alternative matches"]
    errors = []
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if any(key not in value for key in schema.get("required", [])):
            errors.append(f"{path}: missing required properties")
        if schema.get("additionalProperties") is False and any(key not in properties for key in value):
            errors.append(f"{path}: additional properties are forbidden")
        # Paths come only from the submitted schema, not unexpected model keys.
        for key, child_schema in properties.items():
            if key in value and len(errors) < limit:
                child_path = path + "[" + json.dumps(key[:80], ensure_ascii=True) + "]"
                errors.extend(_schema_diagnostics(value[key], child_schema, child_path, limit=limit - len(errors)))
    elif isinstance(value, list) and "items" in schema:
        for index, child in enumerate(value):
            if len(errors) >= limit:
                break
            errors.extend(_schema_diagnostics(child, schema["items"], f"{path}[{index}]", limit=limit - len(errors)))
    if not errors:
        reason = (
            "string pattern/length constraint failed"
            if isinstance(value, str) and "pattern" in schema
            else "schema constraint failed"
        )
        errors.append(f"{path}: {reason}")
    return errors[:limit]


def validate_structured_output(content: str, response_format: ResponseFormat) -> Any:
    """Parse a model response and fail closed unless it satisfies the requested format."""
    if not isinstance(content, str):
        raise APIRequestError(
            "The model returned a non-text structured response.", status=502, code="invalid_model_output"
        )
    try:
        content_bytes = content.encode("utf-8")
    except UnicodeEncodeError:
        raise APIRequestError(
            "The model response contained invalid Unicode.", status=502, code="invalid_model_output"
        ) from None
    if len(content_bytes) > MAX_STRUCTURED_OUTPUT_BYTES:
        raise APIRequestError(
            "The model structured response exceeded the output limit.", status=502, code="invalid_model_output"
        )
    try:
        _check_json_depth(content)
        parsed = json.loads(
            content,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite number")),
            object_pairs_hook=_unique_object,
        )
    except (json.JSONDecodeError, ValueError, RecursionError):
        raise APIRequestError("The model did not return valid JSON.", status=502, code="invalid_model_output") from None

    if _contains_non_finite(parsed):
        raise APIRequestError("The model returned a non-finite JSON number.", status=502, code="invalid_model_output")
    if response_format.kind == "json_object" and not isinstance(parsed, dict):
        raise APIRequestError("The model did not return a JSON object.", status=502, code="invalid_model_output")
    if response_format.kind == "json_schema":
        try:
            matches = _validate_value(parsed, response_format.schema)
        except RecursionError:
            matches = False
        if not matches:
            details = "; ".join(_schema_diagnostics(parsed, response_format.schema))[:2048]
            raise APIRequestError(
                "The model response did not satisfy the requested JSON Schema. " + details,
                status=502,
                code="invalid_model_output",
            )
    return parsed
