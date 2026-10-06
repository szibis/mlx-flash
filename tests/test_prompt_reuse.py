"""Exact-prefix reuse must preserve native state and namespace isolation."""

from dataclasses import dataclass

try:
    from mlx_flash_compress.prompt_reuse import PromptReuse
except ModuleNotFoundError:
    PromptReuse = None


@dataclass
class State:
    values: list
    nbytes: int = 16


def test_exact_prefix_reuses_a_private_copy_and_keeps_one_input_token():
    assert PromptReuse is not None, "native prefix reuse is not implemented"
    cache = PromptReuse(max_bytes=64, max_entries=2)
    original = [State([1, 2])]
    cache.store(("model", "sonnet", "session"), [1, 2], original)
    original[0].values.append(99)
    restored, rest = cache.lookup(("model", "sonnet", "session"), [1, 2, 3])
    assert rest == [3]
    assert restored[0].values == [1, 2]
    restored[0].values.append(4)
    assert cache.lookup(("model", "sonnet", "session"), [1, 2, 5])[0][0].values == [1, 2]
    assert cache.lookup(("model", "sonnet", "session"), [1, 2]) == (None, [1, 2])


def test_different_model_profile_session_or_divergent_prefix_is_a_miss():
    assert PromptReuse is not None, "native prefix reuse is not implemented"
    cache = PromptReuse(max_bytes=64, max_entries=2)
    cache.store(("model", "sonnet", "a"), [1, 2], [State([1, 2])])
    for scope in [("model2", "sonnet", "a"), ("model", "opus", "a"), ("model", "sonnet", "b")]:
        assert cache.lookup(scope, [1, 2, 3]) == (None, [1, 2, 3])
    assert cache.lookup(("model", "sonnet", "a"), [1, 9, 3]) == (None, [1, 9, 3])


def test_byte_limit_lru_eviction_and_disabled_cache_are_truthful():
    assert PromptReuse is not None, "native prefix reuse is not implemented"
    cache = PromptReuse(max_bytes=32, max_entries=4)
    scope = ("m", "p", "s")
    cache.store(scope, [1], [State([1])])
    cache.store(scope, [2], [State([2])])
    cache.lookup(scope, [1, 9])
    cache.store(scope, [3], [State([3])])
    assert cache.lookup(scope, [2, 9])[0] is None
    assert cache.stats()["evictions"] == 1
    assert cache.stats()["memory_bytes"] == 32
    cache.store(scope, [4], [State([4], nbytes=100)])
    assert cache.stats()["memory_bytes"] == 32
    disabled = PromptReuse(max_bytes=0, max_entries=4)
    disabled.store(scope, [1], [State([1])])
    assert disabled.lookup(scope, [1, 9])[0] is None
    assert disabled.stats()["enabled"] is False


def test_reuse_accounting_counts_full_inputs_without_inventing_token_savings():
    assert PromptReuse is not None, "native prefix reuse is not implemented"
    cache = PromptReuse(max_bytes=64, max_entries=2)
    scope = ("m", "p", "s")
    cache.lookup(scope, [1, 2, 3])
    cache.record_completed(0, 3)
    cache.store(scope, [1, 2], [State([1, 2])])
    cache.lookup(scope, [1, 2, 4, 5])
    cache.record_completed(2, 2)
    assert cache.stats()["reused_tokens"] == 2
    assert cache.stats()["processed_tokens"] == 5
    assert cache.stats()["hits"] == 1
    assert cache.stats()["misses"] == 1
    cache.clear()
    assert cache.stats()["entries"] == 0
