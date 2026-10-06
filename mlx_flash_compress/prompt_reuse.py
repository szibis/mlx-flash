"""Bounded exact-token-prefix snapshots of native MLX-LM cache objects.

Never crops a snapshot: recurrent and rotating caches cannot in general be
restored at arbitrary offsets. Callers serialize access with the inference lock.
"""

import copy
from collections import OrderedDict


class PromptReuse:
    def __init__(self, max_bytes=512 * 1024 * 1024, max_entries=8):
        if type(max_bytes) is not int or max_bytes < 0 or type(max_entries) is not int or max_entries < 1:
            raise ValueError("Cache limits must be nonnegative bytes and positive entries")
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self._entries = OrderedDict()
        self._bytes = 0
        self.hits = self.misses = self.reused_tokens = self.processed_tokens = self.evictions = 0

    def lookup(self, scope, tokens):
        best = None
        for key in self._entries:
            namespace, prefix = key
            if namespace == scope and len(prefix) < len(tokens) and tuple(tokens[: len(prefix)]) == prefix:
                if best is None or len(prefix) > len(best[1]):
                    best = key
        reused = len(best[1]) if best else 0
        if best:
            self.hits += 1
            self._entries.move_to_end(best)
            return copy.deepcopy(self._entries[best][0]), tokens[reused:]
        self.misses += 1
        return None, tokens

    def store(self, scope, tokens, native_cache):
        if not tokens or self.max_bytes == 0:
            return
        size = sum(item.nbytes for item in native_cache)
        if size > self.max_bytes:
            return
        key = (scope, tuple(tokens))
        if key in self._entries:
            self._bytes -= self._entries.pop(key)[1]
        # Admit before copying; there is no oversized pending snapshot during
        # decode and no second copy when promoting a prefill snapshot.
        while self._bytes + size > self.max_bytes or len(self._entries) >= self.max_entries:
            _, (_, removed_bytes) = self._entries.popitem(last=False)
            self._bytes -= removed_bytes
            self.evictions += 1
        self._entries[key] = (copy.deepcopy(native_cache), size)
        self._bytes += size

    def clear(self):
        self.evictions += len(self._entries)
        self._entries.clear()
        self._bytes = 0

    def record_completed(self, reused, processed):
        self.reused_tokens += reused
        self.processed_tokens += processed

    def stats(self):
        return {
            "enabled": self.max_bytes > 0,
            "entries": len(self._entries),
            "memory_bytes": self._bytes,
            "max_bytes": self.max_bytes,
            "max_entries": self.max_entries,
            "hits": self.hits,
            "misses": self.misses,
            "reused_tokens": self.reused_tokens,
            "processed_tokens": self.processed_tokens,
            "evictions": self.evictions,
        }
