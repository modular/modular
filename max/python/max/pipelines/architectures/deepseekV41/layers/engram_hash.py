# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Host-side constants for the Engram n-gram hash.

The prime bucket sizes and hash multipliers must be reproduced exactly, with
deterministic integer arithmetic and seeded draws, so that hashed n-grams
land on the same table rows the checkpoint was trained with.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

# Sentinel for a token that takes no part in an n-gram (an image span, or a
# position before the sequence start).
DEAD_ID = -1


def is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin primality test for ``n < 2**32``.

    The witnesses 2, 7 and 61 make the test exact over that range.
    """
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    """The smallest prime above ``start`` not already in ``seen_primes``.

    ``seen_primes`` must be one set threaded across every ``(layer, n-gram
    order, head)`` in :class:`EngramLayout`'s order, or the primes will not
    match the checkpoint.
    """
    candidate = start + 1
    while not is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def compute_hash_multipliers(
    layer_ids: tuple[int, ...],
    max_ngram_size: int,
    compressed_vocab_size: int,
) -> npt.NDArray[np.int64]:
    """One odd, overflow-safe multiplier per ``(layer, shift)``.

    Seeds ``numpy.random.default_rng(10007 * layer_id)`` per layer and
    draws ``2r + 1`` with ``r`` uniform in ``[0, bound)``, bounded so
    ``token_id * multiplier`` cannot overflow int64.

    Returns:
        ``[len(layer_ids), max_ngram_size]`` int64.
    """
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(
            low=0,
            high=multiplier_bound,
            size=(max_ngram_size,),
            dtype=np.int64,
        )
        rows.append(values * 2 + 1)
    return np.stack(rows)


class EngramLayout:
    """Bucket layout of the n-gram hash tables.

    A position hashes to ``max_ngram_size - 1`` n-grams (2-gram..max), each
    over ``n_heads`` heads; every ``(layer, order, head)`` owns a distinct
    prime-sized bucket range.
    """

    primes: npt.NDArray[np.int64]
    """``[n_layers, (max_ngram_size - 1) * n_heads]`` int64 bucket sizes.

    One prime per ``(layer, order, head)``, flattened over ``(order, head)``
    in that iteration order.
    """

    offsets: npt.NDArray[np.int64]
    """``[n_layers, (max_ngram_size - 1) * n_heads]`` int64 bucket offsets.

    The cumulative row offset of each bucket within its layer's table.
    """

    n_hash_cols: int
    """``(max_ngram_size - 1) * n_heads``."""

    def __init__(
        self,
        *,
        layer_ids: tuple[int, ...],
        num_embeddings: tuple[int, ...],
        max_ngram_size: int,
        n_heads: int,
        head_dim: int,
        compressed_vocab_size: int,
        pad_token_id: int,
        vocab_size: int,
    ) -> None:
        self.layer_ids = layer_ids
        self.num_embeddings = num_embeddings
        self.max_ngram_size = max_ngram_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.compressed_vocab_size = compressed_vocab_size
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        if len(layer_ids) != len(num_embeddings):
            raise ValueError(
                "engram_layer_ids and engram_num_embeddings must have the "
                f"same length, got {len(layer_ids)} and {len(num_embeddings)}"
            )

        primes: list[tuple[tuple[int, ...], ...]] = []
        seen: set[int] = set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes: list[int] = []
                current = vocab_size - 1
                for _ in range(n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        self.primes_nested = tuple(primes)
        self.n_hash_cols = (max_ngram_size - 1) * n_heads
        flat = [
            [p for per_ngram in layer for p in per_ngram] for layer in primes
        ]
        self.primes: npt.NDArray[np.int64] = np.array(flat, dtype=np.int64)
        offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
        self.offsets: npt.NDArray[np.int64] = np.array(offsets, dtype=np.int64)

    @classmethod
    def from_config(cls, config: Any) -> EngramLayout | None:
        """Builds a layout from a config exposing the ``engram_*`` fields.

        Returns:
            The layout, or ``None`` if ``engram_layer_ids`` is empty or
            unset.
        """
        layer_ids = getattr(config, "engram_layer_ids", None)
        if not layer_ids:
            return None
        return cls(
            layer_ids=tuple(layer_ids),
            num_embeddings=tuple(config.engram_num_embeddings),
            max_ngram_size=config.engram_max_ngram_size,
            n_heads=config.engram_n_heads,
            head_dim=config.engram_head_dim,
            compressed_vocab_size=config.engram_compressed_vocab_size,
            pad_token_id=config.engram_pad_token_id,
            vocab_size=config.engram_vocab_size,
        )


def engram_lookback_token_ids(
    histories: Sequence[npt.ArrayLike],
    chunk_starts: Sequence[int],
    depth: int,
) -> npt.NDArray[np.int64]:
    """Each request's raw token ids just before its chunk start, host side.

    Column ``j`` of row ``i`` is ``histories[i][chunk_starts[i] - 1 - j]``,
    or ``-1`` before the sequence start.

    Args:
        histories: Per request, at least the first ``chunk_starts[i]`` raw
            token ids of its sequence (prompt plus any generated tokens);
            longer is fine, only that prefix is read.
        chunk_starts: Per request, the absolute position of the first token
            in this step's chunk (its KV cache length).
        depth: ``max_ngram_size - 1``, the lookback width.

    Returns:
        ``[len(histories), depth]`` int64.
    """
    if len(histories) != len(chunk_starts):
        raise ValueError(
            f"got {len(histories)} histories but {len(chunk_starts)} "
            "chunk starts"
        )
    out = np.full((len(histories), depth), -1, dtype=np.int64)
    for i, (history, start) in enumerate(
        zip(histories, chunk_starts, strict=True)
    ):
        tokens = np.asarray(history)
        if start > tokens.shape[0]:
            raise ValueError(
                f"request {i}: chunk start {start} is past its "
                f"{tokens.shape[0]}-token history"
            )
        n = min(depth, start)
        if n:
            # Positions start-1, start-2, ..., start-n.
            out[i, :n] = tokens[start - n : start][::-1]
    return out
