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
"""CPU tests for the Engram host hash machinery.

The trunk reuses the registered pipeline's layout/primality/multiplier
recipes (:mod:`...deepseekV41.layers.engram_hash`) unchanged, so those
are checked against independent implementations (trial division;
directly seeded numpy draws). The n-gram hash oracle -- the CPU
reference the GPU hash-gather tests compare against -- lives here: it
is an independent NumPy transcription of the semantics, pinned by
hand-derived blocked/lookback/dead cases.
"""

from __future__ import annotations

import numpy as np
from max.pipelines.architectures.deepseekV41.layers.engram_hash import (
    DEAD_ID,
    EngramLayout,
    compute_hash_multipliers,
    engram_lookback_token_ids,
    find_next_prime,
    is_prime,
)

PAD_TOKEN_ID = 2


def _is_prime_trial_division(n: int) -> bool:
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def ngram_hash_reference(
    token_map: np.ndarray,
    ids: np.ndarray,
    positions: np.ndarray,
    query_start_loc: np.ndarray,
    dead_mask: np.ndarray | None,
    lookback: np.ndarray | None,
    lookback_dead: np.ndarray | None,
    layout: EngramLayout,
) -> np.ndarray:
    """The bit-exact NumPy hash oracle: per-position n-gram row indices.

    Semantics:
    * sources come from the current chunk where ``positions[t] - shift``
      is at or after the chunk's first position, else from the lookback
      window (column ``chunk_start - 1 - pos``, newest first);
    * a dead source, or a position before the sequence start, blocks the
      n-gram at and after it: blocked positions hash the pad id, and
      blocking is sticky;
    * an unknown lookback id (-1 with a live dead flag) hashes as the pad
      id WITHOUT masking older tokens;
    * the rolling XOR over multiplied sources produces one column per
      (order, head): ``rolling ^= source * multipliers[:, shift]``, then
      ``column = rolling % primes[:, order] + offsets[:, order]``.
    """
    token_map = np.asarray(token_map)
    ids = np.asarray(ids)
    positions = np.asarray(positions)
    query_start_loc = np.asarray(query_start_loc)
    dead = (
        np.zeros(len(ids), dtype=bool)
        if dead_mask is None
        else np.asarray(dead_mask).astype(bool)
    )
    look = (
        None
        if lookback is None
        else np.asarray(lookback).reshape(len(query_start_loc) - 1, -1)
    )
    look_dead = None if lookback_dead is None else np.asarray(lookback_dead)
    pad = int(token_map[layout.pad_token_id]) if len(token_map) else 0
    if len(token_map) == 0:
        pad = DEAD_ID

    multipliers = compute_hash_multipliers(
        layout.layer_ids, layout.max_ngram_size, layout.compressed_vocab_size
    )
    n_layers = len(layout.layer_ids)
    result = np.full(
        (len(ids), n_layers, layout.n_hash_cols), -1, dtype=np.int64
    )
    for t in range(len(ids)):
        if dead[t] or not 0 <= int(ids[t]) < len(token_map):
            continue
        request = int(np.searchsorted(query_start_loc, t, side="right") - 1)
        start = int(positions[int(query_start_loc[request])])
        blocked = False
        sources: list[int] = []
        for back in range(layout.max_ngram_size):
            pos = int(positions[t]) - back
            if pos >= start:
                index = t - back
                token = int(ids[index]) if index >= 0 else DEAD_ID
                source = (
                    int(token_map[token])
                    if 0 <= token < len(token_map)
                    else DEAD_ID
                )
                if index >= 0 and dead[index]:
                    source = DEAD_ID
            elif look is not None:
                col = start - 1 - pos
                token = (
                    int(look[request, col])
                    if 0 <= col < look.shape[1]
                    else DEAD_ID
                )
                source = (
                    int(token_map[token])
                    if 0 <= token < len(token_map)
                    else pad
                )
                if 0 <= token < len(token_map) and look_dead is not None:
                    if look_dead[request, col]:
                        source = DEAD_ID
            else:
                source = DEAD_ID
            blocked |= pos < 0 or source == DEAD_ID
            sources.append(pad if blocked else int(source))
        for layer in range(n_layers):
            rolling = np.int64(0)
            for back in range(layout.max_ngram_size):
                rolling = rolling ^ np.int64(sources[back]) * np.int64(
                    multipliers[layer, back]
                )
                if back:
                    cols = slice(
                        (back - 1) * layout.n_heads, back * layout.n_heads
                    )
                    result[t, layer, cols] = (
                        rolling % layout.primes[layer, cols]
                        + layout.offsets[layer, cols]
                    )
    return result


def test_primes_and_layout() -> None:
    """Bucket primes are distinct primes drawn above the vocab bound."""
    layout = EngramLayout(
        layer_ids=(1, 5),
        num_embeddings=(0, 0),  # only the length is validated
        max_ngram_size=4,
        n_heads=8,
        head_dim=256,
        compressed_vocab_size=99092,
        pad_token_id=2,
        vocab_size=20_000,
    )
    flat = layout.primes.reshape(-1)
    assert len(set(flat.tolist())) == len(flat)  # never collide
    assert (flat > 20_000 - 1).all()
    for p in flat:
        assert _is_prime_trial_division(int(p))
    # Offsets are the cumulative row starts: bucket c starts at the sum
    # of every earlier bucket's size within its layer.
    np.testing.assert_array_equal(
        layout.offsets[:, 1:], np.cumsum(layout.primes[:, :-1], axis=1)
    )
    assert layout.n_hash_cols == 24


def test_multipliers_recipe() -> None:
    """Seeded draws: 2r+1 with r uniform in [0, int64max // vocab // 2)."""
    layer_ids = (1, 14)
    got = compute_hash_multipliers(layer_ids, 4, 99092)
    assert got.shape == (2, 4)
    assert got.dtype == np.int64
    bound = max(1, (np.iinfo(np.int64).max // 99092) // 2)
    for row, layer_id in enumerate(layer_ids):
        rng = np.random.default_rng(10007 * layer_id)
        expected = rng.integers(0, bound, size=(4,), dtype=np.int64) * 2 + 1
        np.testing.assert_array_equal(got[row], expected)
    # Odd and overflow-safe by construction.
    assert (got % 2 == 1).all()
    assert (got * 99092 <= np.iinfo(np.int64).max).all()


def test_is_prime_and_find_next_prime() -> None:
    assert is_prime(2) and is_prime(7) and is_prime(1_000_003)
    assert not is_prime(1) and not is_prime(0) and not is_prime(561)
    seen: set[int] = set()
    first = find_next_prime(10, seen)
    assert first == 11
    seen.add(first)
    assert find_next_prime(10, seen) == 13  # skips the seen prime


def test_lookback_windows() -> None:
    """Column j reads position start-1-j; -1 before the sequence start."""
    histories = [np.array([5, 6, 7, 8]), np.array([9])]
    starts = [2, 0]
    lookback = engram_lookback_token_ids(histories, starts, 3)
    np.testing.assert_array_equal(
        lookback,
        np.array([[6, 5, -1], [-1, -1, -1]]),
    )


def _small_layout() -> EngramLayout:
    return EngramLayout(
        layer_ids=(0, 1),
        num_embeddings=(0, 0),
        max_ngram_size=3,
        n_heads=2,
        head_dim=8,
        compressed_vocab_size=100,
        pad_token_id=PAD_TOKEN_ID,
        vocab_size=1_000,
    )


def _expected_rows(layout: EngramLayout, sources: list[int]) -> np.ndarray:
    """The rolling XOR hash of one position's source list, both layers.

    Column block ``back`` (the (back+1)-gram) accumulates the sources up
    to ``back`` and lands in columns ``[(back-1) * n_heads, back * n_heads)``.
    """
    multipliers = compute_hash_multipliers(
        layout.layer_ids, layout.max_ngram_size, layout.compressed_vocab_size
    )
    out = np.zeros((len(layout.layer_ids), layout.n_hash_cols), np.int64)
    for layer in range(len(layout.layer_ids)):
        rolling: int = 0
        for back in range(layout.max_ngram_size):
            rolling ^= int(sources[back]) * int(multipliers[layer, back])
            if back:
                for head in range(layout.n_heads):
                    col = (back - 1) * layout.n_heads + head
                    out[layer, col] = (
                        rolling % layout.primes[layer, col]
                        + layout.offsets[layer, col]
                    )
    return out


def test_hash_reference_unblocked_positions() -> None:
    """In-chunk sources hash directly; pre-start positions block stickily."""
    layout = _small_layout()
    token_map = np.arange(20, dtype=np.int32)  # identity: pad compressed = 2
    pad = int(token_map[PAD_TOKEN_ID])
    ids = np.array([3, 4, 5, 6], np.int32)
    positions = np.array([0, 1, 2, 3], np.int64)
    qsl = np.array([0, 4], np.int32)
    got = ngram_hash_reference(
        token_map, ids, positions, qsl, None, None, None, layout
    )
    assert got.shape == (4, 2, 4)
    # t = 0: the unigram source is the token itself; backs 1 and 2 are
    # before the sequence start, so blocking turns them into the pad id.
    np.testing.assert_array_equal(got[0], _expected_rows(layout, [3, pad, pad]))
    np.testing.assert_array_equal(got[1], _expected_rows(layout, [4, 3, pad]))
    np.testing.assert_array_equal(got[2], _expected_rows(layout, [5, 4, 3]))
    np.testing.assert_array_equal(got[3], _expected_rows(layout, [6, 5, 4]))


def test_hash_reference_dead_current_token() -> None:
    """A dead current token gets an all -1 row; downstream rows block."""
    layout = _small_layout()
    token_map = np.arange(20, dtype=np.int32)
    pad = int(token_map[PAD_TOKEN_ID])
    ids = np.array([3, 4, 5, 6], np.int32)
    positions = np.array([0, 1, 2, 3], np.int64)
    qsl = np.array([0, 4], np.int32)
    dead = np.array([0, 0, 1, 0], np.uint8)
    got = ngram_hash_reference(
        token_map, ids, positions, qsl, dead, None, None, layout
    )
    assert (got[2] == -1).all()
    # Position 3's 3-gram reaches the dead token at back 1: the source is
    # DEAD there, blocking from that back onward (pad from there on).
    np.testing.assert_array_equal(got[3], _expected_rows(layout, [6, pad, pad]))


def test_hash_reference_lookback_semantics() -> None:
    """Lookback feeds the hash; an unknown id hashes as pad without blocking.

    ``-1`` with a live dead flag is *unknown* (pad id, older sources kept);
    the same column with a dead flag *blocks* at and after it.
    """
    layout = _small_layout()
    token_map = np.arange(20, dtype=np.int32)
    pad = int(token_map[PAD_TOKEN_ID])
    ids = np.array([5, 6], np.int32)
    positions = np.array([2, 3], np.int64)  # chunk starts at position 2
    qsl = np.array([0, 2], np.int32)

    def run(lookback: np.ndarray, lookback_dead: np.ndarray) -> np.ndarray:
        return ngram_hash_reference(
            token_map,
            ids,
            positions,
            qsl,
            None,
            lookback,
            lookback_dead,
            layout,
        )

    # Both lookback sources known and live: backs 1 and 2 read columns
    # 0 and 1 (start-1-pos), newest first.
    got = run(np.array([[7, 8]], np.int32), np.array([[0, 0]], np.uint8))
    np.testing.assert_array_equal(got[0], _expected_rows(layout, [5, 7, 8]))

    # Unknown id (-1, live): hashes as the PAD id, does NOT mask older.
    got = run(np.array([[-1, 8]], np.int32), np.array([[0, 0]], np.uint8))
    np.testing.assert_array_equal(got[0], _expected_rows(layout, [5, pad, 8]))

    # Dead flag on column 0: the source is DEAD, blocking at and after it.
    got = run(np.array([[7, 8]], np.int32), np.array([[1, 0]], np.uint8))
    np.testing.assert_array_equal(got[0], _expected_rows(layout, [5, pad, pad]))
