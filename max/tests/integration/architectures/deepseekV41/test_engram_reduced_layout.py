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
"""Guards the Engram lookup adapter's workspace sizing and table shapes.

The kernels' sort op addresses a fixed production bucket-counter workspace
(BUCKETS 733, 2932 int32) whatever the config's tables hold, so a reduced
layout like the sliced-checkpoint fixture's ([482648, 488250] rows) must reserve the same
counters; a layout past the kernels' static per-layer capacities must be
refused; and a nonempty but too-short table must be refused at graph build,
before the hash can read rows past the allocation. CPU only: the layout,
workspace and shapes are host-side arithmetic.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.graph.dim import Dim
from max.pipelines.architectures.deepseekV41.layers.engram_hash import (
    EngramLayout,
)
from max.pipelines.architectures.deepseekV41.layers.engram_lookup import (
    EngramLookup,
    EngramTokens,
)

PRODUCTION_VOCAB = 16000057
"""The production layout's ``engram_vocab_size``: the primes it generates
are the kernels' comptime ``PRIMES``."""

PRODUCTION_TABLE_ROWS = (384006168, 384016682)
"""The kernels' comptime table capacities, per layer: the production
``TABLE_ROWS`` (Kernels/lib/engram/engram_hash_gather.mojo)."""

PRODUCTION_BUCKETS = (max(PRODUCTION_TABLE_ROWS) >> 19) + 1


def _layout(vocab_size: int) -> EngramLayout:
    """Builds a self-consistent layout for ``vocab_size``.

    ``EngramLookup`` requires ``num_embeddings`` to equal the generated
    primes' row sums, so derive them with a first pass.

    Args:
        vocab_size: The layout's ``engram_vocab_size``.

    Returns:
        The layout, with ``num_embeddings`` taken from its own primes.
    """

    def build(num_embeddings: tuple[int, ...]) -> EngramLayout:
        return EngramLayout(
            layer_ids=(1, 14),
            num_embeddings=num_embeddings,
            max_ngram_size=4,
            n_heads=8,
            head_dim=256,
            compressed_vocab_size=99092,
            pad_token_id=2,
            vocab_size=vocab_size,
        )

    rows = tuple(int(r) for r in build((0, 0)).primes.sum(axis=1))
    return build(rows)


def _lookup(layout: EngramLayout) -> EngramLookup:
    token_map = (np.arange(4096, dtype=np.int64) * 7) % 3000
    return EngramLookup(layout, token_map)


def _token_types(device: DeviceRef) -> list[TensorType]:
    return [
        TensorType(DType.int32, ["T"], device),
        TensorType(DType.int64, ["T"], device),
        TensorType(DType.int32, ["R1"], device),
        TensorType(DType.bool, ["T"], device),
        TensorType(DType.int32, ["R", 3], device),
        TensorType(DType.bool, ["R", 3], device),
    ]


def _table_types(rows: Sequence[int], device: DeviceRef) -> list[TensorType]:
    out = []
    for r in rows:
        out += [
            TensorType(DType.float8_e4m3fn, [r, 256], device),
            TensorType(DType.float8_e8m0fnu, [r, 8], device),
        ]
    return out


def test_reduced_layout_reserves_production_counters() -> None:
    """The sliced-checkpoint fixture's row totals still size the work at production.

    The layout-derived sizing the adapter used before would have reserved
    one bucket per layer (4 counters), and the op would have rejected the
    workspace at execute time.
    """
    layout = _layout(20001)
    rows = [int(r) for r in layout.primes.sum(axis=1)]
    assert rows == [482648, 488250]
    assert (max(rows) >> 19) + 1 == 1
    lookup = _lookup(layout)
    assert lookup._buckets == PRODUCTION_BUCKETS
    work = lookup._work_type(Dim(1024), lookup.n_hash_cols, DeviceRef.GPU(0))
    # The trailing +1 is the gather's claim-counter word
    # (``_work_type``'s own formula, shared with the V2 pipeline model).
    assert [int(d) for d in work.shape] == [
        4 * lookup.n_hash_cols * 1024 + 4 * PRODUCTION_BUCKETS + 1
    ]


def test_production_layout_sizes_the_same() -> None:
    """A drift guard: the production vocab generates the kernels' primes,
    and the layout-derived and fixed derivations agree at capacity."""
    layout = _layout(PRODUCTION_VOCAB)
    rows = [int(r) for r in layout.primes.sum(axis=1)]
    assert rows == list(PRODUCTION_TABLE_ROWS)
    lookup = _lookup(layout)
    assert lookup._buckets == PRODUCTION_BUCKETS
    assert lookup._buckets == (max(rows) >> 19) + 1


def test_over_capacity_layout_is_refused() -> None:
    """A layout past the kernels' static capacities fails at build time,
    without materializing anything."""
    layout = _layout(17000000)
    rows = [int(r) for r in layout.primes.sum(axis=1)]
    assert max(rows) > max(PRODUCTION_TABLE_ROWS)
    with pytest.raises(ValueError, match="production table capacit"):
        _lookup(layout)


def test_short_table_is_refused_before_execution() -> None:
    """A nonempty but too-short table fails at graph build, not execution.

    The hash would otherwise select rows past the allocation: the op can't
    check the runtime primes against the tables without a synchronization,
    so the adapter checks the static shapes where the layout is known.
    """
    lookup = _lookup(_layout(20001))
    gpu = DeviceRef.GPU(0)
    short = [rows - 24 for rows in lookup.table_rows]
    types = _token_types(gpu) + _table_types(short, DeviceRef.CPU())
    with Graph("engram_short_tables", input_types=types) as g:
        x = [v.tensor for v in g.inputs]
        tokens = EngramTokens(
            ids=x[0],
            positions=x[1],
            query_start_loc=x[2],
            dead_mask=x[3],
            lookback=x[4],
            lookback_dead=x[5],
        )
        with pytest.raises(ValueError, match="Engram table 0 must be"):
            lookup.full(tokens, x[6:], gpu)
