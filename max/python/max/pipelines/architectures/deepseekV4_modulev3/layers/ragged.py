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
"""Ragged-batch bookkeeping on ModuleV3, computed in-graph from the offsets.

Segment ids are a prefix sum over segment ends rather than a
``repeat_interleave``, and every range bound is read off a shape, never off a
device tensor, so no forward reads data back to the host. The integer body
runs as one ``F.functional`` op: it sizes ranges by the symbolic token dim
through shape-to-tensor ops that have no eager counterpart, and a graph op
cannot return a dataclass, so the body returns a tuple and the records are
rebuilt here over :class:`Tensor`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from max import functional as F
from max.dtype import DType
from max.graph import DeviceRef, Dim, DimLike, StaticDim, TensorValue, ops
from max.tensor import Tensor
from max.tensor.sharding import (
    AxisAssignment,
    TensorLayout,
    replicated_rows,
)

__all__ = ["RaggedRows", "WindowRows", "window_coff"]


# Extracted from deepseekV4/layers/ragged.py and deepseekV4/layers/cache.py
# at dfc8109fc0c: the graph body of ``RaggedRows.from_offsets`` with
# ``starts`` given and no window count, all the cached path builds.
def window_coff(ratio: int) -> int:
    """Windows a compressed entry pools: two when overlapping (ratio 4)."""
    return 2 if ratio == 4 else 1


def _scalar(value: int, device: DeviceRef) -> TensorValue:
    return ops.constant(value, DType.int32, device)


def _host_scalar(value: DimLike) -> TensorValue:
    """``value`` as an int32 host scalar, read off a shape, never off data."""
    if isinstance(value, str):
        raise TypeError(f"a dim name ({value!r}) has no value; pass the count")
    if isinstance(value, StaticDim):
        value = value.dim
    elif isinstance(value, Dim):
        return ops.cast(TensorValue(value), DType.int32)
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"unsupported range bound {value!r}")
    return ops.constant(int(value), DType.int32, DeviceRef.CPU())


def _arange_to(
    stop: DimLike, out_dim: DimLike, device: DeviceRef
) -> TensorValue:
    return ops.range(
        0,
        _host_scalar(stop),
        1,
        out_dim=out_dim,
        device=device,
        dtype=DType.int32,
    )


def _count_offsets(counts: TensorValue) -> TensorValue:
    """Exclusive prefix sums: ``[b]`` int32 counts -> ``[b + 1]`` int32."""
    zero = ops.reshape(_scalar(0, counts.device), [1])
    return ops.concat([zero, ops.cumsum(counts, axis=0)], axis=0)


def _segment_ids(
    offsets: TensorValue, total: DimLike, device: DeviceRef
) -> tuple[TensorValue, TensorValue]:
    """Which segment each of ``total`` rows falls in, and the row numbers."""
    b = offsets.shape[0] - 1
    # A row's segment is the number of segment ends at or before it. Ends at
    # or past ``total`` land in an extra sink row; zero-length segments
    # repeat an end, hence the add.
    pos = _arange_to(Dim(total) + 1, Dim(total) + 1, device)
    ends = ops.gather(offsets, _arange_to(b, b, device) + 1, axis=0)
    ends = ops.min(ends, pos[-1])
    hist = ops.scatter_add(
        pos * 0, ops.broadcast_to(_scalar(1, device), [b]), ends, axis=0
    )
    ids = ops.rebind(ops.cumsum(hist, axis=0)[:-1], [total])
    return ids, ops.rebind(pos[:-1], [total])


@dataclass
class WindowRows:
    """The candidate windows of one compression ratio over a ragged batch.

    Request ``r``'s chunk of ``s`` tokens at ``P`` touches ``ceil(s / ratio)``
    windows; all requests' windows are laid end to end.
    """

    ratio: int
    base: Tensor
    """``[b]`` int32 entries closed before each request's chunk, ``P // ratio``."""
    woff: Tensor
    """``[b + 1]`` int32 window offsets."""

    @property
    def coff(self) -> int:
        """Windows a compressed entry pools: two when overlapping (ratio 4)."""
        return window_coff(self.ratio)


@dataclass
class RaggedRows:
    """Per-token bookkeeping of one ragged batch."""

    offsets: Tensor
    """``[b + 1]`` uint32 row offsets, as the ragged kernels take them."""
    starts: Tensor
    """``[b]`` int32 ``cache_lengths``, in the cache's own batch dim."""
    bid: Tensor
    """``[T]`` int32 request index of each token."""
    index: Tensor
    """``[T]`` int32 row number of each token."""
    local: Tensor
    """``[T]`` int32 index of each token within its request's chunk."""
    positions: Tensor
    """``[T]`` int32 absolute position of each token."""
    windows: dict[int, WindowRows] = field(default_factory=dict)
    """Per compression ratio, the batch's :class:`WindowRows`."""

    @property
    def total(self) -> DimLike:
        """``T``."""
        return self.index.shape[0]

    @classmethod
    def build(
        cls,
        offsets: Tensor,
        cache_lengths: Tensor,
        total: DimLike,
        ratios: Sequence[int],
    ) -> RaggedRows:
        """From ``input_row_offsets`` and the cache's ``cache_lengths``.

        ``total`` is ``T`` as a graph dim; ``ratios`` the compression ratios
        the batch's attention uses, each getting its :class:`WindowRows`.
        """
        ratios = tuple(ratios)
        out = _ragged_rows(offsets, cache_lengths, total, ratios)
        offsets_u32, starts, bid, index, local, positions = out[:6]
        windows = {
            r: WindowRows(ratio=r, base=out[6 + 2 * i], woff=out[7 + 2 * i])
            for i, r in enumerate(ratios)
        }
        return cls(
            offsets=offsets_u32,
            starts=starts,
            bid=bid,
            index=index,
            local=local,
            positions=positions,
            windows=windows,
        )


def _ragged_rows_graph(
    offsets: TensorValue,
    cache_lengths: TensorValue,
    total: DimLike,
    ratios: tuple[int, ...],
) -> tuple[TensorValue, ...]:
    device = offsets.device
    off = ops.cast(offsets, DType.int32)
    b = off.shape[0] - 1
    i = _arange_to(b, b, device)
    lengths = ops.gather(off, i + 1, axis=0) - ops.gather(off, i, axis=0)
    starts = ops.cast(cache_lengths, DType.int32)
    bid, index = _segment_ids(off, total, device)
    local = index - ops.gather(off, bid, axis=0)
    positions = ops.gather(starts, bid, axis=0) + local
    out = [
        ops.cast(offsets, DType.uint32),
        starts,
        bid,
        index,
        local,
        positions,
    ]
    for r in ratios:
        n_new = (lengths + _scalar(r - 1, device)) // r
        out += [starts // r, _count_offsets(n_new)]
    return tuple(out)


def _ragged_rows_rule(
    offsets: TensorLayout,
    cache_lengths: TensorLayout,
    total: DimLike,
    ratios: tuple[int, ...],
) -> list[AxisAssignment]:
    """Whole on every device: each device's attention reads every row."""
    return replicated_rows(
        offsets, cache_lengths, output_count=6 + 2 * len(ratios)
    )


_ragged_rows = F.functional(_ragged_rows_graph, rule=_ragged_rows_rule)
