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

"""Ragged-batch bookkeeping, computed in-graph from the row offsets.

A serving batch is ``T`` tokens of ``b`` requests laid end to end, described
by ``input_row_offsets`` (``[b + 1]``) and the manager's per-request
``cache_lengths``. Every per-token layer runs on the tokens as one
``[1, T, ...]`` sequence and never looks at the split; attention does, through
two tables built here:

* :class:`RaggedRows` -- per token: which request it belongs to, its index
  within that request's chunk, and its absolute position.
* :class:`WindowRows` -- per compression ratio: the candidate windows every
  request's chunk touches, laid end to end, as per-request offsets.

The window count ``W`` is a sum of per-request ceilings, so it is data, and
reading it back would sync every forward and rule out device graph capture.
The cached path never needs it on the host: one op pools every window, sized
by a bound derived from shapes
(:func:`~max.nn.kernels.compress_windows_ragged`). Only the cache-less
reference path, whose lengths are fixed at graph build, lays the windows out
row by row (:class:`WindowTables`).

Segment ids are a prefix sum over segment ends rather than a
``repeat_interleave``, whose GPU kernel is scalar-repeats only (a
tensor-repeats kernel is still outstanding, AGENT-491).

A :class:`RaggedRows` is a :mod:`max.tree` node, so a block that takes one
can be built as a subgraph. Two compiler limits shape what crosses the call:

- An op sized only by a symbolic dim reads no operand, so inside a body the
  compiler hoists it into a constant subgraph that runs at load with the dim
  unbound. The :class:`WindowRows` are therefore built once per batch in the
  calling graph and passed in.
- A subgraph signature takes no dim expression. ``starts`` keeps the cache's
  own batch dim and is never rebound to the offsets' ``b``, and no table
  sized ``b`` (the per-request lengths or window counts) crosses a call.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, Dim, DimLike, StaticDim, TensorValue, ops

from .cache import arange, row_offsets, scalar


def window_coff(ratio: int) -> int:
    """Windows a compressed entry pools: two when overlapping (ratio 4)."""
    return 2 if ratio == 4 else 1


def host_scalar(value: DimLike) -> TensorValue:
    """``value`` as an int32 scalar on the host, as ``ops.range`` reads bounds.

    Only shapes: a symbolic dim is read off the shape and a static one is a
    constant. A bound held in a device tensor would be read back to the host
    every forward, which no device graph capture can record.
    """
    if isinstance(value, str):
        raise TypeError(f"a dim name ({value!r}) has no value; pass the count")
    if isinstance(value, StaticDim):
        value = value.dim
    elif isinstance(value, Dim):
        return ops.cast(TensorValue(value), DType.int32)
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"unsupported range bound {value!r}")
    return ops.constant(int(value), DType.int32, DeviceRef.CPU())


def window_count(lengths: Sequence[int] | np.ndarray, ratio: int) -> int:
    """``W`` of :class:`WindowRows` for chunks of ``lengths`` tokens."""
    return int(sum(-(-int(n) // ratio) for n in lengths))


def arange_to(
    stop: DimLike, out_dim: DimLike, device: DeviceRef
) -> TensorValue:
    """``[0, stop)`` int32 on ``device`` with length ``out_dim``."""
    return ops.range(
        0,
        host_scalar(stop),
        1,
        out_dim=out_dim,
        device=device,
        dtype=DType.int32,
    )


def count_offsets(counts: TensorValue) -> TensorValue:
    """Exclusive prefix sums: ``[b]`` int32 counts -> ``[b + 1]`` int32."""
    zero = ops.reshape(scalar(0, counts.device), [1])
    return ops.concat([zero, ops.cumsum(counts, axis=0)], axis=0)


def segment_ids(
    offsets: TensorValue,
    total: DimLike,
    device: DeviceRef,
) -> tuple[TensorValue, TensorValue]:
    """Which segment each of ``total`` rows falls in, and the row numbers.

    Args:
        offsets: ``[b + 1]`` int32 segment boundaries; row ``r`` belongs to
            segment ``i`` when ``offsets[i] <= r < offsets[i + 1]``.
        total: The row count ``offsets[b]`` as a graph dim.
    """
    b = offsets.shape[0] - 1
    # A row's segment is the number of segment ends at or before it. Ends at
    # or past ``total`` land in an extra sink row; zero-length segments
    # repeat an end, hence the add.
    pos = arange_to(Dim(total) + 1, Dim(total) + 1, device)
    ends = ops.gather(offsets, arange_to(b, b, device) + 1, axis=0)
    ends = ops.min(ends, pos[-1])
    hist = ops.scatter_add(
        pos * 0, ops.broadcast_to(scalar(1, device), [b]), ends, axis=0
    )
    ids = ops.rebind(ops.cumsum(hist, axis=0)[:-1], [total])
    return ids, ops.rebind(pos[:-1], [total])


@dataclass
class RaggedRows:
    """Per-token bookkeeping of one ragged batch."""

    offsets: TensorValue
    """``[b + 1]`` uint32 row offsets, as the ragged kernels take them."""
    starts: TensorValue
    """``[b]`` int32 tokens already cached per request (``cache_lengths``),
    in the cache's own batch dim."""
    bid: TensorValue
    """``[T]`` int32 request index of each token."""
    index: TensorValue
    """``[T]`` int32 row number of each token, ``0 .. T - 1``."""
    local: TensorValue
    """``[T]`` int32 index of each token within its request's chunk."""
    positions: TensorValue
    """``[T]`` int32 absolute position of each token."""
    total: Dim
    """``T``."""
    device: DeviceRef
    windows: dict[int, WindowRows] = field(default_factory=dict)
    """Per compression ratio, the batch's :class:`WindowRows`."""

    def __tree_flatten__(self) -> tuple[dict[str, Any], None]:
        children = {
            "offsets": self.offsets,
            "starts": self.starts,
            "bid": self.bid,
            "index": self.index,
            "local": self.local,
            "positions": self.positions,
            "windows": self.windows,
        }
        return children, None

    @classmethod
    def __tree_unflatten__(
        cls, _: None, children: Mapping[str, Any]
    ) -> RaggedRows:
        index: TensorValue = children["index"]
        offsets: TensorValue = children["offsets"]
        return cls(
            offsets=offsets,
            starts=children["starts"],
            bid=children["bid"],
            index=index,
            local=children["local"],
            positions=children["positions"],
            total=index.shape[0],
            device=offsets.device,
            windows=dict(children["windows"]),
        )

    @property
    def offsets_i32(self) -> TensorValue:
        return ops.cast(self.offsets, DType.int32)

    @property
    def end(self) -> TensorValue:
        """``T`` as a device scalar (the last offset)."""
        return ops.reshape(ops.max(self.offsets_i32), [])

    @classmethod
    def from_offsets(
        cls,
        offsets: TensorValue,
        total: DimLike,
        starts: TensorValue | None = None,
        windows: Mapping[int, DimLike | None] | None = None,
    ) -> RaggedRows:
        """Build from the graph's ``input_row_offsets`` and ``cache_lengths``.

        ``starts`` keeps the manager's symbolic batch dim; it is only ever
        gathered by request, never combined elementwise with a table sized by
        the offsets (see the module docstring). ``None`` is a fresh batch at
        position 0. ``windows`` maps each compression ratio the batch's
        attention uses to the batch's :func:`window_count`, which only the
        cache-less path needs (``None`` otherwise); each ratio's
        :class:`WindowRows` is built here.
        """
        device = offsets.device
        off = ops.cast(offsets, DType.int32)
        b = off.shape[0] - 1
        i = arange_to(b, b, device)
        lengths = ops.gather(off, i + 1, axis=0) - ops.gather(off, i, axis=0)
        if starts is None:
            starts = ops.broadcast_to(scalar(0, device), [b])
        else:
            starts = ops.cast(starts, DType.int32)
        bid, index = segment_ids(off, total, device)
        local = index - ops.gather(off, bid, axis=0)
        positions = ops.gather(starts, bid, axis=0) + local
        total_dim = index.shape[0]
        rows = cls(
            offsets=ops.cast(offsets, DType.uint32),
            starts=starts,
            bid=bid,
            index=index,
            local=local,
            positions=positions,
            total=total_dim,
            device=device,
        )
        rows.windows = {
            r: WindowRows.build(rows, lengths, r, w)
            for r, w in (windows or {}).items()
        }
        return rows

    @classmethod
    def uniform(
        cls,
        batch: int,
        seq_len: int,
        device: DeviceRef,
        starts: TensorValue | None = None,
        ratios: Sequence[int] = (),
    ) -> RaggedRows:
        """A padded ``[batch, seq_len]`` chunk, flattened row-major."""
        return cls.from_offsets(
            row_offsets(batch, seq_len, device),
            batch * seq_len,
            starts,
            {r: window_count([seq_len] * batch, r) for r in ratios},
        )


@dataclass
class WindowTables:
    """The source rows of every window, laid out for the cache-less path.

    That path pools with graph ops over a table per window, so the window
    count ``W`` has to be known at graph build; the cached path pools in
    place (:func:`~max.nn.kernels.compress_windows_ragged`) and builds none.
    """

    start: TensorValue
    """``[W]`` int32 first position of each window."""
    present: TensorValue
    """``[W, coff * ratio]`` bool, false before position 0."""
    fresh_row: TensorValue
    """``[W, coff * ratio]`` int32 token rows (clamped into the request)."""

    @property
    def total(self) -> Dim:
        """``W``."""
        return self.start.shape[0]


@dataclass
class WindowRows:
    """The candidate windows of one compression ratio over a ragged batch.

    Request ``r``'s chunk of ``s`` tokens at ``P`` touches ``ceil(s / ratio)``
    windows starting at ``A = P - P % ratio``; all requests' windows are laid
    end to end, ``woff`` the per-request offsets. Window ``w`` pools
    ``coff * ratio`` source positions -- its own ``ratio`` and, when
    overlapping, its predecessor's.
    """

    ratio: int
    base: TensorValue
    """``[b]`` int32 entries closed before each request's chunk, ``P // ratio``."""
    woff: TensorValue
    """``[b + 1]`` int32 window offsets."""
    tables: WindowTables | None = None
    """The windows row by row, when :meth:`build` was given their count."""

    @property
    def coff(self) -> int:
        return window_coff(self.ratio)

    def __tree_flatten__(self) -> tuple[dict[str, TensorValue], int]:
        children = {"base": self.base, "woff": self.woff}
        if self.tables is not None:
            children["start"] = self.tables.start
            children["present"] = self.tables.present
            children["fresh_row"] = self.tables.fresh_row
        return children, self.ratio

    @classmethod
    def __tree_unflatten__(
        cls, ratio: int, children: Mapping[str, TensorValue]
    ) -> WindowRows:
        tables = None
        if "start" in children:
            tables = WindowTables(
                start=children["start"],
                present=children["present"],
                fresh_row=children["fresh_row"],
            )
        return cls(
            ratio=ratio,
            base=children["base"],
            woff=children["woff"],
            tables=tables,
        )

    @classmethod
    def build(
        cls,
        rows: RaggedRows,
        lengths: TensorValue,
        ratio: int,
        count: DimLike | None = None,
    ) -> WindowRows:
        """``ratio``'s windows over ``rows``; ``lengths`` is each request's
        ``[b]`` int32 chunk length and ``count``, given for the cache-less
        path's :class:`WindowTables`, is ``W``."""
        device = rows.device
        base = rows.starts // ratio
        n_new = (lengths + scalar(ratio - 1, device)) // ratio
        woff = count_offsets(n_new)
        if count is None:
            return cls(ratio=ratio, base=base, woff=woff)

        coff = window_coff(ratio)
        bid, wid = segment_ids(woff, count, device)
        total = wid.shape[0]
        start = ops.gather(base * ratio, bid, axis=0) + (
            wid - ops.gather(woff, bid, axis=0)
        ) * scalar(ratio, device)

        n = coff * ratio
        rel = ops.reshape(arange(n, device) - (coff - 1) * ratio, [1, n])
        pos = ops.reshape(start, [total, 1]) + rel
        p = ops.reshape(ops.gather(rows.starts, bid, axis=0), [total, 1])
        length = ops.reshape(ops.gather(lengths, bid, axis=0), [total, 1])
        first_row = ops.reshape(
            ops.gather(rows.offsets_i32, bid, axis=0), [total, 1]
        )
        zero = scalar(0, device)
        one = scalar(1, device)
        fresh_row = first_row + ops.min(
            ops.max(pos - p, zero), ops.max(length - one, zero)
        )
        return cls(
            ratio=ratio,
            base=base,
            woff=woff,
            tables=WindowTables(
                start=start, present=pos >= zero, fresh_row=fresh_row
            ),
        )
