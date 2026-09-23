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
  request's chunk touches, laid end to end (``W`` rows), with the source rows
  each window pools -- an absolute token row when the source position is in
  the chunk, a state-leaf slot when it is before it. Chunk and window counts
  differ per request, so both axes are ragged and their lengths symbolic.

There is no ``cumsum`` or ``repeat_interleave`` on the GPU (KERN-1095), so
prefix sums and segment ids are written as compare-and-reduce over ``[n, b]``
masks, which for serving batch sizes is nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, Dim, DimLike, StaticDim, TensorValue, ops

from .cache import arange, idiv, row_offsets, scalar


def host_scalar(value: TensorValue | DimLike) -> TensorValue:
    """``value`` as an int32 scalar on the host, as ``ops.range`` reads bounds.

    A device-resident tensor is transferred (a scalar per call); a symbolic
    dim is read off the shape; a static one is a constant.
    """
    if isinstance(value, TensorValue):
        if not value.device.is_cpu():
            value = ops.transfer_to(value, DeviceRef.CPU())
        return ops.cast(value, DType.int32)
    if isinstance(value, str):
        raise TypeError(f"a dim name ({value!r}) has no value; pass the count")
    if isinstance(value, StaticDim):
        value = value.dim
    elif isinstance(value, Dim):
        return ops.cast(TensorValue(value), DType.int32)
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"unsupported range bound {value!r}")
    return ops.constant(int(value), DType.int32, DeviceRef.CPU())


def arange_to(
    stop: TensorValue | DimLike, out_dim: DimLike, device: DeviceRef
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
    device = counts.device
    b = counts.shape[0]
    rows = arange_to(b + 1, b + 1, device)
    cols = arange_to(b, b, device)
    mask = ops.reshape(cols, [1, b]) < ops.reshape(rows, [b + 1, 1])
    contrib = ops.where(mask, ops.reshape(counts, [1, b]), scalar(0, device))
    return ops.squeeze(ops.sum(contrib, axis=-1), axis=-1)


def segment_ids(
    offsets: TensorValue,
    total: DimLike,
    device: DeviceRef,
    stop: TensorValue | None = None,
) -> tuple[TensorValue, TensorValue]:
    """Which segment each of ``total`` rows falls in, and the row numbers.

    Args:
        offsets: ``[b + 1]`` int32 segment boundaries; row ``r`` belongs to
            segment ``i`` when ``offsets[i] <= r < offsets[i + 1]``.
        total: The row count ``offsets[b]`` as a graph dim, or the name of
            the symbolic dim to give it when it is only known at run time.
        stop: The row count as a tensor when ``total`` is a name.
    """
    b = offsets.shape[0] - 1
    idx = arange_to(total if stop is None else stop, total, device)
    ends = ops.reshape(
        ops.gather(offsets, arange_to(b, b, device) + 1, axis=0), [1, b]
    )
    ids = ops.squeeze(
        ops.sum(
            ops.cast(ops.reshape(idx, [total, 1]) >= ends, DType.int32),
            axis=-1,
        ),
        axis=-1,
    )
    return ids, idx


@dataclass
class RaggedRows:
    """Per-token bookkeeping of one ragged batch."""

    offsets: TensorValue
    """``[b + 1]`` uint32 row offsets, as the ragged kernels take them."""
    starts: TensorValue
    """``[b]`` int32 tokens already cached per request (``cache_lengths``)."""
    lengths: TensorValue
    """``[b]`` int32 tokens of each request in this batch."""
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

    @property
    def batch(self) -> Dim:
        return self.lengths.shape[0]

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
    ) -> RaggedRows:
        """Build from the graph's ``input_row_offsets`` and ``cache_lengths``.

        ``starts`` carries the manager's symbolic batch dim; it is rebound to
        the offsets' so the two can be combined. ``None`` is a fresh batch at
        position 0.
        """
        device = offsets.device
        off = ops.cast(offsets, DType.int32)
        b = off.shape[0] - 1
        i = arange_to(b, b, device)
        lengths = ops.gather(off, i + 1, axis=0) - ops.gather(off, i, axis=0)
        if starts is None:
            starts = ops.broadcast_to(scalar(0, device), [b])
        else:
            starts = ops.rebind(ops.cast(starts, DType.int32), [b])
        bid, index = segment_ids(off, total, device)
        local = index - ops.gather(off, bid, axis=0)
        positions = ops.gather(starts, bid, axis=0) + local
        total_dim = index.shape[0]
        return cls(
            offsets=ops.cast(offsets, DType.uint32),
            starts=starts,
            lengths=lengths,
            bid=bid,
            index=index,
            local=local,
            positions=positions,
            total=total_dim,
            device=device,
        )

    @classmethod
    def uniform(
        cls,
        batch: int,
        seq_len: int,
        device: DeviceRef,
        starts: TensorValue | None = None,
    ) -> RaggedRows:
        """A padded ``[batch, seq_len]`` chunk, flattened row-major."""
        return cls.from_offsets(
            row_offsets(batch, seq_len, device), batch * seq_len, starts
        )


@dataclass
class WindowRows:
    """The candidate windows of one compression ratio over a ragged batch.

    Request ``r``'s chunk of ``s`` tokens at ``P`` touches ``ceil(s / ratio)``
    windows starting at ``A = P - P % ratio``; all requests' windows are laid
    end to end (``W`` rows, ``woff`` the per-request offsets). Window ``w``
    pools ``coff * ratio`` source positions -- its own ``ratio`` and, when
    overlapping, its predecessor's -- and each source is either a row of this
    chunk (``fresh_row``, an absolute token row) or a slot of the state leaf
    (``cache_slot``), told apart by ``from_cache``; ``present`` is false
    before position 0. Dead entries are clamped onto live ones and masked.
    """

    ratio: int
    coff: int
    base: TensorValue
    """``[b]`` int32 entries closed before each request's chunk, ``P // ratio``."""
    n_new: TensorValue
    """``[b]`` int32 windows per request."""
    woff: TensorValue
    """``[b + 1]`` int32 window offsets."""
    bid: TensorValue
    """``[W]`` int32 request of each window."""
    start: TensorValue
    """``[W]`` int32 first position of each window."""
    present: TensorValue
    """``[W, coff * ratio]`` bool."""
    from_cache: TensorValue
    """``[W, coff * ratio]`` bool."""
    fresh_row: TensorValue
    """``[W, coff * ratio]`` int32 token rows (clamped into the request)."""
    cache_slot: TensorValue
    """``[W, coff * ratio]`` int32 state-leaf slots (clamped to live ones)."""
    total: Dim
    """``W``."""

    @classmethod
    def build(cls, rows: RaggedRows, ratio: int, coff: int) -> WindowRows:
        device = rows.device
        base = idiv(rows.starts, ratio)
        aligned = base * ratio
        n_new = idiv(rows.lengths + scalar(ratio - 1, device), ratio)
        woff = count_offsets(n_new)
        bid, wid = segment_ids(
            woff,
            f"windows_r{ratio}",
            device,
            stop=ops.reshape(ops.max(woff), []),
        )
        total = wid.shape[0]
        start = ops.gather(aligned, bid, axis=0) + (
            wid - ops.gather(woff, bid, axis=0)
        ) * scalar(ratio, device)

        n = coff * ratio
        rel = ops.reshape(arange(n, device) - (coff - 1) * ratio, [1, n])
        pos = ops.reshape(start, [total, 1]) + rel
        p = ops.reshape(ops.gather(rows.starts, bid, axis=0), [total, 1])
        length = ops.reshape(ops.gather(rows.lengths, bid, axis=0), [total, 1])
        first_row = ops.reshape(
            ops.gather(rows.offsets_i32, bid, axis=0), [total, 1]
        )
        zero = scalar(0, device)
        one = scalar(1, device)
        fresh_row = first_row + ops.min(
            ops.max(pos - p, zero), ops.max(length - one, zero)
        )
        cache_slot = ops.min(ops.max(pos, zero), ops.max(p - one, zero))
        return cls(
            ratio=ratio,
            coff=coff,
            base=base,
            n_new=n_new,
            woff=woff,
            bid=bid,
            start=start,
            present=pos >= zero,
            from_cache=pos < p,
            fresh_row=fresh_row,
            cache_slot=cache_slot,
            total=total,
        )
