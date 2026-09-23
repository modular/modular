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

"""One compressed stream: its state leaf, its zone leaf, and the windows that
close in the current batch.

Both the attention's compressor and the indexer's run this, each against its
own pair of leaves. Given a ragged batch (:class:`~.ragged.RaggedRows`) and
the windows its chunks touch (:class:`~.ragged.WindowRows`, one per ratio,
shared by the two streams of a layer), it

1. stores the tokens' raw ``wkv`` / ``wgate`` projections into the state leaf;
2. assembles every window's ``coff * ratio`` source rows, from the state leaf
   for positions before the request's ``P`` and from the fresh projections at
   or after it, and pools them into one candidate entry per window;
3. stores the candidates into the zone leaf at entry slots ``P // ratio ...``
   of their request;
4. hands back the candidates and where the closed entries live, so a reader
   can address the zone leaf by entry (:meth:`CompressedStream.entries`, the
   fused kernel) or gather a per-token candidate table
   (:attr:`CompressedStream.table`, the indexer's scoring and the cache-less
   reference path).

Why storing every candidate is exact: the windows that actually close inside
a chunk of ``s`` tokens number ``ceil(s / ratio)`` or one fewer. When one
fewer, the last candidate lands on entry slot ``T // ratio`` (``T = P + s``),
the window still open after the chunk. No reader reaches it -- a query at
position ``t`` sees entries below ``(t + 1) // ratio`` -- and it is overwritten
when the window does close. Its page is allocated: it is only reached when
``T % ratio != 0``, so the slot's first token ``(T // ratio) * ratio`` is below
``T``. Decode is the ``s == 1`` case of the same statement.

Per token the candidate axis is static: ``cap`` zone slots (``ceil(max_seq_len
/ ratio)``) then ``cap`` fresh windows, of which a request uses its first
``n_new``; ``valid`` masks the rest, and a candidate that is valid for a query
is always one of its own request's windows (``base + w < cutoff`` implies
``w < n_new``).
"""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import Dim, TensorValue, ops

from .cache import KEY, VALUE, CacheLeaf, arange, ceildiv, scalar
from .compressor import DeepseekV4Compressor
from .ragged import RaggedRows, WindowRows


@dataclass
class CompressedStream:
    """What a batch sees of one compressed stream.

    Candidates are numbered ``0 .. cap - 1`` for the zone slots and ``cap ..
    2 * cap - 1`` for a request's fresh windows; ``valid`` says which of them
    a query may see. Zone slot ``j`` is live below the request's ``base``;
    fresh candidate ``w`` is entry ``base + w``, already stored in the zone
    leaf, and row ``woff[request] + w`` of ``fresh``.
    """

    fresh: TensorValue
    """``[W, head_dim]`` candidate entries, all requests' windows end to end."""
    windows: WindowRows
    rows: RaggedRows
    cap: int
    """Zone slots a query may reach, ``ceil(max_seq_len / ratio)``."""
    zone_leaf: CacheLeaf | None
    layer: int

    @property
    def n_cand(self) -> int:
        """Static per-token candidate count: ``cap`` cached + ``cap`` fresh."""
        return 2 * self.cap

    @property
    def token_base(self) -> TensorValue:
        """``[T]`` int32 entries closed before each token's chunk."""
        return ops.gather(self.windows.base, self.rows.bid, axis=0)

    @property
    def table(self) -> TensorValue:
        """``[T, n_cand, head_dim]`` per-token candidate table, candidate order.

        Zone rows are gathered per token out of the leaf (a copy of the leaf
        per call, ``CacheLeaf.gather``); only the indexer's scoring and the
        cache-less reference path pay for it, and it scales with
        ``max_seq_len``: a bringup path, not a serving one.
        """
        rows = self.rows
        t = rows.total
        cap = self.cap
        device = rows.device
        head_dim = self.fresh.shape[1]
        zero = scalar(0, device)
        one = scalar(1, device)
        if self.zone_leaf is None:
            cached = ops.broadcast_to(
                ops.constant(0.0, self.fresh.dtype, device), [t, cap, head_dim]
            )
        else:
            # Slots at or past ``base`` are dead; clamp them onto the last
            # live one (or slot 0 when nothing has closed), ``valid`` hides
            # them.
            last = ops.reshape(ops.max(self.token_base - one, zero), [t, 1])
            slots = ops.min(
                ops.broadcast_to(
                    ops.reshape(arange(cap, device), [1, cap]), [t, cap]
                ),
                last,
            )
            cached = ops.cast(
                self.zone_leaf.gather(
                    self.layer, KEY, slots, lut_rows=rows.bid
                ),
                self.fresh.dtype,
            )
        woff = self.windows.woff
        first = ops.reshape(ops.gather(woff, rows.bid, axis=0), [t, 1])
        last_w = (
            ops.reshape(ops.gather(woff, rows.bid + one, axis=0), [t, 1]) - one
        )
        idx = ops.min(
            first + ops.reshape(arange(cap, device), [1, cap]), last_w
        )
        fresh = ops.gather(self.fresh, idx, axis=0)
        return ops.concat([cached, fresh], axis=1)

    def entries(self, candidates: TensorValue) -> TensorValue:
        """Candidate numbers -> zone-leaf entry indices, ``-1`` passing through.

        Args:
            candidates: ``[T, k]`` int32 candidate numbers, ``-1`` unused.
        """
        t = self.rows.total
        device = candidates.device
        cap = scalar(self.cap, device)
        base = ops.reshape(self.token_base, [t, 1])
        entry = ops.where(
            candidates < cap, candidates, base + (candidates - cap)
        )
        return ops.where(
            candidates >= scalar(0, device), entry, scalar(-1, device)
        )

    def fresh_rows(self, candidates: TensorValue) -> TensorValue:
        """Fresh candidate numbers -> rows of :attr:`fresh`, ``-1`` for the rest.

        The cache-less reference path uses this to address a gathered table;
        without a cache no zone slot is ever valid, so those map to ``-1``.
        """
        t = self.rows.total
        device = candidates.device
        cap = scalar(self.cap, device)
        first = ops.reshape(
            ops.gather(self.windows.woff, self.rows.bid, axis=0), [t, 1]
        )
        return ops.where(
            candidates >= cap, first + (candidates - cap), scalar(-1, device)
        )

    def valid(self, cutoff: TensorValue) -> TensorValue:
        """``[T, n_cand]`` bool: candidate closed before each query.

        Args:
            cutoff: ``[T]`` int32, ``(position + 1) // ratio`` per query.
        """
        t = self.rows.total
        device = cutoff.device
        cap = self.cap
        base = ops.reshape(self.token_base, [t, 1])
        cached = ops.reshape(arange(cap, device), [1, cap]) < base
        fresh = (
            base + ops.reshape(arange(cap, device), [1, cap])
        ) < ops.reshape(cutoff, [t, 1])
        return ops.concat([cached, fresh], axis=-1)


def compressed_stream(
    compressor: DeepseekV4Compressor,
    x32: TensorValue,
    rows: RaggedRows,
    windows: WindowRows,
    freqs_table: TensorValue,
    max_seq_len: int,
    state_leaf: CacheLeaf | None,
    zone_leaf: CacheLeaf | None,
    layer: int,
) -> CompressedStream:
    """Run one compressed stream over a ragged batch.

    Args:
        compressor: The stream's compressor (attention's or the indexer's).
        x32: ``[T, hidden]`` float32 block input.
        rows: The batch's token bookkeeping.
        windows: The candidate windows for ``compressor.compress_ratio``.
        freqs_table: The layer's full rotary table.
        max_seq_len: Caps how many closed entries a query can see.
        state_leaf: The stream's open-state leaf, or ``None`` to run without
            a cache (a fresh batch, nothing stored).
        zone_leaf: The stream's compressed-zone leaf, or ``None`` likewise.
        layer: Index of this layer within both leaves.
    """
    ratio = compressor.compress_ratio
    coff = compressor.coff
    proj = compressor.proj_dim
    assert windows.ratio == ratio and windows.coff == coff
    w = windows.total

    kv_proj, score_proj = compressor.projections(x32)
    if state_leaf is not None:
        state_leaf.store(layer, KEY, kv_proj, rows.offsets)
        state_leaf.store(layer, VALUE, score_proj, rows.offsets)

    # Rows at or after the request's ``P`` come from the chunk (the index is
    # clamped so a not-yet-closed window reads a real, finite row); rows
    # before it from the state leaf; ``present`` masks positions below 0.
    fresh_kv = ops.gather(kv_proj, windows.fresh_row, axis=0)
    fresh_score = ops.gather(score_proj, windows.fresh_row, axis=0)
    if state_leaf is not None:
        mask = ops.unsqueeze(windows.from_cache, -1)
        kv_rows = ops.where(
            mask,
            state_leaf.gather(
                layer, KEY, windows.cache_slot, lut_rows=windows.bid
            ),
            fresh_kv,
        )
        score_rows = ops.where(
            mask,
            state_leaf.gather(
                layer, VALUE, windows.cache_slot, lut_rows=windows.bid
            ),
            fresh_score,
        )
    else:
        kv_rows, score_rows = fresh_kv, fresh_score
    shape: list[int | Dim] = [w, coff, ratio, proj]
    fresh = compressor(
        ops.reshape(kv_rows, shape),
        ops.reshape(score_rows, shape),
        ops.reshape(windows.present, [w, coff, ratio]),
        ops.unsqueeze(ops.gather(freqs_table, windows.start, axis=0), 1),
    )
    fresh = ops.reshape(fresh, [w, compressor.head_dim])

    cap = ceildiv(max_seq_len, ratio)
    if zone_leaf is not None:
        zone_leaf.store(
            layer,
            KEY,
            fresh,
            ops.cast(windows.woff, DType.uint32),
            cache_lengths=windows.base,
            rows_per_seq=cap,
            ratio=ratio,
        )
    return CompressedStream(
        fresh=fresh,
        windows=windows,
        rows=rows,
        cap=cap,
        zone_leaf=zone_leaf,
        layer=layer,
    )
