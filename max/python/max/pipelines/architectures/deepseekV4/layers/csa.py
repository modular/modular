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
2. pools every window's ``coff * ratio`` source rows, from the state leaf
   for positions before the request's ``P`` and from the fresh projections at
   or after it, into one candidate entry per window -- in one op that reads
   the leaf in place (:meth:`~.compressor.DeepseekV4Compressor.pool_paged`),
   or without a cache from per-window tables;
3. stores the candidates into the zone leaf at entry slots ``P // ratio ...``
   of their request;
4. hands back the candidates and where the closed entries live, so a reader
   can address the zone leaf by entry (:meth:`CompressedStream.entries`, the
   fused attention and indexer kernels) or, without a cache, build a
   per-token candidate table (:attr:`CompressedStream.table`, the reference
   path).

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

from .cache import KEY, VALUE, CacheLeaf, LayerIndex, arange, ceildiv, scalar
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
    """``[W, head_dim]`` candidate entries, all requests' windows end to end.
    With a cache the row count only bounds ``W``; the rows past it are
    unwritten and never read."""
    windows: WindowRows
    rows: RaggedRows
    cap: int
    """Zone slots a query may reach, ``ceil(max_seq_len / ratio)``."""
    zone_leaf: CacheLeaf | None
    layer: TensorValue
    """This layer within the zone leaf, host uint32 (:class:`LayerIndex`)."""

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

        Only the cache-less reference path builds it, where no zone slot is
        live: the cached half is zeros and the fresh half the request's
        windows. With a cache the indexer reads the zone leaf by entry
        (:meth:`~.indexer.DeepseekV4Indexer.score_cached`).
        """
        assert self.zone_leaf is None
        rows = self.rows
        t = rows.total
        cap = self.cap
        device = rows.device
        head_dim = self.fresh.shape[1]
        one = scalar(1, device)
        cached = ops.broadcast_to(
            ops.constant(0.0, self.fresh.dtype, device), [t, cap, head_dim]
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
    layer: LayerIndex,
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
        layer: Where this layer lives in the leaves; both leaves are
            indexed by ``layer.zone``.
    """
    ratio = compressor.compress_ratio
    assert windows.ratio == ratio
    assert layer.zone is not None

    kv_proj, score_proj = compressor.projections(x32)
    cap = ceildiv(max_seq_len, ratio)
    if state_leaf is None:
        # Every position is in the chunk. Rows past a request's end are
        # clamped to its last one, so a window not yet closed still reads
        # finite rows; ``present`` masks positions below 0.
        tables = windows.tables
        assert tables is not None, "a batch without a cache needs its W"
        w = tables.total
        coff = compressor.coff
        shape: list[int | Dim] = [w, coff, ratio, compressor.proj_dim]
        fresh = compressor(
            ops.reshape(ops.gather(kv_proj, tables.fresh_row, axis=0), shape),
            ops.reshape(
                ops.gather(score_proj, tables.fresh_row, axis=0), shape
            ),
            ops.reshape(tables.present, [w, coff, ratio]),
            ops.unsqueeze(ops.gather(freqs_table, tables.start, axis=0), 1),
        )
        fresh = ops.reshape(fresh, [w, compressor.head_dim])
    else:
        assert zone_leaf is not None
        state_leaf.store(layer.zone, KEY, kv_proj, rows.offsets)
        state_leaf.store(layer.zone, VALUE, score_proj, rows.offsets)
        woff = ops.cast(windows.woff, DType.uint32)
        fresh = compressor.pool_paged(
            kv_proj,
            score_proj,
            rows,
            woff,
            freqs_table,
            state_leaf.values,
            layer.zone,
        )
        zone_leaf.store(
            layer.zone,
            KEY,
            fresh,
            woff,
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
        layer=layer.zone,
    )
