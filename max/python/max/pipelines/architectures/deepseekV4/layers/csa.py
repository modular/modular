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
close in the current chunk.

Both the attention's compressor and the indexer's run this, each against its
own pair of leaves. Given ``s`` new tokens starting at position ``P``
(``cache_lengths``), it

1. stores the tokens' raw ``wkv`` / ``wgate`` projections into the state leaf;
2. assembles the ``ceil(s / ratio)`` candidate windows that begin at or after
   ``A = P - P % ratio`` (plus the predecessor window when overlapping), row
   by row from the state leaf for positions before ``P`` and from the fresh
   projections at or after it;
3. pools them into candidate entries and stores those into the zone leaf at
   entry slots ``P // ratio ...``;
4. hands back the candidates and where the closed entries live, so a reader
   can address the zone leaf by entry (:meth:`CompressedStream.entries`, the
   fused kernel) or gather the closed entries into a table
   (:attr:`CompressedStream.table`, the indexer's scoring and the cache-less
   reference path).

Why storing every candidate is exact: the windows that actually close inside
the chunk number ``ceil(s / ratio)`` or one fewer. When one fewer, the last
candidate lands on entry slot ``T // ratio`` (``T = P + s``), the window still
open after the chunk. No reader reaches it -- a query at position ``t`` sees
entries below ``(t + 1) // ratio`` -- and it is overwritten when the window
does close. Its page is allocated: it is only reached when ``T % ratio != 0``,
so the slot's first token ``(T // ratio) * ratio`` is below ``T``. Decode is
the ``s == 1`` case of the same statement.
"""

from __future__ import annotations

from dataclasses import dataclass

from max.graph import TensorValue, ops

from .cache import (
    KEY,
    VALUE,
    CacheLeaf,
    arange,
    ceildiv,
    idiv,
    row_offsets,
    scalar,
)
from .compressor import DeepseekV4Compressor


@dataclass
class CompressedStream:
    """What a chunk sees of one compressed stream.

    Candidates are numbered ``0 .. cap - 1`` for the zone slots and ``cap ..
    cap + n_new - 1`` for the chunk's fresh windows; ``valid`` says which of
    them a query may see. Zone slot ``j`` is live below ``base``; fresh
    candidate ``w`` is entry ``base + w``, already stored in the zone leaf.
    """

    fresh: TensorValue
    """``[b, n_new, head_dim]``."""
    base: TensorValue
    """``[b]`` int32, entries closed before this chunk (``P // ratio``)."""
    cap: int
    """Zone slots a query may reach, ``ceil(max_seq_len / ratio)``."""
    zone_leaf: CacheLeaf | None
    layer: int

    @property
    def n_new(self) -> int:
        return int(self.fresh.shape[1])

    @property
    def cached(self) -> TensorValue:
        """Zone slots ``[0, cap)`` gathered out of the leaf, ``[b, cap, head_dim]``.

        A copy of the leaf per call (``CacheLeaf.gather``); only the indexer's
        scoring and the cache-less reference path pay for it.
        """
        b, head_dim = int(self.fresh.shape[0]), int(self.fresh.shape[2])
        device = self.fresh.device
        if self.zone_leaf is None:
            return ops.broadcast_to(
                ops.constant(0.0, self.fresh.dtype, device),
                [b, self.cap, head_dim],
            )
        # Slots at or past ``base`` are dead; clamp them onto the last live
        # one (or slot 0 when nothing has closed) and let ``valid`` hide them.
        slots = ops.min(
            ops.broadcast_to(
                ops.reshape(arange(self.cap, device), [1, self.cap]),
                [b, self.cap],
            ),
            ops.reshape(
                ops.max(self.base - scalar(1, device), scalar(0, device)),
                [b, 1],
            ),
        )
        return self.zone_leaf.gather(self.layer, KEY, slots)

    @property
    def table(self) -> TensorValue:
        """``cached`` then ``fresh`` along axis 1, the candidate order."""
        return ops.concat([self.cached, self.fresh], axis=1)

    def entries(self, candidates: TensorValue) -> TensorValue:
        """Candidate numbers -> zone-leaf entry indices, ``-1`` passing through.

        Args:
            candidates: ``[b, s, k]`` int32 candidate numbers, ``-1`` unused.
        """
        b = candidates.shape[0]
        device = candidates.device
        cap = scalar(self.cap, device)
        fresh = ops.reshape(self.base, [b, 1, 1]) + (candidates - cap)
        entry = ops.where(candidates < cap, candidates, fresh)
        return ops.where(
            candidates >= scalar(0, device), entry, scalar(-1, device)
        )

    def valid(self, cutoff: TensorValue) -> TensorValue:
        """``[b, s, cap + n_new]`` bool: candidate closed before each query.

        Args:
            cutoff: ``[b, s]`` int32, ``(position + 1) // ratio`` per query.
        """
        b, s = cutoff.shape[0], cutoff.shape[1]
        device = cutoff.device
        base = ops.reshape(self.base, [b, 1, 1])
        cached = ops.broadcast_to(
            ops.reshape(arange(self.cap, device), [1, 1, self.cap]) < base,
            [b, s, self.cap],
        )
        fresh = (
            base + ops.reshape(arange(self.n_new, device), [1, 1, self.n_new])
        ) < ops.unsqueeze(cutoff, -1)
        return ops.concat([cached, fresh], axis=-1)


def compressed_stream(
    compressor: DeepseekV4Compressor,
    x32: TensorValue,
    positions: TensorValue,
    freqs_table: TensorValue,
    max_seq_len: int,
    state_leaf: CacheLeaf | None,
    zone_leaf: CacheLeaf | None,
    layer: int,
) -> CompressedStream:
    """Run one compressed stream over a chunk.

    Args:
        compressor: The stream's compressor (attention's or the indexer's).
        x32: ``[b, s, hidden]`` float32 block input for the chunk.
        positions: ``[b, s]`` int32 absolute positions of the chunk's tokens;
            ``positions[:, 0]`` is ``P``.
        freqs_table: The layer's full rotary table.
        max_seq_len: Caps how many closed entries a query can see.
        state_leaf: The stream's open-state leaf, or ``None`` to run without
            a cache (a fresh sequence, nothing stored).
        zone_leaf: The stream's compressed-zone leaf, or ``None`` likewise.
        layer: Index of this layer within both leaves.
    """
    ratio = compressor.compress_ratio
    coff = compressor.coff
    proj = compressor.proj_dim
    b, s = int(x32.shape[0]), int(x32.shape[1])
    device = x32.device

    kv_proj, score_proj = compressor.projections(x32)
    if state_leaf is not None:
        offsets = row_offsets(b, s, device)
        state_leaf.store(
            layer, KEY, ops.reshape(kv_proj, [b * s, proj]), offsets
        )
        state_leaf.store(
            layer, VALUE, ops.reshape(score_proj, [b * s, proj]), offsets
        )

    p = positions[:, 0]
    base = idiv(p, ratio)
    aligned = base * ratio
    n_new = ceildiv(s, ratio)
    n_win = n_new + coff - 1
    # Window ``w`` (``w = -(coff - 1) .. n_new - 1``) covers positions
    # ``aligned + w * ratio + r``.
    rel = ops.reshape(
        (arange(n_win, device) - (coff - 1)) * ratio, [n_win, 1]
    ) + ops.reshape(arange(ratio, device), [1, ratio])
    pos = ops.reshape(aligned, [b, 1, 1]) + ops.unsqueeze(rel, 0)
    present = pos >= scalar(0, device)
    p3 = ops.reshape(p, [b, 1, 1])
    from_cache = pos < p3
    flat = [b, n_win * ratio]

    # Rows at or after ``P`` come from the chunk; the index is clamped so a
    # not-yet-closed window reads a real (finite) row and never NaN.
    fresh_idx = ops.reshape(
        ops.min(ops.max(pos - p3, scalar(0, device)), scalar(s - 1, device)),
        flat,
    )
    fresh_kv = ops.gather_nd(
        kv_proj, ops.unsqueeze(fresh_idx, -1), batch_dims=1
    )
    fresh_score = ops.gather_nd(
        score_proj, ops.unsqueeze(fresh_idx, -1), batch_dims=1
    )
    if state_leaf is not None:
        # Rows before ``P`` come from the state leaf; dead positions are
        # clamped onto a live slot and masked by ``present`` / ``from_cache``.
        cache_slot = ops.reshape(
            ops.min(
                ops.max(pos, scalar(0, device)),
                ops.max(p3 - scalar(1, device), scalar(0, device)),
            ),
            flat,
        )
        mask = ops.unsqueeze(ops.reshape(from_cache, flat), -1)
        kv_rows = ops.where(
            mask, state_leaf.gather(layer, KEY, cache_slot), fresh_kv
        )
        score_rows = ops.where(
            mask, state_leaf.gather(layer, VALUE, cache_slot), fresh_score
        )
    else:
        kv_rows, score_rows = fresh_kv, fresh_score
    shape = [b, n_win, ratio, proj]
    window_pos = ops.reshape(aligned, [b, 1]) + ops.reshape(
        arange(n_new, device) * ratio, [1, n_new]
    )
    fresh = compressor(
        ops.reshape(kv_rows, shape),
        ops.reshape(score_rows, shape),
        present,
        ops.gather(freqs_table, window_pos, axis=0),
    )

    if zone_leaf is not None:
        zone_leaf.store(
            layer,
            KEY,
            ops.reshape(fresh, [b * n_new, compressor.head_dim]),
            row_offsets(b, n_new, device),
            cache_lengths=base,
            rows_per_seq=n_new,
            ratio=ratio,
        )
    return CompressedStream(
        fresh=fresh,
        base=base,
        cap=ceildiv(max_seq_len, ratio),
        zone_leaf=zone_leaf,
        layer=layer,
    )
