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
"""One compressed stream on ModuleV3: state leaf in, zone leaf out.

Stores the chunk's raw projections into the state leaf, pools every window
the chunk touches, and stores each candidate entry into the zone leaf at
entry slot ``P // ratio + w``. Storing the trailing, possibly unclosed window
is exact; the proof is in the docstring of the ModuleV2 implementation
(``deepseekV4/layers/csa.py`` at dfc8109fc0c).
"""

from __future__ import annotations

from dataclasses import dataclass

from max import functional as F
from max.dtype import DType
from max.support.math import ceildiv
from max.tensor import Tensor

from .cache import KEY, VALUE, CacheLeaf, LayerIndex
from .compressor import DeepseekV4Compressor
from .ragged import RaggedRows, WindowRows

__all__ = ["CompressedStream", "arange", "compressed_stream"]


def arange(n: int, like: Tensor) -> Tensor:
    """``[0, n)`` int32, whole on every device ``like`` is on."""
    return F.range(0, n, 1, out_dim=n, dtype=DType.int32, device=like.mesh)


def int32(value: int, like: Tensor) -> Tensor:
    return F.constant(value, DType.int32, device=like.mesh)


@dataclass
class CompressedStream:
    """What a batch sees of one compressed stream.

    Candidates ``0 .. cap - 1`` are zone slots (live below the request's
    ``base``), ``cap .. 2 * cap - 1`` the request's fresh windows (fresh
    candidate ``w`` is entry ``base + w``, already in the zone leaf).
    """

    windows: WindowRows
    rows: RaggedRows
    cap: int
    """Zone slots a query may reach, ``ceil(max_seq_len / ratio)``."""
    zone_leaf: CacheLeaf
    layer: Tensor
    """This layer within the zone leaf, host uint32."""

    @property
    def n_cand(self) -> int:
        """Candidates per query: ``cap`` zone slots and ``cap`` fresh ones."""
        return 2 * self.cap

    @property
    def token_base(self) -> Tensor:
        """``[T]`` int32 entries closed before each token's chunk."""
        return F.gather(self.windows.base, self.rows.bid, axis=0)

    def entries(self, candidates: Tensor) -> Tensor:
        """``[T, k]`` candidate numbers -> zone entries, ``-1`` passing through."""
        t = self.rows.total
        cap = int32(self.cap, candidates)
        base = self.token_base.reshape([t, 1])
        entry = F.where(candidates < cap, candidates, base + (candidates - cap))
        return F.where(
            candidates >= int32(0, candidates), entry, int32(-1, candidates)
        )

    def valid(self, cutoff: Tensor) -> Tensor:
        """``[T, n_cand]`` bool: candidate closed before each query.

        ``cutoff`` is ``[T]`` int32, ``(position + 1) // ratio`` per query.
        """
        t = self.rows.total
        cap = self.cap
        base = self.token_base.reshape([t, 1])
        slots = arange(cap, cutoff).reshape([1, cap])
        cached = slots < base
        fresh = (base + slots) < cutoff.reshape([t, 1])
        return F.concat([cached, fresh], axis=-1)


def compressed_stream(
    compressor: DeepseekV4Compressor,
    x32: Tensor,
    rows: RaggedRows,
    windows: WindowRows,
    freqs_table: Tensor,
    max_seq_len: int,
    state_leaf: CacheLeaf,
    zone_leaf: CacheLeaf,
    layer: LayerIndex,
) -> CompressedStream:
    """Run one compressed stream (attention's or the indexer's) over a batch.

    Args:
        compressor: The stream's compressor.
        x32: ``[T, hidden]`` float32 block input.
        rows: The batch's token bookkeeping.
        windows: The candidate windows for ``compressor.compress_ratio``.
        freqs_table: The layer's full rotary table.
        max_seq_len: Caps how many closed entries a query can see.
        state_leaf: The stream's open-state leaf.
        zone_leaf: The stream's compressed-zone leaf.
        layer: Where this layer lives; both leaves use ``layer.zone``.
    """
    ratio = compressor.compress_ratio
    assert windows.ratio == ratio
    assert layer.zone is not None

    kv_proj, score_proj = compressor.projections(x32)
    cap = ceildiv(max_seq_len, ratio)
    state_leaf.store(layer.zone, KEY, kv_proj, rows.offsets)
    state_leaf.store(layer.zone, VALUE, score_proj, rows.offsets)
    woff = windows.woff.cast(DType.uint32)
    fresh = compressor(
        kv_proj, score_proj, rows, woff, freqs_table, state_leaf, layer.zone
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
        windows=windows,
        rows=rows,
        cap=cap,
        zone_leaf=zone_leaf,
        layer=layer.zone,
    )
