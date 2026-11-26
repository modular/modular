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
"""Inkling 2's fused K/V prologue: one launch for both convs and stores.

A local layer's K and V each run a short convolution, then K's heads clip and
V's heads norm, then both stores scatter into the paged cache. This op does
all six in one launch (``Kernels/lib/inkling2_kv_prologue``), for any ragged
batch: decode, prefill chunks, mixed batches and speculative verify, with
``short_conv_ring``'s chunk semantics.
"""

from __future__ import annotations

import os

from max import functional as F
from max.dtype import DType
from max.graph import BufferValue, Dim, TensorValue, ops
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.kv_cache.input_types import KVCacheInputsPerDevice
from max.tensor import Tensor
from max.tensor.sharding.per_shard_dim import is_static, local_dim_at

from .short_convolution import SiteState

_ENV_FLAG = "INKLING2_FUSED_KV_PROLOGUE"
"""Read when the graph is built; default off, so the composed path stays."""

_WARP = 32
"""Threads per warp, as ``Kernels/lib/inkling2_kv_prologue`` splits a head."""

_MAX_THREADS = 1024
"""One CTA holds a thread per channel of K and V."""


def enabled() -> bool:
    """Whether ``INKLING2_FUSED_KV_PROLOGUE=1`` asks for the fused op."""
    return os.environ.get(_ENV_FLAG, "0") == "1"


def _kv_prologue(
    k_in: TensorValue,
    v_in: TensorValue,
    k_taps: TensorValue,
    v_taps: TensorValue,
    k_ring: BufferValue,
    v_ring: BufferValue,
    k_rows: TensorValue,
    v_rows: TensorValue,
    positions: TensorValue,
    input_row_offsets: TensorValue,
    kv_collection: KVCacheInputsPerDevice[TensorValue, BufferValue],
    k_layer_row: TensorValue,
    v_layer_row: TensorValue,
    layer_idx: TensorValue,
    *,
    clip: float,
    clip_eps: float,
    eps: float,
    out_divisor: float,
) -> None:
    ops.inplace_custom(
        "inkling2.kv_prologue",
        device=TensorValue(k_in).device,
        values=[
            k_in,
            v_in,
            k_taps,
            v_taps,
            k_ring,
            v_ring,
            k_rows,
            v_rows,
            positions,
            input_row_offsets,
            kv_collection.kv_blocks,
            kv_collection.page_stride,
            kv_collection.cache_lengths,
            kv_collection.lookup_table,
            k_layer_row,
            v_layer_row,
            layer_idx,
        ],
        parameters={
            "clip": clip,
            "clip_eps": clip_eps,
            "eps": eps,
            "out_divisor": out_divisor,
        },
    )


_kv_prologue_op = F.functional(_kv_prologue)


def _static(dim: Dim) -> int | None:
    """One shard's size of the dim when the graph fixes it, else ``None``.

    Under tensor parallelism the K/V rings are sharded on their channels and
    the cache leaf is replicated, so every shard has the same size, and that
    size, not the global one, is what the op is compiled for.
    """
    local = local_dim_at(dim, 0)
    return int(local) if is_static(local) else None


def can_fuse(
    k_in: Tensor,
    k_site: SiteState,
    v_site: SiteState,
    kv_collection: PagedCacheValues,
    *,
    kernel: int,
    head_dim: int,
) -> bool:
    """Whether the flag is on and the op is compiled for these shapes.

    Only static properties: the op is exact for any ragged batch, so the
    token and sequence counts never matter. The widths are one rank's, as the
    op sees them.

    Args:
        k_in: ``[rows, kv_dim]`` projected keys.
        k_site: K's conv state.
        v_site: V's conv state.
        kv_collection: The layer's cache leaf.
        kernel: The conv's tap count.
        head_dim: The layer's head width, from the config.
    """
    if not enabled():
        return False
    blocks = kv_collection.kv_blocks
    if (
        k_in.dtype != DType.bfloat16
        or blocks.dtype != DType.bfloat16
        or k_site.ring.dtype != DType.float32
        or v_site.ring.dtype != DType.float32
        or blocks.rank != 6
        or kv_collection.split_head_dims is not None
        or kv_collection.kv_scales is not None
    ):
        return False
    dims = [_static(d) for d in blocks.shape[1:]]
    ring_dims = [_static(d) for d in (*k_site.ring.shape, *v_site.ring.shape)]
    if None in dims or None in ring_dims[1:3] or None in ring_dims[4:6]:
        return False
    planes, _, _, heads, cache_head_dim = dims
    ring_len = ring_dims[1]
    if heads is None or cache_head_dim is None or ring_len is None:
        return False
    kv_dim = heads * cache_head_dim
    return (
        planes == 2
        and cache_head_dim == head_dim
        and head_dim % _WARP == 0
        and 2 * kv_dim <= _MAX_THREADS
        and ring_dims[2] == kv_dim
        and ring_dims[4:6] == [ring_len, kv_dim]
        and ring_len >= kernel - 1
    )


def kv_prologue(
    k_in: Tensor,
    v_in: Tensor,
    k_taps: Tensor,
    v_taps: Tensor,
    k_site: SiteState,
    v_site: SiteState,
    kv_collection: PagedCacheValues,
    positions: Tensor,
    *,
    input_row_offsets: Tensor,
    layer_idx: Tensor,
    clip: float,
    clip_eps: float,
    eps: float,
) -> None:
    """Runs both convs, K's clip, V's norm and both stores in one launch.

    The conv divides by ``sqrt(kernel + 1)`` before its bfloat16 rounding,
    as ``ShortConvolution`` does; the head norms read that rounded value.

    Args:
        k_in: ``[rows, kv_dim]`` projected keys before the conv.
        v_in: The same for values.
        k_taps: ``[kv_dim, kernel]`` K taps, the last on the current token.
        v_taps: The same for values.
        k_site: K's ring, slot table and layer row.
        v_site: The same for values.
        kv_collection: The layer's cache leaf, written in place.
        positions: ``[rows]`` absolute position per token.
        input_row_offsets: ``[batch + 1]`` ragged row offsets.
        layer_idx: The layer's index within the cache leaf.
        clip: ``qk_l2_clip_threshold``.
        clip_eps: The ``l2_clip`` denominator's epsilon.
        eps: ``rms_norm_eps``.
    """
    kernel = int(k_taps.shape[1])
    _kv_prologue_op(
        k_in,
        v_in,
        k_taps.cast(DType.bfloat16),
        v_taps.cast(DType.bfloat16),
        k_site.ring,
        v_site.ring,
        k_site.rows,
        v_site.rows,
        positions,
        input_row_offsets,
        kv_collection,
        k_site.layer_row,
        v_site.layer_row,
        layer_idx,
        clip=clip,
        clip_eps=clip_eps,
        eps=eps,
        out_divisor=(kernel + 1) ** 0.5,
    )
