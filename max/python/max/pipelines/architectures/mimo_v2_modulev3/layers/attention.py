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
"""MiMo-V2 attention on ModuleV3: fused chunk-order QKV, partial RoPE,
SWA, sinks.

The paged-cache kernels run through the experimental functional wrappers,
which own the ``Tensor``-to-``TensorValue`` conversion at the call boundary;
the per-device loop stays because each device's chunks carry their own
layout math.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Linear, Module
from max.experimental.nn.common_layers import quant_ops
from max.experimental.nn.common_layers.functional_kernels import (
    flash_attention_ragged,
    fused_chunked_qkv_rope_store_ragged,
    fused_qk_ragged_rope,
    store_k_cache_ragged,
    store_v_cache_ragged,
)
from max.experimental.nn.common_layers.linear import RowParallelLinear
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.nn.common_layers.quant_tensor import FP8BlockTensor
from max.experimental.sharding import NamedMapping
from max.experimental.tensor import Tensor
from max.nn.attention import MHAMaskVariant
from max.nn.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.quant_config import QuantConfig

from ...mimo_v2.layers.attention import QkvChunkLayout
from ..quant import FP8_BLOCK


class ChunkedQkvProj(Module[[], None]):
    """``qkv_proj`` as the checkpoint stores it, padded per chunk in the graph.

    The checkpoint's FP8 scales cover each chunk padded to whole 128-row
    blocks, but its weight holds only the real rows, and the block-scaled
    matmul reads row ``r``'s scale at ``r // 128``. The pad depends on the
    weight alone, so the graph compiler runs it once, at model init. The pad
    rows' outputs are zero.
    """

    def __init__(
        self,
        layout: QkvChunkLayout,
        hidden_size: int,
        quant_config: QuantConfig,
    ) -> None:
        """Initializes the projection's weights, on the host as stored.

        Args:
            layout: The chunk layout this module holds.
            hidden_size: The model width.
            quant_config: The dense FP8 block-scaled config.
        """
        self.layout = layout
        self.hidden_size = hidden_size
        self.quant_config = quant_config
        self.weight = Tensor.zeros(
            [layout.chunks * layout.rows, hidden_size],
            dtype=DType.float8_e4m3fn,
            device=CPU(),
        )
        self.weight_scale = Tensor.zeros(
            [
                layout.chunks * layout.padded_rows // FP8_BLOCK,
                hidden_size // FP8_BLOCK,
            ],
            dtype=quant_config.weight_scale.dtype,
            device=CPU(),
        )

    def matmul(self, x: Tensor, *, shard_over_mesh: bool = True) -> Tensor:
        """Returns ``[tokens, chunks * padded_rows]``.

        The pad columns' outputs are zero. With ``shard_over_mesh`` each
        device holds whole chunks, so its share of the output is its heads'
        rows; without it the weight stays replicated and every device
        projects its own rows with the full chunk set (DP attention).
        """
        layout = self.layout
        weight = self.weight
        if layout.padded_rows != layout.rows:
            weight = weight.reshape(
                [layout.chunks, layout.rows, self.hidden_size]
            )
            weight = F.pad(
                weight, [0, 0, 0, layout.padded_rows - layout.rows, 0, 0]
            )
            weight = weight.reshape(
                [layout.chunks * layout.padded_rows, self.hidden_size]
            )
        quant = FP8BlockTensor(
            weight=weight,
            weight_scale_inv=self.weight_scale,
            block_size=(FP8_BLOCK, FP8_BLOCK),
        )
        if shard_over_mesh:
            quant = quant.shard(0, x.mesh)
        else:
            quant = quant.to(x.mesh)
        return quant_ops.matmul(x, quant)


def _split_heads(x: Tensor, head_dim: int) -> Tensor:
    """Views ``[tokens, chunks, heads * head_dim]`` as its heads.

    Returns ``[tokens, chunks, heads, head_dim]``. When a whole chunk is a
    whole number of ``head_dim`` rows, this stays a view of the chunk slice,
    which the pad's concat reads in place; otherwise the compiler copies the
    slice first. Merging ``chunks`` and ``heads`` before the pad copies it
    either way.
    """
    tokens, chunks, rows = x.shape
    return x.reshape([tokens, chunks, int(rows) // head_dim, head_dim])


def _pad_heads(x: Tensor, depth: int) -> Tensor:
    """Zero-pads each head of ``[tokens, chunks, heads, dim]`` to ``depth``.

    Returns ``[tokens, chunks * heads, depth]``.
    """
    tokens, chunks, heads, dim = x.shape
    if depth != int(dim):
        zeros = F.broadcast_to(
            F.constant(0, x.dtype, device=x.device),
            [tokens, chunks, heads, depth - int(dim)],
        )
        x = F.concat([x, zeros], axis=-1)
    return x.reshape([tokens, int(chunks) * int(heads), depth])


def prepare_qkv_unfused(
    qkv: Tensor,
    input_row_offsets: Tensor,
    freqs_cis: Tensor,
    kv_collection: PagedCacheValues,
    layer_idx: Tensor,
    value_scale: float,
    *,
    kv_params: KVCacheParams,
    layout: QkvChunkLayout,
    head_dim: int,
    v_head_dim: int,
) -> Tensor:
    """Prepares chunked queries and stores rotated keys and scaled values.

    This is the default path and the reference for the fused operator.
    The projection includes FP8 block padding after each chunk's real rows.

    Args:
        qkv: Projection with shape ``[tokens, chunks, padded_chunk_rows]``.
        input_row_offsets: Ragged sequence boundaries on the projection device.
        freqs_cis: Partial NeoX RoPE table on the projection device.
        kv_collection: Paged cache receiving K and V for the current layer.
        layer_idx: CPU scalar selecting the layer within its cache group.
        value_scale: FP32 multiplier applied before V is rounded to BF16.
        kv_params: Parameters of the layer's cache group.
        layout: Chunk counts and Q/K/V row extents on this device.
        head_dim: Unpadded query and key width.
        v_head_dim: Unpadded value width.

    Returns:
        Rotated queries padded to the cache head width.
    """
    assert qkv.shape[1] == layout.chunks
    k_start = layout.q_rows
    v_start = k_start + layout.k_rows
    # Q and K are split into heads together, so a chunk that is not a
    # whole number of heads (a sliding one) costs one copy, not two.
    qk = _split_heads(qkv[:, :, :v_start], head_dim)
    q = qk[:, :, : k_start // head_dim]
    k = qk[:, :, k_start // head_dim :]
    v = _split_heads(qkv[:, :, v_start : layout.rows], v_head_dim)
    # Scaled in float32 and rounded once, as a reference multiplying a
    # BF16 tensor by a Python float does. The cache holds scaled V.
    v = (v.cast(DType.float32) * value_scale).cast(v.dtype)

    depth = kv_params.head_dim
    q = _pad_heads(q, depth)
    store_k_cache_ragged(
        kv_collection,
        _pad_heads(k, depth),
        input_row_offsets,
        layer_idx,
    )
    store_v_cache_ragged(
        kv_collection,
        _pad_heads(v, depth),
        input_row_offsets,
        layer_idx,
    )
    return fused_qk_ragged_rope(
        kv_params,
        q,
        input_row_offsets,
        kv_collection,
        freqs_cis,
        layer_idx,
        interleaved=False,
    )


class MiMoV2Attention(Module[..., Tensor]):
    """Attention for one MiMo-V2 decoder layer, on the device mesh.

    ``qkv_proj`` is FP8 in Xiaomi's chunk order: each chunk holds a slice of
    the query heads and the KV heads they read (see :class:`ChunkedQkvProj`).
    The pad rows' outputs are dropped here. Q and K are ``head_dim`` wide and
    V is ``v_head_dim`` wide; each is zero-padded to the KV cache head dim
    after the projection, which leaves ``QK^T`` unchanged, and the output is
    cut back to ``v_head_dim`` before ``o_proj``. RoPE rotates the leading
    dims that ``freqs_cis`` covers, NeoX style.

    Each device holds whole chunks, so its shard holds its heads in order;
    ``o_proj`` reduces the shards' partial sums. The layer's KV-group index
    is a constant baked into this module, so attention layers do not share a
    compiled subgraph.

    With ``data_parallel`` the layer runs replicated instead: every device
    holds all heads and attends its own rows, so nothing shards over the
    mesh and no partial sums need reducing.
    """

    attention_sink_bias: Tensor | None

    def __init__(
        self,
        *,
        layout: QkvChunkLayout,
        head_dim: int,
        v_head_dim: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        scale: float,
        value_scale: float,
        sliding_window: int | None,
        has_sinks: bool,
        quant_config: QuantConfig,
        data_parallel: bool = False,
        fused_qkv: bool = False,
    ) -> None:
        """Initializes the layer.

        Args:
            layout: The full ``qkv_proj`` chunk layout; whole chunks per
                device are cut from it at run time.
            head_dim: The Q and K head dim.
            v_head_dim: The V head dim.
            hidden_size: The model width.
            kv_params: The KV cache group this layer writes.
            layer_idx: The layer's index within its KV group.
            scale: The softmax scale.
            value_scale: The factor V is multiplied by before the cache
                write.
            sliding_window: Keys attended to, the query's own included;
                ``None`` for full attention.
            has_sinks: Whether the layer has a learned per-head sink logit.
            quant_config: The dense FP8 config of ``qkv_proj``.
            data_parallel: Run replicated per data-parallel replica instead
                of sharding the heads over tensor parallelism.
            fused_qkv: Whether to prepare Q, K and V with the fused chunked
                QKV op, which raises on an unsupported layout. Defaults to
                ``False``.
        """
        self.layout = layout
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.hidden_size = hidden_size
        self.kv_params = kv_params
        self.layer_idx = layer_idx
        self.scale = scale
        self.value_scale = value_scale
        self.sliding_window = sliding_window
        self.data_parallel = data_parallel
        self.fused_qkv = fused_qkv
        self.n_heads = layout.chunks * layout.q_rows // head_dim
        self.qkv_proj = ChunkedQkvProj(layout, hidden_size, quant_config)
        if data_parallel:
            self.o_proj = Linear(
                in_dim=self.n_heads * v_head_dim,
                out_dim=hidden_size,
                bias=False,
            )
        else:
            self.o_proj = RowParallelLinear(
                in_dim=self.n_heads * v_head_dim,
                out_dim=hidden_size,
                bias=False,
            )
        self.attention_sink_bias = None
        if has_sinks:
            sinks = Tensor.zeros([self.n_heads], dtype=DType.bfloat16)
            if data_parallel:
                self.attention_sink_bias = sinks.to(sinks.mesh)
            else:
                self.attention_sink_bias = sinks.to(
                    NamedMapping(sinks.mesh, (TP,))
                )

    def forward(
        self,
        x: Tensor,
        kv_collections: Sequence[PagedCacheValues],
        freqs_cis: Tensor,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Runs the layer's attention on each device's heads.

        Args:
            x: The replicated ``[tokens, hidden]`` input, or the batch
                sharded over data-parallel replicas.
            kv_collections: The layer's KV group, one per device.
            freqs_cis: The group's RoPE table, replicated, in ``x.dtype``.
            input_row_offsets: Each sequence's first row in ``x``,
                replicated or sharded like ``x``.

        Returns:
            This layer's partial ``o_proj`` output, one shard per device;
            complete per shard under data parallelism.
        """
        mesh = x.mesh
        n = mesh.num_devices
        # Whole chunks per device, so whole heads: a plain split is aligned.
        # Replicated (DP) attention keeps every chunk on every device.
        layout = dataclasses.replace(
            self.layout,
            chunks=self.layout.chunks // (1 if self.data_parallel else n),
        )
        local_heads = self.n_heads // (1 if self.data_parallel else n)
        qkv = self.qkv_proj.matmul(x, shard_over_mesh=not self.data_parallel)
        layer_idx = F.constant(self.layer_idx, DType.uint32, device=CPU())
        attns = []
        for d in range(n):
            # Each shard's own rows: one count per device, since DP shards
            # the batch (and its offsets) across the mesh.
            tokens = qkv.local_shards[d].shape[0]
            local_qkv = qkv.local_shards[d].reshape(
                [tokens, layout.chunks, layout.padded_rows]
            )
            local_offsets = input_row_offsets.local_shards[d]
            if self.fused_qkv:
                q = fused_chunked_qkv_rope_store_ragged(
                    local_qkv,
                    local_offsets,
                    freqs_cis.local_shards[d],
                    kv_collections[d],
                    layer_idx,
                    self.value_scale,
                    num_q_heads=local_heads,
                    qk_head_dim=self.head_dim,
                    v_head_dim=self.v_head_dim,
                    chunk_padding=layout.padded_rows - layout.rows,
                )
            else:
                q = prepare_qkv_unfused(
                    local_qkv,
                    local_offsets,
                    freqs_cis.local_shards[d],
                    kv_collections[d],
                    layer_idx,
                    self.value_scale,
                    kv_params=self.kv_params,
                    layout=layout,
                    head_dim=self.head_dim,
                    v_head_dim=self.v_head_dim,
                )
            sinks = self.attention_sink_bias
            attn = flash_attention_ragged(
                self.kv_params,
                input=q,
                input_row_offsets=local_offsets,
                kv_collection=kv_collections[d],
                layer_idx=layer_idx,
                mask_variant=(
                    MHAMaskVariant.CAUSAL_MASK
                    if self.sliding_window is None
                    else MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
                ),
                scale=self.scale,
                local_window_size=self.sliding_window or -1,
                sink_weights=(
                    sinks.local_shards[d] if sinks is not None else None
                ),
            )
            attns.append(
                attn[:, :, : self.v_head_dim].reshape(
                    [tokens, local_heads * self.v_head_dim]
                )
            )
        # Each shard's columns are its heads' ``o_proj`` inputs, in order;
        # the row-parallel projection leaves each device a partial sum, which
        # the residual add reduces. Under DP each shard is a complete
        # projection of its own rows.
        if n == 1 or self.data_parallel:
            heads = Tensor.from_shard_values(attns, x.mapping)
        else:
            heads = Tensor.from_shard_values(
                attns, NamedMapping(mesh, (None, TP))
            )
        return self.o_proj(heads)
