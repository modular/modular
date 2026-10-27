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
"""The MiMo-V2 DFlash drafter on ModuleV3: a non-causal block over windowed
target context.

The block is ``[anchor, mask x (block_size - 1)]``. Each block query attends
to the context keys its window reaches and to every block key, with a learned
sink per head. The layers are pre-norm, with per-head ``q_norm`` and
``k_norm`` and NeoX RoPE on the leading ``rotary_dim`` of each head.

Weight names are the checkpoint's, so the graph-API drafter's weight adapter
loads this module unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from max.driver import Buffer
from max.dtype import DType
from max.experimental.nn import (
    Linear,
    Module,
    ModuleList,
    RMSNorm,
)
from max.experimental.nn.common_layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.nn.common_layers.mlp import MLP
from max.experimental.realization_context import ensure_context
from max.experimental.sharding import NamedMapping
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorValue, ops
from max.graph.weights import WeightData
from max.nn.attention import MHAMaskVariant
from max.nn.kernels import (
    flash_attention_ragged,
    fused_qk_ragged_rope,
    store_k_cache_ragged,
    store_v_cache_ragged,
)
from max.nn.kv_cache import PagedCacheValues

from ..dflash_mimo_v2 import DFlashMiMoV2Config
from .context_writer import DFlashContextWriter, _heads_per_shard

__all__ = [
    "DFlashMiMoV2",
    "DFlashMiMoV2Attention",
    "DFlashMiMoV2DecoderLayer",
]


class DFlashMiMoV2Attention(Module[..., Tensor]):
    """A drafter layer's attention: its K/V projection, shared by context and
    block rows, plus the block's queries.

    The K/V a block query attends to must be computed the same way whether it
    came from a target tap or from the block itself, so both paths go through
    :meth:`store_kv`. The K/V and Q projections are column-parallel over the
    heads, so each device stores its heads' K/V; ``o_proj`` reduces the
    shards' partial sums. Under data parallelism every one-device replica
    holds the full drafter instead: all heads local, plain projections, and
    each replica attends its own rows with nothing to reduce.
    """

    def __init__(self, config: DFlashMiMoV2Config, layer_idx: int) -> None:
        self.config = config
        self.layer_idx = layer_idx
        self.data_parallel = config.data_parallel_degree > 1
        self.num_heads = config.num_attention_heads
        shards = 1 if self.data_parallel else len(config.devices)
        self.local_heads = _heads_per_shard(
            config.num_attention_heads, shards, "query"
        )
        self.num_key_value_heads = config.num_key_value_heads
        self.local_key_value_heads = _heads_per_shard(
            config.num_key_value_heads, shards, "KV"
        )
        head_dim = config.head_dim
        q_dim = self.num_heads * head_dim
        kv_dim = self.num_key_value_heads * head_dim
        proj_cls = Linear if self.data_parallel else ColumnParallelLinear
        self.q_proj = proj_cls(config.hidden_size, q_dim, bias=False)
        self.k_proj = proj_cls(config.hidden_size, kv_dim, bias=False)
        self.v_proj = proj_cls(config.hidden_size, kv_dim, bias=False)
        self.o_proj = (
            Linear(q_dim, config.hidden_size, bias=False)
            if self.data_parallel
            else RowParallelLinear(q_dim, config.hidden_size, bias=False)
        )
        self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        sinks = Tensor.zeros([self.num_heads], dtype=config.dtype)
        self.attention_sink_bias = (
            sinks.to(sinks.mesh)
            if self.data_parallel
            else sinks.to(NamedMapping(sinks.mesh, (TP,)))
        )

    def store_kv(
        self,
        x: Tensor,
        kv_collections: Sequence[PagedCacheValues],
        freqs_cis: Tensor,
        input_row_offsets: Tensor,
        q: Tensor | None = None,
    ) -> list[TensorValue]:
        """Writes the K/V of ``x``'s rows at their positions and ropes ``q``.

        The only kernel that rotates a NeoX prefix of the head rotates the
        cached K together with a query, so without ``q`` the K rows stand in
        for it and the result is discarded.

        Args:
            x: ``[rows, hidden]`` inputs, ragged by ``input_row_offsets``,
                replicated over tensor parallelism or sharded per replica
                under data parallelism.
            kv_collections: The drafter cache, one per device, whose lengths
                place the rows.
            freqs_cis: The ``[positions, rotary_dim]`` RoPE table, replicated
                on the mesh at every device count: each replica indexes its
                own rows' positions out of the whole table.
            input_row_offsets: Ragged offsets of ``x``, replicated over
                tensor parallelism or sharded per replica under data
                parallelism.
            q: ``[rows, heads, head_dim]`` queries to rope, sharded over the
                heads under tensor parallelism and full per replica under
                data parallelism, if any.

        Returns:
            ``q``, or the K rows without it, roped at the rows' positions;
            one entry per device.
        """
        config = self.config
        head_dim = config.head_dim
        heads = self.num_key_value_heads
        k = self.k_norm(self.k_proj(x).reshape([-1, heads, head_dim]))
        v = self.v_proj(x).reshape([-1, heads, head_dim])
        # A bfloat16 constant would round the scale itself (0.612 -> 0.6133).
        v = (v.cast(DType.float32) * config.attention_value_scale).cast(v.dtype)
        layer_idx = ops.constant(
            self.layer_idx, DType.uint32, device=DeviceRef.CPU()
        )
        roped: list[TensorValue] = []
        with ensure_context():
            for d in range(x.mesh.num_devices):
                k_d = TensorValue(k.local_shards[d])
                v_d = TensorValue(v.local_shards[d])
                offsets_d = TensorValue(input_row_offsets.local_shards[d])
                store_k_cache_ragged(
                    kv_collections[d], k_d, offsets_d, layer_idx
                )
                store_v_cache_ragged(
                    kv_collections[d], v_d, offsets_d, layer_idx
                )
                roped.append(
                    fused_qk_ragged_rope(
                        config.kv_params,
                        k_d if q is None else TensorValue(q.local_shards[d]),
                        offsets_d,
                        kv_collections[d],
                        TensorValue(freqs_cis.local_shards[d]),
                        layer_idx,
                        interleaved=False,
                    )
                )
        return roped

    def forward(
        self,
        x: Tensor,
        kv_collections: Sequence[PagedCacheValues],
        freqs_cis: Tensor,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Runs the block's attention on each device's heads.

        Args:
            x: The ``[rows, hidden]`` input, replicated over tensor
                parallelism or sharded per replica under data parallelism.
            kv_collections: The drafter cache, one per device, whose lengths
                are the anchors' positions.
            freqs_cis: The RoPE table, replicated on the mesh at every
                device count.
            input_row_offsets: Ragged offsets of ``x``, replicated over
                tensor parallelism or sharded per replica under data
                parallelism.

        Returns:
            This layer's ``o_proj`` output: partial per device under tensor
            parallelism, complete per replica under data parallelism.
        """
        head_dim = self.config.head_dim
        q = self.q_norm(self.q_proj(x).reshape([-1, self.num_heads, head_dim]))
        roped = self.store_kv(
            x, kv_collections, freqs_cis, input_row_offsets, q=q
        )
        layer_idx = ops.constant(
            self.layer_idx, DType.uint32, device=DeviceRef.CPU()
        )
        attns: list[TensorValue] = []
        with ensure_context():
            for d in range(x.mesh.num_devices):
                # This device's own rows: all of them under tensor
                # parallelism, its replica's under data parallelism.
                rows = x.local_shards[d].shape[0]
                offsets_d = TensorValue(input_row_offsets.local_shards[d])
                attn = flash_attention_ragged(
                    self.config.kv_params,
                    input=roped[d],
                    input_row_offsets=offsets_d,
                    kv_collection=kv_collections[d],
                    layer_idx=layer_idx,
                    mask_variant=(MHAMaskVariant.SLIDING_WINDOW_NONCAUSAL_MASK),
                    scale=head_dim**-0.5,
                    local_window_size=self.config.sliding_window,
                    sink_weights=TensorValue(
                        self.attention_sink_bias.local_shards[d]
                    ),
                )
                attns.append(
                    ops.reshape(attn, [rows, self.local_heads * head_dim])
                )
        if x.mesh.num_devices == 1 or self.data_parallel:
            heads = Tensor.from_shard_values(attns, x.mapping)
        else:
            heads = Tensor.from_shard_values(
                attns, NamedMapping(x.mesh, (None, TP))
            )
        return self.o_proj(heads)


class DFlashMiMoV2DecoderLayer(Module[..., Tensor]):
    """A pre-norm drafter layer: sink attention, then a SiLU MLP.

    The attention's :meth:`DFlashMiMoV2Attention.store_kv` is the context
    writer's path; the block runs through :meth:`forward`.
    """

    def __init__(self, config: DFlashMiMoV2Config, layer_idx: int) -> None:
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.self_attn = DFlashMiMoV2Attention(config, layer_idx)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        h: Tensor,
        kv_collections: Sequence[PagedCacheValues],
        freqs_cis: Tensor,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Runs the layer over the replicated residual stream."""
        h = h + self.self_attn(
            self.input_layernorm(h),
            kv_collections,
            freqs_cis,
            input_row_offsets,
        ).to(h.mapping)
        return h + self.mlp(self.post_attention_layernorm(h)).to(h.mapping)

    def write_context(
        self,
        ctx: Tensor,
        kv_collections: Sequence[PagedCacheValues],
        freqs_cis: Tensor,
        input_row_offsets: Tensor,
    ) -> None:
        """Writes this layer's context K/V, with no block query."""
        self.self_attn.store_kv(
            ctx, kv_collections, freqs_cis, input_row_offsets
        )


class DFlashMiMoV2(DFlashContextWriter):
    """The drafter: the tap projection plus the layer stack and block forward.

    The target's ``embed_tokens`` embeds the anchor and its ``lm_head`` reads
    the drafts; the mask slots take the checkpoint's trained
    ``mask_embedding``, not the target's embedding of ``mask_token_id``,
    which is an untrained padding row.

    No ``forward``: the fused graph drives the drafter through its named
    entries, never ``__call__``.
    """

    def __init__(
        self,
        config: DFlashMiMoV2Config,
        mask_embedding: WeightData | None = None,
    ) -> None:
        """Builds the drafter.

        Args:
            config: The drafter configuration.
            mask_embedding: The trained mask embedding, to bake into the
                graph as a constant, so an exported graph carries it rather
                than naming a weight the engine must supply. ``None``
                declares the ``mask_embedding`` weight instead.

        Raises:
            NotImplementedError: If ``data_parallel_degree`` is neither 1
                nor the device count: data parallelism serves one-device
                replicas only, each holding the full drafter.
        """
        if (
            config.data_parallel_degree > 1
            and config.data_parallel_degree != len(config.devices)
        ):
            raise NotImplementedError(
                f"DFlash MiMo-V2 data_parallel_degree "
                f"({config.data_parallel_degree}) must equal the device "
                f"count ({len(config.devices)}); dp x tp meshes are not "
                "supported yet."
            )
        super().__init__(config)
        self.layers = ModuleList(
            [
                DFlashMiMoV2DecoderLayer(config, i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._mask_values: npt.NDArray[np.float32] | None = None
        if mask_embedding is not None:
            data = mask_embedding.data
            assert isinstance(data, Buffer) and data.dtype == DType.bfloat16
            bits = np.from_dlpack(data.view(DType.uint16)).astype(np.uint32)
            # Widened exactly; the graph casts it back to BFloat16.
            self._mask_values = (bits << 16).view(np.float32)
        else:
            self.mask_embedding = Tensor.zeros(
                [config.hidden_size], dtype=config.dtype
            )

    def write_context(
        self,
        taps: Sequence[Tensor],
        input_row_offsets: Tensor,
        kv_collections: Sequence[PagedCacheValues],
    ) -> None:
        """Writes every layer's context K/V from the target's taps.

        Args:
            taps: The target's residual stream after each
                ``target_layer_ids`` layer, in that order, replicated over
                tensor parallelism or sharded per replica under data
                parallelism.
            input_row_offsets: Ragged offsets of the rows, placed like the
                taps.
            kv_collections: The drafter cache, one per device, whose lengths
                are the rows' first positions.
        """
        ctx = self.combine_taps(taps)
        freqs_cis = self._freqs_cis(ctx)
        for layer in self.layers:
            layer.write_context(
                ctx, kv_collections, freqs_cis, input_row_offsets
            )

    def block_embeddings(self, anchor_embeds: Tensor) -> Tensor:
        """Returns the block rows, ``[batch * block_size, hidden]``.

        Args:
            anchor_embeds: The ``[batch, hidden]`` target embeddings of each
                request's anchor token, replicated over tensor parallelism
                or sharded per replica under data parallelism.
        """
        config = self.config
        hidden = config.hidden_size
        outs = []
        for d in range(anchor_embeds.mesh.num_devices):
            rows = anchor_embeds.local_shards[d].shape[0]
            anchor = TensorValue(anchor_embeds.local_shards[d])
            if self._mask_values is None:
                mask = ops.cast(
                    TensorValue(self.mask_embedding.local_shards[d]),
                    anchor.dtype,
                )
            else:
                mask = ops.cast(
                    ops.constant(
                        self._mask_values,
                        DType.float32,
                        device=anchor.device,
                    ),
                    anchor.dtype,
                )
            masks = ops.broadcast_to(
                mask.reshape((1, 1, hidden)),
                (rows, config.block_size - 1, hidden),
            )
            outs.append(
                ops.concat([ops.unsqueeze(anchor, 1), masks], axis=1).reshape(
                    (-1, hidden)
                )
            )
        return Tensor.from_shard_values(outs, anchor_embeds.mapping)

    def forward_block(
        self,
        block_embeds: Tensor,
        kv_collections: Sequence[PagedCacheValues],
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Runs the block, whose K/V lands at the caches' lengths.

        Args:
            block_embeds: :meth:`block_embeddings`'s rows, replicated over
                tensor parallelism or sharded per replica under data
                parallelism.
            kv_collections: Per device, the drafter cache, whose lengths are
                the anchors' positions and whose dispatch metadata is sized
                for the block.
            input_row_offsets: The block's ragged offsets, placed like the
                block rows.

        Returns:
            The final-normed ``[batch * block_size, hidden]`` hidden states,
            placed like ``block_embeds``.
        """
        h = block_embeds
        freqs_cis = self._freqs_cis(h)
        for layer in self.layers:
            h = layer(h, kv_collections, freqs_cis, input_row_offsets)
        return self.norm(h)
