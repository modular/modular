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

"""Gated sparse MLA with learnable sinks for Hy4 (ModuleV3).

DeepSeek V3.2's sparse MLA at Hy4's widths (``qk_nope 192``, ``v_head 256``,
latent 576), with two additions: one learned softmax logit per head that takes
probability mass and contributes no value (the kernels' ``sparse_attn_sink``),
and an elementwise ``sigmoid(linear_gate(x))`` on the attention output before
``o_proj``.

Only the sparse kernels run: ``mla_decode_graph`` or ``mla_prefill_decode_graph``
with the indexer's top-k. The dense ``mla_prefill_graph`` takes no sink.
"""

from __future__ import annotations

import math
from typing import Any

from max.driver import CPU, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Linear, Module
from max.experimental.nn.common_layers import quant_ops
from max.experimental.nn.common_layers.functional_kernels import (
    flare_mla_prefill_plan,
    mla_decode_graph,
    mla_prefill_decode_graph,
)
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.nn.common_layers.quant_linear import QuantizedLinear
from max.experimental.nn.norm import RMSNorm
from max.experimental.sharding import DeviceMapping, DeviceMesh, NamedMapping
from max.experimental.tensor import Tensor
from max.nn.attention import MHAMaskVariant
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams
from typing_extensions import Self

from ..model_config import HYV4Config
from .indexer import HYV4Indexer

__all__ = ["HYV4Attention"]

_EPS = 1e-6


def _neox_to_interleaved(x: Tensor, rope_dim: int) -> Tensor:
    """Reorders the trailing ``rope_dim`` columns ``[a.., b..]`` to
    ``[a0, b0, a1, b1, ..]``, so an interleaved rotation applies NeoX's."""
    nope, rope = F.split(x, [int(x.shape[-1]) - rope_dim, rope_dim], axis=-1)
    lead = list(rope.shape[:-1])
    rope = rope.reshape([*lead, 2, rope_dim // 2]).transpose(-1, -2)
    return F.concat([nope, rope.reshape([*lead, rope_dim])], axis=-1)


class HYV4Attention(Module[..., tuple[Tensor, Tensor]]):
    """The attention block of decoder layer ``layer_idx``.

    Args:
        config: The model config; ``config.kv_params`` holds the ``mla`` and
            ``indexer`` caches, and ``config.quant_config`` decides whether
            the MLA projections, ``o_proj`` and the indexer's ``wq_b``/``wk``
            are quantized. ``kv_b_proj`` and ``linear_gate`` are BF16 either
            way.
        layer_idx: The decoder layer, which picks its indexer kind.
        graph_mode: ``"auto"`` runs ``mla_prefill_decode_graph``, which picks
            the sparse prefill or decode kernel per batch; ``"decode"`` runs
            ``mla_decode_graph`` for every batch.
        rope_interleave: The checkpoint's rope convention. The kernels rotate
            interleaved pairs; ``False`` reorders the rope columns so they
            rotate NeoX halves instead.
    """

    def __init__(
        self,
        config: HYV4Config,
        layer_idx: int,
        *,
        graph_mode: str = "auto",
        rope_interleave: bool = True,
    ) -> None:
        super().__init__()
        if graph_mode not in ("auto", "decode"):
            raise ValueError(
                f"graph_mode must be 'auto' or 'decode', got {graph_mode!r}:"
                " the dense prefill kernel has no sink operand"
            )
        assert isinstance(config.kv_params, MultiKVCacheParams)
        kv_params = config.kv_params.children["mla"]
        assert isinstance(kv_params, KVCacheParams)
        self.kv_params = kv_params
        self.max_batch_context_length = config.max_batch_context_length
        self.graph_mode = graph_mode
        self.rope_interleave = rope_interleave
        self.n_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_head_dim
        self.v_head_dim = config.v_head_dim
        self.cache_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.scale = 1.0 / math.sqrt(self.qk_head_dim)

        hidden = config.hidden_size
        q_dim = self.n_heads * self.qk_head_dim
        o_dim = self.n_heads * self.v_head_dim

        def weight(out_dim: int, in_dim: int) -> quant_ops.QuantAwareTensor:
            return quant_ops.quantized_weight(
                out_dim, in_dim, config.quant_config
            )

        self.q_a_proj = weight(self.q_lora_rank, hidden)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=_EPS)
        self.q_b_proj = weight(q_dim, self.q_lora_rank)
        self.kv_a_proj_with_mqa = weight(self.cache_head_dim, hidden)
        self.kv_a_proj_layernorm = Tensor.ones([self.kv_lora_rank])
        self.kv_b_proj = Tensor.zeros(
            [
                self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
                self.kv_lora_rank,
            ]
        )
        self.o_proj = QuantizedLinear(
            o_dim, hidden, bias=False, quant_config=config.quant_config
        )
        self.linear_gate = Linear(hidden, o_dim, bias=False)
        self.learnable_sink_param = Tensor.zeros(
            [self.n_heads], dtype=DType.float32
        )
        self.indexer = HYV4Indexer(config, layer_idx)

    def to(self, target: Device | DeviceMesh | DeviceMapping) -> Self:
        """Moves the layer and its quantized projections to ``target`` in place."""
        super().to(target)
        # A quantized weight's ``to`` returns a copy, which ``Module.to``
        # drops for child modules.
        self.q_a_proj = self.q_a_proj.to(target)
        self.q_b_proj = self.q_b_proj.to(target)
        self.kv_a_proj_with_mqa = self.kv_a_proj_with_mqa.to(target)
        return self

    def shard_heads(self) -> None:
        """Splits the heads over the mesh (tensor parallel), as R3 does.

        ``q_b_proj``, ``kv_b_proj``, ``linear_gate`` and the sinks are split by
        head and ``o_proj`` along its input; the down-projections, norms and
        the indexer stay replicated, since every device needs the full
        latent, its own cache copy and the shared top-k.
        """

        mesh = self.kv_b_proj.mesh
        rows = NamedMapping(mesh, (TP, None))
        self.q_b_proj = self.q_b_proj.to(rows)
        self.kv_b_proj = self.kv_b_proj.to(rows)
        self.o_proj.weight = self.o_proj.weight.to(
            NamedMapping(mesh, (None, TP))
        )
        self.linear_gate.weight = self.linear_gate.weight.to(rows)
        self.learnable_sink_param = self.learnable_sink_param.to(
            NamedMapping(mesh, (TP,))
        )

    # The absorbed kv_b_proj views, as LatentAttentionWithRope builds them.
    @property
    def _kv_b(self) -> Tensor:
        return self.kv_b_proj.T.reshape([self.kv_lora_rank, self.n_heads, -1])

    @property
    def w_uk(self) -> Tensor:
        """``[heads, kv_lora_rank, qk_nope]``."""
        return self._kv_b[..., : self.qk_nope_head_dim].transpose(0, 1)

    @property
    def w_uv(self) -> Tensor:
        """``[heads, v_head, kv_lora_rank]``."""
        return self._kv_b[..., self.qk_nope_head_dim :].permute([1, 2, 0])

    @property
    def w_k(self) -> Tensor:
        """``[heads * qk_nope, kv_lora_rank]``."""
        return (
            self._kv_b[..., : self.qk_nope_head_dim]
            .permute([1, 2, 0])
            .reshape([-1, self.kv_lora_rank])
        )

    def _prefill_kwargs(
        self,
        input_row_offsets: Tensor,
        kv_collection: PagedCacheValues,
        buffer_length: Tensor,
    ) -> dict[str, Any]:
        buffer_row_offsets, cache_offsets, _ = flare_mla_prefill_plan(
            self.kv_params,
            input_row_offsets,
            kv_collection,
            F.constant(0, DType.uint32, device=CPU()),
            self.max_batch_context_length,
            max_chunks=1,
        )
        return {
            "buffer_row_offsets": buffer_row_offsets,
            "cache_offsets": cache_offsets,
            "buffer_length": buffer_length,
            "w_k": self.w_k,
        }

    def forward(
        self,
        x: Tensor,
        layer_idx: Tensor,
        indexer_layer: Tensor,
        kv_collection: PagedCacheValues,
        indexer_kv_collection: PagedCacheValues,
        freqs_cis: Tensor,
        input_row_offsets: Tensor,
        prev_topk_indices: Tensor | None,
        buffer_length: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Returns the block output and the top-k positions it attended.

        Args:
            x: ``[tokens, hidden_size]`` normalized hidden states.
            layer_idx: ``[]`` uint32 on the host, the layer's MLA cache slot.
                An input rather than a constant, so layers can share one
                subgraph.
            indexer_layer: The same for the indexer K cache; a ``shared``
                layer does not read it.
            kv_collection: The BF16 MLA latent cache.
            indexer_kv_collection: The FP8 indexer K cache.
            freqs_cis: ``[positions, qk_rope_head_dim]`` interleaved cos/sin.
            input_row_offsets: ``[batch + 1]`` ragged row offsets.
            prev_topk_indices: The previous full layer's top-k, read by a
                ``shared`` layer.
            buffer_length: ``[1]`` int32 on the host, the batch's prefill
                buffer length as the plan kernel computes it; the graph never
                reads the plan's own back, which capture would reject.
        """
        q_resid = self.q_a_layernorm(quant_ops.matmul(x, self.q_a_proj))
        q = quant_ops.matmul(q_resid, self.q_b_proj)
        kv = quant_ops.matmul(x, self.kv_a_proj_with_mqa)
        if not self.rope_interleave:
            q = _neox_to_interleaved(
                q.reshape([-1, self.n_heads, self.qk_head_dim]),
                self.qk_rope_head_dim,
            )
            kv = _neox_to_interleaved(kv, self.qk_rope_head_dim)
        q = q.reshape([q.shape[0], self.n_heads, self.qk_head_dim])

        topk = self.indexer(
            x,
            q_resid,
            freqs_cis,
            input_row_offsets,
            indexer_kv_collection,
            indexer_layer,
            prev_topk_indices,
        )

        assert kv_collection.attention_dispatch_metadata is not None
        assert kv_collection.mla_num_partitions is not None
        kwargs: dict[str, Any] = dict(
            q=q,
            kv=kv,
            input_row_offsets=input_row_offsets,
            freqs_cis=freqs_cis,
            kv_norm_gamma=self.kv_a_proj_layernorm,
            w_uk=self.w_uk,
            w_uv=self.w_uv,
            kv_params=self.kv_params,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
            epsilon=_EPS,
            v_head_dim=self.v_head_dim,
            scalar_args=kv_collection.attention_dispatch_metadata,
            num_partitions_scalar=kv_collection.mla_num_partitions,
            sparse_indices=topk,
            # The indexer's top-k holds logical token positions, which the
            # kernels look up in the page table, not physical cache rows.
            physical_indices=False,
            sparse_attn_sink=self.learnable_sink_param,
            sparse_indices_stride=self.index_topk,
        )
        if self.graph_mode == "decode":
            attn = mla_decode_graph(**kwargs)
        else:
            attn = mla_prefill_decode_graph(
                **kwargs,
                **self._prefill_kwargs(
                    input_row_offsets, kv_collection, buffer_length
                ),
            )
        # The kernels run per device on its heads.
        attn = attn.rebind_mapping(q.mapping)
        attn = attn.reshape([attn.shape[0], self.n_heads * self.v_head_dim])
        out = self.o_proj(attn * F.sigmoid(self.linear_gate(x)))
        if out.mesh.num_devices > 1:
            out = F.allreduce_sum(out)
        return out, topk
