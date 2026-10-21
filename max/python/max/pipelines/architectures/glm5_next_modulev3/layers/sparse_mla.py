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
"""NoPE sparse MLA for GLM-5.3-Flash's 11 sparse layers, on ModuleV3.

GLM-5.2's block with two changes, which the graph-API layer
(:mod:`max.pipelines.architectures.glm5_next.layers.sparse_mla`) explains in
full:

* **NoPE.** The latent is 512 wide, but every MLA kernel compiles against 576,
  so the query and the latent are zero-padded to that geometry on the way into
  the kernel. That is exact: the padded columns are zero in both operands.
* **k-pooled indexing.** :class:`~.indexer.PooledIndexer` selects 2051 token
  positions per query, which the kernel reads as logical positions with the
  k-pool layout.

``kv_b_proj`` is BF16 in this checkpoint while the other projections are FP8,
so the absorbed ``w_uk`` / ``w_uv`` carry no scales and the kernel takes no
quant config.
"""

from __future__ import annotations

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn.common_layers import quant_ops
from max.experimental.nn.common_layers.functional_kernels import (
    mla_decode_graph,
)
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.quant_mla import (
    QuantizedLatentAttention,
    tensor_parallel_latent_attention_with_rope,
)
from max.experimental.tensor import Tensor
from max.nn.attention import MHAMaskVariant
from max.nn.quant_config import QuantConfig

from ..model_config import Glm5NextConfig
from .indexer import KPoolTail, PooledIndexer

__all__ = ["Glm5NextSparseMLA"]


def _zero_pad_last(x: Tensor, width: int) -> Tensor:
    """Appends ``width`` zero columns to ``x``'s last axis."""
    if width == 0:
        return x
    return F.pad(x, [0, 0] * (x.rank - 1) + [0, width])


class Glm5NextSparseMLA(QuantizedLatentAttention):
    """Sparse MLA plus the k-pooled indexer, tensor-parallel over heads.

    Returns ``o_proj``'s partial sums; the decoder layer reduces them.
    """

    def __init__(
        self,
        config: Glm5NextConfig,
        layer_idx: int,
        quant_config: QuantConfig | None,
    ) -> None:
        super().__init__(
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_size=config.hidden_size,
            kv_params=config.mla_kv_params,
            layer_idx=layer_idx,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=0,
            v_head_dim=config.v_head_dim,
            graph_mode="decode",
            quant_config=quant_config,
        )
        if self.kv_params.head_dim != config.mla_latent_pad_to:
            raise ValueError(
                f"The 'mla' cache leaf is {self.kv_params.head_dim} wide but "
                f"the padded latent is {config.mla_latent_pad_to}; the kernel "
                "reads both as one geometry."
            )
        self.latent_pad = config.mla_latent_pad_to - config.kv_lora_rank
        # BF16 in the checkpoint even though the layer's other projections
        # are FP8.
        self.kv_b_proj = quant_ops.quantized_weight(
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            self.kv_lora_rank,
            None,
        )
        assert quant_config is not None
        self.indexer = PooledIndexer(config, quant_config)
        tensor_parallel_latent_attention_with_rope(self)

    @property
    def w_uk(self) -> tuple[Tensor, Tensor | None]:
        w = self._kv_b_proj_weight[..., : self.qk_nope_head_dim]
        return w.transpose(0, 1), None

    @property
    def w_uv(self) -> tuple[Tensor, Tensor | None]:
        w = self._kv_b_proj_weight[..., self.qk_nope_head_dim :]
        return w.permute([1, 2, 0]), None

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        indexer_kv_collection: PagedCacheValues,
        layer_idx: Tensor,
        input_row_offsets: Tensor,
        freqs_cis: Tensor,
        tail: KPoolTail,
    ) -> Tensor:
        if kv_collection.kv_scales is not None:
            raise ValueError(
                "An FP8 latent cache with a BF16 kv_b_proj is not reachable: "
                "the FP8 MLA op requires absorbed weight scales this "
                "checkpoint does not ship. Run the BF16 latent cache, which is "
                "what the accuracy gates use, until that pairing is supported."
            )
        assert self.q_lora_rank is not None
        qkv = quant_ops.matmul(x, self.wqkv)
        q_a_out, kv = qkv.split([self.q_lora_rank, self.cache_head_dim], axis=1)
        q_a_normed = self.q_a_layernorm(q_a_out)
        selection = self.indexer(
            x,
            q_a_normed,
            input_row_offsets,
            indexer_kv_collection,
            layer_idx,
            tail,
        )

        xq = quant_ops.matmul(q_a_normed, self.q_b_proj)
        xq = xq.reshape((xq.shape[0], self.n_heads, self.qk_head_dim))
        xq = _zero_pad_last(xq, self.latent_pad)
        kv = _zero_pad_last(kv, self.latent_pad)

        w_uk, _ = self.w_uk
        w_uv, _ = self.w_uv
        width = self.indexer.selection_width
        assert kv_collection.attention_dispatch_metadata is not None
        assert kv_collection.mla_num_partitions is not None
        attn_out = mla_decode_graph(
            q=xq,
            kv=kv,
            input_row_offsets=input_row_offsets,
            freqs_cis=F.cast(freqs_cis, xq.dtype),
            kv_norm_gamma=self.kv_a_proj_layernorm,
            w_uk=w_uk,
            w_uv=w_uv,
            kv_params=self.kv_params,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
            epsilon=1e-6,
            v_head_dim=self.v_head_dim,
            scalar_args=kv_collection.attention_dispatch_metadata,
            num_partitions_scalar=kv_collection.mla_num_partitions,
            sparse_indices=selection,
            physical_indices=False,
            sparse_topk_lengths=F.full(
                [xq.shape[0]],
                width,
                dtype=DType.int32,
                device=selection.mapping,
            ),
            sparse_indices_stride=width,
            # A pooled row holds fewer valid entries than its 2051 slots once
            # every pool is selected; the kernel masks the padding from this.
            index_kpool=self.indexer.kpool,
            index_pool_topk=self.indexer.pools_selected,
            index_kpool_select_tail=self.indexer.always_select_tail,
        ).rebind_mapping(xq.mapping)
        attn_out = attn_out.reshape(
            (attn_out.shape[0], self.n_heads * self.v_head_dim)
        )
        return self.o_proj(attn_out)
