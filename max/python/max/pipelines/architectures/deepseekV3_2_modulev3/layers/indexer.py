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
"""Lightning Indexer layer for DeepseekV3.2, in the ModuleV3 API."""

from __future__ import annotations

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.functional_kernels import (
    mla_fp8_index_top_k,
    quantize_dynamic_scaled_float8,
    rope_ragged,
    store_k_cache_ragged,
    store_k_scale_cache_ragged,
)
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.norm import LayerNorm
from max.experimental.tensor import Tensor
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)

from ...deepseekV3_modulev3.layers.quant_linear import QuantizedLinear


def act_quant(
    x: Tensor, quant_config: QuantConfig, block_size: int = 128
) -> tuple[Tensor, Tensor]:
    *x_dims, head_dim = x.shape
    x = x.reshape((-1, head_dim))
    assert int(head_dim) % block_size == 0

    x_q, x_scales = quantize_dynamic_scaled_float8(
        x,
        quant_config.input_scale,
        quant_config.weight_scale,
        scales_type=DType.float8_e8m0fnu,
        group_size_or_per_token=block_size,
        out_type=DType.float8_e4m3fn,
    )
    # Both outputs take the input's placements; the scales have a different
    # shape than the quantized tensor, which a placements-only mapping allows.
    x_scales = x_scales.rebind_mapping(x.mapping)
    x = x_q.rebind_mapping(x.mapping)
    num_rows = x.shape[0]
    x = x.reshape((*x_dims, head_dim))

    # Scales layout from ``quantize_dynamic_scaled_float8`` is
    # ``[head_dim // block_size, M_padded]``; ``M`` is padded for TMA (16-byte
    # alignment of the scale row length). Slice to ``num_rows``, then fold
    # multiple K-block scale rows into one per token when ``head_dim > block_size``.
    x_scales = x_scales[:, :num_rows]
    num_k_groups = int(head_dim) // block_size
    if num_k_groups > 1:
        x_scales = F.max(x_scales, axis=0)
    x_scales = x_scales.reshape((*x_dims, 1))

    return x, x_scales


def _indexer_act_quant_config(quant_config: QuantConfig) -> QuantConfig:
    """Return the quant config used for dynamic FP8 activation quant in the indexer.

    Full FP8 checkpoints reuse the model quant config. Mixed-precision paths
    (for example NVFP4 MoE with bf16 MLA) still dynamic-quantize indexer
    activations with block size 128.
    """
    if quant_config.format == QuantFormat.BLOCKSCALED_FP8:
        return quant_config
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
            block_size=(1, 128),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float32,
            block_size=(128, 128),
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        format=QuantFormat.BLOCKSCALED_FP8,
    )


class Indexer(
    Module[
        [
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            PagedCacheValues,
            Tensor,
            MHAMaskVariant,
        ],
        Tensor,
    ]
):
    """Selects the top-k keys that sparse MLA attends to."""

    def __init__(
        self,
        dim: int,
        index_n_heads: int,
        index_head_dim: int,
        qk_rope_head_dim: int,
        index_topk: int,
        q_lora_rank: int,
        quant_config: QuantConfig,
        rope_interleaved: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = index_n_heads
        self.head_dim = index_head_dim
        self.rope_head_dim = qk_rope_head_dim
        self.rope_interleaved = rope_interleaved
        # The rotation covers the leading half of each head, so a non-zero
        # rope width must be exactly half of ``index_head_dim``.
        if qk_rope_head_dim != 0 and qk_rope_head_dim * 2 != index_head_dim:
            raise ValueError(
                "indexer rope width must be 0 or half of index_head_dim; got"
                f" qk_rope_head_dim={qk_rope_head_dim} with"
                f" index_head_dim={index_head_dim}"
            )
        self.index_topk = index_topk
        self.q_lora_rank = q_lora_rank
        self.softmax_scale = self.head_dim**-0.5
        self.quant_config = _indexer_act_quant_config(quant_config)

        indexer_weights_fp8 = quant_config.format == QuantFormat.BLOCKSCALED_FP8
        linear_quant_config = quant_config if indexer_weights_fp8 else None

        self.wq_b = QuantizedLinear(
            in_dim=self.q_lora_rank,
            out_dim=self.n_heads * self.head_dim,
            bias=False,
            quant_config=linear_quant_config,
        )  # lora up projection
        self.wk = QuantizedLinear(
            in_dim=self.dim,
            out_dim=self.head_dim,
            bias=False,
            quant_config=linear_quant_config,
        )
        # V2 runs this norm in float32 and casts back, which ``keep_dtype=False``
        # reproduces; the default would reduce in the activation dtype.
        self.k_norm = LayerNorm(dim=self.head_dim, keep_dtype=False)
        self.weights_proj = QuantizedLinear(
            in_dim=self.dim,
            out_dim=self.n_heads,
            bias=False,
        )  # DS casts to f32

    def forward(
        self,
        x: Tensor,
        qr: Tensor,
        freqs_cis: Tensor,
        input_row_offsets: Tensor,
        indexer_k_collection: PagedCacheValues,
        layer_idx: Tensor,
        mask_variant: MHAMaskVariant = MHAMaskVariant.NULL_MASK,
    ) -> Tensor:
        """
        Args:
            x: Tensor of shape (total_seq_len, dim) Input activations.
            qr: Tensor of shape (total_seq_len, q_lora_rank) Pre-normed queries.
            freqs_cis: Tensor of shape (seq_len, head_dim) RoPE frequencies.
            input_row_offsets: Tensor of shape (total_seq_len + 1) Ragged-tensor
                index that tells where each sequence (batch item) starts and
                ends in a concatenated "ragged" input.
            indexer_k_collection: Indexer's K cache values
            layer_idx: Layer index for cache lookup
            mask_variant: Mask applied to the index scores.

        Returns:
            Tensor of shape (total_seq_len, index_topk): indices of the top k
            keys selected by the indexer for MLA to attend to.
        """
        # qr comes projected to lora rank and pre-normed;
        # q_lora_rank -> self.n_heads * self.head_dim
        q = self.wq_b(qr)
        q = q.reshape((-1, self.n_heads, self.head_dim))
        if self.rope_head_dim:
            q_pe, q_nope = F.chunk(q, chunks=2, axis=-1)
            q_pe = rope_ragged(
                q_pe,
                input_row_offsets,
                indexer_k_collection.cache_lengths,
                freqs_cis,
                interleaved=self.rope_interleaved,
            ).rebind_mapping(q_pe.mapping)
            q = F.concat([q_pe, q_nope], axis=-1)

        k = self.wk(x)  # dim -> head_dim
        k = self.k_norm(k)

        if self.rope_head_dim:
            k_pe, k_nope = F.chunk(k, chunks=2, axis=-1)
            k_pe = F.squeeze(
                rope_ragged(
                    F.unsqueeze(k_pe, axis=-2),
                    input_row_offsets,
                    indexer_k_collection.cache_lengths,
                    freqs_cis,
                    interleaved=self.rope_interleaved,
                ).rebind_mapping(k_pe.mapping),
                axis=-2,
            )
            k = F.concat([k_pe, k_nope], axis=-1)

        q_fp8, q_scale = act_quant(q, self.quant_config)
        k_fp8, k_scale = act_quant(k, self.quant_config)

        store_k_cache_ragged(
            indexer_k_collection,
            F.unsqueeze(k_fp8, axis=1),
            input_row_offsets,
            layer_idx,
        )
        store_k_scale_cache_ragged(
            indexer_k_collection,
            F.cast(F.unsqueeze(k_scale, axis=1), DType.float32),
            input_row_offsets,
            layer_idx,
            quantization_granularity=self.quant_config.scales_granularity_mnk[
                2
            ],
        )

        weights = (
            self.weights_proj(F.cast(x, DType.float32)) * self.n_heads**-0.5
        )  # dim -> n_heads
        weights = F.unsqueeze(weights, axis=-1) * q_scale * self.softmax_scale

        return mla_fp8_index_top_k(
            q_fp8,
            F.squeeze(weights, axis=-1),
            input_row_offsets,
            indexer_k_collection,
            layer_idx,
            self.index_topk,
            self.quant_config.scales_granularity_mnk[2],
            mask_variant,
        ).rebind_mapping(q_fp8.mapping)
