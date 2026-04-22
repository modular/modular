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
"""Lightning Indexer layer for DeepseekV3.2."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn import (
    LayerNorm,
    Linear,
    Module,
    QuantConfig,
)
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import (
    mla_fp8_index_top_k,
    quantize_dynamic_scaled_float8,
    rope_ragged,
    store_k_cache_ragged,
    store_k_scale_cache_ragged,
)
from max.nn.kv_cache import PagedCacheValues

from .transforms import HadamardTransform


def act_quant(
    x: TensorValue, float8_config: QuantConfig, block_size: int = 128
) -> tuple[TensorValue, TensorValue]:
    *x_dims, head_dim = x.shape
    x = x.reshape((-1, head_dim))
    assert int(head_dim) % block_size == 0

    x, x_scales = quantize_dynamic_scaled_float8(
        x,
        float8_config.input_scale,
        float8_config.weight_scale,
        scales_type=DType.float8_e8m0fnu,
        group_size_or_per_token=block_size,
        out_type=DType.float8_e4m3fn,
    )

    x = x.reshape((*x_dims, head_dim))

    # We must rebind -> reshape to teach the shaper resolver the correct
    # shape of the scales tensor.
    x_scales = x_scales.rebind((1, prod(x_dims)))
    x_scales = x_scales.reshape((*x_dims, 1))

    return x, x_scales


class Indexer(Module):
    def __init__(
        self,
        dim: int,
        index_n_heads: int,
        index_head_dim: int,
        qk_rope_head_dim: int,
        index_topk: int,
        q_lora_rank: int,
        devices: Sequence[DeviceRef],
        float8_config: QuantConfig,
    ):
        super().__init__()
        self.dim: int = dim
        self.n_heads: int = index_n_heads
        self.n_local_heads: int = index_n_heads // len(devices)
        self.head_dim: int = index_head_dim
        self.rope_head_dim: int = qk_rope_head_dim
        self.index_topk: int = index_topk
        self.q_lora_rank: int = q_lora_rank
        self.softmax_scale = self.head_dim**-0.5
        self.float8_config = float8_config

        self.wq_b = Linear(
            in_dim=self.q_lora_rank,
            out_dim=self.n_heads * self.head_dim,
            dtype=DType.float8_e4m3fn,
            device=devices[0],
        )  # lora up projection
        self.wk = Linear(
            in_dim=self.dim,
            out_dim=self.head_dim,
            dtype=DType.float8_e4m3fn,
            device=devices[0],
        )
        self.k_norm = LayerNorm(
            dims=self.head_dim, dtype=DType.float32, devices=devices
        )
        self.weights_proj = Linear(
            in_dim=self.dim,
            out_dim=self.n_heads,
            dtype=DType.bfloat16,
            device=devices[0],
        )  # DS casts to f32

        self.hadamard_transform = HadamardTransform(
            scale=self.softmax_scale, device=devices[0]
        )

    def __call__(
        self,
        x: TensorValue,
        qr: TensorValue,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
        indexer_k_collection: PagedCacheValues,
        layer_idx: TensorValue,
        mask_variant: MHAMaskVariant = MHAMaskVariant.NULL_MASK,
    ) -> TensorValue:
        """
        Args:
            x: Tensor of shape (total_seq_len, dim) Input activations.
            qr: Tensor of shape (total_seq_len, q_lora_rank) Pre-normed queries.
            start_pos: Tensor scalar. Used to slice the freqs_cis tensor.
            freqs_cis: Tensor of shape (seq_len, head_dim) RoPE frequencies.
            input_row_offsets: Tensor of shape (total_seq_len + 1) Ragged-tensor
                index that tells where each sequence (batch item) starts and
                ends in a concatenated “ragged” input.
            indexer_k_collection: Indexer's K cache values
            layer_idx: Layer index for cache lookup
        Returns:
            topk_indices, tTensor of shape (total_seq_len, index_topk) indices
                of the top k Keys selected by the Indexer for MLA to attend to.
        """
        # qr comes projected to lora rank and pre-normed; q_lora_rank -> self.n_heads * self.head_dim
        q = self.wq_b(qr)
        q = q.reshape((-1, self.n_heads, self.head_dim))
        assert self.rope_head_dim == self.head_dim - self.rope_head_dim
        q_pe, q_nope = ops.chunk(q, chunks=2, axis=-1)

        q_pe = rope_ragged(
            q_pe,
            input_row_offsets,
            indexer_k_collection.cache_lengths,
            freqs_cis,
            interleaved=False,
        )
        q = ops.concat([q_pe, q_nope], axis=-1)

        k = self.wk(x)  # dim -> head_dim
        k = self.k_norm(k)

        assert self.rope_head_dim == self.head_dim - self.rope_head_dim
        k_pe, k_nope = ops.chunk(k, chunks=2, axis=-1)
        k_pe = ops.squeeze(
            rope_ragged(
                ops.unsqueeze(k_pe, axis=-2),
                input_row_offsets,
                indexer_k_collection.cache_lengths,
                freqs_cis,
                interleaved=False,
            ),
            axis=-2,
        )
        k = ops.concat([k_pe, k_nope], axis=-1)

        q = self.hadamard_transform(q)
        k = self.hadamard_transform(k)

        q_fp8, q_scale = act_quant(q, self.float8_config)
        k_fp8, k_scale = act_quant(k, self.float8_config)

        store_k_cache_ragged(
            indexer_k_collection,
            ops.unsqueeze(k_fp8, axis=1),
            input_row_offsets,
            layer_idx,
        )
        store_k_scale_cache_ragged(
            indexer_k_collection,
            ops.unsqueeze(k_scale, axis=1).cast(DType.float32),
            input_row_offsets,
            layer_idx,
            quantization_granularity=self.float8_config.scales_granularity_mnk[
                2
            ],
        )

        weights = (
            self.weights_proj(x.cast(DType.float32)) * self.n_heads**-0.5
        )  # dim -> n_heads
        weights = ops.unsqueeze(weights, axis=-1) * q_scale * self.softmax_scale

        return mla_fp8_index_top_k(
            q_fp8,
            ops.squeeze(weights, axis=-1),
            input_row_offsets,
            indexer_k_collection,
            layer_idx,
            self.index_topk,
            self.float8_config.scales_granularity_mnk[2],
            mask_variant,
        )
