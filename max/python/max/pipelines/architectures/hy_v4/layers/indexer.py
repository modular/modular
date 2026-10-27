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

"""DeepSeek Sparse Attention (DSA) lightning indexer for Hy4 (ModuleV3).

The DeepSeek V3.2 indexer without the Hadamard rotation, plus a ``k_norm``
bias and Hy4's IndexCache: a ``shared`` layer has no indexer of its own and
reuses the top-k of the most recent ``full`` layer.

The weight adapter moves the 64 rope dims of every ``wq_b`` head, ``wk`` and
``k_norm`` to the front of the head, so the rotation covers the leading dims.
"""

from __future__ import annotations

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Linear, Module
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.quant_kernels import (
    quantize_dynamic_scaled_float8,
)
from max.experimental.nn.common_layers.quant_linear import QuantizedLinear
from max.experimental.nn.norm import LayerNorm
from max.experimental.tensor import Tensor
from max.nn.attention import MHAMaskVariant
from max.nn.quant_config import (
    InputScaleSpec,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)

from ..model_config import HYV4Config
from .functional_kernels import (
    mla_fp8_index_top_k,
    rope_ragged,
    store_k_cache_ragged,
    store_k_scale_cache_ragged,
)

__all__ = ["HYV4Indexer"]

# One block per head: the K cache keeps one scale per token.
_INPUT_SPEC = InputScaleSpec(
    granularity=ScaleGranularity.BLOCK,
    origin=ScaleOrigin.DYNAMIC,
    dtype=DType.float32,
    block_size=(1, 128),
)
_WEIGHT_SPEC = WeightScaleSpec(
    granularity=ScaleGranularity.BLOCK,
    dtype=DType.float32,
    block_size=(128, 128),
)


def _quantize_rows(x: Tensor) -> tuple[Tensor, Tensor]:
    """Quantizes each last-axis row to E4M3 with one power-of-two scale."""
    rows = x.reshape([-1, x.shape[-1]])
    data, scales = quantize_dynamic_scaled_float8(
        rows,
        _INPUT_SPEC,
        _WEIGHT_SPEC,
        group_size_or_per_token=int(x.shape[-1]),
        scales_type=DType.float8_e8m0fnu,
        out_type=DType.float8_e4m3fn,
    )
    # The scale row is padded for TMA alignment.
    scales = F.cast(scales[0, : rows.shape[0]], DType.float32)
    return data.reshape(x.shape), scales.reshape(x.shape[:-1])


class HYV4Indexer(Module[..., Tensor]):
    """The indexer of decoder layer ``layer_idx``.

    A ``full`` layer scores every cached key with ``relu(q . k)`` summed over
    heads by ``weights_proj(x)`` and returns the ``index_topk`` best token
    positions per query row (``-1`` past the end of a short row). Its keys go
    to the FP8 indexer K cache at the slot the caller passes.
    A ``shared`` layer (``skip_topk``) declares no weights and returns the
    ``prev_topk_indices`` it is given.

    Args:
        config: The model config; ``config.quant_config`` decides whether
            ``wq_b`` and ``wk`` are quantized.
        layer_idx: The decoder layer this indexer belongs to.
    """

    def __init__(self, config: HYV4Config, layer_idx: int) -> None:
        super().__init__()
        self.skip_topk = config.indexer_types[layer_idx] != "full"
        if self.skip_topk:
            return
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk

        q_dim = self.n_heads * self.head_dim
        self.wq_b = QuantizedLinear(
            config.q_lora_rank,
            q_dim,
            bias=False,
            quant_config=config.quant_config,
        )
        self.wk = QuantizedLinear(
            config.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=config.quant_config,
        )
        self.k_norm = LayerNorm(
            self.head_dim, eps=config.rms_norm_eps, keep_dtype=False
        )
        self.weights_proj = Linear(config.hidden_size, self.n_heads, bias=False)

    def forward(
        self,
        x: Tensor,
        qr: Tensor,
        freqs_cis: Tensor,
        input_row_offsets: Tensor,
        k_cache: PagedCacheValues,
        cache_layer: Tensor,
        prev_topk_indices: Tensor | None = None,
    ) -> Tensor:
        """Returns ``[tokens, index_topk]`` int32 token positions.

        Args:
            x: ``[tokens, hidden_size]`` normalized hidden states.
            qr: ``[tokens, q_lora_rank]`` output of ``q_a_layernorm``.
            freqs_cis: ``[positions, qk_rope_head_dim]`` interleaved
                cos/sin table.
            input_row_offsets: ``[batch + 1]`` ragged row offsets.
            k_cache: The FP8 indexer K cache.
            cache_layer: ``[]`` uint32 on the host, the layer's slot in
                ``k_cache``.
            prev_topk_indices: The previous ``full`` layer's result; read
                only by a ``shared`` layer.
        """
        if self.skip_topk:
            if prev_topk_indices is None:
                raise ValueError(
                    "a shared indexer layer needs the previous full layer's"
                    " top-k indices"
                )
            return prev_topk_indices

        # The kernels run per device; under tensor parallelism every device
        # computes the same replicated result.
        def rope(t: Tensor) -> Tensor:
            return rope_ragged(
                t,
                input_row_offsets,
                k_cache.cache_lengths,
                freqs_cis,
                interleaved=True,
                rope_first=True,
            ).rebind_mapping(t.mapping)

        q = self.wq_b(qr).reshape([-1, self.n_heads, self.head_dim])
        q_fp8, q_scale = _quantize_rows(rope(q))
        k = rope(self.k_norm(self.wk(x)).unsqueeze(1))
        k_fp8, k_scale = _quantize_rows(k)

        store_k_cache_ragged(k_cache, k_fp8, input_row_offsets, cache_layer)
        store_k_scale_cache_ragged(
            k_cache,
            k_scale.unsqueeze(-1),
            input_row_offsets,
            cache_layer,
            quantization_granularity=self.head_dim,
        )

        # The scoring weights are projected in float32.
        proj = F.cast(self.weights_proj.weight, DType.float32)
        weights = (
            (F.cast(x, DType.float32) @ proj.T)
            * self.n_heads**-0.5
            * q_scale
            * self.head_dim**-0.5
        )
        return mla_fp8_index_top_k(
            q_fp8,
            weights,
            input_row_offsets,
            k_cache,
            cache_layer,
            self.index_topk,
            self.head_dim,
            MHAMaskVariant.CAUSAL_MASK,
        ).rebind_mapping(x.mapping)
