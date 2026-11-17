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
"""The decoder's attention: per-head Q/K RMS norms and M-RoPE."""

from __future__ import annotations

import functools

from max import functional as F
from max.dtype import DType
from max.nn import Module
from max.nn.attention_mask import MHAMaskVariant
from max.nn.common_layers import quant_ops
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.mesh_axis import TP
from max.nn.common_layers.quant_linear import QuantizedLinear
from max.nn.common_layers.quant_tensor import FP8BlockTensor, QuantAwareTensor
from max.nn.kernels import flash_attention_ragged, rope_split_store_ragged
from max.nn.kv_cache import KVCacheParams
from max.nn.quant_config import QuantConfig
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, NamedMapping, Partial

from .norm import RMSNorm

__all__ = ["Qwen3VLAttention"]


def _rows(weight: QuantAwareTensor) -> int:
    """The output width of a ``[out, in]`` projection weight."""
    if isinstance(weight, FP8BlockTensor):
        return int(weight.weight.shape[0])
    assert isinstance(weight, Tensor)
    return int(weight.shape[0])


def _projection(
    in_dim: int,
    out_dim: int,
    quant_config: QuantConfig | None,
    sharding: tuple[str | None, str | None],
) -> QuantizedLinear:
    """A bias-free projection whose weight is split by ``sharding``."""
    linear = QuantizedLinear(
        in_dim, out_dim, bias=False, quant_config=quant_config
    )
    linear.weight = linear.weight.to(NamedMapping(linear.weight.mesh, sharding))
    return linear


def _attention_core(
    x: Tensor,
    qkv_weights: tuple[QuantAwareTensor, QuantAwareTensor, QuantAwareTensor],
    o_weight: QuantAwareTensor,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    kv_collection: PagedCacheValues,
    layer_idx: Tensor,
    freqs_cis: Tensor,
    input_row_offsets: Tensor,
    *,
    kv_params: KVCacheParams,
    head_dim: int,
    scale: float,
) -> Tensor:
    """Runs one device's heads and returns its ``o_proj`` partial sums."""
    total_seq_len = x.shape[0]
    q_weight, k_weight, v_weight = qkv_weights
    q_dim = _rows(q_weight)
    kv_dim = _rows(k_weight)
    n_heads = q_dim // head_dim
    n_kv_heads = kv_dim // head_dim

    qkv = quant_ops.matmul(
        x, quant_ops.concat_weights(q_weight, k_weight, v_weight, axis=0)
    )
    x_q, x_k, x_v = F.split(qkv, [q_dim, kv_dim, kv_dim], axis=-1)
    # Each head's Q and K are normed before RoPE, as the reference does.
    x_q = q_norm(x_q.reshape([-1, n_heads, head_dim])).reshape([-1, q_dim])
    x_k = k_norm(x_k.reshape([-1, n_kv_heads, head_dim])).reshape([-1, kv_dim])
    qkv = F.concat([x_q, x_k, x_v], axis=-1)

    # The table is per token, so each token reads its own row rather than
    # its cache position.
    freqs_cis = freqs_cis.cast(qkv.dtype)
    position_ids = F.range(
        0, total_seq_len, 1, device=qkv.device, dtype=DType.uint32
    ).unsqueeze(0)
    query = rope_split_store_ragged(
        kv_params,
        qkv,
        input_row_offsets,
        freqs_cis,
        kv_collection,
        layer_idx,
        n_heads=n_heads,
        interleaved=False,
        position_ids=position_ids,
    )
    query = query.reshape([-1, n_heads, head_dim])
    attn_out = flash_attention_ragged(
        kv_params,
        input=query,
        kv_collection=kv_collection,
        layer_idx=layer_idx,
        input_row_offsets=input_row_offsets,
        mask_variant=MHAMaskVariant.CAUSAL_MASK,
        scale=scale,
    )
    return quant_ops.matmul(attn_out.reshape([total_seq_len, -1]), o_weight)


class Qwen3VLAttention(Module[..., Tensor]):
    """Full causal attention, split by head across devices.

    Returns ``o_proj``'s partial sums, which the block reduces.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        kv_params: KVCacheParams,
        norm_eps: float,
        quant_config: QuantConfig | None,
    ) -> None:
        self.kv_params = kv_params
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        q_dim = head_dim * num_attention_heads
        kv_dim = head_dim * num_key_value_heads

        # Each device takes its heads' rows of Q, K and V, and the matching
        # columns of the output projection.
        self.q_proj = _projection(hidden_size, q_dim, quant_config, (TP, None))
        self.k_proj = _projection(hidden_size, kv_dim, quant_config, (TP, None))
        self.v_proj = _projection(hidden_size, kv_dim, quant_config, (TP, None))
        self.o_proj = _projection(q_dim, hidden_size, quant_config, (None, TP))
        self.q_norm = RMSNorm(
            head_dim,
            eps=norm_eps,
            dtype=kv_params.dtype,
            multiply_before_cast=True,
        )
        self.k_norm = RMSNorm(
            head_dim,
            eps=norm_eps,
            dtype=kv_params.dtype,
            multiply_before_cast=True,
        )

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        layer_idx: Tensor,
        freqs_cis: Tensor,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Attends over the whole causal context.

        Args:
            x: The normed hidden states, ``[total_seq_len, hidden_size]``.
            kv_collection: The decoder's cache.
            layer_idx: This layer's index in the cache, a host scalar.
            freqs_cis: The per-token M-RoPE table, ``[total_seq_len,
                head_dim]``.
            input_row_offsets: The ragged row offsets.
        """
        core = functools.partial(
            _attention_core,
            kv_params=self.kv_params,
            head_dim=self.head_dim,
            scale=self.scale,
        )
        return F.call_on_mesh(
            core,
            x.mesh,
            out_specs=DeviceMapping(x.mesh, (Partial(),)),
        )(
            x,
            (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight),
            self.o_proj.weight,
            self.q_norm,
            self.k_norm,
            kv_collection,
            layer_idx,
            freqs_cis,
            input_row_offsets,
        )
