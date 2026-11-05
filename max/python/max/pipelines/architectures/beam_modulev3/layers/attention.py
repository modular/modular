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
"""Beam attention layer (ModuleV3)."""

from __future__ import annotations

from max import functional as F
from max.dtype import DType
from max.nn import Module
from max.nn.attention_mask import MHAMaskVariant
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.linear import ColumnParallelLinear, RowParallelLinear
from max.nn.common_layers.mesh_axis import TP
from max.nn.kernels import (
    flash_attention_ragged,
    rope_split_store_ragged,
    store_k_cache_ragged,
    store_v_cache_ragged,
)
from max.nn.kv_cache import KVCacheParams
from max.pipelines.architectures.gemma4_modulev3.layers.rms_norm import (
    Gemma4RMSNorm,
)
from max.tensor import Tensor
from max.tensor.sharding import NamedMapping


class BeamAttention(Module[..., Tensor]):
    """GQA attention with per-head Q/K RMSNorm and a sigmoid output gate.

    Sliding layers rotate Q and K with RoPE inside the fused cache store.
    Full layers have no positional encoding; their queries are scaled per
    token by ``(1 + (position + 1) / 2048) ** 0.1`` instead. The output is
    gated by ``sigmoid(gate_proj(x) + gate_bias)``, one bias per head.

    Heads are sharded over the mesh's ``tp`` axis; ``o_proj`` returns a
    partial sum per device.

    Args:
        num_attention_heads: Query heads.
        num_key_value_heads: KV heads.
        hidden_size: Model width.
        kv_params: The cache leaf this layer addresses.
        is_sliding: Whether this is a sliding-window RoPE layer.
        sliding_window: Keys a sliding-layer query attends to.
        rms_norm_eps: Epsilon of the Q/K norms.
    """

    def __init__(
        self,
        *,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        is_sliding: bool,
        sliding_window: int,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.n_heads = num_attention_heads
        self.n_kv_heads = num_key_value_heads
        self.head_dim = kv_params.head_dim
        self.kv_params = kv_params
        self.is_sliding = is_sliding
        self.sliding_window = sliding_window
        self.scale = self.head_dim**-0.5

        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        self.q_proj = ColumnParallelLinear(hidden_size, q_dim, bias=False)
        self.k_proj = ColumnParallelLinear(hidden_size, kv_dim, bias=False)
        self.v_proj = ColumnParallelLinear(hidden_size, kv_dim, bias=False)
        self.gate_proj = ColumnParallelLinear(hidden_size, q_dim, bias=False)
        self.o_proj = RowParallelLinear(q_dim, hidden_size, bias=False)
        gate_bias = Tensor.zeros([self.n_heads])
        self.gate_bias = gate_bias.to(NamedMapping(gate_bias.mesh, (TP,)))
        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        x: Tensor,
        layer_idx: Tensor,
        kv_collection: PagedCacheValues,
        position_input: Tensor,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Runs attention and returns this device's partial ``o_proj`` sum.

        Args:
            x: Normalized hidden states, ``[total_seq_len, hidden_size]``.
            layer_idx: The layer's index within its cache leaf, a CPU
                ``uint32`` scalar.
            kv_collection: The cache leaf this layer addresses.
            position_input: The float32 RoPE table ``[positions, head_dim]``
                on a sliding layer; the float32 per-token query scale
                ``[total_seq_len]`` on a full layer.
            input_row_offsets: Ragged row offsets.

        Returns:
            The attention output, ``[total_seq_len, hidden_size]``.
        """
        head_dim = self.head_dim
        q = self.q_norm(self.q_proj(x).reshape((-1, self.n_heads, head_dim)))
        k = self.k_norm(self.k_proj(x).reshape((-1, self.n_kv_heads, head_dim)))
        v = self.v_proj(x)

        if self.is_sliding:
            qkv = F.concat(
                [
                    q.reshape((-1, self.n_heads * head_dim)),
                    k.reshape((-1, self.n_kv_heads * head_dim)),
                    v,
                ],
                axis=-1,
            )
            # A float32 table makes the kernel rotate in float32 and round
            # once, as the reference does.
            xq = rope_split_store_ragged(
                kv_params=self.kv_params,
                qkv=qkv,
                input_row_offsets=input_row_offsets,
                freqs_cis=position_input,
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                n_heads=self.n_heads // kv_collection.n_devices,
                interleaved=False,
                q_out_dtype=self.kv_params.dtype,
            ).rebind_mapping(qkv.mapping)
            xq = xq.reshape((-1, self.n_heads, head_dim))
            mask_variant = MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
            local_window_size = self.sliding_window
        else:
            store_k_cache_ragged(kv_collection, k, input_row_offsets, layer_idx)
            store_v_cache_ragged(
                kv_collection,
                v.reshape((-1, self.n_kv_heads, head_dim)),
                input_row_offsets,
                layer_idx,
            )
            # The query rounds to bfloat16 as the reference's does, then to
            # the cache dtype flash_attention_ragged reads Q in (FP8 under an
            # FP8 KV cache), the same two roundings the sliding branch's RoPE
            # kernel makes.
            xq = (
                (q.cast(DType.float32) * position_input.reshape((-1, 1, 1)))
                .cast(q.dtype)
                .cast(self.kv_params.dtype)
            )
            mask_variant = MHAMaskVariant.CAUSAL_MASK
            local_window_size = -1

        attn = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=mask_variant,
            scale=self.scale,
            local_window_size=local_window_size,
            output_dtype=x.dtype,
        ).rebind_mapping(xq.mapping)

        gate = self.gate_proj(x).reshape(
            (-1, self.n_heads, head_dim)
        ) + self.gate_bias.reshape((self.n_heads, 1))
        gated = attn.reshape((-1, self.n_heads, head_dim)) * F.sigmoid(gate)
        return self.o_proj(gated.reshape((-1, self.n_heads * head_dim)))
