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

"""Muse Spark attention layer for the ModuleV3 API."""

from __future__ import annotations

from max import functional as F
from max.driver import CPU, Accelerator
from max.dtype import DType
from max.graph import DeviceRef, TensorType, TensorValue, ops
from max.nn import Module
from max.nn.attention_mask import MHAMaskVariant
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from max.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.nn.kernels import flash_attention_ragged, rope_split_store_ragged
from max.nn.kv_cache import KVCacheParams
from max.nn.kv_cache import PagedCacheValues as GraphPagedCacheValues
from max.tensor import Tensor

from .norms import weightless_rms_norm


class NoPERotaryEmbedding(RotaryEmbedding):
    """Identity RoPE for the full-attention (NoPE) layers.

    Every inverse frequency is zero, so ``freqs_cis`` is all ``(1, 0)``
    pairs and full layers store K and V through the same fused
    ``rope_split_store_ragged`` kernel as sliding layers.
    """

    def _compute_inv_freqs(self) -> Tensor:
        return Tensor.zeros(
            [self.head_dim // 2], dtype=DType.float32, device=CPU()
        )


def _qk_rms_norm_rope_split_store(
    qkv: TensorValue,
    input_row_offsets: TensorValue,
    freqs_cis: TensorValue,
    kv_collection: GraphPagedCacheValues,
    layer_idx: TensorValue,
    n_q_heads: int,
    eps: float,
    q_scale: float,
) -> TensorValue:
    head_dim = int(kv_collection.kv_blocks.shape[-1])
    return ops.inplace_custom(
        "kv_cache.qk_rms_norm_rope_split_store.ragged.paged",
        device=qkv.device,
        values=[
            qkv,
            input_row_offsets,
            freqs_cis,
            *kv_collection.flatten_without_attention_dispatch_metadata(),
            layer_idx,
            *(
                ops.constant(v, DType.float32, device=DeviceRef.CPU())
                for v in (eps, q_scale)
            ),
        ],
        out_types=[
            TensorType(
                qkv.dtype, [qkv.shape[0], n_q_heads * head_dim], qkv.device
            )
        ],
        parameters={"n_q_heads": n_q_heads},
    )[0].tensor


qk_rms_norm_rope_split_store = F.functional(_qk_rms_norm_rope_split_store)


def _head_rms_norm_sigmoid_gate(
    x: TensorValue, qkvg: TensorValue, n_heads: int, eps: float
) -> TensorValue:
    return ops.custom(
        "norm.head_rms_norm_sigmoid_gate",
        device=x.device,
        values=[
            x,
            qkvg,
            ops.constant(eps, DType.float32, device=DeviceRef.CPU()),
        ],
        out_types=[x.type],
        parameters={"n_heads": n_heads},
    )[0].tensor


#: Per-head weightless RMSNorm of ``x`` times the sigmoid of ``qkvg``'s
#: trailing gate columns.
head_rms_norm_sigmoid_gate = F.functional(_head_rms_norm_sigmoid_gate)


class MuseSparkAttention(Module[..., Tensor]):
    """GQA with weightless qk-norm, a q pre-scale, an output norm and a sigmoid gate."""

    def __init__(
        self,
        *,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx_in_cache: int,
        is_sliding: bool,
        norm_eps: float,
        qk_scale_factor: float,
        local_window_size: int,
    ) -> None:
        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.kv_params = kv_params
        self.layer_idx_in_cache = layer_idx_in_cache
        self.use_local = is_sliding
        self.norm_eps = norm_eps
        self.qk_scale_factor = qk_scale_factor
        self.local_window_size = local_window_size

        self.head_dim = kv_params.head_dim
        self.scale = self.head_dim**-0.5
        self.q_weight_dim = self.head_dim * num_attention_heads
        self.kv_weight_dim = self.head_dim * num_key_value_heads

        self.q_proj = ColumnParallelLinear(
            in_dim=hidden_size, out_dim=self.q_weight_dim, bias=False
        )
        self.k_proj = ColumnParallelLinear(
            in_dim=hidden_size, out_dim=self.kv_weight_dim, bias=False
        )
        self.v_proj = ColumnParallelLinear(
            in_dim=hidden_size, out_dim=self.kv_weight_dim, bias=False
        )
        self.gate_proj = ColumnParallelLinear(
            in_dim=hidden_size, out_dim=self.q_weight_dim, bias=False
        )
        self.o_proj = RowParallelLinear(
            in_dim=self.q_weight_dim, out_dim=hidden_size, bias=False
        )

    @property
    def wqkvg(self) -> Tensor:
        """Q, K, V and gate weights stacked for one matmul."""
        return F.concat(
            [
                self.q_proj.weight,
                self.k_proj.weight,
                self.v_proj.weight,
                self.gate_proj.weight,
            ],
            axis=0,
        )

    def _head_norm(self, x: Tensor, n_heads: int) -> Tensor:
        return weightless_rms_norm(
            x.reshape((-1, n_heads, self.head_dim)), self.norm_eps
        ).reshape((-1, n_heads * self.head_dim))

    def rope_store(
        self,
        qkv: Tensor,
        kv_collection: PagedCacheValues,
        layer_idx: Tensor,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Norms, pre-scales and ropes Q and K, stores K and V; returns Q."""
        # Per device with no sharding rule, as the unfused kernel.
        return qk_rms_norm_rope_split_store(
            qkv,
            input_row_offsets,
            self.rope.freqs_cis,
            kv_collection,
            layer_idx,
            self.n_heads // kv_collection.n_devices,
            self.norm_eps,
            self.qk_scale_factor,
        ).rebind_mapping(qkv.mapping)

    def rope_store_unfused(
        self,
        qkv: Tensor,
        kv_collection: PagedCacheValues,
        layer_idx: Tensor,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """:meth:`rope_store` as separate norm, scale and RoPE+store ops."""
        x_q, x_k, x_v = qkv.split(
            [self.q_weight_dim, self.kv_weight_dim, self.kv_weight_dim],
            axis=-1,
        )
        x_q = self._head_norm(x_q, self.n_heads) * self.qk_scale_factor
        x_k = self._head_norm(x_k, self.num_key_value_heads)
        qkv = F.concat([x_q, x_k, x_v], axis=-1)
        # The kernels run per device with no sharding rule: their outputs keep
        # the heads' placement.
        return rope_split_store_ragged(
            kv_params=self.kv_params,
            qkv=qkv,
            input_row_offsets=input_row_offsets,
            freqs_cis=self.rope.freqs_cis,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads // kv_collection.n_devices,
            interleaved=self.rope.interleaved,
            # flash_attention_ragged reads Q in the cache dtype (FP8 under
            # an FP8 KV cache).
            q_out_dtype=self.kv_params.dtype,
        ).rebind_mapping(qkv.mapping)

    def output_gate(
        self, attn_out: Tensor, qkvg: Tensor, n_devices: int
    ) -> Tensor:
        """Per-head norm of ``attn_out`` times the sigmoid of ``qkvg``'s gate."""
        return head_rms_norm_sigmoid_gate(
            attn_out, qkvg, self.n_heads // n_devices, self.norm_eps
        ).rebind_mapping(attn_out.mapping)

    def output_gate_unfused(self, attn_out: Tensor, gate: Tensor) -> Tensor:
        """:meth:`output_gate` as separate norm, sigmoid and multiply ops."""
        return self._head_norm(attn_out, self.n_heads) * F.sigmoid(gate)

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        *,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Runs the attention over ``x``."""
        layer_idx = F.constant(
            self.layer_idx_in_cache, DType.uint32, device=CPU()
        )
        q_dim, kv_dim = self.q_weight_dim, self.kv_weight_dim

        qkvg = x @ self.wqkvg.T
        # The fused ops rope rotate-half heads on GPU and write a bf16 Q only;
        # the DSpark drafter's interleaved RoPE and an FP8 KV cache, whose
        # attention reads Q in FP8, keep the unfused ops.
        fused = (
            not self.rope.interleaved
            and self.kv_params.dtype == DType.bfloat16
            and all(isinstance(d, Accelerator) for d in x.mesh.devices)
        )
        if fused:
            # Both ops read their columns of the projection in place.
            xq = self.rope_store(
                qkvg, kv_collection, layer_idx, input_row_offsets
            )
        else:
            qkv, gate = qkvg.split([q_dim + 2 * kv_dim, q_dim], axis=-1)
            xq = self.rope_store_unfused(
                qkv, kv_collection, layer_idx, input_row_offsets
            )
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq.reshape((-1, self.n_heads, self.head_dim)),
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=(
                MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
                if self.use_local
                else MHAMaskVariant.CAUSAL_MASK
            ),
            scale=self.scale,
            local_window_size=self.local_window_size if self.use_local else -1,
            output_dtype=x.dtype,
        ).rebind_mapping(xq.mapping)
        attn_out = attn_out.reshape((-1, q_dim))
        if fused:
            return self.o_proj(
                self.output_gate(attn_out, qkvg, kv_collection.n_devices)
            )
        return self.o_proj(self.output_gate_unfused(attn_out, gate))
