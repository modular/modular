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

"""Qwen2.5-VL encoder-only attention with bias support (module v2)."""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.rotary_embedding import RotaryEmbedding


class Qwen25VLEncoderAttention(Module):
    """Encoder-only attention with bias for Qwen2.5-VL (module v2)."""

    def __init__(
        self,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        head_dim: int,
        scale: float,
        attention_bias: bool = True,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.n_heads = num_attention_heads
        self.n_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.scale = scale

        q_dim = head_dim * num_attention_heads
        kv_dim = head_dim * num_key_value_heads

        self.q_proj = Linear(
            hidden_size,
            q_dim,
            dtype=dtype,
            device=device,
            has_bias=attention_bias,
        )
        self.k_proj = Linear(
            hidden_size,
            kv_dim,
            dtype=dtype,
            device=device,
            has_bias=attention_bias,
        )
        self.v_proj = Linear(
            hidden_size,
            kv_dim,
            dtype=dtype,
            device=device,
            has_bias=attention_bias,
        )
        self.o_proj = Linear(
            q_dim,
            hidden_size,
            dtype=dtype,
            device=device,
            has_bias=False,
        )

    def _repeat_kv(self, x: TensorValue, n_rep: int) -> TensorValue:
        if n_rep == 1:
            return x
        seq_len = x.shape[0]
        n_kv_heads = x.shape[1]
        head_dim = x.shape[2]
        x = ops.unsqueeze(x, 2)
        x = ops.broadcast_to(x, (seq_len, n_kv_heads, n_rep, head_dim))
        return ops.reshape(x, (seq_len, n_kv_heads * n_rep, head_dim))

    def __call__(
        self,
        x: TensorValue,
        rope: RotaryEmbedding,
    ) -> TensorValue:
        total_seq_len = x.shape[0]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = ops.reshape(q, (total_seq_len, self.n_heads, self.head_dim))
        k = ops.reshape(k, (total_seq_len, self.n_kv_heads, self.head_dim))
        v = ops.reshape(v, (total_seq_len, self.n_kv_heads, self.head_dim))

        q = ops.squeeze(rope(ops.unsqueeze(q, 0)), 0)
        k = ops.squeeze(rope(ops.unsqueeze(k, 0)), 0)

        if self.n_kv_heads != self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            k = self._repeat_kv(k, n_rep)
            v = self._repeat_kv(v, n_rep)

        q = ops.unsqueeze(q, 0)
        k = ops.unsqueeze(k, 0)
        v = ops.unsqueeze(v, 0)

        attn_out = flash_attention_gpu(
            q,
            k,
            v,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
        )

        attn_out = ops.squeeze(attn_out, 0)
        attn_out = ops.reshape(attn_out, (total_seq_len, -1))
        return self.o_proj(attn_out)
