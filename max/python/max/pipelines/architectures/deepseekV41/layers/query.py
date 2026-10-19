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
"""Low-rank query projection for DeepSeek-V4.1-Flash attention."""

from __future__ import annotations

from typing import NamedTuple

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.norm.rms_norm import RMSNorm

from ...deepseekV4.layers.rope import apply_rope_tail


class QueryProjectionOutput(NamedTuple):
    """Normalized low-rank query, its rotated heads, and the raw KV."""

    qr: TensorValue
    q: TensorValue
    kv_raw: TensorValue


class DeepseekV41Query(Module):
    """Projects hidden states to a normalized low-rank query and its heads.

    One fused matmul produces both the low-rank query and the raw KV
    projection. The low-rank query is returned so other modules can share it
    instead of recomputing it. The projected heads are not normalized again.
    """

    def __init__(
        self,
        hidden_size: int,
        q_lora_rank: int,
        head_dim: int,
        local_num_heads: int,
        eps: float,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.q_lora_rank = q_lora_rank
        self.head_dim = head_dim
        self.local_num_heads = local_num_heads
        self.fused_wqa_wkv = Linear(
            hidden_size,
            q_lora_rank + head_dim,
            dtype,
            device,
        )
        self.q_norm = RMSNorm(q_lora_rank, DType.float32, eps)
        self.wq_b = Linear(
            q_lora_rank,
            local_num_heads * head_dim,
            dtype,
            device,
        )

    def __call__(
        self, x: TensorValue, freqs_cis: TensorValue
    ) -> QueryProjectionOutput:
        """Returns the shared normalized ``qr`` and rotated local-head ``q``.

        Args:
            x: ``[1, T, hidden_size]`` hidden states.
            freqs_cis: ``[T, rope_head_dim // 2, 2]``, already gathered at
                these tokens' positions.

        Returns:
            ``qr`` ``[1, T, q_lora_rank]``, ``q``
            ``[1, T, local_num_heads, head_dim]``, and the unnormalized KV
            projection ``kv_raw`` ``[1, T, head_dim]``.
        """
        t = x.shape[-2]
        projected = self.fused_wqa_wkv(x)
        qr_raw = ops.slice_tensor(
            projected, [slice(None), slice(None), slice(0, self.q_lora_rank)]
        )
        kv_raw = ops.slice_tensor(
            projected, [slice(None), slice(None), slice(self.q_lora_rank, None)]
        )
        qr = self.q_norm(qr_raw)
        q = ops.reshape(
            self.wq_b(qr), [1, t, self.local_num_heads, self.head_dim]
        )
        q = apply_rope_tail(q, freqs_cis, int(freqs_cis.shape[-2]) * 2)
        return QueryProjectionOutput(qr=qr, q=q, kv_raw=kv_raw)
