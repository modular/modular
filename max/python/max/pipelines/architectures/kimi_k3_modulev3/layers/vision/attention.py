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
"""Kimi K3 vision attention, derived from K2.5 with wider QKV projections."""

from __future__ import annotations

import math

from max import functional as F
from max.dtype import DType
from max.nn import Linear, Module
from max.nn._graph_ops import (
    flash_attention_ragged_gpu as _flash_attention_ragged_gpu,
)
from max.nn.attention_mask import MHAMaskVariant
from max.tensor import Tensor
from max.tensor.sharding import AxisAssignment, replicated_rows
from max.tensor.sharding.types import TensorLayout


def _attention_rule(
    q: TensorLayout,
    k: TensorLayout,
    v: TensorLayout,
    input_row_offsets: TensorLayout,
    max_seq_len: TensorLayout,
    *extras: object,
) -> list[AxisAssignment]:
    return replicated_rows(q, k, v, input_row_offsets, max_seq_len)


flash_attention_ragged_gpu = F.functional(
    _flash_attention_ragged_gpu, rule=_attention_rule
)


class Attention(Module[[Tensor, Tensor, Tensor, Tensor], Tensor]):
    """Attends within each image; heads split ``qkv_hidden_size`` evenly."""

    def __init__(
        self, num_heads: int, hidden_dim: int, qkv_hidden_size: int
    ) -> None:
        self.num_heads = num_heads
        self.head_dim = qkv_hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.wqkv = Linear(hidden_dim, 3 * qkv_hidden_size, bias=False)
        self.wo = Linear(qkv_hidden_size, hidden_dim, bias=False)

    @staticmethod
    def _apply_rope(x: Tensor, freqs_cis: Tensor) -> Tensor:
        orig_dtype = x.dtype
        complex_x = F.as_interleaved_complex(x.cast(DType.float32))
        freqs = freqs_cis.unsqueeze(1).cast(DType.float32)
        real = (
            complex_x[..., 0] * freqs[..., 0]
            - complex_x[..., 1] * freqs[..., 1]
        )
        imag = (
            complex_x[..., 0] * freqs[..., 1]
            + complex_x[..., 1] * freqs[..., 0]
        )
        return F.stack([real, imag], axis=-1).reshape(x.shape).cast(orig_dtype)

    def forward(
        self,
        x: Tensor,
        input_row_offsets: Tensor,
        max_seq_len: Tensor,
        rope_freqs_cis: Tensor,
    ) -> Tensor:
        """Runs unmasked segmented attention over the packed image patches."""
        rows = x.shape[0]
        qkv = self.wqkv(x).reshape([rows, 3, self.num_heads, self.head_dim])
        q, k, v = [part.squeeze(1) for part in qkv.split([1, 1, 1], axis=1)]
        q = self._apply_rope(q, rope_freqs_cis)
        k = self._apply_rope(k, rope_freqs_cis)
        out = flash_attention_ragged_gpu(
            q=q,
            k=k,
            v=v,
            input_row_offsets=input_row_offsets,
            max_seq_len=max_seq_len,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=self.scale,
        )
        return self.wo(out.reshape([rows, -1]))
