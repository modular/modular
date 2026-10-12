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

import math

from max.experimental import functional as F
from max.experimental.nn import Linear, Module
from max.experimental.nn.norm import RMSNorm
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu as _flash_attention_gpu

from .embeddings import apply_rotary_emb

flash_attention_gpu = F.functional(_flash_attention_gpu)


class ZImageAttention(Module[..., Tensor]):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        qk_norm: bool,
        eps: float,
    ):
        self.head_dim = dim // n_heads
        self.inner_dim = dim
        self.n_heads = n_heads

        self.to_q = Linear(dim, dim, bias=False)
        self.to_k = Linear(dim, dim, bias=False)
        self.to_v = Linear(dim, dim, bias=False)

        self.norm_q = RMSNorm(self.head_dim, eps=eps) if qk_norm else None
        self.norm_k = RMSNorm(self.head_dim, eps=eps) if qk_norm else None

        # Keep ModuleList naming for diffusers-compatible key loading.
        self.to_out = ModuleList([Linear(dim, dim, bias=False)])

    def forward(
        self,
        hidden_states: Tensor,
        freqs_cis: tuple[Tensor, Tensor],
    ) -> Tensor:
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = F.reshape(
            query, [batch_size, seq_len, self.n_heads, self.head_dim]
        )
        key = F.reshape(key, [batch_size, seq_len, self.n_heads, self.head_dim])
        value = F.reshape(
            value, [batch_size, seq_len, self.n_heads, self.head_dim]
        )

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        query = apply_rotary_emb(
            query,
            freqs_cis,
            use_real=True,
            use_real_unbind_dim=-1,
            sequence_dim=1,
        )
        key = apply_rotary_emb(
            key,
            freqs_cis,
            use_real=True,
            use_real_unbind_dim=-1,
            sequence_dim=1,
        )
        query = query.cast(value.dtype)
        key = key.cast(value.dtype)

        out = flash_attention_gpu(
            query,
            key,
            value,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=math.sqrt(1.0 / float(self.head_dim)),
        )

        out = F.reshape(out, [batch_size, seq_len, self.inner_dim])
        return self.to_out[0](out)
