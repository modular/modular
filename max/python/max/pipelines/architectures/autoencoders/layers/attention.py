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

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear
from max.nn.norm import GroupNorm


class VAEAttention(Module):
    """Spatial attention block for the V2 VAE."""

    def __init__(
        self,
        query_dim: int,
        heads: int,
        dim_head: int,
        num_groups: int = 32,
        eps: float = 1e-6,
        device: DeviceRef | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__()
        if dtype is None:
            raise ValueError("dtype must be set for VAEAttention")
        if device is None:
            raise ValueError("device must be set for VAEAttention")

        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.group_norm = GroupNorm(
            num_groups=num_groups,
            num_channels=query_dim,
            eps=eps,
            affine=True,
            device=device,
        )
        self.to_q = Linear(
            in_dim=query_dim,
            out_dim=self.inner_dim,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.to_k = Linear(
            in_dim=query_dim,
            out_dim=self.inner_dim,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.to_v = Linear(
            in_dim=query_dim,
            out_dim=self.inner_dim,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.to_out = LayerList(
            [
                Linear(
                    in_dim=self.inner_dim,
                    out_dim=query_dim,
                    dtype=dtype,
                    device=device,
                    has_bias=True,
                )
            ]
        )
        self.scale = 1.0 / math.sqrt(dim_head)

    def __call__(self, x: TensorValue) -> TensorValue:
        residual = x
        hidden = self.group_norm(x)

        batch, channels, height, width = hidden.shape
        seq_len = height * width
        hidden = ops.reshape(hidden, [batch, channels, seq_len])
        hidden = ops.permute(hidden, [0, 2, 1])

        query = self.to_q(hidden)
        key = self.to_k(hidden)
        value = self.to_v(hidden)

        query = ops.reshape(query, [batch, seq_len, self.heads, self.dim_head])
        query = ops.permute(query, [0, 2, 1, 3])
        key = ops.reshape(key, [batch, seq_len, self.heads, self.dim_head])
        key = ops.permute(key, [0, 2, 1, 3])
        value = ops.reshape(value, [batch, seq_len, self.heads, self.dim_head])
        value = ops.permute(value, [0, 2, 1, 3])

        attn = (query @ ops.permute(key, [0, 1, 3, 2])) * self.scale
        attn = ops.softmax(attn, axis=-1)
        hidden = attn @ value

        hidden = ops.permute(hidden, [0, 2, 1, 3])
        hidden = ops.reshape(hidden, [batch, seq_len, self.inner_dim])
        hidden = self.to_out[0](hidden)
        hidden = ops.permute(hidden, [0, 2, 1])
        hidden = ops.reshape(hidden, [batch, channels, height, width])
        return residual + hidden
