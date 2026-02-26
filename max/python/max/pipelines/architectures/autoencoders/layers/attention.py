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
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import DeviceRef
from max.nn.module_v3 import GroupNorm, Linear, Module, ModuleList


class VAEAttention(Module[[Tensor], Tensor]):
    """Spatial attention module for VAE models.

    This module performs self-attention on 2D spatial features by:
    1. Converting [N, C, H, W] to [N, H*W, C] sequence format
    2. Applying scaled dot-product attention (optimized for small sequences)
    3. Converting back to [N, C, H, W] format

    Note: Manual attention is used instead of flash_attention_gpu because
    VAE attention typically has small sequence lengths (H*W) where flash
    attention overhead outweighs benefits.
    """

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
        """Initialize VAE attention module.

        Args:
            query_dim: Dimension of query (number of channels).
            heads: Number of attention heads.
            dim_head: Dimension of each attention head.
            num_groups: Number of groups for GroupNorm.
            eps: Epsilon value for GroupNorm.
            device: Device reference.
            dtype: Data type.
        """
        super().__init__()
        self.query_dim = query_dim
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head

        self.group_norm = GroupNorm(
            num_groups=num_groups,
            num_channels=query_dim,
            eps=eps,
            affine=True,
        )

        self.to_q = Linear(
            in_dim=query_dim,
            out_dim=self.inner_dim,
            bias=True,
        )
        self.to_k = Linear(
            in_dim=query_dim,
            out_dim=self.inner_dim,
            bias=True,
        )
        self.to_v = Linear(
            in_dim=query_dim,
            out_dim=self.inner_dim,
            bias=True,
        )
        # Use ModuleList to match original weights format (to_out.0.*)
        self.to_out = ModuleList(
            [
                Linear(
                    in_dim=self.inner_dim,
                    out_dim=query_dim,
                    bias=True,
                )
            ]
        )

        self.scale = 1.0 / math.sqrt(dim_head)

    def forward(self, x: Tensor) -> Tensor:
        """Apply spatial attention to 2D image tensor.

        Args:
            x: Input tensor of shape [N, C, H, W].

        Returns:
            Output tensor of shape [N, C, H, W] with residual connection.
        """
        residual = x

        x = self.group_norm(x)

        n, c, h, w = x.shape
        seq_len = h * w

        x = F.reshape(x, [n, c, seq_len])
        x = F.permute(x, [0, 2, 1])

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = F.reshape(q, [n, seq_len, self.heads, self.dim_head])
        q = F.permute(q, [0, 2, 1, 3])
        k = F.reshape(k, [n, seq_len, self.heads, self.dim_head])
        k = F.permute(k, [0, 2, 1, 3])
        v = F.reshape(v, [n, seq_len, self.heads, self.dim_head])
        v = F.permute(v, [0, 2, 1, 3])

        attn = q @ F.permute(k, [0, 1, 3, 2]) * self.scale
        attn = F.softmax(attn, axis=-1)
        out = attn @ v

        out = F.permute(out, [0, 2, 1, 3])
        out = F.reshape(out, [n, seq_len, self.inner_dim])

        out = self.to_out[0](out)

        out = F.permute(out, [0, 2, 1])
        out = F.reshape(out, [n, c, h, w])

        return residual + out
