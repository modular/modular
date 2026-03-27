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
    """Spatial attention module for VAE models.

    This module performs self-attention on 2D spatial features by:
    1. Converting [N, C, H, W] to [N, H*W, C] sequence format
    2. Applying scaled dot-product attention (optimized for small sequences)
    3. Converting back to [N, C, H, W] format

    Note: Manual attention is used instead of flash-attention style kernels
    because VAE attention typically has small sequence lengths (H*W) where
    launch overhead outweighs benefits.
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
        """Apply spatial attention to a 2D image tensor.

        Args:
            x: Input tensor of shape [N, C, H, W].

        Returns:
            Output tensor of shape [N, C, H, W] with residual connection.
        """
        residual = x
        x = self.group_norm(x)

        n, c, h, w = x.shape
        seq_len = h * w
        x = ops.reshape(x, [n, c, seq_len])
        x = ops.permute(x, [0, 2, 1])

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = ops.reshape(q, [n, seq_len, self.heads, self.dim_head])
        q = ops.permute(q, [0, 2, 1, 3])
        k = ops.reshape(k, [n, seq_len, self.heads, self.dim_head])
        k = ops.permute(k, [0, 2, 1, 3])
        v = ops.reshape(v, [n, seq_len, self.heads, self.dim_head])
        v = ops.permute(v, [0, 2, 1, 3])

        attn = (q @ ops.permute(k, [0, 1, 3, 2])) * self.scale
        attn = ops.softmax(attn, axis=-1)
        out = attn @ v

        out = ops.permute(out, [0, 2, 1, 3])
        out = ops.reshape(out, [n, seq_len, self.inner_dim])
        out = self.to_out[0](out)
        out = ops.permute(out, [0, 2, 1])
        out = ops.reshape(out, [n, c, h, w])
        return residual + out
