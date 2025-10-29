# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
from max.nn import Linear
from max.nn.layer import Module

# claude gave us this
class Gemma3VisionAttention(Module):
    """Standard self-attention for SigLIP vision encoder.

    Unlike Pixtral, SigLIP uses:
    - Standard self-attention (no rotary embeddings)
    - No attention masking
    - Absolute position embeddings (added in embedding layer)
    """

    def __init__(
        self,
        n_heads: int,
        dim: int,
        head_dim: int,
        dtype: DType,
        device: DeviceRef,
        has_bias: bool = True,
    ) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = head_dim

        self.q_proj = Linear(
            dim, dim, dtype=dtype, device=device, has_bias=has_bias
        )
        self.k_proj = Linear(
            dim, dim, dtype=dtype, device=device, has_bias=has_bias
        )
        self.v_proj = Linear(
            dim, dim, dtype=dtype, device=device, has_bias=has_bias
        )
        self.o_proj = Linear(
            dim, dim, dtype=dtype, device=device, has_bias=has_bias
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        """Standard self-attention.

        Args:
            x: Input tensor [batch, seq_len, dim]

        Returns:
            Output tensor [batch, seq_len, dim]
        """
        batch_size, n_patches = x.shape[0], x.shape[1]

        # Project to Q, K, V
        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)

        # Reshape to multi-head format
        xq = ops.reshape(xq, [batch_size, n_patches, self.n_heads, self.head_dim])
        xk = ops.reshape(xk, [batch_size, n_patches, self.n_heads, self.head_dim])
        xv = ops.reshape(xv, [batch_size, n_patches, self.n_heads, self.head_dim])

        # Transpose to [batch, n_heads, n_patches, head_dim]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(1.0 / self.head_dim)
        scores = xq @ ops.transpose(xk, 2, 3)
        scores = ops.softmax(scores * scale)

        # Apply attention to values
        output = scores @ xv  # [batch, n_heads, n_patches, head_dim]

        # Transpose back and reshape
        output = output.transpose(1, 2).reshape([batch_size, n_patches, -1])

        # Output projection
        return self.o_proj(output)