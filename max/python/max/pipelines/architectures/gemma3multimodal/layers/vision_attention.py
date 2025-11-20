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
from collections.abc import Sequence

from max.graph import TensorValue, ops
from max.graph.type import DeviceRef
from max.nn import Linear, Module

from ..model_config import VisionConfig


class VisionAttention(Module):
    """Multi-head self-attention for vision encoder."""

    def __init__(
        self,
        config: VisionConfig,
        devices: Sequence[DeviceRef],
        layer_idx: int = 0,
    ):
        """Initialize vision attention layer.

        Args:
            config: Vision configuration.
            devices: Devices to place the weights on.
            layer_idx: Index of the layer (for unique naming).
        """
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.devices = devices
        self.device = devices[0] if devices else DeviceRef.CPU()
        self.layer_idx = layer_idx

        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
            )

        # Q, K, V projections - separate weight matrices
        self.q_proj = Linear(
            in_dim=self.embed_dim,
            out_dim=self.embed_dim,
            dtype=config.dtype,
            device=self.device,
            has_bias=True,
        )

        self.k_proj = Linear(
            in_dim=self.embed_dim,
            out_dim=self.embed_dim,
            dtype=config.dtype,
            device=self.device,
            has_bias=True,
        )

        self.v_proj = Linear(
            in_dim=self.embed_dim,
            out_dim=self.embed_dim,
            dtype=config.dtype,
            device=self.device,
            has_bias=True,
        )

        # Output projection
        self.out_proj = Linear(
            in_dim=self.embed_dim,
            out_dim=self.embed_dim,
            dtype=config.dtype,
            device=self.device,
            has_bias=True,
        )

    def __call__(self, x: TensorValue) -> TensorValue:  # type: ignore[override]
        """Forward pass for attention layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Project to Q, K, V separately
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        # q, k, v shape: (batch_size, seq_len, embed_dim)
        q = q.reshape([batch_size, seq_len, self.num_heads, self.head_dim])
        k = k.reshape([batch_size, seq_len, self.num_heads, self.head_dim])
        v = v.reshape([batch_size, seq_len, self.num_heads, self.head_dim])

        # Permute to (batch_size, num_heads, seq_len, head_dim)
        q = q.permute([0, 2, 1, 3])
        k = k.permute([0, 2, 1, 3])
        v = v.permute([0, 2, 1, 3])

        # Compute attention scores: (batch, num_heads, seq_len, seq_len)
        # Q @ K^T
        k_t = k.transpose(2, 3)
        attn_weights = q @ k_t
        attn_weights = attn_weights * self.scale

        # Apply softmax on last dimension (seq_len)
        attn_weights = ops.softmax(attn_weights)

        # Apply attention to values: (batch, num_heads, seq_len, head_dim)
        attn_output = attn_weights @ v

        # Reshape back to (batch_size, seq_len, embed_dim)
        attn_output = attn_output.permute([0, 2, 1, 3])
        attn_output = attn_output.reshape([batch_size, seq_len, self.embed_dim])

        # Output projection
        output = self.out_proj(attn_output)

        return output
