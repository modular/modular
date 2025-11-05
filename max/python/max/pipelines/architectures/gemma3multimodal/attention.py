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

from max.graph import TensorValue, ops
from max.nn import Linear
from max.nn.layer import Module

from .model_config import Gemma3ForConditionalGenerationConfig


# from Huggingface Transformers
class Gemma3VisionAttention(Module):
    """Standard self-attention for SigLIP vision encoder.

    Unlike Pixtral, SigLIP uses:
    - Standard self-attention (no rotary embeddings)
    - No attention masking
    - Absolute position embeddings (added in embedding layer)
    """

    def __init__(
        self, config: Gemma3ForConditionalGenerationConfig, layer_idx: int
    ) -> None:
        super().__init__()
        vision_config = config.vision_config

        self.layer_type = (
            config.layer_types[layer_idx]
            if hasattr(config, "layer_types")
            else None
        )
        self.config = config
        self.layer_idx = layer_idx
        # Vision encoder uses its own head_dim, not the text model's
        self.head_dim = (
            vision_config.hidden_size // vision_config.num_attention_heads
        )
        self.num_heads = vision_config.num_attention_heads
        self.attention_dropout = vision_config.attention_dropout
        # self.scaling = config.query_pre_attn_scalar**-0.5
        # self.is_causal = not config.use_bidirectional_attention

        self.q_proj = Linear(
            vision_config.hidden_size,  # 1152
            self.num_heads * self.head_dim,  # 16 * 72 = 1152
            has_bias=config.attention_bias,
            dtype=config.dtype,
            device=config.devices[0],
        )
        self.k_proj = Linear(
            vision_config.hidden_size,
            self.num_heads * self.head_dim,
            has_bias=config.attention_bias,
            dtype=config.dtype,
            device=config.devices[0],
        )
        self.v_proj = Linear(
            vision_config.hidden_size,
            self.num_heads * self.head_dim,
            has_bias=config.attention_bias,
            dtype=config.dtype,
            device=config.devices[0],
        )
        self.out_proj = Linear(
            self.num_heads * self.head_dim,
            vision_config.hidden_size,
            has_bias=config.attention_bias,
            dtype=config.dtype,
            device=config.devices[0],
        )

        self.attn_logit_softcapping = config.attn_logit_softcapping
        self.sliding_window = (
            config.sliding_window
            if self.layer_type == "sliding_attention"
            else None
        )
        self.is_sliding = self.layer_type == "sliding_attention"

        # these appear to be part of language model, not vision
        # self.q_norm = Gemma3RMSNorm(
        #     dim=self.head_dim,
        #     dtype=config.dtype,
        #     eps=vision_config.layer_norm_eps,
        # )
        # self.k_norm = Gemma3RMSNorm(
        #     dim=self.head_dim,
        #     dtype=config.dtype,
        #     eps=vision_config.layer_norm_eps,
        # )

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

        # Reshape to multi-head format [batch, n_patches, n_heads, head_dim]
        xq = ops.reshape(
            xq, [batch_size, n_patches, self.num_heads, self.head_dim]
        )
        xk = ops.reshape(
            xk, [batch_size, n_patches, self.num_heads, self.head_dim]
        )
        xv = ops.reshape(
            xv, [batch_size, n_patches, self.num_heads, self.head_dim]
        )

        # Apply per-head QK normalization (Gemma3-specific)
        # appear to be part of language model
        # xq = self.q_norm(xq)
        # xk = self.k_norm(xk)

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
        return self.out_proj(output)
