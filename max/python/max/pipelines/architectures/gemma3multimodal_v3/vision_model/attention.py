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
from __future__ import annotations

from max import functional as F
from max.dtype import DType
from max.graph import DeviceRef
from max.nn import Linear, Module
from max.nn.legacy.attention import MHAMaskVariant
from max.nn.legacy.kernels import flash_attention_gpu
from max.tensor import Tensor

from ..model_config import Gemma3ForConditionalGenerationConfig


class Gemma3VisionAttention(Module[[Tensor], Tensor]):
    """Standard self-attention for SigLIP vision encoder."""

    def __init__(
        self,
        config: Gemma3ForConditionalGenerationConfig,
        layer_idx: int,
        device: DeviceRef | None = None,
    ) -> None:
        """Initialise the vision attention layers for projection and attention"""
        super().__init__()
        self.config = config
        vision_config = config.vision_config
        vision_dtype = (
            DType.bfloat16
        )  # TODO hmmm?  what do with this after V3 move

        self.layer_idx = layer_idx
        self.device = device if device is not None else config.devices[0]
        self.head_dim = (
            vision_config.hidden_size // vision_config.num_attention_heads
        )
        self.num_heads = vision_config.num_attention_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = Linear(
            vision_config.hidden_size,
            self.num_heads * self.head_dim,
            bias=vision_config.attention_bias,
        )
        self.k_proj = Linear(
            vision_config.hidden_size,
            self.num_heads * self.head_dim,
            bias=vision_config.attention_bias,
        )
        self.v_proj = Linear(
            vision_config.hidden_size,
            self.num_heads * self.head_dim,
            bias=vision_config.attention_bias,
        )
        self.out_proj = Linear(
            self.num_heads * self.head_dim,
            vision_config.hidden_size,
            bias=vision_config.attention_bias,
        )

    def __call__(self, x: Tensor) -> Tensor:
        """Process a tensor through the self attention layers and apply scaling"""
        batch_size, n_patches = x.shape[0], x.shape[1]

        # Project to Q, K, V
        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)

        # Reshape to multi-head format [batch, n_patches, n_heads, head_dim]
        xq = F.reshape(
            xq, [batch_size, n_patches, self.num_heads, self.head_dim]
        )
        xk = F.reshape(
            xk, [batch_size, n_patches, self.num_heads, self.head_dim]
        )
        xv = F.reshape(
            xv, [batch_size, n_patches, self.num_heads, self.head_dim]
        )

        output = flash_attention_gpu(
            xq,
            xk,
            xv,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=self.scaling,
        )

        output = output.reshape([batch_size, n_patches, -1])

        return self.out_proj(output)
