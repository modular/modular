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

from __future__ import annotations

from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.linear import Linear
from max.experimental.tensor import Tensor
from max.graph import TensorValue
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu

from ..model_config import Gemma3ForConditionalGenerationConfig


class Gemma3VisionAttention(Module[[Tensor], Tensor]):
    """Standard self-attention for SigLIP vision encoder."""

    def __init__(
        self,
        config: Gemma3ForConditionalGenerationConfig,
        layer_idx: int,
    ) -> None:
        super().__init__()
        vision_config = config.vision_config

        self.layer_idx = layer_idx
        self.head_dim = (
            vision_config.hidden_size // vision_config.num_attention_heads
        )
        self.num_heads = vision_config.num_attention_heads
        self.scaling = self.head_dim**-0.5

        hidden_size = vision_config.hidden_size
        proj_size = self.num_heads * self.head_dim
        has_bias = vision_config.attention_bias

        self.q_proj = Linear(
            in_dim=hidden_size, out_dim=proj_size, bias=has_bias
        )
        self.k_proj = Linear(
            in_dim=hidden_size, out_dim=proj_size, bias=has_bias
        )
        self.v_proj = Linear(
            in_dim=hidden_size, out_dim=proj_size, bias=has_bias
        )
        self.out_proj = Linear(
            in_dim=proj_size, out_dim=hidden_size, bias=has_bias
        )

    def forward(self, x: Tensor) -> Tensor:
        batch_size, n_patches = x.shape[0], x.shape[1]

        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)

        xq = F.reshape(
            xq, [batch_size, n_patches, self.num_heads, self.head_dim]
        )
        xk = F.reshape(
            xk, [batch_size, n_patches, self.num_heads, self.head_dim]
        )
        xv = F.reshape(
            xv, [batch_size, n_patches, self.num_heads, self.head_dim]
        )

        output = Tensor.from_graph_value(
            flash_attention_gpu(
                TensorValue(xq),
                TensorValue(xk),
                TensorValue(xv),
                mask_variant=MHAMaskVariant.NULL_MASK,
                scale=self.scaling,
            )
        )

        output = F.reshape(output, [batch_size, n_patches, -1])

        return self.out_proj(output)
