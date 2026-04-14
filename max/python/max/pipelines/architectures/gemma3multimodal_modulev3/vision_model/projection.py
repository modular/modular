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
from max.pipelines.architectures.gemma3_modulev3.layers.rms_norm import (
    Gemma3RMSNorm,
)

from ..model_config import Gemma3ForConditionalGenerationConfig


class Gemma3MultiModalProjector(Module[[Tensor], Tensor]):
    """Projects vision encoder outputs to text embedding space."""

    def __init__(
        self,
        config: Gemma3ForConditionalGenerationConfig,
    ) -> None:
        super().__init__()

        self.mm_input_projection_weight = Tensor.zeros(
            [
                config.vision_config.hidden_size,
                config.text_config.hidden_size,
            ]
        )

        self.mm_soft_emb_norm = Gemma3RMSNorm(
            config.vision_config.hidden_size,
            eps=config.vision_config.layer_norm_eps,
        )

        self.patches_per_image = int(
            config.vision_config.image_size // config.vision_config.patch_size
        )

        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side

    def forward(self, vision_outputs: Tensor) -> Tensor:
        batch_size, _, seq_length = vision_outputs.shape

        transposed_vision_outputs = vision_outputs.transpose(1, 2)

        reshaped_vision_outputs = F.reshape(
            transposed_vision_outputs,
            [
                batch_size,
                seq_length,
                self.patches_per_image,
                self.patches_per_image,
            ],
        )

        # reshape to NHWC for avg pool
        reshaped_vision_outputs = reshaped_vision_outputs.permute([0, 2, 3, 1])
        pooled_vision_outputs = F.avg_pool2d(
            input=reshaped_vision_outputs,
            kernel_size=(self.kernel_size, self.kernel_size),
            stride=self.kernel_size,
        )

        pooled_vision_outputs = pooled_vision_outputs.permute([0, 3, 1, 2])
        pooled_vision_outputs = F.flatten(pooled_vision_outputs, start_dim=2)
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)

        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

        projected_vision_outputs = (
            normed_vision_outputs @ self.mm_input_projection_weight
        )

        image_hidden_states = F.flatten(
            projected_vision_outputs, start_dim=0, end_dim=1
        )

        return image_hidden_states


class Gemma3VisionMLP(Module[[Tensor], Tensor]):
    """Two-layer MLP with GELU activation for vision encoder."""

    def __init__(
        self,
        config: Gemma3ForConditionalGenerationConfig,
    ) -> None:
        super().__init__()
        self.hidden_act = config.vision_config.hidden_act

        self.fc1 = Linear(
            in_dim=config.vision_config.hidden_size,
            out_dim=config.vision_config.intermediate_size,
            bias=True,
        )

        self.fc2 = Linear(
            in_dim=config.vision_config.intermediate_size,
            out_dim=config.vision_config.hidden_size,
            bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = F.gelu(x, self.hidden_act)
        x = self.fc2(x)
        return x
