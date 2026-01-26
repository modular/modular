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
from max.graph import DeviceRef, Weight
from max.graph.ops import avg_pool2d
from max.nn.linear import Linear
from max.nn.module import Module
from max.tensor import Tensor

from ..model_config import Gemma3ForConditionalGenerationConfig
from .rms_norm import Gemma3RMSNorm


class Gemma3MultiModalProjector(Module):
    """Projects vision encoder outputs to text embedding space."""

    def __init__(
        self,
        config: Gemma3ForConditionalGenerationConfig,
        device: DeviceRef | None = None,
    ):
        """Prepare the normalisation and projection weights based on config"""
        super().__init__()

        self.config = config
        self.device = device if device is not None else config.devices[0]

        vision_dtype = DType.bfloat16

        self.mm_input_projection_weight = Weight(
            "mm_input_projection_weight",
            dtype=vision_dtype,
            shape=(
                config.vision_config.hidden_size,
                config.text_config.hidden_size,
            ),
            device=self.device,
        )

        self.mm_soft_emb_norm = Gemma3RMSNorm(
            config.vision_config.hidden_size,
            eps=config.vision_config.layer_norm_eps,
            dtype=vision_dtype,
        )

        self.patches_per_image = int(
            config.vision_config.image_size // config.vision_config.patch_size
        )

        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side

    def __call__(self, vision_outputs: Tensor) -> Tensor:
        """Process vision outputs through pooling, normalisation, and a
        projection weight"""
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

        # reshape to 0 2 3 1 NHWL (or NHWC) for avg pool
        reshaped_vision_outputs = F.permute(
            reshaped_vision_outputs, [0, 2, 3, 1]
        )
        pooled_vision_outputs = avg_pool2d(
            input=reshaped_vision_outputs,
            kernel_size=(self.kernel_size, self.kernel_size),
            stride=self.kernel_size,
        )

        pooled_vision_outputs = F.permute(pooled_vision_outputs, [0, 3, 1, 2])
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)

        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

        projected_vision_outputs = (
            normed_vision_outputs @ self.mm_input_projection_weight
        )

        image_hidden_states = F.flatten(
            projected_vision_outputs, start_dim=0, end_dim=1
        )

        return image_hidden_states


class Gemma3VisionMLP(Module):
    """Two-layer MLP with GELU activation for vision encoder."""

    def __init__(
        self,
        config: Gemma3ForConditionalGenerationConfig,
        device: DeviceRef | None = None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.vision_config.hidden_size
        self.intermediate_size = config.vision_config.intermediate_size
        self.device = device if device is not None else config.devices[0]

        self.fc1 = Linear(
            self.hidden_size,
            self.intermediate_size,
            has_bias=True,
        )

        self.fc2 = Linear(
            self.intermediate_size,
            self.hidden_size,
            has_bias=True,
        )

    def __call__(self, x: Tensor):
        """Expands hidden states to intermediate size, applies GELU activation,
        then projects back to hidden size."""
        x = self.fc1(x)
        x = F.gelu(x, self.config.vision_config.hidden_act, approximate=True)
        x = self.fc2(x)
        return x
