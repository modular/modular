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

from max.driver import Tensor
from max.experimental.functional import matmul
from max.graph import (
    TensorValue,
    Weight,
    ops,
)
from max.graph.ops import avg_pool2d
from max.nn import (
    Linear,
    Module,
)
from max.pipelines.architectures.gemma3.layers.rms_norm import Gemma3RMSNorm

from ..model_config import Gemma3ForConditionalGenerationConfig


# ✅ working, based on HF and MLX-VLM
class Gemma3MultiModalProjector(Module):
    def __init__(self, config: Gemma3ForConditionalGenerationConfig):
        super().__init__()

        self.devices = config.devices

        self.mm_input_projection_weight = Weight(
            "mm_input_projection_weight",
            dtype=config.dtype,
            shape=(
                config.vision_config.hidden_size,
                config.text_config.hidden_size,
            ),
            device=self.devices[0],
        )

        self.mm_soft_emb_norm = Gemma3RMSNorm(
            config.vision_config.hidden_size,
            eps=config.vision_config.layer_norm_eps,
            dtype=config.dtype,
        )

        self.patches_per_image = int(
            config.vision_config.image_size // config.vision_config.patch_size
        )  # 64

        self.tokens_per_side = int(
            config.mm_tokens_per_image**0.5
        )  # 256 ** 05 = 16
        self.kernel_size = (
            self.patches_per_image // self.tokens_per_side
        )  # 64 / 16 = 4

    def __call__(self, vision_outputs: Tensor):
        batch_size, _, seq_length = (
            vision_outputs.shape
        )  # TensorValue shape: ['batch_size', 4096, 1152]

        reshaped_vision_outputs = vision_outputs.transpose(1, 2)

        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            (
                batch_size,
                seq_length,
                self.patches_per_image,
                self.patches_per_image,
            )
        )

        # reshape to 0 2 3 1 HWBC
        reshaped_vision_outputs = reshaped_vision_outputs.permute(0, 2, 3, 1)
        # avg pool
        pooled_vision_outputs = avg_pool2d(
            input=normed_vision_outputs,
            kernel_size=(self.kernel_size, self.kernel_size),  # (4,4)
            stride=self.kernel_size,  # 4
        )
        # reshape 0 3 1 2.flatten(2)
        pooled_vision_outputs = pooled_vision_outputs.transpose(
            0, 3, 1, 2
        ).pooled_vision_outputs.flatten(2)
        # transpose 0 2 1?
        pooled_vision_outputs = vision_outputs.transpose(1, 2)

        normed_vision_outputs = self.mm_soft_emb_norm(reshaped_vision_outputs)

        projected_vision_outputs = matmul(
            normed_vision_outputs, self.mm_input_projection_weight
        )
        return projected_vision_outputs.type_as(vision_outputs)


# ✅ working, based on HF and MLX-VLM
class Gemma3VisionMLP(Module):
    def __init__(self, config: Gemma3ForConditionalGenerationConfig):
        super().__init__()
        self.hidden_size = config.vision_config.hidden_size
        self.intermediate_size = config.vision_config.intermediate_size

        self.fc1 = Linear(
            self.hidden_size,
            self.intermediate_size,
            dtype=config.dtype,
            device=config.devices[0],
            has_bias=False,
        )

        self.fc2 = Linear(
            self.intermediate_size,
            self.hidden_size,
            dtype=config.dtype,
            device=config.devices[0],
            has_bias=False,
        )

    def __call__(self, x: TensorValue):
        x = self.fc1(x)
        x = ops.gelu(x)
        x = self.fc2(x)
        return x
