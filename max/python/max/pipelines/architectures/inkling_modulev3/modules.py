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
"""Inkling modules implementing the shared decoder and vision contracts."""

from __future__ import annotations

from max import tree
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.nn import Module
from max.pipelines.lib.role_contracts import (
    DecoderInputs,
    DecoderOutputs,
    VisionInputs,
)
from max.tensor import Tensor, default_dtype
from max.tensor.sharding import DeviceMesh

from .inkling import Inkling
from .layers.vision import InklingVisionModel
from .model_config import InklingConfig


class InklingDecoder(Module[[DecoderInputs[Tensor]], DecoderOutputs]):
    """Runs Inkling from semantic tokens, positions, vision and cache inputs."""

    graph_name = "inkling"
    consumes_vision = True
    input_layout = (
        "tokens",
        "input_row_offsets",
        "return_n_logits",
        "vision",
        "kv",
    )

    def __init__(self, config: InklingConfig) -> None:
        super().__init__()
        self.model = Inkling(
            config,
            return_logits=config.outputs.return_logits,
            return_hidden_states=config.outputs.return_hidden_states,
        )
        self.model.name_transparent = True

    @classmethod
    def vision_input_types(
        cls, config: InklingConfig, mesh: DeviceMesh
    ) -> VisionInputs[TensorType]:
        """Keeps the device-zero image inputs the module broadcasts."""
        device = DeviceRef.from_device(mesh.devices[0])
        return VisionInputs(
            embeddings=TensorType(
                config.dtype,
                ["total_image_tokens", config.text_config.hidden_size],
                device=device,
            ),
            scatter_indices=TensorType(
                DType.int32, ["total_image_tokens"], device=device
            ),
        )

    def forward(self, inputs: DecoderInputs[Tensor]) -> DecoderOutputs:
        """Runs both attention and convolution state from the shared KV tree."""
        assert inputs.vision is not None
        output = self.model(
            inputs.tokens,
            inputs.input_row_offsets,
            inputs.return_n_logits,
            inputs.vision.embeddings,
            inputs.vision.scatter_indices,
            *tree.leaves(inputs.kv),
        )
        return DecoderOutputs(
            next_token_logits=output.next_token_logits,
            logits=output.logits,
            logit_offsets=output.logit_offsets,
        )


class InklingVision(InklingVisionModel):
    """Constructs the vision role with the config's image normalization."""

    graph_name = "inkling_vision"

    def __init__(self, config: InklingConfig) -> None:
        with default_dtype(config.dtype):
            super().__init__(
                config.vision_config,
                config.dtype,
                image_mean=config.image_mean,
                image_std=config.image_std,
            )
        self._registered_config = config
        self.embedding_dtype = config.dtype
        self.embedding_dim = config.text_config.hidden_size

    def input_types(
        self, device: DeviceRef | None = None
    ) -> tuple[TensorType, ...]:
        """Returns the raw patch tensor on the encoder device."""
        return super().input_types(device or self._registered_config.devices[0])
