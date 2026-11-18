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
"""Inkling 2 modules implementing the shared decoder contract."""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.nn import Module
from max.pipelines.lib.role_contracts import (
    DecoderInputs,
    DecoderOutputs,
    VisionInputs,
)
from max.tensor import Tensor
from max.tensor.sharding import DeviceMesh

from .inkling2 import Inkling2
from .model_config import Inkling2Config


class Inkling2Decoder(Module[[DecoderInputs[Tensor]], DecoderOutputs]):
    """Runs Inkling 2 from semantic tokens, row offsets and the KV tree."""

    consumes_vision = True
    # Mach's registry entry binds this graph input order: the two image inputs
    # (image_embeddings, image_token_indices) sit before the KV cache.
    input_layout = (
        "tokens",
        "input_row_offsets",
        "return_n_logits",
        "vision",
        "kv",
    )

    def __init__(self, config: Inkling2Config) -> None:
        super().__init__()
        self.model = Inkling2(config)
        self.model.name_transparent = True

    @classmethod
    def vision_input_types(
        cls, config: Inkling2Config, mesh: DeviceMesh
    ) -> VisionInputs[TensorType]:
        """Takes the image inputs on device zero, which the model broadcasts."""
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
        return self.model(
            inputs.tokens,
            inputs.input_row_offsets,
            inputs.return_n_logits,
            inputs.vision,
            inputs.kv,
        )
