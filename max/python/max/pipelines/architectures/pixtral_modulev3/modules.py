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
"""Registered Pixtral text and vision computations."""

from __future__ import annotations

from max import tree
from max.dtype import DType
from max.graph import TensorType
from max.nn import Module
from max.pipelines.lib.role_contracts import (
    DecoderInputs,
    DecoderOutputs,
    VisionInputs,
)
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.sharding import DeviceMesh

from .model_config import PixtralConfig
from .pixtral import PixtralLanguage, PixtralVision


class PixtralDecoder(Module[[DecoderInputs[Tensor]], DecoderOutputs]):
    """Feeds the shared decoder fields to Pixtral's existing text tower."""

    graph_name = "PixtralLanguage"
    consumes_vision = True
    input_layout = (
        "tokens",
        "input_row_offsets",
        "return_n_logits",
        "vision",
        "kv",
    )

    def __init__(self, config: PixtralConfig) -> None:
        super().__init__()
        with default_device(config.devices[0].to_device()):
            self.model = PixtralLanguage(config)
        self.model.name_transparent = True

    @classmethod
    def vision_input_types(
        cls, config: PixtralConfig, mesh: DeviceMesh
    ) -> VisionInputs[TensorType]:
        """Keeps the single-device tensors consumed by Pixtral's graph bridge."""
        del mesh
        return VisionInputs(
            embeddings=TensorType(
                config.dtype,
                ["total_image_tokens", config.hidden_size],
                device=config.devices[0],
            ),
            scatter_indices=TensorType(
                DType.int32,
                ["num_image_token_indices"],
                device=config.devices[0],
            ),
        )

    def forward(self, inputs: DecoderInputs[Tensor]) -> DecoderOutputs:
        """Runs the text tower with the shared KV and vision trees."""
        assert inputs.vision is not None
        outputs = self.model(
            inputs.tokens,
            inputs.input_row_offsets,
            inputs.return_n_logits,
            inputs.vision.embeddings,
            inputs.vision.scatter_indices,
            *tree.leaves(inputs.kv),
        )
        return DecoderOutputs(
            next_token_logits=outputs[0],
            logits=outputs[1] if len(outputs) == 3 else None,
            logit_offsets=outputs[2] if len(outputs) == 3 else None,
        )


class PixtralRegisteredVision(PixtralVision):
    """Declares the ragged patch graph and its embedding row type."""

    graph_name = "PixtralVision"

    def __init__(self, config: PixtralConfig) -> None:
        with default_dtype(config.dtype):
            super().__init__(config)
        self.config = config
        self.embedding_dtype = config.dtype
        self.embedding_dim = config.hidden_size

    def input_types(self) -> tuple[TensorType, ...]:
        """Returns patches, block-diagonal attention, and spatial positions."""
        config = self.config
        device = config.devices[0]
        patch_dim = config.num_channels * config.patch_size**2
        return (
            TensorType(
                DType.float32, ["total_patches", patch_dim], device=device
            ),
            TensorType(
                DType.float32,
                [1, 1, "total_patches", "total_patches"],
                device=device,
            ),
            TensorType(DType.int64, ["total_patches"], device=device),
        )
