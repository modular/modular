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
"""Gemma4 decoder and optional image encoder role implementations."""

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

from .gemma4 import Gemma4
from .model_config import Gemma4Config
from .vision_model.vision_model import Gemma4VisionModel


class Gemma4Decoder(Module[[DecoderInputs[Tensor]], DecoderOutputs]):
    """Runs the existing language tower with the shared semantic input tree."""

    graph_name = "Gemma4"
    consumes_vision = True
    input_layout = (
        "tokens",
        "return_n_logits",
        "input_row_offsets",
        "vision",
        "kv",
    )

    def __init__(self, config: Gemma4Config) -> None:
        super().__init__()
        assert config.runtime is not None
        dtype = (
            config.unquantized_dtype
            if config.text_config.quant_config is not None
            else config.dtype
        )
        with default_dtype(dtype):
            self.model = Gemma4(config, config.kv_params, config.runtime.mesh)
        self.model.name_transparent = True

    @classmethod
    def vision_input_types(
        cls, config: Gemma4Config, mesh: DeviceMesh
    ) -> VisionInputs[TensorType]:
        """Declares device-zero image inputs even for text-only checkpoints."""
        device = DeviceRef.from_device(mesh.devices[0])
        return VisionInputs(
            embeddings=TensorType(
                config.unquantized_dtype,
                ["num_image_tokens", config.text_config.hidden_size],
                device=device,
            ),
            scatter_indices=TensorType(
                DType.int32, ["total_image_tokens"], device=device
            ),
        )

    def forward(self, inputs: DecoderInputs[Tensor]) -> DecoderOutputs:
        """Runs text and image rows, retaining the existing KV layout."""
        assert inputs.vision is not None
        output = self.model(
            inputs.tokens,
            inputs.return_n_logits,
            inputs.input_row_offsets,
            inputs.vision.embeddings,
            inputs.vision.scatter_indices,
            *tree.leaves(inputs.kv),
        )
        return DecoderOutputs(
            next_token_logits=output[0],
            logits=output[1] if len(output) == 3 else None,
            logit_offsets=output[2] if len(output) == 3 else None,
        )


class Gemma4Vision(Gemma4VisionModel):
    """Constructs the optional image tower in its unquantized dtype."""

    graph_name = "Gemma4VisionModel"

    @classmethod
    def is_enabled(cls, config: Gemma4Config) -> bool:
        """Skips the encoder for the text-only Unified checkpoint line."""
        return config.vision_config is not None

    def __init__(self, config: Gemma4Config) -> None:
        with default_dtype(config.unquantized_dtype):
            super().__init__(config)
        self.config = config
        self.embedding_dtype = config.unquantized_dtype
        self.embedding_dim = config.text_config.hidden_size

    def input_types(self) -> tuple[TensorType, ...]:
        """Returns the encoder's patch, position and pooling inputs."""
        config = self.config
        assert config.vision_config is not None
        device = config.devices[0]
        patch_dim = 3 * config.vision_config.patch_size**2
        return (
            TensorType(
                config.unquantized_dtype,
                ["total_patches", patch_dim],
                device=device,
            ),
            TensorType(DType.int32, ["total_patches", 2], device=device),
            TensorType(DType.uint32, ["num_images_plus_1"], device=device),
            TensorType(
                DType.int32,
                ["num_pooled_tokens", "max_pool_patches"],
                device=device,
            ),
            TensorType(DType.uint32, [], device=DeviceRef.CPU()),
        )
