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
"""Gemma 3 decoder and vision modules for role registration."""

from __future__ import annotations

from max import tree
from max.driver import CPU
from max.dtype import DType
from max.graph import TensorType
from max.nn import Module
from max.pipelines.lib.role_contracts import DecoderInputs, DecoderOutputs
from max.tensor import Tensor, default_dtype

from .model_config import Gemma3ForConditionalGenerationConfig
from .vision_model.gemma3multimodal import (
    Gemma3LanguageModel,
    Gemma3VisionModel,
)


def _unquantized_dtype(config: Gemma3ForConditionalGenerationConfig) -> DType:
    """Matches the old default-device builds, independent of weight encoding."""
    return (
        DType.float32
        if isinstance(config.devices[0].to_device(), CPU)
        else DType.bfloat16
    )


class Gemma3Decoder(Module[[DecoderInputs[Tensor]], DecoderOutputs]):
    """Runs the existing multimodal language computation on registered inputs."""

    graph_name = "Gemma3LanguageModel"
    consumes_vision = True

    def __init__(self, config: Gemma3ForConditionalGenerationConfig) -> None:
        super().__init__()
        # FP8 checkpoints still keep the unquantized GPU parameters in bf16.
        with default_dtype(_unquantized_dtype(config)):
            self.model = Gemma3LanguageModel(config, config.kv_params)
        self.model.name_transparent = True

    def forward(self, inputs: DecoderInputs[Tensor]) -> DecoderOutputs:
        """Runs the language tower and names its logits and offsets."""
        assert inputs.vision is not None
        outputs = self.model(
            inputs.tokens,
            inputs.return_n_logits,
            inputs.input_row_offsets,
            inputs.vision.embeddings,
            inputs.vision.scatter_indices,
            *tree.leaves(inputs.kv),
        )
        return DecoderOutputs(
            next_token_logits=outputs[0],
            logits=outputs[1] if len(outputs) == 3 else None,
            logit_offsets=outputs[2] if len(outputs) == 3 else None,
        )


class Gemma3VisionEncoder(Module[[Tensor], Tensor]):
    """The first-device vision tower and multimodal projector."""

    graph_name = "Gemma3VisionModel"

    def __init__(self, config: Gemma3ForConditionalGenerationConfig) -> None:
        super().__init__()
        self.embedding_dtype = _unquantized_dtype(config)
        with default_dtype(self.embedding_dtype):
            self.model = Gemma3VisionModel(config)
        self.model.name_transparent = True
        self.config = config
        self.embedding_dim = config.text_config.hidden_size

    def input_types(self) -> tuple[TensorType, ...]:
        """The image batch consumed by the vision tower."""
        size = self.config.vision_config.image_size
        return (
            TensorType(
                DType.bfloat16,
                ["batch_size", 3, size, size],
                device=self.config.devices[0],
            ),
        )

    def forward(self, pixels: Tensor) -> Tensor:
        """Returns the flattened image embeddings."""
        return self.model(pixels)[0]
