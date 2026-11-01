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
"""Registered Idefics3 decoder and vision modules."""

from __future__ import annotations

from max import tree
from max.dtype import DType
from max.graph import TensorType
from max.nn import Module
from max.pipelines.lib.role_contracts import DecoderInputs, DecoderOutputs
from max.tensor import Tensor, default_dtype

from .model_config import Idefics3Config
from .text_model.idefics3_text import Idefics3Language
from .vision_model.idefics3_vision import Idefics3VisionModel


class Idefics3Decoder(Module[[DecoderInputs[Tensor]], DecoderOutputs]):
    """Runs the text tower with the registered decoder input tree."""

    graph_name = "Idefics3Language"
    consumes_vision = True

    def __init__(self, config: Idefics3Config) -> None:
        super().__init__()
        self.model = Idefics3Language(
            config.text_config, config.image_token_id, config.get_kv_params()
        )
        self.model.name_transparent = True

    def forward(self, inputs: DecoderInputs[Tensor]) -> DecoderOutputs:
        """Runs the language tower with the graph's KV and vision fields."""
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


class Idefics3Vision(Idefics3VisionModel):
    """The vision tower, constructed from the architecture config."""

    graph_name = "Idefics3VisionModel"

    def __init__(self, config: Idefics3Config) -> None:
        with default_dtype(config.vision_config.dtype):
            super().__init__(config.vision_config)
        self.config = config
        self.embedding_dtype = config.vision_config.dtype
        self.embedding_dim = config.text_config.hidden_size

    def input_types(self) -> tuple[TensorType, ...]:
        """The pixel batch consumed by the vision tower."""
        size = self.config.vision_config.image_size
        return (
            TensorType(
                DType.bfloat16,
                ["batch_size", 3, size, size],
                device=self.config.devices[0],
            ),
        )
