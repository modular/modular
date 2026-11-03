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
"""Registered Muse Spark text and vision computations."""

from __future__ import annotations

from max import tree
from max.graph import TensorType
from max.nn import Module
from max.pipelines.lib.role_contracts import DecoderInputs, DecoderOutputs
from max.tensor import Tensor, default_dtype

from .model_config import MuseSparkConfig
from .muse_spark import MuseSpark, compute_dtype
from .vision import MuseSparkVisionModel, vision_input_types


class MuseSparkDecoder(Module[[DecoderInputs[Tensor]], DecoderOutputs]):
    """Runs Spark's text tower on the registered KV and image trees."""

    graph_name = "MuseSpark"
    consumes_vision = True
    input_layout = (
        "tokens",
        "return_n_logits",
        "input_row_offsets",
        "vision",
        "kv",
    )

    def __init__(self, config: MuseSparkConfig) -> None:
        super().__init__()
        assert config.mesh is not None
        with default_dtype(compute_dtype(config)):
            self.model = MuseSpark(config, config.mesh)
        self.model.name_transparent = True

    def forward(self, inputs: DecoderInputs[Tensor]) -> DecoderOutputs:
        """Runs the text computation, scattering the batch's image rows."""
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


class MuseSparkVision(MuseSparkVisionModel):
    """Muse Glimmer's vision tower, built from Spark's config."""

    graph_name = "MuseSparkVisionModel"

    def __init__(self, config: MuseSparkConfig) -> None:
        dtype = compute_dtype(config)
        with default_dtype(dtype):
            super().__init__(
                config.vision_config,
                config.text_config.hidden_size,
                config.text_config.rms_norm_eps,
            )
        self.config = config
        self.embedding_dtype = dtype
        self.embedding_dim = config.text_config.hidden_size

    def input_types(self) -> tuple[TensorType, ...]:
        """Returns the packed image graph's input signature."""
        return vision_input_types(
            self.config.vision_config,
            self.config.devices[0],
            self.embedding_dtype,
        )
