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
"""Registered Muse Glimmer text and optional vision computations."""

from __future__ import annotations

from max import tree
from max.graph import TensorType
from max.nn import Module
from max.pipelines.lib.module_registration import decoder_vision_input_types
from max.pipelines.lib.role_contracts import (
    DecoderInputs,
    DecoderOutputs,
    VisionInputs,
)
from max.tensor import Tensor, default_dtype
from max.tensor.sharding import DeviceMesh, TensorLayout

from .model_config import MuseGlimmerConfig
from .muse_glimmer import MuseGlimmer
from .vision import MuseGlimmerVisionModel, vision_input_types


class MuseGlimmerDecoder(Module[[DecoderInputs[Tensor]], DecoderOutputs]):
    """Runs the existing hybrid-KV text tower with semantic role inputs."""

    graph_name = "MuseGlimmer"
    consumes_vision = True
    input_layout = (
        "tokens",
        "return_n_logits",
        "input_row_offsets",
        "vision",
        "kv",
    )

    def __init__(self, config: MuseGlimmerConfig) -> None:
        super().__init__()
        assert config.mesh is not None
        self.model = MuseGlimmer(config, config.mesh)
        self.model.name_transparent = True

    @classmethod
    def vision_input_types(
        cls, config: MuseGlimmerConfig, mesh: DeviceMesh
    ) -> VisionInputs[TensorLayout]:
        """Preserves empty image fields when a checkpoint has no vision tower."""
        return decoder_vision_input_types(
            mesh, config.dtype, config.text_config.hidden_size
        )

    def forward(self, inputs: DecoderInputs[Tensor]) -> DecoderOutputs:
        """Runs the text computation with the registered KV and image trees."""
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


class MuseGlimmerVision(MuseGlimmerVisionModel):
    """Constructs the image tower only for checkpoints that declare one."""

    graph_name = "MuseGlimmerVisionModel"

    @classmethod
    def is_enabled(cls, config: MuseGlimmerConfig) -> bool:
        """Returns whether this checkpoint includes a vision tower."""
        return config.vision_config is not None

    def __init__(self, config: MuseGlimmerConfig) -> None:
        assert config.vision_config is not None
        with default_dtype(config.dtype):
            super().__init__(
                config.vision_config,
                config.text_config.hidden_size,
                config.text_config.rms_norm_eps,
            )
        self.config = config
        self.embedding_dtype = config.dtype
        self.embedding_dim = config.text_config.hidden_size

    def input_types(self) -> tuple[TensorType, ...]:
        """Returns the existing packed image graph's input signature."""
        assert self.config.vision_config is not None
        return vision_input_types(
            self.config.vision_config, self.config.devices[0], self.config.dtype
        )
