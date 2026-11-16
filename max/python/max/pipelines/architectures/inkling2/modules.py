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
"""Inkling2 decoder and vision modules on the shared contracts."""

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
from .model_config import Inkling2Config


class Inkling2Decoder(Module[[DecoderInputs[Tensor]], DecoderOutputs]):
    """Runs Inkling2 from semantic tokens, positions, vision and cache inputs."""

    graph_name = "inkling2"

    consumes_vision = True

    input_layout = (
        "tokens",
        "return_n_logits",
        "input_row_offsets",
        "token_positions",
        "vision",
        "kv",
    )
    """``Inkling.forward``'s positional order, spelled as role fields.

    The ModuleV3 ragged ABI puts the logit count second and the row offsets
    third (``max/pipelines/lib/interfaces/batch_processor.py:785``), and the
    position ramp follows the offsets, before the image rows and the cache.
    The conv state rides inside ``kv``: the cache tree has one child holding
    every layer's convolution state, so a single walk covers the attention
    pages and the state leaves both (``model_config.STATE_CACHE_KEY``).
    """

    token_position_dtype = DType.uint32

    def __init__(self, config: Inkling2Config) -> None:
        super().__init__()
        self.model = Inkling(
            config,
            return_logits=config.return_logits,
            return_hidden_states=config.return_hidden_states,
        )
        # The decoder's weights are the language model's own leaves, so the
        # graph is name-transparent: the weight rules' role prefix is what
        # places them.
        self.model.name_transparent = True

    @classmethod
    def vision_input_types(
        cls, config: Inkling2Config, mesh: DeviceMesh
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
        """Runs the text model with the position ramp and the image rows."""
        assert inputs.vision is not None
        assert inputs.token_positions is not None
        output = self.model(
            inputs.tokens,
            inputs.return_n_logits,
            inputs.input_row_offsets,
            inputs.token_positions,
            inputs.vision.embeddings,
            inputs.vision.scatter_indices,
            *tree.leaves(inputs.kv),
        )
        return DecoderOutputs(
            next_token_logits=output.next_token_logits,
            logits=output.logits,
            logit_offsets=output.logit_offsets,
        )


class Inkling2Vision(InklingVisionModel):
    """The vision role: a config-taking constructor and a device-free graph."""

    graph_name = "inkling2_vision"

    def __init__(self, config: Inkling2Config) -> None:
        with default_dtype(config.dtype):
            super().__init__(config.vision_config, config.dtype)
        self._registered_config = config
        # What a decoder that declares no ``vision_input_types`` reads off the
        # module: the width and dtype of the rows it emits.
        self.embedding_dtype = config.dtype
        self.embedding_dim = config.text_config.hidden_size

    def input_types(
        self, device: DeviceRef | None = None
    ) -> tuple[TensorType, ...]:
        """Returns the raw patch tensor plus its taps and position ids."""
        return super().input_types(device or self._registered_config.devices[0])


__all__ = ["Inkling2Decoder", "Inkling2Vision"]
