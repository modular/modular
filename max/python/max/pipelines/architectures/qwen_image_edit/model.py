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

"""QwenImage Edit transformer model.

The edit path uses the same MAX-native module_v2 transformer graph as
text-to-image, but derives condition-token masking dynamically from the image
token IDs. That keeps the graph shape-dynamic and avoids recompiles when edit
requests change image resolution or denoising step count across runs.
"""

from collections.abc import Callable
from typing import Any

from max.driver import Buffer, Device
from max.engine import InferenceSession, Model
from max.graph import Graph
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel

from ..qwen_image.model_config import QwenImageConfig
from ..qwen_image.qwen_image import QwenImageTransformer2DModel


class QwenImageEditTransformerModel(ComponentModel):
    """Edit-specific transformer compiled once with dynamic token masking."""

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
        session: InferenceSession,
    ) -> None:
        super().__init__(config, encoding, devices, weights)
        self.session = session
        self.config = QwenImageConfig.generate(config, encoding, devices)
        self.load_model()

    def load_model(self) -> Callable[..., Any]:
        state_dict = {key: value.data() for key, value in self.weights.items()}

        nn_model = QwenImageTransformer2DModel(self.config)
        nn_model.load_state_dict(state_dict, weight_alignment=1, strict=True)
        self.state_dict = nn_model.state_dict()

        with Graph(
            "qwen_image_edit_transformer",
            input_types=nn_model.input_types(),
        ) as graph:
            outputs = nn_model(*(value.tensor for value in graph.inputs))
            if isinstance(outputs, tuple):
                graph.output(*outputs)
            else:
                graph.output(outputs)

        self.model: Model = self.session.load(
            graph,
            weights_registry=self.state_dict,
        )
        return self.model.execute

    def __call__(
        self,
        hidden_states: Buffer,
        encoder_hidden_states: Buffer,
        timestep: Buffer,
        img_ids: Buffer,
        txt_ids: Buffer,
    ) -> Any:
        return self.model.execute(
            hidden_states,
            encoder_hidden_states,
            timestep,
            img_ids,
            txt_ids,
        )
