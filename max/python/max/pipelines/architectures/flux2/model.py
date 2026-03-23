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

from collections.abc import Callable
from typing import Any

from max.driver import Buffer, Device
from max.engine import InferenceSession, Model
from max.graph import Graph
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel

from .flux2 import Flux2Transformer2DModel
from .model_config import Flux2Config


class Flux2TransformerModel(ComponentModel):
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
        self.config = Flux2Config.initialize_from_config(
            config,
            encoding,
            devices,
        )
        self.load_model()

    def load_model(self) -> Callable[..., Any]:
        state_dict = {key: value.data() for key, value in self.weights.items()}

        has_guidance_embedder = any(
            "time_guidance_embed.guidance_embedder." in k for k in state_dict
        )
        if not has_guidance_embedder and getattr(
            self.config, "guidance_embeds", True
        ):
            if hasattr(self.config, "model_copy"):
                self.config = self.config.model_copy(
                    update={"guidance_embeds": False}
                )
            else:
                self.config.guidance_embeds = False

        nn_model = Flux2Transformer2DModel(self.config)
        nn_model.load_state_dict(state_dict, weight_alignment=1, strict=True)
        self.state_dict = nn_model.state_dict()

        with Graph(
            "flux2_transformer",
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
        guidance: Buffer,
    ) -> list[Buffer]:
        return self.model.execute(
            hidden_states,
            encoder_hidden_states,
            timestep,
            img_ids,
            txt_ids,
            guidance,
        )
