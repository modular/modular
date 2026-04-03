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

from .model_config import ZImageConfig
from .weight_adapters import convert_z_image_transformer_state_dict
from .z_image import ZImageTransformer2DModel


class ZImageTransformerModel(ComponentModel):
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
        self.config = ZImageConfig.initialize_from_config(
            config,
            encoding,
            devices,
        )
        self.load_model()

    def load_model(self) -> Callable[..., Any]:
        target_dtype = self.config.dtype
        raw_state_dict = {}
        for key, value in self.weights.items():
            weight = value.data()
            if hasattr(weight, "dtype") and hasattr(weight, "astype"):
                if weight.dtype != target_dtype:
                    if weight.dtype.is_float() and target_dtype.is_float():
                        weight = weight.astype(target_dtype)
            raw_state_dict[key] = weight
        state_dict = convert_z_image_transformer_state_dict(raw_state_dict)

        nn_model = ZImageTransformer2DModel(self.config)
        nn_model.load_state_dict(state_dict, weight_alignment=1, strict=True)
        self.state_dict = nn_model.state_dict()

        with Graph(
            "z_image_transformer",
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
    ) -> list[Buffer]:
        return self.model.execute(
            hidden_states,
            encoder_hidden_states,
            timestep,
            img_ids,
            txt_ids,
        )
