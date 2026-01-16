# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from max.driver import CPU, Accelerator, Device
from max.engine import InferenceSession, Model
from max.graph import Graph
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.max_model import MaxModel

from .flux1 import FluxTransformer2DModel
from .model_config import FluxConfig
from .weight_adapters import convert_safetensor_state_dict


class Flux1Model(MaxModel):
    config_name = FluxConfig.config_name

    def __init__(
        self,
        config: dict,
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
    ) -> None:
        super().__init__(
            config,
            encoding,
            devices,
            weights,
        )
        self.config = FluxConfig.generate(
            config,
            encoding,
            devices,
        )
        self.load_model()

    def load_model(self) -> Model:
        flux = FluxTransformer2DModel(self.config)

        if self.config.device.is_cpu():
            session = InferenceSession([CPU()])
        else:
            session = InferenceSession([Accelerator()])
        state_dict = {key: value.data() for key, value in self.weights.items()}
        state_dict = convert_safetensor_state_dict(state_dict)
        flux.load_state_dict(state_dict)
        with Graph(
            "flux_transformer_2d_model", input_types=flux.input_types()
        ) as graph:
            outputs = flux(
                *graph.inputs,
                joint_attention_kwargs={},
                controlnet_block_samples=None,
                controlnet_single_block_samples=None,
                return_dict=False,
                controlnet_blocks_repeat=False,
            )
            graph.output(*outputs)
            compiled_graph = graph
        self.session = session.load(
            compiled_graph, weights_registry=flux.state_dict()
        )

    def __call__(self, *args, **kwargs):
        return self.session.execute(*args, **kwargs)
