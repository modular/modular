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

from .model_config import T5Config
from .t5 import T5EncoderModel


class T5Model(MaxModel):
    config_name = T5Config.config_name

    def __init__(
        self,
        config: dict,
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
    ) -> None:
        super().__init__(config, encoding, devices, weights)
        self.config = T5Config.generate(
            config,
            encoding,
            devices,
        )
        self.load_model()

    def load_model(self) -> Model:
        t5 = T5EncoderModel(self.config)

        if self.config.device.is_cpu():
            session = InferenceSession([CPU()])
        else:
            session = InferenceSession([Accelerator()])
        state_dict = {key: value.data() for key, value in self.weights.items()}
        t5.load_state_dict(state_dict)
        with Graph("t5_encoder_model", input_types=t5.input_types()) as graph:
            outputs = t5(
                input_ids=graph.inputs[0],
                attention_mask=None,
            )
            graph.output(outputs)
            compiled_graph = graph
        self.session = session.load(
            compiled_graph, weights_registry=t5.state_dict()
        )

    def __call__(self, *args, **kwargs):
        return self.session.execute(*args, **kwargs)
