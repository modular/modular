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

from max.driver import Device
from max.engine import Model
from max.experimental import functional as F
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
        state_dict = {key: value.data() for key, value in self.weights.items()}
        state_dict = convert_safetensor_state_dict(state_dict)
        with F.lazy():
            flux = FluxTransformer2DModel(self.config)
            flux.to(self.devices[0])
        self.model = flux.compile(*flux.input_types(), weights=state_dict)
        return self.model

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
