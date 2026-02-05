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

from max import functional as F
from max.driver import Device
from max.engine import Model
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel
from max.tensor import Tensor

from .flux2 import Flux2Transformer2DModel
from .model_config import Flux2Config


class Flux2TransformerModel(ComponentModel):
    def __init__(
        self,
        config: dict[str, Any],
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
        self.config = Flux2Config.generate(
            config,
            encoding,
            devices,
        )
        self._flux2: Flux2Transformer2DModel | None = None
        self._state_dict: dict[str, Any] | None = None
        self._compiled_model: Model | None = None
        self.load_model()

    def load_model(self) -> Callable[..., Any]:
        self._state_dict = {
            key: value.data() for key, value in self.weights.items()
        }
        with F.lazy():
            self._flux2 = Flux2Transformer2DModel(self.config)
            self._flux2.to(self.devices[0])
        self._compiled_model = self._flux2.compile(
            *self._flux2.input_types(), weights=self._state_dict
        )  # type: ignore[assignment]
        return self.__call__

    def __call__(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        img_ids: Tensor,
        txt_ids: Tensor,
        guidance: Tensor,
    ) -> tuple[Tensor]:
        if self._compiled_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._compiled_model(
            hidden_states,
            encoder_hidden_states,
            timestep,
            img_ids,
            txt_ids,
            guidance,
        )

    @property
    def model(self) -> Model | None:
        return self._compiled_model
