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

from max.driver import Device
from max.experimental import functional as F
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel

from .clip import CLIPTextModel
from .model_config import ClipConfig


class ClipModel(ComponentModel):
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
        self.config = ClipConfig.generate(
            config,
            encoding,
            devices,
        )
        self.load_model()

    def load_model(self) -> Callable[..., Any]:
        state_dict = {key: value.data() for key, value in self.weights.items()}
        with F.lazy():
            clip = CLIPTextModel(self.config)
            clip.to(self.devices[0])
        self.model = clip.compile(*clip.input_types(), weights=state_dict)
        return self.model

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
