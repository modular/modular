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
from max.experimental.tensor import Tensor
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel
from max.profiler import traced

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
        self.config = Flux2Config.initialize_from_config(
            config,
            encoding,
            devices,
        )
        self._enable_fbc = False
        self.load_model()

    @traced
    def load_model(self) -> Callable[..., Any]:
        state_dict = {key: value.data() for key, value in self.weights.items()}
        self._state_dict = state_dict
        # Klein/distilled checkpoints can omit guidance embedder weights.
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
        with F.lazy():
            flux = Flux2Transformer2DModel(self.config)
            flux.to(self.devices[0])
        self._flux_model = flux
        # Model is not yet compiled; compile_model() must be called before use.
        self.model = self._not_compiled
        return self.model

    @staticmethod
    def _not_compiled(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(
            "Flux2 transformer not compiled. Call compile_model() first."
        )

    @traced
    def compile_model(self, enable_fbc: bool) -> None:
        self._enable_fbc = enable_fbc
        self.model = self._flux_model.compile(
            *self._flux_model.input_types(step_cache_enabled=enable_fbc),
            weights=self._state_dict,
        )
        # Free weight dict and graph — no second compilation will happen.
        del self._state_dict
        del self._flux_model

    @traced
    def __call__(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        img_ids: Tensor,
        txt_ids: Tensor,
        guidance: Tensor,
        prev_residual: Tensor | None = None,
        prev_output: Tensor | None = None,
        rdt: Tensor | None = None,
    ) -> Any:
        if self._enable_fbc:
            return self.model(
                hidden_states,
                encoder_hidden_states,
                timestep,
                img_ids,
                txt_ids,
                guidance,
                prev_residual,
                prev_output,
                rdt,
            )

        return self.model(
            hidden_states,
            encoder_hidden_states,
            timestep,
            img_ids,
            txt_ids,
            guidance,
        )
