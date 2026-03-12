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

from .flux1 import FluxTransformer2DModel
from .model_config import FluxConfig
from .weight_adapters import convert_safetensor_state_dict


class Flux1TransformerModel(ComponentModel):
    def _model_not_loaded(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(
            "Flux1 transformer model is not ready yet. "
            "Call use_standard_model or use_step_cache_model first."
        )

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
        self.config = FluxConfig.initialize_from_config(
            config,
            encoding,
            devices,
        )
        self.load_model()

    def load_model(self) -> Callable[..., Any]:
        state_dict = {key: value.data() for key, value in self.weights.items()}
        state_dict = convert_safetensor_state_dict(state_dict)
        self._state_dict = state_dict
        with F.lazy():
            flux = FluxTransformer2DModel(self.config)
            flux.to(self.devices[0])
        self._flux_model = flux
        self._standard_model: Callable[..., Any] | None = None
        self._step_cache_model: Callable[..., Any] | None = None
        self.model = self._model_not_loaded
        return self.model

    def _ensure_standard_model(self) -> Callable[..., Any]:
        if self._standard_model is None:
            self._standard_model = self._flux_model.compile(
                *self._flux_model.input_types(step_cache_enabled=False),
                weights=self._state_dict,
            )
        return self._standard_model

    def _ensure_step_cache_model(self) -> Callable[..., Any]:
        if self._step_cache_model is None:
            self._step_cache_model = self._flux_model.compile(
                *self._flux_model.input_types(step_cache_enabled=True),
                weights=self._state_dict,
            )
        return self._step_cache_model

    def use_standard_model(self) -> None:
        standard_model = self._ensure_standard_model()
        if self.model is self._step_cache_model:
            self._step_cache_model = None
        self.model = standard_model

    def use_step_cache_model(self) -> None:
        step_cache_model = self._ensure_step_cache_model()
        if self.model is self._standard_model:
            self._standard_model = None
        self.model = step_cache_model

    def __call__(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        pooled_projections: Tensor,
        timestep: Tensor,
        img_ids: Tensor,
        txt_ids: Tensor,
        guidance: Tensor | None,
        prev_residual: Tensor | None = None,
        prev_output: Tensor | None = None,
        step_cache_flag: Tensor | None = None,
        rdt: Tensor | None = None,
    ) -> Any:
        if (
            prev_residual is not None
            and prev_output is not None
            and step_cache_flag is not None
            and rdt is not None
        ):
            model = self._ensure_step_cache_model()
            return model(
                hidden_states,
                encoder_hidden_states,
                pooled_projections,
                timestep,
                img_ids,
                txt_ids,
                guidance,
                prev_residual,
                prev_output,
                step_cache_flag,
                rdt,
            )

        model = (
            self._ensure_standard_model()
            if self.model is self._model_not_loaded
            else self.model
        )
        return model(
            hidden_states,
            encoder_hidden_states,
            pooled_projections,
            timestep,
            img_ids,
            txt_ids,
            guidance,
        )

    def call_with_step_cache(self, *args: Any, **kwargs: Any) -> Any:
        return self(*args, **kwargs)
