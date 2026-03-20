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

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from max.driver import Device
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel

if TYPE_CHECKING:
    from max.pipelines.lib.interfaces.cache_mixin import DenoisingCacheConfig

from .flux1 import FluxTransformer2DModel
from .model_config import FluxConfig
from .weight_adapters import convert_safetensor_state_dict


class Flux1TransformerModel(ComponentModel):
    model: Callable[..., Any]

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
        *,
        cache_config: DenoisingCacheConfig | None = None,
    ) -> None:
        super().__init__(
            config,
            encoding,
            devices,
            weights,
            cache_config=cache_config,
        )
        self.config = FluxConfig.initialize_from_config(
            config,
            encoding,
            devices,
        )
        self.load_model()

    def load_model(self) -> None:
        state_dict = {key: value.data() for key, value in self.weights.items()}
        state_dict = convert_safetensor_state_dict(state_dict)
        with F.lazy():
            flux = FluxTransformer2DModel(self.config)
            flux.to(self.devices[0])

        if (
            self.cache_config is not None
            and self.cache_config.first_block_caching
        ):
            assert self.cache_config.residual_threshold is not None
            flux._step_cache_enabled = True
            flux._rdt_value = self.cache_config.residual_threshold
        else:
            flux._step_cache_enabled = False

        self.model = flux.compile(
            *flux.input_types(),
            weights=state_dict,
        )

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
    ) -> Any:
        args: tuple[Any, ...] = (
            hidden_states,
            encoder_hidden_states,
            pooled_projections,
            timestep,
            img_ids,
            txt_ids,
            guidance,
        )
        if prev_residual is not None:
            args = (*args, prev_residual, prev_output)
        return self.model(*args)
