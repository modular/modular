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

from .flux2 import Flux2Transformer2DModel
from .model_config import Flux2Config


class Flux2Model(ComponentModel):
    config_name = Flux2Config.config_name

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
        self._compiled_shapes: tuple[int, int, int] | None = None
        self.load_model()

    def load_model(self) -> Callable[..., Any]:
        self._state_dict = {
            key: value.data() for key, value in self.weights.items()
        }
        with F.lazy():
            self._flux2 = Flux2Transformer2DModel(self.config)
            self._flux2.to(self.devices[0])
        return self.__call__

    def _ensure_compiled(
        self,
        batch_size: int,
        image_seq_len: int,
        text_seq_len: int,
    ) -> Model:
        current_shapes = (batch_size, image_seq_len, text_seq_len)

        # Recompile if shapes changed or not yet compiled
        if (
            self._compiled_model is None
            or self._compiled_shapes != current_shapes
        ):
            if self._flux2 is None or self._state_dict is None:
                raise RuntimeError("Model not loaded. Call load_model() first.")

            input_types = self._flux2.input_types_with_shapes(
                batch_size=batch_size,
                image_seq_len=image_seq_len,
                text_seq_len=text_seq_len,
            )
            compiled = self._flux2.compile(
                *input_types, weights=self._state_dict
            )
            # compile returns a callable, but we need to store it as Model
            # The actual Model is created when the callable is invoked
            self._compiled_model = compiled  # type: ignore[assignment]
            self._compiled_shapes = current_shapes

        if self._compiled_model is None:
            raise RuntimeError("Model compilation failed")
        return self._compiled_model

    def __call__(
        self,
        hidden_states: Any,
        encoder_hidden_states: Any,
        timestep: Any,
        img_ids: Any,
        txt_ids: Any,
        guidance: Any,
    ) -> Any:
        # Extract shapes for compilation
        # Handle both Tensor_v3 and driver.Tensor
        if hasattr(hidden_states, "shape"):
            hs_shape = hidden_states.shape
            batch_size = (
                hs_shape[0].dim if hasattr(hs_shape[0], "dim") else hs_shape[0]
            )
            image_seq_len = (
                hs_shape[1].dim if hasattr(hs_shape[1], "dim") else hs_shape[1]
            )
        else:
            raise ValueError("hidden_states must have a shape attribute")

        if hasattr(encoder_hidden_states, "shape"):
            enc_shape = encoder_hidden_states.shape
            text_seq_len = (
                enc_shape[1].dim
                if hasattr(enc_shape[1], "dim")
                else enc_shape[1]
            )
        else:
            raise ValueError(
                "encoder_hidden_states must have a shape attribute"
            )

        # Ensure model is compiled for these shapes
        model = self._ensure_compiled(
            batch_size=int(batch_size),
            image_seq_len=int(image_seq_len),
            text_seq_len=int(text_seq_len),
        )

        return model(
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
