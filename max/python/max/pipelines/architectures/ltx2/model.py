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

import max.experimental.functional as F
import max.nn.module_v3 as nn
from max.driver import Device
from max.experimental.tensor import Tensor
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel

from .ltx2 import LTX2VideoTransformer3DModel
from .model_config import (
    LTX2TextConnectorsConfig,
    LTX2TransformerConfig,
    LTX2VocoderConfig,
)
from .nn.connectors import LTX2TextConnectors
from .nn.vocoder import LTX2Vocoder


def _reconcile_dtypes(
    state_dict: dict[str, Any], model: nn.Module[..., Any]
) -> dict[str, Any]:
    """Cast state_dict values to match model parameter dtypes.

    Models use mixed dtypes (e.g. BF16 for Linear weights, F32 for
    scale_shift_table). The compile(weights=) path requires exact dtype
    match, so we cast each weight to the model parameter's expected dtype.
    """
    param_dtypes = {name: param.dtype for name, param in model.parameters}
    for key in list(state_dict.keys()):
        if key in param_dtypes:
            w = state_dict[key]
            target_dtype = param_dtypes[key]
            if hasattr(w, "dtype") and w.dtype != target_dtype:
                state_dict[key] = w.astype(target_dtype)
    return state_dict


class LTX2TransformerModel(ComponentModel):
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
        self.config = LTX2TransformerConfig.generate(
            config,
            encoding,
            devices,
        )
        self.load_model()

    def load_model(self) -> Callable[..., Any]:
        state_dict = {
            key: value.data()  # .astype(self.config.dtype)
            for key, value in self.weights.items()
        }
        with F.lazy():
            ltx2transformer = LTX2VideoTransformer3DModel(self.config)
            ltx2transformer.to(self.devices[0])
        state_dict = _reconcile_dtypes(state_dict, ltx2transformer)
        self.model = ltx2transformer.compile(
            *ltx2transformer.input_types(), weights=state_dict
        )
        return self.model

    def __call__(
        self,
        hidden_states: Tensor,
        audio_hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        audio_encoder_hidden_states: Tensor,
        timestep: Tensor,
        audio_timestep: Tensor,
        video_coords: Tensor,
        audio_coords: Tensor,
    ) -> Any:
        return self.model(
            hidden_states,
            audio_hidden_states,
            encoder_hidden_states,
            audio_encoder_hidden_states,
            timestep,
            audio_timestep,
            video_coords,
            audio_coords,
        )


class LTX2VocoderModel(ComponentModel):
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
        self.config = LTX2VocoderConfig.generate(
            config,
            encoding,
            devices,
        )
        self.load_model()

    def load_model(self) -> Callable[..., Any]:
        state_dict = {
            key: value.data()  # .astype(self.config.dtype)
            for key, value in self.weights.items()
        }
        with F.lazy():
            vocoder = LTX2Vocoder(self.config)
            vocoder.to(self.devices[0])
        state_dict = _reconcile_dtypes(state_dict, vocoder)
        self.model = vocoder.compile(*vocoder.input_types(), weights=state_dict)
        return self.model

    def __call__(self, mel_spectrogram: Tensor, **kwargs: Any) -> Any:
        return self.model(mel_spectrogram)


class LTX2TextConnectorsModel(ComponentModel):
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
        self.config = LTX2TextConnectorsConfig.generate(
            config,
            encoding,
            devices,
        )
        self.load_model()

    def load_model(self) -> Callable[..., Any]:
        state_dict = {
            key: value.data()  # .astype(self.config.dtype)
            for key, value in self.weights.items()
        }
        with F.lazy():
            connectors = LTX2TextConnectors(self.config)
            connectors.to(self.devices[0])
        state_dict = _reconcile_dtypes(state_dict, connectors)
        self.model = connectors.compile(
            *connectors.input_types(), weights=state_dict
        )
        return self.model

    def __call__(
        self,
        text_encoder_hidden_states: Tensor,
        valid_length: Tensor,
        **kwargs: Any,
    ) -> Any:
        return self.model(text_encoder_hidden_states, valid_length)
