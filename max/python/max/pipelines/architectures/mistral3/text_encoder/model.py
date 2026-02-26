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

"""Mistral3 text encoder ComponentModel wrapper.

This module provides a ComponentModel wrapper for Mistral3 text encoder.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from max.driver import Device
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel

from ..weight_adapters import MISTRAL_SAFETENSOR_MAP
from .mistral3 import Mistral3TextEncoderTransformer
from .model_config import Mistral3TextEncoderConfig


class Mistral3TextEncoderModel(ComponentModel):
    """Mistral3 text encoder ComponentModel wrapper."""

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
    ) -> None:
        """Initialize Mistral3TextEncoderModel.

        Args:
            config: Configuration dictionary from model config file.
            encoding: Supported encoding for the model.
            devices: List of devices to use.
            weights: Model weights.
        """
        super().__init__(config, encoding, devices, weights)
        self.config = Mistral3TextEncoderConfig.generate(
            config,
            encoding,
            devices,
        )
        self.load_model()

    def load_model(self) -> Callable[..., Any]:
        """Load and compile the Mistral3 text encoder.

        Returns:
            Compiled model callable.
        """
        state_dict = {}
        for key, value in self.weights.items():
            adapted_key = key
            for before, after in MISTRAL_SAFETENSOR_MAP.items():
                adapted_key = adapted_key.replace(before, after)

            state_dict[adapted_key] = value.data()

        with F.lazy():
            model = Mistral3TextEncoderTransformer(self.config)
            model.to(self.devices[0])

        self.model = model.compile(*model.input_types(), weights=state_dict)
        return self.model

    def __call__(self, tokens: Tensor) -> tuple[Tensor, ...]:
        outputs = self.model(tokens)
        if isinstance(outputs, list):
            return tuple(outputs)
        return outputs
