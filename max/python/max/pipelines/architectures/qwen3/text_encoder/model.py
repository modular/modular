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

"""Qwen3 text encoder ComponentModel wrapper.

This module provides a ComponentModel wrapper for Qwen3 text encoder.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from max.driver import Device
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph.weights import Weights
from max.pipelines.architectures.llama3.weight_adapters import (
    LLAMA_SAFETENSOR_MAPPING as QWEN_SAFETENSOR_MAP,
)
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel

from .model_config import Qwen3TextEncoderConfig
from .qwen3 import Qwen3TextEncoderTransformer


class Qwen3TextEncoderModel(ComponentModel):
    """Qwen3 text encoder ComponentModel wrapper."""

    default_hidden_state_layers: tuple[int, ...] | None = None

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
    ) -> None:
        """Initialize Qwen3TextEncoderModel.

        Args:
            config: Configuration dictionary from model config file.
            encoding: Supported encoding for the model.
            devices: List of devices to use.
            weights: Model weights.
        """
        super().__init__(config, encoding, devices, weights)
        self.config = Qwen3TextEncoderConfig.initialize_from_config(
            config,
            encoding,
            devices,
        )
        self.config.hidden_state_layers = self._resolve_hidden_state_layers()
        self.load_model()

    def _resolve_hidden_state_layers(self) -> list[int]:
        raw_layers = list(self.config.hidden_state_layers)
        if not raw_layers:
            if self.default_hidden_state_layers is not None:
                raw_layers = list(self.default_hidden_state_layers)
            else:
                raw_layers = list(range(self.config.num_hidden_layers))
        return self._normalize_hidden_state_layers(
            raw_layers,
            self.config.num_hidden_layers,
        )

    @staticmethod
    def _normalize_hidden_state_layers(
        layers: list[int], num_hidden_layers: int
    ) -> list[int]:
        normalized: list[int] = []
        seen: set[int] = set()
        for layer in layers:
            idx = int(layer)
            if idx < 0:
                idx += num_hidden_layers
            if idx < 0 or idx >= num_hidden_layers:
                raise ValueError(
                    "Invalid `hidden_state_layers` index "
                    f"{layer} for num_hidden_layers={num_hidden_layers}."
                )
            if idx not in seen:
                normalized.append(idx)
                seen.add(idx)

        if not normalized:
            raise ValueError("`hidden_state_layers` cannot be empty.")

        return normalized

    def load_model(self) -> Callable[..., Any]:
        """Load and compile the Qwen3 text encoder.

        Returns:
            Compiled model callable.
        """
        state_dict = {}
        for key, value in self.weights.items():
            adapted_key = key
            for before, after in QWEN_SAFETENSOR_MAP.items():
                adapted_key = adapted_key.replace(before, after)
            # The text-encoder module uses local names without language_model prefix.
            adapted_key = adapted_key.removeprefix("language_model.")

            state_dict[adapted_key] = value.data()

        with F.lazy():
            model = Qwen3TextEncoderTransformer(self.config)
            model.to(self.devices[0])

        self.model = model.compile(*model.input_types(), weights=state_dict)
        return self.model

    def __call__(
        self,
        tokens: Tensor,
        attention_mask: Tensor | None = None,
        *,
        hidden_state_index: int | None = None,
    ):
        if tokens.rank == 2:
            if int(tokens.shape[0]) != 1:
                raise ValueError(
                    "Qwen3TextEncoderModel expects batch_size=1 for 2D token input."
                )
            tokens = tokens[0]

        if attention_mask is not None:
            raise ValueError(
                "Qwen3TextEncoderModel does not support `attention_mask` in "
                "the execution path. Compact tokens with the mask before calling "
                "the encoder."
            )

        outputs = self.model(tokens)
        if isinstance(outputs, list):
            outputs = tuple(outputs)

        if hidden_state_index is None:
            if isinstance(outputs, tuple) and len(outputs) == 1:
                return outputs[0]
            return outputs

        if not isinstance(outputs, tuple):
            raise ValueError(
                "`hidden_state_index` requires model outputs to be tuple/list "
                f"of hidden states, got {type(outputs).__name__}."
            )

        num_layers = len(outputs)
        if hidden_state_index < -num_layers or hidden_state_index >= num_layers:
            raise ValueError(
                f"`hidden_state_index` out of range: {hidden_state_index}. "
                f"Valid range is [{-num_layers}, {num_layers - 1}]."
            )

        return outputs[hidden_state_index]


class Qwen3TextEncoderKleinModel(Qwen3TextEncoderModel):
    """Qwen3 text encoder tuned for Flux2 Klein prompt layers."""

    default_hidden_state_layers = (9, 18, 27)


class Qwen3TextEncoderZImageModel(Qwen3TextEncoderModel):
    """Qwen3 text encoder tuned for Z-Image prompt layers."""

    default_hidden_state_layers = (-2,)
