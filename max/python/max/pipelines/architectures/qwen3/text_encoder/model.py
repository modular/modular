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

import numpy as np
from max.driver import Buffer, Device
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph.weights import Weights
from max.pipelines.architectures.llama3.weight_adapters import (
    LLAMA_SAFETENSOR_MAPPING as QWEN_SAFETENSOR_MAP,
)
from max.pipelines.dataprocessing.causal_attention_mask import (
    causal_attention_mask,
)
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel

from .model_config import Qwen3TextEncoderConfig
from .qwen3 import Qwen3TextEncoderTransformer


class Qwen3TextEncoderModel(ComponentModel):
    """Qwen3 text encoder ComponentModel wrapper."""

    default_hidden_state_layers: tuple[int, ...] | None = None

    @staticmethod
    def _normalize_attention_mask_array(
        attention_mask: np.ndarray,
        *,
        mask_name: str = "attention_mask",
    ) -> np.ndarray:
        """Normalize an attention mask to a 1D bool numpy array."""
        attention_mask_np = np.asarray(attention_mask)
        if attention_mask_np.ndim == 2:
            if attention_mask_np.shape[0] != 1:
                raise ValueError(
                    "Qwen3TextEncoderModel expects batch_size=1 for "
                    f"2D {mask_name} input."
                )
            attention_mask_np = attention_mask_np[0]
        elif attention_mask_np.ndim != 1:
            raise ValueError(
                "Qwen3TextEncoderModel expects rank-1 or rank-2 "
                f"{mask_name} input, got shape {attention_mask_np.shape}."
            )

        return attention_mask_np.astype(np.bool_, copy=False)

    @classmethod
    def valid_length_from_attention_mask_array(
        cls,
        attention_mask: np.ndarray,
        *,
        mask_name: str = "attention_mask",
    ) -> int:
        """Derive a valid length from a 1D/2D right-padded attention mask.

        The current Klein path assumes right padding, so the valid token count
        is derived from the attention-mask sum.
        """
        attention_mask_np = cls._normalize_attention_mask_array(
            attention_mask, mask_name=mask_name
        )
        valid_length = int(attention_mask_np.sum())

        if valid_length <= 0:
            raise ValueError(f"{mask_name} must contain at least one valid token.")

        return valid_length

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
                raw_layers = list(
                    range(1, self.config.num_hidden_layers + 1)
                )
        return self._normalize_hidden_state_layers(
            raw_layers,
            self.config.num_hidden_layers,
        )

    @staticmethod
    def _normalize_hidden_state_layers(
        layers: list[int], num_hidden_layers: int
    ) -> list[int]:
        """Normalize HF-style hidden-state indices.

        Contract:
        - 0 = token embeddings
        - i (1 <= i <= num_hidden_layers) = output after transformer block i-1
        - negative indices follow standard Python indexing over the available
          hidden-states tuple of length ``num_hidden_layers + 1``
        """
        normalized: list[int] = []
        seen: set[int] = set()
        total_hidden_states = num_hidden_layers + 1
        for layer in layers:
            idx = int(layer)
            if idx < 0:
                idx += total_hidden_states
            if idx < 0 or idx >= total_hidden_states:
                raise ValueError(
                    "Invalid `hidden_state_layers` index "
                    f"{layer} for available_hidden_states="
                    f"{total_hidden_states} (num_hidden_layers={num_hidden_layers})."
                )
            if idx not in seen:
                normalized.append(idx)
                seen.add(idx)

        if not normalized:
            raise ValueError("`hidden_state_layers` cannot be empty.")

        return normalized

    def _state_dict(self) -> dict[str, Any]:
        state_dict = {}
        for key, value in self.weights.items():
            adapted_key = key
            for before, after in QWEN_SAFETENSOR_MAP.items():
                adapted_key = adapted_key.replace(before, after)
            # The text-encoder module uses local names without language_model prefix.
            adapted_key = adapted_key.removeprefix("language_model.")

            state_dict[adapted_key] = value.data()
        return state_dict

    def load_model(self) -> Callable[..., Any]:
        """Load and compile the Qwen3 text encoder.

        Returns:
            Compiled model callable.
        """
        state_dict = self._state_dict()

        with F.lazy():
            model = Qwen3TextEncoderTransformer(self.config)
            model.to(self.devices[0])

        self.model = model.compile(*model.input_types(), weights=state_dict)
        return self.model

    @staticmethod
    def attention_bias_from_attention_mask_array(
        attention_mask: np.ndarray,
        *,
        mask_name: str = "attention_mask",
    ) -> np.ndarray:
        attention_mask_np = Qwen3TextEncoderModel._normalize_attention_mask_array(
            attention_mask, mask_name=mask_name
        )
        seq_len = int(attention_mask_np.shape[0])
        additive_mask = causal_attention_mask([0], [seq_len])[0]
        additive_mask[:, ~attention_mask_np] = -10000.0
        return additive_mask[np.newaxis, np.newaxis, :, :].astype(np.float32)

    @classmethod
    def attention_bias_from_valid_length(
        cls,
        *,
        seq_len: int,
        valid_length: int,
    ) -> np.ndarray:
        if valid_length <= 0 or valid_length > seq_len:
            raise ValueError(
                "valid_length must be within sequence length. "
                f"Got valid_length={valid_length}, seq_len={seq_len}."
            )
        attention_mask = np.zeros((seq_len,), dtype=np.bool_)
        attention_mask[:valid_length] = True
        return cls.attention_bias_from_attention_mask_array(attention_mask)

    def __call__(
        self,
        tokens: Tensor,
        attention_mask: Tensor | None = None,
        *,
        valid_length: Tensor | None = None,
        hidden_state_index: int | None = None,
    ):
        if tokens.rank == 2:
            if int(tokens.shape[0]) != 1:
                raise ValueError(
                    "Qwen3TextEncoderModel expects batch_size=1 for 2D token input."
                )
            tokens = tokens[0]

        if attention_mask is not None and valid_length is not None:
            raise ValueError(
                "Pass either `attention_mask` or `valid_length`, not both."
            )

        if attention_mask is not None:
            if attention_mask.storage is not None:
                attention_mask_np = attention_mask.storage.to_numpy()
            else:
                attention_mask_np = np.from_dlpack(attention_mask)
            attention_bias_np = self.attention_bias_from_attention_mask_array(
                attention_mask_np
            )
        else:
            if valid_length is None:
                valid_length_value = int(tokens.shape[0])
            elif valid_length.storage is not None:
                valid_length_value = int(valid_length.storage.to_numpy()[0])
            else:
                valid_length_value = int(np.from_dlpack(valid_length)[0])
            attention_bias_np = self.attention_bias_from_valid_length(
                seq_len=int(tokens.shape[0]),
                valid_length=valid_length_value,
            )

        attention_bias = Tensor(
            storage=Buffer.from_numpy(attention_bias_np).to(self.devices[0])
        )

        outputs = self.model(tokens, attention_bias)
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
