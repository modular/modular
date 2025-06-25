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
"""LoRA-specific classes."""

from __future__ import annotations

import json
import logging
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from max.driver import Tensor
from max.graph.weights import Weights, WeightsFormat, load_weights

from .hf_utils import HuggingFaceRepo

logger = logging.getLogger("max.serve")

ADAPTER_CONFIG_FILE = "adapter_config.json"


class LoRAType(Enum):
    """
    Enumeration for LoRA Types.
    """

    A = "lora_A"
    """Represents the LoRA A matrix (high rank tensor to low rank tensor)."""

    B = "lora_B"
    """Represents the LoRA B matrix. (low rank tensor to high rank tensor)"""

    BIAS = "lora.bias"
    """Represents the LoRA bias matrix. (added to matrix B)"""


def is_lora_kind(key: str) -> bool:
    """
    Whether the key is a lora kind
    """
    if (
        LoRAType.A.value in key
        or LoRAType.B.value in key
        or LoRAType.BIAS.value in key
    ):
        return True
    return False


class LoRAModel:
    """
    Manages LoRA weights and configuration for a single adapter.
    """

    def __init__(self, name: str, path: str, strict: bool = True) -> None:
        """
        Initializes a LoRAModel by loading its configuration and weights.

        .. code-block:: python

            lora = LoRAModel("my_adapter", "/path/to/lora")

        Args:
            name:
                A string identifier for this adapter.
            path:
                Filesystem path is only supported
            strict:
                Whether to enforce strict validation while loading the adapter.

        Raises:
            ValueError: If weight files are not in the supported `safetensors` format,
                or if the keys in the weights are malformed or incomplete.
        """
        self.name = name
        self.path = path
        self.strict = strict
        self._lora_A: dict[str, Tensor] = {}
        self._lora_B: dict[str, Tensor] = {}
        self._lora_bias: dict[str, Tensor] = {}

        self._adapter_config = self._load_weights()

        self.rank = self._adapter_config["r"]
        self.target_modules = self._adapter_config["target_modules"]

    @property
    def lora_A(self) -> dict[str, Tensor]:
        """A dictionary mapping weight keys to LoRA A tensors."""
        return self._lora_A

    @property
    def lora_B(self) -> dict[str, Tensor]:
        """A dictionary mapping weight keys to LoRA B tensors."""
        return self._lora_B

    @property
    def lora_bias(self) -> dict[str, Tensor]:
        """A dictionary mapping weight keys to LoRA bias tensors."""
        return self._lora_bias

    @property
    def adapter_config(self) -> dict:
        """A dictionary containing metadata/configuration for the LoRA adapter."""
        return self._adapter_config

    def _normalize_lora_key(self, key: str) -> str:
        """
        Normalizes LoRA weight keys by extracting the portion starting from `layers.<number>`.

        This ensures that weight keys conform to the expected format used in target models.

        .. code-block:: python

            normalized = lora._normalize_lora_key("model.layers.4.self_attn.q_proj.weight")

        Args:
            key:
                The original key from the weight file.

        Returns:
            A normalized key string suitable for indexing into model layers.
        """
        match = re.search(r"(layers\.\d+\..+)", key)
        if match:
            return match.group(1)
        else:
            return key

    def _load_weights(self) -> dict[str, Any]:
        """
        Loads LoRA adapter weights and configuration from disk.

        This method parses the safetensors weight files and categorizes them
        into A, B, and bias matrices based on their keys. It also reads the
        adapter configuration JSON file.

        .. code-block:: python

            adapter_config = lora._load_weights()

        Returns:
            A dictionary containing the parsed adapter configuration.

        Raises:
            ValueError: If the weight format is not safetensors, or if keys
                are not recognized as valid LoRA components.
        """
        hf_repo = HuggingFaceRepo(repo_id=self.path)

        weight_files = hf_repo.weight_files
        config_path = os.path.join(self.path, ADAPTER_CONFIG_FILE)
        with open(config_path) as f:
            adapter_config = json.load(f)

        if WeightsFormat.safetensors in weight_files:
            weights = load_weights(
                [
                    self.path / Path(p)
                    for p in weight_files[WeightsFormat.safetensors]
                ]
            )
        else:
            # TODO (E2EOPT-279)
            raise ValueError("LoRA only supports files in safetensors format.")

        for key, weight in weights.items():
            key = self._normalize_lora_key(key)
            tensor = Tensor.from_numpy(weight.raw_tensor())
            if LoRAType.A.value in key:
                self._lora_A[key] = tensor
            elif LoRAType.B.value in key:
                self._lora_B[key] = tensor
            elif LoRAType.BIAS.value in key:
                self._lora_bias[key] = tensor
            else:
                raise ValueError(f"Invalid LoRA type got key: {key}")

        return adapter_config


class LoRAManager:
    """
    Manages multiple LoRA models applied to a set of base weights and the
    underlying buffers required for the forward pass.
    """

    def __init__(
        self,
        base_weights: Weights,
        max_num_loras: int,
        lora_paths: Optional[list[str]] = None,
    ) -> None:
        """
        Initializes the LoRAManager with a given base weight structure and maximum number of LoRA models.

        Args:
            base_weights (Weights): The base model weights used as the starting point for LoRA modifications.
            max_num_loras (int): The maximum number of LoRA models to manage concurrently.
            lora_paths: (list[str]): An optional list of local LoRAs to load on initialization.
        """
        self.base_weights = base_weights
        self.max_num_loras = max_num_loras
        self._loras: dict[str, LoRAModel] = dict()
        self._active_loras: dict[str, LoRAModel] = dict()
        self._lora_index_to_id: list[Optional[str]] = [
            None
        ] * self.max_num_loras

        self._A_buffers: dict[str, Tensor] = {}
        self._B_buffers: dict[str, Tensor] = {}
        self._bias_buffers: dict[str, Tensor] = {}

        if lora_paths:
            self.load_adapters(lora_paths)

    def load_adapters(self, lora_paths: list[str]) -> list[str]:
        """
        Loads LoRA adapters from the specified file paths and registers them for use.

        This method is useful when you want to load multiple LoRA adapters in one call.

        .. code-block:: python

            lora_ids = manager.load_adapters(["adapter1=/path/to/lora1", "/path/to/lora2"])

        Args:
            lora_paths:
                A list of file paths (optionally with name prefixes like `name=path`) to LoRA adapter directories.

        Returns:
            A list of strings representing the registered names of the successfully loaded LoRA adapters.

        Raises:
            RuntimeError: If there are no available LoRA slots remaining.
        """
        lora_ids: list[str] = []

        for lora_path in lora_paths:
            lora_id = self.load_adapter(lora_path)
            lora_ids.append(lora_id)

        return lora_ids

    def _next_free_slot(self) -> int:
        """
        Finds and returns the index of the next available slot for a new LoRA adapter.

        This is an internal utility used to manage a fixed number of LoRA slots.

        .. code-block:: python

            slot_index = manager._next_free_slot()

        Returns:
            The integer index of the next available LoRA slot.

        Raises:
            RuntimeError: If no available slots are left.
        """
        for i, slot in enumerate(self._lora_index_to_id):
            if slot is None:
                return i

        raise RuntimeError(
            f"No available LoRA slots left. Current max is: {self.max_num_loras}"
        )

    def load_adapter(self, path: str) -> str:
        """
        Loads a single LoRA adapter from the given path and registers it under a unique name.

        The path can include an explicit name using the format `name=path`. If no name is provided,
        the path itself is used as the name.

        .. code-block:: python

            lora_id = manager.load_adapter("my_adapter=/path/to/lora")
            lora_id = manager.load_adapter("/path/to/another_lora")

        Args:
            path:
                A string in the form `name=path` or just a file path. The adapter is expected to reside at that path.

        Returns:
            A string representing the name under which the LoRA adapter was registered.

        Raises:
            RuntimeError: If there are no available slots remaining.
        """
        # vLLM allows passing in args to cli like: {name}={path} {name}={path}
        if "=" in path:
            name, path = path.split("=", 1)
        else:
            name = path
            path = path

        if name not in self._loras:
            slot = self._next_free_slot()
            lora = LoRAModel(name, path)
            self._lora_index_to_id[slot] = lora.name
            self._loras[lora.name] = lora
            return lora.name
        else:
            logger.warning(
                f"LoRA with name {name} already exists in LoRA registry, not reloading."
            )

            return name

    def unload_adapter(self, lora: str) -> None:
        """
        Unloads the specified LoRA adapter from the internal registry and frees its slot.

        This function is used to release GPU or CPU memory by removing a LoRA model.

        .. code-block:: python

            manager.unload_adapter("my_adapter")

        Args:
            lora:
                The name of the LoRA adapter to unload.

        Returns:
            None

        Raises:
            KeyError: If the specified LoRA adapter is not found in the registry.
        """
        pass

    def activate_adapter(self, lora: str) -> None:
        """
        Moves the specified LoRA adapter to GPU and marks it as active.

        Useful for enabling a specific adapter for use in model inference or training.

        .. code-block:: python

            manager.activate_adapter("my_adapter")

        Args:
            lora:
                The name of the LoRA adapter to activate.

        Returns:
            None

        Raises:
            KeyError: If the specified adapter does not exist in the registry.
        """
        pass

    def set_active_loras(self, loras: list[str]) -> None:
        """
        Set the active LoRAs for the next forward pass.
        """
        self._active_loras.clear()
        for lora in loras:
            model = self._loras.get(lora, None)

            if model is None:
                raise RuntimeError(
                    f"LoRA name should be valid when setting active LoRAs. Attempted to access LoRA: {lora} but doesn't exist."
                )

            self._active_loras[lora] = model
