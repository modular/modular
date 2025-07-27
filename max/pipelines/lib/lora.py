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
from typing import Any, TypeVar

import numpy as np
from max.driver import Device, DLPackArray, Tensor
from max.dtype import DType
from max.graph import Weight
from max.graph.type import DeviceRef, TensorType
from max.graph.value import TensorValue
from max.graph.weights import WeightData, Weights, WeightsFormat, load_weights
from max.graph.weights.weights import _cast_to_dtype
from max.interfaces import InputContext
from max.nn.layer.layer import Module, recursive_named_layers
from max.nn.lora import SupportsLoRA

from .hf_utils import HuggingFaceRepo

logger = logging.getLogger("max.serve")

ADAPTER_CONFIG_FILE = "adapter_config.json"

T = TypeVar("T", bound=InputContext)


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
    return bool(
        LoRAType.A.value in key
        or LoRAType.B.value in key
        or LoRAType.BIAS.value in key
    )


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
        self._lora_A: dict[str, WeightData] = {}
        self._lora_B: dict[str, WeightData] = {}
        self._lora_bias: dict[str, WeightData] = {}

        self._adapter_config = self._load_weights()

        self.rank: int = self._adapter_config["r"]
        self.target_modules: list[str] = self._adapter_config["target_modules"]

    def get(self, key: str) -> WeightData | None:
        """
        Gets the WeightData from the key. If key doesn't exist in model, then None is returned.

        Args:
            key: Key of LoRA

        Returns:
            WeightData for the key or None if it doesn't exist.
        """
        if key in self._lora_A:
            return self.lora_A[key]
        elif key in self._lora_B:
            return self._lora_B[key]
        elif key in self._lora_bias:
            return self._lora_bias[key]

        return None

    @property
    def lora_A(self) -> dict[str, WeightData]:
        """A dictionary mapping weight keys to LoRA A WeightData."""
        return self._lora_A

    @property
    def lora_B(self) -> dict[str, WeightData]:
        """A dictionary mapping weight keys to LoRA B WeightData."""
        return self._lora_B

    @property
    def lora_bias(self) -> dict[str, WeightData]:
        """A dictionary mapping weight keys to LoRA bias WeightData."""
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

        scale = adapter_config["lora_alpha"] / adapter_config["r"]
        for key, weight in weights.items():
            key = self._normalize_lora_key(key)
            data = weight.data()

            if LoRAType.A.value in key:
                self._lora_A[key] = data
            elif LoRAType.B.value in key:
                # A minor optimization so we don't have to multiply scale
                # by LoRA B in the kernel every forward.
                # The loaded safetensors weights are read-only, so we must copy.
                data.data = (
                    Tensor.from_dlpack(data.data).copy().to_numpy() * scale
                )
                self._lora_B[key] = data
            elif LoRAType.BIAS.value in key:
                self._lora_bias[key] = data
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
        base_model_path: str,
        base_weights: Weights,
        max_num_loras: int,
        max_lora_rank: int,
        lora_paths: list[str] | None = None,
    ):
        """
        Initializes the LoRAManager with a given base weight structure and maximum number of LoRA models.

        Args:
            base_weights (Weights): The base model weights used as the starting point for LoRA modifications.
            max_num_loras (int): The maximum number of LoRA models to manage concurrently.
            lora_paths: (list[str]): An optional list of local LoRAs to load on initialization.
        """
        self.base_model_path = base_model_path
        self.base_weights = base_weights
        self.max_num_loras = max_num_loras
        self.max_lora_rank = max_lora_rank

        self._loras: dict[str, LoRAModel] = dict()
        self._active_loras: dict[str, LoRAModel] = dict()
        self._lora_index_to_id: list[str | None] = [None] * self.max_num_loras

        if lora_paths:
            self.load_adapters(lora_paths)

        self._alias_buffers: dict[str, DLPackArray] = {}

    def _name_to_slot(self, name: str):
        """
        Maps the model name to the assigned slot.
        """
        return self._lora_index_to_id.index(name)

    def _model_name_to_id(self, name: str | None) -> int:
        """
        Maps the model name to it's assigned slot id.
        """
        return (
            self._name_to_slot(name)
            if name in self._loras
            else self.max_num_loras - 1
        )

    def _model_name_to_rank(self, name: str | None) -> int:
        """
        Maps the model name to it's rank.
        """
        return self._loras[name].rank if name in self._loras else 0

    def _model_names_to_ids(self, model_names: list[str | None]) -> list[int]:
        """
        Maps the list of model names to their assigned slots.
        If a model isn't a valid loaded LoRA, we assume the base model is
        selected and set id to max_num_loras.
        """
        return [self._model_name_to_id(name) for name in model_names]

    def _model_names_to_ranks(self, model_names: list[str | None]) -> list[int]:
        """
        Maps the list of model names to their assigned ranks.
        If a model isn't a valid loaded LoRA, we assume the base model is
        selected and set rank to 0.
        """
        return [self._model_name_to_rank(name) for name in model_names]

    def get_lora_graph_inputs(
        self,
        model_names: list[str | None],
        device: Device,
    ) -> tuple[Tensor, ...]:
        """
        Gets the LoRA graph inputs

        Args:
            model_names: List of model names
            input_row_offsets: The offsets for each sequence in the batch
            device: The device
        """
        for name in model_names:
            if name and name not in self._loras:
                raise RuntimeError(
                    "Issuing a request with a non-existent LoRA. "
                    f"Requested LoRA with name: {name}. Valid LoRA names are: "
                    f"{list(self._loras.keys())}"
                )

        ids = self._model_names_to_ids(model_names)
        ranks = self._model_names_to_ranks(model_names)

        lora_ids = Tensor.from_numpy(np.array(ids, dtype=np.uint32)).to(device)
        lora_ranks = Tensor.from_numpy(np.array(ranks, dtype=np.uint32)).to(
            device
        )

        return lora_ids, lora_ranks

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
            if lora_id := self.load_adapter(lora_path):
                lora_ids.append(lora_id)

        return lora_ids

    def _next_free_slot(self) -> int:
        """
        Finds and returns the index of the next available slot for a new LoRA adapter.

        This is an internal utility used to manage a fixed number of LoRA slots.
        The last slot (max_num_loras) is reserved for inactive LoRAs and
        should always contain zeros.

        .. code-block:: python

            slot_index = manager._next_free_slot()

        Returns:
            The integer index of the next available LoRA slot.

        Raises:
            RuntimeError: If no available slots are left.
        """
        # Reserve the last slot (max_num_loras - 1) for inactive LoRAs
        for i, slot in enumerate(self._lora_index_to_id[:-1]):
            if slot is None:
                return i

        raise RuntimeError(
            f"No available LoRA slots left. Current max is: {self.max_num_loras - 1}"
        )

    def load_adapter(self, path: str) -> str | None:
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

        raise RuntimeError(
            f"LoRA with name {name} already exists in LoRA registry."
        )

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

    def _get_lora_weights(self, key: str, base_weight: Weight) -> WeightData:
        """
        Get's the LoRA weights for the specified key for each LoRA that is loaded.
        If the LoRA's don't contain the weight for the key, a zero-weight is returned.

        Args:
            key: Key for LoRA selection.
            base_weight: Weight used to provide WeightData properties.

        Returns:
            A WeightData object with the weights from the loaded LoRAs.
        """
        weight_np = np.zeros(base_weight.shape.static_dims, dtype=np.float32)

        for name, lora in self._loras.items():
            if lora_weight := lora.get(key):
                slot = self._name_to_slot(name)

                if LoRAType.A.value in key:
                    weight_np[slot, : lora.rank, :] = lora_weight.data
                elif LoRAType.B.value in key:
                    weight_np[slot, :, : lora.rank] = lora_weight.data
                elif LoRAType.BIAS.value in key:
                    weight_np[slot, :] = lora_weight.data

        # cast from fp32 -> target dtype
        # if target dtype is bfloat16, this technically returns a float16 np.ndarray
        # we then view the MAX Tensor to get the correct dtype
        weight = _cast_to_dtype(
            Tensor.from_numpy(weight_np), DType.float32, base_weight.dtype
        ).copy(base_weight.device.to_device())

        lora_weights = WeightData(
            weight,
            key,
            base_weight.dtype,
            base_weight.shape,
            base_weight.quantization_encoding,
        )
        return lora_weights

    def _get_lora_leaf_layers(self, model: Module) -> dict[str, Module]:
        """
        Uses recursive_named_layers(model) to return only the leaf module names
        that are instances of SupportsLoRA — skipping containers.

        Args:
            model: The model to scan.

        Returns:
            List of dot-names for leaf LoRA modules.
        """
        lora_layers: list[tuple[str, Module]] = [
            (name, layer)
            for name, layer in recursive_named_layers(model)
            if isinstance(layer, SupportsLoRA)
        ]

        # Make a set of all parent module names (e.g. 'layers.0.self_attn')
        parent_names = set()
        for name, _ in lora_layers:
            parts = name.split(".")
            for i in range(1, len(parts)):
                parent_names.add(".".join(parts[:i]))

        # Only keep layers that are not parents of other LoRA layers
        leaf_lora_layers = {
            name: layer
            for name, layer in lora_layers
            if name not in parent_names
        }

        return leaf_lora_layers

    def init_weights(
        self, model: Module, state_dict: dict[str, WeightData]
    ) -> None:
        """
        Recursively collect all leaf modules in the model that are instances of SupportsLoRA.
        Init's their weights with the loaded LoRAs and adds them to the `state_dict`.

        Acquires the alias-able buffers for dynamic LoRA swapping.

        Must be called to initialize the base model properly.

        Args:
            model: The top-level Module.
            state_dict: Model state_dict to be loaded into model.
            device: The device the base model resides in.
        """
        self._lora_layers = self._get_lora_leaf_layers(model)
        for key, layer in self._lora_layers.items():
            for weight_key, weight in layer.layer_weights.items():
                if not is_lora_kind(weight_key):
                    continue

                state_key = f"{key}.{weight_key}"
                state_dict[state_key] = self._get_lora_weights(
                    state_key, weight
                )

                self._alias_buffers[state_key] = state_dict[state_key].data

    def input_symbols(self, device_ref: DeviceRef) -> list[TensorType]:
        """
        Returns the input symbols needed for the graph inputs

        Args:
            device_ref: Symbolic device to be used for the symbols.

        Returns:
            The graph input symbols.
        """
        lora_ids_type = TensorType(
            DType.uint32, shape=["lora_ids"], device=device_ref
        )
        lora_ranks_type = TensorType(
            DType.uint32, shape=["lora_ranks"], device=device_ref
        )

        return [lora_ids_type, lora_ranks_type]

    def set_graph_info(
        self,
        lora_ids: TensorValue,
        lora_ranks: TensorValue,
    ) -> None:
        """
        Sets the lora batch info required for the forward-pass.

        Args:
            lora_ids: IDs of the LoRAs used in the batch.
            lora_ranks: Ranks of the LoRAs used in the batch.
        """
        for _, layer in self._lora_layers.items():
            if isinstance(layer, SupportsLoRA):
                layer.set_lora_batch_info(lora_ids, lora_ranks)

    def sort_lora_batch(self, batch: dict[str, T]) -> dict[str, T]:
        """ "
        Sorts the LoRA batch by name
        Args:
            batch: The context batch to sort
        """
        batch_by_model_names = {
            req_id: batch[req_id]
            for req_id, _ in sorted(
                batch.items(),
                key=lambda item: self._model_name_to_rank(
                    getattr(item[1], "model_name")  # noqa: B009
                ),
            )
        }
        return batch_by_model_names
