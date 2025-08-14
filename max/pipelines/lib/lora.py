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
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from max.driver import Device, DLPackArray, Tensor
from max.dtype import DType
from max.graph import Weight
from max.graph.type import DeviceRef, TensorType
from max.graph.value import TensorValue
from max.graph.weights import WeightData, WeightsFormat, load_weights
from max.graph.weights.weights import _cast_to_dtype
from max.interfaces import InputContext, LoRAStatus, LoRAType
from max.nn.layer.layer import Module, recursive_named_layers
from max.nn.lora import SupportsLoRA
from max.pipelines.lib.config import LoRAConfig

from .hf_utils import HuggingFaceRepo
from .lora_request_processor import LoRARequestProcessor

logger = logging.getLogger("max.serve")

ADAPTER_CONFIG_FILE = "adapter_config.json"

T = TypeVar("T", bound=InputContext)


class LoRALRUCache:
    """
    LRU cache for managing active LoRA models and their slot assignments.

    This cache maintains a maximum number of active LoRA models and evicts
    the least recently used model when the cache is full. It also manages
    slot assignments for GPU buffer placement.
    """

    def __init__(self, max_size: int):
        """
        Initialize the LRU cache.

        Args:
            max_size: Maximum number of LoRA models to keep in the cache.
        """
        self._cache: OrderedDict[str, tuple[LoRAModel, int]] = OrderedDict()
        self._max_size = max_size
        self._free_slots: set[int] = set(range(max_size))
        self._name_to_slot: dict[str, int] = {}

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        return key in self._cache

    def __len__(self) -> int:
        """Return the number of items in the cache."""
        return len(self._cache)

    def get(self, key: str) -> tuple[LoRAModel | None, int | None]:
        """
        Get a LoRA model and its slot from the cache and mark it as recently used.

        Args:
            key: The name of the LoRA model.

        Returns:
            A tuple of (LoRA model, slot) if found, None otherwise.
        """
        if key not in self._cache:
            return None, None

        self._cache.move_to_end(key)
        return self._cache[key]

    def get_slot(self, key: str) -> int | None:
        """
        Get the slot assignment for a LoRA model.

        Args:
            key: The name of the LoRA model.

        Returns:
            The slot number if the model is active, None otherwise.
        """
        return self._name_to_slot.get(key)

    def next_slot(self) -> int | None:
        """
        Get the next available slot for a new LoRA.

        Returns:
            The next available slot number, or None if no slots are available.
        """
        if not self._free_slots:
            return None
        return min(self._free_slots)

    def put(
        self, key: str, value: LoRAModel, slot: int | None = None
    ) -> tuple[str | None, int | None]:
        """
        Add or update a LoRA model in the cache with slot assignment.

        Args:
            key: The name of the LoRA model.
            value: The LoRA model to cache.
            slot: Optional slot assignment. If None, assigns next available slot.

        Returns:
            A tuple of (evicted_key, freed_slot) if eviction occurred, (None, None) otherwise.
        """
        evicted_key = None
        freed_slot = None

        if key in self._cache:
            self._cache.move_to_end(key)
            return (None, None)

        # Need to add new entry
        if slot is None:
            slot = self.next_slot()
            if slot is None:
                # No free slots, need to evict
                if len(self._cache) >= self._max_size:
                    # Evict least recently used (first item)
                    evicted_key, (_, freed_slot) = self._cache.popitem(
                        last=False
                    )
                    del self._name_to_slot[evicted_key]
                    self._free_slots.add(freed_slot)
                    slot = freed_slot

        if slot is not None:
            self._cache[key] = (value, slot)
            self._name_to_slot[key] = slot
            self._free_slots.discard(slot)

        return (evicted_key, freed_slot)

    def remove(self, key: str) -> tuple[bool, int | None]:
        """
        Remove a LoRA model from the cache.

        Args:
            key: The name of the LoRA model to remove.

        Returns:
            A tuple of (success, freed_slot).
        """
        if key in self._cache:
            _, slot = self._cache[key]
            del self._cache[key]
            del self._name_to_slot[key]
            self._free_slots.add(slot)
            return (True, slot)
        return (False, None)

    def clear(self) -> None:
        """Clear all entries from the cache."""
        self._cache.clear()
        self._name_to_slot.clear()
        self._free_slots = set(range(self._max_size))

    def keys(self) -> list[str]:
        """Return all keys in the cache, ordered from least to most recently used."""
        return list(self._cache.keys())

    def values(self) -> list[tuple[LoRAModel, int]]:
        """Return all values in the cache, ordered from least to most recently used."""
        return list(self._cache.values())

    def items(self) -> list[tuple[str, tuple[LoRAModel, int]]]:
        """Return all key-value pairs with slots in the cache, ordered from least to most recently used."""
        return list(self._cache.items())


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

    def __init__(
        self,
        name: str,
        path: str,
        base_dtype: DType,
        max_lora_rank: int,
        strict: bool = True,
    ) -> None:
        """
        Initializes a LoRAModel by loading its configuration and weights.

        .. code-block:: python

            lora = LoRAModel("my_adapter", "/path/to/lora", base_dtype, max_lora_rank)

        Args:
            name:
                A string identifier for this adapter.
            path:
                Filesystem path is only supported
            base_dtype:
                The base model dtype.
            max_lora_rank:
                The maximum LoRA rank supported by the system.
            strict:
                Whether to enforce strict validation while loading the adapter.

        Raises:
            ValueError: If weight files are not in the supported `safetensors` format,
                or if the keys in the weights are malformed or incomplete.
        """
        self.name = name
        self.path = path
        self.strict = strict
        self.max_lora_rank = max_lora_rank
        self._lora_A: dict[str, WeightData] = {}
        self._lora_B: dict[str, WeightData] = {}
        self._lora_bias: dict[str, WeightData] = {}

        self._adapter_config = self._load_weights(base_dtype)

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

    def _load_weights(self, base_dtype: DType) -> dict[str, Any]:
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
        if not os.path.exists(config_path):
            raise ValueError(f"Adapter config file not found: {config_path}")

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

            # TODO: Need to pad the tensors since max.driver.Tensors don't allow for
            #  non-contiguous copies. Move padding to max, if possible.
            if LoRAType.A.value in key:
                # Convert to numpy array
                weight_np = Tensor.from_dlpack(data.data).to_numpy()

                # Pad LoRA A weights from [rank, in_features] to [max_rank, in_features]
                if adapter_config["r"] < self.max_lora_rank:
                    padded = np.zeros(
                        (self.max_lora_rank, weight_np.shape[-1]),
                        dtype=weight_np.dtype,
                    )
                    padded[: adapter_config["r"], :] = weight_np
                    weight_np = padded

                # Cast to base dtype after padding
                data.data = _cast_to_dtype(weight_np, data.dtype, base_dtype)
                self._lora_A[key] = data
            elif LoRAType.B.value in key:
                # A minor optimization so we don't have to multiply scale
                # by LoRA B in the kernel every forward.
                # The loaded safetensors weights are read-only, so we must copy.
                weight_np = (
                    Tensor.from_dlpack(data.data).copy().to_numpy() * scale
                )

                # Pad LoRA B weights from [out_features, rank] to [out_features, max_rank]
                if adapter_config["r"] < self.max_lora_rank:
                    padded = np.zeros(
                        (weight_np.shape[0], self.max_lora_rank),
                        dtype=weight_np.dtype,
                    )
                    padded[:, : adapter_config["r"]] = weight_np
                    weight_np = padded

                # Cast to base dtype after padding
                data.data = _cast_to_dtype(weight_np, data.dtype, base_dtype)
                self._lora_B[key] = data
            elif LoRAType.BIAS.value in key:
                data.data = _cast_to_dtype(data.data, data.dtype, base_dtype)
                self._lora_bias[key] = data
            else:
                raise ValueError(f"Invalid LoRA type got key: {key}")

        return adapter_config


class LoRAManager:
    """
    Manages multiple LoRA models applied to a set of base weights and the
    underlying buffers required for the forward pass.
    """

    # -1 is used to indicate that there is no active LoRA for a given request
    # downstream kernels use this to exit early.
    _NO_ACTIVE_LORA = -1

    def __init__(
        self,
        config: LoRAConfig,
        base_model_path: str,
        base_dtype: DType,
    ):
        """
        Initializes the LoRAManager with a given base weight structure and maximum number of LoRA models.

        Args:
            base_model_path (str): The name/path of the base model.
            base_dtype (DType): The base model dtype.
            max_num_loras (int): The maximum number of LoRA models to manage concurrently.
            max_lora_rank (int): The maximum rank of all LoRAs loadable on the server.
            lora_paths: (list[str]): An optional list of local LoRAs to load on initialization.
        """
        self.base_model_path = base_model_path
        self.base_dtype = base_dtype
        self.max_num_loras = config.max_num_loras
        self.max_lora_rank = config.max_lora_rank

        self._loras: dict[str, LoRAModel] = dict()
        self._active_loras: LoRALRUCache = LoRALRUCache(
            max_size=self.max_num_loras
        )

        self._lora_lock = threading.RLock()
        self._request_processor: LoRARequestProcessor = LoRARequestProcessor(
            self, config.lora_request_endpoint, config.lora_response_endpoint
        )

        if config.lora_paths:
            self._load_adapters(config.lora_paths)

        self._alias_buffers: dict[str, DLPackArray] = {}

    @property
    def loras(self) -> list[str]:
        with self._lora_lock:
            return list(self._loras.keys())

    def _model_name_to_id(self, name: str | None) -> int:
        """
        Maps the model name to its assigned slot id.

        Active LoRAs get their assigned slot (0 to max_num_loras-2).
        Base model or non-existent LoRAs get the last slot (max_num_loras-1).
        """
        if name and name in self._loras:
            slot = self._active_loras.get_slot(name)
            if slot is not None:
                return slot
        return self._NO_ACTIVE_LORA

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
            if (
                name
                and name != self.base_model_path
                and name not in self._loras
            ):
                raise RuntimeError(
                    "Issuing a request with a non-existent LoRA. "
                    f"Requested LoRA with name: {name}. Valid LoRA names are: "
                    f"{list(self._loras.keys())}"
                )

        ids = self._model_names_to_ids(model_names)
        ranks = self._model_names_to_ranks(model_names)

        lora_ids = Tensor.from_numpy(np.array(ids, dtype=np.int32)).to(device)
        lora_ranks = Tensor.from_numpy(np.array(ranks, dtype=np.uint32)).to(
            device
        )

        return lora_ids, lora_ranks

    def _validate_lora_path(self, path: str) -> LoRAStatus:
        """
        Validates that a LoRA adapter path exists locally.

        Remote HuggingFace repositories are not supported and must be downloaded
        to a local directory first.

        Args:
            path: The path to validate.

        """
        if not os.path.exists(path):
            return LoRAStatus.LOAD_INVALID_PATH

        return LoRAStatus.SUCCESS

    def _load_adapters(self, lora_paths: list[str]) -> None:
        """
        Internal method to load LoRA adapters during initialization.

        This method raises exceptions on any errors to fail during startup.

        Args:
            lora_paths: List of LoRA adapter paths to load.

        Raises:
            RuntimeError: If any adapter fails to load.
        """
        for lora_path in lora_paths:
            status = self.load_adapter(lora_path)
            if status != LoRAStatus.SUCCESS:
                error_messages = {
                    LoRAStatus.LOAD_NAME_EXISTS: f"LoRA adapter name already exists with different path: {lora_path}",
                    LoRAStatus.LOAD_INVALID_PATH: f"Invalid LoRA adapter path: {lora_path}",
                    LoRAStatus.LOAD_INVALID_ADAPTER: f"Invalid LoRA adapter format: {lora_path}",
                    LoRAStatus.LOAD_ERROR: f"Unexpected error loading LoRA adapter: {lora_path}",
                }
                raise RuntimeError(error_messages.get(status))

    def load_adapter(self, path: str) -> LoRAStatus:
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
            LoRAStatus indicating the result of the load operation.
        """
        with self._lora_lock:
            try:
                if "=" in path:
                    name, path = path.split("=", 1)
                else:
                    name = path
                    path = path

                # Check if name already exists first
                if name in self._loras:
                    existing_lora = self._loras[name]
                    if existing_lora.path == path:
                        return LoRAStatus.SUCCESS
                    else:
                        return LoRAStatus.LOAD_NAME_EXISTS

                if (
                    status := self._validate_lora_path(path)
                ) != LoRAStatus.SUCCESS:
                    return status

                try:
                    lora = LoRAModel(
                        name, path, self.base_dtype, self.max_lora_rank
                    )
                except ValueError as e:
                    return LoRAStatus.LOAD_INVALID_ADAPTER

                self._loras[lora.name] = lora
                return LoRAStatus.SUCCESS

            except Exception as e:
                logger.exception(
                    f"Unexpected error loading LoRA adapter from '{path}': {e}"
                )
                return LoRAStatus.LOAD_ERROR

    def unload_adapter(self, name: str) -> LoRAStatus:
        """
        Unloads the specified LoRA adapter from the internal registry and frees its slot.

        This function is used to release GPU or CPU memory by removing a LoRA model.

        .. code-block:: python

            manager.unload_adapter("my_adapter")

        Args:
            lora:
                The name of the LoRA adapter to unload.

        Returns:
            LoRAStatus indicating the result of the unload operation.
        """
        with self._lora_lock:
            try:
                if name not in self._loras:
                    return LoRAStatus.UNLOAD_NAME_NONEXISTENT

                # Remove from registries
                del self._loras[name]
                # Remove from active cache (if present)
                self._active_loras.remove(name)

                return LoRAStatus.SUCCESS
            except Exception as e:
                logger.exception(f"Error unloading LoRA adapter '{name}': {e}")
                return LoRAStatus.UNLOAD_ERROR

    def activate_adapter(self, name: str) -> None:
        """
        Moves the specified LoRA adapter to GPU and marks it as active.

        Useful for enabling a specific adapter for use in model inference.

        .. code-block:: python

            manager.activate_adapter("my_adapter")

        Args:
            name:
                The name of the LoRA adapter to activate.

        Returns:
            None

        Raises:
            KeyError: If the specified adapter does not exist in the registry.
        """
        with self._lora_lock:
            if name not in self._loras:
                raise KeyError(f"LoRA adapter '{name}' not found in registry")

            # Check if already active before putting
            is_active = name in self._active_loras
            # if it is active already, we still need to update the lru cache
            self._active_loras.put(name, self._loras[name])

            # Only update buffers if the LoRA wasn't already active
            if not is_active:
                # Get the current LoRA and its slot
                (lora, slot) = self._active_loras.get(name)

                if lora is None or slot is None:
                    raise RuntimeError(
                        "LoRA or slot is None even after it has been added to cache..."
                        " This shouldn't happen."
                    )

                # Update alias buffers with the newly activated LoRA
                self._update_alias_buffers_for_lora(lora, slot)

    def _update_alias_buffers_for_lora(
        self, lora: LoRAModel, slot: int
    ) -> None:
        """
        Updates the alias buffers with weights from a newly activated LoRA.

        This function copies the LoRA weights (A, B, and bias) into the appropriate
        slot in the alias buffers, which are used for dynamic LoRA swapping during
        inference.

        Args:
            lora: The LoRAModel instance containing the weights.
            slot: The slot index where the LoRA weights should be placed.
        """
        for state_key in self._alias_buffers:
            buffer = Tensor.from_dlpack(self._alias_buffers[state_key])

            if lora_weight := lora.get(state_key):
                weight = Tensor.from_dlpack(lora_weight.data)

                if (
                    LoRAType.A.value in state_key
                    or LoRAType.B.value in state_key
                ):
                    buffer[slot, :, :].inplace_copy_from(weight)
                elif LoRAType.BIAS.value in state_key:
                    buffer[slot, :].inplace_copy_from(weight)
            else:
                # If this LoRA doesn't have this weight, zero out the slot
                if LoRAType.A.value in state_key:
                    zeros = Tensor.zeros(
                        (self.max_lora_rank, buffer.shape[-1]),
                        dtype=buffer.dtype,
                        device=buffer.device,
                    )
                    buffer[slot, :, :].inplace_copy_from(zeros)
                elif LoRAType.B.value in state_key:
                    zeros = Tensor.zeros(
                        (buffer.shape[1], self.max_lora_rank),
                        dtype=buffer.dtype,
                        device=buffer.device,
                    )
                    buffer[slot, :, :].inplace_copy_from(zeros)
                elif LoRAType.BIAS.value in state_key:
                    zeros = Tensor.zeros(
                        buffer.shape[1:],
                        dtype=buffer.dtype,
                        device=buffer.device,
                    )
                    buffer[slot, :].inplace_copy_from(zeros)

    def _get_lora_weights(self, key: str, base_weight: Weight) -> WeightData:
        """
        Gets the LoRA weights for the specified key for each LoRA that is loaded.
        If the LoRAs don't contain the weight for the key, a zero-weight is returned.

        Args:
            key: Key for LoRA selection.
            base_weight: Weight used to provide WeightData properties.

        Returns:
            A WeightData object with the weights from the loaded LoRAs.
        """
        weight_np = np.zeros(base_weight.shape.static_dims, dtype=np.float32)

        for lora, slot in self._active_loras.values():
            if lora_weight := lora.get(key):
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
            DType.int32, shape=["lora_ids"], device=device_ref
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
        """
        Sorts the LoRA batch by name and activates any new LoRAs in the batch.

        This method ensures that all LoRAs referenced in the batch are moved to
        the active cache, implementing the LRU policy by evicting least recently
        used LoRAs if necessary.

        Args:
            batch: The context batch to sort
        """
        # First, activate any LoRAs referenced in the batch that aren't already active
        with self._lora_lock:
            for context in batch.values():
                model_name = getattr(context, "model_name", None)
                if (
                    model_name
                    and model_name != self.base_model_path
                    and model_name in self._loras
                ):
                    self.activate_adapter(model_name)

            # Then sort the batch by slot ID (active LoRAs by their slot, inactive ones to the start)
            batch_by_model_names = {
                req_id: batch[req_id]
                for req_id, _ in sorted(
                    batch.items(),
                    key=lambda item: self._model_name_to_id(
                        getattr(item[1], "model_name", None)
                    ),
                )
            }
            return batch_by_model_names
