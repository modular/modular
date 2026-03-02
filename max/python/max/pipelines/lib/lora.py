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

"""LoRA-specific classes."""

from __future__ import annotations

import json
import logging
import os
import re
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from max.driver import (
    CPU,
    Buffer,
    Device,
    DLPackArray,
    is_virtual_device_mode,
)
from max.dtype import DType
from max.graph.buffer_utils import cast_dlpack_to, cast_tensor_to
from max.graph.quantization import QuantizationEncoding
from max.graph.type import DeviceRef, Shape, TensorType
from max.graph.value import TensorValue
from max.graph.weights import WeightData, WeightsFormat, load_weights
from max.interfaces import (
    LoRAStatus,
    LoRAType,
    TextGenerationContextType,
)
from max.interfaces.pipeline import (
    Pipeline,
    PipelineInputsType,
    PipelineOutputType,
)
from max.nn.layer.layer import Module, recursive_named_layers
from max.nn.lora import SupportsLoRA
from max.pipelines.lib.config.lora_config import LoRAConfig

from .hf_utils import HuggingFaceRepo
from .lora_request_processor import LoRARequestProcessor

logger = logging.getLogger("max.serve")

ADAPTER_CONFIG_FILE = "adapter_config.json"


class LoRALRUCache:
    """LRU cache for managing active LoRA models and their slot assignments.

    This cache maintains a maximum number of active LoRA models and evicts
    the least recently used model when the cache is full. It also manages
    slot assignments for GPU buffer placement.
    """

    def __init__(self, max_size: int):
        """Initialize the LRU cache.

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
        """Get a LoRA model and its slot from the cache and mark it as recently used.

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
        """Get the slot assignment for a LoRA model.

        Args:
            key: The name of the LoRA model.

        Returns:
            The slot number if the model is active, None otherwise.
        """
        return self._name_to_slot.get(key)

    def next_slot(self) -> int | None:
        """Get the next available slot for a new LoRA.

        Returns:
            The next available slot number, or None if no slots are available.
        """
        if not self._free_slots:
            return None
        return min(self._free_slots)

    def put(
        self, key: str, value: LoRAModel, slot: int | None = None
    ) -> tuple[str | None, int | None]:
        """Add or update a LoRA model in the cache with slot assignment.

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
        """Remove a LoRA model from the cache.

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
    """Returns whether the key denotes a LoRA kind."""
    return bool(
        LoRAType.A.value in key
        or LoRAType.B.value in key
        or LoRAType.BIAS.value in key
    )


class LoRAModel:
    """Manages LoRA weights and configuration for a single adapter."""

    def __init__(
        self,
        name: str,
        path: str,
        base_dtype: DType,
        max_lora_rank: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        strict: bool = True,
    ) -> None:
        """Initializes a LoRAModel by loading its configuration and weights.

        .. code-block:: python

            lora = LoRAModel("my_adapter", "/path/to/lora", base_dtype, max_lora_rank,
                             n_heads=32, n_kv_heads=8, head_dim=128)

        Args:
            name:
                A string identifier for this adapter.
            path:
                Filesystem path is only supported
            base_dtype:
                The base model dtype.
            max_lora_rank:
                The maximum LoRA rank supported by the system.
            n_heads:
                Number of attention heads in the base model.
            n_kv_heads:
                Number of key-value heads in the base model.
            head_dim:
                Dimension of each attention head.
            strict:
                Whether to enforce strict validation while loading the adapter.

        Raises:
            ValueError: If weight files are not in the supported `safetensors` format,
                or if the keys in the weights are malformed or incomplete.
        """
        self.name = name
        self.path = path
        self.base_dtype = (
            base_dtype if not base_dtype.is_float8() else DType.bfloat16
        )
        self.max_lora_rank = max_lora_rank
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.strict = strict
        self._lora_A: dict[str, WeightData] = {}
        self._lora_B: dict[str, WeightData] = {}
        self._lora_bias: dict[str, WeightData] = {}

        self._adapter_config = self._load_weights(self.base_dtype)

        self.rank: int = self._adapter_config["r"]
        self.target_modules: list[str] = self._adapter_config["target_modules"]

        # Validate that target modules are supported
        self._validate_target_modules()

    def get(self, key: str) -> WeightData | None:
        """Gets the WeightData from the key. If key doesn't exist in model, then None is returned.

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
    def adapter_config(self) -> dict[str, Any]:
        """A dictionary containing metadata/configuration for the LoRA adapter."""
        return self._adapter_config

    def _validate_target_modules(self) -> None:
        """Validates that all target modules in the LoRA adapter are supported.

        Currently supported target modules:
        - Attention modules: q_proj, k_proj, v_proj, o_proj

        Raises:
            ValueError: If any target module is not supported.
        """
        supported_modules = {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",  # Attention modules
            # TODO E2EOPT-526
            # "gate_proj",
            # "up_proj",
            # "down_proj",  # MLP modules
        }

        unsupported_modules = []
        for module in self.target_modules:
            if module not in supported_modules:
                unsupported_modules.append(module)

        if unsupported_modules:
            supported_list = ", ".join(sorted(supported_modules))
            unsupported_list = ", ".join(unsupported_modules)
            raise ValueError(
                f"LoRA adapter contains unsupported target modules: {unsupported_list}. "
                f"Currently supported modules are: {supported_list}."
            )

    def _normalize_lora_key(self, key: str) -> str:
        """Normalizes LoRA weight keys by extracting the portion starting from `layers.<number>`.

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

    def _pad_lora_a_weight(
        self, weight_np: npt.NDArray[Any], rank: int
    ) -> npt.NDArray[Any]:
        """Pad LoRA A weights from [rank, in_features] to [max_rank, in_features]."""
        if rank < self.max_lora_rank:
            padded = np.zeros(
                (self.max_lora_rank, weight_np.shape[-1]),
                dtype=weight_np.dtype,
            )
            padded[:rank, :] = weight_np
            return padded
        return weight_np

    def _pad_lora_b_weight(
        self, weight_np: npt.NDArray[Any], rank: int
    ) -> npt.NDArray[Any]:
        """Pad LoRA B weights from [out_features, rank] to [out_features, max_rank]."""
        if rank < self.max_lora_rank:
            padded = np.zeros(
                (weight_np.shape[0], self.max_lora_rank),
                dtype=weight_np.dtype,
            )
            padded[:, :rank] = weight_np
            return padded
        return weight_np

    def _cast_all_weights(self, base_dtype: DType) -> None:
        """Cast all LoRA weights to base_dtype.

        Called after _combine_qkv_weights() so that all weights (including
        the concatenated QKV weights) are cast in one place.

        In virtual device mode (warm-cache/cross-compilation), casting is skipped
        since weights won't be used for inference - only compilation matters.
        """
        # Skip casting in virtual device mode since weights aren't needed
        if is_virtual_device_mode():
            return

        for data in self._lora_A.values():
            if isinstance(data.data, np.ndarray):
                weight_tensor = Buffer.from_numpy(data.data)
                data.data = cast_tensor_to(weight_tensor, base_dtype)

        for data in self._lora_B.values():
            if isinstance(data.data, np.ndarray):
                weight_tensor = Buffer.from_numpy(data.data)
                data.data = cast_tensor_to(weight_tensor, base_dtype)

    def _combine_qkv_weights(self) -> None:
        """Combines separate q_proj, k_proj, v_proj LoRA weights into qkv_lora weights.

        This method identifies sets of Q, K, V weights for the same layer and combines them:
        - For lora_A: concatenates across rank dimension (dim 0)
        - For lora_B: concatenates across output dimension (dim 0)

        The combined weights are stored with 'qkv_lora' keys, making them ready for use
        by the fused QKV attention layers.
        """
        # Find all unique layer prefixes that have q_proj, k_proj, v_proj
        # LoRA A and B always come in pairs, so we only need to check one
        layer_prefixes: set[str] = set()
        for key in self._lora_A:
            # Extract the layer prefix (e.g., "layers.0.self_attn")
            match = re.match(
                r"(layers\.\d+\.self_attn)\.(q_proj|k_proj|v_proj)", key
            )
            if match:
                layer_prefixes.add(match.group(1))

        # Determine default dtype from first available weight, or use base-dtype
        if self._lora_A:
            first_data = next(iter(self._lora_A.values()))
            default_dtype = first_data.dtype
        else:
            default_dtype = self.base_dtype

        # For each layer, combine q, k, v weights
        for layer_prefix in layer_prefixes:
            self._combine_qkv_for_layer(layer_prefix, default_dtype)

    def _weight_to_numpy(self, weight_data: WeightData) -> npt.NDArray[Any]:
        """Convert WeightData to numpy array (data may already be numpy or dlpack)."""
        if isinstance(weight_data.data, np.ndarray):
            return weight_data.data
        return Buffer.from_dlpack(weight_data.data).to_numpy()

    def _create_weight_data(
        self,
        np_array: npt.NDArray[Any],
        key: str,
        dtype: DType,
        quantization_encoding: QuantizationEncoding | None,
    ) -> WeightData:
        """Create WeightData from numpy array."""
        return WeightData(
            np_array,
            key,
            dtype,
            Shape(np_array.shape),
            quantization_encoding,
        )

    def _get_qkv_keys(
        self, layer_prefix: str, lora_type: LoRAType
    ) -> tuple[str, str, str]:
        """Generate Q, K, V weight keys for a layer and LoRA type."""
        return (
            f"{layer_prefix}.q_proj.{lora_type.value}.weight",
            f"{layer_prefix}.k_proj.{lora_type.value}.weight",
            f"{layer_prefix}.v_proj.{lora_type.value}.weight",
        )

    def _combine_qkv_for_layer(
        self, layer_prefix: str, default_dtype: DType
    ) -> None:
        """Combines Q, K, V weights for a specific layer.

        Args:
            layer_prefix: The layer prefix (e.g., "layers.0.self_attn")
            default_dtype: Default DType to use if no weights are present.
        """
        self._combine_lora_a_weights(layer_prefix, default_dtype)
        self._combine_lora_b_weights(layer_prefix, default_dtype)

    def _combine_lora_a_weights(
        self, layer_prefix: str, default_dtype: DType
    ) -> None:
        """Combine Q, K, V lora_A weights into a single concatenated weight."""
        q_key, k_key, v_key = self._get_qkv_keys(layer_prefix, LoRAType.A)
        keys = (q_key, k_key, v_key)
        present_keys = [k for k in keys if k in self._lora_A]

        # LoRA A input dimension is the hidden size (n_heads * head_dim)
        in_features = self.n_heads * self.head_dim

        # Use first present key as reference for dtype/quantization, or defaults
        if present_keys:
            ref_data = self._lora_A[present_keys[0]]
            src_dtype = ref_data.dtype
            quantization_encoding = ref_data.quantization_encoding
        else:
            src_dtype = default_dtype
            quantization_encoding = None

        np_dtype = src_dtype.to_numpy()

        # LoRA A shape: [max_rank, in_features]
        def get_or_zeros(key: str) -> npt.NDArray[Any]:
            if key in self._lora_A:
                return self._weight_to_numpy(self._lora_A[key])
            return np.zeros((self.max_lora_rank, in_features), dtype=np_dtype)

        q_np = get_or_zeros(q_key)
        k_np = get_or_zeros(k_key)
        v_np = get_or_zeros(v_key)

        # Shape: [rank, in_features] -> [3*rank, in_features]
        combined_np = np.concatenate([q_np, k_np, v_np], axis=0)

        combined_key = f"{layer_prefix}.qkv_lora.{LoRAType.A.value}.weight"
        self._lora_A[combined_key] = self._create_weight_data(
            combined_np,
            combined_key,
            src_dtype,
            quantization_encoding,
        )

        for key in present_keys:
            del self._lora_A[key]

    def _combine_lora_b_weights(
        self, layer_prefix: str, default_dtype: DType
    ) -> None:
        """Combine Q, K, V lora_B weights into separate Q and stacked KV weights."""
        q_key, k_key, v_key = self._get_qkv_keys(layer_prefix, LoRAType.B)
        keys = (q_key, k_key, v_key)
        present_keys = [k for k in keys if k in self._lora_B]

        # Compute output dimensions from model config
        q_out_features = self.n_heads * self.head_dim
        kv_out_features = self.n_kv_heads * self.head_dim

        # Use first present key as reference for dtype/quantization, or defaults
        if present_keys:
            ref_data = self._lora_B[present_keys[0]]
            src_dtype = ref_data.dtype
            quantization_encoding = ref_data.quantization_encoding
        else:
            src_dtype = default_dtype
            quantization_encoding = None

        np_dtype = src_dtype.to_numpy()

        # LoRA B shape: [out_features, max_rank]
        def get_q_or_zeros() -> npt.NDArray[Any]:
            if q_key in self._lora_B:
                return self._weight_to_numpy(self._lora_B[q_key])
            return np.zeros(
                (q_out_features, self.max_lora_rank), dtype=np_dtype
            )

        def get_kv_or_zeros(key: str) -> npt.NDArray[Any]:
            if key in self._lora_B:
                return self._weight_to_numpy(self._lora_B[key])
            return np.zeros(
                (kv_out_features, self.max_lora_rank), dtype=np_dtype
            )

        q_np = get_q_or_zeros()
        k_np = get_kv_or_zeros(k_key)
        v_np = get_kv_or_zeros(v_key)

        # Q weight: shape [q_out_features, rank]
        q_combined_key = f"{layer_prefix}.qkv_lora.{LoRAType.B.value}_q.weight"
        self._lora_B[q_combined_key] = self._create_weight_data(
            q_np,
            q_combined_key,
            src_dtype,
            quantization_encoding,
        )

        # KV weight: shape [2, kv_out_features, rank]
        kv_np = np.stack([k_np, v_np])
        kv_combined_key = (
            f"{layer_prefix}.qkv_lora.{LoRAType.B.value}_kv.weight"
        )
        self._lora_B[kv_combined_key] = self._create_weight_data(
            kv_np,
            kv_combined_key,
            src_dtype,
            quantization_encoding,
        )

        for key in present_keys:
            del self._lora_B[key]

    def _load_weights(self, base_dtype: DType) -> dict[str, Any]:
        """Loads LoRA adapter weights and configuration from disk.

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

        # Check for bias configuration which is not currently supported
        bias_config = adapter_config.get("bias", "none")
        if bias_config != "none":
            raise ValueError(
                f"LoRA bias training is not currently supported. "
                f"Found bias='{bias_config}' in LoRA adapter '{self.name}'. "
                f"Please use a LoRA adapter with bias='none' or without bias configuration."
            )

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
        rank = adapter_config["r"]

        if rank > self.max_lora_rank:
            raise ValueError(
                f"LoRA of rank {rank} exceeds maximum rank of {self.max_lora_rank}."
            )

        # load all weights as numpy arrays
        for key, weight in weights.items():
            key = self._normalize_lora_key(key)
            data = weight.data()

            if LoRAType.A.value in key:
                weight_np = Buffer.from_dlpack(data.data).to_numpy()
                data.data = self._pad_lora_a_weight(weight_np, rank)
                self._lora_A[key] = data

            elif LoRAType.B.value in key:
                # Pre-multiply scale to avoid doing it in the kernel every forward.
                # The loaded safetensors weights are read-only, so we must copy.
                weight_np = (
                    Buffer.from_dlpack(data.data).copy().to_numpy() * scale
                )
                data.data = self._pad_lora_b_weight(weight_np, rank)
                self._lora_B[key] = data

            elif LoRAType.BIAS.value in key:
                # Skip casting in virtual device mode (warm-cache)
                if not is_virtual_device_mode():
                    data.data = cast_dlpack_to(
                        data.data, data.dtype, base_dtype, CPU()
                    )
                self._lora_bias[key] = data

            else:
                raise ValueError(f"Invalid LoRA type got key: {key}")

        # Combine Q, K, V weights into fused QKV weights
        self._combine_qkv_weights()
        self._cast_all_weights(base_dtype)

        return adapter_config


class LoRAManager:
    """Manages multiple LoRA models and buffers for the forward pass.

    Applies multiple LoRA models to a set of base weights and manages the
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
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        zmq_endpoint_base: str,
    ):
        """Initializes the LoRAManager with a given base weight structure and maximum number of LoRA models.

        Args:
            config (LoRAConfig): The LoRA config.
            base_model_path (str): The name/path of the base model.
            base_dtype (DType): The base model dtype.
            n_heads (int): Number of attention heads in the base model.
            n_kv_heads (int): Number of key-value heads in the base model.
            head_dim (int): Dimension of each attention head.
            zmq_endpoint_base (str): The ZMQ endpoint base used to construct ZMQ lora request and response endpoints.
        """
        self.base_model_path = base_model_path
        self.base_dtype = base_dtype
        self.max_num_loras = config.max_num_loras
        self.max_lora_rank = config.max_lora_rank
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self._loras: dict[str, LoRAModel] = dict()
        self._active_loras: LoRALRUCache = LoRALRUCache(
            max_size=self.max_num_loras
        )

        self._request_processor = LoRARequestProcessor(self, zmq_endpoint_base)

        if config.lora_paths:
            self._load_adapters(config.lora_paths)

        self._alias_buffers: dict[str, DLPackArray] = {}

    def process_lora_requests(self) -> None:
        """Check for new LoRA requests and processes them."""
        self._request_processor.process_lora_requests()

    @property
    def loras(self) -> list[str]:
        """Returns the list of loaded LoRA adapter names."""
        return list(self._loras.keys())

    def _model_name_to_id(self, name: str | None) -> int:
        """Maps the model name to its assigned slot id.

        Base model requests are ID == _NO_ACTIVE_LORA (-1)
        Empty string or base model path maps to base model.
        """
        # Empty string, None, or base model path all map to base model
        if not name or name == self.base_model_path:
            return self._NO_ACTIVE_LORA

        if name in self._loras:
            slot = self._active_loras.get_slot(name)
            if slot is not None:
                return slot
        return self._NO_ACTIVE_LORA

    def _model_name_to_rank(self, name: str | None) -> int:
        """Maps the model name to its rank."""
        return self._loras[name].rank if name in self._loras else 0

    def get_lora_graph_inputs(
        self,
        context_batch: Sequence[TextGenerationContextType],
        input_row_offsets: npt.NDArray[np.integer[Any]],
        device: Device,
    ) -> tuple[Buffer, ...]:
        """Returns the LoRA graph inputs for the batch.

        Args:
            context_batch: The generation contexts for the batch.
            input_row_offsets: The offsets for each sequence in the batch.
            device: The device.
        """
        ids = []
        ranks = []

        for ctx in context_batch:
            name = getattr(ctx, "model_name", None)
            # Empty string, None, or base model path are all valid (use base model)
            # Only raise error if a non-empty name that's not the base model and not a loaded LoRA
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

            ids.append(self._model_name_to_id(name))
            ranks.append(self._model_name_to_rank(name))

        grouped_ids = []
        grouped_ranks = []
        grouped_offsets = []

        prev_id = None

        # group contexts that use the same LoRA
        for i, id_ in enumerate(ids):
            if id_ != prev_id:
                grouped_ids.append(id_)
                grouped_ranks.append(ranks[i])
                grouped_offsets.append(input_row_offsets[i])
                prev_id = id_

        grouped_offsets.append(input_row_offsets[-1])
        last_lora_idx = (
            grouped_ids.index(-1) if -1 in grouped_ids else len(grouped_ids)
        )

        grouped_offsets = grouped_offsets[: last_lora_idx + 1]
        grouped_ids = grouped_ids[:last_lora_idx]
        grouped_ranks = grouped_ranks[:last_lora_idx]

        # For KV cache iadd optimization with shape [2m, N]
        # We need offsets for both K (first m rows) and V (next m rows)
        # Create duplicate offsets: first for K portion, then for V portion
        lora_end_idx = grouped_offsets[-1]
        grouped_offsets_kv = []
        grouped_ids_kv = []

        # Add K portion: G groups with G+1 offsets (includes lora_end_idx)
        for offset in grouped_offsets:
            grouped_offsets_kv.append(offset)
        for id_ in grouped_ids:
            grouped_ids_kv.append(id_)

        # Add V portion: G groups with G offsets (skip first to avoid duplicate lora_end_idx)
        for offset in grouped_offsets[1:]:
            grouped_offsets_kv.append(lora_end_idx + offset)
        for id_ in grouped_ids:
            grouped_ids_kv.append(id_ + self.max_num_loras)

        lora_ids = Buffer.from_numpy(np.array(grouped_ids, dtype=np.int32)).to(
            device
        )
        lora_ranks = Buffer.from_numpy(np.array(grouped_ranks, dtype=np.uint32))
        lora_grouped_offsets = Buffer.from_numpy(
            np.array(grouped_offsets, dtype=np.uint32)
        ).to(device)
        num_active_loras = Buffer.from_numpy(
            np.array([last_lora_idx], dtype=np.int64)
        )
        # TODO: This is a hacky workaround for creating a dynamic shaped output
        #  tensor inside of the graph and is a byproduct of not being able
        #  to use TensorValues as shapes without slicing. Aka we can slice:
        #  x[[slice(None, TensorValue), "dynamic"]] but incurs a slice which has overhead.
        #  The approach below is working around slicing and just creating a
        #  zero'd tensor of shape "dynamic" and setting the first element to
        #  this "dynamic" value since we also need it for ops within the LoRA
        #  computation.
        lora_end_zeros = np.zeros([lora_end_idx], dtype=np.int64)
        if lora_end_idx != 0:
            lora_end_zeros[0] = lora_end_idx
        lora_end = Buffer.from_numpy(lora_end_zeros)
        batch_seq_len = Buffer.from_numpy(
            np.array([input_row_offsets[-1]], dtype=np.int64)
        )

        lora_ids_kv = Buffer.from_numpy(
            np.array(grouped_ids_kv, dtype=np.int32)
        ).to(device)
        lora_grouped_offsets_kv = Buffer.from_numpy(
            np.array(grouped_offsets_kv, dtype=np.uint32)
        ).to(device)

        return (
            lora_ids,
            lora_ranks,
            lora_grouped_offsets,
            num_active_loras,
            lora_end,
            batch_seq_len,
            lora_ids_kv,
            lora_grouped_offsets_kv,
        )

    def _validate_lora_path(self, path: str) -> LoRAStatus:
        """Validates that a LoRA adapter path exists locally.

        Remote Hugging Face repositories are not supported and must be downloaded
        to a local directory first.

        Args:
            path: The path to validate.

        """
        if not os.path.exists(path):
            return LoRAStatus.LOAD_INVALID_PATH

        return LoRAStatus.SUCCESS

    def _load_adapters(self, lora_paths: list[str]) -> None:
        """Internal method to load LoRA adapters during initialization.

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
        """Loads a single LoRA adapter from the given path and registers it under a unique name.

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

            if (status := self._validate_lora_path(path)) != LoRAStatus.SUCCESS:
                return status

            try:
                lora = LoRAModel(
                    name,
                    path,
                    self.base_dtype,
                    self.max_lora_rank,
                    self.n_heads,
                    self.n_kv_heads,
                    self.head_dim,
                )
            except ValueError:
                return LoRAStatus.LOAD_INVALID_ADAPTER

            self._loras[lora.name] = lora
            return LoRAStatus.SUCCESS

        except Exception as e:
            logger.exception(
                f"Unexpected error loading LoRA adapter from '{path}': {e}"
            )
            return LoRAStatus.LOAD_ERROR

    def unload_adapter(self, name: str) -> LoRAStatus:
        """Unloads the specified LoRA adapter from the internal registry and frees its slot.

        This function is used to release GPU or CPU memory by removing a LoRA model.

        .. code-block:: python

            manager.unload_adapter("my_adapter")

        Args:
            name: The name of the LoRA adapter to unload.

        Returns:
            LoRAStatus indicating the result of the unload operation.
        """
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
        """Moves the specified LoRA adapter to GPU and marks it as active.

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
        """Updates the alias buffers with weights from a newly activated LoRA.

        This function copies the LoRA weights (A, B, and bias) into the appropriate
        slot in the alias buffers, which are used for dynamic LoRA swapping during
        inference.

        Args:
            lora: The LoRAModel instance containing the weights.
            slot: The slot index where the LoRA weights should be placed.
        """
        for state_key in self._alias_buffers:
            buffer = Buffer.from_dlpack(self._alias_buffers[state_key])

            lora_weight = lora.get(state_key)
            weight: Buffer
            if lora_weight:
                weight = Buffer.from_dlpack(lora_weight.data)
            else:
                weight = Buffer.zeros(
                    buffer.shape[1:],
                    dtype=buffer.dtype,
                    device=buffer.device,
                )

            if LoRAType.B_KV.value in state_key:
                if lora_weight:
                    buffer[slot, :, :].inplace_copy_from(weight[0, :, :])
                    buffer[slot + self.max_num_loras, :, :].inplace_copy_from(
                        weight[1, :, :]
                    )
                else:
                    buffer[slot, :, :].inplace_copy_from(weight)
                    buffer[slot + self.max_num_loras, :, :].inplace_copy_from(
                        weight
                    )
            elif LoRAType.BIAS.value in state_key:
                buffer[slot, :].inplace_copy_from(weight)
            else:
                buffer[slot, :, :].inplace_copy_from(weight)

    def _get_lora_leaf_layers(self, model: Module) -> dict[str, Module]:
        """Returns leaf module names that are instances of SupportsLoRA.

        Uses recursive_named_layers(model), skipping containers.

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
        """Recursively collects leaf SupportsLoRA modules and inits their weights.

        Inits their weights with the loaded LoRAs and adds them to the
        ``state_dict``.

        Acquires the alias-able buffers for dynamic LoRA swapping.

        Must be called to initialize the base model properly.

        Args:
            model: The top-level Module.
            state_dict: Model state_dict to be loaded into model.
            device: The device the base model resides in.
        """
        self._lora_layers = self._get_lora_leaf_layers(model)

        for key, layer in self._lora_layers.items():
            for weight_key, base_weight in layer.layer_weights.items():
                if not is_lora_kind(weight_key):
                    continue

                state_key = f"{key}.{weight_key}"
                weight = Buffer.zeros(
                    base_weight.shape.static_dims, base_weight.dtype
                ).copy(base_weight.device.to_device())
                state_dict[state_key] = WeightData(
                    weight,
                    key,
                    base_weight.dtype,
                    base_weight.shape,
                    base_weight.quantization_encoding,
                )

                self._alias_buffers[state_key] = state_dict[state_key].data

    def get_symbolic_inputs(self, device_ref: DeviceRef) -> list[TensorType]:
        """Returns the input symbols needed for the graph inputs.

        Args:
            device_ref: Symbolic device to be used for the symbols.

        Returns:
            The graph input symbols.
        """
        lora_ids_type = TensorType(
            DType.int32, shape=["lora_ids"], device=device_ref
        )
        lora_ranks_type = TensorType(
            DType.uint32, shape=["lora_ranks"], device=DeviceRef.CPU()
        )
        lora_grouped_offsets_type = TensorType(
            DType.uint32, shape=["lora_grouped_offsets"], device=device_ref
        )
        num_active_loras_type = TensorType(
            DType.int64, shape=[1], device=DeviceRef.CPU()
        )
        lora_end_idx_type = TensorType(
            DType.int64, shape=["lora_end_idx"], device=DeviceRef.CPU()
        )
        batch_seq_len_type = TensorType(
            DType.int64, shape=[1], device=DeviceRef.CPU()
        )
        lora_ids_kv_type = TensorType(
            DType.int32, shape=["lora_ids_kv"], device=device_ref
        )
        lora_grouped_offsets_kv_type = TensorType(
            DType.uint32, shape=["lora_grouped_offsets_kv"], device=device_ref
        )

        return [
            lora_ids_type,
            lora_ranks_type,
            lora_grouped_offsets_type,
            num_active_loras_type,
            lora_end_idx_type,
            batch_seq_len_type,
            lora_ids_kv_type,
            lora_grouped_offsets_kv_type,
        ]

    def set_graph_info(
        self,
        lora_ids: TensorValue,
        lora_ranks: TensorValue,
        lora_grouped_offsets: TensorValue,
        num_active_loras: TensorValue,
        lora_end_idx: TensorValue,
        batch_seq_len: TensorValue,
        lora_ids_kv: TensorValue,
        lora_grouped_offsets_kv: TensorValue,
    ) -> None:
        """Sets the lora batch info required for the forward-pass.

        Args:
            lora_ids: IDs of the LoRAs used in the batch.
            lora_ranks: Ranks of the LoRAs used in the batch.
            lora_grouped_offsets: Cumulative offsets for each LoRA group.
            num_active_loras: Number of active LoRA adapters in the batch.
            lora_end_idx: End index of LoRA token portion.
            batch_seq_len: Total sequence length in the batch.
            lora_ids_kv: LoRA IDs for KV cache (includes K and V portions).
            lora_grouped_offsets_kv: Cumulative offsets for KV LoRA groups.
        """
        for _, layer in self._lora_layers.items():
            if isinstance(layer, SupportsLoRA):
                layer.set_lora_batch_info(
                    lora_ids,
                    lora_ranks,
                    lora_grouped_offsets,
                    num_active_loras,
                    lora_end_idx,
                    batch_seq_len,
                    lora_ids_kv,
                    lora_grouped_offsets_kv,
                )

    def sort_lora_batch(
        self, context_batch: list[TextGenerationContextType]
    ) -> list[TextGenerationContextType]:
        """Sorts the LoRA batch by LRU cache id.

        Args:
            context_batch: The context batch to sort
        """
        return sorted(
            context_batch,
            key=lambda item: self._model_name_to_id(
                getattr(item, "model_name", None)
            ),
            reverse=True,
        )

    def is_lora(self, name: str) -> bool:
        """Returns whether the given name is a loaded LoRA adapter."""
        return name in self._loras

    def is_active_lora(self, name: str) -> bool:
        """Returns whether the given name is an active LoRA adapter."""
        return name in self._active_loras

    @staticmethod
    def get_lora_manager(
        pipeline: Pipeline[PipelineInputsType, PipelineOutputType],
    ) -> LoRAManager | None:
        """Returns the LoRAManager from the pipeline if LoRA is enabled."""
        manager: LoRAManager | None = None

        if hasattr(pipeline, "_pipeline_model"):
            manager = pipeline._pipeline_model._lora_manager
        elif hasattr(pipeline, "speech_lm_pipeline"):
            manager = pipeline.speech_lm_pipeline._pipeline_model._lora_manager
        elif hasattr(pipeline, "pipeline_model"):
            manager = pipeline.pipeline_model._lora_manager

        return manager
