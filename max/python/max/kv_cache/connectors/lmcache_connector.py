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

"""LMCache connector for external KV cache integration.

Provides a connector implementation that integrates with LMCache for tiered
external caching of KV blocks. This enables:
- Sharing KV cache blocks across multiple model instances and machines
- Tiered storage: Local CPU → Local Disk → Remote (Redis, etc.)
- Async fetch/store operations with optimized GPU transfers
- Automatic write-back from remote tiers to local CPU

Configuration:
    LMCache is configured via the `lmcache_config_file` field on KVCacheParams.
    If not set, default settings are used.

    Example YAML config (lmcache_config.yaml):
    ```yaml
    chunk_size: 256
    local_cpu: true
    max_local_cpu_size: 10  # GB
    local_disk: /path/to/disk/cache  # Optional
    max_local_disk_size: 50  # GB
    remote_url: redis://localhost:6379  # Optional
    ```

    See https://docs.lmcache.ai/api_reference/configurations.html for full options.

Architecture:
    The connector-based architecture separates concerns:
    - Manager (_TPPagedKVCacheManager) owns device tensors and block allocation
    - Connector (LMCacheConnector) handles external tier operations via LMCacheEngine

    The connector uses MAXGPUConnector to bridge MAX's Buffer tensors with
    LMCache's MemoryObj-based storage using custom Mojo kernels for efficient
    GPU data transfers.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from typing import Any

import torch  # type: ignore[import-not-found]
from lmcache.utils import CacheEngineKey  # type: ignore[import-not-found]
from lmcache.v1.cache_engine import (  # type: ignore[import-not-found]
    GPUConnectorInterface,
    LMCacheEngine,
    LMCacheEngineBuilder,
)
from lmcache.v1.config import (  # type: ignore[import-not-found]
    LMCacheEngineConfig,
)
from lmcache.v1.memory_management import (  # type: ignore[import-not-found]
    MemoryObj,
)
from lmcache.v1.metadata import (  # type: ignore[import-not-found]
    LMCacheMetadata,
)
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, Graph, TensorType, Type
from max.graph.type import DeviceRef
from max.interfaces import RequestID, TextGenerationContext
from max.nn.kernels import lmcache_offload, lmcache_onload
from max.nn.kv_cache import KVCacheBuffer, KVCacheParams
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.profiler import traced

logger = logging.getLogger("max.pipelines")


def _max_dtype_to_torch(dtype: DType) -> torch.dtype:
    """Convert MAX DType to torch dtype."""
    dtype_map = {
        DType.float32: torch.float32,
        DType.float16: torch.float16,
        DType.bfloat16: torch.bfloat16,
        DType.int8: torch.int8,
        DType.int32: torch.int32,
        DType.int64: torch.int64,
        DType.float8_e4m3fn: torch.float8_e4m3fn,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype for LMCache: {dtype}")
    return dtype_map[dtype]


class MAXGPUConnector(GPUConnectorInterface):
    """GPU connector for MAX Buffer tensors using optimized Mojo kernels.

    This connector bridges MAX's Buffer-based KV cache with LMCache's
    MemoryObj-based storage. It uses custom Mojo kernels for efficient
    GPU data transfer.

    MAX paged KV cache layout:
        [total_num_blocks, kv_dim, num_layers, page_size, num_kv_heads, head_dim]
        where kv_dim = 2 for standard attention, 1 for MLA

    LMCache's expected memory format (KV_2LTD):
        [kv_dim, num_layers, num_tokens, hidden_dim]
        where hidden_dim = num_kv_heads * head_dim

    The slot_mapping kwarg maps token positions to cache slots:
        slot = block_id * page_size + offset_in_block
    """

    def __init__(
        self,
        params: KVCacheParams,
        device_buffer: KVCacheBuffer,
        devices: Sequence[Device],
        total_num_blocks: int,
        session: InferenceSession | None = None,
    ) -> None:
        """Initialize the MAX GPU connector.

        Args:
            params: KV cache parameters containing model configuration.
            device_buffer: Device buffer for KV cache (owned by manager).
            devices: Devices for the KV cache buffers.
            total_num_blocks: Total number of blocks in the paged cache.
            session: Inference session for loading kernels.
        """
        self._device_buffer = device_buffer
        self._num_layers = params.num_layers
        self._num_kv_heads = params.n_kv_heads_per_device
        self._head_dim = params.head_dim
        self._block_size = params.page_size
        self._kv_dtype = _max_dtype_to_torch(params.dtype)
        self._max_dtype = params.dtype
        self._hidden_dim = params.n_kv_heads_per_device * params.head_dim
        self._kv_dim = 1 if params.is_mla else 2
        self._session = session
        self._devices = list(devices)
        self._total_num_blocks = total_num_blocks

        # Track the current TP shard being used
        self._current_tp_idx = 0

        # kvcaches pointer (set by initialize_kvcaches_ptr)
        self.kvcaches: list[torch.Tensor] | None = None

        # Build and compile transfer graphs if session is provided
        self._offload_models: list[Model] | None = None
        self._onload_models: list[Model] | None = None
        if session is not None:
            self._build_transfer_graphs(session)

    def _build_transfer_graphs(self, session: InferenceSession) -> None:
        """Build and compile graphs for offload/onload operations.

        Uses named dimensions for dynamic shapes (num_tokens, slot_mapping_len).
        """
        self._offload_models = []
        self._onload_models = []

        for device_idx, _ in enumerate(self._devices):
            device_ref = DeviceRef.GPU(device_idx)

            # Build offload graph (GPU paged cache -> external contiguous)
            offload_graph = self._build_offload_graph(device_ref, device_idx)
            self._offload_models.append(session.load(offload_graph))

            # Build onload graph (external contiguous -> GPU paged cache)
            onload_graph = self._build_onload_graph(device_ref, device_idx)
            self._onload_models.append(session.load(onload_graph))

        logger.debug(f"Built transfer graphs for {len(self._devices)} devices")

    def _build_offload_graph(
        self, device_ref: DeviceRef, device_idx: int
    ) -> Graph:
        """Build graph for offloading KV data from paged cache to external format.

        Args:
            device_ref: Device reference for the graph.
            device_idx: Index of the device for tensor shapes.

        Returns:
            Graph that copies from paged cache to contiguous output.
        """
        # Paged cache shape: [total_blocks, kv_dim, num_layers, page_size, num_kv_heads, head_dim]
        paged_cache_shape = [
            self._total_num_blocks,
            self._kv_dim,
            self._num_layers,
            self._block_size,
            self._num_kv_heads,
            self._head_dim,
        ]

        input_types: list[Type[Any]] = [
            # Output buffer [kv_dim, num_layers, num_tokens, hidden_dim]
            # Must be BufferType for inplace_custom to work
            BufferType(
                self._max_dtype,
                [
                    self._kv_dim,
                    self._num_layers,
                    "num_tokens",
                    self._hidden_dim,
                ],
                device=device_ref,
            ),
            # Paged cache tensor (static shape, read-only for offload)
            TensorType(
                self._max_dtype,
                paged_cache_shape,
                device=device_ref,
            ),
            # Slot mapping [slot_mapping_len]
            TensorType(DType.int64, ["slot_mapping_len"], device=device_ref),
            # Start token scalar
            TensorType(DType.int64, [1], device=DeviceRef.CPU()),
            # End token scalar
            TensorType(DType.int64, [1], device=DeviceRef.CPU()),
        ]

        with Graph(
            f"lmcache_offload_{device_idx}", input_types=input_types
        ) as graph:
            (
                output_val,
                paged_cache_val,
                slot_mapping_val,
                start_token_val,
                end_token_val,
            ) = graph.inputs

            lmcache_offload(
                output=output_val.buffer,
                paged_cache=paged_cache_val.tensor,
                slot_mapping=slot_mapping_val.tensor,
                start_token=start_token_val.tensor,
                end_token=end_token_val.tensor,
                page_size=self._block_size,
                num_kv_heads=self._num_kv_heads,
                head_dim=self._head_dim,
                kv_dim=self._kv_dim,
                device_ref=device_ref,
            )

            graph.output(output_val)

        return graph

    def _build_onload_graph(
        self, device_ref: DeviceRef, device_idx: int
    ) -> Graph:
        """Build graph for onloading KV data from external format to paged cache.

        Args:
            device_ref: Device reference for the graph.
            device_idx: Index of the device for tensor shapes.

        Returns:
            Graph that copies from contiguous input to paged cache.
        """
        # Paged cache shape: [total_blocks, kv_dim, num_layers, page_size, num_kv_heads, head_dim]
        paged_cache_shape = [
            self._total_num_blocks,
            self._kv_dim,
            self._num_layers,
            self._block_size,
            self._num_kv_heads,
            self._head_dim,
        ]

        # Define input types
        onload_input_types: list[Type[Any]] = [
            # Paged cache buffer (static shape, modified in-place)
            # Must be BufferType for inplace_custom to work
            BufferType(
                self._max_dtype,
                paged_cache_shape,
                device=device_ref,
            ),
            # Input tensor [kv_dim, num_layers, num_tokens, hidden_dim]
            TensorType(
                self._max_dtype,
                [
                    self._kv_dim,
                    self._num_layers,
                    "num_tokens",
                    self._hidden_dim,
                ],
                device=device_ref,
            ),
            # Slot mapping [slot_mapping_len]
            TensorType(DType.int64, ["slot_mapping_len"], device=device_ref),
            # Start token scalar
            TensorType(DType.int64, [1], device=DeviceRef.CPU()),
            # End token scalar
            TensorType(DType.int64, [1], device=DeviceRef.CPU()),
        ]

        with Graph(
            f"lmcache_onload_{device_idx}",
            input_types=onload_input_types,
        ) as graph:
            (
                paged_cache_val,
                input_tensor_val,
                slot_mapping_val,
                start_token_val,
                end_token_val,
            ) = graph.inputs

            lmcache_onload(
                paged_cache=paged_cache_val.buffer,
                input_tensor=input_tensor_val.tensor,
                slot_mapping=slot_mapping_val.tensor,
                start_token=start_token_val.tensor,
                end_token=end_token_val.tensor,
                page_size=self._block_size,
                num_kv_heads=self._num_kv_heads,
                head_dim=self._head_dim,
                kv_dim=self._kv_dim,
                device_ref=device_ref,
            )

            graph.output(paged_cache_val)

        return graph

    def set_tp_shard(self, tp_idx: int) -> None:
        """Set the current TP shard for operations."""
        self._current_tp_idx = tp_idx

    def initialize_kvcaches_ptr(self, **kwargs: Any) -> None:
        """Initialize the kvcaches pointers if provided in kwargs."""
        if "kvcaches" in kwargs:
            self.kvcaches = kwargs["kvcaches"]

    def get_shape(self, num_tokens: int) -> torch.Size:
        """Get the shape of KV data for the given number of tokens.

        Returns shape in LMCache's KV_2LTD format:
            [kv_dim, num_layers, num_tokens, hidden_dim]
        """
        return torch.Size(
            [self._kv_dim, self._num_layers, num_tokens, self._hidden_dim]
        )

    def from_gpu(
        self, memory_obj: MemoryObj, start: int, end: int, **kwargs: Any
    ) -> None:
        """Copy KV cache data from GPU to memory object.

        Args:
            memory_obj: Target LMCache memory object.
            start: Starting token index.
            end: Ending token index (exclusive).
            **kwargs: Must contain 'slot_mapping' tensor.
        """
        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' must be provided in kwargs")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        num_tokens = end - start
        assert memory_obj.tensor is not None

        if num_tokens <= 0:
            return

        if self._offload_models is None:
            raise RuntimeError(
                "MAXGPUConnector requires a session for GPU transfers. "
                "Pass session to LMCacheConnector constructor."
            )

        device_buffer = self._device_buffer.values[self._current_tp_idx]
        model = self._offload_models[self._current_tp_idx]

        # TODO: SERVOPT-1026
        # Create output buffer from the memory object tensor.
        # LMCache uses pinned memory tensors, but Buffer.from_dlpack
        # doesn't support pinned CPU tensors, so clone to unpin first.
        cpu_tensor = memory_obj.tensor.cpu().contiguous().clone()
        output_buffer = Buffer.from_dlpack(cpu_tensor).to(device_buffer.device)

        # Route slot_mapping through CPU to avoid torch's __dlpack__ device
        # The tensor is small (num_tokens \times int64), so we might have to pay
        # the copy cost here (which is negligible)
        slot_mapping_buffer = Buffer.from_dlpack(slot_mapping.cpu()).to(
            device_buffer.device
        )

        start_buffer = Buffer.from_numpy(
            torch.tensor([start], dtype=torch.int64).numpy()
        )
        end_buffer = Buffer.from_numpy(
            torch.tensor([end], dtype=torch.int64).numpy()
        )

        model.execute(
            output_buffer,
            device_buffer,
            slot_mapping_buffer,
            start_buffer,
            end_buffer,
        )

        result_tensor = torch.from_dlpack(output_buffer)
        memory_obj.tensor.copy_(result_tensor)

    def to_gpu(
        self, memory_obj: MemoryObj, start: int, end: int, **kwargs: Any
    ) -> None:
        """Copy KV cache data from memory object to GPU.

        Args:
            memory_obj: Source LMCache memory object.
            start: Starting token index.
            end: Ending token index (exclusive).
            **kwargs: Must contain 'slot_mapping' tensor.
        """
        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' must be provided in kwargs")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        num_tokens = end - start
        assert memory_obj.tensor is not None

        if num_tokens <= 0:
            return

        if self._onload_models is None:
            raise RuntimeError(
                "MAXGPUConnector requires a session for GPU transfers. "
                "Pass session to LMCacheConnector constructor."
            )

        device_buffer = self._device_buffer.values[self._current_tp_idx]
        model = self._onload_models[self._current_tp_idx]

        # TODO: SERVOPT-1026
        # Create input buffer from the memory object tensor.
        # LMCache uses pinned memory tensors — clone to unpin for from_dlpack.
        cpu_tensor = memory_obj.tensor.cpu().contiguous().clone()
        input_buffer = Buffer.from_dlpack(cpu_tensor).to(device_buffer.device)

        slot_mapping_buffer = Buffer.from_dlpack(slot_mapping.cpu()).to(
            device_buffer.device
        )

        start_buffer = Buffer.from_numpy(
            torch.tensor([start], dtype=torch.int64).numpy()
        )
        end_buffer = Buffer.from_numpy(
            torch.tensor([end], dtype=torch.int64).numpy()
        )

        model.execute(
            device_buffer,
            input_buffer,
            slot_mapping_buffer,
            start_buffer,
            end_buffer,
        )

    def batched_from_gpu(
        self,
        memory_objs: list[MemoryObj],
        starts: list[int],
        ends: list[int],
        **kwargs: Any,
    ) -> None:
        """Batched copy from GPU to memory objects."""
        # TODO: SERVOPT-1025
        for memory_obj, start, end in zip(
            memory_objs, starts, ends, strict=True
        ):
            self.from_gpu(memory_obj, start, end, **kwargs)

    def batched_to_gpu(
        self,
        memory_objs: list[MemoryObj],
        starts: list[int],
        ends: list[int],
        **kwargs: Any,
    ) -> None:
        """Batched copy from memory objects to GPU."""

        # TODO: SERVOPT-1025
        for memory_obj, start, end in zip(
            memory_objs, starts, ends, strict=True
        ):
            self.to_gpu(memory_obj, start, end, **kwargs)


class LMCacheConnector:
    """LMCache connector for external KV cache integration.

    This connector implementation uses LMCache's full engine for tiered
    external caching of KV blocks. The engine handles:
    1. StorageManager: Tiered storage (CPU → Disk → Remote) with write-back
    2. TokenDatabase: Token hashing (we pass pre-computed hashes)
    3. GPUConnector: Efficient GPU↔CPU transfers (we provide MAXGPUConnector)

    Configuration is loaded from a YAML file specified via the
    `lmcache_config_file` field on KVCacheParams.

    Key design decisions:
    - The manager owns device tensors and block allocation
    - This connector only handles external tier operations
    - LMCacheEngine handles all tiered storage management
    - Keys include TP shard info for correct multi-GPU handling

    Attributes:
        block_size: Number of tokens stored per block.
    """

    @traced
    def __init__(
        self,
        params: KVCacheParams,
        devices: Sequence[Device],
        device_buffer: KVCacheBuffer,
        total_num_blocks: int,
        session: InferenceSession | None = None,
    ) -> None:
        """Initialize the LMCache connector.

        Args:
            params: KV cache parameters containing configuration.
            devices: List of devices for tensor parallelism.
            device_buffer: Device buffer for KV cache (owned by manager).
            total_num_blocks: Total number of device blocks.
            session: Optional inference session for loading kernels.

        Raises:
            ValueError: If prefix caching is disabled (required for LMCache).
            ImportError: If lmcache is not installed.
        """
        self._session = session

        if not params.enable_prefix_caching:
            raise ValueError(
                "LMCacheConnector requires prefix caching to be enabled"
            )

        self._devices = list(devices)
        self._device_buffer = device_buffer
        self._block_size = params.page_size
        self._total_num_blocks = total_num_blocks
        self.params = params
        self._world_size = len(devices)
        self._model_name = f"max-model-{uuid.uuid4().hex[:8]}"

        self._gpu_connector = MAXGPUConnector(
            params=params,
            device_buffer=device_buffer,
            devices=devices,
            total_num_blocks=total_num_blocks,
            session=session,
        )
        self._engine = self._create_engine()

        # Pending saves (block_id, block_hash) tuples
        self._pending_saves: list[tuple[int, int]] = []

        self._pending_loads: dict[str, list[tuple[int, int]]] = {}

        self._h2d_blocks_copied: int = 0
        self._d2h_blocks_copied: int = 0

        self._is_shutdown: bool = False

    def _create_engine(self) -> LMCacheEngine:
        """Create LMCacheEngine using YAML config file.

        LMCache reads configuration from `self.params.lmcache_config_file`.
        If not set, default settings are used.

        Returns:
            A LMCacheEngine instance configured from YAML or defaults.

        Raises:
            ImportError: If lmcache is not installed.
        """

        kv_dtype = _max_dtype_to_torch(self.params.dtype)

        # kv_shape format: (num_layers, kv_dim, chunk_size, num_kv_heads, head_dim)
        # MLA caches a single fused latent vector (kv_dim=1) instead of
        # separate K and V tensors (kv_dim=2).
        # Use per-device heads since each TP shard operates independently.
        kv_dim = 1 if self.params.is_mla else 2
        kv_shape = (
            self.params.num_layers,
            kv_dim,
            self._block_size,
            self.params.n_kv_heads_per_device,
            self.params.head_dim,
        )

        # Load config from lmcache_config_file if set, else use defaults
        # Then override chunk_size to match our block size
        config_file = self.params.lmcache_config_file
        if config_file:
            config = LMCacheEngineConfig.from_file(config_file)
            config.chunk_size = self._block_size
        else:
            config = LMCacheEngineConfig(chunk_size=self._block_size)

        # MAXGPUConnector uses all-layers format (KV_2LTD / KV_MLA_FMT),
        # not per-layer formats (KV_T2D / KV_2TD).
        if config.use_layerwise:
            raise ValueError(
                "use_layerwise=True is not supported by MAXGPUConnector. "
                "Remove 'use_layerwise' from the LMCache config file."
            )

        # Create metadata with required KV cache information
        # Note: chunk_size is set in config, not metadata
        metadata = LMCacheMetadata(
            model_name=self._model_name,
            world_size=self._world_size,
            local_world_size=self._world_size,  # Same as world_size for single-node
            worker_id=0,  # We handle TP ourselves
            local_worker_id=0,  # Same as worker_id for single-node
            kv_dtype=kv_dtype,
            kv_shape=kv_shape,
            # LMCache derives its internal MemoryFormat from use_mla:
            #   MLA:      KV_MLA_FMT  [1, num_layers, num_tokens, hidden_dim]
            #   Non-MLA:  KV_2LTD     [2, num_layers, num_tokens, hidden_dim]
            # where hidden_dim = num_kv_heads * head_dim
            use_mla=self.params.is_mla,
            chunk_size=self._block_size,
        )

        # LMCache expects broadcast functions for cross-rank coordination
        # in multi-process tensor-parallel (TP) setups (e.g., vLLM uses
        # torch.distributed.broadcast). MAX uses a single-process TP model
        # where all shards are managed in-process, so no inter-process
        # communication is needed.
        def noop_broadcast(_tensor: torch.Tensor, _src: int) -> None:
            pass

        def noop_broadcast_object(obj: Any, _src: int) -> Any:
            return obj

        instance_id = f"max-{self._model_name}-{id(self)}"
        engine = LMCacheEngineBuilder.get_or_create(
            instance_id=instance_id,
            config=config,
            metadata=metadata,
            gpu_connector=self._gpu_connector,
            broadcast_fn=noop_broadcast,
            broadcast_object_fn=noop_broadcast_object,
        )

        engine.post_init()

        logger.info(
            f"Created LMCacheEngine '{instance_id}' for model '{self._model_name}' "
            f"with chunk_size={self._block_size}, "
            f"config_file={config_file or '(using defaults)'}"
        )

        return engine

    # =========================================================================
    # KVConnector Protocol Implementation
    # =========================================================================

    @property
    def name(self) -> str:
        """Connector name for logging/debugging."""
        return "LMCacheConnector"

    @traced
    def lookup(
        self,
        ctx: TextGenerationContext,
        block_hashes: list[int],
    ) -> int:
        """Look up blocks in LMCache.

        Args:
            ctx: The request context.
            block_hashes: Hashes to look up in LMCache.

        Returns:
            Number of tokens available from LMCache.
        """
        if not block_hashes:
            return 0

        request_id = str(ctx.request_id)

        # Clear any previous lookup state for this request
        self._pending_loads.pop(request_id, None)

        offsets = [self._block_size] * len(block_hashes)
        tokens_found = self._engine.lookup(hashes=block_hashes, offsets=offsets)
        num_available = tokens_found // self._block_size

        if num_available > 0:
            # Store the available hashes for load()
            self._pending_loads[request_id] = [
                (block_hashes[i], i) for i in range(num_available)
            ]

        return num_available * self._block_size

    @traced
    def load(
        self,
        ctx: TextGenerationContext,
        target_block_ids: list[int],
    ) -> list[int]:
        """Load data from LMCache into device blocks.

        Args:
            ctx: The request context.
            target_block_ids: Device block IDs to load data into.

        Returns:
            List of block hashes for the loaded blocks.
        """
        request_id = str(ctx.request_id)
        pending = self._pending_loads.pop(request_id, None)

        if not pending or not target_block_ids:
            return []

        # Limit to available blocks
        num_to_load = min(len(pending), len(target_block_ids))
        hashes_to_load = [pending[i][0] for i in range(num_to_load)]
        offsets = [self._block_size] * num_to_load

        keys: list[CacheEngineKey] = []
        for _start, _end, key in self._engine.token_database.process_tokens(
            hashes=hashes_to_load,
            offsets=offsets,
        ):
            keys.append(key)

        if not keys:
            return []

        # Get memory objects from LMCache storage
        memory_objs = self._engine.storage_manager.batched_get(keys=keys)

        if memory_objs is None:
            logger.warning("Failed to get memory objects from LMCache")
            return []

        # Build slot mapping for data fetch
        slot_mapping: list[int] = []
        valid_memory_objs: list[MemoryObj] = []
        loaded_hashes: list[int] = []
        starts: list[int] = []
        ends: list[int] = []

        for i, (block_hash, memory_obj) in enumerate(
            zip(hashes_to_load, memory_objs, strict=False)
        ):
            if memory_obj is None:
                # Stop at first missing block (contiguous prefix required)
                break

            if i >= len(target_block_ids):
                break

            target_block_id = target_block_ids[i]

            # Build slot mapping for this block
            for offset in range(self._block_size):
                slot_mapping.append(target_block_id * self._block_size + offset)

            valid_memory_objs.append(memory_obj)
            loaded_hashes.append(block_hash)
            starts.append(i * self._block_size)
            ends.append((i + 1) * self._block_size)

        # Fetch data from LMCache to GPU
        if valid_memory_objs and slot_mapping:
            for tp_idx in range(self._world_size):
                self._gpu_connector.set_tp_shard(tp_idx)
                slot_mapping_tensor = torch.tensor(
                    slot_mapping,
                    dtype=torch.long,
                    device=f"cuda:{self._devices[tp_idx].id}",
                )
                self._gpu_connector.batched_to_gpu(
                    valid_memory_objs,
                    starts,
                    ends,
                    slot_mapping=slot_mapping_tensor,
                )

            # Release memory objects back to LMCache allocator
            for memory_obj in valid_memory_objs:
                memory_obj.ref_count_down()

            self._h2d_blocks_copied += len(valid_memory_objs)

        return loaded_hashes

    @traced
    def save(
        self,
        block_ids: list[int],
        block_hashes: list[int],
    ) -> None:
        """Queue device blocks for save to LMCache.

        Args:
            block_ids: Device block IDs to save.
            block_hashes: Hashes for the blocks being saved.
        """
        for block_id, block_hash in zip(block_ids, block_hashes, strict=True):
            self._pending_saves.append((block_id, block_hash))

    @traced
    def sync(self) -> None:
        """Wait for pending loads to complete.

        LMCacheEngine handles async operations internally via StorageManager,
        so this is typically a no-op unless there are pending prefetch operations.
        """
        # LMCacheEngine handles async internally
        pass

    @traced
    def flush(self) -> None:
        """Execute pending saves to LMCache."""
        if not self._pending_saves:
            return

        hashes = []
        offsets = []
        slot_mapping: list[int] = []

        for block_id, block_hash in self._pending_saves:
            hashes.append(block_hash)
            offsets.append(self._block_size)
            for offset in range(self._block_size):
                slot_mapping.append(block_id * self._block_size + offset)

        for tp_idx in range(self._world_size):
            self._gpu_connector.set_tp_shard(tp_idx)
            slot_mapping_tensor = torch.tensor(
                slot_mapping,
                dtype=torch.long,
                device=f"cuda:{self._devices[tp_idx].id}",
            )
            try:
                self._engine.store(
                    hashes=hashes,
                    offsets=offsets,
                    slot_mapping=slot_mapping_tensor,
                )
            except FileNotFoundError:
                # LMCache disk backend TOCTOU race: concurrent eviction
                # deletes a file between existence check and os.remove().
                logger.warning(
                    "LMCache disk eviction race during store for TP shard %d.",
                    tp_idx,
                )

        self._d2h_blocks_copied += len(self._pending_saves)
        self._pending_saves.clear()

    def on_request_complete(
        self,
        request_id: RequestID,
        block_ids: list[int],
    ) -> None:
        """Clean up request-specific state."""
        self._pending_loads.pop(str(request_id), None)

    def shutdown(self) -> None:
        """Clean shutdown of connector resources.

        This method is idempotent and can be called multiple times safely.
        """
        if self._is_shutdown:
            return

        # Flush any pending saves
        self.flush()

        instance_id = f"max-{self._model_name}-{id(self)}"
        LMCacheEngineBuilder.destroy(instance_id)

        self._pending_saves.clear()
        self._pending_loads.clear()
        self._is_shutdown = True

    # =========================================================================
    # Optional Protocol Properties
    # =========================================================================

    @property
    def num_host_blocks(self) -> int:
        """Number of host blocks available in external cache.

        Returns a large value to signal to the block manager that external
        blocks are available and lookup()/load() should be called. The actual
        storage capacity is managed internally by LMCache.
        """
        return 2**31 - 1

    @property
    def num_used_host_blocks(self) -> int:
        """Number of used host blocks.

        Note: Host blocks are managed by LMCache internally.
        """
        return 0

    def reset_prefix_cache(self) -> None:
        """Reset prefix cache.

        Note: This only affects local tracking state. LMCache state
        is not affected.
        """
        self._pending_loads.clear()
        self._pending_saves.clear()

    @property
    def metrics(self) -> KVCacheMetrics:
        """Transfer metrics for LMCache operations."""
        return KVCacheMetrics(
            h2d_blocks_copied=self._h2d_blocks_copied,
            d2h_blocks_copied=self._d2h_blocks_copied,
        )
