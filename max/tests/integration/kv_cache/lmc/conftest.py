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

"""Shared fixtures and utilities for LMCache tests."""

from __future__ import annotations

import logging
import os
import pathlib
import tempfile
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.kv_cache.connectors.lmcache_connector import (
    LMCacheConnector,
    MAXGPUConnector,
)
from max.kv_cache.paged_kv_cache.cache_manager import PagedKVCacheManager
from max.kv_cache.paged_kv_cache.tp_cache_manager import (
    _TPPagedKVCacheManager,
)
from max.nn.legacy.kv_cache import KVCacheParams
from max.pipelines.core import TextContext
from test_common.context_utils import create_text_context

logger = logging.getLogger(__name__)


# --- Data classes and constants ---


@dataclass(frozen=True)
class KVCacheTestConfig:
    """Configuration for KV cache test dimensions.

    Provides standardized test parameters with computed properties for
    derived dimensions.
    """

    num_blocks: int
    kv_dim: int = 2
    num_layers: int = 2
    page_size: int = 4
    num_kv_heads: int = 2
    head_dim: int = 4
    dtype: DType = DType.float32

    @property
    def hidden_dim(self) -> int:
        """Compute hidden dimension from head configuration."""
        return self.num_kv_heads * self.head_dim

    @property
    def paged_cache_shape(self) -> list[int]:
        """Compute paged cache tensor shape."""
        return [
            self.num_blocks,
            self.kv_dim,
            self.num_layers,
            self.page_size,
            self.num_kv_heads,
            self.head_dim,
        ]


# Page size used by the integration test fixtures (kv_cache_manager, etc.)
INTEGRATION_PAGE_SIZE = 16

# Default test configs for parameterized tests
SMALL_CONFIG = KVCacheTestConfig(
    num_blocks=8,
    kv_dim=2,
    num_layers=2,
    page_size=4,
    num_kv_heads=2,
    head_dim=4,
)

MODEL_CONFIG = KVCacheTestConfig(
    num_blocks=16,
    kv_dim=2,
    num_layers=4,
    page_size=64,
    num_kv_heads=4,
    head_dim=64,
)


# --- Utility functions ---


def create_memory_obj(
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    fmt: MemoryFormat | None = None,
) -> TensorMemoryObj:
    """Create a real LMCache TensorMemoryObj for testing.

    Args:
        shape: Shape of the tensor to create.
        dtype: PyTorch dtype for the tensor.
        fmt: LMCache memory format (defaults to KV_2LTD if not specified).

    Returns:
        Initialized TensorMemoryObj with zeroed data.
    """
    if fmt is None:
        fmt = MemoryFormat.KV_2LTD

    numel = 1
    for dim in shape:
        numel *= dim
    size_bytes = numel * dtype.itemsize
    raw_data = torch.zeros(size_bytes, dtype=torch.uint8)
    metadata = MemoryObjMetadata(
        shape=torch.Size(shape),
        dtype=dtype,
        address=raw_data.data_ptr(),
        phy_size=size_bytes,
        ref_count=1,
        fmt=fmt,
    )
    return TensorMemoryObj(
        raw_data=raw_data, metadata=metadata, parent_allocator=None
    )


def cleanup_gpu_connector(connector: MAXGPUConnector) -> None:
    """Release compiled models held by a MAXGPUConnector.

    Each connector compiles Mojo offload/onload graphs via session.load().
    If these Model objects are left for Python's garbage collector at shutdown,
    the process hangs. Explicitly clearing them prevents this.
    """
    connector._offload_models = None
    connector._onload_models = None


def create_max_gpu_connector(
    config: KVCacheTestConfig,
    device: Accelerator | None = None,
    session: InferenceSession | None = None,
) -> tuple[MAXGPUConnector, Buffer, Accelerator, InferenceSession]:
    """Create a MAXGPUConnector with associated resources.

    Args:
        config: KV cache test configuration.
        device: Existing accelerator to reuse (creates new if None).
        session: Existing session to reuse (creates new if None).

    Returns:
        Tuple of (connector, paged_cache buffer, device, session).
    """
    if device is None:
        device = Accelerator()
    if session is None:
        session = InferenceSession(devices=[device])

    paged_cache = Buffer.zeros(
        shape=config.paged_cache_shape,
        dtype=config.dtype,
        device=device,
    )

    params = KVCacheParams(
        dtype=config.dtype,
        n_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        num_layers=config.num_layers,
        devices=[DeviceRef.GPU()],
        page_size=config.page_size,
    )

    connector = MAXGPUConnector(
        params=params,
        device_tensors=[paged_cache],
        devices=[device],
        total_num_blocks=config.num_blocks,
        session=session,
    )

    return connector, paged_cache, device, session


def fill_paged_cache(
    paged_cache: Buffer,
    config: KVCacheTestConfig,
) -> np.ndarray:
    """Fill paged cache with a deterministic, position-encoded pattern.

    Each element's value encodes its position:
    ``block_id * 1000 + kv_idx * 100 + layer_idx * 10 + offset``.

    Args:
        paged_cache: Buffer to fill.
        config: KV cache test configuration.

    Returns:
        The numpy array used to fill the cache.
    """
    paged_cache_np = np.zeros(config.paged_cache_shape, dtype=np.float32)

    for block_id in range(config.num_blocks):
        for kv_idx in range(config.kv_dim):
            for layer_idx in range(config.num_layers):
                for offset in range(config.page_size):
                    value = (
                        block_id * 1000 + kv_idx * 100 + layer_idx * 10 + offset
                    )
                    paged_cache_np[
                        block_id, kv_idx, layer_idx, offset, :, :
                    ] = value

    paged_cache.inplace_copy_from(Buffer.from_numpy(paged_cache_np))
    return paged_cache_np


def shutdown_connector_with_timeout(
    connector: LMCacheConnector, timeout: float = 15.0
) -> None:
    """Shutdown LMCache connector with timeout to avoid hanging threads.

    LMCache's observability.log_worker thread doesn't exit cleanly during
    shutdown. This uses vLLM's pattern of wrapping shutdown in a
    ThreadPoolExecutor with timeout.

    Args:
        connector: The LMCacheConnector to shutdown.
        timeout: Maximum seconds to wait for shutdown (default: 15s).
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(connector.shutdown)
        try:
            future.result(timeout=timeout)
        except TimeoutError:
            logger.warning(
                f"LMCache connector shutdown timed out after {timeout}s. "
                "Continuing with process exit."
            )


def make_dummy_context() -> TextContext:
    """Create a dummy TextContext with tokens [1, 2, 3]."""
    return create_text_context(np.array([1, 2, 3], dtype=np.int64))


def get_tp_manager(
    manager: PagedKVCacheManager,
) -> _TPPagedKVCacheManager:
    """Get the underlying _TPPagedKVCacheManager from a PagedKVCacheManager."""
    return manager._replica_managers[0]


# --- Config templates ---

LMCACHE_TEST_CONFIG = """
chunk_size: {chunk_size}
local_cpu: true
max_local_cpu_size: 1
"""

# Tiered storage config: CPU -> Disk
# Uses 1GB CPU limit to ensure all test data fits without triggering eviction.
# Note: LMCache eviction DROPS data (doesn't write to disk), so we need sufficient
# CPU capacity. Disk writes happen via batched_put() during store(), not eviction.
LMCACHE_TIERED_TEST_CONFIG = """
chunk_size: {chunk_size}
local_cpu: true
max_local_cpu_size: 1
local_disk: {disk_path}
max_local_disk_size: 1
"""


# --- Internal helpers ---


def _make_kv_cache_manager(
    session: InferenceSession,
    lmcache_config_file: str | None = None,
) -> PagedKVCacheManager:
    """Create a PagedKVCacheManager with standard integration test params.

    Args:
        session: Inference session with GPU device.
        lmcache_config_file: Path to LMCache YAML config file. When set,
            the manager creates an LMCacheConnector instead of NullConnector.
    """
    kv_params = KVCacheParams(
        dtype=DType.float32,
        num_layers=2,
        n_kv_heads=4,
        head_dim=64,
        cache_strategy="paged",
        enable_prefix_caching=True,
        enable_kvcache_swapping_to_host=False,
        page_size=INTEGRATION_PAGE_SIZE,
        devices=[DeviceRef.GPU()],
        lmcache_config_file=lmcache_config_file,
    )
    return PagedKVCacheManager(
        params=kv_params,
        session=session,
        total_num_pages=64,
        total_num_host_pages=0,
        max_batch_size=64,
    )


# --- Fixtures (config file creation) ---


@pytest.fixture(scope="module")
def lmcache_config_file() -> Generator[str, None, None]:
    """Create a temporary LMCache config file for testing."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        # chunk_size will be overridden by connector to match page_size
        f.write(LMCACHE_TEST_CONFIG.format(chunk_size=INTEGRATION_PAGE_SIZE))
        config_path = f.name

    yield config_path

    # Cleanup
    os.unlink(config_path)


@pytest.fixture(scope="function")
def lmcache_disk_config_file(
    tmp_path: str,
) -> Generator[tuple[str, str], None, None]:
    """Create a temporary LMCache config file with disk tier enabled.

    The config uses a very small CPU tier (1MB) to force spilling to disk.

    Yields:
        Tuple of (config_path, disk_directory_path).
    """
    disk_dir = pathlib.Path(tmp_path) / "lmcache_disk"
    disk_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        # chunk_size will be overridden by connector to match page_size
        f.write(
            LMCACHE_TIERED_TEST_CONFIG.format(
                chunk_size=INTEGRATION_PAGE_SIZE,
                disk_path=str(disk_dir),
            )
        )
        config_path = f.name

    yield config_path, str(disk_dir)

    # Cleanup
    os.unlink(config_path)


# --- Fixtures (KV cache managers) ---


@pytest.fixture(scope="module")
def kv_cache_manager(
    lmcache_config_file: str,
) -> Generator[tuple[PagedKVCacheManager, _TPPagedKVCacheManager], None, None]:
    """Module-scoped PagedKVCacheManager backed by LMCache (CPU-only).

    Yields:
        Tuple of (outer manager, tp manager). Most tests only need the
        tp manager; manager-integration tests also use the outer manager.
    """
    if accelerator_count() == 0:
        pytest.skip("No GPU available")

    device = Accelerator()
    session = InferenceSession(devices=[device])
    manager = _make_kv_cache_manager(
        session, lmcache_config_file=lmcache_config_file
    )
    tp_mgr = get_tp_manager(manager)

    yield manager, tp_mgr

    if isinstance(tp_mgr.connector, LMCacheConnector):
        shutdown_connector_with_timeout(tp_mgr.connector)


@pytest.fixture(scope="function")
def kv_cache_manager_with_disk(
    lmcache_disk_config_file: tuple[str, str],
) -> Generator[_TPPagedKVCacheManager, None, None]:
    """Function-scoped _TPPagedKVCacheManager backed by LMCache (CPU + Disk)."""
    if accelerator_count() == 0:
        pytest.skip("No GPU available")

    config_path, _ = lmcache_disk_config_file

    device = Accelerator()
    session = InferenceSession(devices=[device])
    manager = _make_kv_cache_manager(session, lmcache_config_file=config_path)
    tp_mgr = get_tp_manager(manager)

    yield tp_mgr

    if isinstance(tp_mgr.connector, LMCacheConnector):
        shutdown_connector_with_timeout(tp_mgr.connector)


# --- Session hooks ---


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Force exit to avoid hangs from LMCache's background threads.

    LMCache's observability.log_worker thread doesn't exit cleanly.
    shutdown_connector_with_timeout() handles the graceful attempt, but
    if it times out the lingering thread blocks normal process exit.
    """
    os._exit(exitstatus)
