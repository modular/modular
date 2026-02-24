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

"""KV cache connectors for external cache tiers.

- `NullConnector`: No-op connector when external caching is disabled
- `LocalConnector`: Host memory offloading
- `TieredConnector`: GPU <-> CPU <-> Disk offloading
- `LMCacheConnector`: LMCache integration for tiered external caching
- `create_connector()`: Factory function
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from max.driver import Buffer, Device
from max.engine import InferenceSession
from max.kv_cache.kv_connector import KVConnector
from max.nn.kv_cache import KVCacheParams

from .local_connector import LocalConnector
from .null_connector import NullConnector
from .tiered_connector import TieredConnector

logger = logging.getLogger("max.pipelines")


def create_connector(
    params: KVCacheParams,
    devices: Sequence[Device],
    device_tensors: list[Buffer],
    device_scale_tensors: list[Buffer] | None,
    total_num_host_blocks: int,
    total_num_blocks: int = 0,
    session: InferenceSession | None = None,
) -> KVConnector:
    """Create a KV cache connector instance.

    Returns a connector appropriate for the configuration:
    - If `params.lmcache_config_file` is set: LMCacheConnector
    - If swapping enabled + disk_offload_dir set: TieredConnector
    - If swapping enabled (no disk): LocalConnector
    - Otherwise: NullConnector

    Args:
        params: KV cache parameters containing configuration.
        devices: Devices for the KV cache tensors.
        device_tensors: Device tensors for KV cache (owned by manager).
        device_scale_tensors: Device scale tensors for FP8, or None.
        total_num_host_blocks: Total number of host blocks for swapping.
        total_num_blocks: Total number of device blocks (required for LMCache).
        session: Optional inference session for loading custom kernels.

    Returns:
        A connector instance implementing KVConnectorProtocol.
    """
    # TODO: SERVOPT-1020
    # Check for LMCache configuration
    if params.lmcache_config_file:
        try:
            from .lmcache_connector import LMCacheConnector
        except ImportError as e:
            raise ImportError(
                "lmcache and torch are required for LMCache integration. "
                "Install them with: pip install lmcache torch"
            ) from e

        return LMCacheConnector(
            params=params,
            devices=devices,
            device_tensors=device_tensors,
            device_scale_tensors=device_scale_tensors,
            total_num_blocks=total_num_blocks
            or (device_tensors[0].shape[0] if device_tensors else 0),
            session=session,
        )

    if params.enable_kvcache_swapping_to_host and total_num_host_blocks > 0:
        if params.disk_offload_dir is not None:
            logger.info(
                "Creating TieredConnector: "
                f"host_blocks={total_num_host_blocks}, "
                f"disk_dir={params.disk_offload_dir}, "
                f"disk_max_gb={params.disk_offload_max_gb}"
            )
            return TieredConnector(
                params=params,
                devices=devices,
                device_tensors=device_tensors,
                device_scale_tensors=device_scale_tensors,
                total_num_host_blocks=total_num_host_blocks,
                disk_cache_dir=params.disk_offload_dir,
                max_disk_size_gb=params.disk_offload_max_gb,
                use_direct_io=params.disk_offload_direct_io,
            )
        else:
            logger.info(
                f"Creating LocalConnector: host_blocks={total_num_host_blocks}"
            )
            return LocalConnector(
                params=params,
                devices=devices,
                device_tensors=device_tensors,
                device_scale_tensors=device_scale_tensors,
                total_num_host_blocks=total_num_host_blocks,
            )

    logger.info(
        "Creating NullConnector: external KV cache swapping disabled or no host blocks allocated"
    )
    return NullConnector()


__all__ = [
    "KVConnector",
    "LMCacheConnector",
    "LocalConnector",
    "NullConnector",
    "TieredConnector",
    "create_connector",
]
