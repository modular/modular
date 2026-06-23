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
- `create_connector()`: Factory function
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from max.driver import Device
from max.nn.kv_cache.cache_params import (
    KVCacheBufferInterface,
    KVCacheMemory,
    KVConnectorType,
)
from max.pipelines.kv_cache.kv_connector import KVConnector

from .local_connector import LocalConnector
from .null_connector import NullConnector
from .tiered_connector import TieredConnector

if TYPE_CHECKING:
    from max.pipelines.kv_cache.config import KVConnectorConfig

logger = logging.getLogger("max.pipelines")


def create_connector(
    kv_connector: KVConnectorType | None,
    kv_connector_config: KVConnectorConfig | None,
    devices: Sequence[Device],
    kv_buffers: KVCacheBufferInterface,
    total_num_host_blocks: int,
) -> KVConnector:
    """Create a KV cache connector instance based on ``kv_connector``.

    Args:
        kv_connector: Connector type to instantiate (or None for no-op).
        kv_connector_config: Connector-specific configuration object.
        devices: Devices for the KV cache tensors.
        kv_buffers: The replica's KV buffer (a single leaf or a tree of
            leaves) describing all caches to offload.
        total_num_host_blocks: Total number of host blocks for swapping.

    Returns:
        A connector instance implementing the KVConnector protocol.
    """
    connector = kv_connector
    kv_memory: list[KVCacheMemory] = kv_buffers.to_memory()

    if connector == KVConnectorType.dkv:
        from .dkv import DKVConnector

        if (
            kv_connector_config is None
            or not kv_connector_config.block_store_endpoint
        ):
            raise ValueError(
                "kv_connector_config must include 'block_store_endpoint' "
                "when kv_connector is 'dkv'"
            )
        logger.info(
            "Creating DKVConnector: endpoint=%s",
            kv_connector_config.block_store_endpoint,
        )
        return DKVConnector(
            devices=devices,
            kv_memory=kv_memory,
            local_block_store_endpoint=kv_connector_config.block_store_endpoint,
        )

    if connector == KVConnectorType.tiered:
        cfg = kv_connector_config
        if cfg is None or cfg.disk_offload_dir is None:
            raise ValueError(
                "kv_connector_config must include 'disk_offload_dir' "
                "when kv_connector is 'tiered'"
            )
        logger.debug(
            "Creating TieredConnector: "
            f"host_blocks={total_num_host_blocks}, "
            f"disk_dir={cfg.disk_offload_dir}, "
            f"disk_max_gb={cfg.disk_offload_max_gb}"
        )

        return TieredConnector(
            devices=devices,
            kv_memory=kv_memory,
            total_num_host_blocks=total_num_host_blocks,
            disk_cache_dir=cfg.disk_offload_dir,
            max_disk_size_gb=cfg.disk_offload_max_gb,
            use_direct_io=cfg.disk_offload_direct_io,
        )

    if connector == KVConnectorType.local:
        logger.debug(
            f"Creating LocalConnector: host_blocks={total_num_host_blocks}"
        )
        return LocalConnector(
            kv_memory=kv_memory,
            total_num_host_blocks=total_num_host_blocks,
        )

    logger.debug("Creating NullConnector: no KV cache connector configured")
    return NullConnector()


__all__ = [
    "DKVConnector",
    "KVConnector",
    "KVConnectorType",
    "LocalConnector",
    "NullConnector",
    "TieredConnector",
    "create_connector",
]
