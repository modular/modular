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

from max.driver import Buffer, Device
from max.kv_cache.kv_connector import KVConnector
from max.nn.kv_cache import KVCacheParams
from max.nn.kv_cache.cache_params import KVConnectorType

from .local_connector import LocalConnector
from .null_connector import NullConnector
from .tiered_connector import TieredConnector

logger = logging.getLogger("max.pipelines")


def create_connector(
    params: KVCacheParams,
    devices: Sequence[Device],
    device_buffers: list[Buffer],
    total_num_host_blocks: int,
    total_num_blocks: int,
) -> KVConnector:
    """Create a KV cache connector instance based on ``params.kv_connector``.

    Args:
        params: KV cache parameters containing configuration.
        devices: Devices for the KV cache tensors.
        device_buffer: Device buffer for KV cache (owned by manager).
        total_num_host_blocks: Total number of host blocks for swapping.
        total_num_blocks: Total number of device blocks.
        session: Optional inference session for loading custom kernels.

    Returns:
        A connector instance implementing KVConnectorProtocol.
    """
    connector = params.kv_connector

    if connector == KVConnectorType.dkv:
        from .dkv import DKVConnector

        cfg = params.kv_connector_config
        if cfg is None or not getattr(cfg, "block_store_endpoint", None):
            raise ValueError(
                "kv_connector_config must include 'block_store_endpoint' "
                "when kv_connector is 'dkv'"
            )
        logger.info(
            "Creating DKVConnector: endpoint=%s",
            cfg.block_store_endpoint,
        )
        # DKVConnector is temporarily disabled. We need to implement proper support
        # later down the line. For now, we raise an error.
        # Note that mantaining backward compatibility is hard since we are
        # changing the core KVConnector protocol in order to better support
        # the other connectors.
        raise NotImplementedError("DKVConnector is not implemented")
        # return DKVConnector(
        #     params=params,
        #     devices=devices,
        #     device_buffers=device_buffers,
        #     total_num_blocks=total_num_blocks,
        #     local_block_store_endpoint=cfg.block_store_endpoint,
        # )

    if connector == KVConnectorType.tiered:
        cfg = params.kv_connector_config
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
            params=params,
            devices=devices,
            device_buffers=device_buffers,
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
            params=params,
            device_buffers=device_buffers,
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
