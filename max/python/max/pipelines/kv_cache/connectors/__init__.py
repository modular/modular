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
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from max.driver import Device, accelerator_api
from max.nn.kv_cache.cache_params import (
    KVCacheBufferInterface,
    KVCacheMemory,
    KVCacheParamInterface,
    KVConnectorType,
)
from max.pipelines.kv_cache.kv_connector import KVConnector

from .local_connector import LocalConnector
from .null_connector import NullConnector
from .tiered_connector import TieredConnector

if TYPE_CHECKING:
    from max.pipelines.kv_cache.config import KVConnectorConfig

logger = logging.getLogger("max.pipelines")

# Prefix for auto-created tiered-connector disk offload directories. Owned by
# the connectors package (which creates, warns about, and cleans up these
# dirs); the pipeline config imports it only to name the mkdtemp it creates.
KV_OFFLOAD_DIR_PREFIX = "max_kv_tiered_"


def warn_stale_offload_dirs(offload_dir: str) -> None:
    """Warns about leftover KV cache offload directories from previous runs.

    The tiered connectors delete their own offload directory on graceful
    shutdown, but a forceful shutdown (SIGKILL, OOM-kill, or a crash) skips
    that cleanup and leaves the directory (and its cached blocks) on disk.
    Scan the sibling directory for such leftovers and warn so operators can
    reclaim the space.

    Args:
        offload_dir: The offload directory this run will use. Its siblings
            matching ``{KV_OFFLOAD_DIR_PREFIX}*`` are treated as leftovers.
    """
    parent = Path(offload_dir).parent
    try:
        stale = sorted(
            str(p)
            for p in parent.glob(f"{KV_OFFLOAD_DIR_PREFIX}*")
            if p.is_dir() and str(p) != offload_dir
        )
    except OSError:
        return
    if not stale:
        return
    logger.warning(
        "Found %d leftover KV cache offload director%s from a previous run "
        "in %s:\n  %s\n"
        "MAX Serve deletes its offload directory on graceful shutdown, but a "
        "forceful shutdown (SIGKILL / OOM-kill) leaves it behind. If no MAX "
        "Serve process is currently using them, delete these directories to "
        "reclaim disk space.",
        len(stale),
        "y" if len(stale) == 1 else "ies",
        parent,
        "\n  ".join(stale),
    )


def _resolve_disk_offload_dir(cfg: KVConnectorConfig) -> str:
    """Returns the disk offload dir, auto-creating one if unset.

    A single connector serves every DP replica, so the directory is created
    once here (not per replica). Warns about leftovers from previous runs.
    """
    if cfg.disk_offload_dir is None:
        cfg.disk_offload_dir = tempfile.mkdtemp(prefix=KV_OFFLOAD_DIR_PREFIX)
        logger.info(
            "Tiered connector: auto-created disk offload dir %s",
            cfg.disk_offload_dir,
        )
    warn_stale_offload_dirs(cfg.disk_offload_dir)
    return cfg.disk_offload_dir


def create_connector(
    kv_connector: KVConnectorType | None,
    kv_connector_config: KVConnectorConfig | None,
    devices: Sequence[Device],
    replica_kv_memory: Sequence[Sequence[KVCacheMemory]],
    total_num_host_blocks: int,
    params: KVCacheParamInterface,
) -> KVConnector:
    """Create a KV cache connector instance based on ``kv_connector``.

    A single connector serves every DP replica for all connector types:
    ``replica_kv_memory`` holds each replica's device buffers, and load/offload
    select the replica via ``replica_idx`` (SERVOPT-1501). The host/disk tiers
    (``local``/``tiered``) back this with one shared pinned host buffer / disk
    cache; the distributed ``dkv`` connector owns one Rust client per replica
    internally.

    Args:
        kv_connector: Connector type to instantiate (or None for no-op).
        kv_connector_config: Connector-specific configuration object.
        devices: Devices for the KV cache tensors (all participating devices).
        replica_kv_memory: Per-replica offload-ready KV memory units (one inner
            sequence per DP replica).
        total_num_host_blocks: Total number of host blocks for swapping (the
            full shared pool across replicas for ``local``/``tiered``).
        params: KV-cache parameters; the ``dkv`` connector uses them to derive
            its multi-tenant per-GPU handshake identity.

    Returns:
        A connector instance implementing the KVConnector protocol.
    """
    connector = kv_connector

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
            replica_kv_memory=replica_kv_memory,
            local_block_store_endpoint=kv_connector_config.block_store_endpoint,
            devices=devices,
            params=params,
        )

    if connector == KVConnectorType.tiered:
        cfg = kv_connector_config
        if cfg is None:
            raise ValueError(
                "kv_connector_config is required when kv_connector is 'tiered'"
            )
        disk_dir = _resolve_disk_offload_dir(cfg)
        logger.debug(
            "Creating TieredConnector: "
            f"host_blocks={total_num_host_blocks}, "
            f"disk_dir={disk_dir}, "
            f"disk_max_gb={cfg.disk_offload_max_gb}, "
            f"num_disk_workers={cfg.num_disk_workers}"
        )

        return TieredConnector(
            devices=devices,
            replica_kv_memory=replica_kv_memory,
            total_num_host_blocks=total_num_host_blocks,
            disk_cache_dir=disk_dir,
            max_disk_size_gb=cfg.disk_offload_max_gb,
            num_disk_workers=cfg.num_disk_workers,
        )

    if connector == KVConnectorType.rust_tiered:
        cfg = kv_connector_config
        if cfg is None:
            raise ValueError(
                "kv_connector_config is required when kv_connector is "
                "'rust_tiered'"
            )
        # The Rust connector drives the GPU copy engines directly via its own
        # dlopen'd driver shim, supporting NVIDIA (CUDA) and AMD (HIP) but not
        # Metal/CPU.
        api = accelerator_api()
        if api not in ("cuda", "hip"):
            raise ValueError(
                f"kv_connector 'rust_tiered' requires a CUDA or HIP GPU, found "
                f"incompatible accelerator API: '{api}'."
            )
        from .rust_tier_connector import RustTierConnector

        disk_dir = _resolve_disk_offload_dir(cfg)
        logger.debug(
            "Creating RustTierConnector: "
            f"host_blocks={total_num_host_blocks}, "
            f"disk_dir={disk_dir}, "
            f"disk_max_gb={cfg.disk_offload_max_gb}, "
            f"num_disk_workers={cfg.num_disk_workers}"
        )
        return RustTierConnector(
            replica_kv_memory=replica_kv_memory,
            total_num_host_blocks=total_num_host_blocks,
            kv_hash_algo=params.kv_hash_algo,
            disk_cache_dir=disk_dir,
            max_disk_size_gb=cfg.disk_offload_max_gb,
            num_disk_workers=cfg.num_disk_workers,
        )

    if connector == KVConnectorType.local:
        logger.debug(
            f"Creating LocalConnector: host_blocks={total_num_host_blocks}"
        )
        return LocalConnector(
            replica_kv_memory=replica_kv_memory,
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
