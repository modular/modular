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
- `RustTierConnector`: GPU <-> CPU <-> Disk offloading, backed by the Rust
  ``kv_tier_connector`` extension. Also serves the ``tiered`` alias, whose
  Python implementation it replaced.
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
    KVCacheMemory,
    KVCacheParamInterface,
    KVConnectorType,
)
from max.pipelines.kv_cache.kv_connector import KVConnector

from .null_connector import NullConnector

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
    back this with one shared pinned host buffer / disk cache; the distributed
    ``dkv`` connector owns one Rust client per replica internally.

    ``tiered`` is a backward-compatible alias for the Rust ``rust_tiered``
    connector, which replaced its deleted Python implementation, and therefore
    inherits ``rust_tiered``'s CUDA/HIP requirement.

    Args:
        kv_connector: Connector type to instantiate (or None for no-op).
        kv_connector_config: Connector-specific configuration object.
        devices: Devices for the KV cache tensors (all participating devices).
        replica_kv_memory: Per-replica offload-ready KV memory units (one inner
            sequence per DP replica).
        total_num_host_blocks: Total number of host blocks for swapping (the
            full shared pool across replicas for ``tiered``).
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

    # ``tiered`` is a backward-compatible alias for ``rust_tiered``, kept after
    # its Python implementation was deleted.
    if connector in (KVConnectorType.tiered, KVConnectorType.rust_tiered):
        cfg = kv_connector_config
        # Check the KV memory's own device before the build's accelerator API,
        # so a CPU-device pipeline fails the same way on every host rather than
        # reporting "no CUDA/HIP" only on GPU-less ones.
        if (
            replica_kv_memory
            and replica_kv_memory[0][0].buffers[0].device.is_host
        ):
            raise ValueError("KVCacheMemory is on the CPU; cannot offload")
        # The Rust connector drives the GPU copy engines directly via its own
        # dlopen'd driver shim, supporting NVIDIA (CUDA) and AMD (HIP) but not
        # Metal/CPU.
        api = accelerator_api()
        if api not in ("cuda", "hip"):
            raise ValueError(
                f"kv_connector '{connector.value}' requires a CUDA or HIP GPU, "
                f"found incompatible accelerator API: '{api}'."
            )
        if cfg is None:
            raise ValueError(
                "kv_connector_config is required when kv_connector is "
                f"'{connector.value}'"
            )
        from .rust_tier_connector import RustTierConnector

        disk_dir = _resolve_disk_offload_dir(cfg)
        if connector != KVConnectorType.rust_tiered:
            logger.warning(
                "kv_connector '%s' is deprecated: its Python implementation "
                "was removed and it now runs the Rust 'rust_tiered' connector. "
                "Pass '--kv-connector rust_tiered' instead.",
                connector.value,
            )
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

    logger.debug("Creating NullConnector: no KV cache connector configured")
    return NullConnector()


__all__ = [
    "KVConnector",
    "KVConnectorType",
    "NullConnector",
    "create_connector",
]
