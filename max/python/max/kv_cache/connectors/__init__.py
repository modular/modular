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
- `create_connector()`: Factory function
"""

from __future__ import annotations

from collections.abc import Sequence

from max.driver import Buffer, Device
from max.engine import InferenceSession
from max.kv_cache.kv_connector import KVConnector
from max.nn.legacy.kv_cache import KVCacheParams

from .null_connector import NullConnector


def create_connector(
    params: KVCacheParams,
    devices: Sequence[Device],
    device_tensors: list[Buffer],
    device_scale_tensors: list[Buffer] | None,
    total_num_host_blocks: int,
    session: InferenceSession | None = None,
) -> KVConnector:
    """Create a KV cache connector instance.

    Returns a connector appropriate for the configuration:
    - If `params.enable_kvcache_swapping_to_host` is True: LocalConnector
    - Otherwise: NullConnector

    Args:
        params: KV cache parameters containing configuration.
        devices: Devices for the KV cache tensors.
        device_tensors: Device tensors for KV cache (owned by manager).
        device_scale_tensors: Device scale tensors for FP8, or None.
        total_num_host_blocks: Total number of host blocks for swapping.
        session: Optional inference session for loading custom kernels.

    Returns:
        A connector instance implementing KVConnectorProtocol.
    """
    if params.enable_kvcache_swapping_to_host and total_num_host_blocks > 0:
        from .local_connector import LocalConnector

        return LocalConnector(
            params=params,
            devices=devices,
            device_tensors=device_tensors,
            device_scale_tensors=device_scale_tensors,
            total_num_host_blocks=total_num_host_blocks,
        )

    return NullConnector()


__all__ = [
    "KVConnector",
    "NullConnector",
    "create_connector",
]
