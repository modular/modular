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

"""Host-capacity preflight shared by the KV cache host/disk offload tiers."""

from __future__ import annotations

import logging

import psutil

_logger = logging.getLogger("max.pipelines")

_GIB = 1024**3


def _check_host_memory_capacity(requested_bytes: int) -> None:
    """Raises when a pinned host allocation exceeds host availability."""
    try:
        available_bytes = psutil.virtual_memory().available
    except (OSError, RuntimeError) as error:
        _logger.warning(
            "Unable to determine available host memory; skipping KV cache "
            "host capacity preflight: %s",
            error,
        )
        return
    if requested_bytes > available_bytes:
        raise RuntimeError(
            "KV cache host offload buffer requires "
            f"{requested_bytes / _GIB:.1f} GiB of pinned host memory but only "
            f"{available_bytes / _GIB:.1f} GiB is available. Reduce "
            "host_offload_max_gb or provision more host memory."
        )
