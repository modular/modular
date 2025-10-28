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

"""Public types returned by the GPU diagnostics API."""

from __future__ import annotations

import msgspec


class GPUStats(msgspec.Struct):
    """Comprehensive GPU state snapshot containing memory and utilization metrics.

    This class provides a complete view of a GPU's current state, including
    detailed memory usage statistics and utilization percentages. It serves
    as the primary data structure returned by GPU diagnostic queries.
    """

    """Detailed memory usage statistics for the GPU."""
    memory: MemoryStats
    """Current GPU compute and memory utilization percentages."""
    utilization: UtilizationStats


class MemoryStats(msgspec.Struct):
    """Detailed GPU memory usage statistics including total, free, used, and reserved memory.

    This class provides comprehensive memory information for a GPU, allowing
    developers to monitor memory consumption and identify potential memory
    bottlenecks during model inference or training.
    """

    """Total GPU memory capacity in bytes."""
    total_bytes: int
    """Currently available GPU memory in bytes."""
    free_bytes: int
    """Currently allocated GPU memory in bytes."""
    used_bytes: int
    """Memory reserved by the driver, if available from the GPU vendor."""
    reserved_bytes: int | None


class UtilizationStats(msgspec.Struct):
    """GPU compute and memory activity utilization percentages.

    This class captures the current utilization levels of a GPU's compute
    units and memory subsystem, providing insights into how effectively
    the GPU resources are being utilized during workload execution.
    """

    gpu_usage_percent: int
    """Current GPU compute utilization as a percentage (0-100)."""

    memory_activity_percent: int | None
    """Memory controller activity percentage, if available from the GPU vendor."""
