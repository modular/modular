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
    """A snapshot-in-time representation of a GPU's state."""

    memory: MemoryStats
    utilization: UtilizationStats


class MemoryStats(msgspec.Struct):
    """A snapshot-in-time representation of a GPU's memory usage."""

    total_bytes: int
    free_bytes: int
    used_bytes: int
    reserved_bytes: int | None


class UtilizationStats(msgspec.Struct):
    """A snapshot-in-time representation of the utilization of a GPU."""

    gpu_usage_percent: int
    memory_activity_percent: int | None
