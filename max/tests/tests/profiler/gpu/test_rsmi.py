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

from __future__ import annotations

import ctypes

from max.profiler.gpu._rsmi import (
    _RSMI_STATUS_NOT_SUPPORTED,
    _RSMI_STATUS_SUCCESS,
    RSMIContext,
)


class _NotSupportedBusyLibrary:
    """Fake ROCm-SMI library whose busy-percent query is unsupported.

    Memory total/usage succeed so the rest of the snapshot still populates,
    matching consumer Radeon cards that implement VRAM queries but not
    ``rsmi_dev_memory_busy_percent_get``.
    """

    def rsmi_status_string(self, status: int, string: object) -> int:
        return 1

    def rsmi_dev_memory_total_get(
        self, index: int, mem_type: int, total: ctypes.c_uint64
    ) -> int:
        # `_bindtools.byref` passes the ctypes value itself, not a pointer.
        total.value = 24 * 1024 * 1024 * 1024
        return _RSMI_STATUS_SUCCESS

    def rsmi_dev_memory_usage_get(
        self, index: int, mem_type: int, used: ctypes.c_uint64
    ) -> int:
        used.value = 0
        return _RSMI_STATUS_SUCCESS

    def rsmi_dev_memory_busy_percent_get(
        self, index: int, busy_percent: object
    ) -> int:
        return _RSMI_STATUS_NOT_SUPPORTED

    def rsmi_dev_busy_percent_get(
        self, index: int, busy_percent: ctypes.c_uint32
    ) -> int:
        busy_percent.value = 12
        return _RSMI_STATUS_SUCCESS

    def rsmi_dev_gpu_clk_freq_get(
        self, index: int, clk_type: int, freqs: object
    ) -> int:
        return _RSMI_STATUS_NOT_SUPPORTED


def test_memory_busy_not_supported_degrades() -> None:
    ctx = RSMIContext()
    ctx._library = _NotSupportedBusyLibrary()  # type: ignore[assignment]
    stats = ctx._get_device_stats(0)
    assert stats.memory is not None
    assert stats.memory.total_bytes == 24 * 1024 * 1024 * 1024
    assert stats.utilization.gpu_usage_percent == 12
    assert stats.utilization.memory_activity_percent is None
    assert stats.clocks is None
