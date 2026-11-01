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

from max.profiler.gpu._nvml import NVMLContext


class _NotSupportedLibrary:
    """Fake NVML library whose memory query fails like a GB10's driver."""

    def nvmlErrorString(self, result: int) -> bytes:
        return b"Not Supported"

    def nvmlDeviceGetMemoryInfo_v2(self, device: object, mem: object) -> int:
        return 3  # NVML_ERROR_NOT_SUPPORTED

    def nvmlDeviceGetUtilizationRates(
        self, device: object, util: object
    ) -> int:
        return 0  # NVML_SUCCESS


def test_memory_info_not_supported_degrades() -> None:
    ctx = NVMLContext()
    ctx._library = _NotSupportedLibrary()  # type: ignore[assignment]
    ctx._clock_library = None
    stats = ctx._get_device_stats(ctypes.c_void_p(0))
    assert stats.memory is None
    assert stats.utilization.gpu_usage_percent == 0
    assert stats.clocks is None
