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

import msgspec
from max.profiler.gpu import (
    HARDWARE_THROTTLE_REASONS,
    ClockStats,
    GPUStats,
    MemoryStats,
    UtilizationStats,
)


def test_legacy_record_decodes_with_no_clocks() -> None:
    # Records serialized before `clocks` was added must still decode, with
    # `clocks` defaulting to None.
    legacy = (
        b'{"memory":{"total_bytes":1,"free_bytes":1,"used_bytes":0,'
        b'"reserved_bytes":null},'
        b'"utilization":{"gpu_usage_percent":0,"memory_activity_percent":null}}'
    )
    decoded = msgspec.json.decode(legacy, type=GPUStats)
    assert decoded.clocks is None
    assert decoded.memory.total_bytes == 1


def test_clock_stats_round_trip() -> None:
    stats = GPUStats(
        memory=MemoryStats(
            total_bytes=1, free_bytes=1, used_bytes=0, reserved_bytes=None
        ),
        utilization=UtilizationStats(
            gpu_usage_percent=10, memory_activity_percent=None
        ),
        clocks=ClockStats(
            core_mhz=1500,
            core_max_mhz=2100,
            mem_mhz=2000,
            mem_max_mhz=2500,
            throttle_reasons=["gpu_idle", "hw_thermal_slowdown"],
        ),
    )
    decoded = msgspec.json.decode(msgspec.json.encode(stats), type=GPUStats)
    assert decoded.clocks is not None
    assert decoded.clocks.core_mhz == 1500
    assert decoded.clocks.mem_max_mhz == 2500
    assert decoded.clocks.throttle_reasons == [
        "gpu_idle",
        "hw_thermal_slowdown",
    ]


def test_clock_stats_optional_memory() -> None:
    # Memory clock fields can be None when the vendor library doesn't
    # report them (e.g. ROCm-SMI failing on the MEM domain).
    stats = ClockStats(
        core_mhz=1500,
        core_max_mhz=2100,
        mem_mhz=None,
        mem_max_mhz=None,
        throttle_reasons=None,
    )
    decoded = msgspec.json.decode(msgspec.json.encode(stats), type=ClockStats)
    assert decoded.mem_mhz is None
    assert decoded.mem_max_mhz is None
    assert decoded.throttle_reasons is None


def test_clock_stats_empty_throttle_reasons_round_trip() -> None:
    # Empty list means "API works, no reasons reported" -- distinct from
    # None ("API unavailable"). Make sure the distinction survives
    # serialization.
    stats = ClockStats(
        core_mhz=1500,
        core_max_mhz=2100,
        mem_mhz=None,
        mem_max_mhz=None,
        throttle_reasons=[],
    )
    decoded = msgspec.json.decode(msgspec.json.encode(stats), type=ClockStats)
    assert decoded.throttle_reasons == []


def test_hardware_throttle_reasons_excludes_gpu_idle() -> None:
    # "gpu_idle" is reported whenever the GPU is idle and is NOT a real
    # throttle; consumers must filter it out.
    assert "gpu_idle" not in HARDWARE_THROTTLE_REASONS


def test_decode_throttle_bits() -> None:
    # Importing _nvml is safe without libnvidia-ml because the CDLL load
    # only happens inside NVMLContext.__enter__.
    from max.profiler.gpu._nvml import _decode_throttle_bits

    assert _decode_throttle_bits(0x0) == []
    assert _decode_throttle_bits(0x1) == ["gpu_idle"]
    # GPU idle (0x1) | HW thermal slowdown (0x40) -- order matches the bit
    # table, not the input.
    assert _decode_throttle_bits(0x41) == ["gpu_idle", "hw_thermal_slowdown"]
    # Unknown bits are dropped silently.
    assert _decode_throttle_bits(0x1 | (1 << 40)) == ["gpu_idle"]
