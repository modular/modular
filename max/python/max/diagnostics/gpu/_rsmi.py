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

"""Interface for accessing GPU statistics of AMD GPUs."""

from __future__ import annotations

import ctypes
from types import TracebackType
from typing import Annotated, Any, Protocol, runtime_checkable

from . import _bindtools
from ._types import GPUStats, MemoryStats, UtilizationStats

_rsmi_status_t = Annotated[int, ctypes.c_int]
_rsmi_memory_type_t = Annotated[int, ctypes.c_int]
_rsmi_device_index_t = Annotated[int, ctypes.c_uint32]

_RSMI_STATUS_SUCCESS = 0

_RSMI_MEM_TYPE_VRAM = 0
_RSMI_MEM_TYPE_VIS_VRAM = 1
_RSMI_MEM_TYPE_GTT = 2


@runtime_checkable
class _RSMILibrary(Protocol):
    def rsmi_init(
        self, init_flags: Annotated[int, ctypes.c_uint64]
    ) -> _rsmi_status_t: ...
    def rsmi_shut_down(self) -> None: ...
    def rsmi_num_monitor_devices(
        self, num_devices: ctypes._Pointer[ctypes.c_uint32]
    ) -> _rsmi_status_t: ...
    def rsmi_dev_memory_total_get(
        self,
        device_index: _rsmi_device_index_t,
        mem_type: _rsmi_memory_type_t,
        total: ctypes._Pointer[ctypes.c_uint64],
    ) -> _rsmi_status_t: ...
    def rsmi_dev_memory_usage_get(
        self,
        device_index: _rsmi_device_index_t,
        mem_type: _rsmi_memory_type_t,
        total: ctypes._Pointer[ctypes.c_uint64],
    ) -> _rsmi_status_t: ...
    def rsmi_dev_memory_busy_percent_get(
        self,
        device_index: _rsmi_device_index_t,
        busy_percent: ctypes._Pointer[ctypes.c_uint32],
    ) -> _rsmi_status_t: ...
    def rsmi_dev_busy_percent_get(
        self,
        device_index: _rsmi_device_index_t,
        busy_percent: ctypes._Pointer[ctypes.c_uint32],
    ) -> _rsmi_status_t: ...
    def rsmi_status_string(
        self, status: _rsmi_status_t, string: ctypes._Pointer[ctypes.c_char_p]
    ) -> _rsmi_status_t: ...


class RSMIError(Exception):
    def __init__(self, code: _rsmi_status_t, message: str, /) -> None:
        super().__init__(message)
        self.code = code


def _check_rsmi_status(library: _RSMILibrary, status: _rsmi_status_t) -> None:
    if status != _RSMI_STATUS_SUCCESS:
        error_ptr = ctypes.c_char_p()
        if (
            library.rsmi_status_string(status, _bindtools.byref(error_ptr))
            != _RSMI_STATUS_SUCCESS
            or (error_bytes := error_ptr.value) is None
        ):
            error_string = "(Unknown)"
        else:
            error_string = error_bytes.decode()
        raise RSMIError(status, error_string)


class RSMIContext:
    """Context for accessing ROCm-SMI and accessing GPU information."""

    def __init__(self) -> None:
        self._library: _RSMILibrary | None = None

    def __enter__(self) -> RSMIContext:
        if self._library is not None:
            raise AssertionError("Context already active")
        cdll = ctypes.CDLL("librocm_smi64.so")
        lib = _bindtools.bind_protocol(cdll, _RSMILibrary)
        _check_rsmi_status(lib, lib.rsmi_init(0))
        self._library = lib
        return self

    def __exit__(
        self,
        exc_type: type[Any] | None,
        exc_value: Any,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._library is not None:
            self._library.rsmi_shut_down()
        self._library = None

    def _get_count(self) -> int:
        if self._library is None:
            return 0
        count = ctypes.c_uint32()
        _check_rsmi_status(
            self._library,
            self._library.rsmi_num_monitor_devices(_bindtools.byref(count)),
        )
        return count.value

    def _get_memory_total(self, index: int) -> int:
        assert self._library is not None
        result = ctypes.c_uint64()
        _check_rsmi_status(
            self._library,
            self._library.rsmi_dev_memory_total_get(
                index, _RSMI_MEM_TYPE_VRAM, _bindtools.byref(result)
            ),
        )
        return result.value

    def _get_memory_usage(self, index: int) -> int:
        assert self._library is not None
        result = ctypes.c_uint64()
        _check_rsmi_status(
            self._library,
            self._library.rsmi_dev_memory_usage_get(
                index, _RSMI_MEM_TYPE_VRAM, _bindtools.byref(result)
            ),
        )
        return result.value

    def _get_memory_busy_percent(self, index: int) -> int:
        assert self._library is not None
        result = ctypes.c_uint32()
        _check_rsmi_status(
            self._library,
            self._library.rsmi_dev_memory_busy_percent_get(
                index, _bindtools.byref(result)
            ),
        )
        return result.value

    def _get_device_busy_percent(self, index: int) -> int:
        assert self._library is not None
        result = ctypes.c_uint32()
        _check_rsmi_status(
            self._library,
            self._library.rsmi_dev_busy_percent_get(
                index, _bindtools.byref(result)
            ),
        )
        return result.value

    def _get_device_stats(self, index: int) -> GPUStats:
        mem_total = self._get_memory_total(index)
        mem_used = self._get_memory_usage(index)
        mem_busy_pct = self._get_memory_busy_percent(index)
        dev_busy_pct = self._get_device_busy_percent(index)
        return GPUStats(
            memory=MemoryStats(
                total_bytes=mem_total,
                free_bytes=mem_total - mem_used,
                used_bytes=mem_used,
                reserved_bytes=None,
            ),
            utilization=UtilizationStats(
                gpu_usage_percent=dev_busy_pct,
                memory_activity_percent=mem_busy_pct,
            ),
        )

    def get_stats(self) -> dict[int, GPUStats]:
        """Get GPU statistics for all GPUs."""
        stats: dict[int, GPUStats] = {}
        count = self._get_count()
        for i in range(count):
            stats[i] = self._get_device_stats(i)
        return stats
