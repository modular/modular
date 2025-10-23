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

"""Interface for accessing GPU statistics of NVIDIA GPUs."""

from __future__ import annotations

import ctypes
from types import TracebackType
from typing import Annotated, Any, Protocol, runtime_checkable

from . import _bindtools
from ._types import GPUStats, MemoryStats, UtilizationStats

_nvmlReturn_t = Annotated[int, ctypes.c_int]
_nvmlDevice_t = ctypes.c_void_p

_NVML_SUCCESS = 0
_NVML_ERROR_NO_PERMISSION = 4
_NVML_ERROR_GPU_NOT_FOUND = 28


class _nvmlMemory_v2_t(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint),
        ("total", ctypes.c_ulonglong),
        ("reserved", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]

    version: int
    total: int
    reserved: int
    free: int
    used: int


class _nvmlUtilization_t(ctypes.Structure):
    _fields_ = [
        ("gpu", ctypes.c_uint),
        ("memory", ctypes.c_uint),
    ]

    gpu: int
    memory: int


@runtime_checkable
class _NVMLLibrary(Protocol):
    def nvmlInit_v2(self) -> _nvmlReturn_t: ...
    def nvmlShutdown(self) -> _nvmlReturn_t: ...
    def nvmlErrorString(
        self, result: _nvmlReturn_t
    ) -> Annotated[bytes, ctypes.c_char_p]: ...
    def nvmlDeviceGetCount_v2(
        self, device_count: ctypes._Pointer[ctypes.c_uint]
    ) -> _nvmlReturn_t: ...
    def nvmlDeviceGetHandleByIndex_v2(
        self, index: ctypes.c_uint, device: ctypes._Pointer[_nvmlDevice_t]
    ) -> _nvmlReturn_t: ...
    def nvmlDeviceGetMemoryInfo_v2(
        self, device: _nvmlDevice_t, memory: ctypes._Pointer[_nvmlMemory_v2_t]
    ) -> _nvmlReturn_t: ...
    def nvmlDeviceGetUtilizationRates(
        self,
        device: _nvmlDevice_t,
        utilization: ctypes._Pointer[_nvmlUtilization_t],
    ) -> _nvmlReturn_t: ...


class NVMLError(Exception):
    def __init__(self, code: _nvmlReturn_t, message: str, /) -> None:
        super().__init__(message)
        self.code = code


class NoPermissionError(NVMLError):
    pass


class GPUNotFoundError(NVMLError):
    pass


_SPECIFIC_ERROR_TYPES: dict[int, type[NVMLError]] = {
    _NVML_ERROR_NO_PERMISSION: NoPermissionError,
    _NVML_ERROR_GPU_NOT_FOUND: GPUNotFoundError,
}


def _check_nvml_return(library: _NVMLLibrary, result: _nvmlReturn_t) -> None:
    if result != _NVML_SUCCESS:
        cls = _SPECIFIC_ERROR_TYPES.get(result, NVMLError)
        error_bytes = library.nvmlErrorString(result)
        if error_bytes is None:
            error_string = "(Unknown)"
        else:
            error_string = error_bytes.decode()
        raise cls(result, error_string)


class NVMLContext:
    """Context for accessing NVML and accessing GPU information."""

    def __init__(self) -> None:
        self._library: _NVMLLibrary | None = None

    def __enter__(self) -> NVMLContext:
        if self._library is not None:
            raise AssertionError("Context already active")
        cdll = ctypes.CDLL("libnvidia-ml.so.1")
        lib = _bindtools.bind_protocol(cdll, _NVMLLibrary)
        _check_nvml_return(lib, lib.nvmlInit_v2())
        self._library = lib
        return self

    def __exit__(
        self,
        exc_type: type[Any] | None,
        exc_value: Any,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._library is not None:
            _check_nvml_return(self._library, self._library.nvmlShutdown())
        self._library = None

    def _get_count(self) -> int:
        if self._library is None:
            return 0
        count = ctypes.c_uint()
        _check_nvml_return(
            self._library,
            self._library.nvmlDeviceGetCount_v2(_bindtools.byref(count)),
        )
        return count.value

    def _get_device(self, index: int) -> _nvmlDevice_t:
        if self._library is None:
            raise IndexError("All indices are invalid when uninitialized")
        device = _nvmlDevice_t()
        _check_nvml_return(
            self._library,
            self._library.nvmlDeviceGetHandleByIndex_v2(
                ctypes.c_uint(index), _bindtools.byref(device)
            ),
        )
        return device

    def _get_device_stats(self, device: _nvmlDevice_t) -> GPUStats:
        assert self._library is not None
        mem = _nvmlMemory_v2_t()
        # Documentation says version must be 2, but the required value is
        # actually more complicated -- see
        # https://github.com/NVIDIA/nvidia-settings/issues/78#issuecomment-1012837988
        mem.version = ctypes.sizeof(_nvmlMemory_v2_t) | (2 << 24)
        _check_nvml_return(
            self._library,
            self._library.nvmlDeviceGetMemoryInfo_v2(
                device, _bindtools.byref(mem)
            ),
        )
        util = _nvmlUtilization_t()
        _check_nvml_return(
            self._library,
            self._library.nvmlDeviceGetUtilizationRates(
                device, _bindtools.byref(util)
            ),
        )
        return GPUStats(
            memory=MemoryStats(
                total_bytes=mem.total,
                free_bytes=mem.free,
                used_bytes=mem.used,
                reserved_bytes=mem.reserved,
            ),
            utilization=UtilizationStats(
                gpu_usage_percent=util.gpu,
                memory_activity_percent=util.memory,
            ),
        )

    def get_stats(self) -> dict[int, GPUStats]:
        """Get GPU statistics for all GPUs."""
        stats: dict[int, GPUStats] = {}
        count = self._get_count()
        for i in range(count):
            try:
                device = self._get_device(i)
            except NoPermissionError:
                continue
            stats[i] = self._get_device_stats(device)
        return stats
