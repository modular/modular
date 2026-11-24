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
"""Tests the driver C API handle a `Buffer` hands out."""

import ctypes

import max._core
import numpy as np
import pytest
from max._core.driver import _release_buffers_to_borrowed
from max.driver import CPU, Buffer
from max.dtype import DType

M_DRIVER_OK = 0


def _driver_capi() -> ctypes.CDLL:
    """Opens the driver C API the bindings already loaded.

    The handle comes from the same library, so it is only valid with these
    entry points, never with another copy of the driver.
    """
    capi = ctypes.CDLL(max._core.__file__)
    capi.M_DriverBufferPointer.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    capi.M_DriverBufferPointer.restype = ctypes.c_int
    capi.M_DriverBufferByteSize.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    capi.M_DriverBufferByteSize.restype = ctypes.c_int
    capi.M_DriverBufferDestroy.argtypes = [ctypes.c_void_p]
    capi.M_DriverBufferDestroy.restype = ctypes.c_int
    return capi


def _describe(capi: ctypes.CDLL, handle: int) -> tuple[int, int]:
    pointer = ctypes.c_void_p()
    size = ctypes.c_uint64()
    assert capi.M_DriverBufferPointer(handle, ctypes.byref(pointer)) == (
        M_DRIVER_OK
    )
    assert capi.M_DriverBufferByteSize(handle, ctypes.byref(size)) == (
        M_DRIVER_OK
    )
    assert pointer.value is not None
    return pointer.value, size.value


def test_handle_names_the_buffers_memory() -> None:
    capi = _driver_capi()
    buffer = Buffer.zeros((16, 32), DType.float32, device=CPU())

    handle = buffer._driver_buffer_handle()
    try:
        assert _describe(capi, handle) == (buffer._data_ptr(), 16 * 32 * 4)
    finally:
        assert capi.M_DriverBufferDestroy(handle) == M_DRIVER_OK


def test_handle_keeps_the_memory_after_the_buffer_is_gone() -> None:
    capi = _driver_capi()
    # A deep copy owns its memory, so only the handle keeps it alive below.
    buffer = Buffer.from_numpy(np.arange(64, dtype=np.uint8)).copy()
    # The copy runs on the CPU stream, and the raw read below does not wait.
    buffer.device.synchronize()
    address = buffer._data_ptr()

    handle = buffer._driver_buffer_handle()
    del buffer
    try:
        pointer, size = _describe(capi, handle)
        assert (pointer, size) == (address, 64)
        assert bytes((ctypes.c_uint8 * 64).from_address(pointer)) == bytes(
            range(64)
        )
    finally:
        assert capi.M_DriverBufferDestroy(handle) == M_DRIVER_OK


def test_a_strided_buffer_has_no_handle() -> None:
    buffer = Buffer.zeros((4, 4), DType.uint8, device=CPU())

    with pytest.raises(ValueError, match="strided"):
        buffer[:, ::2]._driver_buffer_handle()


@pytest.mark.parametrize("start,stop", [(1, 4), (0, 3)])
def test_a_sub_byte_slice_that_splits_a_byte_has_no_handle(
    start: int, stop: int
) -> None:
    buffer = Buffer.zeros((8,), DType.float4_e2m1fn, device=CPU())

    with pytest.raises(ValueError, match="mid-byte"):
        buffer[start:stop]._driver_buffer_handle()


def test_a_borrowed_buffer_has_no_handle() -> None:
    owner = Buffer.zeros((8,), DType.uint8, device=CPU())
    (borrowed,) = _release_buffers_to_borrowed([owner])

    with pytest.raises(ValueError, match="borrowed"):
        borrowed._driver_buffer_handle()
    with pytest.raises(ValueError, match="borrowed"):
        borrowed[2:6]._driver_buffer_handle()
