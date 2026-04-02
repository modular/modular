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

from std.sys import (
    CompilationTarget,
    has_nvidia_gpu_accelerator,
    has_amd_gpu_accelerator,
    size_of,
)
from std.ffi import external_call

from std.gpu.host import DeviceContext, HostBuffer
from std.gpu.host.device_context import _checked, _CString, _DeviceContextPtr

from .shmem_api import shmem_free, shmem_malloc
from std.builtin.device_passable import DevicePassable


struct SHMEMBuffer[dtype: DType](DevicePassable, Sized):
    var _data: UnsafePointer[Scalar[Self.dtype], MutExternalOrigin]
    var _ctx_ptr: _DeviceContextPtr[mut=True]
    var _size: Int

    comptime device_type: AnyType = UnsafePointer[
        Scalar[Self.dtype], MutAnyOrigin
    ]

    def _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self._data

    @staticmethod
    def get_type_name() -> String:
        return String(t"SHMEMBuffer[{Self.dtype}]")

    @doc_hidden
    @always_inline
    def __init__(
        out self,
        ctx: DeviceContext,
        size: Int,
    ) raises:
        comptime if has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator():
            self._data = shmem_malloc[Self.dtype](UInt(size))
            self._ctx_ptr = ctx._handle
            self._size = size
        else:
            CompilationTarget.unsupported_target_error[
                operation="SHMEMBuffer.__init__",
            ]()

    @doc_hidden
    @always_inline
    def __init__(
        out self,
        ctx: DeviceContext,
        data: UnsafePointer[Scalar[Self.dtype], MutExternalOrigin],
        size: Int,
    ):
        self._data = data
        self._ctx_ptr = ctx._handle
        self._size = size

    def __del__(deinit self):
        shmem_free(self._data)

    def __len__(self) -> Int:
        return self._size

    def unsafe_ptr(self) -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]:
        return self._data

    def enqueue_copy_to(
        self, dst_ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]
    ) raises:
        """Enqueues an asynchronous copy from this buffer to host memory.

        This method schedules a memory copy operation from this device buffer to the
        specified host memory location. The operation is asynchronous and will be
        executed in the stream associated with this buffer's context.

        Args:
            dst_ptr: Pointer to the destination host memory location.
        """
        _checked(
            external_call[
                "AsyncRT_DeviceContext_DtoH_async_sized",
                _CString[],
                _DeviceContextPtr[mut=True],
                UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
                UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
                Int,
            ](
                self._ctx_ptr,
                dst_ptr,
                self._data,
                self._size * size_of[Self.dtype](),
            )
        )

    def enqueue_copy_to(self, dst: HostBuffer[Self.dtype]) raises:
        """Enqueues an asynchronous copy from this buffer to host memory.

        This method schedules a memory copy operation from this device buffer to the
        specified host memory location. The operation is asynchronous and will be
        executed in the stream associated with this buffer's context.

        Args:
            dst: Host buffer to copy to.

        Raises:
            If the copy operation fails.
        """
        _checked(
            external_call[
                "AsyncRT_DeviceContext_DtoH_async_sized",
                _CString[],
                _DeviceContextPtr[mut=True],
                UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
                UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
                Int,
            ](
                self._ctx_ptr,
                dst.unsafe_ptr(),
                self._data,
                self._size * size_of[Self.dtype](),
            )
        )

    def enqueue_copy_from(
        self, src_ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]
    ) raises:
        """Enqueues an asynchronous copy from host memory to this buffer.

        This method schedules a memory copy operation from the specified host memory
        location to this device buffer. The operation is asynchronous and will be
        executed in the stream associated with this buffer's context.

        Args:
            src_ptr: Pointer to the source host memory location.

        Raises:
            If the copy operation fails.
        """
        _checked(
            external_call[
                "AsyncRT_DeviceContext_HtoD_async_sized",
                _CString[],
                _DeviceContextPtr[mut=True],
                UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
                UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
                Int,
            ](
                self._ctx_ptr,
                self._data,
                src_ptr,
                self._size * size_of[Self.dtype](),
            )
        )

    def enqueue_copy_from(self, src: HostBuffer[Self.dtype]) raises:
        """Enqueues an asynchronous copy from host memory to this buffer.

        This method schedules a memory copy operation from the specified host memory
        location to this device buffer. The operation is asynchronous and will be
        executed in the stream associated with this buffer's context.

        Args:
            src: Host buffer to copy from.

        Raises:
            If the copy operation fails.
        """
        _checked(
            external_call[
                "AsyncRT_DeviceContext_HtoD_async_sized",
                _CString[],
                _DeviceContextPtr[mut=True],
                UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
                UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
                Int,
            ](
                self._ctx_ptr,
                self._data,
                src.unsafe_ptr(),
                self._size * size_of[Self.dtype](),
            )
        )
