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

# Implementation of the C++ backed DeviceContext in Mojo
"""This module provides functionality for interacting with accelerators. In
particular the
[`DeviceContext`](/mojo/stdlib/gpu/host/device_context/DeviceContext) struct,
which represents a single stream of execution on a given accelerator. You can
use this struct to allocate accelerator memory, copy data to and from the
accelerator, and compile and execute functions on the accelerator."""

from collections import List, Optional
from collections.string import StaticString, StringSlice
from pathlib import Path
from sys import env_get_int, env_get_string, external_call, is_defined, sizeof
from sys.ffi import c_char
from sys.compile import DebugLevel, OptimizationLevel
from sys.info import _get_arch, has_nvidia_gpu_accelerator, is_triple
from sys.param_env import _is_bool_like
from sys.intrinsics import _type_is_eq

from builtin._location import __call_location, _SourceLocation
from compile.compile import Info
from gpu.host._compile import (
    _compile_code,
    _compile_code_asm,
    _cross_compilation,
    _get_gpu_target,
    _ptxas_compile,
    _to_sass,
)
from memory import stack_allocation

from utils import Variant
from utils._serialize import _serialize_elements, _serialize_elements_compact

from builtin.device_passable import DevicePassable

from .info import DEFAULT_GPU


# Create empty structs to ensure type checking when using the C++ handles.
struct _DeviceContextCpp:
    pass


struct _DeviceBufferCpp:
    pass


struct _DeviceFunctionCpp:
    pass


struct _DeviceMulticastBufferCpp:
    pass


struct _DeviceStreamCpp:
    pass


struct _DeviceTimerCpp:
    pass


alias _DeviceContextPtr = UnsafePointer[_DeviceContextCpp]
alias _DeviceBufferPtr = UnsafePointer[_DeviceBufferCpp]
alias _DeviceFunctionPtr = UnsafePointer[_DeviceFunctionCpp]
alias _DeviceMulticastBufferPtr = UnsafePointer[_DeviceMulticastBufferCpp]
alias _DeviceStreamPtr = UnsafePointer[_DeviceStreamCpp]
alias _DeviceTimerPtr = UnsafePointer[_DeviceTimerCpp]
alias _CharPtr = UnsafePointer[UInt8]
alias _IntPtr = UnsafePointer[Int32]
alias _VoidPtr = UnsafePointer[NoneType]
alias _SizeT = UInt

alias _DumpPath = Variant[Bool, Path, StaticString, fn () capturing -> Path]

# Define helper methods to call AsyncRT bindings.


fn _string_from_owned_charptr(c_str: _CharPtr) -> String:
    var result = String(unsafe_from_utf8_ptr=c_str)
    # void AsyncRT_DeviceContext_strfree(const char* ptr)
    external_call["AsyncRT_DeviceContext_strfree", NoneType, _CharPtr](c_str)
    return result


@always_inline
fn _checked(
    err: _CharPtr,
    *,
    msg: String = "",
    location: OptionalReg[_SourceLocation] = None,
) raises:
    if err:
        _raise_checked_impl(err, msg, location.or_else(__call_location[1]()))


@no_inline
fn _raise_checked_impl(
    err_msg: _CharPtr, msg: String, location: _SourceLocation
) raises:
    var err = _string_from_owned_charptr(err_msg)
    raise Error(location.prefix(err + ((" " + msg) if msg else "")))


struct _DeviceTimer:
    var _handle: _DeviceTimerPtr

    @implicit
    fn __init__(out self, ptr: _DeviceTimerPtr):
        self._handle = ptr

    fn __del__(owned self):
        # void AsyncRT_DeviceTimer_release(const DviceTimer *timer)
        external_call["AsyncRT_DeviceTimer_release", NoneType, _DeviceTimerPtr](
            self._handle
        )


@value
struct _DeviceBufferMode:
    var _mode: Int

    alias _SYNC = _DeviceBufferMode(0)
    alias _ASYNC = _DeviceBufferMode(1)

    fn __eq__(self, other: Self) -> Bool:
        return self._mode == other._mode


struct HostBuffer[type: DType](Sized, Stringable, Writable):
    """Represents a block of host-resident storage. For GPU devices, a host
    buffer is allocated in the host's global memory.

    To allocate a `HostBuffer`, use one of the methods provided by
    `DeviceContext`, such as
    [`enqueue_create_host_buffer()`](/mojo/stdlib/gpu/host/device_context/DeviceContext#enqueue_create_host_buffer).

    Parameters:
        type: Data type to be stored in the buffer.
    """

    alias _HostPtr = UnsafePointer[Scalar[type]]

    # We cache the pointer of the buffer here to provide access to elements.
    var _host_ptr: Self._HostPtr
    var _handle: _DeviceBufferPtr

    @doc_private
    fn __init__(
        out self,
        ctx: DeviceContext,
        size: Int,
    ) raises:
        """This init takes in a constructed `DeviceContext` and schedules an
        owned buffer allocation using the stream in the device context.
        """
        constrained[
            not is_gpu(),
            "HostBuffer is not supported on GPUs",
        ]()
        alias elem_size = sizeof[type]()
        var cpp_handle = _DeviceBufferPtr()
        var host_ptr = Self._HostPtr()

        # const char *AsyncRT_DeviceContext_createHostBuffer(const DeviceBuffer **result, void **device_ptr, const DeviceContext *ctx, size_t len, size_t elem_size)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_createHostBuffer",
                _CharPtr,
                UnsafePointer[_DeviceBufferPtr],
                UnsafePointer[Self._HostPtr],
                _DeviceContextPtr,
                _SizeT,
                _SizeT,
            ](
                UnsafePointer(to=cpp_handle),
                UnsafePointer(to=host_ptr),
                ctx._handle,
                size,
                elem_size,
            )
        )

        self._host_ptr = host_ptr
        self._handle = cpp_handle

    @doc_private
    fn __init__(out self, handle: _DeviceBufferPtr, host_ptr: Self._HostPtr):
        constrained[
            not is_gpu(),
            "HostBuffer is not supported on GPUs",
        ]()
        self._host_ptr = host_ptr
        self._handle = handle

    @doc_private
    fn __init__(
        out self,
        ctx: DeviceContext,
        host_ptr: Self._HostPtr,
        size: Int,
        *,
        owning: Bool,
    ):
        constrained[
            not is_gpu(),
            "HostBuffer is not supported on GPUs",
        ]()
        alias elem_size = sizeof[type]()
        var cpp_handle = _DeviceBufferPtr()
        # void AsyncRT_DeviceContext_createBuffer_owning(
        #     const DeviceBuffer **result, const DeviceContext *ctx,
        #     void *device_ptr, size_t len, size_t elem_size, bool owning)
        external_call[
            "AsyncRT_DeviceContext_createBuffer_owning",
            NoneType,
            UnsafePointer[_DeviceBufferPtr],
            _DeviceContextPtr,
            Self._HostPtr,
            _SizeT,
            _SizeT,
            Bool,
        ](
            UnsafePointer(to=cpp_handle),
            ctx._handle,
            host_ptr,
            size,
            elem_size,
            owning,
        )

        self._host_ptr = host_ptr
        self._handle = cpp_handle

    fn __copyinit__(out self, existing: Self):
        """Creates a copy of an existing host buffer by incrementing its reference count.

        This copy constructor creates a new reference to the same underlying host buffer
        by incrementing the reference count of the native buffer object. Both the original
        and the copy will refer to the same memory on the device.

        Args:
            existing: The host buffer to copy.
        """
        constrained[
            not is_gpu(),
            "HostBuffer is not supported on GPUs",
        ]()
        # Increment the reference count before copying the handle.
        #
        # void AsyncRT_DeviceBuffer_retain(const DeviceBuffer *buffer)
        external_call[
            "AsyncRT_DeviceBuffer_retain",
            NoneType,
            _DeviceBufferPtr,
        ](existing._handle)
        self._host_ptr = existing._host_ptr
        self._handle = existing._handle

    @always_inline
    fn copy(self) -> Self:
        """Explicitly construct a copy of self.

        Returns:
            A copy of this value.
        """
        constrained[
            not is_gpu(),
            "HostBuffer is not supported on GPUs",
        ]()
        return self

    fn __moveinit__(out self, owned existing: Self):
        """Initializes this buffer by taking ownership of an existing buffer.

        This move constructor transfers ownership of the device buffer from the existing
        instance to the new instance without incrementing the reference count.

        Args:
            existing: The buffer to move from, which will no longer be valid after this call.
        """
        constrained[
            not is_gpu(),
            "HostBuffer is not supported on GPUs",
        ]()
        self._host_ptr = existing._host_ptr
        self._handle = existing._handle

    fn __del__(owned self):
        """Releases resources associated with this host buffer.

        This function schedules an owned buffer free using the stream in the
        device context. The actual deallocation may occur asynchronously after
        all operations using this buffer have completed.
        """
        constrained[
            not is_gpu(),
            "HostBuffer is not supported on GPUs",
        ]()
        # void AsyncRT_DeviceBuffer_release(const DeviceBuffer *buffer)
        external_call[
            "AsyncRT_DeviceBuffer_release", NoneType, _DeviceBufferPtr
        ](
            self._handle,
        )

    fn __len__(self) -> Int:
        """Returns the number of elements in this buffer.

        This method calculates the number of elements by dividing the total byte size
        of the buffer by the size of each element.

        Returns:
            The number of elements in the buffer.
        """
        constrained[
            not is_gpu(),
            "HostBuffer is not supported on GPUs",
        ]()
        # int64_t AsyncRT_DeviceBuffer_bytesize(const DeviceBuffer *buffer)
        return (
            external_call[
                "AsyncRT_DeviceBuffer_bytesize", Int, _DeviceBufferPtr
            ](self._handle)
            // sizeof[type]()
        )

    fn create_sub_buffer[
        view_type: DType
    ](self, offset: Int, size: Int) raises -> HostBuffer[view_type]:
        """Creates a sub-buffer view of this buffer with a different element type.

        This method creates a new buffer that references a subset of the memory in this
        buffer, potentially with a different element type. The sub-buffer shares the
        underlying memory with the original buffer.

        Parameters:
            view_type: The data type for elements in the new sub-buffer.

        Args:
            offset: The starting offset in elements from the beginning of this buffer.
            size: The number of elements in the new sub-buffer.

        Returns:
            A new HostBuffer referencing the specified region with the specified element type.
        """
        constrained[
            not is_gpu(),
            "HostBuffer is not supported on GPUs",
        ]()
        alias elem_size = sizeof[view_type]()
        var new_handle = _DeviceBufferPtr()
        var new_host_ptr = UnsafePointer[Scalar[view_type]]()
        # const char *AsyncRT_DeviceBuffer_createSubBuffer(
        #     const DeviceBuffer **result, void **device_ptr,
        #     const DeviceBuffer *buf, size_t offset, size_t len, size_t elem_size)
        _checked(
            external_call[
                "AsyncRT_DeviceBuffer_createSubBuffer",
                _CharPtr,
                UnsafePointer[_DeviceBufferPtr],
                UnsafePointer[UnsafePointer[Scalar[view_type]]],
                _DeviceBufferPtr,
                _SizeT,
                _SizeT,
                _SizeT,
            ](
                UnsafePointer(to=new_handle),
                UnsafePointer(to=new_host_ptr),
                self._handle,
                offset,
                size,
                elem_size,
            )
        )
        return HostBuffer[view_type](new_handle, new_host_ptr)

    fn enqueue_copy_to(self, dst: HostBuffer[type, **_]) raises:
        """Enqueues an asynchronous copy from this buffer to another host buffer.

        This method schedules a memory copy operation from this buffer to the destination
        buffer. The operation is asynchronous and will be executed in the stream associated
        with this buffer's context.

        Args:
            dst: The destination host buffer to copy data to.
        """
        dst.context().enqueue_copy(dst, self)

    fn enqueue_copy_to(self, dst: DeviceBuffer[type, **_]) raises:
        """Enqueues an asynchronous copy from this buffer to a device buffer.

        This method schedules a memory copy operation from this buffer to the destination
        buffer. The operation is asynchronous and will be executed in the stream associated
        with this buffer's context.

        Args:
            dst: The destination device buffer to copy data to.
        """
        constrained[
            not is_gpu(),
            "HostBuffer is not supported on GPUs",
        ]()
        dst.context().enqueue_copy(dst, self)

    fn enqueue_copy_to(self, dst_ptr: UnsafePointer[Scalar[type]]) raises:
        """Enqueues an asynchronous copy from this buffer to host memory.

        This method schedules a memory copy operation from this device buffer to the
        specified host memory location. The operation is asynchronous and will be
        executed in the stream associated with this buffer's context.

        Args:
            dst_ptr: Pointer to the destination host memory location.
        """
        constrained[
            not is_gpu(),
            "HostBuffer is not supported on GPUs",
        ]()
        self.context().enqueue_copy(dst_ptr, self)

    fn enqueue_copy_from(self, src: HostBuffer[type, **_]) raises:
        """Enqueues an asynchronous copy to this buffer from another host buffer.

        This method schedules a memory copy operation to this buffer from the source
        buffer. The operation is asynchronous and will be executed in the stream
        associated with this buffer's context.

        Args:
            src: The source host buffer to copy data from.
        """
        constrained[
            not is_gpu(),
            "HostBuffer is not supported on GPUs",
        ]()
        self.context().enqueue_copy(self, src)

    fn enqueue_copy_from(self, src: DeviceBuffer[type, **_]) raises:
        """Enqueues an asynchronous copy to this buffer from a device buffer.

        This method schedules a memory copy operation to this buffer from the source
        buffer. The operation is asynchronous and will be executed in the stream
        associated with this buffer's context.

        Args:
            src: The source device buffer to copy data from.
        """
        constrained[
            not is_gpu(),
            "HostBuffer is not supported on GPUs",
        ]()
        self.context().enqueue_copy(self, src)

    fn enqueue_copy_from(self, src_ptr: UnsafePointer[Scalar[type]]) raises:
        """Enqueues an asynchronous copy to this buffer from host memory.

        This method schedules a memory copy operation to this device buffer from the
        specified host memory location. The operation is asynchronous and will be
        executed in the stream associated with this buffer's context.

        Args:
            src_ptr: Pointer to the source host memory location.
        """
        constrained[
            not is_gpu(),
            "HostBuffer is not supported on GPUs",
        ]()
        self.context().enqueue_copy(self, src_ptr)

    fn enqueue_fill(self, val: Scalar[type]) raises -> Self:
        """Enqueues an operation to fill this buffer with a specified value.

        This method schedules a memory set operation that fills the entire buffer
        with the specified value. The operation is asynchronous and will be executed
        in the stream associated with this buffer's context.

        Args:
            val: The value to fill the buffer with.

        Returns:
            Self reference for method chaining.
        """
        constrained[
            not is_gpu(),
            "HostBuffer is not supported on GPUs",
        ]()
        self.context().enqueue_memset(self, val)
        return self

    fn reassign_ownership_to(self, ctx: DeviceContext) raises:
        """Transfers ownership of this buffer to another device context.

        This method changes the device context that owns this buffer. This can be
        useful when sharing buffers between different contexts or when migrating
        workloads between devices.

        Args:
            ctx: The new device context to take ownership of this buffer.
        """
        constrained[
            not is_gpu(),
            "HostBuffer is not supported on GPUs",
        ]()
        # const char * AsyncRT_DeviceBuffer_reassignOwnershipTo(const DeviceBuffer *buf, const DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceBuffer_reassignOwnershipTo",
                _CharPtr,
                _DeviceBufferPtr,
                _DeviceContextPtr,
            ](self._handle, ctx._handle)
        )

    fn take_ptr(
        owned self,
    ) -> Self._HostPtr:
        """Takes ownership of the device pointer from this buffer.

        This method releases the device pointer from the buffer's control and
        returns it to the caller. After this call, the buffer no longer owns
        the pointer, and the caller is responsible for managing its lifecycle.

        Returns:
            The raw device pointer that was owned by this buffer.
        """
        constrained[
            not is_gpu(),
            "HostBuffer is not supported on GPUs",
        ]()
        # void AsyncRT_DeviceBuffer_release_ptr(const DeviceBuffer *buffer)
        external_call[
            "AsyncRT_DeviceBuffer_release_ptr", NoneType, _DeviceBufferPtr
        ](self._handle)
        var result = self._host_ptr
        self._host_ptr = Self._HostPtr()
        return result

    @always_inline
    fn unsafe_ptr(
        self,
    ) -> Self._HostPtr:
        """Returns the raw device pointer without transferring ownership.

        This method provides direct access to the underlying device pointer
        for advanced use cases. The buffer retains ownership of the pointer.

        Returns:
            The raw device pointer owned by this buffer.
        """
        constrained[
            not is_gpu(),
            "HostBuffer is not supported on GPUs",
        ]()
        return self._host_ptr

    fn context(self) raises -> DeviceContext:
        """Returns the device context associated with this buffer.

        This method retrieves the device context that owns this buffer and is
        responsible for managing its lifecycle and operations.

        Returns:
            The device context associated with this buffer.
        """
        constrained[
            not is_gpu(),
            "HostBuffer is not supported on GPUs",
        ]()
        # const DeviceContext *AsyncRT_DeviceBuffer_context(const DeviceBuffer *buffer)
        var ctx_ptr: _DeviceContextPtr = external_call[
            "AsyncRT_DeviceBuffer_context", _DeviceContextPtr, _DeviceBufferPtr
        ](self._handle)
        return DeviceContext(ctx_ptr)

    fn write_to[W: Writer](self, mut writer: W):
        """Writes a string representation of this buffer to the provided writer.

        This method formats the buffer's contents as a string and writes it to
        the specified writer. For large buffers, a compact representation is used.

        Parameters:
            W: The writer type.

        Args:
            writer: The writer to output the formatted string to.
        """
        constrained[
            not is_gpu(),
            "HostBuffer is not supported on GPUs",
        ]()
        writer.write("HostBuffer")
        writer.write("(")

        @parameter
        fn serialize[T: Writable](val: T):
            writer.write(val)

        var size = len(self)

        if size < 1000:
            writer.write("[")
            _serialize_elements[serialize_fn=serialize](
                self.unsafe_ptr(), len(self)
            )
            writer.write("]")
        else:
            _serialize_elements[serialize_fn=serialize, compact=True](
                self.unsafe_ptr(), size
            )
        writer.write(")")

    fn __str__(self) -> String:
        """Returns a string representation of the `HostBuffer`.

        This method creates a human-readable string representation of the buffer's contents
        by mapping the device memory to host memory and formatting the elements.

        Returns:
            A string containing the formatted buffer contents.
        """
        constrained[
            not is_gpu(),
            "HostBuffer is not supported on GPUs",
        ]()
        return String.write(self)

    fn __getitem__(self, idx: Int) -> Scalar[type]:
        """Retrieves the element at the specified index from the host buffer.

        This operator allows direct access to individual elements in the host buffer
        using array indexing syntax.

        Args:
            idx: The index of the element to retrieve.

        Returns:
            The scalar value at the specified index.
        """
        constrained[
            not is_gpu(),
            "HostBuffer is not supported on GPUs",
        ]()
        return self._host_ptr[idx]

    fn __setitem__(self: HostBuffer[type], idx: Int, val: Scalar[type]):
        """Sets the element at the specified index in the host buffer.

        This operator allows direct modification of individual elements in the host buffer
        using array indexing syntax.

        Args:
            idx: The index of the element to modify.
            val: The new value to store at the specified index.
        """
        constrained[
            not is_gpu(),
            "HostBuffer is not supported on GPUs",
        ]()
        self._host_ptr[idx] = val

    fn as_span(ref self, out span: Span[Scalar[type], __origin_of(self)]):
        """
        Returns a `Span` pointing to the underlying memory of the `HostBuffer`.

        Returns:
            A `Span` pointing to the underlying memory of the `HostBuffer`.
        """
        return __type_of(span)(ptr=self._host_ptr, length=len(self))


struct DeviceBuffer[type: DType](
    Sized,
    Stringable,
    Writable,
    CollectionElement,
    DevicePassable,
):
    """Represents a block of device-resident storage. For GPU devices, a device
    buffer is allocated in the device's global memory.

    To allocate a `DeviceBuffer`, use one of the methods provided by
    `DeviceContext`, such as
    [`enqueue_create_buffer()`](/mojo/stdlib/gpu/host/device_context/DeviceContext#enqueue_create_buffer).

    Parameters:
        type: Data type to be stored in the buffer.
    """

    # Implementation of `DevicePassable`
    alias device_type: AnyTrivialRegType = UnsafePointer[Scalar[type]]
    """DeviceBuffer types are remapped to UnsafePointer when passed to accelerator devices."""

    alias _DevicePtr = UnsafePointer[Scalar[type]]
    # _device_ptr must be the first word in the struct to enable passing of
    # DeviceBuffer to kernels. The first word is passed to the kernel and
    # it needs to contain the value registered with the driver.
    var _device_ptr: Self._DevicePtr
    var _handle: _DeviceBufferPtr

    @doc_private
    @always_inline
    fn __init__(
        out self,
        ctx: DeviceContext,
        size: Int,
        mode: _DeviceBufferMode,
    ) raises:
        """This init takes in a constructed `DeviceContext` and schedules an
        owned buffer allocation using the stream in the device context.
        """
        constrained[
            not is_gpu(),
            "DeviceBuffer is not supported on GPUs",
        ]()
        alias elem_size = sizeof[type]()
        var cpp_handle = _DeviceBufferPtr()
        var device_ptr = Self._DevicePtr()

        if mode == _DeviceBufferMode._SYNC:
            # const char *AsyncRT_DeviceContext_createBuffer_sync(const DeviceBuffer **result, void **device_ptr, const DeviceContext *ctx, size_t len, size_t elem_size)
            _checked(
                external_call[
                    "AsyncRT_DeviceContext_createBuffer_sync",
                    _CharPtr,
                    UnsafePointer[_DeviceBufferPtr],
                    UnsafePointer[Self._DevicePtr],
                    _DeviceContextPtr,
                    _SizeT,
                    _SizeT,
                ](
                    UnsafePointer(to=cpp_handle),
                    UnsafePointer(to=device_ptr),
                    ctx._handle,
                    size,
                    elem_size,
                )
            )
        elif mode == _DeviceBufferMode._ASYNC:
            # const char *AsyncRT_DeviceContext_createBuffer_async(const DeviceBuffer **result, void **device_ptr, const DeviceContext *ctx, size_t len, size_t elem_size)
            _checked(
                external_call[
                    "AsyncRT_DeviceContext_createBuffer_async",
                    _CharPtr,
                    UnsafePointer[_DeviceBufferPtr],
                    UnsafePointer[Self._DevicePtr],
                    _DeviceContextPtr,
                    _SizeT,
                    _SizeT,
                ](
                    UnsafePointer(to=cpp_handle),
                    UnsafePointer(to=device_ptr),
                    ctx._handle,
                    size,
                    elem_size,
                )
            )
        else:
            raise Error(
                "DeviceBuffer.__init__: Unsupported _DeviceBufferMode(",
                mode._mode,
                ")",
            )

        self._device_ptr = device_ptr
        self._handle = cpp_handle

    @doc_private
    fn __init__(
        out self, handle: _DeviceBufferPtr, device_ptr: Self._DevicePtr
    ):
        constrained[
            not is_gpu(),
            "DeviceBuffer is not supported on GPUs",
        ]()
        self._device_ptr = device_ptr
        self._handle = handle

    @doc_private
    fn __init__(
        out self,
        ctx: DeviceContext,
        ptr: Self._DevicePtr,
        size: Int,
        *,
        owning: Bool,
    ):
        constrained[
            not is_gpu(),
            "DeviceBuffer is not supported on GPUs",
        ]()
        alias elem_size = sizeof[type]()
        var cpp_handle = _DeviceBufferPtr()
        # void AsyncRT_DeviceContext_createBuffer_owning(
        #     const DeviceBuffer **result, const DeviceContext *ctx,
        #     void *device_ptr, size_t len, size_t elem_size, bool owning)
        external_call[
            "AsyncRT_DeviceContext_createBuffer_owning",
            NoneType,
            UnsafePointer[_DeviceBufferPtr],
            _DeviceContextPtr,
            Self._DevicePtr,
            _SizeT,
            _SizeT,
            Bool,
        ](
            UnsafePointer(to=cpp_handle),
            ctx._handle,
            ptr,
            size,
            elem_size,
            owning,
        )

        self._device_ptr = ptr
        self._handle = cpp_handle

    fn __copyinit__(out self, existing: Self):
        """Creates a copy of an existing device buffer by incrementing its reference count.

        This copy constructor creates a new reference to the same underlying device buffer
        by incrementing the reference count of the native buffer object. Both the original
        and the copy will refer to the same memory on the device.

        Args:
            existing: The device buffer to copy.
        """
        constrained[
            not is_gpu(),
            "DeviceBuffer is not supported on GPUs",
        ]()
        # Increment the reference count before copying the handle.
        #
        # void AsyncRT_DeviceBuffer_retain(const DeviceBuffer *buffer)
        external_call[
            "AsyncRT_DeviceBuffer_retain",
            NoneType,
            _DeviceBufferPtr,
        ](existing._handle)
        self._device_ptr = existing._device_ptr
        self._handle = existing._handle

    @always_inline
    fn copy(self) -> Self:
        """Explicitly construct a copy of self.

        Returns:
            A copy of this value.
        """
        constrained[
            not is_gpu(),
            "DeviceBuffer is not supported on GPUs",
        ]()
        return self

    fn __moveinit__(out self, owned existing: Self):
        """Initializes this buffer by taking ownership of an existing buffer.

        This move constructor transfers ownership of the device buffer from the existing
        instance to the new instance without incrementing the reference count.

        Args:
            existing: The buffer to move from, which will no longer be valid after this call.
        """
        constrained[
            not is_gpu(),
            "DeviceBuffer is not supported on GPUs",
        ]()
        self._device_ptr = existing._device_ptr
        self._handle = existing._handle

    @always_inline
    fn __del__(owned self):
        """Releases resources associated with this device buffer.

        This function schedules an owned buffer free using the stream in the
        device context. The actual deallocation may occur asynchronously after
        all operations using this buffer have completed.
        """
        constrained[
            not is_gpu(),
            "DeviceBuffer is not supported on GPUs",
        ]()
        # void AsyncRT_DeviceBuffer_release(const DeviceBuffer *buffer)
        external_call[
            "AsyncRT_DeviceBuffer_release", NoneType, _DeviceBufferPtr
        ](
            self._handle,
        )

    fn __len__(self) -> Int:
        """Returns the number of elements in this buffer.

        This method calculates the number of elements by dividing the total byte size
        of the buffer by the size of each element.

        Returns:
            The number of elements in the buffer.
        """
        constrained[
            not is_gpu(),
            "DeviceBuffer is not supported on GPUs",
        ]()
        # int64_t AsyncRT_DeviceBuffer_bytesize(const DeviceBuffer *buffer)
        return (
            external_call[
                "AsyncRT_DeviceBuffer_bytesize", Int, _DeviceBufferPtr
            ](self._handle)
            // sizeof[type]()
        )

    @always_inline
    fn create_sub_buffer[
        view_type: DType
    ](self, offset: Int, size: Int) raises -> DeviceBuffer[view_type]:
        """Creates a sub-buffer view of this buffer with a different element type.

        This method creates a new buffer that references a subset of the memory in this
        buffer, potentially with a different element type. The sub-buffer shares the
        underlying memory with the original buffer.

        Parameters:
            view_type: The data type for elements in the new sub-buffer.

        Args:
            offset: The starting offset in elements from the beginning of this buffer.
            size: The number of elements in the new sub-buffer.

        Returns:
            A new DeviceBuffer referencing the specified region with the specified element type.
        """
        constrained[
            not is_gpu(),
            "DeviceBuffer is not supported on GPUs",
        ]()
        alias elem_size = sizeof[view_type]()
        var new_handle = _DeviceBufferPtr()
        var new_device_ptr = UnsafePointer[Scalar[view_type]]()
        # const char *AsyncRT_DeviceBuffer_createSubBuffer(
        #     const DeviceBuffer **result, void **device_ptr,
        #     const DeviceBuffer *buf, size_t offset, size_t len, size_t elem_size)
        _checked(
            external_call[
                "AsyncRT_DeviceBuffer_createSubBuffer",
                _CharPtr,
                UnsafePointer[_DeviceBufferPtr],
                UnsafePointer[UnsafePointer[Scalar[view_type]]],
                _DeviceBufferPtr,
                _SizeT,
                _SizeT,
                _SizeT,
            ](
                UnsafePointer(to=new_handle),
                UnsafePointer(to=new_device_ptr),
                self._handle,
                offset,
                size,
                elem_size,
            )
        )
        return DeviceBuffer[view_type](new_handle, new_device_ptr)

    fn enqueue_copy_to(self, dst: DeviceBuffer[type, **_]) raises:
        """Enqueues an asynchronous copy from this buffer to another device buffer.

        This method schedules a memory copy operation from this buffer to the destination
        buffer. The operation is asynchronous and will be executed in the stream associated
        with this buffer's context.

        Args:
            dst: The destination device buffer to copy data to.
        """
        constrained[
            not is_gpu(),
            "DeviceBuffer is not supported on GPUs",
        ]()
        dst.context().enqueue_copy(dst, self)

    fn enqueue_copy_to(self, dst: HostBuffer[type, **_]) raises:
        """Enqueues an asynchronous copy from this buffer to a host buffer.

        This method schedules a memory copy operation from this buffer to the destination
        buffer. The operation is asynchronous and will be executed in the stream associated
        with this buffer's context.

        Args:
            dst: The destination host buffer to copy data to.
        """
        constrained[
            not is_gpu(),
            "DeviceBuffer is not supported on GPUs",
        ]()
        dst.context().enqueue_copy(dst, self)

    fn enqueue_copy_to(self, dst_ptr: UnsafePointer[Scalar[type]]) raises:
        """Enqueues an asynchronous copy from this buffer to host memory.

        This method schedules a memory copy operation from this device buffer to the
        specified host memory location. The operation is asynchronous and will be
        executed in the stream associated with this buffer's context.

        Args:
            dst_ptr: Pointer to the destination host memory location.
        """
        constrained[
            not is_gpu(),
            "DeviceBuffer is not supported on GPUs",
        ]()
        self.context().enqueue_copy(dst_ptr, self)

    fn enqueue_copy_from(self, src: DeviceBuffer[type, **_]) raises:
        """Enqueues an asynchronous copy to this buffer from another device buffer.

        This method schedules a memory copy operation to this buffer from the source
        buffer. The operation is asynchronous and will be executed in the stream
        associated with this buffer's context.

        Args:
            src: The source device buffer to copy data from.
        """
        constrained[
            not is_gpu(),
            "DeviceBuffer is not supported on GPUs",
        ]()
        self.context().enqueue_copy(self, src)

    fn enqueue_copy_from(self, src: HostBuffer[type, **_]) raises:
        """Enqueues an asynchronous copy to this buffer from a host buffer.

        This method schedules a memory copy operation to this buffer from the source
        buffer. The operation is asynchronous and will be executed in the stream
        associated with this buffer's context.

        Args:
            src: The source host buffer to copy data from.
        """
        constrained[
            not is_gpu(),
            "DeviceBuffer is not supported on GPUs",
        ]()
        self.context().enqueue_copy(self, src)

    fn enqueue_copy_from(self, src_ptr: UnsafePointer[Scalar[type]]) raises:
        """Enqueues an asynchronous copy to this buffer from host memory.

        This method schedules a memory copy operation to this device buffer from the
        specified host memory location. The operation is asynchronous and will be
        executed in the stream associated with this buffer's context.

        Args:
            src_ptr: Pointer to the source host memory location.
        """
        constrained[
            not is_gpu(),
            "DeviceBuffer is not supported on GPUs",
        ]()
        self.context().enqueue_copy(self, src_ptr)

    fn enqueue_fill(self, val: Scalar[type]) raises -> Self:
        """Enqueues an operation to fill this buffer with a specified value.

        This method schedules a memory set operation that fills the entire buffer
        with the specified value. The operation is asynchronous and will be executed
        in the stream associated with this buffer's context.

        Args:
            val: The value to fill the buffer with.

        Returns:
            Self reference for method chaining.
        """
        constrained[
            not is_gpu(),
            "DeviceBuffer is not supported on GPUs",
        ]()
        self.context().enqueue_memset(self, val)
        return self

    fn reassign_ownership_to(self, ctx: DeviceContext) raises:
        """Transfers ownership of this buffer to another device context.

        This method changes the device context that owns this buffer. This can be
        useful when sharing buffers between different contexts or when migrating
        workloads between devices.

        Args:
            ctx: The new device context to take ownership of this buffer.
        """
        constrained[
            not is_gpu(),
            "DeviceBuffer is not supported on GPUs",
        ]()
        # const char * AsyncRT_DeviceBuffer_reassignOwnershipTo(const DeviceBuffer *buf, const DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceBuffer_reassignOwnershipTo",
                _CharPtr,
                _DeviceBufferPtr,
                _DeviceContextPtr,
            ](self._handle, ctx._handle)
        )

    @always_inline
    fn take_ptr(
        owned self,
    ) -> Self._DevicePtr:
        """Takes ownership of the device pointer from this buffer.

        This method releases the device pointer from the buffer's control and
        returns it to the caller. After this call, the buffer no longer owns
        the pointer, and the caller is responsible for managing its lifecycle.

        Returns:
            The raw device pointer that was owned by this buffer.
        """
        return self._take_ptr()

    fn _take_ptr(
        owned self,
    ) -> Self._DevicePtr:
        constrained[
            not is_gpu(),
            "DeviceBuffer is not supported on GPUs",
        ]()
        # void AsyncRT_DeviceBuffer_release_ptr(const DeviceBuffer *buffer)
        external_call[
            "AsyncRT_DeviceBuffer_release_ptr", NoneType, _DeviceBufferPtr
        ](self._handle)
        var result = self._device_ptr
        self._device_ptr = Self._DevicePtr()
        return result

    @always_inline
    fn unsafe_ptr(
        self,
    ) -> Self._DevicePtr:
        """Returns the raw device pointer without transferring ownership.

        This method provides direct access to the underlying device pointer
        for advanced use cases. The buffer retains ownership of the pointer.

        Returns:
            The raw device pointer owned by this buffer.
        """
        return self._unsafe_ptr()

    fn _unsafe_ptr(
        self,
    ) -> Self._DevicePtr:
        constrained[
            not is_gpu(),
            "DeviceBuffer is not supported on GPUs",
        ]()
        return self._device_ptr

    fn context(self) raises -> DeviceContext:
        """Returns the device context associated with this buffer.

        This method retrieves the device context that owns this buffer and is
        responsible for managing its lifecycle and operations.

        Returns:
            The device context associated with this buffer.
        """
        constrained[
            not is_gpu(),
            "DeviceBuffer is not supported on GPUs",
        ]()
        # const DeviceContext *AsyncRT_DeviceBuffer_context(const DeviceBuffer *buffer)
        var ctx_ptr: _DeviceContextPtr = external_call[
            "AsyncRT_DeviceBuffer_context", _DeviceContextPtr, _DeviceBufferPtr
        ](self._handle)
        return DeviceContext(ctx_ptr)

    fn map_to_host(
        self,
        out mapped_buffer: _HostMappedBuffer[type, __origin_of(self)],
    ) raises:
        """Maps this device buffer to host memory for CPU access.

        This method creates a host-accessible view of the device buffer's contents.
        The mapping operation may involve copying data from device to host memory.

        Returns:
            A host-mapped buffer that provides CPU access to the device buffer's
            contents inside a with-statement.

        Raises:
            If there's an error during buffer creation or data transfer.

        Notes:

        Values modified inside the `with` statement are updated on the
        device when the `with` statement exits.

        Example:

        ```mojo
        from gpu.host import DeviceContext

        var ctx = DeviceContext()
        var length = 1024
        var in_dev = ctx.enqueue_create_buffer[DType.float32](length)
        var out_dev = ctx.enqueue_create_buffer[DType.float32](length)

        # Initialize the input and output with known values.
        with in_dev.map_to_host() as in_host, out_dev.map_to_host() as out_host:
            for i in range(length):
                in_host[i] = i
                out_host[i] = 255
        ```
        """
        constrained[
            not is_gpu(),
            "DeviceBuffer is not supported on GPUs",
        ]()
        mapped_buffer = _HostMappedBuffer[type, __origin_of(self)](
            self.context(), self
        )

    fn write_to[W: Writer](self, mut writer: W):
        """Writes a string representation of this buffer to the provided writer.

        This method formats the buffer's contents as a string and writes it to
        the specified writer. For large buffers, a compact representation is used.

        Parameters:
            W: The writer type.

        Args:
            writer: The writer to output the formatted string to.
        """
        constrained[
            not is_gpu(),
            "DeviceBuffer is not supported on GPUs",
        ]()
        try:
            with self.map_to_host() as host_buffer:
                writer.write("DeviceBuffer")
                writer.write("(")

                @parameter
                fn serialize[T: Writable](val: T):
                    writer.write(val)

                var size = len(self)

                if size < 1000:
                    writer.write("[")
                    _serialize_elements[serialize_fn=serialize](
                        host_buffer.unsafe_ptr(), len(self)
                    )
                    writer.write("]")
                else:
                    _serialize_elements[serialize_fn=serialize, compact=True](
                        host_buffer.unsafe_ptr(), size
                    )
                writer.write(")")
        except e:
            abort("failed to write DeviceBuffer:", e)

    fn __str__(self) -> String:
        """Returns a string representation of the `DeviceBuffer`.

        This method creates a human-readable string representation of the buffer's contents
        by mapping the device memory to host memory and formatting the elements.

        Returns:
            A string containing the formatted buffer contents.
        """
        constrained[
            not is_gpu(),
            "DeviceBuffer is not supported on GPUs",
        ]()
        return String.write(self)


# @doc_private does not work on structs - see MOTO-992.
struct DeviceStream:
    """Represents a CUDA/HIP stream for asynchronous GPU operations.

    A DeviceStream provides a queue for GPU operations that can execute concurrently
    with operations in other streams. Operations within a single stream execute in
    the order they are issued, but operations in different streams may execute in
    any relative order or concurrently.

    This abstraction allows for better utilization of GPU resources by enabling
    overlapping of computation and data transfers.

    Example:

    ```mojo
    from gpu.host import DeviceContext, DeviceStream
    var ctx = DeviceContext(0)  # Select first GPU
    var stream = DeviceStream(ctx)

    # Launch operations on the stream
    # ...

    # Wait for all operations in the stream to complete
    stream.synchronize()
    ```
    """

    var _handle: _DeviceStreamPtr
    """Internal handle to the native stream object."""

    @doc_private
    @always_inline
    fn __init__(out self, ctx: DeviceContext) raises:
        """Creates a new stream associated with the given device context.

        Args:
            ctx: The device context to associate this stream with.

        Raises:
            - If stream creation fails.
        """
        var result = _DeviceStreamPtr()
        # const char *AsyncRT_DeviceContext_stream(const DeviceStream **result, const DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_stream",
                _CharPtr,
                UnsafePointer[_DeviceStreamPtr],
                _DeviceContextPtr,
            ](UnsafePointer(to=result), ctx._handle)
        )
        self._handle = result

    @doc_private
    fn __copyinit__(out self, existing: Self):
        """Creates a copy of an existing stream by incrementing its reference count.

        Args:
            existing: The stream to copy.
        """
        # void AsyncRT_DeviceStream_retain(const DeviceStream *stream)
        external_call[
            "AsyncRT_DeviceStream_retain",
            NoneType,
            _DeviceStreamPtr,
        ](existing._handle)
        self._handle = existing._handle

    @doc_private
    fn __moveinit__(out self, owned existing: Self):
        """Moves an existing stream into this one.

        Args:
            existing: The stream to move from.
        """
        self._handle = existing._handle

    @doc_private
    @always_inline
    fn __del__(owned self):
        """Releases resources associated with this stream."""
        # void AsyncRT_DeviceStream_release(const DeviceStream *stream)
        external_call[
            "AsyncRT_DeviceStream_release", NoneType, _DeviceStreamPtr
        ](
            self._handle,
        )

    @always_inline
    fn synchronize(self) raises:
        """Blocks the calling CPU thread until all operations in this stream complete.

        This function waits until all previously issued commands in this stream
        have completed execution. It provides a synchronization point between
        host and device code.

        Raises:
            If synchronization fails.

        Example:

        ```mojo
        # Launch kernel or memory operations on the stream
        # ...

        # Wait for completion
        stream.synchronize()

        # Now it's safe to use results on the host
        ```
        """
        # const char *AsyncRT_DeviceStream_synchronize(const DeviceStream *stream)
        _checked(
            external_call[
                "AsyncRT_DeviceStream_synchronize",
                _CharPtr,
                _DeviceStreamPtr,
            ](self._handle)
        )


fn _is_nvidia_gpu[target: __mlir_type.`!kgen.target`]() -> Bool:
    return is_triple["nvptx64-nvidia-cuda", target]()


fn _is_path_like(ss: StringSlice) -> Bool:
    # Ideally we want to use `val.start_with` but we hit a compiler bug if we do
    # that. So, instead we implement the function inline, since we only care
    # about whether the string starts with a `/`, `~`, or "./".
    if len(ss) == 0:
        return False
    if len(ss) >= 1:
        if ss[0] == "/" or ss[0] == "~":
            return True
    if len(ss) >= 2:
        if ss[0] == "." and ss[1] == "/":
            return True
    return False


struct DeviceFunction[
    func_type: AnyTrivialRegType, //,
    func: func_type,
    declared_arg_types: Optional[__mlir_type[`!kgen.variadic<`, AnyType, `>`]],
    *,
    target: __mlir_type.`!kgen.target` = _get_gpu_target(),
    _ptxas_info_verbose: Bool = False,
]:
    """Represents a compiled device function for GPU execution.

    This struct encapsulates a compiled GPU function that can be launched on a device.
    It handles the compilation, loading, and resource management of device functions.

    Parameters:
        func_type: The type of the function to compile.
        func: The function to compile for GPU execution.
        declared_arg_types: An optional containing a variadic of the declared types of the kernel signature.
        target: The target architecture for compilation. Defaults to the current GPU target.
        _ptxas_info_verbose: Whether to enable verbose PTX assembly output. Defaults to False.

    Example:

    ```mojo
    from gpu.host import DeviceContext, DeviceFunction

    fn my_kernel(x: Int, y: Int):
        # Kernel implementation
        pass

    var ctx = DeviceContext()
    var kernel = ctx.compile_function[my_kernel]()
    ctx.enqueue_function(kernel, grid_dim=(1,1,1), block_dim=(32,1,1))
    ```
    """

    # emit asm if cross compiling for nvidia gpus.
    alias _emission_kind = "asm" if (
        _cross_compilation() and _is_nvidia_gpu[target]()
    ) else "object"
    var _handle: _DeviceFunctionPtr
    """Internal handle to the compiled device function."""

    var _func_impl: Info[func_type, func]
    """Compilation information for the function."""

    fn __copyinit__(out self, existing: Self):
        """Creates a copy of an existing DeviceFunction.

        This increases the reference count of the underlying device function handle.

        Args:
            existing: The DeviceFunction to copy from.
        """
        # Increment the reference count before copying the handle.
        #
        # void AsyncRT_DeviceFunction_retain(const DeviceFunction *ctx)
        external_call[
            "AsyncRT_DeviceFunction_retain",
            NoneType,
            _DeviceFunctionPtr,
        ](existing._handle)
        self._handle = existing._handle
        self._func_impl = existing._func_impl

    fn __moveinit__(out self, owned existing: Self):
        """Moves an existing DeviceFunction into this one.

        Args:
            existing: The DeviceFunction to move from.
        """
        self._handle = existing._handle
        self._func_impl = existing._func_impl

    fn __del__(owned self):
        """Releases resources associated with this DeviceFunction.

        This decrements the reference count of the underlying device function handle.
        """
        # Decrement the reference count held by this struct.
        #
        # void AsyncRT_DeviceFunction_release(const DeviceFunction *ctx)
        external_call[
            "AsyncRT_DeviceFunction_release",
            NoneType,
            _DeviceFunctionPtr,
        ](self._handle)

    @doc_private
    @always_inline
    fn __init__(
        out self,
        ctx: DeviceContext,
        *,
        func_attribute: OptionalReg[FuncAttribute] = None,
    ) raises:
        """Initializes a new DeviceFunction by compiling the function for the specified device.

        Args:
            ctx: The device context to compile the function for.
            func_attribute: Optional attributes to apply to the function, such as shared memory size.

        Raises:
            Error: If compilation fails or if an unsupported function attribute is provided.
        """
        var max_dynamic_shared_size_bytes: Int32 = -1
        if func_attribute:
            if (
                func_attribute.value().attribute
                == Attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES
            ):
                max_dynamic_shared_size_bytes = func_attribute.value().value
            else:
                raise Error(
                    "the function attribute '",
                    func_attribute.value().attribute,
                    "' is not currently supported",
                )

        # const char *AsyncRT_DeviceContext_loadFunction(
        #     const DeviceFunction **result, const DeviceContext *ctx,
        #     const char *moduleName, const char *functionName, const char *data,
        #     size_t dataLen, int32_t maxDynamicSharedBytes, const char *debugLevel,
        #     int32_t optimizationLevel)
        var result = _DeviceFunctionPtr()
        self._func_impl = _compile_code[
            func,
            emission_kind = self._emission_kind,
            target=target,
        ]()
        var debug_level = String(DebugLevel)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_loadFunction",
                _CharPtr,
                UnsafePointer[_DeviceFunctionPtr],
                _DeviceContextPtr,
                _CharPtr,
                _CharPtr,
                _CharPtr,
                _SizeT,
                Int32,
                _CharPtr,
                Int32,
            ](
                UnsafePointer(to=result),
                ctx._handle,
                self._func_impl.module_name.unsafe_ptr(),
                self._func_impl.function_name.unsafe_ptr(),
                self._func_impl.asm.unsafe_ptr(),
                len(self._func_impl.asm),
                max_dynamic_shared_size_bytes,
                debug_level.unsafe_cstr_ptr().bitcast[UInt8](),
                Int(OptimizationLevel),
            )
        )
        self._handle = result

    fn _copy_to_constant_memory(self, mapping: ConstantMemoryMapping) raises:
        # const char *AsyncRT_DeviceFunction_copyToConstantMemory(
        #     const DeviceFunction *func,
        #     const void *name, size_t nameSize,
        #     const void *data, size_t dataSize)
        _checked(
            external_call[
                "AsyncRT_DeviceFunction_copyToConstantMemory",
                _CharPtr,
                _DeviceFunctionPtr,
                _CharPtr,
                _SizeT,
                _VoidPtr,
                _SizeT,
            ](
                self._handle,
                mapping.name.unsafe_ptr(),
                len(mapping.name),
                mapping.ptr,
                mapping.byte_count,
            )
        )

    @staticmethod
    fn _dump_q[name: String, val: _DumpPath]() -> (Bool, _DumpPath):
        alias env_var = "DUMP_GPU_" + name.upper()

        @parameter
        if is_defined[env_var]():
            alias env_val = env_get_string[env_var]()

            @parameter
            if _is_bool_like[env_val]():
                alias env_bool_val = env_get_bool[env_var]()
                return env_bool_val, _DumpPath(env_bool_val)

            @parameter
            if _is_path_like(env_val):
                return True, _DumpPath(Path(env_val))

            constrained[
                False,
                "the environment variable '",
                env_var,
                (
                    "' is not a valid value. The value should either be"
                    " a boolean value or a path like value, but got '"
                ),
                env_val,
                "'",
            ]()
            return False, val

        @parameter
        if val.isa[Bool]():
            return val.unsafe_get[Bool](), val

        @parameter
        if val.isa[Path]():
            return val.unsafe_get[Path]() != Path(""), val

        @parameter
        if val.isa[StaticString]():
            return val.unsafe_get[StaticString]() != "", val

        return val.isa[fn () capturing -> Path](), val

    @staticmethod
    fn _cleanup_asm(s: StringSlice) -> String:
        return (
            String(s)
            .replace("\t// begin inline asm\n", "")
            .replace("\t// end inline asm\n", "")
            .replace("\t;;#ASMSTART\n", "")
            .replace("\t;;#ASMEND\n", "")
        )

    fn _expand_path(self, path: Path) -> Path:
        """If the path contains a `%` character, it is replaced with the module
        name. This allows one to dump multiple kernels which are disambiguated
        by the module name.
        """
        return String(path).replace("%", self._func_impl.module_name)

    fn _expand_path(self, path: StaticString) -> Path:
        """If the path contains a `%` character, it is replaced with the module
        name. This allows one to dump multiple kernels which are disambiguated
        by the module name.
        """
        return String(path).replace("%", self._func_impl.module_name)

    @no_inline
    fn dump_rep[
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
    ](self) raises:
        """Dumps various representations of the compiled device function.

        This method dumps the assembly, LLVM IR, and/or SASS code for the compiled
        device function based on the provided parameters. The output can be directed
        to stdout or written to files.

        Parameters:
            dump_asm: Controls dumping of assembly code. Can be a boolean, a file path,
                or a function returning a file path.
            dump_llvm: Controls dumping of LLVM IR. Can be a boolean, a file path,
                or a function returning a file path.
            _dump_sass: Controls dumping of SASS code (internal use). Can be a boolean,
                a file path, or a function returning a file path.

        Raises:
            If any file operations fail during the dumping process.

        Notes:

        When a path contains '%', it will be replaced with the module name to
        help disambiguate multiple kernel dumps.
        """

        fn get_asm() -> StaticString:
            @parameter
            if Self._emission_kind == "asm":
                return self._func_impl.asm
            return _compile_code_asm[
                func,
                emission_kind="asm",
                target=target,
            ]()

        @parameter
        if _ptxas_info_verbose:
            print(_ptxas_compile[target](String(get_asm()), options="-v"))

        alias dump_asm_tup = Self._dump_q["asm", dump_asm]()
        alias do_dump_asm = dump_asm_tup[0]
        alias dump_asm_val = dump_asm_tup[1]

        @parameter
        if do_dump_asm:
            var asm = self._cleanup_asm(get_asm())

            @parameter
            if dump_asm_val.isa[fn () capturing -> Path]():
                alias dump_asm_fn = dump_asm_val.unsafe_get[
                    fn () capturing -> Path
                ]()
                dump_asm_fn().write_text(asm)
            elif dump_asm_val.isa[Path]():
                self._expand_path(dump_asm_val.unsafe_get[Path]()).write_text(
                    asm
                )
            elif dump_asm_val.isa[StaticString]():
                self._expand_path(
                    dump_asm_val.unsafe_get[StaticString]()
                ).write_text(asm)
            else:
                print(asm)

        alias dump_sass_tup = Self._dump_q["sass", _dump_sass]()
        alias do_dump_sass = dump_sass_tup[0]
        alias dump_sass_val = dump_sass_tup[1]

        @parameter
        if do_dump_sass:
            var ptx = Self._cleanup_asm(get_asm())
            var sass = _to_sass[target](ptx)

            @parameter
            if dump_sass_val.isa[fn () capturing -> Path]():
                alias _dump_sass_fn = dump_sass_val.unsafe_get[
                    fn () capturing -> Path
                ]()
                _dump_sass_fn().write_text(sass)
            elif dump_sass_val.isa[Path]():
                self._expand_path(dump_sass_val.unsafe_get[Path]()).write_text(
                    sass
                )
            elif dump_sass_val.isa[StaticString]():
                self._expand_path(
                    dump_sass_val.unsafe_get[StaticString]()
                ).write_text(sass)
            else:
                print(sass)

        alias dump_llvm_tup = Self._dump_q["llvm", dump_llvm]()
        alias do_dump_llvm = dump_llvm_tup[0]
        alias dump_llvm_val = dump_llvm_tup[1]

        @parameter
        if do_dump_llvm:
            var llvm = _compile_code_asm[
                Self.func,
                emission_kind="llvm-opt",
                target=target,
            ]()

            @parameter
            if dump_llvm_val.isa[fn () capturing -> Path]():
                alias dump_llvm_fn = dump_llvm_val.unsafe_get[
                    fn () capturing -> Path
                ]()
                dump_llvm_fn().write_text(llvm)
            elif dump_llvm_val.isa[Path]():
                self._expand_path(dump_llvm_val.unsafe_get[Path]()).write_text(
                    llvm
                )
            elif dump_llvm_val.isa[StaticString]():
                self._expand_path(
                    dump_llvm_val.unsafe_get[StaticString]()
                ).write_text(llvm)
            else:
                print(llvm)

    @always_inline
    @parameter
    fn _call_with_pack[
        *Ts: AnyType
    ](
        self,
        ctx: DeviceContext,
        args: VariadicPack[_, _, AnyType, *Ts],
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        owned attributes: List[LaunchAttribute] = List[LaunchAttribute](),
        owned constant_memory: List[ConstantMemoryMapping] = List[
            ConstantMemoryMapping
        ](),
    ) raises:
        alias num_args = len(VariadicList(Ts))
        var num_captures = self._func_impl.num_captures
        alias populate = __type_of(self._func_impl).populate
        alias num_captures_static = 16

        var dense_args_addrs = stack_allocation[
            num_captures_static + num_args, UnsafePointer[NoneType]
        ]()

        if num_captures > num_captures_static:
            dense_args_addrs = UnsafePointer[UnsafePointer[NoneType]].alloc(
                num_captures + num_args
            )

        @parameter
        for i in range(num_args):
            var first_word_addr = UnsafePointer(to=args[i])
            dense_args_addrs[i] = first_word_addr.bitcast[NoneType]()

        if cluster_dim:
            attributes.append(
                LaunchAttribute.from_cluster_dim(cluster_dim.value())
            )

        if constant_memory:
            for i in range(len(constant_memory)):
                self._copy_to_constant_memory(constant_memory[i])

        # const char *AsyncRT_DeviceContext_enqueueFunctionDirect(const DeviceContext *ctx, const DeviceFunction *func,
        #                                                         uint32_t gridX, uint32_t gridY, uint32_t gridZ,
        #                                                         uint32_t blockX, uint32_t blockY, uint32_t blockZ,
        #                                                         uint32_t sharedMemBytes, void *attrs, uint32_t num_attrs,
        #                                                         void **args)

        if num_captures > 0:
            # Call the populate function to initialize the captured values in the arguments array.
            # The captured values are always at the end of the argument list.
            # This function (generated by the compiler) has to be inlined here
            # and be in the same scope as the user of dense_args_addr
            # (i.e. the following external_call).
            # Because this closure uses stack allocated ptrs
            # to store the captured values in dense_args_addrs, they need to
            # not go out of the scope before dense_args_addr is being use.
            var capture_args_start = dense_args_addrs.offset(num_args)
            populate(
                rebind[UnsafePointer[NoneType]](
                    capture_args_start.bitcast[NoneType]()
                )
            )

            _checked(
                external_call[
                    "AsyncRT_DeviceContext_enqueueFunctionDirect",
                    _CharPtr,
                    _DeviceContextPtr,
                    _DeviceFunctionPtr,
                    UInt32,
                    UInt32,
                    UInt32,
                    UInt32,
                    UInt32,
                    UInt32,
                    UInt32,
                    UnsafePointer[LaunchAttribute],
                    UInt32,
                    UnsafePointer[UnsafePointer[NoneType]],
                ](
                    ctx._handle,
                    self._handle,
                    grid_dim.x(),
                    grid_dim.y(),
                    grid_dim.z(),
                    block_dim.x(),
                    block_dim.y(),
                    block_dim.z(),
                    shared_mem_bytes.or_else(0),
                    attributes.unsafe_ptr(),
                    len(attributes),
                    dense_args_addrs,
                )
            )
        else:
            _checked(
                external_call[
                    "AsyncRT_DeviceContext_enqueueFunctionDirect",
                    _CharPtr,
                    _DeviceContextPtr,
                    _DeviceFunctionPtr,
                    UInt32,
                    UInt32,
                    UInt32,
                    UInt32,
                    UInt32,
                    UInt32,
                    UInt32,
                    UnsafePointer[LaunchAttribute],
                    UInt32,
                    UnsafePointer[UnsafePointer[NoneType]],
                ](
                    ctx._handle,
                    self._handle,
                    grid_dim.x(),
                    grid_dim.y(),
                    grid_dim.z(),
                    block_dim.x(),
                    block_dim.y(),
                    block_dim.z(),
                    shared_mem_bytes.or_else(0),
                    attributes.unsafe_ptr(),
                    len(attributes),
                    dense_args_addrs,
                )
            )

        if num_captures > num_captures_static:
            dense_args_addrs.free()

    @always_inline
    @parameter
    fn _call_with_pack_checked[
        *Ts: DevicePassable
    ](
        self,
        ctx: DeviceContext,
        args: VariadicPack[_, _, DevicePassable, *Ts],
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        owned attributes: List[LaunchAttribute] = List[LaunchAttribute](),
        owned constant_memory: List[ConstantMemoryMapping] = List[
            ConstantMemoryMapping
        ](),
    ) raises:
        alias num_args = len(VariadicList(Ts))

        @parameter
        if declared_arg_types:
            alias declared_num_args = len(
                VariadicList(declared_arg_types.value())
            )
            alias actual_num_args = len(VariadicList(Ts))
            constrained[
                declared_num_args == actual_num_args,
                "Wrong number of arguments to enqueue",
            ]()

            @parameter
            for i in range(num_args):
                alias declared_arg_type = declared_arg_types.value()[i]
                alias actual_arg_type = Ts[i]

                constrained[
                    _type_is_eq[
                        declared_arg_type, actual_arg_type.device_type
                    ](),
                    "Handed in wrong argument type for argument #",
                    String(i),
                ]()

        var num_captures = self._func_impl.num_captures
        alias populate = __type_of(self._func_impl).populate
        alias num_captures_static = 16

        var dense_args_addrs = stack_allocation[
            num_captures_static + num_args, UnsafePointer[NoneType]
        ]()

        if num_captures > num_captures_static:
            dense_args_addrs = UnsafePointer[UnsafePointer[NoneType]].alloc(
                num_captures + num_args
            )

        @parameter
        for i in range(num_args):
            var first_word_addr = UnsafePointer(to=args[i])
            dense_args_addrs[i] = first_word_addr.bitcast[NoneType]()

        if cluster_dim:
            attributes.append(
                LaunchAttribute.from_cluster_dim(cluster_dim.value())
            )

        if constant_memory:
            for i in range(len(constant_memory)):
                self._copy_to_constant_memory(constant_memory[i])

        # const char *AsyncRT_DeviceContext_enqueueFunctionDirect(const DeviceContext *ctx, const DeviceFunction *func,
        #                                                         uint32_t gridX, uint32_t gridY, uint32_t gridZ,
        #                                                         uint32_t blockX, uint32_t blockY, uint32_t blockZ,
        #                                                         uint32_t sharedMemBytes, void *attrs, uint32_t num_attrs,
        #                                                         void **args)

        if num_captures > 0:
            # Call the populate function to initialize the captured values in the arguments array.
            # The captured values are always at the end of the argument list.
            # This function (generated by the compiler) has to be inlined here
            # and be in the same scope as the user of dense_args_addr
            # (i.e. the following external_call).
            # Because this closure uses stack allocated ptrs
            # to store the captured values in dense_args_addrs, they need to
            # not go out of the scope before dense_args_addr is being use.
            var capture_args_start = dense_args_addrs.offset(num_args)
            populate(
                rebind[UnsafePointer[NoneType]](
                    capture_args_start.bitcast[NoneType]()
                )
            )

            _checked(
                external_call[
                    "AsyncRT_DeviceContext_enqueueFunctionDirect",
                    _CharPtr,
                    _DeviceContextPtr,
                    _DeviceFunctionPtr,
                    UInt32,
                    UInt32,
                    UInt32,
                    UInt32,
                    UInt32,
                    UInt32,
                    UInt32,
                    UnsafePointer[LaunchAttribute],
                    UInt32,
                    UnsafePointer[UnsafePointer[NoneType]],
                ](
                    ctx._handle,
                    self._handle,
                    grid_dim.x(),
                    grid_dim.y(),
                    grid_dim.z(),
                    block_dim.x(),
                    block_dim.y(),
                    block_dim.z(),
                    shared_mem_bytes.or_else(0),
                    attributes.unsafe_ptr(),
                    len(attributes),
                    dense_args_addrs,
                )
            )
        else:
            _checked(
                external_call[
                    "AsyncRT_DeviceContext_enqueueFunctionDirect",
                    _CharPtr,
                    _DeviceContextPtr,
                    _DeviceFunctionPtr,
                    UInt32,
                    UInt32,
                    UInt32,
                    UInt32,
                    UInt32,
                    UInt32,
                    UInt32,
                    UnsafePointer[LaunchAttribute],
                    UInt32,
                    UnsafePointer[UnsafePointer[NoneType]],
                ](
                    ctx._handle,
                    self._handle,
                    grid_dim.x(),
                    grid_dim.y(),
                    grid_dim.z(),
                    block_dim.x(),
                    block_dim.y(),
                    block_dim.z(),
                    shared_mem_bytes.or_else(0),
                    attributes.unsafe_ptr(),
                    len(attributes),
                    dense_args_addrs,
                )
            )

        if num_captures > num_captures_static:
            dense_args_addrs.free()

    @always_inline
    fn get_attribute(self, attr: Attribute) raises -> Int:
        """Retrieves a specific attribute value from the compiled device function.

        This method queries the device function for information about its resource
        requirements, execution capabilities, or other properties defined by the
        specified attribute.

        Args:
            attr: The attribute to query, defined in the Attribute enum.

        Returns:
            The integer value of the requested attribute.

        Raises:
            If the attribute query fails or the attribute is not supported.

        Example:

        ```mojo
        from gpu.host import Attribute, DeviceFunction

        var device_function = DeviceFunction(...)

        # Get the maximum number of threads per block for this function
        var max_threads = device_function.get_attribute(Attribute.MAX_THREADS_PER_BLOCK)
        ```
        """
        var result: Int32 = 0
        # const char *AsyncRT_DeviceFunction_getAttribute(int32_t *result, const DeviceFunction *func, int32_t attr_code)
        _checked(
            external_call[
                "AsyncRT_DeviceFunction_getAttribute",
                _CharPtr,
                UnsafePointer[Int32],
                _DeviceFunctionPtr,
                Int32,
            ](
                UnsafePointer(to=result),
                self._handle,
                attr.code,
            )
        )
        return Int(result)


struct DeviceExternalFunction:
    """Represents an external device function loaded from PTX/SASS assembly.

    This class provides functionality to load and execute pre-compiled GPU functions
    from assembly code rather than compiling them from Mojo source. This is useful
    for integrating with existing CUDA/HIP code or for using specialized assembly
    optimizations.

    The `DeviceExternalFunction` handles reference counting of the underlying device
    function handle and provides methods for launching the function on a GPU with
    specified execution configuration.
    """

    var _handle: _DeviceFunctionPtr
    """Internal handle to the native device function object."""

    fn __copyinit__(out self, existing: Self):
        """Creates a copy of an existing device function by incrementing its reference count.

        Args:
            existing: The device function to copy.
        """
        # Increment the reference count before copying the handle.
        #
        # void AsyncRT_DeviceFunction_retain(const DeviceFunction *ctx)
        external_call[
            "AsyncRT_DeviceFunction_retain",
            NoneType,
            _DeviceFunctionPtr,
        ](existing._handle)
        self._handle = existing._handle

    fn __moveinit__(out self, owned existing: Self):
        """Moves an existing device function into this one.

        Args:
            existing: The device function to move from.
        """
        self._handle = existing._handle

    fn __del__(owned self):
        """Releases resources associated with this device function."""
        # Decrement the reference count held by this struct.
        #
        # void AsyncRT_DeviceFunction_release(const DeviceFunction *ctx)
        external_call[
            "AsyncRT_DeviceFunction_release",
            NoneType,
            _DeviceFunctionPtr,
        ](self._handle)

    @doc_private
    @always_inline
    fn __init__(
        out self,
        ctx: DeviceContext,
        *,
        function_name: StringSlice,
        asm: StringSlice,
        func_attribute: OptionalReg[FuncAttribute] = None,
    ) raises:
        """Initializes a new device function from assembly code.

        Args:
            ctx: The device context to associate this function with.
            function_name: The name of the function in the assembly code.
            asm: The assembly code (PTX/SASS) containing the function.
            func_attribute: Optional function attributes like shared memory size.

        Raises:
            If function loading fails or if an unsupported attribute is provided.
        """
        var max_dynamic_shared_size_bytes: Int32 = -1
        if func_attribute:
            if (
                func_attribute.value().attribute
                == Attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES
            ):
                max_dynamic_shared_size_bytes = func_attribute.value().value
            else:
                raise Error(
                    "the function attribute '",
                    func_attribute.value().attribute,
                    "' is not currently supported",
                )

        # const char *AsyncRT_DeviceContext_loadFunction(
        #     const DeviceFunction **result, const DeviceContext *ctx,
        #     const char *moduleName, const char *functionName, const char *data,
        #     size_t dataLen, int32_t maxDynamicSharedBytes, const char *debugLevel,
        #     int32_t optimizationLevel)
        var module_name: String = ""
        var result = _DeviceFunctionPtr()
        var debug_level = String(DebugLevel)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_loadFunction",
                _CharPtr,
                UnsafePointer[_DeviceFunctionPtr],
                _DeviceContextPtr,
                _CharPtr,
                _CharPtr,
                _CharPtr,
                _SizeT,
                Int32,
                _CharPtr,
                Int32,
            ](
                UnsafePointer(to=result),
                ctx._handle,
                module_name.unsafe_ptr(),
                function_name.unsafe_ptr(),
                asm.unsafe_ptr(),
                len(asm),
                max_dynamic_shared_size_bytes,
                debug_level.unsafe_cstr_ptr().bitcast[UInt8](),
                Int(OptimizationLevel),
            )
        )
        self._handle = result

    @always_inline
    fn _copy_to_constant_memory(self, mapping: ConstantMemoryMapping) raises:
        """Copies data to constant memory for use by the device function.

        Args:
            mapping: A mapping describing the constant memory to copy.

        Raises:
            If the copy operation fails.
        """
        # const char *AsyncRT_DeviceFunction_copyToConstantMemory(
        #     const DeviceFunction *func,
        #     const void *name, size_t nameSize,
        #     const void *data, size_t dataSize)
        _checked(
            external_call[
                "AsyncRT_DeviceFunction_copyToConstantMemory",
                _CharPtr,
                _DeviceFunctionPtr,
                _CharPtr,
                _SizeT,
                _VoidPtr,
                _SizeT,
            ](
                self._handle,
                mapping.name.unsafe_ptr(),
                len(mapping.name),
                mapping.ptr,
                mapping.byte_count,
            )
        )

    @always_inline
    @parameter
    fn _call_with_pack[
        *Ts: AnyType
    ](
        self,
        ctx: DeviceContext,
        args: VariadicPack[_, _, AnyType, *Ts],
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        owned attributes: List[LaunchAttribute] = List[LaunchAttribute](),
        owned constant_memory: List[ConstantMemoryMapping] = List[
            ConstantMemoryMapping
        ](),
    ) raises:
        """Launches the device function with the specified arguments and configuration.

        Parameters:
            Ts: Types of the arguments to pass to the device function.

        Args:
            ctx: The device context to launch the function on.
            args: Arguments to pass to the device function.
            grid_dim: Grid dimensions for the kernel launch.
            block_dim: Block dimensions for the kernel launch.
            cluster_dim: Optional cluster dimensions for multi-GPU execution.
            shared_mem_bytes: Optional amount of shared memory to allocate.
            attributes: Optional list of additional launch attributes.
            constant_memory: Optional list of constant memory mappings.

        Raises:
            If the function launch fails.
        """
        alias num_args = len(VariadicList(Ts))

        var dense_args_addrs = stack_allocation[
            num_args, UnsafePointer[NoneType]
        ]()

        @parameter
        for i in range(num_args):
            var first_word_addr = UnsafePointer(to=args[i])
            dense_args_addrs[i] = first_word_addr.bitcast[NoneType]()

        if cluster_dim:
            attributes.append(
                LaunchAttribute.from_cluster_dim(cluster_dim.value())
            )

        if constant_memory:
            for i in range(len(constant_memory)):
                self._copy_to_constant_memory(constant_memory[i])

        # const char *AsyncRT_DeviceContext_enqueueFunctionDirect(const DeviceContext *ctx, const DeviceFunction *func,
        #                                                         uint32_t gridX, uint32_t gridY, uint32_t gridZ,
        #                                                         uint32_t blockX, uint32_t blockY, uint32_t blockZ,
        #                                                         uint32_t sharedMemBytes, void *attrs, uint32_t num_attrs,
        #                                                         void **args)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_enqueueFunctionDirect",
                _CharPtr,
                _DeviceContextPtr,
                _DeviceFunctionPtr,
                UInt32,
                UInt32,
                UInt32,
                UInt32,
                UInt32,
                UInt32,
                UInt32,
                UnsafePointer[LaunchAttribute],
                UInt32,
                UnsafePointer[UnsafePointer[NoneType]],
            ](
                ctx._handle,
                self._handle,
                grid_dim.x(),
                grid_dim.y(),
                grid_dim.z(),
                block_dim.x(),
                block_dim.y(),
                block_dim.z(),
                shared_mem_bytes.or_else(0),
                attributes.unsafe_ptr(),
                len(attributes),
                dense_args_addrs,
            )
        )

    @always_inline
    fn get_attribute(self, attr: Attribute) raises -> Int:
        """Retrieves a specific attribute of this device function.

        Args:
            attr: The attribute to query.

        Returns:
            The value of the requested attribute.

        Raises:
            If the attribute query fails.
        """
        var result: Int32 = 0
        # const char *AsyncRT_DeviceFunction_getAttribute(int32_t *result, const DeviceFunction *func, int32_t attr_code)
        _checked(
            external_call[
                "AsyncRT_DeviceFunction_getAttribute",
                _CharPtr,
                UnsafePointer[Int32],
                _DeviceFunctionPtr,
                Int32,
            ](
                UnsafePointer(to=result),
                self._handle,
                attr.code,
            )
        )
        return Int(result)


@register_passable
struct DeviceContext(CollectionElement):
    """Represents a single stream of execution on a particular accelerator
    (GPU).

    A `DeviceContext` serves as the low-level interface to the
    accelerator inside a MAX [custom operation](/max/custom-ops/) and provides
    methods for allocating buffers on the device, copying data between host and
    device, and for compiling and running functions (also known as kernels) on
    the device.

    The device context can be used as a
    [context manager](/mojo/manual/errors#use-a-context-manager). For example:

    ```mojo
    from gpu.host import DeviceContext
    from gpu import thread_idx

    fn kernel():
        print("hello from thread:", thread_idx.x, thread_idx.y, thread_idx.z)

    with DeviceContext() as ctx:
        ctx.enqueue_function[kernel](grid_dim=1, block_dim=(2, 2, 2))
        ctx.synchronize()
    ```

    A custom operation receives an opaque `DeviceContextPtr`, which provides
    a `get_device_context()` method to retrieve the device context:

    ```mojo
    from runtime.asyncrt import DeviceContextPtr

    @register("custom_op")
    struct CustomOp:
        @staticmethod
        fn execute(ctx_ptr: DeviceContextPtr) raises:
            var ctx = ctx_ptr.get_device_context()
            ctx.enqueue_function[kernel](grid_dim=1, block_dim=(2, 2, 2))
            ctx.synchronize()
    ```
    """

    alias device_info = DEFAULT_GPU
    """`gpu.info.Info` object for the default accelerator."""

    alias device_api = Self.device_info.api
    """Device API for the default accelerator (for example, "cuda" or
    "hip")."""

    var _handle: _DeviceContextPtr

    @always_inline
    fn __init__(
        out self,
        device_id: Int = 0,
        *,
        owned api: String = String(Self.device_api),
    ) raises:
        """Constructs a `DeviceContext` for the specified device.

        This initializer creates a new device context for the specified accelerator device.
        The device context provides an interface for interacting with the GPU, including
        memory allocation, data transfer, and kernel execution.

        Args:
            device_id: ID of the accelerator device. If not specified, uses
                the default accelerator (device 0).
            api: Requested device API (for example, "cuda" or "hip"). Defaults to the
                device API specified by the DeviceContext class.

        Raises:
            If device initialization fails or the specified device is not available.

        Example:

        ```mojo
        from gpu.host import DeviceContext

        # Create a context for the default GPU
        var ctx = DeviceContext()

        # Create a context for a specific GPU (device 1)
        var ctx2 = DeviceContext(1)
        ```
        """
        # const char *AsyncRT_DeviceContext_create(const DeviceContext **result, const char *api, int id)
        var result = _DeviceContextPtr()
        _checked(
            external_call[
                "AsyncRT_DeviceContext_create",
                _CharPtr,
                UnsafePointer[_DeviceContextPtr],
                UnsafePointer[c_char],
                Int32,
            ](
                UnsafePointer(to=result),
                api.unsafe_cstr_ptr(),
                device_id,
            )
        )
        self._handle = result

    fn _retain(self):
        # Increment the reference count.
        #
        # void AsyncRT_DeviceContext_retain(const DeviceContext *ctx)
        external_call[
            "AsyncRT_DeviceContext_retain",
            NoneType,
            _DeviceContextPtr,
        ](self._handle)

    @doc_private
    @implicit
    fn __init__(out self, handle: UnsafePointer[NoneType]):
        """Create a Mojo DeviceContext from a pointer to an existing C++ object.
        """
        self._handle = handle.bitcast[_DeviceContextCpp]()
        self._retain()

    @doc_private
    @implicit
    fn __init__(out self, ctx_ptr: _DeviceContextPtr):
        """Create a Mojo DeviceContext from a pointer to an existing C++ object.
        """
        self._handle = ctx_ptr
        self._retain()

    fn __copyinit__(out self, existing: Self):
        """Creates a copy of an existing device context by incrementing its reference count.

        This copy constructor creates a new reference to the same underlying device context
        by incrementing the reference count of the native context object. Both the original
        and the copy will refer to the same device context.

        Args:
            existing: The device context to copy.
        """
        # Increment the reference count before copying the handle.
        existing._retain()
        self._handle = existing._handle

    @always_inline
    fn copy(self) -> Self:
        """Explicitly constructs a copy of this device context.

        This method creates a new reference to the same underlying device context
        by incrementing the reference count of the native context object.

        Returns:
            A copy of this device context that refers to the same underlying context.
        """
        return self

    fn __del__(owned self):
        """Releases resources associated with this device context.

        This destructor decrements the reference count of the native device context.
        When the reference count reaches zero, the underlying resources are released,
        including any cached memory buffers and compiled device functions.
        """
        # Decrement the reference count held by this struct.
        #
        # void AsyncRT_DeviceContext_release(const DeviceContext *ctx)
        external_call[
            "AsyncRT_DeviceContext_release",
            NoneType,
            _DeviceContextPtr,
        ](self._handle)

    fn __enter__(owned self) -> Self:
        """Enables the use of DeviceContext in a 'with' statement context manager.

        This method allows DeviceContext to be used with Python-style context managers,
        which ensures proper resource management and cleanup when the context exits.

        Returns:
            The DeviceContext instance to be used within the context manager block.

        Example:

        ```mojo
        from gpu.host import DeviceContext

        # Using DeviceContext as a context manager
        with DeviceContext() as ctx:
            # Perform GPU operations
            # Resources are automatically released when exiting the block
        ```
        """
        return self^

    fn name(self) -> String:
        """Returns the device name, an ASCII string identifying this device,
        defined by the native device API.

        This method queries the underlying GPU device for its name, which typically
        includes the model and other identifying information. This can be useful for
        logging, debugging, or making runtime decisions based on the specific GPU hardware.

        Returns:
            A string containing the device name.

        Example:

        ```mojo
        from gpu.host import DeviceContext

        var ctx = DeviceContext()
        print("Running on device:", ctx.name())
        ```
        """
        # const char *AsyncRT_DeviceContext_deviceName(const DeviceContext *ctx)
        var name_ptr = external_call[
            "AsyncRT_DeviceContext_deviceName",
            _CharPtr,
            _DeviceContextPtr,
        ](
            self._handle,
        )
        return _string_from_owned_charptr(name_ptr)

    fn api(self) -> String:
        """Returns the name of the API used to program the device.

        This method queries the underlying device context to determine which GPU programming
        API is being used for the current device. This information is useful for writing
        code that can adapt to different GPU architectures and programming models.

        Possible values are:

        - "cpu": Generic host device (CPU).
        - "cuda": NVIDIA GPUs.
        - "hip": AMD GPUs.

        Returns:
            A string identifying the device API.

        Example:

        ```mojo
        from gpu.host import DeviceContext

        var ctx = DeviceContext()
        var api_name = ctx.api()
        print("Using device API:", api_name)

        # Conditionally execute code based on the API
        if api_name == "cuda":
            print("Running on NVIDIA GPU")
        elif api_name == "hip":
            print("Running on AMD GPU")
        ```
        """
        # void AsyncRT_DeviceContext_deviceApi(llvm::StringRef *result, const DeviceContext *ctx)
        var api_ptr = StaticString(ptr=UnsafePointer[Byte](), length=0)
        external_call[
            "AsyncRT_DeviceContext_deviceApi",
            NoneType,
            UnsafePointer[StaticString],
            _DeviceContextPtr,
        ](
            UnsafePointer(to=api_ptr),
            self._handle,
        )
        return String(api_ptr)

    fn enqueue_create_buffer[
        type: DType
    ](self, size: Int) raises -> DeviceBuffer[type]:
        """Enqueues a buffer creation using the `DeviceBuffer` constructor.

        For GPU devices, the space is allocated in the device's global memory.

        Parameters:
            type: The data type to be stored in the allocated memory.

        Args:
            size: The number of elements of `type` to allocate memory for.

        Returns:
            The allocated buffer.
        """
        return DeviceBuffer[type](self, size, _DeviceBufferMode._ASYNC)

    fn create_buffer_sync[
        type: DType
    ](self, size: Int) raises -> DeviceBuffer[type]:
        """Creates a buffer synchronously using the `DeviceBuffer` constructor.

        Parameters:
            type: The data type to be stored in the allocated memory.

        Args:
            size: The number of elements of `type` to allocate memory for.

        Returns:
            The allocated buffer.
        """
        var result = DeviceBuffer[type](self, size, _DeviceBufferMode._SYNC)
        self.synchronize()
        return result

    fn enqueue_create_host_buffer[
        type: DType
    ](self, size: Int) raises -> HostBuffer[type]:
        """Enqueues the creation of a HostBuffer.

        This function allocates memory on the host that is accessible by the device.
        The memory is page-locked (pinned) for efficient data transfer between host and device.

        Pinned memory is guaranteed to remain resident in the host's RAM, not be
        paged/swapped out to disk. Memory allocated normally (for example, using
        [`UnsafePointer.alloc()`](/mojo/stdlib/memory/unsafe_ptr/UnsafePointer#alloc))
        is pageable—individual pages of memory can be moved to secondary storage
        (disk/SSD) when main memory fills up.

        Using pinned memory allows devices to make fast transfers
        between host memory and device memory, because they can use direct
        memory access (DMA) to transfer data without relying on the CPU.

        Allocating too much pinned memory can cause performance issues, since it
        reduces the amount of memory available for other processes.

        Parameters:
            type: The data type to be stored in the allocated memory.

        Args:
            size: The number of elements of `type` to allocate memory for.

        Returns:
            A `HostBuffer` object that wraps the allocated host memory.

        Raises:
            If memory allocation fails or if the device context is invalid.

        Example:

        ```mojo
        from gpu.host import DeviceContext

        with DeviceContext() as ctx:
            # Allocate host memory accessible by the device
            var host_buffer = ctx.enqueue_create_host_buffer[DType.float32](1024)

            # Use the host buffer for device operations
            # ...
        ```
        """
        return HostBuffer[type](self, size)

    @always_inline
    fn compile_function[
        func_type: AnyTrivialRegType, //,
        func: func_type,
        *,
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
        _target: __mlir_type.`!kgen.target` = Self.device_info.target(),
    ](
        self,
        *,
        func_attribute: OptionalReg[FuncAttribute] = None,
        out result: DeviceFunction[
            func,
            Optional[__mlir_type[`!kgen.variadic<`, AnyType, `>`]](None),
            target=_target,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ],
    ) raises:
        """Compiles the provided function for execution on this device.

        Parameters:
            func_type: Type of the function.
            func: The function to compile.
            dump_asm: To dump the compiled assembly, pass `True`, or a file
                path to dump to, or a function returning a file path.
            dump_llvm: To dump the generated LLVM code, pass `True`, or a file
                path to dump to, or a function returning a file path.
            _dump_sass: Only runs on NVIDIA targets, and requires CUDA Toolkit
                to be installed. Pass `True`, or a file path to dump to, or a
                function returning a file path.
            _ptxas_info_verbose: Only runs on NVIDIA targets, and requires CUDA
                Toolkit to be installed. Changes `dump_asm` to output verbose
                PTX assembly (default `False`).
            _target: Change the target to different device type than the
                one associated with this `DeviceContext`.

        Args:
            func_attribute: An attribute to use when compiling the code (such
                as maximum shared memory size).

        Returns:
            The compiled function.
        """
        result = self.compile_function_unchecked[
            func,
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            _dump_sass=_dump_sass,
            _ptxas_info_verbose=_ptxas_info_verbose,
            _target=_target,
        ](func_attribute=func_attribute)

    @always_inline
    fn compile_function_unchecked[
        func_type: AnyTrivialRegType, //,
        func: func_type,
        *,
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
        _target: __mlir_type.`!kgen.target` = Self.device_info.target(),
    ](
        self,
        *,
        func_attribute: OptionalReg[FuncAttribute] = None,
        out result: DeviceFunction[
            func,
            Optional[__mlir_type[`!kgen.variadic<`, AnyType, `>`]](None),
            target=_target,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ],
    ) raises:
        """Compiles the provided function for execution on this device.

        Parameters:
            func_type: Type of the function.
            func: The function to compile.
            dump_asm: To dump the compiled assembly, pass `True`, or a file
                path to dump to, or a function returning a file path.
            dump_llvm: To dump the generated LLVM code, pass `True`, or a file
                path to dump to, or a function returning a file path.
            _dump_sass: Only runs on NVIDIA targets, and requires CUDA Toolkit
                to be installed. Pass `True`, or a file path to dump to, or a
                function returning a file path.
            _ptxas_info_verbose: Only runs on NVIDIA targets, and requires CUDA
                Toolkit to be installed. Changes `dump_asm` to output verbose
                PTX assembly (default `False`).
            _target: Change the target to different device type than the
                one associated with this `DeviceContext`.

        Args:
            func_attribute: An attribute to use when compiling the code (such
                as maximum shared memory size).

        Returns:
            The compiled function.
        """
        debug_assert(
            not func_attribute
            or func_attribute.value().attribute
            != Attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES
            or func_attribute.value().value
            <= self.device_info.shared_memory_per_multiprocessor,
            "Requested more than available shared memory.",
        )
        alias result_type = __type_of(result)
        result = result_type(
            self,
            func_attribute=func_attribute,
        )

        result.dump_rep[
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            _dump_sass=_dump_sass,
        ]()

    @always_inline
    fn compile_function_checked[
        func_type: AnyTrivialRegType,
        declared_arg_types: __mlir_type[`!kgen.variadic<`, AnyType, `>`], //,
        func: func_type,
        signature_func: fn (* args: * declared_arg_types) -> None,
        *,
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
        _target: __mlir_type.`!kgen.target` = Self.device_info.target(),
    ](
        self,
        *,
        func_attribute: OptionalReg[FuncAttribute] = None,
        out result: DeviceFunction[
            func,
            declared_arg_types,
            target=_target,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ],
    ) raises:
        """Compiles the provided function for execution on this device.

        Parameters:
            func_type: Type of the function.
            declared_arg_types: Types of the arguments to pass to the device function.
            func: The function to compile.
            signature_func: The function to compile, passed in again. Used for
                checking argument types later.
                Note: This will disappear in future versions.
            dump_asm: To dump the compiled assembly, pass `True`, or a file
                path to dump to, or a function returning a file path.
            dump_llvm: To dump the generated LLVM code, pass `True`, or a file
                path to dump to, or a function returning a file path.
            _dump_sass: Only runs on NVIDIA targets, and requires CUDA Toolkit
                to be installed. Pass `True`, or a file path to dump to, or a
                function returning a file path.
            _ptxas_info_verbose: Only runs on NVIDIA targets, and requires CUDA
                Toolkit to be installed. Changes `dump_asm` to output verbose
                PTX assembly (default `False`).
            _target: Change the target to different device type than the
                one associated with this `DeviceContext`.

        Args:
            func_attribute: An attribute to use when compiling the code (such
                as maximum shared memory size).

        Returns:
            The compiled function.
        """
        debug_assert(
            not func_attribute
            or func_attribute.value().attribute
            != Attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES
            or func_attribute.value().value
            <= self.device_info.shared_memory_per_multiprocessor,
            "Requested more than available shared memory.",
        )
        alias result_type = __type_of(result)
        result = result_type(
            self,
            func_attribute=func_attribute,
        )

        result.dump_rep[
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            _dump_sass=_dump_sass,
        ]()

    @always_inline
    fn compile_function_checked[
        func_type: AnyTrivialRegType,
        declared_arg_types: __mlir_type[`!kgen.variadic<`, AnyType, `>`], //,
        func: func_type,
        signature_func: fn (* args: * declared_arg_types) capturing -> None,
        *,
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
        _target: __mlir_type.`!kgen.target` = Self.device_info.target(),
    ](
        self,
        *,
        func_attribute: OptionalReg[FuncAttribute] = None,
        out result: DeviceFunction[
            func,
            declared_arg_types,
            target=_target,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ],
    ) raises:
        """Compiles the provided function for execution on this device.

        Parameters:
            func_type: Type of the function.
            declared_arg_types: Types of the arguments to pass to the device function.
            func: The function to compile.
            signature_func: The function to compile, passed in again. Used for
                checking argument types later.
                Note: This will disappear in future versions.
            dump_asm: To dump the compiled assembly, pass `True`, or a file
                path to dump to, or a function returning a file path.
            dump_llvm: To dump the generated LLVM code, pass `True`, or a file
                path to dump to, or a function returning a file path.
            _dump_sass: Only runs on NVIDIA targets, and requires CUDA Toolkit
                to be installed. Pass `True`, or a file path to dump to, or a
                function returning a file path.
            _ptxas_info_verbose: Only runs on NVIDIA targets, and requires CUDA
                Toolkit to be installed. Changes `dump_asm` to output verbose
                PTX assembly (default `False`).
            _target: Change the target to different device type than the
                one associated with this `DeviceContext`.

        Args:
            func_attribute: An attribute to use when compiling the code (such
                as maximum shared memory size).

        Returns:
            The compiled function.
        """
        debug_assert(
            not func_attribute
            or func_attribute.value().attribute
            != Attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES
            or func_attribute.value().value
            <= self.device_info.shared_memory_per_multiprocessor,
            "Requested more than available shared memory.",
        )
        alias result_type = __type_of(result)
        result = result_type(
            self,
            func_attribute=func_attribute,
        )

        result.dump_rep[
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            _dump_sass=_dump_sass,
        ]()

    fn load_function[
        func_type: AnyTrivialRegType, //,
        func: func_type,
    ](
        self,
        *,
        function_name: StringSlice,
        asm: StringSlice,
        func_attribute: OptionalReg[FuncAttribute] = None,
        out result: DeviceExternalFunction,
    ) raises:
        """Loads a pre-compiled device function from assembly code.

        This method loads an external GPU function from provided assembly code (PTX/SASS)
        rather than compiling it from Mojo source. This is useful for integrating with
        existing CUDA/HIP code or for using specialized assembly optimizations.

        Parameters:
            func_type: The type of the function to load.
            func: The function reference.

        Args:
            function_name: The name of the function in the assembly code.
            asm: The assembly code (PTX/SASS) containing the function.
            func_attribute: Optional attribute to apply to the function (such as
                maximum shared memory size).

        Returns:
            The loaded function is stored in the `result` parameter.

        Raises:
            If loading the function fails or the assembly code is invalid.

        Example:

        ```mojo
        from gpu.host import DeviceContext
        from gpu.host.device_context import DeviceExternalFunction

        fn func_signature(
            # Arguments being passed to the assembly code
            # e.g. two pointers and a length
            input: UnsafePointer[Float32],
            output: UnsafePointer[Float32],
            len: Int,
        ):
            # No body because that is passed as assembly code below.
            pass

        var ctx = DeviceContext()
        var ptx_code = "..."  # PTX assembly code
        var ext_func = ctx.load_function[func_signature](
            function_name="my_kernel",
            asm=ptx_code,
        )
        ```
        """
        alias result_type = __type_of(result)
        result = result_type(
            self,
            function_name=function_name,
            asm=asm,
            func_attribute=func_attribute,
        )

    @parameter
    @always_inline
    fn enqueue_function[
        func_type: AnyTrivialRegType, //,
        func: func_type,
        *Ts: AnyType,
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
    ](
        self,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        owned attributes: List[LaunchAttribute] = List[LaunchAttribute](),
        owned constant_memory: List[ConstantMemoryMapping] = List[
            ConstantMemoryMapping
        ](),
        func_attribute: OptionalReg[FuncAttribute] = None,
    ) raises:
        """Compiles and enqueues a kernel for execution on this device.

        Parameters:
            func_type: The type of the function to launch.
            func: The function to launch.
            Ts: The types of the arguments being passed to the function.
            dump_asm: To dump the compiled assembly, pass `True`, or a file
                path to dump to, or a function returning a file path.
            dump_llvm: To dump the generated LLVM code, pass `True`, or a file
                path to dump to, or a function returning a file path.
            _dump_sass: Only runs on NVIDIA targets, and requires CUDA Toolkit
                to be installed. Pass `True`, or a file path to dump to, or a
                function returning a file path.
            _ptxas_info_verbose: Only runs on NVIDIA targets, and requires CUDA
                Toolkit to be installed. Changes `dump_asm` to output verbose
                PTX assembly (default `False`).

        Args:
            args: Variadic arguments which are passed to the `func`.
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            cluster_dim: The cluster dimensions.
            shared_mem_bytes: Per-block memory shared between blocks.
            attributes: A `List` of launch attributes.
            constant_memory: A `List` of constant memory mappings.
            func_attribute: `CUfunction_attribute` enum.

        You can pass the function directly to `enqueue_function` without
        compiling it first:

        ```mojo
        from gpu.host import DeviceContext

        fn kernel():
            print("hello from the GPU")

        with DeviceContext() as ctx:
            ctx.enqueue_function[kernel](grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        If you are reusing the same function and parameters multiple times, this
        incurs 50-500 nanoseconds of overhead per enqueue, so you can compile it
        first to remove the overhead:

        ```mojo
        with DeviceContext() as ctx:
            var compile_func = ctx.compile_function[kernel]()
            ctx.enqueue_function(compile_func, grid_dim=1, block_dim=1)
            ctx.enqueue_function(compile_func, grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```
        """
        var gpu_kernel = self.compile_function[
            func,
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            _dump_sass=_dump_sass,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ](func_attribute=func_attribute)

        self._enqueue_function_unchecked(
            gpu_kernel,
            args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
        )

    @parameter
    @always_inline
    fn enqueue_function_unchecked[
        func_type: AnyTrivialRegType, //,
        func: func_type,
        *Ts: AnyType,
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
    ](
        self,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        owned attributes: List[LaunchAttribute] = List[LaunchAttribute](),
        owned constant_memory: List[ConstantMemoryMapping] = List[
            ConstantMemoryMapping
        ](),
        func_attribute: OptionalReg[FuncAttribute] = None,
    ) raises:
        """Compiles and enqueues a kernel for execution on this device.

        Parameters:
            func_type: The type of the function to launch.
            func: The function to launch.
            Ts: The types of the arguments being passed to the function.
            dump_asm: To dump the compiled assembly, pass `True`, or a file
                path to dump to, or a function returning a file path.
            dump_llvm: To dump the generated LLVM code, pass `True`, or a file
                path to dump to, or a function returning a file path.
            _dump_sass: Only runs on NVIDIA targets, and requires CUDA Toolkit
                to be installed. Pass `True`, or a file path to dump to, or a
                function returning a file path.
            _ptxas_info_verbose: Only runs on NVIDIA targets, and requires CUDA
                Toolkit to be installed. Changes `dump_asm` to output verbose
                PTX assembly (default `False`).

        Args:
            args: Variadic arguments which are passed to the `func`.
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            cluster_dim: The cluster dimensions.
            shared_mem_bytes: Per-block memory shared between blocks.
            attributes: A `List` of launch attributes.
            constant_memory: A `List` of constant memory mappings.
            func_attribute: `CUfunction_attribute` enum.

        You can pass the function directly to `enqueue_function` without
        compiling it first:

        ```mojo
        from gpu.host import DeviceContext

        fn kernel():
            print("hello from the GPU")

        with DeviceContext() as ctx:
            ctx.enqueue_function[kernel](grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        If you are reusing the same function and parameters multiple times, this
        incurs 50-500 nanoseconds of overhead per enqueue, so you can compile it
        first to remove the overhead:

        ```mojo
        with DeviceContext() as ctx:
            var compile_func = ctx.compile_function[kernel]()
            ctx.enqueue_function(compile_func, grid_dim=1, block_dim=1)
            ctx.enqueue_function(compile_func, grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```
        """
        var gpu_kernel = self.compile_function[
            func,
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            _dump_sass=_dump_sass,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ](func_attribute=func_attribute)

        self._enqueue_function_unchecked(
            gpu_kernel,
            args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
        )

    @parameter
    @always_inline
    fn enqueue_function[
        *Ts: AnyType
    ](
        self,
        f: DeviceFunction,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        owned attributes: List[LaunchAttribute] = List[LaunchAttribute](),
        owned constant_memory: List[ConstantMemoryMapping] = List[
            ConstantMemoryMapping
        ](),
    ) raises:
        """Enqueues a compiled function for execution on this device.

        Parameters:
            Ts: Argument types.

        Args:
            f: The compiled function to execute.
            args: Arguments to pass to the function.
            grid_dim: Dimensions of the compute grid, made up of thread
                blocks.
            block_dim: Dimensions of each thread block in the grid.
            cluster_dim: Dimensions of clusters (if the thread blocks are
                grouped into clusters).
            shared_mem_bytes: Amount of shared memory per thread block.
            attributes: Launch attributes.
            constant_memory: Constant memory mapping.

        You can pass the function directly to `enqueue_function` without
        compiling it first:

        ```mojo
        from gpu.host import DeviceContext

        fn kernel():
            print("hello from the GPU")

        with DeviceContext() as ctx:
            ctx.enqueue_function[kernel](grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        If you are reusing the same function and parameters multiple times, this
        incurs 50-500 nanoseconds of overhead per enqueue, so you can compile
        the function first to remove the overhead:

        ```mojo
        from gpu.host import DeviceContext

        with DeviceContext() as ctx:
            var compiled_func = ctx.compile_function[kernel]()
            ctx.enqueue_function(compiled_func, grid_dim=1, block_dim=1)
            ctx.enqueue_function(compiled_func, grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        """
        constrained[
            not f.declared_arg_types,
            (
                "A checked DeviceFunction should be called with"
                " `enqueue_function_checked`."
            ),
        ]()
        self._enqueue_function_unchecked(
            f,
            args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
        )

    @parameter
    @always_inline
    fn enqueue_function_unchecked[
        *Ts: AnyType
    ](
        self,
        f: DeviceFunction,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        owned attributes: List[LaunchAttribute] = List[LaunchAttribute](),
        owned constant_memory: List[ConstantMemoryMapping] = List[
            ConstantMemoryMapping
        ](),
    ) raises:
        """Enqueues a compiled function for execution on this device.

        Parameters:
            Ts: Argument types.

        Args:
            f: The compiled function to execute.
            args: Arguments to pass to the function.
            grid_dim: Dimensions of the compute grid, made up of thread
                blocks.
            block_dim: Dimensions of each thread block in the grid.
            cluster_dim: Dimensions of clusters (if the thread blocks are
                grouped into clusters).
            shared_mem_bytes: Amount of shared memory per thread block.
            attributes: Launch attributes.
            constant_memory: Constant memory mapping.

        You can pass the function directly to `enqueue_function` without
        compiling it first:

        ```mojo
        from gpu.host import DeviceContext

        fn kernel():
            print("hello from the GPU")

        with DeviceContext() as ctx:
            ctx.enqueue_function[kernel](grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        If you are reusing the same function and parameters multiple times, this
        incurs 50-500 nanoseconds of overhead per enqueue, so you can compile
        the function first to remove the overhead:

        ```mojo
        from gpu.host import DeviceContext

        with DeviceContext() as ctx:
            var compiled_func = ctx.compile_function[kernel]()
            ctx.enqueue_function(compiled_func, grid_dim=1, block_dim=1)
            ctx.enqueue_function(compiled_func, grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        """
        constrained[
            not f.declared_arg_types,
            (
                "A checked DeviceFunction should be called with"
                " `enqueue_function_checked`."
            ),
        ]()
        self._enqueue_function_unchecked(
            f,
            args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
        )

    @parameter
    @always_inline
    fn enqueue_function_checked[
        *Ts: DevicePassable
    ](
        self,
        f: DeviceFunction,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        owned attributes: List[LaunchAttribute] = List[LaunchAttribute](),
        owned constant_memory: List[ConstantMemoryMapping] = List[
            ConstantMemoryMapping
        ](),
    ) raises:
        """Enqueues a compiled function for execution on this device.

        Parameters:
            Ts: Argument types.

        Args:
            f: The compiled function to execute.
            args: Arguments to pass to the function.
            grid_dim: Dimensions of the compute grid, made up of thread
                blocks.
            block_dim: Dimensions of each thread block in the grid.
            cluster_dim: Dimensions of clusters (if the thread blocks are
                grouped into clusters).
            shared_mem_bytes: Amount of shared memory per thread block.
            attributes: Launch attributes.
            constant_memory: Constant memory mapping.

        You can pass the function directly to `enqueue_function` without
        compiling it first:

        ```mojo
        from gpu.host import DeviceContext

        fn kernel():
            print("hello from the GPU")

        with DeviceContext() as ctx:
            ctx.enqueue_function[kernel](grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        If you are reusing the same function and parameters multiple times, this
        incurs 50-500 nanoseconds of overhead per enqueue, so you can compile
        the function first to remove the overhead:

        ```mojo
        from gpu.host import DeviceContext

        with DeviceContext() as ctx:
            var compiled_func = ctx.compile_function[kernel]()
            ctx.enqueue_function(compiled_func, grid_dim=1, block_dim=1)
            ctx.enqueue_function(compiled_func, grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        """
        constrained[
            Bool(f.declared_arg_types), "Calling a non-checked function."
        ]()
        self._enqueue_function_checked(
            f,
            args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
        )

    @parameter
    @always_inline
    fn enqueue_function_checked[
        func_type: AnyTrivialRegType,
        declared_arg_types: __mlir_type[`!kgen.variadic<`, AnyType, `>`], //,
        func: func_type,
        signature_func: fn (* args: * declared_arg_types) -> None,
        *actual_arg_types: DevicePassable,
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
    ](
        self,
        *args: *actual_arg_types,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        owned attributes: List[LaunchAttribute] = List[LaunchAttribute](),
        owned constant_memory: List[ConstantMemoryMapping] = List[
            ConstantMemoryMapping
        ](),
        func_attribute: OptionalReg[FuncAttribute] = None,
    ) raises:
        """Compiles and enqueues a kernel for execution on this device.

        Parameters:
            func_type: The type of the function to launch.
            declared_arg_types: Types of the arguments to pass to the device function.
            func: The function to compile and launch.
            signature_func: The function to compile and launch, passed in
                again. Used for checking argument types later.
                Note: This will disappear in future versions.
            actual_arg_types: The types of the arguments being passed to the function.
            dump_asm: To dump the compiled assembly, pass `True`, or a file
                path to dump to, or a function returning a file path.
            dump_llvm: To dump the generated LLVM code, pass `True`, or a file
                path to dump to, or a function returning a file path.
            _dump_sass: Only runs on NVIDIA targets, and requires CUDA Toolkit
                to be installed. Pass `True`, or a file path to dump to, or a
                function returning a file path.
            _ptxas_info_verbose: Only runs on NVIDIA targets, and requires CUDA
                Toolkit to be installed. Changes `dump_asm` to output verbose
                PTX assembly (default `False`).

        Args:
            args: Variadic arguments which are passed to the `func`.
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            cluster_dim: The cluster dimensions.
            shared_mem_bytes: Per-block memory shared between blocks.
            attributes: A `List` of launch attributes.
            constant_memory: A `List` of constant memory mappings.
            func_attribute: `CUfunction_attribute` enum.

        You can pass the function directly to `enqueue_function` without
        compiling it first:

        ```mojo
        from gpu.host import DeviceContext

        fn kernel():
            print("hello from the GPU")

        with DeviceContext() as ctx:
            ctx.enqueue_function[kernel](grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        If you are reusing the same function and parameters multiple times, this
        incurs 50-500 nanoseconds of overhead per enqueue, so you can compile it
        first to remove the overhead:

        ```mojo
        with DeviceContext() as ctx:
            var compile_func = ctx.compile_function[kernel]()
            ctx.enqueue_function(compile_func, grid_dim=1, block_dim=1)
            ctx.enqueue_function(compile_func, grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```
        """
        var gpu_kernel = self.compile_function_checked[
            func,
            signature_func,
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            _dump_sass=_dump_sass,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ](func_attribute=func_attribute)

        self._enqueue_function_checked(
            gpu_kernel,
            args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
        )

    @parameter
    @always_inline
    fn enqueue_function_checked[
        func_type: AnyTrivialRegType,
        declared_arg_types: __mlir_type[`!kgen.variadic<`, AnyType, `>`], //,
        func: func_type,
        signature_func: fn (* args: * declared_arg_types) capturing -> None,
        *actual_arg_types: DevicePassable,
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
    ](
        self,
        *args: *actual_arg_types,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        owned attributes: List[LaunchAttribute] = List[LaunchAttribute](),
        owned constant_memory: List[ConstantMemoryMapping] = List[
            ConstantMemoryMapping
        ](),
        func_attribute: OptionalReg[FuncAttribute] = None,
    ) raises:
        """Compiles and enqueues a kernel for execution on this device. This
        overload takes in a function that's `capturing`.

        Parameters:
            func_type: The type of the function to launch.
            declared_arg_types: Types of the arguments to pass to the device function.
            func: The function to compile and launch.
            signature_func: The function to compile and launch, passed in
                again. Used for checking argument types later.
                Note: This will disappear in future versions.
            actual_arg_types: The types of the arguments being passed to the function.
            dump_asm: To dump the compiled assembly, pass `True`, or a file
                path to dump to, or a function returning a file path.
            dump_llvm: To dump the generated LLVM code, pass `True`, or a file
                path to dump to, or a function returning a file path.
            _dump_sass: Only runs on NVIDIA targets, and requires CUDA Toolkit
                to be installed. Pass `True`, or a file path to dump to, or a
                function returning a file path.
            _ptxas_info_verbose: Only runs on NVIDIA targets, and requires CUDA
                Toolkit to be installed. Changes `dump_asm` to output verbose
                PTX assembly (default `False`).

        Args:
            args: Variadic arguments which are passed to the `func`.
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            cluster_dim: The cluster dimensions.
            shared_mem_bytes: Per-block memory shared between blocks.
            attributes: A `List` of launch attributes.
            constant_memory: A `List` of constant memory mappings.
            func_attribute: `CUfunction_attribute` enum.

        You can pass the function directly to `enqueue_function` without
        compiling it first:

        ```mojo
        from gpu.host import DeviceContext

        fn kernel():
            print("hello from the GPU")

        with DeviceContext() as ctx:
            ctx.enqueue_function[kernel](grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        If you are reusing the same function and parameters multiple times, this
        incurs 50-500 nanoseconds of overhead per enqueue, so you can compile it
        first to remove the overhead:

        ```mojo
        with DeviceContext() as ctx:
            var compile_func = ctx.compile_function[kernel]()
            ctx.enqueue_function(compile_func, grid_dim=1, block_dim=1)
            ctx.enqueue_function(compile_func, grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```
        """
        var gpu_kernel = self.compile_function_checked[
            func,
            signature_func,
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            _dump_sass=_dump_sass,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ](func_attribute=func_attribute)

        self._enqueue_function_checked(
            gpu_kernel,
            args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
        )

    @parameter
    @always_inline
    fn _enqueue_function_unchecked[
        *Ts: AnyType
    ](
        self,
        f: DeviceFunction,
        args: VariadicPack[_, _, AnyType, *Ts],
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        owned attributes: List[LaunchAttribute] = List[LaunchAttribute](),
        owned constant_memory: List[ConstantMemoryMapping] = List[
            ConstantMemoryMapping
        ](),
    ) raises:
        f._call_with_pack(
            self,
            args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
        )

    @parameter
    @always_inline
    fn _enqueue_function_checked[
        *Ts: DevicePassable
    ](
        self,
        f: DeviceFunction,
        args: VariadicPack[_, _, DevicePassable, *Ts],
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        owned attributes: List[LaunchAttribute] = List[LaunchAttribute](),
        owned constant_memory: List[ConstantMemoryMapping] = List[
            ConstantMemoryMapping
        ](),
    ) raises:
        f._call_with_pack_checked(
            self,
            args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
        )

    @parameter
    @always_inline
    fn enqueue_function[
        *Ts: AnyType
    ](
        self,
        f: DeviceExternalFunction,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        owned attributes: List[LaunchAttribute] = List[LaunchAttribute](),
        owned constant_memory: List[ConstantMemoryMapping] = List[
            ConstantMemoryMapping
        ](),
    ) raises:
        """Enqueues an external device function for asynchronous execution on the GPU.

        This method schedules an external device function to be executed on the GPU with the
        specified execution configuration. The function and its arguments are passed to the
        underlying GPU runtime, which will execute them when resources are available.

        Parameters:
            Ts: The types of the arguments to be passed to the device function.

        Args:
            f: The external device function to execute.
            args: The arguments to pass to the device function.
            grid_dim: The dimensions of the grid (number of thread blocks).
            block_dim: The dimensions of each thread block (number of threads per block).
            cluster_dim: Optional dimensions for thread block clusters (for newer GPU architectures).
            shared_mem_bytes: Optional amount of dynamic shared memory to allocate per block.
            attributes: Optional list of launch attributes for fine-grained control.
            constant_memory: Optional list of constant memory mappings to use during execution.

        Raises:
            If there's an error enqueuing the function or if the function execution fails.

        Example:

        ```mojo
        from gpu.host import DeviceContext
        from gpu.host.device_context import DeviceExternalFunction

        # Create a device context and load an external function
        with DeviceContext() as ctx:
            var ext_func = DeviceExternalFunction("my_kernel")

            # Enqueue the external function with execution configuration
            ctx.enqueue_function(
                ext_func,
                grid_dim=Dim(16),
                block_dim=Dim(256)
            )

            # Wait for completion
            ctx.synchronize()
        ```
        """
        self._enqueue_external_function(
            f,
            args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
        )

    @parameter
    @always_inline
    fn _enqueue_external_function[
        *Ts: AnyType
    ](
        self,
        f: DeviceExternalFunction,
        args: VariadicPack[_, _, AnyType, *Ts],
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        owned attributes: List[LaunchAttribute] = List[LaunchAttribute](),
        owned constant_memory: List[ConstantMemoryMapping] = List[
            ConstantMemoryMapping
        ](),
    ) raises:
        f._call_with_pack(
            self,
            args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
        )

    @always_inline
    fn execution_time[
        func: fn (Self) raises capturing [_] -> None
    ](self, num_iters: Int) raises -> Int:
        """Measures the execution time of a function that takes a DeviceContext parameter.

        This method times the execution of a provided function that requires the
        DeviceContext as a parameter. It runs the function for the specified number
        of iterations and returns the total elapsed time in nanoseconds.

        Parameters:
            func: A function that takes a DeviceContext parameter to execute and time.

        Args:
            num_iters: The number of iterations to run the function.

        Returns:
            The total elapsed time in nanoseconds for all iterations.

        Raises:
            If the timer operations fail or if the function raises an exception.

        Example:

        ```mojo
        from gpu.host import DeviceContext

        fn gpu_operation(ctx: DeviceContext) raises capturing [_] -> None:
            # Perform some GPU operation using ctx
            pass

        with DeviceContext() as ctx:
            # Measure execution time of a function that uses the context
            var time_ns = ctx.execution_time[gpu_operation](10)
            print("Execution time for 10 iterations:", time_ns, "ns")
        ```
        """
        var timer_ptr = _DeviceTimerPtr()
        # const char* AsyncRT_DeviceContext_startTimer(const DeviceTimer **result, const DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_startTimer",
                _CharPtr,
                UnsafePointer[_DeviceTimerPtr],
                _DeviceContextPtr,
            ](
                UnsafePointer(to=timer_ptr),
                self._handle,
            )
        )
        var timer = _DeviceTimer(timer_ptr)
        for _ in range(num_iters):
            func(self)
        var elapsed_nanos: Int = 0
        # const char *AsyncRT_DeviceContext_stopTimer(int64_t *result, const DeviceContext *ctx, const DeviceTimer *timer)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_stopTimer",
                _CharPtr,
                UnsafePointer[Int],
                _DeviceContextPtr,
                _DeviceTimerPtr,
            ](
                UnsafePointer(to=elapsed_nanos),
                self._handle,
                timer._handle,
            )
        )
        return elapsed_nanos

    @always_inline
    fn execution_time[
        func: fn () raises capturing [_] -> None
    ](self, num_iters: Int) raises -> Int:
        """Measures the execution time of a function over multiple iterations.

        This method times the execution of a provided function that doesn't require
        the DeviceContext as a parameter. It runs the function for the specified
        number of iterations and returns the total elapsed time in nanoseconds.

        Parameters:
            func: A function with no parameters to execute and time.

        Args:
            num_iters: The number of iterations to run the function.

        Returns:
            The total elapsed time in nanoseconds for all iterations.

        Raises:
            If the timer operations fail or if the function raises an exception.

        Example:

        ```mojo
        from gpu.host import DeviceContext

        fn some_gpu_operation() raises capturing [_] -> None:
            # Perform some GPU operation
            pass

        with DeviceContext() as ctx:
            # Measure execution time of a function
            var time_ns = ctx.execution_time[some_gpu_operation]
            print("Execution time:", time_ns, "ns")
        ```
        """
        var timer_ptr = _DeviceTimerPtr()
        # const char* AsyncRT_DeviceContext_startTimer(const DeviceTimer **result, const DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_startTimer",
                _CharPtr,
                UnsafePointer[_DeviceTimerPtr],
                _DeviceContextPtr,
            ](
                UnsafePointer(to=timer_ptr),
                self._handle,
            )
        )
        var timer = _DeviceTimer(timer_ptr)
        for _ in range(num_iters):
            func()
        var elapsed_nanos: Int = 0
        # const char *AsyncRT_DeviceContext_stopTimer(int64_t *result, const DeviceContext *ctx, const DeviceTimer *timer)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_stopTimer",
                _CharPtr,
                UnsafePointer[Int],
                _DeviceContextPtr,
                _DeviceTimerPtr,
            ](
                UnsafePointer(to=elapsed_nanos),
                self._handle,
                timer._handle,
            )
        )
        return elapsed_nanos

    @always_inline
    fn execution_time_iter[
        func: fn (Self, Int) raises capturing [_] -> None
    ](self, num_iters: Int) raises -> Int:
        """Measures the execution time of a function that takes iteration index as input.

        This method times the execution of a provided function that requires both the
        DeviceContext and the current iteration index as parameters. It runs the function
        for the specified number of iterations, passing the iteration index to each call,
        and returns the total elapsed time in nanoseconds.

        Parameters:
            func: A function that takes the DeviceContext and an iteration index.

        Args:
            num_iters: The number of iterations to run the function.

        Returns:
            The total elapsed time in nanoseconds for all iterations.

        Raises:
            If the timer operations fail or if the function raises an exception.

        Example:

        ```mojo
        from gpu.host import DeviceContext

        var my_kernel = DeviceFunction(...)

        fn benchmark_kernel(ctx: DeviceContext, i: Int) raises capturing [_] -> None:
            # Run kernel with different parameters based on iteration
            ctx.enqueue_function[my_kernel](grid_dim=Dim(i), block_dim=Dim(256))

        with DeviceContext() as ctx:
            # Measure execution time with iteration awareness
            var time_ns = ctx.execution_time_iter[benchmark_kernel](10)
            print("Total execution time:", time_ns, "ns")
        ```
        """
        var timer_ptr = _DeviceTimerPtr()
        # const char* AsyncRT_DeviceContext_startTimer(const DeviceTimer **result, const DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_startTimer",
                _CharPtr,
                UnsafePointer[_DeviceTimerPtr],
                _DeviceContextPtr,
            ](
                UnsafePointer(to=timer_ptr),
                self._handle,
            )
        )
        var timer = _DeviceTimer(timer_ptr)
        for i in range(num_iters):
            func(self, i)
        var elapsed_nanos: Int = 0
        # const char *AsyncRT_DeviceContext_stopTimer(int64_t *result, const DeviceContext *ctx, const DeviceTimer *timer)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_stopTimer",
                _CharPtr,
                UnsafePointer[Int],
                _DeviceContextPtr,
                _DeviceTimerPtr,
            ](
                UnsafePointer(to=elapsed_nanos),
                self._handle,
                timer._handle,
            )
        )
        return elapsed_nanos

    @always_inline
    fn enqueue_copy[
        type: DType
    ](
        self,
        dst_buf: DeviceBuffer[type, **_],
        src_ptr: UnsafePointer[Scalar[type]],
    ) raises:
        """Enqueues an async copy from the host to the provided device
        buffer. The number of bytes copied is determined by the size of the
        device buffer.

        Parameters:
            type: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_ptr: Host pointer to copy from.
        """
        # const char * AsyncRT_DeviceContext_HtoD_async(const DeviceContext *ctx, const DeviceBuffer *dst, const void *src)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_HtoD_async",
                _CharPtr,
                _DeviceContextPtr,
                _DeviceBufferPtr,
                UnsafePointer[Scalar[type]],
            ](
                self._handle,
                dst_buf._handle,
                src_ptr,
            )
        )

    @always_inline
    fn enqueue_copy[
        type: DType
    ](
        self,
        dst_buf: HostBuffer[type, **_],
        src_ptr: UnsafePointer[Scalar[type]],
    ) raises:
        """Enqueues an async copy from the host to the provided device
        buffer. The number of bytes copied is determined by the size of the
        device buffer.

        Parameters:
            type: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_ptr: Host pointer to copy from.
        """
        # const char * AsyncRT_DeviceContext_HtoD_async(const DeviceContext *ctx, const DeviceBuffer *dst, const void *src)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_HtoD_async",
                _CharPtr,
                _DeviceContextPtr,
                _DeviceBufferPtr,
                UnsafePointer[Scalar[type]],
            ](
                self._handle,
                dst_buf._handle,
                src_ptr,
            )
        )

    @always_inline
    fn enqueue_copy[
        type: DType
    ](
        self,
        dst_ptr: UnsafePointer[Scalar[type]],
        src_buf: DeviceBuffer[type, **_],
    ) raises:
        """Enqueues an async copy from the device to the host. The
        number of bytes copied is determined by the size of the device buffer.

        Parameters:
            type: Type of the data being copied.

        Args:
            dst_ptr: Host pointer to copy to.
            src_buf: Device buffer to copy from.
        """
        # const char * AsyncRT_DeviceContext_DtoH_async(const DeviceContext *ctx, void *dst, const DeviceBuffer *src)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_DtoH_async",
                _CharPtr,
                _DeviceContextPtr,
                UnsafePointer[Scalar[type]],
                _DeviceBufferPtr,
            ](
                self._handle,
                dst_ptr,
                src_buf._handle,
            )
        )

    @always_inline
    fn enqueue_copy[
        type: DType
    ](
        self,
        dst_ptr: UnsafePointer[Scalar[type]],
        src_buf: HostBuffer[type, **_],
    ) raises:
        """Enqueues an async copy from the device to the host. The
        number of bytes copied is determined by the size of the device buffer.

        Parameters:
            type: Type of the data being copied.

        Args:
            dst_ptr: Host pointer to copy to.
            src_buf: Device buffer to copy from.
        """
        # const char * AsyncRT_DeviceContext_DtoH_async(const DeviceContext *ctx, void *dst, const DeviceBuffer *src)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_DtoH_async",
                _CharPtr,
                _DeviceContextPtr,
                UnsafePointer[Scalar[type]],
                _DeviceBufferPtr,
            ](
                self._handle,
                dst_ptr,
                src_buf._handle,
            )
        )

    @always_inline
    fn enqueue_copy[
        type: DType
    ](
        self,
        dst_ptr: UnsafePointer[Scalar[type]],
        src_ptr: UnsafePointer[Scalar[type]],
        size: Int,
    ) raises:
        """Enqueues an async copy of `size` elements from a device pointer to
        another device pointer.

        Parameters:
            type: Type of the data being copied.

        Args:
            dst_ptr: Host pointer to copy to.
            src_ptr: Device pointer to copy from.
            size: Number of elements (of the specified `DType`) to copy.
        """
        # Not directly implemented on DeviceContext, wrap in buffers first
        var dst_buf = DeviceBuffer(self, dst_ptr, size, owning=False)
        var src_buf = DeviceBuffer(self, src_ptr, size, owning=False)
        self.enqueue_copy(dst_buf, src_buf)

    @always_inline
    fn enqueue_copy[
        type: DType
    ](
        self, dst_buf: DeviceBuffer[type, **_], src_buf: DeviceBuffer[type, **_]
    ) raises:
        """Enqueues an async copy from one device buffer to another. The amount
        of data transferred is determined by the size of the destination buffer.

        Parameters:
            type: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_buf: Device buffer to copy from. Must be at least as large as
                `dst`.
        """
        # const char * AsyncRT_DeviceContext_DtoD_async(const DeviceContext *ctx, const DeviceBuffer *dst, const DeviceBuffer *src)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_DtoD_async",
                _CharPtr,
                _DeviceContextPtr,
                _DeviceBufferPtr,
                _DeviceBufferPtr,
            ](
                self._handle,
                dst_buf._handle,
                src_buf._handle,
            )
        )

    @always_inline
    fn enqueue_copy[
        type: DType
    ](
        self, dst_buf: DeviceBuffer[type, **_], src_buf: HostBuffer[type, **_]
    ) raises:
        """Enqueues an async copy from one device buffer to another. The amount
        of data transferred is determined by the size of the destination buffer.

        Parameters:
            type: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_buf: Device buffer to copy from. Must be at least as large as
                `dst`.
        """
        # const char * AsyncRT_DeviceContext_DtoD_async(const DeviceContext *ctx, const DeviceBuffer *dst, const DeviceBuffer *src)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_DtoD_async",
                _CharPtr,
                _DeviceContextPtr,
                _DeviceBufferPtr,
                _DeviceBufferPtr,
            ](
                self._handle,
                dst_buf._handle,
                src_buf._handle,
            )
        )

    @always_inline
    fn enqueue_copy[
        type: DType
    ](
        self, dst_buf: HostBuffer[type, **_], src_buf: DeviceBuffer[type, **_]
    ) raises:
        """Enqueues an async copy from one device buffer to another. The amount
        of data transferred is determined by the size of the destination buffer.

        Parameters:
            type: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_buf: Device buffer to copy from. Must be at least as large as
                `dst`.
        """
        # const char * AsyncRT_DeviceContext_DtoD_async(const DeviceContext *ctx, const DeviceBuffer *dst, const DeviceBuffer *src)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_DtoD_async",
                _CharPtr,
                _DeviceContextPtr,
                _DeviceBufferPtr,
                _DeviceBufferPtr,
            ](
                self._handle,
                dst_buf._handle,
                src_buf._handle,
            )
        )

    @always_inline
    fn enqueue_copy[
        type: DType
    ](
        self, dst_buf: HostBuffer[type, **_], src_buf: HostBuffer[type, **_]
    ) raises:
        """Enqueues an async copy from one device buffer to another. The amount
        of data transferred is determined by the size of the destination buffer.

        Parameters:
            type: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_buf: Device buffer to copy from. Must be at least as large as
                `dst`.
        """
        # const char * AsyncRT_DeviceContext_DtoD_async(const DeviceContext *ctx, const DeviceBuffer *dst, const DeviceBuffer *src)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_DtoD_async",
                _CharPtr,
                _DeviceContextPtr,
                _DeviceBufferPtr,
                _DeviceBufferPtr,
            ](
                self._handle,
                dst_buf._handle,
                src_buf._handle,
            )
        )

    @always_inline
    fn enqueue_memset[
        type: DType
    ](self, dst: DeviceBuffer[type, **_], val: Scalar[type]) raises:
        """Enqueues an async memset operation, setting all of the elements in
        the destination device buffer to the specified value.

        Parameters:
            type: Type of the data stored in the buffer.

        Args:
            dst: Destination buffer.
            val: Value to set all elements of `dst` to.
        """
        alias bitwidth = bitwidthof[type]()
        constrained[
            bitwidth == 8 or bitwidth == 16 or bitwidth == 32 or bitwidth == 64,
            "bitwidth of memset type must be one of [8,16,32,64]",
        ]()
        var value: UInt64

        @parameter
        if bitwidth == 8:
            value = UInt64(Int(bitcast[DType.uint8, 1](val)))
        elif bitwidth == 16:
            value = UInt64(Int(bitcast[DType.uint16, 1](val)))
        elif bitwidth == 32:
            value = UInt64(bitcast[DType.uint32, 1](val))
        else:
            value = bitcast[DType.uint64, 1](val)

        # const char *AsyncRT_DeviceContext_setMemory_async(const DeviceContext *ctx, const DeviceBuffer *dst, uint64_t val, size_t val_size)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_setMemory_async",
                _CharPtr,
                _DeviceContextPtr,
                _DeviceBufferPtr,
                UInt64,
                _SizeT,
            ](
                self._handle,
                dst._handle,
                value,
                sizeof[type](),
            )
        )

    fn enqueue_memset[
        type: DType
    ](self, dst: HostBuffer[type, **_], val: Scalar[type]) raises:
        """Enqueues an async memset operation, setting all of the elements in
        the destination host buffer to the specified value.

        Parameters:
            type: Type of the data stored in the buffer.

        Args:
            dst: Destination buffer.
            val: Value to set all elements of `dst` to.
        """
        alias bitwidth = bitwidthof[type]()
        constrained[
            bitwidth == 8 or bitwidth == 16 or bitwidth == 32 or bitwidth == 64,
            "bitwidth of memset type must be one of [8,16,32,64]",
        ]()
        var value: UInt64

        @parameter
        if bitwidth == 8:
            value = UInt64(Int(bitcast[DType.uint8, 1](val)))
        elif bitwidth == 16:
            value = UInt64(Int(bitcast[DType.uint16, 1](val)))
        elif bitwidth == 32:
            value = UInt64(bitcast[DType.uint32, 1](val))
        else:
            value = bitcast[DType.uint64, 1](val)

        # const char *AsyncRT_DeviceContext_setMemory_async(const DeviceContext *ctx, const DeviceBuffer *dst, uint64_t val, size_t val_size)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_setMemory_async",
                _CharPtr,
                _DeviceContextPtr,
                _DeviceBufferPtr,
                UInt64,
                _SizeT,
            ](
                self._handle,
                dst._handle,
                value,
                sizeof[type](),
            )
        )

    @doc_private
    @always_inline
    fn stream(self) raises -> DeviceStream:
        return DeviceStream(self)

    @always_inline
    fn synchronize(self) raises:
        """Blocks until all asynchronous calls on the stream associated with
        this device context have completed.

        This should never be necessary when writing a custom operation."""
        # const char * AsyncRT_DeviceContext_synchronize(const DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_synchronize",
                _CharPtr,
                _DeviceContextPtr,
            ](
                self._handle,
            )
        )

    fn enqueue_wait_for(self, other: DeviceContext) raises:
        """Enqueues a wait operation for another device context to complete its work.

        This method creates a dependency between two device contexts, ensuring that operations
        in the current context will not begin execution until all previously enqueued operations
        in the other context have completed. This is useful for synchronizing work across
        multiple devices or streams.

        Args:
            other: The device context whose operations must complete before operations in this context can proceed.

        Raises:
            If there's an error enqueuing the wait operation or if the operation
            is not supported by the underlying device API.

        Example:

        ```mojo
        from gpu.host import DeviceContext

        # Create two device contexts
        var ctx1 = DeviceContext(0)  # First GPU
        var ctx2 = DeviceContext(1)  # Second GPU

        # Enqueue operations on ctx1
        # ...

        # Make ctx2 wait for ctx1 to complete before proceeding
        ctx2.enqueue_wait_for(ctx1)

        # Enqueue operations on ctx2 that depend on ctx1's completion
        # ...
        ```
        """
        # const char * AsyncRT_DeviceContext_enqueue_wait_for_context(const DeviceContext *ctx, const DeviceContext *other)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_enqueue_wait_for_context",
                _CharPtr,
                _DeviceContextPtr,
                _DeviceContextPtr,
            ](self._handle, other._handle)
        )

    @always_inline
    fn get_api_version(self) raises -> Int:
        """Returns the API version associated with this device.

        This method retrieves the version number of the GPU driver currently installed
        on the system for the device associated with this context. The version is
        returned as an integer that can be used to check compatibility with specific
        features or to troubleshoot driver-related issues.

        Returns:
            An integer representing the driver version.

        Raises:
            If the driver version cannot be retrieved or if the device context is invalid.

        Example:

        ```mojo
        from gpu.host import DeviceContext

        with DeviceContext() as ctx:
            # Get the API version
            var api_version = ctx.get_api_version()
            print("GPU API version:", api_version)
        ```
        """
        var value: Int32 = 0
        # const char * AsyncRT_DeviceContext_getApiVersion(int *result, const DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_getApiVersion",
                _CharPtr,
                _IntPtr,
                _DeviceContextPtr,
            ](
                UnsafePointer(to=value),
                self._handle,
            )
        )
        return Int(value)

    @always_inline
    fn get_attribute(self, attr: DeviceAttribute) raises -> Int:
        """Returns the specified attribute for this device.

        Use the aliases defined by
        [DeviceAttribute](/mojo/stdlib/gpu/host/device_attribute/DeviceAttribute)
        to specify attributes. For example:

        ```mojo
        from gpu.host import DeviceAttribute, DeviceContext

        def main():
            var ctx = DeviceContext()
            var attr = DeviceAttribute.MAX_BLOCKS_PER_MULTIPROCESSOR
            var max_blocks = ctx.get_attribute(attr)
            print(max_blocks)
        ```

        Args:
            attr: The device attribute to query.

        Returns:
            The value for `attr` on this device.
        """
        var value: Int32 = 0
        # const char * AsyncRT_DeviceContext_getAttribute(int *result, const DeviceContext *ctx, int attr)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_getAttribute",
                _CharPtr,
                _IntPtr,
                _DeviceContextPtr,
                Int,
            ](
                UnsafePointer(to=value),
                self._handle,
                Int(attr._value),
            )
        )
        return Int(value)

    @always_inline
    fn is_compatible(self) -> Bool:
        """Returns True if this device is compatible with MAX.

        This method checks whether the current device is compatible with the
        Modular Accelerated Execution (MAX) runtime. It's useful for validating
        that the device can execute the compiled code before attempting operations.

        Returns:
            True if the device is compatible with MAX, False otherwise.

        Example:

        ```mojo
        from gpu.host import DeviceContext

        var ctx = DeviceContext()
        print("Device is compatible with MAX:", ctx.is_compatible())
        ```
        """
        # const char * AsyncRT_DeviceContext_isCompatible(const DeviceContext *ctx)
        try:
            _checked(
                external_call[
                    "AsyncRT_DeviceContext_isCompatible",
                    _CharPtr,
                    _DeviceContextPtr,
                ](
                    self._handle,
                )
            )
            return True
        except:
            return False

    @always_inline
    fn id(self) raises -> Int64:
        """Returns the ID associated with this device.

        This method retrieves the unique identifier for the current device.
        Device IDs are used to distinguish between multiple devices in a system
        and are often needed for multi-GPU programming.

        Returns:
            The unique device ID as an Int64.

        Raises:
            If there's an error retrieving the device ID.

        Example:

        ```mojo
        var ctx = DeviceContext()
        try:
            var device_id = ctx.id()
            print("Using device with ID:", device_id)
        except:
            print("Failed to get device ID")
        ```
        """
        # int64_t AsyncRT_DeviceContext_id(const DeviceContext *ctx)
        return external_call[
            "AsyncRT_DeviceContext_id", Int64, _DeviceContextPtr
        ](self._handle)

    @doc_private
    @always_inline
    fn compute_capability(self) raises -> Int:
        """Returns the compute capability of this NVIDIA GPU device.

        This internal method retrieves the compute capability version of the current
        NVIDIA GPU device. The compute capability is a version number that identifies
        the features supported by the CUDA hardware.

        Returns:
            The compute capability as an integer (e.g., 70 for 7.0, 86 for 8.6).

        Raises:
            If there's an error retrieving the compute capability.

        Notes:

        This is a private method intended for internal use only.
        """
        var compute_capability: Int32 = 0
        # const char * AsyncRT_DeviceContext_computeCapability(int32_t *result, const DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_computeCapability",
                _CharPtr,
                _IntPtr,
                _DeviceContextPtr,
            ](UnsafePointer(to=compute_capability), self._handle)
        )
        return Int(compute_capability)

    @always_inline
    fn get_memory_info(self) raises -> (_SizeT, _SizeT):
        """Returns the free and total memory size for this device.

        This method queries the current state of device memory, providing information
        about how much memory is available and the total memory capacity of the device.
        This is useful for memory management and determining if there's enough space
        for planned operations.

        Returns:
            A tuple of (free memory, total memory) in bytes.

        Raises:
            If there's an error retrieving the memory information.

        Example:

        ```mojo
        from gpu.host import DeviceContext

        var ctx = DeviceContext()
        try:
            (free, total) = ctx.get_memory_info()
            print("Free memory:", free / (1024*1024), "MB")
            print("Total memory:", total / (1024*1024), "MB")
        except:
            print("Failed to get memory information")
        ```
        """
        var free = _SizeT(0)
        var total = _SizeT(0)
        # const char *AsyncRT_DeviceContext_getMemoryInfo(const DeviceContext *ctx, size_t *free, size_t *total)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_getMemoryInfo",
                _CharPtr,
                _DeviceContextPtr,
                UnsafePointer[_SizeT],
                UnsafePointer[_SizeT],
            ](
                self._handle,
                UnsafePointer(to=free),
                UnsafePointer(to=total),
            )
        )

        return (free, total)

    @always_inline
    fn can_access(self, peer: DeviceContext) raises -> Bool:
        """Returns True if this device can access the identified peer device.

        This method checks whether the current device can directly access memory on
        the specified peer device. Peer-to-peer access allows for direct memory transfers
        between devices without going through host memory, which can significantly
        improve performance in multi-GPU scenarios.

        Args:
            peer: The peer device to check for accessibility.

        Returns:
            True if the current device can access the peer device, False otherwise.

        Raises:
            If there's an error checking peer access capability.

        Example:

        ```mojo
        from gpu.host import DeviceContext
        var ctx1 = DeviceContext(0)  # First GPU
        var ctx2 = DeviceContext(1)  # Second GPU

        try:
            if ctx1.can_access(ctx2):
                print("Direct peer access is possible")
                ctx1.enable_peer_access(ctx2)
            else:
                print("Direct peer access is not supported")
        except:
            print("Failed to check peer access capability")
        ```
        """
        var result: Bool = False
        # const char *AsyncRT_DeviceContext_canAccess(bool *result, const DeviceContext *ctx, const DeviceContext *peer)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_canAccess",
                _CharPtr,
                UnsafePointer[Bool],
                _DeviceContextPtr,
                _DeviceContextPtr,
            ](
                UnsafePointer(to=result),
                self._handle,
                peer._handle,
            )
        )
        return result

    @always_inline
    fn enable_peer_access(self, peer: DeviceContext) raises:
        """Enables direct memory access to the peer device.

        This method establishes peer-to-peer access from the current device to the
        specified peer device. Once enabled, the current device can directly read from
        and write to memory allocated on the peer device without going through host memory,
        which can significantly improve performance for multi-GPU operations.

        Args:
            peer: The peer device to enable access to.

        Raises:
            If there's an error enabling peer access or if peer access is not supported
            between the devices.

        Notes:

        - It's recommended to call `can_access()` first to check if peer access is possible.
        - Peer access is not always symmetric; you may need to enable access in both directions.

        Example:

        ```mojo
        from gpu.host import DeviceContext

        var ctx1 = DeviceContext(0)  # First GPU
        var ctx2 = DeviceContext(1)  # Second GPU

        try:
            if ctx1.can_access(ctx2):
                ctx1.enable_peer_access(ctx2)
                print("Peer access enabled from device 0 to device 1")

                # For bidirectional access
                if ctx2.can_access(ctx1):
                    ctx2.enable_peer_access(ctx1)
                    print("Peer access enabled from device 1 to device 0")
            else:
                print("Peer access not supported between these devices")
        except:
            print("Failed to enable peer access")
        ```
        """
        # const char *AsyncRT_DeviceContext_enablePeerAccess(const DeviceContext *ctx, const DeviceContext *peer)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_enablePeerAccess",
                _CharPtr,
                _DeviceContextPtr,
                _DeviceContextPtr,
            ](
                self._handle,
                peer._handle,
            )
        )

    fn supports_multicast(self) raises -> Bool:
        """Returns True if this device supports multicast memory mappings.

        Returns:
            True if the current device supports multicast memory, False otherwise.

        Raises:
            If there's an error checking peer access capability.
        """
        var result: Bool = False
        # const char *AsyncRT_DeviceContext_supportsMulticast(bool *result, const DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_supportsMulticast",
                _CharPtr,
                UnsafePointer[Bool],
                _DeviceContextPtr,
            ](
                UnsafePointer(to=result),
                self._handle,
            )
        )
        return result

    @staticmethod
    @always_inline
    fn number_of_devices(*, api: String = String(Self.device_api)) -> Int:
        """Returns the number of devices available that support the specified API.

        This function queries the system for available devices that support the
        requested API (such as CUDA or HIP). It's useful for determining how many
        accelerators are available before allocating resources or distributing work.

        Args:
            api: Requested device API (for example, "cuda" or "hip"). Defaults to the
                device API specified by the DeviceContext class.

        Returns:
            The number of available devices supporting the specified API.

        Example:

        ```mojo
        from gpu.host import DeviceContext

        # Get number of CUDA devices
        var num_cuda_devices = DeviceContext.number_of_devices(api="cuda")

        # Get number of devices for the default API
        var num_devices = DeviceContext.number_of_devices()
        ```
        """
        # int32_t *AsyncRT_DeviceContext_numberOfDevices(const char* kind)
        return Int(
            external_call[
                "AsyncRT_DeviceContext_numberOfDevices",
                Int32,
                _CharPtr,
            ](
                api.unsafe_ptr(),
            )
        )


struct DeviceMulticastBuffer[type: DType]:
    """Represents a muticast memory object enables special memory operations to be broadcast
    across a group of devices.

    Parameters:
        type: Data type to be stored in the associated memory regions.
    """

    var _handle: _DeviceMulticastBufferPtr

    @doc_private
    fn __init__(
        out self,
        owned contexts: List[DeviceContext],
        size: Int,
    ) raises:
        alias elem_size = sizeof[type]()
        var handle = _DeviceMulticastBufferPtr()

        var ctxs_len = len(contexts)
        var ctxs = UnsafePointer[_DeviceContextPtr].alloc(ctxs_len)
        for i in range(ctxs_len):
            ctxs[i] = contexts[i]._handle

        # const char* AsyncRT_DeviceMulticastBuffer_allocate(const DeviceMulticastBuffer **result, size_t ctxsLen, const DeviceContext **ctxs, size_t len, size_t elemSize)
        _checked(
            external_call[
                "AsyncRT_DeviceMulticastBuffer_allocate",
                _CharPtr,
                UnsafePointer[_DeviceMulticastBufferPtr],
                _SizeT,
                UnsafePointer[_DeviceContextPtr],
                _SizeT,
                _SizeT,
            ](
                UnsafePointer(to=handle),
                ctxs_len,
                ctxs,
                size,
                elem_size,
            )
        )

        self._handle = handle

    @doc_private
    fn unicast_buffer_for(
        self, ctx: DeviceContext
    ) raises -> DeviceBuffer[type]:
        # const char* AsyncRT_DeviceMulticastBuffer_unicastBufferFor(const DeviceBuffer **result, void **devicePtr, const DeviceMulticastBuffer *multiBuffer, const DeviceContext* ctx)
        var buf_handle = _DeviceBufferPtr()
        var buf_ptr = UnsafePointer[Scalar[type]]()

        _checked(
            external_call[
                "AsyncRT_DeviceMulticastBuffer_unicastBufferFor",
                _CharPtr,
                UnsafePointer[_DeviceBufferPtr],
                UnsafePointer[UnsafePointer[Scalar[type]]],
                _DeviceMulticastBufferPtr,
                _DeviceContextPtr,
            ](
                UnsafePointer(to=buf_handle),
                UnsafePointer(to=buf_ptr),
                self._handle,
                ctx._handle,
            )
        )

        return DeviceBuffer[type](buf_handle, buf_ptr)

    @doc_private
    fn multicast_buffer_for(
        self, ctx: DeviceContext
    ) raises -> DeviceBuffer[type]:
        # const char* AsyncRT_DeviceMulticastBuffer_multicastBufferFor(const DeviceBuffer **result, void **devicePtr, const DeviceMulticastBuffer *multiBuffer, const DeviceContext* ctx)
        var buf_handle = _DeviceBufferPtr()
        var buf_ptr = UnsafePointer[Scalar[type]]()

        _checked(
            external_call[
                "AsyncRT_DeviceMulticastBuffer_multicastBufferFor",
                _CharPtr,
                UnsafePointer[_DeviceBufferPtr],
                UnsafePointer[UnsafePointer[Scalar[type]]],
                _DeviceMulticastBufferPtr,
                _DeviceContextPtr,
            ](
                UnsafePointer(to=buf_handle),
                UnsafePointer(to=buf_ptr),
                self._handle,
                ctx._handle,
            )
        )

        return DeviceBuffer[type](buf_handle, buf_ptr)


struct _HostMappedBuffer[
    mut: Bool, //,
    type: DType,
    origin: Origin[mut],
]:
    var _ctx: DeviceContext
    var _dev_buf: Pointer[DeviceBuffer[type], origin]
    var _cpu_buf: HostBuffer[type]

    fn __init__(
        out self,
        ctx: DeviceContext,
        ref [origin]buf: DeviceBuffer[type],
    ) raises:
        var cpu_buf = ctx.enqueue_create_host_buffer[type](len(buf))
        self._ctx = ctx
        self._dev_buf = Pointer(to=buf)
        self._cpu_buf = cpu_buf

    fn __del__(owned self):
        pass

    fn __enter__(mut self) raises -> HostBuffer[type]:
        self._dev_buf[].enqueue_copy_to(self._cpu_buf)
        self._ctx.synchronize()
        return self._cpu_buf

    fn __exit__(mut self) raises:
        self._ctx.synchronize()
        self._cpu_buf.enqueue_copy_to(self._dev_buf[])
        self._ctx.synchronize()
