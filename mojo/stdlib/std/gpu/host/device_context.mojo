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

# Implementation of the C++ backed DeviceContext in Mojo
"""This module provides functionality for interacting with accelerators. In
particular the
[`DeviceContext`](/docs/std/gpu/host/device_context/DeviceContext/) struct,
which represents a single stream of execution on a given accelerator. You can
use this struct to allocate accelerator memory, copy data to and from the
accelerator, and compile and execute functions on the accelerator."""

from std.collections.optional import OptionalReg
from std.math import align_up
from std.os import abort
from std.pathlib import Path
from std.ffi import (
    c_char,
    c_int,
    c_uint,
    c_size_t,
    external_call,
    CStringSlice,
    _CPointer,
)
from std.sys import (
    bit_width_of,
    get_defined_bool,
    get_defined_string,
    is_defined,
    is_gpu,
    size_of,
)
from std.sys.compile import DebugLevel, OptimizationLevel
from std.sys.info import (
    CompilationTarget,
    _accelerator_arch,
    _TargetType,
    is_triple,
)
from std.sys.intrinsics import _type_is_eq
from std.sys.defines import _is_bool_like

from std.reflection import call_location, SourceLocation
from std.builtin.device_passable import DevicePassable
from std.compile.compile import CompiledFunctionInfo
from std.reflection import get_linkage_name, reflect, reflect_fn
from std.gpu.host.compile import (
    _compile_code,
    _cross_compilation,
    _ptxas_compile,
    _to_sass,
    get_gpu_target,
)
from std.memory import stack_allocation
from std.memory.unsafe import bitcast
from std.memory.unsafe_pointer import alloc
from std.builtin.rebind import downcast

from std.builtin.coroutine import (
    AnyCoroutine,
    _coro_resume_fn,
    _coro_destroy_fn,
)

from std.utils import Variant
from std.utils._serialize import _serialize_elements

from .info import GPUInfo


# Create empty structs to ensure dtype checking when using the C++ handles.
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


struct _DeviceEventCpp:
    pass


struct _DeviceTimerCpp:
    pass


struct _DeviceContextScopeCpp:
    pass


struct _DeviceGraphBuilderCpp:
    pass


struct _DeviceGraphCpp:
    pass


comptime _DeviceContextPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = ExternalOrigin[mut=mut],
] = _CPointer[_DeviceContextCpp, origin]

comptime _DeviceBufferPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = ExternalOrigin[mut=mut],
] = _CPointer[_DeviceBufferCpp, origin]

comptime _DeviceFunctionPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = ExternalOrigin[mut=mut],
] = _CPointer[_DeviceFunctionCpp, origin]

comptime _DeviceMulticastBufferPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = ExternalOrigin[mut=mut],
] = _CPointer[_DeviceMulticastBufferCpp, origin]

comptime _DeviceStreamPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = ExternalOrigin[mut=mut],
] = _CPointer[_DeviceStreamCpp, origin]

comptime _DeviceEventPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = ExternalOrigin[mut=mut],
] = _CPointer[_DeviceEventCpp, origin]

comptime _DeviceTimerPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = ExternalOrigin[mut=mut],
] = _CPointer[_DeviceTimerCpp, origin]

comptime _DeviceContextScopePtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = ExternalOrigin[mut=mut],
] = _CPointer[_DeviceContextScopeCpp, origin]

comptime _DeviceGraphBuilderPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = ExternalOrigin[mut=mut],
] = _CPointer[_DeviceGraphBuilderCpp, origin]

comptime _DeviceGraphPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = ExternalOrigin[mut=mut],
] = _CPointer[_DeviceGraphCpp, origin]

comptime _CString[
    origin: Origin[mut=False] = ExternalOrigin[mut=False]
] = Optional[CStringSlice[origin]]

comptime _DumpPath = Variant[Bool, Path, StaticString, def() capturing -> Path]

# Define helper methods to call AsyncRT bindings.


def _string_from_owned_charptr(c_str: _CString) -> String:
    var result = String()
    if c_str:
        result = String(unsafe_from_utf8_ptr=c_str.unsafe_value().unsafe_ptr())
    # void AsyncRT_DeviceContext_strfree(const char* ptr)
    external_call["AsyncRT_DeviceContext_strfree", NoneType](c_str)
    return result^


@always_inline
def _checked(
    err: _CString,
    *,
    msg: String = "",
    location: OptionalReg[SourceLocation] = None,
) raises:
    if err:
        _raise_checked_impl(err, msg, location.or_else(call_location()))


@always_inline
def _checked_call[
    func: Some[TrivialRegisterPassable]
](
    err: _CString,
    *,
    device_context: DeviceContext,
    location: SourceLocation,
) raises:
    # Use the source-level function name for the error message. This is purely
    # a display string; kernel launch/compilation continues to use the mangled
    # linkage name elsewhere.
    comptime func_name = reflect_fn[func].display_name()
    if err:
        var err_msg = _string_from_owned_charptr(err)
        raise Error(
            location,
            " failed calling '",
            func_name,
            "' on device ",
            device_context.api(),
            ":",
            device_context.id(),
            " with error '",
            err_msg,
            "'",
        )


@no_inline
def _raise_checked_impl(
    err_msg: _CString, msg: String, location: SourceLocation
) raises:
    var err = _string_from_owned_charptr(err_msg)
    raise Error(location.prefix(err + ((" " + msg) if msg else "")))


# Checks that the given `dim` has only positive integers in them.
def _check_dim[
    func_name_for_msg: StringLiteral, dim_name_for_msg: StringLiteral
](dim: Dim, *, location: SourceLocation) raises:
    if dim.x() <= 0:
        comptime msg = String(
            func_name_for_msg,
            ": Dim value ",
            dim_name_for_msg,
            ".x must be a positive number.",
        )
        raise Error(location.prefix(msg))
    if dim.y() <= 0:
        comptime msg = String(
            func_name_for_msg,
            ": Dim value ",
            dim_name_for_msg,
            ".y must be a positive number.",
        )
        raise Error(location.prefix(msg))
    if dim.z() <= 0:
        comptime msg = String(
            func_name_for_msg,
            ": Dim value ",
            dim_name_for_msg,
            ".z must be a positive number.",
        )
        raise Error(location.prefix(msg))


struct _DeviceTimer:
    var _handle: _DeviceTimerPtr[mut=True]

    def __init__(out self, ptr: _DeviceTimerPtr[mut=True]):
        self._handle = ptr

    def __del__(deinit self):
        # void AsyncRT_DeviceTimer_release(const DviceTimer *timer)
        external_call["AsyncRT_DeviceTimer_release", NoneType](self._handle)


@fieldwise_init
struct StreamPriorityRange(TrivialRegisterPassable, Writable):
    """Represents the range of valid stream priorities for a GPU device.

    Stream priorities control the scheduling of GPU operations, with higher
    priority streams being executed preferentially over lower priority streams.
    """

    var least: Int
    """The lowest (numerically smallest) priority value."""

    var greatest: Int
    """The highest (numerically largest) priority value."""

    @always_inline
    def write_to(self, mut writer: Some[Writer]):
        """Writes the stream priority range to the given writer.

        Args:
            writer: The writer to output the stream priority range to.
        """
        writer.write(
            "StreamPriorityRange(least=",
            self.least,
            ", greatest=",
            self.greatest,
            ")",
        )


@fieldwise_init
struct _DeviceBufferMode(TrivialRegisterPassable):
    var _mode: Int

    comptime _SYNC = _DeviceBufferMode(0)
    comptime _ASYNC = _DeviceBufferMode(1)

    def __eq__(self, other: Self) -> Bool:
        return self._mode == other._mode


struct HostBuffer[dtype: DType](ImplicitlyCopyable, Sized, Writable):
    """Represents a block of host-resident storage. For GPU devices, a host
    buffer is allocated in the host's global memory.

    To allocate a `HostBuffer`, use one of the methods provided by
    `DeviceContext`, such as
    [`enqueue_create_host_buffer()`](/docs/std/gpu/host/device_context/DeviceContext/#enqueue_create_host_buffer).

    Parameters:
        dtype: Data type to be stored in the buffer.
    """

    comptime _HostPtr = UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]

    # We cache the pointer of the buffer here to provide access to elements.
    var _host_ptr: Self._HostPtr
    var _handle: _DeviceBufferPtr[mut=True]

    @doc_hidden
    def __init__(
        out self,
        ctx: DeviceContext,
        size: Int,
    ) raises:
        """This init takes in a constructed `DeviceContext` and schedules an
        owned buffer allocation using the stream in the device context.
        """
        comptime assert not is_gpu(), "HostBuffer is not supported on GPUs"
        comptime elem_size = size_of[Self.dtype]()
        var cpp_handle: _DeviceBufferPtr[mut=True] = {}
        var host_ptr: Optional[Self._HostPtr] = {}

        # const char *AsyncRT_DeviceContext_createHostBuffer(const DeviceBuffer **result, void **device_ptr, const DeviceContext *ctx, size_t len, size_t elem_size)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_createHostBuffer",
                _CString[],
            ](
                UnsafePointer(to=cpp_handle),
                UnsafePointer(to=host_ptr),
                ctx._handle,
                c_size_t(size),
                c_size_t(elem_size),
            )
        )

        self._host_ptr = host_ptr._unsafe_nullable()
        self._handle = cpp_handle

    @doc_hidden
    def __init__(
        out self, handle: _DeviceBufferPtr[mut=True], host_ptr: Self._HostPtr
    ):
        comptime assert not is_gpu(), "HostBuffer is not supported on GPUs"
        self._host_ptr = host_ptr
        self._handle = handle

    @doc_hidden
    def __init__(
        out self,
        ctx: DeviceContext,
        host_ptr: Self._HostPtr,
        size: Int,
        *,
        owning: Bool,
    ):
        comptime assert not is_gpu(), "HostBuffer is not supported on GPUs"
        comptime elem_size = size_of[Self.dtype]()
        var cpp_handle: _DeviceBufferPtr[mut=True] = {}
        # void AsyncRT_DeviceContext_createBuffer_owning(
        #     const DeviceBuffer **result, const DeviceContext *ctx,
        #     void *device_ptr, size_t len, size_t elem_size, bool owning)
        external_call[
            "AsyncRT_DeviceContext_createBuffer_owning",
            NoneType,
            UnsafePointer[_DeviceBufferPtr[mut=True], origin_of(cpp_handle)],
            _DeviceContextPtr[mut=True],
            Self._HostPtr,
            c_size_t,
            c_size_t,
            Bool,
        ](
            UnsafePointer(to=cpp_handle),
            ctx._handle,
            host_ptr,
            c_size_t(size),
            c_size_t(elem_size),
            owning,
        )

        self._host_ptr = host_ptr
        self._handle = cpp_handle

    def __init__(out self, *, copy: Self):
        """Creates a copy of an existing host buffer by incrementing its reference count.

        This copy constructor creates a new reference to the same underlying host buffer
        by incrementing the reference count of the native buffer object. Both the original
        and the copy will refer to the same memory on the device.

        Args:
            copy: The host buffer to copy.
        """
        comptime assert not is_gpu(), "HostBuffer is not supported on GPUs"
        # Increment the reference count before copying the handle.
        #
        # void AsyncRT_DeviceBuffer_retain(const DeviceBuffer *buffer)
        external_call[
            "AsyncRT_DeviceBuffer_retain",
            NoneType,
            _DeviceBufferPtr[mut=True],
        ](copy._handle)
        self._host_ptr = copy._host_ptr
        self._handle = copy._handle

    def __del__(deinit self):
        """Releases resources associated with this host buffer.

        This function schedules an owned buffer free using the stream in the
        device context. The actual deallocation may occur asynchronously after
        all operations using this buffer have completed.
        """
        comptime assert not is_gpu(), "HostBuffer is not supported on GPUs"
        # void AsyncRT_DeviceBuffer_release(const DeviceBuffer *buffer)
        external_call[
            "AsyncRT_DeviceBuffer_release", NoneType, _DeviceBufferPtr[mut=True]
        ](
            self._handle,
        )

    def __len__(self) -> Int:
        """Returns the number of elements in this buffer.

        This method calculates the number of elements by dividing the total byte size
        of the buffer by the size of each element.

        Returns:
            The number of elements in the buffer.
        """
        comptime assert not is_gpu(), "HostBuffer is not supported on GPUs"
        # int64_t AsyncRT_DeviceBuffer_bytesize(const DeviceBuffer *buffer)
        return (
            external_call[
                "AsyncRT_DeviceBuffer_bytesize", Int, _DeviceBufferPtr[mut=True]
            ](self._handle)
            // size_of[Self.dtype]()
        )

    def create_sub_buffer[
        view_type: DType
    ](self, offset: Int, size: Int) raises -> HostBuffer[view_type]:
        """Creates a sub-buffer view of this buffer with a different element dtype.

        This method creates a new buffer that references a subset of the memory in this
        buffer, potentially with a different element dtype. The sub-buffer shares the
        underlying memory with the original buffer.

        Parameters:
            view_type: The data type for elements in the new sub-buffer.

        Args:
            offset: The starting offset in elements from the beginning of this buffer.
            size: The number of elements in the new sub-buffer.

        Returns:
            A new HostBuffer referencing the specified region with the specified element dtype.

        Raises:
            If the operation fails.
        """
        comptime assert not is_gpu(), "HostBuffer is not supported on GPUs"
        comptime elem_size = size_of[view_type]()
        var new_handle: _DeviceBufferPtr[mut=True] = {}
        var new_host_ptr = Optional[
            UnsafePointer[Scalar[view_type], MutAnyOrigin]
        ]()
        # const char *AsyncRT_DeviceBuffer_createSubBuffer(
        #     const DeviceBuffer **result, void **device_ptr,
        #     const DeviceBuffer *buf, size_t offset, size_t len, size_t elem_size)
        _checked(
            external_call[
                "AsyncRT_DeviceBuffer_createSubBuffer",
                _CString[],
            ](
                UnsafePointer(to=new_handle),
                UnsafePointer(to=new_host_ptr),
                self._handle,
                c_size_t(offset),
                c_size_t(size),
                c_size_t(elem_size),
            )
        )
        return HostBuffer[view_type](
            new_handle, new_host_ptr._unsafe_nullable()
        )

    def enqueue_copy_to(self, dst: HostBuffer[Self.dtype, ...]) raises:
        """Enqueues an asynchronous copy from this buffer to another host buffer.

        This method schedules a memory copy operation from this buffer to the destination
        buffer. The operation is asynchronous and will be executed in the stream associated
        with this buffer's context.

        Args:
            dst: The destination host buffer to copy data to.

        Raises:
            If the operation fails.
        """
        dst.context().enqueue_copy(dst, self)

    def enqueue_copy_to(self, dst: DeviceBuffer[Self.dtype, ...]) raises:
        """Enqueues an asynchronous copy from this buffer to a device buffer.

        This method schedules a memory copy operation from this buffer to the destination
        buffer. The operation is asynchronous and will be executed in the stream associated
        with this buffer's context.

        Args:
            dst: The destination device buffer to copy data to.

        Raises:
            If the operation fails.
        """
        comptime assert not is_gpu(), "HostBuffer is not supported on GPUs"
        dst.context().enqueue_copy(dst, self)

    def enqueue_copy_to(
        self, dst_ptr: OptionalUnsafePointer[mut=True, Scalar[Self.dtype], _]
    ) raises:
        """Enqueues an asynchronous copy from this buffer to host memory.

        This method schedules a memory copy operation from this device buffer to the
        specified host memory location. The operation is asynchronous and will be
        executed in the stream associated with this buffer's context.

        Args:
            dst_ptr: Pointer to the destination host memory location.

        Raises:
            If the operation fails.
        """
        comptime assert not is_gpu(), "HostBuffer is not supported on GPUs"
        self.context().enqueue_copy(dst_ptr, self)

    def enqueue_copy_from(self, src: HostBuffer[Self.dtype, ...]) raises:
        """Enqueues an asynchronous copy to this buffer from another host buffer.

        This method schedules a memory copy operation to this buffer from the source
        buffer. The operation is asynchronous and will be executed in the stream
        associated with this buffer's context.

        Args:
            src: The source host buffer to copy data from.

        Raises:
            If the operation fails.
        """
        comptime assert not is_gpu(), "HostBuffer is not supported on GPUs"
        self.context().enqueue_copy(self, src)

    def enqueue_copy_from(self, src: DeviceBuffer[Self.dtype, ...]) raises:
        """Enqueues an asynchronous copy to this buffer from a device buffer.

        This method schedules a memory copy operation to this buffer from the source
        buffer. The operation is asynchronous and will be executed in the stream
        associated with this buffer's context.

        Args:
            src: The source device buffer to copy data from.

        Raises:
            If the operation fails.
        """
        comptime assert not is_gpu(), "HostBuffer is not supported on GPUs"
        self.context().enqueue_copy(self, src)

    def enqueue_copy_from(
        self, src_ptr: OptionalUnsafePointer[Scalar[Self.dtype], _]
    ) raises:
        """Enqueues an asynchronous copy to this buffer from host memory.

        This method schedules a memory copy operation to this device buffer from the
        specified host memory location. The operation is asynchronous and will be
        executed in the stream associated with this buffer's context.

        Args:
            src_ptr: Pointer to the source host memory location.

        Raises:
            If the operation fails.
        """
        comptime assert not is_gpu(), "HostBuffer is not supported on GPUs"
        self.context().enqueue_copy(self, src_ptr)

    def enqueue_copy_from(self, src: Span[Scalar[Self.dtype], _]) raises:
        """Enqueues an asynchronous copy to this buffer from a `Span`.

        This method schedules a memory copy operation to this buffer from the
        source span. The operation is asynchronous and will be executed in the
        stream associated with this buffer's context. The span must contain at
        least as many elements as this buffer; this invariant is checked via
        `debug_assert`.

        Args:
            src: The source span to copy data from. Must have at least as many
                elements as this buffer.

        Raises:
            If the operation fails.
        """
        comptime assert not is_gpu(), "HostBuffer is not supported on GPUs"
        self.context().enqueue_copy(self, src)

    def enqueue_copy_to(
        self, dst: Span[mut=True, Scalar[Self.dtype], _]
    ) raises:
        """Enqueues an asynchronous copy from this buffer to a `Span`.

        This method schedules a memory copy operation from this buffer to the
        destination span. The operation is asynchronous and will be executed in
        the stream associated with this buffer's context. The span must contain
        at least as many elements as this buffer; this invariant is checked via
        `debug_assert`.

        Args:
            dst: The destination span to copy data to. Must have at least as
                many elements as this buffer.

        Raises:
            If the operation fails.
        """
        comptime assert not is_gpu(), "HostBuffer is not supported on GPUs"
        self.context().enqueue_copy(dst, self)

    def enqueue_fill(self, val: Scalar[Self.dtype]) raises:
        """Enqueues an operation to fill this buffer with a specified value.

        This method schedules a memory set operation that fills the entire buffer
        with the specified value. The operation is asynchronous and will be executed
        in the stream associated with this buffer's context.

        Args:
            val: The value to fill the buffer with.

        Raises:
            If the operation fails.
        """
        comptime assert not is_gpu(), "HostBuffer is not supported on GPUs"
        self.context().enqueue_memset(self, val)

    def reassign_ownership_to(self, ctx: DeviceContext) raises:
        """Transfers ownership of this buffer to another device context.

        This method changes the device context that owns this buffer. This can be
        useful when sharing buffers between different contexts or when migrating
        workloads between devices.

        Args:
            ctx: The new device context to take ownership of this buffer.

        Raises:
            If the operation fails.
        """
        comptime assert not is_gpu(), "HostBuffer is not supported on GPUs"
        # const char * AsyncRT_DeviceBuffer_reassignOwnershipTo(const DeviceBuffer *buf, const DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceBuffer_reassignOwnershipTo",
                _CString[],
                _DeviceBufferPtr[mut=True],
                _DeviceContextPtr[mut=True],
            ](self._handle, ctx._handle)
        )

    def take_ptr(
        deinit self,
    ) -> Self._HostPtr:
        """Takes ownership of the device pointer from this buffer.

        This method releases the device pointer from the buffer's control and
        returns it to the caller. After this call, the buffer no longer owns
        the pointer, and the caller is responsible for managing its lifecycle.

        Returns:
            The raw device pointer that was owned by this buffer.
        """
        comptime assert not is_gpu(), "HostBuffer is not supported on GPUs"
        # void AsyncRT_DeviceBuffer_release_ptr(const DeviceBuffer *buffer)
        external_call[
            "AsyncRT_DeviceBuffer_release_ptr",
            NoneType,
            _DeviceBufferPtr[mut=True],
        ](self._handle)
        return self._host_ptr

    @always_inline
    def unsafe_ptr(
        self,
    ) -> Self._HostPtr:
        """Returns the raw device pointer without transferring ownership.

        This method provides direct access to the underlying device pointer
        for advanced use cases. The buffer retains ownership of the pointer.

        Returns:
            The raw device pointer owned by this buffer.
        """
        comptime assert not is_gpu(), "HostBuffer is not supported on GPUs"
        return self._host_ptr

    def context(self) raises -> DeviceContext:
        """Returns the device context associated with this buffer.

        This method retrieves the device context that owns this buffer and is
        responsible for managing its lifecycle and operations.

        Returns:
            The device context associated with this buffer.

        Raises:
            If the operation fails.
        """
        comptime assert not is_gpu(), "HostBuffer is not supported on GPUs"
        # const DeviceContext *AsyncRT_DeviceBuffer_context(const DeviceBuffer *buffer)
        var ctx_ptr: _DeviceContextPtr[mut=True] = external_call[
            "AsyncRT_DeviceBuffer_context",
            _DeviceContextPtr[mut=True],
            _DeviceBufferPtr[mut=True],
        ](self._handle)
        return DeviceContext(ctx_ptr)

    def write_to(self, mut writer: Some[Writer]):
        """Writes a string representation of this buffer to the provided writer.

        This method formats the buffer's contents as a string and writes it to
        the specified writer. For large buffers, a compact representation is used.

        Args:
            writer: The writer to output the formatted string to.
        """
        comptime assert not is_gpu(), "HostBuffer is not supported on GPUs"
        writer.write("HostBuffer")
        writer.write("(")

        @parameter
        def serialize[T: Writable](val: T):
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

    @always_inline
    def __getitem__(self, idx: Int) -> Scalar[Self.dtype]:
        """Retrieves the element at the specified index from the host buffer.

        This operator allows direct access to individual elements in the host buffer
        using array indexing syntax.

        Args:
            idx: The index of the element to retrieve.

        Returns:
            The scalar value at the specified index.
        """
        comptime assert not is_gpu(), "HostBuffer is not supported on GPUs"
        return self._host_ptr[idx]

    @always_inline
    def __setitem__(
        self: HostBuffer[Self.dtype], idx: Int, val: Scalar[Self.dtype]
    ):
        """Sets the element at the specified index in the host buffer.

        This operator allows direct modification of individual elements in the host buffer
        using array indexing syntax.

        Args:
            idx: The index of the element to modify.
            val: The new value to store at the specified index.
        """
        comptime assert not is_gpu(), "HostBuffer is not supported on GPUs"
        self._host_ptr[idx] = val

    def as_span[
        mut: Bool, origin: Origin[mut=mut], //
    ](ref[origin] self) -> Span[Scalar[Self.dtype], origin]:
        """Returns a `Span` pointing to the underlying memory of the `HostBuffer`.

        Parameters:
            mut: Whether the span should be mutable.
            origin: The origin of the buffer reference.

        Returns:
            A `Span` pointing to the underlying memory of the `HostBuffer`.
        """
        # Safety: We are casting the pointer to the mutability and origin of
        # self and `_host_ptr` is already mutable.
        return {
            ptr = self._host_ptr.unsafe_mut_cast[mut]().unsafe_origin_cast[
                origin
            ](),
            length = len(self),
        }


struct DeviceBuffer[dtype: DType](
    DevicePassable, ImplicitlyCopyable, Sized, Writable
):
    """Represents a block of device-resident storage. For GPU devices, a device
    buffer is allocated in the device's global memory.

    To allocate a `DeviceBuffer`, use one of the methods provided by
    `DeviceContext`, such as
    [`enqueue_create_buffer()`](/docs/std/gpu/host/device_context/DeviceContext/#enqueue_create_buffer).

    Parameters:
        dtype: Data dtype to be stored in the buffer.
    """

    # Implementation of `DevicePassable`
    comptime device_type: AnyType = UnsafePointer[
        mut=True, Scalar[Self.dtype], AnyOrigin[mut=True]
    ]
    """DeviceBuffer dtypes are remapped to UnsafePointer when passed to accelerator devices."""

    def _to_device_type(self, target: MutOpaquePointer[_]):
        """Device dtype mapping from DeviceBuffer to the device's UnsafePointer.
        """
        # TODO: Allow the low-level DeviceContext implementation to intercept
        # these translations.
        target.bitcast[Self.device_type]()[] = self._device_ptr

    @staticmethod
    def get_type_name() -> String:
        """
        Gets this dtype's name, for use in error messages when handing arguments
        to kernels.
        TODO: This will go away soon, when we get better error messages for
        kernel calls.

        Returns:
            This dtype's name.
        """
        return String(t"DeviceBuffer[{Self.dtype}]")

    comptime _DevicePtr = UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]
    # _device_ptr must be the first word in the struct to enable passing of
    # DeviceBuffer to kernels. The first word is passed to the kernel and
    # it needs to contain the value registered with the driver.
    var _device_ptr: Self._DevicePtr
    var _handle: _DeviceBufferPtr[mut=True]

    @doc_hidden
    @always_inline
    def __init__(
        out self,
        ctx: DeviceContext,
        size: Int,
        mode: _DeviceBufferMode,
    ) raises:
        """This init takes in a constructed `DeviceContext` and schedules an
        owned buffer allocation using the stream in the device context.
        """
        comptime assert not is_gpu(), "DeviceBuffer is not supported on GPUs"
        comptime elem_size = size_of[Self.dtype]()
        var cpp_handle: _DeviceBufferPtr[mut=True] = {}
        var device_ptr: Optional[Self._DevicePtr] = {}

        # TODO: Remove this if statement.
        # As of GEX-3005, Driver only supports async allocation. For
        # sync allocation, we need to explicitly synchronize after this step.
        # See DeviceContext.create_buffer_sync() for example.
        if mode == _DeviceBufferMode._ASYNC:
            # const char *AsyncRT_DeviceContext_createBuffer_async(const DeviceBuffer **result, void **device_ptr, const DeviceContext *ctx, size_t len, size_t elem_size)
            _checked(
                external_call[
                    "AsyncRT_DeviceContext_createBuffer_async",
                    _CString[],
                ](
                    UnsafePointer(to=cpp_handle),
                    UnsafePointer(to=device_ptr),
                    ctx._handle,
                    c_size_t(size),
                    c_size_t(elem_size),
                ),
                location=call_location(),
            )
        else:
            raise Error(
                "DeviceBuffer.__init__: Unsupported _DeviceBufferMode(",
                mode._mode,
                ")",
            )

        self._device_ptr = device_ptr.value()
        self._handle = cpp_handle

    @doc_hidden
    def __init__(
        out self,
        handle: _DeviceBufferPtr[mut=True],
        device_ptr: Self._DevicePtr,
    ):
        comptime assert not is_gpu(), "DeviceBuffer is not supported on GPUs"
        self._device_ptr = device_ptr
        self._handle = handle

    @doc_hidden
    def __init__(
        out self,
        ctx: DeviceContext,
        ptr: Self._DevicePtr,
        size: Int,
        *,
        owning: Bool,
    ):
        comptime assert not is_gpu(), "DeviceBuffer is not supported on GPUs"
        comptime elem_size = size_of[Self.dtype]()
        var cpp_handle: _DeviceBufferPtr[mut=True] = {}
        # void AsyncRT_DeviceContext_createBuffer_owning(
        #     const DeviceBuffer **result, const DeviceContext *ctx,
        #     void *device_ptr, size_t len, size_t elem_size, bool owning)
        external_call[
            "AsyncRT_DeviceContext_createBuffer_owning",
            NoneType,
            UnsafePointer[_DeviceBufferPtr[mut=True], origin_of(cpp_handle)],
            _DeviceContextPtr[mut=True],
            Self._DevicePtr,
            c_size_t,
            c_size_t,
            Bool,
        ](
            UnsafePointer(to=cpp_handle),
            ctx._handle,
            ptr,
            c_size_t(size),
            c_size_t(elem_size),
            owning,
        )

        self._device_ptr = ptr
        self._handle = cpp_handle

    @doc_hidden
    def __init__[
        _dtype: DType,
    ](
        out self: DeviceBuffer[_dtype],
        ctx: DeviceContext,
        ptr: UnsafePointer[Scalar[_dtype], ...],
        size: Int,
        *,
        owning: Bool,
    ):
        """Constructs a DeviceBuffer from any pointer.

        This constructor accepts pointers with any origin and converts them
        internally to MutAnyOrigin. This is a stepping stone API that allows
        existing code using specific origins to work while the codebase
        transitions to proper origin tracking.

        Parameters:
            _dtype: The element type of the buffer.

        Args:
            ctx: The device context.
            ptr: Pointer to device memory with any origin.
            size: Number of elements.
            owning: Whether this buffer owns the memory.
        """
        comptime assert not is_gpu(), "DeviceBuffer is not supported on GPUs"
        comptime elem_size = size_of[_dtype]()
        var cpp_handle: _DeviceBufferPtr[mut=True] = {}
        var device_ptr = rebind[UnsafePointer[Scalar[_dtype], MutAnyOrigin]](
            ptr
        )
        external_call[
            "AsyncRT_DeviceContext_createBuffer_owning",
            NoneType,
            UnsafePointer[_DeviceBufferPtr[mut=True], origin_of(cpp_handle)],
            _DeviceContextPtr[mut=True],
            UnsafePointer[Scalar[_dtype], MutAnyOrigin],
            c_size_t,
            c_size_t,
            Bool,
        ](
            UnsafePointer(to=cpp_handle),
            ctx._handle,
            device_ptr,
            c_size_t(size),
            c_size_t(elem_size),
            owning,
        )

        self._device_ptr = device_ptr
        self._handle = cpp_handle

    def __init__(out self, *, copy: Self):
        """Creates a copy of an existing device buffer by incrementing its reference count.

        This copy constructor creates a new reference to the same underlying device buffer
        by incrementing the reference count of the native buffer object. Both the original
        and the copy will refer to the same memory on the device.

        Args:
            copy: The device buffer to copy.
        """
        comptime assert not is_gpu(), "DeviceBuffer is not supported on GPUs"
        # Increment the reference count before copying the handle.
        #
        # void AsyncRT_DeviceBuffer_retain(const DeviceBuffer *buffer)
        external_call[
            "AsyncRT_DeviceBuffer_retain",
            NoneType,
            _DeviceBufferPtr[mut=True],
        ](copy._handle)
        self._device_ptr = copy._device_ptr
        self._handle = copy._handle

    @always_inline
    def __del__(deinit self):
        """Releases resources associated with this device buffer.

        This function schedules an owned buffer free using the stream in the
        device context. The actual deallocation may occur asynchronously after
        all operations using this buffer have completed.
        """
        comptime assert not is_gpu(), "DeviceBuffer is not supported on GPUs"
        # void AsyncRT_DeviceBuffer_release(const DeviceBuffer *buffer)
        external_call[
            "AsyncRT_DeviceBuffer_release", NoneType, _DeviceBufferPtr[mut=True]
        ](
            self._handle,
        )

    @staticmethod
    @doc_hidden
    def empty(context: DeviceContext) -> Self:
        return Self(
            context,
            Self._DevicePtr.unsafe_dangling(),
            0,
            owning=False,
        )

    def __len__(self) -> Int:
        """Returns the number of elements in this buffer.

        This method calculates the number of elements by dividing the total byte size
        of the buffer by the size of each element.

        Returns:
            The number of elements in the buffer.
        """
        comptime assert not is_gpu(), "DeviceBuffer is not supported on GPUs"
        # int64_t AsyncRT_DeviceBuffer_bytesize(const DeviceBuffer *buffer)
        return (
            external_call[
                "AsyncRT_DeviceBuffer_bytesize", Int, _DeviceBufferPtr[mut=True]
            ](self._handle)
            // size_of[Self.dtype]()
        )

    @always_inline
    def create_sub_buffer[
        view_type: DType
    ](self, offset: Int, size: Int) raises -> DeviceBuffer[view_type]:
        """Creates a sub-buffer view of this buffer with a different element dtype.

        This method creates a new buffer that references a subset of the memory in this
        buffer, potentially with a different element dtype. The sub-buffer shares the
        underlying memory with the original buffer.

        Parameters:
            view_type: The data type for elements in the new sub-buffer.

        Args:
            offset: The starting offset in elements from the beginning of this buffer.
            size: The number of elements in the new sub-buffer.

        Returns:
            A new DeviceBuffer referencing the specified region with the specified element dtype.

        Raises:
            If the operation fails.
        """
        comptime assert not is_gpu(), "DeviceBuffer is not supported on GPUs"
        comptime elem_size = size_of[view_type]()
        var new_handle: _DeviceBufferPtr[mut=True] = {}
        var new_device_ptr: Optional[
            UnsafePointer[Scalar[view_type], MutAnyOrigin]
        ] = {}
        # const char *AsyncRT_DeviceBuffer_createSubBuffer(
        #     const DeviceBuffer **result, void **device_ptr,
        #     const DeviceBuffer *buf, size_t offset, size_t len, size_t elem_size)
        _checked(
            external_call[
                "AsyncRT_DeviceBuffer_createSubBuffer",
                _CString[],
            ](
                UnsafePointer(to=new_handle),
                UnsafePointer(to=new_device_ptr),
                self._handle,
                c_size_t(offset),
                c_size_t(size),
                c_size_t(elem_size),
            ),
            location=call_location(),
        )
        return DeviceBuffer[view_type](new_handle, new_device_ptr.value())

    def enqueue_copy_to(self, dst: DeviceBuffer[Self.dtype, ...]) raises:
        """Enqueues an asynchronous copy from this buffer to another device buffer.

        This method schedules a memory copy operation from this buffer to the destination
        buffer. The operation is asynchronous and will be executed in the stream associated
        with this buffer's context.

        Args:
            dst: The destination device buffer to copy data to.

        Raises:
            If the operation fails.
        """
        comptime assert not is_gpu(), "DeviceBuffer is not supported on GPUs"
        dst.context().enqueue_copy(dst, self)

    def enqueue_copy_to(self, dst: HostBuffer[Self.dtype, ...]) raises:
        """Enqueues an asynchronous copy from this buffer to a host buffer.

        This method schedules a memory copy operation from this buffer to the destination
        buffer. The operation is asynchronous and will be executed in the stream associated
        with this buffer's context.

        Args:
            dst: The destination host buffer to copy data to.

        Raises:
            If the operation fails.
        """
        comptime assert not is_gpu(), "DeviceBuffer is not supported on GPUs"
        dst.context().enqueue_copy(dst, self)

    def enqueue_copy_to(
        self, dst_ptr: OptionalUnsafePointer[mut=True, Scalar[Self.dtype], _]
    ) raises:
        """Enqueues an asynchronous copy from this buffer to host memory.

        This method schedules a memory copy operation from this device buffer to the
        specified host memory location. The operation is asynchronous and will be
        executed in the stream associated with this buffer's context.

        Args:
            dst_ptr: Pointer to the destination host memory location.

        Raises:
            If the operation fails.
        """
        comptime assert not is_gpu(), "DeviceBuffer is not supported on GPUs"
        self.context().enqueue_copy(dst_ptr, self)

    def enqueue_copy_from(self, src: DeviceBuffer[Self.dtype, ...]) raises:
        """Enqueues an asynchronous copy to this buffer from another device buffer.

        This method schedules a memory copy operation to this buffer from the source
        buffer. The operation is asynchronous and will be executed in the stream
        associated with this buffer's context.

        Args:
            src: The source device buffer to copy data from.

        Raises:
            If the operation fails.
        """
        comptime assert not is_gpu(), "DeviceBuffer is not supported on GPUs"
        self.context().enqueue_copy(self, src)

    def enqueue_copy_from(self, src: HostBuffer[Self.dtype, ...]) raises:
        """Enqueues an asynchronous copy to this buffer from a host buffer.

        This method schedules a memory copy operation to this buffer from the source
        buffer. The operation is asynchronous and will be executed in the stream
        associated with this buffer's context.

        Args:
            src: The source host buffer to copy data from.

        Raises:
            If the operation fails.
        """
        comptime assert not is_gpu(), "DeviceBuffer is not supported on GPUs"
        self.context().enqueue_copy(self, src)

    def enqueue_copy_from(
        self, src_ptr: OptionalUnsafePointer[Scalar[Self.dtype], _]
    ) raises:
        """Enqueues an asynchronous copy to this buffer from host memory.

        This method schedules a memory copy operation to this device buffer from the
        specified host memory location. The operation is asynchronous and will be
        executed in the stream associated with this buffer's context.

        Args:
            src_ptr: Pointer to the source host memory location.

        Raises:
            If the operation fails.
        """
        comptime assert not is_gpu(), "DeviceBuffer is not supported on GPUs"
        self.context().enqueue_copy(self, src_ptr)

    def enqueue_copy_from(self, src: Span[Scalar[Self.dtype], _]) raises:
        """Enqueues an asynchronous copy to this buffer from a `Span`.

        This method schedules a memory copy operation to this buffer from the
        source span. The operation is asynchronous and will be executed in the
        stream associated with this buffer's context. The span must contain at
        least as many elements as this buffer; this invariant is checked via
        `debug_assert`.

        Args:
            src: The source span to copy data from. Must have at least as many
                elements as this buffer.

        Raises:
            If the operation fails.
        """
        comptime assert not is_gpu(), "DeviceBuffer is not supported on GPUs"
        self.context().enqueue_copy(self, src)

    def enqueue_copy_to(
        self, dst: Span[mut=True, Scalar[Self.dtype], _]
    ) raises:
        """Enqueues an asynchronous copy from this buffer to a `Span`.

        This method schedules a memory copy operation from this buffer to the
        destination span. The operation is asynchronous and will be executed in
        the stream associated with this buffer's context. The span must contain
        at least as many elements as this buffer; this invariant is checked via
        `debug_assert`.

        Args:
            dst: The destination span to copy data to. Must have at least as
                many elements as this buffer.

        Raises:
            If the operation fails.
        """
        comptime assert not is_gpu(), "DeviceBuffer is not supported on GPUs"
        self.context().enqueue_copy(dst, self)

    def enqueue_fill(self, val: Scalar[Self.dtype]) raises:
        """Enqueues an operation to fill this buffer with a specified value.

        This method schedules a memory set operation that fills the entire buffer
        with the specified value. The operation is asynchronous and will be executed
        in the stream associated with this buffer's context.

        Args:
            val: The value to fill the buffer with.

        Raises:
            If the operation fails.
        """
        comptime assert not is_gpu(), "DeviceBuffer is not supported on GPUs"
        self.context().enqueue_memset(self, val)

    def reassign_ownership_to(self, ctx: DeviceContext) raises:
        """Transfers ownership of this buffer to another device context.

        This method changes the device context that owns this buffer. This can be
        useful when sharing buffers between different contexts or when migrating
        workloads between devices.

        Args:
            ctx: The new device context to take ownership of this buffer.

        Raises:
            If the operation fails.
        """
        comptime assert not is_gpu(), "DeviceBuffer is not supported on GPUs"
        # const char * AsyncRT_DeviceBuffer_reassignOwnershipTo(const DeviceBuffer *buf, const DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceBuffer_reassignOwnershipTo",
                _CString[],
                _DeviceBufferPtr[mut=True],
                _DeviceContextPtr[mut=True],
            ](self._handle, ctx._handle)
        )

    # NOTE: This is var self and not deinit self, since we still need
    # the destructor to run otherwise we hit memory leaks.
    @always_inline
    def take_ptr(
        var self,
    ) -> Self._DevicePtr:
        """Takes ownership of the device pointer from this buffer.

        This method releases the device pointer from the buffer's control and
        returns it to the caller. After this call, the buffer no longer owns
        the pointer, and the caller is responsible for managing its lifecycle.

        Returns:
            The raw device pointer that was owned by this buffer.
        """
        comptime assert not is_gpu(), "DeviceBuffer is not supported on GPUs"
        # void AsyncRT_DeviceBuffer_release_ptr(const DeviceBuffer *buffer)
        external_call[
            "AsyncRT_DeviceBuffer_release_ptr",
            NoneType,
            _DeviceBufferPtr[mut=True],
        ](self._handle)
        return self._device_ptr

    @always_inline
    def unsafe_ptr(
        self,
    ) -> Self._DevicePtr:
        """Returns the raw device pointer without transferring ownership.

        This method provides direct access to the underlying device pointer
        for advanced use cases. The buffer retains ownership of the pointer.

        Returns:
            The raw device pointer owned by this buffer.
        """
        comptime assert not is_gpu(), "DeviceBuffer is not supported on GPUs"
        return self._device_ptr

    def context(self) raises -> DeviceContext:
        """Returns the device context associated with this buffer.

        This method retrieves the device context that owns this buffer and is
        responsible for managing its lifecycle and operations.

        Returns:
            The device context associated with this buffer.

        Raises:
            If the operation fails.
        """
        comptime assert not is_gpu(), "DeviceBuffer is not supported on GPUs"
        # const DeviceContext *AsyncRT_DeviceBuffer_context(const DeviceBuffer *buffer)
        var ctx_ptr: _DeviceContextPtr[mut=True] = external_call[
            "AsyncRT_DeviceBuffer_context",
            _DeviceContextPtr[mut=True],
            _DeviceBufferPtr[mut=True],
        ](self._handle)
        return DeviceContext(ctx_ptr)

    def map_to_host(
        self,
        out mapped_buffer: _HostMappedBuffer[Self.dtype],
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
        from std.gpu.host import DeviceContext

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
        comptime assert not is_gpu(), "DeviceBuffer is not supported on GPUs"
        mapped_buffer = _HostMappedBuffer[Self.dtype](self.context(), self)

    def write_to(self, mut writer: Some[Writer]):
        """Writes a string representation of this buffer to the provided writer.

        This method formats the buffer's contents as a string and writes it to
        the specified writer. For large buffers, a compact representation is used.

        Args:
            writer: The writer to output the formatted string to.
        """
        comptime assert not is_gpu(), "DeviceBuffer is not supported on GPUs"
        try:
            with self.map_to_host() as host_buffer:
                writer.write("DeviceBuffer")
                writer.write("(")

                @parameter
                def serialize[T: Writable](val: T):
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
            abort(t"failed to write DeviceBuffer:{e}")


trait _FunctionEnqueuer:
    """Trait for contexts (DeviceContext or DeviceStream) that can enqueue device functions.
    """

    comptime enqueue_fn_name: StaticString

    def handle(self) -> UnsafePointer[NoneType, MutExternalOrigin]:
        ...


@doc_hidden
struct DeviceStream(ImplicitlyCopyable, _FunctionEnqueuer):
    """Represents a CUDA/HIP stream for asynchronous GPU operations.

    A DeviceStream provides a queue for GPU operations that can execute concurrently
    with operations in other streams. Operations within a single stream execute in
    the order they are issued, but operations in different streams may execute in
    any relative order or concurrently.

    This abstraction allows for better utilization of GPU resources by enabling
    overlapping of computation and data transfers.

    Example:

    ```mojo
    from std.gpu.host import DeviceContext, DeviceStream
    var ctx = DeviceContext(0)  # Select first GPU
    var stream = DeviceStream(ctx)

    # Launch operations on the stream
    # ...

    # Wait for all operations in the stream to complete
    stream.synchronize()
    ```
    """

    comptime enqueue_fn_name: StaticString = (
        "AsyncRT_DeviceStream_enqueueFunctionDirect"
    )
    """C runtime function name for enqueueing a device function on this stream."""

    var _handle: _DeviceStreamPtr[mut=True]
    """Internal handle to the native stream object."""

    @always_inline
    def handle(self) -> UnsafePointer[NoneType, MutExternalOrigin]:
        """Get underlying handle.

        Returns:
            The underlying C context handle as an opaque pointer.
        """
        return self._handle.value().bitcast[NoneType]()

    @doc_hidden
    @always_inline
    def __init__(out self, handle: _DeviceStreamPtr[mut=True]):
        """Initializes a new DeviceStream with the given stream handle.

        Args:
            handle: The stream handle to initialize the DeviceStream with.
        """
        self._handle = handle

    @doc_hidden
    @always_inline
    def __init__(out self, ctx: DeviceContext) raises:
        """Retrieves the stream associated with the given device context.

        Args:
            ctx: The device context to retrieve the stream from.

        Raises:
            If stream creation fails.
        """
        var result: _DeviceStreamPtr[mut=True] = {}
        # const char *AsyncRT_DeviceContext_stream(const DeviceStream **result, const DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_stream",
                _CString[],
            ](UnsafePointer(to=result), ctx._handle)
        )
        self._handle = result

    @doc_hidden
    def __init__(out self, *, copy: Self):
        """Creates a copy of an existing stream by incrementing its reference count.

        Args:
            copy: The stream to copy.
        """
        # void AsyncRT_DeviceStream_retain(const DeviceStream *stream)
        external_call[
            "AsyncRT_DeviceStream_retain",
            NoneType,
        ](copy._handle)
        self._handle = copy._handle

    @doc_hidden
    @always_inline
    def __del__(deinit self):
        """Releases resources associated with this stream."""
        # void AsyncRT_DeviceStream_release(const DeviceStream *stream)
        external_call["AsyncRT_DeviceStream_release", NoneType](
            self._handle,
        )

    @always_inline
    def synchronize(self) raises:
        """Blocks the calling CPU thread until all operations in this stream complete.

        This function waits until all previously issued commands in this stream
        have completed execution. It provides a synchronization point between
        host and device code.

        Raises:
            If synchronization fails.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext

        var ctx = DeviceContext()
        var stream = ctx.create_stream()

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
                _CString[],
            ](self._handle)
        )

    @always_inline
    def enqueue_wait_for(self, event: DeviceEvent) raises:
        """Makes this stream wait for the specified event.

        This function inserts a wait operation into this stream that will
        block all subsequent operations in the stream until the specified
        event has been recorded and completed.

        Args:
            event: The event to wait for.

        Raises:
            If the wait operation fails.
        """
        # const char *AsyncRT_DeviceStream_waitForEvent(const DeviceStream *stream, const DeviceEvent *event)
        _checked(
            external_call[
                "AsyncRT_DeviceStream_waitForEvent",
                _CString[],
            ](self._handle, event._handle)
        )

    @always_inline
    def record_event(self, event: DeviceEvent) raises:
        """Records an event in this stream.

        This function records the given event at the current point in this stream.
        All operations in the stream that were enqueued before this call will
        complete before the event is triggered.

        Args:
            event: The event to record.

        Raises:
            If event recording fails.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext

        var ctx = DeviceContext()

        var default_stream = ctx.stream()
        var new_stream = ctx.create_stream()

        # Create event on the context
        var event = ctx.create_event()

        # Wait for the event on the new stream
        new_stream.enqueue_wait_for(event)

        # Stream 2 can continue
        default_stream.record_event(event)
        ```
        """
        # const char *AsyncRT_DeviceStream_eventRecord(const DeviceStream *stream, const DeviceEvent *event)
        _checked(
            external_call[
                "AsyncRT_DeviceStream_eventRecord",
                _CString[],
            ](self._handle, event._handle)
        )

    @always_inline
    def enqueue_host_func(
        self,
        func: def(OpaquePointer[MutAnyOrigin]) thin -> None,
        user_data: OpaquePointer[MutAnyOrigin],
    ) raises:
        """Enqueues a host callback to run on this stream.

        This corresponds to CUDA's `cuLaunchHostFunc`. The callback `func`
        runs on a driver thread once all preceding work on this stream has
        completed, and receives `user_data` as its only argument. Per the
        CUDA contract, the callback must not call any device APIs.

        Currently only implemented for CUDA streams; other backends raise.

        Args:
            func: A `thin` C-compatible function pointer that accepts a
                single `void*` argument.
            user_data: An opaque pointer passed through to `func` when it
                runs.

        Raises:
            If the underlying device does not support host callbacks, or if
            the driver rejects the enqueue.
        """
        # const char *AsyncRT_DeviceStream_enqueueHostFunc(const DeviceStream *stream, void (*fn)(void *), void *userData)
        _checked(
            external_call[
                "AsyncRT_DeviceStream_enqueueHostFunc",
                _CString[],
            ](self._handle, func, user_data)
        )

    @always_inline
    def enqueue_function[
        *Ts: DevicePassable
    ](
        self,
        f: DeviceFunction,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
    ) raises:
        """Enqueues a checked compiled function for execution on this stream.

        Parameters:
            Ts: Argument types (must be DevicePassable).

        Args:
            f: The checked compiled function to execute.
            args: Arguments to pass to the function.
            grid_dim: Dimensions of the compute grid, made up of thread blocks.
            block_dim: Dimensions of each thread block in the grid.
            cluster_dim: Dimensions of clusters (if the thread blocks are
                grouped into clusters).
            shared_mem_bytes: Amount of shared memory per thread block.
            attributes: Launch attributes.
            constant_memory: Constant memory mapping.

        Raises:
            If the operation fails.
        """
        _check_dim["DeviceStream.enqueue_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceStream.enqueue_function", "block_dim"](
            block_dim, location=call_location()
        )

        f._call_with_pack_checked(
            self,
            *args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=call_location(),
        )


struct EventFlags(TrivialRegisterPassable):
    """Provides flags for creating events.

    These flags can be combined using the bitwise OR operator (`|`, `|=`).
    """

    var _flags: c_uint
    """The flags to pass when creating an event."""

    comptime default = Self(0x00)
    """Default event flags, with timing enabled."""
    comptime blocking_sync = Self(0x01)
    """Allows `event.synchronize()` to block until the event has been recorded."""
    comptime disable_timing = Self(0x02)
    """Removes timing overhead."""
    comptime interprocess = Self(0x04)
    """Enable interprocess synchronization, currently unimplemented."""

    def __init__(out self, flags: c_uint):
        """Initializes a new EventFlags.

        Args:
            flags: The flags to initialize the EventFlags with.
        """
        self._flags = flags

    def __ior__(mut self, other: Self):
        """Combines the current flags with another flag in-place.

        Args:
            other: The flag to combine with the current flags.
        """
        self._flags |= other._flags

    def __or__(self, other: Self) -> Self:
        """Returns the current flags combined with another flag.

        Args:
            other: The flag to combine with the current flags.

        Returns:
            A new EventFlags instance with the combined flags.
        """
        return Self(self._flags | other._flags)


struct DeviceEvent(ImplicitlyCopyable):
    """Represents a GPU event for synchronization between streams.

    A DeviceEvent allows for fine-grained synchronization between different
    GPU streams. Events can be recorded in one stream and waited for in another,
    enabling efficient coordination of asynchronous GPU operations.

    Example:

    ```mojo
    from std.gpu.host import DeviceContext

    var ctx = DeviceContext()

    var default_stream = ctx.stream()
    var new_stream = ctx.create_stream()

    # Create event in default_stream
    var event = ctx.create_event()

    # Wait for the event in new_stream
    new_stream.enqueue_wait_for(event)

    # Stream 2 can continue
    default_stream.record_event(event)
    ```
    """

    var _handle: _DeviceEventPtr[mut=True]
    """Internal handle to the native event object."""

    @doc_hidden
    @always_inline
    def __init__(out self, ctx: DeviceContext) raises:
        """Creates a new event recorded on the given context's default stream.

        Args:
            ctx: The device context to record the event on.

        Raises:
            If event creation or recording fails.
        """
        var result: _DeviceEventPtr[mut=True] = {}
        # const char *AsyncRT_DeviceContext_enqueue_event(const DeviceEvent **result, const DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_enqueue_event",
                _CString[],
            ](UnsafePointer(to=result), ctx._handle)
        )
        self._handle = result

    @doc_hidden
    @always_inline
    def __init__(out self, existing: _DeviceEventPtr[mut=True]):
        """Creates a DeviceEvent from an existing pointer.

        Args:
            existing: Pointer to existing DeviceEvent.
        """
        # Increment the reference count.
        external_call["AsyncRT_DeviceEvent_retain", NoneType](existing)
        self._handle = existing

    @doc_hidden
    def __init__(out self, *, copy: Self):
        """Creates a copy of an existing event by incrementing its reference count.

        Args:
            copy: The event to copy.
        """
        # Increment the reference count.
        external_call["AsyncRT_DeviceEvent_retain", NoneType](copy._handle)
        self._handle = copy._handle

    def __del__(deinit self):
        """Releases resources associated with this event."""
        # void AsyncRT_DeviceEvent_release(const DeviceEvent *event)
        external_call["AsyncRT_DeviceEvent_release", NoneType](
            self._handle,
        )

    @always_inline
    def synchronize(self) raises:
        """Blocks the calling CPU thread until this event completes.

        This function waits until the event has been recorded and all
        operations before the event in the stream have completed.

        Raises:
            If synchronization fails.
        """
        # const char *AsyncRT_DeviceEvent_synchronize(const DeviceEvent *event)
        _checked(
            external_call[
                "AsyncRT_DeviceEvent_synchronize",
                _CString[],
            ](self._handle)
        )


def _is_nvidia_gpu[target: _TargetType]() -> Bool:
    return is_triple["nvptx64-nvidia-cuda", target]()


def _is_path_like(ss: StringSlice) -> Bool:
    return ss.startswith("/") or ss.startswith("~") or ss.startswith("./")


struct DeviceFunction[
    func_type: TrivialRegisterPassable,
    //,
    func: func_type,
    declared_arg_types: Optional[TypeList.of[Trait=AnyType]()._mlir_type],
    *,
    target: _TargetType = get_gpu_target(),
    compile_options: StaticString = CompilationTarget[
        target
    ].default_compile_options(),
    link_options: StaticString = "",
    _ptxas_info_verbose: Bool = False,
](ImplicitlyCopyable):
    """Represents a compiled device function for GPU execution.

    This struct encapsulates a compiled GPU function that can be launched on a device.
    It handles the compilation, loading, and resource management of device functions.

    Parameters:
        func_type: The dtype of the function to compile.
        func: The function to compile for GPU execution.
        declared_arg_types: An optional containing a variadic of the declared dtypes of the kernel signature.
        target: The target architecture for compilation. Defaults to the current GPU target.
        compile_options: The string of compilation options to pass to the compiler.
        link_options: The string of linker options to pass to the linker.
        _ptxas_info_verbose: Whether to enable verbose PTX assembly output. Defaults to False.

    Example:

    ```mojo
    from std.gpu.host import DeviceContext

    def my_kernel():
        # Kernel implementation
        pass

    var ctx = DeviceContext()
    ctx.enqueue_function[my_kernel, my_kernel](grid_dim=1, block_dim=32)
    ctx.synchronize()
    ```
    """

    # emit asm if cross compiling for nvidia gpus.
    comptime _emission_kind = "asm" if (
        _cross_compilation() and _is_nvidia_gpu[Self.target]()
    ) else "object"
    var _handle: _DeviceFunctionPtr[mut=True]
    """Internal handle to the compiled device function."""

    var _func_impl: CompiledFunctionInfo[Self.func_type, Self.func, Self.target]
    """Compilation information for the function."""

    var _context: DeviceContext
    """The device context backing the function."""

    def __init__(out self, *, copy: Self):
        """Creates a copy of an existing DeviceFunction.

        This increases the reference count of the underlying device function handle.

        Args:
            copy: The DeviceFunction to copy from.
        """
        # Increment the reference count before copying the handle.
        #
        # void AsyncRT_DeviceFunction_retain(const DeviceFunction *ctx)
        external_call[
            "AsyncRT_DeviceFunction_retain",
            NoneType,
            _DeviceFunctionPtr[mut=True],
        ](copy._handle)
        self._handle = copy._handle
        self._func_impl = copy._func_impl
        self._context = copy._context

    def __del__(deinit self):
        """Releases resources associated with this DeviceFunction.

        This decrements the reference count of the underlying device function handle.
        """
        # Decrement the reference count held by this struct.
        #
        # void AsyncRT_DeviceFunction_release(const DeviceFunction *ctx)
        external_call[
            "AsyncRT_DeviceFunction_release",
            NoneType,
            _DeviceFunctionPtr[mut=True],
        ](self._handle)

    @doc_hidden
    @always_inline
    def __init__(
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
            If compilation fails or if an unsupported function attribute is provided.
        """
        self._context = ctx

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
        var result: _DeviceFunctionPtr[mut=True] = {}
        self._func_impl = _compile_code[
            Self.func,
            emission_kind=self._emission_kind,
            target=Self.target,
            compile_options=Self.compile_options,
            link_options=Self.link_options,
        ]()
        var debug_level = String(DebugLevel)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_loadFunction",
                _CString[],
                UnsafePointer[_DeviceFunctionPtr[mut=True], origin_of(result)],
                _DeviceContextPtr[mut=True],
                CStringSlice[StaticConstantOrigin],
                CStringSlice[StaticConstantOrigin],
                CStringSlice[StaticConstantOrigin],
                c_size_t,
                Int32,
                CStringSlice[origin_of(debug_level)],
                Int32,
            ](
                UnsafePointer(to=result),
                ctx._handle,
                self._func_impl.module_name.as_c_string_slice(),
                self._func_impl.function_name.as_c_string_slice(),
                self._func_impl.asm.as_c_string_slice(),
                c_size_t(self._func_impl.asm.byte_length()),
                max_dynamic_shared_size_bytes,
                debug_level.as_c_string_slice(),
                Int32(Int(OptimizationLevel)),
            ),
        )
        self._handle = result

    def _copy_to_constant_memory(
        read self, mapping: ConstantMemoryMapping
    ) raises:
        # const char *AsyncRT_DeviceFunction_copyToConstantMemory(
        #     const DeviceFunction *func,
        #     const void *name, size_t nameSize,
        #     const void *data, size_t dataSize)
        _checked(
            external_call[
                "AsyncRT_DeviceFunction_copyToConstantMemory",
                _CString[],
                _DeviceFunctionPtr[mut=True],
                CStringSlice[StaticConstantOrigin],
                c_size_t,
                OpaquePointer[ImmutAnyOrigin],
                c_size_t,
            ](
                self._handle,
                mapping.name.as_c_string_slice(),
                c_size_t(mapping.name.byte_length()),
                mapping.ptr,
                c_size_t(mapping.byte_count),
            )
        )

    @staticmethod
    def _dump_q[name: String, val: _DumpPath]() -> Tuple[Bool, _DumpPath]:
        comptime env_var = "DUMP_GPU_" + name.upper()

        comptime if is_defined[env_var]():
            comptime env_val = get_defined_string[env_var]()

            comptime if _is_bool_like[env_val]():
                comptime env_bool_val = get_defined_bool[env_var]()
                return env_bool_val, _DumpPath(env_bool_val)
            elif _is_path_like(env_val):
                return True, _DumpPath(Path(env_val))
            else:
                comptime assert False, String(
                    "the environment variable '",
                    env_var,
                    (
                        "' is not a valid value. The value should either be"
                        " a boolean value or a path like value, but got '"
                    ),
                    env_val,
                    "'",
                )

        elif val.isa[Bool]():
            return val.unsafe_get[Bool](), val

        elif val.isa[Path]():
            return val.unsafe_get[Path]() != Path(""), val

        elif val.isa[StaticString]():
            return val.unsafe_get[StaticString]() != "", val

        else:
            return val.isa[def() capturing -> Path](), val

    @staticmethod
    def _cleanup_asm(s: StringSlice) -> String:
        return (
            String(s)
            .replace("\t// begin inline asm\n", "")
            .replace("\t// end inline asm\n", "")
            .replace("\t;;#ASMSTART\n", "")
            .replace("\t;;#ASMEND\n", "")
        )

    def _expand_path(read self, path: Path) -> Path:
        """If the path contains a `%` character, it is replaced with the module
        name. This allows one to dump multiple kernels which are disambiguated
        by the module name.
        """
        return String(path).replace("%", self._func_impl.module_name)

    def _expand_path(read self, path: StaticString) -> Path:
        """If the path contains a `%` character, it is replaced with the module
        name. This allows one to dump multiple kernels which are disambiguated
        by the module name.
        """
        return String(path).replace("%", self._func_impl.module_name)

    @no_inline
    def dump_rep[
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
    ](read self) raises:
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

        # Get ASM - either from the pre-compiled func_impl or by compiling now
        @parameter
        def get_asm() -> StaticString:
            comptime if Self._emission_kind == "asm":
                return self._func_impl.asm
            return _compile_code[
                Self.func,
                emission_kind="asm",
                target=Self.target,
                compile_options=Self.compile_options,
                link_options=Self.link_options,
            ]().asm

        comptime if Self._ptxas_info_verbose:
            print(_ptxas_compile[Self.target](String(get_asm()), options="-v"))

        comptime dump_asm_tup = Self._dump_q["asm", dump_asm]()
        comptime do_dump_asm = dump_asm_tup[0]
        comptime dump_asm_val = dump_asm_tup[1]

        comptime if do_dump_asm:
            var asm = self._cleanup_asm(get_asm())

            comptime if dump_asm_val.isa[def() capturing -> Path]():
                comptime dump_asm_fn = dump_asm_val.unsafe_get[
                    def() capturing -> Path
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

        comptime dump_sass_tup = Self._dump_q["sass", _dump_sass]()
        comptime do_dump_sass = dump_sass_tup[0]
        comptime dump_sass_val = dump_sass_tup[1]

        comptime if do_dump_sass:
            var ptx = Self._cleanup_asm(get_asm())
            var sass = _to_sass[Self.target](ptx)

            comptime if dump_sass_val.isa[def() capturing -> Path]():
                comptime _dump_sass_fn = dump_sass_val.unsafe_get[
                    def() capturing -> Path
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

        comptime dump_llvm_tup = Self._dump_q["llvm", dump_llvm]()
        comptime do_dump_llvm = dump_llvm_tup[0]
        comptime dump_llvm_val = dump_llvm_tup[1]

        comptime if do_dump_llvm:
            var llvm = _compile_code[
                Self.func,
                emission_kind="llvm-opt",
                target=Self.target,
                compile_options=Self.compile_options,
                link_options=Self.link_options,
            ]().asm

            comptime if dump_llvm_val.isa[def() capturing -> Path]():
                comptime dump_llvm_fn = dump_llvm_val.unsafe_get[
                    def() capturing -> Path
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

    # Enqueue function on a stream
    @always_inline
    @parameter
    def _call_with_pack[
        *Ts: AnyType,
        ContextT: _FunctionEnqueuer,
    ](
        read self,
        ctx: ContextT,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        comptime num_args = Ts.size
        var num_captures = max(0, self._func_impl.num_captures)
        comptime populate = type_of(self._func_impl).populate
        comptime num_captures_static = 16

        # NOTE: Manual short buffer optimization. We could use a
        # Variant[List, InlineArray] instead, but it would look a lot more
        # verbose. This way, however, we need to conditionally free at the end.
        var dense_args_addrs: UnsafePointer[
            OpaquePointer[MutAnyOrigin], MutAnyOrigin
        ]
        var dense_args_sizes: UnsafePointer[UInt64, MutAnyOrigin]
        if num_captures > num_captures_static:
            dense_args_addrs = alloc[OpaquePointer[MutAnyOrigin]](
                num_captures + num_args
            )
            dense_args_sizes = alloc[UInt64](num_captures + num_args)
            for i in range(num_captures + num_args):
                dense_args_sizes[i] = 0
        else:
            dense_args_addrs = stack_allocation[
                num_captures_static + num_args, OpaquePointer[MutAnyOrigin]
            ]()
            dense_args_sizes = stack_allocation[
                num_captures_static + num_args, UInt64
            ]()
            for i in range(num_captures_static + num_args):
                dense_args_sizes[i] = 0

        comptime for i in range(num_args):
            # TODO(MSTDL-1904): Validate the safety of this.
            dense_args_addrs[i] = (
                UnsafePointer(to=args[i])
                .bitcast[NoneType]()
                .unsafe_mut_cast[True]()
            )

        @parameter
        def _populate_arg_sizes[i: Int]():
            dense_args_sizes[i] = UInt64(size_of[Ts[i]]())

        comptime for i in range(num_args):
            _populate_arg_sizes[i]()

        if cluster_dim:
            attributes.append(
                LaunchAttribute.from_cluster_dim(cluster_dim.value())
            )

        for i in range(len(constant_memory)):
            self._copy_to_constant_memory(constant_memory[i])

        if num_captures > 0:
            for i in range(num_captures):
                dense_args_sizes[num_args + i] = self._func_impl.capture_sizes[
                    i
                ]
            # Call the populate function to initialize the captured values in the arguments array.
            # The captured values are always at the end of the argument list.
            # This function (generated by the compiler) has to be inlined here
            # and be in the same scope as the user of dense_args_addr
            # (i.e. the following external_call).
            # Because this closure uses stack allocated ptrs
            # to store the captured values in dense_args_addrs, they need to
            # not go out of the scope before dense_args_addr is being use.
            var capture_args_start = dense_args_addrs + num_args
            populate(capture_args_start.bitcast[NoneType]())

        _checked_call[Self.func](
            external_call[ContextT.enqueue_fn_name, _CString[]](
                ctx.handle(),
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
                UInt32(num_args + num_captures),
                dense_args_sizes,
            ),
            device_context=self._context,
            location=location.or_else(call_location()),
        )

        if num_captures > num_captures_static:
            dense_args_addrs.free()
            dense_args_sizes.free()

    @always_inline
    @staticmethod
    def _validate_arguments[
        *Ts: DevicePassable,
        num_args: Int,
    ]() -> Tuple[Int, InlineArray[Int, num_args]]:
        comptime declared_num_args = TypeList[
            Self.declared_arg_types.value()
        ].size

        comptime assert (
            declared_num_args == num_args
        ), "Wrong number of arguments to enqueue"

        # For each argument determine the size of the device dtype and
        # calculate the offset into a contiguous memory area which will
        # be used to remap the passed arguments into the device dtypes.
        var tmp_arg_offset = 0
        var translated_arg_offsets = InlineArray[Int, num_args](
            uninitialized=True
        )
        var num_translated_args = 0

        comptime for i in range(num_args):
            comptime declared_arg_type = TypeList[
                Self.declared_arg_types.value()
            ]()[i]
            comptime actual_arg_type = Ts[i]

            def declared_arg_type_name() -> String:
                comptime if conforms_to(declared_arg_type, DevicePassable):
                    return downcast[
                        declared_arg_type, DevicePassable
                    ].get_type_name()
                else:
                    return reflect[declared_arg_type].name()

            # Now we'll check if the given argument's device_type is
            # what the kernel expects.

            # First, check if they're handing in a device dtype, in other
            # words, a dtype that can be passed directly and doesn't need to
            # be mapped. For example, Int, IndexList, etc.
            comptime is_convertible: Bool = actual_arg_type._is_convertible_to_device_type[
                declared_arg_type
            ]()

            comptime if _type_is_eq[
                actual_arg_type, actual_arg_type.device_type
            ]():
                # Now check if they handed in the *correct* device dtype.
                comptime assert is_convertible, String(
                    "argument #",
                    i,
                    " of type '",
                    actual_arg_type.get_type_name(),
                    "' does not match the declared function argument type '",
                    declared_arg_type_name(),
                    "'",
                )
            else:
                # They handed in a host dtype, in other words, a dtype that
                # needs to be mapped before handing it to the device. In
                # this case, we use a more informative error message.
                comptime assert is_convertible, String(
                    "argument #",
                    i,
                    " of type '",
                    actual_arg_type.get_type_name(),
                    "' (which became device of type '",
                    declared_arg_type_name(),
                    "') does not match the declared function argument type",
                )
            var aligned_type_size = align_up(
                size_of[actual_arg_type.device_type](), 8
            )
            if aligned_type_size != 0:
                num_translated_args += 1
                translated_arg_offsets[i] = tmp_arg_offset
                tmp_arg_offset += aligned_type_size
            else:
                translated_arg_offsets[i] = -1

        return (num_translated_args, translated_arg_offsets^)

    @always_inline
    @parameter
    def _call_with_pack_checked[
        *Ts: DevicePassable,
        ContextT: _FunctionEnqueuer,
    ](
        read self,
        ctx: ContextT,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        # We need to keep track of both the number of arguments pushed by the
        # caller and the number of translated arguments expected by the kernel.
        comptime num_passed_args = Ts.size
        var num_translated_args = 0

        var translated_arg_offsets = InlineArray[Int, num_passed_args](
            uninitialized=True
        )

        # Validate that all actual arguments do remap to the declared device
        # dtype in the kernel.
        comptime if Self.declared_arg_types:
            var validated_args = Self._validate_arguments[
                *Ts, num_args=num_passed_args
            ]()
            num_translated_args = validated_args[0]
            translated_arg_offsets = validated_args[1].copy()

        var num_captures = max(0, self._func_impl.num_captures)
        comptime populate = type_of(self._func_impl).populate
        comptime num_captures_static = 16

        # We need the total byte size of arguments as a compile time constant,
        # so we break out the calculation into a function executed at compile
        # time.
        @parameter
        def calculate_args_size() -> Int:
            var tmp_args_size = 8  # always reserve 8 extra bytes for alignment.

            comptime for i in range(num_passed_args):
                comptime actual_arg_type = Ts[i]
                tmp_args_size += align_up(
                    size_of[actual_arg_type.device_type](), 8
                )
            return tmp_args_size

        comptime args_size = calculate_args_size()

        # Space to store the arguments to the kernel that have been converted
        # from host dtype to device dtype.
        var translated_args = InlineArray[Byte, args_size](uninitialized=True)
        var start_addr = Int(translated_args.unsafe_ptr())
        var extra_align = align_up(start_addr, 8) - start_addr

        # NOTE: Manual short buffer optimization. We could use a
        # Variant[List, InlineArray] instead, but it would look a lot more
        # verbose. This way, however, we need to conditionally free at the end.
        var dense_args_addrs: UnsafePointer[
            OpaquePointer[MutAnyOrigin], MutAnyOrigin
        ]
        var dense_args_sizes: UnsafePointer[UInt64, MutAnyOrigin]
        if num_captures > num_captures_static:
            dense_args_addrs = alloc[OpaquePointer[MutAnyOrigin]](
                num_captures + num_passed_args
            )
            dense_args_sizes = alloc[UInt64](num_captures + num_passed_args)
            for i in range(num_captures + num_passed_args):
                dense_args_sizes[i] = 0
        else:
            dense_args_addrs = stack_allocation[
                num_captures_static + num_passed_args,
                OpaquePointer[MutAnyOrigin],
            ]()
            dense_args_sizes = stack_allocation[
                num_captures_static + num_passed_args, UInt64
            ]()
            for i in range(num_captures_static + num_passed_args):
                dense_args_sizes[i] = 0

        # Since we skip over zero sized declared dtypes when passing arguments
        # we need to know the current count of arguments pushed.
        var translated_arg_idx = 0

        comptime for i in range(num_passed_args):
            # If the arg offset is negative then the corresponding declared
            # dtype is zero sized and we do not push the argument to the kernel.
            var translated_arg_offset = translated_arg_offsets[i]
            if translated_arg_offset >= 0:
                comptime actual_arg_type = Ts[i]
                var first_word_addr = UnsafePointer(
                    to=translated_args.unsafe_ptr()[
                        translated_arg_offset + extra_align
                    ]
                ).bitcast[NoneType]()
                args[i]._to_device_type(first_word_addr)
                dense_args_addrs[translated_arg_idx] = first_word_addr
                dense_args_sizes[translated_arg_idx] = UInt64(
                    size_of[actual_arg_type.device_type]()
                )
                translated_arg_idx += 1

        if cluster_dim:
            attributes.append(
                LaunchAttribute.from_cluster_dim(cluster_dim.value())
            )

        if constant_memory:
            for i in range(len(constant_memory)):
                self._copy_to_constant_memory(constant_memory[i])

        if num_captures > 0:
            for i in range(num_captures):
                dense_args_sizes[
                    num_passed_args + i
                ] = self._func_impl.capture_sizes[i]
            # Call the populate function to initialize the captured values in the arguments array.
            # The captured values are always at the end of the argument list.
            # This function (generated by the compiler) has to be inlined here
            # and be in the same scope as the user of dense_args_addr
            # (i.e. the following external_call).
            # Because this closure uses stack allocated ptrs
            # to store the captured values in dense_args_addrs, they need to
            # not go out of the scope before dense_args_addr is being use.
            var capture_args_start = dense_args_addrs + num_translated_args
            populate(capture_args_start.bitcast[NoneType]())

        _checked_call[Self.func](
            external_call[ContextT.enqueue_fn_name, _CString[]](
                ctx.handle(),
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
                UInt32(num_translated_args + num_captures),
                dense_args_sizes,
            ),
            device_context=self._context,
            location=location.or_else(call_location()),
        )

        if num_captures > num_captures_static:
            dense_args_addrs.free()
            dense_args_sizes.free()

    @always_inline
    def get_attribute(self, attr: Attribute) raises -> Int:
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
        from std.gpu.host import Attribute, DeviceContext

        def kernel():
            pass

        var ctx = DeviceContext()
        var device_function = ctx.compile_function[kernel, kernel]()

        # Get the maximum number of threads per block for this function
        var max_threads = device_function.get_attribute(Attribute.MAX_THREADS_PER_BLOCK)
        ```
        """
        var result: Int32 = 0
        # const char *AsyncRT_DeviceFunction_getAttribute(int32_t *result, const DeviceFunction *func, int32_t attr_code)
        _checked(
            external_call[
                "AsyncRT_DeviceFunction_getAttribute",
                _CString[],
                UnsafePointer[Int32, origin_of(result)],
                _DeviceFunctionPtr[mut=True],
                Int32,
            ](
                UnsafePointer(to=result),
                self._handle,
                attr.code,
            )
        )
        return Int(result)

    @always_inline
    def occupancy_max_active_blocks_per_multiprocessor(
        self, block_size: Int, dynamic_shared_mem_size: Int
    ) raises -> Int:
        """Returns the maximum number of active blocks per multiprocessor for the given function.

        Args:
            block_size: The number of threads per block.
            dynamic_shared_mem_size: The size of dynamically allocated shared memory in bytes.

        Returns:
            The maximum number of active blocks that can run concurrently per multiprocessor.

        Raises:
            If the occupancy calculation fails.
        """
        var result: Int32 = 0
        # const char *AsyncRT_occupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const DeviceContext *ctx, const DeviceFunction *func, int blockSize, size_t dynamicSharedMemSize)
        _checked(
            external_call[
                "AsyncRT_occupancyMaxActiveBlocksPerMultiprocessor",
                _CString[],
                UnsafePointer[Int32, origin_of(result)],
                _DeviceFunctionPtr[mut=True],
                Int32,
                c_size_t,
            ](
                UnsafePointer(to=result),
                self._handle,
                Int32(block_size),
                c_size_t(dynamic_shared_mem_size),
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

    var _handle: _DeviceFunctionPtr[mut=True]
    """Internal handle to the native device function object."""

    def __init__(out self, *, copy: Self):
        """Creates a copy of an existing device function by incrementing its reference count.

        Args:
            copy: The device function to copy.
        """
        # Increment the reference count before copying the handle.
        #
        # void AsyncRT_DeviceFunction_retain(const DeviceFunction *ctx)
        external_call[
            "AsyncRT_DeviceFunction_retain",
            NoneType,
            _DeviceFunctionPtr[mut=True],
        ](copy._handle)
        self._handle = copy._handle

    def __del__(deinit self):
        """Releases resources associated with this device function."""
        # Decrement the reference count held by this struct.
        #
        # void AsyncRT_DeviceFunction_release(const DeviceFunction *ctx)
        external_call[
            "AsyncRT_DeviceFunction_release",
            NoneType,
            _DeviceFunctionPtr[mut=True],
        ](self._handle)

    @doc_hidden
    @always_inline
    def __init__(
        out self,
        ctx: DeviceContext,
        info: CompiledFunctionInfo,
        *,
        func_attribute: OptionalReg[FuncAttribute] = None,
    ) raises:
        """Initializes a new device function from CompileInfo object.

        Args:
            ctx: The device context to associate this function with.
            info: The result from the compile command (must be compiled to object).
            func_attribute: Optional function attributes like shared memory size.

        Raises:
            If function loading fails or if an unsupported attribute is provided.
        """
        if info.emission_kind != "object":
            raise Error(
                "the function is not compiled to object code",
            )
        return {
            ctx,
            function_name = info.function_name,
            asm = info.asm,
            func_attribute = func_attribute,
        }

    @doc_hidden
    @always_inline
    def __init__(
        out self,
        ctx: DeviceContext,
        *,
        var function_name: String,
        var asm: String,
        func_attribute: OptionalReg[FuncAttribute] = None,
    ) raises:
        """Initializes a new device function from assembly code.

        Args:
            ctx: The device context to associate this function with.
            function_name: The name of the function in the assembly code.
            asm: The assembly code containing the function.
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
        var module_name = StaticString("")
        var result: _DeviceFunctionPtr[mut=True] = {}
        var debug_level = String(DebugLevel)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_loadFunction",
                _CString[],
                UnsafePointer[_DeviceFunctionPtr[mut=True], origin_of(result)],
                _DeviceContextPtr[mut=True],
                CStringSlice[StaticConstantOrigin],
                CStringSlice[origin_of(function_name)],
                CStringSlice[origin_of(asm)],
                c_size_t,
                Int32,
                CStringSlice[origin_of(debug_level)],
                Int32,
            ](
                UnsafePointer(to=result),
                ctx._handle,
                module_name.as_c_string_slice(),
                function_name.as_c_string_slice(),
                asm.as_c_string_slice(),
                c_size_t(asm.byte_length()),
                max_dynamic_shared_size_bytes,
                debug_level.as_c_string_slice(),
                Int32(Int(OptimizationLevel)),
            )
        )
        self._handle = result

    @always_inline
    def _copy_to_constant_memory(
        read self, mapping: ConstantMemoryMapping
    ) raises:
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
                _CString[],
                _DeviceFunctionPtr[mut=True],
                type_of(mapping.name.as_c_string_slice()),
                c_size_t,
                OpaquePointer[MutAnyOrigin],
                c_size_t,
            ](
                self._handle,
                mapping.name.as_c_string_slice(),
                c_size_t(mapping.name.byte_length()),
                mapping.ptr,
                c_size_t(mapping.byte_count),
            )
        )

    @always_inline
    @parameter
    def _call_with_pack[
        *Ts: AnyType
    ](
        read self,
        ctx: DeviceContext,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        location: OptionalReg[SourceLocation] = None,
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
            location: Source location for the function call.

        Raises:
            If the function launch fails.
        """
        comptime num_args = Ts.size

        var dense_args_addrs = InlineArray[
            OpaquePointer[MutAnyOrigin], num_args
        ](uninitialized=True)

        comptime for i in range(num_args):
            # TODO(MSTDL-1904): Validate the safety of this.
            dense_args_addrs[i] = (
                UnsafePointer(to=args[i])
                .bitcast[NoneType]()
                .unsafe_mut_cast[True]()
            )

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
        #                                                         void **args, uint32_t argCount, const size_t *argSizes)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_enqueueFunctionDirect",
                _CString[],
                _DeviceContextPtr[mut=True],
                _DeviceFunctionPtr[mut=True],
                UInt32,
                UInt32,
                UInt32,
                UInt32,
                UInt32,
                UInt32,
                UInt32,
                UnsafePointer[LaunchAttribute, MutAnyOrigin],
                UInt32,
                UnsafePointer[OpaquePointer[MutAnyOrigin], MutAnyOrigin],
                UInt32,
                Optional[UnsafePointer[UInt64, MutAnyOrigin]],
            ](
                ctx._handle,
                self._handle,
                UInt32(grid_dim.x()),
                UInt32(grid_dim.y()),
                UInt32(grid_dim.z()),
                UInt32(block_dim.x()),
                UInt32(block_dim.y()),
                UInt32(block_dim.z()),
                UInt32(shared_mem_bytes.or_else(0)),
                attributes.unsafe_ptr(),
                UInt32(len(attributes)),
                dense_args_addrs.unsafe_ptr(),
                UInt32(num_args),
                None,
            )
        )

    @always_inline
    def get_attribute(self, attr: Attribute) raises -> Int:
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
                _CString[],
                UnsafePointer[Int32, origin_of(result)],
                _DeviceFunctionPtr[mut=True],
                Int32,
            ](
                UnsafePointer(to=result),
                self._handle,
                attr.code,
            )
        )
        return Int(result)


struct DeviceGraph(ImplicitlyCopyable):
    """Represents an instantiated device graph that can be replayed.

    A `DeviceGraph` captures a sequence of GPU operations (such as kernel
    launches) as a reusable graph. Once instantiated from a
    `DeviceGraphBuilder`, the graph can be replayed multiple times at a
    lower overhead than re-enqueueing each operation individually.

    To obtain a `DeviceGraph`, use
    [`DeviceGraphBuilder.instantiate()`](/docs/std/gpu/host/device_context/DeviceGraphBuilder/#instantiate).
    """

    var _handle: _DeviceGraphPtr[mut=True]

    @doc_hidden
    def __init__(out self, handle: _DeviceGraphPtr[mut=True]):
        self._handle = handle

    def __init__(out self, *, copy: Self):
        """Creates a copy of an existing device graph by incrementing its
        reference count.

        Args:
            copy: The device graph to copy.
        """
        # void AsyncRT_DeviceGraph_retain(DeviceGraph *graph)
        external_call[
            "AsyncRT_DeviceGraph_retain", NoneType, _DeviceGraphPtr[mut=True]
        ](copy._handle)
        self._handle = copy._handle

    def __del__(deinit self):
        """Releases resources associated with this device graph."""
        # void AsyncRT_DeviceGraph_release(DeviceGraph *graph)
        external_call[
            "AsyncRT_DeviceGraph_release", NoneType, _DeviceGraphPtr[mut=True]
        ](self._handle)

    def replay(self) raises:
        """Replays the captured sequence of GPU operations.

        Submits the pre-captured sequence of operations for execution on the
        device. This is more efficient than re-enqueueing each operation
        individually because the graph has already been compiled and
        instantiated by the driver.

        Raises:
            If replay fails.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext

        def kernel():
            print("replaying")

        with DeviceContext() as ctx:
            var compiled_fn = ctx.compile_function[kernel, kernel]()
            var builder = ctx.create_graph_builder()
            builder.add_function(compiled_fn, grid_dim=1, block_dim=1)
            var graph = builder^.instantiate()
            graph.replay()
            graph.replay()  # replay as many times as needed
            ctx.synchronize()
        ```
        """
        # const char *AsyncRT_DeviceGraph_replay(DeviceGraph *graph)
        _checked(
            external_call[
                "AsyncRT_DeviceGraph_replay",
                _CString[],
                _DeviceGraphPtr[mut=True],
            ](self._handle)
        )


struct DeviceGraphBuilder(Movable, _FunctionEnqueuer):
    """Builder for explicit device graph construction.

    A `DeviceGraphBuilder` is obtained from
    [`DeviceContext.create_graph_builder()`](/docs/std/gpu/host/device_context/DeviceContext/#create_graph_builder).
    Callers add kernel nodes via `add_function()` and then call
    `instantiate()` to produce a reusable `DeviceGraph`.

    Example:

    ```mojo
    from std.gpu.host import DeviceContext

    def kernel(x: Int):
        print("Value:", x)

    with DeviceContext() as ctx:
        var compiled_fn = ctx.compile_function[kernel, kernel]()
        var builder = ctx.create_graph_builder()
        builder.add_function(compiled_fn, 42, grid_dim=1, block_dim=1)
        var graph = builder^.instantiate()
        graph.replay()
        ctx.synchronize()
    ```
    """

    comptime enqueue_fn_name: StaticString = (
        "AsyncRT_DeviceGraphBuilder_addFunctionDirect"
    )
    """C runtime function name used by `_FunctionEnqueuer` to add a kernel node."""

    var _handle: _DeviceGraphBuilderPtr[mut=True]
    var _ctx: DeviceContext

    @always_inline
    def handle(self) -> UnsafePointer[NoneType, MutExternalOrigin]:
        """Gets the underlying C builder handle.

        Returns:
            The underlying C builder handle as an opaque pointer.
        """
        return self._handle.value().bitcast[NoneType]()

    @doc_hidden
    def __init__(
        out self,
        handle: _DeviceGraphBuilderPtr[mut=True],
        ctx: DeviceContext,
    ):
        self._handle = handle
        self._ctx = ctx

    def __init__(out self, *, copy: Self):
        """Creates a copy of an existing graph builder by incrementing its
        reference count.

        Args:
            copy: The graph builder to copy.
        """
        # void AsyncRT_DeviceGraphBuilder_retain(DeviceGraphBuilder *builder)
        external_call[
            "AsyncRT_DeviceGraphBuilder_retain",
            NoneType,
            _DeviceGraphBuilderPtr[mut=True],
        ](copy._handle)
        self._handle = copy._handle
        self._ctx = copy._ctx

    def __del__(deinit self):
        """Releases resources associated with this graph builder."""
        # void AsyncRT_DeviceGraphBuilder_release(DeviceGraphBuilder *builder)
        external_call[
            "AsyncRT_DeviceGraphBuilder_release",
            NoneType,
            _DeviceGraphBuilderPtr[mut=True],
        ](self._handle)

    @parameter
    @always_inline
    def add_function[
        *Ts: DevicePassable
    ](
        self,
        f: DeviceFunction,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
    ) raises:
        """Adds a type-checked compiled kernel function as a node in this graph.

        Parameters:
            Ts: Argument types (must be `DevicePassable`).

        Args:
            f: The type-checked compiled function to add. Must have been
                compiled via `DeviceContext.compile_function()`.
            args: Arguments to pass to the kernel.
            grid_dim: Dimensions of the compute grid.
            block_dim: Dimensions of each thread block.
            cluster_dim: Cluster dimensions (optional).
            shared_mem_bytes: Amount of dynamic shared memory per block.
            attributes: Launch attributes.
            constant_memory: Constant memory mappings.

        Raises:
            If adding the node fails.
        """
        _check_dim["DeviceGraphBuilder.add_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceGraphBuilder.add_function", "block_dim"](
            block_dim, location=call_location()
        )
        comptime assert Bool(
            f.declared_arg_types
        ), "Calling a non-checked DeviceFunction; use the unchecked overload."
        f._call_with_pack_checked(
            self,
            *args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=call_location(),
        )

    @always_inline
    def add_function[
        FuncType: def() register_passable -> None,
        //,
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
    ](
        self,
        func: FuncType,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
    ) raises:
        """Compiles and adds a capturing kernel closure as a node in this graph.

        This overload is for kernels that capture variables from their
        enclosing scope using the `{var}` capture syntax. Compilation is
        performed automatically using the `DeviceContext` that created this
        builder, so no separate compile step is needed.

        Parameters:
            FuncType: The type of the closure function (usually inferred).
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
            func: The capturing kernel closure to compile and add as a graph
                node.
            grid_dim: Dimensions of the compute grid.
            block_dim: Dimensions of each thread block.
            cluster_dim: Cluster dimensions (optional).
            shared_mem_bytes: Amount of dynamic shared memory per block.
            attributes: Launch attributes.
            constant_memory: Constant memory mappings.

        Raises:
            If adding the node fails.

        Example:

        ```mojo
        from std.gpu import global_idx
        from std.gpu.host import DeviceContext

        with DeviceContext() as ctx:
            var scale: Float32 = 2.0
            var buf = ctx.enqueue_create_buffer[DType.float32](256)
            var ptr = buf.unsafe_ptr()

            def scale_kernel() {var}:
                var i = global_idx.x
                ptr[i] = Float32(i) * scale

            var builder = ctx.create_graph_builder()
            builder.add_function(scale_kernel, grid_dim=1, block_dim=256)
            var graph = builder^.instantiate()
            graph.replay()
            ctx.synchronize()
        ```
        """
        _check_dim["DeviceGraphBuilder.add_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceGraphBuilder.add_function", "block_dim"](
            block_dim, location=call_location()
        )
        var compiled = self._ctx.compile_function_unchecked[
            FuncType.__call__,
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            _dump_sass=_dump_sass,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ]()
        compiled._call_with_pack(
            self,
            func,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=call_location(),
        )

    @always_inline
    def add_copy[
        dtype: DType
    ](
        self,
        dst_buf: DeviceBuffer[dtype, ...],
        src_buf: HostBuffer[dtype, ...],
    ) raises:
        """Adds a host-to-device memcpy node to the graph.

        The number of bytes copied is determined by the size of the device
        buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_buf: Host buffer to copy from.

        Raises:
            If adding the node fails.
        """
        # const char *AsyncRT_DeviceGraphBuilder_addCopyHostToDevice(
        #     DeviceGraphBuilder *builder, DeviceBuffer *dst, const void *src)
        _checked(
            external_call[
                "AsyncRT_DeviceGraphBuilder_addCopyHostToDevice",
                _CString[],
            ](
                self._handle,
                dst_buf._handle,
                src_buf._host_ptr,
            )
        )

    @always_inline
    def add_copy[
        dtype: DType
    ](
        self,
        dst_buf: HostBuffer[dtype, ...],
        src_buf: DeviceBuffer[dtype, ...],
    ) raises:
        """Adds a device-to-host memcpy node to the graph.

        The number of bytes copied is determined by the size of the device
        buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Host buffer to copy to.
            src_buf: Device buffer to copy from.

        Raises:
            If adding the node fails.
        """
        # const char *AsyncRT_DeviceGraphBuilder_addCopyDeviceToHost(
        #     DeviceGraphBuilder *builder, void *dst, DeviceBuffer *src)
        _checked(
            external_call[
                "AsyncRT_DeviceGraphBuilder_addCopyDeviceToHost",
                _CString[],
            ](
                self._handle,
                dst_buf._host_ptr,
                src_buf._handle,
            )
        )

    @always_inline
    def add_copy[
        dtype: DType
    ](
        self,
        dst_buf: DeviceBuffer[dtype, ...],
        src_buf: DeviceBuffer[dtype, ...],
    ) raises:
        """Adds a device-to-device memcpy node to the graph.

        Both buffers must belong to the same context as this builder;
        cross-context copies are not supported in graphs. The number of bytes
        copied is determined by the size of the source buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_buf: Device buffer to copy from. Must be the same size as
                `dst_buf`.

        Raises:
            If adding the node fails.
        """
        # const char *AsyncRT_DeviceGraphBuilder_addCopyDeviceToDevice(
        #     DeviceGraphBuilder *builder, DeviceBuffer *dst, DeviceBuffer *src)
        _checked(
            external_call[
                "AsyncRT_DeviceGraphBuilder_addCopyDeviceToDevice",
                _CString[],
                _DeviceGraphBuilderPtr[mut=True],
                _DeviceBufferPtr[mut=True],
                _DeviceBufferPtr[mut=True],
            ](
                self._handle,
                dst_buf._handle,
                src_buf._handle,
            )
        )

    @always_inline
    def add_memset[
        dtype: DType
    ](self, dst: DeviceBuffer[dtype, ...], val: Scalar[dtype]) raises:
        """Adds a memset node to the graph that sets all elements of `dst` to
        `val`.

        Parameters:
            dtype: Type of the data stored in the buffer.

        Args:
            dst: Destination buffer.
            val: Value to set all elements of `dst` to.

        Raises:
            If adding the node fails. The underlying graph APIs cannot express
            an 8-byte memset whose high and low 32-bit halves differ as a
            single node, so such patterns will return an error.
        """
        comptime bitwidth = bit_width_of[dtype]()
        comptime assert (
            bitwidth == 8 or bitwidth == 16 or bitwidth == 32 or bitwidth == 64
        ), "bitwidth of memset dtype must be one of [8,16,32,64]"
        var value: UInt64

        comptime if bitwidth == 8:
            value = UInt64(Int(bitcast[DType.uint8, 1](val)))
        elif bitwidth == 16:
            value = UInt64(Int(bitcast[DType.uint16, 1](val)))
        elif bitwidth == 32:
            value = UInt64(bitcast[DType.uint32, 1](val))
        else:
            value = bitcast[DType.uint64, 1](val)

        # const char *AsyncRT_DeviceGraphBuilder_addSetMemory(
        #     DeviceGraphBuilder *builder, DeviceBuffer *dst, uint64_t val,
        #     size_t valSize)
        _checked(
            external_call[
                "AsyncRT_DeviceGraphBuilder_addSetMemory",
                _CString[],
                _DeviceGraphBuilderPtr[mut=True],
                _DeviceBufferPtr[mut=True],
                UInt64,
                c_size_t,
            ](
                self._handle,
                dst._handle,
                value,
                c_size_t(size_of[dtype]()),
            )
        )

    def instantiate(var self) raises -> DeviceGraph:
        """Instantiates the constructed graph into an executable device graph.

        Finalizes the graph construction and produces a `DeviceGraph` that
        can be replayed multiple times.

        Returns:
            The instantiated device graph.

        Raises:
            If instantiation fails.
        """
        var result: _DeviceGraphPtr[mut=True] = {}
        # const char *AsyncRT_DeviceGraphBuilder_instantiate(
        #     DeviceGraph **result, DeviceGraphBuilder *builder)
        _checked(
            external_call[
                "AsyncRT_DeviceGraphBuilder_instantiate",
                _CString[],
                UnsafePointer[_DeviceGraphPtr[mut=True], origin_of(result)],
                _DeviceGraphBuilderPtr[mut=True],
            ](
                UnsafePointer(to=result),
                self._handle,
            )
        )
        return DeviceGraph(result)


struct DeviceContext(ImplicitlyCopyable, RegisterPassable, _FunctionEnqueuer):
    """Represents a single stream of execution on a particular accelerator
    (GPU).

    A `DeviceContext` serves as the low-level interface to the
    accelerator inside a MAX [custom operation](/max/develop/custom-ops/) and provides
    methods for allocating buffers on the device, copying data between host and
    device, and for compiling and running functions (also known as kernels) on
    the device.

    The device context can be used as a
    [context manager](/docs/manual/errors/#use-a-context-manager). For example:

    ```mojo
    from std.gpu.host import DeviceContext
    from std.gpu import thread_idx

    def kernel():
        print("hello from thread:", thread_idx.x, thread_idx.y, thread_idx.z)

    with DeviceContext() as ctx:
        ctx.enqueue_function[kernel, kernel](grid_dim=1, block_dim=(2, 2, 2))
        ctx.synchronize()
    ```

    A custom operation receives an opaque `DeviceContextPtr`, which provides
    a `get_device_context()` method to retrieve the device context:

    ```text
    from std.runtime.asyncrt import DeviceContextPtr
    from compiler import register

    @register("custom_op")
    struct CustomOp:
        @staticmethod
        def execute(ctx_ptr: DeviceContextPtr) raises:
            var ctx = ctx_ptr.get_device_context()
            ctx.enqueue_function[kernel, kernel](grid_dim=1, block_dim=(2, 2, 2))
            ctx.synchronize()
    ```
    """

    comptime enqueue_fn_name: StaticString = (
        "AsyncRT_DeviceContext_enqueueFunctionDirect"
    )
    """C runtime function name for enqueueing a device function on this context."""

    comptime default_device_info = GPUInfo.from_name[_accelerator_arch()]()
    """`GPUInfo` object for the default accelerator."""

    var _handle: _DeviceContextPtr[mut=True]
    var _owning: Bool

    @always_inline
    def handle(self) -> UnsafePointer[NoneType, MutExternalOrigin]:
        """Get underlying handle.

        Returns:
            The underlying C context handle as an opaque pointer.
        """
        return self._handle.value().bitcast[NoneType]()

    @always_inline
    def __init__(
        out self,
        device_id: Int = 0,
        *,
        var api: String = String(Self.default_device_info.api),
    ) raises:
        """Constructs a `DeviceContext` for the specified device.

        This initializer creates a new device context for the specified accelerator device.
        The device context provides an interface for interacting with the GPU, including
        memory allocation, data transfer, and kernel execution.

        Args:
            device_id: ID of the accelerator device. If not specified, uses
                the default accelerator (device 0).
            api: Requested device API (for example, "cuda" or "hip"). Defaults
                to the device API specified by current target accelerator.

        Raises:
            If device initialization fails or the specified device is not available.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext

        # Create a context for the default GPU
        var ctx = DeviceContext()

        # Create a context for a specific GPU (device 1)
        var ctx2 = DeviceContext(1)
        ```
        """
        # const char *AsyncRT_DeviceContext_create(const DeviceContext **result, const char *api, int id)
        var result: _DeviceContextPtr[mut=True] = {}
        _checked(
            external_call[
                "AsyncRT_DeviceContext_create",
                _CString[],
                UnsafePointer[_DeviceContextPtr[mut=True], origin_of(result)],
                UnsafePointer[c_char, ImmutAnyOrigin],
                Int32,
            ](
                UnsafePointer(to=result),
                api.as_c_string_slice().unsafe_ptr(),
                Int32(device_id),
            )
        )
        self._handle = result
        self._owning = True

    def _retain(self):
        # Increment the reference count.
        #
        # void AsyncRT_DeviceContext_retain(const DeviceContext *ctx)
        external_call[
            "AsyncRT_DeviceContext_retain",
            NoneType,
            _DeviceContextPtr[mut=True],
        ](self._handle)

    @doc_hidden
    def __init__(out self, ctx_ptr: _DeviceContextPtr[mut=True]):
        """Create a Mojo DeviceContext from a pointer to an existing C++ object.
        """
        self._handle = ctx_ptr
        self._owning = False

    def __init__(out self, *, copy: Self):
        """Creates a copy of an existing device context by incrementing its reference count.

        This copy constructor creates a new reference to the same underlying device context
        by incrementing the reference count of the native context object. Both the original
        and the copy will refer to the same device context.

        Args:
            copy: The device context to copy.
        """
        # Increment the reference count before copying the handle.
        if copy._owning:
            copy._retain()
        self._handle = copy._handle
        self._owning = copy._owning

    def __del__(deinit self):
        """Releases resources associated with this device context.

        This destructor decrements the reference count of the native device context.
        When the reference count reaches zero, the underlying resources are released,
        including any cached memory buffers and compiled device functions.
        """
        if not self._owning:
            return
        # Decrement the reference count held by this struct.
        #
        # void AsyncRT_DeviceContext_release(const DeviceContext *ctx)
        external_call[
            "AsyncRT_DeviceContext_release",
            NoneType,
            _DeviceContextPtr[mut=True],
        ](self._handle)

    def __enter__(var self) -> Self:
        """Enables the use of DeviceContext in a 'with' statement context manager.

        This method allows DeviceContext to be used with Python-style context managers,
        which ensures proper resource management and cleanup when the context exits.

        Returns:
            The DeviceContext instance to be used within the context manager block.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext

        # Using DeviceContext as a context manager
        with DeviceContext() as ctx:
            # Perform GPU operations
            # Resources are automatically released when exiting the block
            pass
        ```
        """
        return self^

    def name(self) -> String:
        """Returns the device name, an ASCII string identifying this device,
        defined by the native device API.

        This method queries the underlying GPU device for its name, which typically
        includes the model and other identifying information. This can be useful for
        logging, debugging, or making runtime decisions based on the specific GPU hardware.

        Returns:
            A string containing the device name.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext

        var ctx = DeviceContext()
        print("Running on device:", ctx.name())
        ```
        """
        # const char *AsyncRT_DeviceContext_deviceName(const DeviceContext *ctx)
        var name_ptr = external_call[
            "AsyncRT_DeviceContext_deviceName",
            _CString[],
            _DeviceContextPtr[mut=True],
        ](
            self._handle,
        )
        return _string_from_owned_charptr(name_ptr)

    def api(self) -> String:
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
        from std.gpu.host import DeviceContext

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
        var api_ptr = StaticString()
        external_call["AsyncRT_DeviceContext_deviceApi", NoneType](
            UnsafePointer(to=api_ptr),
            self._handle,
        )
        return String(api_ptr)

    def _get_max_dynamic_shared_memory_bytes(
        self, requested_bytes: Int
    ) -> UInt32:
        """Gets the maximum dynamic shared memory bytes for this device.

        For NVIDIA GPUs, dynamic shared memory defaults to 48KB max. For larger
        allocations, we set MAX_DYNAMIC_SHARED_SIZE_BYTES to the minimum of:
        - The device's maximum opt-in shared memory per block
        - The requested size rounded up to nearest 1KB boundary

        For smaller allocations (<= 48KB), we return 0 to skip setting the
        attribute (avoiding unnecessary API calls and potential errors).

        For AMD GPUs, the MAX_SHARED_MEMORY_PER_BLOCK_OPTIN attribute doesn't
        exist, so we return 0 (no automatic inference) and rely on explicit
        func_attribute settings when needed.

        Args:
            requested_bytes: The amount of shared memory requested by the kernel.

        Returns:
            Maximum dynamic shared memory bytes to set, or 0 if not needed.
        """
        # NVIDIA GPUs have a 48KB default limit for dynamic shared memory
        comptime NVIDIA_DEFAULT_DYNAMIC_SHARED_LIMIT = 48 * 1024

        # Only set the attribute if we need more than the default limit
        if requested_bytes <= NVIDIA_DEFAULT_DYNAMIC_SHARED_LIMIT:
            return 0

        # Try to query the maximum opt-in shared memory limit from the device.
        # This attribute is NVIDIA-specific (via cudaFuncSetAttribute) and may
        # not be available on AMD GPUs or other vendors.
        try:
            var capacity = self.get_attribute(
                DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
            )

            # Sanity check: capacity should be reasonable (at least 48KB)
            if capacity < NVIDIA_DEFAULT_DYNAMIC_SHARED_LIMIT:
                # If the opt-in capacity is less than the default, something is wrong.
                # Fall back to not setting the attribute.
                return 0

            # Round requested_bytes up to nearest 1KB and use the minimum of
            # that and the device capacity minus 1KB system reservation
            var rounded_request = ((requested_bytes + 1023) // 1024) * 1024
            return UInt32(min(rounded_request, capacity - 1024))
        except:
            # Attribute not available (e.g., on AMD GPUs). Return 0 to skip
            # automatic inference. Code that needs >48KB on AMD should explicitly
            # set func_attribute.
            return 0

    def enqueue_create_buffer[
        dtype: DType
    ](self, size: Int) raises -> DeviceBuffer[dtype]:
        """Enqueues a buffer creation using the `DeviceBuffer` constructor.

        For GPU devices, the space is allocated in the device's global memory.

        Parameters:
            dtype: The data type to be stored in the allocated memory.

        Args:
            size: The number of elements of `type` to allocate memory for.

        Returns:
            The allocated buffer.

        Raises:
            If the operation fails.
        """
        return DeviceBuffer[dtype](self, size, _DeviceBufferMode._ASYNC)

    def create_buffer_sync[
        dtype: DType
    ](self, size: Int) raises -> DeviceBuffer[dtype]:
        """Creates a buffer synchronously using the `DeviceBuffer` constructor.

        Parameters:
            dtype: The data type to be stored in the allocated memory.

        Args:
            size: The number of elements of `type` to allocate memory for.

        Returns:
            The allocated buffer.

        Raises:
            If the operation fails.
        """
        var result = DeviceBuffer[dtype](self, size, _DeviceBufferMode._ASYNC)
        self.synchronize()
        return result

    def enqueue_create_host_buffer[
        dtype: DType
    ](self, size: Int) raises -> HostBuffer[dtype]:
        """Enqueues the creation of a HostBuffer.

        This function allocates memory on the host that is accessible by the device.
        The memory is page-locked (pinned) for efficient data transfer between host and device.

        Pinned memory is guaranteed to remain resident in the host's RAM, not be
        paged/swapped out to disk. Memory allocated normally (for example, using
        [`alloc()`](/docs/std/memory/unsafe_pointer/alloc/))
        is pageable—individual pages of memory can be moved to secondary storage
        (disk/SSD) when main memory fills up.

        Using pinned memory allows devices to make fast transfers
        between host memory and device memory, because they can use direct
        memory access (DMA) to transfer data without relying on the CPU.

        Allocating too much pinned memory can cause performance issues, since it
        reduces the amount of memory available for other processes.

        Parameters:
            dtype: The data type to be stored in the allocated memory.

        Args:
            size: The number of elements of `type` to allocate memory for.

        Returns:
            A `HostBuffer` object that wraps the allocated host memory.

        Raises:
            If memory allocation fails or if the device context is invalid.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext

        with DeviceContext() as ctx:
            # Allocate host memory accessible by the device
            var host_buffer = ctx.enqueue_create_host_buffer[DType.float32](1024)

            # Use the host buffer for device operations
            # ...
        ```
        """
        return HostBuffer[dtype](self, size)

    @always_inline
    def compile_function_unchecked[
        func_type: TrivialRegisterPassable,
        //,
        func: func_type,
        *,
        compile_options: StaticString = CompilationTarget[
            Self.default_device_info.target()
        ].default_compile_options(),
        link_options: StaticString = "",
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
    ](
        self,
        *,
        func_attribute: OptionalReg[FuncAttribute] = None,
        out result: DeviceFunction[
            func,
            None,
            target=Self.default_device_info.target(),
            compile_options=compile_options,
            link_options=link_options,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ],
    ) raises:
        """Compiles the provided function for execution on this device.

        Parameters:
            func_type: Type of the function.
            func: The function to compile.
            compile_options: Change the compile options to different options
                than the ones associated with this `DeviceContext`.
            link_options: Additional linker flags and options as a string.
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
            func_attribute: An attribute to use when compiling the code (such
                as maximum shared memory size).

        Returns:
            The compiled function via the `result` output parameter.

        Raises:
            If the operation fails.

        Notes:

        - This method doesn't perform compile-time type-checking of the kernel
          function arguments. You will encounter run-time errors if the values
          you pass don't conform to the expected argument types.
        - This method will be deprecated and eventually removed.
          Use `compile_function()` instead for type-checked kernel compilation.
        """
        assert (
            not func_attribute
            or func_attribute.value().attribute
            != Attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES
            or func_attribute.value().value
            <= Int32(self.default_device_info.shared_memory_per_multiprocessor)
        ), "Requested more than available shared memory."
        comptime result_type = type_of(result)
        result = result_type(
            self,
            func_attribute=func_attribute,
        )

        result.dump_rep[
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            _dump_sass=_dump_sass,
        ]()

    @deprecated("Pass the kernel only once: `compile_function[func]`.")
    @always_inline
    def compile_function[
        func_type: TrivialRegisterPassable,
        declared_arg_types: TypeList[Trait=AnyType, ...],
        //,
        func: func_type,
        signature_func: def(* args: * declared_arg_types) thin -> None,
        *,
        compile_options: StaticString = CompilationTarget[
            Self.default_device_info.target()
        ].default_compile_options(),
        link_options: StaticString = "",
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
    ](
        self,
        *,
        func_attribute: OptionalReg[FuncAttribute] = None,
        out result: DeviceFunction[
            func,
            declared_arg_types.values,
            compile_options=compile_options,
            link_options=link_options,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ],
    ) raises:
        """Compiles the provided function for execution on this device.

        Parameters:
            func_type: Type of the function.
            declared_arg_types: Types of the arguments to pass to the device function.
            func: The function to compile.
            signature_func: The function to compile, passed in again. Used for
                checking argument dtypes later.
                Note: This will disappear in future versions.
            compile_options: Change the compile options to different options
                than the ones associated with this `DeviceContext`.
            link_options: Additional linker flags and options as a string.
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
            func_attribute: An attribute to use when compiling the code (such
                as maximum shared memory size).

        Returns:
            The compiled function via the `result` output parameter.

        Raises:
            If the operation fails.
        """
        assert (
            not func_attribute
            or func_attribute.value().attribute
            != Attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES
            or func_attribute.value().value
            <= Int32(self.default_device_info.shared_memory_per_multiprocessor)
        ), "Requested more than available shared memory."
        comptime result_type = type_of(result)
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
    def compile_function[
        declared_arg_types: TypeList[Trait=AnyType, ...],
        //,
        func: def(* args: * declared_arg_types) thin -> None,
        *,
        compile_options: StaticString = CompilationTarget[
            Self.default_device_info.target()
        ].default_compile_options(),
        link_options: StaticString = "",
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
    ](
        self,
        *,
        func_attribute: OptionalReg[FuncAttribute] = None,
        out result: DeviceFunction[
            func,
            declared_arg_types.values,
            target=Self.default_device_info.target(),
            compile_options=compile_options,
            link_options=link_options,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ],
    ) raises:
        """Compiles the provided function for execution on this device.

        Parameters:
            declared_arg_types: Types of the arguments to pass to the device function.
            func: The function to compile.
            compile_options: Change the compile options to different options
                than the ones associated with this `DeviceContext`.
            link_options: Additional linker flags and options as a string.
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
            func_attribute: An attribute to use when compiling the code (such
                as maximum shared memory size).

        Returns:
            The compiled function via the `result` output parameter.

        Raises:
            If the operation fails.
        """
        assert (
            not func_attribute
            or func_attribute.value().attribute
            != Attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES
            or func_attribute.value().value
            <= Int32(self.default_device_info.shared_memory_per_multiprocessor)
        ), "Requested more than available shared memory."
        comptime result_type = type_of(result)
        result = result_type(
            self,
            func_attribute=func_attribute,
        )

        result.dump_rep[
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            _dump_sass=_dump_sass,
        ]()

    @deprecated("Use `compile_function[func]` instead.")
    @always_inline
    def compile_function_experimental[
        declared_arg_types: TypeList[Trait=AnyType, ...],
        //,
        func: def(* args: * declared_arg_types) thin -> None,
        *,
        compile_options: StaticString = CompilationTarget[
            Self.default_device_info.target()
        ].default_compile_options(),
        link_options: StaticString = "",
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
    ](
        self,
        *,
        func_attribute: OptionalReg[FuncAttribute] = None,
        out result: DeviceFunction[
            func,
            declared_arg_types.values,
            target=Self.default_device_info.target(),
            compile_options=compile_options,
            link_options=link_options,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ],
    ) raises:
        """Deprecated alias for `compile_function[func]`.

        Parameters:
            declared_arg_types: Types of the arguments to pass to the device function.
            func: The function to compile.
            compile_options: Compile options.
            link_options: Additional linker flags and options as a string.
            dump_asm: To dump the compiled assembly.
            dump_llvm: To dump the generated LLVM code.
            _dump_sass: NVIDIA SASS dump.
            _ptxas_info_verbose: Verbose PTX assembly.

        Args:
            func_attribute: An attribute to use when compiling the code.

        Returns:
            The compiled function via the `result` output parameter.

        Raises:
            If the operation fails.
        """
        result = self.compile_function[
            func,
            compile_options=compile_options,
            link_options=link_options,
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            _dump_sass=_dump_sass,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ](func_attribute=func_attribute)

    @deprecated("Pass the kernel only once: `compile_function[func]`.")
    @always_inline
    def compile_function[
        func_type: TrivialRegisterPassable,
        declared_arg_types: TypeList[Trait=AnyType, ...],
        //,
        func: func_type,
        signature_func: def(* args: * declared_arg_types) capturing -> None,
        *,
        compile_options: StaticString = CompilationTarget[
            Self.default_device_info.target()
        ].default_compile_options(),
        link_options: StaticString = "",
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
    ](
        self,
        *,
        func_attribute: OptionalReg[FuncAttribute] = None,
        out result: DeviceFunction[
            func,
            declared_arg_types.values,
            target=Self.default_device_info.target(),
            compile_options=compile_options,
            link_options=link_options,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ],
    ) raises:
        """Compiles the provided function for execution on this device.

        Parameters:
            func_type: Type of the function.
            declared_arg_types: Types of the arguments to pass to the device function.
            func: The function to compile.
            signature_func: The function to compile, passed in again. Used for
                checking argument dtypes later.
                Note: This will disappear in future versions.
            compile_options: Change the compile options to different options
                than the ones associated with this `DeviceContext`.
            link_options: Additional linker flags and options as a string.
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
            func_attribute: An attribute to use when compiling the code (such
                as maximum shared memory size).

        Returns:
            The compiled function via the `result` output parameter.

        Raises:
            If the operation fails.
        """
        assert (
            not func_attribute
            or func_attribute.value().attribute
            != Attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES
            or func_attribute.value().value
            <= Int32(self.default_device_info.shared_memory_per_multiprocessor)
        ), "Requested more than available shared memory."
        comptime result_type = type_of(result)
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
    def compile_function[
        declared_arg_types: TypeList[Trait=AnyType, ...],
        //,
        func: def(* args: * declared_arg_types) capturing -> None,
        *,
        compile_options: StaticString = CompilationTarget[
            Self.default_device_info.target()
        ].default_compile_options(),
        link_options: StaticString = "",
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
    ](
        self,
        *,
        func_attribute: OptionalReg[FuncAttribute] = None,
        out result: DeviceFunction[
            func,
            declared_arg_types.values,
            target=Self.default_device_info.target(),
            compile_options=compile_options,
            link_options=link_options,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ],
    ) raises:
        """Compiles the provided function for execution on this device.

        Parameters:
            declared_arg_types: Types of the arguments to pass to the device function.
            func: The function to compile.
            compile_options: Change the compile options to different options
                than the ones associated with this `DeviceContext`.
            link_options: Additional linker flags and options as a string.
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
            func_attribute: An attribute to use when compiling the code (such
                as maximum shared memory size).

        Returns:
            The compiled function via the `result` output parameter.

        Raises:
            If the operation fails.
        """
        assert (
            not func_attribute
            or func_attribute.value().attribute
            != Attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES
            or func_attribute.value().value
            <= Int32(self.default_device_info.shared_memory_per_multiprocessor)
        ), "Requested more than available shared memory."
        comptime result_type = type_of(result)
        result = result_type(
            self,
            func_attribute=func_attribute,
        )

        result.dump_rep[
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            _dump_sass=_dump_sass,
        ]()

    @deprecated("Use `compile_function[func]` instead.")
    @always_inline
    def compile_function_experimental[
        declared_arg_types: TypeList[Trait=AnyType, ...],
        //,
        func: def(* args: * declared_arg_types) capturing -> None,
        *,
        compile_options: StaticString = CompilationTarget[
            Self.default_device_info.target()
        ].default_compile_options(),
        link_options: StaticString = "",
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
    ](
        self,
        *,
        func_attribute: OptionalReg[FuncAttribute] = None,
        out result: DeviceFunction[
            func,
            declared_arg_types.values,
            target=Self.default_device_info.target(),
            compile_options=compile_options,
            link_options=link_options,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ],
    ) raises:
        """Deprecated alias for `compile_function[func]` (capturing).

        Parameters:
            declared_arg_types: Types of the arguments to pass to the device function.
            func: The function to compile.
            compile_options: Compile options.
            link_options: Additional linker flags and options as a string.
            dump_asm: To dump the compiled assembly.
            dump_llvm: To dump the generated LLVM code.
            _dump_sass: NVIDIA SASS dump.
            _ptxas_info_verbose: Verbose PTX assembly.

        Args:
            func_attribute: An attribute to use when compiling the code.

        Returns:
            The compiled function via the `result` output parameter.

        Raises:
            If the operation fails.
        """
        result = self.compile_function[
            func,
            compile_options=compile_options,
            link_options=link_options,
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            _dump_sass=_dump_sass,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ](func_attribute=func_attribute)

    def load_function[
        func_type: TrivialRegisterPassable,
        //,
        func: func_type,
    ](
        self,
        *,
        var function_name: String,
        var asm: String,
        func_attribute: OptionalReg[FuncAttribute] = None,
        out result: DeviceExternalFunction,
    ) raises:
        """Loads a pre-compiled device function from assembly code.

        This method loads an external GPU function from provided assembly code (PTX/SASS)
        rather than compiling it from Mojo source. This is useful for integrating with
        existing CUDA/HIP code or for using specialized assembly optimizations.

        Parameters:
            func_type: The dtype of the function to load.
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

        ```text
        from std.gpu.host import DeviceContext
        from std.gpu.host.device_context import DeviceExternalFunction

        def func_signature(
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
        comptime result_type = type_of(result)
        result = result_type(
            self,
            function_name=function_name^,
            asm=asm^,
            func_attribute=func_attribute,
        )

    @parameter
    @always_inline
    def enqueue_function_unchecked[
        func_type: TrivialRegisterPassable,
        //,
        func: func_type,
        *Ts: AnyType,
        link_options: StaticString = "",
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
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        func_attribute: OptionalReg[FuncAttribute] = None,
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        """Compiles and enqueues a kernel for execution on this device.

        Parameters:
            func_type: The dtype of the function to launch.
            func: The function to launch.
            Ts: The dtypes of the arguments being passed to the function.
            link_options: Additional linker flags and options as a string.
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
            location: Source location for the function call.

        You can pass the function directly to `enqueue_function_unchecked`
        without compiling it first:

        ```mojo
        from std.gpu.host import DeviceContext

        def kernel():
            print("hello from the GPU")

        with DeviceContext() as ctx:
            ctx.enqueue_function_unchecked[kernel](grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        If you are reusing the same function and parameters multiple times, this
        incurs 50-500 nanoseconds of overhead per enqueue, so you can compile it
        first to remove the overhead:

        ```mojo
        from std.gpu.host import DeviceContext

        def kernel():
            print("hello from the GPU")

        with DeviceContext() as ctx:
            var compiled_func = ctx.compile_function_unchecked[kernel]()
            ctx.enqueue_function_unchecked(compiled_func, grid_dim=1, block_dim=1)
            ctx.enqueue_function_unchecked(compiled_func, grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        Raises:
            If the operation fails.

        Notes:

        - This method doesn't perform compile-time type-checking of the kernel
          function arguments. You will encounter run-time errors if the values
          you pass don't conform to the expected argument types.
        - This method will be deprecated and eventually removed.
          Use `enqueue_function()` instead for type-checked kernel execution.
        """
        _check_dim["DeviceContext.enqueue_function_unchecked", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceContext.enqueue_function_unchecked", "block_dim"](
            block_dim, location=call_location()
        )

        var gpu_kernel = self.compile_function_unchecked[
            func,
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            link_options=link_options,
            _dump_sass=_dump_sass,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ](func_attribute=func_attribute)

        gpu_kernel._call_with_pack(
            self,
            *args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=location.or_else(call_location()),
        )

    @parameter
    @always_inline
    def enqueue_function_unchecked[
        *Ts: AnyType
    ](
        self,
        f: DeviceFunction,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        """Enqueues a compiled function for execution on this device.

        Parameters:
            Ts: Argument dtypes.

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
            location: Source location for the function call.

        You can pass the function directly to `enqueue_function_unchecked`
        without compiling it first:

        ```mojo
        from std.gpu.host import DeviceContext

        def kernel():
            print("hello from the GPU")

        with DeviceContext() as ctx:
            ctx.enqueue_function_unchecked[kernel](grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        If you are reusing the same function and parameters multiple times, this
        incurs 50-500 nanoseconds of overhead per enqueue, so you can compile
        the function first to remove the overhead:

        ```mojo
        from std.gpu.host import DeviceContext

        def kernel():
            print("hello from the GPU")

        with DeviceContext() as ctx:
            var compiled_func = ctx.compile_function_unchecked[kernel]()
            ctx.enqueue_function_unchecked(compiled_func, grid_dim=1, block_dim=1)
            ctx.enqueue_function_unchecked(compiled_func, grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        Raises:
            If the operation fails.

        Notes:

        - This method doesn't perform compile-time type-checking of the kernel
          function arguments. You will encounter run-time errors if the values
          you pass don't conform to the expected argument types.
        - This method will be deprecated and eventually removed.
          Use `enqueue_function()` instead for type-checked kernel execution.
        """
        _check_dim["DeviceContext.enqueue_function_unchecked", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceContext.enqueue_function_unchecked", "block_dim"](
            block_dim, location=call_location()
        )

        comptime assert (
            not f.declared_arg_types
        ), "A checked DeviceFunction should be called with `enqueue_function`."
        f._call_with_pack(
            self,
            *args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=location.or_else(call_location()),
        )

    @parameter
    @always_inline
    def enqueue_function[
        *Ts: DevicePassable
    ](
        self,
        f: DeviceFunction,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        """Enqueues a pre-compiled checked function for execution on this device.

        This overload requires a `DeviceFunction` that was compiled with
        type checking enabled (via `compile_function`). The function
        will verify that the argument types match the declared types at
        compile time.

        Parameters:
            Ts: Argument dtypes.

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
            location: Source location for the function call.

        ```mojo
        from std.gpu.host import DeviceContext

        def kernel(x: Int):
            print("Value:", x)

        with DeviceContext() as ctx:
            var compiled_func = ctx.compile_function[kernel, kernel]()
            ctx.enqueue_function(compiled_func, 42, grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        Raises:
            If the operation fails.
        """
        _check_dim["DeviceContext.enqueue_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceContext.enqueue_function", "block_dim"](
            block_dim, location=call_location()
        )

        comptime assert Bool(
            f.declared_arg_types
        ), "Calling a non-checked function."
        f._call_with_pack_checked(
            self,
            *args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=location.or_else(call_location()),
        )

    @always_inline
    def enqueue_function[
        *Ts: AnyType
    ](
        self,
        f: DeviceExternalFunction,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        """Enqueues an external device function for execution on this device.

        This overload accepts a `DeviceExternalFunction` that was loaded from
        assembly code (PTX/SASS). External functions are pre-compiled GPU kernels
        that can be integrated with Mojo code.

        Parameters:
            Ts: Argument types to pass to the external function.

        Args:
            f: The external device function to execute.
            args: Arguments to pass to the function.
            grid_dim: Dimensions of the compute grid, made up of thread blocks.
            block_dim: Dimensions of each thread block in the grid.
            cluster_dim: Dimensions of clusters (if the thread blocks are
                grouped into clusters).
            shared_mem_bytes: Amount of shared memory per thread block.
            attributes: Launch attributes.
            constant_memory: Constant memory mapping.
            location: Source location for the function call.

        Example:

        ```text
        from std.gpu.host import DeviceContext

        def vec_add_sig(
            in0: UnsafePointer[Float32],
            in1: UnsafePointer[Float32],
            out: UnsafePointer[Float32],
            len: Int,
        ):
            pass

        with DeviceContext() as ctx:
            var func = ctx.load_function[vec_add_sig](
                function_name="vectorAdd",
                asm=ptx_code,
            )
            ctx.enqueue_function(
                func,
                in0_buf,
                in1_buf,
                out_buf,
                1024,
                grid_dim=Dim(32),
                block_dim=Dim(32),
            )
            ctx.synchronize()
        ```

        Raises:
            If the operation fails.
        """
        _check_dim["DeviceContext.enqueue_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceContext.enqueue_function", "block_dim"](
            block_dim, location=call_location()
        )

        f._call_with_pack(
            self,
            *args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=location.or_else(call_location()),
        )

    @deprecated("Pass the kernel only once: `enqueue_function[func]`.")
    @parameter
    @always_inline
    def enqueue_function[
        func_type: TrivialRegisterPassable,
        declared_arg_types: TypeList[Trait=AnyType, ...],
        //,
        func: func_type,
        signature_func: def(* args: * declared_arg_types) thin -> None,
        *actual_arg_types: DevicePassable,
        link_options: StaticString = "",
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
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        func_attribute: OptionalReg[FuncAttribute] = None,
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        """Compiles and enqueues a kernel for execution on this device with type checking.

        This function performs compile-time type checking on the kernel arguments,
        ensuring that the types passed match the declared signature. Both `func` and
        `signature_func` should typically be the same kernel function (this redundancy
        is required for type checking and will be removed in future versions).

        Parameters:
            func_type: The type of the function to launch (usually inferred).
            declared_arg_types: The declared argument types from the function
                signature (usually inferred).
            func: The kernel function to compile and launch.
            signature_func: The kernel function, passed again for type checking.
                Typically the same as `func`.
            actual_arg_types: The types of the arguments being passed (usually inferred).
            link_options: Additional linker flags and options as a string.
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
            args: Variadic arguments which are passed to the kernel function.
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            cluster_dim: The cluster dimensions.
            shared_mem_bytes: Per-block memory shared between blocks.
            attributes: A `List` of launch attributes.
            constant_memory: A `List` of constant memory mappings.
            func_attribute: `CUfunction_attribute` enum.
            location: Source location for the function call.

        Most parameters are inferred automatically. In typical usage, you only
        need to pass the kernel function twice (as both `func` and `signature_func`):

        ```text
        from std.gpu.host import DeviceContext
        from layout import Layout, LayoutTensor

        def vector_add(
            a: LayoutTensor[DType.float32, Layout.row_major(1000), MutAnyOrigin],
            b: LayoutTensor[DType.float32, Layout.row_major(1000), MutAnyOrigin],
            c: LayoutTensor[DType.float32, Layout.row_major(1000), MutAnyOrigin],
        ):
            # ... kernel implementation ...
            pass

        with DeviceContext() as ctx:
            # Create tensors a, b, c...
            # Most parameters are inferred automatically:
            ctx.enqueue_function[vector_add, vector_add](
                a, b, c,
                grid_dim=4,
                block_dim=256
            )
            ctx.synchronize()
        ```

        Raises:
            If the operation fails.
        """
        _check_dim["DeviceContext.enqueue_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceContext.enqueue_function", "block_dim"](
            block_dim, location=call_location()
        )

        # If shared_mem_bytes is specified but func_attribute is not,
        # automatically set MAX_DYNAMIC_SHARED_SIZE_BYTES if needed (>48KB)
        var inferred_func_attribute = func_attribute
        if not func_attribute and shared_mem_bytes:
            var max_shared = self._get_max_dynamic_shared_memory_bytes(
                shared_mem_bytes.value()
            )
            if max_shared > 0:
                inferred_func_attribute = (
                    FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(max_shared)
                )

        var gpu_kernel = self.compile_function[
            signature_func,
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            link_options=link_options,
            _dump_sass=_dump_sass,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ](func_attribute=inferred_func_attribute)

        gpu_kernel._call_with_pack_checked(
            self,
            *args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=location.or_else(call_location()),
        )

    @parameter
    @always_inline
    def enqueue_function[
        declared_arg_types: TypeList[Trait=AnyType, ...],
        //,
        func: def(* args: * declared_arg_types) thin -> None,
        *actual_arg_types: DevicePassable,
        link_options: StaticString = "",
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
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        func_attribute: OptionalReg[FuncAttribute] = None,
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        """Compiles and enqueues a kernel for execution on this device.

        Parameters:
            declared_arg_types: Types of the arguments to pass to the device function.
            func: The function to compile and launch.
            actual_arg_types: The dtypes of the arguments being passed to the function.
            link_options: Additional linker flags and options as a string.
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
            location: Source location for the function call.

        You can pass the function directly to `enqueue_function`
        without compiling it first:

        ```mojo
        from std.gpu.host import DeviceContext

        def kernel():
            print("hello from the GPU")

        with DeviceContext() as ctx:
            ctx.enqueue_function[kernel](grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        If you are reusing the same function and parameters multiple times, this
        incurs 50-500 nanoseconds of overhead per enqueue, so you can compile it
        first to remove the overhead:

        ```mojo
        from std.gpu.host import DeviceContext

        def kernel():
            print("hello from the GPU")

        with DeviceContext() as ctx:
            var compiled_func = ctx.compile_function[kernel]()
            ctx.enqueue_function(compiled_func, grid_dim=1, block_dim=1)
            ctx.enqueue_function(compiled_func, grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        Raises:
            If the operation fails.
        """
        _check_dim["DeviceContext.enqueue_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceContext.enqueue_function", "block_dim"](
            block_dim, location=call_location()
        )

        # If shared_mem_bytes is specified but func_attribute is not,
        # automatically set MAX_DYNAMIC_SHARED_SIZE_BYTES if needed (>48KB)
        var inferred_func_attribute = func_attribute
        if not func_attribute and shared_mem_bytes:
            var max_shared = self._get_max_dynamic_shared_memory_bytes(
                shared_mem_bytes.value()
            )
            if max_shared > 0:
                inferred_func_attribute = (
                    FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(max_shared)
                )

        var gpu_kernel = self.compile_function[
            func,
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            link_options=link_options,
            _dump_sass=_dump_sass,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ](func_attribute=inferred_func_attribute)

        gpu_kernel._call_with_pack_checked(
            self,
            *args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=location.or_else(call_location()),
        )

    @deprecated("Use `enqueue_function[func]` instead.")
    @parameter
    @always_inline
    def enqueue_function_experimental[
        declared_arg_types: TypeList[Trait=AnyType, ...],
        //,
        func: def(* args: * declared_arg_types) thin -> None,
        *actual_arg_types: DevicePassable,
        link_options: StaticString = "",
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
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        func_attribute: OptionalReg[FuncAttribute] = None,
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        """Deprecated alias for `enqueue_function[func]`.

        Parameters:
            declared_arg_types: Types of the arguments to pass to the device function.
            func: The function to compile and launch.
            actual_arg_types: The dtypes of the arguments being passed.
            link_options: Additional linker flags and options as a string.
            dump_asm: To dump the compiled assembly.
            dump_llvm: To dump the generated LLVM code.
            _dump_sass: NVIDIA SASS dump.
            _ptxas_info_verbose: Verbose PTX assembly.

        Args:
            args: Variadic arguments which are passed to the `func`.
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            cluster_dim: The cluster dimensions.
            shared_mem_bytes: Per-block memory shared between blocks.
            attributes: A `List` of launch attributes.
            constant_memory: A `List` of constant memory mappings.
            func_attribute: `CUfunction_attribute` enum.
            location: Source location for the function call.

        Raises:
            If the operation fails.
        """
        self.enqueue_function[
            func,
            link_options=link_options,
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            _dump_sass=_dump_sass,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ](
            *args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            func_attribute=func_attribute,
            location=location.or_else(call_location()),
        )

    @deprecated("Pass the kernel only once: `enqueue_function[func]`.")
    @parameter
    @always_inline
    def enqueue_function[
        func_type: TrivialRegisterPassable,
        declared_arg_types: TypeList[Trait=AnyType, ...],
        //,
        func: func_type,
        signature_func: def(* args: * declared_arg_types) capturing -> None,
        *actual_arg_types: DevicePassable,
        link_options: StaticString = "",
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
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        func_attribute: OptionalReg[FuncAttribute] = None,
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        """Compiles and enqueues a capturing kernel for execution on this device with type checking.

        This overload is for kernels that capture variables from their enclosing scope.
        The `capturing` annotation on the signature function indicates that the kernel
        can access variables from the surrounding context. Like the non-capturing overload,
        both `func` and `signature_func` should typically be the same kernel function.

        Parameters:
            func_type: The type of the function to launch (usually inferred).
            declared_arg_types: The declared argument types from the function
                signature (usually inferred).
            func: The capturing kernel function to compile and launch.
            signature_func: The kernel function, passed again for type checking.
                Typically the same as `func`.
            actual_arg_types: The types of the arguments being passed (usually inferred).
            link_options: Additional linker flags and options as a string.
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
            args: Variadic arguments which are passed to the kernel function.
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            cluster_dim: The cluster dimensions.
            shared_mem_bytes: Per-block memory shared between blocks.
            attributes: A `List` of launch attributes.
            constant_memory: A `List` of constant memory mappings.
            func_attribute: `CUfunction_attribute` enum.
            location: Source location for the function call.

        Most parameters are inferred automatically. This overload is selected when
        your kernel captures variables from its surrounding scope:

        ```text
        from std.gpu.host import DeviceContext
        from layout import Layout, LayoutTensor

        def main():
            with DeviceContext() as ctx:
                var scale_factor = 2.0

                # This kernel captures 'scale_factor' from the enclosing scope
                def scale_kernel(data: LayoutTensor[DType.float32, Layout.row_major(100), MutAnyOrigin]):
                    # Uses captured scale_factor variable
                    pass

                # Create tensor 'data'...
                # Most parameters are inferred:
                ctx.enqueue_function[scale_kernel, scale_kernel](
                    data,
                    grid_dim=1,
                    block_dim=256
                )
                ctx.synchronize()
        ```

        Raises:
            If the operation fails.
        """
        _check_dim["DeviceContext.enqueue_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceContext.enqueue_function", "block_dim"](
            block_dim, location=call_location()
        )

        # If shared_mem_bytes is specified but func_attribute is not,
        # automatically set MAX_DYNAMIC_SHARED_SIZE_BYTES if needed (>48KB)
        var inferred_func_attribute = func_attribute
        if not func_attribute and shared_mem_bytes:
            var max_shared = self._get_max_dynamic_shared_memory_bytes(
                shared_mem_bytes.value()
            )
            if max_shared > 0:
                inferred_func_attribute = (
                    FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(max_shared)
                )

        var gpu_kernel = self.compile_function[
            signature_func,
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            link_options=link_options,
            _dump_sass=_dump_sass,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ](func_attribute=inferred_func_attribute)

        gpu_kernel._call_with_pack_checked(
            self,
            *args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=location.or_else(call_location()),
        )

    @always_inline
    def enqueue_function[
        FuncType: def() register_passable -> None,
        //,
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
    ](
        self,
        func: FuncType,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        func_attribute: OptionalReg[FuncAttribute] = None,
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        """Compiles and enqueues a capturing kernel for execution on this device with type checking.

        This overload is for kernels that capture variables from their enclosing scope.
        The `capturing` annotation on the signature function indicates that the kernel
        can access variables from the surrounding context. Like the non-capturing overload,
        both `func` and `signature_func` should typically be the same kernel function.

        Parameters:
            FuncType: The type of the function to launch (usually inferred).
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
            func: The capturing kernel function to compile and launch.
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            cluster_dim: The cluster dimensions.
            shared_mem_bytes: Per-block memory shared between blocks.
            attributes: A `List` of launch attributes.
            constant_memory: A `List` of constant memory mappings.
            func_attribute: `CUfunction_attribute` enum.
            location: Source location for the function call.

        Most parameters are inferred automatically. This overload is selected when
        your kernel captures variables from its surrounding scope:

        ```text
        from std.gpu import DeviceContext, global_idx
        from layout import TileTensor, row_major


        def main() raises:
            with DeviceContext() as ctx:
                var scale_factor: Float32 = 2.0

                var data_buffer = ctx.enqueue_create_buffer[DType.float32](100)
                var data = TileTensor(data_buffer, row_major[100]())
                with data_buffer.map_to_host() as h:
                    for i in range(data.num_elements()):
                        h[i] = Float32(i)

                # This kernel captures 'scale_factor' from the enclosing scope
                def scale_kernel() {var}:
                    var i = global_idx.x
                    if i >= UInt(100):
                        return
                    data[i] = data[i] * scale_factor

                ctx.enqueue_function(scale_kernel, grid_dim=1, block_dim=256)
                ctx.synchronize()
                with data_buffer.map_to_host() as h:
                    for i in range(data.num_elements()):
                        print(h[i])
        ```

        Raises:
            If the operation fails.
        """
        _check_dim["DeviceContext.enqueue_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceContext.enqueue_function", "block_dim"](
            block_dim, location=call_location()
        )

        self.enqueue_function_unchecked[
            FuncType.__call__,
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            _dump_sass=_dump_sass,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ](
            func,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=location.or_else(call_location()),
        )

    @parameter
    @always_inline
    def enqueue_function[
        declared_arg_types: TypeList[Trait=AnyType, ...],
        //,
        func: def(* args: * declared_arg_types) capturing -> None,
        *actual_arg_types: DevicePassable,
        link_options: StaticString = "",
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
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        func_attribute: OptionalReg[FuncAttribute] = None,
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        """Compiles and enqueues a kernel for execution on this device. This
        overload takes in a function that's `capturing`.

        Parameters:
            declared_arg_types: Types of the arguments to pass to the device function.
            func: The function to compile and launch.
            actual_arg_types: The dtypes of the arguments being passed to the function.
            link_options: Additional linker flags and options as a string.
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
            location: Source location for the function call.

        You can pass the function directly to `enqueue_function`
        without compiling it first:

        ```mojo
        from std.gpu.host import DeviceContext

        def kernel():
            print("hello from the GPU")

        with DeviceContext() as ctx:
            ctx.enqueue_function[kernel](grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        If you are reusing the same function and parameters multiple times, this
        incurs 50-500 nanoseconds of overhead per enqueue, so you can compile it
        first to remove the overhead:

        ```mojo
        from std.gpu.host import DeviceContext

        def kernel():
            print("hello from the GPU")

        with DeviceContext() as ctx:
            var compiled_func = ctx.compile_function[kernel]()
            ctx.enqueue_function(compiled_func, grid_dim=1, block_dim=1)
            ctx.enqueue_function(compiled_func, grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        Raises:
            If the operation fails.
        """
        _check_dim["DeviceContext.enqueue_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceContext.enqueue_function", "block_dim"](
            block_dim, location=call_location()
        )

        # If shared_mem_bytes is specified but func_attribute is not,
        # automatically set MAX_DYNAMIC_SHARED_SIZE_BYTES if needed (>48KB)
        var inferred_func_attribute = func_attribute
        if not func_attribute and shared_mem_bytes:
            var max_shared = self._get_max_dynamic_shared_memory_bytes(
                shared_mem_bytes.value()
            )
            if max_shared > 0:
                inferred_func_attribute = (
                    FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(max_shared)
                )

        var gpu_kernel = self.compile_function[
            func,
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            link_options=link_options,
            _dump_sass=_dump_sass,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ](func_attribute=inferred_func_attribute)

        gpu_kernel._call_with_pack_checked(
            self,
            *args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=location.or_else(call_location()),
        )

    @deprecated("Use `enqueue_function[func]` instead.")
    @parameter
    @always_inline
    def enqueue_function_experimental[
        declared_arg_types: TypeList[Trait=AnyType, ...],
        //,
        func: def(* args: * declared_arg_types) capturing -> None,
        *actual_arg_types: DevicePassable,
        link_options: StaticString = "",
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
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        func_attribute: OptionalReg[FuncAttribute] = None,
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        """Deprecated alias for `enqueue_function[func]` (capturing).

        Parameters:
            declared_arg_types: Types of the arguments to pass to the device function.
            func: The function to compile and launch.
            actual_arg_types: The dtypes of the arguments being passed.
            link_options: Additional linker flags and options as a string.
            dump_asm: To dump the compiled assembly.
            dump_llvm: To dump the generated LLVM code.
            _dump_sass: NVIDIA SASS dump.
            _ptxas_info_verbose: Verbose PTX assembly.

        Args:
            args: Variadic arguments which are passed to the `func`.
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            cluster_dim: The cluster dimensions.
            shared_mem_bytes: Per-block memory shared between blocks.
            attributes: A `List` of launch attributes.
            constant_memory: A `List` of constant memory mappings.
            func_attribute: `CUfunction_attribute` enum.
            location: Source location for the function call.

        Raises:
            If the operation fails.
        """
        self.enqueue_function[
            func,
            link_options=link_options,
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            _dump_sass=_dump_sass,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ](
            *args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            func_attribute=func_attribute,
            location=location.or_else(call_location()),
        )

    @deprecated("Use `enqueue_function(func, ...)` instead.")
    @parameter
    @always_inline
    def enqueue_function_experimental[
        func_type: TrivialRegisterPassable,
        //,
        func: func_type,
        *Ts: DevicePassable,
    ](
        self,
        f: DeviceFunction[func, ...],
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        location: OptionalReg[SourceLocation] = None,
    ) raises:
        """Enqueues a compiled function for execution on this device.

        Parameters:
            func_type: The type of the function type.
            func: The function to execute.
            Ts: Argument dtypes.

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
            location: Source location for the function call.

        You can pass the function directly to `enqueue_function` without
        compiling it first:

        ```mojo
        from std.gpu.host import DeviceContext

        def kernel():
            print("hello from the GPU")

        with DeviceContext() as ctx:
            ctx.enqueue_function_experimental[kernel](grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        If you are reusing the same function and parameters multiple times, this
        incurs 50-500 nanoseconds of overhead per enqueue, so you can compile
        the function first to remove the overhead:

        ```mojo
        from std.gpu.host import DeviceContext

        def kernel():
            print("hello from the GPU")

        with DeviceContext() as ctx:
            var compiled_func = ctx.compile_function_experimental[kernel]()
            ctx.enqueue_function_experimental(compiled_func, grid_dim=1, block_dim=1)
            ctx.enqueue_function_experimental(compiled_func, grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        Raises:
            If the operation fails.
        """
        _check_dim["DeviceContext.enqueue_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceContext.enqueue_function", "block_dim"](
            block_dim, location=call_location()
        )

        comptime assert Bool(
            f.declared_arg_types
        ), "Calling a non-checked function."
        f._call_with_pack_checked(
            self,
            *args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=location.or_else(call_location()),
        )

    @always_inline
    def enqueue_cpu_function[
        func: def() capturing -> None,
    ](self) raises:
        """Enqueues a function for execution on CPU.

        Parameters:
            func: The function to execute.

        Raises:
            If the operation fails.
            If self is not a CPU DeviceContext.
        """
        if self.api() != "cpu":
            raise Error(
                "enqueue_cpu_function is only supported on CPU DeviceContexts"
            )

        async def wrapper() capturing -> None:
            func()

        var coro = wrapper()
        coro._set_noop_callback()
        _checked(
            external_call[
                "AsyncRT_DeviceContext_enqueueHostFunction",
                _CString[],
            ](
                self._handle,
                _coro_resume_fn,
                _coro_destroy_fn,
                coro^._take_handle(),
            )
        )

    @always_inline
    def enqueue_cpu_function[
        FuncType: def() -> None,
    ](self, func: FuncType) raises:
        """Enqueues a function for execution on CPU.

        Parameters:
            FuncType: The function type.

        Args:
            func: The function to execute.

        Raises:
            If the operation fails.
            If self is not a CPU DeviceContext.
        """
        if self.api() != "cpu":
            raise Error(
                "enqueue_cpu_function is only supported on CPU DeviceContexts"
            )

        async def wrapper() capturing -> None:
            func()

        var coro = wrapper()
        coro._set_noop_callback()
        _checked(
            external_call[
                "AsyncRT_DeviceContext_enqueueHostFunction",
                _CString[],
            ](
                self._handle,
                _coro_resume_fn,
                _coro_destroy_fn,
                coro^._take_handle(),
            )
        )

    @always_inline
    def enqueue_cpu_range[
        func: def(count: Int) capturing -> None,
    ](self, count: Int,) raises:
        """Enqueues a function to be executed in parallel over a 1D range.

        The function is called as `func(i)` for each `i` in `range(count)`.

        Instances of the function are executed in parallel, but it is not
        guaranteed that all instances will execute simultaneously.

        Parameters:
            func: The function to execute.

        Args:
            count: The number of parallel instances of the function to enqueue.

        Raises:
            If the operation fails.
            If self is not a CPU DeviceContext.
        """
        if self.api() != "cpu":
            raise Error(
                "enqueue_cpu_range is only supported on CPU DeviceContexts"
            )

        var handles = alloc[AnyCoroutine](count)

        @always_inline
        @parameter
        async def wrapper(idx: Int) capturing -> None:
            func(idx)

        for j in range(count):
            var coro = wrapper(j)
            coro._set_noop_callback()
            handles[j] = coro^._take_handle()

        _checked(
            external_call[
                "AsyncRT_DeviceContext_enqueueHostFunctionRange",
                _CString[],
            ](
                self._handle,
                _coro_resume_fn,
                _coro_destroy_fn,
                handles,
                count,
            )
        )
        handles.free()

    @always_inline
    def enqueue_cpu_range[
        FuncType: def(Int) -> None,
    ](self, func: FuncType, count: Int,) raises:
        """Enqueues a function to be executed in parallel over a 1D range.

        The function is called as `func(i)` for each `i` in `range(count)`.

        Instances of the function are executed in parallel, but it is not
        guaranteed that all instances will execute simultaneously.

        Parameters:
            FuncType: The type of function to execute.

        Args:
            func: The function closure to execute.
            count: The number of parallel instances of the function to enqueue.

        Raises:
            If the operation fails.
            If self is not a CPU DeviceContext.
        """
        if self.api() != "cpu":
            raise Error(
                "enqueue_cpu_range is only supported on CPU DeviceContexts"
            )

        var handles = alloc[AnyCoroutine](count)

        async def wrapper(idx: Int) capturing -> None:
            func(idx)

        for j in range(count):
            var coro = wrapper(j)
            coro._set_noop_callback()
            handles[j] = coro^._take_handle()

        _checked(
            external_call[
                "AsyncRT_DeviceContext_enqueueHostFunctionRange",
                _CString[],
            ](
                self._handle,
                _coro_resume_fn,
                _coro_destroy_fn,
                handles,
                count,
            )
        )
        handles.free()

    @parameter
    @always_inline
    def execution_time[
        func: def(Self) raises capturing[_] -> None
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

        ```text
        from std.gpu.host import DeviceContext

        def gpu_operation(ctx: DeviceContext) raises -> None:
            # Perform some GPU operation using ctx
            pass

        with DeviceContext() as ctx:
            # Measure execution time of a function that uses the context
            var time_ns = ctx.execution_time[gpu_operation](10)
            print("Execution time for 10 iterations:", time_ns, "ns")
        ```
        """

        @always_inline
        def func_unified(ctx: Self):
            try:
                func(ctx)
            except e:
                abort(String(e))

        return self.execution_time(func_unified, num_iters)

    @always_inline
    def execution_time[
        FuncType: def(Self) raises -> None,
    ](self, func: FuncType, num_iters: Int) raises -> Int:
        """Measures the execution time of a function that takes a DeviceContext parameter.

        Parameters:
            FuncType: The body function type.

        Args:
            func: The closure carrying the captured state of the body function.
            num_iters: The number of iterations to run the function.

        Returns:
            The total elapsed time in nanoseconds for all iterations.

        Raises:
            If the timer operations fail.
        """
        var timer_ptr: _DeviceTimerPtr[mut=True] = {}
        _checked(
            external_call[
                "AsyncRT_DeviceContext_startTimer",
                _CString[],
            ](
                UnsafePointer(to=timer_ptr),
                self._handle,
            )
        )
        var timer = _DeviceTimer(timer_ptr)
        for _ in range(num_iters):
            func(self)
        var elapsed_nanos: Int = 0
        _checked(
            external_call[
                "AsyncRT_DeviceContext_stopTimer",
                _CString[],
            ](
                UnsafePointer(to=elapsed_nanos),
                self._handle,
                timer._handle,
            )
        )
        return elapsed_nanos

    def push_context(self) raises -> _DeviceContextScope:
        """Returns a context manager that ensures this device's driver context is active.

        This method returns a context manager that pushes this device's driver
        context as the current context on entry and restores the previous context
        on exit. This is useful for operations that require a specific GPU context
        to be active, such as cuDNN operations on multi-GPU systems.

        Returns:
            A context manager that manages the driver context stack.

        Raises:
            If there's an error switching contexts.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext

        var ctx = DeviceContext(device_id=1)
        # Ensure GPU 1's context is active for these operations.
        with ctx.push_context():
            # All GPU operations here will use GPU 1's context.
            ...  # call external stateful APIs, such as cudnn.
        # Previous context is automatically restored
        ```
        """
        comptime assert not is_gpu(), "DeviceContext is not supported on GPUs"
        return _DeviceContextScope(self)

    def set_as_current(self) raises:
        """For use with libraries that require a specific GPU context to be
        active. Sets the current device to the one associated with this
        DeviceContext.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext
        var ctx = DeviceContext(device_id=1)
        ctx.set_as_current()
        ```

        Raises:
            If there's an error setting the current device.
        """

        _checked(
            external_call["AsyncRT_DeviceContext_setAsCurrent", _CString[]](
                self._handle,
            )
        )

    @always_inline
    def execution_time[
        func: def() raises capturing[_] -> None
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

        ```text
        from std.gpu.host import DeviceContext

        def some_gpu_operation() raises -> None:
            # Perform some GPU operation
            pass

        with DeviceContext() as ctx:
            # Measure execution time of a function
            var time_ns = ctx.execution_time[some_gpu_operation](10)
            print("Execution time:", time_ns, "ns")
        ```
        """

        @always_inline
        def func_unified():
            try:
                func()
            except e:
                abort(String(e))

        return self.execution_time(func_unified, num_iters)

    @always_inline
    def execution_time[
        FuncType: def() raises -> None,
    ](self, func: FuncType, num_iters: Int) raises -> Int:
        """Measures the execution time of a function over multiple iterations.

        Parameters:
            FuncType: The body function type.

        Args:
            func: The closure carrying the captured state of the body function.
            num_iters: The number of iterations to run the function.

        Returns:
            The total elapsed time in nanoseconds for all iterations.

        Raises:
            If the timer operations fail.
        """
        var timer_ptr: _DeviceTimerPtr[mut=True] = {}
        _checked(
            external_call[
                "AsyncRT_DeviceContext_startTimer",
                _CString[],
            ](
                UnsafePointer(to=timer_ptr),
                self._handle,
            )
        )
        var timer = _DeviceTimer(timer_ptr)
        for _ in range(num_iters):
            func()
        var elapsed_nanos: Int = 0
        _checked(
            external_call[
                "AsyncRT_DeviceContext_stopTimer",
                _CString[],
            ](
                UnsafePointer(to=elapsed_nanos),
                self._handle,
                timer._handle,
            )
        )
        return elapsed_nanos

    @always_inline
    def execution_time_iter[
        func: def(Self, Int) raises capturing[_] -> None
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

        ```text
        from std.gpu.host import DeviceContext

        def benchmark_kernel(ctx: DeviceContext, i: Int) raises -> None:
            # Perform GPU operations using ctx, potentially varying by iteration
            pass

        with DeviceContext() as ctx:
            # Measure execution time with iteration awareness
            var time_ns = ctx.execution_time_iter[benchmark_kernel](10)
            print("Total execution time:", time_ns, "ns")
        ```
        """

        @always_inline
        def func_unified(ctx: Self, i: Int):
            try:
                func(ctx, i)
            except e:
                abort(String(e))

        return self.execution_time_iter(func_unified, num_iters)

    @always_inline
    def execution_time_iter[
        FuncType: def(Self, Int) raises -> None,
    ](self, func: FuncType, num_iters: Int) raises -> Int:
        """Measures the execution time of a function that takes iteration index as input.

        Parameters:
            FuncType: The body function type.

        Args:
            func: The closure carrying the captured state of the body function.
            num_iters: The number of iterations to run the function.

        Returns:
            The total elapsed time in nanoseconds for all iterations.

        Raises:
            If the timer operations fail.
        """
        var timer_ptr: _DeviceTimerPtr[mut=True] = {}
        _checked(
            external_call[
                "AsyncRT_DeviceContext_startTimer",
                _CString[],
            ](
                UnsafePointer(to=timer_ptr),
                self._handle,
            )
        )
        var timer = _DeviceTimer(timer_ptr)
        for i in range(num_iters):
            func(self, i)
        var elapsed_nanos: Int = 0
        _checked(
            external_call[
                "AsyncRT_DeviceContext_stopTimer",
                _CString[],
            ](
                UnsafePointer(to=elapsed_nanos),
                self._handle,
                timer._handle,
            )
        )
        return elapsed_nanos

    @always_inline
    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst_buf: DeviceBuffer[dtype, ...],
        src_ptr: OptionalUnsafePointer[Scalar[dtype], ...],
    ) raises:
        """Enqueues an async copy from the host to the provided device
        buffer. The number of bytes copied is determined by the size of the
        device buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_ptr: Host pointer to copy from.

        Raises:
            If the operation fails.
        """
        # const char * AsyncRT_DeviceContext_HtoD_async(const DeviceContext *ctx, const DeviceBuffer *dst, const void *src)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_HtoD_async",
                _CString[],
            ](
                self._handle,
                dst_buf._handle,
                src_ptr,
            )
        )

    @always_inline
    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst_buf: HostBuffer[dtype, ...],
        src_ptr: OptionalUnsafePointer[Scalar[dtype], ...],
    ) raises:
        """Enqueues an async copy from the host to the provided device
        buffer. The number of bytes copied is determined by the size of the
        device buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_ptr: Host pointer to copy from.

        Raises:
            If the operation fails.
        """
        # const char * AsyncRT_DeviceContext_HtoD_async(const DeviceContext *ctx, const DeviceBuffer *dst, const void *src)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_HtoD_async",
                _CString[],
            ](
                self._handle,
                dst_buf._handle,
                src_ptr,
            )
        )

    @always_inline
    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst_ptr: OptionalUnsafePointer[mut=True, Scalar[dtype], ...],
        src_buf: DeviceBuffer[dtype, ...],
    ) raises:
        """Enqueues an async copy from the device to the host. The
        number of bytes copied is determined by the size of the device buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_ptr: Host pointer to copy to.
            src_buf: Device buffer to copy from.

        Raises:
            If the operation fails.
        """
        # const char * AsyncRT_DeviceContext_DtoH_async(const DeviceContext *ctx, void *dst, const DeviceBuffer *src)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_DtoH_async",
                _CString[],
            ](
                self._handle,
                dst_ptr,
                src_buf._handle,
            )
        )

    @always_inline
    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst_ptr: OptionalUnsafePointer[mut=True, Scalar[dtype], ...],
        src_buf: HostBuffer[dtype, ...],
    ) raises:
        """Enqueues an async copy from the device to the host. The
        number of bytes copied is determined by the size of the device buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_ptr: Host pointer to copy to.
            src_buf: Device buffer to copy from.

        Raises:
            If the operation fails.
        """
        # const char * AsyncRT_DeviceContext_DtoH_async(const DeviceContext *ctx, void *dst, const DeviceBuffer *src)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_DtoH_async",
                _CString[],
            ](
                self._handle,
                dst_ptr,
                src_buf._handle,
            )
        )

    @always_inline
    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst_ptr: OptionalUnsafePointer[mut=True, Scalar[dtype], ...],
        src_ptr: OptionalUnsafePointer[Scalar[dtype], ...],
        size: Int,
    ) raises:
        """Enqueues an async copy of `size` elements from a device pointer to
        another device pointer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_ptr: Host pointer to copy to.
            src_ptr: Device pointer to copy from.
            size: Number of elements (of the specified `DType`) to copy.

        Raises:
            If the operation fails.
        """

        def to_device_buffer(
            pointer: OptionalUnsafePointer[Scalar[dtype], ...]
        ) {read} -> DeviceBuffer[dtype]:
            if pointer:
                return DeviceBuffer[dtype](
                    self,
                    pointer.unsafe_value(),
                    size,
                    owning=False,
                )
            else:
                return DeviceBuffer[dtype].empty(self)

        self.enqueue_copy(to_device_buffer(dst_ptr), to_device_buffer(src_ptr))

    @always_inline
    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst_buf: DeviceBuffer[dtype, ...],
        src: Span[Scalar[dtype], _],
    ) raises:
        """Enqueues an async copy from a host `Span` to a device buffer.

        The number of bytes copied is determined by the size of the device
        buffer. The span must contain at least as many elements as the
        destination buffer; this invariant is checked via `debug_assert`.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src: Host span to copy from.

        Raises:
            If the operation fails.
        """
        debug_assert(
            len(src) >= len(dst_buf),
            "source span length must be >= destination buffer length",
        )
        self.enqueue_copy(dst_buf, src.unsafe_ptr())

    @always_inline
    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst_buf: HostBuffer[dtype, ...],
        src: Span[Scalar[dtype], _],
    ) raises:
        """Enqueues an async copy from a host `Span` to a host buffer.

        The number of bytes copied is determined by the size of the
        destination buffer. The span must contain at least as many elements
        as the destination buffer; this invariant is checked via
        `debug_assert`.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Host buffer to copy to.
            src: Host span to copy from.

        Raises:
            If the operation fails.
        """
        debug_assert(
            len(src) >= len(dst_buf),
            "source span length must be >= destination buffer length",
        )
        self.enqueue_copy(dst_buf, src.unsafe_ptr())

    @always_inline
    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst: Span[mut=True, Scalar[dtype], _],
        src_buf: DeviceBuffer[dtype, ...],
    ) raises:
        """Enqueues an async copy from a device buffer to a host `Span`.

        The number of bytes copied is determined by the size of the device
        buffer. The span must contain at least as many elements as the source
        buffer; this invariant is checked via `debug_assert` (debug builds
        only).

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst: Host span to copy to.
            src_buf: Device buffer to copy from.

        Raises:
            If the operation fails.
        """
        debug_assert(
            len(dst) >= len(src_buf),
            "destination span length must be >= source buffer length",
        )
        self.enqueue_copy(dst.unsafe_ptr(), src_buf)

    @always_inline
    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst: Span[mut=True, Scalar[dtype], _],
        src_buf: HostBuffer[dtype, ...],
    ) raises:
        """Enqueues an async copy from a host buffer to a host `Span`.

        The number of bytes copied is determined by the size of the source
        buffer. The span must contain at least as many elements as the source
        buffer; this invariant is checked via `debug_assert` (debug builds
        only).

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst: Host span to copy to.
            src_buf: Host buffer to copy from.

        Raises:
            If the operation fails.
        """
        debug_assert(
            len(dst) >= len(src_buf),
            "destination span length must be >= source buffer length",
        )
        self.enqueue_copy(dst.unsafe_ptr(), src_buf)

    @always_inline
    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst_buf: DeviceBuffer[dtype, ...],
        src_buf: DeviceBuffer[dtype, ...],
    ) raises:
        """Enqueues an async copy from one device buffer to another. The amount
        of data transferred is determined by the size of the destination buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_buf: Device buffer to copy from. Must be at least as large as
                `dst`.

        Raises:
            If the operation fails.
        """
        # const char * AsyncRT_DeviceContext_DtoD_async(const DeviceContext *ctx, const DeviceBuffer *dst, const DeviceBuffer *src)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_DtoD_async",
                _CString[],
                _DeviceContextPtr[mut=True],
                _DeviceBufferPtr[mut=True],
                _DeviceBufferPtr[mut=True],
            ](
                self._handle,
                dst_buf._handle,
                src_buf._handle,
            )
        )

    @always_inline
    def enqueue_copy[
        dtype: DType
    ](
        self, dst_buf: DeviceBuffer[dtype, ...], src_buf: HostBuffer[dtype, ...]
    ) raises:
        """Enqueues an async copy from one device buffer to another. The amount
        of data transferred is determined by the size of the destination buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_buf: Device buffer to copy from. Must be at least as large as
                `dst`.

        Raises:
            If the operation fails.
        """
        # const char * AsyncRT_DeviceContext_DtoD_async(const DeviceContext *ctx, const DeviceBuffer *dst, const DeviceBuffer *src)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_DtoD_async",
                _CString[],
                _DeviceContextPtr[mut=True],
                _DeviceBufferPtr[mut=True],
                _DeviceBufferPtr[mut=True],
            ](
                self._handle,
                dst_buf._handle,
                src_buf._handle,
            )
        )

    @always_inline
    def enqueue_copy[
        dtype: DType
    ](
        self, dst_buf: HostBuffer[dtype, ...], src_buf: DeviceBuffer[dtype, ...]
    ) raises:
        """Enqueues an async copy from one device buffer to another. The amount
        of data transferred is determined by the size of the destination buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_buf: Device buffer to copy from. Must be at least as large as
                `dst`.

        Raises:
            If the operation fails.
        """
        # const char * AsyncRT_DeviceContext_DtoD_async(const DeviceContext *ctx, const DeviceBuffer *dst, const DeviceBuffer *src)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_DtoD_async",
                _CString[],
                _DeviceContextPtr[mut=True],
                _DeviceBufferPtr[mut=True],
                _DeviceBufferPtr[mut=True],
            ](
                self._handle,
                dst_buf._handle,
                src_buf._handle,
            )
        )

    @always_inline
    def enqueue_copy[
        dtype: DType
    ](
        self, dst_buf: HostBuffer[dtype, ...], src_buf: HostBuffer[dtype, ...]
    ) raises:
        """Enqueues an async copy from one device buffer to another. The amount
        of data transferred is determined by the size of the destination buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_buf: Device buffer to copy from. Must be at least as large as
                `dst`.

        Raises:
            If the operation fails.
        """
        # const char * AsyncRT_DeviceContext_DtoD_async(const DeviceContext *ctx, const DeviceBuffer *dst, const DeviceBuffer *src)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_DtoD_async",
                _CString[],
                _DeviceContextPtr[mut=True],
                _DeviceBufferPtr[mut=True],
                _DeviceBufferPtr[mut=True],
            ](
                self._handle,
                dst_buf._handle,
                src_buf._handle,
            )
        )

    @always_inline
    def enqueue_memset[
        dtype: DType
    ](self, dst: DeviceBuffer[dtype, ...], val: Scalar[dtype]) raises:
        """Enqueues an async memset operation, setting all of the elements in
        the destination device buffer to the specified value.

        Parameters:
            dtype: Type of the data stored in the buffer.

        Args:
            dst: Destination buffer.
            val: Value to set all elements of `dst` to.

        Raises:
            If the operation fails.
        """
        comptime bitwidth = bit_width_of[dtype]()
        comptime assert (
            bitwidth == 8 or bitwidth == 16 or bitwidth == 32 or bitwidth == 64
        ), "bitwidth of memset dtype must be one of [8,16,32,64]"
        var value: UInt64

        comptime if bitwidth == 8:
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
                _CString[],
                _DeviceContextPtr[mut=True],
                _DeviceBufferPtr[mut=True],
                UInt64,
                c_size_t,
            ](
                self._handle,
                dst._handle,
                value,
                c_size_t(size_of[dtype]()),
            )
        )

    def enqueue_memset[
        dtype: DType
    ](self, dst: HostBuffer[dtype, ...], val: Scalar[dtype]) raises:
        """Enqueues an async memset operation, setting all of the elements in
        the destination host buffer to the specified value.

        Parameters:
            dtype: Type of the data stored in the buffer.

        Args:
            dst: Destination buffer.
            val: Value to set all elements of `dst` to.

        Raises:
            If the operation fails.
        """
        comptime bitwidth = bit_width_of[dtype]()
        comptime assert (
            bitwidth == 8 or bitwidth == 16 or bitwidth == 32 or bitwidth == 64
        ), "bitwidth of memset dtype must be one of [8,16,32,64]"
        var value: UInt64

        comptime if bitwidth == 8:
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
                _CString[],
                _DeviceContextPtr[mut=True],
                _DeviceBufferPtr[mut=True],
                UInt64,
                c_size_t,
            ](
                self._handle,
                dst._handle,
                value,
                UInt(size_of[dtype]()),
            )
        )

    @doc_hidden
    @always_inline
    def stream(self) raises -> DeviceStream:
        return DeviceStream(self)

    @always_inline
    def create_event[
        *,
        blocking_sync: Bool = False,
        disable_timing: Bool = True,
        interprocess: Bool = False,
    ](self) raises -> DeviceEvent:
        """Creates a new event for synchronization between streams.

        Provides the best performance by default, disabling timing and blocking sync.
        `DeviceContext.execution_time()` provides the functionality required for
        timing kernels by passing it a closure, and is functionally equivalent to
        recording start and end events, then calculating the elapsed time.

        Parameters:
            blocking_sync: Enable `event.synchronize()` to block until the event
                has been recorded. Incurs overhead compared to
                `stream.enqueue_wait_for(event)` (default: False).
            disable_timing: Remove timing overhead (default: True).
            interprocess: Enable interprocess synchronization, currently
                unimplemented. (default: False).

        Returns:
            A DeviceEvent that can be used for synchronization.

        Raises:
            If event creation fails.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext

        var ctx = DeviceContext()

        var default_stream = ctx.stream()
        var new_stream = ctx.create_stream()

        # Create an event
        var event = ctx.create_event()

        # Wait for the event in new_stream
        new_stream.enqueue_wait_for(event)

        # new_stream can continue
        default_stream.record_event(event)
        default_stream.synchronize()
        ```
        """
        var result: _DeviceEventPtr[mut=True] = {}
        var flags = EventFlags.default

        comptime if blocking_sync:
            flags |= EventFlags.blocking_sync

        comptime if disable_timing:
            flags |= EventFlags.disable_timing

        comptime if interprocess:
            flags |= EventFlags.interprocess

        # const char *AsyncRT_DeviceContext_eventCreate(const DeviceEvent **result, const DeviceContext *ctx, unsigned int flags)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_eventCreate",
                _CString[],
                UnsafePointer[_DeviceEventPtr[mut=True], origin_of(result)],
                _DeviceContextPtr[mut=True],
                EventFlags,
            ](UnsafePointer(to=result), self._handle, flags)
        )
        return DeviceEvent(result)

    def stream_priority_range(self) raises -> StreamPriorityRange:
        """Returns the range of stream priorities supported by this device context.

        Returns:
            A StreamPriorityRange object containing the minimum and maximum stream priorities.

        Raises:
            If the operation fails.
        """
        var least_priority = c_int(0)
        var greatest_priority = c_int(0)
        # const char *AsyncRT_DeviceContext_streamPriorityRange(int *leastPriority, int *greatestPriority, const DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_streamPriorityRange",
                _CString[],
            ](
                UnsafePointer(to=least_priority),
                UnsafePointer(to=greatest_priority),
                self._handle,
            )
        )
        return StreamPriorityRange(Int(least_priority), Int(greatest_priority))

    def create_stream(self, *, priority: Int = 0) raises -> DeviceStream:
        """Creates a new stream associated with the given device context.

        To create a stream with the highest priority, use:

        ```mojo
        from std.gpu.host import DeviceContext
        var ctx = DeviceContext()
        var priority = ctx.stream_priority_range().greatest
        var stream = ctx.create_stream(priority=priority)
        ```

        Args:
            priority: The priority of the stream (default: 0).

        Returns:
            The newly created device stream with the specified priority.

        Raises:
            If stream creation fails.
        """
        var result: _DeviceStreamPtr[mut=True] = {}

        # const char *AsyncRT_DeviceContext_createStream(const DeviceStream **stream, int priority, const DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_createStream",
                _CString[],
            ](UnsafePointer(to=result), c_int(priority), self._handle)
        )
        return DeviceStream(result)

    def create_external_stream(
        self, external_stream: _CPointer[mut=True, NoneType, _]
    ) raises -> DeviceStream:
        """Creates a non-owning stream wrapper around an externally managed GPU stream.

        The returned `DeviceStream` does not
        take ownership of the underlying stream. The caller is responsible for
        ensuring the external stream remains valid for the lifetime of the
        returned wrapper.

        Args:
            external_stream: An opaque pointer to the external GPU stream handle
                (e.g., a `CUstream` or `hipStream_t` cast to `void*`).

        Returns:
            A `DeviceStream` wrapping the external stream without taking
            ownership of it.

        Raises:
            If wrapping the external stream fails.
        """
        var result: _DeviceStreamPtr[mut=True] = {}

        # const char *AsyncRT_DeviceContext_createExternalStream(const DeviceStream **stream, void *externalStream, const DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_createExternalStream",
                _CString[],
            ](UnsafePointer(to=result), external_stream, self._handle)
        )
        return DeviceStream(result)

    @always_inline
    def synchronize(self) raises:
        """Blocks until all asynchronous calls on the stream associated with
        this device context have completed.


        Raises:
            If the operation fails. This should never be necessary when
            writing a custom operation.
        """
        # const char * AsyncRT_DeviceContext_synchronize(const DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_synchronize",
                _CString[],
                _DeviceContextPtr[mut=True],
            ](
                self._handle,
            ),
            location=call_location(),
        )

    def enqueue_wait_for(self, other: DeviceContext) raises:
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
        from std.gpu.host import DeviceContext

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
                _CString[],
                _DeviceContextPtr[mut=True],
                _DeviceContextPtr[mut=True],
            ](self._handle, other._handle)
        )

    @always_inline
    def get_api_version(self) raises -> Int:
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
        from std.gpu.host import DeviceContext

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
                _CString[],
            ](
                UnsafePointer(to=value),
                self._handle,
            ),
            location=call_location(),
        )
        return Int(value)

    @always_inline
    def get_attribute(self, attr: DeviceAttribute) raises -> Int:
        """Returns the specified attribute for this device.

        Use the aliases defined by
        [DeviceAttribute](/docs/std/gpu/host/device_attribute/DeviceAttribute/)
        to specify attributes. For example:

        ```mojo
        from std.gpu.host import DeviceAttribute, DeviceContext

        def main() raises:
            var ctx = DeviceContext()
            var attr = DeviceAttribute.MAX_BLOCKS_PER_MULTIPROCESSOR
            var max_blocks = ctx.get_attribute(attr)
            print(max_blocks)
        ```

        Args:
            attr: The device attribute to query.

        Returns:
            The value for `attr` on this device.

        Raises:
            If the operation fails.
        """
        var value: c_int = 0
        # const char * AsyncRT_DeviceContext_getAttribute(int *result, const DeviceContext *ctx, int attr)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_getAttribute",
                _CString[],
            ](
                UnsafePointer(to=value),
                self._handle,
                c_int(attr._value),
            ),
            location=call_location(),
        )
        return Int(value)

    @always_inline
    def is_compatible(self) -> Bool:
        """Returns True if this device is compatible with MAX.

        This method checks whether the current device is compatible with the
        Modular Accelerated Execution (MAX) runtime. It's useful for validating
        that the device can execute the compiled code before attempting operations.

        Returns:
            True if the device is compatible with MAX, False otherwise.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext

        var ctx = DeviceContext()
        print("Device is compatible with MAX:", ctx.is_compatible())
        ```
        """
        # const char * AsyncRT_DeviceContext_isCompatible(const DeviceContext *ctx)
        try:
            _checked(
                external_call[
                    "AsyncRT_DeviceContext_isCompatible",
                    _CString[],
                    _DeviceContextPtr[mut=True],
                ](
                    self._handle,
                ),
                location=call_location(),
            )
            return True
        except:
            return False

    @always_inline
    def run_healthcheck(self) raises:
        """Runs lightweight GPU health validation.

        Checks for hardware throttling, uncorrectable ECC errors, and stuck
        VRAM. Raises an error if the GPU is unhealthy. The healthcheck runs
        automatically during device initialization; this method allows
        re-running it explicitly.

        Disable with `MODULAR_DEVICE_CONTEXT_DISABLE_HEALTHCHECK=true`.

        Raises:
            Error: If the GPU is in an unhealthy state.
        """
        # const char *AsyncRT_DeviceContext_runHealthcheck(DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_runHealthcheck",
                _CString[],
                _DeviceContextPtr[mut=True],
            ](self._handle),
            location=call_location(),
        )

    @always_inline
    def id(self) raises -> Int64:
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
        from std.gpu.host import DeviceContext

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
            "AsyncRT_DeviceContext_id", Int64, _DeviceContextPtr[mut=True]
        ](self._handle)

    @doc_hidden
    @always_inline
    def compute_capability(self) raises -> Int:
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
                _CString[],
            ](UnsafePointer(to=compute_capability), self._handle),
            location=call_location(),
        )
        return Int(compute_capability)

    @doc_hidden
    @always_inline
    def arch_name(self) raises -> String:
        """Returns the architecture name of this device.

        This internal method retrieves the architecture name of AMD GPUs.

        Returns:
            The compute capability as a string (e.g., `gfx942` for `MI300`).

        Raises:
            If there's an error retrieving the compute capability.

        Notes:

        This is a private method intended for internal use only.
        """
        var arch_name = StaticString()
        external_call[
            "AsyncRT_DeviceContext_archName",
            NoneType,
        ](UnsafePointer(to=arch_name), self._handle)
        return String(arch_name)

    @always_inline
    def get_memory_info(self) raises -> Tuple[c_size_t, c_size_t]:
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
        from std.gpu.host import DeviceContext

        var ctx = DeviceContext()
        try:
            (free, total) = ctx.get_memory_info()
            print("Free memory:", free / (1024*1024), "MB")
            print("Total memory:", total / (1024*1024), "MB")
        except:
            print("Failed to get memory information")
        ```
        """
        var free = c_size_t(0)
        var total = c_size_t(0)
        # const char *AsyncRT_DeviceContext_getMemoryInfo(const DeviceContext *ctx, size_t *free, size_t *total)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_getMemoryInfo",
                _CString[],
                _DeviceContextPtr[mut=True],
                UnsafePointer[c_size_t, origin_of(free)],
                UnsafePointer[c_size_t, origin_of(total)],
            ](
                self._handle,
                UnsafePointer(to=free),
                UnsafePointer(to=total),
            ),
            location=call_location(),
        )

        return (free, total)

    @always_inline
    def can_access(self, peer: DeviceContext) raises -> Bool:
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
        from std.gpu.host import DeviceContext
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
                _CString[],
                UnsafePointer[Bool, origin_of(result)],
                _DeviceContextPtr[mut=True],
                _DeviceContextPtr[mut=True],
            ](
                UnsafePointer(to=result),
                self._handle,
                peer._handle,
            ),
            location=call_location(),
        )
        return result

    @always_inline
    def enable_peer_access(self, peer: DeviceContext) raises:
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
        from std.gpu.host import DeviceContext

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
                _CString[],
                _DeviceContextPtr[mut=True],
                _DeviceContextPtr[mut=True],
            ](
                self._handle,
                peer._handle,
            ),
            location=call_location(),
        )

    @always_inline
    def supports_multicast(self) raises -> Bool:
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
                _CString[],
                UnsafePointer[Bool, origin_of(result)],
                _DeviceContextPtr[mut=True],
            ](
                UnsafePointer(to=result),
                self._handle,
            ),
            location=call_location(),
        )
        return result

    @staticmethod
    @always_inline
    def number_of_devices(
        *, var api: String = String(Self.default_device_info.api)
    ) -> Int:
        """Returns the number of devices available that support the specified API.

        This function queries the system for available devices that support the
        requested API (such as CUDA or HIP). It's useful for determining how many
        accelerators are available before allocating resources or distributing work.

        Args:
            api: Requested device API (for example, "cuda" or "hip"). Defaults
                to the device API specified by current target accelerator.

        Returns:
            The number of available devices supporting the specified API.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext

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
            ](api.as_c_string_slice())
        )

    @staticmethod
    def enable_all_peer_access() raises:
        """Enable peer-to-peer memory access between all available accelerators.

        This function detects all available accelerators in the system and enables
        peer-to-peer (P2P) memory access between every pair of devices.

        When peer access is enabled, kernels running on one device can directly access
        memory allocated on another device without going through host memory. This is
        crucial for efficient multi-GPU operations like allreduce.

        The function is a no-op when:
        - No accelerators are available
        - Only one accelerator is available
        - Peer access is already enabled between devices

        Raises:
            If peer access cannot be enabled between any pair of devices.
                   This can happen if the hardware doesn't support P2P access or if
                   there's a configuration issue.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext

        # Enable P2P access between all GPUs
        DeviceContext.enable_all_peer_access()

        # Now GPUs can directly access each other's memory
        ```
        """
        # const char *AsyncRT_DeviceContext_enableAllPeerAccess()
        _checked(
            external_call[
                "AsyncRT_DeviceContext_enableAllPeerAccess",
                _CString[],
            ]()
        )

    @staticmethod
    @always_inline
    def all_peer_access_enabled() raises -> Bool:
        """Check whether peer-to-peer memory access is enabled between all GPU pairs.

        This function queries whether P2P access has been successfully enabled
        between all pairs of GPUs in the system. It returns True only if every
        GPU can directly access every other GPU's memory.

        Returns:
            True if P2P access is enabled between all GPU pairs, False otherwise.
            Returns False if there are fewer than 2 GPUs or if P2P is not
            supported between any pair.

        Raises:
            If there's an error querying the P2P access status.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext

        # P2P access is automatically enabled when devices are constructed.
        # Check if it was successful for all pairs.
        if DeviceContext.all_peer_access_enabled():
            print("P2P access enabled between all GPUs")
        else:
            print("P2P access not available for all GPU pairs")
        ```
        """
        var result: Bool = False
        # const char *AsyncRT_DeviceContext_allPeerAccessEnabled(bool *result)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_allPeerAccessEnabled",
                _CString[],
                UnsafePointer[Bool, origin_of(result)],
            ](UnsafePointer(to=result)),
            location=call_location(),
        )
        return result

    def create_graph_builder(self) raises -> DeviceGraphBuilder:
        """Creates a graph builder for explicit device graph construction.

        Returns a `DeviceGraphBuilder` that can be used to add kernel nodes
        and then instantiate a reusable `DeviceGraph`.

        Returns:
            A new `DeviceGraphBuilder` associated with this device context.

        Raises:
            If graph builder creation fails or is not supported on this device.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext

        def kernel(x: Int):
            print("Value:", x)

        with DeviceContext() as ctx:
            var compiled_fn = ctx.compile_function[kernel, kernel]()
            var builder = ctx.create_graph_builder()
            builder.add_function(compiled_fn, 42, grid_dim=1, block_dim=1)
            var graph = builder^.instantiate()
            graph.replay()
            ctx.synchronize()
        ```
        """
        var result: _DeviceGraphBuilderPtr[mut=True] = {}
        # const char *AsyncRT_DeviceContext_createGraphBuilder(
        #     DeviceGraphBuilder **result, DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_createGraphBuilder",
                _CString[],
                UnsafePointer[
                    _DeviceGraphBuilderPtr[mut=True], origin_of(result)
                ],
                _DeviceContextPtr[mut=True],
            ](
                UnsafePointer(to=result),
                self._handle,
            )
        )
        return DeviceGraphBuilder(result, self)


struct DeviceMulticastBuffer[dtype: DType]:
    """Represents a multicast memory object enables special memory operations to be broadcast
    across a group of devices.

    Parameters:
        dtype: Data dtype to be stored in the associated memory regions.
    """

    var _handle: _DeviceMulticastBufferPtr[mut=True]

    @doc_hidden
    def __init__(
        out self,
        var contexts: List[DeviceContext],
        size: Int,
    ) raises:
        comptime elem_size = size_of[Self.dtype]()
        var handle: _DeviceMulticastBufferPtr[mut=True] = {}

        var ctxs_len = len(contexts)
        var ctxs = alloc[_DeviceContextPtr[mut=True]](ctxs_len)
        for i in range(ctxs_len):
            ctxs[i] = contexts[i]._handle

        # const char* AsyncRT_DeviceMulticastBuffer_allocate(const DeviceMulticastBuffer **result, size_t ctxsLen, const DeviceContext **ctxs, size_t len, size_t elemSize)
        _checked(
            external_call[
                "AsyncRT_DeviceMulticastBuffer_allocate",
                _CString[],
            ](
                UnsafePointer(to=handle),
                c_size_t(ctxs_len),
                ctxs,
                c_size_t(size),
                c_size_t(elem_size),
            )
        )

        self._handle = handle

    @doc_hidden
    def unicast_buffer_for(
        self, ctx: DeviceContext
    ) raises -> DeviceBuffer[Self.dtype]:
        # const char* AsyncRT_DeviceMulticastBuffer_unicastBufferFor(const DeviceBuffer **result, void **devicePtr, const DeviceMulticastBuffer *multiBuffer, const DeviceContext* ctx)
        var buf_handle = _DeviceBufferPtr[mut=True]()
        var buf_ptr = Optional[
            UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]
        ]()

        _checked(
            external_call[
                "AsyncRT_DeviceMulticastBuffer_unicastBufferFor",
                _CString[],
            ](
                UnsafePointer(to=buf_handle),
                UnsafePointer(to=buf_ptr),
                self._handle,
                ctx._handle,
            )
        )

        return DeviceBuffer[Self.dtype](buf_handle, buf_ptr.value())

    @doc_hidden
    def multicast_buffer_for(
        self, ctx: DeviceContext
    ) raises -> DeviceBuffer[Self.dtype]:
        # const char* AsyncRT_DeviceMulticastBuffer_multicastBufferFor(const DeviceBuffer **result, void **devicePtr, const DeviceMulticastBuffer *multiBuffer, const DeviceContext* ctx)
        var buf_handle = _DeviceBufferPtr[mut=True]()
        var buf_ptr = Optional[
            UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]
        ]()

        _checked(
            external_call[
                "AsyncRT_DeviceMulticastBuffer_multicastBufferFor",
                _CString[],
            ](
                UnsafePointer(to=buf_handle),
                UnsafePointer(to=buf_ptr),
                self._handle,
                ctx._handle,
            )
        )

        return DeviceBuffer[Self.dtype](buf_handle, buf_ptr.value())


struct _HostMappedBuffer[dtype: DType]:
    var _ctx: DeviceContext
    var _dev_buf: DeviceBuffer[Self.dtype]
    var _cpu_buf: HostBuffer[Self.dtype]

    def __init__(
        out self, ctx: DeviceContext, buf: DeviceBuffer[Self.dtype]
    ) raises:
        var cpu_buf = ctx.enqueue_create_host_buffer[Self.dtype](len(buf))
        self._ctx = ctx
        self._dev_buf = buf
        self._cpu_buf = cpu_buf

    def __del__(deinit self):
        pass

    def __enter__(mut self) raises -> HostBuffer[Self.dtype]:
        self._dev_buf.enqueue_copy_to(self._cpu_buf)
        self._ctx.synchronize()
        return self._cpu_buf

    def __exit__(mut self) raises:
        self._ctx.synchronize()
        self._cpu_buf.enqueue_copy_to(self._dev_buf)
        self._ctx.synchronize()


struct _DeviceContextScope:
    var _ctx: DeviceContext
    var _handle: _DeviceContextScopePtr[mut=True]

    def __init__(out self, ctx: DeviceContext):
        self._ctx = ctx
        self._handle = {}

    def __del__(deinit self):
        # Ensure that the C++ scope is removed in all cases.
        if self._handle:
            self._release()

    def __enter__(mut self) raises -> DeviceContext:
        # Create a C++ DeviceContextScope
        var cpp_handle: _DeviceContextScopePtr[mut=True] = {}

        # const char *AsyncRT_DeviceContextScope_create(const DeviceContextScope **result, const DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceContextScope_create",
                _CString[],
            ](
                UnsafePointer(to=cpp_handle),
                self._ctx._handle,
            )
        )
        self._handle = cpp_handle

        return self._ctx

    def __exit__(mut self) raises:
        # Release the C++ DeviceContextScope
        self._release()
        self._handle = {}

    def _release(mut self):
        # void AsyncRT_DeviceContextScope_release(const DeviceContextScope *scope)
        external_call[
            "AsyncRT_DeviceContextScope_release",
            NoneType,
        ](
            self._handle,
        )
