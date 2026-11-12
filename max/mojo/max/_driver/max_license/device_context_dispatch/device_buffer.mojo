# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Dispatching `DeviceBuffer`, and the host-mapped view of one."""

from std.builtin.device_passable import (
    DevicePassable,
    DeviceTypeEncoder,
)
from std.format import Writer
from std.memory import Pointer
from std.sys import is_gpu

from max._driver.max_license.device_context_hal.device_context_hal import (
    HALDeviceBuffer,
)
from max.gpu.host._driver_handles import (
    _DeviceBufferPtr as _HALDeviceBufferPtr,
)
from max._driver.max_license.device_context_dispatch.device_pointer import (
    DevicePointer,
)
from max.gpu.host._legacy.device_context import LegacyDeviceBuffer
from max.gpu.host._driver_handles import (
    _DeviceBufferPtr as _LegacyDeviceBufferPtr,
)
from .device_context import use_legacy_backend
from max.gpu.host.info import default_accelerator
from .host_types import Device
from max._driver.max_license.device_context_dispatch.device_context import (
    DeviceContext,
)
from max._driver.max_license.device_context_dispatch.host_buffer import (
    HostBuffer,
)


struct DeviceBuffer[dtype: DType](
    DevicePassable, ImplicitlyCopyable, Sized, Writable
):
    """Represents a block of device-resident storage. For GPU devices, a device
    buffer is allocated in the device's global memory.

    To allocate a `DeviceBuffer`, use one of the methods provided by
    `DeviceContext`, such as
    [`enqueue_create_buffer()`](/api/mojo/max/gpu/host/device_context/DeviceContext/#enqueue_create_buffer).

    Parameters:
        dtype: Data dtype to be stored in the buffer.
    """

    # ===-------------------------------------------------------------------===#
    # Backend selection
    # ===-------------------------------------------------------------------===#

    # `DeviceBuffer` carries no target, so it takes the default accelerator's
    # backend. A context that produces or consumes one asserts agreement.
    comptime _Legacy = LegacyDeviceBuffer[Self.dtype]
    comptime _HAL = HALDeviceBuffer[Self.dtype]
    comptime _Impl = Self._Legacy if use_legacy_backend() else Self._HAL

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    # The backend buffer must stay the only field, and it must keep the device
    # address as its own first word: kernel dispatch reinterprets the start of a
    # `DeviceBuffer` as the device pointer.
    var _impl: Self._Impl

    # ===-------------------------------------------------------------------===#
    # Aliases
    # ===-------------------------------------------------------------------===#

    comptime _DevicePtr = Pointer[Scalar[Self.dtype], MutUntrackedOrigin]

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @doc_hidden
    @inline(.always)
    def __init__(out self, ctx: DeviceContext, size: Int) raises:
        """This init takes in a constructed `DeviceContext` and schedules an
        owned buffer allocation using the stream in the device context.
        """
        comptime if use_legacy_backend():
            self._impl = Self._Legacy(
                ctx._impl,
                size=size,
            )
        else:
            self._impl = Self._HAL(
                ctx._impl,
                size=size,
            )

    @doc_hidden
    def __init__(
        out self,
        ctx: DeviceContext,
        ptr: Pointer[Scalar[Self.dtype], ...],
        size: Int,
        *,
        owning: Bool,
    ):
        """Constructs a `DeviceBuffer` that wraps an externally-managed device
        pointer.

        Args:
            ctx: The device context the pointer belongs to.
            ptr: The device pointer to wrap.
            size: Number of elements addressable through the pointer.
            owning: Whether releasing this buffer should free the pointer.
        """
        comptime if use_legacy_backend():
            self._impl = Self._Legacy(
                ctx._impl,
                rebind[Self._DevicePtr](ptr),
                size,
                owning=owning,
            )
        else:
            self._impl = Self._HAL(
                ctx._impl,
                ptr,
                size,
                owning=owning,
            )

    def __init__(out self, *, var _impl: Self._Impl):
        """Wraps an already-constructed backend buffer.

        Args:
            _impl: The backend buffer to take ownership of.
        """
        self._impl = _impl^

    # ===-------------------------------------------------------------------===#
    # Factory methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    @doc_hidden
    def empty(context: DeviceContext) -> Self:
        comptime if use_legacy_backend():
            return Self(_impl=Self._Legacy.empty(context._impl))
        else:
            return Self(_impl=Self._HAL.empty(context._impl))

    # ===-------------------------------------------------------------------===#
    # Trait: DevicePassable
    # ===-------------------------------------------------------------------===#

    comptime device_type: AnyType = Self._Impl.device_type
    """DeviceBuffer dtypes are remapped to a device `Pointer` when passed to accelerator devices."""

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        """Device dtype mapping from DeviceBuffer to the device's `Pointer`."""
        comptime if use_legacy_backend():
            self._impl._to_device_type(encoder, target)
        else:
            self._impl._to_device_type(encoder, target)

    @staticmethod
    def get_type_name() -> String:
        """Gets this dtype's name, for use in error messages when handing
        arguments to kernels.

        Returns:
            This dtype's name.
        """
        return String(t"DeviceBuffer[{Self.dtype}]")

    # ===-------------------------------------------------------------------===#
    # Escape hatches
    # ===-------------------------------------------------------------------===#

    def unwrap_legacy(self) -> Self._Legacy:
        """Returns the underlying legacy buffer.

        Only valid in a build whose backend is the legacy one.

        Returns:
            The legacy buffer this wrapper holds.
        """
        comptime assert (
            use_legacy_backend()
        ), "unwrap_legacy is only available in a legacy DeviceContext build"
        return self._impl

    def unwrap_hal(self) -> Self._HAL:
        """Returns the underlying HAL buffer.

        Only valid in a build whose backend is the HAL one.

        Returns:
            The HAL buffer this wrapper holds.
        """
        comptime assert (
            not use_legacy_backend()
        ), "unwrap_hal is only available in a HAL DeviceContext build"
        return self._impl

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    def __len__(self) -> Int:
        """Returns the number of elements in this buffer.

        This method calculates the number of elements by dividing the total byte size
        of the buffer by the size of each element.

        Returns:
            The number of elements in the buffer.
        """
        comptime if use_legacy_backend():
            return len(self._impl)
        else:
            return len(self._impl)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @inline(.always)
    def create_sub_buffer[
        view_type: DType
    ](self, offset: Int, size: Int) raises -> DeviceBuffer[view_type]:
        """Creates a sub-buffer view of this buffer with a different element dtype.

        This method creates a new buffer that references a subset of the memory in this
        buffer, potentially with a different element dtype. The sub-buffer shares the
        underlying memory with the original buffer.

        Parameters:
            view_type: Element dtype of the sub-buffer.

        Args:
            offset: Offset in bytes from the start of this buffer.
            size: Size of the sub-buffer in elements of `view_type`.

        Returns:
            A device buffer viewing the requested range.

        Raises:
            If the range is invalid.
        """
        comptime if use_legacy_backend():
            return DeviceBuffer[view_type](
                _impl=self._impl.create_sub_buffer[view_type](offset, size)
            )
        else:
            return DeviceBuffer[view_type](
                _impl=self._impl.create_sub_buffer[view_type](offset, size)
            )

    def unsafe_ptr[
        mut: Bool, //, origin: Origin[mut=mut]
    ](ref[origin] self) -> Pointer[Scalar[Self.dtype], origin]:
        """Returns the raw device pointer without transferring ownership.

        This method provides direct access to the underlying device pointer
        for advanced use cases. The buffer retains ownership of the pointer.

        Parameters:
            mut: Whether the returned pointer is mutable.
            origin: The origin of the buffer reference.

        Returns:
            The raw device pointer owned by this buffer.
        """
        comptime if use_legacy_backend():
            return rebind[Pointer[Scalar[Self.dtype], origin]](
                self._impl.unsafe_ptr()
            )
        else:
            # The HAL reports an untracked origin; rebuild it against the
            # caller's borrow so the result behaves like the legacy pointer.
            return Pointer[Scalar[Self.dtype], origin](
                unsafe_from_address=Int(self._impl.unsafe_ptr())
            )

    def unsafe_host_ptr[
        mut: Bool, //, origin: Origin[mut=mut]
    ](ref[origin] self) raises -> Pointer[Scalar[Self.dtype], origin]:
        """Returns the host-accessible pointer for this device buffer.

        Only unified-memory devices expose one; other devices raise.

        Parameters:
            mut: Whether the returned pointer is mutable.
            origin: The origin of the buffer reference.

        Returns:
            A pointer to the same memory, addressable from the host.

        Raises:
            If this device's memory is not host-accessible.
        """
        comptime assert (
            use_legacy_backend()
        ), "unsafe_host_ptr is only available on the legacy DeviceContext"
        return rebind[Pointer[Scalar[Self.dtype], origin]](
            self._impl.unsafe_host_ptr()
        )

    def context(self) raises -> DeviceContext[]:
        """Returns the device context associated with this buffer.

        Returns:
            The device context that owns this buffer.

        Raises:
            If the query fails.
        """
        comptime if use_legacy_backend():
            return DeviceContext[](_impl=self._impl.context())
        else:
            return DeviceContext[](_impl=self._impl.context())

    @inline(.always)
    def take_ptr(var self) -> Self._DevicePtr:
        """Takes ownership of the device pointer from this buffer.

        This method releases the device pointer from the buffer's control and
        returns it to the caller. After this call, the buffer no longer owns
        the pointer, and the caller is responsible for managing its lifecycle.

        Returns:
            The raw device pointer that was owned by this buffer.
        """
        comptime if use_legacy_backend():
            return self._impl.take_ptr()
        else:
            return self._impl.take_ptr()

    def enqueue_copy_to(self, dst: DeviceBuffer[Self.dtype]) raises:
        """Enqueues an asynchronous copy to another device buffer.

        Args:
            dst: Device buffer to copy to.

        Raises:
            If the operation fails.
        """
        comptime if use_legacy_backend():
            self._impl.enqueue_copy_to(dst._impl)
        else:
            # The HAL host buffer copies through spans; it has no direct
            # host-to-host overload.
            self._impl.enqueue_copy_to(dst._impl)

    def enqueue_copy_to(self, dst: HostBuffer[Self.dtype]) raises:
        """Enqueues an asynchronous copy to host memory.

        Args:
            dst: Host buffer to copy to.

        Raises:
            If the operation fails.
        """
        comptime if use_legacy_backend():
            self._impl.enqueue_copy_to(dst._impl)
        else:
            self._impl.enqueue_copy_to(dst._impl)

    def enqueue_copy_to(
        self, dst_ptr: Pointer[mut=True, Scalar[Self.dtype], _]
    ) raises:
        """Enqueues an asynchronous copy from this buffer to host memory.

        Args:
            dst_ptr: Pointer to the destination host memory location.

        Raises:
            If the operation fails.
        """
        self.context().enqueue_copy(dst_ptr, self)

    def enqueue_copy_from(self, src: DeviceBuffer[Self.dtype]) raises:
        """Enqueues an asynchronous copy from another device buffer.

        Args:
            src: Device buffer to copy from.

        Raises:
            If the operation fails.
        """
        comptime if use_legacy_backend():
            self._impl.enqueue_copy_from(src._impl)
        else:
            # The HAL host buffer copies through spans; it has no direct
            # host-to-host overload.
            self._impl.enqueue_copy_from(src._impl)

    def enqueue_copy_from(self, src: HostBuffer[Self.dtype]) raises:
        """Enqueues an asynchronous copy from host memory.

        Args:
            src: Host buffer to copy from.

        Raises:
            If the operation fails.
        """
        comptime if use_legacy_backend():
            self._impl.enqueue_copy_from(src._impl)
        else:
            self._impl.enqueue_copy_from(src._impl)

    def enqueue_copy_from(
        self, src_ptr: Pointer[mut=False, Scalar[Self.dtype], _]
    ) raises:
        """Enqueues an asynchronous copy from host memory.

        Args:
            src_ptr: Pointer to the source host memory.

        Raises:
            If the operation fails.
        """
        comptime if use_legacy_backend():
            self._impl.enqueue_copy_from(src_ptr)
        else:
            self._impl.enqueue_copy_from(src_ptr)

    def enqueue_copy_from(
        self, src: Span[mut=False, Scalar[Self.dtype], _]
    ) raises:
        """Enqueues an asynchronous copy from a host span.

        Args:
            src: Host span to copy from.

        Raises:
            If the operation fails.
        """
        comptime if use_legacy_backend():
            self._impl.enqueue_copy_from(src)
        else:
            self._impl.enqueue_copy_from(src)

    def enqueue_copy_to(
        self, dst: Span[mut=True, Scalar[Self.dtype], _]
    ) raises:
        """Enqueues an asynchronous copy to a host span.

        Args:
            dst: Host span to copy to.

        Raises:
            If the operation fails.
        """
        comptime if use_legacy_backend():
            self._impl.enqueue_copy_to(dst)
        else:
            self._impl.enqueue_copy_to(dst)

    def enqueue_fill(self, val: Scalar[Self.dtype]) raises:
        """Enqueues an operation to fill this buffer with a specified value.

        Args:
            val: The value to fill the buffer with.

        Raises:
            If the operation fails.
        """
        comptime if use_legacy_backend():
            self._impl.enqueue_fill(val)
        else:
            self._impl.enqueue_fill(val)

    comptime _Handle = (
        _LegacyDeviceBufferPtr[
            mut=True
        ] if use_legacy_backend() else _HALDeviceBufferPtr[mut=True]
    )

    @doc_hidden
    def __init__(
        out self,
        handle: Self._Handle,
        device_ptr: Self._DevicePtr,
    ):
        """Wraps a native buffer handle and its device pointer.

        Args:
            handle: The native handle of the buffer.
            device_ptr: The device pointer the buffer holds.
        """
        comptime assert (
            use_legacy_backend()
        ), "this constructor is only available on the legacy DeviceContext"
        self._impl = Self._Legacy(handle, device_ptr)

    def device_ptr(
        ref self,
    ) raises -> DevicePointer[Self.dtype, origin_of(self)]:
        """Returns a pointer to this buffer's device memory.

        Returns:
            A device pointer that borrows from this buffer.

        Raises:
            If the pointer cannot be created.
        """
        comptime if use_legacy_backend():
            return rebind_var[DevicePointer[Self.dtype, origin_of(self)]](
                self._impl.device_ptr()
            )
        else:
            return rebind_var[DevicePointer[Self.dtype, origin_of(self)]](
                self._impl.device_ptr()
            )

    def take_handle(deinit self) -> Self._Handle:
        """Transfers the owning native handle out without releasing it.

        Unlike `take_ptr()`, the buffer's destructor does not run, so the
        caller adopts the reference this buffer held.

        Returns:
            The native handle of the buffer.
        """
        comptime if use_legacy_backend():
            return self._impl^.take_handle()
        else:
            return self._impl^.take_handle()

    def _tensor_map_encode_tiled(
        self,
        tensor_map: MutOpaquePointer[_],
        data_type: Int32,
        rank: Int32,
        global_dim: Pointer[mut=False, Int64, _],
        global_strides: Pointer[mut=False, Int64, _],
        box_dim: Pointer[mut=False, Int32, _],
        element_strides: Pointer[mut=False, Int32, _],
        interleave: Int32,
        swizzle: Int32,
        l2_promotion: Int32,
        oob_fill: Int32,
    ) raises:
        """Encodes a tiled TMA descriptor for this buffer."""
        comptime if use_legacy_backend():
            self._impl._tensor_map_encode_tiled(
                tensor_map,
                data_type,
                rank,
                global_dim,
                global_strides,
                box_dim,
                element_strides,
                interleave,
                swizzle,
                l2_promotion,
                oob_fill,
            )
        else:
            self._impl._tensor_map_encode_tiled(
                tensor_map,
                data_type,
                rank,
                global_dim,
                global_strides,
                box_dim,
                element_strides,
                interleave,
                swizzle,
                l2_promotion,
                oob_fill,
            )

    def _tensor_map_encode_im2col(
        self,
        tensor_map: MutOpaquePointer[_],
        data_type: Int32,
        rank: Int32,
        global_dim: Pointer[mut=False, Int64, _],
        global_strides: Pointer[mut=False, Int64, _],
        pixel_box_lower_corner: Pointer[mut=False, Int32, _],
        pixel_box_upper_corner: Pointer[mut=False, Int32, _],
        channels_per_pixel: Int32,
        pixels_per_column: Int32,
        element_strides: Pointer[mut=False, Int32, _],
        interleave: Int32,
        swizzle: Int32,
        l2_promotion: Int32,
        oob_fill: Int32,
    ) raises:
        """Encodes an im2col TMA descriptor for this buffer."""
        comptime if use_legacy_backend():
            self._impl._tensor_map_encode_im2col(
                tensor_map,
                data_type,
                rank,
                global_dim,
                global_strides,
                pixel_box_lower_corner,
                pixel_box_upper_corner,
                channels_per_pixel,
                pixels_per_column,
                element_strides,
                interleave,
                swizzle,
                l2_promotion,
                oob_fill,
            )
        else:
            self._impl._tensor_map_encode_im2col(
                tensor_map,
                data_type,
                rank,
                global_dim,
                global_strides,
                pixel_box_lower_corner,
                pixel_box_upper_corner,
                channels_per_pixel,
                pixels_per_column,
                element_strides,
                interleave,
                swizzle,
                l2_promotion,
                oob_fill,
            )

    def map_to_host(
        self,
        out mapped_buffer: _HostMappedBuffer[Self.dtype],
    ) raises:
        """Maps this device buffer to host memory for CPU access.

        This method creates a host-accessible view of the device buffer's
        contents. The mapping operation may involve copying data from device to
        host memory.

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
        from max.gpu.host import DeviceContext

        var ctx = DeviceContext()
        var length = 1024
        var in_dev = ctx.enqueue_create_buffer[.float32](length)
        var out_dev = ctx.enqueue_create_buffer[.float32](length)

        # Initialize the input and output with known values.
        with in_dev.map_to_host() as in_host, out_dev.map_to_host() as out_host:
            for i in range(length):
                in_host[i] = Float32(i)
                out_host[i] = 255
        ```
        """
        comptime assert not is_gpu(), "DeviceBuffer is not supported on GPUs"
        mapped_buffer = _HostMappedBuffer[Self.dtype](self.context(), self)

    def write_to(self, mut writer: Some[Writer]):
        """Writes a string representation of this buffer to the provided writer.

        Only the legacy backend prints buffer contents.

        Args:
            writer: The writer to output the formatted string to.
        """
        comptime assert (
            use_legacy_backend()
        ), "write_to is only available on the legacy DeviceContext"
        self._impl.write_to(writer)


struct _HostMappedBuffer[dtype: DType]:
    """Copies a device buffer to the host on entry and back on exit.

    Written against the wrapper types, so both backends share it.
    """

    var _ctx: DeviceContext[]
    var _dev_buf: DeviceBuffer[Self.dtype]
    var _cpu_buf: HostBuffer[Self.dtype]

    def __init__(
        out self,
        ctx: DeviceContext[],
        buf: DeviceBuffer[Self.dtype],
    ) raises:
        var cpu_buf = ctx.enqueue_create_host_buffer[Self.dtype](len(buf))
        self._ctx = ctx
        self._dev_buf = buf
        self._cpu_buf = cpu_buf

    def __deinit__(deinit self):
        pass

    def __enter__(mut self) raises -> HostBuffer[Self.dtype]:
        self._dev_buf.enqueue_copy_to(self._cpu_buf)
        self._ctx.synchronize()
        return self._cpu_buf

    def __exit__(mut self) raises:
        self._ctx.synchronize()
        self._cpu_buf.enqueue_copy_to(self._dev_buf)
        self._ctx.synchronize()
