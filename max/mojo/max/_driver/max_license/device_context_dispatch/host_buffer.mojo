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
"""Dispatching `HostBuffer`."""

from std.format import Writer
from std.memory import Pointer

from max._driver.max_license.device_context_hal.device_context_hal import (
    HALHostBuffer,
)
from max.gpu.host._legacy.device_context import LegacyHostBuffer
from .device_context import use_legacy_backend
from max.gpu.host.info import default_accelerator
from .host_types import Device
from max._driver.max_license.device_context_dispatch.device_context import (
    DeviceContext,
)
from max._driver.max_license.device_context_dispatch.device_buffer import (
    DeviceBuffer,
)


struct HostBuffer[dtype: DType](ImplicitlyCopyable, Sized, Writable):
    """Represents a block of host-resident storage. For GPU devices, a host
    buffer is allocated in the host's global memory.

    To allocate a `HostBuffer`, use one of the methods provided by
    `DeviceContext`, such as
    [`enqueue_create_host_buffer()`](/api/mojo/max/gpu/host/device_context/DeviceContext/#enqueue_create_host_buffer).

    Parameters:
        dtype: Data type to be stored in the buffer.
    """

    # ===-------------------------------------------------------------------===#
    # Backend selection
    # ===-------------------------------------------------------------------===#

    comptime _Legacy = LegacyHostBuffer[Self.dtype]
    comptime _HAL = HALHostBuffer[Self.dtype]
    comptime _Impl = Self._Legacy if use_legacy_backend() else Self._HAL

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var _impl: Self._Impl

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @doc_hidden
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
        host_ptr: Self._HostPtr,
        size: Int,
        *,
        owning: Bool,
    ):
        """Constructs a `HostBuffer` that wraps an externally-managed pointer.

        Args:
            ctx: The device context the pointer belongs to.
            host_ptr: The host pointer to wrap.
            size: Number of elements addressable through the pointer.
            owning: Whether releasing this buffer should free the pointer.
        """
        comptime assert use_legacy_backend(), (
            "wrapping an external host pointer is only available on the"
            " legacy DeviceContext"
        )
        self._impl = Self._Legacy(
            ctx._impl,
            host_ptr,
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
    # Aliases
    # ===-------------------------------------------------------------------===#

    comptime _HostPtr = Pointer[Scalar[Self.dtype], MutUntrackedOrigin]

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
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @inline(.always)
    def __getitem__(self, idx: Int) -> Scalar[Self.dtype]:
        """Retrieves the element at the specified index from the host buffer.

        Args:
            idx: The index of the element to retrieve.

        Returns:
            The scalar value at the specified index.
        """
        comptime if use_legacy_backend():
            return self._impl[idx]
        else:
            return self._impl[idx]

    @inline(.always)
    def __setitem__(self, idx: Int, val: Scalar[Self.dtype]):
        """Sets the element at the specified index in the host buffer.

        Args:
            idx: The index of the element to set.
            val: The value to store.
        """
        comptime if use_legacy_backend():
            self._impl[idx] = val
        else:
            self._impl[idx] = val

    def __len__(self) -> Int:
        """Returns the number of elements in this buffer.

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

    def unsafe_ptr(self) -> Self._HostPtr:
        """Returns the raw host pointer without transferring ownership.

        Returns:
            The raw host pointer owned by this buffer.
        """
        comptime if use_legacy_backend():
            return rebind[Self._HostPtr](self._impl.unsafe_ptr())
        else:
            return rebind[Self._HostPtr](self._impl.unsafe_ptr())

    @__unsafe_nested_origins_read_only
    def as_span[
        origin: Origin, //
    ](ref[origin] self) -> Span[
        Scalar[Self.dtype], origin_of(self)._get_owned_interior["buffer"]
    ]:
        """Returns a `Span` pointing to the underlying memory of the `HostBuffer`.

        Parameters:
            origin: The origin of the buffer reference.

        Returns:
            A `Span` over the buffer's memory. The span carries an interior
            origin derived from `self`, so any subsequent mutation of the
            `HostBuffer` invalidates it at compile time.
        """
        # Built from the buffer's own pointer rather than forwarded: the
        # interior origin belongs to this wrapper, so neither backend's span
        # carries it.
        return {
            unsafe_ptr = Pointer[
                Scalar[Self.dtype],
                origin_of(self)._get_owned_interior["buffer"],
            ](
                to=self.unsafe_ptr()._get_ref_with_unsafe_interior_origin[
                    "buffer", origin_of(self)
                ]()
            ),
            length = len(self),
        }

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

    def create_sub_buffer[
        view_type: DType
    ](self, offset: Int, size: Int) raises -> HostBuffer[view_type]:
        """Creates a sub-buffer view of this buffer with a different element dtype.

        Parameters:
            view_type: Element dtype of the sub-buffer.

        Args:
            offset: Offset in bytes from the start of this buffer.
            size: Size of the sub-buffer in elements of `view_type`.

        Returns:
            A host buffer viewing the requested range.

        Raises:
            If the range is invalid.
        """
        comptime if use_legacy_backend():
            return HostBuffer[view_type](
                _impl=self._impl.create_sub_buffer[view_type](offset, size)
            )
        else:
            return HostBuffer[view_type](
                _impl=self._impl.create_sub_buffer[view_type](offset, size)
            )

    def enqueue_copy_to(self, dst: HostBuffer[Self.dtype]) raises:
        """Enqueues an asynchronous copy to another host buffer.

        Args:
            dst: Host buffer to copy to.

        Raises:
            If the operation fails.
        """
        comptime if use_legacy_backend():
            self._impl.enqueue_copy_to(dst._impl)
        else:
            # The HAL host buffer copies through spans; it has no direct
            # host-to-host overload.
            dst._impl.enqueue_copy_from(self._impl.as_span())

    def enqueue_copy_to(self, dst: DeviceBuffer[Self.dtype]) raises:
        """Enqueues an asynchronous copy to a device buffer.

        Args:
            dst: Device buffer to copy to.

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
        """Enqueues an asynchronous copy to host memory.

        Args:
            dst_ptr: Pointer to the destination host memory.

        Raises:
            If the operation fails.
        """
        comptime assert use_legacy_backend(), (
            "enqueue_copy_to a raw pointer is only available on the legacy"
            " DeviceContext"
        )
        self._impl.enqueue_copy_to(dst_ptr)

    def enqueue_copy_from(self, src: HostBuffer[Self.dtype]) raises:
        """Enqueues an asynchronous copy from another host buffer.

        Args:
            src: Host buffer to copy from.

        Raises:
            If the operation fails.
        """
        comptime if use_legacy_backend():
            self._impl.enqueue_copy_from(src._impl)
        else:
            # The HAL host buffer copies through spans; it has no direct
            # host-to-host overload.
            self._impl.enqueue_copy_from(src._impl.as_span())

    def enqueue_copy_from(self, src: DeviceBuffer[Self.dtype]) raises:
        """Enqueues an asynchronous copy from a device buffer.

        Args:
            src: Device buffer to copy from.

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
        comptime assert use_legacy_backend(), (
            "enqueue_copy_from a raw pointer is only available on the legacy"
            " DeviceContext"
        )
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
        comptime assert (
            use_legacy_backend()
        ), "enqueue_fill is only available on the legacy DeviceContext"
        self._impl.enqueue_fill(val)

    def take_ptr(deinit self) -> Self._HostPtr:
        """Takes ownership of the host pointer from this buffer.

        Returns:
            The raw host pointer that was owned by this buffer.
        """
        comptime assert (
            use_legacy_backend()
        ), "take_ptr is only available on the legacy DeviceContext"
        return rebind[Self._HostPtr](self._impl.take_ptr())

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
