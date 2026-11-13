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
"""Backend-dispatching wrapper for `DevicePointer`.

A `DevicePointer` borrows a `DeviceBuffer` and an offset into it. The wrapper
holds the backend's pointer and forwards to it, so each backend keeps its own
notion of buffer identity for comparisons.
"""

from std.builtin.device_passable import (
    DevicePassable,
    DevicePointerLike,
    DeviceTypeEncoder,
)
from std.format import Writer
from std.sys.info import CompilationTarget

from max._driver.max_license.device_context_dispatch.device_buffer import (
    DeviceBuffer,
)
from max._driver.max_license.device_context_hal.device_context_hal import (
    HALDevicePointer,
)
from .device_context import use_legacy_backend
from max.gpu.host._legacy.device_context import LegacyDevicePointer
from max.gpu.host.info import default_accelerator


struct DevicePointer[
    mut: Bool,
    //,
    dtype: DType,
    origin: Origin[mut=mut],
](
    DevicePassable,
    DevicePointerLike,
    Equatable,
    ImplicitlyCopyable,
    TrivialRegisterPassable,
    Writable,
):
    """A pointer into a device buffer.

    Parameters:
        mut: Whether the borrowed buffer is mutable.
        dtype: The element type of the buffer.
        origin: The origin of the borrowed buffer.
    """

    comptime _Legacy = LegacyDevicePointer[Self.dtype, Self.origin]
    comptime _HAL = HALDevicePointer[Self.dtype, Self.origin]
    comptime _Impl = Self._Legacy if use_legacy_backend() else Self._HAL

    var _impl: Self._Impl

    comptime PointeeType: AnyType = Scalar[Self.dtype]
    """The type this pointer points to."""

    comptime device_type: AnyType = Pointer[
        mut=True, Scalar[Self.dtype], AnyOrigin[mut=True]
    ]
    """The type this pointer has inside a kernel."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    def _legacy_buffer(
        ref[Self.origin] buffer: DeviceBuffer[Self.dtype],
    ) -> ref[Self.origin] DeviceBuffer[Self.dtype]._Legacy:
        return rebind[Pointer[DeviceBuffer[Self.dtype]._Legacy, Self.origin]](
            Pointer(to=buffer._impl)
        )[]

    @staticmethod
    def _hal_buffer(
        ref[Self.origin] buffer: DeviceBuffer[Self.dtype],
    ) -> ref[Self.origin] DeviceBuffer[Self.dtype]._HAL:
        return rebind[Pointer[DeviceBuffer[Self.dtype]._HAL, Self.origin]](
            Pointer(to=buffer._impl)
        )[]

    def __init__(
        out self, ref[Self.origin] buffer: DeviceBuffer[Self.dtype]
    ) raises:
        """Points at the start of `buffer`.

        Args:
            buffer: The buffer to point into.

        Raises:
            If `buffer` is empty.
        """
        comptime if use_legacy_backend():
            self._impl = rebind_var[Self._Impl](
                Self._Legacy(Self._legacy_buffer(buffer))
            )
        else:
            self._impl = rebind_var[Self._Impl](
                Self._HAL(Self._hal_buffer(buffer))
            )

    def __init__(
        out self,
        ref[Self.origin] buffer: DeviceBuffer[Self.dtype],
        offset: Int,
    ) raises:
        """Points at `offset` elements into `buffer`.

        Args:
            buffer: The buffer to point into.
            offset: The element offset into the buffer.

        Raises:
            If `buffer` is empty or `offset` is out of range.
        """
        comptime if use_legacy_backend():
            self._impl = rebind_var[Self._Impl](
                Self._Legacy(Self._legacy_buffer(buffer), offset)
            )
        else:
            self._impl = rebind_var[Self._Impl](
                Self._HAL(Self._hal_buffer(buffer), offset)
            )

    @doc_hidden
    def __init__(out self, *, var _impl: Self._Impl):
        """Wraps an already-constructed backend pointer.

        Args:
            _impl: The backend pointer to wrap.
        """
        self._impl = _impl

    # ===-------------------------------------------------------------------===#
    # Accessors
    # ===-------------------------------------------------------------------===#

    def buffer(self) -> ref[Self.origin] DeviceBuffer[Self.dtype]:
        """Returns the buffer this pointer points into.

        Returns:
            A reference to the buffer.
        """
        comptime if use_legacy_backend():
            return rebind[Pointer[DeviceBuffer[Self.dtype], Self.origin]](
                Pointer(to=rebind[Self._Legacy](self._impl).buffer())
            )[]
        else:
            return rebind[Pointer[DeviceBuffer[Self.dtype], Self.origin]](
                Pointer(to=rebind[Self._HAL](self._impl).buffer())
            )[]

    def offset(self) -> Int:
        """Returns the element offset into the buffer.

        Returns:
            The offset.
        """
        comptime if use_legacy_backend():
            return rebind[Self._Legacy](self._impl).offset()
        else:
            return rebind[Self._HAL](self._impl).offset()

    @doc_hidden
    def unsafe_ptr(
        ref self,
    ) -> Pointer[Scalar[Self.dtype], MutAnyOrigin]:
        """Returns the raw device address this pointer refers to.

        Returns:
            A raw pointer to the element.
        """
        comptime if use_legacy_backend():
            return rebind[Self._Legacy](self._impl).unsafe_ptr()
        else:
            return rebind[Self._HAL](self._impl).unsafe_ptr()

    def _buffer_handle(
        ref self,
    ) -> Optional[OpaquePointer[MutUntrackedOrigin]]:
        comptime if use_legacy_backend():
            return rebind[Self._Legacy](self._impl)._buffer_handle()
        else:
            return rebind[Self._HAL](self._impl)._buffer_handle()

    # ===-------------------------------------------------------------------===#
    # Origin and mutability casts
    # ===-------------------------------------------------------------------===#

    comptime _OriginCastType[
        target_mut: Bool, //, target_origin: Origin[mut=target_mut]
    ] = DevicePointer[Self.dtype, target_origin]

    @always_inline("builtin")
    def unsafe_mut_cast[
        target_mut: Bool
    ](self) -> Self._OriginCastType[Self.origin.unsafe_mut_cast[target_mut]()]:
        """Changes the mutability of the borrowed buffer.

        Parameters:
            target_mut: The mutability to cast to.

        Returns:
            This pointer with the new mutability.
        """
        return rebind[
            Self._OriginCastType[Self.origin.unsafe_mut_cast[target_mut]()]
        ](self)

    @always_inline("builtin")
    def unsafe_origin_cast[
        target_origin: Origin[mut=Self.mut]
    ](self) -> Self._OriginCastType[target_origin]:
        """Changes the origin of the borrowed buffer.

        Parameters:
            target_origin: The origin to cast to.

        Returns:
            This pointer with the new origin.
        """
        return rebind[Self._OriginCastType[target_origin]](self)

    @always_inline("builtin")
    def as_imm(self) -> Self._OriginCastType[ImmOrigin(Self.origin)]:
        """Returns this pointer with an immutable origin.

        Returns:
            This pointer, immutable.
        """
        return self.unsafe_mut_cast[False]()

    @always_inline("builtin")
    def as_unsafe_any_origin(
        self,
    ) -> Self._OriginCastType[UnsafeAnyOrigin[mut=Self.mut]]:
        """Returns this pointer with the any-origin escape hatch.

        Returns:
            This pointer, with its origin erased.
        """
        return self.unsafe_origin_cast[UnsafeAnyOrigin[mut=Self.mut]]()

    # ===-------------------------------------------------------------------===#
    # Pointer arithmetic
    # ===-------------------------------------------------------------------===#

    def __add__(self, n: Int) raises -> Self:
        """Returns this pointer advanced by `n` elements.

        Args:
            n: The number of elements to advance by.

        Returns:
            The advanced pointer.

        Raises:
            If the result is outside the buffer.
        """
        comptime if use_legacy_backend():
            return Self(
                _impl=rebind_var[Self._Impl](
                    rebind[Self._Legacy](self._impl) + n
                )
            )
        else:
            return Self(
                _impl=rebind_var[Self._Impl](rebind[Self._HAL](self._impl) + n)
            )

    def __sub__(self, n: Int) raises -> Self:
        """Returns this pointer moved back by `n` elements.

        Args:
            n: The number of elements to move back by.

        Returns:
            The moved pointer.

        Raises:
            If the result is outside the buffer.
        """
        comptime if use_legacy_backend():
            return Self(
                _impl=rebind_var[Self._Impl](
                    rebind[Self._Legacy](self._impl) - n
                )
            )
        else:
            return Self(
                _impl=rebind_var[Self._Impl](rebind[Self._HAL](self._impl) - n)
            )

    def __iadd__(mut self, n: Int) raises:
        """Advances this pointer by `n` elements.

        Args:
            n: The number of elements to advance by.

        Raises:
            If the result is outside the buffer.
        """
        comptime if use_legacy_backend():
            var impl = rebind[Self._Legacy](self._impl)
            impl.__iadd__(n)
            self._impl = rebind_var[Self._Impl](impl)
        else:
            var impl = rebind[Self._HAL](self._impl)
            impl.__iadd__(n)
            self._impl = rebind_var[Self._Impl](impl)

    def __isub__(mut self, n: Int) raises:
        """Moves this pointer back by `n` elements.

        Args:
            n: The number of elements to move back by.

        Raises:
            If the result is outside the buffer.
        """
        comptime if use_legacy_backend():
            var impl = rebind[Self._Legacy](self._impl)
            impl.__isub__(n)
            self._impl = rebind_var[Self._Impl](impl)
        else:
            var impl = rebind[Self._HAL](self._impl)
            impl.__isub__(n)
            self._impl = rebind_var[Self._Impl](impl)

    # ===-------------------------------------------------------------------===#
    # Comparison
    # ===-------------------------------------------------------------------===#

    @__unsafe_nested_origins_read_only
    def __eq__(self, other: Self) -> Bool:
        """Returns whether both pointers refer to the same element.

        Args:
            other: The pointer to compare with.

        Returns:
            Whether the pointers are equal.
        """
        comptime if use_legacy_backend():
            var lhs = rebind[Self._Legacy](self._impl)
            var rhs = rebind[Self._Legacy](other._impl)
            return lhs == rhs
        else:
            var lhs = rebind[Self._HAL](self._impl)
            var rhs = rebind[Self._HAL](other._impl)
            return lhs == rhs

    @__unsafe_nested_origins_read_only
    def __eq__(self, other: DevicePointer[Self.dtype, _]) -> Bool:
        """Returns whether both pointers refer to the same element.

        Args:
            other: The pointer to compare with.

        Returns:
            Whether the pointers are equal.
        """
        comptime if use_legacy_backend():
            var lhs = rebind[Self._Legacy](self._impl)
            var rhs = rebind[type_of(other)._Legacy](other._impl)
            return lhs == rhs
        else:
            var lhs = rebind[Self._HAL](self._impl)
            var rhs = rebind[type_of(other)._HAL](other._impl)
            return lhs == rhs

    @__unsafe_nested_origins_read_only
    def __ne__(self, other: Self) -> Bool:
        """Returns whether the pointers refer to different elements.

        Args:
            other: The pointer to compare with.

        Returns:
            Whether the pointers differ.
        """
        return not (self == other)

    @__unsafe_nested_origins_read_only
    def __ne__(self, other: DevicePointer[Self.dtype, _]) -> Bool:
        """Returns whether the pointers refer to different elements.

        Args:
            other: The pointer to compare with.

        Returns:
            Whether the pointers differ.
        """
        return not (self == other)

    @__unsafe_nested_origins_read_only
    def __lt__(self, other: DevicePointer[Self.dtype, _]) raises -> Bool:
        """Returns whether this pointer is before `other` in the same buffer.

        Args:
            other: The pointer to compare with.

        Returns:
            Whether this pointer is before `other`.

        Raises:
            If the pointers are into different buffers.
        """
        comptime if use_legacy_backend():
            var lhs = rebind[Self._Legacy](self._impl)
            var rhs = rebind[type_of(other)._Legacy](other._impl)
            return lhs < rhs
        else:
            var lhs = rebind[Self._HAL](self._impl)
            var rhs = rebind[type_of(other)._HAL](other._impl)
            return lhs < rhs

    @__unsafe_nested_origins_read_only
    def __le__(self, other: DevicePointer[Self.dtype, _]) raises -> Bool:
        """Returns whether this pointer is at or before `other`.

        Args:
            other: The pointer to compare with.

        Returns:
            Whether this pointer is at or before `other`.

        Raises:
            If the pointers are into different buffers.
        """
        comptime if use_legacy_backend():
            var lhs = rebind[Self._Legacy](self._impl)
            var rhs = rebind[type_of(other)._Legacy](other._impl)
            return lhs <= rhs
        else:
            var lhs = rebind[Self._HAL](self._impl)
            var rhs = rebind[type_of(other)._HAL](other._impl)
            return lhs <= rhs

    @__unsafe_nested_origins_read_only
    def __gt__(self, other: DevicePointer[Self.dtype, _]) raises -> Bool:
        """Returns whether this pointer is after `other` in the same buffer.

        Args:
            other: The pointer to compare with.

        Returns:
            Whether this pointer is after `other`.

        Raises:
            If the pointers are into different buffers.
        """
        comptime if use_legacy_backend():
            var lhs = rebind[Self._Legacy](self._impl)
            var rhs = rebind[type_of(other)._Legacy](other._impl)
            return lhs > rhs
        else:
            var lhs = rebind[Self._HAL](self._impl)
            var rhs = rebind[type_of(other)._HAL](other._impl)
            return lhs > rhs

    @__unsafe_nested_origins_read_only
    def __ge__(self, other: DevicePointer[Self.dtype, _]) raises -> Bool:
        """Returns whether this pointer is at or after `other`.

        Args:
            other: The pointer to compare with.

        Returns:
            Whether this pointer is at or after `other`.

        Raises:
            If the pointers are into different buffers.
        """
        comptime if use_legacy_backend():
            var lhs = rebind[Self._Legacy](self._impl)
            var rhs = rebind[type_of(other)._Legacy](other._impl)
            return lhs >= rhs
        else:
            var lhs = rebind[Self._HAL](self._impl)
            var rhs = rebind[type_of(other)._HAL](other._impl)
            return lhs >= rhs

    # ===-------------------------------------------------------------------===#
    # Writable and DevicePassable
    # ===-------------------------------------------------------------------===#

    def write_to(self, mut writer: Some[Writer]):
        """Writes a description of this pointer to `writer`.

        Args:
            writer: The writer to write to.
        """
        comptime if use_legacy_backend():
            rebind[Self._Legacy](self._impl).write_to(writer)
        else:
            rebind[Self._HAL](self._impl).write_to(writer)

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_device_ptr(self, target)

    @staticmethod
    def get_type_name() -> String:
        """Returns the name of this type for kernel signatures.

        Returns:
            The type name.
        """
        return String(t"DevicePointer[{Self.dtype}]")
