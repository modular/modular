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
from .device_context import use_legacy_backend
from max.gpu.host._legacy.device_context import LegacyDevicePointer
from max.gpu.host.info import default_accelerator


@fieldwise_init
struct _HALDevicePointer[
    mut: Bool,
    //,
    dtype: DType,
    origin: Origin[mut=mut],
](ImplicitlyCopyable, TrivialRegisterPassable):
    """The state of a `DevicePointer` on the HAL backend.

    Parameters:
        mut: Whether the borrow of the underlying `DeviceBuffer` is mutable
            (inferred from `origin`).
        dtype: Data dtype to be stored in the pointer.
        origin: The origin of the borrowed `DeviceBuffer`.
    """

    var _buffer: Pointer[DeviceBuffer[Self.dtype], Self.origin]
    var _offset: Int
    var _size: Int


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
    """A host-side representation of a pointer to device memory that resides
    within a `DeviceBuffer`.

    A `DevicePointer` is a non-owning borrow of a `DeviceBuffer`; it must not
    outlive the buffer it points into.

    - Supports pointer arithmetic which may result in a new `DevicePointer`
      instance referring to the same `DeviceBuffer` with a new offset.
    - Supports equality comparison and ordering of `DevicePointer`s pointing
      into the same `DeviceBuffer`.
    - May support accessing device pointer address on supported hardware.
    - Does not support load/store operations.

    At the device function execution boundary a `DevicePointer` is transformed
    into an `Pointer` on the device at the point of being handed over to
    the device driver.

    The wrapper holds the backend's pointer, so each backend keeps its own
    notion of buffer identity for comparisons.

    Parameters:
        mut: Whether the borrow of the underlying `DeviceBuffer` is mutable
            (inferred from `origin`).
        dtype: Data dtype to be stored in the pointer.
        origin: The origin of the borrowed `DeviceBuffer`.
    """

    comptime _Legacy = LegacyDevicePointer[Self.dtype, Self.origin]
    comptime _HAL = _HALDevicePointer[Self.dtype, Self.origin]
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

    def __init__(
        out self, ref[Self.origin] buffer: DeviceBuffer[Self.dtype]
    ) raises:
        """Constructs a `DevicePointer` referencing the start of `buffer`.

        Args:
            buffer: The `DeviceBuffer` this pointer references. Must outlive
                the resulting `DevicePointer`.

        Raises:
            If `buffer` has size 0.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            self._impl = Self._Legacy(buffer._legacy_buffer())
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        var size = len(buffer)
        if size == 0:
            raise Error("DevicePointer: size of DeviceBuffer must not be 0")
        self._impl = Self._HAL(Pointer(to=buffer), 0, size)

    def __init__(
        out self,
        ref[Self.origin] buffer: DeviceBuffer[Self.dtype],
        offset: Int,
    ) raises:
        """Constructs a `DevicePointer` into `buffer` at `offset`.

        Args:
            buffer: The `DeviceBuffer` this pointer references. Must outlive
                the resulting `DevicePointer`.
            offset: Element offset from the start of `buffer`.

        Raises:
            If `buffer` has size 0, or if `offset` is outside the half-open
            range `[0, len(buffer))`.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            self._impl = Self._Legacy(buffer._legacy_buffer(), offset)
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        var size = len(buffer)
        if size == 0:
            raise Error("DevicePointer: invalid DeviceBuffer of size '0'")
        if offset < 0 or offset >= size:
            raise Error(
                t"DevicePointer: invalid offset '{offset}' for DeviceBuffer of"
                t" size '{size}'"
            )
        self._impl = Self._HAL(Pointer(to=buffer), offset, size)

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
        """Returns a reference to the `DeviceBuffer` this pointer references.

        The reference is non-owning; the underlying `DeviceBuffer` must
        outlive `self`.

        Returns:
            A reference to the referenced `DeviceBuffer`.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            return rebind[Pointer[DeviceBuffer[Self.dtype], Self.origin]](
                Pointer(to=self._impl.buffer())
            )[]
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        return self._impl._buffer[]

    def offset(self) -> Int:
        """Returns the element offset from the start of the owning buffer.

        Returns:
            The element offset.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            return self._impl.offset()
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        return self._impl._offset

    @doc_hidden
    def unsafe_ptr(
        ref self,
    ) -> Pointer[Scalar[Self.dtype], MutAnyOrigin]:
        """Returns the raw device pointer, if supported by the target.

        On targets that expose raw device pointers (for example CUDA and HIP),
        this returns the underlying address adjusted by the current offset.
        On targets that do not (for example Metal), this raises an error.

        Returns:
            The raw device pointer.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            return self._impl.unsafe_ptr()
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        # TODO: GEX-3693: Assert/raise when target doesn't support raw device
        # pointer access
        # `DeviceBuffer.unsafe_ptr()` now ties its mutability to the borrow of
        # the buffer; force mutable to preserve this helper's `MutAnyOrigin`
        # contract.
        return (
            self._impl._buffer[]
            ._impl._device_ptr.unsafe_offset(self._impl._offset)
            .as_unsafe_any_origin()
        )

    def _buffer_handle(
        ref self,
    ) -> Optional[OpaquePointer[MutUntrackedOrigin]]:
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            return self._impl._buffer_handle()
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        return None

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
        """Returns a new `DevicePointer` offset forward by `n` elements.

        Args:
            n: Number of elements to offset by.

        Returns:
            A new `DevicePointer` referencing the same `DeviceBuffer` at the
            new offset.

        Raises:
            If the resulting offset is outside the bounds of the owning
            `DeviceBuffer`.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            return Self(_impl=self._impl + n)
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        var offset = self._impl._offset + n
        if offset < 0 or offset >= self._impl._size:
            raise Error(
                t"DevicePointer: addition of '{n}' results in invalid offset of"
                t" '{offset}' for DeviceBuffer of size {self._impl._size}"
            )
        return Self(
            _impl=Self._HAL(self._impl._buffer, offset, self._impl._size)
        )

    def __sub__(self, n: Int) raises -> Self:
        """Returns a new `DevicePointer` offset backward by `n` elements.

        Args:
            n: Number of elements to offset by.

        Returns:
            A new `DevicePointer` referencing the same `DeviceBuffer` at the
            new offset.

        Raises:
            If the resulting offset is outside the bounds of the owning
            `DeviceBuffer`.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            return Self(_impl=self._impl - n)
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        var offset = self._impl._offset - n
        if offset < 0 or offset >= self._impl._size:
            raise Error(
                t"DevicePointer: subtraction of '{n}' results in invalid offset"
                t" of '{offset}' for DeviceBuffer of size {self._impl._size}"
            )
        return Self(
            _impl=Self._HAL(self._impl._buffer, offset, self._impl._size)
        )

    def __iadd__(mut self, n: Int) raises:
        """Offsets this `DevicePointer` forward by `n` elements in place.

        Args:
            n: Number of elements to offset by.

        Raises:
            If the resulting offset is outside the bounds of the owning
            `DeviceBuffer`.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            var impl = self._impl
            impl.__iadd__(n)
            self._impl = impl
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        var offset = self._impl._offset + n
        if offset < 0 or offset >= self._impl._size:
            raise Error(
                t"DevicePointer: addition of '{n}' results in invalid offset of"
                t" '{offset}' for DeviceBuffer of size {self._impl._size}"
            )
        self._impl._offset = offset

    def __isub__(mut self, n: Int) raises:
        """Offsets this `DevicePointer` backward by `n` elements in place.

        Args:
            n: Number of elements to offset by.

        Raises:
            If the resulting offset is outside the bounds of the owning
            `DeviceBuffer`.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            var impl = self._impl
            impl.__isub__(n)
            self._impl = impl
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        var offset = self._impl._offset - n
        if offset < 0 or offset >= self._impl._size:
            raise Error(
                t"DevicePointer: subtraction of '{n}' results in invalid offset"
                t" of '{offset}' for DeviceBuffer of size {self._impl._size}"
            )
        self._impl._offset = offset

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
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            var lhs = self._impl
            var rhs = other._impl
            return lhs == rhs
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        # Buffer identity = base device address (HAL buffers have no C++
        # handle).
        return (
            self._impl._buffer[]._impl._device_ptr
            == other._impl._buffer[]._impl._device_ptr
            and self._impl._offset == other._impl._offset
        )

    @__unsafe_nested_origins_read_only
    def __eq__(self, other: DevicePointer[Self.dtype, _]) -> Bool:
        """Returns whether both pointers refer to the same element.

        Args:
            other: The pointer to compare with.

        Returns:
            Whether the pointers are equal.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            var lhs = self._impl
            var rhs = other._impl
            return lhs == rhs
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        # Buffer identity = base device address (HAL buffers have no C++
        # handle).
        return (
            self._impl._buffer[]._impl._device_ptr
            == other._impl._buffer[]._impl._device_ptr
            and self._impl._offset == other._impl._offset
        )

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
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            var lhs = self._impl
            var rhs = other._impl
            return lhs < rhs
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        if (
            self._impl._buffer[]._impl._device_ptr
            != other._impl._buffer[]._impl._device_ptr
        ):
            raise Error(
                "DevicePointer: less than comparison not supported when the"
                " underlying DeviceBuffer does not match"
            )
        return self._impl._offset < other._impl._offset

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
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            var lhs = self._impl
            var rhs = other._impl
            return lhs <= rhs
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        if (
            self._impl._buffer[]._impl._device_ptr
            != other._impl._buffer[]._impl._device_ptr
        ):
            raise Error(
                "DevicePointer: less than or equal comparison not supported"
                " when the underlying DeviceBuffer does not match"
            )
        return self._impl._offset <= other._impl._offset

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
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            var lhs = self._impl
            var rhs = other._impl
            return lhs > rhs
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        if (
            self._impl._buffer[]._impl._device_ptr
            != other._impl._buffer[]._impl._device_ptr
        ):
            raise Error(
                "DevicePointer: greater than comparison not supported when the"
                " underlying DeviceBuffer does not match"
            )
        return self._impl._offset > other._impl._offset

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
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            var lhs = self._impl
            var rhs = other._impl
            return lhs >= rhs
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        if (
            self._impl._buffer[]._impl._device_ptr
            != other._impl._buffer[]._impl._device_ptr
        ):
            raise Error(
                "DevicePointer: greater than or equal comparison not supported"
                " when the underlying DeviceBuffer does not match"
            )
        return self._impl._offset >= other._impl._offset

    # ===-------------------------------------------------------------------===#
    # Writable and DevicePassable
    # ===-------------------------------------------------------------------===#

    def write_to(self, mut writer: Some[Writer]):
        """Writes a description of this pointer to `writer`.

        Args:
            writer: The writer to write to.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            self._impl.write_to(writer)
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        writer.write(
            t"DevicePointer[{Self.dtype}]("
            t"buffer=DeviceBuffer(size={len(self._impl._buffer[])}), "
            t"offset={self._impl._offset})"
        )

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
