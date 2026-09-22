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
"""TileTensor type for structured memory access with compile-time layout information."""

from std.bit import log2_floor
from std.math import align_up, ceildiv, nan
from std.math.uutils import umod
from std.sys import align_of, simd_width_of, is_gpu, size_of
from std.os import abort

from std.builtin.builtin_slice import ContiguousSlice
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.builtin.int import index as _index
from std.collections._conditional import _ComptimeConditional
from std.memory import unsafe_stack_allocation as _std_stack_allocation
from std.memory.unsafe_pointer import unsafe_cast
from std.reflection import call_location
from max.gpu.host import DeviceBuffer, DeviceContext, DevicePointer, HostBuffer
from max.gpu.memory import CacheEviction, Fill, async_copy
from layout._fillers import BATCH_SIZE
from layout.layout_tensor import LayoutTensor
from std.sys import prefetch
from std.sys.intrinsics import PrefetchOptions
from std.utils import IndexList, StaticTuple
from std.utils.coord import _coerce_dynamic

from .swizzle import Swizzle

from .tensor_engine import (
    DevicePointerEngine,
    TensorOps,
    TensorEngine,
    DefaultEngine,
)
from .tile_layout import (
    Layout,
    RowMajorLayout,
    TensorLayout,
    WeaklyCompatible,
    _RowMajor,
    row_major,
)
from layout.coord import (
    ComptimeInt,
    Idx,
    Coord,
    CoordLike,
    _IntToComptimeInt,
    coord,
    coord_to_index_list,
    _CoordToDynamic,
    _Multiply,
    _Divide,
    _CeilDiv,
    _Flattened,
    DynamicCoord,
)
from .int_tuple import coord_to_int_tuple, _IntTupleToCoordLike


@inline(.always)
def _default_invariant[mut: Bool]() -> Bool:
    return is_gpu() and mut == False


@inline(.always)
def _async_fill_value[dtype: DType, fill: Fill]() -> Optional[Scalar[dtype]]:
    """Returns the value written to the bytes a masked async copy skips, or
    `None` when `Fill.NONE` asks for those bytes to be left as they are."""
    comptime if fill == Fill.NONE:
        return None
    elif fill == Fill.NAN:
        comptime assert (
            dtype.is_floating_point()
        ), "Fill.NAN requires a floating-point dtype"
        return nan[dtype]()
    else:
        return Scalar[dtype](0)


struct _IndexOrSlice[static: Int = -1](
    EnumLike, ImplicitlyCopyable, _IndexOrSliceLike
):
    """A single `TileTensor.slice` argument: an index that fixes (drops) a
    dimension, or a `ContiguousSlice` that selects a rank-preserving subrange.

    An index is an `Int`, or a `ComptimeInt` whose value the type records in
    `static`, so a `TypeList` of these types can tell the type system which
    axes a subscript fixes to a compile-time position. The runtime `Int` and
    slice cases leave `static` at its `-1` default.

    This thin wrapper over the `Variant` adds the slice-literal and implicit
    constructors that subscript syntax requires, so both `t.slice[2, 0:4]()`
    (fix axis 0 to 2, subslice axis 1) and `t.slice[0:4, 0:4]()` parse.

    Parameters:
        static: The index a `ComptimeInt` argument fixes its axis to, or `-1`.
    """

    comptime static_index = Self.static

    var _value: std.utils.Variant[
        Int, ContiguousSlice, ComptimeInt[Self.static]
    ]
    """The wrapped index (`Int` or `ComptimeInt`) or subrange
    (`ContiguousSlice`)."""

    comptime _enum_case_length = 3
    comptime _enum_case_names = ParameterList.of[
        "index".value, "static".value, "slice".value
    ].values
    comptime _enum_case_types = TypeList.of[
        Trait=AnyType, Int, ComptimeInt[Self.static], ContiguousSlice
    ].values

    @implicit
    @inline(.nodebug)
    def __init__(out self, index: Int):
        """Wraps an `Int`, fixing and dropping the corresponding dimension.

        Args:
            index: The index to fix the dimension to.
        """
        self._value = index

    @implicit
    @inline(.nodebug)
    def __init__(out self, index: ComptimeInt[Self.static]):
        """Wraps a compile-time index, fixing and dropping the corresponding
        dimension.

        Args:
            index: The index to fix the dimension to; its value is `static`.
        """
        self._value = index

    @implicit
    @inline(.nodebug)
    def __init__(out self, slice: ContiguousSlice):
        """Wraps a `ContiguousSlice`, slicing the corresponding dimension.

        Args:
            slice: The range to slice the dimension.
        """
        self._value = slice

    @inline(.nodebug)
    def __init__(
        out self,
        start: Optional[Int],
        end: Optional[Int],
        stride: NoneType,
        __slice_literal__: NoneType = None,
    ):
        """Wraps a slice literal as a rank-preserving `ContiguousSlice`.

        Args:
            start: The start of the subrange.
            end: The end of the subrange.
            stride: Always none; disambiguates from strided slices.
            __slice_literal__: Enables slice-literal subscript syntax.
        """
        self._value = ContiguousSlice(start, end, stride)

    def _get_enum_discriminant(self) -> Int:
        if self._value.isa[Int]():
            return 0
        return 1 if self._value.isa[ComptimeInt[Self.static]]() else 2

    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[Trait=AnyType, Self._enum_case_types]()[
        id
    ]:
        return (
            Pointer(to=self._value)
            .unsafe_bitcast[
                TypeList[Trait=AnyType, Self._enum_case_types]()[id]
            ]()
            .unsafe_origin_cast[origin_of(self)]()[]
        )

    @staticmethod
    def index(index: Int) -> Self:
        return Self(index=index)

    @staticmethod
    def slice(slice: ContiguousSlice) -> Self:
        return Self(slice=slice)

    def is_slice(self) -> Bool:
        return self._value.isa[ContiguousSlice]()

    def unsafe_get_slice(self) -> ContiguousSlice:
        return self._value[ContiguousSlice]

    def unsafe_get_index(self) -> Int:
        if self._value.isa[Int]():
            return self._value[Int]
        return Self.static


trait _IndexOrSliceLike(ImplicitlyCopyable):
    """One subscript argument: an index that fixes and drops its axis, or a
    slice that keeps and narrows it.

    The argument's compile-time face is `static_index`. A `TypeList` of
    conforming types therefore describes to the type system which axes a
    subscript fixes at a known position, while the values carry the runtime
    indices and slice bounds.
    """

    comptime static_index: Int
    """The index the argument fixes its axis to when known at compile time,
    `-1` for a runtime index or a slice."""

    comptime is_static_index = Self.static_index != -1
    """Whether the argument fixes its axis at a compile-time index."""

    def is_slice(self) -> Bool:
        """Returns whether the argument keeps its axis as a subrange."""
        ...

    def unsafe_get_slice(self) -> ContiguousSlice:
        """Returns the subrange. Only valid when `is_slice()` holds."""
        ...

    def unsafe_get_index(self) -> Int:
        """Returns the fixed index. Only valid when `is_slice()` does not
        hold."""
        ...


@inline(.nodebug)
def _count_slice_dims[
    slices: ParameterList[type=_IndexOrSlice[], ...]
]() -> Int:
    """Returns the number of `ContiguousSlice` (rank-preserving) arguments."""
    var count = 0
    comptime for i in range(slices.size):
        __match slices[i]:
        case .slice:
            count += 1
        case .index:
            pass
        case .static:
            pass
    return count


@inline(.nodebug)
def _kept_slice_axis_for_output[
    slices: ParameterList[type=_IndexOrSlice[], ...], out_axis: Int
]() -> Int:
    """Maps an output axis to the original axis holding its `ContiguousSlice`.

    `Int` arguments fix (drop) their dimension, so only the axes carrying a
    `ContiguousSlice` survive into the result; this skips the dropped axes.
    """
    var kept_axes_seen = 0
    comptime for axis in range(slices.size):
        __match slices[axis]:
        case .slice:
            if kept_axes_seen == out_axis:
                return axis
            kept_axes_seen += 1
        case .index:
            pass
        case .static:
            pass
    abort("invalid sliced-axis mapping")


@inline(.nodebug)
def _indexed_extent[
    slices: ParameterList[type=_IndexOrSlice[], ...],
    layout: TensorLayout,
    out_axis: Int,
]() -> Int:
    """Returns the extent of output axis `out_axis` after fixing/subslicing."""
    comptime orig = _kept_slice_axis_for_output[slices, out_axis]()
    comptime s = slices[orig].unsafe_get_slice()
    return comptime (
        s.end.or_else(layout.static_shape[orig]) - s.start.or_else(0)
    )


comptime _indexed_extent_at[
    slices: ParameterList[type=_IndexOrSlice[], ...],
    layout: TensorLayout,
    out_axis: Int,
] = _indexed_extent[slices, layout, out_axis]()


@inline(.nodebug)
def _slice_start[
    slices: ParameterList[type=_IndexOrSlice[], ...], axis: Int
]() -> Int:
    """Returns the first element of `axis` that the view selects: a slice's
    start, or the index a rank-reducing (`Int`) argument fixes the axis to."""
    var start = 0
    __match slices[axis]:
    case .slice:
        start = slices[axis].unsafe_get_slice().start.or_else(0)
    case .index:
        start = slices[axis].unsafe_get_index()
    case .static:
        start = slices[axis].unsafe_get_index()
    return start


@inline(.nodebug)
def _slice_storage_offset[
    slices: ParameterList[type=_IndexOrSlice[], ...], layout: TensorLayout
]() -> Int:
    """Returns the scalar-element offset of the view's first element.

    Every dropped axis and every slice start shifts the view's base; none of
    them survives in the view's layout, so the whole shift is folded into its
    storage handle instead.
    """
    var offset = 0
    comptime for axis in range(slices.size):
        offset += _slice_start[slices, axis]() * layout.static_stride[axis]
    return offset


comptime _sliced_stride_at[
    slices: ParameterList[type=_IndexOrSlice[], ...],
    layout: TensorLayout,
    out_axis: Int,
] = layout.static_stride[_kept_slice_axis_for_output[slices, out_axis]()]
"""Flat stride a sliced view inherits for output axis `out_axis`.

A rank-reducing (`Int`) argument drops its axis, so the surviving axes shift
left; the stride has to come from the original axis that carried the slice,
exactly like `_indexed_extent_at` does for the extent. Identity for a
rank-preserving (all-slice) view."""


comptime _SlicedOffset[
    slices: ParameterList[type=_IndexOrSlice[], ...],
    layout: TensorLayout,
] = ComptimeInt[_slice_storage_offset[slices, layout]()]
"""The view's base offset, as a `ComptimeInt` so the engine keeps it static."""


comptime _SlicedLayout[
    slices: ParameterList[type=_IndexOrSlice[], ...],
    layout: TensorLayout,
] = Layout[
    shape_types=_IntToComptimeInt[
        *ParameterList.tabulate[
            _count_slice_dims[slices](),
            _indexed_extent_at[slices, layout, _],
        ]()
    ],
    stride_types=_IntToComptimeInt[
        *ParameterList.tabulate[
            _count_slice_dims[slices](),
            _sliced_stride_at[slices, layout, _],
        ]()
    ],
]
"""The layout of the view `slices` cuts out of `layout`.

Its rank is the number of slice arguments, since every `Int` argument drops
its axis. Extents are the sliced subranges and strides are inherited from the
surviving axes; the element offsets the slices start at live in the view's
storage handle, not here."""


struct TileTensor[
    mut: Bool,
    //,
    dtype: DType,
    LayoutType: TensorLayout,
    origin: Origin[mut=mut],
    *,
    Engine: TensorEngine = DefaultEngine[element_width=1],
    address_space: AddressSpace = .GENERIC,
    linear_idx_type: DType = _get_index_type[LayoutType](address_space),
](DevicePassable, ImplicitlyCopyable, TrivialRegisterPassable, Writable):
    """A tensor type with trait-based layouts supporting nested and hierarchical
    indexing.

    `TileTensor` provides a flexible abstraction for multi-dimensional data with
    layouts expressed via the `TensorLayout` trait. Unlike `LayoutTensor` which
    uses a concrete `Layout` type, `TileTensor` accepts any type implementing
    `TensorLayout`, enabling more flexible compile-time layout composition.

    When to use `TileTensor` vs `LayoutTensor`:

    - Use `TileTensor` when you need trait-based layout composition, nested
      layouts, or when working with the newer `Coord`-based layout system.
    - Use `LayoutTensor` when you need established operations like `copy_dma`,
      `collective_load`, or compatibility with existing code using
      `IntTuple`-based layouts.
    - Both types can interoperate via `to_layout_tensor()`.

    Parameters:
        mut: The inferred mutability of the underlying pointer.
        dtype: The data type of tensor elements (e.g., `DType.float32`).
        LayoutType: A type implementing `TensorLayout` that defines the tensor's
            shape and stride structure. Common types include `Layout` (with
            `Coord`-based shapes/strides) and `RowMajorLayout`.
        origin: The origin of the underlying pointer for lifetime tracking.
        Engine: A type implementing `TensorEngine` that supplies the storage
            handle and the load/store/offset operations acting on it. Defaults
            to `DefaultEngine[element_width=1]`, a plain `Pointer` handle over
            non-vectorized elements.
        address_space: Memory address space (GENERIC, SHARED, CONSTANT, etc.).
            Defaults to GENERIC.
        linear_idx_type: Integer type for memory indexing. Defaults to int32 for
            shared/constant memory, int64 otherwise.

    Example:

    ```mojo
    from layout.tile_layout import row_major
    from layout import TileTensor
    from layout import Idx

    # Create a 4x4 tensor with row-major layout
    var storage = Array[Float32, 16](uninitialized=True)
    var tensor = TileTensor(storage, row_major[4, 4]()).fill(0.0)

    # Access elements using flat indices
    tensor[0, 0] = 1.0
    tensor[1, 2] = 2.0

    # Extract a 2x2 tile at position (1, 0)
    var tile = tensor.tile[2, 2](1, 0)

    # Vectorize for SIMD operations (shape becomes 4x1, element size 1x4)
    var vec = tensor.vectorize[1, 4]()
    ```
    """

    comptime rank = Self.LayoutType.rank
    """The number of dimensions in the tensor's layout."""

    comptime flat_rank = _Flattened[*Self.LayoutType._shape_types].length
    """The flattened rank - total number of dimensions after flattening nested Coords.

    For non-nested layouts, flat_rank == rank.
    For nested layouts (e.g., from blocked_product), flat_rank > rank.
    """

    comptime element_size = Self.Engine.element_size
    """Number of scalar elements per logical element, derived from `Engine`."""

    comptime ElementType = SIMD[Self.dtype, Self.element_size]
    """The SIMD type used for element access.

    For scalar tensors, this is `SIMD[dtype, 1]` (equivalent to `Scalar[dtype]`).
    For vectorized tensors, this reflects the vector width.
    """

    comptime TileResultType[
        tile_shape_types: TypeList[Trait=CoordLike, ...],
        *,
        linear_idx_type: DType = .int,
    ] = TileTensor[
        Self.dtype,
        Layout[
            shape_types=tile_shape_types,
            stride_types=_NestedTileResultStrideTypes[Self.LayoutType],
        ],
        Self.origin,
        Engine=Self.Engine.OffsetResultType[
            TypeList.of[Scalar[linear_idx_type]]()
        ],
        address_space=Self.address_space,
    ]
    """Result type of `.tile[]`. Per outer mode, the result stride is
    parent's innermost sub-stride (CuTe `local_tile`): identity for
    scalar parent strides, last-sub-element for tuple parent strides.

    Trade-off: the `_NestedTileResultStrideTypes[Self.LayoutType]` wrap
    is identity-equivalent for flat parents but nominally a different
    `TypeList` than parent's literal `_stride_types`. Cascaded
    `.tile[].tile[].tile[]` chains pay one `param_list.tabulate(...)`
    wrap per level (~+20% ASAN compile on linalg matmul kernels with
    deep cascades). Until Mojo gets dependent return types for
    parametric aliases (so the flat path could keep parent's literal
    name) this is the cost of a single unified `.tile[]` API.

    Parameters:
        tile_shape_types: The result tile's shape `TypeList` (typically
            built from the variadic `tile_sizes` of the calling
            `.tile[]` method).
        linear_idx_type: Integer type keying the offset-derived storage of the
            result (see `OffsetViewType`). Defaults to `DType.int`.
    """

    comptime shape_known = Self.LayoutType.shape_known
    """True if all shape dimensions are compile-time constants."""

    comptime stride_known = Self.LayoutType.stride_known
    """True if all stride dimensions are compile-time constants."""

    comptime all_dims_known = Self.LayoutType.all_dims_known
    """True if both shape and stride are fully known at compile time.

    Required for operations like `vectorize()` and `distribute()`.
    """

    comptime is_compatible_with[
        C: TypeList[Trait=CoordLike, ...]
    ] = WeaklyCompatible[Self.LayoutType, C]
    """True if coordinate types `C` are structurally compatible with this
    tensor's layout shape.

    A scalar coordinate element is always compatible. A tuple coordinate
    element requires the corresponding layout shape element to also be a
    tuple of the same length, checked recursively up to 4 levels of
    nesting.

    Parameters:
        C: The coordinate element types to check against.
    """

    comptime static_shape[i: Int] = Self.LayoutType.static_shape[i]
    """Get the compile-time shape value for dimension i, or -1 if dynamic.

    Parameters:
        i: The dimension index.
    """

    comptime static_stride[i: Int] = Self.LayoutType.static_stride[i]
    """Get the compile-time stride value for dimension i, or -1 if dynamic.

    Parameters:
        i: The dimension index.
    """

    var _storage: Self.Engine.StorageType[
        Self.dtype, Self.origin, Self.address_space
    ]
    """Pointer to the tensor's underlying data storage."""

    var layout: Self.LayoutType
    """The layout instance defining shape and stride mappings."""

    comptime device_type = Self
    """Device-side type for GPU kernel parameter passing."""

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def _is_convertible_to_device_type[T: AnyType]() -> Bool:
        comptime if Self.mut:
            return TypeList.of[
                Self,
                Self.OriginCastType[ImmOrigin(Self.origin)],
            ]().contains[T]()
        else:
            return TypeList.of[Self]().contains[T]()

    @staticmethod
    def get_type_name() -> String:
        """
        Gets the name of the host type (the one implementing this trait).

        Returns:
            The host type's name.
        """
        var writer = String()
        t"TileTensor[mut={Self.mut}, dtype={Self.dtype}, Engine=".write_to(
            writer
        )
        Self.Engine.write_type_name_to(writer)
        (
            t", address_space={Self.address_space},"
            t" linear_idx_type={Self.linear_idx_type}]"
        ).write_to(writer)
        return writer^

    comptime GenericType = TileTensor[
        Self.dtype,
        Self.LayoutType,
        Self.origin,
        Engine=DefaultEngine[element_width=1],
        address_space=.GENERIC,
        linear_idx_type=Self.linear_idx_type,
    ]
    """Type alias for this tensor with GENERIC address space.

    Used by constructors that create tensors from Span or HostBuffer, which
    produce GENERIC address space tensors.
    """

    comptime DeviceGenericType[origin: Origin] = TileTensor[
        Self.dtype,
        Self.LayoutType,
        origin,
        Engine=DevicePointerEngine[element_width=1],
        address_space=.GENERIC,
        linear_idx_type=Self.linear_idx_type,
    ]
    """Type alias for this tensor backed by `DevicePointerEngine`.

    Used by the `DeviceBuffer` and `DevicePointer` constructors, which carry the
    buffer's `DevicePointer` (its owning reference plus offset and size) to the
    kernel boundary instead of a bare pointer.

    Parameters:
        origin: The pointer origin for the returned device-pointer-backed
            tensor.
    """

    @inline(.always)
    def __init__(
        out self,
        var storage: Self.Engine.StorageType[
            Self.dtype, Self.origin, Self.address_space
        ],
        var layout: Self.LayoutType,
        /,
    ):
        """Create a TileTensor from a storage handle and layout.

        Args:
            storage: The storage handle referencing the tensor data.
            layout: The layout defining the tensor's shape and strides.
        """
        self._storage = storage
        self.layout = layout

    @inline(.always)
    def __init__(
        out self,
        *,
        var ptr: Pointer[
            Scalar[Self.dtype], Self.origin, address_space=Self.address_space
        ],
        var layout: Self.LayoutType,
    ):
        """Create a TileTensor from a `Pointer` and layout.

        Args:
            ptr: The pointer to the tensor data.
            layout: The layout defining the tensor's shape and strides.
        """
        comptime assert (
            Self.Engine == DefaultEngine[element_width=1]
        ), "TileTensor.__init__ from Pointer requires DefaultEngine"
        self._storage = rebind[type_of(self._storage)](ptr)
        self.layout = layout

    def __init__(
        out self: Self.GenericType,
        var span: Span[Scalar[Self.dtype], Self.origin],
        var layout: Self.LayoutType,
    ):
        """Create a TileTensor from a Span and layout.

        Args:
            span: The memory span containing the tensor data.
            layout: The layout defining the tensor's shape and strides.
        """
        comptime assert (
            Self.Engine == DefaultEngine[element_width=1]
        ), "TileTensor.__init__ from Span requires DefaultEngine"
        self._storage = rebind[type_of(self._storage)](span.unsafe_ptr())
        self.layout = layout

    @inline(.always)
    def __init__(
        out self: Self.GenericType,
        ref[Self.origin] device_buffer: DeviceBuffer[Self.dtype],
        var layout: Self.LayoutType,
    ):
        """Create a `LayoutTensor` from a `DeviceBuffer`. The layout must have
        statically known dimensions.

        Note that the device buffer memory is on the accelerator device (GPU
        global memory). Code running on the CPU can use the
        [`DeviceContext`](/api/mojo/max/gpu/host/device_context/DeviceContext/) to
        allocate a `DeviceBuffer` and use that to construct a `LayoutTensor`
        that can be accessed on the GPU. You cannot directly access data in the
        `DeviceBuffer` or `LayoutTensor` from the CPU.

        The following example shows a typical pattern for using `DeviceBuffer`
        to construct a `LayoutTensor` that you can use on the GPU.

        ```mojo
        from max.gpu.host import DeviceContext, DeviceBuffer
        from layout.tile_layout import row_major
        from layout import TileTensor
        from layout import Idx

        comptime dtype = DType.float32

        var ctx = DeviceContext()
        # Allocate buffers
        var dev_buf = ctx.enqueue_create_buffer[dtype](16)
        var host_buf = ctx.enqueue_create_host_buffer[dtype](16)
        # Ensure buffers have been created
        ctx.synchronize()

        # Initialize host buffer and copy to device buffer
        for i in range(16):
            host_buf[i] = Scalar[dtype](i)
        ctx.enqueue_copy(dev_buf, host_buf)

        # Create TileTensor to use on device
        var tensor = TileTensor(
             dev_buf,
             row_major(Idx[4], Idx[4]),
        )
        ...
        ```
        Args:
            device_buffer: Contains the underlying data to point to.
            layout: The layout of the tensor.
        """
        comptime assert (
            Self.Engine == DefaultEngine[element_width=1]
        ), "TileTensor.__init__ from DeviceBuffer requires DefaultEngine"
        self._storage = rebind[type_of(self._storage)](
            device_buffer.unsafe_ptr()
        )
        self.layout = layout

    @inline(.always)
    def __init__(
        out self: Self.DeviceGenericType[Self.origin],
        var device_pointer: DevicePointer[Self.dtype, Self.origin],
        var layout: Self.LayoutType,
    ):
        """Create a `DevicePointerEngine`-backed `TileTensor` from a
        `DevicePointer`.

        Like the `DeviceBuffer` constructor, this produces a
        `DevicePointerEngine`-backed tile that carries the full `DevicePointer`
        (its non-owning reference to the owning `DeviceBuffer` plus an element
        offset and size) to the kernel boundary, where
        `DevicePointer._to_device_type` encodes it to a bare device pointer.
        Use this overload when you already hold a `DevicePointer` (for example
        an offset one); construct it with `TileTensor(buffer.device_ptr(),
        layout)`.

        The tile borrows the `DevicePointer`'s origin; the backing
        `DeviceBuffer` must outlive the tile.

        Args:
            device_pointer: The device pointer referencing the tensor data.
            layout: The layout of the tensor.
        """
        self._storage = device_pointer
        self.layout = layout

    @inline(.always)
    def __init__(
        out self: Self.GenericType,
        ref[Self.origin] host_buffer: HostBuffer[Self.dtype],
        var layout: Self.LayoutType,
    ):
        """Create a `LayoutTensor` from a `HostBuffer`. The layout must have
        statically known dimensions.

        The resulting tensor's data can only be accessed on the CPU.

        ```mojo
        from max.gpu.host import DeviceContext, HostBuffer
        from layout.tile_layout import row_major
        from layout import TileTensor
        from layout import Idx

        comptime dtype = DType.float32

        var ctx = DeviceContext()
        var host_buf = ctx.enqueue_create_host_buffer[dtype](8)

        var tensor = TileTensor(
            host_buf,
            row_major(Idx[4], Idx[4]),
        )
        ```

        Args:
            host_buffer: Contains the underlying data to point to.
            layout: The layout of the tensor.
        """
        comptime assert (
            Self.Engine == DefaultEngine[element_width=1]
        ), "TileTensor.__init__ from HostBuffer requires DefaultEngine"
        self._storage = rebind[type_of(self._storage)](host_buffer.unsafe_ptr())
        self.layout = layout

    @inline(.nodebug)
    @implicit
    def __init__(
        other: TileTensor,
        out self: type_of(other).Immut,
    ):
        """Implicitly cast a mutable TileTensor to immutable.

        Args:
            other: The mutable TileTensor to cast from.
        """
        self._storage = other._unsafe_storage_cast[
            to_origin=type_of(self).origin
        ]()
        self.layout = other.layout

    @inline(.nodebug)
    @implicit
    def __init__(
        other: TileTensor[mut=Self.mut, ...],
        out self: type_of(other).OriginCastType[AnyOrigin[mut=Self.mut]],
    ):
        """Implicitly cast a TileTensor to have an `AnyOrigin`.

        Args:
            other: The TileTensor to cast from.
        """
        self._storage = other._unsafe_storage_cast[
            to_origin=type_of(self).origin
        ]()
        self.layout = other.layout

    @doc_hidden
    @inline(.always)
    def __getattr_param__[
        name: StringLiteral
    ](
        self,
        out result: Pointer[
            Scalar[Self.dtype], Self.origin, address_space=Self.address_space
        ],
    ):
        comptime assert (
            name == "ptr"
        ), "TileTensor.__getattr_param__ only support 'ptr'"
        try:
            result = Self.Engine.unsafe_ptr(self._storage)
        except e:
            abort(t"TileTensor.ptr access not possible: {e}")

    @inline(.nodebug)
    def _unsafe_storage_cast[
        to_mut: Bool = Self.mut,
        //,
        to_dtype: DType = Self.dtype,
        to_origin: Origin[mut=to_mut] = Self.origin.unsafe_mut_cast[to_mut](),
        to_address_space: AddressSpace = Self.address_space,
    ](self) -> Self.Engine.StorageType[to_dtype, to_origin, to_address_space]:
        return Self.Engine.unsafe_cast[
            to_dtype,
            to_origin,
            to_address_space,
        ](self._storage)

    @inline(.nodebug)
    def _offset_storage(
        self, offset: Some[CoordLike]
    ) -> Self.Engine.OffsetResultType[
        TypeList.of[type_of(offset)]()
    ].StorageType[Self.dtype, Self.origin, Self.address_space]:
        """Advances `self`'s storage handle by `offset` elements via the
        engine.
        """
        return Self.Engine.offset(self._storage, Coord(offset))

    @inline(.nodebug)
    def _load_storage[
        width: SIMDLength,
        alignment: Int,
        invariant: Bool = False,
        non_temporal: Bool = False,
    ](self, offset: Some[Indexer]) -> SIMD[Self.dtype, width]:
        """Loads `width` elements from `self`'s storage handle at `offset` via
        the engine."""
        return Self.Engine.load[
            width=width,
            alignment=alignment,
            invariant=invariant,
            non_temporal=non_temporal,
        ](
            self._unsafe_storage_cast[to_mut=False](),
            offset,
        )

    @inline(.nodebug)
    def _store_storage[
        alignment: Int,
        non_temporal: Bool = False,
    ](self, offset: Some[Indexer], value: SIMD[Self.dtype, _]) where Self.mut:
        """Stores `value` into `self`'s storage handle at `offset` via the
        engine."""
        Self.Engine.store[
            alignment=alignment,
            non_temporal=non_temporal,
        ](self._unsafe_storage_cast[to_mut=True](), offset, value)

    @inline(.nodebug)
    def __getitem__(self, i0: Some[CoordLike]) -> Self.ElementType:
        """Retrieve the element at the given index or coordinate.

        Args:
            i0: The index along axis 0, or a `Coord` holding every index.

        Returns:
            The element at the specified position.
        """
        comptime if type_of(i0).is_tuple:
            return self.load(i0.tuple())
        else:
            return self.load(Coord(i0))

    @inline(.nodebug)
    def __getitem__(
        self, i0: Some[CoordLike], i1: Some[CoordLike]
    ) -> Self.ElementType:
        """Retrieve the element at the given indices.

        Args:
            i0: The index along axis 0.
            i1: The index along axis 1.

        Returns:
            The element at the specified position.
        """
        return self.load(Coord(i0, i1))

    @inline(.nodebug)
    def __getitem__(
        self, i0: Some[CoordLike], i1: Some[CoordLike], i2: Some[CoordLike]
    ) -> Self.ElementType:
        """Retrieve the element at the given indices.

        Args:
            i0: The index along axis 0.
            i1: The index along axis 1.
            i2: The index along axis 2.

        Returns:
            The element at the specified position.
        """
        return self.load(Coord(i0, i1, i2))

    @inline(.nodebug)
    def __getitem__(
        self,
        i0: Some[CoordLike],
        i1: Some[CoordLike],
        i2: Some[CoordLike],
        i3: Some[CoordLike],
    ) -> Self.ElementType:
        """Retrieve the element at the given indices.

        Args:
            i0: The index along axis 0.
            i1: The index along axis 1.
            i2: The index along axis 2.
            i3: The index along axis 3.

        Returns:
            The element at the specified position.
        """
        return self.load(Coord(i0, i1, i2, i3))

    @inline(.nodebug)
    def __getitem__(
        self,
        i0: Some[CoordLike],
        i1: Some[CoordLike],
        i2: Some[CoordLike],
        i3: Some[CoordLike],
        i4: Some[CoordLike],
    ) -> Self.ElementType:
        """Retrieve the element at the given indices.

        Args:
            i0: The index along axis 0.
            i1: The index along axis 1.
            i2: The index along axis 2.
            i3: The index along axis 3.
            i4: The index along axis 4.

        Returns:
            The element at the specified position.
        """
        return self.load(Coord(i0, i1, i2, i3, i4))

    @inline(.nodebug)
    def __getitem__(
        self,
        i0: Some[CoordLike],
        i1: Some[CoordLike],
        i2: Some[CoordLike],
        i3: Some[CoordLike],
        i4: Some[CoordLike],
        i5: Some[CoordLike],
    ) -> Self.ElementType:
        """Retrieve the element at the given indices.

        Args:
            i0: The index along axis 0.
            i1: The index along axis 1.
            i2: The index along axis 2.
            i3: The index along axis 3.
            i4: The index along axis 4.
            i5: The index along axis 5.

        Returns:
            The element at the specified position.
        """
        return self.load(Coord(i0, i1, i2, i3, i4, i5))

    @inline(.nodebug)
    def __getitem__[
        *CoordLikes: CoordLike
    ](self, *coords: *CoordLikes) -> Self.ElementType:
        """Retrieve a single element from the tensor at the specified coordinates.

        Accepts either a single `Coord` argument or multiple scalar
        `CoordLike` arguments packed into a `Coord`. Passing a slice produces
        a view instead.

        The fixed-arity overloads above serve ranks up to six; this pack is
        the fallback beyond that. Overload resolution ranks a fixed arity
        over any pack, which is what keeps an element load unambiguous
        against the slicing pack below inside a `comptime if` branch that is
        not taken, where `where` clauses are ignored.

        Parameters:
            CoordLikes: The types of each index argument (`CoordLike`).

        Args:
            coords: The coordinates specifying the element's position.

        Returns:
            The element at the specified position.
        """
        comptime if CoordLikes.length == 1 and CoordLikes[0].is_tuple:
            return self.load(coords[0].tuple())
        else:
            var coord = Coord[*CoordLikes]()
            comptime for i in range(CoordLikes.length):
                Pointer(to=coord[i]).write(coords[i])
            return self.load(coord)

    @inline(.nodebug)
    def __getitem__[
        *CoordLikes: CoordLike
    ](self, coords: Tuple[*CoordLikes]) -> Self.ElementType:
        """Retrieve a single element from the tensor at the specified coordinates.

        Accepts either a single `Coord` argument or multiple scalar
        `CoordLike` arguments packed into a `Coord`.

        Parameters:
            CoordLikes: The types of each index argument (`CoordLike`).

        Args:
            coords: The tuple containing coordinates specifying the element's position.

        Returns:
            The element at the specified position.
        """
        return self[Coord(coords)]

    @inline(.always)
    def __getitem__[
        *arg_types: AnyType
    ](self, *args: *arg_types) -> Self.OffsetViewType[
        _SubscriptOffset[
            _SubscriptArgs[arg_types](),
            Self.LayoutType,
            Self.linear_idx_type,
        ](),
        _SubscriptLayout[arg_types, Self.LayoutType, Self.linear_idx_type],
    ] where _SubscriptHasSlice[arg_types]:
        """Fix or narrow each dimension, returning a view.

        Applies when at least one argument is a slice; indexing every
        dimension loads a single element instead. Each argument is either an
        index (`n` / `Idx[n]`), which fixes that dimension to the given
        element and drops it from the result, or a slice (`a:b`), which keeps
        the dimension and narrows it to that subrange. Every dimension takes
        exactly one argument.

        Inside a `comptime if` branch that is not taken, overload resolution
        ignores `where` clauses, so this pack alone cannot be told apart from
        a variadic element overload there. The fixed-arity element overloads
        above outrank any pack by arity instead, and a conditional return
        type is no way out either: the compiler's IR verifier does not fold
        it when an argument's type is an element of a generic `Coord`.

        Compile-time and runtime arguments go through the same path; what is
        known at compile time is folded there. An `Idx[n]` index on an axis
        with a compile-time stride contributes to the view's offset as a
        `ComptimeInt` component, and only the remainder is computed at
        runtime. Slice bounds are runtime values, so a sliced dimension's
        extent is runtime too -- `:` is `0:dim`, not a marker.

        Note:
            Only works with flat (non-nested) layouts where every shape and
            stride element is a scalar `CoordLike` (e.g., layouts produced by
            `row_major`, `col_major`, or manual `Layout` construction). Does
            **not** support nested/hierarchical layouts (e.g., from
            `blocked_product`) where shape or stride elements are `Coord`
            tuples.

        Parameters:
            arg_types: The type of each argument: a `CoordLike` index, or a
                `ContiguousSlice` for a dimension to keep.

        Args:
            args: One argument per dimension, in dimension order.

        Returns:
            A strided view over the same backing storage, of rank equal to
            the number of slice arguments. Strides are inherited from the
            surviving axes. The view's offset has a `ComptimeInt` component
            for the compile-time indices and a `Scalar` component for the
            rest.

        Example:

        ```mojo
        from layout import TileTensor
        from layout.tile_layout import row_major

        # 4D tensor: (batch=2, N=8, heads=4, head_dim=16)
        var storage = Array[Float32, 2 * 8 * 4 * 16](fill=0)
        var t = TileTensor(storage, row_major[2, 8, 4, 16]())

        var batch = 1

        # Fix batch and heads, keep N and head_dim -> 2D (8, 16). The
        # heads index is compile-time, so its share of the offset is too.
        var selected = t[batch, :, Idx[2], :]

        # Narrow N to a runtime subrange, keep heads and head_dim whole
        # -> 3D (n, 4, 16)
        var head = t[batch, 0:n, :, :]
        ```
        """
        comptime assert (
            arg_types.length == Self.rank
        ), "subscript takes exactly one argument per dimension"
        comptime assert (
            Self.rank == Self.flat_rank
        ), "subscript slicing requires a flat (non-nested) layout"

        comptime Args = _SubscriptArgs[arg_types]()

        # The runtime share of the pointer offset: the runtime indices, the
        # slice starts, and any compile-time index whose stride is not.
        # Compile-time indices on compile-time strides are already summed
        # into the `ComptimeInt` component of the return type. Narrow-first
        # multiply at `linear_idx_type` precision keeps index arithmetic out
        # of 64-bit Int on GPUs with narrow `linear_idx_type` (e.g. uint32).
        # Callers are responsible for picking a `linear_idx_type` wide enough
        # to hold the maximum offset.
        var offset = Scalar[Self.linear_idx_type](0)

        comptime ShapeTypes = _SubscriptShape[arg_types, Self.linear_idx_type]()
        comptime StrideTypes = _SubscriptStride[
            arg_types, Self.LayoutType._stride_types
        ]()

        var new_shape = Coord[*ShapeTypes]()
        var new_stride = Coord[*StrideTypes]()

        comptime for axis in range(Self.rank):
            var stride = Scalar[Self.linear_idx_type](
                self.layout.stride[axis]().value()
            )

            comptime if _IsSliceArg[arg_types[axis], axis]:
                comptime assert (
                    arg_types[axis] == ContiguousSlice
                ), "subscript arguments must be indices or slices"
                comptime out_axis = _SubscriptOutAxis[arg_types, axis]
                # `indices` resolves the open ends against the parent extent
                # and folds in Python's negative-index wrap.
                var bounds = rebind[ContiguousSlice](args[axis]).indices(
                    Int(self.layout.shape[axis]().value())
                )
                offset += Scalar[Self.linear_idx_type](bounds[0]) * stride
                Pointer(to=new_shape[out_axis]).write(
                    rebind[ShapeTypes[out_axis]](
                        Scalar[Self.linear_idx_type](bounds[1] - bounds[0])
                    )
                )
                Pointer(to=new_stride[out_axis]).write(
                    rebind[StrideTypes[out_axis]](self.layout.stride[axis]())
                )
            elif not _IsStaticOffsetAxis[Args, Self.LayoutType, axis]:
                # Re-states what `_IsSliceArg` established, which is what
                # lets the index's `CoordLike` interface be used here.
                comptime assert conforms_to(arg_types[axis], CoordLike)
                offset += (
                    Scalar[Self.linear_idx_type](args[axis].value()) * stride
                )

        var new_layout = Layout(new_shape, new_stride)
        var static_offset = ComptimeInt[
            _subscript_static_offset[Args, Self.LayoutType]()
        ]()

        return {
            Self.Engine.offset(self._storage, Coord(static_offset, offset)),
            new_layout,
        }

    @inline(.nodebug)
    def __setitem__(self, coord: Coord, value: Self.ElementType) where Self.mut:
        """Set a single element in the tensor at the specified coordinates.

        Accepts Coords of flat_rank (flattened).

        Args:
            coord: The coordinates specifying the element's position.
            value: The value to store at the specified position.
        """
        self.store(coord, value)

    @inline(.nodebug)
    def __setitem__[
        *IndexTypes: Indexer & Copyable
    ](self, *items: *IndexTypes, value: Self.ElementType) where (
        IndexTypes.length == Self.flat_rank
    ) & Self.mut:
        """Sets a single element in the tensor at the specified indices.

        Uses flat indexing based on flat_rank. For non-nested layouts,
        flat_rank == rank, so tensor[i, j, k] = value works normally. For
        nested layouts (e.g., from blocked_product), use all flat_rank indices:
        tensor[i0, i1, i2, i3] = value for a tensor with flat_rank == 4.

        Parameters:
            IndexTypes: The types of the indices (must implement Indexer).

        Args:
            items: The indices specifying the element's position.
            value: The value to store.
        """
        comptime arg_count = IndexTypes.length
        var linear_tuple = DynamicCoord[Self.linear_idx_type, arg_count]()

        comptime for i in range(arg_count):
            Pointer(to=linear_tuple[i]).write(
                rebind[type_of(linear_tuple).element_types[i]](
                    Scalar[Self.linear_idx_type](index(items[i]))
                )
            )

        # Inline store logic to avoid constraint propagation issues
        comptime alignment = align_of[
            SIMD[Self.dtype, Self.element_size]
        ]() if is_gpu() else align_of[Self.dtype]()
        self._store_storage[alignment=alignment](
            self.layout[linear_idx_type=Self.linear_idx_type](linear_tuple),
            value,
        )

    @inline(.nodebug)
    def load[
        width: SIMDLength = Self.element_size,
        alignment: Int = align_of[
            SIMD[Self.dtype, width]
        ]() if is_gpu() else align_of[Self.dtype](),
        invariant: Bool = _default_invariant[Self.mut](),
        non_temporal: Bool = False,
    ](self, coord: Coord) -> SIMD[Self.dtype, width]:
        """Load elements from the tensor at the specified coordinates.

        Supports both hierarchical indexing (rank indices) and flat indexing
        (flat_rank indices) for nested layouts.

        Parameters:
            width: Number of elements to load (default: element_size).
            alignment: Memory alignment for the load.
            invariant: If True, the compiler may assume the memory won't be
                modified during the kernel, enabling load hoisting and caching.
            non_temporal: If True, indicates the data will not be reused soon,
                allowing the hardware to bypass caches (e.g., streaming loads).

        Args:
            coord: The coordinates specifying the element's position.

        Returns:
            A SIMD vector containing the loaded elements.
        """
        comptime assert (
            Self.is_compatible_with[coord.element_types]
            or coord.rank == Self.flat_rank
            or coord.rank == 1
        )

        return self.raw_load[
            width=width,
            alignment=alignment,
            invariant=invariant,
            non_temporal=non_temporal,
        ](self.layout[linear_idx_type=Self.linear_idx_type](coord))

    @inline(.nodebug)
    def store[
        width: SIMDLength = Self.element_size,
        alignment: Int = align_of[
            SIMD[Self.dtype, width]
        ]() if is_gpu() else align_of[Self.dtype](),
        non_temporal: Bool = False,
    ](self, coord: Coord, value: SIMD[Self.dtype, width]) where Self.mut:
        """Store elements to the tensor at the specified coordinates.

        Supports both hierarchical indexing (rank indices) and flat indexing
        (flat_rank indices) for nested layouts.

        Parameters:
            width: Number of elements to store (default: element_size).
            alignment: Memory alignment for the store.
            non_temporal: If True, indicates the data will not be reused soon,
                allowing the hardware to bypass caches (e.g., streaming stores).

        Args:
            coord: The coordinates specifying the element's position.
            value: The SIMD vector to store.
        """
        comptime assert Self.is_compatible_with[coord.element_types]

        self._store_storage[alignment=alignment, non_temporal=non_temporal](
            self.layout[linear_idx_type=Self.linear_idx_type](coord),
            value,
        )

    @inline(.nodebug)
    def _linear_offset(
        self, idx: IndexList[_, ...]
    ) -> Scalar[Self.linear_idx_type]:
        """Compute a linear memory offset from an IndexList using the layout
        strides.

        This is for flat (non-nested) layouts where rank == flat_rank. It
        computes the inner product of the index and stride vectors.

        Args:
            idx: The multi-dimensional index.

        Returns:
            The linear memory offset.
        """
        comptime assert (
            idx.size == Self.rank
        ), "IndexList rank must match tensor rank"
        comptime assert (
            Self.rank == Self.flat_rank
        ), "load_linear/store_linear only support flat layouts"
        var offset = Scalar[Self.linear_idx_type](0)
        var stride_coord = self.layout.stride_coord()

        comptime for i in range(idx.size):
            offset += Scalar[Self.linear_idx_type](idx[i]) * Scalar[
                Self.linear_idx_type
            ](stride_coord[i].value())
        return offset

    @inline(.nodebug)
    def load_linear[
        width: SIMDLength = Self.element_size,
        alignment: Int = align_of[SIMD[Self.dtype, width]](),
        invariant: Bool = _default_invariant[Self.mut](),
    ](self, idx: IndexList[_, ...]) -> SIMD[Self.dtype, width]:
        """Load elements using an IndexList index (for flat layouts).

        This enables TileTensor to be used directly with `_elementwise_impl_gpu`
        callbacks which pass IndexList coordinates.

        Parameters:
            width: Number of elements to load.
            alignment: Memory alignment for the load.
            invariant: If True, enables load hoisting.

        Args:
            idx: The multi-dimensional index.

        Returns:
            A SIMD vector containing the loaded elements.
        """
        return self.raw_load[
            width=width, alignment=alignment, invariant=invariant
        ](self._linear_offset(idx))

    @__allow_legacy_custom_self_type
    @inline(.nodebug)
    def store_linear[
        width: SIMDLength = Self.element_size,
        alignment: Int = align_of[SIMD[Self.dtype, width]](),
    ](
        self: TileTensor[mut=True, Self.dtype, ...],
        idx: IndexList[_, ...],
        value: SIMD[Self.dtype, width],
    ):
        """Store elements using an IndexList index (for flat layouts).

        This enables TileTensor to be used directly with `_elementwise_impl_gpu`
        callbacks which pass IndexList coordinates.

        Parameters:
            width: Number of elements to store.
            alignment: Memory alignment for the store.

        Args:
            idx: The multi-dimensional index.
            value: The SIMD vector to store.
        """
        self.raw_store[alignment=alignment](self._linear_offset(idx), value)

    @inline(.nodebug)
    def raw_load[
        width: SIMDLength = 1,
        alignment: Int = align_of[Self.dtype](),
        invariant: Bool = _default_invariant[Self.mut](),
        non_temporal: Bool = False,
    ](self, offset: Some[Indexer]) -> SIMD[Self.dtype, width]:
        """Load `width` elements starting at `ptr[offset]`, bypassing the layout.

        This is a raw read against the underlying storage: the caller
        is responsible for ensuring `offset` is a valid index into
        the backing buffer, independent of the tensor's layout. Useful for
        kernels that treat the buffer as a contiguous array (copies, fills,
        reductions over contiguous storage).

        Parameters:
            width: Number of elements to load.
            alignment: Memory alignment for the load.
            invariant: If True, enables load hoisting.
            non_temporal: If True, indicates the data will not be reused
                soon, allowing the hardware to bypass caches (e.g.,
                streaming loads).

        Args:
            offset: Linear element offset into the underlying storage.

        Returns:
            A SIMD vector containing the loaded elements.
        """
        return self._load_storage[
            width=width,
            alignment=alignment,
            invariant=invariant,
            non_temporal=non_temporal,
        ](_index(offset))

    @inline(.nodebug)
    def raw_store[
        width: SIMDLength = 1,
        alignment: Int = align_of[Self.dtype](),
        non_temporal: Bool = False,
    ](
        self,
        offset: Some[Indexer],
        value: SIMD[Self.dtype, width],
    ) where Self.mut:
        """Store `width` elements at `ptr[offset]`, bypassing the layout.

        This is a raw write against the underlying storage: the caller
        is responsible for ensuring `offset` is a valid index into
        the backing buffer, independent of the tensor's layout.

        Parameters:
            width: Number of elements to store.
            alignment: Memory alignment for the store.
            non_temporal: If True, indicates the data will not be reused
                soon, allowing the hardware to bypass caches (e.g.,
                streaming stores).

        Args:
            offset: Linear element offset into the underlying storage.
            value: The SIMD vector to store.
        """
        self._store_storage[alignment=alignment, non_temporal=non_temporal](
            _index(offset), value
        )

    @inline(.always)
    def as_span(
        self,
        out result: Span[
            Scalar[Self.dtype], Self.origin, address_space=Self.address_space
        ],
    ):
        """Get a `Span` over the tensor's elements.

        Constraints:
            The tensor must have row-major (contiguous) strides, so storage
            order and layout order agree and the span visits every element
            exactly once.

        Returns:
            A `Span` of `num_elements()` scalars over the tensor's storage.
        """
        comptime assert (
            Self.Engine == DefaultEngine[element_width=1]
        ), "TileTensor.as_span requires DefaultEngine"
        comptime assert (
            Self.is_row_major
        ), "TileTensor.as_span requires row-major (contiguous) strides"
        return {
            unsafe_ptr = rebind[
                Pointer[
                    Scalar[Self.dtype],
                    Self.origin,
                    address_space=Self.address_space,
                ]
            ](self._storage),
            length = self.num_elements(),
        }

    @inline(.nodebug)
    def bitcast[
        target_dtype: DType,
    ](self) -> TileTensor[
        target_dtype,
        Self.LayoutType,
        Self.origin,
        Engine=Self.Engine,
        address_space=Self.address_space,
        linear_idx_type=Self.linear_idx_type,
    ]:
        """Reinterprets the tensor's element dtype, preserving layout.

        Returns a new `TileTensor` that shares the same underlying storage
        and layout as `self` but views elements as `target_dtype` rather
        than `Self.dtype`.

        Parameters:
            target_dtype: The new element dtype to view the storage as.

        Returns:
            A `TileTensor[target_dtype, ...]` backed by the same pointer
            and layout as `self`.
        """
        return {
            self._unsafe_storage_cast[
                to_dtype=target_dtype, to_origin=Self.origin
            ](),
            self.layout,
        }

    @inline(.always)
    def ptr_at_offset(
        self,
        coords: Coord[...],
        out result: Pointer[
            Scalar[Self.dtype], Self.origin, address_space=Self.address_space
        ],
    ) where coords.flat_rank == Self.flat_rank or coords.flat_rank == 1:
        """Get a pointer offset at the given flattened coordinates.

        Args:
            coords: A flattened list of the offset coordinates.

        Returns:
            A pointer offset at the given flattened coordinates.
        """
        comptime assert (
            Self.Engine == DefaultEngine[element_width=1]
        ), "TileTensor.ptr_at_offset requires DefaultEngine"
        return rebind[type_of(result)](self._storage).unsafe_offset(
            self.layout[linear_idx_type=Self.linear_idx_type](coords)
        )

    @inline(.always)
    def prefetch(
        self, coords: Coord[...]
    ) where coords.flat_rank == Self.flat_rank:
        """Prefetch tensor data at the specified coordinates into cache.

        Issues a software prefetch hint to the processor to load the data at
        coords into the cache hierarchy. This can improve performance
        by reducing memory latency for subsequent accesses to the same location.

        Args:
            coords: The indices.

        Performance:

        - Prefetching is a performance hint and does not guarantee data will be
            cached.
        - Most effective when issued sufficiently ahead of the actual data
            access.
        - Uses high locality prefetch to the data cache, optimized for data that
            will be accessed multiple times.
        - Can reduce memory access latency by 50-90% when used correctly.

        Notes:

        - Excessive prefetching can pollute the cache and degrade performance.
        - Most beneficial for predictable access patterns that would otherwise
            cause cache misses.
        - No operation is performed on the prefetched data.
        """
        prefetch[PrefetchOptions().for_read().high_locality().to_data_cache()](
            self.ptr_at_offset(coords)
        )

    def num_elements(self) -> Int:
        """Returns the total number of elements in the tensor.

        Computes the product of all shape dimensions.

        Returns:
            The total element count.
        """
        var result = 1

        comptime for i in range(Self.rank):
            result *= Int(self.layout.shape[i]().value())
        return result

    @inline(.nodebug)
    def copy_from(self, other: TileTensor) where Self.mut:
        """Copy data from another tensor into this tensor.

        Performs an element-by-element copy from `other` into `self`,
        respecting the layouts of both tensors. Each logical element is
        loaded from `other` using its layout and stored into `self` using
        `self`'s layout, so the copy works correctly even when the tensors
        have different shapes or strides (as long as they agree on total
        element count).

        When both tensors have fully static, row-major layouts,
        the copy widens to SIMD load + cast + SIMD store,
        using the narrower of the two dtypes' native SIMD widths.

        The copy loop lives in the engine (`Self.Engine.copy_from`);
        this forwards `self` and `other` as `(storage, layout)` pairs.

        Constraints:

        - Both tensors must have statically known shapes with matching total
            element count.
        - Source and destination dtypes may differ; each logical element is
            cast to the destination dtype.

        Args:
            other: The source tensor to copy data from. Must have the same
                total number of elements as `self`.
        """
        # `other` may carry a different (e.g. offset-derived) engine;
        # the storage-level copy takes it as a distinct `OtherEngine` operand.
        Self.Engine.copy_from(
            (self._unsafe_storage_cast[to_mut=True](), self.layout),
            (other._storage, other.layout),
        )

    @inline(.always)
    def copy_from_async[
        is_masked: Bool = False,
        swizzle: Optional[Swizzle] = None,
        fill: Fill = Fill.NONE,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        src: TileTensor,
        src_idx_bound: Scalar[src.linear_idx_type] = 0,
        base_offset: Scalar[Self.linear_idx_type] = 0,
    ) where Self.mut:
        """Asynchronously copy data from another tensor to this tensor using GPU
        hardware.

        This method performs an asynchronous copy from the source tensor to this
        tensor using GPU hardware acceleration. It's specifically designed for
        copying data from global memory to shared memory in GPU kernels,
        leveraging hardware-specific asynchronous copy mechanisms for improved
        performance.

        For optimal performance, you need to arrange the copy correctly. Use the
        [`distribute()`](/api/mojo/layout/tile_tensor/TileTensor/#distribute)
        method to create thread-local fragments of the source and destination
        tensors, assigning each thread one or more elements to copy.

        Optionally, use the
        [`vectorize()`](/api/mojo/layout/tile_tensor/TileTensor/#vectorize)
        method to get vectorized views of both tensors before calling
        `distribute()`. This allows each thread to copy multiple elements of the
        tensor. For example:

        ```mojo
        var fragment = tensor.vectorize[1, simd_width]().distribute[
            thread_layout
        ](thread_id)
        ```

        The copy operation is asynchronous, so you must call
        [`async_copy_wait_all()`](/api/mojo/max/gpu/memory/memory/async_copy_wait_all/)
        or
        [`async_copy_wait_group()`](/api/mojo/max/gpu/memory/memory/async_copy_wait_group/)
        to ensure the copy has completed before using the data.

        Unlike `LayoutTensor`, a `TileTensor`'s logical element is always a
        contiguous run of `element_size` scalars (the engine's
        `element_width`), so there is no non-vectorizable element layout to
        fall back on: every copy issues one `cp.async` per logical element.

        Constraints:
            - Destination must be in shared memory.
            - Source must be in the generic or global address space.
            - Source and destination data types must match.
            - Element size must be 4, 8, or 16 bytes.
            - Destination tensor must have a static layout.
            - `Fill.NAN` requires a floating-point dtype and 16-byte elements.

        Parameters:
            is_masked: Whether to perform a masked copy, where elements outside
                the `src_idx_bound` are not copied and are handled according to
                `fill` instead.
            swizzle: Optional swizzling function to rearrange the destination
                indices, which can improve memory access patterns.
            fill: What a masked copy does with the bytes it skips. `Fill.NONE`
                leaves them untouched, so the destination keeps whatever it
                already held. `Fill.ZERO` zeroes them, byte-granularly, so a
                partially valid element is part copy and part zero.
                `Fill.NAN` writes NaN, but only whole elements at a time: a
                partially valid element is filled rather than partly copied.
            eviction_policy: Cache eviction policy for the source data.

        Args:
            src: The source tensor to copy data from.
            src_idx_bound: For masked copies, the upper bound index for valid
                source elements.
            base_offset: Base offset for swizzling calculations.

        Example:

        ```mojo
        from layout import Idx, TileTensor, row_major
        from layout.tile_tensor import stack_allocation
        from max.gpu import thread_idx
        from max.gpu.memory import async_copy_commit_group, async_copy_wait_all
        from max.gpu.sync import barrier

        def kernel(src_ptr: MutPointer[Float32, MutAnyOrigin]):
            comptime thread_layout = row_major(Idx[2], Idx[2])

            var src = TileTensor(src_ptr, row_major[4, 4]())
            var smem = stack_allocation[
                dtype = DType.float32, address_space = .SHARED
            ](row_major[4, 4]())

            # Each of the 4 threads copies its own 2x2 fragment.
            var tid = thread_idx.x
            smem.distribute[thread_layout](tid).copy_from_async(
                src.distribute[thread_layout](tid)
            )
            async_copy_commit_group()
            async_copy_wait_all()
            barrier()
            # ... read the shared tile
        ```

        Performance:

        - Supports vectorized copies for 4, 8, or 16-byte elements for better
            throughput.
        - Can bypass L1 cache with appropriate eviction policies for specific
            access patterns.
        - Swizzling can improve memory access patterns and reduce bank
            conflicts.

        Notes:

        - Asynchronous copies allow computation to overlap with memory
            transfers.
        - A synchronization barrier is required before using the copied data.
        """
        comptime assert (
            Self.address_space == .SHARED
        ), "Async is only supported for destinations in shared memory"

        comptime assert (
            src.address_space == AddressSpace.GENERIC
            or src.address_space == AddressSpace.GLOBAL
        ), (
            "Async source must be in the generic or global address space;"
            " cp.async reads from global memory"
        )

        comptime assert (
            src.dtype == Self.dtype
        ), "src dtype must be the same as dst dtype."

        comptime assert (
            Self.element_size == src.element_size
        ), "copy_from_async should move data of the same element size"

        # Eligibility for 4, 8, 16 bytes async load.
        comptime element_size_bytes = size_of[Self.dtype]() * Self.element_size
        comptime assert element_size_bytes in (
            4,
            8,
            16,
        ), "copy_from_async only allows 4, 8, 16 bytes element"

        # The swizzle's `base` parameter sets how many least-significant bits
        # of the offset are kept constant. cp.async requires the destination
        # address to be aligned to `element_size` scalars, so the swizzle must
        # not permute bits below `log2(element_size)`.
        # `make_swizzle[..., access_size=element_size]` satisfies this; a
        # hand-rolled `Swizzle(bits, base, shift)` with
        # `base < log2_floor(element_size)` would silently produce misaligned
        # offsets and trigger CUDA_ERROR_MISALIGNED_ADDRESS.
        comptime if swizzle:
            comptime assert swizzle.value().base >= log2_floor(
                Self.element_size
            ), (
                "swizzle.base is too small for the requested element_size:"
                " cp.async would receive misaligned offsets. Construct the"
                " swizzle with `make_swizzle[..., access_size=element_size]`"
                " (or a hand-rolled `Swizzle(bits, base, shift)` with"
                " `base >= log2_floor(element_size)`)."
            )

        # Shared memory must always have a static layout.
        comptime assert (
            Self.LayoutType.all_dims_known
        ), "dst tensor must have static layout"

        comptime num_vecs = Self.LayoutType.static_product

        comptime assert (
            not src.LayoutType.shape_known
            or src.LayoutType.static_product == num_vecs
        ), "copy_from_async requires matching total element count"

        # The trailing `bitcast` materializes both pointee types as a concrete
        # `Scalar[Self.dtype]` so `async_copy`'s `dtype` parameter infers
        # cleanly; without it the inferred dtype is a comptime expression that
        # fails to unify across the two pointer arguments.
        var dst_ptr = (
            self.ptr.address_space_cast[.SHARED]()
            .unsafe_mut_cast[True]()
            .bitcast[Scalar[Self.dtype]]()
        )
        var src_ptr = src.ptr.address_space_cast[.GLOBAL]().bitcast[
            Scalar[Self.dtype]
        ]()

        comptime for i in range(num_vecs):
            var src_idx = src.layout[linear_idx_type=src.linear_idx_type](
                Idx[i]
            )
            var dst_idx = self.layout[linear_idx_type=Self.linear_idx_type](
                Idx[i]
            )

            var swizzled_idx: Scalar[Self.linear_idx_type]
            comptime if swizzle:
                comptime swizzle_fn = swizzle.value()
                # The destination address is swizzled in absolute tile
                # coordinates (`base_offset` + the in-window position), then
                # rebased back onto the fragment. Only the in-window position
                # is permuted, so the shuffle stays local to one window.
                var dst_idx_base = umod(
                    dst_idx, Scalar[Self.linear_idx_type](swizzle_fn.size())
                )
                swizzled_idx = (
                    swizzle_fn(base_offset + dst_idx_base)
                    + (dst_idx - dst_idx_base)
                    - base_offset
                )
            else:
                swizzled_idx = dst_idx

            comptime if is_masked:
                comptime fill_value = _async_fill_value[Self.dtype, fill]()
                var in_bounds = Bool(src_idx < src_idx_bound)

                comptime if not fill_value:
                    # `Fill.NONE`: leave the skipped bytes as the caller left
                    # them. `async_copy` only forwards `src_size` to the
                    # hardware when it has a fill value to write, so a
                    # zero-size copy would widen back to a full-width read of
                    # out-of-bounds source; skip the instruction instead.
                    if in_bounds:
                        async_copy[
                            element_size_bytes,
                            eviction_policy=eviction_policy,
                        ](src_ptr + src_idx, dst_ptr + swizzled_idx)
                elif fill_value.value() == 0:
                    # Zero fill is byte-granular: `cp.async` copies
                    # `src_size` bytes and zeroes the rest of the element.
                    async_copy[
                        element_size_bytes,
                        fill=fill_value,
                        eviction_policy=eviction_policy,
                    ](
                        src_ptr + src_idx,
                        dst_ptr + swizzled_idx,
                        Int32(element_size_bytes) if in_bounds else 0,
                    )
                else:
                    # A non-zero fill has no hardware path, so `cp.async`
                    # predicates between a whole-element copy and a
                    # whole-element store of the fill value. That makes it
                    # all-or-nothing per element rather than byte-granular,
                    # and 16-byte elements only.
                    #
                    # `cp.async` reads `predicate` and ignores `src_size`
                    # here; the AMD/Apple emulation does the reverse. Pass
                    # both so the element is either wholly copied or wholly
                    # filled on every target.
                    async_copy[
                        element_size_bytes,
                        fill=fill_value,
                        eviction_policy=eviction_policy,
                    ](
                        src_ptr + src_idx,
                        dst_ptr + swizzled_idx,
                        Int32(element_size_bytes) if in_bounds else 0,
                        predicate=in_bounds,
                    )
            else:
                async_copy[
                    element_size_bytes,
                    eviction_policy=eviction_policy,
                ](src_ptr + src_idx, dst_ptr + swizzled_idx)

    @__allow_legacy_custom_self_type
    def _distance(
        self: Self.Immut,
        other: TileTensor[
            mut=False, Self.dtype, _, address_space=Self.address_space, ...
        ],
    ) -> Scalar[Self.linear_idx_type]:
        """Calculate the element-wise distance between this tensor's storage
        and another tensor's storage.

        Computes the number of elements (not bytes) between this tensor's
        storage and `other`. Useful for determining offsets within a larger
        memory allocation.

        Args:
            other: The tensor to calculate the distance to.

        Returns:
            The number of elements between `self` and `other`.
        """
        # Storages are assumed copy-compatible: `other` may carry a different
        # (e.g. offset-derived) `Engine`, so reinterpret its handle as
        # `self`'s before measuring the scalar-element distance.
        return Scalar[Self.linear_idx_type](
            Self.Engine.distance(
                self._storage,
                rebind[type_of(self._storage)](other._storage),
            )
        )

    def write_to(self, mut w: Some[Writer]):
        """Format and write the tensor's contents to a writer.

        Uses bracket-delimited, comma-separated format. For 2D tensors,
        the output shows nested row structure. For other ranks, values are
        printed as a flat bracketed list in column-major coordinate order.

        Args:
            w: The writer instance to write the formatted output to.

        Example:

        ```mojo
        from layout import TileTensor
        from layout.tile_layout import row_major

        def main():
            var storage = Array[Float32, 4](uninitialized=True)
            var vec = TileTensor(storage, row_major[4]()).fill(1.0)
            print(vec)   # [1.0, 1.0, 1.0, 1.0]

            var storage2 = Array[Float32, 6](uninitialized=True)
            var mat = TileTensor(storage2, row_major[2, 3]()).fill(1.0)
            print(mat)   # [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        ```
        """

        if Int(self.layout.product()) == 0:
            return

        comptime if Self.flat_rank == 2:
            comptime assert Self.flat_rank == 2

            comptime if Self.static_shape[0] > -1 and Self.static_shape[1] > -1:
                _pretty_print_2d_tensor(self, w)
                return

        _pretty_print_elementwise(self, w)

    @inline(.nodebug)
    def tile[
        *tile_sizes: Int
    ](self, coordinates: Coord) -> Self.TileResultType[
        _IntToComptimeInt[*tile_sizes], linear_idx_type=Self.linear_idx_type
    ]:
        """Extract a sub-tile (CuTe `local_tile`). Works on both flat
        and nested parent layouts.

        On a flat parent, returns the rank-`Self.rank` sub-tile whose
        strides are the parent's strides and shape is `tile_sizes`. On
        a nested parent of shape `((outer_h, inner_h), (outer_w, inner_w))`,
        slices one outer index per mode and returns a flat rank-2
        sub-tile whose strides are each parent mode's innermost
        sub-strides.

        Parameters:
            tile_sizes: The dimensions of the tile along each axis.

        Args:
            coordinates: The tile coordinates as a `Coord`.

        Returns:
            A view into the original tensor representing the sub-tile.
        """
        comptime assert tile_sizes.size == Self.rank, String(
            t"tile requires exactly one tile size per tensor dimension; got"
            t" {tile_sizes.size} tile sizes for tensor of rank {Self.rank}"
        )
        return _tile(self, coord[*tile_sizes], coordinates)

    @inline(.nodebug)
    def tile[
        *tile_sizes: Int, stride_layout: TensorLayout
    ](self, coordinates: Coord) -> Self.OffsetViewType[
        TypeList.of[Scalar[Self.linear_idx_type]](),
        Layout[
            shape_types=_IntToComptimeInt[*tile_sizes],
            stride_types=stride_layout._shape_types,
        ],
    ]:
        """Tile with explicit static strides (flat parents only).

        Use when the parent tensor has dynamic (Scalar) strides but
        the actual stride values are known at compile time. This produces
        a tile with all_dims_known=True, enabling vectorize/distribute.

        This is needed because TensorLayout trait parameters erase concrete
        stride types -- the compiler cannot prove all_dims_known through
        a trait-bounded parameter even when the underlying strides are static.

        Parameters:
            tile_sizes: Tile dimensions along each axis.
            stride_layout: The layout providing static stride types.

        Args:
            coordinates: Tile coordinates in the grid.

        Returns:
            A view into the original tensor representing the specified tile.
        """
        comptime assert tile_sizes.size == Self.rank, String(
            t"tile requires exactly one tile size per tensor dimension; got"
            t" {tile_sizes.size} tile sizes for tensor of rank {Self.rank}"
        )
        comptime assert stride_layout.rank == Self.rank, String(
            t"stride_layout rank {stride_layout.rank} must match tensor rank"
            t" {Self.rank}"
        )
        return _tile[stride_layout=stride_layout](
            self, coord[*tile_sizes], coordinates
        )

    @inline(.nodebug)
    def tile[
        tile_shape_types: TypeList[Trait=CoordLike, ...],
        //,
    ](
        self, tile_shape: Coord[*tile_shape_types], coordinates: Coord
    ) -> Self.TileResultType[
        tile_shape_types, linear_idx_type=Self.linear_idx_type
    ]:
        """Extract a tile (sub-tensor) with shape specified as a Coord argument.

        This overload accepts the tile shape as a Coord value rather than
        compile-time Int parameters, enabling use cases where tile shapes
        are constructed programmatically or passed as values.

        Parameters:
            tile_shape_types: Types of the tile shape elements (inferred).

        Args:
            tile_shape: The dimensions of the tile as a Coord.
            coordinates: The tile coordinates as a Coord.

        Returns:
            A view into the original tensor representing the specified tile.

        Example:

        ```mojo
        from layout.tile_layout import row_major
        from layout import TileTensor
        from layout.coord import coord

        var storage = Array[Float32, 16](uninitialized=True)
        var tensor = TileTensor(storage, row_major[4, 4]()).fill(1.0)

        # Extract the tile at position (1, 0) with tile size 2x2
        var t = tensor.tile(coord[2, 2], coord[1, 0])
        ```
        """
        comptime assert tile_shape_types.length == Self.rank, String(
            t"tile_shape rank {tile_shape_types.length} must match tensor rank"
            t" {Self.rank}"
        )
        return _tile(self, tile_shape, coordinates)

    @inline(.nodebug)
    def tile_with_offset[
        *tile_sizes: Int
    ](self, coordinates: Coord) -> Tuple[
        Self.OffsetViewType[
            TypeList.of[Int](),
            Layout[
                shape_types=_IntToComptimeInt[*tile_sizes],
                stride_types=Self.LayoutType._stride_types,
            ],
        ],
        IndexList[coordinates.element_types.length],
        Int,
    ]:
        """Like tile(), but also returns corner coordinates and linear
        offset. Flat-layout parents only.

        Parameters:
            tile_sizes: Tile dimensions along each axis.

        Args:
            coordinates: Tile coordinates in the grid.

        Returns:
            Tuple of (tile, corner_coords, offset).
        """
        comptime assert tile_sizes.size == Self.rank, String(
            t"tile_with_offset requires one tile size per tensor dimension;"
            t" got {tile_sizes.size} tile sizes for tensor of rank {Self.rank}"
        )
        return _tile_with_offset(self, coord[*tile_sizes], coordinates)

    @inline(.nodebug)
    def tile_with_offset[
        *tile_sizes: Int, stride_layout: TensorLayout
    ](self, coordinates: Coord) -> Tuple[
        Self.OffsetViewType[
            TypeList.of[Int](),
            Layout[
                shape_types=_IntToComptimeInt[*tile_sizes],
                stride_types=stride_layout._shape_types,
            ],
        ],
        IndexList[coordinates.element_types.length],
        Int,
    ]:
        """Like tile(), but with explicit static strides. Flat-layout parents
        only.

        Use when the parent has dynamic strides but the values are known
        at compile time. See tile[stride_layout=...] for details.

        Parameters:
            tile_sizes: Tile dimensions along each axis.
            stride_layout: The layout providing static stride types.

        Args:
            coordinates: Tile coordinates in the grid.

        Returns:
            Tuple of (tile, corner_coords, offset).
        """
        comptime assert tile_sizes.size == Self.rank, String(
            t"tile_with_offset requires one tile size per tensor dimension;"
            t" got {tile_sizes.size} tile sizes for tensor of rank {Self.rank}"
        )
        comptime assert stride_layout.rank == Self.rank, String(
            t"stride_layout rank {stride_layout.rank} must match tensor rank"
            t" {Self.rank}"
        )
        return _tile_with_offset[stride_layout=stride_layout](
            self, coord[*tile_sizes], coordinates
        )

    comptime ViewType[new_layout: TensorLayout] = TileTensor[
        Self.dtype,
        LayoutType=new_layout,
        origin=Self.origin,
        Engine=Self.Engine,
        address_space=Self.address_space,
    ]
    """A TileTensor type with the same data properties but a different layout.

    Preserves dtype, origin, address_space, and other properties while
    replacing LayoutType. Use this to name the return type of reshape()
    and other layout-changing operations in helper functions.

    Parameters:
        new_layout: The new TensorLayout type for the view.
    """
    comptime OffsetViewType[
        offsets: TypeList[Trait=CoordLike, ...],
        LayoutType: TensorLayout = Self.LayoutType,
    ] = TileTensor[
        Self.dtype,
        LayoutType=LayoutType,
        origin=Self.origin,
        Engine=Self.Engine.OffsetResultType[offsets],
        address_space=Self.address_space,
    ]
    """The TileTensor type produced by offsetting into this tensor's storage.

    Names the return type of offset-producing operations (slicing, tiling,
    distribution). It preserves dtype, origin, and address_space, optionally
    changes the layout, and carries the engine's
    `OffsetResultType[offsets]` so an offset that yields a different storage
    handle is reflected in the view's type.

    Parameters:
        offsets: The coordinate types of the offset applied to the storage.
        LayoutType: The layout type of the resulting view. Defaults to this
            tensor's `LayoutType`.
    """

    @inline(.nodebug)
    def reshape[
        new_layout: TensorLayout,
    ](self, layout_val: new_layout) -> Self.ViewType[new_layout]:
        """Create a view of the tensor with a different layout.

        Returns a new TileTensor sharing the same pointer but with
        a different layout. This is a zero-cost operation -- only the
        layout type changes, no data is moved.

        Parameters:
            new_layout: The target layout type (inferred from layout_val).

        Args:
            layout_val: The layout instance to use for the new view.

        Returns:
            A TileTensor with the new layout viewing the same memory.
        """
        return {self._storage, layout_val}

    @inline(.nodebug)
    def transpose(
        self,
    ) -> Self.ViewType[
        Layout[
            Self.LayoutType._shape_types.reverse(),
            Self.LayoutType._stride_types.reverse(),
        ],
    ]:
        """Create a transposed view of the tensor.

        Returns a new TileTensor sharing the same pointer but with the
        layout dimensions reversed. For 2D tensors, this swaps rows and
        columns. This is a zero-cost operation -- no data is moved.

        Returns:
            A TileTensor with transposed layout viewing the same memory.
        """
        return {self._storage, self.layout.transpose()}

    # flatten_leading is defined as a standalone function below the
    # struct. As a method, Self.LayoutType._shape_types[i] in the return
    # type is symbolic and can't match value-level types. As a standalone
    # function, type_of(tensor).LayoutType resolves correctly.

    @inline(.nodebug)
    def tile[
        *tile_sizes: Int
    ](self, *tile_coords: Int) -> Self.TileResultType[
        _IntToComptimeInt[*tile_sizes], linear_idx_type=Self.linear_idx_type
    ]:
        """Variadic-`Int`-coords form of `.tile[]`. Works on both flat
        and nested parents: see the `Coord`-arg sibling above.

        Parameters:
            tile_sizes: The dimensions of each tile along each axis.

        Args:
            tile_coords: The coordinates of the specific tile to extract.

        Returns:
            A view into the original tensor representing the sub-tile.

        Example:

        ```mojo
        from layout.tile_layout import row_major
        from layout import TileTensor

        var storage = Array[Float32, 16](uninitialized=True)
        var tensor = TileTensor(storage, row_major[4, 4]()).fill(1.0)

        # Extract the tile at position (1, 0) with tile size 2x2
        var t = tensor.tile[2, 2](1, 0)
        ```
        """
        comptime assert tile_sizes.size == Self.rank, String(
            t"tile requires exactly one tile size per tensor dimension; got"
            t" {tile_sizes.size} tile sizes for tensor of rank {Self.rank}"
        )
        var coordinates = DynamicCoord[Self.linear_idx_type, Self.rank]()

        comptime for i in range(Self.rank):
            Pointer(to=coordinates[i]).write(
                rebind[coordinates.element_types[i]](
                    Scalar[Self.linear_idx_type](tile_coords[i])
                )
            )

        return _tile(self, coord[*tile_sizes], coordinates)

    @inline(.nodebug)
    def distribute[
        thread_layout: Layout,
        swizzle: Optional[Swizzle] = None,
    ](self, thread_id: Int) -> Self.OffsetViewType[
        TypeList.of[Scalar[Self.linear_idx_type]](),
        Layout[
            shape_types=_Divide[
                Self.LayoutType._shape_types,
                thread_layout.shape_types,
            ],
            stride_types=_Multiply[
                Self.LayoutType._stride_types,
                thread_layout.shape_types,
            ],
        ],
    ]:
        """Distribute tensor workload across multiple threads in a structured
        pattern.

        This method partitions a tensor across multiple threads for parallel
        processing, assigning each thread a specific portion of the tensor. The
        distribution pattern is determined by the thread_layout parameter,
        which defines the logical arrangement of threads.

        Parameters:
            thread_layout: Defines the logical arrangement of threads (e.g.,
                2x2 grid of 4 threads). This layout determines how the tensor is
                partitioned.
            swizzle: Optional. A function that remaps the distribution pattern
                to improve memory access patterns or cache locality.

        Args:
            thread_id: The ID of the current thread (0-based).

        Returns:
            A view into the original tensor representing the portion assigned to
            this thread.
        """
        return _distribute[thread_layout, swizzle](self, thread_id)

    @inline(.nodebug)
    def distribute_with_offset[
        thread_layout: Layout,
        swizzle: Optional[Swizzle] = None,
    ](self, thread_id: Int) -> Tuple[
        TileTensor[
            Self.dtype,
            origin=Self.origin,
            LayoutType=Layout[
                shape_types=_Divide[
                    Self.LayoutType._shape_types, thread_layout.shape_types
                ],
                stride_types=_Multiply[
                    Self.LayoutType._stride_types, thread_layout.shape_types
                ],
            ],
            Engine=Self.Engine.OffsetResultType[TypeList.of[Int]()],
            address_space=Self.address_space,
        ],
        IndexList[thread_layout.shape_types.length],
        Int,
    ]:
        """Like distribute(), but also returns thread coordinates and offset.

        Parameters:
            thread_layout: Defines the logical arrangement of threads.
            swizzle: Optional swizzle function.

        Args:
            thread_id: The ID of the current thread (0-based).

        Returns:
            Tuple of (distributed_tensor, thread_coords, offset).
        """
        return _distribute_with_offset[thread_layout, swizzle](self, thread_id)

    @inline(.always)
    def fill[
        *,
        use_runtime_layout: Bool = (
            not Self.all_dims_known
            or Coord[*Self.LayoutType._shape_types].static_product > BATCH_SIZE
        ),
    ](self, val: Scalar[Self.dtype]) -> Self where Self.mut:
        """Fill the entire tensor with a single value.

        This method sets all elements of the tensor to the specified value. It
        works with both statically and dynamically shaped tensors.

        For statically known layouts, the fill operation is unrolled at compile
        time. For dynamic layouts, a runtime loop is used. No vectorization is
        applied, so performance may be suboptimal for large tensors. Consider
        using hardware-specific fill operations for better performance with
        large tensors.

        This method can be used with tensors of any rank and shape. The
        fill operation respects the tensor's layout, filling all
        elements regardless of how they are arranged in memory. For
        tensors with `element_layout`, all elements within each logical element
        are filled with the same value.

        Parameters:
            use_runtime_layout: Whether to use the runtime layout for filling.
                This parameter is defaulted to `True` if the layout is not
                statically known. If loop bounds are too large, it's better to
                use the runtime layout to avoid long compilation time.

        Args:
            val: The value to fill the tensor with. Must be of the same data
                type as the tensor.

        Returns:
            The tensor itself (self), allowing for method chaining.

        Example:

        ```mojo
        from layout.tile_layout import row_major
        from layout import TileTensor

        def main() raises:
            var storage = Array[Float32, 3 * 4](uninitialized=True)
            var tensor = TileTensor(storage, row_major[3,4]()).fill(0.0)
            print(tensor)
        ```

        If not using method chaining, you can either reassign the result to the
        tensor variable, or assign the result to the discard pattern (`_`) to
        avoid warnings about an unused value:

        ```mojo
        from layout.tile_layout import row_major
        from layout import TileTensor

        var storage = Array[Float32, 3 * 4](uninitialized=True)
        var tensor = TileTensor(storage, row_major[3,4]()).fill(0.0)
        tensor = tensor.fill(0.0)
        # or
        _ = tensor.fill(0.0)
        ```
        """

        comptime if not use_runtime_layout:
            comptime num_elements = Coord[
                *Self.LayoutType._shape_types
            ].static_product

            # TODO: MSTDL-1352 we can use memory element to fill the tensor.
            comptime for i in range(num_elements):
                var idx = self.layout(Idx[i])
                self._store_storage[alignment=align_of[Self.dtype]()](idx, val)
        else:
            var num_elements = self.num_elements()

            for i in range(num_elements):
                var idx = self.layout(i)
                self._store_storage[alignment=align_of[Self.dtype]()](idx, val)
        return self

    @inline(.nodebug)
    def dim[i: Int](self) -> Scalar[Self.linear_idx_type]:
        """Returns the size of outer-mode dimension `i`.

        For a flat layout this is `shape[i]`. For a nested layout (where
        `shape[i]` is itself a `Coord`) this is the product of all leaf
        dims under outer-mode `i`: the i-th mode's extent under CuTe
        Layout Algebra. For shape `((a, b), (c, d))`: `dim[0] = a*b`,
        `dim[1] = c*d`.

        Parameters:
            i: The dimension index (compile-time constant).

        Returns:
            The product of all leaf dims under outer-mode `i`.
        """
        comptime assert 0 <= i < Self.rank, String(
            t"dim index {i} is out of bounds for tensor rank [0, {Self.rank})"
        )
        comptime if Self.LayoutType._shape_types[i].is_tuple:
            return Scalar[Self.linear_idx_type](
                self.layout.shape[i]().product()
            )
        else:
            return Scalar[Self.linear_idx_type](self.layout.shape[i]().value())

    @inline(.nodebug)
    def dim[
        IndexType: Indexer
    ](self, index: IndexType) -> Scalar[Self.linear_idx_type]:
        """Returns the size of the specified dimension.

        Parameters:
            IndexType: The type of the index argument.

        Args:
            index: The dimension index (runtime value).

        Returns:
            The size of the specified dimension as a scalar.
        """
        var idx = _index(index)

        comptime for i in range(Self.rank):
            if idx == i:
                return Scalar[Self.linear_idx_type](
                    self.layout.shape[i]().value()
                )
        # Should this raise instead?
        abort("attempt to dynamically index out of bounds")

    @inline(.nodebug)
    def dynamic_stride[
        IndexType: Indexer
    ](self, index: IndexType) -> Scalar[Self.linear_idx_type]:
        """Returns the stride of the specified dimension.

        Parameters:
            IndexType: The type of the index argument.

        Args:
            index: The dimension index (runtime value).

        Returns:
            The stride of the specified dimension as a scalar.
        """
        var idx = _index(index)

        comptime for i in range(Self.rank):
            if idx == i:
                return Scalar[Self.linear_idx_type](
                    self.layout.stride[i]().value()
                )
        # Should this raise instead?
        abort("attempt to dynamically index out of bounds")

    comptime SplitElementType[
        count: Int,
        axis: Int = 0,
    ] = TileTensor[
        Self.dtype,
        Layout[
            _StaticSplitShape[count, axis, Self.LayoutType._shape_types](),
            Self.LayoutType._stride_types,
        ],
        ImmOrigin(Self.origin),
        Engine=Self.Engine.OffsetResultType[TypeList.of[Int]()],
        address_space=Self.address_space,
        linear_idx_type=Self.linear_idx_type,
    ]
    """Type alias for equal-sized split element tensors.

    The result has an immutable origin.

    Parameters:
        count: The number of equal-sized partitions.
        axis: The axis along which the tensor is split.
    """

    comptime StaticSplitType[
        count: Int,
        axis: Int = 0,
    ] = StaticTuple[
        Self.SplitElementType[count, axis],
        count,
    ]
    """Type alias for static split result tuples.

    Each tuple element is an immutable view.

    Parameters:
        count: The number of equal-sized partitions.
        axis: The axis along which the tensor is split.
    """

    @inline(.nodebug)
    def split[
        count: Int,
        axis: Int = 0,
    ](self) -> Self.StaticSplitType[count, axis] where not Self.mut:
        """Splits the tensor into equal-sized views along an axis.

        Split views are immutable. Call `as_immut().split[count]()` on a
        mutable tensor before splitting.

        Parameters:
            count: The number of partitions to split into.
            axis: The axis along which to split.

        Constraints:
            The tensor shape must be statically known. The split axis must
            have static, scalar shape and stride values. The split-axis shape
            must be evenly divisible by `count`.

        Returns:
            A `StaticTuple` containing `count` non-overlapping `TileTensor`
            views into this tensor.

        See also:
            Use `split(count, idx)` to return a single partition with a
            runtime-sized split axis. The dynamic overload takes `axis` before
            `split_alignment` as compile-time parameters, while this overload
            takes `count` before `axis`.
        """
        comptime assert (
            axis >= 0 and axis < Self.rank
        ), "TileTensor.split axis out of bounds"
        comptime assert (
            Self.LayoutType.shape_known
        ), "TileTensor.split[count]() requires statically known shapes"
        comptime assert Self.LayoutType._shape_types[
            axis
        ].is_value, "TileTensor.split only supports scalar dimensions"
        comptime assert (
            Self.LayoutType._shape_types[axis].is_static_value
            and Self.LayoutType._stride_types[axis].is_static_value
        ), "TileTensor.split requires static shape and stride on the split axis"
        comptime assert (
            Self.LayoutType._shape_types[axis].static_value % count == 0
        ), "The input dimension must be divisible by the input count"

        comptime tile_size = (
            Self.LayoutType._shape_types[axis].static_value // count
        )
        comptime axis_stride = Self.LayoutType._stride_types[axis].static_value

        var tiles = Self.StaticSplitType[count, axis]()
        var split_layout = self._split_layout[count, axis]()

        comptime for i in range(count):
            tiles[i] = Self.SplitElementType[count, axis](
                Self.Engine.offset(
                    self._unsafe_storage_cast[to_mut=False](),
                    Coord(i * tile_size * axis_stride),
                ),
                split_layout,
            )

        return tiles

    comptime DynamicSplitType[
        axis: Int = 0,
    ] = TileTensor[
        Self.dtype,
        Layout[
            _DynamicSplitShape[
                Self.linear_idx_type, axis, Self.LayoutType._shape_types
            ](),
            Self.LayoutType._stride_types,
        ],
        ImmOrigin(Self.origin),
        Engine=Self.Engine.OffsetResultType[TypeList.of[Int]()],
        address_space=Self.address_space,
        linear_idx_type=Self.linear_idx_type,
    ]
    """Type alias for runtime-sized split element tensors.

    The result has an immutable origin.

    Parameters:
        axis: The axis along which the tensor is split.
    """

    @inline(.nodebug)
    def split[
        axis: Int = 0,
        split_alignment: Int = 1,
    ](self, count: Int, idx: Int) -> Self.DynamicSplitType[
        axis
    ] where not Self.mut:
        """Returns one partition of the tensor after splitting along an axis.

        The returned partition is immutable. Call
        `as_immut().split(count, idx)` on a mutable tensor before splitting.

        The base partition size is `align_up(ceildiv(axis_dim, count),
        split_alignment)`. This can make the first `count - 1` partitions
        larger than `ceildiv(axis_dim, count)`; each returned view is clamped
        to the remaining elements. If the aligned partition offsets exhaust
        the axis before all `count` partitions are assigned, trailing
        partitions have size 0.

        Parameters:
            axis: The axis along which to split.
            split_alignment: Alignment for the partition size.

        Args:
            count: The number of partitions.
            idx: The partition index to return.

        Returns:
            An immutable `TileTensor` view whose split axis has runtime shape.

        See also:
            Use `split[count]()` to split into a `StaticTuple` of equal-sized
            views when the partition count is known at compile time.
        """
        comptime assert (
            axis >= 0 and axis < Self.rank
        ), "TileTensor.split axis out of bounds"
        comptime assert Self.LayoutType._shape_types[
            axis
        ].is_value, "TileTensor.split only supports scalar dimensions"
        comptime assert (
            Self.LayoutType._shape_types[axis].is_static_value
            and Self.LayoutType._stride_types[axis].is_static_value
        ), "TileTensor.split requires static shape and stride on the split axis"
        debug_assert(count > 0, "split requires count > 0")
        debug_assert(idx >= 0 and idx < count, "split idx out of range")

        comptime axis_dim = Self.LayoutType._shape_types[axis].static_value
        comptime axis_stride = Self.LayoutType._stride_types[axis].static_value

        var axis_partition_dim = align_up(
            ceildiv(axis_dim, count), split_alignment
        )
        var raw_remaining = axis_dim - idx * axis_partition_dim
        var partition_dim = max(0, min(axis_partition_dim, raw_remaining))
        comptime NewShapeTypes = _DynamicSplitShape[
            Self.linear_idx_type, axis, Self.LayoutType._shape_types
        ]()
        var new_shape = Coord[*NewShapeTypes]()

        comptime for i in range(Self.rank):
            comptime NewShapeType = NewShapeTypes[i]
            comptime if i == axis:
                Pointer(to=new_shape[i]).write(
                    rebind[NewShapeType](
                        Scalar[Self.linear_idx_type](partition_dim)
                    )
                )
            else:
                Pointer(to=new_shape[i]).write(
                    rebind[NewShapeType](self.layout.shape[i]())
                )

        return Self.DynamicSplitType[axis](
            Self.Engine.offset(
                self._unsafe_storage_cast[to_mut=False](),
                Coord(idx * axis_partition_dim * axis_stride),
            ),
            Layout(new_shape, self.layout.stride_coord()),
        )

    @inline(.nodebug)
    def _split_layout[
        count: Int,
        axis: Int = 0,
    ](self) -> Layout[
        _StaticSplitShape[count, axis, Self.LayoutType._shape_types](),
        Self.LayoutType._stride_types,
    ]:
        comptime NewShapeTypes = _StaticSplitShape[
            count, axis, Self.LayoutType._shape_types
        ]()
        var new_shape = Coord[*NewShapeTypes]()

        comptime for i in range(Self.rank):
            comptime NewShapeType = NewShapeTypes[i]
            Pointer(to=new_shape[i]).write(NewShapeType())

        return Layout(new_shape, self.layout.stride_coord())

    @inline(.always)
    def slice[
        *slices: _IndexOrSlice[]
    ](self) -> Self.OffsetViewType[
        TypeList.of[_SlicedOffset[slices, Self.LayoutType]](),
        _SlicedLayout[slices, Self.LayoutType],
    ] where (slices.size == Self.flat_rank and Self.all_dims_known):
        """Extract a view of the tensor, fixing or subslicing each dimension.

        Each parameter is either an `Int`, which fixes that dimension to the
        given index and drops it from the result (rank reduction, like
        `squeeze`), or a slice literal selecting a rank-preserving subrange --
        e.g. `t.slice[0:5, 2]()`. The output rank equals the number of slice
        parameters. Unlike `tile`, whose coordinate indexes a grid of
        tile-sized blocks, slice bounds are element offsets, so a view need
        not be aligned to its own extent (e.g. the shorter trailing tile of a
        `tile_iterator` walk). The bounds must be compile-time values: they are
        folded into the view's storage handle as a `ComptimeInt` offset, so a
        view of a fully static tensor stays fully static.

        Parameters:
            slices: One `Int` or slice literal per tensor dimension.

        Returns:
            A strided sub-view over the same backing storage. Its extents are
            fresh `ComptimeInt`s, since a sliced extent is a new compile-time
            value rather than the parent's; its strides are the surviving
            axes' own.

        Example:

        ```mojo
        from layout.tile_layout import row_major
        from layout import TileTensor

        comptime layout_3d = row_major[16, 16, 16]()
        var stack = Array[UInt8, layout_3d.static_product](fill=0)
        var tensor_3d = TileTensor(stack, layout_3d)

        # Rank-preserving: a 2x2x4 view.
        var sub = tensor_3d.slice[0:2, 1:3, 0:4]()

        # Rank-reducing: plane 3, then a 2x4 view of it.
        var plane = tensor_3d.slice[3, 1:3, 0:4]()
        ```

        Performance:

        - Creates a view without copying data, making it very efficient.
        - Maintains the original tensor's stride information for efficient
            memory access.
        - Free at runtime: the whole view, offset included, is computed in
            the type system and emits no index arithmetic.

        Notes:

        - The slice is a view into the original tensor, so modifications to
            the slice will affect the original tensor.
        - The step size must be 1 for all dimensions (no gaps allowed).
        - Slice bounds are checked at compile time against the parent's static
            shape; out-of-range bounds are a compile error, not a runtime one.
        """
        comptime for i in range(slices.size):
            comptime if slices[i].is_slice():
                comptime s = slices[i].unsafe_get_slice()
                comptime start = s.start.or_else(0)
                comptime end = s.end.or_else(Self.static_shape[i])
                comptime assert (
                    start >= 0 and end >= start and end <= Self.static_shape[i]
                ), "`slice`: slice bounds out of range"
            else:
                comptime idx = slices[i].unsafe_get_index()
                comptime assert (
                    idx >= 0 and idx < Self.static_shape[i]
                ), "`slice`: fixed index out of range"

        return {
            self._offset_storage(_SlicedOffset[slices, Self.LayoutType]()),
            _SlicedLayout[slices, Self.LayoutType](),
        }

    # ===------------------------------------------------------------------=== #
    # Vectorization
    # ===------------------------------------------------------------------=== #

    comptime VectorizedType[*vector_shape: Int] = TileTensor[
        Self.dtype,
        origin=Self.origin,
        LayoutType=Layout[
            shape_types=_CeilDiv[
                Self.LayoutType._shape_types,
                _IntToComptimeInt[*vector_shape],
            ],
            stride_types=_Multiply[
                Self.LayoutType._stride_types, _IntToComptimeInt[*vector_shape]
            ],
        ],
        Engine=DefaultEngine[
            element_width=Coord[
                *_IntToComptimeInt[*vector_shape]
            ].static_product
        ],
        address_space=Self.address_space,
        linear_idx_type=Self.linear_idx_type,
    ]
    """Type alias for vectorized tensor types.

    Parameters:
        vector_shape: The shape of each vector unit along each axis.
    """

    comptime SIMDVectorizedType = Self.VectorizedType[
        1, simd_width_of[Self.dtype]()
    ]
    """Result type for SIMD-width vectorization."""

    @inline(.nodebug)
    def vectorize[
        *vector_shape: Int
    ](self) -> Self.VectorizedType[*vector_shape]:
        """Reshape a tensor into a vectorized form for efficient SIMD operations.

        This method transforms the tensor's logical layout to enable efficient
        vectorized processing, treating blocks of elements as vector units. The
        transformation is particularly useful for SIMD (Single Instruction
        Multiple Data) operations and hardware acceleration.

        The vector shape is tracked in `element_size`.

        Parameters:
            vector_shape: The dimensions of each vector unit along each axis of
                the tensor. For example, in a 2D tensor, `vectorize[4, 4]` treats
                4x4 blocks as vector units.

        Returns:
            A view of the tensor with a vectorized layout, where each element in
            the resulting tensor represents the start of a vector block from the
            original tensor. The element layout is tracked via
            `element_size` (the vector shape).

        Example:

        For a 16x16 tensor, `vectorize[4, 4]` will produce a 4x4 tensor
        where each element position is the starting point of a 4x4 block
        from the original tensor. The strides are scaled by the vector shape
        so that adjacent elements in the vectorized tensor are spaced apart
        by the vector dimensions.

        Performance:

        - Creates a view without copying data, making it very efficient.
        - Enables strided access patterns suitable for SIMD vector loads.
        - Zero-cost abstraction at compile time when used with static shapes.
        """
        comptime assert (
            Self.Engine == DefaultEngine[element_width=1]
            or Self.Engine == DevicePointerEngine[element_width=1]
        ), "TileTensor.vectorize requires DefaultEngine or DevicePointerEngine"

        return _vectorize(self, coord[*vector_shape])

    @inline(.nodebug)
    def vectorize(self) -> Self.VectorizedType[1, simd_width_of[Self.dtype]()]:
        """Return a SIMD-width vectorized view of this tensor.

        This is a convenience method that vectorizes along the last dimension
        by the SIMD width for the tensor's dtype.

        Returns:
            A `Self.VectorizedType[1, simd_width_of[Self.dtype]()]` view whose
            last dimension stride equals the SIMD width for the tensor's dtype.
        """
        return self.vectorize[1, simd_width_of[Self.dtype]()]()

    # ===------------------------------------------------------------------=== #
    # Coalescing
    # ===------------------------------------------------------------------=== #

    comptime CoalescedType = Self.ViewType[
        Layout[
            shape_types=Coord[
                ComptimeInt[
                    Coord[*Self.LayoutType._shape_types].static_product
                ],
            ].element_types,
            stride_types=Coord[ComptimeInt[1]].element_types,
        ],
    ]
    """Type alias for coalesced (flattened to rank-1) tensor types.

    The coalesced tensor has:
    - shape: product of all original dimensions
    - stride: 1 (contiguous)
    - element shape: product of all original element dimensions
    - element stride: 1 (contiguous)
    """

    comptime is_row_major = _IsRowMajor[
        Self.LayoutType._shape_types, Self.LayoutType._stride_types
    ]
    """True if the tensor has row-major (contiguous) strides."""

    # ===------------------------------------------------------------------=== #
    # Reshape
    # ===------------------------------------------------------------------=== #

    comptime ReshapedType[*new_shape_types: CoordLike] = Self.ViewType[
        Layout[
            shape_types=new_shape_types,
            stride_types=_RowMajor[*new_shape_types],
        ],
    ]
    """Type alias for reshaped tensor types.

    Parameters:
        new_shape_types: The shape types for the reshaped tensor.
    """

    @inline(.nodebug)
    def reshape[
        *new_shape: Int
    ](self) -> Self.ReshapedType[*_IntToComptimeInt[*new_shape]] where (
        Self.all_dims_known
        and Self.is_row_major
        and Coord[*Self.LayoutType._shape_types].static_product
        == Coord[*_IntToComptimeInt[*new_shape]].static_product
    ):
        """Reshape the tensor to a new shape with compile-time dimensions.

        This method creates a view of the tensor with a different logical shape
        while preserving the underlying data. The total number of elements must
        remain the same, and the tensor must have row-major (contiguous) strides.

        Parameters:
            new_shape: The new shape dimensions as compile-time integers.

        Returns:
            A TileTensor with the new shape and row-major strides, sharing
            the same underlying data as the original tensor.

        Constraints:
            - All dimensions must be statically known (`all_dims_known`).
            - The tensor must have row-major strides (`is_row_major`).
            - The product of the new shape must equal the product of the
              original shape.

        Example:

        ```mojo
        from layout.tile_layout import row_major
        from layout import TileTensor

        var storage = Array[Float32, 12](uninitialized=True)
        var tensor = TileTensor(storage, row_major[3, 4]()).fill(1.0)
        # tensor has shape (3, 4)

        var reshaped = tensor.reshape[2, 6]()
        # reshaped has shape (2, 6), same underlying data

        var reshaped_1d = tensor.reshape[12]()
        # reshaped_1d has shape (12,), equivalent to coalesce
        ```

        Performance:

        - Creates a view without copying data.
        - Zero-cost abstraction at compile time when used with static shapes.
        """
        comptime NewShapeTypes = _IntToComptimeInt[*new_shape]
        comptime NewStrideTypes = _RowMajor[*NewShapeTypes]

        var new_layout = Layout(
            Coord[*NewShapeTypes](),
            Coord[*NewStrideTypes](),
        )

        return Self.ReshapedType[*NewShapeTypes](self._storage, new_layout)

    @inline(.nodebug)
    def reshape[
        *new_shape_types: CoordLike
    ](self, new_shape: Coord[*new_shape_types]) -> Self.ReshapedType[
        *new_shape_types
    ] where Self.is_row_major:
        """Reshape the tensor to a new shape specified as a Coord.

        This method creates a view of the tensor with a different logical shape
        while preserving the underlying data. The total number of elements must
        remain the same, and the tensor must have row-major (contiguous) strides.

        This overload accepts shapes with runtime dimensions, performing the
        element count validation at runtime when needed.

        Parameters:
            new_shape_types: The types of the new shape dimensions (inferred).

        Args:
            new_shape: The new shape as a Coord.

        Returns:
            A TileTensor with the new shape and row-major strides, sharing
            the same underlying data as the original tensor.

        Constraints:
            - The tensor must have row-major strides (`is_row_major`).
            - The product of the new shape must equal the product of the
              original shape (validated at runtime for dynamic shapes).

        Example:

        ```mojo
        from layout.tile_layout import row_major
        from layout import TileTensor
        from layout import Idx, Coord

        var storage = Array[Float32, 12](uninitialized=True)
        var tensor = TileTensor(storage, row_major[3, 4]()).fill(1.0)

        # Reshape with runtime-determined dimensions
        var rows = 2
        var cols = 6
        var reshaped = tensor.reshape(Coord(rows, cols))
        ```

        Performance:

        - Creates a view without copying data.
        - May include runtime validation for dynamic shapes.
        """
        # Runtime validation for element count
        assert self.num_elements() == Int(
            new_shape.product()
        ), "reshape: total number of elements must match"

        var new_layout = row_major(new_shape)

        return Self.ReshapedType[*new_shape_types](self._storage, new_layout)

    @inline(.nodebug)
    def coalesce(
        self,
    ) -> Self.CoalescedType where Self.all_dims_known and Self.is_row_major:
        """Creates a rank-1 tensor by flattening all dimensions.

        Coalescing combines all dimensions into a single contiguous dimension.
        This is useful for operations that need to iterate over all elements
        sequentially.

        Returns:
            A rank-1 tensor with shape equal to the product of all original
            dimensions and stride 1. Element layout is also coalesced.

        Constraints:
            All dimensions must be statically known (`all_dims_known`).
            The tensor must have row-major (contiguous) strides (`is_row_major`).

        Example:

        For a 4x4 tensor, `coalesce()` produces a 16-element rank-1 tensor.
        For a vectorized tensor with shape (4, 4) and element shape (4, 4),
        coalescing produces shape (16,) with element shape (16,).

        Performance:

        - Creates a view without copying data.
        - Enables simple sequential iteration over all elements.
        - Zero-cost abstraction at compile time.
        """
        comptime total_size = Coord[
            *Self.LayoutType._shape_types
        ].static_product

        var new_layout = Layout(
            Coord(ComptimeInt[total_size]()),
            Coord(ComptimeInt[1]()),
        )

        return Self.CoalescedType(self._storage, new_layout)

    comptime DynamicType[dyn_dtype: DType] = Self.ViewType[
        Layout[
            shape_types=_CoordToDynamic[
                dyn_dtype, Self.LayoutType._shape_types
            ],
            stride_types=_CoordToDynamic[
                dyn_dtype, Self.LayoutType._stride_types
            ],
        ],
    ]
    """Type alias for dynamic tensor types.

    Parameters:
        dyn_dtype: The data type for Scalar values in the dynamic tensor.
    """

    @inline(.nodebug)
    def make_dynamic[dyn_dtype: DType](self) -> Self.DynamicType[dyn_dtype]:
        """Convert all elements in shape and stride to Scalar[dyn_dtype].

        Parameters:
            dyn_dtype: The data type for the resulting Scalar values.

        Returns:
            A new TileTensor where all elements in shape and stride
            are converted to Scalar[dyn_dtype].

        Examples:
            ```mojo
            from layout import TileTensor
            from layout.tile_layout import row_major
            var storage = Array[Float32, 12](uninitialized=True)
            var tensor = TileTensor(Span(storage), row_major[3, 4]())
            var dynamic = tensor.make_dynamic[.int64]()
            # dynamic has Int64 for all shape/stride dimensions
            ```
        """
        return Self.DynamicType[dyn_dtype](
            self._storage,
            Layout(
                self.layout.shape_coord(), self.layout.stride_coord()
            ).make_dynamic[dyn_dtype](),
        )

    @inline(.nodebug)
    def to_layout_tensor(
        self,
        out result: LayoutTensor[
            Self.dtype,
            _LegacyLayout(
                coord_to_int_tuple[*Self.LayoutType._shape_types](),
                coord_to_int_tuple[*Self.LayoutType._stride_types](),
            ),
            Self.origin,
            address_space=Self.address_space,
        ],
    ):
        """Return a LayoutTensor with the same shape, stride, and address space
        of this tensor.

        This is a utility to help with porting LayoutTensor methods to this type.

        Supports `DefaultEngine` and `DevicePointerEngine`-backed tiles. For a
        `DevicePointerEngine`-backed tile the raw device pointer is recovered
        from the handle (via `Engine.unsafe_ptr`), so the resulting
        `LayoutTensor` no longer carries the owning `DevicePointer`. This is a
        temporary workaround until `LayoutTensor` support is removed as part of
        GPUA-6.

        Returns:
            A LayoutTensor with the same shape, stride, and address space of
            this tensor.
        """
        comptime assert (
            Self.Engine == DefaultEngine[element_width=1]
            or Self.Engine == DevicePointerEngine[element_width=1]
        ), (
            "TileTensor.to_layout_tensor requires DefaultEngine or"
            " DevicePointerEngine"
        )
        return {
            self.ptr,
            type_of(result.runtime_layout)(
                # A `RuntimeTuple` stores one entry per leaf, so a nested mode
                # has to be flattened to supply them in the order it expects.
                # `flatten()` is the identity on a flat `Coord`.
                coord_to_index_list(self.layout.shape_coord().flatten()).cast[
                    result.layout_int_type
                ](),
                coord_to_index_list(self.layout.stride_coord().flatten()).cast[
                    result.linear_idx_type
                ](),
            ),
        }

    comptime Immut = Self.OriginCastType[ImmOrigin(Self.origin)]
    """Type alias for an immutably-casted tensor."""

    comptime OriginCastType[
        mut: Bool,
        //,
        origin: Origin[mut=mut],
    ] = TileTensor[
        Self.dtype,
        Self.LayoutType,
        origin,
        Engine=Self.Engine,
        address_space=Self.address_space,
        linear_idx_type=Self.linear_idx_type,
    ]
    """Type alias for origin-cast result tensors.

    Parameters:
        mut: Whether the result tensor is mutable.
        origin: The origin for the result tensor.
    """

    @inline(.nodebug)
    def as_unsafe_any_origin(
        self,
    ) -> Self.OriginCastType[UnsafeAnyOrigin[mut=Self.mut]]:
        """Casts the origin of the `TileTensor` to `UnsafeAnyOrigin`.

        Returns:
            A tensor with the origin set to `UnsafeAnyOrigin`.

        Safety:

        It is **always** preferred to maintain a concrete origin values instead of
        using `UnsafeAnyOrigin`. Casting to `UnsafeAnyOrigin` is an inherently unsafe
        operation that will silently extend unrelated lifetimes and turn off
        exclusivity checking.
        """
        return {
            self._unsafe_storage_cast[
                to_origin=UnsafeAnyOrigin[mut=Self.mut]
            ](),
            self.layout,
        }

    @doc_hidden
    @inline(.nodebug)
    @deprecated(use=as_unsafe_any_origin)
    def as_any_origin(self) -> Self.OriginCastType[AnyOrigin[mut=Self.mut]]:
        return self.as_unsafe_any_origin()

    @inline(.always)
    def as_immut(
        self,
    ) -> Self.OriginCastType[ImmOrigin(Self.origin)]:
        """
        Return an immutable version of this tensor.

        Returns:
            A `LayoutTensor` covering the same elements, but without mutability.
        """
        return {
            self._unsafe_storage_cast[to_origin=ImmOrigin(Self.origin)](),
            self.layout,
        }

    comptime AddressSpaceCastType[address_space: AddressSpace] = TileTensor[
        Self.dtype,
        origin=Self.origin,
        LayoutType=Self.LayoutType,
        Engine=Self.Engine,
        address_space=address_space,
        linear_idx_type=Self.linear_idx_type,
    ]
    """Type alias for address-space-cast result tensors.

    Parameters:
        address_space: The address_space for the result tensor.
    """

    @inline(.always)
    def address_space_cast[
        target_address_space: AddressSpace
    ](self,) -> Self.AddressSpaceCastType[target_address_space]:
        """Return a version of this tensor cast to a new address space.

        Parameters:
            target_address_space: The target address space to cast to.

        Returns:
            A TileTensor covering the same elements in the new address space.
        """
        return self.unsafe_address_space_cast[target_address_space]()

    @inline(.always)
    def unsafe_address_space_cast[
        target_address_space: AddressSpace
    ](self) -> Self.AddressSpaceCastType[target_address_space]:
        """Return a version of this tensor cast to a new address space.

        Parameters:
            target_address_space: The target address space to cast to.

        Returns:
            A TileTensor covering the same elements in the new address space.
        """
        return {
            self._unsafe_storage_cast[
                to_origin=Self.origin, to_address_space=target_address_space
            ](),
            self.layout,
        }

    @inline(.always)
    def to_device_buffer(self, ctx: DeviceContext) -> DeviceBuffer[Self.dtype]:
        """Convert the tensor to a `DeviceBuffer`.

        Works for tensors backed by either `DefaultEngine` or
        `DevicePointerEngine`. In both cases the base pointer is recovered
        through the engine (`self.ptr`), so the resulting non-owning
        `DeviceBuffer` covers exactly this tensor's elements, honoring any
        offset baked into the storage handle.

        Args:
            ctx: The device context to use.

        Returns:
            A `DeviceBuffer` containing the tensor's data.
        """
        comptime assert (
            Self.Engine == DefaultEngine[element_width=1]
            or Self.Engine == DevicePointerEngine[element_width=1]
        ), (
            "TileTensor.to_device_buffer requires DefaultEngine or"
            " DevicePointerEngine"
        )
        comptime assert (
            Self.address_space == Self.address_space.GENERIC
        ), "DeviceBuffer is only used on GENERIC address space"
        return DeviceBuffer[Self.dtype](
            ctx,
            self.ptr,
            self.num_elements(),
            owning=False,
        )

    @inline(.always)
    def __iadd__(
        self, rhs: TileTensor[Self.dtype, ...]
    ) where Self.mut and conforms_to(Self.Engine, TensorOps):
        """Adds `rhs` into this tensor elementwise, in place.

        Args:
            rhs: The tensor to add, broadcast against this tensor's layout.
        """
        comptime assert (
            Self.Engine._BASE_TYPE_NAME == rhs.Engine._BASE_TYPE_NAME
        ), "in-place binary ops require operands with the same engine"
        comptime assert (
            self.element_size == rhs.element_size
        ), "in-place binary ops require operands with the same element size"
        Self.Engine.iadd(
            (self._unsafe_storage_cast[to_mut=True](), self.layout),
            (rhs._storage, rhs.layout),
        )

    @inline(.always)
    def __imul__(
        self, rhs: TileTensor[Self.dtype, ...]
    ) where Self.mut and conforms_to(Self.Engine, TensorOps):
        """Multiplies this tensor by `rhs` elementwise, in place.

        Args:
            rhs: The tensor to multiply by, broadcast against this tensor's
                layout.
        """
        comptime assert (
            Self.Engine._BASE_TYPE_NAME == rhs.Engine._BASE_TYPE_NAME
        ), "in-place binary ops require operands with the same engine"
        comptime assert (
            self.element_size == rhs.element_size
        ), "in-place binary ops require operands with the same element size"
        Self.Engine.imul(
            (self._unsafe_storage_cast[to_mut=True](), self.layout),
            (rhs._storage, rhs.layout),
        )

    @inline(.always)
    def __isub__(
        self, rhs: TileTensor[Self.dtype, ...]
    ) where Self.mut and conforms_to(Self.Engine, TensorOps):
        """Subtracts `rhs` from this tensor elementwise, in place.

        Args:
            rhs: The tensor to subtract, broadcast against this tensor's layout.
        """
        comptime assert (
            Self.Engine._BASE_TYPE_NAME == rhs.Engine._BASE_TYPE_NAME
        ), "in-place binary ops require operands with the same engine"
        comptime assert (
            self.element_size == rhs.element_size
        ), "in-place binary ops require operands with the same element size"
        Self.Engine.isub(
            (self._unsafe_storage_cast[to_mut=True](), self.layout),
            (rhs._storage, rhs.layout),
        )

    @inline(.always)
    def __ifloordiv__(
        self, rhs: TileTensor[Self.dtype, ...]
    ) where Self.mut and conforms_to(Self.Engine, TensorOps):
        """Floor-divides this tensor by `rhs` elementwise, in place.

        Args:
            rhs: The tensor to floor-divide by, broadcast against this tensor's
                layout.
        """
        comptime assert (
            Self.Engine._BASE_TYPE_NAME == rhs.Engine._BASE_TYPE_NAME
        ), "in-place binary ops require operands with the same engine"
        comptime assert (
            self.element_size == rhs.element_size
        ), "in-place binary ops require operands with the same element size"
        Self.Engine.ifloordiv(
            (self._unsafe_storage_cast[to_mut=True](), self.layout),
            (rhs._storage, rhs.layout),
        )

    @inline(.always)
    def __itruediv__(
        self, rhs: TileTensor[Self.dtype, ...]
    ) where Self.mut and conforms_to(Self.Engine, TensorOps):
        """True-divides this tensor by `rhs` elementwise, in place.

        Args:
            rhs: The tensor to true-divide by, broadcast against this tensor's
                layout.
        """
        comptime assert (
            Self.Engine._BASE_TYPE_NAME == rhs.Engine._BASE_TYPE_NAME
        ), "in-place binary ops require operands with the same engine"
        comptime assert (
            self.element_size == rhs.element_size
        ), "in-place binary ops require operands with the same element size"
        Self.Engine.itruediv(
            (self._unsafe_storage_cast[to_mut=True](), self.layout),
            (rhs._storage, rhs.layout),
        )

    @inline(.always)
    def min(
        self, rhs: TileTensor[Self.dtype, ...]
    ) where Self.mut and conforms_to(Self.Engine, TensorOps):
        """Takes the elementwise minimum with `rhs`, in place.

        Args:
            rhs: The tensor to take the minimum against, broadcast against this
                tensor's layout.
        """
        comptime assert (
            Self.Engine._BASE_TYPE_NAME == rhs.Engine._BASE_TYPE_NAME
        ), "in-place binary ops require operands with the same engine"
        comptime assert (
            self.element_size == rhs.element_size
        ), "in-place binary ops require operands with the same element size"
        Self.Engine.imin(
            (self._unsafe_storage_cast[to_mut=True](), self.layout),
            (rhs._storage, rhs.layout),
        )

    @inline(.always)
    def max(
        self, rhs: TileTensor[Self.dtype, ...]
    ) where Self.mut and conforms_to(Self.Engine, TensorOps):
        """Takes the elementwise maximum with `rhs`, in place.

        Args:
            rhs: The tensor to take the maximum against, broadcast against this
                tensor's layout.
        """
        comptime assert (
            Self.Engine._BASE_TYPE_NAME == rhs.Engine._BASE_TYPE_NAME
        ), "in-place binary ops require operands with the same engine"
        comptime assert (
            self.element_size == rhs.element_size
        ), "in-place binary ops require operands with the same element size"
        Self.Engine.imax(
            (self._unsafe_storage_cast[to_mut=True](), self.layout),
            (rhs._storage, rhs.layout),
        )

    @inline(.always)
    def abs(self) where Self.mut and conforms_to(Self.Engine, TensorOps):
        """Takes the elementwise absolute value of this tensor, in place.

        For unsigned dtypes this is the identity.
        """
        Self.Engine.iabs(self._unsafe_storage_cast[to_mut=True](), self.layout)

    @inline(.always)
    def recip(self) where Self.mut and conforms_to(Self.Engine, TensorOps):
        """Replaces each element of this tensor with its reciprocal, in place.

        Elements equal to zero produce infinity, following IEEE 754 division
        semantics.

        Constraints:
            The tensor's dtype must be a floating-point type.
        """
        Self.Engine.irecip(
            self._unsafe_storage_cast[to_mut=True](), self.layout
        )

    @inline(.always)
    def exp[
        scale_dtype: DType = Self.dtype, //, scale: Scalar[scale_dtype] = 1
    ](self) where Self.mut and conforms_to(Self.Engine, TensorOps):
        """Replaces each element `x` of this tensor with `exp(scale * x)`,
        in place.

        The scale factor is applied before exponentiation so that scaled
        exponentials (for example softmax logit scaling) fuse into a single
        pass over the elements. The default scale of `1` gives a plain
        exponential.

        Parameters:
            scale_dtype: The data type of the scale factor. Defaults to the
                tensor's dtype; the scale is cast to the tensor's dtype
                before the multiplication.
            scale: The compile-time factor each element is multiplied by
                before exponentiation.

        Constraints:
            The tensor's dtype must be a floating-point type.
        """
        Self.Engine.iexp[scale](
            self._unsafe_storage_cast[to_mut=True](), self.layout
        )


@fieldwise_init
struct NullableTileTensor[
    mut: Bool,
    //,
    dtype: DType,
    LayoutType: TensorLayout,
    origin: Origin[mut=mut],
    *,
    Engine: TensorEngine,
    address_space: AddressSpace = .GENERIC,
    linear_idx_type: DType = _get_index_type[LayoutType](address_space),
](ImplicitlyCopyable, RegisterPassable):
    """A TileTensor variant whose pointer may be absent (null).

    `NullableTileTensor` carries the same layout metadata as `TileTensor` but
    explicitly represents its pointer as nullable.

    Only layout-query methods are provided.  To perform loads, stores, or other
    data operations, first check `self.ptr` and then call `value()` to
    obtain a regular `TileTensor`.

    Parameters:
        mut: The inferred mutability of the underlying pointer.
        dtype: The data type of tensor elements.
        LayoutType: A type implementing `TensorLayout` that defines the
            tensor's shape and stride structure.
        origin: The origin of the underlying pointer for lifetime tracking.
        Engine: A type implementing `TensorEngine` that supplies the storage
            handle. Defaults to `DefaultEngine[element_width=1]`, a plain
            `Pointer` handle over non-vectorized elements.
        address_space: Memory address space. Defaults to GENERIC.
        linear_idx_type: Integer type for memory indexing.
    """

    comptime rank = Self.LayoutType.rank
    """The number of dimensions in the tensor's layout."""

    comptime flat_rank = _Flattened[*Self.LayoutType._shape_types].length
    """The flattened rank."""

    comptime element_size = Self.Engine.element_size
    """Number of scalar elements per logical element, derived from `Engine`."""

    comptime ElementType = SIMD[Self.dtype, Self.element_size]
    """The SIMD type used for element access."""

    comptime shape_known = Self.LayoutType.shape_known
    """True if all shape dimensions are compile-time constants."""

    comptime stride_known = Self.LayoutType.stride_known
    """True if all stride dimensions are compile-time constants."""

    comptime all_dims_known = Self.LayoutType.all_dims_known
    """True if both shape and stride are fully known at compile time."""

    comptime is_compatible_with[
        C: TypeList[Trait=CoordLike, ...]
    ] = WeaklyCompatible[Self.LayoutType, C]
    """True if coordinate types `C` are structurally compatible with this
    tensor's layout shape.

    A scalar coordinate element is always compatible. A tuple coordinate
    element requires the corresponding layout shape element to also be a
    tuple of the same length, checked recursively up to 4 levels of
    nesting.

    Parameters:
        C: The coordinate element types to check against.
    """

    comptime static_shape[i: Int] = Self.LayoutType.static_shape[i]
    """Get the compile-time shape value for dimension i, or -1 if dynamic.

    Parameters:
        i: The dimension index.
    """

    comptime static_stride[i: Int] = Self.LayoutType.static_stride[i]
    """Get the compile-time stride value for dimension i, or -1 if dynamic.

    Parameters:
        i: The dimension index.
    """

    comptime is_row_major = _IsRowMajor[
        Self.LayoutType._shape_types, Self.LayoutType._stride_types
    ]
    """True if the tensor has row-major (contiguous) strides."""

    var _storage: Optional[
        Self.Engine.StorageType[Self.dtype, Self.origin, Self.address_space]
    ]
    """Optional handle to the tensor's underlying data storage.

    When `None`, represents a tensor with layout metadata but no backing
    memory (e.g. an output buffer that the callee should allocate).
    """

    var layout: Self.LayoutType
    """The layout instance defining shape and stride mappings."""

    comptime GenericType = NullableTileTensor[
        Self.dtype,
        Self.LayoutType,
        Self.origin,
        Engine=Self.Engine,
        address_space=.GENERIC,
        linear_idx_type=Self.linear_idx_type,
    ]
    """Type alias for this tensor with GENERIC address space.

    Used by constructors that create tensors from Span, DeviceBuffer, or
    HostBuffer, which all produce GENERIC address space tensors.
    """

    @inline(.always)
    @implicit
    def __init__(
        other: NullableTileTensor,
        out self: NullableTileTensor[
            other.dtype,
            other.LayoutType,
            ImmOrigin(other.origin),
            Engine=other.Engine,
            address_space=other.address_space,
            linear_idx_type=other.linear_idx_type,
        ],
    ):
        """Implicitly cast a mutable NullableTileTensor to immutable.

        Args:
            other: The mutable NullableTileTensor to cast from.
        """
        self._storage = other._unsafe_storage_cast[
            to_origin=type_of(self).origin
        ]()
        self.layout = other.layout

    @inline(.always)
    @implicit
    def __init__(
        other: TileTensor[mut=True, ...],
        out self: NullableTileTensor[
            other.dtype,
            other.LayoutType,
            other.origin,
            Engine=other.Engine,
            address_space=other.address_space,
            linear_idx_type=other.linear_idx_type,
        ],
    ):
        """Implicitly cast a TileTensor to a NullableTileTensor.

        Args:
            other: The TileTensor to cast from.
        """
        self._storage = other._storage
        self.layout = other.layout

    @inline(.always)
    @implicit
    def __init__(
        other: TileTensor,
        out self: NullableTileTensor[
            other.dtype,
            other.LayoutType,
            ImmOrigin(other.origin),
            Engine=other.Engine,
            address_space=other.address_space,
            linear_idx_type=other.linear_idx_type,
        ],
    ):
        """Implicitly cast a mutable TileTensor to an immutable NullableTileTensor.

        Args:
            other: The mutable TileTensor to cast from.
        """
        self._storage = other._unsafe_storage_cast[
            to_origin=type_of(self).origin
        ]()
        self.layout = other.layout

    @doc_hidden
    @inline(.always)
    def __getattr_param__[
        name: StringLiteral
    ](
        self,
        out result: Optional[
            Pointer[
                Scalar[Self.dtype],
                Self.origin,
                address_space=Self.address_space,
            ]
        ],
    ):
        comptime assert (
            name == "ptr"
        ), "NullableTileTensor.__getattr_param__ only support 'ptr'"
        if not self._storage:
            result = None
            return
        try:
            result = Self.Engine.unsafe_ptr(self._storage.unsafe_value())
        except e:
            abort(t"NullableTileTensor.ptr access not possible: {e}")

    @inline(.nodebug)
    def _unsafe_storage_cast[
        to_mut: Bool = Self.mut,
        //,
        to_dtype: DType = Self.dtype,
        to_origin: Origin[mut=to_mut] = Self.origin.unsafe_mut_cast[to_mut](),
        to_address_space: AddressSpace = Self.address_space,
    ](self) -> Optional[
        Self.Engine.StorageType[to_dtype, to_origin, to_address_space]
    ]:
        if not self._storage:
            return None
        return Self.Engine.unsafe_cast[to_dtype, to_origin, to_address_space](
            self._storage.unsafe_value()
        )

    @inline(.always)
    def value(
        self,
    ) -> TileTensor[
        Self.dtype,
        Self.LayoutType,
        Self.origin,
        Engine=Self.Engine,
        address_space=Self.address_space,
        linear_idx_type=Self.linear_idx_type,
    ]:
        """Returns a regular TileTensor with the underlying storage handle.

        The caller must ensure the storage handle is present before calling
        this method.

        Returns:
            A `TileTensor` backed by the stored handle and layout.
        """
        assert Bool(self._storage), "TileTensor cannot be null"
        return {self._storage.unsafe_value(), self.layout}

    # ===------------------------------------------------------------------=== #
    # Layout query methods
    # ===------------------------------------------------------------------=== #

    @inline(.nodebug)
    def dim[i: Int](self) -> Scalar[Self.linear_idx_type]:
        """Returns the size of outer-mode dimension `i`.

        For a flat layout this is `shape[i]`. For a nested layout (where
        `shape[i]` is itself a `Coord`) this is the product of all leaf
        dims under outer-mode `i`: the i-th mode's extent under CuTe
        Layout Algebra. For shape `((a, b), (c, d))`: `dim[0] = a*b`,
        `dim[1] = c*d`.

        Parameters:
            i: The dimension index (compile-time constant).

        Returns:
            The product of all leaf dims under outer-mode `i`.
        """
        comptime assert 0 <= i < Self.rank, String(
            t"dim index {i} is out of bounds for tensor rank [0, {Self.rank})"
        )
        comptime if Self.LayoutType._shape_types[i].is_tuple:
            return Scalar[Self.linear_idx_type](
                self.layout.shape[i]().product()
            )
        else:
            return Scalar[Self.linear_idx_type](self.layout.shape[i]().value())

    @inline(.nodebug)
    def dim[
        IndexType: Indexer
    ](self, index: IndexType) -> Scalar[Self.linear_idx_type]:
        """Returns the size of the specified dimension.

        Parameters:
            IndexType: The type of the index argument.

        Args:
            index: The dimension index (runtime value).

        Returns:
            The size of the specified dimension as a scalar.
        """
        var idx = _index(index)

        comptime for i in range(Self.rank):
            if idx == i:
                return Scalar[Self.linear_idx_type](
                    self.layout.shape[i]().value()
                )
        abort("attempt to dynamically index out of bounds")

    def num_elements(self) -> Int:
        """Returns the total number of elements in the tensor.

        Computes the product of all shape dimensions.

        Returns:
            The total element count.
        """
        var result = 1

        comptime for i in range(Self.rank):
            result *= Int(self.layout.shape[i]().value())
        return result

    @inline(.nodebug)
    def to_layout_tensor(
        self,
        out result: LayoutTensor[
            Self.dtype,
            _LegacyLayout(
                coord_to_int_tuple[*Self.LayoutType._shape_types](),
                coord_to_int_tuple[*Self.LayoutType._stride_types](),
            ),
            Self.origin,
            address_space=Self.address_space,
        ],
    ):
        """Return a LayoutTensor with the same shape, stride, and address space
        of this tensor.

        This is a utility to help with porting LayoutTensor methods to this type.

        Returns:
            A LayoutTensor with the same shape, stride, and address space of
            this tensor.
        """
        return {
            # This is totally a hack casting nullable pointer to non-nullable,
            # however this works as they have the same size/layout
            # and this is much simpler than moving LayoutTensor over to
            # nullable pointers since TileTensor is the preferred alternative now.
            Pointer(to=self.ptr).unsafe_bitcast[type_of(result.ptr)]()[],
            type_of(result.runtime_layout)(
                # A `RuntimeTuple` stores one entry per leaf, so a nested mode
                # has to be flattened to supply them in the order it expects.
                # `flatten()` is the identity on a flat `Coord`.
                coord_to_index_list(self.layout.shape_coord().flatten()).cast[
                    result.layout_int_type
                ](),
                coord_to_index_list(self.layout.stride_coord().flatten()).cast[
                    result.linear_idx_type
                ](),
            ),
        }


comptime _ComptimeConditionalTileTensor[
    mut: Bool,
    //,
    dtype: DType,
    LayoutType: TensorLayout,
    origin: Origin[mut=mut],
    *,
    engaged: Bool = False,
    Engine: TensorEngine = DefaultEngine[element_width=1],
    address_space: AddressSpace = .GENERIC,
    linear_idx_type: DType = _get_index_type[LayoutType](address_space),
] = _ComptimeConditional[
    TileTensor[
        dtype,
        LayoutType,
        origin,
        Engine=Engine,
        address_space=address_space,
        linear_idx_type=linear_idx_type,
    ],
    engaged=engaged,
]


@inline(.nodebug)
def stack_allocation[
    LayoutType: TensorLayout,
    //,
    dtype: DType,
    address_space: AddressSpace = .GENERIC,
    alignment: Int = align_of[dtype](),
](var layout: LayoutType) -> TileTensor[
    dtype, LayoutType, MutUntrackedOrigin, address_space=address_space
] where LayoutType.all_dims_known:
    """Allocate a TileTensor on the stack with the given layout.

    Creates a stack-allocated buffer sized for the layout and returns a
    TileTensor pointing to it. The layout must have all dimensions known
    at compile time.

    Parameters:
        LayoutType: The layout type (inferred from layout argument).
        dtype: The data type of tensor elements.
        address_space: Memory address space (default: GENERIC).
        alignment: Allocation alignment in bytes (default: natural type
            alignment from `align_of[dtype]()`). Pass an explicit value
            when downstream loads/stores require larger alignment (for
            example, AMD `ds_read_b128` requires 16 B, and sub-block
            swizzles can require alignment up to the sub-block size).
            Forwarded to the underlying `std.memory.stack_allocation`.

    Args:
        layout: The layout instance defining shape and strides.

    Returns:
        A mutable TileTensor backed by stack-allocated memory.

    Constraints:
        All layout dimensions must be statically known.
    """
    return TileTensor[
        dtype, LayoutType, MutUntrackedOrigin, address_space=address_space
    ](
        _std_stack_allocation[
            Coord[*LayoutType._shape_types].static_product,
            Scalar[dtype],
            alignment=alignment,
            address_space=address_space,
        ](),
        layout,
    )


@inline(.always)
def _pretty_print_elementwise[W: Writer](tensor: TileTensor, mut writer: W):
    var n = Int(tensor.layout.product())
    writer.write("[")
    for i in range(n):
        var offset = tensor.layout[linear_idx_type=tensor.linear_idx_type](
            Scalar[tensor.linear_idx_type](i)
        )
        writer.write(tensor.raw_load[width=tensor.element_size](offset))
        if i < n - 1:
            writer.write(", ")
    writer.write("]")


@inline(.always)
def _pretty_print_2d_tensor[
    W: Writer
](tensor: TileTensor, mut writer: W) where tensor.flat_rank == 2:
    # Provide evidence to the constraint system
    comptime assert tensor.flat_rank == 2
    var m_dim = tensor.layout.shape[0]()
    var n_dim = tensor.layout.shape[1]()
    writer.write("[")
    for m in range(Int(m_dim.value())):
        writer.write("[")
        for n in range(Int(n_dim.value())):
            writer.write(tensor[m, n])
            if n < Int(n_dim.value()) - 1:
                writer.write(", ")
        writer.write("]")
        if m < Int(m_dim.value()) - 1:
            writer.write(", ")
    writer.write("]")


@inline(.nodebug)
def _distribute[
    thread_layout: Layout,
    swizzle: Optional[Swizzle] = None,
](
    data_layout_tensor: TileTensor,
    thread_id: Int,
) -> data_layout_tensor.OffsetViewType[
    TypeList.of[Scalar[data_layout_tensor.linear_idx_type]](),
    Layout[
        shape_types=_Divide[
            data_layout_tensor.LayoutType._shape_types,
            thread_layout.shape_types,
        ],
        stride_types=_Multiply[
            data_layout_tensor.LayoutType._stride_types,
            thread_layout.shape_types,
        ],
    ],
]:
    """A simplified implementation of LayoutTensor.distribute on TileTensor.

    Parameters:
        thread_layout: Defines the logical arrangement of threads.
        swizzle: Optional swizzle function to remap the distribution pattern
            for improved memory access patterns.

    Args:
        data_layout_tensor: The tensor to distribute.
        thread_id: The ID of the current thread (0-based).

    Returns:
        A view into the tensor for the specified thread.
    """

    # Narrow-first multiply: accumulate the offset at the tensor's
    # `linear_idx_type` so GPU codegen with narrow index types (e.g. uint32)
    # doesn't route through 64-bit Int for every dim.
    var offset = Scalar[data_layout_tensor.linear_idx_type](0)

    comptime for i in range(thread_layout.stride_types.length):
        comptime stride_i = thread_layout.stride_types[i].static_value
        comptime shape_i = thread_layout.shape_types[i].static_value
        var thread_coord_i = (thread_id // stride_i) % shape_i
        offset += Scalar[data_layout_tensor.linear_idx_type](
            thread_coord_i
        ) * Scalar[data_layout_tensor.linear_idx_type](
            data_layout_tensor.layout.stride[i]().value()
        )

    # Swizzling applies to the index of elements rather than scalars because
    # the former is the unit in distribution. Swizzle functions take Int, so
    # widen at that boundary and narrow back after.
    var swizzled_offset = offset

    comptime if swizzle:
        comptime swizzle_fn = swizzle.value()
        comptime element_size = data_layout_tensor.element_size
        swizzled_offset = Scalar[data_layout_tensor.linear_idx_type](
            swizzle_fn(Int(offset) // element_size) * element_size
        )

    comptime NewShapeTypes = _Divide[
        data_layout_tensor.LayoutType._shape_types,
        thread_layout.shape_types,
    ]
    comptime NewStrideTypes = _Multiply[
        data_layout_tensor.LayoutType._stride_types,
        thread_layout.shape_types,
    ]
    var shape = Coord[*NewShapeTypes]()
    var stride = Coord[*NewStrideTypes]()

    # Populate runtime values for dimensions that aren't statically known.
    comptime for i in range(NewShapeTypes.length):
        comptime if not NewShapeTypes[i].is_static_value:
            Pointer(to=shape[i]).write(
                _coerce_dynamic[NewShapeTypes[i]](
                    Int(data_layout_tensor.layout.shape_coord()[i].value())
                    // thread_layout.shape_types[i].static_value
                )
            )
        comptime if not NewStrideTypes[i].is_static_value:
            Pointer(to=stride[i]).write(
                _coerce_dynamic[NewStrideTypes[i]](
                    Int(data_layout_tensor.layout.stride_coord()[i].value())
                    * thread_layout.shape_types[i].static_value
                )
            )

    var layout = Layout(shape, stride)

    comptime ResultLayout = Layout[
        shape_types=NewShapeTypes,
        stride_types=NewStrideTypes,
    ]
    return {
        data_layout_tensor._offset_storage(swizzled_offset),
        layout,
    }


@inline(.nodebug)
def _distribute_with_offset[
    thread_layout: Layout,
    swizzle: Optional[Swizzle] = None,
](
    data_layout_tensor: TileTensor,
    thread_id: Int,
) -> Tuple[
    data_layout_tensor.OffsetViewType[
        TypeList.of[Int](),
        Layout[
            shape_types=_Divide[
                data_layout_tensor.LayoutType._shape_types,
                thread_layout.shape_types,
            ],
            stride_types=_Multiply[
                data_layout_tensor.LayoutType._stride_types,
                thread_layout.shape_types,
            ],
        ],
    ],
    IndexList[thread_layout.shape_types.length],
    Int,
]:
    """Like _distribute, but also returns thread coordinates and offset.

    The thread coordinates are the position of the current thread in
    the thread grid. The offset is the linear element offset (after
    optional swizzling) used to advance the pointer.
    """

    # Use shape_types consistently for the IndexList size (must match return type)
    # This variant returns the offset as `Int` so callers can consume it at
    # full precision. Index arithmetic stays at `Int` here deliberately —
    # narrowing to `linear_idx_type` first would truncate before the widening
    # at the return boundary. Use `_distribute` (above) if narrow-precision
    # pointer-offset arithmetic is what you want.
    var offset: Int = 0
    var thread_coords = IndexList[thread_layout.shape_types.length]()

    comptime for i in range(thread_layout.shape_types.length):
        comptime stride_i = thread_layout.stride_types[i].static_value
        comptime shape_i = thread_layout.shape_types[i].static_value
        var thread_coord_i = (thread_id // stride_i) % shape_i
        thread_coords[i] = thread_coord_i
        offset += thread_coord_i * Int(
            data_layout_tensor.layout.stride[i]().value()
        )

    # Swizzling applies to the index of elements rather than scalars because
    # the former is the unit in distribution.
    var swizzled_offset = offset

    comptime if swizzle:
        comptime swizzle_fn = swizzle.value()
        comptime element_size = data_layout_tensor.element_size
        swizzled_offset = swizzle_fn(offset // element_size) * element_size

    comptime NewShapeTypes = _Divide[
        data_layout_tensor.LayoutType._shape_types,
        thread_layout.shape_types,
    ]
    comptime NewStrideTypes = _Multiply[
        data_layout_tensor.LayoutType._stride_types,
        thread_layout.shape_types,
    ]
    var shape = Coord[*NewShapeTypes]()
    var stride = Coord[*NewStrideTypes]()

    # Populate runtime values for dimensions that aren't statically known.
    comptime for i in range(NewShapeTypes.length):
        comptime if not NewShapeTypes[i].is_static_value:
            Pointer(to=shape[i]).write(
                _coerce_dynamic[NewShapeTypes[i]](
                    Int(data_layout_tensor.layout.shape_coord()[i].value())
                    // thread_layout.shape_types[i].static_value
                )
            )
        comptime if not NewStrideTypes[i].is_static_value:
            Pointer(to=stride[i]).write(
                _coerce_dynamic[NewStrideTypes[i]](
                    Int(data_layout_tensor.layout.stride_coord()[i].value())
                    * thread_layout.shape_types[i].static_value
                )
            )

    var layout = Layout(shape, stride)

    comptime ResultLayout = Layout[
        shape_types=NewShapeTypes,
        stride_types=NewStrideTypes,
    ]
    return (
        data_layout_tensor.OffsetViewType[TypeList.of[Int](), ResultLayout](
            data_layout_tensor._offset_storage(swizzled_offset),
            layout,
        ),
        thread_coords,
        swizzled_offset,
    )


# ===-------------------------------------------------------------------=== #
# Result-stride machinery for `.tile[]` (CuTe `local_tile`).
#
# `.tile[a, b](i, j)` slices the sub-tile at outer coord `(i, j)`. Per
# outer mode, the result's stride is the parent's innermost sub-stride
# (identity for scalar parent strides, last sub-element for tuple
# parent strides). One uniform helper handles both flat and nested
# parents — the per-mode dispatch happens via `_InnermostStride`.
# ===-------------------------------------------------------------------=== #


comptime _InnermostStride[T: CoordLike]: CoordLike = (
    T.ParamListType[T.ParamListType.length - 1] if T.is_tuple else T
)
"""For a tuple parent stride, pick the innermost (last) sub-element. For
a scalar parent stride, identity."""


comptime _TileResultStrideTabulator[
    ParentLayoutType: TensorLayout,
    idx: Int,
]: CoordLike = _InnermostStride[ParentLayoutType._stride_types[idx]]


comptime _NestedTileResultStrideTypes[
    ParentLayoutType: TensorLayout,
] = TypeList.tabulate[
    Trait=CoordLike,
    ParentLayoutType._stride_types.length,
    _TileResultStrideTabulator[ParentLayoutType, _],
]()
"""Result stride `TypeList` for `.tile[]`: each outer mode contributes
its innermost sub-stride. Identity-equivalent to parent's
`_stride_types` for flat parents; innermost-extracted for nested
parents."""


@inline(.nodebug)
def _tile[
    dtype: DType,
    coord_types: TypeList[Trait=CoordLike, ...],
    tile_shape_types: TypeList[Trait=CoordLike, ...],
    //,
](
    data_layout_tensor: TileTensor[dtype, ...],
    tile_shape: Coord[*tile_shape_types],
    tile_coords: Coord[*coord_types],
) -> data_layout_tensor.TileResultType[
    tile_shape_types, linear_idx_type=data_layout_tensor.linear_idx_type
]:
    """Extract a sub-tile from a TileTensor (CuTe `local_tile`). Works
    on both flat and nested parents via per-mode comptime dispatch.

    Per outer mode `i`:

    - Nested mode (parent stride at mode `i` is a tuple): offset +=
      `tile_coords[i] * stride[i].tuple()[0]` (outermost sub-stride).
      Result `stride[i]` = parent's innermost sub-stride.
    - Flat mode (parent stride at mode `i` is a scalar): offset +=
      `tile_coords[i] * tile_shape[i] * stride[i]`. Result `stride[i]`
      = parent stride (identity through `_InnermostStride`).

    Parameters:
        dtype: Data type of the tensor elements (inferred from tensor argument).
        coord_types: Types of the tile coordinates (inferred from coordinates argument).
        tile_shape_types: Types of the tile dimensions (inferred from tile_shape argument).

    Args:
        data_layout_tensor: The source tensor to extract the tile from.
        tile_shape: The shape that the layout should be tiled into.
        tile_coords: The index of the tile to extract as a Coord.

    Returns:
        A TileTensor view of the sub-tile region.
    """
    var offset = Scalar[data_layout_tensor.linear_idx_type](0)
    comptime for i in range(Coord[*coord_types].__len__()):
        comptime if data_layout_tensor.LayoutType._stride_types[i].is_tuple:
            offset += Scalar[data_layout_tensor.linear_idx_type](
                tile_coords[i].value()
            ) * Scalar[data_layout_tensor.linear_idx_type](
                data_layout_tensor.layout.stride[i]().tuple()[0].value()
            )
        else:
            offset += (
                Scalar[data_layout_tensor.linear_idx_type](
                    tile_coords[i].value()
                )
                * Scalar[data_layout_tensor.linear_idx_type](
                    tile_shape[i].value()
                )
                * Scalar[data_layout_tensor.linear_idx_type](
                    data_layout_tensor.layout.stride[i]().value()
                )
            )

    comptime ResultType = data_layout_tensor.TileResultType[
        tile_shape_types, linear_idx_type=data_layout_tensor.linear_idx_type
    ]
    comptime ParentIsFlat = (
        data_layout_tensor.LayoutType.rank
        == data_layout_tensor.LayoutType.flat_rank
    )
    comptime if ParentIsFlat:
        # Flat parent: result stride_types is identity-equivalent to
        # parent's, MLIR layouts match. Propagate parent's runtime
        # stride values via rebind — required for dynamic-stride
        # layouts (e.g. matmul ND-buffer tiles); default-construct
        # would zero the runtime scalars and miscompute offsets
        # downstream.
        var tile_layout = Layout(
            shape=tile_shape,
            stride=data_layout_tensor.layout.stride_coord(),
        )
        return ResultType(
            data_layout_tensor._offset_storage(offset),
            rebind[ResultType.LayoutType](tile_layout),
        )
    else:
        # Nested parent: result stride_types is innermost-extracted —
        # different MLIR storage than parent's nested stride_coord, so
        # a flat-style rebind isn't safe. All in-tree nested-layout
        # uses are static (values in the type), so default-construct
        # is sufficient. A future dynamic-nested caller would need a
        # per-mode innermost extraction here.
        return ResultType(
            data_layout_tensor._offset_storage(offset),
            ResultType.LayoutType(),
        )


@inline(.nodebug)
def _tile_with_offset[
    dtype: DType,
    coord_types: TypeList[Trait=CoordLike, ...],
    tile_shape_types: TypeList[Trait=CoordLike, ...],
    //,
](
    data_layout_tensor: TileTensor[dtype, ...],
    tile_shape: Coord[*tile_shape_types],
    tile_coords: Coord[*coord_types],
) -> Tuple[
    data_layout_tensor.OffsetViewType[
        TypeList.of[Int](),
        Layout[
            shape_types=tile_shape_types,
            stride_types=data_layout_tensor.LayoutType._stride_types,
        ],
    ],
    IndexList[coord_types.length],
    Int,
]:
    """Like _tile, but also returns corner coordinates and linear offset.

    The corner coordinates are the element-space coordinates of the tile's
    origin: corner_coords[i] = tile_coords[i] * tile_sizes[i].
    The offset is the linear element offset used to advance the pointer.
    """

    # Use TypeList[coord_types].length consistently (must match return type)
    # `offset` and `corner_coords` are part of the return type (`Int` and
    # `IndexList[...].element_type=int64`), so arithmetic stays at `Int` here.
    # See `_tile` above for the narrow-precision variant.
    var offset: Int = 0
    var corner_coords = IndexList[coord_types.length]()

    comptime for i in range(coord_types.length):
        corner_coords[i] = Int(tile_coords[i].value()) * Int(
            tile_shape[i].value()
        )
        offset += (
            Int(tile_coords[i].value())
            * Int(tile_shape[i].value())
            * Int(data_layout_tensor.layout.stride[i]().value())
        )

    var tile_layout = Layout(
        shape=tile_shape,
        stride=data_layout_tensor.layout.stride_coord(),
    )

    return (
        data_layout_tensor.OffsetViewType[
            TypeList.of[Int](),
            Layout[
                shape_types=tile_shape_types,
                stride_types=data_layout_tensor.LayoutType._stride_types,
            ],
        ](
            data_layout_tensor._offset_storage(offset),
            tile_layout,
        ),
        corner_coords,
        offset,
    )


@inline(.nodebug)
def _tile[
    dtype: DType,
    coord_types: TypeList[Trait=CoordLike, ...],
    tile_shape_types: TypeList[Trait=CoordLike, _],
    //,
    *,
    stride_layout: TensorLayout,
](
    data_layout_tensor: TileTensor[dtype, ...],
    tile_shape: Coord[*tile_shape_types],
    tile_coords: Coord[*coord_types],
) -> data_layout_tensor.OffsetViewType[
    TypeList.of[Scalar[data_layout_tensor.linear_idx_type]](),
    Layout[
        shape_types=tile_shape_types,
        stride_types=stride_layout._shape_types,
    ],
]:
    """Like _tile, but with explicit static strides.

    Use when the parent tensor has dynamic strides (e.g. from a
    TensorLayout trait parameter) but the stride values are known at
    compile time. The resulting tile has ComptimeInt strides, enabling
    vectorize/distribute which require all_dims_known.
    """

    # Narrow-first multiply: accumulate offset at `linear_idx_type` precision.
    var offset = Scalar[data_layout_tensor.linear_idx_type](0)

    comptime for i in range(Coord[*coord_types].__len__()):
        offset += (
            Scalar[data_layout_tensor.linear_idx_type](tile_coords[i].value())
            * Scalar[data_layout_tensor.linear_idx_type](tile_shape[i].value())
            * Scalar[data_layout_tensor.linear_idx_type](
                data_layout_tensor.layout.stride[i]().value()
            )
        )

    var tile_layout = Layout(
        shape=tile_shape,
        stride=Coord[*stride_layout._shape_types](),
    )

    return data_layout_tensor.OffsetViewType[
        TypeList.of[Scalar[data_layout_tensor.linear_idx_type]](),
        Layout[
            shape_types=tile_shape_types,
            stride_types=stride_layout._shape_types,
        ],
    ](
        data_layout_tensor._offset_storage(offset),
        tile_layout,
    )


@inline(.nodebug)
def _tile_with_offset[
    dtype: DType,
    coord_types: TypeList[Trait=CoordLike, ...],
    tile_shape_types: TypeList[Trait=CoordLike, ...],
    //,
    *,
    stride_layout: TensorLayout,
](
    data_layout_tensor: TileTensor[dtype, ...],
    tile_shape: Coord[*tile_shape_types],
    tile_coords: Coord[*coord_types],
) -> Tuple[
    data_layout_tensor.OffsetViewType[
        TypeList.of[Int](),
        Layout[
            shape_types=tile_shape_types,
            stride_types=stride_layout._shape_types,
        ],
    ],
    IndexList[coord_types.length],
    Int,
]:
    """Like _tile_with_offset, but with explicit static strides."""

    # `offset` and `corner_coords` are part of the return type; index
    # arithmetic stays at `Int` for the same reason as `_tile_with_offset`.
    var offset: Int = 0
    var corner_coords = IndexList[coord_types.length]()

    comptime for i in range(coord_types.length):
        corner_coords[i] = Int(tile_coords[i].value()) * Int(
            tile_shape[i].value()
        )
        offset += (
            Int(tile_coords[i].value())
            * Int(tile_shape[i].value())
            * Int(data_layout_tensor.layout.stride[i]().value())
        )

    var tile_layout = Layout(
        shape=tile_shape,
        stride=Coord[*stride_layout._shape_types](),
    )

    return (
        data_layout_tensor.OffsetViewType[
            TypeList.of[Int](),
            Layout[
                shape_types=tile_shape_types,
                stride_types=stride_layout._shape_types,
            ],
        ](
            data_layout_tensor._offset_storage(offset),
            tile_layout,
        ),
        corner_coords,
        offset,
    )


@inline(.nodebug)
def _vectorize[
    dtype: DType,
    vector_shape_types: TypeList[Trait=CoordLike, ...],
    //,
](
    data_layout_tensor: TileTensor[dtype, ...],
    vector_shape: Coord[*vector_shape_types],
) -> TileTensor[
    dtype,
    Layout[
        shape_types=_CeilDiv[
            data_layout_tensor.LayoutType._shape_types,
            vector_shape_types,
        ],
        stride_types=_Multiply[
            data_layout_tensor.LayoutType._stride_types,
            vector_shape_types,
        ],
    ],
    data_layout_tensor.origin,
    address_space=data_layout_tensor.address_space,
    linear_idx_type=data_layout_tensor.linear_idx_type,
    Engine=DefaultEngine[
        element_width=Coord[*vector_shape_types].static_product
    ],
]:
    """Create a vectorized view of a TileTensor.

    This function creates a new view where the shape is divided by the vector
    shape (ceiling division) and strides are multiplied by the vector shape.
    This effectively groups elements into vector-sized blocks. The element
    layout is tracked via element_size.

    Parameters:
        dtype: Data type of the tensor elements.
        vector_shape_types: Types of the vector shape dimensions.

    Args:
        data_layout_tensor: The source tensor to vectorize.
        vector_shape: The shape of each vector unit as a Coord.

    Returns:
        A TileTensor representing a vectorized view. Each logical element
        in the result corresponds to a vector block in the original tensor.
        The element layout shape and strides are set to the vector shape
        with row-major strides.
    """
    comptime NewShapeTypes = _CeilDiv[
        data_layout_tensor.LayoutType._shape_types,
        vector_shape_types,
    ]
    comptime NewStrideTypes = _Multiply[
        data_layout_tensor.LayoutType._stride_types,
        vector_shape_types,
    ]

    var new_shape = Coord[*NewShapeTypes]()
    var new_stride = Coord[*NewStrideTypes]()

    # Populate runtime values for dimensions that aren't statically known.
    comptime for i in range(NewShapeTypes.length):
        comptime if not NewShapeTypes[i].is_static_value:
            Pointer(to=new_shape[i]).write(
                rebind[NewShapeTypes[i]](
                    Scalar[NewShapeTypes[i].DTYPE](
                        ceildiv(
                            Scalar[NewShapeTypes[i].DTYPE](
                                data_layout_tensor.layout.shape_coord()[
                                    i
                                ].value()
                            ),
                            Scalar[NewShapeTypes[i].DTYPE](
                                vector_shape[i].value()
                            ),
                        )
                    )
                )
            )
        comptime if not NewStrideTypes[i].is_static_value:
            Pointer(to=new_stride[i]).write(
                rebind[NewStrideTypes[i]](
                    Scalar[NewStrideTypes[i].DTYPE](
                        data_layout_tensor.layout.stride_coord()[i].value()
                    )
                    * Scalar[NewStrideTypes[i].DTYPE](vector_shape[i].value())
                )
            )

    var new_layout = Layout(new_shape, new_stride)

    comptime ResultLayout = Layout[
        shape_types=_CeilDiv[
            data_layout_tensor.LayoutType._shape_types,
            vector_shape_types,
        ],
        stride_types=_Multiply[
            data_layout_tensor.LayoutType._stride_types,
            vector_shape_types,
        ],
    ]
    return {
        rebind[
            Pointer[
                SIMD[
                    data_layout_tensor.dtype,
                    Coord[*vector_shape_types].static_product,
                ],
                address_space=data_layout_tensor.address_space,
                origin=data_layout_tensor.origin,
            ]
        ](data_layout_tensor.ptr),
        new_layout,
    }


def _get_index_type[
    LayoutType: TensorLayout
](address_space: AddressSpace) -> DType:
    """Returns int32 for shared/constant GPU memory or small known layouts,
    int64 otherwise."""
    comptime if LayoutType.all_dims_known and Int64(
        LayoutType.static_cosize
    ) >> 31 == 0:
        return DType.int32

    if address_space in (
        AddressSpace.SHARED,
        AddressSpace.CONSTANT,
    ):
        return DType.int32
    else:
        return DType.int64


comptime _ToRuntimeMapper[
    dtype: DType,
    element_types: TypeList[Trait=CoordLike, _],
    idx: Int,
] = Scalar[dtype]
"""Convert shape types to Scalar for slicing operations.

When slicing, compile-time dimensions become runtime dimensions because
we can't change ComptimeInt[4] to ComptimeInt[2] in the type system.

Parameters:
    dtype: The default data type to use for Scalar conversions.
    element_types: The variadic sequence of types to convert (wrapped in values).
    idx: The current index being processed.
"""

comptime _StaticSplitShapeTabulator[
    count: Int,
    axis: Int,
    element_types: TypeList[Trait=CoordLike, ...],
    idx: Int,
]: CoordLike = (
    ComptimeInt[element_types[idx].static_value // count] if idx
    == axis else element_types[idx]
)


comptime _StaticSplitShape[
    count: Int,
    axis: Int,
    element_types: TypeList[Trait=CoordLike, ...],
] = TypeList.tabulate[
    element_types.length,
    _StaticSplitShapeTabulator[count, axis, element_types, _],
]


comptime _DynamicSplitShapeTabulator[
    dtype: DType,
    axis: Int,
    element_types: TypeList[Trait=CoordLike, ...],
    idx: Int,
]: CoordLike = Scalar[dtype] if idx == axis else element_types[idx]


comptime _DynamicSplitShape[
    dtype: DType,
    axis: Int,
    element_types: TypeList[Trait=CoordLike, ...],
] = TypeList.tabulate[
    element_types.length,
    _DynamicSplitShapeTabulator[dtype, axis, element_types, _],
]

# ===-----------------------------------------------------------------------===#
# Subscript helpers — classify each argument as an index or a slice
# ===-----------------------------------------------------------------------===#
#
# Everything a subscript's result type depends on is spelled as a `comptime`
# alias rather than a `def`: the constraint solver folds aliases, but cannot
# evaluate a call, and it has to decide these to pick between the element and
# view overloads of `__getitem__`.


comptime _IsSliceArg[Elt: AnyType, idx: Int] = not conforms_to(Elt, CoordLike)
"""Predicate over a subscript's argument types: is this one a slice?

Asking what the argument is *not* keeps the answer decidable: an index type
reaches here through a `CoordLike`-bounded parameter, so conformance holds
whatever the concrete type turns out to be, whereas the solver cannot rule
out that same opaque type being `ContiguousSlice`."""


comptime _KeepStrideWhereArgIsSlice[
    arg_types: TypeList[Trait=AnyType, ...],
    element: CoordLike,
    idx: Int,
] = not conforms_to(arg_types[idx], CoordLike)
"""Predicate over a layout's stride types: does the argument at the same axis
keep that axis?"""


comptime _ArgHeadTabulator[
    arg_types: TypeList[Trait=AnyType, ...], idx: Int
]: AnyType = arg_types[idx]


comptime _DynamicExtentTabulator[dtype: DType, idx: Int]: CoordLike = Scalar[
    dtype
]


comptime _SliceArgs[
    arg_types: TypeList[Trait=AnyType, ...], up_to: Int = arg_types.length
] = TypeList.tabulate[up_to, _ArgHeadTabulator[arg_types, _]]().filter_idx[
    _IsSliceArg
]()
"""The slice arguments among the first `up_to`, in order."""


comptime _SubscriptHasSlice[
    arg_types: TypeList[Trait=AnyType, ...]
] = _SliceArgs[arg_types].length > 0
"""Whether any subscript argument is a slice, which makes the subscript a
view rather than an element load."""


comptime _SubscriptOutAxis[
    arg_types: TypeList[Trait=AnyType, ...], axis: Int
] = _SliceArgs[arg_types, axis].length
"""The output axis a sliced parent `axis` lands on, once the axes that index
arguments dropped have shifted the survivors left."""


# TODO(lukas): once `:` materializes as a zero-sized `FullSlice`, give it a
# case here and in `_IndexOrSlice` that keeps the parent's extent type, so a
# fully-sliced axis stays static.
comptime _SubscriptShape[
    arg_types: TypeList[Trait=AnyType, ...], dtype: DType
] = TypeList.tabulate[
    _SliceArgs[arg_types].length, _DynamicExtentTabulator[dtype, _]
]
"""A slice carries runtime bounds, so every surviving extent is a `Scalar`."""


comptime _SubscriptStride[
    arg_types: TypeList[Trait=AnyType, ...],
    stride_types: TypeList[Trait=CoordLike, ...],
] = stride_types.filter_idx[_KeepStrideWhereArgIsSlice[arg_types, _, _]]
"""Strides are inherited whole -- neither fixing an axis nor narrowing one
changes the step between the elements that survive."""


comptime _SubscriptLayout[
    arg_types: TypeList[Trait=AnyType, ...],
    layout: TensorLayout,
    dtype: DType,
] = Layout[
    shape_types=_SubscriptShape[arg_types, dtype](),
    stride_types=_SubscriptStride[arg_types, layout._stride_types](),
]
"""The layout of the view a subscript cuts out of `layout`."""


comptime _StaticIndexOf[T: AnyType] = T.static_value if conforms_to(
    T, CoordLike
) else -1
"""The compile-time value of an index argument type, `-1` when it is a
runtime index or not an index at all."""


comptime _SubscriptArgTabulator[
    arg_types: TypeList[Trait=AnyType, ...], idx: Int
]: _IndexOrSliceLike = _IndexOrSlice[_StaticIndexOf[arg_types[idx]]]


comptime _SubscriptArgs[
    arg_types: TypeList[Trait=AnyType, ...]
] = TypeList.tabulate[arg_types.length, _SubscriptArgTabulator[arg_types, _]]
"""The subscript's arguments as `_IndexOrSliceLike` descriptors, one per
axis, carrying what each argument's type says at compile time."""


comptime _IsStaticOffsetAxis[
    args: TypeList[Trait=_IndexOrSliceLike, ...],
    layout: TensorLayout,
    axis: Int,
] = args[axis].is_static_index and layout.static_stride[axis] != -1
"""Whether `axis` contributes to the view's offset entirely at compile time:
a compile-time index times a compile-time stride."""


@inline(.nodebug)
def _subscript_static_offset[
    args: TypeList[Trait=_IndexOrSliceLike, ...], layout: TensorLayout
]() -> Int:
    """Returns the compile-time share of a subscript view's element offset."""
    var offset = 0
    comptime for axis in range(args.length):
        comptime if _IsStaticOffsetAxis[args, layout, axis]:
            offset += args[axis].static_index * layout.static_stride[axis]
    return offset


comptime _SubscriptOffset[
    args: TypeList[Trait=_IndexOrSliceLike, ...],
    layout: TensorLayout,
    dtype: DType,
] = TypeList.of[
    Trait=CoordLike,
    ComptimeInt[_subscript_static_offset[args, layout]()],
    Scalar[dtype],
]
"""The component types of a subscript view's offset: the compile-time share
as a `ComptimeInt`, then the runtime remainder."""


comptime _IsRowMajorTabulator[
    expected_strides: TypeList[Trait=CoordLike, ...],
    element_types: TypeList[Trait=CoordLike, ...],
    idx: Int,
]: Bool = expected_strides[idx].static_value == element_types[idx].static_value
"""Check if stride at index matches expected row-major stride."""

comptime _ReturnBool[value: Bool]: Bool = value

comptime _IsRowMajor[
    shape_types: TypeList[Trait=CoordLike, ...],
    stride_types: TypeList[Trait=CoordLike, ...],
]: Bool = ParameterList.tabulate[
    stride_types.length,
    _IsRowMajorTabulator[_RowMajor[*shape_types], stride_types, _],
]().all[
    _ReturnBool
]()
"""Check if stride_types match row-major strides for shape_types.

Returns True if all strides match the expected row-major pattern,
False otherwise. For row-major, stride[i] = product(shape[i+1:]).
"""


# =============================================================================
# Standalone reshape helpers
# =============================================================================


comptime _FlatLeadingLayout[L: TensorLayout] = RowMajorLayout[
    *Coord[Int64, L._shape_types[L.rank - 1]].element_types
]
"""Layout type after merging leading two dims: (A, B, C) -> (A*B, C).

The merged dimension is always Scalar. The last dimension preserves
its original static/dynamic type.
"""


@inline(.nodebug)
def flatten_leading[
    dtype: DType,
    layout: TensorLayout,
    //,
](
    tensor: TileTensor[dtype, LayoutType=layout, ...],
) -> tensor.ViewType[
    RowMajorLayout[
        *Coord[Int64, layout._shape_types[layout.rank - 1]].element_types
    ]
]:
    """Merge the first two dimensions of a rank-3 TileTensor: (A, B, C) -> (A*B, C).

    Returns a new TileTensor sharing the same pointer with row-major
    strides computed from the merged shape. Zero-cost operation.

    Common use case: converting 3D batched tensors (num_experts, N, K)
    to 2D (num_experts*N, K) for TMA descriptor creation in MoE kernels.

    Parameters:
        dtype: Element type (inferred from tensor).
        layout: Layout type (inferred from tensor).

    Args:
        tensor: A rank-3 TileTensor.

    Returns:
        A rank-2 TileTensor where dim[0] = old dim[0] * dim[1].
    """
    comptime assert type_of(tensor).rank == 3, "flatten_leading requires rank 3"
    var merged = Int64(tensor.layout.shape[0]().value()) * Int64(
        tensor.layout.shape[1]().value()
    )
    comptime ResultLayout = RowMajorLayout[
        *Coord[Int64, layout._shape_types[layout.rank - 1]].element_types
    ]
    return rebind[tensor.ViewType[ResultLayout]](
        tensor.reshape(row_major(Coord(merged, tensor.layout.shape[2]())))
    )


# ============================================================================
# lt_to_tt -- Convert a LayoutTensor to a TileTensor
# ============================================================================

from .int_tuple import IntTuple as _IntTuple, product_each as _product_each
from .layout import Layout as _LegacyLayout
from .layout_tensor import LayoutTensor as _LayoutTensor


comptime LTToTTLayout[lt_layout: _LegacyLayout] = Layout[
    shape_types=_IntTupleToCoordLike[
        DType.int64, _product_each(lt_layout.shape)
    ],
    stride_types=_IntTupleToCoordLike[
        DType.int64, _product_each(lt_layout.stride)
    ],
]
"""Derive a TileTensor Layout from a legacy Layout.

Known dimensions become ComptimeInt, UNKNOWN_VALUE dimensions become
Scalar.  Hierarchical layouts (e.g. from ``tile_to_shape``) are
collapsed via ``product_each`` so each mode becomes a single value.

Parameters:
    lt_layout: The legacy Layout to convert.
"""


@inline(.always)
def lt_to_tt[
    dtype: DType,
    lt_layout: _LegacyLayout,
    //,
    ResultLayout: TensorLayout = LTToTTLayout[lt_layout],
](lt: _LayoutTensor[dtype, lt_layout, ...]) -> TileTensor[
    dtype,
    Layout[
        shape_types=ResultLayout._shape_types,
        stride_types=ResultLayout._stride_types,
    ],
    lt.origin,
    address_space=lt.address_space,
]:
    """Convert a LayoutTensor to a TileTensor.

    Static dimensions (known at compile time) are preserved as ComptimeInt.
    Dynamic dimensions (UNKNOWN_VALUE) become Scalar, filled from the
    LayoutTensor's runtime layout.  The address space is preserved from the
    source LayoutTensor.  Works for any flat rank.

    By default the TileTensor layout is derived automatically from the
    LayoutTensor's legacy layout.  Pass an explicit ``ResultLayout`` to
    override which dimensions are static vs runtime.

    The result's `linear_idx_type` is the `TileTensor` default
    (`int32` for SHARED/CONSTANT or small static cosize, `int64`
    otherwise). This may differ from `lt.linear_idx_type` -- in
    particular, when the parent `LayoutTensor` has a runtime dimension
    in GENERIC space, the parent uses `int64` but the converted tile
    (with all dims static after tiling) defaults to `int32`. To force
    the result to match `lt.linear_idx_type`, use `lt_to_tt_idx`.

    Parameters:
        dtype: Element type of the tensor.
        lt_layout: The legacy Layout of the LayoutTensor.
        ResultLayout: The target TileTensor layout type.  Defaults to
            ``LTToTTLayout[lt_layout]``.

    Args:
        lt: The LayoutTensor to convert.

    Returns:
        A TileTensor with the same data, equivalent layout, and matching
        address space.
    """
    comptime ConcLayout = Layout[
        shape_types=ResultLayout._shape_types,
        stride_types=ResultLayout._stride_types,
    ]
    comptime rank = ConcLayout.rank
    var shape_c = Coord[*ConcLayout.shape_types]()
    var stride_c = Coord[*ConcLayout.stride_types]()

    comptime for i in range(rank):
        comptime if not shape_c.element_types[i].is_static_value:
            shape_c[i] = rebind[shape_c.element_types[i]](
                Int64(lt.runtime_layout.shape.value[i])
            )

        comptime if not stride_c.element_types[i].is_static_value:
            stride_c[i] = rebind[stride_c.element_types[i]](
                Int64(lt.runtime_layout.stride.value[i])
            )

    var ptr = Pointer[Scalar[dtype], lt.origin, address_space=lt.address_space](
        unsafe_from_address=Int(lt.ptr)
    )
    return TileTensor[
        dtype, ConcLayout, lt.origin, address_space=lt.address_space
    ](
        ptr=ptr,
        layout=ConcLayout(shape_c, stride_c),
    )


@inline(.always)
def lt_to_tt_idx[
    dtype: DType,
    lt_layout: _LegacyLayout,
    //,
    ResultLayout: TensorLayout = LTToTTLayout[lt_layout],
    linear_idx_type: DType = .int64,
](lt: _LayoutTensor[dtype, lt_layout, ...]) -> TileTensor[
    dtype,
    Layout[
        shape_types=ResultLayout._shape_types,
        stride_types=ResultLayout._stride_types,
    ],
    lt.origin,
    address_space=lt.address_space,
    linear_idx_type=linear_idx_type,
]:
    """Like `lt_to_tt` but with an explicit `linear_idx_type` override.

    Use this variant when the default `TileTensor` index-type heuristic
    (`int32` for SHARED/CONSTANT or small static cosize, `int64`
    otherwise) does not match what callers downstream want. The most
    common reason is preserving `int64` indexing for DRAM tiles derived
    from a tensor with runtime dimensions: the parent `LayoutTensor`
    uses `int64` for its address arithmetic, but a tile coming out of
    `lt_to_tt` (whose own layout is fully static post-tiling) defaults
    to `int32` -- forcing `_distribute()` to do narrow-then-widen index
    arithmetic for every offset, which is measurable in tight inner
    loops.

    Parameters:
        dtype: Element type of the tensor.
        lt_layout: The legacy Layout of the LayoutTensor.
        ResultLayout: The target TileTensor layout type.
        linear_idx_type: Integer type used for the result's offset
            arithmetic. Defaults to `DType.int64`.

    Args:
        lt: The LayoutTensor to convert.

    Returns:
        A TileTensor with the same data and the requested
        `linear_idx_type`.
    """
    comptime ConcLayout = Layout[
        shape_types=ResultLayout._shape_types,
        stride_types=ResultLayout._stride_types,
    ]
    comptime rank = ConcLayout.rank
    var shape_c = Coord[*ConcLayout.shape_types]()
    var stride_c = Coord[*ConcLayout.stride_types]()

    comptime for i in range(rank):
        comptime if not shape_c.element_types[i].is_static_value:
            shape_c[i] = rebind[shape_c.element_types[i]](
                Int64(lt.runtime_layout.shape.value[i])
            )

        comptime if not stride_c.element_types[i].is_static_value:
            stride_c[i] = rebind[stride_c.element_types[i]](
                Int64(lt.runtime_layout.stride.value[i])
            )

    var ptr = Pointer[Scalar[dtype], lt.origin, address_space=lt.address_space](
        unsafe_from_address=Int(lt.ptr)
    )
    return TileTensor[
        dtype,
        ConcLayout,
        lt.origin,
        address_space=lt.address_space,
        linear_idx_type=linear_idx_type,
    ](
        ptr=ptr,
        layout=ConcLayout(shape_c, stride_c),
    )
