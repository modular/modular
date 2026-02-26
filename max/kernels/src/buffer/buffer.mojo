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
"""Implements the NDBuffer struct.

You can import these APIs from the `buffer` package. For example:

```mojo
from buffer import NDBuffer
```
"""

from math import align_down, fma, iota
from pathlib import Path
from sys.info import align_of, is_gpu, is_nvidia_gpu, simd_width_of, size_of
from sys.intrinsics import PrefetchOptions, masked_load, masked_store, prefetch

from buffer.dimlist import Dim, DimList, _make_tuple
from builtin.device_passable import DevicePassable
from memory import (
    # Keep LegacyOpaquePointer for DevicePassable trait compatibility
    LegacyOpaquePointer,
    memset_zero,
    stack_allocation,
)

from utils._serialize import _serialize
from utils.index import IndexList
from utils.static_tuple import StaticTuple

comptime _MAX_RANK = 8
"""The maximum tensor rank for any tensor shape.
This value must match kMaxRank in Support/include/Support/ML/TensorShape.h
"""


# ===-----------------------------------------------------------------------===#
# NDBuffer Utilities
# ===-----------------------------------------------------------------------===#


@always_inline
fn _use_32bit_indexing[address_space: AddressSpace]() -> Bool:
    return is_gpu() and address_space in (
        AddressSpace.SHARED,
        AddressSpace.LOCAL,
        AddressSpace.CONSTANT,
    )


@always_inline
fn _compute_nd_index(buf: NDBuffer, index: Int) -> IndexList[buf.rank]:
    """Computes the NDBuffer's offset using the index positions provided.

    Args:
        buf: The NDBuffer.
        index: The flat index position.

    Returns:
        The index positions.
    """

    comptime if buf.rank == 0:
        return IndexList[buf.rank](0)

    var result = IndexList[buf.rank]()

    var curr_index = index

    comptime for i in reversed(range(buf.rank)):
        var dim = buf.dim[i]()
        result[i] = curr_index._positive_rem(dim)
        curr_index = curr_index / dim

    return result


@always_inline
fn _compute_ndbuffer_offset(
    buf: NDBuffer,
    index: VariadicList[Int],
) -> Int:
    """Computes the NDBuffer's offset using the index positions provided.

    Args:
        buf: The NDBuffer.
        index: The index positions.

    Returns:
        The offset into the NDBuffer given the indices.
    """

    comptime rank = buf.rank

    comptime if buf.rank == 0:
        return 0

    comptime if _use_32bit_indexing[buf.address_space]():
        var result: Int32 = 0

        comptime for i in range(buf.rank):
            result = fma(Int32(buf.stride[i]()), Int32(index[i]), result)

        return Int(result)

    else:
        var result: Int = 0

        comptime for i in range(buf.rank):
            result = fma(buf.stride[i](), index[i], result)

        return result


@always_inline
fn _compute_ndbuffer_offset(
    buf: NDBuffer,
    index: StaticTuple[Int, buf.rank],
) -> Int:
    """Computes the NDBuffer's offset using the index positions provided.

    Args:
        buf: The NDBuffer.
        index: The index positions.

    Returns:
        The offset into the NDBuffer given the indices.
    """

    comptime rank = buf.rank

    comptime if rank == 0:
        return 0

    comptime if _use_32bit_indexing[buf.address_space]():
        var result: Int32 = 0

        comptime for i in range(rank):
            result = fma(Int32(buf.stride[i]()), Int32(index[i]), result)

        return Int(result)

    else:
        var result: Int = 0

        comptime for i in range(rank):
            result = fma(buf.stride[i](), index[i], result)

        return result


@always_inline
fn _compute_ndbuffer_offset(
    buf: NDBuffer,
    idx: IndexList[buf.rank, ...],
) -> Int:
    """Computes the NDBuffer's offset using the index positions provided.

    Args:
        buf: The NDBuffer.
        idx: The index positions.

    Returns:
        The offset into the NDBuffer given the indices.
    """
    return _compute_ndbuffer_offset(buf, idx.as_tuple())


@always_inline
fn _compute_ndbuffer_stride[
    rank: Int
](shape: IndexList[rank, ...]) -> type_of(shape):
    """Computes the NDBuffer's default dynamic strides using the input shape.
    The default strides correspond to contiguous memory layout.

    Parameters:
        rank: The rank of the NDBuffer.

    Args:
        shape: The shape of the NDBuffer.

    Returns:
        The default strides of the NDBuffer.
    """
    comptime assert rank > 0

    comptime if rank == 1:
        return {1}

    var stride = shape
    stride[rank - 1] = 1

    comptime for i in reversed(range(1, rank)):
        stride[i - 1] = shape[i] * stride[i]

    return stride


# ===-----------------------------------------------------------------------===#
# NDBuffer
# ===-----------------------------------------------------------------------===#


# This type is "async safe" (see _async_parallelize).
@fieldwise_init
struct NDBuffer[
    mut: Bool,
    //,
    dtype: DType,
    rank: Int,
    origin: Origin[mut=mut],
    shape: DimList = DimList.create_unknown[rank](),
    strides: DimList = DimList.create_unknown[rank](),
    *,
    alignment2: Int = 1,
    address_space: AddressSpace = AddressSpace.GENERIC,
    exclusive: Bool = True,
](
    Defaultable,
    DevicePassable,
    ImplicitlyCopyable,
    Sized,
    Stringable,
    TrivialRegisterPassable,
    Writable,
):
    """An N-dimensional buffer.

    NDBuffer can be parametrized on rank, static dimensions and Dtype. It does
    not own its underlying pointer.

    Parameters:
        mut: The inferred mutability.
        dtype: The element dtype of the buffer.
        rank: The rank of the buffer.
        origin: The origin of the memory being addressed.
        shape: The static size (if known) of the buffer.
        strides: The strides (if known) of the buffer.
        alignment2: The preferred address alignment of the buffer.
        address_space: The address space of the buffer.
        exclusive: The underlying memory allocation of the tensor is known
            only to be accessible through this pointer.
    """

    comptime type = Self.dtype
    """The data type of elements in this buffer."""

    var data: UnsafePointer[
        Scalar[Self.dtype],
        Self.origin,
        address_space = Self.address_space,
    ]
    """The underlying data for the buffer. The pointer is not owned by the
    NDBuffer."""
    var dynamic_shape: IndexList[Self.rank, element_type = DType.uint64]
    """The dynamic value of the shape."""
    var dynamic_stride: IndexList[Self.rank, element_type = DType.uint64]
    """The dynamic stride of the buffer."""

    @staticmethod
    fn _default_alignment[width: Int = 1]() -> Int:
        return align_of[SIMD[Self.dtype, width]]() if is_nvidia_gpu() else 1

    @always_inline
    fn __init__(out self):
        """Default initializer for NDBuffer. By default the fields are all
        initialized to 0.
        """
        self.data = {}
        self.dynamic_shape = {}
        self.dynamic_stride = {}

    @always_inline
    fn __init__(
        out self,
        ptr: UnsafePointer[
            Scalar[Self.dtype],
            Self.origin,
            address_space = Self.address_space,
        ],
    ):
        """Constructs an NDBuffer with statically known rank, shapes and
        dtype.

        Constraints:
            The rank, shapes, and type are known.

        Args:
            ptr: Pointer to the data.
        """
        comptime assert Self.shape.all_known[
            Self.rank
        ](), "dimensions must all be known"

        self.data = ptr
        self.dynamic_shape = _make_tuple[
            Self.rank, element_type = DType.uint64
        ](Self.shape)
        self.dynamic_stride = _compute_ndbuffer_stride[Self.rank](
            self.dynamic_shape
        )

    @always_inline
    @implicit
    fn __init__(
        other: NDBuffer[mut=True, ...],
        out self: NDBuffer[
            other.dtype,
            other.rank,
            ImmutOrigin(other.origin),
            other.shape,
            other.strides,
            alignment2 = other.alignment2,
            address_space = other.address_space,
            exclusive = other.exclusive,
        ],
    ):
        """Implicitly cast a mutable NDBuffer to immutable.

        Args:
            other: The mutable NDBuffer to cast from.
        """
        self.data = other.data
        self.dynamic_shape = other.dynamic_shape
        self.dynamic_stride = other.dynamic_stride

    @always_inline
    @implicit
    fn __init__(
        other: NDBuffer,
        out self: NDBuffer[
            other.dtype,
            other.rank,
            other.origin,
            other.shape,
        ],
    ):
        """Rebinds the NDBuffer to one with unknown strides.

        Args:
            other: The source NDBuffer to rebind.
        """
        return rebind[type_of(self)](other)

    @always_inline
    fn __init__(
        out self,
        ptr: UnsafePointer[
            Scalar[Self.dtype],
            Self.origin,
            address_space = Self.address_space,
        ],
        dynamic_shape: IndexList[Self.rank, ...],
    ):
        """Constructs an NDBuffer with statically known rank, but dynamic
        shapes and type.

        Constraints:
            The rank is known.

        Args:
            ptr: Pointer to the data.
            dynamic_shape: A static tuple of size 'rank' representing shapes.
        """
        self.data = ptr
        self.dynamic_shape = rebind[type_of(self.dynamic_shape)](
            dynamic_shape.cast[type_of(self.dynamic_shape).element_type]()
        )
        self.dynamic_stride = _compute_ndbuffer_stride[Self.rank](
            self.dynamic_shape
        )

    @always_inline
    fn __init__(
        out self: Self.OriginCastType[MutAnyOrigin],
        ptr: UnsafePointer[Scalar[Self.dtype], ...],
        dynamic_shape: IndexList[Self.rank, ...],
    ):
        """Constructs an NDBuffer with MutAnyOrigin from any pointer.

        This overload allows origin erasure when constructing an NDBuffer with
        MutAnyOrigin, accepting pointers with any origin.

        Args:
            ptr: Pointer to the data.
            dynamic_shape: A static tuple of size 'rank' representing shapes.
        """
        self.data = rebind[type_of(self.data)](ptr)
        self.dynamic_shape = rebind[type_of(self.dynamic_shape)](
            dynamic_shape.cast[type_of(self.dynamic_shape).element_type]()
        )
        self.dynamic_stride = _compute_ndbuffer_stride[Self.rank](
            self.dynamic_shape
        )

    @always_inline
    fn __init__(
        out self,
        ptr: UnsafePointer[
            Scalar[Self.dtype],
            Self.origin,
            address_space = Self.address_space,
        ],
        dynamic_shape: DimList,
    ):
        """Constructs an NDBuffer with statically known rank, but dynamic
        shapes and type.

        Constraints:
            The rank is known.

        Args:
            ptr: Pointer to the data.
            dynamic_shape: A static tuple of size 'rank' representing shapes.
        """
        self = Self(ptr, _make_tuple[Self.rank](dynamic_shape))

    @always_inline
    fn __init__(
        out self: Self.OriginCastType[MutAnyOrigin],
        ptr: UnsafePointer[Scalar[Self.dtype], ...],
        dynamic_shape: DimList,
    ):
        """Constructs an NDBuffer with MutAnyOrigin from any pointer.

        This overload allows origin erasure when constructing an NDBuffer with
        MutAnyOrigin, accepting pointers with any origin.

        Args:
            ptr: Pointer to the data.
            dynamic_shape: A DimList of size 'rank' representing shapes.
        """
        var shape_tuple = _make_tuple[Self.rank](dynamic_shape)
        self.data = rebind[type_of(self.data)](ptr)
        self.dynamic_shape = rebind[type_of(self.dynamic_shape)](
            shape_tuple.cast[type_of(self.dynamic_shape).element_type]()
        )
        self.dynamic_stride = _compute_ndbuffer_stride[Self.rank](
            self.dynamic_shape
        )

    @always_inline
    fn __init__(
        out self,
        ptr: UnsafePointer[
            Scalar[Self.dtype],
            Self.origin,
            address_space = Self.address_space,
        ],
        dynamic_shape: IndexList[Self.rank, ...],
        dynamic_stride: IndexList[Self.rank, ...],
    ):
        """Constructs a strided NDBuffer with statically known rank, but
        dynamic shapes and type.

        Constraints:
            The rank is known.

        Args:
            ptr: Pointer to the data.
            dynamic_shape: A static tuple of size 'rank' representing shapes.
            dynamic_stride: A static tuple of size 'rank' representing strides.
        """
        self.data = ptr
        self.dynamic_shape = rebind[type_of(self.dynamic_shape)](
            dynamic_shape.cast[type_of(self.dynamic_shape).element_type]()
        )
        self.dynamic_stride = rebind[type_of(self.dynamic_stride)](
            dynamic_stride.cast[type_of(self.dynamic_shape).element_type]()
        )

    @always_inline
    fn __init__(
        out self: Self.OriginCastType[MutAnyOrigin],
        ptr: UnsafePointer[Scalar[Self.dtype], ...],
        dynamic_shape: IndexList[Self.rank, ...],
        dynamic_stride: IndexList[Self.rank, ...],
    ):
        """Constructs a strided NDBuffer with MutAnyOrigin from any pointer.

        This overload allows origin erasure when constructing an NDBuffer with
        MutAnyOrigin, accepting pointers with any origin.

        Args:
            ptr: Pointer to the data.
            dynamic_shape: A static tuple of size 'rank' representing shapes.
            dynamic_stride: A static tuple of size 'rank' representing strides.
        """
        self.data = rebind[type_of(self.data)](ptr)
        self.dynamic_shape = rebind[type_of(self.dynamic_shape)](
            dynamic_shape.cast[type_of(self.dynamic_shape).element_type]()
        )
        self.dynamic_stride = rebind[type_of(self.dynamic_stride)](
            dynamic_stride.cast[type_of(self.dynamic_shape).element_type]()
        )

    @always_inline
    fn __init__(
        out self,
        ptr: UnsafePointer[
            Scalar[Self.dtype],
            Self.origin,
            address_space = Self.address_space,
        ],
        dynamic_shape: DimList,
        dynamic_stride: IndexList[Self.rank, ...],
    ):
        """Constructs a strided NDBuffer with statically known rank, but
        dynamic shapes and type.

        Constraints:
            The rank is known.

        Args:
            ptr: Pointer to the data.
            dynamic_shape: A DimList of size 'rank' representing shapes.
            dynamic_stride: A static tuple of size 'rank' representing strides.
        """
        self = Self(
            ptr=ptr,
            dynamic_shape=_make_tuple[Self.rank](dynamic_shape),
            dynamic_stride=dynamic_stride,
        )

    @always_inline
    fn __init__(
        out self: Self.OriginCastType[MutAnyOrigin],
        ptr: UnsafePointer[Scalar[Self.dtype], ...],
        dynamic_shape: DimList,
        dynamic_stride: IndexList[Self.rank, ...],
    ):
        """Constructs a strided NDBuffer with MutAnyOrigin from any pointer.

        This overload allows origin erasure when constructing an NDBuffer with
        MutAnyOrigin, accepting pointers with any origin.

        Args:
            ptr: Pointer to the data.
            dynamic_shape: A DimList of size 'rank' representing shapes.
            dynamic_stride: A static tuple of size 'rank' representing strides.
        """
        var shape_tuple = _make_tuple[Self.rank](dynamic_shape)
        self.data = rebind[type_of(self.data)](ptr)
        self.dynamic_shape = rebind[type_of(self.dynamic_shape)](
            shape_tuple.cast[type_of(self.dynamic_shape).element_type]()
        )
        self.dynamic_stride = rebind[type_of(self.dynamic_stride)](
            dynamic_stride.cast[type_of(self.dynamic_shape).element_type]()
        )

    comptime OriginCastType[target_origin: Origin] = NDBuffer[
        Self.dtype,
        Self.rank,
        target_origin,
        Self.shape,
        Self.strides,
        alignment2 = Self.alignment2,
        address_space = Self.address_space,
        exclusive = Self.exclusive,
    ]
    """Type alias for casting this buffer to a different origin.

    Parameters:
        target_origin: The target origin to cast to.
    """

    @always_inline("nodebug")
    fn get_immutable(
        self,
    ) -> Self.OriginCastType[ImmutOrigin(Self.origin)]:
        """Changes the mutability of the `NDBuffer` to immutable.

        Returns:
            A buffer with the mutability set to immutable.
        """
        return {
            self.data.as_immutable(),
            self.dynamic_shape,
            self.dynamic_stride,
        }

    @always_inline("nodebug")
    fn as_any_origin(
        self: NDBuffer[mut=True, ...],
    ) -> type_of(self).OriginCastType[MutAnyOrigin]:
        """Changes the origin of the `NDBuffer` to `MutAnyOrigin`.

        Returns:
            A buffer with the origin set to `MutAnyOrigin`.

        This requires the buffer to already be mutable as casting mutability
        is inherently very unsafe.

        It is usually preferred to maintain concrete origin values instead of
        using `MutAnyOrigin`. However, if it is needed, keep in mind that
        `MutAnyOrigin` can alias any memory value, so Mojo's ASAP
        destruction will not apply during the lifetime of the buffer.
        """
        return {
            self.data.as_any_origin(),
            self.dynamic_shape,
            self.dynamic_stride,
        }

    @always_inline("nodebug")
    fn as_any_origin(
        self: NDBuffer[mut=False, ...],
    ) -> type_of(self).OriginCastType[ImmutAnyOrigin]:
        """Changes the origin of the `NDBuffer` to `ImmutAnyOrigin`.

        Returns:
            A buffer with the origin set to `ImmutAnyOrigin`.

        It is usually preferred to maintain concrete origin values instead of
        using `ImmutAnyOrigin`. However, if it is needed, keep in mind that
        `ImmutAnyOrigin` can alias any memory value, so Mojo's ASAP
        destruction will not apply during the lifetime of the buffer.
        """
        return {
            self.data.as_any_origin(),
            self.dynamic_shape,
            self.dynamic_stride,
        }

    @always_inline
    fn get_rank(self) -> Int:
        """Returns the rank of the buffer.

        Returns:
            The rank of NDBuffer.
        """
        return Self.rank

    @always_inline
    fn get_shape(self) -> IndexList[Self.rank]:
        """Returns the shapes of the buffer.

        Returns:
            A static tuple of size 'rank' representing shapes of the NDBuffer.
        """
        var res = IndexList[Self.rank]()

        comptime for i in range(Self.rank):
            res[i] = self.dim[i]()
        return res

    @always_inline
    fn get_strides(self) -> IndexList[Self.rank]:
        """Returns the strides of the buffer.

        Returns:
            A static tuple of size 'rank' representing strides of the NDBuffer.
        """
        var res = IndexList[Self.rank]()

        comptime for i in range(Self.rank):
            res[i] = self.stride[i]()
        return res

    @always_inline
    fn get_nd_index(self, idx: Int) -> IndexList[Self.rank]:
        """Computes the NDBuffer's ND-index based on the flat index.

        Args:
            idx: The flat index.

        Returns:
            The index positions.
        """
        return _compute_nd_index(self, idx)

    @always_inline
    fn __len__(self) -> Int:
        """Computes the NDBuffer's number of elements.

        Returns:
            The total number of elements in the NDBuffer.
        """
        return self.size()

    @always_inline
    fn num_elements(self) -> Int:
        """Computes the NDBuffer's number of elements.

        Returns:
            The total number of elements in the NDBuffer.
        """
        return self.size()

    @always_inline
    fn size(self) -> Int:
        """Computes the NDBuffer's number of elements.

        Returns:
            The total number of elements in the NDBuffer.
        """
        var product: Int = 1

        comptime for i in range(Self.rank):
            product *= self.dim(i)

        return product

    @no_inline
    fn __str__(self) -> String:
        """Gets the buffer as a string.

        Returns:
          A compact string of the buffer.
        """
        return String.write(self)

    fn write_to(self, mut writer: Some[Writer]):
        """
        Formats this buffer to the provided Writer.

        Args:
            writer: The object to write to.
        """
        writer.write("NDBuffer(")

        @parameter
        fn serialize[T: Writable](val: T):
            writer.write(val)

        var dyn_shape = List[Int]()
        for i in range(Self.rank):
            dyn_shape.append(self.dynamic_shape[i])

        _serialize[serialize_fn=serialize, serialize_end_line=False](
            self.data, dyn_shape
        )

        writer.write(")")

    @no_inline
    fn __repr__(self) -> String:
        """Gets the buffer as a string.

        Returns:
          A compact string representation of the buffer.
        """
        return self.__str__()

    @always_inline
    fn _offset(
        self, idx: VariadicList[Int]
    ) -> UnsafePointer[
        Scalar[Self.dtype],
        Self.origin,
        address_space = Self.address_space,
    ]:
        """Computes the NDBuffer's offset using the index positions provided.

        Args:
            idx: The index positions.

        Returns:
            The offset into the NDBuffer given the indices.
        """
        comptime assert Self.rank <= _MAX_RANK
        return self.data + _compute_ndbuffer_offset(self, idx)

    @always_inline
    fn _offset(
        self, idx: IndexList[Self.rank, ...]
    ) -> UnsafePointer[
        Scalar[Self.dtype],
        Self.origin,
        address_space = Self.address_space,
    ]:
        comptime assert Self.rank <= _MAX_RANK
        return self.data + _compute_ndbuffer_offset(self, idx.as_tuple())

    @always_inline
    fn _offset(
        self, idx: StaticTuple[Int, Self.rank]
    ) -> UnsafePointer[
        Scalar[Self.dtype],
        Self.origin,
        address_space = Self.address_space,
    ]:
        """Computes the NDBuffer's offset using the index positions provided.

        Args:
            idx: The index positions.

        Returns:
            The offset into the NDBuffer given the indices.
        """
        comptime assert Self.rank <= _MAX_RANK
        return self.data + _compute_ndbuffer_offset(self, idx)

    @always_inline
    fn __getitem__(self, *idx: Int) -> Scalar[Self.dtype]:
        """Gets an element from the buffer from the specified index.

        Args:
            idx: Index of the element to retrieve.

        Returns:
            The value of the element.
        """
        return self.load[width=1](idx)

    @always_inline
    fn __getitem__(self, idx: IndexList[Self.rank, ...]) -> Scalar[Self.dtype]:
        """Gets an element from the buffer from the specified index.

        Args:
            idx: Index of the element to retrieve.

        Returns:
            The value of the element.
        """
        return self.load[width=1](idx)

    @always_inline
    fn tile[
        *tile_sizes: Dim
    ](self, tile_coords: IndexList[Self.rank, ...]) -> NDBuffer[
        Self.dtype,
        Self.rank,
        Self.origin,
        DimList(tile_sizes),
        address_space = Self.address_space,
    ]:
        """Returns an n-d tile "slice" of the buffer of size tile_sizes at
           coords.

        Parameters:
            tile_sizes: The size of the tiles.

        Args:
            tile_coords: The tile index.

        Returns:
            The tiled buffer at tile_coords.
        """

        comptime num_tile_sizes = std.builtin.Variadic.size(tile_sizes)

        comptime assert (
            num_tile_sizes == Self.rank
        ), "The tile should have the same rank as the buffer"

        comptime assert DimList(tile_sizes).all_known[
            Self.rank
        ](), "Static tile sizes are only supported"

        var offset = 0
        var dyn_shape = IndexList[Self.rank]()

        comptime for i in range(Self.rank):
            comptime tile_size_i = tile_sizes[i].get()
            dyn_shape[i] = tile_size_i
            var coord_i = tile_coords[i]
            offset += coord_i * tile_size_i * self.stride[i]()

        # The tile buffer has the same stride and an offset calculated as
        # computed above, why?
        # Consider the 2d case, tile(i, j) of size tile_m, tile_n can be accessed
        # at buffer(i + m * tile_m, j + n * tile_n) =
        #    dot((i + m * tile_m, j + n * tile_n), stride)
        # = dot((i, j), stride) + dot(((m * tile_m), (n * tile_n)), stride)
        # which tells us the tile has a stride of the original buffer stride and
        # offset = dot(((m * tile_m), (n * tile_n)), stride).
        var tile = NDBuffer[
            Self.dtype,
            Self.rank,
            Self.origin,
            DimList(tile_sizes),
            address_space = Self.address_space,
        ](
            self.data + offset,
            dynamic_shape=dyn_shape,
            dynamic_stride=self.dynamic_stride,
        )
        return tile

    @always_inline("nodebug")
    fn load[
        *, width: Int = 1, alignment: Int = Self._default_alignment[width]()
    ](self, *idx: Int) -> SIMD[Self.dtype, width]:
        """Loads a simd value from the buffer at the specified index.

        Constraints:
            The buffer must be contiguous or width must be 1.

        Parameters:
            width: The simd_width of the load.
            alignment: The alignment value.

        Args:
            idx: The index into the NDBuffer.

        Returns:
            The simd value starting at the `idx` position and ending at
            `idx+width`.
        """
        return self.load[width=width, alignment=alignment](idx)

    @always_inline("nodebug")
    fn load[
        *, width: Int = 1, alignment: Int = Self._default_alignment[width]()
    ](self, idx: VariadicList[Int]) -> SIMD[Self.dtype, width]:
        """Loads a simd value from the buffer at the specified index.

        Constraints:
            The buffer must be contiguous or width must be 1.

        Parameters:
            width: The simd_width of the load.
            alignment: The alignment value.

        Args:
            idx: The index into the NDBuffer.

        Returns:
            The simd value starting at the `idx` position and ending at
            `idx+width`.
        """
        debug_assert(
            self.is_contiguous() or width == 1,
            "Function requires contiguous buffer.",
        )
        return self._offset(idx).load[width=width, alignment=alignment]()

    @always_inline("nodebug")
    fn load[
        *, width: Int = 1, alignment: Int = Self._default_alignment[width]()
    ](self, idx: IndexList) -> SIMD[Self.dtype, width]:
        """Loads a simd value from the buffer at the specified index.

        Constraints:
            The buffer must be contiguous or width must be 1.

        Parameters:
            width: The simd_width of the load.
            alignment: The alignment value.

        Args:
            idx: The index into the NDBuffer.

        Returns:
            The simd value starting at the `idx` position and ending at
            `idx+width`.
        """
        comptime assert idx.size == Self.rank, "invalid index size"
        return self.load[width=width, alignment=alignment](
            rebind[IndexList[Self.rank, element_type = idx.element_type]](
                idx
            ).as_tuple()
        )

    @always_inline("nodebug")
    fn load[
        *, width: Int = 1, alignment: Int = Self._default_alignment[width]()
    ](self, idx: StaticTuple[Int, Self.rank]) -> SIMD[Self.dtype, width]:
        """Loads a simd value from the buffer at the specified index.

        Constraints:
            The buffer must be contiguous or width must be 1.

        Parameters:
            width: The simd_width of the load.
            alignment: The alignment value.

        Args:
            idx: The index into the NDBuffer.

        Returns:
            The simd value starting at the `idx` position and ending at
            `idx+width`.
        """
        debug_assert(
            self.is_contiguous() or width == 1,
            "Function requires contiguous buffer.",
        )
        return self._offset(idx).load[width=width, alignment=alignment]()

    @always_inline
    fn __setitem__(
        self: NDBuffer[
            mut=True,
            Self.dtype,
            Self.rank,
            _,
            shape = Self.shape,
            strides = Self.strides,
            alignment2 = Self.alignment2,
            address_space = Self.address_space,
            exclusive = Self.exclusive,
        ],
        idx: IndexList[Self.rank, ...],
        val: Scalar[Self.dtype],
    ):
        """Stores a single value into the buffer at the specified index.

        Args:
            idx: The index into the buffer.
            val: The value to store.
        """
        self.store[width=1](idx, val)

    @always_inline
    fn __setitem__(
        self: NDBuffer[
            mut=True,
            Self.dtype,
            Self.rank,
            _,
            shape = Self.shape,
            strides = Self.strides,
            alignment2 = Self.alignment2,
            address_space = Self.address_space,
            exclusive = Self.exclusive,
        ],
        *idx: Int,
        val: Scalar[Self.dtype],
    ):
        """Stores a single value into the buffer at the specified index.

        Args:
            idx: Index of the element to retrieve.
            val: The value to store.
        """
        self.store[width=1](IndexList[Self.rank](idx), val)

    @always_inline("nodebug")
    fn store[
        _alignment: Int,
        //,
        *,
        width: Int = 1,
        alignment: Int = Self._default_alignment[width](),
    ](
        self: NDBuffer[
            mut=True,
            Self.dtype,
            Self.rank,
            _,
            shape = Self.shape,
            strides = Self.strides,
            alignment2=_alignment,
            address_space = Self.address_space,
            exclusive = Self.exclusive,
        ],
        idx: IndexList[Self.rank, ...],
        val: SIMD[Self.dtype, width],
    ):
        """Stores a simd value into the buffer at the specified index.

        Constraints:
            The buffer must be contiguous or width must be 1.

        Parameters:
            _alignment: The inferred alignment of self.
            width: The width of the simd vector.
            alignment: The alignment value.

        Args:
            idx: The index into the buffer.
            val: The value to store.
        """
        self.store[width=width, alignment=alignment](idx.as_tuple(), val)

    @always_inline("nodebug")
    fn store[
        _alignment: Int,
        //,
        *,
        width: Int = 1,
        alignment: Int = Self._default_alignment[width](),
    ](
        self: NDBuffer[
            mut=True,
            Self.dtype,
            Self.rank,
            _,
            shape = Self.shape,
            strides = Self.strides,
            alignment2=_alignment,
            address_space = Self.address_space,
            exclusive = Self.exclusive,
        ],
        idx: StaticTuple[Int, Self.rank],
        val: SIMD[Self.dtype, width],
    ):
        """Stores a simd value into the buffer at the specified index.

        Constraints:
            The buffer must be contiguous or width must be 1.

        Parameters:
            _alignment: The inferred alignment of self.
            width: The width of the simd vector.
            alignment: The alignment value.

        Args:
            idx: The index into the buffer.
            val: The value to store.
        """
        debug_assert(
            self.is_contiguous() or width == 1,
            "Function requires contiguous buffer.",
        )
        self._offset(idx).store[alignment=alignment](val)

    @always_inline
    fn dim[index: Int](self) -> Int:
        """Gets the buffer dimension at the given index.

        Parameters:
            index: The number of dimension to get.

        Returns:
            The buffer size at the given dimension.
        """
        # First try to extract the static info on this dimension, could be either a
        # meta constant or an unknown.
        comptime static_dim_value = Self.shape.at[index]()

        comptime if static_dim_value.has_value():
            return static_dim_value.get()
        return self.dynamic_shape[index]

    @always_inline
    fn dim(self, index: Int) -> Int:
        """Gets the buffer dimension at the given index.

        Args:
            index: The number of dimension to get.

        Returns:
            The buffer size at the given dimension.
        """
        return self.dynamic_shape[index]

    @always_inline
    fn stride[index: Int](self) -> Int:
        """Gets the buffer stride at the given index.

        Parameters:
            index: The number of dimension to get the stride for.

        Returns:
            The stride at the given dimension.
        """
        # First try to extract the static info on this stride, could be either a
        # meta constant or an unknown.
        comptime static_stride_value = Self.strides.at[index]()

        comptime if static_stride_value.has_value():
            return static_stride_value.get()
        return self.dynamic_stride[index]

    @always_inline
    fn stride(self, index: Int) -> Int:
        """Gets the buffer stride at the given index.

        Args:
            index: The number of dimension to get the stride for.

        Returns:
            The stride at the given dimension.
        """
        return self.dynamic_stride[index]

    @always_inline
    fn is_contiguous(self) -> Bool:
        """Checks if the buffer is contiguous in memory.

        Returns:
            True if the buffer is contiguous in memory and False otherwise.
        """

        comptime assert Self.rank > 0, "rank must be positive"
        return self.stride[Self.rank - 1]() == 1

    @always_inline
    fn flatten(
        self,
        out result: NDBuffer[
            Self.dtype,
            1,
            Self.origin,
            Self.shape.product(),
            address_space = Self.address_space,
        ],
    ):
        """Constructs a flattened buffer counterpart for this NDBuffer.

        Constraints:
            The buffer must be contiguous.

        Returns:
            Constructed buffer object.
        """
        debug_assert(
            self.is_contiguous(), "Function requires contiguous buffer."
        )
        return {self.data, self.size()}

    @always_inline
    fn make_dims_unknown(
        self,
        out result: NDBuffer[
            Self.dtype,
            Self.rank,
            address_space = Self.address_space,
            origin = Self.origin,
        ],
    ):
        """Rebinds the NDBuffer to one with unknown shape.

        Returns:
            The rebound NDBuffer with unknown shape.
        """
        return rebind[type_of(result)](self)

    @always_inline
    fn bytecount(self) -> Int:
        """Returns the size of the NDBuffer in bytes.

        Returns:
            The size of the NDBuffer in bytes.
        """
        return self.size() * size_of[Self.dtype]()

    @always_inline
    fn zero(self: NDBuffer[mut=True, ...]):
        """Sets all bytes of the NDBuffer to 0.

        Constraints:
            The buffer must be contiguous.
        """
        debug_assert(
            self.is_contiguous(), "Function requires contiguous buffer."
        )

        comptime if Self.shape.all_known[Self.rank]():
            comptime count = Int(Self.shape.product())
            memset_zero[count=count](self.data)
        else:
            memset_zero(self.data, len(self))

    @always_inline
    fn _simd_fill[
        simd_width: Int
    ](
        self: NDBuffer[
            mut=True,
            Self.dtype,
            Self.rank,
            _,
            shape = Self.shape,
            strides = Self.strides,
            alignment2 = Self.alignment2,
            address_space = Self.address_space,
            exclusive = Self.exclusive,
        ],
        val: Scalar[Self.dtype],
    ):
        """Assigns val to all elements in chunks of size simd_width.

        Parameters:
            simd_width: The simd_width of the fill.

        Args:
            val: The value to store.
        """

        comptime if Self.rank > 1:
            if val == 0:
                self.zero()
                return
            self.flatten()._simd_fill[simd_width](val)
        else:
            if val == 0:
                self.zero()
                return

            var vec_end = align_down(len(self), simd_width)
            for i in range(0, vec_end, simd_width):
                self.store[width=simd_width](IndexList[Self.rank](i), val)
            for i in range(vec_end, len(self)):
                self.store(IndexList[Self.rank](i), val)

    @always_inline
    fn tofile(self, path: Path) raises:
        """Write values to a file.

        Args:
            path: Path to the output file.

        Raises:
            If the operation fails.
        """
        with open(String(path), "w") as f:
            var ptr = self.data.bitcast[UInt8]()
            f._write(ptr, self.bytecount())

    @always_inline
    fn fill(
        self: NDBuffer[
            mut=True,
            Self.dtype,
            Self.rank,
            _,
            shape = Self.shape,
            strides = Self.strides,
            alignment2 = Self.alignment2,
            address_space = Self.address_space,
            exclusive = Self.exclusive,
        ],
        val: Scalar[Self.dtype],
    ):
        """Assigns val to all elements in the buffer.

        The fill is performed in chunks of size N, where N is the native SIMD
        width of type on the system.

        Args:
            val: The value to store.
        """
        debug_assert(
            self.is_contiguous(), "Function requires contiguous buffer."
        )
        self._simd_fill[simd_width_of[Self.dtype]()](val)

    @staticmethod
    @always_inline("nodebug")
    fn stack_allocation[*, alignment: Int = align_of[Self.dtype]()]() -> Self:
        """Constructs an NDBuffer instance backed by stack allocated memory space.

        Parameters:
            alignment: Address alignment requirement for the allocation.

        Returns:
            Constructed NDBuffer with the allocated space.
        """
        comptime assert Self.shape.all_known[Self.rank](), (
            "the shape of the NDBuffer must be known to allow for stack"
            " allocation"
        )
        var data_pointer = (
            stack_allocation[
                Self.shape.product[Self.rank]().get(),
                Self.dtype,
                alignment=alignment,
                address_space = Self.address_space,
            ]()
            .unsafe_mut_cast[Self.mut]()
            .unsafe_origin_cast[Self.origin]()
        )
        return Self(data_pointer)

    @always_inline
    fn prefetch[params: PrefetchOptions](self, *idx: Int):
        """Prefetches the data at the given index.

        Parameters:
            params: The prefetch configuration.

        Args:
            idx: The N-D index of the prefetched location.
        """
        prefetch[params](self._offset(idx))

    @always_inline
    fn prefetch[params: PrefetchOptions](self, indices: IndexList[Self.rank]):
        """Prefetches the data at the given index.

        Parameters:
            params: The prefetch configuration.

        Args:
            indices: The N-D index of the prefetched location.
        """
        prefetch[params](self._offset(indices))

    # `trait DevicePassable` implementation
    comptime device_type: AnyType = Self
    """The device-side type for this buffer."""

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        """Convert the host type object to a device_type and store it at the
        target address.

        Args:
            target: The target address to store the device type.
        """
        target.bitcast[Self.device_type]()[] = self

    @no_inline
    @staticmethod
    fn get_type_name() -> String:
        """Gets the name of the host type (the one implementing this trait).

        Returns:
            The host type's name.
        """
        return String(
            "NDBuffer[mut = ",
            String(Self.mut),
            ", dtype = ",
            String(Self.dtype),
            ", rank = ",
            String(Self.rank),
            ", address_space = ",
            String(Self.address_space),
            "]",
        )


@always_inline
fn partial_simd_load[
    dtype: DType, //, width: Int
](
    storage: UnsafePointer[mut=False, Scalar[dtype], ...],
    lbound: Int,
    rbound: Int,
    pad_value: Scalar[dtype],
) -> SIMD[dtype, width]:
    """Loads a vector with dynamic bound.

    Out of bound data will be filled with pad value. Data is valid if
    lbound <= idx < rbound for idx from 0 to (simd_width-1). For example:

        addr 0  1  2  3
        data x 42 43  x

        partial_simd_load[4](addr0, 1, 3) #gives [0 42 43 0]

    Parameters:
        dtype: The DType of storage.
        width: The system simd vector size.

    Args:
        storage: Pointer to the address to perform load.
        lbound: Lower bound of valid index within simd (inclusive).
        rbound: Upper bound of valid index within simd (non-inclusive).
        pad_value: Value to fill for out of bound indices.

    Returns:
        The SIMD vector loaded and zero-filled.
    """
    # Create a mask based on input bounds.
    var effective_lbound = SIMD[DType.int32, width](max(lbound, 0))
    var effective_rbound = SIMD[DType.int32, width](min(width, rbound))
    var incr = iota[DType.int32, width]()
    var mask = incr.ge(effective_lbound) & incr.lt(effective_rbound)

    return masked_load[width](storage, mask, pad_value)


@always_inline
fn partial_simd_store[
    dtype: DType, //, width: Int
](
    storage: UnsafePointer[mut=True, Scalar[dtype], ...],
    lbound: Int,
    rbound: Int,
    data: SIMD[dtype, width],
):
    """Stores a vector with dynamic bound.

    Out of bound data will ignored. Data is valid if lbound <= idx < rbound for
    idx from 0 to (simd_width-1).

    e.g.
        addr 0 1 2  3
        data 0 0 0  0

        partial_simd_load[4](addr0, 1, 3, [-1, 42, 43, -1]) #gives [0 42 43 0]

    Parameters:
        dtype: The DType of storage.
        width: The system simd vector size.

    Args:
        storage: Pointer to the address to perform load.
        lbound: Lower bound of valid index within simd (inclusive).
        rbound: Upper bound of valid index within simd (non-inclusive).
        data: The vector value to store.
    """
    # Create a mask based on input bounds.
    var effective_lbound = SIMD[DType.int32, width](max(lbound, 0))
    var effective_rbound = SIMD[DType.int32, width](min(width, rbound))
    var incr = iota[DType.int32, width]()
    var mask = incr.ge(effective_lbound) & incr.lt(effective_rbound)

    return masked_store(data, storage, mask)


@always_inline
fn prod_dims[start_dim: Int, end_dim: Int](x: NDBuffer) -> Int:
    """Computes the product of a slice of the given buffer's dimensions.

    Parameters:
        start_dim: The index at which to begin computing the product.
        end_dim: The index at which to stop computing the product.

    Args:
        x: The NDBuffer whose dimensions will be multiplied.

    Returns:
        The product of the specified slice of the buffer's dimensions.
    """

    var product: Int = 1

    comptime for i in range(start_dim, end_dim):
        product *= x.dim[i]()

    return product
