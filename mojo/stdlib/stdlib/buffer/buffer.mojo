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
"""Implements the NDBuffer struct.

You can import these APIs from the `memory` package. For example:

```mojo
from buffer import NDBuffer
```
"""

from math import align_down, fma, iota
from pathlib import Path
from sys.info import alignof, is_gpu, is_nvidia_gpu, simdwidthof, sizeof
from sys.intrinsics import PrefetchOptions, masked_load, masked_store, prefetch

from buffer.dimlist import Dim, DimList, _make_tuple
from memory import memset_zero, stack_allocation
from memory.pointer import AddressSpace, _GPUAddressSpace

from utils._serialize import _serialize
from utils.index import IndexList
from utils.static_tuple import StaticTuple

alias _MAX_RANK = 8
"""The maximum tensor rank for any tensor shape.
This value must match kMaxRank in Support/include/Support/ML/TensorShape.h
"""


# ===-----------------------------------------------------------------------===#
# NDBuffer Utilities
# ===-----------------------------------------------------------------------===#


@always_inline
fn _use_32bit_indexing[address_space: AddressSpace]() -> Bool:
    return is_gpu() and address_space in (
        _GPUAddressSpace.SHARED,
        _GPUAddressSpace.LOCAL,
        _GPUAddressSpace.CONSTANT,
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

    @parameter
    if buf.rank == 0:
        return IndexList[buf.rank](0)

    var result = IndexList[buf.rank]()

    var curr_index = index

    @parameter
    for i in reversed(range(buf.rank)):
        var dim = buf.dim[i]()
        result[i] = curr_index._positive_rem(dim)
        curr_index = curr_index._positive_div(dim)

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

    alias rank = buf.rank

    @parameter
    if buf.rank == 0:
        return 0

    @parameter
    if _use_32bit_indexing[buf.address_space]():
        var result: Int32 = 0

        @parameter
        for i in range(buf.rank):
            result = fma(Int32(buf.stride[i]()), Int32(index[i]), result)

        return Int(result)

    else:
        var result: Int = 0

        @parameter
        for i in range(buf.rank):
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

    alias rank = buf.rank

    @parameter
    if rank == 0:
        return 0

    @parameter
    if _use_32bit_indexing[buf.address_space]():
        var result: Int32 = 0

        @parameter
        for i in range(rank):
            result = fma(Int32(buf.stride[i]()), Int32(index[i]), result)

        return Int(result)

    else:
        var result: Int = 0

        @parameter
        for i in range(rank):
            result = fma(buf.stride[i](), index[i], result)

        return result


@always_inline
fn _compute_ndbuffer_offset(
    buf: NDBuffer,
    idx: IndexList[buf.rank, **_],
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
](shape: IndexList[rank, **_]) -> __type_of(shape):
    """Computes the NDBuffer's default dynamic strides using the input shape.
    The default strides correspond to contiguous memory layout.

    Parameters:
        rank: The rank of the NDBuffer.

    Args:
        shape: The shape of the NDBuffer.

    Returns:
        The default strides of the NDBuffer.
    """
    constrained[rank > 0]()

    @parameter
    if rank == 1:
        return __type_of(shape)(1)

    var stride = shape
    stride[rank - 1] = 1

    @parameter
    for i in reversed(range(1, rank)):
        stride[i - 1] = shape[i] * stride[i]

    return stride


# ===-----------------------------------------------------------------------===#
# NDBuffer
# ===-----------------------------------------------------------------------===#


# This type is "async safe" (see _async_parallelize).
@fieldwise_init
@register_passable("trivial")
struct NDBuffer[
    mut: Bool, //,
    type: DType,
    rank: Int,
    origin: Origin[mut],
    shape: DimList = DimList.create_unknown[rank](),
    strides: DimList = DimList.create_unknown[rank](),
    *,
    alignment: Int = 1,
    address_space: AddressSpace = AddressSpace.GENERIC,
    exclusive: Bool = True,
](Sized, Stringable, Writable, Copyable, Movable, Defaultable):
    """An N-dimensional buffer.

    NDBuffer can be parametrized on rank, static dimensions and Dtype. It does
    not own its underlying pointer.

    Parameters:
        mut: The inferred mutability.
        type: The element type of the buffer.
        rank: The rank of the buffer.
        origin: The origin of the memory being addressed.
        shape: The static size (if known) of the buffer.
        strides: The strides (if known) of the buffer.
        alignment: The preferred address alignment of the buffer.
        address_space: The address space of the buffer.
        exclusive: The underlying memory allocation of the tensor is known
            only to be accessible through this pointer.
    """

    var data: UnsafePointer[
        Scalar[type], address_space=address_space, mut=mut, origin=origin
    ]
    """The underlying data for the buffer. The pointer is not owned by the
    NDBuffer."""
    var dynamic_shape: IndexList[rank, element_type = DType.uint64]
    """The dynamic value of the shape."""
    var dynamic_stride: IndexList[rank, element_type = DType.uint64]
    """The dynamic stride of the buffer."""

    @staticmethod
    fn _default_alignment[width: Int = 1]() -> Int:
        return alignof[SIMD[type, width]]() if is_nvidia_gpu() else 1

    @always_inline
    fn __init__(out self):
        """Default initializer for NDBuffer. By default the fields are all
        initialized to 0.
        """
        self.data = {}
        self.dynamic_shape = {}
        self.dynamic_stride = {}

    @always_inline
    @implicit
    fn __init__(
        out self,
        ptr: UnsafePointer[
            Scalar[type],
            address_space=address_space,
            mut=mut,
            origin=origin, **_,
        ],
    ):
        """Constructs an NDBuffer with statically known rank, shapes and
        type.

        Constraints:
            The rank, shapes, and type are known.

        Args:
            ptr: Pointer to the data.
        """
        constrained[
            shape.all_known[rank](),
            "dimensions must all be known",
        ]()

        self.data = ptr
        self.dynamic_shape = _make_tuple[rank, element_type = DType.uint64](
            shape
        )
        self.dynamic_stride = _compute_ndbuffer_stride[rank](self.dynamic_shape)

    @always_inline
    @implicit
    fn __init__(
        out self,
        span: Span[
            Scalar[type],
            address_space=address_space,
            origin=origin, **_,
        ],
    ):
        """Constructs an NDBuffer with statically known rank, shapes and
        type.

        Constraints:
            The rank, shapes, and type are known.

        Args:
            span: Span of the data.
        """
        constrained[
            shape.all_known[rank](),
            "dimensions must all be known",
        ]()
        self = Self(span.unsafe_ptr())

    @always_inline
    @implicit
    fn __init__(
        out self,
        # For functions
        other: NDBuffer[type, rank, *_, **_],
    ):
        """Converts NDBuffers between different variants which do not effect
        the underlying memory representation.

        E.g. this allows implicit conversion between

        `NDBuffer[type, rank, DimList(1, 2, 3), DimList(6, 6, 1), alignment=16]`
          to
        `NDBuffer[type, rank, DimList(1, 2, 3), DimList.create_unknown[rank](), alignment=4]`

        Args:
            other: The other NDBuffer type.
        """
        # It is probably unsafe to convert between address spaces
        constrained[other.address_space == address_space]()

        # We can only downgrade our alignment
        constrained[
            other.alignment >= alignment and other.alignment % alignment == 0
        ]()

        # Exclusivity can only be lost
        constrained[other.exclusive == exclusive or not exclusive]()

        # We can lose information about shape/stride, but not gain information
        alias unknown_dim_list = DimList.create_unknown[rank]()
        constrained[other.shape == shape or shape == unknown_dim_list]()
        constrained[other.strides == strides or strides == unknown_dim_list]()

        self.data = rebind[__type_of(self.data)](other.data)
        self.dynamic_shape = other.dynamic_shape
        self.dynamic_stride = other.dynamic_stride

    @always_inline
    fn __init__(
        out self,
        ptr: UnsafePointer[
            Scalar[type]._mlir_type,
            address_space=address_space,
            mut=mut,
            origin=origin,
        ],
        dynamic_shape: IndexList[rank, **_],
    ):
        """Constructs an NDBuffer with statically known rank, but dynamic
        shapes and type.

        Constraints:
            The rank is known.

        Args:
            ptr: Pointer to the data.
            dynamic_shape: A static tuple of size 'rank' representing shapes.
        """
        self.data = ptr.bitcast[Scalar[type]]()
        self.dynamic_shape = rebind[__type_of(self.dynamic_shape)](
            dynamic_shape.cast[__type_of(self.dynamic_shape).element_type]()
        )
        self.dynamic_stride = _compute_ndbuffer_stride[rank](self.dynamic_shape)

    @always_inline
    fn __init__(
        out self,
        ptr: UnsafePointer[
            Scalar[type], address_space=address_space, mut=mut, origin=origin
        ],
        dynamic_shape: IndexList[rank, **_],
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
        self.dynamic_shape = rebind[__type_of(self.dynamic_shape)](
            dynamic_shape.cast[__type_of(self.dynamic_shape).element_type]()
        )
        self.dynamic_stride = _compute_ndbuffer_stride[rank](self.dynamic_shape)

    @always_inline
    fn __init__(
        out self,
        span: Span[Scalar[type], address_space=address_space, origin=origin],
        dynamic_shape: IndexList[rank, **_],
    ):
        """Constructs an NDBuffer with statically known rank, but dynamic
        shapes and type.

        Constraints:
            The rank is known.

        Args:
            span: Span of the data.
            dynamic_shape: A static tuple of size 'rank' representing shapes.
        """
        self = Self(span.unsafe_ptr(), dynamic_shape)

    @always_inline
    fn __init__(
        out self,
        ptr: UnsafePointer[
            Scalar[type], address_space=address_space, mut=mut, origin=origin
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
        self = Self(ptr, _make_tuple[rank](dynamic_shape))

    @always_inline
    fn __init__(
        out self,
        span: Span[Scalar[type], address_space=address_space, origin=origin],
        dynamic_shape: DimList,
    ):
        """Constructs an NDBuffer with statically known rank, but dynamic
        shapes and type.

        Constraints:
            The rank is known.

        Args:
            span: Span of the data.
            dynamic_shape: A static tuple of size 'rank' representing shapes.
        """
        self = Self(span.unsafe_ptr(), dynamic_shape)

    @always_inline
    fn __init__(
        out self,
        ptr: UnsafePointer[
            Scalar[type], address_space=address_space, mut=mut, origin=origin
        ],
        dynamic_shape: IndexList[rank, **_],
        dynamic_stride: IndexList[rank, **_],
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
        self.dynamic_shape = rebind[__type_of(self.dynamic_shape)](
            dynamic_shape.cast[__type_of(self.dynamic_shape).element_type]()
        )
        self.dynamic_stride = rebind[__type_of(self.dynamic_stride)](
            dynamic_stride.cast[__type_of(self.dynamic_shape).element_type]()
        )

    @always_inline
    fn __init__(
        out self,
        span: Span[Scalar[type], address_space=address_space, origin=origin],
        dynamic_shape: IndexList[rank, **_],
        dynamic_stride: IndexList[rank, **_],
    ):
        """Constructs a strided NDBuffer with statically known rank, but
        dynamic shapes and type.

        Constraints:
            The rank is known.

        Args:
            span: Span over the data.
            dynamic_shape: A static tuple of size 'rank' representing shapes.
            dynamic_stride: A static tuple of size 'rank' representing strides.
        """
        self = Self(span.unsafe_ptr(), dynamic_shape, dynamic_stride)

    @always_inline
    fn __init__(
        out self,
        ptr: UnsafePointer[
            Scalar[type], address_space=address_space, mut=mut, origin=origin
        ],
        dynamic_shape: DimList,
        dynamic_stride: IndexList[rank, **_],
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
            dynamic_shape=_make_tuple[rank](dynamic_shape),
            dynamic_stride=dynamic_stride,
        )

    @always_inline
    fn __init__(
        out self,
        span: Span[Scalar[type], address_space=address_space, origin=origin],
        dynamic_shape: DimList,
        dynamic_stride: IndexList[rank, **_],
    ):
        """Constructs a strided NDBuffer with statically known rank, but
        dynamic shapes and type.

        Constraints:
            The rank is known.

        Args:
            span: Pointer to the data.
            dynamic_shape: A DimList of size 'rank' representing shapes.
            dynamic_stride: A static tuple of size 'rank' representing strides.
        """
        self = Self(span.unsafe_ptr(), dynamic_shape, dynamic_stride)

    @always_inline("nodebug")
    fn origin_cast[
        mut: Bool = Self.mut,
        origin: Origin[mut] = Origin[mut].cast_from[Self.origin].result,
    ](
        self,
        out result: NDBuffer[
            type,
            rank,
            origin,
            shape,
            strides,
            alignment=alignment,
            address_space=address_space,
            exclusive=exclusive,
        ],
    ):
        """Changes the origin or mutability of a pointer.

        Parameters:
            mut: Whether the origin is mutable.
            origin: Origin of the destination pointer.

        Returns:
            A new `NDBuffer` object with the same type and the same address,
            as the original `NDBuffer` and the new specified mutability and origin.
        """
        result = __type_of(result)(
            self.data.origin_cast[mut, origin](),
            self.dynamic_shape,
            self.dynamic_stride,
        )

    @always_inline
    fn get_rank(self) -> Int:
        """Returns the rank of the buffer.

        Returns:
            The rank of NDBuffer.
        """
        return rank

    @always_inline
    fn get_shape(self) -> IndexList[rank]:
        """Returns the shapes of the buffer.

        Returns:
            A static tuple of size 'rank' representing shapes of the NDBuffer.
        """
        var res = IndexList[rank]()

        @parameter
        for i in range(rank):
            res[i] = self.dim[i]()
        return res

    @always_inline
    fn get_strides(self) -> IndexList[rank]:
        """Returns the strides of the buffer.

        Returns:
            A static tuple of size 'rank' representing strides of the NDBuffer.
        """
        var res = IndexList[rank]()

        @parameter
        for i in range(rank):
            res[i] = self.stride[i]()
        return res

    @always_inline
    fn get_nd_index(self, idx: Int) -> IndexList[rank]:
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

        @parameter
        for i in range(rank):
            product *= self.dim(i)

        return product

    @no_inline
    fn __str__(self) -> String:
        """Gets the buffer as a string.

        Returns:
          A compact string of the buffer.
        """
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this buffer to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """
        writer.write("NDBuffer(")

        @parameter
        fn serialize[T: Writable](val: T):
            writer.write(val)

        var shape = List[Int, hint_trivial_type=True]()
        for i in range(rank):
            shape.append(self.dynamic_shape[i])

        _serialize[serialize_fn=serialize, serialize_end_line=False](
            self.data, shape
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
        Scalar[type], address_space=address_space, mut=mut, origin=origin, **_
    ]:
        """Computes the NDBuffer's offset using the index positions provided.

        Args:
            idx: The index positions.

        Returns:
            The offset into the NDBuffer given the indices.
        """
        constrained[rank <= _MAX_RANK]()
        return self.data.offset(_compute_ndbuffer_offset(self, idx))

    @always_inline
    fn _offset(
        self, idx: IndexList[rank, **_]
    ) -> UnsafePointer[
        Scalar[type], address_space=address_space, mut=mut, origin=origin, **_
    ]:
        constrained[rank <= _MAX_RANK]()
        return self.data.offset(_compute_ndbuffer_offset(self, idx))

    @always_inline
    fn _offset(
        self, idx: StaticTuple[Int, rank]
    ) -> UnsafePointer[
        Scalar[type], address_space=address_space, mut=mut, origin=origin, **_
    ]:
        """Computes the NDBuffer's offset using the index positions provided.

        Args:
            idx: The index positions.

        Returns:
            The offset into the NDBuffer given the indices.
        """
        constrained[rank <= _MAX_RANK]()
        return self.data.offset(_compute_ndbuffer_offset(self, idx))

    @always_inline
    fn __getitem__(self, *idx: Int) -> Scalar[type]:
        """Gets an element from the buffer from the specified index.

        Args:
            idx: Index of the element to retrieve.

        Returns:
            The value of the element.
        """
        return self.load[width=1](idx)

    @always_inline
    fn __getitem__(self, idx: IndexList[rank, **_]) -> Scalar[type]:
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
    ](self, tile_coords: IndexList[rank, **_]) -> NDBuffer[
        type,
        rank,
        origin,
        DimList(tile_sizes),
        address_space=address_space,
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

        @parameter
        fn num_tile_sizes() -> Int:
            return __mlir_op.`pop.variadic.size`(tile_sizes)

        constrained[
            num_tile_sizes() == rank,
            "The tile should have the same rank as the buffer",
        ]()

        constrained[
            DimList(tile_sizes).all_known[rank](),
            "Static tile sizes are only supported",
        ]()

        var offset = 0
        var shape = IndexList[rank]()

        @parameter
        for i in range(rank):
            alias tile_size_i = tile_sizes[i].get()
            shape[i] = tile_size_i
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
            type,
            rank,
            origin,
            DimList(tile_sizes),
            address_space=address_space,
        ](
            self.data.offset(offset),
            dynamic_shape=shape,
            dynamic_stride=self.dynamic_stride,
        )
        return tile

    @always_inline("nodebug")
    fn load[
        *, width: Int = 1, alignment: Int = Self._default_alignment[width]()
    ](self, *idx: Int) -> SIMD[type, width]:
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
    ](self, idx: VariadicList[Int]) -> SIMD[type, width]:
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
    ](self, idx: IndexList) -> SIMD[type, width]:
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
        constrained[idx.size == rank, "invalid index size"]()
        return self.load[width=width, alignment=alignment](
            rebind[IndexList[rank, element_type = idx.element_type]](
                idx
            ).as_tuple()
        )

    @always_inline("nodebug")
    fn load[
        *, width: Int = 1, alignment: Int = Self._default_alignment[width]()
    ](self, idx: StaticTuple[Int, rank]) -> SIMD[type, width]:
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
            type,
            rank,
            _,
            shape=shape,
            strides=strides,
            alignment=alignment,
            address_space=address_space,
            exclusive=exclusive,
        ],
        idx: IndexList[rank, **_],
        val: Scalar[type],
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
            type,
            rank,
            _,
            shape=shape,
            strides=strides,
            alignment=alignment,
            address_space=address_space,
            exclusive=exclusive,
        ],
        *idx: Int,
        val: Scalar[type],
    ):
        """Stores a single value into the buffer at the specified index.

        Args:
            idx: Index of the element to retrieve.
            val: The value to store.
        """
        self.store[width=1](IndexList[rank](idx), val)

    @always_inline("nodebug")
    fn store[
        _alignment: Int, //,
        *,
        width: Int = 1,
        alignment: Int = Self._default_alignment[width](),
    ](
        self: NDBuffer[
            mut=True,
            type,
            rank,
            _,
            shape=shape,
            strides=strides,
            alignment=_alignment,
            address_space=address_space,
            exclusive=exclusive,
        ],
        idx: IndexList[rank, **_],
        val: SIMD[type, width],
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
        _alignment: Int, //,
        *,
        width: Int = 1,
        alignment: Int = Self._default_alignment[width](),
    ](
        self: NDBuffer[
            mut=True,
            type,
            rank,
            _,
            shape=shape,
            strides=strides,
            alignment=_alignment,
            address_space=address_space,
            exclusive=exclusive,
        ],
        idx: StaticTuple[Int, rank],
        val: SIMD[type, width],
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
        alias static_dim_value = shape.at[index]()

        @parameter
        if static_dim_value.has_value():
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
        alias static_stride_value = strides.at[index]()

        @parameter
        if static_stride_value.has_value():
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

        constrained[rank > 0, "rank must be positive"]()
        return self.stride[rank - 1]() == 1

    @always_inline
    fn flatten(
        self,
        out result: NDBuffer[
            type, 1, origin, shape.product(), address_space=address_space
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
        return __type_of(result)(self.data, self.size())

    @always_inline
    fn make_dims_unknown(
        self,
        out result: NDBuffer[
            type, rank, address_space=address_space, origin=origin
        ],
    ):
        """Rebinds the NDBuffer to one with unknown shape.

        Returns:
            The rebound NDBuffer with unknown shape.
        """
        return rebind[__type_of(result)](self)

    @always_inline
    fn bytecount(self) -> Int:
        """Returns the size of the NDBuffer in bytes.

        Returns:
            The size of the NDBuffer in bytes.
        """
        return self.size() * sizeof[type]()

    @always_inline
    fn zero(self: NDBuffer[mut=True, *_, **_]):
        """Sets all bytes of the NDBuffer to 0.

        Constraints:
            The buffer must be contiguous.
        """
        debug_assert(
            self.is_contiguous(), "Function requires contiguous buffer."
        )

        @parameter
        if shape.all_known[rank]():
            alias count = Int(shape.product())
            memset_zero[count=count](self.data)
        else:
            memset_zero(self.data, len(self))

    @always_inline
    fn _simd_fill[
        simd_width: Int
    ](
        self: NDBuffer[
            mut=True,
            type,
            rank,
            _,
            shape=shape,
            strides=strides,
            alignment=alignment,
            address_space=address_space,
            exclusive=exclusive,
        ],
        val: Scalar[type],
    ):
        """Assigns val to all elements in chunks of size simd_width.

        Parameters:
            simd_width: The simd_width of the fill.

        Args:
            val: The value to store.
        """

        @parameter
        if rank > 1:
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
                self.store[width=simd_width](i, val)
            for i in range(vec_end, len(self)):
                self.store(i, val)

    @always_inline
    fn tofile(self, path: Path) raises:
        """Write values to a file.

        Args:
            path: Path to the output file.
        """
        with open(path.__str__(), "w") as f:
            var ptr = self.data.bitcast[UInt8]()
            f._write(ptr, self.bytecount())

    @always_inline
    fn fill(
        self: NDBuffer[
            mut=True,
            type,
            rank,
            _,
            shape=shape,
            strides=strides,
            alignment=alignment,
            address_space=address_space,
            exclusive=exclusive,
        ],
        val: Scalar[type],
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
        self._simd_fill[simdwidthof[type]()](val)

    @staticmethod
    @always_inline("nodebug")
    fn stack_allocation[*, alignment: Int = alignof[type]()]() -> Self:
        """Constructs an NDBuffer instance backed by stack allocated memory space.

        Parameters:
            alignment: Address alignment requirement for the allocation.

        Returns:
            Constructed NDBuffer with the allocated space.
        """
        constrained[
            shape.all_known[rank](),
            (
                "the shape of the NDBuffer must be known to allow for stack"
                " allocation"
            ),
        ]()
        var data_pointer = stack_allocation[
            shape.product[rank]().get(),
            type,
            alignment=alignment,
            address_space=address_space,
        ]()
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
    fn prefetch[params: PrefetchOptions](self, indices: IndexList[rank]):
        """Prefetches the data at the given index.

        Parameters:
            params: The prefetch configuration.

        Args:
            indices: The N-D index of the prefetched location.
        """
        prefetch[params](self._offset(indices))


@always_inline
fn partial_simd_load[
    type: DType, //, width: Int
](
    storage: UnsafePointer[Scalar[type], **_],
    lbound: Int,
    rbound: Int,
    pad_value: Scalar[type],
) -> SIMD[type, width]:
    """Loads a vector with dynamic bound.

    Out of bound data will be filled with pad value. Data is valid if
    lbound <= idx < rbound for idx from 0 to (simd_width-1). For example:

        addr 0  1  2  3
        data x 42 43  x

        partial_simd_load[4](addr0, 1, 3) #gives [0 42 43 0]

    Parameters:
        type: The DType of storage.
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
    var effective_lbound = max(0, lbound)
    var effective_rbound = min(width, rbound)
    var incr = iota[DType.int32, width]()
    var mask = (incr >= effective_lbound) & (incr < effective_rbound)

    return masked_load[width](storage, mask, pad_value)


@always_inline
fn partial_simd_store[
    type: DType, //, width: Int
](
    storage: UnsafePointer[Scalar[type], **_],
    lbound: Int,
    rbound: Int,
    data: SIMD[type, width],
):
    """Stores a vector with dynamic bound.

    Out of bound data will ignored. Data is valid if lbound <= idx < rbound for
    idx from 0 to (simd_width-1).

    e.g.
        addr 0 1 2  3
        data 0 0 0  0

        partial_simd_load[4](addr0, 1, 3, [-1, 42, 43, -1]) #gives [0 42 43 0]

    Parameters:
        type: The DType of storage.
        width: The system simd vector size.

    Args:
        storage: Pointer to the address to perform load.
        lbound: Lower bound of valid index within simd (inclusive).
        rbound: Upper bound of valid index within simd (non-inclusive).
        data: The vector value to store.
    """
    # Create a mask based on input bounds.
    var effective_lbound = max(0, lbound)
    var effective_rbound = min(width, rbound)
    var incr = iota[DType.int32, width]()
    var mask = (incr >= effective_lbound) & (incr < effective_rbound)

    # Rebind for the inconsistency between (1) `ptr: UnsafePointer` deduces
    # address_space as ptr1 and (2) `UnsafePointer[Scalar[type]]` sets address_space to
    # generic by default. The `masked_store` takes (2) to enforce the same type
    # between data and storage. #28834.
    return masked_store(
        data, rebind[UnsafePointer[Scalar[type]]](storage), mask
    )


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

    @parameter
    for i in range(start_dim, end_dim):
        product *= x.dim[i]()

    return product
