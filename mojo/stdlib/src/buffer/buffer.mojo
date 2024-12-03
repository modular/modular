# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements the Buffer class.

You can import these APIs from the `memory` package. For example:

```mojo
from buffer import Buffer
```
"""

from math import align_down, fma, iota
from pathlib import Path
from sys.info import alignof, is_nvidia_gpu, simdwidthof, sizeof
from sys.intrinsics import PrefetchOptions, masked_load, masked_store, prefetch

from buffer.dimlist import Dim, DimList, _make_tuple
from memory import UnsafePointer, memset_zero, stack_allocation
from memory.pointer import AddressSpace, _GPUAddressSpace

from utils._serialize import _serialize
from utils.index import IndexList
from utils.index import product as tuple_product
from utils.loop import unroll
from utils.static_tuple import StaticTuple

alias _MAX_RANK = 8
"""The maximum tensor rank for any tensor shape.
This value must match kMaxRank in Support/include/Support/ML/TensorShape.h
"""

# ===-----------------------------------------------------------------------===#
# Buffer
# ===-----------------------------------------------------------------------===#


# This type is "async safe" (see _async_parallelize).
@value
@register_passable("trivial")
struct Buffer[
    type: DType,
    /,
    size: Dim = Dim(),
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
    origin: Origin[True].type = MutableAnyOrigin,
](Sized):
    """Defines a Buffer which can be parametrized on a static size and Dtype.

    The Buffer does not own its underlying pointer.

    Parameters:
      type: The element type of the Buffer.
      size: The static size (if known) of the Buffer.
      address_space: The address space of the Buffer.
      origin: The origin of the memory being addressed.
    """

    var data: UnsafePointer[
        Scalar[type], address_space=address_space, origin=origin
    ]
    """The underlying data pointer of the data."""
    var dynamic_size: Int
    """The dynamic size of the buffer."""
    var dtype: DType
    """The dynamic data type of the buffer."""

    @staticmethod
    fn _default_alignment[width: Int = 1]() -> Int:
        return alignof[SIMD[type, width]]() if is_nvidia_gpu() else 1

    @always_inline
    fn __init__(out self):
        """Default initializer for Buffer. By default the fields are all
        initialized to 0.
        """

        self.data = UnsafePointer[Scalar[type], address_space=address_space]()
        self.dynamic_size = 0
        self.dtype = type

    @always_inline
    @implicit
    fn __init__(
        out self, ptr: UnsafePointer[Scalar[type], address_space=address_space]
    ):
        """Constructs a Buffer with statically known size and type.

        Constraints:
            The size is known.

        Args:
            ptr: Pointer to the data.
        """
        # Construct a Buffer type with statically known size
        constrained[size.has_value(), "must have known size"]()
        self.data = ptr
        self.dynamic_size = size.get()
        self.dtype = type

    @always_inline
    fn __init__(
        mut self,
        ptr: UnsafePointer[Scalar[type], address_space=address_space],
        in_size: Int,
    ):
        """Constructs a Buffer with statically known type.

        Constraints:
            The size is unknown.

        Args:
            ptr: Pointer to the data.
            in_size: Dynamic size of the buffer.
        """

        @parameter
        if size:
            debug_assert(
                in_size == size.get(),
                "if static size is known, static size must equal dynamic size",
            )
        self.data = ptr
        self.dynamic_size = in_size
        self.dtype = type

    @always_inline
    fn __len__(self) -> Int:
        """Gets the size if it is a known constant, otherwise it gets the
        dynamic_size.

        This method is used by `Buffer.__len__` to get the size of the buffer.
        If the Buffer size is a known constant, then the size is returned.
        Otherwise, the dynamic_size is returned.

        Returns:
            The size if static otherwise dynamic_size.
        """

        @parameter
        if not size:
            return self.dynamic_size

        return size.get()

    @always_inline
    fn __getitem__(self, idx: Int) -> Scalar[type]:
        """Loads a single element (SIMD of size 1) from the buffer at the
        specified index.

        Args:
            idx: The index into the Buffer.

        Returns:
            The value at the `idx` position.
        """
        return self.load[width=1](idx)

    @always_inline
    fn load[
        *, width: Int = 1, alignment: Int = Self._default_alignment[width]()
    ](self, idx: Int) -> SIMD[type, width]:
        """Loads a simd value from the buffer at the specified index.

        Parameters:
            width: The simd_width of the load.
            alignment: The alignment value.

        Args:
            idx: The index into the Buffer.

        Returns:
            The simd value starting at the `idx` position and ending at
            `idx+width`.
        """
        return self.data.load[width=width, alignment=alignment](idx)

    @always_inline
    fn __setitem__(
        self,
        idx: Int,
        val: __mlir_type[`!pop.scalar<`, type.value, `>`],
    ):
        """Stores a single value into the buffer at the specified index.

        Args:
            idx: The index into the Buffer.
            val: The value to store.
        """
        self.store[width=1](idx, Scalar[type](val))

    @always_inline
    fn __setitem__(self, idx: Int, val: Scalar[type]):
        """Stores a single value into the buffer at the specified index.

        Args:
            idx: The index into the Buffer.
            val: The value to store.
        """
        self.store[width=1](idx, val)

    @always_inline
    fn store[
        *, width: Int = 1, alignment: Int = Self._default_alignment[width]()
    ](self, idx: Int, val: SIMD[type, width]):
        """Stores a simd value into the buffer at the specified index.

        Parameters:
            width: The width of the simd vector.
            alignment: The alignment value.

        Args:
            idx: The index into the Buffer.
            val: The value to store.
        """
        self.data.store[alignment=alignment](idx, val)

    @always_inline
    fn prefetch[params: PrefetchOptions](self, idx: Int):
        """Prefetches the data at the given index.

        Parameters:
            params: The prefetch configuration.

        Args:
            idx: The index of the prefetched location.
        """
        prefetch[params](self.data.offset(idx))

    @always_inline
    fn bytecount(self) -> Int:
        """Returns the size of the Buffer in bytes.

        Returns:
            The size of the Buffer in bytes.
        """
        return len(self) * sizeof[type]()

    @always_inline
    fn zero(self):
        """Sets all bytes of the Buffer to 0."""
        memset_zero(self.data, len(self))

    @always_inline
    fn _simd_fill[simd_width: Int](self, val: Scalar[type]):
        """Assigns val to all elements in chunks of size simd_width.

        Parameters:
            simd_width: The simd_width of the fill.

        Args:
            val: The value to store.
        """
        if val == 0:
            self.zero()
            return

        var vec_end = align_down(len(self), simd_width)
        for i in range(0, vec_end, simd_width):
            self.store[width=simd_width](i, val)
        for i in range(vec_end, len(self)):
            self.store(i, val)

    @always_inline
    fn fill(self, val: Scalar[type]):
        """Assigns val to all elements in the Buffer.

        The fill is performed in chunks of size N, where N is the native SIMD
        width of type on the system.

        Args:
            val: The value to store.
        """
        self._simd_fill[simdwidthof[type]()](val)

    @always_inline
    fn tofile(self, path: Path) raises:
        """Write values to a file.

        Args:
            path: Path to the output file.
        """
        with open(path.__str__(), "w") as f:
            var ptr = self.data.bitcast[Scalar[DType.uint8]]()
            f._write(ptr, self.bytecount())

    @staticmethod
    @always_inline("nodebug")
    fn stack_allocation[*, alignment: Int = alignof[type]()]() -> Self:
        """Constructs a buffer instance backed by stack allocated memory space.

        Parameters:
            alignment: Address alignment requirement for the allocation.

        Returns:
            Constructed buffer with the allocated space.
        """
        constrained[size.has_value(), "must have known size"]()
        var data_pointer = stack_allocation[
            size.get(), type, alignment=alignment, address_space=address_space
        ]()
        return Self(data_pointer)


# ===-----------------------------------------------------------------------===#
# NDBuffer Utilities
# ===-----------------------------------------------------------------------===#


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
    if is_nvidia_gpu() and buf.address_space in (
        _GPUAddressSpace.SHARED,
        _GPUAddressSpace.LOCAL,
        _GPUAddressSpace.CONSTANT,
        _GPUAddressSpace.PARAM,
    ):
        var result: Int32 = 0

        @parameter
        for i in range(buf.rank):
            result = fma(Int32(buf.stride[i]()), Int32(index[i]), result)

        return int(result)

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
    if is_nvidia_gpu() and buf.address_space == _GPUAddressSpace.SHARED:
        var result: Int32 = 0

        @parameter
        for i in range(rank):
            result = fma(Int32(buf.stride[i]()), Int32(index[i]), result)

        return int(result)

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
@value
@register_passable("trivial")
struct NDBuffer[
    type: DType,
    rank: Int,
    shape: DimList = DimList.create_unknown[rank](),
    strides: DimList = DimList.create_unknown[rank](),
    *,
    alignment: Int = 1,
    address_space: AddressSpace = AddressSpace.GENERIC,
    exclusive: Bool = True,
](Sized, Stringable, Writable):
    """An N-dimensional Buffer.

    NDBuffer can be parametrized on rank, static dimensions and Dtype. It does
    not own its underlying pointer.

    Parameters:
        type: The element type of the buffer.
        rank: The rank of the buffer.
        shape: The static size (if known) of the buffer.
        strides: The strides (if known) of the buffer.
        alignment: The preferred address alignment of the buffer.
        address_space: The address space of the buffer.
        exclusive: The underlying memory allocation of the tensor is known
            only to be accessible through this pointer.
    """

    var data: UnsafePointer[Scalar[type], address_space=address_space]
    """The underlying data for the buffer. The pointer is not owned by the
    NDBuffer."""
    var dynamic_shape: IndexList[rank, unsigned=True]
    """The dynamic value of the shape."""
    var dynamic_stride: IndexList[rank, unsigned=True]
    """The dynamic stride of the buffer."""

    @staticmethod
    fn _default_alignment[width: Int = 1]() -> Int:
        return alignof[SIMD[type, width]]() if is_nvidia_gpu() else 1

    @always_inline
    fn __init__(out self):
        """Default initializer for NDBuffer. By default the fields are all
        initialized to 0.
        """

        self.data = UnsafePointer[Scalar[type], address_space=address_space]()
        self.dynamic_shape = __type_of(self.dynamic_shape)()
        self.dynamic_stride = __type_of(self.dynamic_stride)()

    @always_inline
    @implicit
    fn __init__(
        mut self,
        ptr: UnsafePointer[Scalar[type], address_space=address_space, **_],
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
        self.dynamic_shape = _make_tuple[rank, unsigned=True](shape)
        self.dynamic_stride = _compute_ndbuffer_stride[rank](self.dynamic_shape)

    @always_inline
    fn __init__(
        mut self,
        ptr: UnsafePointer[
            __mlir_type[`!pop.scalar<`, type.value, `>`],
            address_space=address_space,
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
            dynamic_shape.cast[
                element_bitwidth = __type_of(
                    self.dynamic_shape
                ).element_bitwidth,
                unsigned = __type_of(self.dynamic_shape).unsigned,
            ]()
        )
        self.dynamic_stride = _compute_ndbuffer_stride[rank](self.dynamic_shape)

    @always_inline
    fn __init__(
        mut self,
        ptr: UnsafePointer[Scalar[type], address_space=address_space],
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
            dynamic_shape.cast[
                element_bitwidth = __type_of(
                    self.dynamic_shape
                ).element_bitwidth,
                unsigned = __type_of(self.dynamic_shape).unsigned,
            ]()
        )
        self.dynamic_stride = _compute_ndbuffer_stride[rank](self.dynamic_shape)

    @always_inline
    fn __init__(
        mut self,
        ptr: UnsafePointer[Scalar[type], address_space=address_space],
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
        self.__init__(ptr, _make_tuple[rank](dynamic_shape))

    @always_inline
    fn __init__(
        mut self,
        ptr: UnsafePointer[Scalar[type], address_space=address_space],
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
            dynamic_shape.cast[
                element_bitwidth = __type_of(
                    self.dynamic_shape
                ).element_bitwidth,
                unsigned = __type_of(self.dynamic_shape).unsigned,
            ]()
        )
        self.dynamic_stride = rebind[__type_of(self.dynamic_stride)](
            dynamic_stride.cast[
                element_bitwidth = __type_of(
                    self.dynamic_stride
                ).element_bitwidth,
                unsigned = __type_of(self.dynamic_stride).unsigned,
            ]()
        )

    @always_inline
    fn __init__(
        mut self,
        ptr: UnsafePointer[Scalar[type], address_space=address_space],
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
        self.__init__(
            ptr=ptr,
            dynamic_shape=_make_tuple[rank](dynamic_shape),
            dynamic_stride=dynamic_stride,
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
    ) -> UnsafePointer[Scalar[type], address_space=address_space, **_]:
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
    ) -> UnsafePointer[Scalar[type], address_space=address_space, **_]:
        constrained[rank <= _MAX_RANK]()
        return self.data.offset(_compute_ndbuffer_offset(self, idx))

    @always_inline
    fn _offset(
        self, idx: StaticTuple[Int, rank]
    ) -> UnsafePointer[Scalar[type], address_space=address_space, **_]:
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
        type, rank, DimList(tile_sizes), address_space=address_space
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
            "The tile sould have the same rank as the buffer",
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
            type, rank, DimList(tile_sizes), address_space=address_space
        ](
            self.data.offset(offset),
            dynamic_shape=shape,
            dynamic_stride=self.dynamic_stride,
        )
        return tile

    @always_inline
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

    @always_inline
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

    @always_inline
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
            rebind[
                IndexList[
                    rank,
                    element_bitwidth = idx.element_bitwidth,
                    unsigned = idx.unsigned,
                ]
            ](idx).as_tuple()
        )

    @always_inline
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
    fn __setitem__(self, idx: IndexList[rank, **_], val: Scalar[type]):
        """Stores a single value into the buffer at the specified index.

        Args:
            idx: The index into the Buffer.
            val: The value to store.
        """
        self.store[width=1](idx, val)

    @always_inline
    fn __setitem__(self, *idx: Int, val: Scalar[type]):
        """Stores a single value into the buffer at the specified index.

        Args:
            idx: Index of the element to retrieve.
            val: The value to store.
        """
        self.store[width=1](IndexList[rank](idx), val)

    @always_inline
    fn store[
        *, width: Int = 1, alignment: Int = Self._default_alignment[width]()
    ](self, idx: IndexList[rank, **_], val: SIMD[type, width]):
        """Stores a simd value into the buffer at the specified index.

        Constraints:
            The buffer must be contiguous or width must be 1.

        Parameters:
            width: The width of the simd vector.
            alignment: The alignment value.

        Args:
            idx: The index into the Buffer.
            val: The value to store.
        """
        self.store[width=width, alignment=alignment](idx.as_tuple(), val)

    @always_inline
    fn store[
        *, width: Int = 1, alignment: Int = Self._default_alignment[width]()
    ](self, idx: StaticTuple[Int, rank], val: SIMD[type, width]):
        """Stores a simd value into the buffer at the specified index.

        Constraints:
            The buffer must be contiguous or width must be 1.

        Parameters:
            width: The width of the simd vector.
            alignment: The alignment value.

        Args:
            idx: The index into the Buffer.
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
    ) -> Buffer[type, shape.product(), address_space=address_space] as result:
        """Constructs a flattened Buffer counterpart for this NDBuffer.

        Constraints:
            The buffer must be contiguous.

        Returns:
            Constructed Buffer object.
        """
        debug_assert(
            self.is_contiguous(), "Function requires contiguous buffer."
        )
        return __type_of(result)(self.data, self.size())

    @always_inline
    fn make_dims_unknown(
        self,
    ) -> NDBuffer[type, rank, address_space=address_space] as result:
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
    fn zero(self):
        """Sets all bytes of the NDBuffer to 0.

        Constraints:
            The buffer must be contiguous.
        """
        debug_assert(
            self.is_contiguous(), "Function requires contiguous buffer."
        )

        @parameter
        if shape.all_known[rank]():
            alias count = int(shape.product())
            memset_zero[count=count](self.data)
        else:
            memset_zero(self.data, len(self))

    @always_inline
    fn _simd_fill[simd_width: Int](self, val: Scalar[type]):
        """Assigns val to all elements in chunks of size simd_width.

        Parameters:
            simd_width: The simd_width of the fill.

        Args:
            val: The value to store.
        """
        if val == 0:
            self.zero()
            return
        self.flatten()._simd_fill[simd_width](val)

    @always_inline
    fn tofile(self, path: Path) raises:
        """Write values to a file.

        Args:
            path: Path to the output file.
        """
        with open(path.__str__(), "w") as f:
            var ptr = self.data.bitcast[Scalar[DType.uint8]]()
            f._write(ptr, self.bytecount())

    @always_inline
    fn fill(self, val: Scalar[type]):
        """Assigns val to all elements in the Buffer.

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

    @always_inline("nodebug")
    fn __imul__(mut self, rhs: Float32):
        """In-place multiplies a scalar.

        Args:
            rhs: The RHS of the mul operation.
        """
        constrained[rank == 2]()
        constrained[shape.all_known[2]()]()
        constrained[type is DType.float32]()

        alias TM = shape.at[0]().get()
        alias TN = shape.at[1]().get()

        alias simd_size = simdwidthof[DType.float32]()

        @parameter
        for i in range(TM):

            @parameter
            for j in range(0, TN, simd_size):
                var idx = i * TN + j
                var vec = self.data.load[width=simd_size](i * TN + j)
                self.data.store(idx, vec * rhs.cast[type]())

    @always_inline("nodebug")
    fn __imul__(mut self, rhs: NDBuffer):
        """In-place multiplies a NDBuffer.

        Args:
            rhs: The RHS of the mul operation.
        """
        constrained[rank == 2]()
        constrained[shape.all_known[2]()]()
        constrained[type is DType.float32]()

        alias TM = shape.at[0]().get()
        alias TN = shape.at[1]().get()

        alias simd_size = simdwidthof[type]()

        @parameter
        for i in range(TM):

            @parameter
            for j in range(0, TN, simd_size):
                var idx = i * TN + j
                var vec = self.data.load[width=simd_size](idx)
                var rhs_vec = SIMD[type, simd_size](
                    rhs.data.load(i).cast[type]()
                )
                self.data.store(idx, vec * rhs_vec)

    @always_inline("nodebug")
    fn __itruediv__(mut self, rhs: NDBuffer):
        """In-place divides a NDBuffer.

        Args:
            rhs: The RHS of the div operation.
        """
        constrained[rank == 2]()
        constrained[shape.all_known[2]()]()
        constrained[type is DType.float32]()

        alias TM = shape.at[0]().get()
        alias TN = shape.at[1]().get()

        alias simd_size = simdwidthof[type]()

        @parameter
        for i in range(TM):

            @parameter
            for j in range(0, TN, simd_size):
                var idx = i * TN + j
                var vec = self.data.load[width=simd_size](idx)
                var rhs_vec = SIMD[type, simd_size](
                    rhs.data.load(i).cast[type]()
                )
                self.data.store(idx, vec / rhs_vec)

    @always_inline("nodebug")
    fn __mul__(self, rhs: Self) -> Self:
        """Multiplies a NDBuffer.

        Args:
            rhs: The RHS of the mul operation.

        Returns:
            The division result.
        """
        constrained[rank == 1]()
        constrained[shape.all_known[1]()]()
        constrained[type is DType.float32]()

        alias m = shape.at[0]().get()

        alias simd_size = simdwidthof[type]()

        var res = Self.stack_allocation()

        @parameter
        for i in range(m):
            res[i] = self.data.load(i) * rhs.data.load(i)

        return res

    @always_inline("nodebug")
    fn __add__(self, rhs: NDBuffer) -> Self:
        """Adds a NDBuffer.

        Args:
            rhs: The RHS of the add operation.

        Returns:
            The addition result.
        """
        constrained[rank == 1]()
        constrained[shape.all_known[1]()]()
        constrained[type is DType.float32]()

        alias m = shape.at[0]().get()

        alias simd_size = simdwidthof[type]()

        var res = Self.stack_allocation()

        @parameter
        for i in range(m):
            res[i] = self[i] + rhs[i].cast[type]()

        return res

    @always_inline("nodebug")
    fn __sub__(self, rhs: Self) -> Self:
        """Subtracts a scalar.

        Args:
            rhs: The RHS of the sub operation.

        Returns:
            The subtraction result.
        """
        constrained[rank == 1]()
        constrained[shape.all_known[1]()]()
        constrained[type is DType.float32]()

        alias simd_size = simdwidthof[type]()

        alias m = shape.at[0]().get()

        var res = Self.stack_allocation()

        @parameter
        for i in range(m):
            res.data.store(
                i,
                self.data.load(i) - rhs.data.load(i),
            )

        return res

    @always_inline("nodebug")
    fn __sub__[
        rhs_shape: DimList
    ](self, rhs: NDBuffer[type, 1, rhs_shape]) -> Self:
        """Subtracts a NDBuffer.

        Parameters:
            rhs_shape: Shape of RHS.

        Args:
            rhs: The RHS of the sub operation.

        Returns:
            The subtraction result.
        """
        constrained[rank == 2]()
        constrained[shape.all_known[2]()]()
        constrained[type is DType.float32]()

        alias m = shape.at[0]().get()
        alias n = shape.at[1]().get()
        alias simd_size = simdwidthof[type]()

        var res = Self.stack_allocation()

        @parameter
        for i in range(m):

            @parameter
            for j in range(0, n, simd_size):
                var idx = i * n + j
                res.data.store(
                    idx,
                    self.data.load[width=simd_size](idx) - rhs.data.load(i),
                )

        return res


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


# ===-----------------------------------------------------------------------===#
# DynamicRankBuffer
# ===-----------------------------------------------------------------------===#


# This struct must match DynamicRankBuffer in Kernels/lib/MojoKernels/Kernels.cpp
@register_passable("trivial")
struct DynamicRankBuffer:
    """DynamicRankBuffer represents a buffer with unknown rank, shapes and dtype.

    It is not as efficient as the statically ranked buffer, but is useful when
    interacting with external functions. In particular the shape is represented
    as a fixed (ie _MAX_RANK) array of dimensions to simplify the ABI."""

    var data: UnsafePointer[NoneType]
    """The pointer to the buffer."""
    var rank: Int
    """The buffer rank. Has a max value of `_MAX_RANK`."""
    var shape: IndexList[_MAX_RANK]
    """The dynamic shape of the buffer."""
    var type: DType
    """The dynamic dtype of the buffer."""

    @always_inline
    fn __init__(
        mut self,
        data: UnsafePointer[NoneType],
        rank: Int,
        shape: IndexList[_MAX_RANK],
        type: DType,
    ):
        """Construct DynamicRankBuffer.

        Args:
            data: Pointer to the underlying data.
            rank: Rank of the buffer.
            shape: Shapes of the buffer.
            type: `dtype` of the buffer.
        """
        self.data = data
        self.rank = rank
        self.shape = shape
        self.type = type.value

    @always_inline
    fn to_buffer[type: DType](self) -> Buffer[type]:
        """Casts DynamicRankBuffer to Buffer.

        Parameters:
            type: `dtype` of the buffer.

        Returns:
            Constructed Buffer.
        """
        return Buffer[type](
            self.data.bitcast[Scalar[type]](),
            tuple_product(self.shape, self.rank),
        )

    @always_inline
    fn to_ndbuffer[type: DType, rank: Int](self) -> NDBuffer[type, rank]:
        """Casts the buffer to NDBuffer.

        Constraints:
            Rank of DynamicRankBuffer must equal rank of NDBuffer.

        Parameters:
            type: `dtype` of the buffer.
            rank: Rank of the buffer.

        Returns:
            Constructed NDBuffer.
        """
        debug_assert(
            self.type == type,
            "type of DynamicRankBuffer must equal type of NDBuffer",
        )
        debug_assert(
            self.rank == rank,
            "rank of DynamicRankBuffer must equal rank of NDBuffer",
        )
        return NDBuffer[type, rank](
            self.data.bitcast[Scalar[type]](),
            self._shape_to_static_tuple[rank](),
        )

    @always_inline
    fn to_ndbuffer[
        type: DType, rank: Int
    ](self, stride: IndexList[rank]) -> NDBuffer[type, rank]:
        """Casts the buffer to NDBuffer.

        Constraints:
            Rank of DynamicRankBuffer must equal rank of NDBuffer.

        Parameters:
            type: `dtype` of the buffer.
            rank: Rank of the buffer.

        Args:
            stride: Strides of the buffer.

        Returns:
            Constructed NDBuffer.
        """
        debug_assert(
            self.rank == rank,
            "rank of DynamicRankBuffer must equal rank of NDBuffer",
        )
        return NDBuffer[type, rank](
            self.data.bitcast[Scalar[type]](),
            self._shape_to_static_tuple[rank](),
            stride,
        )

    @always_inline
    fn rank_dispatch[func: fn[rank: Int] () capturing [_] -> None](self):
        """Dispatches the function call based on buffer rank.

        Constraints:
            Rank must be positive and less or equal to 8.

        Parameters:
            func: Function to dispatch. The function should be parametrized on
              an index parameter, which will be used for rank when the function
              will be called.
        """
        debug_assert(
            self.rank > 0 and self.rank <= _MAX_RANK,
            "rank must be positive and less or equal to 8",
        )

        if self.rank == 1:
            func[1]()
            return

        if self.rank == 2:
            func[2]()
            return

        if self.rank == 3:
            func[3]()
            return

        if self.rank == 4:
            func[4]()
            return

        if self.rank == 5:
            func[5]()
            return

        if self.rank == 6:
            func[6]()
            return

        if self.rank == 7:
            func[7]()
            return

        if self.rank == 8:
            func[8]()
            return

    @always_inline
    fn num_elements(self) -> Int:
        """Gets number of elements in the buffer.

        Returns:
            The number of elements in the buffer.
        """
        return tuple_product(self.shape, self.rank)

    @always_inline
    fn get_shape[rank: Int](self) -> IndexList[rank]:
        """Gets a static tuple representing the buffer shape.

        Parameters:
            rank: Rank of the buffer.

        Returns:
            A static tuple of size 'Rank' filled with buffer shapes.
        """
        return self._shape_to_static_tuple[rank]()

    @always_inline
    fn dim(self, idx: Int) -> Int:
        """Gets given dimension.

        Args:
            idx: The dimension index.

        Returns:
            The buffer size on the given dimension.
        """
        debug_assert(idx < self.rank, "dimension index is out of bounds")
        return self.shape[idx]

    @always_inline
    fn _shape_to_static_tuple[rank: Int](self) -> IndexList[rank]:
        var result = IndexList[rank]()

        @parameter
        for idx in range(rank):
            result[idx] = self.dim(idx)
        return result


fn _collapse_batch_dim(input: DynamicRankBuffer) -> DynamicRankBuffer:
    """Collapse the batch so that the tensor has rank of at most 3. The output
    shape will therefore be
    [input.shape[0] * ... * input.shape[input.rank-2],
     input.shape[rank-2],
     input.shape[rank-1],
     0, ..., 0
    ]

    Args:
      input: The input DynamicRankBuffer.

    Returns:
      The DynamicRankBuffer where the batch dim has been flattened.
    """
    var batch_size = 1
    for i in range(0, input.rank - 2):
        batch_size *= input.shape[i]
    var collapsed_shape = IndexList[_MAX_RANK]()
    collapsed_shape[0] = batch_size
    collapsed_shape[1] = input.shape[input.rank - 2]
    collapsed_shape[2] = input.shape[input.rank - 1]
    return DynamicRankBuffer(input.data, 3, collapsed_shape, input.type)


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
