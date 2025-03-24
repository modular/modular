# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides element-based access to memory using layout-driven vectorization.

This module implements efficient memory access patterns for multi-dimensional data
using the layout system. It provides abstractions for loading and storing data with
specific memory layouts, enabling vectorized operations that respect the underlying
memory organization.

Key components:
- `Element`: A wrapper around SIMD types that provides layout-driven vectorized
  operations
- `MemoryElement`: Represents data in memory organized according to a specific layout

These components enable efficient tensor operations by ensuring memory accesses
follow optimal patterns defined by the layout system.
"""

from sys import alignof, bitwidthof

from memory import AddressSpace, UnsafePointer

from utils import IndexList

from . import Layout, RuntimeLayout
from .int_tuple import UNKNOWN_VALUE, _get_index_type


@always_inline
fn _get_offset[
    i: Int
](runtime_layout: RuntimeLayout) -> Scalar[runtime_layout.linear_idx_type]:
    """Returns the offset for a single index into the runtime layout.

    Parameters:
        i: The index to get the offset for.

    Args:
        runtime_layout: The runtime layout to get the offset from.

    Returns:
        The offset value for the given index.
    """

    @parameter
    if runtime_layout.layout.all_dims_known():
        alias offset = runtime_layout.layout(i)
        return offset
    else:
        return runtime_layout(i)


@always_inline
fn _get_offset[
    i: Int, j: Int
](runtime_layout: RuntimeLayout) -> Scalar[runtime_layout.linear_idx_type]:
    """Returns the offset for a 2D index into the runtime layout.

    Parameters:
        i: The first index to get the offset for.
        j: The second index to get the offset for.

    Args:
        runtime_layout: The runtime layout to get the offset from.

    Returns:
        The offset value for the given indices.
    """

    @parameter
    if runtime_layout.layout.all_dims_known():
        alias offset = runtime_layout.layout(IntTuple(i, j))
        return offset
    else:
        return runtime_layout(
            RuntimeTuple[IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE)](i, j)
        )


struct Element[
    dtype: DType,
    layout: Layout,
    /,
    index_type: DType = _get_index_type(layout),
](Stringable, Writable):
    """A wrapper around SIMD types that provides layout-driven vectorized operations.

    The `Element` struct extends SIMD types with layout-aware load and store
    operations, enabling efficient vectorized access to multi-dimensional data.
    It maps between logical tensor coordinates and physical memory locations
    according to the specified layout.

    Parameters:
        dtype: The data type of the elements.
        layout: The memory layout describing how elements are organized.
        index_type: The integer type of the index pointing to each element.
    """

    alias element_data_type = SIMD[dtype, size = layout.size()]
    """The SIMD type used to store and process the element data.

    This type alias defines a SIMD vector with the specified data type and size
    matching the layout's total element count, enabling efficient vectorized operations.
    """

    var element_data: Self.element_data_type
    """The actual SIMD data stored in this element.

    This field contains the vectorized data values that can be processed
    efficiently using SIMD operations.
    """

    var runtime_layout: RuntimeLayout[
        layout,
        bitwidth = bitwidthof[UInt32](),
        linear_idx_type = Self.index_type,
    ]
    """The runtime layout information for memory access patterns.

    This field stores the layout information needed to map between logical tensor
    coordinates and physical memory locations, supporting both compile-time and
    runtime-determined access patterns.
    """

    @implicit
    fn __init__(out self, element_data: Self.element_data_type):
        """Initializes an Element with the given SIMD data.

        Args:
            element_data: The SIMD data to initialize the element with.
        """
        self.element_data = element_data
        self.runtime_layout = __type_of(self.runtime_layout)()

    fn __init__(
        out self,
        element_data: Self.element_data_type,
        runtime_layout: RuntimeLayout[
            layout,
            bitwidth = bitwidthof[UInt32](),
            linear_idx_type = Self.index_type,
        ],
    ):
        """Initializes an Element with the given SIMD data and runtime layout.

        Args:
            element_data: The SIMD data to initialize the element with.
            runtime_layout: The runtime layout to use for memory access.
        """
        self.element_data = element_data
        self.runtime_layout = runtime_layout

    @always_inline("nodebug")
    @staticmethod
    fn load(
        ptr: UnsafePointer[Scalar[dtype], **_],
        runtime_layout: RuntimeLayout[
            layout,
            bitwidth = bitwidthof[UInt32](),
            linear_idx_type = Self.index_type,
        ] = RuntimeLayout[
            layout,
            bitwidth = bitwidthof[UInt32](),
            linear_idx_type = Self.index_type,
        ](),
    ) -> Self:
        """Loads data from memory according to the specified layout.

        This method loads data from memory using the layout information to determine
        the memory access pattern. It supports both rank-1 and rank-2 layouts with
        various stride patterns, optimizing for contiguous memory access when
        possible.

        Args:
            ptr: Pointer to the memory location to load from.
            runtime_layout: The runtime layout to use for memory access.

        Returns:
            A new `Element` containing the loaded data.
        """
        constrained[layout.rank() <= 2, "Only supports rank <= 2"]()

        var element_data = Self.element_data_type()

        @parameter
        if layout.rank() == 1:
            alias size = layout.size()

            @parameter
            if layout.stride[0] == 1:
                alias alignment = alignof[Self.element_data_type]()
                return ptr.load[
                    width = Self.element_data_type.size, alignment=alignment
                ]()

            @parameter
            for i in range(size):
                element_data[i] = ptr[_get_offset[i](runtime_layout)]
            return Element(element_data, runtime_layout)

        @parameter
        if layout.stride[0] == 1:
            alias size = Int(layout.shape[0])
            alias elements = Int(layout.shape[1])
            alias vec_type = SIMD[dtype, size]
            alias alignment = alignof[vec_type]()

            @parameter
            for i in range(elements):
                var vec_i = ptr.load[width=size, alignment=alignment](
                    _get_offset[0, i](runtime_layout)
                )
                element_data = element_data.insert[offset = i * size](vec_i)
            return Element(element_data, runtime_layout)

        elif layout.stride[1] == 1:
            alias size = Int(layout.shape[1])
            alias elements = Int(layout.shape[0])
            alias vec_type = SIMD[dtype, size]
            alias alignment = alignof[vec_type]()

            @parameter
            for i in range(elements):
                var vec_i = ptr.load[width=size, alignment=alignment](
                    _get_offset[i, 0](runtime_layout)
                )
                element_data = element_data.insert[offset = i * size](vec_i)
            return Element(element_data, runtime_layout)

        alias dim_0 = Int(layout.shape[0])
        alias dim_1 = Int(layout.shape[1])

        @parameter
        for i in range(dim_0):

            @parameter
            for j in range(dim_1):
                element_data[i + j * dim_0] = ptr[
                    _get_offset[i, j](runtime_layout)
                ]
        return Element(element_data, runtime_layout)

    @always_inline("nodebug")
    @staticmethod
    fn masked_load(
        ptr: UnsafePointer[Scalar[dtype], **_],
        runtime_layout: RuntimeLayout[
            layout,
            bitwidth = bitwidthof[UInt32](),
            linear_idx_type = Self.index_type,
        ] = RuntimeLayout[
            layout,
            bitwidth = bitwidthof[UInt32](),
            linear_idx_type = Self.index_type,
        ](),
    ) -> Self:
        """Loads data from memory with masking for partial loads.

        This method loads data from memory using the layout information, but also
        handles cases where the runtime dimensions are smaller than the static
        layout dimensions. It ensures that only valid memory locations are accessed.

        Args:
            ptr: Pointer to the memory location to load from.
            runtime_layout: The runtime layout to use for memory access.

        Returns:
            A new `Element` containing the loaded data, with zeros in positions
            beyond the runtime dimensions.
        """
        # TODO: Use partial_simd_load after closing KERN-729.
        constrained[layout.rank() <= 2, "Only supports rank <= 2"]()
        var element_data = Self.element_data_type()

        @parameter
        if layout.rank() == 1:
            alias size = layout.size()

            @parameter
            if layout.stride[0] == 1:
                alias alignment = alignof[Self.element_data_type]()
                if runtime_layout.dim(0) < size:

                    @parameter
                    for i in range(size):
                        if i >= runtime_layout.dim(0):
                            break
                        element_data[i] = ptr[_get_offset[i](runtime_layout)]
                    return Element(element_data, runtime_layout)

                return ptr.load[
                    width = Self.element_data_type.size, alignment=alignment
                ](0)

            @parameter
            for i in range(size):
                if i >= runtime_layout.dim(0):
                    break
                element_data[i] = ptr[_get_offset[i](runtime_layout)]
            return Element(element_data, runtime_layout)

        # rank-2 element.
        @parameter
        if layout.stride[0] == 1:
            alias size = Int(layout.shape[0])
            alias elements = Int(layout.shape[1])
            alias vec_type = SIMD[dtype, size]
            alias alignment = alignof[vec_type]
            var element_data = Self.element_data_type()
            if runtime_layout.dim(0) < size:
                alias dim_0 = Int(layout.shape[0])
                alias dim_1 = Int(layout.shape[1])

                @parameter
                for i in range(dim_0):
                    if i >= runtime_layout.dim(0):
                        break

                    @parameter
                    for j in range(dim_1):
                        if j >= runtime_layout.dim(1):
                            break
                        element_data[i + j * dim_0] = ptr[
                            _get_offset[i, j](runtime_layout)
                        ]
                return Element(element_data, runtime_layout)

            @parameter
            for i in range(elements):
                if i >= runtime_layout.dim(0):
                    break
                var vec_i = ptr.load[width=size](
                    _get_offset[0, i](runtime_layout)
                )
                element_data = element_data.insert[offset = i * size](vec_i)
            return Element(element_data, runtime_layout)

        elif layout.stride[1] == 1:
            alias size = Int(layout.shape[1])
            alias elements = Int(layout.shape[0])
            alias vec_type = SIMD[dtype, size]
            alias alignment = alignof[vec_type]
            var element_data = Self.element_data_type()
            if runtime_layout.dim(1) < size:
                alias dim_0 = Int(layout.shape[0])
                alias dim_1 = Int(layout.shape[1])

                @parameter
                for i in range(dim_0):
                    if i >= runtime_layout.dim(0):
                        break

                    @parameter
                    for j in range(dim_1):
                        if j >= runtime_layout.dim(1):
                            break
                        element_data[i + j * dim_0] = ptr[
                            _get_offset[i, j](runtime_layout)
                        ]
                return Element(element_data, runtime_layout)

            @parameter
            for i in range(elements):
                if i >= runtime_layout.dim(0):
                    break
                var vec_i = ptr.load[width=size](
                    _get_offset[i, 0](runtime_layout)
                )
                element_data = element_data.insert[offset = i * size](vec_i)
            return Element(element_data, runtime_layout)

        alias dim_0 = Int(layout.shape[0])
        alias dim_1 = Int(layout.shape[1])

        @parameter
        for i in range(dim_0):
            if i >= runtime_layout.dim(0):
                break

            @parameter
            for j in range(dim_1):
                if j >= runtime_layout.dim(1):
                    break
                element_data[i + j * dim_0] = ptr[
                    _get_offset[i, j](runtime_layout)
                ]
        return Element(element_data, runtime_layout)

    @always_inline("nodebug")
    fn store(self, ptr: UnsafePointer[Scalar[dtype], mut=True, **_]):
        """Stores element data to memory according to the specified layout.

        This method performs a layout-aware store operation, writing data to memory
        following the access patterns defined by the layout. It optimizes memory
        writes based on the layout's stride patterns to maximize performance.

        The method handles different memory layout patterns:
        - For rank-1 tensors with contiguous memory (stride=1), it uses vectorized stores
        - For rank-2 tensors with contiguous rows or columns, it uses optimized slice-based stores
        - For non-contiguous memory layouts, it performs element-by-element stores

        Unlike `masked_store()`, this method assumes the full static dimensions will be written
        and does not perform runtime dimension boundary checking.

        Args:
            ptr: Mutable pointer to the memory location where data will be stored.

        Note:
            This method is constrained to layouts with rank <= 2. For higher-rank
            tensors, consider decomposing the operation.
        """
        constrained[layout.rank() <= 2, "Only supports rank <= 2"]()

        @parameter
        if layout.rank() == 1:
            alias size = layout.size()

            @parameter
            if layout.stride[0] == 1:
                alias alignment = alignof[Self.element_data_type]()
                ptr.store[alignment=alignment](self.element_data)
                return

            @parameter
            for i in range(size):
                ptr[_get_offset[i](self.runtime_layout)] = self.element_data[i]
            return

        @parameter
        if layout.stride[0] == 1:
            alias size = Int(layout.shape[0])
            alias elements = Int(layout.shape[1])
            alias vec_type = SIMD[dtype, size]
            alias alignment = alignof[vec_type]()

            @parameter
            for i in range(elements):
                ptr.store[alignment=alignment](
                    _get_offset[0, i](self.runtime_layout),
                    self.element_data.slice[size, offset = i * size](),
                )
            return

        elif layout.stride[1] == 1:
            alias size = Int(layout.shape[1])
            alias elements = Int(layout.shape[0])
            alias vec_type = SIMD[dtype, size]
            alias alignment = alignof[vec_type]()

            @parameter
            for i in range(elements):
                ptr.store[alignment=alignment](
                    _get_offset[i, 0](self.runtime_layout),
                    self.element_data.slice[size, offset = i * size](),
                )
            return

        alias dim_0 = Int(layout.shape[0])
        alias dim_1 = Int(layout.shape[1])

        @parameter
        for i in range(dim_0):

            @parameter
            for j in range(dim_1):
                (ptr + _get_offset[i, j](self.runtime_layout)).store(
                    self.element_data[i + j * dim_0]
                )

    @always_inline("nodebug")
    fn masked_store(self, ptr: UnsafePointer[Scalar[dtype], mut=True, **_]):
        """Stores element data to memory with masking for partial stores.

        This method performs a layout-aware store operation with boundary checking.
        It ensures that only valid memory locations are written to when the runtime
        dimensions are smaller than the static layout dimensions, preventing out-of-bounds
        memory access.

        The method optimizes for different memory layouts:
        - For contiguous memory (stride=1), it uses vectorized stores when possible
        - For non-contiguous memory, it performs element-by-element stores
        - For all patterns, it respects runtime dimension bounds

        Args:
            ptr: Pointer to the memory location where data will be stored.

        Note:
            This method is constrained to layouts with rank <= 2. For higher-rank
            tensors, consider decomposing the operation.
        """
        constrained[layout.rank() <= 2, "Only supports rank <= 2"]()

        @parameter
        if layout.rank() == 1:
            alias size = layout.size()

            @parameter
            if layout.stride[0] == 1:
                if self.runtime_layout.dim(0) < size:

                    @parameter
                    for i in range(size):
                        if i >= self.runtime_layout.dim(0):
                            break
                        ptr[
                            _get_offset[i](self.runtime_layout)
                        ] = self.element_data[i]
                    return

                alias alignment = alignof[Self.element_data_type]()
                ptr.store(self.element_data)
                return

            @parameter
            for i in range(size):
                if i >= self.runtime_layout.dim(0):
                    break
                ptr[_get_offset[i](self.runtime_layout)] = self.element_data[i]
            return

        @parameter
        if layout.stride[0] == 1:
            alias size = Int(layout.shape[0])
            alias elements = Int(layout.shape[1])
            alias vec_type = SIMD[dtype, size]
            alias alignment = alignof[vec_type]()
            if self.runtime_layout.dim(1) < size:
                alias dim_0 = Int(layout.shape[0])
                alias dim_1 = Int(layout.shape[1])

                @parameter
                for i in range(dim_0):
                    if i >= self.runtime_layout.dim(0):
                        break

                    @parameter
                    for j in range(dim_1):
                        if j >= self.runtime_layout.dim(1):
                            break
                        (ptr + _get_offset[i, j](self.runtime_layout)).store(
                            self.element_data[i + j * dim_0],
                        )
                return

            @parameter
            for i in range(elements):
                if i >= self.runtime_layout.dim(0):
                    break
                (ptr + _get_offset[i, 0](self.runtime_layout)).store[
                    alignment=alignment
                ](
                    self.element_data.slice[size, offset = i * size](),
                )
            return

        elif layout.stride[1] == 1:
            alias size = Int(layout.shape[1])
            alias elements = Int(layout.shape[0])
            alias vec_type = SIMD[dtype, size]
            alias alignment = alignof[vec_type]()
            if self.runtime_layout.dim(1) < size:
                alias dim_0 = Int(layout.shape[0])
                alias dim_1 = Int(layout.shape[1])

                @parameter
                for i in range(dim_0):
                    if i >= self.runtime_layout.dim(0):
                        break

                    @parameter
                    for j in range(dim_1):
                        if j >= self.runtime_layout.dim(1):
                            break
                        (ptr + _get_offset[i, j](self.runtime_layout)).store(
                            self.element_data[i + j * dim_0],
                        )
                return

            @parameter
            for i in range(elements):
                if i >= self.runtime_layout.dim(0):
                    break
                (ptr + _get_offset[i, 0](self.runtime_layout)).store[
                    alignment=alignment
                ](
                    self.element_data.slice[size, offset = i * size](),
                )
            return

        alias dim_0 = Int(layout.shape[0])
        alias dim_1 = Int(layout.shape[1])

        @parameter
        for i in range(dim_0):
            if i >= self.runtime_layout.dim(0):
                break

            @parameter
            for j in range(dim_1):
                if j >= self.runtime_layout.dim(1):
                    break
                (ptr + _get_offset[i, j](self.runtime_layout)).store(
                    self.element_data[i + j * dim_0]
                )

    @no_inline
    fn __str__(self) -> String:
        """Returns a string representation of the element.

        Returns:
            A string representation of the element's data.
        """
        return String.write(self)

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        """Writes the element to the specified writer.

        Parameters:
            W: Type parameter representing a Writer implementation.

        Args:
            writer: The writer to output the element representation to.
        """
        writer.write(self.element_data)


struct MemoryElement[
    dtype: DType,
    layout: Layout,
    address_space: AddressSpace,
    alignment: Int,
    /,
    *,
    index_type: DType = _get_index_type(layout, address_space),
]:
    """Represents data in memory organized according to a specific layout.

    The `MemoryElement` struct provides a high-level interface for accessing data
    in memory with a specific layout. It encapsulates a pointer to the memory
    location and the runtime layout information needed to access the data correctly.

    This abstraction enables efficient memory operations that respect the underlying
    memory organization, supporting vectorized loads and stores while handling
    different memory layouts transparently.

    Parameters:
        dtype: The data type of the elements.
        layout: The memory layout describing how elements are organized.
        address_space: The memory address space where the data is located.
        alignment: The memory alignment requirement for the data.
        index_type: The integer type of the index pointing to each memory element.
    """

    var ptr: UnsafePointer[
        Scalar[dtype], address_space=address_space, alignment=alignment
    ]
    """Pointer to the memory location where the data is stored.

    This pointer provides access to the underlying memory with the specified
    address space and alignment requirements. It points to the first element
    of the data structure in memory.
    """

    var runtime_layout: RuntimeLayout[
        layout,
        bitwidth = bitwidthof[UInt32](),
        linear_idx_type = Self.index_type,
    ]
    """Runtime layout information used for memory access calculations.

    This field stores the runtime layout information needed to compute memory
    offsets for accessing elements according to the specified layout pattern.
    It handles both compile-time known dimensions and runtime-determined dimensions.
    """

    fn __init__(
        out self,
        ptr: UnsafePointer[
            Scalar[dtype], address_space=address_space, alignment=alignment
        ],
        runtime_layout: RuntimeLayout[
            layout,
            bitwidth = bitwidthof[UInt32](),
            linear_idx_type = Self.index_type,
        ],
    ):
        """Initializes a `MemoryElement` with the given pointer and runtime layout.

        Args:
            ptr: Pointer to the memory location of the element.
            runtime_layout: The runtime layout to use for memory access.
        """
        self.ptr = ptr
        self.runtime_layout = runtime_layout

    @always_inline("nodebug")
    fn load(
        self, out result: Element[dtype, layout, index_type = Self.index_type]
    ):
        """Loads data from memory according to the specified layout.

        This method performs a layout-aware load operation, reading data from memory
        following the access patterns defined by the layout. It optimizes memory
        reads based on the layout's stride patterns to maximize performance.

        The method leverages the underlying `Element.load` implementation which handles
        different memory layout patterns including contiguous and strided access.

        Returns:
            An `Element` containing the loaded data organized according to the layout.
        """
        return __type_of(result).load(self.ptr, self.runtime_layout)

    @always_inline("nodebug")
    fn store(self, src: Element[dtype, layout, **_]):
        """Stores element data to the memory location of this MemoryElement.

        This method performs a layout-aware store operation, writing data to memory
        following the access patterns defined by the layout. It optimizes memory
        writes based on the layout's stride patterns to maximize performance.

        The method delegates to the `Element.store` implementation which handles
        different memory layout patterns including vectorized stores for contiguous memory
        and element-by-element stores for non-contiguous layouts.

        Args:
            src: The `Element` containing the data to store.
        """
        return src.store(self.ptr)

    @always_inline("nodebug")
    fn transfer(self, src: MemoryElement):
        """Transfers data from another `MemoryElement` to this one.

        This method efficiently transfers data between memory locations with potentially
        different layouts and data types. It performs the following operations:
        1. Loads data from the source `MemoryElement` using its layout
        2. Converts the data to the destination data type if necessary
        3. Stores the converted data to the destination memory location using its layout

        This provides a high-performance way to copy and convert data between different
        memory representations while respecting both source and destination memory layouts.

        Args:
            src: The source `MemoryElement` to transfer data from.
        """
        # Load source element and convert to destination dtype if needed
        var src_element = src.load()
        var converted_element = Element[
            dtype, src.layout, index_type = src.index_type
        ](src_element.element_data.cast[dtype](), src_element.runtime_layout)
        self.store(
            rebind[Element[dtype, layout, index_type = src_element.index_type]](
                converted_element
            )
        )
