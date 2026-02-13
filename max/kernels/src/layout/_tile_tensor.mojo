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

from sys import align_of, simd_width_of
from os import abort

from buffer import Dim, DimList, NDBuffer
from builtin.builtin_slice import ContiguousSlice
from builtin.device_passable import DevicePassable
from builtin.variadics import (
    Variadic,
    _MapValuesAndIdxToType,
    _MapVariadicAndIdxToType,
)
from builtin.dtype import _unsigned_integral_type_of
from gpu import thread_idx, block_dim, lane_id
from gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from utils.numerics import max_finite
from layout._fillers import BATCH_SIZE
from sys import prefetch
from sys.intrinsics import PrefetchOptions, readfirstlane
from utils import IndexList

from .swizzle import Swizzle, make_ldmatrix_swizzle

from ._layout import (
    Layout,
    RowMajorLayout,
    TensorLayout,
    ZippedDivideLayout,
    _RowMajor,
    row_major,
)
from ._coord import (
    ComptimeInt,
    RuntimeInt,
    Idx,
    Coord,
    CoordLike,
    _AllEqual,
    _IntToComptimeInt,
    coord,
    coord_to_int_tuple,
    coord_to_index_list,
    _CoordToDimList,
    _DimsToCoordLike,
    _CoordToDynamic,
    _IntTupleToCoordLike,
    _Multiply,
    _Divide,
    _CeilDiv,
    _Flattened,
    DynamicCoord,
    StaticCoord,
)


@fieldwise_init
struct TileTensor[
    mut: Bool,
    //,
    dtype: DType,
    LayoutType: TensorLayout,
    origin: Origin[mut=mut],
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
    linear_idx_type: DType = _get_index_type(address_space),
    element_shape_types: Variadic.TypesOfTrait[CoordLike] = Variadic.types[
        ComptimeInt[1]
    ],
](DevicePassable, ImplicitlyCopyable, TrivialRegisterPassable, Writable):
    comptime rank = Self.LayoutType.rank
    comptime flat_rank = Variadic.size(
        _Flattened[*Self.LayoutType._shape_types]
    )
    """The flattened rank - total number of dimensions after flattening nested Coords.

    For non-nested layouts, flat_rank == rank.
    For nested layouts (e.g., from blocked_product), flat_rank > rank.
    """
    comptime element_size = Coord[*Self.element_shape_types].static_product
    comptime ElementType = SIMD[Self.dtype, Self.element_size]
    comptime shape_known = Self.LayoutType.shape_known
    comptime stride_known = Self.LayoutType.stride_known
    comptime all_dims_known = Self.LayoutType.all_dims_known
    comptime static_shape[i: Int] = Self.LayoutType.static_shape[i]
    comptime static_stride[i: Int] = Self.LayoutType.static_stride[i]

    var ptr: UnsafePointer[
        Scalar[Self.dtype], Self.origin, address_space = Self.address_space
    ]

    var layout: Self.LayoutType

    comptime device_type = Self

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        """
        Gets the name of the host type (the one implementing this trait).

        Returns:
            The host type's name.
        """

        return String(
            "TileTensor[mut = ",
            Self.mut,
            ", dtype = ",
            Self.dtype,
            ", address_space = ",
            Self.address_space,
            ", linear_idx_type = ",
            Self.linear_idx_type,
            "]",
        )

    comptime GenericType = TileTensor[
        Self.dtype,
        Self.LayoutType,
        Self.origin,
        address_space = AddressSpace.GENERIC,
        linear_idx_type = Self.linear_idx_type,
    ]

    fn __init__(
        out self: Self.GenericType,
        var span: Span[Scalar[Self.dtype], Self.origin],
        var layout: Self.LayoutType,
    ):
        self.ptr = span.unsafe_ptr()
        self.layout = layout

    @always_inline("nodebug")
    fn __init__(
        buffer: NDBuffer[...],
        out self: TileTensor[
            buffer.dtype,
            Layout[
                shape_types = _DimsToCoordLike[DType.int64, buffer.shape],
                stride_types = _DimsToCoordLike[DType.int64, buffer.strides],
            ],
            buffer.origin,
            address_space = buffer.address_space,
        ],
    ):
        self.ptr = buffer.data
        var shape = Coord[*_DimsToCoordLike[DType.int64, buffer.shape]]()
        var stride = Coord[*_DimsToCoordLike[DType.int64, buffer.strides]]()

        @parameter
        for i in range(buffer.rank):

            @parameter
            if not shape.element_types[i].is_static_value:
                shape[i] = rebind[shape.element_types[i]](
                    Scalar[DType.int64](buffer.dynamic_shape[i])
                )

            @parameter
            if not stride.element_types[i].is_static_value:
                stride[i] = rebind[stride.element_types[i]](
                    Scalar[DType.int64](buffer.dynamic_stride[i])
                )
        self.layout = Layout(shape, stride)

    @always_inline
    fn __init__(
        out self: Self.GenericType,
        ref[Self.origin] device_buffer: DeviceBuffer[Self.dtype],
        var layout: Self.LayoutType,
    ):
        """Create a `LayoutTensor` from a `DeviceBuffer`. The layout must have
        statically known dimensions.

        Note that the device buffer memory is on the accelerator device (GPU
        global memory). Code running on the CPU can use the
        [`DeviceContext`](/mojo/std/gpu/host/device_context/DeviceContext) to
        allocate a `DeviceBuffer` and use that to construct a `LayoutTensor`
        that can be accessed on the GPU. You cannot directly access data in the
        `DeviceBuffer` or `LayoutTensor` from the CPU.

        The following example shows a typical pattern for using `DeviceBuffer`
        to construct a `LayoutTensor` that you can use on the GPU.

        ```mojo
        from gpu.host import DeviceContext, DeviceBuffer
        from layout._layout import row_major
        from layout._tile_tensor import TileTensor
        from layout._coord import Idx

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
             row_major((Idx[4](), Idx[4]())),
        )
        ...
        ```
        Args:
            device_buffer: Contains the underlying data to point to.
            layout: The layout of the tensor.
        """
        self = Self.GenericType(
            device_buffer.unsafe_ptr()
            .mut_cast[Self.mut]()
            .unsafe_origin_cast[Self.origin](),
            layout,
        )

    @always_inline
    fn __init__(
        out self: Self.GenericType,
        ref[Self.origin] host_buffer: HostBuffer[Self.dtype],
        var layout: Self.LayoutType,
    ):
        """Create a `LayoutTensor` from a `HostBuffer`. The layout must have
        statically known dimensions.

        The resulting tensor's data can only be accessed on the CPU.

        ```mojo
        from gpu.host import DeviceContext, HostBuffer
        from layout._layout import row_major
        from layout._tile_tensor import TileTensor
        from layout._coord import Idx

        comptime dtype = DType.float32

        var ctx = DeviceContext()
        var host_buf = ctx.enqueue_create_host_buffer[dtype](8)

        var tensor = TileTensor(
            host_buf,
            row_major((Idx[4](), Idx[4]())),
        )
        ```

        Args:
            host_buffer: Contains the underlying data to point to.
            layout: The layout of the tensor.
        """
        self = Self.GenericType(
            host_buffer.unsafe_ptr()
            .mut_cast[Self.mut]()
            .unsafe_origin_cast[Self.origin](),
            layout,
        )

    @always_inline("nodebug")
    fn __getitem__(
        self, coord: Coord
    ) -> Self.ElementType where coord.flat_rank == Self.flat_rank:
        """Retrieves a single element from the tensor at the specified coordinates.

        Accepts Coords of flat_rank (flattened).
        """
        return self.load(coord)

    @always_inline("nodebug")
    fn __getitem__[
        *IndexTypes: Indexer & Copyable
    ](self, *items: *IndexTypes) -> Self.ElementType where (
        Variadic.size(IndexTypes) == Self.flat_rank
    ):
        """Retrieves a single element from the tensor at the specified indices.

        Uses flat indexing based on flat_rank. For non-nested layouts,
        flat_rank == rank, so tensor[i, j, k] works normally. For nested
        layouts (e.g., from blocked_product), use all flat_rank indices:
        tensor[i0, i1, i2, i3] for a tensor with flat_rank == 4.

        Parameters:
            IndexTypes: The types of the indices (must implement Indexer).

        Args:
            items: The indices specifying the element's position.

        Returns:
            The element at the specified position.
        """
        comptime arg_count = Variadic.size(IndexTypes)
        var linear_tuple = DynamicCoord[Self.linear_idx_type, arg_count]()

        @parameter
        for i in range(arg_count):
            UnsafePointer(to=linear_tuple[i]).init_pointee_copy(
                rebind[type_of(linear_tuple).element_types[i]](
                    RuntimeInt[Self.linear_idx_type](
                        Scalar[Self.linear_idx_type](index(items[i]))
                    )
                )
            )

        # Inline load logic to avoid constraint propagation issues
        return self.ptr.load[width = Self.element_size](
            self.layout[linear_idx_type = Self.linear_idx_type](linear_tuple)
        )

    @always_inline("nodebug")
    fn __setitem__(
        self, coord: Coord, value: Self.ElementType
    ) where coord.flat_rank == Self.flat_rank and Self.mut:
        self.store(coord, value)

    @always_inline("nodebug")
    fn __setitem__[
        *IndexTypes: Indexer & Copyable
    ](self, *items: *IndexTypes, value: Self.ElementType) where (
        Variadic.size(IndexTypes) == Self.flat_rank
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
        comptime arg_count = Variadic.size(IndexTypes)
        var linear_tuple = DynamicCoord[Self.linear_idx_type, arg_count]()

        @parameter
        for i in range(arg_count):
            UnsafePointer(to=linear_tuple[i]).init_pointee_copy(
                rebind[type_of(linear_tuple).element_types[i]](
                    RuntimeInt[Self.linear_idx_type](
                        Scalar[Self.linear_idx_type](index(items[i]))
                    )
                )
            )

        # Inline store logic to avoid constraint propagation issues
        self.ptr.mut_cast[True]().store(
            self.layout[linear_idx_type = Self.linear_idx_type](linear_tuple),
            value,
        )

    @always_inline("nodebug")
    fn load[
        width: Int = Self.element_size,
        alignment: Int = align_of[SIMD[Self.dtype, Self.element_size]](),
    ](self, coord: Coord) -> SIMD[Self.dtype, width] where (
        coord.flat_rank == Self.flat_rank
    ):
        """Load elements from the tensor at the specified coordinates.

        Supports both hierarchical indexing (rank indices) and flat indexing
        (flat_rank indices) for nested layouts.

        Parameters:
            width: Number of elements to load (default: element_size).
            alignment: Memory alignment for the load.

        Args:
            coord: The coordinates specifying the element's position.

        Returns:
            A SIMD vector containing the loaded elements.
        """
        return self.ptr.load[width=width, alignment=alignment](
            self.layout[linear_idx_type = Self.linear_idx_type](coord)
        )

    @always_inline("nodebug")
    fn store[
        width: Int = Self.element_size,
        alignment: Int = align_of[SIMD[Self.dtype, Self.element_size]](),
    ](self, coord: Coord, value: SIMD[Self.dtype, width]) where (
        coord.flat_rank == Self.flat_rank and Self.mut
    ):
        """Store elements to the tensor at the specified coordinates.

        Supports both hierarchical indexing (rank indices) and flat indexing
        (flat_rank indices) for nested layouts.

        Parameters:
            width: Number of elements to store (default: element_size).
            alignment: Memory alignment for the store.

        Args:
            coord: The coordinates specifying the element's position.
            value: The SIMD vector to store.
        """
        self.ptr.mut_cast[True]().store[alignment=alignment](
            self.layout[linear_idx_type = Self.linear_idx_type](coord), value
        )

    @always_inline
    fn ptr_at_offset(
        self, coords: Coord[...]
    ) -> UnsafePointer[
        Scalar[Self.dtype], Self.origin, address_space = Self.address_space
    ] where (coords.rank == Self.rank):
        """Get a pointer offset at the given flattened coordinates.

        Args:
            coords: A flattened list of the offset coordinates.

        Returns:
            A pointer offset at the given flattened coordinates.
        """

        return self.ptr + self.layout[linear_idx_type = Self.linear_idx_type](
            coords
        )

    @always_inline
    fn prefetch(self, coords: Coord[...]) where coords.rank == Self.rank:
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

    fn numel(self) -> Int:
        var result = 1

        @parameter
        for i in range(Self.rank):
            result *= self.layout.shape[i]().value()
        return result

    fn write_to(self, mut w: Some[Writer]):
        """Format and write the tensor's contents to a writer.

        This method formats the tensor's contents and writes them to the
        provided writer. For 2D tensors, it formats the output in a 2D grid. For
        tensors of other ranks, it prints all values in column-major coordinate
        order.

        Args:
            w: The writer instance to write the formatted output to.

        Example:

        ```mojo
        from layout._tile_tensor import TileTensor
        from layout._layout import row_major

        def main():
            var storage = InlineArray[Float32, 2 * 3](uninitialized=True)
            var tensor = TileTensor(storage, row_major[2, 3]()).fill(1.0)
            print(tensor)  # Internally calls `write_to` with a StringWriter
        ```

        Output for a 2x3 tensor:

        ```
        [[1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]]
        ```

        Notes:

        - For 2D tensors, the output is formatted as a 2D grid with rows and
            columns.
        - For tensors of other ranks, values are printed in column-major
            coordinate order.
        - Empty tensors (size 0) produce no output.
        - This method is used by the `__str__` method to convert the tensor to a
            string.
        - The formatting is designed for human readability rather than parsing.
        - For large tensors, the output may be truncated to avoid excessive
            output.
        """

        if self.layout.product() == 0:
            return

        # The 2D print works only for layout shape (M, N).
        # Check both original and coalesced layouts so that (M, 1) and
        # ((M), (N)) can all be printed in 2D. Shapes like ((2, 2), 2) will be
        # printed elementwise.

        @parameter
        if Self.flat_rank == 2:
            comptime assert Self.flat_rank == 2

            @parameter
            if Self.static_shape[0] > -1 and Self.static_shape[1] > -1:
                _pretty_print_2d_tensor(self, w)
                return

    @always_inline("nodebug")
    fn tile[
        *tile_sizes: Int
    ](self, coordinates: Coord) -> TileTensor[
        dtype = Self.dtype,
        origin = Self.origin,
        LayoutType = Layout[
            shape_types = _IntToComptimeInt[*tile_sizes],
            stride_types = Self.LayoutType._stride_types,
        ],
        address_space = Self.address_space,
        linear_idx_type = Self.linear_idx_type,
        element_shape_types = Self.element_shape_types,
    ]:
        return _tile(self, coord[*tile_sizes](), coordinates)

    @always_inline("nodebug")
    fn tile[
        *tile_sizes: Int, stride_layout: TensorLayout
    ](self, coordinates: Coord) -> TileTensor[
        dtype = Self.dtype,
        origin = Self.origin,
        LayoutType = Layout[
            shape_types = _IntToComptimeInt[*tile_sizes],
            stride_types = stride_layout._shape_types,
        ],
        address_space = Self.address_space,
        linear_idx_type = Self.linear_idx_type,
        element_shape_types = Self.element_shape_types,
    ]:
        """Tile with explicit static strides.

        Use when the parent tensor has dynamic (RuntimeInt) strides but
        the actual stride values are known at compile time. This produces
        a tile with all_dims_known=True, enabling vectorize/distribute.

        This is needed because TensorLayout trait parameters erase concrete
        stride types -- the compiler cannot prove all_dims_known through
        a trait-bounded parameter even when the underlying strides are static.
        """
        return _tile[stride_layout=stride_layout](
            self, coord[*tile_sizes](), coordinates
        )

    @always_inline("nodebug")
    fn tile_with_offset[
        *tile_sizes: Int
    ](self, coordinates: Coord) -> Tuple[
        TileTensor[
            dtype = Self.dtype,
            origin = Self.origin,
            LayoutType = Layout[
                shape_types = _IntToComptimeInt[*tile_sizes],
                stride_types = Self.LayoutType._stride_types,
            ],
            address_space = Self.address_space,
            linear_idx_type = Self.linear_idx_type,
            element_shape_types = Self.element_shape_types,
        ],
        IndexList[Variadic.size(coordinates.element_types)],
        UInt,
    ]:
        """Like tile(), but also returns corner coordinates and linear offset.

        Parameters:
            tile_sizes: Tile dimensions along each axis.

        Args:
            coordinates: Tile coordinates in the grid.

        Returns:
            Tuple of (tile, corner_coords, offset).
        """
        return _tile_with_offset(self, coord[*tile_sizes](), coordinates)

    @always_inline("nodebug")
    fn tile_with_offset[
        *tile_sizes: Int, stride_layout: TensorLayout
    ](self, coordinates: Coord) -> Tuple[
        TileTensor[
            dtype = Self.dtype,
            origin = Self.origin,
            LayoutType = Layout[
                shape_types = _IntToComptimeInt[*tile_sizes],
                stride_types = stride_layout._shape_types,
            ],
            address_space = Self.address_space,
            linear_idx_type = Self.linear_idx_type,
            element_shape_types = Self.element_shape_types,
        ],
        IndexList[Variadic.size(coordinates.element_types)],
        UInt,
    ]:
        """Like tile(), but with explicit static strides.

        Use when the parent has dynamic strides but the values are known
        at compile time. See tile[stride_layout=...] for details.
        """
        return _tile_with_offset[stride_layout=stride_layout](
            self, coord[*tile_sizes](), coordinates
        )

    comptime ViewType[new_layout: TensorLayout] = TileTensor[
        dtype = Self.dtype,
        LayoutType=new_layout,
        origin = Self.origin,
        address_space = Self.address_space,
        linear_idx_type = Self.linear_idx_type,
        element_shape_types = Self.element_shape_types,
    ]
    """A TileTensor type with the same data properties but a different layout.

    Preserves dtype, origin, address_space, and other properties while
    replacing LayoutType. Use this to name the return type of reshape()
    and other layout-changing operations in helper functions.

    Parameters:
        new_layout: The new TensorLayout type for the view.
    """

    @always_inline("nodebug")
    fn reshape[
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
        return {self.ptr, layout_val}

    # flatten_leading is defined as a standalone function below the
    # struct. As a method, Self.LayoutType._shape_types[i] in the return
    # type is symbolic and can't match value-level types. As a standalone
    # function, type_of(tensor).LayoutType resolves correctly.

    @always_inline("nodebug")
    fn tile[
        *tile_sizes: Int
    ](self, *tile_coords: Int) -> TileTensor[
        dtype = Self.dtype,
        origin = Self.origin,
        LayoutType = Layout[
            shape_types = _IntToComptimeInt[*tile_sizes],
            stride_types = Self.LayoutType._stride_types,
        ],
        address_space = Self.address_space,
        linear_idx_type = Self.linear_idx_type,
        element_shape_types = Self.element_shape_types,
    ]:
        """Extract a tile (sub-tensor) from this tensor with specified
        dimensions and position.

        This overload accepts tile coordinates as variadic Int arguments,
        providing API compatibility with LayoutTensor.

        Parameters:
            tile_sizes: The dimensions of each tile along each axis.

        Args:
            tile_coords: The coordinates of the specific tile to extract.

        Returns:
            A view into the original tensor representing the specified tile.

        Example:

        ```mojo
        from layout._layout import row_major
        from layout._tile_tensor import TileTensor

        var storage = InlineArray[Float32, 16](uninitialized=True)
        var tensor = TileTensor(storage, row_major[4, 4]()).fill(1.0)

        # Extract the tile at position (1, 0) with tile size 2x2
        var t = tensor.tile[2, 2](1, 0)
        ```
        """
        var coordinates = DynamicCoord[Self.linear_idx_type, Self.rank]()

        @parameter
        for i in range(Self.rank):
            UnsafePointer(to=coordinates[i]).init_pointee_copy(
                rebind[coordinates.element_types[i]](
                    RuntimeInt[Self.linear_idx_type](
                        Scalar[Self.linear_idx_type](tile_coords[i])
                    )
                )
            )

        return _tile(self, coord[*tile_sizes](), coordinates)

    @always_inline("nodebug")
    fn distribute[
        thread_layout: Layout,
        swizzle: Optional[Swizzle] = None,
    ](self, thread_id: Int) -> TileTensor[
        dtype = Self.dtype,
        origin = Self.origin,
        LayoutType = Layout[
            shape_types = _Divide[
                Self.LayoutType._shape_types, thread_layout.shape_types
            ],
            stride_types = _Multiply[
                Self.LayoutType._stride_types, thread_layout.shape_types
            ],
        ],
        address_space = Self.address_space,
        linear_idx_type = Self.linear_idx_type,
        element_shape_types = Self.element_shape_types,
    ] where Self.all_dims_known:
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

    @always_inline("nodebug")
    fn distribute_with_offset[
        thread_layout: Layout,
        swizzle: Optional[Swizzle] = None,
    ](self, thread_id: Int) -> Tuple[
        TileTensor[
            dtype = Self.dtype,
            origin = Self.origin,
            LayoutType = Layout[
                shape_types = _Divide[
                    Self.LayoutType._shape_types, thread_layout.shape_types
                ],
                stride_types = _Multiply[
                    Self.LayoutType._stride_types, thread_layout.shape_types
                ],
            ],
            address_space = Self.address_space,
            linear_idx_type = Self.linear_idx_type,
            element_shape_types = Self.element_shape_types,
        ],
        IndexList[Variadic.size(thread_layout.shape_types)],
        UInt,
    ] where Self.all_dims_known:
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

    @always_inline
    fn fill[
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
        from layout._layout import row_major
        from layout._tile_tensor import TileTensor

        def main():
            var storage = InlineArray[Float32, 3 * 4](uninitialized=True)
            var tensor = TileTensor(storage, row_major[3,4]()).fill(0.0)
            print(tensor)
        ```

        If not using method chaining, you can either reassign the result to the
        tensor variable, or assign the result to the discard pattern (`_`) to
        avoid warnings about an unused value:

        ```mojo
        from layout._layout import row_major
        from layout._tile_tensor import TileTensor

        var storage = InlineArray[Float32, 3 * 4](uninitialized=True)
        var tensor = TileTensor(storage, row_major[3,4]()).fill(0.0)
        tensor = tensor.fill(0.0)
        # or
        _ = tensor.fill(0.0)
        ```
        """

        @parameter
        if not use_runtime_layout:
            comptime num_elements = Coord[
                *Self.LayoutType._shape_types
            ].static_product

            # TODO: MSTDL-1352 we can use memory element to fill the tensor.
            @parameter
            for i in range(num_elements):
                var idx = self.layout(Idx[i]())
                self.ptr.mut_cast[True]()[idx] = val
        else:
            var num_elements = self.numel()

            for i in range(num_elements):
                var idx = self.layout(Idx(i))
                self.ptr.mut_cast[True]()[idx] = val
        return self

    @always_inline("nodebug")
    fn dim[i: Int](self) -> Scalar[Self.linear_idx_type]:
        return Scalar[Self.linear_idx_type](self.layout.shape[i]().value())

    @always_inline("nodebug")
    fn dim[
        IndexType: Indexer
    ](self, index: IndexType) -> Scalar[Self.linear_idx_type]:
        var idx = std.builtin.int.index(index)

        @parameter
        for i in range(Self.rank):
            if idx == i:
                return Scalar[Self.linear_idx_type](
                    self.layout.shape[i]().value()
                )
        # Should this raise instead?
        std.os.abort("attempt to dynamically index out of bounds")

    @always_inline("nodebug")
    fn dynamic_stride[
        IndexType: Indexer
    ](self, index: IndexType) -> Scalar[Self.linear_idx_type]:
        var idx = std.builtin.int.index(index)

        @parameter
        for i in range(Self.rank):
            if idx == i:
                return Scalar[Self.linear_idx_type](
                    self.layout.stride[i]().value()
                )
        # Should this raise instead?
        std.os.abort("attempt to dynamically index out of bounds")

    @always_inline
    fn slice[
        *slices: ContiguousSlice
    ](self) -> TileTensor[
        Self.dtype,
        Layout[
            shape_types = _Slice[slices, Self.LayoutType._shape_types],
            stride_types = Self.LayoutType._stride_types,
        ],
        Self.origin,
        address_space = Self.address_space,
        linear_idx_type = Self.linear_idx_type,
        element_shape_types = Self.element_shape_types,
    ] where (Variadic.size(slices) == Self.flat_rank and Self.all_dims_known):
        """Extract a slice from the tensor using slice objects.

        This method creates a view into a subset of the tensor defined by the
        slice specifications for each dimension. The slice is a continuous
        region of the tensor with no gaps (step size must be 1 for all dimensions).

        The number of slice arguments must match the tensor rank.

        Parameters:
            slices: Slice specifications for each dimension. Each slice defines
                the start and end indices for that dimension.

        Returns:
            A view into the original tensor representing the specified slice.
            The returned tensor has the same rank but smaller dimensions.

        Example:


        For a 3D tensor, you can slice all three dimensions:

        ```mojo
        from layout._layout import row_major
        from layout._tile_tensor import TileTensor
        comptime layout_3d = row_major[16, 16, 16]()
        var stack = InlineArray[UInt8, layout_3d.static_product](fill=0)
        var tensor_3d = TileTensor(stack, layout_3d)
        var slice = tensor_3d.slice[0:2, 1:3, 0:4]()
        ```

        Performance:

        - Creates a view without copying data, making it very efficient.
        - Maintains the original tensor's stride information for efficient
            memory access.
        - Zero-cost abstraction at runtime when used with compile-time constant
            slices.

        Notes:

        - The slice is a view into the original tensor, so modifications to the
            slice will affect the original tensor.
        - Works with tensors of any rank (must provide one slice per dimension).
        - The step size must be 1 for all dimensions (no gaps allowed).
        - Slice bounds are not checked at runtime; accessing out-of-bounds
            indices will result in undefined behavior.
        - Shape and stride types are converted to RuntimeInt in the sliced
            tensor, even if the original tensor had ComptimeInt dimensions.
            This is necessary because we can't change ComptimeInt[4] to
            ComptimeInt[2] in the type system.
        """

        # Compute offset based on slice start indices and strides
        var offset = 0

        @parameter
        for i in range(Variadic.size(slices)):
            comptime slice_i = slices[i]
            comptime slice_start = slice_i.start.or_else(0)
            var stride_i = self.layout.stride[i]().value()
            offset += slice_start * stride_i

        # Build new shape tuple with runtime types
        # Even though slice bounds are compile-time known, we use RuntimeInt
        # because we can't change ComptimeInt[4] to ComptimeInt[2] in the type system
        comptime NewShapeTypes = _Slice[slices, Self.LayoutType._shape_types]
        var new_shape = Coord[*NewShapeTypes]()

        @parameter
        for i in range(Self.rank):
            comptime slice_i = slices[i]
            comptime slice_start = slice_i.start.or_else(0)

            var shape_ptr = UnsafePointer(to=new_shape[i])
            comptime NewShapeType = NewShapeTypes[i]

            shape_ptr.init_pointee_copy(
                rebind[NewShapeType](ComptimeInt[NewShapeType.static_value]())
            )

        # Strides remain unchanged
        var new_layout = Layout(new_shape, self.layout.stride_coord())

        return TileTensor[
            Self.dtype,
            Layout[
                shape_types=NewShapeTypes,
                stride_types = Self.LayoutType._stride_types,
            ],
            Self.origin,
            address_space = Self.address_space,
            linear_idx_type = Self.linear_idx_type,
            element_shape_types = Self.element_shape_types,
        ](self.ptr + offset, new_layout)

    # ===------------------------------------------------------------------=== #
    # Vectorization
    # ===------------------------------------------------------------------=== #

    comptime VectorizedType[*vector_shape: Int] = TileTensor[
        dtype = Self.dtype,
        origin = Self.origin,
        LayoutType = Layout[
            shape_types = _CeilDiv[
                Self.LayoutType._shape_types, _IntToComptimeInt[*vector_shape]
            ],
            stride_types = _Multiply[
                Self.LayoutType._stride_types, _IntToComptimeInt[*vector_shape]
            ],
        ],
        address_space = Self.address_space,
        linear_idx_type = Self.linear_idx_type,
        element_shape_types = _IntToComptimeInt[*vector_shape],
    ]
    """Type alias for vectorized tensor types.

    Parameters:
        vector_shape: The shape of each vector unit along each axis.
    """

    comptime SIMDVectorizedType = Self.VectorizedType[
        1, simd_width_of[Self.dtype]()
    ]
    """Result type for SIMD-width vectorization."""

    @always_inline("nodebug")
    fn vectorize[
        *vector_shape: Int
    ](self) -> Self.VectorizedType[*vector_shape] where Self.all_dims_known:
        """Reshape a tensor into a vectorized form for efficient SIMD operations.

        This method transforms the tensor's logical layout to enable efficient
        vectorized processing, treating blocks of elements as vector units. The
        transformation is particularly useful for SIMD (Single Instruction
        Multiple Data) operations and hardware acceleration.

        The vector shape is tracked in `element_shape_types`.

        Parameters:
            vector_shape: The dimensions of each vector unit along each axis of
                the tensor. For example, in a 2D tensor, `vectorize[4, 4]` treats
                4x4 blocks as vector units.

        Returns:
            A view of the tensor with a vectorized layout, where each element in
            the resulting tensor represents the start of a vector block from the
            original tensor. The element layout is tracked via
            `element_shape_types` (the vector shape).

        Constraints:
            All dimensions must be statically known (`all_dims_known`).

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
        return _vectorize(self, coord[*vector_shape]())

    @always_inline("nodebug")
    fn vectorize(self) -> Self.SIMDVectorizedType where Self.all_dims_known:
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

    comptime CoalescedType = TileTensor[
        dtype = Self.dtype,
        origin = Self.origin,
        LayoutType = Layout[
            shape_types = Variadic.types[
                ComptimeInt[Coord[*Self.LayoutType._shape_types].static_product]
            ],
            stride_types = Variadic.types[ComptimeInt[1]],
        ],
        address_space = Self.address_space,
        linear_idx_type = Self.linear_idx_type,
        element_shape_types = Variadic.types[
            ComptimeInt[Coord[*Self.element_shape_types].static_product]
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

    comptime ReshapedType[*new_shape_types: CoordLike] = TileTensor[
        dtype = Self.dtype,
        origin = Self.origin,
        LayoutType = Layout[
            shape_types = Variadic.types[*new_shape_types],
            stride_types = _RowMajor[*new_shape_types],
        ],
        address_space = Self.address_space,
        linear_idx_type = Self.linear_idx_type,
        element_shape_types = Self.element_shape_types,
    ]
    """Type alias for reshaped tensor types.

    Parameters:
        new_shape_types: The shape types for the reshaped tensor.
    """

    @always_inline("nodebug")
    fn reshape[
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
        from layout._layout import row_major
        from layout._tile_tensor import TileTensor

        var storage = InlineArray[Float32, 12](uninitialized=True)
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

        return Self.ReshapedType[*NewShapeTypes](self.ptr, new_layout)

    @always_inline("nodebug")
    fn reshape[
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
        from layout._layout import row_major
        from layout._tile_tensor import TileTensor
        from layout._coord import Idx, Coord

        var storage = InlineArray[Float32, 12](uninitialized=True)
        var tensor = TileTensor(storage, row_major[3, 4]()).fill(1.0)

        # Reshape with runtime-determined dimensions
        var rows = 2
        var cols = 6
        var reshaped = tensor.reshape(Coord(Idx(rows), Idx(cols)))
        ```

        Performance:

        - Creates a view without copying data.
        - May include runtime validation for dynamic shapes.
        """
        # Runtime validation for element count
        debug_assert(
            self.numel() == new_shape.product(),
            "reshape: total number of elements must match",
        )

        var new_layout = row_major(new_shape)

        return Self.ReshapedType[*new_shape_types](self.ptr, new_layout)

    @always_inline("nodebug")
    fn coalesce(
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
        comptime element_size = Coord[*Self.element_shape_types].static_product

        var new_layout = Layout(
            Coord(ComptimeInt[total_size]()),
            Coord(ComptimeInt[1]()),
        )

        return Self.CoalescedType(self.ptr, new_layout)

    comptime DynamicType[dyn_dtype: DType] = TileTensor[
        dtype = Self.dtype,
        origin = Self.origin,
        LayoutType = Layout[
            shape_types = _CoordToDynamic[
                dyn_dtype, *Self.LayoutType._shape_types
            ],
            stride_types = _CoordToDynamic[
                dyn_dtype, *Self.LayoutType._stride_types
            ],
        ],
        address_space = Self.address_space,
        linear_idx_type = Self.linear_idx_type,
    ]
    """Type alias for dynamic tensor types.

    Parameters:
        dyn_dtype: The data type for RuntimeInt values in the dynamic tensor.
    """

    @always_inline("nodebug")
    fn make_dynamic[dyn_dtype: DType](self) -> Self.DynamicType[dyn_dtype]:
        """Convert all elements in shape and stride to RuntimeInt[dyn_dtype].

        Parameters:
            dyn_dtype: The data type for the resulting RuntimeInt values.

        Returns:
            A new TileTensor where all elements in shape and stride
            are converted to RuntimeInt[dyn_dtype].

        Examples:
            ```mojo
            from layout._tile_tensor import TileTensor
            from layout._layout import row_major
            var storage = InlineArray[Float32, 12](uninitialized=True)
            var tensor = TileTensor(Span(storage), row_major[3, 4]())
            var dynamic = tensor.make_dynamic[DType.int64]()
            # dynamic has RuntimeInt[DType.int64] for all shape/stride dimensions
            ```
        """
        return Self.DynamicType[dyn_dtype](
            self.ptr,
            Layout(
                self.layout.shape_coord(), self.layout.stride_coord()
            ).make_dynamic[dyn_dtype](),
        )

    @always_inline("nodebug")
    fn to_layout_tensor(
        self,
        out result: LayoutTensor[
            Self.dtype,
            layout.Layout(
                coord_to_int_tuple[*Self.LayoutType._shape_types](),
                coord_to_int_tuple[*Self.LayoutType._stride_types](),
            ),
            Self.origin,
            address_space = Self.address_space,
        ],
    ):
        """Return a LayoutTensor with the same shape, stride, and address space
        of this tensor. Currently it expects flat layouts.

        This is a utility to help with porting LayoutTensor methods to this type.

        Returns:
            A LayoutTensor with the same shape, stride, and address space of
            this tensor.
        """
        return {
            self.ptr,
            layout.RuntimeLayout[result.layout](
                coord_to_index_list(self.layout.shape_coord()),
                coord_to_index_list(self.layout.stride_coord()),
            ),
        }

    @always_inline("nodebug")
    fn _to_ndbuffer(
        self,
        out result: NDBuffer[
            Self.dtype,
            Self.rank,
            Self.origin,
            _CoordToDimList[*Self.LayoutType._shape_types],
            _CoordToDimList[*Self.LayoutType._stride_types],
            address_space = Self.address_space,
        ],
    ):
        """Return an NDBuffer with the same shape, stride, and address space
        of this tensor. Currently it expects flat layouts.

        This is a utility to help with porting NDBuffer methods to this type.

        Returns:
            An NDBuffer with the same shape, stride, and address space of
            this tensor.
        """

        return {
            self.ptr,
            rebind[IndexList[Self.rank]](
                coord_to_index_list(self.layout.shape_coord())
            ),
            rebind[IndexList[Self.rank]](
                coord_to_index_list(self.layout.stride_coord())
            ),
        }

    comptime OriginCastType[
        mut: Bool,
        //,
        origin: Origin[mut=mut],
    ] = TileTensor[
        dtype = Self.dtype,
        origin=origin,
        LayoutType = Self.LayoutType,
        address_space = Self.address_space,
        linear_idx_type = Self.linear_idx_type,
    ]
    """Type alias for origin-cast result tensors.

    Parameters:
        mut: Whether the result tensor is mutable.
        origin: The origin for the result tensor.
    """

    @always_inline("nodebug")
    fn as_any_origin(self) -> Self.OriginCastType[AnyOrigin[mut = Self.mut]]:
        """Casts the origin of the mutable `LayoutTensor` to `MutAnyOrigin`.

        Returns:
            A pointer with the origin set to `MutAnyOrigin`.

        This requires the tensor to already be mutable as casting mutability
        is inherently very unsafe.

        It is usually preferred to maintain concrete origin values instead of
        using `MutAnyOrigin`. However, if it is needed, keep in mind that
        `MutAnyOrigin` can alias any memory value, so Mojo's ASAP
        destruction will not apply during the lifetime of the tensor.
        """
        return {self.ptr.as_any_origin(), self.layout}

    @always_inline
    fn as_immut(
        self,
    ) -> Self.OriginCastType[ImmutOrigin(Self.origin)]:
        """
        Return an immutable version of this tensor.

        Returns:
            A `LayoutTensor` covering the same elements, but without mutability.
        """
        return {self.ptr.as_immutable(), self.layout}

    @always_inline
    fn to_device_buffer(self, ctx: DeviceContext) -> DeviceBuffer[Self.dtype]:
        """Convert the tensor to a `DeviceBuffer`.

        Args:
            ctx: The device context to use.

        Returns:
            A `DeviceBuffer` containing the tensor's data.
        """
        comptime assert (
            Self.address_space == Self.address_space.GENERIC
        ), "DeviceBuffer is only used on GENERIC address space"
        return DeviceBuffer[Self.dtype](
            ctx,
            self.ptr,
            self.numel(),
            owning=False,
        )


@always_inline("nodebug")
fn stack_allocation[
    LayoutType: TensorLayout,
    //,
    dtype: DType,
    address_space: AddressSpace = AddressSpace.GENERIC,
](var layout: LayoutType) -> TileTensor[
    dtype,
    LayoutType,
    MutExternalOrigin,
    address_space=address_space,
] where LayoutType.all_dims_known:
    return TileTensor[
        dtype,
        LayoutType,
        MutExternalOrigin,
        address_space=address_space,
    ](
        std.memory.stack_allocation[
            Coord[*LayoutType._shape_types].static_product,
            Scalar[dtype],
            address_space=address_space,
        ](),
        layout,
    )


@always_inline
fn _pretty_print_2d_tensor[
    W: Writer
](tensor: TileTensor, mut writer: W) where tensor.flat_rank == 2:
    # Provide evidence to the constraint system
    comptime assert tensor.flat_rank == 2
    var m_dim = tensor.layout.shape[0]()
    var n_dim = tensor.layout.shape[1]()
    for m in range(m_dim.value()):
        for n in range(n_dim.value()):
            writer.write(tensor[m, n], " ")
        if m < m_dim.value() - 1:
            writer.write("\n")


@always_inline("nodebug")
fn _distribute[
    thread_layout: Layout,
    swizzle: Optional[Swizzle] = None,
](
    data_layout_tensor: TileTensor,
    thread_id: Int,
) -> TileTensor[
    data_layout_tensor.dtype,
    Layout[
        shape_types = _Divide[
            data_layout_tensor.LayoutType._shape_types,
            thread_layout.shape_types,
        ],
        stride_types = _Multiply[
            data_layout_tensor.LayoutType._stride_types,
            thread_layout.shape_types,
        ],
    ],
    data_layout_tensor.origin,
    address_space = data_layout_tensor.address_space,
    linear_idx_type = data_layout_tensor.linear_idx_type,
    element_shape_types = data_layout_tensor.element_shape_types,
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

    var offset: UInt = 0

    @parameter
    for i in range(Variadic.size(thread_layout.stride_types)):
        comptime stride_i = thread_layout.stride_types[i].static_value
        comptime shape_i = thread_layout.shape_types[i].static_value
        var thread_coord_i = (thread_id // stride_i) % shape_i
        offset += UInt(
            thread_coord_i * data_layout_tensor.layout.stride[i]().value()
        )

    # Swizzling applies to the index of elements rather than scalars because
    # the former is the unit in distribution.
    var swizzled_offset = offset

    @parameter
    if swizzle:
        comptime swizzle_fn = swizzle.value()
        comptime element_size = data_layout_tensor.element_size
        swizzled_offset = UInt(
            swizzle_fn(Int(offset) // element_size) * element_size
        )

    comptime ShapeType = Coord[
        *_Divide[
            data_layout_tensor.LayoutType._shape_types,
            thread_layout.shape_types,
        ]
    ]
    comptime StrideType = Coord[
        *_Multiply[
            data_layout_tensor.LayoutType._stride_types,
            thread_layout.shape_types,
        ]
    ]
    # Since the thread layout and tensor layout have all_dims_known this is safe
    comptime assert ShapeType.all_dims_known
    var shape = ShapeType()
    comptime assert StrideType.all_dims_known
    var stride = StrideType()

    var layout = Layout(shape, stride)

    return TileTensor[
        data_layout_tensor.dtype,
        _,
        data_layout_tensor.origin,
        address_space = data_layout_tensor.address_space,
        linear_idx_type = data_layout_tensor.linear_idx_type,
        element_shape_types = data_layout_tensor.element_shape_types,
    ](
        UnsafePointer(to=data_layout_tensor.ptr[swizzled_offset]),
        layout,
    )


@always_inline("nodebug")
fn _distribute_with_offset[
    thread_layout: Layout,
    swizzle: Optional[Swizzle] = None,
](
    data_layout_tensor: TileTensor,
    thread_id: Int,
) -> Tuple[
    TileTensor[
        data_layout_tensor.dtype,
        Layout[
            shape_types = _Divide[
                data_layout_tensor.LayoutType._shape_types,
                thread_layout.shape_types,
            ],
            stride_types = _Multiply[
                data_layout_tensor.LayoutType._stride_types,
                thread_layout.shape_types,
            ],
        ],
        data_layout_tensor.origin,
        address_space = data_layout_tensor.address_space,
        linear_idx_type = data_layout_tensor.linear_idx_type,
        element_shape_types = data_layout_tensor.element_shape_types,
    ],
    IndexList[Variadic.size(thread_layout.shape_types)],
    UInt,
]:
    """Like _distribute, but also returns thread coordinates and offset.

    The thread coordinates are the position of the current thread in
    the thread grid. The offset is the linear element offset (after
    optional swizzling) used to advance the pointer.
    """

    # Use shape_types consistently for the IndexList size (must match return type)
    var offset: UInt = 0
    var thread_coords = IndexList[Variadic.size(thread_layout.shape_types)]()

    @parameter
    for i in range(Variadic.size(thread_layout.shape_types)):
        comptime stride_i = thread_layout.stride_types[i].static_value
        comptime shape_i = thread_layout.shape_types[i].static_value
        var thread_coord_i = (thread_id // stride_i) % shape_i
        thread_coords[i] = Int(thread_coord_i)
        offset += UInt(
            thread_coord_i * data_layout_tensor.layout.stride[i]().value()
        )

    # Swizzling applies to the index of elements rather than scalars because
    # the former is the unit in distribution.
    var swizzled_offset = offset

    @parameter
    if swizzle:
        comptime swizzle_fn = swizzle.value()
        comptime element_size = data_layout_tensor.element_size
        swizzled_offset = UInt(
            swizzle_fn(Int(offset) // element_size) * element_size
        )

    comptime ShapeType = Coord[
        *_Divide[
            data_layout_tensor.LayoutType._shape_types,
            thread_layout.shape_types,
        ]
    ]
    comptime StrideType = Coord[
        *_Multiply[
            data_layout_tensor.LayoutType._stride_types,
            thread_layout.shape_types,
        ]
    ]
    comptime assert ShapeType.all_dims_known
    var shape = ShapeType()
    comptime assert StrideType.all_dims_known
    var stride = StrideType()

    var layout = Layout(shape, stride)

    return (
        TileTensor[
            data_layout_tensor.dtype,
            _,
            data_layout_tensor.origin,
            address_space = data_layout_tensor.address_space,
            linear_idx_type = data_layout_tensor.linear_idx_type,
            element_shape_types = data_layout_tensor.element_shape_types,
        ](
            UnsafePointer(to=data_layout_tensor.ptr[swizzled_offset]),
            layout,
        ),
        thread_coords,
        swizzled_offset,
    )


@always_inline("nodebug")
fn _tile[
    dtype: DType,
    coord_types: Variadic.TypesOfTrait[CoordLike],
    tile_shape_types: Variadic.TypesOfTrait[CoordLike],
    element_shape_types: Variadic.TypesOfTrait[CoordLike],
    //,
](
    data_layout_tensor: TileTensor[
        dtype,
        element_shape_types=element_shape_types,
        ...,
    ],
    tile_shape: Coord[*tile_shape_types],
    tile_coords: Coord[*coord_types],
) -> TileTensor[
    dtype,
    Layout[
        shape_types=tile_shape_types,
        stride_types = data_layout_tensor.LayoutType._stride_types,
    ],
    data_layout_tensor.origin,
    address_space = data_layout_tensor.address_space,
    linear_idx_type = data_layout_tensor.linear_idx_type,
    element_shape_types=element_shape_types,
]:
    """Extract a tile (sub-tensor) from a TileTensor at specified coordinates.

    This function creates a view into a specific rectangular region of the source tensor
    without copying data. It computes the memory offset for the tile and creates a new
    TileTensor with the tile dimensions while preserving the original stride pattern.

    Difference from LayoutTensor.tile:
        This simplified implementation returns a tile with the original tensor's
        stride information rather than creating a hierarchical (blocked/tiled)
        layout with an appropriate stride.

        It is incorrect for non-divisible tile shapes (like dividing a 16x16 tensor
        into 3x3 tiles).

    Parameters:
        dtype: Data type of the tensor elements (inferred from tensor argument).
        coord_types: Types of the tile coordinates (inferred from coordinates argument).
        tile_shape_types: Types of the tile dimensions (inferred from tile_shape argument).
        element_shape_types: Element layout shape types (inferred from tensor argument).

    Args:
        data_layout_tensor: The source tensor to extract the tile from.
        tile_shape: The shape that the layout should be tiled into.
        tile_coords: The index of the tile to extract as a Coord.

    Returns:
        A TileTensor representing a view into the specified tile region.
        The returned tensor has the tile_shape as its dimensions and shares memory
        with the original tensor. Element types are propagated from the source tensor.
    """

    var offset: UInt = 0

    @parameter
    for i in range(Coord[*coord_types].__len__()):
        offset += UInt(
            tile_coords[i].value()
            * tile_shape[i].value()
            * data_layout_tensor.layout.stride[i]().value()
        )

    var tile_layout = Layout(
        shape=tile_shape,
        stride=data_layout_tensor.layout.stride_coord(),
    )

    return TileTensor[
        dtype,
        Layout[
            shape_types=tile_shape_types,
            stride_types = data_layout_tensor.LayoutType._stride_types,
        ],
        data_layout_tensor.origin,
        address_space = data_layout_tensor.address_space,
        linear_idx_type = data_layout_tensor.linear_idx_type,
        element_shape_types=element_shape_types,
    ](
        UnsafePointer(to=data_layout_tensor.ptr[offset]),
        tile_layout,
    )


@always_inline("nodebug")
fn _tile_with_offset[
    dtype: DType,
    coord_types: Variadic.TypesOfTrait[CoordLike],
    tile_shape_types: Variadic.TypesOfTrait[CoordLike],
    element_shape_types: Variadic.TypesOfTrait[CoordLike],
    //,
](
    data_layout_tensor: TileTensor[
        dtype,
        element_shape_types=element_shape_types,
        ...,
    ],
    tile_shape: Coord[*tile_shape_types],
    tile_coords: Coord[*coord_types],
) -> Tuple[
    TileTensor[
        dtype,
        Layout[
            shape_types=tile_shape_types,
            stride_types = data_layout_tensor.LayoutType._stride_types,
        ],
        data_layout_tensor.origin,
        address_space = data_layout_tensor.address_space,
        linear_idx_type = data_layout_tensor.linear_idx_type,
        element_shape_types=element_shape_types,
    ],
    IndexList[Variadic.size(coord_types)],
    UInt,
]:
    """Like _tile, but also returns corner coordinates and linear offset.

    The corner coordinates are the element-space coordinates of the tile's
    origin: corner_coords[i] = tile_coords[i] * tile_sizes[i].
    The offset is the linear element offset used to advance the pointer.
    """

    # Use Variadic.size(coord_types) consistently (must match return type)
    var offset: UInt = 0
    var corner_coords = IndexList[Variadic.size(coord_types)]()

    @parameter
    for i in range(Variadic.size(coord_types)):
        corner_coords[i] = Int(tile_coords[i].value() * tile_shape[i].value())
        offset += UInt(
            tile_coords[i].value()
            * tile_shape[i].value()
            * data_layout_tensor.layout.stride[i]().value()
        )

    var tile_layout = Layout(
        shape=tile_shape,
        stride=data_layout_tensor.layout.stride_coord(),
    )

    return (
        TileTensor[
            dtype,
            Layout[
                shape_types=tile_shape_types,
                stride_types = data_layout_tensor.LayoutType._stride_types,
            ],
            data_layout_tensor.origin,
            address_space = data_layout_tensor.address_space,
            linear_idx_type = data_layout_tensor.linear_idx_type,
            element_shape_types=element_shape_types,
        ](
            UnsafePointer(to=data_layout_tensor.ptr[offset]),
            tile_layout,
        ),
        corner_coords,
        offset,
    )


@always_inline("nodebug")
fn _tile[
    dtype: DType,
    coord_types: Variadic.TypesOfTrait[CoordLike],
    tile_shape_types: Variadic.TypesOfTrait[CoordLike],
    element_shape_types: Variadic.TypesOfTrait[CoordLike],
    //,
    *,
    stride_layout: TensorLayout,
](
    data_layout_tensor: TileTensor[
        dtype,
        element_shape_types=element_shape_types,
        ...,
    ],
    tile_shape: Coord[*tile_shape_types],
    tile_coords: Coord[*coord_types],
) -> TileTensor[
    dtype,
    Layout[
        shape_types=tile_shape_types,
        stride_types = stride_layout._shape_types,
    ],
    data_layout_tensor.origin,
    address_space = data_layout_tensor.address_space,
    linear_idx_type = data_layout_tensor.linear_idx_type,
    element_shape_types=element_shape_types,
]:
    """Like _tile, but with explicit static strides.

    Use when the parent tensor has dynamic strides (e.g. from a
    TensorLayout trait parameter) but the stride values are known at
    compile time. The resulting tile has ComptimeInt strides, enabling
    vectorize/distribute which require all_dims_known.
    """

    var offset: UInt = 0

    @parameter
    for i in range(Coord[*coord_types].__len__()):
        offset += UInt(
            tile_coords[i].value()
            * tile_shape[i].value()
            * data_layout_tensor.layout.stride[i]().value()
        )

    var tile_layout = Layout(
        shape=tile_shape,
        stride=Coord[*stride_layout._shape_types](),
    )

    return TileTensor[
        dtype,
        Layout[
            shape_types=tile_shape_types,
            stride_types = stride_layout._shape_types,
        ],
        data_layout_tensor.origin,
        address_space = data_layout_tensor.address_space,
        linear_idx_type = data_layout_tensor.linear_idx_type,
        element_shape_types=element_shape_types,
    ](
        UnsafePointer(to=data_layout_tensor.ptr[offset]),
        tile_layout,
    )


@always_inline("nodebug")
fn _tile_with_offset[
    dtype: DType,
    coord_types: Variadic.TypesOfTrait[CoordLike],
    tile_shape_types: Variadic.TypesOfTrait[CoordLike],
    element_shape_types: Variadic.TypesOfTrait[CoordLike],
    //,
    *,
    stride_layout: TensorLayout,
](
    data_layout_tensor: TileTensor[
        dtype,
        element_shape_types=element_shape_types,
        ...,
    ],
    tile_shape: Coord[*tile_shape_types],
    tile_coords: Coord[*coord_types],
) -> Tuple[
    TileTensor[
        dtype,
        Layout[
            shape_types=tile_shape_types,
            stride_types = stride_layout._shape_types,
        ],
        data_layout_tensor.origin,
        address_space = data_layout_tensor.address_space,
        linear_idx_type = data_layout_tensor.linear_idx_type,
        element_shape_types=element_shape_types,
    ],
    IndexList[Variadic.size(coord_types)],
    UInt,
]:
    """Like _tile_with_offset, but with explicit static strides."""

    var offset: UInt = 0
    var corner_coords = IndexList[Variadic.size(coord_types)]()

    @parameter
    for i in range(Variadic.size(coord_types)):
        corner_coords[i] = Int(tile_coords[i].value() * tile_shape[i].value())
        offset += UInt(
            tile_coords[i].value()
            * tile_shape[i].value()
            * data_layout_tensor.layout.stride[i]().value()
        )

    var tile_layout = Layout(
        shape=tile_shape,
        stride=Coord[*stride_layout._shape_types](),
    )

    return (
        TileTensor[
            dtype,
            Layout[
                shape_types=tile_shape_types,
                stride_types = stride_layout._shape_types,
            ],
            data_layout_tensor.origin,
            address_space = data_layout_tensor.address_space,
            linear_idx_type = data_layout_tensor.linear_idx_type,
            element_shape_types=element_shape_types,
        ](
            UnsafePointer(to=data_layout_tensor.ptr[offset]),
            tile_layout,
        ),
        corner_coords,
        offset,
    )


@always_inline("nodebug")
fn _vectorize[
    dtype: DType,
    vector_shape_types: Variadic.TypesOfTrait[CoordLike],
    //,
](
    data_layout_tensor: TileTensor[dtype, ...],
    vector_shape: Coord[*vector_shape_types],
) -> TileTensor[
    dtype,
    Layout[
        shape_types = _CeilDiv[
            data_layout_tensor.LayoutType._shape_types, vector_shape_types
        ],
        stride_types = _Multiply[
            data_layout_tensor.LayoutType._stride_types,
            vector_shape_types,
        ],
    ],
    data_layout_tensor.origin,
    address_space = data_layout_tensor.address_space,
    linear_idx_type = data_layout_tensor.linear_idx_type,
    element_shape_types=vector_shape_types,
]:
    """Create a vectorized view of a TileTensor.

    This function creates a new view where the shape is divided by the vector
    shape (ceiling division) and strides are multiplied by the vector shape.
    This effectively groups elements into vector-sized blocks. The element
    layout is tracked via element_shape_types.

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
        data_layout_tensor.LayoutType._shape_types, vector_shape_types
    ]
    comptime NewStrideTypes = _Multiply[
        data_layout_tensor.LayoutType._stride_types, vector_shape_types
    ]

    # Since all_dims_known is required, we can use compile-time values directly
    comptime assert Coord[*NewShapeTypes].all_dims_known
    comptime assert Coord[*NewStrideTypes].all_dims_known
    var new_shape = Coord[*NewShapeTypes]()
    var new_stride = Coord[*NewStrideTypes]()

    var new_layout = Layout(new_shape, new_stride)

    return TileTensor[
        dtype,
        Layout[
            shape_types=NewShapeTypes,
            stride_types=NewStrideTypes,
        ],
        data_layout_tensor.origin,
        address_space = data_layout_tensor.address_space,
        linear_idx_type = data_layout_tensor.linear_idx_type,
        element_shape_types=vector_shape_types,
    ](data_layout_tensor.ptr, new_layout)


fn _get_index_type(address_space: AddressSpace) -> DType:
    """Returns int32 for shared/constant GPU memory, index otherwise."""
    if address_space in (
        AddressSpace.SHARED,
        AddressSpace.CONSTANT,
    ):
        return DType.int32
    else:
        return DType.int64


comptime _ToRuntimeMapper[
    dtype: DType,
    element_types: Variadic.TypesOfTrait[CoordLike],
    idx: Int,
] = RuntimeInt[dtype]
"""Convert shape types to RuntimeInt for slicing operations.

When slicing, compile-time dimensions become runtime dimensions because
we can't change ComptimeInt[4] to ComptimeInt[2] in the type system.

Parameters:
    dtype: The default data type to use for RuntimeInt conversions.
    element_types: The variadic sequence of types to convert (wrapped in values).
    idx: The current index being processed.
"""


comptime _ToRuntimeInts[
    element_types: Variadic.TypesOfTrait[CoordLike], dtype: DType
] = _MapVariadicAndIdxToType[
    To=CoordLike,
    VariadicType=element_types,
    Mapper = _ToRuntimeMapper[dtype],
]
"""Convert all shape types to RuntimeInt for slicing operations.

Parameters:
    element_types: The original shape types (may include ComptimeInt).
    dtype: The data type to use for RuntimeInt conversions.
"""

comptime _SliceMapper[
    slices: Variadic.ValuesOfType[ContiguousSlice],
    From: Variadic.TypesOfTrait[CoordLike],
    idx: Int,
] = ComptimeInt[
    slices[idx].end.or_else(From[idx].static_value)
    - slices[idx].start.or_else(0)
]

comptime _Slice[
    slices: Variadic.ValuesOfType[ContiguousSlice],
    element_types: Variadic.TypesOfTrait[CoordLike],
] = _MapVariadicAndIdxToType[
    To=CoordLike,
    VariadicType=element_types,
    Mapper = _SliceMapper[slices=slices],
]


comptime _IsRowMajorMapper[
    expected_strides: Variadic.TypesOfTrait[CoordLike],
    element_types: Variadic.TypesOfTrait[CoordLike],
    idx: Int,
] = ComptimeInt[
    1 if element_types[idx].static_value
    == expected_strides[idx].static_value else 0
]
"""Check if stride at index matches expected row-major stride."""


comptime _IsRowMajorHelper[
    shape_types: Variadic.TypesOfTrait[CoordLike],
    stride_types: Variadic.TypesOfTrait[CoordLike],
] = _MapVariadicAndIdxToType[
    To=CoordLike,
    VariadicType=stride_types,
    Mapper = _IsRowMajorMapper[expected_strides = _RowMajor[*shape_types]],
]
"""Returns variadic of ComptimeInt[1] if strides match, ComptimeInt[0] if not."""


comptime _IsRowMajor[
    shape_types: Variadic.TypesOfTrait[CoordLike],
    stride_types: Variadic.TypesOfTrait[CoordLike],
] = Coord[*_IsRowMajorHelper[shape_types, stride_types]].static_product == (
    1 if Variadic.size(shape_types)
    == 0 else StaticCoord[1, Variadic.size(shape_types)].static_product
)
"""Check if stride_types match row-major strides for shape_types.

Returns True if all strides match the expected row-major pattern,
False otherwise. For row-major, stride[i] = product(shape[i+1:]).
"""


# =============================================================================
# Standalone reshape helpers
# =============================================================================


comptime _FlatLeadingLayout[L: TensorLayout] = RowMajorLayout[
    RuntimeInt[DType.int64], L._shape_types[L.rank - 1]
]
"""Layout type after merging leading two dims: (A, B, C) -> (A*B, C).

The merged dimension is always RuntimeInt. The last dimension preserves
its original static/dynamic type.
"""


@always_inline("nodebug")
fn flatten_leading[
    dtype: DType,
    layout: TensorLayout,
    //,
](
    tensor: TileTensor[dtype=dtype, LayoutType=layout, ...],
) -> tensor.ViewType[
    RowMajorLayout[
        RuntimeInt[DType.int64], layout._shape_types[layout.rank - 1]
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
    constrained[type_of(tensor).rank == 3, "flatten_leading requires rank 3"]()
    var merged = RuntimeInt[DType.int64](
        Int64(tensor.layout.shape[0]().value())
        * Int64(tensor.layout.shape[1]().value())
    )
    comptime ResultLayout = RowMajorLayout[
        RuntimeInt[DType.int64], layout._shape_types[layout.rank - 1]
    ]
    return rebind[tensor.ViewType[ResultLayout]](
        tensor.reshape(row_major(Coord(merged, tensor.layout.shape[2]())))
    )
