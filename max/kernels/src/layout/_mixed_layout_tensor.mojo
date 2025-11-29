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

from sys import align_of

from builtin.variadics import VariadicOf, variadic_size
from builtin.dtype import _unsigned_integral_type_of
from gpu.host import DeviceBuffer, HostBuffer
from utils.numerics import max_finite

from ._mixed_layout import MixedLayout
from ._mixed_tuple import (
    ComptimeInt,
    RuntimeInt,
    Idx,
    MixedTuple,
    MixedTupleLike,
    _AllEqual,
)


@fieldwise_init
struct MixedLayoutTensor[
    mut: Bool,
    dtype: DType,
    shape_types: VariadicOf[MixedTupleLike],
    stride_types: VariadicOf[MixedTupleLike], //,
    origin: Origin[mut],
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
    linear_idx_type: DType = _get_index_type(address_space),
](Copyable, Movable, Writable):
    comptime rank = variadic_size(Self.shape_types)
    comptime ALL_DIMS_KNOWN = MixedTuple[
        *Self.shape_types
    ].ALL_DIMS_KNOWN and MixedTuple[*Self.stride_types].ALL_DIMS_KNOWN

    var ptr: UnsafePointer[
        Scalar[Self.dtype], Self.origin, address_space = Self.address_space
    ]

    var layout: MixedLayout[
        shape_types = Self.shape_types,
        stride_types = Self.stride_types,
    ]

    comptime GenericType = MixedLayoutTensor[
        dtype = Self.dtype,
        shape_types = Self.shape_types,
        stride_types = Self.stride_types,
        origin = Self.origin,
        address_space = AddressSpace.GENERIC,
        linear_idx_type = Self.linear_idx_type,
    ]

    fn __init__(
        out self: Self.GenericType,
        var span: Span[Scalar[Self.dtype], Self.origin],
        var layout: MixedLayout[Self.shape_types, Self.stride_types],
    ):
        self.ptr = span.unsafe_ptr()
        self.layout = layout^

    @always_inline
    fn __init__(
        out self: Self.GenericType,
        ref [Self.origin]device_buffer: DeviceBuffer[Self.dtype],
        var layout: MixedLayout[Self.shape_types, Self.stride_types],
    ):
        """Create a `LayoutTensor` from a `DeviceBuffer`. The layout must have
        statically known dimensions.

        Note that the device buffer memory is on the accelerator device (GPU
        global memory). Code running on the CPU can use the
        [`DeviceContext`](/mojo/stdlib/gpu/host/device_context/DeviceContext) to
        allocate a `DeviceBuffer` and use that to construct a `LayoutTensor`
        that can be accessed on the GPU. You cannot directly access data in the
        `DeviceBuffer` or `LayoutTensor` from the CPU.

        The following example shows a typical pattern for using `DeviceBuffer`
        to construct a `LayoutTensor` that you can use on the GPU.

        ```mojo
        from gpu.host import DeviceContext, DeviceBuffer
        from layout._mixed_layout import row_major
        from layout._mixed_layout_tensor import MixedLayoutTensor
        from layout._mixed_tuple import Idx

        comptime dtype = DType.float32

        var ctx = DeviceContext()
        # Allocate buffers
        var dev_buf = ctx.enqueue_create_buffer[dtype](16)
        var host_buf = ctx.enqueue_create_host_buffer[dtype](16)
        # Ensure buffers have been created
        ctx.synchronize()

        # Initialize host buffer and copy to device buffer
        for i in range(16):
            host_buf[i] = i
        ctx.enqueue_copy(dev_buf, host_buf)

        # Create MixedLayoutTensor to use on device
        var tensor = MixedLayoutTensor(
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
            layout^,
        )

    @always_inline
    fn __init__(
        out self: Self.GenericType,
        ref [Self.origin]host_buffer: HostBuffer[Self.dtype],
        var layout: MixedLayout[Self.shape_types, Self.stride_types],
    ):
        """Create a `LayoutTensor` from a `HostBuffer`. The layout must have
        statically known dimensions.

        The resulting tensor's data can only be accessed on the CPU.

        ```mojo
        from gpu.host import DeviceContext, HostBuffer
        from layout._mixed_layout import row_major
        from layout._mixed_layout_tensor import MixedLayoutTensor
        from layout._mixed_tuple import Idx

        comptime dtype = DType.float32

        var ctx = DeviceContext()
        var host_buf = ctx.enqueue_create_host_buffer[dtype](8)

        var tensor = MixedLayoutTensor(
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
            layout^,
        )

    @always_inline("nodebug")
    fn __getitem__(
        self, tuple: MixedTuple
    ) -> Scalar[Self.dtype] where variadic_size(
        tuple.element_types
    ) == variadic_size(Self.shape_types):
        return self.ptr[
            self.layout[linear_idx_type = Self.linear_idx_type](tuple)
        ]

    @always_inline("nodebug")
    fn __getitem__(
        self, tuple: Tuple
    ) -> Scalar[Self.dtype] where variadic_size(
        tuple.element_types
    ) == variadic_size(Self.shape_types) where _AllEqual[
        Int, *tuple.element_types
    ]:
        var linear_tuple: MixedTuple[
            *_Splatted[RuntimeInt[Self.linear_idx_type], Self.rank]
        ]
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(linear_tuple)
        )

        @parameter
        for i in range(variadic_size(tuple.element_types)):
            UnsafePointer(to=linear_tuple[i]).init_pointee_copy(
                rebind[type_of(linear_tuple).element_types[i]](
                    RuntimeInt[Self.linear_idx_type](rebind[Int](tuple[i]))
                )
            )
        return self.ptr[
            self.layout[linear_idx_type = Self.linear_idx_type](linear_tuple)
        ]

    @always_inline("nodebug")
    fn __setitem__(
        self: MixedLayoutTensor[
            mut=True,
            dtype = Self.dtype,
            shape_types = Self.shape_types,
            stride_types = Self.stride_types,
            address_space = Self.address_space,
            linear_idx_type = Self.linear_idx_type,
        ],
        tuple: MixedTuple,
        value: Scalar[Self.dtype],
    ) where variadic_size(tuple.element_types) == variadic_size(
        Self.shape_types
    ):
        self.ptr[
            self.layout[linear_idx_type = Self.linear_idx_type](tuple)
        ] = value

    fn numel(self) -> Int:
        var result = 1

        @parameter
        for i in range(Self.rank):
            result *= self.layout.shape[i].value()
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
        from layout._mixed_layout_tensor import MixedLayoutTensor
        from layout._mixed_layout import row_major
        from layout._mixed_tuple import Idx

        def main():
            var storage = InlineArray[Float32, 2 * 3](uninitialized=True)
            var tensor = MixedLayoutTensor(storage, (Idx[2], Idx[3])).fill(1.0)
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

        if self.layout.size() == 0:
            return

        # The 2D print works only for layout shape (M, N).
        # Check both original and coalesced layouts so that (M, 1) and
        # ((M), (N)) can all be printed in 2D. Shapes like ((2, 2), 2) will be
        # printed elementwise.

        @parameter
        if Self.rank == 2:

            @parameter
            if (
                Self.shape_types[0].STATIC_VALUE > -1
                and Self.shape_types[1].STATIC_VALUE > -1
            ):
                _pretty_print_2d_tensor(self, w)
                return


@always_inline
fn _pretty_print_2d_tensor[
    W: Writer
](tensor: MixedLayoutTensor, mut writer: W) where tensor.rank == 2:
    var m_dim = tensor.layout.shape[0]
    var n_dim = tensor.layout.shape[1]
    for m in range(m_dim.value()):
        for n in range(n_dim.value()):
            writer.write(tensor[(m, n)], " ")
        if m < m_dim.value() - 1:
            writer.write("\n")


fn distribute[
    thread_shape_0: Int,
    thread_shape_1: Int,
    thread_stride_0: Int,
    thread_stride_1: Int,
    data_shape_0: Int,
    data_shape_1: Int,
    data_stride_0: Int,
    data_stride_1: Int, //,
    dtype: DType,
    thread_layout: MixedLayout[
        Tuple[
            ComptimeInt[thread_shape_0], ComptimeInt[thread_shape_1]
        ].element_types,
        Tuple[
            ComptimeInt[thread_stride_0], ComptimeInt[thread_stride_1]
        ].element_types,
    ],
](
    data_layout_tensor: MixedLayoutTensor[
        dtype=dtype,
        shape_types = Tuple[
            ComptimeInt[data_shape_0], ComptimeInt[data_shape_1]
        ].element_types,
        stride_types = Tuple[
            ComptimeInt[data_stride_0], ComptimeInt[data_stride_1]
        ].element_types,
    ],
    thread_id: Int,
) -> MixedLayoutTensor[
    dtype = data_layout_tensor.dtype,
    shape_types = MixedTuple[
        ComptimeInt[data_shape_0 // thread_shape_0],
        ComptimeInt[data_shape_1 // thread_shape_1],
    ].element_types,
    stride_types = Tuple[
        ComptimeInt[data_stride_0 * thread_shape_0],
        ComptimeInt[data_stride_1 * thread_shape_1],
    ].element_types,
    data_layout_tensor.origin,
    address_space = data_layout_tensor.address_space,
    linear_idx_type = data_layout_tensor.linear_idx_type,
]:
    """A simplified implementation of LayoutTensor.distribute on MixedLayoutTensor.
    """

    var offset: UInt = 0

    @parameter
    for i in range(len(thread_layout.stride)):
        comptime stride_i = Int(thread_layout.stride[i].value())
        comptime shape_i = Int(thread_layout.shape[i].value())
        var thread_coord_i = (thread_id // stride_i) % shape_i
        offset += UInt(
            thread_coord_i * Int(data_layout_tensor.layout.stride[i].value())
        )

    comptime shape = MixedTuple(
        ComptimeInt[data_shape_0 // thread_shape_0](),
        ComptimeInt[data_shape_1 // thread_shape_1](),
    )

    comptime stride = MixedTuple(
        ComptimeInt[data_stride_0 * thread_shape_0](),
        ComptimeInt[data_stride_1 * thread_shape_1](),
    )

    var frag_layout = MixedLayout(
        shape=shape,
        stride=stride,
    )

    return MixedLayoutTensor[dtype = data_layout_tensor.dtype,](
        UnsafePointer(to=data_layout_tensor.ptr[offset]),
        rebind[
            MixedLayout[
                shape_types = type_of(shape._storage).element_types,
                stride_types = type_of(stride._storage).element_types,
            ]
        ](frag_layout),
    )


fn tile[
    dtype: DType,
    shape_types: VariadicOf[MixedTupleLike],
    stride_types: VariadicOf[MixedTupleLike],
    coord_types: VariadicOf[MixedTupleLike],
    tile_shape_types: VariadicOf[MixedTupleLike], //,
](
    data_layout_tensor: MixedLayoutTensor[
        dtype=dtype, shape_types=shape_types, stride_types=stride_types
    ],
    tile_shape: MixedTuple[*tile_shape_types],
    tile_coords: MixedTuple[*coord_types],
) -> MixedLayoutTensor[
    dtype=dtype,
    shape_types=tile_shape_types,
    stride_types=stride_types,
    data_layout_tensor.origin,
    address_space = data_layout_tensor.address_space,
    linear_idx_type = data_layout_tensor.linear_idx_type,
]:
    """Extract a tile (sub-tensor) from a MixedLayoutTensor at specified coordinates.

    This function creates a view into a specific rectangular region of the source tensor
    without copying data. It computes the memory offset for the tile and creates a new
    MixedLayoutTensor with the tile dimensions while preserving the original stride pattern.

    Difference from LayoutTensor.tile:
        This simplified implementation returns a tile with the original tensor's
        stride information rather than creating a hierarchical (blocked/tiled)
        layout with an appropriate stride.

        It is incorrect for non-divisible tile shapes (like dividing a 16x16 tensor
        into 3x3 tiles).

    Parameters:
        dtype: Data type of the tensor elements (inferred from tensor argument).
        shape_types: Shape types of the source tensor (inferred from tensor argument).
        stride_types: Stride types of the source tensor (inferred from tensor argument).
        coord_types: Types of the tile coordinates (inferred from coordinates argument).
        tile_shape_types: Types of the tile dimensions (inferred from tile_shape argument).

    Args:
        data_layout_tensor: The source tensor to extract the tile from.
        tile_shape: The shape that the layout should be tiled into.
        tile_coords: The index of the tile to extract as a MixedTuple.

    Returns:
        A MixedLayoutTensor representing a view into the specified tile region.
        The returned tensor has the tile_shape as its dimensions and shares memory
        with the original tensor.
    """

    var offset: UInt = 0

    @parameter
    for i in range(MixedTuple[*coord_types].__len__()):
        offset += UInt(
            tile_coords[i].value()
            * tile_shape[i].value()
            * Int(data_layout_tensor.layout.stride[i].value())
        )

    var tile_layout = MixedLayout(
        shape=tile_shape,
        stride=data_layout_tensor.layout.stride,
    )

    return MixedLayoutTensor[
        dtype=dtype,
        shape_types=tile_shape_types,
        stride_types=stride_types,
    ](
        UnsafePointer(to=data_layout_tensor.ptr[offset]),
        tile_layout,
    )


struct MixedLayoutTensorIter[
    mut: Bool,
    dtype: DType,
    shape_types: VariadicOf[MixedTupleLike],
    stride_types: VariadicOf[MixedTupleLike], //,
    origin: Origin[mut],
    /,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
    axis: Optional[Int] = None,
    linear_idx_type: DType = _get_index_type(address_space),
](Copyable, ImplicitlyCopyable, Iterable, Iterator, Movable):
    """Iterator for traversing a memory buffer with a specific layout.

    `MixedLayoutTensorIter` provides a way to iterate through memory according to a
    specific layout pattern, constructing layout tensors at each position. This
    enables efficient traversal of multi-dimensional data structures with custom
    memory layouts.

    Parameters:
        mut: Whether the iterator allows mutation of the underlying data.
        dtype: The data type of the tensor elements.
        shape_types: The inferred shape types from the layout.
        stride_types: The inferred stride types from the layout.
        origin: Origin tracking for memory safety.
        address_space: The memory address space (`GLOBAL`, `SHARED`, etc.).
        axis: Optional axis for dimension-specific operations.
        linear_idx_type: Integer type used for indexing into memory.

    Notes:

    The returned layout tensor is NOT vectorized. Users should explicitly vectorize
    if needed for performance-critical operations.
    """

    alias IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[iterable_mut]
    ]: Iterator = Self
    alias Element = Self.MixedLayoutTensorType

    comptime linear_uint_type = Scalar[
        _unsigned_integral_type_of[Self.linear_idx_type]()
    ]
    """The unsigned integer type used for indexing into memory."""

    var ptr: UnsafePointer[
        Scalar[Self.dtype], Self.origin, address_space = Self.address_space
    ]
    """Pointer to the memory region being iterated, with appropriate type and memory attributes."""

    var offset: Self.linear_uint_type
    """Current offset from the base pointer, representing the iterator's position in memory."""

    var stride: Self.linear_uint_type
    """Step size between consecutive elements or blocks in memory during iteration."""

    var bound: Self.linear_uint_type
    """Upper bound of the memory region, limiting the iteration range."""

    comptime MixedLayoutType = MixedLayout[Self.shape_types, Self.stride_types]
    var layout: Self.MixedLayoutType
    """Representation of the layout pattern used for mapping logical indices to memory locations."""

    var idx: Self.linear_uint_type
    """Current logical index position within the iteration sequence."""

    @always_inline
    fn __init__(
        out self: MixedLayoutTensorIter[
            dtype = Self.dtype,
            shape_types = Self.shape_types,
            stride_types = Self.stride_types,
            Self.origin,
            address_space = AddressSpace.GENERIC,
            axis = Self.axis,
            linear_idx_type = Self.linear_idx_type,
        ],
        span: Span[Scalar[Self.dtype], Self.origin],
        var layout: MixedLayout[Self.shape_types, Self.stride_types],
        offset: Self.linear_uint_type = 0,
        idx: Self.linear_uint_type = 0,
    ):
        """Initialize an iterator with a runtime layout.

        Creates an iterator with a runtime-determined layout, allowing for more
        flexible memory traversal patterns.

        Args:
            span: Span containing the memory region.
            layout: Layout determined at runtime.
            offset: Initial offset from the base pointer.
            idx: Initial index position.

        Constraints:
            The runtime layout must have the same bitwidth as specified for the
            iterator.
        """
        self = {span.unsafe_ptr(), len(span), layout^, offset, idx}

    @always_inline
    fn __init__(
        out self,
        ptr: UnsafePointer[
            Scalar[Self.dtype], Self.origin, address_space = Self.address_space
        ],
        bound: Self.linear_uint_type,
        var layout: MixedLayout[Self.shape_types, Self.stride_types],
        offset: Self.linear_uint_type = 0,
        idx: Self.linear_uint_type = 0,
    ):
        """Initialize an iterator with a runtime layout.

        Creates an iterator with a runtime-determined layout, allowing for more
        flexible memory traversal patterns.

        Args:
            ptr: Pointer to the beginning of the memory region.
            bound: Upper bound of the memory region.
            layout: Layout determined at runtime.
            offset: Initial offset from the base pointer.
            idx: Initial index position.

        Constraints:
            The runtime layout must have the same bitwidth as specified for the
            iterator.
        """

        constrained[
            Self.linear_idx_type.is_signed(),
            "Linear index type must be signed.",
        ]()

        self.ptr = ptr
        self.offset = offset
        self.bound = bound
        self.layout = layout
        self.stride = layout.size()
        self.idx = idx

    comptime MixedLayoutTensorType = MixedLayoutTensor[
        dtype = Self.dtype,
        shape_types = Self.shape_types,
        stride_types = Self.stride_types,
        Self.origin,
        address_space = Self.address_space,
        linear_idx_type = Self.linear_idx_type,
    ]

    @always_inline
    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    @always_inline
    fn get(self) -> Self.MixedLayoutTensorType:
        """Get the layout tensor at the current iterator position.

        Returns a layout tensor representing the data at the current position
        of the iterator.

        Returns:
            A tensor view at the current iterator position with the
            same type, layout, and memory characteristics as specified by the
            output parameter.
        """

        return Self.MixedLayoutTensorType(
            self.ptr + Int(self.offset),
            self.layout,
        )

    @always_inline
    fn _incr(mut self):
        """Increment the iterator by 1.

        Advances the iterator by a single position. This is equivalent to
        `iter += 1` but without the division operation, making it more
        efficient.
        """
        self.offset += self.stride

    @always_inline
    fn __next__(mut self) -> Self.Element:
        """Return an iterator pointing to a position ahead by rhs steps.

        Creates a new iterator that points rhs positions ahead of the current
        one.


        Returns:
           A MixedLayoutTensor at the given offset.
        """
        var next_idx = Self.linear_uint_type(0)
        var next_offset = self.offset + self.stride
        var item = self.get()

        @parameter
        if Self.axis:
            next_idx = self.idx + 1
        self.idx = next_idx
        self.offset = next_offset

        return item^

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.offset < self.bound

    comptime BitcastType[
        new_type: DType, *, address_space: AddressSpace = Self.address_space
    ] = MixedLayoutTensorIter[
        dtype=new_type,
        shape_types = Self.shape_types,
        stride_types = Self.stride_types,
        Self.origin,
        address_space=address_space,
        linear_idx_type = Self.linear_idx_type,
    ]

    @always_inline
    fn bitcast[
        new_type: DType,
        *,
        target_address_space: AddressSpace = Self.address_space,
    ](self) -> Self.BitcastType[new_type, address_space = Self.address_space]:
        """Reinterpret the iterator's underlying pointer as a different data
        type.

        This method performs a bitcast operation, allowing you to view the same
        memory location as a different data type without copying or converting
        the data.

        Parameters:
            new_type: The target data type to cast to.
            target_address_space: The memory address space for the new
                iterator (defaults to current).

        Returns:
            A new MixedLayoutTensorIter with the same layout but different data type.
        """
        return Self.BitcastType[new_type, address_space = Self.address_space](
            self.ptr.bitcast[Scalar[new_type]]().address_space_cast[
                Self.address_space
            ](),
            Int(self.bound),
            self.layout,
            Int(self.offset),
            idx=Int(self.idx),
        )


fn _get_index_type(address_space: AddressSpace) -> DType:
    """Returns int32 for shared/constant GPU memory, index otherwise."""
    if address_space in (
        AddressSpace.SHARED,
        AddressSpace.CONSTANT,
    ):
        return DType.int32
    else:
        return DType.int64


comptime _Splatted[T: MixedTupleLike, count: Int] = __mlir_attr[
    `#kgen.variadic.splat<`,
    T,
    `,`,
    count._mlir_value,
    `> : `,
    VariadicOf[type_of(T)],
]
