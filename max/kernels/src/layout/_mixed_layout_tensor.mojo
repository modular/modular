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

from sys import align_of, simd_width_of
from os import abort

from builtin.builtin_slice import ContiguousSlice
from builtin.device_passable import DevicePassable
from builtin.variadics import (
    Variadic,
    _MapValuesAndIdxToType,
    _MapVariadicAndIdxToType,
)
from builtin.dtype import _unsigned_integral_type_of
from gpu import thread_idx, block_dim, lane_id
from gpu.host import DeviceBuffer, HostBuffer
from utils.numerics import max_finite
from layout._fillers import BATCH_SIZE

from .swizzle import Swizzle, make_ldmatrix_swizzle

from ._mixed_layout import MixedLayout
from ._mixed_tuple import (
    ComptimeInt,
    RuntimeInt,
    Idx,
    MixedTuple,
    MixedTupleLike,
    _AllEqual,
    _IntToComptimeInt,
    mixed_tuple,
    mixed_int_tuple_to_int_tuple,
    mixed_int_tuple_to_index_list,
)


@fieldwise_init
struct MixedLayoutTensor[
    mut: Bool,
    shape_types: Variadic.TypesOfTrait[MixedTupleLike],
    stride_types: Variadic.TypesOfTrait[MixedTupleLike],
    //,
    dtype: DType,
    origin: Origin[mut=mut],
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
    linear_idx_type: DType = _get_index_type(address_space),
](Copyable, DevicePassable, Writable):
    comptime rank = Variadic.size(Self.shape_types)
    comptime SHAPE_KNOWN = MixedTuple[*Self.shape_types].ALL_DIMS_KNOWN
    comptime STRIDE_KNOWN = MixedTuple[*Self.shape_types].ALL_DIMS_KNOWN
    comptime ALL_DIMS_KNOWN = Self.SHAPE_KNOWN and Self.STRIDE_KNOWN

    var ptr: UnsafePointer[
        Scalar[Self.dtype], Self.origin, address_space = Self.address_space
    ]

    var layout: MixedLayout[
        shape_types = Self.shape_types,
        stride_types = Self.stride_types,
    ]

    comptime device_type = Self

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self.copy()

    @staticmethod
    fn get_type_name() -> String:
        return "MixedLayoutTensor"

    @staticmethod
    fn get_device_type_name() -> String:
        return Self.get_type_name()

    comptime GenericType = MixedLayoutTensor[
        shape_types = Self.shape_types,
        stride_types = Self.stride_types,
        Self.dtype,
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
        [`DeviceContext`](/mojo/std/gpu/host/device_context/DeviceContext) to
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
    ) -> Scalar[Self.dtype] where Variadic.size(
        tuple.element_types
    ) == Variadic.size(Self.shape_types):
        return self.ptr[
            self.layout[linear_idx_type = Self.linear_idx_type](tuple)
        ]

    @always_inline("nodebug")
    fn __getitem__[
        *IndexTypes: Indexer & Copyable
    ](self, tuple: Tuple[*IndexTypes]) -> Scalar[
        Self.dtype
    ] where Variadic.size(tuple.element_types) == Variadic.size(
        Self.shape_types
    ):
        var linear_tuple: MixedTuple[
            *_Splatted[RuntimeInt[Self.linear_idx_type], Self.rank]
        ]
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(linear_tuple)
        )

        @parameter
        for i in range(Variadic.size(tuple.element_types)):
            UnsafePointer(to=linear_tuple[i]).init_pointee_copy(
                rebind[type_of(linear_tuple).element_types[i]](
                    RuntimeInt[Self.linear_idx_type](index(tuple[i]))
                )
            )
        return self.ptr[
            self.layout[linear_idx_type = Self.linear_idx_type](linear_tuple)
        ]

    @always_inline("nodebug")
    fn __setitem__(
        self, tuple: MixedTuple, value: Scalar[Self.dtype]
    ) where (tuple.rank == Self.rank) & Self.mut:
        self.ptr.mut_cast[True]()[
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

        def main():
            var storage = InlineArray[Float32, 2 * 3](uninitialized=True)
            var tensor = MixedLayoutTensor(storage, row_major[2, 3]()).fill(1.0)
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

    @always_inline("nodebug")
    fn tile[
        *tile_sizes: Int
    ](self, coordinates: MixedTuple) -> MixedLayoutTensor[
        shape_types = _IntToComptimeInt[*tile_sizes],
        stride_types = Self.stride_types,
        dtype = Self.dtype,
        origin = Self.origin,
        address_space = Self.address_space,
        linear_idx_type = Self.linear_idx_type,
    ]:
        return _tile(self, mixed_tuple[*tile_sizes](), coordinates)

    @always_inline("nodebug")
    fn distribute[
        thread_layout: MixedLayout,
    ](self, thread_id: Int) -> MixedLayoutTensor[
        shape_types = _Divide[Self.shape_types, thread_layout.shape_types],
        stride_types = _Multiply[Self.stride_types, thread_layout.shape_types],
        dtype = Self.dtype,
        origin = Self.origin,
        address_space = Self.address_space,
        linear_idx_type = Self.linear_idx_type,
    ] where Self.ALL_DIMS_KNOWN:
        return _distribute[thread_layout](self, thread_id)

    @always_inline
    fn fill[
        *,
        use_runtime_layout: Bool = (
            not Self.ALL_DIMS_KNOWN
            or MixedTuple[*Self.shape_types].STATIC_PRODUCT > BATCH_SIZE
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
        from layout import Layout, LayoutTensor

        def main():
            var storage = InlineArray[Float32, 3 * 4](uninitialized=True)
            var tensor = LayoutTensor[
                DType.float32,
                Layout([3, 4]),
            ](storage).fill(0.0)
            print(tensor)
        ```

        If not using method chaining, you can either reassign the result to the
        tensor variable, or assign the result to the discard pattern (`_`) to
        avoid warnings about an unused value:

        ```mojo
        tensor = tensor.fill(0.0)
        # or
        _ = tensor.fill(0.0)
        ```
        """

        @parameter
        if not use_runtime_layout:
            comptime num_elements = MixedTuple[*Self.shape_types].STATIC_PRODUCT

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
        return self.copy()

    @always_inline("nodebug")
    fn dim[i: Int](self) -> Scalar[Self.linear_idx_type]:
        return Scalar[Self.linear_idx_type](self.layout.shape[i].value())

    @always_inline
    fn slice[
        *slices: ContiguousSlice
    ](self) -> MixedLayoutTensor[
        shape_types = _Slice[slices, Self.shape_types],
        stride_types = Self.stride_types,
        Self.dtype,
        Self.origin,
        address_space = Self.address_space,
        linear_idx_type = Self.linear_idx_type,
    ] where (Variadic.size(slices) == Self.rank) & Self.ALL_DIMS_KNOWN:
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

        For a 4x4 tensor `t` with values:

        ```
        [1 2 3 4]
        [5 6 7 8]
        [9 10 11 12]
        [13 14 15 16]
        ```

        ```mojo
        t.slice[Slice(1, 3), Slice(0, 2)]()
        ```

        will extract:

        ```
        [5 6]
        [9 10]
        ```

        For a 3D tensor, you can slice all three dimensions:

        ```mojo
        tensor_3d.slice[Slice(0, 2), Slice(1, 3), Slice(0, 4)]()
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
            var stride_i = self.layout.stride[i].value()
            offset += slice_start * stride_i

        # Build new shape tuple with runtime types
        # Even though slice bounds are compile-time known, we use RuntimeInt
        # because we can't change ComptimeInt[4] to ComptimeInt[2] in the type system
        comptime NewShapeTypes = _Slice[slices, Self.shape_types]
        var new_shape: MixedTuple[*NewShapeTypes]
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(new_shape)
        )

        @parameter
        for i in range(Self.rank):
            comptime slice_i = slices[i]
            comptime slice_start = slice_i.start.or_else(0)

            var shape_ptr = UnsafePointer(to=new_shape[i])
            comptime NewShapeType = NewShapeTypes[i]

            shape_ptr.init_pointee_copy(
                rebind[NewShapeType](ComptimeInt[NewShapeType.STATIC_VALUE]())
            )

        # Strides remain unchanged
        var new_layout = MixedLayout(new_shape^, self.layout.stride)

        return MixedLayoutTensor[
            shape_types=NewShapeTypes,
            stride_types = Self.stride_types,
            Self.dtype,
            Self.origin,
            address_space = Self.address_space,
            linear_idx_type = Self.linear_idx_type,
        ](self.ptr + offset, new_layout^)

    # ===------------------------------------------------------------------=== #
    # Vectorization
    # ===------------------------------------------------------------------=== #

    comptime VectorizedType[*vector_shape: Int] = MixedLayoutTensor[
        shape_types = _CeilDiv[
            Self.shape_types, _IntToComptimeInt[*vector_shape]
        ],
        stride_types = _Multiply[
            Self.stride_types, _IntToComptimeInt[*vector_shape]
        ],
        dtype = Self.dtype,
        origin = Self.origin,
        address_space = Self.address_space,
        linear_idx_type = Self.linear_idx_type,
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
    ](self) -> Self.VectorizedType[*vector_shape] where Self.ALL_DIMS_KNOWN:
        """Reshape a tensor into a vectorized form for efficient SIMD operations.

        This method transforms the tensor's logical layout to enable efficient
        vectorized processing, treating blocks of elements as vector units. The
        transformation is particularly useful for SIMD (Single Instruction
        Multiple Data) operations and hardware acceleration.

        Unlike `LayoutTensor.vectorize`, this simplified implementation does not
        track the vector shape in an `element_layout`. Users must manually handle
        loading SIMD vectors from each logical element position.

        Parameters:
            vector_shape: The dimensions of each vector unit along each axis of
                the tensor. For example, in a 2D tensor, `vectorize[4, 4]` treats
                4x4 blocks as vector units.

        Returns:
            A view of the tensor with a vectorized layout, where each element in
            the resulting tensor represents the start of a vector block from the
            original tensor.

        Constraints:
            All dimensions must be statically known (`ALL_DIMS_KNOWN`).

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
        return _vectorize(self, mixed_tuple[*vector_shape]())

    @always_inline("nodebug")
    fn vectorize(self) -> Self.SIMDVectorizedType where Self.ALL_DIMS_KNOWN:
        """Return a SIMD-width vectorized view of this tensor.

        This is a convenience method that vectorizes along the last dimension
        by the SIMD width for the tensor's dtype.

        Returns:
            A `Self.VectorizedType[1, simd_width_of[Self.dtype]()]` view whose
            last dimension stride equals the SIMD width for the tensor's dtype.
        """
        return self.vectorize[1, simd_width_of[Self.dtype]()]()

    @always_inline("nodebug")
    fn to_layout_tensor(
        self,
        out result: LayoutTensor[
            Self.dtype,
            Layout(
                mixed_int_tuple_to_int_tuple[*Self.shape_types](),
                mixed_int_tuple_to_int_tuple[*Self.stride_types](),
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
                mixed_int_tuple_to_index_list(self.layout.shape),
                mixed_int_tuple_to_index_list(self.layout.stride),
            ),
        }


@always_inline("nodebug")
fn stack_allocation[
    shape_types: Variadic.TypesOfTrait[MixedTupleLike],
    stride_types: Variadic.TypesOfTrait[MixedTupleLike],
    //,
    dtype: DType,
    address_space: AddressSpace = AddressSpace.GENERIC,
](var layout: MixedLayout[shape_types, stride_types]) -> MixedLayoutTensor[
    shape_types=shape_types,
    stride_types=stride_types,
    dtype,
    MutExternalOrigin,
    address_space=address_space,
] where layout.ALL_DIMS_KNOWN:
    return MixedLayoutTensor[
        shape_types=shape_types,
        stride_types=stride_types,
        dtype,
        MutExternalOrigin,
        address_space=address_space,
    ](
        std.memory.stack_allocation[
            MixedTuple[*shape_types].STATIC_PRODUCT,
            Scalar[dtype],
            address_space=address_space,
        ](),
        layout^,
    )


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


@always_inline("nodebug")
fn _distribute[
    thread_layout: MixedLayout,
](
    data_layout_tensor: MixedLayoutTensor,
    thread_id: Int,
) -> MixedLayoutTensor[
    shape_types = _Divide[
        data_layout_tensor.shape_types, thread_layout.shape_types
    ],
    stride_types = _Multiply[
        data_layout_tensor.stride_types, thread_layout.shape_types
    ],
    data_layout_tensor.dtype,
    data_layout_tensor.origin,
    address_space = data_layout_tensor.address_space,
    linear_idx_type = data_layout_tensor.linear_idx_type,
]:
    """A simplified implementation of LayoutTensor.distribute on MixedLayoutTensor.
    """

    var offset: UInt = 0

    @parameter
    for i in range(Variadic.size(thread_layout.stride_types)):
        comptime stride_i = thread_layout.stride_types[i].STATIC_VALUE
        comptime shape_i = thread_layout.shape_types[i].STATIC_VALUE
        var thread_coord_i = (thread_id // stride_i) % shape_i
        offset += UInt(
            thread_coord_i * Int(data_layout_tensor.layout.stride[i].value())
        )

    comptime ShapeType = MixedTuple[
        *_Divide[data_layout_tensor.shape_types, thread_layout.shape_types]
    ]
    comptime StrideType = MixedTuple[
        *_Multiply[data_layout_tensor.stride_types, thread_layout.shape_types]
    ]
    # Since the thread layout and tensor layout have ALL_DIMS_KNOWN this is safe
    __comptime_assert ShapeType.ALL_DIMS_KNOWN
    var shape = ShapeType()
    __comptime_assert StrideType.ALL_DIMS_KNOWN
    var stride = StrideType()

    var layout = MixedLayout(shape^, stride^)

    return MixedLayoutTensor[
        data_layout_tensor.dtype,
        data_layout_tensor.origin,
        address_space = data_layout_tensor.address_space,
        linear_idx_type = data_layout_tensor.linear_idx_type,
    ](
        UnsafePointer(to=data_layout_tensor.ptr[offset]),
        layout^,
    )


@always_inline("nodebug")
fn _tile[
    dtype: DType,
    shape_types: Variadic.TypesOfTrait[MixedTupleLike],
    stride_types: Variadic.TypesOfTrait[MixedTupleLike],
    coord_types: Variadic.TypesOfTrait[MixedTupleLike],
    tile_shape_types: Variadic.TypesOfTrait[MixedTupleLike],
    //,
](
    data_layout_tensor: MixedLayoutTensor[
        shape_types=shape_types, stride_types=stride_types, dtype, ...
    ],
    tile_shape: MixedTuple[*tile_shape_types],
    tile_coords: MixedTuple[*coord_types],
) -> MixedLayoutTensor[
    shape_types=tile_shape_types,
    stride_types=stride_types,
    dtype,
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
        shape_types=tile_shape_types,
        stride_types=stride_types,
        dtype,
        data_layout_tensor.origin,
        address_space = data_layout_tensor.address_space,
        linear_idx_type = data_layout_tensor.linear_idx_type,
    ](
        UnsafePointer(to=data_layout_tensor.ptr[offset]),
        tile_layout,
    )


@always_inline("nodebug")
fn _vectorize[
    dtype: DType,
    shape_types: Variadic.TypesOfTrait[MixedTupleLike],
    stride_types: Variadic.TypesOfTrait[MixedTupleLike],
    vector_shape_types: Variadic.TypesOfTrait[MixedTupleLike],
    //,
](
    data_layout_tensor: MixedLayoutTensor[
        shape_types=shape_types, stride_types=stride_types, dtype, ...
    ],
    vector_shape: MixedTuple[*vector_shape_types],
) -> MixedLayoutTensor[
    shape_types = _CeilDiv[shape_types, vector_shape_types],
    stride_types = _Multiply[stride_types, vector_shape_types],
    dtype,
    data_layout_tensor.origin,
    address_space = data_layout_tensor.address_space,
    linear_idx_type = data_layout_tensor.linear_idx_type,
]:
    """Create a vectorized view of a MixedLayoutTensor.

    This function creates a new view where the shape is divided by the vector
    shape (ceiling division) and strides are multiplied by the vector shape.
    This effectively groups elements into vector-sized blocks.

    Parameters:
        dtype: Data type of the tensor elements.
        shape_types: Shape types of the source tensor.
        stride_types: Stride types of the source tensor.
        vector_shape_types: Types of the vector shape dimensions.

    Args:
        data_layout_tensor: The source tensor to vectorize.
        vector_shape: The shape of each vector unit as a MixedTuple.

    Returns:
        A MixedLayoutTensor representing a vectorized view. Each logical element
        in the result corresponds to a vector block in the original tensor.
    """
    comptime NewShapeTypes = _CeilDiv[shape_types, vector_shape_types]
    comptime NewStrideTypes = _Multiply[stride_types, vector_shape_types]

    # Since ALL_DIMS_KNOWN is required, we can use compile-time values directly
    __comptime_assert MixedTuple[*NewShapeTypes].ALL_DIMS_KNOWN
    __comptime_assert MixedTuple[*NewStrideTypes].ALL_DIMS_KNOWN
    var new_shape = MixedTuple[*NewShapeTypes]()
    var new_stride = MixedTuple[*NewStrideTypes]()

    var new_layout = MixedLayout(new_shape^, new_stride^)

    return MixedLayoutTensor[
        shape_types=NewShapeTypes,
        stride_types=NewStrideTypes,
        dtype,
        data_layout_tensor.origin,
        address_space = data_layout_tensor.address_space,
        linear_idx_type = data_layout_tensor.linear_idx_type,
    ](data_layout_tensor.ptr, new_layout^)


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
    Variadic.TypesOfTrait[type_of(T)],
]


comptime _MultiplyMapper[
    Rhs: Variadic.TypesOfTrait[MixedTupleLike],
    element_types: Variadic.TypesOfTrait[MixedTupleLike],
    idx: Int,
] = ComptimeInt[element_types[idx].STATIC_VALUE * Rhs[idx].STATIC_VALUE]


comptime _Multiply[
    Lhs: Variadic.TypesOfTrait[MixedTupleLike],
    Rhs: Variadic.TypesOfTrait[MixedTupleLike],
] = _MapVariadicAndIdxToType[
    To=MixedTupleLike,
    VariadicType=Lhs,
    Mapper = _MultiplyMapper[Rhs=Rhs],
]

comptime _DivideMapper[
    Rhs: Variadic.TypesOfTrait[MixedTupleLike],
    element_types: Variadic.TypesOfTrait[MixedTupleLike],
    idx: Int,
] = ComptimeInt[element_types[idx].STATIC_VALUE // Rhs[idx].STATIC_VALUE]


comptime _Divide[
    Lhs: Variadic.TypesOfTrait[MixedTupleLike],
    Rhs: Variadic.TypesOfTrait[MixedTupleLike],
] = _MapVariadicAndIdxToType[
    To=MixedTupleLike,
    VariadicType=Lhs,
    Mapper = _DivideMapper[Rhs=Rhs],
]

comptime _CeilDivMapper[
    Rhs: Variadic.TypesOfTrait[MixedTupleLike],
    element_types: Variadic.TypesOfTrait[MixedTupleLike],
    idx: Int,
] = ComptimeInt[
    (element_types[idx].STATIC_VALUE + Rhs[idx].STATIC_VALUE - 1)
    // Rhs[idx].STATIC_VALUE
]


comptime _CeilDiv[
    Lhs: Variadic.TypesOfTrait[MixedTupleLike],
    Rhs: Variadic.TypesOfTrait[MixedTupleLike],
] = _MapVariadicAndIdxToType[
    To=MixedTupleLike,
    VariadicType=Lhs,
    Mapper = _CeilDivMapper[Rhs=Rhs],
]


comptime _ToRuntimeMapper[
    dtype: DType,
    element_types: Variadic.TypesOfTrait[MixedTupleLike],
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
    element_types: Variadic.TypesOfTrait[MixedTupleLike], dtype: DType
] = _MapVariadicAndIdxToType[
    To=MixedTupleLike,
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
    From: Variadic.TypesOfTrait[MixedTupleLike],
    idx: Int,
] = ComptimeInt[
    slices[idx].end.or_else(From[idx].STATIC_VALUE)
    - slices[idx].start.or_else(0)
]

comptime _Slice[
    slices: Variadic.ValuesOfType[ContiguousSlice],
    element_types: Variadic.TypesOfTrait[MixedTupleLike],
] = _MapVariadicAndIdxToType[
    To=MixedTupleLike,
    VariadicType=element_types,
    Mapper = _SliceMapper[slices=slices],
]
