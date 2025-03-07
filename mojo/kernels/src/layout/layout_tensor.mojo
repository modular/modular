# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import Optional, OptionalReg
from math import align_up, ceildiv
from sys import (
    alignof,
    bitwidthof,
    is_nvidia_gpu,
    is_amd_gpu,
    prefetch,
    simdwidthof,
    sizeof,
)

from sys.intrinsics import PrefetchOptions
from algorithm import vectorize
from bit import log2_floor
from gpu.host import DeviceBuffer
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.id import block_idx, thread_idx, lane_id
from gpu.memory import CacheEviction, Fill, async_copy
import gpu.memory as gpu_memory
from layout.element import Element, MemoryElement
from layout.tma_async import _tma_desc_tile_layout
from memory import UnsafePointer, memcpy, memset_zero, stack_allocation
from memory.pointer import AddressSpace, _GPUAddressSpace

from utils import IndexList, StaticTuple
from utils.index import Index
from utils.numerics import max_finite

from .fillers import arange
from .int_tuple import depth, fill_like, flatten, idx2crd, product
from .layout import *
from .runtime_layout import RuntimeLayout
from .runtime_layout import coalesce as runtime_coalesce
from .runtime_layout import make_layout as make_runtime_layout
from .runtime_tuple import RuntimeTuple
from .swizzle import Swizzle, make_ldmatrix_swizzle
from gpu.intrinsics import buffer_load, buffer_store
from ._utils import get_amd_buffer_descriptor


fn _compute_distribute_layout[
    data_layout: Layout,
    threads_layout: Layout,
    axis: OptionalReg[Int] = None,
]() -> Layout:
    """Computes a layout for distributing threads across data.

    Distributes thread_layout into data_layout. If axis is provided, distributes
    into threads_layout projected into this axis.

    Parameters:
        data_layout: The layout of the data to be distributed.
        threads_layout: The layout of the threads.
        axis: Optional axis for projection-based distribution.

    Returns:
        A layout representing the distribution of threads across data.
    """
    var thread_tile = LayoutList()

    @parameter
    if axis:
        return zipped_divide(
            data_layout, Layout(threads_layout.shape[axis.value()])
        )

    else:
        for dim in threads_layout.shape:
            thread_tile.append(Layout(dim))

        return zipped_divide(data_layout, thread_tile)


fn _project_on_axis[
    axis: Int, submode_axis: OptionalReg[Int] = None
](t: IntTuple) -> IntTuple:
    """Projects an IntTuple onto a specific axis.

    Creates an IntTuple with zeros in all positions except the specified axis,
    which contains ones. When submode_axis is provided, the projection happens
    only on that specific submode.

    Parameters:
        axis: The axis to project onto.
        submode_axis: Optional submode axis for nested projection.

    Args:
        t: The input IntTuple to project.

    Returns:
        A projected IntTuple.
    """
    if not submode_axis:
        var p_t = fill_like(t, 0)
        # p_t[axis] = fill_like(t[axis], 1)
        p_t = p_t.replace_entry(axis, fill_like(t[axis], 1))
        return p_t
    var p_t = fill_like(t, 1)
    # p_t[axis] = fill_like(t[axis], 0)
    # p_t[axis][submode_axis.value()] = 1
    var filled = fill_like(t[axis], 0)
    filled = filled.replace_entry(submode_axis.value(), 1)
    p_t = p_t.replace_entry(axis, filled)
    return p_t


fn _get_index_type(address_space: AddressSpace) -> DType:
    """Determines the appropriate index type based on address space.

    Returns `DType.int32` for shared or constant GPU memory, and `DType.index` otherwise.

    Args:
        address_space: The address space to determine the index type for.

    Returns:
        The appropriate `DType` for indexing in the given address space.
    """
    if address_space in (
        _GPUAddressSpace.SHARED,
        _GPUAddressSpace.CONSTANT,
    ):
        return DType.int32
    else:
        return DType.index


fn _get_index_type(layout: Layout, address_space: AddressSpace) -> DType:
    """Determines the appropriate index type based on layout and address space.

    Uses `DType.int32` if the layout size fits within int32 range, otherwise uses the
    address space's default index type.

    Args:
        layout: The layout to determine the index type for.
        address_space: The address space to determine the index type for.

    Returns:
        The appropriate `DType` for indexing with the given layout and address space.
    """
    if layout.all_dims_known() and layout.cosize() < Int(
        max_finite[DType.int32]()
    ):
        return DType.int32
    else:
        return _get_index_type(address_space)


fn _get_unsigned_type(layout: Layout, address_space: AddressSpace) -> DType:
    """Determines the appropriate unsigned index type for a layout and address space.

    Uses `DType.uint32` if the layout size fits within uint32 range or if the index type
    is int32, otherwise uses the unsigned index type.

    Args:
        layout: The layout to determine the unsigned type for.
        address_space: The address space to determine the unsigned type for.

    Returns:
        The appropriate unsigned DType for the given layout and address space.
    """
    if layout.all_dims_known() and layout.cosize() < Int(
        max_finite[DType.uint32]()
    ):
        return DType.uint32
    else:
        var dtype = _get_index_type(address_space)
        return DType.uint32 if dtype is DType.int32 else DType.index


alias _swizzle_signature = fn[type: DType] (Scalar[type]) -> Scalar[type]


fn _get_len[*values: Int]() -> Int:
    """Returns the number of variadic integer parameters.

    This utility function counts the number of integer parameters passed to a
    variadic template function. It's used internally for handling variable
    numbers of dimensions, tile sizes, or other integer parameters.

    Parameters:
        values: Variadic integer parameters to count.

    Returns:
        The count of variadic integer parameters.
    """
    return __mlir_op.`pop.variadic.size`(values)


fn _get_slice_size(layout: Layout, slc: Slice, dim: Int) -> Int:
    """Calculates the size of a slice in a specific layout dimension.

    Computes the number of elements in a slice for a given dimension of the layout.
    This function handles the conversion between slice notation and actual element counts.

    Args:
        layout: The layout containing the dimension information.
        slc: The slice specification (start:end:step).
        dim: The dimension index to slice.

    Returns:
        The number of elements in the slice for the specified dimension.
    """
    var start: Int
    var end: Int
    start, end, _ = slc.indices(Int(layout.shape[dim]))
    return end - start


fn _not_in_tuple[n: Int, size: Int, tuple: IndexList[size]]() -> Bool:
    """Checks if a value is *not* present in an `IndexList`.

    This utility function searches through an `IndexList` to determine if a specific
    value is absent. Used for dimension validation and filtering operations.

    Parameters:
        n: The value to check for in the `IndexList`.
        size: The size of the `IndexList`.
        tuple: The `IndexList` to search in.

    Returns:
        True if the value is not found in the `IndexList`, False if it is present.
    """

    @parameter
    for i in range(size):

        @parameter
        if tuple[i] == n:
            return False
    return True


fn _tile_is_masked[layout: Layout, *tile_sizes: Int]() -> Bool:
    """Determines if a tiled layout requires masked access.

    When tiling a tensor, this function checks if any dimension of the layout is not
    evenly divisible by its corresponding tile size. If any dimension requires padding,
    masked access is needed to prevent out-of-bounds memory accesses.

    Parameters:
        layout: The layout to check for divisibility.
        tile_sizes: The tile sizes for each dimension of the layout.

    Returns:
        True if masked access is required (any dimension not evenly divisible),
        False if all dimensions are perfectly divisible by their tile sizes.
    """
    if not layout.all_dims_known():
        return True

    @parameter
    for axis in range(layout.rank()):
        alias dim = product(layout.shape[axis])
        if dim % tile_sizes[axis] != 0:
            return True
    return False


fn _distribute_is_masked[
    layout: Layout, threads_layout: Layout, axis: OptionalReg[Int] = None
]() -> Bool:
    """Determines if a distributed layout requires masked access.

    When distributing computation across threads, this function checks if the layout's
    dimensions are evenly divisible by the corresponding thread dimensions. Masked access
    is required when dimensions don't divide evenly to prevent out-of-bounds accesses.

    Parameters:
        layout: The layout to distribute across threads.
        threads_layout: The layout representing thread organization.
        axis: Optional axis for projection-based distribution. When specified,
              distribution occurs along this axis only.

    Returns:
        True if masked access is required (dimensions not evenly divisible),
        False if all dimensions are perfectly divisible by thread dimensions.
    """

    # TODO: relax this constraint
    @parameter
    if depth(threads_layout.shape) > 1:
        return False

    @parameter
    if axis:
        return False

    @parameter
    if not layout.all_dims_known():
        return True

    @parameter
    for i in range(layout.rank()):
        alias layout_dim = Int(layout.shape[i])
        alias thread_dim = Int(threads_layout.shape[i])

        @parameter
        if layout_dim % thread_dim != 0:
            return True

    return False


@register_passable("trivial")
struct LayoutTensor[
    mut: Bool, //,
    dtype: DType,
    layout: Layout,
    origin: Origin[mut],
    /,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
    element_layout: Layout = Layout(1, 1),
    layout_bitwidth: Int = bitwidthof[_get_index_type(address_space)](),
    masked: Bool = False,
    alignment: Int = alignof[dtype](),
](CollectionElement, CollectionElementNew, Stringable, Writable):
    """A high-performance tensor with explicit memory layout and hardware-optimized access patterns.

    LayoutTensor provides a powerful abstraction for multi-dimensional data with precise control
    over memory organization. It supports various memory layouts (row-major, column-major, tiled),
    hardware-specific optimizations, and efficient parallel access patterns.

    Parameters:
        mut: The inferred mutability of the underlying pointer.
        dtype: The data type of the underlying pointer.
        layout: The memory layout of the Tensor.
        origin: The origin of the underlying pointer.
        address_space: The address space of the underlying pointer.
        element_layout: The memory layout of each element in the Tensor.
        layout_bitwidth: The bitwidth of each dimension of runtime layout.
        masked: If true the tensor is masked and runtime layouts determine the shape.
        alignment: Alignment of the data pointer.

    Example:
    ```mojo
    from layout import Layout, LayoutTensor

    var storage = InlineArray[Scalar[DType.float32], 5 * 4](uninitialized = True)
    var tensor_5x4 = LayoutTensor[DType.float32, Layout.row_major(5,4)](storage)
    ```
    """

    alias rank = layout.rank()

    alias index_type: DType = _get_index_type(layout, address_space)
    alias uint_type = Scalar[_get_unsigned_type(layout, address_space)]

    var ptr: UnsafePointer[
        Scalar[dtype],
        address_space=address_space,
        alignment=alignment,
        mut=mut,
        origin=origin,
    ]

    var runtime_layout: RuntimeLayout[layout, bitwidth=layout_bitwidth]

    var runtime_element_layout: RuntimeLayout[element_layout]

    alias element_size = element_layout.size()
    alias element_type = SIMD[dtype, Self.element_size]

    # ===------------------------------------------------------------------=== #
    # Life cycle methods
    # ===------------------------------------------------------------------=== #
    @always_inline
    @implicit
    fn __init__(
        out self,
        span: Span[
            Scalar[dtype],
            origin,
            address_space=address_space, **_,
        ],
    ):
        """Create a LayoutTensor with an UnsafePointer. Expect layout to be
        fully static.

        Args:
            span: The UnsafePointer pointing to the underlying data.
        """

        constrained[layout.all_dims_known(), "Layout must be fully static"]()
        self.ptr = span.unsafe_ptr()
        self.runtime_layout = RuntimeLayout[
            layout, bitwidth = Self.layout_bitwidth
        ]()
        self.runtime_element_layout = RuntimeLayout[element_layout]()

    @always_inline
    fn __init__(
        mut self,
        span: Span[
            Scalar[dtype],
            origin,
            address_space=address_space, **_,
        ],
        runtime_layout: RuntimeLayout[layout, **_],
    ):
        """Create a LayoutTensor with an UnsafePointer. Expect element layout
        to be fully static.

        Args:
            span: The UnsafePointer pointing to the underlying data.
            runtime_layout: The runtime layout of the LayoutTensor.
        """

        constrained[
            element_layout.all_dims_known(), "Layout must be fully static"
        ]()

        constrained[
            runtime_layout.bitwidth == Self.layout_bitwidth,
            "Mismatch of bitwidth for RuntimeLayout and LayoutTensor.",
        ]()

        self.ptr = span.unsafe_ptr()
        self.runtime_layout = rebind[
            RuntimeLayout[layout, bitwidth = Self.layout_bitwidth]
        ](runtime_layout)
        self.runtime_element_layout = RuntimeLayout[element_layout]()

    @always_inline
    fn __init__(
        mut self,
        span: Span[
            Scalar[dtype],
            origin,
            address_space=address_space, **_,
        ],
        runtime_layout: RuntimeLayout[layout, bitwidth = Self.layout_bitwidth],
        element_runtime_layout: RuntimeLayout[element_layout],
    ):
        """Create a LayoutTensor with an UnsafePointer, a runtime layout of the
        Tensor, the runtime layout of each element.

        Args:
            span: The UnsafePointer pointing to the underlying data.
            runtime_layout: The runtime layout of the LayoutTensor.
            element_runtime_layout: The runtime layout of each element.
        """
        self.ptr = span.unsafe_ptr()
        self.runtime_layout = runtime_layout
        self.runtime_element_layout = element_runtime_layout

    @always_inline
    @implicit
    fn __init__(
        out self,
        ptr: UnsafePointer[
            Scalar[dtype],
            address_space=address_space,
            mut=mut,
            origin=origin, **_,
        ],
    ):
        """Create a LayoutTensor with an UnsafePointer. Expect layout to be
        fully static.

        Args:
            ptr: The UnsafePointer pointing to the underlying data.
        """

        constrained[layout.all_dims_known(), "Layout must be fully static"]()
        self.ptr = ptr
        self.runtime_layout = RuntimeLayout[
            layout, bitwidth = Self.layout_bitwidth
        ]()
        self.runtime_element_layout = RuntimeLayout[element_layout]()

    @always_inline
    fn __init__(
        mut self,
        ptr: UnsafePointer[
            Scalar[dtype],
            address_space=address_space,
            mut=mut,
            origin=origin, **_,
        ],
        runtime_layout: RuntimeLayout[layout, **_],
    ):
        """Create a LayoutTensor with an UnsafePointer. Expect element layout
        to be fully static.

        Args:
            ptr: The UnsafePointer pointing to the underlying data.
            runtime_layout: The runtime layout of the LayoutTensor.
        """

        constrained[
            element_layout.all_dims_known(), "Layout must be fully static"
        ]()

        constrained[
            runtime_layout.bitwidth == Self.layout_bitwidth,
            "Mismatch of bitwidth for RuntimeLayout and LayoutTensor.",
        ]()

        self.ptr = ptr
        self.runtime_layout = rebind[
            RuntimeLayout[layout, bitwidth = Self.layout_bitwidth]
        ](runtime_layout)
        self.runtime_element_layout = RuntimeLayout[element_layout]()

    @always_inline
    fn __init__(
        mut self,
        ptr: UnsafePointer[
            Scalar[dtype],
            address_space=address_space,
            mut=mut,
            origin=origin, **_,
        ],
        runtime_layout: RuntimeLayout[layout, bitwidth = Self.layout_bitwidth],
        element_runtime_layout: RuntimeLayout[element_layout],
    ):
        """Create a LayoutTensor with an UnsafePointer, a runtime layout of the
        Tensor, the runtime layout of each element.

        Args:
            ptr: The UnsafePointer pointing to the underlying data.
            runtime_layout: The runtime layout of the LayoutTensor.
            element_runtime_layout: The runtime layout of each element.
        """
        self.ptr = ptr
        self.runtime_layout = runtime_layout
        self.runtime_element_layout = element_runtime_layout

    @always_inline
    @implicit
    fn __init__(
        out self,
        device_buffer: DeviceBuffer[
            dtype,
            address_space=address_space,
            mut=mut,
            origin=origin, **_,
        ],
    ):
        """Create a LayoutTensor from a `DeviceBuffer`. The layout must have
        statically known dimensions.

        ```mojo
        from gpu.host import DeviceContext, DeviceBuffer
        from layout import Layout, LayoutTensor

        alias dtype = DType.float32

        var ctx = DeviceContext()
        var dev_buf = ctx.enqueue_create_buffer[dtype](8)

        alias layout = Layout.row_major(4, 4)
        var tensor = LayoutTensor[dtype, layout](dev_buf)
        ```

        Args:
            device_buffer: Contains the underlying data to point to.
        """
        self = Self(device_buffer.unsafe_ptr())

    @always_inline
    fn __init__(
        mut self,
        device_buffer: DeviceBuffer[
            dtype,
            address_space=address_space,
            mut=mut,
            origin=origin, **_,
        ],
        runtime_layout: RuntimeLayout[layout, **_],
    ):
        """Create a LayoutTensor from a `DeviceBuffer`. The layout must have
        statically known dimensions.

        Args:
            device_buffer: The DeviceBuffer containing to the underlying data.
            runtime_layout: The runtime layout of the LayoutTensor.
        """
        self = Self(device_buffer.unsafe_ptr(), runtime_layout)

    @always_inline
    fn __init__(
        mut self,
        device_buffer: DeviceBuffer[
            dtype,
            address_space=address_space,
            mut=mut,
            origin=origin, **_,
        ],
        runtime_layout: RuntimeLayout[layout, bitwidth = Self.layout_bitwidth],
        element_runtime_layout: RuntimeLayout[element_layout],
    ):
        """Create a LayoutTensor from a DeviceBuffer, a runtime layout of the
        Tensor, and the runtime layout of each element.

        Args:
            device_buffer: The DeviceBuffer containing to the underlying data.
            runtime_layout: The runtime layout of the LayoutTensor.
            element_runtime_layout: The runtime layout of each element.
        """
        self = Self(
            device_buffer.unsafe_ptr(), runtime_layout, element_runtime_layout
        )

    fn copy(self) -> Self:
        """Explicitly copy the other LayoutTensor.

        Returns:
            A copy of the value.
        """
        return self

    @always_inline
    fn bitcast[
        new_type: DType,
        /,
        address_space: AddressSpace = Self.address_space,
        element_layout: Layout = Self.element_layout,
    ](
        self,
        out result: LayoutTensor[
            new_type,
            layout,
            origin,
            address_space=address_space,
            element_layout=element_layout,
            masked=masked,
        ],
    ):
        """Bitcast the underlying pointer to a new data type.

        Parameters:
            new_type: The new data type it is casting to.
            address_space: The address space of the returned LayoutTensor.
            element_layout: The element layout of the returned LayoutTensor.
        """
        return __type_of(result)(
            self.ptr.bitcast[Scalar[new_type]]().address_space_cast[
                address_space
            ](),
            rebind[RuntimeLayout[layout, bitwidth = result.layout_bitwidth]](
                self.runtime_layout
            ),
        )

    @always_inline("nodebug")
    fn origin_cast[
        mut: Bool = Self.mut,
        origin: Origin[mut] = Origin[mut].cast_from[Self.origin].result,
    ](
        self,
        out result: LayoutTensor[
            dtype,
            layout,
            origin,
            address_space=address_space,
            element_layout=element_layout,
            layout_bitwidth=layout_bitwidth,
            masked=masked,
            alignment=alignment,
        ],
    ):
        """Changes the origin or mutability of a pointer.

        Parameters:
            mut: Whether the origin is mutable.
            origin: Origin of the destination pointer.

        Returns:
            A new UnsafePointer object with the same type and the same address,
            as the original UnsafePointer and the new specified mutability and origin.
        """
        result = __type_of(result)(
            self.ptr.origin_cast[mut, origin](),
            self.runtime_layout,
            self.runtime_element_layout,
        )

    @always_inline
    fn get_immutable(
        self,
    ) -> LayoutTensor[
        dtype,
        layout,
        ImmutableOrigin.cast_from[origin].result,
        address_space=address_space,
        element_layout=element_layout,
        layout_bitwidth=layout_bitwidth,
        masked=masked,
        alignment=alignment,
    ]:
        """
        Return an immutable version of this tensor.

        Returns:
            A LayoutTensor covering the same elements, but without mutability.
        """
        return LayoutTensor[
            dtype,
            layout,
            ImmutableOrigin.cast_from[origin].result,
            address_space=address_space,
            element_layout=element_layout,
            layout_bitwidth=layout_bitwidth,
            masked=masked,
            alignment=alignment,
        ](self.ptr, self.runtime_layout, self.runtime_element_layout)

    @always_inline
    fn _offset(self, m: Int, n: Int) -> Int:
        return Self.stride[0]() * m + Self.stride[1]() * n

    @always_inline
    fn _elementwise_unary[
        func: fn (Self.element_type) capturing -> (Self.element_type),
    ](self) -> Self:
        constrained[
            layout.all_dims_known(),
            (
                "__elmentwise_unary must operates on tensors of statically know"
                " layouts"
            ),
        ]()

        @parameter
        for i in range(self.layout.size()):
            alias idx = self.layout(i)
            self.ptr.store(
                idx, func(self.ptr.load[width = Self.element_size](idx))
            )
        return self

    @always_inline
    fn _elementwise_binary_with_broadcast[
        func: fn (Self.element_type, Self.element_type) capturing -> (
            Self.element_type
        ),
        other_layout: Layout,
        other_mut: Bool,
        other_origin: Origin[other_mut],
        other_masked: Bool,
        other_alignment: Int,
    ](
        self,
        other: LayoutTensor[
            dtype,
            other_layout,
            other_origin,
            address_space=address_space,
            element_layout=element_layout,
            layout_bitwidth=layout_bitwidth,
            masked=other_masked,
            alignment=other_alignment,
        ],
    ) -> Self:
        @parameter
        if Self.rank == other.rank:

            @parameter
            for axis in range(Self.rank):
                constrained[
                    other.shape[axis]() == self.shape[axis](),
                    (
                        "_elementwise_binary_with_broadcast requires shape to"
                        " be the same for tensors of the same rank"
                    ),
                ]()

        constrained[
            layout.all_dims_known(),
            (
                "_elementwise_binary_with_broadcast must operates on tensors"
                " of statically know layouts"
            ),
        ]()
        constrained[
            other.rank <= Self.rank,
            (
                "_elementwise_binary_with_broadcast must operates on tensor of"
                " equal of lower rank"
            ),
        ]()

        # TODO(KERN-812): Support numpy like broadcasting and relax rank-2
        # constrain.
        constrained[
            Self.rank == 2 or Self.rank == other.rank,
            "Only supports rank-2 tensor, or same rank",
        ]()

        @parameter
        if other.rank == 1:
            constrained[
                other.shape[0]() == self.shape[0](),
                (
                    "_elementwise_binary_with_broadcast 1d tensor operand must"
                    " have a dim that matches the tensors"
                ),
            ]()

            @parameter
            for i in range(self.layout.size()):
                alias other_size = other.layout.size()

                alias lhs_idx = self.layout(i)
                alias rhs_idx = other.layout(i % other_size)

                self.ptr.store(
                    lhs_idx,
                    func(
                        self.ptr.load[width = Self.element_size](lhs_idx),
                        other.ptr.load[width = Self.element_size](rhs_idx),
                    ),
                )
            return self

        @parameter
        for i in range(self.layout.size()):
            alias idx = self.layout(i)
            self.ptr.store(
                idx,
                func(
                    self.ptr.load[width = Self.element_size](idx),
                    other.ptr.load[width = Self.element_size](idx),
                ),
            )
        return self

    @always_inline
    fn __add__(
        self, other: Scalar[dtype]
    ) -> __type_of(self.origin_cast[True, MutableAnyOrigin]()):
        """Add the LayoutTensor with a scalar value. The scalar value will be
        broadcasted to the entire tensor.

        Args:
            other: The scalar value.
        """

        @parameter
        fn add_val(val: Self.element_type) -> Self.element_type:
            return Self.element_type(other) + val

        return self._stack_copy()._elementwise_unary[add_val]()

    @always_inline
    fn __iadd__(self, other: Scalar[dtype]):
        """Adds scalar value to the LayoutTensor. The scalar value will be
        broadcasted to the entire tensor.

        Args:
            other: The scalar value.
        """

        @parameter
        fn add_val(val: Self.element_type) -> Self.element_type:
            return Self.element_type(other) + val

        _ = self._elementwise_unary[add_val]()

    @always_inline
    fn __add__[
        other_layout: Layout
    ](
        self,
        other: LayoutTensor[
            dtype,
            other_layout,
            address_space=address_space,
            element_layout=element_layout,
            layout_bitwidth=layout_bitwidth,
        ],
    ) -> __type_of(self.origin_cast[True, MutableAnyOrigin]()):
        """Do an addition with another LayoutTensor and return the added
        tensor. Currently only support tensors of the same shape if the rank
        is the same and also tensors of rank-2.

        Parameters:
            other_layout: The layout of the other tensor.

        Args:
            other: The other tensor to be added to.
        """

        fn add_val(
            lhs: Self.element_type, rhs: Self.element_type
        ) capturing -> Self.element_type:
            return lhs + rhs

        return self._stack_copy()._elementwise_binary_with_broadcast[add_val](
            other
        )

    @always_inline
    fn __iadd__[
        other_layout: Layout
    ](
        self,
        other: LayoutTensor[
            dtype,
            other_layout,
            address_space=address_space,
            element_layout=element_layout,
            layout_bitwidth=layout_bitwidth,
        ],
    ):
        """Do an addition with another LayoutTensor in place.
        Currently only support tensors of the same shape if the rank
        is the same and also tensors of rank-2.

        Parameters:
            other_layout: The layout of the other tensor.

        Args:
            other: The other tensor to be added to.
        """

        fn add_val(
            lhs: Self.element_type, rhs: Self.element_type
        ) capturing -> Self.element_type:
            return lhs + rhs

        _ = self._elementwise_binary_with_broadcast[add_val](other)

    @always_inline
    fn __mul__(
        self, other: Scalar[dtype]
    ) -> __type_of(self.origin_cast[True, MutableAnyOrigin]()):
        """Multiply the LayoutTensor with a scalar value. The scalar value will
        be broadcasted to the entire tensor.

        Args:
            other: The scalar value.
        """

        @parameter
        fn mul_val(val: Self.element_type) -> Self.element_type:
            return Self.element_type(other) * val

        return self._stack_copy()._elementwise_unary[mul_val]()

    @always_inline
    fn __mul__[
        other_layout: Layout
    ](
        self,
        other: LayoutTensor[
            dtype,
            other_layout,
            address_space=address_space,
            element_layout=element_layout,
            layout_bitwidth=layout_bitwidth,
        ],
    ) -> __type_of(self.origin_cast[True, MutableAnyOrigin]()):
        """Perform a multiplication with another LayoutTensor and return the
        resulting tensor.

        Currently, only tensors of the same shape are supported if the ranks are
        the same. Additionally, tensors of rank-2 are supported.

        Parameters:
            other_layout: The layout of the other tensor.

        Args:
            other: The other tensor to be multiplied with.

        Returns:
            The resulting tensor after multiplication.
        """

        fn mul_val(
            lhs: Self.element_type, rhs: Self.element_type
        ) capturing -> Self.element_type:
            return lhs * rhs

        return self._stack_copy()._elementwise_binary_with_broadcast[mul_val](
            other
        )

    @always_inline
    fn __imul__(self, other: Scalar[dtype]):
        """Multiply the LayoutTensor with a scalar value inplace.
        The scalar value will be broadcasted to the entire tensor.

        Args:
            other: The scalar value.
        """

        @parameter
        fn mul_val(val: Self.element_type) -> Self.element_type:
            return Self.element_type(other) * val

        _ = self._elementwise_unary[mul_val]()

    @always_inline
    fn __imul__[
        other_layout: Layout
    ](
        self,
        other: LayoutTensor[
            dtype,
            other_layout,
            address_space=address_space,
            element_layout=element_layout,
            layout_bitwidth=layout_bitwidth,
        ],
    ):
        """Do a multiplication with another LayoutTensor in place.
        Currently only support tensors of the same shape if the rank
        is the same and also tensors of rank-2.

        Parameters:
            other_layout: The layout of the other tensor.

        Args:
            other: The other tensor to be added to.
        """

        fn mul_val(
            lhs: Self.element_type, rhs: Self.element_type
        ) capturing -> Self.element_type:
            return lhs * rhs

        _ = self._elementwise_binary_with_broadcast[mul_val](other)

    @always_inline
    fn __sub__(
        self, other: Scalar[dtype]
    ) -> __type_of(self.origin_cast[True, MutableAnyOrigin]()):
        """Subtract the LayoutTensor with a scalar value. The scalar value will be
        broadcasted to the entire tensor.

        Args:
            other: The scalar value.
        """

        @parameter
        fn sub_val(val: Self.element_type) -> Self.element_type:
            return val - Self.element_type(other)

        return self._stack_copy()._elementwise_unary[sub_val]()

    @always_inline
    fn __sub__[
        other_layout: Layout,
    ](
        self,
        other: LayoutTensor[
            dtype,
            other_layout,
            address_space=address_space,
            element_layout=element_layout,
            layout_bitwidth=layout_bitwidth,
        ],
    ) -> __type_of(self.origin_cast[True, MutableAnyOrigin]()):
        """Do an subtraction with another LayoutTensor and return the subtracted
        tensor. Currently only support tensors of the same shape if the rank
        is the same and also tensors of rank-2.

        Parameters:
            other_layout: The layout of the other tensor.

        Args:
            other: The other tensor to be subtract from.
        """

        fn sub_val(
            lhs: Self.element_type, rhs: Self.element_type
        ) capturing -> Self.element_type:
            return lhs - rhs

        return self._stack_copy()._elementwise_binary_with_broadcast[sub_val](
            other
        )

    @always_inline
    fn __isub__(self, other: Scalar[dtype]):
        """Subtract scalar value from the LayoutTensor. The scalar value will
        be broadcasted to the entire tensor.

        Args:
            other: The scalar value.
        """

        @parameter
        fn sub_val(val: Self.element_type) -> Self.element_type:
            return val - Self.element_type(other)

        _ = self._elementwise_unary[sub_val]()

    @always_inline
    fn __isub__[
        other_layout: Layout
    ](
        self,
        other: LayoutTensor[
            dtype,
            other_layout,
            address_space=address_space,
            element_layout=element_layout,
            layout_bitwidth=layout_bitwidth,
        ],
    ):
        """Subtracts other from the LayoutTensor. Currently only support tensors
        of the same shape if the rank is the same and also tensors of rank-2.

        Parameters:
            other_layout: The layout of the other tensor.

        Args:
            other: The other tensor to be subtract from.
        """

        fn sub_val(
            lhs: Self.element_type, rhs: Self.element_type
        ) capturing -> Self.element_type:
            return lhs - rhs

        _ = self._elementwise_binary_with_broadcast[sub_val](other)

    @always_inline
    fn __truediv__(
        self, other: Scalar[dtype]
    ) -> __type_of(self.origin_cast[True, MutableAnyOrigin]()):
        """Truediv the LayoutTensor with a scalar value. The scalar value will be
        broadcasted to the entire tensor.

        Args:
            other: The scalar value.
        """

        @parameter
        fn div_val(val: Self.element_type) -> Self.element_type:
            return val / Self.element_type(other)

        return self._stack_copy()._elementwise_unary[div_val]()

    @always_inline
    fn __truediv__[
        other_layout: Layout
    ](
        self,
        other: LayoutTensor[
            dtype,
            other_layout,
            address_space=address_space,
            element_layout=element_layout,
            layout_bitwidth=layout_bitwidth,
        ],
    ) -> __type_of(self.origin_cast[True, MutableAnyOrigin]()):
        """Do an truediv with another LayoutTensor and return the divided
        tensor. Currently only support tensors of the same shape if the rank
        is the same and also tensors of rank-2.

        Parameters:
            other_layout: The layout of the other tensor.

        Args:
            other: The other tensor to be subtract from.
        """

        fn div_val(
            lhs: Self.element_type, rhs: Self.element_type
        ) capturing -> Self.element_type:
            return lhs / rhs

        return self._stack_copy()._elementwise_binary_with_broadcast[div_val](
            other
        )

    fn __itruediv__(self, other: Scalar[dtype]):
        """Truediv the LayoutTensor with a scalar value. The scalar value will be
        broadcasted to the entire tensor.

        Args:
            other: The scalar value.
        """

        @parameter
        fn div_val(val: Self.element_type) -> Self.element_type:
            return val / Self.element_type(other)

        _ = self._elementwise_unary[div_val]()

    @always_inline
    fn __itruediv__[
        other_layout: Layout
    ](
        self,
        other: LayoutTensor[
            dtype,
            other_layout,
            address_space=address_space,
            element_layout=element_layout,
            layout_bitwidth=layout_bitwidth,
        ],
    ):
        """Do an truediv with another LayoutTensor and return the divided
        tensor. Currently only support tensors of the same shape if the rank
        is the same and also tensors of rank-2.

        Parameters:
            other_layout: The layout of the other tensor.

        Args:
            other: The other tensor to be subtract from.
        """

        fn div_val(
            lhs: Self.element_type, rhs: Self.element_type
        ) capturing -> Self.element_type:
            return lhs / rhs

        _ = self._elementwise_binary_with_broadcast[div_val](other)

    @always_inline("nodebug")
    fn __getitem__(self, *dims: Int) -> Self.element_type:
        """Get the element of the tensor with a specified index. Note that the
        size of index has to match the rank of the tensor.

        Args:
            dims: The indexes that specify which element to retrieve.
        """

        var strides = self.runtime_layout.stride.value
        var offset = Self._get_offset(strides, dims)

        return (
            Element[dtype, Self.element_layout]
            .load(self.ptr.offset(offset), self.runtime_element_layout)
            .element_data
        )

    @always_inline("nodebug")
    fn __setitem__(self, d0: Int, val: Self.element_type):
        """Set the element of the tensor with a specified index and value.

        Args:
            d0: The first dimensional index.
            val: The value writing to the tensor.
        """

        var strides = self.runtime_layout.stride.value
        var offset = Self._get_offset(strides, VariadicList[Int](d0))

        Element[dtype, Self.element_layout](
            val, self.runtime_element_layout
        ).store(self.ptr.offset(offset))

    @always_inline("nodebug")
    fn __setitem__(self, d0: Int, d1: Int, val: Self.element_type):
        """Set the element of the tensor with a specified index and value.

        Args:
            d0: The first dimensional index.
            d1: The second dimensional index.
            val: The value writing to the tensor.
        """

        var strides = self.runtime_layout.stride.value
        var offset = Self._get_offset(strides, VariadicList[Int](d0, d1))

        Element[dtype, Self.element_layout](
            val, self.runtime_element_layout
        ).store(self.ptr.offset(offset))

    @always_inline("nodebug")
    fn __setitem__(self, d0: Int, d1: Int, d2: Int, val: Self.element_type):
        """Set the element of the tensor with a specified index and value.

        Args:
            d0: The first dimensional index.
            d1: The second dimensional index.
            d2: The third dimensional index.
            val: The value writing to the tensor.
        """

        var strides = self.runtime_layout.stride.value
        var offset = Self._get_offset(strides, VariadicList[Int](d0, d1, d2))

        Element[dtype, Self.element_layout](
            val, self.runtime_element_layout
        ).store(self.ptr.offset(offset))

    @always_inline("nodebug")
    fn __setitem__(
        self, d0: Int, d1: Int, d2: Int, d3: Int, val: Self.element_type
    ):
        """Set the element of the tensor with a specified index and value.

        Args:
            d0: The first dimensional index.
            d1: The second dimensional index.
            d2: The third dimensional index.
            d3: The fourth dimensional index.
            val: The value writing to the tensor.
        """

        var strides = self.runtime_layout.stride.value
        var offset = Self._get_offset(
            strides, VariadicList[Int](d0, d1, d2, d3)
        )

        Element[dtype, Self.element_layout](
            val, self.runtime_element_layout
        ).store(self.ptr.offset(offset))

    @always_inline("nodebug")
    fn load[width: Int](self, m: Int, n: Int) -> SIMD[dtype, width]:
        """Load a value from a specified location.

        Parameters:
            width: The simd width of the returned value.

        Args:
            m: The m dimension of the value.
            n: The n dimension of the value.
        """

        return self.ptr.load[width=width](self._offset(m, n))

    @always_inline
    fn prefetch(self, m: Int, n: Int):
        """Do software prefetching of a value from a specified location.

        Args:
            m: The m dimension of the value.
            n: The n dimension of the value.
        """
        prefetch[PrefetchOptions().for_read().high_locality().to_data_cache()](
            self.ptr.offset(self._offset(m, n))
        )

    @always_inline("nodebug")
    fn aligned_load[width: Int](self, m: Int, n: Int) -> SIMD[dtype, width]:
        """Do a load with a specified alignment base on the dtype and simd width.

        Parameters:
            width: The simd width if the returned value.

        Args:
            m: The m dimension of the value.
            n: The n dimension of the value.
        """

        alias alignment = alignof[SIMD[dtype, width]]()
        return self.ptr.load[width=width, alignment=alignment](
            self._offset(m, n)
        )

    @always_inline("nodebug")
    fn store[width: Int](self, m: Int, n: Int, val: SIMD[dtype, width]):
        """Store a value to a specified location.

        Parameters:
            width: The simd width of the stored value.

        Args:
            m: The m dimensional index to the tensor.
            n: The n dimensional index to the tensor.
            val: The value to be stored.
        """

        return self.ptr.store(self._offset(m, n), val)

    @always_inline("nodebug")
    fn aligned_store[width: Int](self, m: Int, n: Int, val: SIMD[dtype, width]):
        """Do a store with a specified alignment base on the dtype and simd width.

        Parameters:
            width: The simd width if the stored value.

        Args:
            m: The m dimensional index to the tensor.
            n: The n dimensional index to the tensor.
            val: The value to be stored.
        """

        alias alignment = alignof[SIMD[dtype, width]]()
        return self.ptr.store[alignment=alignment](self._offset(m, n), val)

    @staticmethod
    @always_inline("nodebug")
    fn stack_allocation[
        *, alignment: Int = Self.alignment
    ]() -> LayoutTensor[
        dtype,
        layout,
        MutableAnyOrigin,
        address_space=address_space,
        element_layout=element_layout,
        layout_bitwidth=layout_bitwidth,
        masked=masked,
        alignment=alignment,
    ]:
        """Allocates stack memory for a LayoutTensor. Expects layout to be
        fully static.

        Parameters:
            alignment: Alignment of the allocation. It must be multiple of
              the tensor's alignment, which is the minimum required by
              arch, instruction, etc.
        """

        constrained[layout.all_dims_known(), "Requires fully static layout"]()
        constrained[
            alignment % Self.alignment == 0,
            "Stack allocation alignment "
            + String(alignment)
            + " must be multiple of tensor alignment "
            + String(Self.alignment),
        ]()

        var ptr = stack_allocation[
            layout.size(),
            dtype,
            alignment=alignment,
            address_space=address_space,
        ]()

        return ptr

    @always_inline("nodebug")
    fn _stack_copy(
        self,
    ) -> LayoutTensor[
        dtype,
        layout,
        MutableAnyOrigin,
        address_space=address_space,
        element_layout=element_layout,
        layout_bitwidth=layout_bitwidth,
        masked=masked,
        alignment=alignment,
    ]:
        var copy = self.stack_allocation()

        fn self_value(
            lhs: Self.element_type, rhs: Self.element_type
        ) capturing -> Self.element_type:
            return rhs

        return copy._elementwise_binary_with_broadcast[self_value](self)

    @staticmethod
    @always_inline("nodebug")
    fn _to_static[t: IntTuple]() -> IndexList[len(t)]:
        var st = IndexList[len(t)]()

        @parameter
        for i in range(len(t)):
            st[i] = Int(t[i])
        return st

    @staticmethod
    @always_inline("nodebug")
    fn _get_offset[
        rank: Int
    ](stride: IndexList[rank, **_], vals: VariadicList[Int]) -> Int:
        var offset = 0

        @parameter
        for i in range(rank):
            offset += vals[i] * stride[i]
        return offset

    @staticmethod
    @always_inline("nodebug")
    fn _get_offset[
        rank_1: Int, rank_2: Int
    ](stride: IndexList[rank_1, **_], vals: IndexList[rank_2]) -> Int:
        # In theory we should be able to verify this at compile time but it not happening now!
        constrained[
            rank_1 == rank_2, "shape and stride should be the same rank!"
        ]()
        var offset = 0

        @parameter
        for i in range(rank_1):
            offset += vals[i] * stride[i]
        return offset

    @always_inline
    @staticmethod
    fn shape[idx: Int]() -> Int:
        """Returns the shape of the tensor given a index.

        Parameters:
            idx: The index to the shape of the tensor.
        """

        # FIXME: having to specify the origin is kind of weird
        alias shape = Self._to_static[layout.shape.origin, layout.shape]()
        return shape[idx]

    @always_inline
    @staticmethod
    fn stride[idx: Int]() -> Int:
        """Returns the stride of the tensor given a index.

        Parameters:
            idx: The index to the stride of the tensor.
        """

        # FIXME: having to specify the origin is kind of weird
        alias stride = Self._to_static[layout.stride.origin, layout.stride]()
        return stride[idx]

    @always_inline
    fn dim(self, idx: Int) -> Int:
        """Returns the dimension of the tensor given a index.

        Arguments:
            idx: The index to the dimension of the tensor.
        """
        constrained[
            depth(layout.shape) == 1,
            "Dim is defined for depth-1 layouts",
        ]()
        return self.runtime_layout.shape.value[idx]

    @always_inline
    fn coalesce(
        self,
        out result: LayoutTensor[
            dtype,
            coalesce(layout),
            origin,
            address_space=address_space,
            element_layout = self.element_layout,
        ],
    ):
        """Returns a LayoutTensor with a coalesced Layout."""
        return __type_of(result)(self.ptr)

    @staticmethod
    fn _compute_tile_layout[*tile_sizes: Int]() -> Layout:
        return Self._divide_tiles[*tile_sizes]()

    @staticmethod
    fn _divide_tiles[*tile_sizes: Int]() -> Layout:
        alias tiler = MakeTileLayoutList[*tile_sizes]()
        return zipped_divide(layout, tiler)

    @staticmethod
    @always_inline
    fn _prop_unknown_shape[idx: Int, src: Layout, target: Layout]() -> Layout:
        """Propagate all unknown dim from target to a new layout."""
        var new_shape = src.shape
        # new_shape[idx] = propagate_unknown(src.shape[idx], target.shape)
        new_shape = new_shape.replace_entry(
            idx, propagate_unknown(src.shape[idx], target.shape)
        )
        return Layout(new_shape, src.stride)

    @staticmethod
    fn _compute_tile_layout[tile_size: Int, axis: Int]() -> Layout:
        var tiler = LayoutList()
        var i = 0
        for dim in layout.shape:
            if i == axis:
                tiler.append(Layout(tile_size))
            else:
                tiler.append(Layout(dim))
            i += 1
        return zipped_divide(layout, tiler)

    @always_inline
    fn tile[
        *tile_sizes: Int,
    ](
        self,
        *tile_coords: Int,
        out result: LayoutTensor[
            dtype,
            Self._compute_tile_layout[*tile_sizes]()[0],
            origin,
            address_space=address_space,
            element_layout=element_layout,
            masked = masked or _tile_is_masked[layout, *tile_sizes](),
        ],
    ):
        """Tiles the layout and returns a tensor tile with the specified
        tile_sizes at specific tile coordinates.

        Parameters:
            tile_sizes: The tile sizes of the returned LayoutTensor.

        Args:
            tile_coords: The tile coordinate. This refer to the coordinate of
                         the tile after the tiled layout. Consider the following
                         example.

        Example:

            Memory Layout of
                            [1 2 3 4]
                            [2 3 4 5]
                            [5 4 3 2]
                            [1 1 1 1]

            tile[2, 2](1, 0) will give you
                            [5 4]
                            [1 1]
        """

        alias num_tiles = _get_len[*tile_sizes]()

        # need to calculate this again because __tiled_layout[1] is required for the offset calculation
        alias __tiled_layout = Self._compute_tile_layout[*tile_sizes]()

        constrained[
            __tiled_layout[1].rank() == num_tiles,
            "Number of tiles should match the rank",
        ]()

        # Static layout tiling
        # TODO: Consider merge the two cases in away that won't slowdown the fully static layout.
        @parameter
        if result.layout.all_dims_known():
            var offset = 0

            runtime_shape = RuntimeTuple[
                result.layout.shape,
                element_bitwidth = result.layout_bitwidth,
                unsigned=True,
            ]()

            var runtime_stride = RuntimeTuple[
                result.layout.stride, unsigned=True
            ]()

            @parameter
            for i in range(num_tiles):
                alias stride = Int(__tiled_layout[1].stride[i])
                offset += tile_coords[i] * stride

            var runtime_layout = RuntimeLayout(runtime_shape, runtime_stride)

            # Adjust runtime layout, so the shape is clipped to the unmasked sizes.
            @parameter
            if result.masked:

                @parameter
                for i in range(result.layout.rank()):
                    cur_dim = self.dim(i) - (tile_coords[i] * tile_sizes[i])
                    shape_i = max(0, min(tile_sizes[i], cur_dim))
                    runtime_layout.shape.value[i] = shape_i

            return __type_of(result)(self.ptr.offset(offset), runtime_layout)

        else:
            # Dynamic layout, use strides
            var offset = 0

            dynamic_shape = RuntimeTuple[
                result.layout.shape,
                element_bitwidth = result.layout_bitwidth,
                unsigned=True,
            ]()

            var dynamic_stride = RuntimeTuple[
                result.layout.stride, unsigned=True
            ]()

            @parameter
            for i in range(num_tiles):
                var stride = self.runtime_layout.stride.value[i] * tile_sizes[i]
                dynamic_stride.value[i] = self.runtime_layout.stride.value[i]
                offset += tile_coords[i] * stride

            var runtime_layout = RuntimeLayout(dynamic_shape, dynamic_stride)

            # Adjusts the runtime layout so that the shape is clipped to the unmasked sizes.
            @parameter
            for i in range(result.layout.rank()):
                cur_dim = self.dim(i) - (tile_coords[i] * tile_sizes[i])
                shape_i = max(0, min(tile_sizes[i], cur_dim))
                runtime_layout.shape.value[i] = shape_i

            return __type_of(result)(self.ptr.offset(offset), runtime_layout)

    @always_inline
    fn tiled_iterator[
        *tile_sizes: Int,
        axis: Int = 0,
    ](
        self,
        *tile_coords: Int,
        out result: LayoutTensorIter[
            dtype,
            Self._compute_tile_layout[*tile_sizes]()[0],
            origin,
            address_space=address_space,
            circular=False,
            axis=axis,
            layout_bitwidth = Self.layout_bitwidth,
            masked = masked or _tile_is_masked[layout, *tile_sizes](),
        ],
    ):
        """Returns the tiled iterator of the LayoutTensor.

        Parameters:
            tile_sizes: Tile sizes of each tile the iterator will iterate through.
            axis: Axis of the LayoutTensor the iterator will iterate through.

        Args:
            tile_coords: The tile coordinate that the iterator will point to.
        """

        alias tiles_rank = _get_len[*tile_sizes]()
        alias __tiled_layout = Self._compute_tile_layout[*tile_sizes]()
        constrained[
            __tiled_layout[1].rank() == tiles_rank,
            "Number of tiles should match the rank",
        ]()

        var ptr_offset = 0

        @parameter
        if layout.all_dims_known():
            var runtime_shape = RuntimeTuple[
                result.layout.shape,
                element_bitwidth = Self.layout_bitwidth,
                unsigned=True,
            ]()
            var runtime_stride = RuntimeTuple[
                result.layout.stride, unsigned=True
            ]()

            @parameter
            for i in range(tiles_rank):
                alias stride = Int(__tiled_layout[1].stride[i])
                ptr_offset += tile_coords[i] * stride

            # fmt: off

            # A nested LayoutTensor may have shape=(16, 64) and stride=(1, 16)
            # In order to calculate the bound we only need to use the last
            # element in the IntTuple.
            alias is_axis_val = layout.shape[axis].is_value()
            alias bound = layout.shape[axis].value() * layout.stride[axis].value() \
                if is_axis_val \
                else layout.shape[axis][-1].value() * layout.stride[axis][-1].value()
            alias dim_bound = Self.shape[axis]() \
                if is_axis_val \
                else product(Self.layout.shape[axis])
            alias stride = __tiled_layout[1].stride[axis].value()
            # fmt: on

            @parameter
            if result.masked:

                @parameter
                for i in range(result.layout.rank()):
                    cur_dim = self.dim(i) - (tile_coords[i] * tile_sizes[i])
                    shape_i = max(0, min(tile_sizes[i], cur_dim))
                    runtime_shape.value[i] = shape_i

                return __type_of(result)(
                    self.ptr + ptr_offset,
                    bound,
                    RuntimeLayout(runtime_shape, runtime_stride),
                    stride=stride,
                    offset=0,
                    dimension_bound=dim_bound,
                    idx=tile_coords[axis],
                )
            else:
                return __type_of(result)(
                    self.ptr + ptr_offset,
                    bound,
                    stride=stride,
                    offset=0,
                )

        else:
            var runtime_shape = RuntimeTuple[
                result.layout.shape,
                element_bitwidth = Self.layout_bitwidth,
                unsigned=True,
            ]()
            var runtime_stride = RuntimeTuple[
                result.layout.stride, unsigned=True
            ]()

            @parameter
            for i in range(tiles_rank):
                var stride = self.runtime_layout.stride.value[i] * tile_sizes[i]
                runtime_stride.value[i] = self.runtime_layout.stride.value[i]
                ptr_offset += tile_coords[i] * stride

            var axis_dim = self.runtime_layout.shape.value[axis]
            var axis_stride = self.runtime_layout.stride.value[axis]
            var iter_bound = axis_dim * axis_stride
            var iter_stride = tile_sizes[axis] * axis_stride

            @parameter
            for i in range(result.layout.rank()):
                cur_dim = self.dim(i) - (tile_coords[i] * tile_sizes[i])
                shape_i = max(0, min(tile_sizes[i], cur_dim))
                runtime_shape.value[i] = shape_i

            return __type_of(result)(
                self.ptr + ptr_offset,
                iter_bound,
                stride=iter_stride,
                offset=0,
                runtime_layout=RuntimeLayout(runtime_shape, runtime_stride),
                dimension_bound=self.dim(axis),
                idx=tile_coords[axis],
            )

    @always_inline
    fn split[
        count: Int,
        axis: Int = 0,
    ](
        self,
        out result: StaticTuple[
            LayoutTensor[
                dtype,
                Self._compute_tile_layout[
                    layout.shape[axis].value() // count, axis
                ]()[0],
                origin,
                address_space=address_space,
                element_layout=element_layout,
                alignment=alignment,
            ],
            count,
        ],
    ):
        """Split the LayoutTensor along a axis and return an InlineArray of
        LayoutTensor.

        Parameters:
            count: Number of portion to split.
            axis: The axis where the split is applied to.
        """

        constrained[
            layout.shape[axis].is_value(),
            "Only support partition modes that are plain values.",
        ]()

        constrained[
            layout.shape[axis].value() % count == 0,
            "The input dimension must be divisible over the input count.",
        ]()

        alias stride = layout.stride[axis].value()
        var tiles = __type_of(result)()

        @parameter
        for i in range(count):
            # Need tile_size alias to ensure that the ptr passed to LayoutTensor is
            # known at compile time. Otherwise we get compile time failure.
            # The compiler can't allocate LayoutTensor on stack if ptr is not known at compile time.
            # See MOCO-1081 for more details.
            alias tile_size = layout.shape[axis].value() // count
            tiles[i] = LayoutTensor[
                dtype,
                Self._compute_tile_layout[
                    layout.shape[axis].value() // count, axis
                ]()[0],
                origin,
                address_space=address_space,
                element_layout=element_layout,
                alignment=alignment,
            ](self.ptr.offset(i * tile_size * stride))

        return tiles

    @always_inline
    fn split[
        axis: Int = 0,
        alignment: Int = 1,
    ](
        self,
        count: Int,
        idx: Int,
        out result: LayoutTensor[
            dtype,
            layout.make_shape_unknown[axis](),
            origin,
            address_space=address_space,
            element_layout=element_layout,
        ],
    ):
        constrained[
            layout.shape[axis].is_value(), "Can't split non-scalar dimension."
        ]()

        # We can split dynamic dimension but that should be audited carefully with
        # other parts when we really want to support arbitrary K, N in matmul.
        # Restrict to static case for now.
        constrained[
            layout.shape[axis].value() != UNKNOWN_VALUE
            and layout.stride[axis].value() != UNKNOWN_VALUE,
            "Shouldn't split dynamic dimension.",
        ]()

        alias axis_dim = layout.shape[axis].value()
        alias axis_stride = layout.stride[axis].value()
        alias flatten_rank = len(flatten(layout.shape))
        alias axis_in_flatten_tuple = runtime_shape.offset_until[axis]()

        var runtime_shape = RuntimeTuple[
            result.layout.shape,
            element_bitwidth = result.layout_bitwidth,
            unsigned=True,
        ]()
        var axis_partition_dim = align_up(axis_dim // count, alignment)

        @parameter
        for i in range(flatten_rank):
            var shape_i = self.runtime_layout.shape.value[i]

            @parameter
            if i == axis_in_flatten_tuple:
                runtime_shape.value[i] = min(
                    axis_partition_dim, shape_i - idx * axis_partition_dim
                )
            else:
                runtime_shape.value[i] = shape_i

        return __type_of(result)(
            # Only the last partition can have size other than axis_partition_dim.
            self.ptr + idx * axis_partition_dim * axis_stride,
            RuntimeLayout[result.layout, bitwidth = result.layout_bitwidth](
                runtime_shape,
                rebind[RuntimeTuple[result.layout.stride, unsigned=True]](
                    self.runtime_layout.stride
                ),
            ),
        )

    @always_inline
    fn _clamp_distribute_shape[
        thread_layout: Layout,
    ](self, thread_id: UInt) -> IndexList[Self.rank]:
        constrained[
            len(flatten(thread_layout.shape)) <= 2
            and len(flatten(thread_layout.stride)) <= 2,
            "Only supporting rank-2 or less thread layout for dynamic tile.",
        ]()

        # clamp IndexList using thread_id and thread_layout
        var tile_shape = IndexList[Self.rank]()
        alias thread_shape = thread_layout.shape
        alias thread_stride = thread_layout.stride

        # this would only work for rank-2 thread layout, need to extend this
        # to support thread layout such as Layout((2, 2), 2)
        @parameter
        for i in range(Self.rank):
            alias thread_stride_i = Int(thread_stride[i])
            alias thread_shape_i = Int(thread_shape[i])
            var tile_idx = (thread_id // thread_stride_i) % thread_shape_i
            var tile_shape_i = ceildiv(self.dim(i), thread_shape_i)
            var bound_i = Int((tile_shape_i - 1) * thread_shape_i + tile_idx)
            tile_shape[i] = min(self.dim(i) - bound_i, tile_shape_i)

        return tile_shape

    @always_inline
    fn distribute[
        threads_layout: Layout,
        axis: OptionalReg[Int] = None,
        swizzle: OptionalReg[Swizzle] = None,
        submode_axis: OptionalReg[Int] = None,
    ](
        self,
        thread_id: UInt,
        out result: LayoutTensor[
            dtype,
            _compute_distribute_layout[
                layout,
                threads_layout,
                axis,
            ]()[1],
            origin,
            address_space=address_space,
            element_layout=element_layout,
            masked = masked
            or _distribute_is_masked[layout, threads_layout, axis](),
        ],
    ):
        """Distribute tiled workload to threads.

        If the `axis` is given, for example, using `axis = 0` for 4 threads:
        TH_0 TH_2
        TH_1 TH_3
        This means the tensor is only distributed to threads in axis = 0, i.e.,
        threads 0 and 1. Threads 2 and 3 gets the same tile as 0 and 1, respectively.
        This is useful when threads load same vectors from a row in A matrix and
        some threads share the same vector.
        """

        alias distributed_layout = _compute_distribute_layout[
            layout,
            threads_layout,
            axis,
        ]()

        @parameter
        if result.masked:
            runtime_shape = RuntimeTuple[
                result.layout.shape,
                element_bitwidth = result.layout_bitwidth,
                unsigned=True,
            ](self._clamp_distribute_shape[threads_layout](thread_id))
        else:
            runtime_shape = RuntimeTuple[
                result.layout.shape,
                element_bitwidth = result.layout_bitwidth,
                unsigned=True,
            ]()

        var runtime_stride = RuntimeTuple[result.layout.stride, unsigned=True]()

        # Static layout tiling
        # TODO: Consider merge the two cases in away that won't slowdown the fully static layout.
        @parameter
        if layout.all_dims_known():
            alias fragments_layout_stride = flatten(
                distributed_layout[0].stride
            )

            # Only extract coordinates in the given axis.
            # Example: axis = 0 for 2x2 threads, we only need thread 0 and 1's
            # coordinates since thread 2 and 3 are getting the same tile.
            alias thread_projected_stride = flatten(
                threads_layout.stride[
                    axis.value()
                ] if axis else threads_layout.stride
            )
            alias thread_projected_shape = flatten(
                threads_layout.shape[
                    axis.value()
                ] if axis else threads_layout.shape
            )

            var offset: Scalar[Self.index_type] = 0

            @parameter
            for i in range(len(fragments_layout_stride)):
                alias fragments_stride_i: UInt = Int(
                    fragments_layout_stride[i]
                ).value
                alias shape_i: UInt = Int(thread_projected_shape[i])
                alias stride_i: UInt = Int(thread_projected_stride[i])
                var thread_coord_i: UInt = (thread_id // stride_i) % shape_i
                offset += thread_coord_i * fragments_stride_i

            # Swizzling applies to the index of elements rather than scalars because
            # the former is the unit in distribution.
            var swizzled_offset = offset

            @parameter
            if swizzle:
                alias swizzle_fn = swizzle.value()
                swizzled_offset = (
                    swizzle_fn(offset // self.element_size) * self.element_size
                )

            @parameter
            if result.masked:
                return __type_of(result)(
                    self.ptr.offset(Int(swizzled_offset)),
                    RuntimeLayout(runtime_shape, runtime_stride),
                )
            else:
                return __type_of(result)(
                    self.ptr.offset(Int(swizzled_offset)),
                )

        else:
            constrained[
                layout.known_shape() and threads_layout.all_dims_known(),
                (
                    "Distribute expecting layout with static shapes and"
                    " fully static threads_layout"
                ),
            ]()

            # Only extract coordinates in the given axis.
            # Example: axis = 0 for 2x2 threads, we only need thread 0 and 1's
            # coordinates since thread 2 and 3 are getting the same tile.
            alias thread_projected_stride = flatten(
                threads_layout.stride[
                    axis.value()
                ] if axis else threads_layout.stride
            )
            alias thread_projected_shape = flatten(
                threads_layout.shape[
                    axis.value()
                ] if axis else threads_layout.shape
            )

            var offset: Scalar[Self.index_type] = 0

            @parameter
            for i in range(runtime_shape.scalar_length):
                alias thread_shape_i = threads_layout[i].size()
                runtime_stride.value[i] = (
                    self.runtime_layout.stride.value[i] * thread_shape_i
                )

            @parameter
            for i in range(len(flatten(Self.layout.stride))):
                var fragments_stride_i = self.runtime_layout.stride.value[i]
                alias shape_i: UInt = Int(thread_projected_shape[i]).value
                alias stride_i: UInt = Int(thread_projected_stride[i]).value
                var thread_coord_i: UInt = (thread_id // stride_i) % shape_i
                offset += thread_coord_i * fragments_stride_i

            # Swizzling applies to the index of elements rather than scalars because
            # the former is the unit in distribution.
            var swizzled_offset = offset

            @parameter
            if swizzle:
                alias swizzle_fn = swizzle.value()
                swizzled_offset = (
                    swizzle_fn(offset // self.element_size) * self.element_size
                )

            @parameter
            if self.element_layout.all_dims_known():
                return __type_of(result)(
                    self.ptr.offset(Int(swizzled_offset)),
                    RuntimeLayout(runtime_shape, runtime_stride),
                )
            else:
                return __type_of(result)(
                    self.ptr.offset(Int(swizzled_offset)),
                    RuntimeLayout(runtime_shape, runtime_stride),
                    self.runtime_element_layout,
                )

    @always_inline
    fn vectorize[
        *vector_shape: Int
    ](
        self,
        out result: LayoutTensor[
            dtype,
            coalesce(
                Self._compute_tile_layout[*vector_shape]()[1], keep_rank=True
            ),
            origin,
            address_space=address_space,
            element_layout = Self._divide_tiles[*vector_shape]()[0],
            masked=masked,
        ],
    ):
        @parameter
        @always_inline
        fn _check_vector_shape[*vec_shape: Int]():
            @parameter
            for i in range(_get_len[*vec_shape]()):
                alias shape_i = Int(self.layout.shape[i])

                @parameter
                if shape_i == UNKNOWN_VALUE:
                    constrained[
                        vec_shape[i] == 1,
                        "vector dim has to be 1 when layout shape is unknown.",
                    ]()
                else:
                    constrained[
                        shape_i % vec_shape[i] == 0,
                        (
                            "tensor dim has to be an integer multiple of vector"
                            " dim."
                        ),
                    ]()

                    constrained[
                        shape_i >= vec_shape[i],
                        "vectorize shape has to be smaller than tensor shape.",
                    ]()

        runtime_shape = RuntimeTuple[
            result.layout.shape,
            element_bitwidth = result.layout_bitwidth,
            unsigned=True,
        ]()
        runtime_stride = RuntimeTuple[result.layout.stride, unsigned=True]()

        @parameter
        if result.masked or not layout.all_dims_known():

            @parameter
            for i in range(runtime_shape.scalar_length):
                runtime_shape.value[i] = ceildiv(
                    self.runtime_layout.shape.value[i], vector_shape[i]
                )
                runtime_stride.value[i] = (
                    self.runtime_layout.stride.value[i] * vector_shape[i]
                )

        @parameter
        if layout.all_dims_known():

            @parameter
            if result.masked:
                return __type_of(result)(
                    self.ptr,
                    RuntimeLayout(runtime_shape, runtime_stride),
                )
            else:
                return __type_of(result)(self.ptr)
        else:
            constrained[
                coalesce(result.element_layout).known_shape(),
                "Result element layout should have known shape",
            ]()

            runtime_element_layout_shape = RuntimeTuple[
                result.element_layout.shape, unsigned=True
            ]()
            runtime_element_layout_stride = RuntimeTuple[
                result.element_layout.stride, unsigned=True
            ](self.runtime_layout.stride.value)

            return __type_of(result)(
                self.ptr,
                RuntimeLayout(runtime_shape, runtime_stride),
                rebind[RuntimeLayout[result.element_layout]](
                    RuntimeLayout(
                        runtime_element_layout_shape,
                        runtime_element_layout_stride,
                    )
                ),
            )

    @staticmethod
    fn _compute_slice_layout(d0_slice: Slice, d1_slice: Slice) -> Layout:
        constrained[
            layout.shape.__len__() == 2,
            "Only rank-2 tensors slices are supported for now!",
        ]()
        return Layout(
            IntTuple(
                _get_slice_size(Self.layout, d0_slice, 0),
                _get_slice_size(Self.layout, d1_slice, 1),
            ),
            layout.stride,
        )

    @staticmethod
    fn _compute_slice_layout(
        slice_0: Slice, slice_1: Slice, slice_0_axis: Int, slice_1_axis: Int
    ) -> Layout:
        constrained[
            layout.shape.__len__() > 2,
            "Rank should be >= 2",
        ]()
        var sliced_layout = sublayout(Self.layout, slice_0_axis, slice_1_axis)
        return Layout(
            IntTuple(
                _get_slice_size(sliced_layout, slice_0, 0),
                _get_slice_size(sliced_layout, slice_1, 1),
            ),
            sliced_layout.stride,
        )

    @staticmethod
    fn _compute_slice_layout(slice_0: Slice, slice_0_axis: Int) -> Layout:
        constrained[
            layout.shape.__len__() > 1,
            "Rank should be >= 1",
        ]()
        var sliced_layout = sublayout(Self.layout, slice_0_axis)
        return Layout(
            IntTuple(
                _get_slice_size(sliced_layout, slice_0, 0),
            ),
            sliced_layout.stride[0],
        )

    @always_inline
    fn slice[
        d0_slice: Slice,
        d1_slice: Slice,
    ](
        self,
        out result: LayoutTensor[
            dtype,
            Self._compute_slice_layout(
                d0_slice,
                d1_slice,
            ),
            origin,
            address_space=address_space,
            element_layout=element_layout,
        ],
    ):
        constrained[
            d0_slice.step.or_else(1) == 1 and d1_slice.step.or_else(1) == 1,
            "Slice should have no gaps",
        ]()
        alias stride_m = Int(result.layout.stride[0])
        alias stride_n = Int(result.layout.stride[1])

        alias d0_slice_start = d0_slice.start.or_else(0)
        alias d1_slice_start = d1_slice.start.or_else(0)

        var offset = d0_slice_start * stride_m + d1_slice_start * stride_n

        return __type_of(result)(self.ptr.offset(offset))

    @always_inline
    fn slice[
        d0_slice: Slice,
        d1_slice: Slice,
        slice_indices: IndexList[2],
        __offset_dims: Int = Self.rank - 2,
    ](
        self,
        offsets: IndexList[__offset_dims],
        out result: LayoutTensor[
            dtype,
            Self._compute_slice_layout(
                d0_slice, d1_slice, slice_indices[0], slice_indices[1]
            ),
            origin,
            address_space=address_space,
            element_layout=element_layout,
        ],
    ):
        constrained[
            d0_slice.step.or_else(1) == 1 and d1_slice.step.or_else(1) == 1,
            "Slice should have no gaps",
        ]()
        constrained[
            slice_indices[0] < slice_indices[1],
            "Slice indices should be ordered",
        ]()
        alias stride_0 = Int(result.layout.stride[0])
        alias stride_1 = Int(result.layout.stride[1])

        alias d0_slice_start = d0_slice.start.or_else(0)
        alias d1_slice_start = d1_slice.start.or_else(0)

        var slice_offset = d0_slice_start * stride_0 + d1_slice_start * stride_1

        var idx = 0

        @parameter
        for i in range(Self.rank):
            alias stride_i = Int(Self.layout.stride[i])

            alias offset_index = _not_in_tuple[i, 2, slice_indices]()

            @parameter
            if offset_index:
                slice_offset += offsets[idx] * stride_i
                idx += 1

        return __type_of(result)(self.ptr.offset(slice_offset))

    # FIXME: Can't overload slice, hitting compiler issue.
    # https://linear.app/modularml/issue/MOCO-174
    @always_inline
    fn slice_1d[
        d0_slice: Slice,
        slice_indices: IndexList[1],
        __offset_dims: Int = Self.rank - 1,
    ](
        self,
        offsets: IndexList[__offset_dims],
        out result: LayoutTensor[
            dtype,
            Self._compute_slice_layout(d0_slice, slice_indices[0]),
            origin,
            address_space=address_space,
            element_layout=element_layout,
        ],
    ):
        constrained[
            d0_slice.step.or_else(1) == 1,
            "Slice should have no gaps",
        ]()

        alias stride_0 = Int(result.layout.stride[0])

        alias d0_slice_start = d0_slice.start.or_else(0)

        var slice_offset = d0_slice_start * stride_0

        var idx = 0

        @parameter
        for i in range(Self.rank):
            alias stride_i = Int(Self.layout.stride[i])

            alias offset_index = _not_in_tuple[i, 1, slice_indices]()

            @parameter
            if offset_index:
                slice_offset += offsets[idx] * stride_i
                idx += 1

        return __type_of(result)(self.ptr.offset(slice_offset))

    @always_inline
    fn transpose[
        M: Int = Self.shape[0](),
        N: Int = Self.shape[1](),
    ](
        self,
        out result: LayoutTensor[
            dtype,
            composition(
                layout,
                Layout(IntTuple(N, M), IntTuple(M, 1)),
            ),
            origin,
            address_space=address_space,
            element_layout=element_layout,
        ],
    ):
        return __type_of(result)(self.ptr)

    @always_inline
    fn reshape[
        dst_layout: Layout,
    ](
        self,
        out result: LayoutTensor[
            dtype,
            dst_layout,
            origin,
            address_space=address_space,
            element_layout=element_layout,
            masked=masked,
        ],
    ):
        constrained[not masked, "Masked tensor does not support reshape."]()
        return __type_of(result)(self.ptr)

    @always_inline
    fn composition[
        rhs_layout: Layout,
        dst_layout: Layout = composition(layout, rhs_layout),
    ](
        self,
        out result: LayoutTensor[
            dtype,
            dst_layout,
            origin,
            address_space=address_space,
            element_layout=element_layout,
        ],
    ):
        return __type_of(result)(self.ptr)

    @always_inline
    fn distance[
        # Unsafe to infer uint type from self's layout cosize because the
        # input ptr could come from a much larger base tensor.
        _uint_dtype: DType = DType.uint32 if address_space
        == _GPUAddressSpace.SHARED else DType.uint64,
    ](
        self,
        addr: UnsafePointer[Scalar[dtype], address_space=address_space, *_],
    ) -> Scalar[_uint_dtype]:
        """Returns the distance from the input address."""

        return Scalar[_uint_dtype](Int(self.ptr) - Int(addr)) // sizeof[dtype]()

    @always_inline
    fn distance[
        _layout: Layout,
        _uint_dtype: DType = _get_unsigned_type(_layout, address_space),
    ](
        self, src: LayoutTensor[dtype, _layout, address_space=address_space]
    ) -> Scalar[_uint_dtype]:
        """Returns the distance from the input address."""

        return Scalar[_uint_dtype](
            (Int(self.ptr) - Int(src.ptr)) // sizeof[dtype]()
        )

    # Returns the linear index of an elem_i 0 ... size(layout).
    #
    @always_inline
    fn _get_element_idx[elem_i: Int](self) -> Int:
        alias element_size = Int(self.element_size)

        @parameter
        if layout.all_dims_known():
            alias idx = make_layout(element_layout, layout)(
                elem_i * element_size
            )
            return idx
        else:
            # FIXME: this used to be simpler
            var rt = RuntimeTuple[IntTuple(UNKNOWN_VALUE)](
                elem_i * element_size
            )
            var idx = make_runtime_layout(
                self.runtime_element_layout, self.runtime_layout
            )(rt)
            return idx

    @always_inline("nodebug")
    fn copy_from(self, other: LayoutTensor):
        alias other_layout = other.layout

        alias dst_element_size = Int(self.element_size)
        alias src_element_size = Int(other.element_size)

        alias dst_size = layout.size()
        alias src_size = other_layout.size()

        constrained[
            layout.known_shape() and other_layout.known_shape(),
            "copy_from must move data of statically known shape",
        ]()

        constrained[
            dst_size == src_size,
            "copy_from should move data of the same size, getting dst size "
            + String(dst_size)
            + " and src size "
            + String(src_size),
        ]()

        constrained[
            dst_element_size == src_element_size, "copy_from should move"
        ]()

        @parameter
        for i in range(dst_size):
            src_idx = other._get_element_idx[i]()
            dst_idx = self._get_element_idx[i]()

            src_element = MemoryElement(
                other.ptr.offset(src_idx), other.runtime_element_layout
            )

            dst_element = MemoryElement(
                self.ptr.offset(dst_idx), self.runtime_element_layout
            )

            dst_element.transfer(src_element)

    @always_inline("nodebug")
    fn copy_from_async[
        is_masked: Bool = False,
        swizzle: OptionalReg[Swizzle] = None,
        fill: Fill = Fill.NONE,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        src: LayoutTensor,
        src_idx_bound: Scalar[__type_of(src).index_type] = 0,
        base_offset: Self.uint_type = 0,
    ):
        constrained[
            self.address_space == _GPUAddressSpace.SHARED,
            "Async is only supported for destinations in shared memory",
        ]()

        constrained[
            src.dtype == dtype, "src dtype must be the same as dst dtype."
        ]()

        alias dst_size = layout.size()
        alias src_size = src.layout.size()

        alias dst_element_size = Int(self.element_size)
        alias src_element_size = Int(src.element_size)
        constrained[
            dst_element_size == src_element_size,
            "copy_from_async should move data of the same element size",
        ]()

        # Eligibility for 4, 8, 16 bytes async load.
        alias element_size_bytes = sizeof[dtype]() * src_element_size
        constrained[
            element_size_bytes == 4
            or element_size_bytes == 8
            or element_size_bytes == 16,
            "copy_from_async only allows 4, 8, 16 bytes element",
        ]()

        # Share memory must always have static layout.
        alias dst_dims_known = (
            self.layout.all_dims_known()
            and self.element_layout.all_dims_known()
        )
        constrained[dst_dims_known, "dst tensor must have static layout"]()

        alias src_dims_known = (
            src.layout.all_dims_known() and src.element_layout.all_dims_known()
        )

        var dst_ptr = self.ptr.address_space_cast[_GPUAddressSpace.SHARED]()
        var src_ptr = src.ptr.address_space_cast[_GPUAddressSpace.GLOBAL]()

        # Coalesce element layouts to simplify vectorization condition.
        alias coalesce_src_element_layout = coalesce(src.element_layout)
        alias coalesce_dst_element_layout = coalesce(self.element_layout)

        @parameter
        if (
            src.element_layout.all_dims_known()
            and coalesce_src_element_layout.rank() == 1
            and coalesce_src_element_layout.stride[0] == 1
            and coalesce_dst_element_layout.rank() == 1
            and coalesce_dst_element_layout.stride[0] == 1
        ):
            alias num_vecs = layout.size()

            @parameter
            for i in range(num_vecs):
                var src_idx: Scalar[src.index_type] = 0
                alias src_static_idx: Scalar[src.index_type] = src.layout(i)

                @parameter
                if src_dims_known:
                    src_idx = src_static_idx
                else:
                    src_idx = src.runtime_layout(i)
                alias dst_idx = layout(i)
                var swizzled_idx: Scalar[self.index_type] = 0

                @parameter
                if swizzle:
                    alias swizzle_fn = swizzle.value()
                    alias dst_idx_base = dst_idx % swizzle_fn.size()
                    alias dst_idx_diff = dst_idx - dst_idx_base
                    swizzled_idx = (
                        swizzle_fn(base_offset + dst_idx_base)
                        + dst_idx_diff
                        - base_offset
                    ).cast[self.index_type]()
                else:
                    swizzled_idx = dst_idx

                @parameter
                if is_masked:
                    var src_copy_size = Int32(
                        element_size_bytes
                    ) if src_idx < src_idx_bound else 0
                    async_copy[element_size_bytes, fill = Scalar[dtype](0.0)](
                        src_ptr.bitcast[Scalar[dtype]]() + src_idx,
                        dst_ptr + Int(swizzled_idx),
                        src_copy_size,
                    )
                else:
                    async_copy[
                        element_size_bytes,
                        eviction_policy=eviction_policy,
                    ](
                        src_ptr.bitcast[Scalar[dtype]]() + src_idx,
                        dst_ptr + swizzled_idx,
                    )

        # Async copy should only be used for 16B vector for bypassing L1.
        # Scalar path is only for kernel tests.
        else:
            constrained[not swizzle, "Should not swizzle scalar copy."]()

            @parameter
            for i in range(dst_size * dst_element_size):
                var src_idx = 0
                alias src_static_idx = make_layout(
                    src.element_layout, src.layout
                )(i)
                alias dst_idx = make_layout(self.element_layout, self.layout)(i)

                @parameter
                if src_dims_known:
                    src_idx = src_static_idx
                else:
                    # FIXME: this used to be simpler
                    var rt = RuntimeTuple[IntTuple(UNKNOWN_VALUE)](i)
                    src_idx = make_runtime_layout(
                        src.runtime_element_layout, src.runtime_layout
                    )(rt)

                async_copy[4, eviction_policy=eviction_policy](
                    src_ptr.bitcast[Scalar[dtype]]() + src_idx,
                    dst_ptr + dst_idx,
                )

    @always_inline
    fn fill(
        self: LayoutTensor[mut=True, dtype, **_], val: Scalar[dtype]
    ) -> __type_of(self):
        @parameter
        if layout.all_dims_known():
            alias num_elements = layout.size() * Self.element_size

            @parameter
            for i in range(num_elements):
                self.ptr[i] = val
        else:
            var num_elements = self.runtime_layout.size() * Self.element_size
            for i in range(num_elements):
                self.ptr[i] = val
        return self

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        """Format 2D tensor in 2D, otherwise print all values in column major
        coordinate order."""

        if self.runtime_layout.size() == 0:
            return

        @always_inline
        fn is_2d_print(layout: Layout) -> Bool:
            return (
                len(layout) == 2
                and layout.shape[0].is_value()
                and layout.shape[1].is_value()
            )

        # The 2D print works only for layout shape (M, N).
        # Check both original and coalesced layouts so that (M, 1) and
        # ((M), (N)) can all be printed in 2D. Shapes like ((2, 2), 2) will be
        # printed elementwise.
        @parameter
        if is_2d_print(layout):
            _pretty_print_2d_tensor(self, writer)
            return
        elif is_2d_print(coalesce(layout)):
            _pretty_print_2d_tensor(self.coalesce(), writer)
            return

        for i in range(self.runtime_layout.size()):
            var vec_offset = self.runtime_layout(i)
            var vec = SIMD[dtype, Self.element_size]()

            @parameter
            for idx in range(Self.element_size):
                alias element_offset = self.element_layout(idx)
                vec[idx] = self.ptr.load(vec_offset + element_offset)

            writer.write(vec)
            if i != layout.size() - 1:
                writer.write(" ")

    @staticmethod
    @always_inline
    fn _offset(idxs: VariadicList[Int]) -> Int:
        # fn _offset[*idxs: Int]() -> Int:
        constrained[layout.all_dims_known()]()
        var offset: Int = 0
        alias r = Self.layout.rank()

        @parameter
        for i in range(r):
            alias li = Self.layout[i]
            offset += li(idxs[i])
        return offset

    @always_inline
    fn _get[
        *idxs: Int, size: Int = Self.element_size
    ](self) -> SIMD[dtype, size]:
        """Get an element, guaranteeing that the ptr offset is computed at
        compile time.
        Indices are passed as parameters.
        Optional `size` parameter must equal the `element_size` (the default)
        or `element_size == 1`, and the loaded element is broadcast across
        the returned vector.
        """
        constrained[Self.element_size == size or Self.element_size == 1]()
        alias offset = Self._offset(idxs)
        # alias offset = Self._offset[*idxs]()
        var val: Self.element_type = (
            Element[dtype, Self.element_layout]
            .load(self.ptr.offset(offset), self.runtime_element_layout)
            .element_data
        )

        @parameter
        if Self.element_size == size:
            return rebind[SIMD[dtype, size]](val)
        else:
            return SIMD[dtype, size](val[0])

    @always_inline("nodebug")
    fn _set[*idxs: Int](self, val: Self.element_type):
        """Set an element, guaranteeing that the ptr offset is computed at
        compile time.
        Indices are passed as parameters.
        """
        alias offset = Self._offset(idxs)
        # alias offset = Self._offset[*idxs]()
        Element[dtype, Self.element_layout](
            val, self.runtime_element_layout
        ).store(self.ptr.offset(offset))


@always_inline
fn _pretty_print_2d_tensor[W: Writer](tensor: LayoutTensor, mut writer: W):
    constrained[tensor.layout.rank() == 2]()

    var m_dim = tensor.runtime_layout.shape[0].value[0]
    var n_dim = tensor.runtime_layout.shape[1].value[0]
    for m in range(m_dim):
        for n in range(n_dim):
            writer.write(tensor[m, n], " ")
        if m < m_dim - 1:
            writer.write("\n")


fn stack_allocation_like[
    layout: Layout,
    dtype: DType,
    *,
    address_space: AddressSpace,
    target_address_space: AddressSpace = AddressSpace.GENERIC,
](
    in_tensor: LayoutTensor[dtype, layout, address_space=address_space, **_],
    out result: LayoutTensor[
        dtype,
        layout,
        MutableAnyOrigin,
        address_space=target_address_space,
        masked = in_tensor.masked,
    ],
):
    return __type_of(result).stack_allocation()


@value
@register_passable("trivial")
struct ThreadScope:
    var _value: Int32
    alias BLOCK = Self(0)
    alias WARP = Self(1)

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __str__(self) -> String:
        if self == Self.BLOCK:
            return "BLOCK"
        if self == Self.WARP:
            return "WARP"
        return abort[String]("invalid ThreadScope entry")

    fn __int__(self) -> Int:
        return Int(self._value)


@always_inline("nodebug")
fn _copy_dram_to_sram_validate_args(dst: LayoutTensor, src: LayoutTensor):
    constrained[
        dst.dtype == src.dtype, "src dtype and dst dtype must be the same."
    ]()

    constrained[
        src.address_space
        in (_GPUAddressSpace.GENERIC, _GPUAddressSpace.GLOBAL),
        "src address space must be GENERIC or GLOBAL.",
    ]()

    constrained[
        dst.address_space == _GPUAddressSpace.SHARED,
        "dst address space must be SHARED.",
    ]()


# Synchronous copy from DRAM -> SRAM, this requires w/r thread affinity mapping.
#
@always_inline("nodebug")
fn copy_dram_to_sram[
    src_thread_layout: Layout,
    dst_thread_layout: Layout = src_thread_layout,
    swizzle: OptionalReg[Swizzle] = None,
    num_threads: Int = src_thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](dst: LayoutTensor, src: LayoutTensor):
    _copy_dram_to_sram_validate_args(dst, src)
    alias num_busy_threads = src_thread_layout.size()

    var worker_idx = thread_idx.x if thread_scope == ThreadScope.BLOCK else lane_id()

    @parameter
    if num_threads > num_busy_threads:
        if worker_idx >= num_busy_threads:
            return

    var src_fragments = src.distribute[src_thread_layout](worker_idx)
    var dst_fragments = dst.distribute[dst_thread_layout, swizzle=swizzle](
        worker_idx
    )

    alias simd_width = simdwidthof[dst.dtype]()
    alias src_align = alignof[SIMD[src.dtype, simd_width]]()
    alias dst_align = alignof[SIMD[dst.dtype, simd_width]]()

    alias coalesce_src_element_layout = coalesce(src.element_layout)
    alias coalesce_dst_element_layout = coalesce(dst.element_layout)

    alias is_scalar = not (
        src.element_layout.all_dims_known()
        and coalesce_src_element_layout.rank() == 1
        and coalesce_src_element_layout.stride[0] == 1
        and coalesce_dst_element_layout.rank() == 1
        and coalesce_dst_element_layout.stride[0] == 1
    )

    @parameter
    if not src_fragments.masked or is_scalar:
        constrained[
            dst_fragments.layout.size() == src_fragments.layout.size(),
            "Fragment size mismatch: dst fragments size ("
            + String(dst_fragments.layout.size())
            + ") does not match src fragments size ("
            + String(src_fragments.layout.size())
            + ")",
        ]()

        dst_fragments.copy_from(src_fragments)
    else:
        alias num_stores_per_thread = dst_fragments.layout.size()
        alias static_stride = src.layout.stride[0].value()

        @parameter
        if src.layout.all_dims_known():
            stride = static_stride
        else:
            stride = src.runtime_layout.stride.value[0]
        var src_frag_offset = src_fragments.distance(src.ptr)

        var src_idx_bound = (src.dim(0) * stride - src_frag_offset).cast[
            src_fragments.index_type
        ]()

        @parameter
        for i in range(num_stores_per_thread):
            alias src_static_idx = src_fragments.layout(i)

            alias dst_idx = dst_fragments.layout(i)

            var src_idx: Scalar[src_fragments.index_type] = 0

            @parameter
            if src.layout.all_dims_known():
                src_idx = src_static_idx
            else:
                src_idx = src_fragments.runtime_layout(i)

            if src_idx < src_idx_bound:
                var src_vec = (src.ptr).load[
                    width=simd_width, alignment=src_align
                ](Int(src_frag_offset) + src_idx)
                dst_fragments.ptr.store[alignment=dst_align](
                    dst_idx, src_vec.cast[dst.dtype]()
                )


@always_inline("nodebug")
fn copy_dram_to_sram[
    src_thread_layout: Layout,
    dst_thread_layout: Layout = src_thread_layout,
    swizzle: OptionalReg[Swizzle] = None,
    num_threads: Int = src_thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](dst: LayoutTensor, src: LayoutTensor, src_base: LayoutTensor):
    """
    Used to copy data from DRAM to SRAM for AMD GPUs. It uses buffer_load intrinsic
    to load data and can check for bounds. In addition to dst and src, it takes
    src_base as an argument to construct the buffer descriptor of the src tensor.
    src_base is the original global memory tensor from which src is derived.
    """
    constrained[is_amd_gpu(), "This function is only supported on AMD GPUs."]()
    _copy_dram_to_sram_validate_args(dst, src)
    alias num_busy_threads = src_thread_layout.size()

    var worker_idx = thread_idx.x if thread_scope == ThreadScope.BLOCK else lane_id()

    @parameter
    if num_threads > num_busy_threads:
        if worker_idx >= num_busy_threads:
            return

    var src_fragments = src.distribute[src_thread_layout](worker_idx)
    var dst_fragments = dst.distribute[dst_thread_layout, swizzle=swizzle](
        worker_idx
    )

    alias simd_width = simdwidthof[dst.dtype]()
    alias dst_align = alignof[SIMD[dst.dtype, simd_width]]()

    alias num_stores_per_thread = dst_fragments.layout.size()
    # TODO: Use distance function, it gives parameter mismatch error
    var offset = (Int(src.ptr) - Int(src_base.ptr)) // sizeof[src.dtype]()
    var descriptor = get_amd_buffer_descriptor(src_base)
    var src_frag_offset = src_fragments.distance(src.ptr) + offset

    @parameter
    for i in range(num_stores_per_thread):
        alias src_static_idx = src_fragments.layout(i)
        alias dst_idx = dst_fragments.layout(i)
        var src_idx: Scalar[src_fragments.index_type] = 0

        @parameter
        if src.layout.all_dims_known():
            src_idx = src_static_idx
        else:
            src_idx = src_fragments.runtime_layout(i)
        var src_vec = buffer_load[src.dtype, simd_width](
            descriptor,
            Int32(src_idx + Int(src_frag_offset)),
        )
        dst_fragments.ptr.store[alignment=dst_align](
            dst_idx, src_vec.cast[dst.dtype]()
        )


@always_inline("nodebug")
fn cp_async_k_major[
    type: DType
](
    dst: LayoutTensor[
        type, _, address_space = gpu_memory.AddressSpace.SHARED, *_, **_
    ],
    src: LayoutTensor[
        type, _, address_space = gpu_memory.AddressSpace.GENERIC, *_, **_
    ],
):
    alias dst_layout = dst.layout

    alias src_layout = src.layout
    alias src_shape0 = src_layout.shape[0].value()
    alias src_shape1 = src_layout.shape[1].value()

    alias desc_layout = _tma_desc_tile_layout[
        type,
        2,
        Index(src_shape0, src_shape1),
        is_k_major=True,
        swizzle_mode = TensorMapSwizzle.SWIZZLE_128B,
    ]()
    alias desc_shape0 = desc_layout.shape[0].value()
    alias desc_shape1 = desc_layout.shape[1].value()
    alias desc_size = desc_layout.size()

    constrained[
        desc_shape0 == src_shape0, "k-major desc layout shouldn't alter 1st dim"
    ]()

    alias num_tiles = src_shape1 // desc_shape1
    alias simd_size = simdwidthof[type]()
    # single warp group
    alias thread_layout = Layout.row_major(
        128 * simd_size // desc_shape1, desc_shape1 // simd_size
    )

    @parameter
    for tile_id in range(num_tiles):
        src_tile = src.tile[desc_shape0, desc_shape1](0, tile_id)
        dst_tile = LayoutTensor[
            type, desc_layout, address_space = gpu_memory.AddressSpace.SHARED
        ](dst.ptr + tile_id * desc_size)

        copy_dram_to_sram_async[thread_layout, swizzle=True](
            dst_tile.vectorize[1, simd_size](),
            src_tile.vectorize[1, simd_size](),
        )


@always_inline("nodebug")
fn cp_async_mn_major[
    type: DType
](
    dst: LayoutTensor[
        type, _, address_space = gpu_memory.AddressSpace.SHARED, *_, **_
    ],
    src: LayoutTensor[
        type, _, address_space = gpu_memory.AddressSpace.GENERIC, *_, **_
    ],
):
    alias dst_layout = dst.layout

    alias src_layout = src.layout
    alias src_shape0 = src_layout.shape[0].value()
    alias src_shape1 = src_layout.shape[1].value()

    alias desc_layout = _tma_desc_tile_layout[
        type,
        2,
        Index(src_shape0, src_shape1),
        is_k_major=False,
        swizzle_mode = TensorMapSwizzle.SWIZZLE_128B,
    ]()
    alias desc_shape0 = desc_layout.shape[0].value()
    alias desc_shape1 = desc_layout.shape[1].value()
    alias desc_size = desc_layout.size()

    alias num_tiles0 = src_shape0 // desc_shape0
    alias num_tiles1 = src_shape1 // desc_shape1
    alias num_warps = 4  # single warp group
    alias num_tiles_per_warp = (num_tiles0 * num_tiles1) // num_warps

    alias simd_size = simdwidthof[type]()
    alias thread_layout_per_warp = Layout.row_major(
        gpu_memory.WARP_SIZE * simd_size // desc_shape1,
        desc_shape1 // simd_size,
    )

    warp_id = thread_idx.x // gpu_memory.WARP_SIZE

    @parameter
    for tile_id_per_warp in range(num_tiles_per_warp):
        tile_id = warp_id + UInt(tile_id_per_warp) * num_warps
        tile_coord0, tile_coord1 = divmod(tile_id, UInt(num_tiles1))
        src_tile = src.tile[desc_shape0, desc_shape1](tile_coord0, tile_coord1)
        dst_tile = LayoutTensor[
            type, desc_layout, address_space = gpu_memory.AddressSpace.SHARED
        ](dst.ptr + tile_id * desc_size)

        copy_dram_to_sram_async[thread_layout_per_warp, swizzle=True](
            dst_tile.vectorize[1, simd_size](),
            src_tile.vectorize[1, simd_size](),
        )


# Synchronous copy from DRAM -> SRAM, this requires w/r thread affinity mapping.
#
@always_inline("nodebug")
fn copy_dram_to_sram[
    thread_layout: Layout,
    swizzle: OptionalReg[Swizzle] = None,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](dst: LayoutTensor, src: LayoutTensor, src_base: LayoutTensor):
    copy_dram_to_sram[
        src_thread_layout=thread_layout,
        dst_thread_layout=thread_layout,
        swizzle=swizzle,
        num_threads=num_threads,
        thread_scope=thread_scope,
    ](dst, src, src_base)


@always_inline("nodebug")
fn copy_dram_to_sram[
    thread_layout: Layout,
    swizzle: OptionalReg[Swizzle] = None,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](dst: LayoutTensor, src: LayoutTensor):
    copy_dram_to_sram[
        src_thread_layout=thread_layout,
        dst_thread_layout=thread_layout,
        swizzle=swizzle,
        num_threads=num_threads,
        thread_scope=thread_scope,
    ](dst, src)


# Asynchronous copy from DRAM -> SRAM, this requires w/r thread affinity mapping.
#
@always_inline("nodebug")
fn copy_dram_to_sram_async[
    src_thread_layout: Layout,
    dst_thread_layout: Layout,
    swizzle: Bool = False,
    fill: Fill = Fill.NONE,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    num_threads: Int = src_thread_layout.size(),
](dst: LayoutTensor, src: LayoutTensor):
    constrained[
        src.address_space
        in (_GPUAddressSpace.GENERIC, _GPUAddressSpace.GLOBAL),
        "src address space must be GENERIC or GLOBAL.",
    ]()

    constrained[
        dst.address_space == _GPUAddressSpace.SHARED,
        "dst address space must be SHARED.",
    ]()

    constrained[
        src_thread_layout.size() == dst_thread_layout.size(),
        String(
            "src thread layout size ",
            src_thread_layout.size(),
            " does not match dst thread layout size ",
            dst_thread_layout.size(),
        ),
    ]()

    alias num_busy_threads = src_thread_layout.size()

    # We know at compile time that only partial threads copy based on the size
    # of input tensors. Return if current thread doesn't have work.
    @parameter
    if num_threads > num_busy_threads:
        if thread_idx.x >= num_busy_threads:
            return

    alias row_size = dst.stride[0]()
    # See make_ldmatrix_swizzle in Swizzle.mojo for `conflict_ways`.
    # TODO: use the above when MOCO-1048 is fixed.
    alias bytes_32_banks = 128
    alias conflict_ways = min(
        8 * row_size * sizeof[dst.dtype]() // bytes_32_banks, 8
    )
    constrained[
        (swizzle and (conflict_ways in (4, 8))) or not swizzle,
        "Only support swizzle for 4 or 8 ways conflict.",
    ]()

    constrained[
        (swizzle and row_size in (16, 32, 64, 128, 256)) or not swizzle,
        (
            "Only support 2^4-2^8 elements per row in shared memory tile for"
            " async copy with swizzling."
        ),
    ]()

    alias swizzle_option = None if not swizzle else (
        OptionalReg[Swizzle](
            make_ldmatrix_swizzle[
                dst.dtype, row_size, log2_floor(dst_fragments.element_size)
            ]()
        )
    )

    var src_fragments = src.distribute[src_thread_layout](thread_idx.x)
    var dst_fragments = dst.distribute[dst_thread_layout](thread_idx.x)

    var dst_frag_offset = rebind[dst_fragments.uint_type](
        dst_fragments.distance(dst.ptr) if swizzle else 0
    )

    @parameter
    if not src_fragments.masked:
        dst_fragments.copy_from_async[
            swizzle=swizzle_option, eviction_policy=eviction_policy
        ](src_fragments, base_offset=Int(dst_frag_offset))
    else:
        var src_frag_offset = src_fragments.distance(src.ptr)
        alias src_uint_dtype = _get_unsigned_type(
            src_fragments.layout, src_fragments.address_space
        )

        # Stride between two rows
        alias static_row_stride = Scalar[src_uint_dtype](
            src.layout.stride[0].value()
        )
        var row_stride = static_row_stride

        @parameter
        if src.layout.stride[0].value() == UNKNOWN_VALUE:
            row_stride = Scalar[src_uint_dtype](
                src.runtime_layout.stride.value[0]
            )

        var src_idx_bound = (
            Scalar[src_uint_dtype](src.dim(0)) * row_stride
            - src_frag_offset.cast[src_uint_dtype]()
        ).cast[src_fragments.index_type]()

        dst_fragments.copy_from_async[
            is_masked=True,
            swizzle=swizzle_option,
            eviction_policy=eviction_policy,
        ](
            src_fragments,
            src_idx_bound=rebind[Scalar[src_fragments.index_type]](
                src_idx_bound
            ),
            base_offset=dst_frag_offset,
        )


# Asynchronous copy from DRAM -> SRAM, this requires w/r thread affinity mapping.
#
@always_inline("nodebug")
fn copy_dram_to_sram_async[
    thread_layout: Layout,
    swizzle: Bool = False,
    masked: Bool = False,
    fill: Fill = Fill.NONE,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    num_threads: Int = thread_layout.size(),
](dst: LayoutTensor, src: LayoutTensor):
    copy_dram_to_sram_async[
        src_thread_layout=thread_layout,
        dst_thread_layout=thread_layout,
        swizzle=swizzle,
        eviction_policy=eviction_policy,
        num_threads=num_threads,
    ](dst, src)


alias binary_op_type = fn[type: DType, width: Int] (
    lhs: SIMD[type, width], rhs: SIMD[type, width]
) -> SIMD[type, width]


@always_inline("nodebug")
fn copy_sram_to_dram[
    thread_layout: Layout,
    swizzle: OptionalReg[Swizzle] = None,
    num_threads: Int = thread_layout.size(),
    binary_op: OptionalReg[binary_op_type] = None,
](dst: LayoutTensor, src: LayoutTensor):
    constrained[
        dst.address_space
        in (_GPUAddressSpace.GENERIC, _GPUAddressSpace.GLOBAL),
        "dst address space must be GENERIC or GLOBAL.",
    ]()

    constrained[
        src.address_space == _GPUAddressSpace.SHARED,
        "src address space must be SHARED.",
    ]()

    constrained[
        src.layout.all_dims_known(), "Shared memory must have static layout"
    ]()

    alias num_busy_threads = thread_layout.size()

    @parameter
    if num_threads > num_busy_threads:
        if thread_idx.x >= num_busy_threads:
            return

    var src_fragments = src.distribute[thread_layout](thread_idx.x)
    var dst_fragments = dst.distribute[thread_layout](thread_idx.x)

    # TODO: copy_from only allows static layout
    @parameter
    if src.dtype == dst.dtype and not swizzle:
        dst_fragments.copy_from(src_fragments)
    else:
        constrained[
            src.dtype == dst.dtype
            or (src.dtype == DType.float32 and dst.dtype.is_half_float()),
            "Only support FP32 -> half precision downcast during copy.",
        ]()

        alias simd_size = simdwidthof[dst.dtype]()
        # TODO: generalize the copy to non-scalar case if possible.
        constrained[
            src.element_layout.size() == simd_size
            and dst.element_layout.size() == simd_size,
            "Only FP32 -> half precision downcast for vectorized copy.",
        ]()

        alias src_align = alignof[SIMD[src.dtype, simdwidthof[src.dtype]()]]()
        alias dst_align = alignof[SIMD[dst.dtype, simd_size]]()

        var src_frag_offset = src_fragments.distance(src.ptr)

        alias num_stores_per_thread = dst_fragments.layout.size()

        @parameter
        if not dst_fragments.masked:

            @parameter
            for i in range(num_stores_per_thread):
                alias src_idx = src_fragments.layout(i)
                alias dst_idx = dst_fragments.layout(i)
                var swizzled_idx = src_frag_offset + src_idx

                @parameter
                if swizzle:
                    alias swizzle_fn = swizzle.value()
                    alias src_idx_base = src_idx % swizzle_fn.size()
                    alias src_idx_diff = src_idx - src_idx_base
                    # `src_frag_offset + src_idx_base` should be a value already seen
                    # in the unrolled loop. Hopefully compiler can eleminate the duplicated
                    # xor computation.
                    swizzled_idx = (
                        swizzle_fn(src_frag_offset + src_idx_base)
                        + src_idx_diff
                    )

                var src_vec = src.ptr.load[
                    width=simd_size, alignment=src_align
                ](swizzled_idx).cast[dst.dtype]()

                @parameter
                if binary_op:
                    alias binop = binary_op.value()
                    var dst_vec = dst_fragments.ptr.load[
                        width=simd_size, alignment=dst_align
                    ](dst_idx)
                    src_vec = binop(src_vec, dst_vec)

                dst_fragments.ptr.store[alignment=dst_align](dst_idx, src_vec)
        else:
            alias static_stride = dst.layout.stride[0].value()

            @parameter
            if dst.layout.all_dims_known():
                stride = static_stride
            else:
                stride = dst.runtime_layout.stride.value[0]
            var dst_frag_offset = dst_fragments.distance(dst.ptr)
            var dst_idx_bound = (dst.dim(0) * stride - dst_frag_offset).cast[
                dst_fragments.index_type
            ]()

            @parameter
            for i in range(num_stores_per_thread):
                alias src_idx = src_fragments.layout(i)

                alias dst_uint_dtype = _get_unsigned_type(
                    dst_fragments.layout, dst_fragments.address_space
                )
                alias dst_static_idx = dst_fragments.layout(i)

                var dst_idx: Scalar[dst_fragments.index_type] = 0

                @parameter
                if dst.layout.all_dims_known():
                    dst_idx = dst_static_idx
                else:
                    dst_idx = dst_fragments.runtime_layout(i)

                var swizzled_idx = src_frag_offset + src_idx

                @parameter
                if swizzle:
                    alias swizzle_fn = swizzle.value()
                    alias src_idx_base = src_idx % swizzle_fn.size()
                    alias src_idx_diff = src_idx - src_idx_base
                    # `src_frag_offset + src_idx_base` should be a value already seen
                    # in the unrolled loop. Hopefully compiler can eleminate the duplicated
                    # xor computation.
                    swizzled_idx = (
                        swizzle_fn(src_frag_offset + src_idx_base)
                        + src_idx_diff
                    )

                if dst_idx < dst_idx_bound:
                    var src_vec = (src.ptr).load[
                        width=simd_size, alignment=src_align
                    ](swizzled_idx).cast[dst.dtype]()

                    @parameter
                    if binary_op:
                        alias binop = binary_op.value()
                        var dst_vec = dst_fragments.ptr.load[
                            width=simd_size, alignment=dst_align
                        ](dst_idx)
                        src_vec = binop(src_vec, dst_vec)

                    dst_fragments.ptr.store[alignment=dst_align](
                        dst_idx, src_vec
                    )


# Copy from SRAM to local memory.
#
@always_inline("nodebug")
fn copy_sram_to_local[
    src_warp_layout: Layout,
    axis: OptionalReg[Int] = None,
](dst: LayoutTensor, src: LayoutTensor):
    constrained[
        dst.dtype == src.dtype, "dst dtype must be the same as src dtype."
    ]()

    constrained[
        src.address_space == _GPUAddressSpace.SHARED,
        "src address space must be SHARED.",
    ]()

    constrained[
        dst.address_space == _GPUAddressSpace.LOCAL,
        "dst address space must be LOCAL.",
    ]()

    @parameter
    if axis:
        var src_fragments = src.distribute[
            src_warp_layout, axis = axis.value()
        ](thread_idx.x)
        dst.copy_from(src_fragments)
    else:
        var src_fragments = src.distribute[src_warp_layout](thread_idx.x)
        dst.copy_from(src_fragments)


@always_inline("nodebug")
fn _copy_local_to_dram_validate_args(dst: LayoutTensor, src: LayoutTensor):
    constrained[
        src.address_space == _GPUAddressSpace.LOCAL,
        "src address space must be LOCAL.",
    ]()

    constrained[
        dst.address_space
        in (_GPUAddressSpace.GENERIC, _GPUAddressSpace.GLOBAL),
        "dst address space must be GENERIC or GLOBAL.",
    ]()


# Copy local memory to DRAM, thread affinity is needed only for dst fragments.
#
@always_inline("nodebug")
fn copy_local_to_dram[
    dst_thread_layout: Layout,
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](dst: LayoutTensor, src: LayoutTensor):
    _copy_local_to_dram_validate_args(dst, src)

    var worker_idx = thread_idx.x if thread_scope == ThreadScope.BLOCK else lane_id()
    var dst_fragments = dst.distribute[dst_thread_layout](worker_idx)

    @parameter
    if not dst_fragments.masked:
        dst_fragments.copy_from(src)
    else:
        var dst_frag_offset = dst_fragments.distance(dst.ptr)
        alias static_stride = dst.layout.stride[0].value()

        @parameter
        if dst.layout.all_dims_known():
            stride = static_stride
        else:
            stride = dst.runtime_layout.stride.value[0]
        var dst_idx_bound = (dst.dim(0) * stride - dst_frag_offset).cast[
            dst_fragments.index_type
        ]()

        alias num_stores_per_thread = dst_fragments.layout.size()

        @parameter
        for i in range(num_stores_per_thread):
            alias src_idx = src.layout(i)
            alias dst_uint_dtype = _get_unsigned_type(
                dst_fragments.layout, dst_fragments.address_space
            )
            alias dst_static_idx = dst_fragments.layout(i)

            var dst_idx: Scalar[dst_fragments.index_type] = 0

            @parameter
            if dst_fragments.layout.all_dims_known():
                dst_idx = dst_static_idx
            else:
                dst_idx = dst_fragments.runtime_layout(i)

            if dst_idx < dst_idx_bound:
                var src_element = Element[src.dtype, src.element_layout].load(
                    src.ptr.offset(src_idx),
                    src.runtime_element_layout,
                )
                alias dst_element_type = Element[dst.dtype, dst.element_layout]
                dst_element_type(
                    rebind[dst_element_type.element_data_type](
                        src_element.element_data.cast[dst.dtype]()
                    )
                ).store(dst_fragments.ptr.offset(dst_idx))


@always_inline("nodebug")
fn copy_local_to_dram[
    dst_thread_layout: Layout,
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](dst: LayoutTensor, src: LayoutTensor, dst_base: LayoutTensor):
    """
    Used to copy data from registers to DRAM for AMD GPUs. It uses buffer_store intrinsic
    to store data and can check for bounds. In addition to dst and src, it takes
    dst_base as an argument to construct the buffer descriptor of the dst tensor.
    dst_base is the original global memory tensor from which dst is derived.
    """
    constrained[is_amd_gpu(), "This function is only supported on AMD GPUs."]()
    _copy_local_to_dram_validate_args(dst, src)

    var worker_idx = thread_idx.x if thread_scope == ThreadScope.BLOCK else lane_id()
    var dst_fragments = dst.distribute[dst_thread_layout](worker_idx)

    var offset = (Int(dst.ptr) - Int(dst_base.ptr)) // sizeof[dst.dtype]()
    var descriptor = get_amd_buffer_descriptor(dst_base)
    var dst_frag_offset = dst_fragments.distance(dst.ptr) + offset
    alias num_stores_per_thread = dst_fragments.layout.size()

    @parameter
    for i in range(num_stores_per_thread):
        alias src_idx = src.layout(i)
        alias dst_uint_dtype = _get_unsigned_type(
            dst_fragments.layout, dst_fragments.address_space
        )
        alias dst_static_idx = dst_fragments.layout(i)
        var dst_idx: Scalar[dst_fragments.index_type] = Int(dst_frag_offset)

        @parameter
        if dst_fragments.layout.all_dims_known():
            dst_idx += dst_static_idx
        else:
            dst_idx += dst_fragments.runtime_layout(i)

        var src_element = Element[src.dtype, src.element_layout].load(
            src.ptr.offset(src_idx),
            src.runtime_element_layout,
        )
        alias dst_element_type = Element[dst.dtype, dst.element_layout]

        @parameter
        for i in range(dst_fragments.element_layout.size()):
            alias element_offset = dst_fragments.element_layout(i)
            var src = src_element.element_data[i].cast[dst.dtype]()

            buffer_store(
                descriptor,
                Int32(dst_idx + element_offset),
                src,
            )


@always_inline("nodebug")
fn copy_dram_to_local[
    src_thread_layout: Layout,
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](dst: LayoutTensor, src: LayoutTensor, src_base: LayoutTensor):
    """
    Used to copy data from DRAM to registers for AMD GPUs. It uses buffer_load intrinsic
    to load data and can check for bounds. In addition to dst and src, it takes
    src_base as an argument to construct the buffer descriptor of the src tensor.
    src_base is the original global memory tensor from which src is derived.
    """
    constrained[is_amd_gpu(), "This function is only supported on AMD GPUs."]()
    alias simd_width = simdwidthof[src.dtype]()
    constrained[
        dst.element_layout.size() == simd_width,
        "dst element size must be the same as simd width.",
    ]()
    _copy_local_to_dram_validate_args(src, dst)

    var worker_idx = thread_idx.x if thread_scope == ThreadScope.BLOCK else lane_id()
    var src_fragments = src.distribute[src_thread_layout](worker_idx)
    # the offset calculation using pointer leads to loss of ~7-8 TFlops vs
    # calculating offset and passing it as an argument, using this for now
    # but we may want to revisit this
    var offset = (Int(src.ptr) - Int(src_base.ptr)) // sizeof[src.dtype]()
    var descriptor = get_amd_buffer_descriptor(src_base)
    var src_frag_offset = src_fragments.distance(src.ptr) + offset
    alias num_stores_per_thread = src_fragments.layout.size()

    alias M = src_fragments.shape[0]()
    alias N = src_fragments.shape[1]()

    constrained[
        src_fragments.layout.rank() == 2,
        "src_fragments must be rank 2.",
    ]()

    constrained[
        src_fragments.layout.all_dims_known(),
        "src_fragments must have known layout.",
    ]()

    # These loads need to be row-major for L1 cache performance
    @parameter
    for i in range(M):

        @parameter
        for j in range(N):
            alias dst_idx = Layout.col_major(M, N)(IntTuple(i, j))
            alias src_static_idx = src_fragments.layout(IntTuple(i, j))
            var src_idx = Int32(src_frag_offset) + src_static_idx
            dst[dst_idx] = rebind[dst.element_type](
                buffer_load[src.dtype, simd_width](
                    descriptor,
                    src_idx,
                )
            )


@always_inline("nodebug")
fn copy_local_to_sram[
    thread_layout: Layout,
    swizzle: OptionalReg[Swizzle] = None,
    thread_scope: ThreadScope = ThreadScope.BLOCK,
    row_major: Bool = False,
    # row_major is used when using prefetching from dram to sram via registers for AMD GPUs
](dst: LayoutTensor, src: LayoutTensor):
    constrained[
        dst.address_space == _GPUAddressSpace.SHARED,
        "dst address space must be SHARED.",
    ]()

    constrained[
        src.address_space == _GPUAddressSpace.LOCAL,
        "src address space must be LOCAL.",
    ]()

    var worker_idx = thread_idx.x if thread_scope == ThreadScope.BLOCK else lane_id()

    constrained[
        src.dtype == dst.dtype
        or (src.dtype == DType.float32 and dst.dtype.is_half_float()),
        "Only support FP32 -> half precision downcast during copy.",
    ]()
    constrained[
        src.element_size == dst.element_size,
        "src and dst element size mismatch.",
    ]()

    @parameter
    if not row_major:
        var dst_frag = dst.distribute[thread_layout](worker_idx)

        @parameter
        if swizzle:
            alias swizzle_fn = swizzle.value()
            alias num_vecs = src.layout.size()
            alias align_src = alignof[SIMD[src.dtype, src.element_size]]()
            alias align_dst = alignof[SIMD[dst.dtype, dst.element_size]]()
            var dst_frag_offset = dst_frag.distance(dst.ptr)

            @parameter
            for i in range(num_vecs):
                alias src_idx = src.layout(i)
                alias dst_idx = dst_frag.layout(i)
                alias dst_idx_base = dst_idx % swizzle_fn.size()
                alias dst_idx_diff = dst_idx - dst_idx_base
                var swizzled_idx = swizzle_fn(
                    dst_frag_offset + dst_idx_base
                ) + dst_idx_diff
                var src_vec = src.ptr.load[
                    width = src.element_size, alignment=align_src
                ](src_idx).cast[dst.dtype]()
                dst.ptr.store[alignment=align_dst](
                    swizzled_idx, src_vec.cast[dst.dtype]()
                )

        else:
            dst_frag.copy_from(src)
    else:
        constrained[
            is_amd_gpu(), "This function is only supported on AMD GPUs."
        ]()
        var dst_frag = dst.distribute[thread_layout, swizzle=swizzle](
            worker_idx
        )
        alias M = dst_frag.shape[0]()
        alias N = dst_frag.shape[1]()

        constrained[
            dst_frag.layout.rank() == 2,
            "dst_frag must be rank 2.",
        ]()

        @parameter
        for i in range(M):

            @parameter
            for j in range(N):
                # The order here needs to match the order of the loads in copy_dram_to_local
                alias idx = Layout.col_major(M, N)(IntTuple(i, j))
                var src_idx = src._get_element_idx[idx]()
                var dst_idx = dst_frag._get_element_idx[idx]()

                var src_element = MemoryElement(
                    src.ptr.offset(src_idx), src.runtime_element_layout
                )

                var dst_element = MemoryElement(
                    dst_frag.ptr.offset(dst_idx),
                    dst_frag.runtime_element_layout,
                )
                dst_element.transfer(src_element)


@always_inline
fn copy_local_to_local(dst: LayoutTensor, src: LayoutTensor):
    constrained[
        dst.address_space == _GPUAddressSpace.LOCAL,
        "dst address space must be LOCAL.",
    ]()

    constrained[
        src.address_space == _GPUAddressSpace.LOCAL,
        "src address space must be LOCAL.",
    ]()

    constrained[
        dst.dtype.is_half_float() and src.dtype == DType.float32,
        "Only support copy float32 to bfloat16 for now",
    ]()

    constrained[
        dst.layout.size() == src.layout.size(),
        "dst and src should have the same size.",
    ]()

    # Fast for 2D fragments
    @parameter
    if (
        dst.rank == 2
        and src.rank == 2
        and dst.stride[1]() == 1
        and src.stride[1]() == 1
    ):
        # This path is to map 16x8x16 mma output (16x8) to 16x8x16 mma input (16x16).
        # Output fragment has layout [2 * num_m_mmas, 4]
        # Input  fragment has layout [num_m_mmas, 8]
        alias num_mmas = src.layout.shape[0].value()
        alias src_frag_size = src.layout.shape[1].value()
        alias a_frag_layout = composition(
            src.layout,
            make_layout(Layout.row_major(num_mmas // 2, 2), src.layout[1]),
        )
        # [num_m_mmas, 8] vectorized and transposed to [2, num_m_mmas] x 4
        var dst_vectorized = dst.vectorize[1, src_frag_size]().transpose()
        # [2*num_m_mmas, 4] reshaped and vectorized row_major(num_m_mmas, 2) x 4
        var src_vectorized = src.reshape[a_frag_layout]().vectorize[
            1, src_frag_size
        ]()

        @parameter
        for i in range(dst_vectorized.layout.size()):
            alias dst_idx = dst_vectorized.layout(i)
            alias src_idx = src_vectorized.layout(i)

            dst_vectorized.ptr.store(
                dst_idx,
                src_vectorized.ptr.load[width=src_frag_size](src_idx).cast[
                    dst.dtype
                ](),
            )

    # Default elementwise copy
    else:

        @parameter
        for i in range(dst.layout.size()):
            alias dst_idx = dst.layout(i)
            alias src_idx = src.layout(i)
            dst.ptr.store(dst_idx, src.ptr[src_idx].cast[dst.dtype]())


# ===-----------------------------------------------------------------------===#
# LayoutTensorIter                                                             #
# ===-----------------------------------------------------------------------===#


@register_passable("trivial")
struct LayoutTensorIter[
    mut: Bool, //,
    type: DType,
    layout: Layout,
    origin: Origin[mut],
    /,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
    alignment: Int = alignof[type]() if is_nvidia_gpu() else 1,
    circular: Bool = False,
    axis: OptionalReg[Int] = None,
    layout_bitwidth: Int = bitwidthof[_get_index_type(address_space)](),
    masked: Bool = False,
]:
    """Iterate through a memory buffer and construct layout tensor.

    The returned layout tensor is NOT vectorized. User should explicitly vectorize.
    """

    alias uint_type = Scalar[_get_unsigned_type(layout, address_space)]

    var ptr: UnsafePointer[
        Scalar[type],
        address_space=address_space,
        alignment=alignment,
        mut=mut,
        origin=origin,
    ]
    var offset: Self.uint_type
    var stride: Self.uint_type
    var bound: Self.uint_type
    var runtime_layout: RuntimeLayout[layout, bitwidth=layout_bitwidth]
    var dimension_bound: Self.uint_type
    var idx: Self.uint_type

    @always_inline
    fn __init__(out self):
        """Empty iterator, used as default value."""

        @parameter
        if axis:
            constrained[
                not circular,
                "Circular use case is not supported if an axis is defined.",
            ]()

        self.ptr = __type_of(self.ptr)()
        self.offset = 0
        self.stride = 0
        self.bound = 0
        self.runtime_layout = RuntimeLayout[layout, bitwidth=layout_bitwidth]()
        self.dimension_bound = 0
        self.idx = 0

    @always_inline
    fn __init__(
        out self,
        ptr: UnsafePointer[
            Scalar[type],
            address_space=address_space,
            alignment=alignment,
            mut=mut,
            origin=origin,
        ],
        bound: Self.uint_type,
        stride: Self.uint_type = layout.size(),
        offset: Self.uint_type = 0,
    ):
        constrained[
            layout.all_dims_known(),
            "Cannot construct LayoutTensorIter with unknown layout.",
        ]()

        self.ptr = ptr
        self.bound = bound
        self.stride = stride
        self.runtime_layout = RuntimeLayout[layout, bitwidth=layout_bitwidth]()
        self.offset = offset
        self.dimension_bound = 0
        self.idx = 0

    @always_inline
    fn __init__(
        out self,
        ptr: UnsafePointer[
            Scalar[type],
            address_space=address_space,
            alignment=alignment,
            mut=mut,
            origin=origin,
        ],
        bound: Self.uint_type,
        runtime_layout: RuntimeLayout[layout, **_],
        stride: Self.uint_type = layout.size() if layout.all_dims_known() else UNKNOWN_VALUE,
        offset: Self.uint_type = 0,
        dimension_bound: Self.uint_type = 0,
        idx: Self.uint_type = 0,
    ):
        constrained[
            runtime_layout.bitwidth == layout_bitwidth,
            "Mismatch of bitwidth for RuntimeLayout and LayoutTensorIter.",
        ]()

        @parameter
        if axis:
            constrained[
                not circular,
                "Circular use case is not supported if an axis is defined.",
            ]()

        self.ptr = ptr
        self.offset = offset
        self.stride = (
            runtime_layout.size() if stride == UNKNOWN_VALUE else stride
        )
        self.bound = bound
        self.runtime_layout = rebind[
            RuntimeLayout[layout, bitwidth=layout_bitwidth]
        ](runtime_layout)
        self.dimension_bound = dimension_bound
        self.idx = idx

    @always_inline
    fn get(
        self,
        out result: LayoutTensor[
            type,
            layout,
            origin,
            address_space=address_space,
            masked=masked,
            alignment=alignment,
        ],
    ):
        """Return the layout tensor at current iterator."""
        # TODO: Use deref `[]` to be consistent with mojo feature.

        return __type_of(result)(
            self.ptr + Int(self.offset),
            rebind[RuntimeLayout[layout, bitwidth = result.layout_bitwidth]](
                self.runtime_layout
            ),
        )

    @always_inline
    fn __getitem__(
        self,
    ) -> LayoutTensor[
        type,
        layout,
        origin,
        address_space=address_space,
        masked=masked,
        alignment=alignment,
    ]:
        """Return the layout tensor at current iterator."""
        return self.get()

    @always_inline
    fn _clip_shape(self) -> RuntimeLayout[layout, bitwidth=layout_bitwidth]:
        new_shape = self.runtime_layout.shape
        var cur_dim = new_shape.value[axis.value()]
        new_shape.value[axis.value()] = max(
            0, min(Int(self.dimension_bound - self.idx * cur_dim), cur_dim)
        )
        return RuntimeLayout(new_shape, self.runtime_layout.stride)

    @always_inline
    fn __iadd__[T: Intable](mut self, rhs: T):
        """Increment the iterator.

        This function is unsafe. It omits bound checking for performance reasons.
        Caller must make sure index doesn't go out-of-bound.
        """
        self += Self.uint_type(Int(rhs))

    @always_inline
    fn __iadd__(mut self, rhs: Self.uint_type):
        """Increment the iterator.

        This function is unsafe. It omits bound checking for performance reasons.
        Caller must make sure index doesn't go out-of-bound.
        """
        self.offset += rhs * self.stride

        @parameter
        if axis:
            self.idx += rhs

        @parameter
        if masked:
            self.runtime_layout = self._clip_shape()

        @parameter
        if circular:
            self.offset = self.offset % self.bound

    @always_inline
    fn _incr(mut self):
        """Increment the iterator by 1. Equivalent to `iter += 1` but w/o the division.
        """
        self.offset += self.stride

        @parameter
        if circular:
            self.offset = (
                self.offset - self.bound if self.offset
                >= self.bound else self.offset
            )

    @always_inline
    fn next[T: Intable](self, rhs: T) -> Self:
        """Return an iterator pointing to the next `rhs` layout tensor."""

        var next_idx = Self.uint_type(0)
        var next_offset = self.offset + Int(rhs) * self.stride

        @parameter
        if axis:
            next_idx = self.idx + Int(rhs)

        @parameter
        if masked:
            runtime_layout = self._clip_shape()
        else:
            runtime_layout = self.runtime_layout

        @parameter
        if circular:
            next_offset = next_offset % self.bound

        return Self(
            self.ptr,
            self.bound,
            stride=self.stride,
            offset=Int(next_offset),
            runtime_layout=runtime_layout,
            dimension_bound=self.dimension_bound,
            idx=next_idx,
        )

    @always_inline
    fn next(self, rhs: Self.uint_type = 1) -> Self:
        return self.next(Int(rhs))

    @always_inline
    fn next_unsafe(self, rhs: Self.uint_type = 1) -> Self:
        """Return an iterator pointing to the next `rhs` layout tensor.
        This is the unsafe version and user must ensure rhs < bound / stride.
        """
        constrained[
            not masked, "Cannot use unsafe increment for masked iterator."
        ]()

        var next_offset = self.offset + rhs * self.stride

        @parameter
        if circular:
            next_offset = (
                next_offset - self.bound if next_offset
                >= self.bound else next_offset
            )

        return Self(
            self.ptr,
            self.bound,
            stride=self.stride,
            offset=next_offset,
        )

    @always_inline
    fn reshape[
        dst_layout: Layout,
    ](
        self,
        out result: LayoutTensorIter[
            type,
            dst_layout,
            origin,
            address_space=address_space,
            alignment=alignment,
            circular=circular,
            layout_bitwidth=layout_bitwidth,
            masked=masked,
        ],
    ):
        """Reshape the iterator to a new layout.

        This method creates a new iterator with a different layout while preserving the
        underlying data. The new layout must have the same total size as the original.

        Parameters:
            dst_layout: The target layout to reshape to.

        Returns:
            A new iterator with the specified layout.

        Constraints:
            - The destination layout must have the same total size as the original.
            - Both layouts must be contiguous.
            - Both layouts must have compile-time known dimensions.
        """
        constrained[
            dst_layout.size() == layout.size(),
            "Destination layout doesn't match the original.",
        ]()

        constrained[
            dst_layout.size() == dst_layout.cosize()
            and layout.size() == layout.cosize(),
            "Iterator reshape only supports contiguous layout.",
        ]()

        constrained[
            layout.all_dims_known() and dst_layout.all_dims_known(),
            "Iterator reshape only supports compile time layout.",
        ]()

        return __type_of(result)(
            self.ptr,
            Int(self.bound),
            RuntimeLayout[dst_layout, bitwidth=layout_bitwidth](),
            Int(self.stride),
            Int(self.offset),
            dimension_bound=Int(self.dimension_bound),
            idx=Int(self.idx),
        )

    @always_inline
    fn bitcast[
        new_type: DType,
        *,
        address_space: AddressSpace = Self.address_space,
        alignment: Int = Self.alignment,
    ](
        self,
        out result: LayoutTensorIter[
            new_type,
            layout,
            origin,
            address_space=address_space,
            alignment=alignment,
            circular = Self.circular,
            layout_bitwidth=layout_bitwidth,
            masked=masked,
        ],
    ):
        """Reinterpret the iterator's underlying pointer as a different data type.

        This method performs a bitcast operation, allowing you to view the same
        memory location as a different data type without copying or converting
        the data.

        Parameters:
            new_type: The target data type to cast to.
            address_space: The memory address space for the new
              iterator (defaults to current).
            alignment: Memory alignment requirement for the new
              iterator (defaults to current).

        Returns:
            A new LayoutTensorIter with the same layout but different data type.
        """
        return __type_of(result)(
            self.ptr.bitcast[Scalar[new_type]]()
            .address_space_cast[address_space]()
            .static_alignment_cast[alignment](),
            Int(self.bound),
            self.runtime_layout,
            Int(self.stride),
            Int(self.offset),
            dimension_bound=Int(self.dimension_bound),
            idx=Int(self.idx),
        )
