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
"""Provides the `LayoutTensor` type for representing multidimensional data.
"""
from collections import OptionalReg
from math import align_up, ceildiv, exp
from math.math import _Expable
from sys import (
    align_of,
    is_amd_gpu,
    is_nvidia_gpu,
    prefetch,
    simd_width_of,
    size_of,
)
from sys.intrinsics import PrefetchOptions, readfirstlane

import gpu.memory as gpu_memory
from algorithm import vectorize
from bit import log2_floor
from builtin.device_passable import DevicePassable
from builtin.dtype import _unsigned_integral_type_of
from gpu.host import DeviceBuffer, HostBuffer, DeviceContext
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.id import block_dim, block_idx, lane_id, thread_idx
from gpu.intrinsics import AMDBufferResource
from gpu.memory import CacheEviction, CacheOperation, Fill, async_copy
from layout._fillers import BATCH_SIZE
from layout._utils import make_amd_buffer_resource
from layout.element import Element, MemoryElement
from layout.tma_async import _tma_desc_tile_layout
from memory import stack_allocation
from memory.pointer import _GPUAddressSpace

from utils import IndexList, StaticTuple
from utils.index import Index

from .int_tuple import (
    _get_index_type,
    _get_layout_type,
    _get_unsigned_type,
    congruent,
    depth,
    fill_like,
    flatten,
    product,
    propagate_unknown,
    to_nest,
)
from .layout import *
from .runtime_layout import RuntimeLayout
from .runtime_layout import make_layout as make_runtime_layout
from .runtime_tuple import RuntimeTuple
from .swizzle import Swizzle, make_ldmatrix_swizzle


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


alias _swizzle_signature = fn[dtype: DType] (Scalar[dtype]) -> Scalar[dtype]


fn _get_slice_size(layout: Layout, slc: Slice, dim: Int) -> Int:
    """Calculates the size of a slice in a specific layout dimension.

    Computes the number of elements in a slice for a given dimension of the
    layout. This function handles the conversion between slice notation and
    actual element counts.

    Args:
        layout: The layout containing the dimension information.
        slc: The slice specification (start:end:step).
        dim: The dimension index to slice.

    Returns:
        The number of elements in the slice for the specified dimension.
    """
    var start, end, _ = slc.indices(Int(layout.shape[dim]))
    return end - start


fn _not_in_tuple[n: Int, size: Int, tuple: IndexList[size]]() -> Bool:
    """Checks if a value is *not* present in an `IndexList`.

    This utility function searches through an `IndexList` to determine if a
    specific value is absent. Used for dimension validation and filtering
    operations.

    Parameters:
        n: The value to check for in the `IndexList`.
        size: The size of the `IndexList`.
        tuple: The `IndexList` to search in.

    Returns:
        True if the value is not found in the `IndexList`, False if it is
        present.
    """

    @parameter
    for i in range(size):

        @parameter
        if tuple[i] == n:
            return False
    return True


fn _tile_is_masked[layout: Layout, *tile_sizes: Int]() -> Bool:
    """Determines if a tiled layout requires masked access.

    When tiling a tensor, this function checks if any dimension of the layout is
    not evenly divisible by its corresponding tile size. If any dimension
    requires padding, masked access is needed to prevent out-of-bounds memory
    accesses.

    Parameters:
        layout: The layout to check for divisibility.
        tile_sizes: The tile sizes for each dimension of the layout.

    Returns:
        True if masked access is required (any dimension not evenly divisible),
        False if all dimensions are perfectly divisible by their tile sizes.
    """

    @parameter
    if not layout.all_dims_known():
        return True

    @parameter
    for axis in range(layout.rank()):
        alias dim = product(layout.shape[axis])

        @parameter
        if dim % tile_sizes[axis] != 0:
            return True
    return False


fn _distribute_is_masked[
    layout: Layout, threads_layout: Layout, axis: OptionalReg[Int] = None
]() -> Bool:
    """Determines if a distributed layout requires masked access.

    When distributing computation across threads, this function checks if the
    layout's dimensions are evenly divisible by the corresponding thread
    dimensions. Masked access is required when dimensions don't divide evenly to
    prevent out-of-bounds accesses.

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
        alias layout_dim = Int(product(layout.shape[i]))
        alias thread_dim = Int(product(threads_layout.shape[i]))

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
    layout_int_type: DType = _get_layout_type(layout, address_space),
    linear_idx_type: DType = _get_index_type(layout, address_space),
    masked: Bool = False,
    alignment: Int = align_of[dtype](),
](
    ImplicitlyCopyable,
    DevicePassable,
    Movable,
    Stringable,
    Writable,
    _Expable,
):
    """A high-performance tensor with explicit memory layout and
    hardware-optimized access patterns.

    `LayoutTensor` provides a powerful abstraction for multi-dimensional data
    with precise control over memory organization. It supports various memory
    layouts (row-major, column-major, tiled), hardware-specific optimizations,
    and efficient parallel access patterns.

    Parameters:
        mut: The inferred mutability of the underlying pointer.
        dtype: The data type of the underlying pointer.
        layout: The memory layout of the tensor.
        origin: The origin of the underlying pointer.
        address_space: The address space of the underlying pointer.
        element_layout: The memory layout of each element in the tensor.
        layout_int_type: The integer type of each dimension of runtime layout.
        linear_idx_type: The integer type of the index pointing to memory
            locations.
        masked: If true the tensor is masked and runtime layouts determine the
            shape.
        alignment: Alignment of the data pointer.

    Example:

    ```mojo
    from layout import Layout, LayoutTensor

    # Create tensor on CPU using InlineArray to allocate storage space.
    var storage = InlineArray[Float32, 5 * 4](uninitialized=True)
    var tensor_5x4 = LayoutTensor[DType.float32, Layout.row_major(5, 4)](storage)
    ```
    """

    # `trait DevicePassable` implementation, to allow LayoutTensor to be passed directly to kernels
    alias device_type: AnyTrivialRegType = Self

    fn _to_device_type(self, target: OpaquePointer):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        """
        Gets the name of the host type (the one implementing this trait).

        Returns:
            The host type's name.
        """
        return (
            "LayoutTensor[mut = "
            + String(mut)
            + ", dtype = "
            + String(dtype)
            + ", layout = "
            + String(layout)
            + ", address_space = "
            + String(address_space)
            + "]"
        )

    @staticmethod
    fn get_device_type_name() -> String:
        """
        Gets device_type's name.

        Returns:
            The device type's name.
        """
        return Self.get_type_name()

    alias rank = layout.rank()
    """The number of dimensions in the tensor's layout."""

    var ptr: UnsafePointer[
        Scalar[dtype],
        address_space=address_space,
        mut=mut,
        origin=origin,
    ]
    """Pointer to the underlying memory buffer containing the tensor data.

    This pointer respects the specified address space, alignment, mutability,
    and origin tracking for memory safety and performance optimization."""

    alias RuntimeLayoutType = RuntimeLayout[
        layout,
        element_type=layout_int_type,
        linear_idx_type=linear_idx_type,
    ]

    var runtime_layout: Self.RuntimeLayoutType
    """Runtime representation of the tensor's memory layout.

    Handles both compile-time and runtime-determined dimensions, enabling
    efficient mapping between logical tensor coordinates and physical memory
    locations."""

    alias RuntimeElementLayoutType = RuntimeLayout[
        element_layout,
        element_type = DType.int32,
        linear_idx_type=linear_idx_type,
    ]

    var runtime_element_layout: Self.RuntimeElementLayoutType
    """Runtime representation of each element's internal layout.

    Used when elements themselves have structure, such as in blocked or tiled
    layouts."""

    alias element_size = element_layout.size()
    """The number of scalar values in each element of the tensor."""

    alias element_type = SIMD[dtype, Self.element_size]
    """The SIMD vector type used for vectorized operations on tensor elements."""

    alias num_strides: Int = Self.RuntimeLayoutType.StrideType.scalar_length
    alias idx_list_t[rank: Int = Self.rank] = IndexList[
        rank, element_type = Self.linear_idx_type
    ]

    # ===------------------------------------------------------------------=== #
    # Life cycle methods
    # ===------------------------------------------------------------------=== #
    @always_inline
    fn __init__(
        out self,
        span: Span[
            Scalar[dtype],
            origin,
            address_space=address_space, **_,
        ],
    ):
        """Create a `LayoutTensor` with a `Span`.

        Constraints:
            Layout must be fully static.

        Args:
            span: The `Span` pointing to the underlying data.
        """
        self = Self(span.unsafe_ptr())

    @always_inline
    fn __init__(
        out self,
        span: Span[
            Scalar[dtype],
            origin,
            address_space=address_space, **_,
        ],
        runtime_layout: RuntimeLayout[layout, **_],
    ):
        """Create a `LayoutTensor` with a `Span` and a runtime layout
        for the tensor. The runtime layout element type will be casted to the
        layout tensor layout integer type.

        Constraints:
            - Element layout must be fully static.

        Args:
            span: The `Span` pointing to the underlying data.
            runtime_layout: The runtime layout of the LayoutTensor.
        """
        self = Self(span.unsafe_ptr(), runtime_layout)

    @always_inline
    fn __init__(
        out self,
        span: Span[
            Scalar[dtype],
            origin,
            address_space=address_space, **_,
        ],
        runtime_layout: RuntimeLayout[layout, **_],
        element_runtime_layout: RuntimeLayout[element_layout, **_],
    ):
        """Create a `LayoutTensor` with a `Span`, a runtime layout of
        the tensor, and the runtime layout of each element. The runtime layout
        element type will be casted to the layout tensor layout integer type.

        Constraints:
            - Runtime layout and `LayoutTensor` must have the same bitwidth and
                index type.

        Args:
            span: The `Span` pointing to the underlying data.
            runtime_layout: The runtime layout of the `LayoutTensor`.
            element_runtime_layout: The runtime layout of each element.
        """
        self = Self(span.unsafe_ptr(), runtime_layout, element_runtime_layout)

    @always_inline
    fn __init__(
        out self,
        unsafe_ptr: UnsafePointer[
            Scalar[dtype],
            address_space=address_space,
            mut=mut,
            origin=origin, **_,
        ],
    ):
        """Create a `LayoutTensor` with an `UnsafePointer`.

        Constraints:
            Layout must be fully static.

        Args:
            unsafe_ptr: The `UnsafePointer` pointing to the underlying data.
        """

        constrained[layout.all_dims_known(), "Layout must be fully static"]()

        constrained[
            layout_int_type.is_signed() and linear_idx_type.is_signed(),
            "Layout integer type and linear index type must be signed.",
        ]()

        self.ptr = unsafe_ptr
        self.runtime_layout = {}
        self.runtime_element_layout = {}

    @always_inline
    fn __init__(
        out self,
        unsafe_ptr: UnsafePointer[
            Scalar[dtype],
            address_space=address_space,
            mut=mut,
            origin=origin, **_,
        ],
        runtime_layout: RuntimeLayout[layout, **_],
    ):
        """Create a `LayoutTensor` with an `UnsafePointer` and a runtime layout
        for the tensor. The runtime layout element type will be casted to the
        layout tensor layout integer type.

        Constraints:
            Element layout must be fully static.

        Args:
            unsafe_ptr: The UnsafePointer pointing to the underlying data.
            runtime_layout: The runtime layout of the LayoutTensor.
        """

        constrained[
            element_layout.all_dims_known(), "Layout must be fully static"
        ]()

        self.ptr = unsafe_ptr
        self.runtime_layout = runtime_layout.cast[
            layout_int_type, target_linear_idx_type=linear_idx_type
        ]()
        self.runtime_element_layout = {}

    @always_inline
    fn __init__(
        out self,
        unsafe_ptr: UnsafePointer[
            Scalar[dtype],
            address_space=address_space,
            mut=mut,
            origin=origin, **_,
        ],
        runtime_layout: RuntimeLayout[layout, **_],
        element_runtime_layout: RuntimeLayout[element_layout, **_],
    ):
        """Create a `LayoutTensor` with an `UnsafePointer`, a runtime layout for
        the tensor, and the runtime layout of each element. The runtime layout
        element type will be casted to the layout tensor layout integer type.

        Args:
            unsafe_ptr: The `UnsafePointer` pointing to the underlying data.
            runtime_layout: The runtime layout of the `LayoutTensor`.
            element_runtime_layout: The runtime layout of each element.
        """

        self.ptr = unsafe_ptr
        self.runtime_layout = runtime_layout.cast[
            layout_int_type, target_linear_idx_type=linear_idx_type
        ]()
        self.runtime_element_layout = element_runtime_layout.cast[
            DType.int32, target_linear_idx_type=linear_idx_type
        ]()

    alias GenericLayoutTensorType = LayoutTensor[
        dtype,
        layout,
        origin,
        address_space = AddressSpace.GENERIC,
        element_layout=element_layout,
        layout_int_type=layout_int_type,
        linear_idx_type=linear_idx_type,
        masked=masked,
        alignment=alignment,
    ]

    @always_inline
    fn __init__(
        out self: Self.GenericLayoutTensorType,
        ref [origin]device_buffer: DeviceBuffer[dtype],
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
        from layout import Layout, LayoutTensor

        alias dtype = DType.float32

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

        # Create LayoutTensor to use on device
        alias layout = Layout.row_major(4, 4)
        var tensor = LayoutTensor[dtype, layout](dev_buf)
        ...
        ```

        Constraints:
            - Layout must be fully static.

        Args:
            device_buffer: Contains the underlying data to point to.
        """
        self = Self.GenericLayoutTensorType(device_buffer.unsafe_ptr())

    @always_inline
    fn __init__(
        out self: Self.GenericLayoutTensorType,
        ref [origin]host_buffer: HostBuffer[dtype],
    ):
        """Create a `LayoutTensor` from a `HostBuffer`. The layout must have
        statically known dimensions.

        The resulting tensor's data can only be accessed on the CPU.

        ```mojo
        from gpu.host import DeviceContext, HostBuffer
        from layout import Layout, LayoutTensor

        alias dtype = DType.float32

        var ctx = DeviceContext()
        var dev_buf = ctx.enqueue_create_host_buffer[dtype](8)

        alias layout = Layout.row_major(4, 4)
        var tensor = LayoutTensor[dtype, layout](dev_buf)
        ```

        Constraints:
            - Layout must be fully static.

        Args:
            host_buffer: Contains the underlying data to point to.
        """
        self = Self.GenericLayoutTensorType(host_buffer.unsafe_ptr())

    @always_inline
    fn __init__(
        out self: Self.GenericLayoutTensorType,
        ref [origin]device_buffer: DeviceBuffer[dtype],
        runtime_layout: RuntimeLayout[layout, **_],
    ):
        """Create a `LayoutTensor` from a `DeviceBuffer` and a runtime layout.
        The runtime layout element type will be casted to the layout tensor layout
        integer type.

        The resulting tensor's data can only be accessed on the GPU.

        Constraints:
            - Element layout must be fully static.

        Args:
            device_buffer: The `DeviceBuffer` containing to the underlying data.
            runtime_layout: The runtime layout of the LayoutTensor.
        """
        self = Self.GenericLayoutTensorType(
            device_buffer.unsafe_ptr(), runtime_layout
        )

    @always_inline
    fn __init__(
        out self: Self.GenericLayoutTensorType,
        ref [origin]host_buffer: HostBuffer[dtype],
        runtime_layout: RuntimeLayout[layout, **_],
    ):
        """Create a `LayoutTensor` from a `HostBuffer` and a runtime layout.
        The runtime layout element type will be casted to the layout tensor layout
        integer type.

        The resulting tensor's data can only be accessed on the CPU.

        Constraints:
            - Element layout must be fully static.

        Args:
            host_buffer: The `HostBuffer` containing to the underlying data.
            runtime_layout: The runtime layout of the `LayoutTensor`.
        """
        self = Self.GenericLayoutTensorType(
            host_buffer.unsafe_ptr(), runtime_layout
        )

    @always_inline
    fn __init__(
        out self: Self.GenericLayoutTensorType,
        ref [origin]device_buffer: DeviceBuffer[dtype],
        runtime_layout: RuntimeLayout[layout, **_],
        element_runtime_layout: RuntimeLayout[element_layout, **_],
    ):
        """Create a `LayoutTensor` from a `DeviceBuffer`, a runtime layout for
        the tensor, and the runtime layout of each element. The runtime layout
        element type will be casted to the layout tensor layout integer type.

        The resulting tensor's data can only be accessed on the GPU.

        Args:
            device_buffer: The `DeviceBuffer` containing to the underlying data.
            runtime_layout: The runtime layout of the `LayoutTensor`.
            element_runtime_layout: The runtime layout of each element.
        """
        self = Self.GenericLayoutTensorType(
            device_buffer.unsafe_ptr(), runtime_layout, element_runtime_layout
        )

    @always_inline
    fn __init__(
        out self: Self.GenericLayoutTensorType,
        ref [origin]host_buffer: HostBuffer[dtype],
        runtime_layout: RuntimeLayout[layout, **_],
        element_runtime_layout: RuntimeLayout[element_layout, **_],
    ):
        """Create a `LayoutTensor` from a `HostBuffer`, a runtime layout for the
        tensor, and the runtime layout of each element. The runtime layout
        element type will be casted to the layout tensor layout integer type.

        The resulting tensor's data can only be accessed on the CPU.

        Args:
            host_buffer: The `HostBuffer` containing to the underlying data.
            runtime_layout: The runtime layout of the `LayoutTensor`.
            element_runtime_layout: The runtime layout of each element.
        """
        self = Self.GenericLayoutTensorType(
            host_buffer.unsafe_ptr(), runtime_layout, element_runtime_layout
        )

    @always_inline("nodebug")
    fn __merge_with__[
        other_type: __type_of(
            LayoutTensor[
                dtype,
                layout,
                _,
                address_space=address_space,
                alignment=alignment,
                element_layout=element_layout,
                layout_int_type=layout_int_type,
                linear_idx_type=linear_idx_type,
                masked=masked,
            ]
        ),
    ](
        self,
        out result: LayoutTensor[
            mut = mut & other_type.origin.mut,
            dtype,
            layout,
            __origin_of(origin, other_type.origin),
            alignment=alignment,
            address_space=address_space,
            element_layout=element_layout,
            layout_int_type=layout_int_type,
            linear_idx_type=linear_idx_type,
            masked=masked,
        ],
    ):
        """Returns a tensor merged with the specified `other_type`.

        Parameters:
            other_type: The type of the tensor to merge with.

        Returns:
            A tensor merged with the specified `other_type`.
        """
        return {
            self.ptr.unsafe_mut_cast[result.mut]().unsafe_origin_cast[
                result.origin
            ](),
            self.runtime_layout,
            self.runtime_element_layout,
        }

    alias BitcastType[
        new_dtype: DType,
        /,
        address_space: AddressSpace = Self.address_space,
        element_layout: Layout = Self.element_layout,
    ] = LayoutTensor[
        new_dtype,
        layout,
        origin,
        address_space=address_space,
        element_layout=element_layout,
        layout_int_type=layout_int_type,
        linear_idx_type=linear_idx_type,
        masked=masked,
    ]

    @always_inline
    fn bitcast[
        new_dtype: DType,
        /,
        target_address_space: AddressSpace = Self.address_space,
        _element_layout: Layout = Self.element_layout,
    ](self) -> Self.BitcastType[
        new_dtype, target_address_space, _element_layout
    ]:
        """Bitcast the underlying pointer to a new data type.

        Parameters:
            new_dtype: The new data type it is casting to.
            target_address_space: The address space of the returned `LayoutTensor`.
            _element_layout: The element layout of the returned `LayoutTensor`.

        Returns:
            A new `LayoutTensor` with the same memory location but with the
            specified data type, address space, and element layout.
        """
        return Self.BitcastType[
            new_dtype, target_address_space, _element_layout
        ](
            self.ptr.bitcast[Scalar[new_dtype]]().address_space_cast[
                target_address_space
            ](),
            self.runtime_layout,
        )

    alias OriginCastType[
        mut: Bool,
        origin: Origin[mut],
    ] = LayoutTensor[
        dtype,
        layout,
        origin,
        address_space=address_space,
        element_layout=element_layout,
        layout_int_type=layout_int_type,
        linear_idx_type=linear_idx_type,
        masked=masked,
        alignment=alignment,
    ]

    alias MutableAnyType = Self.OriginCastType[True, MutableAnyOrigin]

    @deprecated(
        "`origin_cast` is no longer supported for `LayoutTensor`. Use the safer"
        " `as_any_origin`, `as_immutable` or binding the mutability instead."
    )
    @always_inline("nodebug")
    fn origin_cast[
        target_mut: Bool = Self.mut,
        target_origin: Origin[target_mut] = Origin[target_mut].cast_from[
            Self.origin
        ],
    ](self) -> Self.OriginCastType[target_mut, target_origin]:
        """Changes the origin or mutability of a pointer.

        Parameters:
            target_mut: Whether the origin is mutable.
            target_origin: Origin of the destination pointer.

        Returns:
            A new `LayoutTensor` object with the same type and the same address,
            as the original `LayoutTensor`, and the new specified mutability and
            origin.
        """
        return Self.OriginCastType[target_mut, target_origin](
            self.ptr.unsafe_mut_cast[target_mut]().unsafe_origin_cast[
                target_origin
            ](),
            self.runtime_layout,
            self.runtime_element_layout,
        )

    @always_inline("nodebug")
    fn as_any_origin(
        self: LayoutTensor[mut=True, *_, **_],
    ) -> __type_of(self).OriginCastType[True, MutableAnyOrigin]:
        """Casts the origin of the mutable `LayoutTensor` to `MutableAnyOrigin`.

        Returns:
            A pointer with the origin set to `MutableAnyOrigin`.

        This requires the tensor to already be mutable as casting mutability
        is inherently very unsafe.

        It is usually preferred to maintain concrete origin values instead of
        using `MutableAnyOrigin`. However, if it is needed, keep in mind that
        `MutableAnyOrigin` can alias any memory value, so Mojo's ASAP
        destruction will not apply during the lifetime of the tensor.
        """
        return {
            self.ptr.as_any_origin(),
            self.runtime_layout,
            self.runtime_element_layout,
        }

    @always_inline("nodebug")
    fn as_any_origin(
        self: LayoutTensor[mut=False, *_, **_],
    ) -> __type_of(self).OriginCastType[False, ImmutableAnyOrigin]:
        """Casts the origin of the immutable `LayoutTensor` to `ImmutableAnyOrigin`.

        Returns:
            A tensor with the origin set to `ImmutableAnyOrigin`.

        It is usually preferred to maintain concrete origin values instead of
        using `ImmutableAnyOrigin`. However, if it is needed, keep in mind that
        `ImmutableAnyOrigin` can alias any memory value, so Mojo's ASAP
        destruction will not apply during the lifetime of the tensor.
        """
        return {
            self.ptr.as_any_origin(),
            self.runtime_layout,
            self.runtime_element_layout,
        }

    @doc_private
    fn as_any_origin(
        self: LayoutTensor[*_, **_],
        out result: __type_of(self).OriginCastType[False, ImmutableAnyOrigin],
    ):
        constrained[
            False,
            (
                "A LayoutTensor with unbound mutability cannot be cast to"
                " 'AnyOrigin'. Consider using `as_immutable` or binding the"
                " mutability explicitly before calling this function."
            ),
        ]()
        result = abort[__type_of(result)]()

    alias AddressSpaceCastType[
        address_space: AddressSpace = Self.address_space,
    ] = LayoutTensor[
        dtype,
        layout,
        origin,
        address_space=address_space,
        element_layout=element_layout,
        layout_int_type=layout_int_type,
        linear_idx_type=linear_idx_type,
        masked=masked,
        alignment=alignment,
    ]

    @always_inline("nodebug")
    fn address_space_cast[
        target_address_space: AddressSpace = Self.address_space,
    ](self) -> Self.AddressSpaceCastType[target_address_space]:
        """Changes the address space of the `LayoutTensor`.

        Parameters:
            target_address_space: The new address space.

        Returns:
            A new `LayoutTensor` object with the same type and origin
            as the original `LayoutTensor`, and the new specified address_space.
        """
        return Self.AddressSpaceCastType[target_address_space](
            self.ptr.address_space_cast[target_address_space](),
            self.runtime_layout,
            self.runtime_element_layout,
        )

    @always_inline
    fn get_immutable(
        self,
    ) -> Self.OriginCastType[False, ImmutableOrigin.cast_from[origin]]:
        """
        Return an immutable version of this tensor.

        Returns:
            A `LayoutTensor` covering the same elements, but without mutability.
        """
        return {
            self.ptr.as_immutable(),
            self.runtime_layout,
            self.runtime_element_layout,
        }

    @always_inline
    fn _offset(self, m: Int, n: Int) -> Int:
        """Calculate the memory offset for a 2D tensor element.

        Computes the linear memory offset based on the tensor's stride
        configuration.

        Args:
            m: The row index.
            n: The column index.

        Returns:
            The calculated memory offset as an integer.
        """
        return Self.stride[0]() * m + Self.stride[1]() * n

    @always_inline
    fn _offset(self, coords: IndexList) -> Int:
        """Calculate the memory offset for a row-major tensor element.

        Computes the linear memory offset based on the tensor's stride
        configuration.

        Args:
            coords: The coordinates for the index.

        Returns:
            The calculated memory offset as an integer.
        """
        return Int(
            self.runtime_layout(
                RuntimeTuple[fill_like(self.layout.shape, UNKNOWN_VALUE)](
                    coords
                )
            )
        )

    @always_inline
    fn _elementwise_unary[
        func: fn (Self.element_type) capturing -> (Self.element_type),
    ](self) -> Self:
        """Apply an elementwise unary operation to all elements in the tensor.

        This is an internal method that applies the provided function to each
        element in the tensor. The operation is performed in-place and optimized
        for the tensor's memory layout.

        Parameters:
            func: A function that takes a single element and returns a
                transformed element. The function should be pure with no side
                effects for predictable results.

        Returns:
            Self: The modified tensor with the unary operation applied.

        Notes:

        This method requires the tensor to have a statically known layout
        for compile-time optimization.
        """
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
        other_layout_int_type: DType,
        other_linear_idx_type: DType,
    ](
        self,
        other: LayoutTensor[
            dtype,
            other_layout,
            other_origin,
            address_space=address_space,
            element_layout=element_layout,
            layout_int_type=other_layout_int_type,
            linear_idx_type=other_linear_idx_type,
            masked=other_masked,
            alignment=other_alignment,
        ],
    ) -> Self:
        """Apply an elementwise binary operation with broadcasting support.

        This internal method applies a binary operation between elements of this
        tensor and another tensor, with support for limited broadcasting
        patterns. The operation is performed in-place on this tensor.

        Parameters:
            func: A binary function that takes two elements (one from each
                tensor) and returns a single element as the result of the
                operation.
            other_layout: The layout of the other tensor.
            other_mut: Whether the other tensor is mutable.
            other_origin: The origin type of the other tensor.
            other_masked: Whether the other tensor is masked.
            other_alignment: The memory alignment of the other tensor.
            other_layout_int_type: The dimension type of the other tensor.
            other_linear_idx_type: The linear idx type of the other tensor.

        Args:
            other: The second tensor operand for the binary operation.

        Returns:
            Self: The modified tensor with the binary operation applied.

        Notes:

        - Currently supports only rank-2 tensors or tensors of the same rank.
        - For tensors of the same rank, shapes must match exactly.
        - For rank-1 to rank-2 broadcasting, the rank-1 tensor's dimension must
            match the corresponding dimension of the rank-2 tensor.
        - The operation is optimized based on the memory layout of both tensors.
        """

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
    ) -> Self.OriginCastType[True, MutableAnyOrigin]:
        """Add a scalar value to each element of the tensor.

        Performs an elementwise addition operation, adding the scalar value to
        each element in the tensor. This operation creates a new tensor with the
        results.

        Args:
            other: The scalar value to add to each element.

        Returns:
            A new tensor containing the results of the addition operation.

        Performance:

        - This operation creates a copy of the tensor before performing the
            addition.
        - For in-place addition, use the `__iadd__` method instead (`+=`
            operator).
        """

        @parameter
        fn add_val(val: Self.element_type) -> Self.element_type:
            return Self.element_type(other) + val

        return self._stack_copy()._elementwise_unary[add_val]()

    @always_inline
    fn __iadd__(self, other: Scalar[dtype]):
        """Add a scalar value to each element of the tensor in-place.

        Performs an elementwise addition operation, adding the scalar value to
        each element in the tensor. This operation modifies the tensor in-place.

        Args:
            other: The scalar value to add to each element.

        Performance:

        - This operation modifies the tensor directly without creating a copy.
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
            element_layout=element_layout, **_,
        ],
    ) -> Self.OriginCastType[True, MutableAnyOrigin]:
        """Add another tensor to this tensor elementwise.

        Performs an elementwise addition between this tensor and another tensor.
        This operation creates a new tensor with the results.

        Limited broadcasting is supported:
        - For tensors of the same rank, shapes must match exactly.
        - For rank-1 to rank-2 broadcasting, the rank-1 tensor's dimension must
          match the corresponding dimension of the rank-2 tensor.

        Parameters:
            other_layout: The layout of the other tensor.

        Args:
            other: The tensor to add to this tensor.

        Returns:
            A new tensor containing the results of the addition operation.

        Performance:

        - This operation creates a copy of the tensor before performing the
            addition.
        - For in-place addition, use the `__iadd__` method instead (`+=`
            operator).
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
            element_layout=element_layout, **_,
        ],
    ):
        """Add another tensor to this tensor elementwise in-place.

        Performs an elementwise addition between this tensor and another tensor.
        This operation modifies the tensor in-place.

        Limited broadcasting is supported:
        - For tensors of the same rank, shapes must match exactly.
        - For rank-1 to rank-2 broadcasting, the rank-1 tensor's dimension must
          match the corresponding dimension of the rank-2 tensor.

        Parameters:
            other_layout: The layout of the other tensor.

        Args:
            other: The tensor to add to this tensor.

        Performance:

        - This operation modifies the tensor directly without creating a
            copy.
        """

        fn add_val(
            lhs: Self.element_type, rhs: Self.element_type
        ) capturing -> Self.element_type:
            return lhs + rhs

        _ = self._elementwise_binary_with_broadcast[add_val](other)

    @always_inline
    fn __mul__(
        self, other: Scalar[dtype]
    ) -> Self.OriginCastType[True, MutableAnyOrigin]:
        """Multiply each element of the tensor by a scalar value.

        Performs an elementwise multiplication operation, multiplying each
        element in the tensor by the scalar value. This operation creates a new
        tensor with the results.

        Args:
            other: The scalar value to multiply with each element.

        Returns:
            A new tensor containing the results of the multiplication operation.

        Performance:

        - This operation creates a copy of the tensor before performing the
            multiplication.
        - For in-place multiplication, use the `__imul__` method instead
            (`*=` operator).
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
            element_layout=element_layout, **_,
        ],
    ) -> Self.OriginCastType[True, MutableAnyOrigin]:
        """Multiply this tensor with another tensor elementwise.

        Performs an elementwise multiplication (Hadamard product) between this tensor
        and another tensor. This operation creates a new tensor with the results.

        Limited broadcasting is supported:
        - For tensors of the same rank, shapes must match exactly.
        - For rank-1 to rank-2 broadcasting, the rank-1 tensor's dimension must
          match the corresponding dimension of the rank-2 tensor.

        Note: This is NOT a matrix multiplication operation. For matrix
        multiplication, use the appropriate matmul function instead.

        Parameters:
            other_layout: The layout of the other tensor.

        Args:
            other: The tensor to multiply with this tensor.

        Returns:
            A new tensor containing the results of the elementwise
            multiplication.

        Performance:

        - This operation creates a copy of the tensor before performing the
            multiplication.
        - For in-place multiplication, use the `__imul__` method instead
            (`*=` operator).
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
        """Multiply each element of the tensor by a scalar value in-place.

        Performs an elementwise multiplication operation, multiplying each
        element in the tensor by the scalar value. This operation modifies the
        tensor in-place.

        Args:
            other: The scalar value to multiply with each element.

        Performance:

        - This operation modifies the tensor directly without creating a copy.
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
            element_layout=element_layout, **_,
        ],
    ):
        """Multiply this tensor with another tensor elementwise in-place.

        Performs an elementwise multiplication (Hadamard product) between this
        tensor and another tensor. This operation modifies the tensor in-place.

        Limited broadcasting is supported:
        - For tensors of the same rank, shapes must match exactly.
        - For rank-1 to rank-2 broadcasting, the rank-1 tensor's dimension must
          match the corresponding dimension of the rank-2 tensor.

        Note: This is NOT a matrix multiplication operation. For matrix
        multiplication, use the appropriate matmul function instead.

        Parameters:
            other_layout: The layout of the other tensor.

        Args:
            other: The tensor to multiply with this tensor.

        Performance:

        - This operation modifies the tensor directly without creating a copy.
        """

        fn mul_val(
            lhs: Self.element_type, rhs: Self.element_type
        ) capturing -> Self.element_type:
            return lhs * rhs

        _ = self._elementwise_binary_with_broadcast[mul_val](other)

    @always_inline
    fn __sub__(
        self, other: Scalar[dtype]
    ) -> Self.OriginCastType[True, MutableAnyOrigin]:
        """Subtract a scalar value from each element of the tensor.

        Performs an elementwise subtraction operation, subtracting the scalar
        value from each element in the tensor. This operation creates a new
        tensor with the results.

        Args:
            other: The scalar value to subtract from each element.

        Returns:
            A new tensor containing the results of the subtraction operation.

        Performance:

        - This operation creates a copy of the tensor before performing the
            subtraction.
        - For in-place subtraction, use the `__isub__` method instead (`-=`
            operator).
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
            element_layout=element_layout, **_,
        ],
    ) -> Self.OriginCastType[True, MutableAnyOrigin]:
        """Subtract another tensor from this tensor elementwise.

        Performs an elementwise subtraction between this tensor and another
        tensor. This operation creates a new tensor with the results.

        Limited broadcasting is supported:
        - For tensors of the same rank, shapes must match exactly.
        - For rank-1 to rank-2 broadcasting, the rank-1 tensor's dimension must
          match the corresponding dimension of the rank-2 tensor.

        Parameters:
            other_layout: The layout of the other tensor.

        Args:
            other: The tensor to subtract from this tensor.

        Returns:
            A new tensor containing the results of the subtraction operation.

        Performance:

        - This operation creates a copy of the tensor before performing the
            subtraction.
        - For in-place subtraction, use the `__isub__` method instead (`-=`
            operator).
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
        """Subtract a scalar value from each element of the tensor in-place.

        Performs an elementwise subtraction operation, subtracting the scalar
        value from each element in the tensor. This operation modifies the
        tensor in-place.

        Args:
            other: The scalar value to subtract from each element.

        Performance:

        - This operation modifies the tensor directly without creating a copy.
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
            element_layout=element_layout, **_,
        ],
    ):
        """Subtract another tensor from this tensor elementwise in-place.

        Performs an elementwise subtraction between this tensor and another
        tensor. This operation modifies the tensor in-place.

        Limited broadcasting is supported:
        - For tensors of the same rank, shapes must match exactly.
        - For rank-1 to rank-2 broadcasting, the rank-1 tensor's dimension must
          match the corresponding dimension of the rank-2 tensor.

        Parameters:
            other_layout: The layout of the other tensor.

        Args:
            other: The tensor to subtract from this tensor.

        Performance:

        - This operation modifies the tensor directly without creating a copy.
        """

        fn sub_val(
            lhs: Self.element_type, rhs: Self.element_type
        ) capturing -> Self.element_type:
            return lhs - rhs

        _ = self._elementwise_binary_with_broadcast[sub_val](other)

    @always_inline
    fn __truediv__(
        self, other: Scalar[dtype]
    ) -> Self.OriginCastType[True, MutableAnyOrigin]:
        """Divide each element of the tensor by a scalar value.

        Performs an elementwise division operation, dividing each element in the
        tensor by the scalar value. This operation creates a new tensor with the
        results.

        Args:
            other: The scalar value to divide each element by.

        Returns:
            A new tensor containing the results of the division operation.

        Performance:

        - This operation creates a copy of the tensor before performing the
            division.
        - For in-place division, use the `__itruediv__` method instead
            (`/=` operator).

        Notes:

        - Division by zero will result in undefined behavior or errors
            depending on the dtype.
        - For integer dtypes, this performs integer division.
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
            element_layout=element_layout, **_,
        ],
    ) -> Self.OriginCastType[True, MutableAnyOrigin]:
        """Divide this tensor by another tensor elementwise.

        Performs an elementwise division between this tensor and another tensor.
        This operation creates a new tensor with the results.

        Limited broadcasting is supported:
        - For tensors of the same rank, shapes must match exactly.
        - For rank-1 to rank-2 broadcasting, the rank-1 tensor's dimension must
          match the corresponding dimension of the rank-2 tensor.

        Parameters:
            other_layout: The layout of the other tensor.

        Args:
            other: The tensor to divide this tensor by.

        Returns:
            A new tensor containing the results of the division operation.

        Performance:

        - This operation creates a copy of the tensor before performing the
            division.
        - For in-place division, use the `__itruediv__` method instead
            (`/=` operator).

        Notes:

        - Division by zero will result in undefined behavior or errors depending on the dtype.
        - For integer dtypes, this performs integer division.
        """

        fn div_val(
            lhs: Self.element_type, rhs: Self.element_type
        ) capturing -> Self.element_type:
            return lhs / rhs

        return self._stack_copy()._elementwise_binary_with_broadcast[div_val](
            other
        )

    fn __itruediv__(self, other: Scalar[dtype]):
        """Divide each element of the tensor by a scalar value in-place.

        Performs an elementwise division operation, dividing each element in the
        tensor by the scalar value. This operation modifies the tensor in-place.

        Args:
            other: The scalar value to divide each element by.

        Performance:

        - This operation modifies the tensor directly without creating a copy.

        Notes:

        - Division by zero will result in undefined behavior or errors depending on the dtype.
        - For integer dtypes, this performs integer division.
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
            element_layout=element_layout, **_,
        ],
    ):
        """Divide this tensor by another tensor elementwise in-place.

        Performs an elementwise division between this tensor and another tensor.
        This operation modifies the tensor in-place.

        Limited broadcasting is supported:
        - For tensors of the same rank, shapes must match exactly.
        - For rank-1 to rank-2 broadcasting, the rank-1 tensor's dimension must
          match the corresponding dimension of the rank-2 tensor.

        Parameters:
            other_layout: The layout of the other tensor.

        Args:
            other: The tensor to divide this tensor by.

        Performance:

        - This operation modifies the tensor directly without creating a copy.

        Notes:

        - Division by zero will result in undefined behavior or errors depending on the dtype.
        - For integer dtypes, this performs integer division.
        """

        fn div_val(
            lhs: Self.element_type, rhs: Self.element_type
        ) capturing -> Self.element_type:
            return lhs / rhs

        _ = self._elementwise_binary_with_broadcast[div_val](other)

    @always_inline
    fn __exp__(self) -> Self:
        """Computes element-wise exponential function.

        Returns a new tensor containing the
        [element-wise exponential](/mojo/stdlib/math/math/exp/) of the input tensor.

        Returns:
            A new tensor containing the element-wise exponential.
        """

        @parameter
        fn exp_func(val: Self.element_type) -> Self.element_type:
            return exp(val)

        return {
            self._stack_copy()
            ._elementwise_unary[exp_func]()
            .ptr.unsafe_mut_cast[mut]()
            .unsafe_origin_cast[origin](),
            self.runtime_layout,
            self.runtime_element_layout,
        }

    @always_inline("nodebug")
    fn _load_offset(
        self, offset: Scalar[Self.linear_idx_type]
    ) -> Self.element_type:
        """Retrieves a single element from the tensor at the specified offset.

        This method provides array-like linear indexing for the tensor.

        Args:
            offset: The integer offset for array indexing.

        Returns:
            The element at the specified offset with the tensor's data type.
        """

        return (
            Element[index_type=linear_idx_type]
            .load(self.ptr.offset(offset), self.runtime_element_layout)
            .element_data
        )

    @always_inline("nodebug")
    fn __getitem__[*Tys: Indexer](self, *args: *Tys) -> Self.element_type:
        """Retrieves a single element from the tensor at the specified indices.

        This method provides array-like indexing for the tensor. The number of
        indices provided must match the rank of the tensor, otherwise an error
        will occur at runtime.

        Parameters:
            Tys: The type of the indices. Must implement the `Indexer` trait,
                and match the rank of the tensor.

        Args:
            args: The indices specifying the element's position in the tensor.

        Returns:
            The element at the specified position with the tensor's data type.
        """
        alias arg_count = args.__len__()

        constrained[
            Self.rank == arg_count or Self.num_strides == arg_count,
            "Indexed with "
            + String(arg_count)
            + " dims, but Self.rank, Self.num_strides = "
            + String(Self.rank)
            + ", "
            + String(self.num_strides),
        ]()

        var index_list = Self.idx_list_t[arg_count](fill=0)

        @parameter
        for arg_idx in range(arg_count):
            index_list[arg_idx] = index(args[arg_idx])

        var strides = self.runtime_layout.stride.value
        var offset = Self._get_offset[rank=arg_count](strides, index_list)
        return self._load_offset(offset)

    @always_inline("nodebug")
    fn __getitem__(self, crd: RuntimeTuple) -> Self.element_type:
        """Retrieves a single element from the tensor at the specified indices.

        This method provides array-like indexing for the tensor. The number of
        indices provided must match the rank of the tensor, otherwise an error
        will occur at runtime.

        Args:
            crd: The coordinate specifying the element's position in each dimension. For example, in a 3D tensor, you would use (i, j, k).

        Returns:
            The element at the specified position with the tensor's data type.
        """

        var offset = self.runtime_layout(crd)
        return self._load_offset(offset)

    @always_inline("nodebug")
    fn __setitem__[*Tys: Indexer](self, *args: *Tys, val: Self.element_type):
        """Sets a single element in a tensor at the specified indices.

        This method provides array-like element assignment for tensors.

        Parameters:
            Tys: The type of the indices. Must implement the `Indexer` trait,
                and match the rank of the tensor.

        Args:
            args: The indices specifying the element's position in the tensor.
            val: The value to write to the tensor at the specified position.

        Notes:

        - No bounds checking is performed. Accessing out-of-bounds indices
            will result in undefined behavior.
        """

        alias arg_count = args.__len__()

        constrained[
            Self.rank == arg_count or Self.num_strides == arg_count,
            "Indexed with "
            + String(arg_count)
            + " dims, but Self.rank, Self.num_strides = "
            + String(Self.rank)
            + ", "
            + String(self.num_strides),
        ]()

        var index_list = Self.idx_list_t[arg_count](fill=0)

        @parameter
        for arg_idx in range(arg_count):
            index_list[arg_idx] = index(args[arg_idx])

        var strides = self.runtime_layout.stride.value
        var offset = Self._get_offset(strides, index_list)

        Element[index_type=linear_idx_type](
            val, self.runtime_element_layout
        ).store(self.ptr.offset(offset))

    @always_inline("nodebug")
    fn load[width: Int](self, m: Int, n: Int) -> SIMD[dtype, width]:
        """Load a SIMD vector from the tensor at the specified 2D coordinates.

        Performs a vectorized load operation from the tensor's memory,
        retrieving `width` consecutive elements starting at position (m, n).
        This method enables efficient SIMD operations on tensor data.

        Parameters:
            width: The number of elements to load into the SIMD vector. Should match
                  the target hardware's vector width for optimal performance.

        Args:
            m: The row index (first dimension).
            n: The column index (second dimension).

        Returns:
            A SIMD vector containing 'width' consecutive elements from the tensor.

        Performance:

        - Uses unaligned memory access which may be slower on some
            architectures.
        - For aligned access, use `aligned_load` instead when data alignment is
            guaranteed.
        - The load operation is optimized based on the tensor's memory layout.

        Notes:

        - No bounds checking is performed. Accessing out-of-bounds indices will
            result in undefined behavior.
        - The elements are loaded according to the tensor's stride configuration.
        """

        return self.ptr.load[width=width, alignment = Self.alignment](
            self._offset(m, n)
        )

    @always_inline("nodebug")
    fn load[width: Int](self, coords: IndexList[**_]) -> SIMD[dtype, width]:
        """Load a SIMD vector from the tensor at the specified 2D coordinates.

        Performs a vectorized load operation from the tensor's memory,
        retrieving `width` consecutive elements starting at position (m, n).
        This method enables efficient SIMD operations on tensor data.

        Parameters:
            width: The number of elements to load into the SIMD vector. Should match
                    the target hardware's vector width for optimal performance.

        Args:
            coords: The coordinates to index.

        Returns:
            A SIMD vector containing 'width' consecutive elements from the tensor.

        Performance:

        - Uses unaligned memory access which may be slower on some
            architectures.
        - For aligned access, use `aligned_load` instead when data alignment is
            guaranteed.
        - The load operation is optimized based on the tensor's memory layout.

        Notes:

        - No bounds checking is performed. Accessing out-of-bounds indices will
            result in undefined behavior.
        - The elements are loaded according to the tensor's stride configuration.
        """
        constrained[self.rank == coords.size]()
        debug_assert(Int(self.runtime_layout.stride.value[self.rank - 1]) == 1)

        var idx = self.runtime_layout(
            RuntimeTuple[fill_like(self.layout.shape, UNKNOWN_VALUE)](coords)
        )

        return self.ptr.load[width=width, alignment = Self.alignment](idx)

    @always_inline
    fn prefetch(self, m: Int, n: Int):
        """Prefetch tensor data at the specified 2D coordinates into cache.

        Issues a software prefetch hint to the processor to load the data at
        position (m, n) into the cache hierarchy. This can improve performance
        by reducing memory latency for subsequent accesses to the same location.

        Args:
            m: The row index (first dimension).
            n: The column index (second dimension).

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
            self.ptr.offset(self._offset(m, n))
        )

    @always_inline
    fn prefetch(self, coords: IndexList):
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
            self.ptr.offset(self._offset(coords))
        )

    @always_inline("nodebug")
    fn aligned_load[width: Int](self, m: Int, n: Int) -> SIMD[dtype, width]:
        """Load a SIMD vector with alignment guarantees from the tensor.

        Performs an aligned vectorized load operation from the tensor's memory,
        retrieving `width` consecutive elements starting at position (m, n). The
        alignment is automatically calculated based on the SIMD width and dtype.

        Parameters:
            width: The number of elements to load into the SIMD vector. Should
                match the target hardware's vector width for optimal performance.

        Args:
            m: The row index (first dimension).
            n: The column index (second dimension).

        Returns:
            A SIMD vector containing 'width' consecutive elements from the tensor.

        Performance:

        - Uses aligned memory access which is faster than unaligned access on
            most architectures.
        - The alignment is automatically calculated based on the SIMD width and
            dtype.
        - Can be up to 2x faster than unaligned loads on architectures that
            require alignment.

        Notes:

        - The caller must ensure that the memory at (m, n) is properly aligned.
            Misaligned access with this method may cause hardware exceptions on
            some architectures.
        - No bounds checking is performed. Accessing out-of-bounds indices will
            result in undefined behavior.
        """

        alias _alignment = align_of[SIMD[dtype, width]]()
        return self.ptr.load[width=width, alignment=_alignment](
            self._offset(m, n)
        )

    @always_inline("nodebug")
    fn store[width: Int](self, m: Int, n: Int, val: SIMD[dtype, width]):
        """Store a SIMD vector to the tensor at the specified 2D coordinates.

        Performs a vectorized store operation to the tensor's memory, writing
        'width' consecutive elements starting at position (m, n). This method
        enables efficient SIMD operations on tensor data.

        Parameters:
            width: The number of elements in the SIMD vector to store. Should
                match the target hardware's vector width for optimal performance.

        Args:
            m: The row index (first dimension) where the store operation begins.
            n: The column index (second dimension) where the store operation
                begins.
            val: The SIMD vector containing the values to store in the tensor.

        Performance:

        - Uses unaligned memory access which may be slower on some
            architectures.
        - For aligned access, use aligned_store instead when data alignment is
            guaranteed.
        - The store operation is optimized based on the tensor's memory layout.

        Notes:

        - No bounds checking is performed. Accessing out-of-bounds indices will
            result in undefined behavior.
        - The elements are stored according to the tensor's stride configuration.
        - This operation modifies the tensor's data in-place.
        """
        return self.ptr.store[alignment = Self.alignment](
            self._offset(m, n), val
        )

    @always_inline("nodebug")
    fn store[
        width: Int
    ](self, coords: IndexList[*_, **_], val: SIMD[dtype, width]):
        """Store a SIMD vector to the tensor at the specified 2D coordinates.

        Performs a vectorized store operation to the tensor's memory, writing
        'width' consecutive elements starting at position (m, n). This method
        enables efficient SIMD operations on tensor data.

        Parameters:
            width: The number of elements in the SIMD vector to store. Should
                match the target hardware's vector width for optimal performance.

        Args:
            coords: The coordinates to index.
            val: The SIMD vector containing the values to store in the tensor.

        Performance:

        - Uses unaligned memory access which may be slower on some
            architectures.
        - For aligned access, use aligned_store instead when data alignment is
            guaranteed.
        - The store operation is optimized based on the tensor's memory layout.

        Notes:

        - No bounds checking is performed. Accessing out-of-bounds indices will
            result in undefined behavior.
        - The elements are stored according to the tensor's stride configuration.
        - This operation modifies the tensor's data in-place.
        """
        constrained[self.rank == coords.size]()
        debug_assert(Int(self.runtime_layout.stride.value[self.rank - 1]) == 1)

        var idx = self.runtime_layout(
            RuntimeTuple[fill_like(self.layout.shape, UNKNOWN_VALUE)](coords)
        )

        return self.ptr.store[alignment = Self.alignment](idx, val)

    @always_inline("nodebug")
    fn aligned_store[width: Int](self, m: Int, n: Int, val: SIMD[dtype, width]):
        """Store a SIMD vector with alignment guarantees to the tensor.

        Performs an aligned vectorized store operation to the tensor's memory,
        writing `width` consecutive elements starting at position (m, n). The
        alignment is automatically calculated based on the SIMD width and dtype.

        Parameters:
            width: The number of elements in the SIMD vector to store. Should
                match the target hardware's vector width for optimal performance.

        Args:
            m: The row index (first dimension) where the store operation begins.
            n: The column index (second dimension) where the store operation
                begins.
            val: The SIMD vector containing the values to store in the tensor.

        Performance:

        - Uses aligned memory access which is faster than unaligned access on
            most architectures.
        - The alignment is automatically calculated based on the SIMD width and
            dtype.
        - Can be up to 2x faster than unaligned stores on architectures that
            require alignment.
        - Particularly important for streaming stores that bypass the cache.

        Notes:

        - The caller must ensure that the memory at (m, n) is properly aligned.
            Misaligned access with this method may cause hardware exceptions on
            some architectures.
        - No bounds checking is performed. Accessing out-of-bounds indices will
            result in undefined behavior.
        - This operation modifies the tensor's data in-place.
        """

        alias _alignment = align_of[SIMD[dtype, width]]()
        return self.ptr.store[alignment=_alignment](self._offset(m, n), val)

    @always_inline("nodebug")
    fn size(self) -> Int:
        """
        Get the total number of elements that the tensor can contain.

        Returns:
          The total number of elements that can be stores in the tensor.
        """

        @parameter
        if layout.all_dims_known():
            alias size = layout.size()
            return size
        else:
            return self.runtime_layout.size()

    @staticmethod
    @always_inline("nodebug")
    fn stack_allocation[
        *, stack_alignment: Int = Self.alignment
    ]() -> Self.StackTensorType:
        """Allocates stack memory for a `LayoutTensor` with a fully static
        layout.

        Creates a new `LayoutTensor` instance with memory allocated on the stack
        rather than the heap. This provides deterministic memory management and
        potentially better performance for tensors with known sizes at compile
        time.

        Constraints:
            - The layout must be fully static (all dimensions known at compile
                time).
            - The alignment must be a multiple of the tensor's minimum required
                alignment.

        Parameters:
             stack_alignment: Memory alignment value for the allocation in bytes. Must
                be a multiple of the tensor's minimum required alignment.
                Default is the tensor's natural alignment based on its data type
                and layout.

        Returns:
            A new `LayoutTensor` instance with memory allocated on the stack.

        Performance:

        - Stack allocation is typically faster than heap allocation.
        - Proper alignment can significantly improve memory access performance,
            especially for vectorized operations.
        - No dynamic memory management overhead (no malloc/free calls).

        Notes:

        - Only works with tensors that have fully static layouts known at
            compile time.
        - Stack memory is limited, so this should only be used for reasonably
            sized tensors.
        - The allocated memory is automatically freed when the function returns.
        """

        constrained[layout.all_dims_known(), "Requires fully static layout"]()
        constrained[
            stack_alignment % Self.alignment == 0,
            "Stack allocation alignment ",
            String(stack_alignment),
            " must be multiple of tensor alignment ",
            String(Self.alignment),
        ]()

        return Self.StackTensorType(
            stack_allocation[
                layout.size() * element_layout.size(),
                dtype,
                alignment=stack_alignment,
                address_space=address_space,
            ]()
        )

    @staticmethod
    @always_inline("nodebug")
    fn null() -> Self.StackTensorType:
        """
        Returns a null `LayoutTensor` object.

        Returns:
            A null `LayoutTensor` object.
        """
        return Self.StackTensorType(
            UnsafePointer[
                Scalar[dtype],
                address_space=address_space,
                origin = MutableOrigin.empty,
            ]()
        )

    alias StackTensorType = LayoutTensor[
        dtype,
        layout,
        MutableAnyOrigin,
        address_space=address_space,
        element_layout=element_layout,
        layout_int_type=layout_int_type,
        linear_idx_type=linear_idx_type,
        masked=masked,
        alignment=alignment,
    ]

    @always_inline
    fn to_device_buffer(self, ctx: DeviceContext) -> DeviceBuffer[dtype]:
        """Convert the tensor to a `DeviceBuffer`.

        Args:
            ctx: The device context to use.

        Returns:
            A `DeviceBuffer` containing the tensor's data.
        """
        constrained[
            address_space == address_space.GENERIC,
            "DeviceBuffer is only used on GENERIC address space",
        ]()
        return DeviceBuffer[dtype](
            ctx,
            rebind[UnsafePointer[Scalar[dtype]]](self.ptr),
            self.size(),
            owning=False,
        )

    @always_inline("nodebug")
    fn _stack_copy(
        self,
    ) -> Self.StackTensorType:
        @parameter
        if Self.layout.all_dims_known():
            copy = self.stack_allocation()
        else:
            copy = Self.StackTensorType(self.ptr, self.runtime_layout)

        fn self_value(
            lhs: Self.element_type, rhs: Self.element_type
        ) capturing -> Self.element_type:
            return rhs

        return copy._elementwise_binary_with_broadcast[self_value](self)

    @staticmethod
    @always_inline("nodebug")
    fn _to_static[
        t: IntTuple[__origin_of()], element_type: DType
    ]() -> IndexList[len(t), element_type=element_type]:
        var st = IndexList[len(t), element_type=element_type]()

        @parameter
        for i in range(len(t)):
            st[i] = Int(t[i])
        return st

    @staticmethod
    @always_inline("nodebug")
    fn _get_rank_stride_offset(rank_idx: Int) -> Int:
        offset = 0
        for i in range(rank_idx):
            offset += len(flatten(layout.shape[i]))
        return offset

    @staticmethod
    @always_inline("nodebug")
    fn _get_rank_offset[
        num_strides: Int, rank: Int, //, rank_idx: Int
    ](stride: IndexList[num_strides, **_], vals: IndexList[rank, **_]) -> Int:
        alias sub_layout = layout[rank_idx]
        alias stride_idx = Self._get_rank_stride_offset(rank_idx)

        @parameter
        if len(sub_layout) == 1:
            return stride[stride_idx] * vals[rank_idx]
        return 0

    @staticmethod
    @always_inline("nodebug")
    fn _expand_indices(
        ridx: Self.idx_list_t[Self.rank],
    ) -> Self.idx_list_t[Self.num_strides]:
        eidx = IndexList[
            Self.num_strides, element_type = Self.linear_idx_type
        ]()
        eidx_offset = 0

        @parameter
        for rank_idx in range(Self.rank):
            alias sub_layout = flatten(layout.shape[rank_idx])
            alias sub_layout_size = len(sub_layout)
            constrained[sub_layout_size > 0]()

            @parameter
            if sub_layout_size == 1:
                # not nested
                eidx[eidx_offset] = ridx[rank_idx]
                eidx_offset += 1
            else:
                # map from linear to column-major cartesian indices
                idx = ridx[rank_idx]

                @parameter
                for i in range(sub_layout_size - 1):
                    alias sz: Int = sub_layout[i].value()
                    constrained[
                        sz != UNKNOWN_VALUE,
                        (
                            "unknown shapes not supported in non-trailing"
                            " positions of nested dimensions"
                        ),
                    ]()
                    idx, r = divmod(idx, sz)
                    eidx[eidx_offset] = r
                    eidx_offset += 1
                eidx[eidx_offset] = idx
                eidx_offset += 1

        return eidx

    @staticmethod
    @always_inline("nodebug")
    fn _get_offset[
        rank: Int,
    ](
        stride: Self.idx_list_t[Self.num_strides],
        vals: Self.idx_list_t[rank],
    ) -> Int:
        constrained[
            rank == Self.rank or rank == Self.num_strides,
            "idx rank = "
            + String(rank)
            + "\nTensor rank = "
            + String(Self.rank)
            + "\nnum_strides = "
            + String(Self.num_strides),
        ]()

        var offset: Scalar[Self.linear_idx_type] = 0

        var idxs: Self.idx_list_t[Self.num_strides]

        @parameter
        if Self.num_strides == rank:
            idxs = rebind[Self.idx_list_t[Self.num_strides]](vals)
        else:
            idxs = Self._expand_indices(
                rebind[Self.idx_list_t[Self.rank]](vals)
            )

        @parameter
        for i in range(Self.num_strides):
            offset += idxs[i] * stride[i]
        return Int(offset)

    @always_inline
    @staticmethod
    fn shape[idx: Int]() -> Int:
        """Returns the size of the tensor along the specified dimension.

        Provides static access to the tensor's shape information. This method
        returns the size of a specific dimension without requiring an instance
        of the tensor, as the shape is part of the tensor's static type
        information.

        Parameters:
            idx: The dimension index to query (0-based).
                 For example, in a 3D tensor with shape [10, 20, 30]:
                 - `shape[0]()` returns 10 (first dimension).
                 - `shape[1]()` returns 20 (second dimension).
                 - `shape[2]()` returns 30 (third dimension).

        Returns:
            The size of the tensor along the specified dimension as an integer.

        Performance:

        - This is a compile-time operation with no runtime cost when used
            with static dimensions.

        Notes:

        - This is a static method that operates on the tensor's type information,
            not on a specific tensor instance.
        """

        # FIXME: having to specify the origin is kind of weird
        alias shape = Self._to_static[layout.shape, layout_int_type]()
        return shape[idx]

    @always_inline
    @staticmethod
    fn stride[idx: Int]() -> Int:
        """Returns the memory stride of the tensor along the specified
        dimension.

        Provides static access to the tensor's stride information. The stride
        represents the number of elements to skip in memory to move one position
        along a particular dimension. This method returns the stride without
        requiring an instance of the tensor, as the stride is part of the
        tensor's static type information.

        Parameters:
            idx: The dimension index to query (0-based).
                 For example, in a 2D tensor with shape [10, 20] and row-major
                 layout:
                 - `stride[0]()` might return 20 (moving one row requires
                   skipping 20 elements).
                 - `stride[1]()` might return 1 (moving one column requires
                   skipping 1 element).

        Returns:
            The memory stride of the tensor along the specified dimension as an
            integer.

        Performance:

        - This is a compile-time operation with no runtime cost when used
            with static dimensions.
        - Understanding stride patterns is crucial for optimizing memory access
            patterns in performance-critical code.

        Notes:

        - Strides depend on the memory layout (row-major, column-major, or
            custom).
        - For non-contiguous tensors (e.g., tensor slices), strides may not
            follow a simple pattern.
        """

        # FIXME: having to specify the origin is kind of weird
        alias stride = Self._to_static[layout.stride, linear_idx_type]()
        return stride[idx]

    @always_inline
    fn dim(self, idx: Int) -> Int:
        """Returns the runtime dimension size of the tensor along the specified
        axis.

        Unlike the static `dim` method, this instance method takes a runtime
        dimension index.

        Args:
            idx: The dimension index to query (0-based).
                 For example, in a 3D tensor with shape `[10, 20, 30]`:
                 - `dim(0)` returns 10 (first dimension).
                 - `dim(1)` returns 20 (second dimension).
                 - `dim(2)` returns 30 (third dimension).

        Returns:
            The dimension of the tensor along the specified axis as an integer.
        """

        constrained[
            0 <= depth(layout.shape) <= 1,
            String(
                (
                    "This method only works with tensors that have depth-1"
                    " layouts (no nested shapes). Received: "
                ),
                layout,
            ),
        ]()

        return self.runtime_layout.shape.value[idx]

    @always_inline
    fn stride(self, idx: Int) -> Int:
        """Returns the runtime stride of the tensor along the specified
        axis.

        Unlike the static `stride` method, this instance method takes a runtime
        dimension index.

        Args:
            idx: The dimension index to query (0-based).
                 For example, in a row-major 3D tensor with shape `[10, 20, 30]`:
                 - `stride(0)` returns 600 (first dimension).
                 - `stride(1)` returns 30 (second dimension).
                 - `stride(2)` returns 1 (third dimension).

        Returns:
            The dimension of the tensor along the specified axis as an integer.
        """

        constrained[
            0 <= depth(layout.stride) <= 1,
            String(
                (
                    "This method only works with tensors that have depth-1"
                    " layouts (no nested shapes). Received: "
                ),
                layout,
            ),
        ]()

        return self.runtime_layout.stride.value[idx]

    @always_inline
    fn dim[idx: Int](self) -> Int:
        """Returns the dimension size of the tensor along the specified
        axis.

        Unlike the static `shape` method, this instance method provides access
        to the tensor's actual dimension sizes. If the dimension is unknown,
        the runtime layout is used to get the dimension size.

        Parameters:
            idx: The dimension index to query (0-based).
                 For example, in a 3D tensor with shape `[10, 20, 30]`:
                 - `dim[0]()` returns 10 (first dimension).
                 - `dim[1]()` returns 20 (second dimension).
                 - `dim[2]()` returns 30 (third dimension).

        Constraints:
            - Only works with tensors that have depth-1 layouts (no nested
                shapes).

        Returns:
            The size of the tensor along the specified dimension as an integer.

        Performance:

        - For static dimensions known at compile time, prefer the static
            `shape` method when possible for better performance.

        Notes:

        - This method works with both static and dynamic dimensions.
        - For tensors with masked or partial views, this returns the actual
            size of the view, not the original tensor.
        """
        constrained[
            0 <= depth(layout.shape) <= 1,
            String(
                (
                    "This method only works with tensors that have depth-1"
                    " layouts (no nested shapes). Received: "
                ),
                layout,
            ),
        ]()

        alias shape = Self._to_static[layout.shape, layout_int_type]()

        @parameter
        if not layout.shape[idx].all_known() or Self.masked:
            return self.runtime_layout.shape.value[idx]
        else:
            return shape[idx]

    alias CoalesceType[element_layout: Layout] = LayoutTensor[
        dtype,
        coalesce(layout),
        origin,
        address_space=address_space,
        element_layout=element_layout,
    ]

    @always_inline
    fn coalesce(self) -> Self.CoalesceType[Self.element_layout]:
        """Creates a tensor with a coalesced memory layout from this tensor.

        Coalescing a tensor's layout means reorganizing its memory
        representation to be as contiguous as possible, which can improve memory
        access patterns and performance. This operation does not move or copy
        data; it only changes how the same memory is interpreted.

        Returns:
            A tensor with the same data but with a coalesced memory layout.
            The returned tensor has type `LayoutTensor` with the same dtype but
            with a coalesced layout.

        Performance:

        - Coalesced layouts typically provide better cache utilization and
            memory access patterns.
        - This operation is zero-cost at runtime as it only changes the
            layout information, not the actual data.
        - Particularly beneficial before operations that perform sequential
            memory access or vectorized operations.

        Notes:

        - The coalesced tensor shares the same memory as the original tensor,
            so modifications to one will affect the other.
        - The shape of the tensor remains the same, only the stride information
            is optimized.
        - For already optimally coalesced tensors, this operation has no effect.
        """
        return Self.CoalesceType[Self.element_layout](self.ptr)

    @staticmethod
    fn _compute_tile_layout[*tile_sizes: Int]() -> Layout:
        return Self._divide_tiles[*tile_sizes]()

    @staticmethod
    fn _divide_tiles[*tile_sizes: Int]() -> Layout:
        alias tiler = MakeTileLayoutList[*tile_sizes]()
        return zipped_divide(layout, materialize[tiler]())

    @staticmethod
    fn _fast_varying_dim_tiler(shape: Int) -> Layout:
        var flat_stride = flatten(layout.stride)
        var min_stride = Int.MAX
        var min_idx = -1
        for i in range(len(flat_stride)):
            var s = flat_stride[i]
            if s == UNKNOWN_VALUE:
                continue
            if s < min_stride:
                min_stride = Int(flat_stride[i])
                min_idx = i
        if min_stride != 1:
            abort(
                "Linear vectorization is limited to tensors with a contiguous"
                " dimension"
            )
        if min_idx == -1:
            abort("No known stride found to vectorize!")
        var flat_tiler_shape = IntTuple()
        for i in range(len(flat_stride)):
            if i == min_idx:
                flat_tiler_shape.append(shape)
            else:
                flat_tiler_shape.append(Int(1))
        var tiler_shape = to_nest(layout.stride, flat_tiler_shape)
        var unit_stride = fill_like(layout.shape, 1)
        return Layout(tiler_shape, unit_stride)

    @staticmethod
    fn _tuple_divide_tiler(
        shape: IntTuple, linear_vectorize: Bool = False
    ) -> Layout:
        if is_int(shape):
            if linear_vectorize:
                # If the shape is a single int, and we are vectorizing wrt the
                # linear indexing. We should then vectorize the fastest varying
                # dimension.
                return Self._fast_varying_dim_tiler(Int(shape))
            else:
                # If the shape is a single int, we need to use the LayoutList
                # dispatch
                return Layout(shape, 1)
        else:
            # Otherwise, the shape should be compatible, so we can use the
            # nested layout dispatch
            var tiler_stride = fill_like(shape, 1)
            return Layout(shape, tiler_stride)

    @staticmethod
    fn _tuple_divide_tiles(
        shape: IntTuple, linear_vectorize: Bool = False
    ) -> Layout:
        var tiler = Self._tuple_divide_tiler(shape, linear_vectorize)
        if is_int(shape) and not linear_vectorize:
            # legacy behavior
            return zipped_divide(layout, LayoutList(tiler))
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

    alias TileType[*tile_sizes: Int] = LayoutTensor[
        dtype,
        Self._compute_tile_layout[*tile_sizes]()[0],
        origin,
        address_space=address_space,
        element_layout=element_layout,
        layout_int_type=layout_int_type,
        linear_idx_type=linear_idx_type,
        masked = masked or _tile_is_masked[layout, *tile_sizes](),
        alignment=alignment,
    ]
    """The tile type returned by the `tile()` method given
    the specified set of tile sizes.

    Parameters:
        tile_sizes: The dimensions of each tile along each axis of the
            tensor.
    """

    @always_inline
    fn tile[
        *tile_sizes: Int
    ](self, *tile_coords: Int) -> self.TileType[*tile_sizes]:
        """Extract a tile (sub-tensor) from this tensor with specified
        dimensions and position.

        Tiling is a fundamental operation for high-performance tensor
        computations that divides a tensor into smaller blocks for better cache
        locality and parallelism. This method extracts a specific tile at the
        given coordinates without copying data.

        Parameters:
            tile_sizes: The dimensions of each tile along each axis of the
                tensor. For example, in a 2D tensor, `tile[32, 32]` creates
                32x32 tiles.

        Args:
            tile_coords: The coordinates of the specific tile to extract. For
                example, `tile[32, 32](1, 2)` extracts the tile at position
                (1, 2) in the grid of 32x32 tiles.

        Returns:
            A view into the original tensor representing the specified tile.

        Example:

        For a 4x4 tensor with values:

        ```
        [1 2 3 4]
        [2 3 4 5]
        [5 4 3 2]
        [1 1 1 1]
        ```

        `tile[2, 2](1, 0)` will extract the tile:

        ```
        [5 4]
        [1 1]
        ```

        Performance:

        - Creates a view without copying data, making it very efficient.
        - Optimized for both static and dynamic layouts with different code paths.
        - Properly handles edge cases where tiles may be partially outside the tensor.
        - Maintains stride information for efficient memory access within the tile.

        Notes:

        - The resulting tile is a view into the original tensor, so modifications
            to the tile will affect the original tensor.
        - For tiles at the edges of the tensor, the actual dimensions may be smaller
            than the requested tile_sizes if masking is enabled.
        - The implementation automatically selects between static and dynamic tiling
            based on the tensor's layout properties.
        """

        alias num_tiles = stdlib.builtin.variadic_size(tile_sizes)

        # need to calculate this again because _tiled_layout[1] is required for the offset calculation
        alias _tiled_layout = Self._compute_tile_layout[*tile_sizes]()

        constrained[
            _tiled_layout[1].rank() == num_tiles,
            "Number of tiles should match the rank",
        ]()

        alias tile_type = self.TileType[*tile_sizes]

        var offset = 0
        var runtime_shape = tile_type.RuntimeLayoutType.ShapeType()
        var runtime_stride = tile_type.RuntimeLayoutType.StrideType()

        # Static layout tiling
        # TODO: Consider merge the two cases in away that won't slowdown the fully static layout.
        @parameter
        if tile_type.layout.all_dims_known():

            @parameter
            for i in range(num_tiles):
                alias stride = Int(_tiled_layout[1].stride[i])
                offset += tile_coords[i] * stride

            var runtime_layout = tile_type.RuntimeLayoutType(
                runtime_shape, runtime_stride
            )

            # Adjust runtime layout, so the shape is clipped to the unmasked sizes.
            @parameter
            if tile_type.masked:

                @parameter
                for i in range(tile_type.layout.rank()):
                    cur_dim = self.dim[i]() - (tile_coords[i] * tile_sizes[i])
                    shape_i = max(0, min(tile_sizes[i], cur_dim))
                    runtime_layout.shape.value[i] = shape_i

            return tile_type(self.ptr.offset(offset), runtime_layout)

        else:
            # Dynamic layout, use strides

            @parameter
            for i in range(num_tiles):
                var stride = self.runtime_layout.stride.value[i] * tile_sizes[i]
                runtime_stride.value[i] = self.runtime_layout.stride.value[i]
                offset += tile_coords[i] * stride

            var runtime_layout = tile_type.RuntimeLayoutType(
                runtime_shape, runtime_stride
            )

            # Adjusts the runtime layout so that the shape is clipped to the unmasked sizes.
            @parameter
            for i in range(tile_type.layout.rank()):
                cur_dim = self.dim[i]() - (tile_coords[i] * tile_sizes[i])
                shape_i = max(0, min(tile_sizes[i], cur_dim))
                runtime_layout.shape.value[i] = shape_i

            return tile_type(self.ptr.offset(offset), runtime_layout)

    alias SIMDTileType[tile_size: Int] = Self.TileType[
        tile_size, simd_width_of[Self.dtype]()
    ]

    @always_inline
    fn simd_tile[
        tile_size: Int
    ](self, tile_idx: Int) -> Self.SIMDTileType[tile_size]:
        """Return a SIMD[dtype] sized tile of size `tile_size` at `tile_idx`.

        Parameters:
            tile_size: The size of the tile along the tiled axis used for
                vectorization.

        Args:
            tile_idx: The index of the tile to extract along the tiled axis.

        Returns:
            A SIMD[dtype] tile of size `tile_size` at `tile_idx`
        """
        return self.tile[tile_size, simd_width_of[Self.dtype]()](tile_idx)

    alias CornerCoordsType = IndexList[
        len(flatten(Self.layout.shape)),
        element_type = Self.layout_int_type,
    ]

    @always_inline
    fn tile_with_offset[
        *tile_sizes: Int,
    ](
        self,
        *tile_coords: Int,
    ) -> Tuple[
        Self.TileType[*tile_sizes],
        Self.CornerCoordsType,
        Scalar[Self.linear_idx_type],
    ]:
        """Similar to `tile`, but also returns the corner coordinates of the
        tile as well as the offset.

        Parameters:
            tile_sizes: The dimensions of each tile along each axis of the
                tensor.

        Args:
            tile_coords: The coordinates of the specific tile to extract.

        Returns:
            A tuple containing:
                - The extracted tile as a `LayoutTensor`.
                - The corner coordinates of the tile.
                - The offset of the tile.
        """
        alias num_tiles = stdlib.builtin.variadic_size(tile_sizes)

        # need to calculate this again because _tiled_layout[1] is required for the offset calculation
        alias _tiled_layout = Self._compute_tile_layout[*tile_sizes]()

        constrained[
            _tiled_layout[1].rank() == num_tiles,
            "Number of tiles should match the rank",
        ]()

        alias tile_type = self.TileType[*tile_sizes]

        # Static layout tiling
        # TODO: Consider merge the two cases in away that won't slowdown the fully static layout.
        var corner_coords = IndexList[
            len(flatten(self.layout.shape)), element_type = Self.layout_int_type
        ]()
        var offset: Scalar[Self.linear_idx_type] = 0
        var runtime_shape = tile_type.RuntimeLayoutType.ShapeType()
        var runtime_stride = tile_type.RuntimeLayoutType.StrideType()

        @parameter
        if tile_type.layout.all_dims_known():

            @parameter
            for i in range(num_tiles):
                alias stride = Int(_tiled_layout[1].stride[i])
                offset += tile_coords[i] * stride
                corner_coords[i] = tile_coords[i] * tile_sizes[i]

            var runtime_layout = tile_type.RuntimeLayoutType(
                runtime_shape, runtime_stride
            )

            # Adjust runtime layout, so the shape is clipped to the unmasked sizes.
            @parameter
            if tile_type.masked:

                @parameter
                for i in range(tile_type.layout.rank()):
                    cur_dim = self.dim[i]() - (tile_coords[i] * tile_sizes[i])
                    shape_i = max(0, min(tile_sizes[i], cur_dim))
                    runtime_layout.shape.value[i] = shape_i

            return (
                tile_type(self.ptr.offset(offset), runtime_layout),
                corner_coords,
                offset,
            )

        else:
            # Dynamic layout, use strides
            @parameter
            for i in range(num_tiles):
                var corner_coord = tile_coords[i] * tile_sizes[i]
                corner_coords[i] = corner_coord
                runtime_stride.value[i] = self.runtime_layout.stride.value[i]
                offset += self.runtime_layout.stride.value[i] * corner_coord

            var runtime_layout = tile_type.RuntimeLayoutType(
                runtime_shape, runtime_stride
            )

            # Adjusts the runtime layout so that the shape is clipped to the unmasked sizes.
            @parameter
            for i in range(tile_type.layout.rank()):
                cur_dim = self.dim[i]() - (tile_coords[i] * tile_sizes[i])
                shape_i = max(0, min(tile_sizes[i], cur_dim))
                runtime_layout.shape.value[i] = shape_i

            return (
                tile_type(self.ptr.offset(offset), runtime_layout),
                corner_coords,
                offset,
            )

    alias TiledIteratorType[
        *tile_sizes: Int,
        axis: Int = 0,
    ] = LayoutTensorIter[
        dtype,
        Self._compute_tile_layout[*tile_sizes]()[0],
        origin,
        address_space=address_space,
        circular=False,
        axis=axis,
        layout_int_type=layout_int_type,
        linear_idx_type=linear_idx_type,
        masked = masked or _tile_is_masked[layout, *tile_sizes](),
    ]

    @always_inline
    fn tiled_iterator[
        *tile_sizes: Int,
        axis: Int = 0,
    ](self, *tile_coords: Int) -> Self.TiledIteratorType[
        *tile_sizes, axis=axis
    ]:
        """Create an iterator that traverses tiles along a specified axis.

        This method creates an iterator that allows efficient traversal of tiles
        within a tensor. The iterator starts at the specified tile coordinates
        and can move along the specified axis, providing access to consecutive
        tiles.

        Parameters:
            tile_sizes: The dimensions of each tile along each axis of the
                tensor. For example, in a 2D tensor, `tiled_iterator[32, 32]`
                creates an iterator over 32x32 tiles.
            axis: The axis along which the iterator will traverse. Default is 0
                (first dimension). For example, with axis=0, the iterator will
                move vertically through tiles.

        Args:
            tile_coords: The starting coordinates of the tile where iteration
                begins.

        Returns:
            A `LayoutTensorIter` that can be used to traverse tiles along the
                specified axis.

        Performance:

        - Provides efficient sequential access to tiles with good cache
            locality.
        - Optimized for both static and dynamic layouts with different code
            paths.
        - Maintains stride information for efficient memory access within each
            tile.
        - Properly handles edge cases where tiles may be partially outside the
            tensor.

        Notes:

        - The iterator provides views into the original tensor, so modifications
            through the iterator will affect the original tensor.
        - For tiles at the edges of the tensor, the actual dimensions may be smaller
            than the requested tile_sizes if masking is enabled.
        - The iterator is not circular by default, meaning it will not wrap around
            when reaching the end of the tensor along the iteration axis.
        - The implementation automatically selects between static and dynamic tiling
            based on the tensor's layout properties.

        Example:

        ```mojo
        var iter = tensor.tiled_iterator[16, 16, axis=0](0, 0)
        for i in range(num_tiles_along_axis):
            var tile = iter.get()
            # Process tile
            iter.next()
        ```
        """

        alias tiles_rank = stdlib.builtin.variadic_size(tile_sizes)
        alias __tiled_layout = Self._compute_tile_layout[*tile_sizes]()
        constrained[
            __tiled_layout[1].rank() == tiles_rank,
            "Number of tiles should match the rank",
        ]()

        alias tiled_iterator_type = Self.TiledIteratorType[
            *tile_sizes, axis=axis
        ]

        var ptr_offset = 0

        @parameter
        if layout.all_dims_known():
            var runtime_shape = (
                tiled_iterator_type.RuntimeLayoutType.ShapeType()
            )
            var runtime_stride = (
                tiled_iterator_type.RuntimeLayoutType.StrideType()
            )

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
            if tiled_iterator_type.masked:

                @parameter
                for i in range(tiled_iterator_type.layout.rank()):
                    cur_dim = self.dim[i]() - (tile_coords[i] * tile_sizes[i])
                    shape_i = max(0, min(tile_sizes[i], cur_dim))
                    runtime_shape.value[i] = shape_i

                return tiled_iterator_type(
                    self.ptr + ptr_offset,
                    bound,
                    tiled_iterator_type.RuntimeLayoutType(
                        runtime_shape, runtime_stride
                    ),
                    stride=stride,
                    offset=0,
                    dimension_bound=dim_bound,
                    idx=tile_coords[axis],
                )
            else:
                return tiled_iterator_type(
                    self.ptr + ptr_offset,
                    bound,
                    stride=stride,
                    offset=0,
                )

        else:
            var runtime_shape = (
                tiled_iterator_type.RuntimeLayoutType.ShapeType()
            )
            var runtime_stride = (
                tiled_iterator_type.RuntimeLayoutType.StrideType()
            )

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
            for i in range(tiled_iterator_type.layout.rank()):
                cur_dim = self.dim[i]() - (tile_coords[i] * tile_sizes[i])
                shape_i = max(0, min(tile_sizes[i], cur_dim))
                runtime_shape.value[i] = shape_i

            return tiled_iterator_type(
                self.ptr + ptr_offset,
                iter_bound,
                stride=iter_stride,
                offset=0,
                runtime_layout=tiled_iterator_type.RuntimeLayoutType(
                    runtime_shape, runtime_stride
                ),
                dimension_bound=self.dim[axis](),
                idx=tile_coords[axis],
            )

    alias SplitElementType[
        count: Int,
        axis: Int = 0,
    ] = LayoutTensor[
        dtype,
        Self._compute_tile_layout[layout.shape[axis].value() // count, axis]()[
            0
        ],
        origin,
        address_space=address_space,
        element_layout=element_layout,
        alignment=alignment,
    ]

    alias StaticSplitType[
        count: Int,
        axis: Int = 0,
    ] = StaticTuple[
        Self.SplitElementType[count, axis],
        count,
    ]

    @always_inline
    fn split[
        count: Int,
        axis: Int = 0,
    ](self) -> Self.StaticSplitType[count, axis]:
        """Split the `LayoutTensor` along a axis and return a `StaticTuple` of
        `LayoutTensor`.

        Parameters:
            count: Number of portion to split.
            axis: The axis where the split is applied to.

        Returns:
            A `StaticTuple` containing `count` `LayoutTensors`, each
            representing an equal-sized partition of the original tensor along
            the specified axis. Each partition has the same data type and memory
            characteristics as the original tensor, but with a reduced size
            along the split axis.
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
        var tiles = Self.StaticSplitType[count, axis]()

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

    alias DynamicSplitType[
        axis: Int = 0,
    ] = LayoutTensor[
        dtype,
        layout.make_shape_unknown[axis](),
        origin,
        address_space=address_space,
        element_layout=element_layout,
        layout_int_type=layout_int_type,
        linear_idx_type=linear_idx_type,
    ]

    @always_inline
    fn split[
        axis: Int = 0,
        split_alignment: Int = 1,
    ](self, count: Int, idx: Int) -> Self.DynamicSplitType[axis]:
        """Retrieve a specific partition of the tensor after splitting along a
        specified axis.

        This method divides the tensor into 'count' partitions along the
        specified axis and returns the partition at index 'idx'. The
        partitioning is done with alignment considerations to optimize memory
        access patterns.

        Unlike the overloaded split method that returns all partitions, this
        method returns only a single partition, making it more memory-efficient
        for cases where only one partition is needed at a time.

        Constraints:
            - The dimension being split must have a statically known size.
            - Cannot split dimensions with unknown or dynamic sizes.

        Parameters:
            axis: The axis along which to split the tensor. Defaults to 0 (first
                dimension).
            split_alignment: Memory alignment value for the partition size. Defaults
                to 1.

        Args:
            count: The number of partitions to divide the tensor into.
            idx: The index of the partition to return (0-based).

        Returns:
            A `LayoutTensor` representing the requested partition.

        Notes:

        - The shape along the split axis becomes unknown at compile time.
        - Only works with dimensions that have statically known sizes.
        - The last partition may be smaller than others if the dimension size
            is not evenly divisible by `count`.
        - Partition sizes are aligned up to the specified alignment value,
            which can improve performance for vectorized operations.

        Performance:

        - Uses aligned partitioning to improve memory access patterns.
        - Avoids creating all partitions in memory, reducing memory usage.
        - Maintains the original tensor's stride information for efficient
            element access within the partition.
        """
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

        var runtime_shape = Self.DynamicSplitType[
            axis
        ].RuntimeLayoutType.ShapeType()
        var axis_partition_dim = align_up(axis_dim // count, split_alignment)

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

        return Self.DynamicSplitType[axis](
            # Only the last partition can have size other than axis_partition_dim.
            self.ptr + idx * axis_partition_dim * axis_stride,
            Self.DynamicSplitType[axis].RuntimeLayoutType(
                runtime_shape,
                rebind[
                    Self.DynamicSplitType[axis].RuntimeLayoutType.StrideType
                ](self.runtime_layout.stride),
            ),
        )

    @always_inline
    fn _clamp_distribute_shape[
        thread_layout: Layout,
    ](self, thread_id: UInt) -> IndexList[
        Self.rank, element_type=layout_int_type
    ]:
        constrained[
            len(flatten(thread_layout.shape)) <= 2
            and len(flatten(thread_layout.stride)) <= 2,
            "Only supporting rank-2 or less thread layout for dynamic tile.",
        ]()

        # clamp IndexList using thread_id and thread_layout
        var tile_shape = IndexList[Self.rank, element_type=layout_int_type]()
        alias thread_shape = thread_layout.shape
        alias thread_stride = thread_layout.stride

        # this would only work for rank-2 thread layout, need to extend this
        # to support thread layout such as Layout((2, 2), 2)
        @parameter
        for i in range(Self.rank):
            alias thread_stride_i = Int(thread_stride[i])
            alias thread_shape_i = Int(thread_shape[i])
            var tile_idx = (thread_id // UInt(thread_stride_i)) % UInt(
                thread_shape_i
            )
            var tile_shape_i = ceildiv(self.dim[i](), thread_shape_i)
            var bound_i = Int((tile_shape_i - 1) * thread_shape_i + tile_idx)
            tile_shape[i] = min(self.dim[i]() - bound_i, tile_shape_i)

        return tile_shape

    alias DistributeType[
        threads_layout: Layout,
        axis: OptionalReg[Int] = None,
    ] = LayoutTensor[
        dtype,
        _compute_distribute_layout[
            layout,
            threads_layout,
            axis,
        ]()[1],
        origin,
        address_space=address_space,
        element_layout=element_layout,
        layout_int_type=layout_int_type,
        linear_idx_type=linear_idx_type,
        # TODO: This is a workaround as we don't need masking support for AMD GPU
        # if we use buffer stores and loads. Probably need a better solution
        # in the long term, if someone ends up using global loads and stores
        # it may lead to out of bounds access.
        masked = (
            masked or _distribute_is_masked[layout, threads_layout, axis]()
        ) if is_nvidia_gpu() else False,
    ]

    @always_inline
    fn distribute[
        threads_layout: Layout,
        axis: OptionalReg[Int] = None,
        swizzle: OptionalReg[Swizzle] = None,
        submode_axis: OptionalReg[Int] = None,
    ](self, thread_id: UInt,) -> Self.DistributeType[threads_layout, axis]:
        """Distribute tensor workload across multiple threads in a structured
        pattern.

        This method partitions a tensor across multiple threads for parallel
        processing, assigning each thread a specific portion of the tensor. The
        distribution pattern is determined by the threads_layout parameter,
        which defines the logical arrangement of threads.

        Constraints:
            - For dynamic layouts, the shape must be known at runtime and the
                threads_layout must be fully static.

        Parameters:
            threads_layout: Defines the logical arrangement of threads (e.g.,
                2x2 grid of 4 threads). This layout determines how the tensor is
                partitioned.
            axis: Optional. If specified, restricts distribution to only this
                axis. For example, with axis=0 in a 2D thread layout, threads
                that differ only in their second coordinate will receive the
                same data.
            swizzle: Optional. A function that remaps the distribution pattern
                to improve memory access patterns or cache locality.
            submode_axis: Optional. Specifies an axis for specialized
                distribution modes.

        Args:
            thread_id: The ID of the current thread (0-based).

        Returns:
            A view into the original tensor representing the portion assigned to
            this thread.

        Example:

        For a 4x4 row-major tensor distributed across 4 threads in a 2x2 row-major grid:

        - Thread 0 will receive a LayoutTensor with a view into
            (0,0), (0,2), (2,0), (2,2) of the original tensor.
        - Thread 1 will receive a LayoutTensor with a view into
            (0,1), (0,3), (2,1), (2,3) of the original tensor.
        - Thread 2 will receive a LayoutTensor with a view into
            (1,0), (1,2), (3,0), (3,2) of the original tensor.
        - Thread 3 will receive a LayoutTensor with a view into
            (1,1), (1,3), (3,1), (3,3) of the original tensor.

        If axis=0 is specified with the same setup:

        - Thread (0, 0) and Thread (0, 1) would get the same data (top half)
        - Thread (1, 0) and Thread (1, 1) would get the same data (bottom half)

        Performance:

        - Creates a view without copying data, making it very efficient for
            parallel processing.
        - The swizzle parameter can significantly improve cache locality and
            memory access patterns.
        - Optimized for both static and dynamic layouts with different code
            paths.

        Notes:

        - The resulting tensor is a view into the original tensor, so
            modifications will affect the original tensor.
        - For optimal performance, the `threads_layout` should match the
            hardware's thread organization (e.g., warp/wavefront size and shape).
        - When using swizzling, carefully consider the memory access patterns to
            avoid cache thrashing or bank conflicts.
        - This function is particularly useful for GPU programming where threads
            are organized in structured grids.
        """

        alias distribute_type = Self.DistributeType[threads_layout, axis]
        alias runtime_layout_type = distribute_type.RuntimeLayoutType
        alias runtime_shape_type = runtime_layout_type.ShapeType
        alias runtime_stride_type = runtime_layout_type.StrideType

        alias distributed_layout = _compute_distribute_layout[
            layout,
            threads_layout,
            axis,
        ]()

        var runtime_shape: runtime_shape_type

        @parameter
        if distribute_type.masked:
            runtime_shape = runtime_shape_type(
                self._clamp_distribute_shape[threads_layout](thread_id)
            )
        else:
            runtime_shape = runtime_shape_type()

        var runtime_stride = runtime_stride_type()

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

            var offset: Scalar[linear_idx_type] = 0

            @parameter
            for i in range(len(fragments_layout_stride)):
                alias fragments_stride_i: UInt = UInt(
                    mlir_value=Int(fragments_layout_stride[i])._mlir_value
                )
                alias shape_i: UInt = UInt(Int(thread_projected_shape[i]))
                alias stride_i: UInt = UInt(Int(thread_projected_stride[i]))
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
            if distribute_type.masked:
                return distribute_type(
                    self.ptr.offset(Int(swizzled_offset)),
                    runtime_layout_type(runtime_shape, runtime_stride),
                )
            else:
                return distribute_type(
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

            var offset: Scalar[linear_idx_type] = 0

            @parameter
            for i in range(runtime_shape.scalar_length):
                alias thread_shape_i = threads_layout[i].size()
                runtime_stride.value[i] = (
                    self.runtime_layout.stride.value[i] * thread_shape_i
                )

            @parameter
            for i in range(len(flatten(Self.layout.stride))):
                var fragments_stride_i = self.runtime_layout.stride.value[i]
                alias shape_i: UInt = UInt(
                    mlir_value=Int(thread_projected_shape[i])._mlir_value
                )
                alias stride_i: UInt = UInt(
                    mlir_value=Int(thread_projected_stride[i])._mlir_value
                )
                var thread_coord_i: UInt = (thread_id // stride_i) % shape_i
                offset += thread_coord_i * UInt(fragments_stride_i)

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
                return distribute_type(
                    self.ptr.offset(Int(swizzled_offset)),
                    runtime_layout_type(runtime_shape, runtime_stride),
                )
            else:
                return distribute_type(
                    self.ptr.offset(Int(swizzled_offset)),
                    runtime_layout_type(runtime_shape, runtime_stride),
                    self.runtime_element_layout,
                )

    @always_inline
    fn distribute_with_offset[
        threads_layout: Layout,
        axis: OptionalReg[Int] = None,
        swizzle: OptionalReg[Swizzle] = None,
        submode_axis: OptionalReg[Int] = None,
    ](
        self,
        thread_id: UInt,
    ) -> Tuple[
        Self.DistributeType[threads_layout, axis],
        IndexList[threads_layout.rank(), element_type=layout_int_type],
        Scalar[linear_idx_type],
    ]:
        """Similar to `distribute`, but also returns the corner coordinates of
        the tile as well as the offset.

        Parameters:
            threads_layout: The layout of the threads.
            axis: The axis to distribute along.
            swizzle: An optional swizzle function.
            submode_axis: An optional submode axis.

        Args:
            thread_id: The ID of the current thread (0-based).

        Returns:
            A tuple containing:
                - The distributed tensor.
                - The corner coordinates of the tile.
                - The offset of the tile.
        """
        alias ret_tensor_type = Self.DistributeType[threads_layout, axis]
        alias distributed_layout = _compute_distribute_layout[
            layout,
            threads_layout,
            axis,
        ]()

        @parameter
        if ret_tensor_type.masked:
            runtime_shape = ret_tensor_type.RuntimeLayoutType.ShapeType(
                self._clamp_distribute_shape[threads_layout](thread_id)
            )
        else:
            runtime_shape = ret_tensor_type.RuntimeLayoutType.ShapeType()

        var runtime_stride = ret_tensor_type.RuntimeLayoutType.StrideType()
        var offset_coords = IndexList[
            threads_layout.rank(), element_type=layout_int_type
        ]()
        var offset: Scalar[linear_idx_type] = 0

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

            @parameter
            for i in range(len(fragments_layout_stride)):
                alias fragments_stride_i: UInt = UInt(
                    mlir_value=Int(fragments_layout_stride[i])._mlir_value
                )
                alias shape_i: UInt = UInt(Int(thread_projected_shape[i]))
                alias stride_i: UInt = UInt(Int(thread_projected_stride[i]))
                var thread_coord_i: UInt = (thread_id // stride_i) % shape_i
                offset_coords[i] = Int(thread_coord_i)
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
            if ret_tensor_type.masked:
                return (
                    ret_tensor_type(
                        self.ptr.offset(Int(swizzled_offset)),
                        ret_tensor_type.RuntimeLayoutType(
                            runtime_shape, runtime_stride
                        ),
                    ),
                    offset_coords,
                    swizzled_offset,
                )
            else:
                return (
                    ret_tensor_type(
                        self.ptr.offset(Int(swizzled_offset)),
                    ),
                    offset_coords,
                    swizzled_offset,
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

            @parameter
            for i in range(runtime_shape.scalar_length):
                alias thread_shape_i = threads_layout[i].size()
                runtime_stride.value[i] = (
                    self.runtime_layout.stride.value[i] * thread_shape_i
                )

            @parameter
            for i in range(len(flatten(Self.layout.stride))):
                var fragments_stride_i = self.runtime_layout.stride.value[i]
                alias shape_i: UInt = UInt(
                    mlir_value=Int(thread_projected_shape[i])._mlir_value
                )
                alias stride_i: UInt = UInt(
                    mlir_value=Int(thread_projected_stride[i])._mlir_value
                )
                var thread_coord_i: UInt = (thread_id // stride_i) % shape_i
                offset_coords[i] = Int(thread_coord_i)
                offset += thread_coord_i * UInt(fragments_stride_i)

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
                return (
                    ret_tensor_type(
                        self.ptr.offset(Int(swizzled_offset)),
                        ret_tensor_type.RuntimeLayoutType(
                            runtime_shape, runtime_stride
                        ),
                    ),
                    offset_coords,
                    swizzled_offset,
                )
            else:
                return (
                    ret_tensor_type(
                        self.ptr.offset(Int(swizzled_offset)),
                        ret_tensor_type.RuntimeLayoutType(
                            runtime_shape, runtime_stride
                        ),
                        self.runtime_element_layout,
                    ),
                    offset_coords,
                    swizzled_offset,
                )

    alias ShapeVectorizedType[
        origin: ImmutableOrigin,
        vector_shape: IntTuple[origin],
        linear_vectorize: Bool,
    ] = LayoutTensor[
        dtype,
        coalesce(
            Self._tuple_divide_tiles(vector_shape, linear_vectorize)[1],
            keep_rank=True,
        ),
        origin,
        address_space=address_space,
        element_layout = Self._tuple_divide_tiles(
            vector_shape, linear_vectorize
        )[0],
        layout_int_type=layout_int_type,
        linear_idx_type=linear_idx_type,
        masked=masked,
    ]

    @always_inline
    fn _vectorize_2[
        vector_len: Int,
        linear_vectorize: Bool = True,
    ](self) -> Self.ShapeVectorizedType[
        __origin_of(),
        IntTuple(vector_len),
        linear_vectorize=linear_vectorize,
    ]:
        """Wrap the integer `vector_len` in an `IntTuple` and call the
        `_vectorize_2` function.

        Parameters:
            vector_len: The length of vectorization.
            linear_vectorize: Whether to vectorize in a linear manner. Defaults to True.

        Returns:
            A view of the tensor with a vectorized layout based on the specified
            vector length.
        """
        return self._vectorize_2[
            __origin_of(),
            IntTuple(vector_len),
            linear_vectorize=linear_vectorize,
        ]()

    @always_inline
    fn _vectorize_2[
        _origin: ImmutableOrigin,  # FIXME: MOCO-1912
        vector_shape: IntTuple[_origin],
        check_rank: Bool = True,
        linear_vectorize: Bool = vector_shape.is_value(),
    ](self) -> Self.ShapeVectorizedType[
        _origin, vector_shape, linear_vectorize
    ]:
        """Experimental implementation of the generalized vectorize operation
        using IntTuple.

        This function creates a vectorized view of the tensor using an IntTuple
        to specify the vector dimensions rather than variadic parameters.

        Parameters:
            _origin: The origin of the IntTuple.
            vector_shape: The dimensions of each vector unit as an IntTuple.
            check_rank: Whether to verify that vector_shape is congruent with
                the tensor's shape. Defaults to True.
            linear_vectorize: Whether to vectorize in a linear manner. Defaults to True.

        Returns:
            A view of the tensor with a vectorized layout based on the specified
            vector shape.
        """
        constrained[
            (vector_shape.is_value() and linear_vectorize)
            or (not linear_vectorize),
            (
                "Only contiguous vectorization or vectorization of a"
                " congruent shape is supported!"
            ),
        ]()

        alias vectorized_type = Self.ShapeVectorizedType[
            _origin, vector_shape, linear_vectorize
        ]
        runtime_shape = vectorized_type.RuntimeLayoutType.ShapeType()
        runtime_stride = vectorized_type.RuntimeLayoutType.StrideType()

        @parameter
        if check_rank:
            constrained[
                is_int(vector_shape) or congruent(vector_shape, layout.shape),
                "vector_shape has to be congruent to layout.shape = ",
                String(layout.shape),
            ]()

        alias tiler = Self._tuple_divide_tiler(vector_shape, linear_vectorize)
        alias flat_vector_shape = flatten(tiler.shape)

        @parameter
        if vectorized_type.masked or not layout.all_dims_known():

            @parameter
            for i in range(len(flat_vector_shape)):
                alias vector_shape_i = Int(flat_vector_shape[i])
                runtime_shape.value[i] = ceildiv(
                    self.runtime_layout.shape.value[i], vector_shape_i
                )
                runtime_stride.value[i] = (
                    self.runtime_layout.stride.value[i] * vector_shape_i
                )

        @parameter
        if layout.all_dims_known():

            @parameter
            if vectorized_type.masked:
                return vectorized_type(
                    self.ptr,
                    vectorized_type.RuntimeLayoutType(
                        runtime_shape, runtime_stride
                    ),
                )
            else:
                return vectorized_type(self.ptr)
        else:
            constrained[
                coalesce(vectorized_type.element_layout).known_shape(),
                "Result element layout should have known shape",
            ]()

            runtime_element_layout_shape = (
                vectorized_type.RuntimeElementLayoutType.ShapeType()
            )
            runtime_element_layout_stride = (
                vectorized_type.RuntimeElementLayoutType.StrideType(
                    self.runtime_layout.stride.value
                )
            )

            return Self.ShapeVectorizedType[
                _origin, vector_shape, linear_vectorize
            ](
                self.ptr,
                vectorized_type.RuntimeLayoutType(
                    runtime_shape, runtime_stride
                ),
                vectorized_type.RuntimeElementLayoutType(
                    runtime_element_layout_shape,
                    runtime_element_layout_stride,
                ),
            )

    alias VectorizedType[*vector_shape: Int] = LayoutTensor[
        dtype,
        coalesce(Self._compute_tile_layout[*vector_shape]()[1], keep_rank=True),
        origin,
        address_space=address_space,
        element_layout = Self._divide_tiles[*vector_shape]()[0],
        layout_int_type=layout_int_type,
        linear_idx_type=linear_idx_type,
        masked=masked,
    ]

    @always_inline
    fn vectorize[
        *vector_shape: Int
    ](self) -> Self.VectorizedType[*vector_shape]:
        """Reshape a tensor into a vectorized form for efficient SIMD
        operations.

        This method transforms the tensor's logical layout to enable efficient
        vectorized processing, treating blocks of elements as vector units. The
        transformation is particularly useful for SIMD (Single Instruction
        Multiple Data) operations and hardware acceleration.

        Constraints:
            - Each tensor dimension must be divisible by the corresponding
                vector dimension.
            - Vector dimensions must be smaller than or equal to the
                corresponding tensor dimensions.
            - For dimensions with unknown size, the vector dimension must be 1.

        Parameters:
            vector_shape: The dimensions of each vector unit along each axis of
                the tensor. or example, in a 2D tensor, `vectorize[4, 4]` treats
                4x4 blocks as vector units.

        Returns:
            A view of the tensor with a vectorized layout, where each element in
            the resulting tensor represents a vector of elements from the
            original tensor.

        Example:

        For a 16x16 tensor, `vectorize[4, 4]` will produce a 4x4 tensor
        where each element represents a 4x4 block from the original tensor.

        Performance:

        - Creates a view without copying data, making it very efficient.
        - Enables hardware-accelerated vector operations on blocks of data.
        - Improves cache locality by grouping related elements together.
        - Particularly beneficial for operations that can leverage SIMD
            instructions.

        Notes:

        - The tensor dimensions must be divisible by the corresponding vector
            dimensions.
        - For dimensions with unknown size, the corresponding vector dimension
            must be 1.
        - The resulting tensor has the same data but a different logical
            organization.
        - Modifications to the vectorized tensor affect the original tensor.
        - This transformation is particularly useful for GPU and vector
            processor optimizations.
        """

        alias shape = IntTuple(vector_shape)
        alias _origin = __origin_of()  # FIXME: MOCO-1912
        var ret = self._vectorize_2[
            _origin,
            shape,
            check_rank=False,
            linear_vectorize=False,
        ]()
        # FIXME: this is ugly, is there a simpler way to do this?
        return rebind[Self.VectorizedType[*vector_shape]](ret)

    alias SIMDVectorizedType = Self.VectorizedType[1, simd_width_of[dtype]()]

    @always_inline
    fn vectorize(self) -> Self.SIMDVectorizedType:
        """Return a SIMD[dtype] vectorized view of this tensor.

        Returns:
            A `Self.VectorizedType[1, simd_width_of[Self.dtype]()]` view whose
            width equals the SIMD width for the tensor's dtype.
        """
        return self.vectorize[1, simd_width_of[Self.dtype]()]()

    @staticmethod
    fn _compute_slice_layout(d0_slice: Slice, d1_slice: Slice) -> Layout:
        constrained[
            layout.shape.__len__() == 2,
            "Only rank-2 tensors slices are supported for now!",
        ]()
        return Layout(
            [
                _get_slice_size(Self.layout, d0_slice, 0),
                _get_slice_size(Self.layout, d1_slice, 1),
            ],
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
            [
                _get_slice_size(sliced_layout, slice_0, 0),
                _get_slice_size(sliced_layout, slice_1, 1),
            ],
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
            [_get_slice_size(sliced_layout, slice_0, 0)],
            sliced_layout.stride[0],
        )

    alias SliceType[
        d0_slice: Slice,
        d1_slice: Slice,
    ] = LayoutTensor[
        dtype,
        Self._compute_slice_layout(
            d0_slice,
            d1_slice,
        ),
        origin,
        address_space=address_space,
        element_layout=element_layout,
        layout_int_type=layout_int_type,
        linear_idx_type=linear_idx_type,
    ]

    @always_inline
    fn slice[
        d0_slice: Slice, d1_slice: Slice
    ](self) -> Self.SliceType[d0_slice, d1_slice]:
        """Extract a slice from a rank-2 tensor using slice objects.

        This method creates a view into a subset of the tensor defined by the
        slice specifications for each dimension. The slice is a continuous
        region of the tensor with no gaps (step size must be 1).

        Constraints:
            - Only works with rank-2 tensors.

        Parameters:
            d0_slice: Slice specification for the first dimension (rows).
                Defines the start and end indices for the slice along this
                dimension.
            d1_slice: Slice specification for the second dimension (columns).
                Defines the start and end indices for the slice along this
                dimension.

        Returns:
            A view into the original tensor representing the specified slice.

        Example:

        For a 4x4 tensor, `t` with values:

        ```
        [1 2 3 4]
        [5 6 7 8]
        [9 10 11 12]
        [13 14 15 16]
        ```

        ```mojo
        t.slice[Slice(1, 3), Slice(0, 2)]
        ```

        will extract:

        ```
        [5 6]
        [9 10]
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
        - Only supports rank-2 tensors. For higher-rank tensors, use the
            overloaded version with slice indices.
        - The step size must be 1 (no gaps allowed in the slice).
        - Slice bounds are not checked at runtime; accessing out-of-bounds
            indices will result in undefined behavior.
        """
        constrained[
            d0_slice.step.or_else(1) == 1 and d1_slice.step.or_else(1) == 1,
            "Slice should have no gaps",
        ]()

        alias return_type = Self.SliceType[d0_slice, d1_slice]
        alias stride_m = Int(return_type.layout.stride[0])
        alias stride_n = Int(return_type.layout.stride[1])

        alias d0_slice_start = d0_slice.start.or_else(0)
        alias d1_slice_start = d1_slice.start.or_else(0)

        var offset = d0_slice_start * stride_m + d1_slice_start * stride_n

        return Self.SliceType[d0_slice, d1_slice](self.ptr.offset(offset))

    alias SliceType2D[
        d0_slice: Slice,
        d1_slice: Slice,
        slice_indices: IndexList[2],
        __offset_dims: Int = Self.rank - 2,
    ] = LayoutTensor[
        dtype,
        Self._compute_slice_layout(
            d0_slice, d1_slice, slice_indices[0], slice_indices[1]
        ),
        origin,
        address_space=address_space,
        element_layout=element_layout,
        layout_int_type=layout_int_type,
        linear_idx_type=linear_idx_type,
    ]

    @always_inline
    fn slice[
        d0_slice: Slice,
        d1_slice: Slice,
        slice_indices: IndexList[2],
        __offset_dims: Int = Self.rank - 2,
    ](
        self,
        offsets: IndexList[__offset_dims],
    ) -> Self.SliceType2D[
        d0_slice, d1_slice, slice_indices, __offset_dims
    ]:
        """Extract a 2D slice from a higher-rank tensor at specific indices.

        This method creates a view into a 2D subset of a higher-rank tensor:

        Selecting two dimensions to slice using the slice_indices parameter.
        Applying slice specifications to those dimensions.
        Using fixed offsets for all other dimensions.

        Constraints:
            - Slice step size must be 1 (no gaps).
            - Slice indices must be ordered (ascending).
            - Tensor rank must be at least 2.

        Parameters:
            d0_slice: Slice specification for the first selected dimension.
            d1_slice: Slice specification for the second selected dimension.
            slice_indices: Indices of the two dimensions to slice (must be
                ordered).
            __offset_dims: Internal parameter representing number of fixed
                dimensions.

        Args:
            offsets: Fixed index values for all dimensions not being sliced.

        Returns:
            A 2D view into the original tensor representing the specified slice.

        Example:

        Given a 3x4x5 tensor, `t`, the following example extracts a 2x2 slice
        from dimensions 0 and 2, with dimension 1 fixed at index 1.

        ```mojo
        t.slice = t.slice[Slice(1, 3), Slice(0, 2), IndexList[2](0, 2)](1)
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
        - The slice indices must be ordered (e.g., [0, 2] is valid, [2, 0] is
            not).
        - The step size must be 1 (no gaps allowed in the slice).
        - Slice bounds are not checked at runtime; accessing out-of-bounds
            indices will result in undefined behavior.
        """
        constrained[
            d0_slice.step.or_else(1) == 1 and d1_slice.step.or_else(1) == 1,
            "Slice should have no gaps",
        ]()
        constrained[
            slice_indices[0] < slice_indices[1],
            "Slice indices should be ordered",
        ]()
        alias slice_type = Self.SliceType2D[
            d0_slice, d1_slice, slice_indices, __offset_dims
        ]

        alias stride_0 = Int(slice_type.layout.stride[0])
        alias stride_1 = Int(slice_type.layout.stride[1])

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

        return Self.SliceType2D[
            d0_slice, d1_slice, slice_indices, __offset_dims
        ](self.ptr.offset(slice_offset))

    alias SliceType1D[
        d0_slice: Slice,
        slice_indices: IndexList[1],
        __offset_dims: Int = Self.rank - 1,
    ] = LayoutTensor[
        dtype,
        Self._compute_slice_layout(d0_slice, slice_indices[0]),
        origin,
        address_space=address_space,
        element_layout=element_layout,
        layout_int_type=layout_int_type,
        linear_idx_type=linear_idx_type,
    ]

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
    ) -> Self.SliceType1D[
        d0_slice, slice_indices, __offset_dims
    ]:
        """Extract a 1D slice from a higher-rank tensor at a specific index.

        This method creates a view into a 1D subset of a higher-rank tensor by:
        1. Selecting one dimension to slice using the slice_indices parameter
        2. Applying a slice specification to that dimension
        3. Using fixed offsets for all other dimensions

        Constraints:
            - Slice step size must be 1 (no gaps).
            - Tensor rank must be at least 1.

        Parameters:
            d0_slice: Slice specification for the selected dimension.
            slice_indices: Index of the dimension to slice.
            __offset_dims: Internal parameter representing number of fixed
                dimensions.

        Args:
            offsets: Fixed index values for all dimensions not being sliced.

        Returns:
            A 1D view into the original tensor representing the specified slice.

        Example:

        For a 3x4x5 tensor, `t`, the following example extracts a 1D slice from
        dimension 0, with dimensions 1 and 2 fixed at indices 1 and 2:

        ```mojo
        t.slice_1d[Slice(1, 3), IndexList[1](0)](1, 2)`
        ```

        Performance:

        - Creates a view without copying data, making it very efficient.
        - Maintains the original tensor's stride information for efficient
            memory access.
        - Zero-cost abstraction at runtime when used with compile-time constant
            slices.

        Notes:

        - The slice is a view into the original tensor, so modifications
            to the slice will affect the original tensor.
        - The step size must be 1 (no gaps allowed in the slice).
        - Slice bounds are not checked at runtime; accessing out-of-bounds
            indices will result in undefined behavior.
        - This function exists as a workaround for compiler limitations with
            overloading.
        """
        constrained[
            d0_slice.step.or_else(1) == 1,
            "Slice should have no gaps",
        ]()

        alias slice_type = Self.SliceType1D[
            d0_slice, slice_indices, __offset_dims
        ]

        alias stride_0 = Int(slice_type.layout.stride[0])

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

        return Self.SliceType1D[d0_slice, slice_indices, __offset_dims](
            self.ptr.offset(slice_offset)
        )

    alias TransposeType = LayoutTensor[
        dtype,
        layout.transpose(),
        origin,
        address_space=address_space,
        element_layout=element_layout,
        layout_int_type=layout_int_type,
        linear_idx_type=linear_idx_type,
    ]

    @always_inline
    fn transpose(self) -> Self.TransposeType:
        """Create a transposed view of a tensor.

        This method creates a view of the tensor with its dimensions swapped, effectively
        converting rows to columns and columns to rows. The transposition is performed
        without copying data, by adjusting the tensor's layout information.

        Returns:
            A view of the tensor with dimensions transposed (rows become columns and vice versa).

        Example:

        For a 2x3 tensor with values:

        ```
        [1 2 3]
        [4 5 6]
        ```

        `transpose()` will produce a 3x2 tensor:

        ```
        [1 4]
        [2 5]
        [3 6]
        ```

        Performance:

        - Creates a view without copying data, making it very efficient.
        - The operation is zero-cost at runtime as it only changes the layout
            information.
        - Memory access patterns may be less efficient in the transposed view
            due to non-contiguous memory access, especially for row-major
            storage.

        Notes:

        - The transposed tensor shares the same memory as the original tensor,
            so modifications to one will affect the other.
        - For optimal performance when repeatedly accessing the transposed data,
            consider creating a physical copy with the transposed layout.
        - Transpose only works with statically known shapes.
        """
        constrained[
            layout.all_dims_known(),
            "Transpose only works with statically known shapes.",
        ]()
        return Self.TransposeType(self.ptr)

    alias ReshapeType[
        dst_layout: Layout,
    ] = LayoutTensor[
        dtype,
        dst_layout,
        origin,
        address_space=address_space,
        element_layout=element_layout,
        layout_int_type=layout_int_type,
        linear_idx_type=linear_idx_type,
        masked=masked,
        alignment=alignment,
    ]

    @always_inline
    fn reshape[
        dst_layout: Layout,
    ](self) -> Self.ReshapeType[dst_layout]:
        """Create a view of the tensor with a different shape.

        This method creates a view of the tensor with a new shape, without changing
        the underlying data. The total number of elements must remain the same.

        Constraints:
            - Cannot reshape masked tensors.
            - The total number of elements must be the same in both layouts.

        Parameters:
            dst_layout: The target layout for the reshaped tensor. Must have the same
                       total number of elements as the original tensor.

        Returns:
            A view of the tensor with the new shape specified by dst_layout.

        Example:

        Given a 2x6 row-major tensor, `reshape[Layout.col_major(3, 4)]()`
        produces a 3x4 tensor with the same elements in column-major order.

        Performance:

        - Creates a view without copying data, making it very efficient.
        - The operation is zero-cost at runtime as it only changes the layout
            information.
        - Memory access patterns may change, potentially affecting performance
            depending on the original and target layouts.

        Notes:

        - The reshaped tensor shares the same memory as the original tensor,
            so modifications to one will affect the other.
        - The total number of elements must remain the same after reshaping.
        - The reshape operation assumes a row-major (C-style) memory layout.
        - For tensors with complex strides or non-contiguous memory, reshaping
            may not produce the expected results.
        - Masked tensors cannot be reshaped.
        """
        constrained[not masked, "Masked tensor does not support reshape."]()
        return Self.ReshapeType[dst_layout](self.ptr)

    alias CompositionType[
        rhs_layout: Layout,
        dst_layout: Layout = composition(layout, rhs_layout),
    ] = LayoutTensor[
        dtype,
        dst_layout,
        origin,
        address_space=address_space,
        element_layout=element_layout,
        layout_int_type=layout_int_type,
        linear_idx_type=linear_idx_type,
    ]

    @always_inline
    fn composition[
        rhs_layout: Layout,
        dst_layout: Layout = composition(layout, rhs_layout),
    ](self, out result: self.CompositionType[rhs_layout, dst_layout]):
        """Create a view of the tensor with a composed layout.

        This method creates a view of the tensor with a new layout that is the
        composition of the original layout with another layout. Layout
        composition allows for complex transformations of the tensor's logical
        structure without copying data.

        Constraints:
            - The layouts must be compatible for composition.
            - The total number of elements must remain the same after
                composition.

        Parameters:
            rhs_layout: The layout to compose with the tensor's current layout.
            dst_layout: The resulting layout after composition. Defaults to the
                       composition of the tensor's layout with rhs_layout.

        Returns:
            A view of the tensor with the composed layout.

        Example:

        For a 4x4 tensor with a standard row-major layout, composing with a
        layout that represents a 2x2 tiling would result in a tensor that
        logically views the data as 2x2 blocks.

        Performance:

        - Creates a view without copying data, making it very efficient.
        - The operation is zero-cost at runtime as it only changes the layout information.
        - Can be used to optimize memory access patterns for specific algorithms.

        Notes:

        - The composed tensor shares the same memory as the original tensor,
            so modifications to one will affect the other.
        - Layout composition is a powerful tool for expressing complex data
            transformations like tiling, transposition, and reshaping in a
            unified framework.
        - Understanding the mathematical properties of layout composition is
            important for correctly using this function.
        """
        return self.CompositionType[rhs_layout, dst_layout](self.ptr)

    @always_inline
    fn distance(
        self,
        addr: UnsafePointer[Scalar[dtype], address_space=address_space, *_],
    ) -> Scalar[linear_idx_type]:
        """Calculate the element-wise distance between this tensor's pointer
        and another pointer.

        This method computes the number of elements (not bytes) between the
        tensor's pointer and the provided address. This is useful for
        determining offsets within a larger memory allocation or for pointer
        arithmetic operations.

        Args:
            addr: The target pointer to calculate the distance to.

        Returns:
            The number of elements between this tensor's pointer and the
            provided address. The result is of type `_uint_dtype`.

        Example:

        If `tensor.ptr` points to an element at index 100 in a buffer, and
        `addr` points to element at index 50, then `distance(addr)` returns 50.

        Performance:

        - This is a lightweight operation that only involves pointer arithmetic.
        - The operation is optimized based on the address space, using smaller
            integer types for shared memory to improve efficiency.

        Notes:

        - The distance is calculated in elements, not bytes.
        - The result can be positive or negative depending on the relative positions
            of the pointers.
        - This function is particularly useful for GPU programming where understanding
            memory offsets is critical for performance.
        - Care should be taken when using this with pointers from different allocations,
            as the result would be meaningless.
        """
        return (
            Scalar[linear_idx_type](Int(self.ptr) - Int(addr))
            // size_of[dtype]()
        )

    @always_inline
    fn distance[
        _layout: Layout,
        _uint_dtype: DType = _get_unsigned_type(_layout, address_space),
    ](
        self, src: LayoutTensor[dtype, _layout, address_space=address_space]
    ) -> Scalar[_uint_dtype]:
        """Calculate the element-wise distance between this tensor and another
        tensor.

        This method computes the number of elements (not bytes) between this
        tensor's pointer and another tensor's pointer. This is useful for
        determining the relative positions of tensors within a larger memory
        allocation.

        Parameters:
            _layout: The layout of the source tensor.
            _uint_dtype: The unsigned integer type to use for the result.
                Automatically determined based on the layout and address space.

        Args:
            src: The source tensor to calculate the distance to.

        Returns:
            The number of elements between this tensor's pointer and the source
            tensor's pointer. The result is of type _uint_dtype.

        Example:

        If tensor1 points to element at index 100 in a buffer, and tensor2 points
        to element at index 50, then `tensor1.distance(tensor2)` would return 50.

        Performance:

        - This is a lightweight operation that only involves pointer arithmetic.
        - The operation is optimized based on the address space and layout,
            using appropriate integer types for efficiency.

        Notes:

        - The distance is calculated in elements, not bytes.
        - The result can be positive or negative depending on the relative
            positions of the tensors.
        - This function is particularly useful for GPU programming where
            understanding memory offsets is critical for performance.
        - Both tensors must be in the same address space for the result to be
            meaningful.
        - This overload is more type-safe than the pointer-based version as it
            ensures the tensors have compatible data types and address spaces.
        """

        return Scalar[_uint_dtype](
            (Int(self.ptr) - Int(src.ptr)) // size_of[dtype]()
        )

    # Returns the linear index of an elem_i 0 ... size(layout).
    #
    @always_inline
    fn _get_element_idx[elem_i: Int](self) -> Scalar[linear_idx_type]:
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
            var idx = make_runtime_layout[linear_idx_type=linear_idx_type](
                self.runtime_element_layout, self.runtime_layout
            )(rt)
            return idx

    @always_inline("nodebug")
    fn copy_from(self, other: LayoutTensor):
        """Copy data from another tensor to this tensor.

        This method performs an element-by-element copy from the source tensor
        to this tensor, respecting the layouts of both tensors. The copy
        operation handles different memory layouts correctly, ensuring that
        elements are copied to their proper positions regardless of how the data
        is arranged in memory.

        Constraints:
        - Both tensors must have statically known shapes.
        - The total number of elements must be the same in both tensors.
        - The element sizes must match between the tensors.

        Args:
            other: The source tensor to copy data from. Must have the same total
                number of elements as this tensor.

        Example:

        ```mojo
        from layout import LayoutTensor, Layout

        var src_storage = InlineArray[Float32, 2 * 3](uninitialized=True)
        var dst_storage = InlineArray[Float32, 3 * 2](uninitialized=True)
        var src = LayoutTensor[
            DType.float32,
            Layout([2, 3]),
        ](src_storage).fill(1.0)

        var dst = LayoutTensor[
            DType.float32,
            Layout([3, 2]),
        ](dst_storage)

        dst.copy_from(src)  # Copies all elements from src to dst
        ```

        Performance:

        - Performs element-by-element copying, which may be less efficient than
            vectorized or bulk memory operations.
        - The copy respects the memory layout of both tensors, which may involve
            non-contiguous memory access patterns.
        - For optimal performance with large tensors, consider using specialized
            copy functions that can leverage hardware acceleration.

        Notes:

        - Both tensors must have statically known shapes.
        - The total number of elements must be the same in both tensors.
        - The element sizes must match between the tensors.
        - This function handles different memory layouts correctly, making it suitable
            for copying between tensors with different shapes or strides.
        - The copy is performed element by element, not as a bulk memory copy.
        """
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
            "copy_from should move data of the same size, getting dst size ",
            String(dst_size),
            " and src size ",
            String(src_size),
        ]()

        constrained[
            dst_element_size == src_element_size, "copy_from should move"
        ]()

        @parameter
        for i in range(dst_size):
            src_idx = other._get_element_idx[i]()
            dst_idx = self._get_element_idx[i]()

            src_element = MemoryElement[index_type = other.linear_idx_type](
                other.ptr.offset(src_idx), other.runtime_element_layout
            )

            dst_element = MemoryElement[index_type=linear_idx_type](
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
        src_idx_bound: Scalar[src.linear_idx_type] = 0,
        base_offset: Scalar[linear_idx_type] = 0,
    ):
        """Asynchronously copy data from another tensor to this tensor using GPU
        hardware.

        This method performs an asynchronous copy from the source tensor to this
        tensor using GPU hardware acceleration. It's specifically designed for
        copying data from global memory to shared memory in GPU kernels,
        leveraging hardware-specific asynchronous copy mechanisms for improved
        performance.

        For optimal performance, you need to arrange the copy correctly. Use the
        [`distribute()`](/mojo/kernels/layout/layout_tensor/LayoutTensor/#distribute)
        method to create thread-local fragments of the source and
        destination tensors, assigning each thread one or more elements to copy.

        Optionally, use the
        [`vectorize()`]((/mojo/kernels/layout/layout_tensor/LayoutTensor/#vectorize)
        method to get vectorized views of both tensors before calling
        `distribute()`. This allows each thread to copy multiple elements of the
        tensor. For example:

        ```mojo
        var fragment = tensor.vectorize[1, simd_width]().distribute[
            thread_layout
        ](thread_id)
        ```

        The copy operation is asynchronous, so you must call
        [`async_copy_wait_all()`](/mojo/stdlib/gpu/memory/async_copy_wait_all/)
        or
        [`async_copy_wait_group()`](/mojo/stdlib/gpu/memory/async_copy_wait_group/)
        to ensure the copy has completed before using the data.

        Constraints:
            - Destination must be in shared memory.
            - Source and destination data types must match.
            - Element size must be 4, 8, or 16 bytes.
            - Destination tensor must have a static layout.

        Parameters:
            is_masked: Whether to perform a masked copy, where elements outside
                the `src_idx_bound` are not copied or filled with zeros.
            swizzle: Optional swizzling function to rearrange the destination
                indices, which can improve memory access patterns.
            fill: Fill policy for elements that are not copied (only used with
                masked copies).
            eviction_policy: Cache eviction policy for the source data.

        Args:
            src: The source tensor to copy data from.
            src_idx_bound: For masked copies, the upper bound index for valid
                source elements.
            base_offset: Base offset for swizzling calculations.

        Example:

        ```mojo
        from layout import LayoutTensor, Layout
        from gpu import thread_idx, block_idx, block_dim
        from gpu.memory import AddressSpace, async_copy_wait_all

        alias dtype = DType.float32
        alias in_size = 128
        alias block_size = 16
        num_blocks = in_size // block_size
        alias input_layout = Layout.row_major(in_size, in_size)

        fn kernel(tensor: LayoutTensor[dtype, input_layout, MutableAnyOrigin]):
            # extract a tile from the input tensor.
            var global_tile = tensor.tile[block_size, block_size](block_idx.x, block_idx.y)

            # allocate a shared memory tile
            alias tile_layout = Layout.row_major(block_size, block_size)
            var shared_tile = LayoutTensor[
                dtype,
                tile_layout,
                MutableAnyOrigin,
                address_space = AddressSpace.SHARED,
            ].stack_allocation()

            # Create per-thread tile fragments for copying
            var tid = thread_idx.y + thread_idx.x * block_dim.x
            alias thread_layout = Layout.row_major(block_size, block_size)
            var global_fragment = global_tile.distribute[thread_layout](tid)
            var shared_fragment = shared_tile.distribute[thread_layout](tid)

            # async copy to shared memory
            shared_fragment.copy_from_async(global_fragment)
            async_copy_wait_all()
            # ... do something with the shared tile
        ```

        Performance:

        - Supports vectorized copies for 4, 8, or 16-byte elements for better
            throughput.
        - Can bypass L1 cache with appropriate eviction policies for specific
            access patterns.
        - Swizzling can improve memory access patterns and reduce bank
            conflicts.

        Notes:

        - For vectorized copies, both tensors must have contiguous element
            layouts.
        - Asynchronous copies allow computation to overlap with memory
            transfers.
        - A synchronization barrier is required before using the copied data.
        """
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
        alias element_size_bytes = size_of[dtype]() * src_element_size
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
                var src_idx: Scalar[src.linear_idx_type]
                alias src_static_idx: Scalar[src.linear_idx_type] = src.layout(
                    i
                )

                @parameter
                if src_dims_known:
                    src_idx = src_static_idx
                else:
                    src_idx = src.runtime_layout(i)
                alias dst_idx = layout(i)
                var swizzled_idx: Scalar[self.linear_idx_type]

                @parameter
                if swizzle:
                    alias swizzle_fn = swizzle.value()
                    alias dst_idx_base = dst_idx % swizzle_fn.size()
                    alias dst_idx_diff = dst_idx - dst_idx_base
                    swizzled_idx = (
                        swizzle_fn(base_offset + dst_idx_base)
                        + dst_idx_diff
                        - base_offset
                    ).cast[linear_idx_type]()
                else:
                    swizzled_idx = dst_idx

                @parameter
                if is_masked:
                    var src_copy_size = (
                        Int32(element_size_bytes) if src_idx
                        < src_idx_bound else 0
                    )
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
                var src_idx: Scalar[src.linear_idx_type]
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
                    src_idx = make_runtime_layout[
                        linear_idx_type = src.linear_idx_type
                    ](src.runtime_element_layout, src.runtime_layout)(rt)

                async_copy[4, eviction_policy=eviction_policy](
                    src_ptr.bitcast[Scalar[dtype]]() + src_idx,
                    dst_ptr + dst_idx,
                )

    @always_inline
    fn fill[
        *,
        use_runtime_layout: Bool = (
            not layout.all_dims_known() or layout.size() > BATCH_SIZE
        ),
    ](
        self: LayoutTensor[mut=True, dtype, **_], val: Scalar[dtype]
    ) -> __type_of(self):
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
            alias num_elements = layout.size()

            # TODO: MSTDL-1352 we can use memory element to fill the tensor.
            @parameter
            for i in range(num_elements):
                alias idx = layout(i)

                @parameter
                for j in range(Self.element_size):
                    alias element_offset = element_layout(j)
                    self.ptr[idx + element_offset] = val
        else:
            var num_elements = self.runtime_layout.size()

            for i in range(num_elements):
                var idx = self.runtime_layout(i)

                @parameter
                if element_layout.all_dims_known():

                    @parameter
                    for j in range(Self.element_size):
                        alias element_offset = element_layout(j)
                        self.ptr[idx + element_offset] = val
                else:
                    for j in range(self.runtime_element_layout.size()):
                        var element_offset = self.runtime_element_layout(j)
                        self.ptr[idx + element_offset] = val
        return self

    @no_inline
    fn __str__(self) -> String:
        """Convert the tensor to a string representation.

        This method converts the tensor to a human-readable string
        representation by writing its contents to a string. It delegates to the
        `write_to` method which formats the tensor appropriately based on its
        rank and shape.

        Returns:
            A string representation of the tensor.
        """
        return String.write(self)

    fn write_to(self, mut writer: Some[Writer]):
        """Format and write the tensor's contents to a writer.

        This method formats the tensor's contents and writes them to the
        provided writer. For 2D tensors, it formats the output in a 2D grid. For
        tensors of other ranks, it prints all values in column-major coordinate
        order.

        Args:
            writer: The writer instance to write the formatted output to.

        Example:

        ```mojo
        from layout import Layout, LayoutTensor

        def main():
            var storage = InlineArray[Float32, 2 * 3](uninitialized=True)
            var tensor = LayoutTensor[
                DType.float32,
                Layout([2, 3]),
            ](storage).fill(1.0)
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
) -> LayoutTensor[
    dtype,
    layout,
    MutableAnyOrigin,
    address_space=target_address_space,
    masked = in_tensor.masked,
]:
    """Create a stack-allocated tensor with the same layout as an existing
    tensor.

    This function creates a new tensor on the stack with the same layout, data
    type, and masking properties as the input tensor, but potentially with a
    different address space. This is useful for creating temporary tensors that
    match the structure of existing tensors.

    Parameters:
        layout: The layout of the tensor to allocate.
        dtype: The data type of the tensor elements.
        address_space: The address space of the input tensor.
        target_address_space: The address space for the new tensor. Defaults to
            GENERIC.

    Args:
        in_tensor: The input tensor to match the layout of.

    Returns:
        A new tensor allocated on the stack with the same layout as the input
        tensor.

    Example:

    ```mojo
    from layout import LayoutTensor, Layout
    from layout.layout_tensor import stack_allocation_like
    from gpu.memory import AddressSpace

    var global_tensor = LayoutTensor[
        DType.float32,
        Layout([10, 10]),
        MutableAnyOrigin,
        address_space=AddressSpace.GLOBAL
    ].stack_allocation()

    var shared_tensor = stack_allocation_like[
        target_address_space=AddressSpace.SHARED
    ](global_tensor)
    ```

    Performance:

    - Creates a tensor on the stack, which is typically faster to allocate and
        access than heap-allocated memory.
    - Stack allocations have automatic lifetime management, reducing memory
        management overhead.
    - Stack size is limited, so be cautious with large tensor allocations.

    Notes:

    - The new tensor will have the same layout, data type, and masking properties
        as the input tensor.
    - The address space can be changed, which is useful for moving data between
        different memory regions (e.g., from global to shared memory).
    - Stack allocations are automatically freed when they go out of scope.
    - The function uses the stack_allocation method of the result tensor type.
    """
    return LayoutTensor[
        dtype,
        layout,
        MutableAnyOrigin,
        address_space=target_address_space,
        masked = in_tensor.masked,
    ].stack_allocation()


@register_passable("trivial")
struct ThreadScope(ImplicitlyCopyable, Movable):
    """Represents the scope of thread operations in GPU programming.

    This struct defines the scope at which thread operations are performed,
    particularly for operations like tensor distribution and synchronization.
    It provides two main scopes: `BLOCK` and `WARP`, which correspond to
    different levels of thread grouping in GPU programming models.

    Example:

    ```mojo
    from layout.layout_tensor import copy_dram_to_sram, ThreadScope

    # Distribute tensor at block level (all threads in block participate)
    copy_dram_to_sram[layout, thread_scope=ThreadScope.BLOCK](dst, src)

    # Distribute tensor at warp level (only threads in same warp participate)
    copy_dram_to_sram[layout, thread_scope=ThreadScope.WARP](dst, src)
    ```

    Performance:

    - WARP scope operations typically have lower synchronization overhead
        than BLOCK scope operations.
    - BLOCK scope operations allow coordination across all threads in a block,
        which is necessary for certain algorithms.
    - The choice of scope can significantly impact performance and correctness
        of parallel algorithms.

    Notes:

    - The appropriate scope depends on the specific algorithm and hardware.
    - WARP scope operations may be more efficient for operations that only
        require coordination within a warp.
    - BLOCK scope operations are necessary when threads from different warps
        need to coordinate.
    - The actual size of a warp or block is hardware-dependent.
    """

    var _value: Int32
    """The internal integer value representing the thread scope."""

    alias BLOCK = Self(0)
    """Represents operations at the thread block level, where all threads in a
    block participate."""

    alias WARP = Self(1)
    """Represents operations at the warp level, where only threads within the
    same warp participate."""

    fn __init__(out self, value: Int):
        """Initialize a `ThreadScope` with the given integer value.

        Args:
            value: An integer representing the thread scope (0 for `BLOCK`,
                1 for `WARP`).
        """
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        """Compare two `ThreadScope` objects for equality.

        Args:
            other: Another `ThreadScope` object to compare with.

        Returns:
            True if the thread scopes are equal, False otherwise.
        """
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        """Compare two `ThreadScope` objects for inequality.

        Args:
            other: Another `ThreadScope` object to compare with.

        Returns:
            True if the thread scopes are not equal, False otherwise.
        """
        return not (self == other)

    fn __str__(self) -> String:
        """Convert the `ThreadScope` to a human-readable string representation.

        Returns:
            A string representation of the thread scope ("BLOCK" or "WARP").

        Aborts:
            If the thread scope has an invalid value.
        """
        if self == Self.BLOCK:
            return "BLOCK"
        if self == Self.WARP:
            return "WARP"
        return abort[String]("invalid ThreadScope entry")

    fn __int__(self) -> Int:
        """Convert the `ThreadScope` to an integer value.

        Returns:
            The integer value of the thread scope (0 for BLOCK, 1 for WARP).
        """
        return Int(self._value)


@always_inline("nodebug")
fn _get_worker_idx[
    thread_scope: ThreadScope, block_dim_count: Int = 1
]() -> UInt:
    """
    Returns the worker index for the current thread scope.

    This function determines the index of the current worker (thread) based on the
    specified thread scope. If the scope is `BLOCK`, it returns the thread's index
    within the block (`thread_idx.x`). If the scope is `WARP`, it returns the lane
    ID within the warp (`lane_id()`).

    Parameters:
        thread_scope: The scope at which the worker index is determined.
        block_dim_count: The number of dimensions in the thread block.

    Returns:
        UInt: The worker index within the specified scope.

    """

    constrained[
        block_dim_count >= 1 and block_dim_count <= 3,
        "block_dim_count = ",
        String(block_dim_count),
        ". Thread blocks contain between 1 (x) and 3 (x,y,z) dimensions",
    ]()

    @parameter
    if thread_scope == ThreadScope.BLOCK:

        @parameter
        if block_dim_count == 1:
            return thread_idx.x
        elif block_dim_count == 2:
            return thread_idx.y * block_dim.x + thread_idx.x
        else:
            return (
                thread_idx.z * block_dim.y * block_dim.x
                + thread_idx.y * block_dim.x
                + thread_idx.x
            )
    else:
        return lane_id()


@always_inline("nodebug")
fn _copy_dram_to_sram_validate_args(dst: LayoutTensor, src: LayoutTensor):
    """Validate arguments for DRAM to SRAM copy operations.

    This internal function validates that the source and destination tensors
    have compatible properties for a DRAM to SRAM copy operation. It checks
    data types and address spaces to ensure the copy operation can be performed
    correctly.

    Constraints:
        - Source and destination tensors must have the same data type.
        - Source tensor must be in GENERIC or GLOBAL address space.
        - Destination tensor must be in SHARED address space.

    Args:
        dst: The destination tensor, which must be in shared memory (SRAM).
        src: The source tensor, which must be in global or generic memory
            (DRAM).

    Notes:

    - This is an internal helper function used by copy_dram_to_sram.
    - The function enforces that the source and destination tensors have
        the same data type.
    - The source tensor must be in GENERIC or GLOBAL address space (DRAM).
    - The destination tensor must be in SHARED address space (SRAM).
    - These constraints ensure that the copy operation follows the expected
        memory hierarchy flow from slower global memory to faster shared memory.
    """
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


@always_inline("nodebug")
fn copy_dram_to_sram[
    src_thread_layout: Layout,
    dst_thread_layout: Layout = src_thread_layout,
    swizzle: OptionalReg[Swizzle] = None,
    num_threads: Int = src_thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
    block_dim_count: Int = 1,
](dst: LayoutTensor, src: LayoutTensor):
    """Synchronously copy data from DRAM (global memory) to SRAM (shared memory)
    in a GPU context.

    This function performs a synchronous copy operation from global memory
    (DRAM) to shared memory (SRAM) in a GPU context, distributing the workload
    across multiple threads for parallel execution. It uses thread affinity
    mapping to ensure efficient work distribution and supports vectorized memory
    operations for optimal performance.

    Constraints:
        - Source and destination tensors must have the same data type.
        - Source tensor must be in GENERIC or GLOBAL address space.
        - Destination tensor must be in SHARED address space.
        - For non-masked tensors, the fragment sizes must match.

    Parameters:
        src_thread_layout: Layout defining how threads are organized for the
            source tensor. This determines how the workload is distributed among
            threads.
        dst_thread_layout: Layout defining how threads are organized for the
            destination tensor. Defaults to the same as `src_thread_layout` if
            not specified.
        swizzle: Optional swizzling function to rearrange the destination
            indices, which can improve memory access patterns and reduce bank
            conflicts.
        num_threads: Total number of threads participating in the copy
            operation. Defaults to the size of `src_thread_layout`.
        thread_scope: Scope at which thread operations are performed (`BLOCK` or
            `WARP`). Defaults to `ThreadScope.BLOCK`, where all threads in a
            block participate.
        block_dim_count: The number of dimensions in the thread block.

    Args:
        dst: The destination tensor, which must be in shared memory (SRAM).
        src: The source tensor, which must be in global or generic memory
            (DRAM).

    Performance:

    - Distributes the copy workload across multiple threads for parallel
        execution.
    - Supports vectorized loads and stores for better memory throughput.
    - Can use swizzling to optimize memory access patterns and reduce bank
        conflicts.
    - Thread affinity mapping ensures efficient work distribution.
    - For masked tensors, performs bounds checking to handle edge cases
        correctly.

    Notes:

    - The source tensor must be in GENERIC or GLOBAL address space (DRAM).
    - The destination tensor must be in SHARED address space (SRAM).
    - Both tensors must have the same data type.
    - This function is synchronous, meaning all threads must complete their
        copy operations before proceeding.
    - For optimal performance, the thread layouts should match the memory
        access patterns of the tensors.
    - This function is particularly useful in GPU kernels for loading data
        from global memory to shared memory for faster access.
    """
    _copy_dram_to_sram_validate_args(dst, src)

    alias num_busy_threads = src_thread_layout.size()
    var worker_idx = _get_worker_idx[thread_scope, block_dim_count]()

    @parameter
    if num_threads > num_busy_threads:
        if worker_idx >= UInt(num_busy_threads):
            return

    var src_fragments = src.distribute[src_thread_layout](worker_idx)
    var dst_fragments = dst.distribute[dst_thread_layout, swizzle=swizzle](
        worker_idx
    )

    alias simd_width = simd_width_of[dst.dtype]()
    alias src_align = align_of[SIMD[src.dtype, simd_width]]()
    alias dst_align = align_of[SIMD[dst.dtype, simd_width]]()

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
            "Fragment size mismatch: dst fragments size (",
            String(dst_fragments.layout.size()),
            ") does not match src fragments size (",
            String(src_fragments.layout.size()),
            ")",
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

        # NOTE: This can be a negative number, so we cannot use unsigned type
        # in layout tensor.
        var src_idx_bound = (src.dim[0]() * stride - src_frag_offset).cast[
            src_fragments.linear_idx_type
        ]()

        @parameter
        for i in range(num_stores_per_thread):
            alias src_static_idx = src_fragments.layout(i)

            alias dst_idx = dst_fragments.layout(i)

            var src_idx: Scalar[src_fragments.linear_idx_type]

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
    block_dim_count: Int = 1,
](dst: LayoutTensor, src_iter: LayoutTensorIter, bound: Int):
    """Efficiently copy data from global memory (DRAM) to shared memory (SRAM)
    on AMD GPUs.

    This function implements an optimized memory transfer operation specifically
    for AMD GPU architectures. It utilizes the hardware's `buffer_load`
    intrinsic to efficiently transfer data while handling bounds checking. The
    function distributes the copy operation across multiple threads for maximum
    throughput.

    Parameters:
        src_thread_layout: The layout used to distribute the source tensor
            across threads. This determines how the workload is divided among
            participating threads.
        dst_thread_layout: The layout used to distribute the destination tensor
            across threads. Defaults to the same layout as `src_thread_layout`.
        swizzle: Optional swizzling pattern to apply when distributing the
            destination tensor. This can improve memory access patterns and
            reduce bank conflicts. Defaults to None (no swizzling).
        num_threads: The total number of threads participating in the copy
            operation. Defaults to the size of `src_thread_layout`.
        thread_scope: Defines whether operations are performed at `BLOCK` or
            `WARP` level. `BLOCK` scope involves all threads in a thread block,
            while `WARP` scope restricts operations to threads within the same
            warp. Defaults to `ThreadScope.BLOCK`.
        block_dim_count: The number of dimensions in the thread block.

    Args:
        dst: The destination tensor in shared memory (SRAM).
        src_iter: The source tensor iterator in global memory (DRAM) to be
            copied.
        bound: The bound of the source tensor iterator.
    """
    constrained[is_amd_gpu(), "This function is only supported on AMD GPUs."]()

    var src_tensor = src_iter[].vectorize[
        dst.element_layout.shape[0].value(), dst.element_layout.shape[1].value()
    ]()
    _copy_dram_to_sram_validate_args(dst, src_tensor)

    alias num_busy_threads = src_thread_layout.size()
    var worker_idx = _get_worker_idx[thread_scope, block_dim_count]()

    @parameter
    if num_threads > num_busy_threads:
        if worker_idx >= UInt(num_busy_threads):
            return

    var src_fragments = src_tensor.distribute[src_thread_layout](worker_idx)
    var dst_fragments = dst.distribute[dst_thread_layout, swizzle=swizzle](
        worker_idx
    )

    alias simd_width = src_tensor.element_layout.size()
    alias dst_align = align_of[SIMD[dst.dtype, simd_width]]()

    alias num_stores_per_thread = dst_fragments.layout.size()
    var buffer = make_amd_buffer_resource(src_iter, bound)
    var src_frag_offset = src_fragments.distance(src_tensor.ptr) + Int(
        src_iter.offset
    )

    @parameter
    for i in range(num_stores_per_thread):
        var src_frag_idx: Scalar[src_fragments.linear_idx_type]

        @parameter
        if src_tensor.layout.all_dims_known():
            alias frag_layout = src_fragments.layout(i)
            src_frag_idx = frag_layout
        else:
            src_frag_idx = src_fragments.runtime_layout(i)

        alias dst_frag_idx = dst_fragments.layout(i)
        dst_fragments.ptr.store[alignment=dst_align](
            dst_frag_idx,
            buffer.load[src_tensor.dtype, simd_width](
                Int32(src_frag_offset),
                scalar_offset=Int32(src_frag_idx),
            ).cast[dst.dtype](),
        )


@always_inline("nodebug")
fn cp_async_k_major[
    dtype: DType,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
](
    dst: LayoutTensor[
        dtype, _, address_space = gpu_memory.AddressSpace.SHARED, *_, **_
    ],
    src: LayoutTensor[
        dtype, _, address_space = gpu_memory.AddressSpace.GENERIC, *_, **_
    ],
):
    """Asynchronously copy data from DRAM to SRAM using TMA (Tensor Memory
    Accelerator) with K-major layout.

    This function performs an asynchronous copy operation from global memory
    (DRAM) to shared memory (SRAM) using NVIDIA's Tensor Memory Accelerator
    (TMA) hardware. It optimizes for K-major memory access patterns, which is
    particularly beneficial for certain tensor operations like matrix
    multiplications where the inner dimension (K) is accessed contiguously.

    The function automatically determines the optimal tile size and thread
    distribution based on the tensor shapes and hardware capabilities,
    leveraging TMA's efficient memory transfer mechanisms.

    Constraints:
        - Requires NVIDIA GPUs with TMA support (compute capability 9.0+).
        - Source tensor must be in GENERIC or GLOBAL address space.
        - Destination tensor must be in SHARED address space.
        - Both tensors must have the same data type.
        - Source and destination tensors must be 2D.

    Parameters:
        dtype: The data type of the tensor elements.
        eviction_policy: The cache eviction policy to use. Default is `CacheEviction.EVICT_NORMAL`.

    Args:
        dst: The destination tensor, which must be in shared memory (SRAM).
        src: The source tensor, which must be in global or generic memory
            (DRAM).

    Performance:

    - Uses TMA hardware acceleration for optimal memory transfer performance.
    - Optimizes for K-major access patterns, which can significantly improve
        performance for certain tensor operations like matrix multiplications.
    - Performs asynchronous transfers, allowing computation to overlap with
        memory operations.
    - Automatically determines optimal tile sizes based on tensor dimensions.
    - Uses hardware-accelerated swizzling to reduce shared memory bank
        conflicts.

    Notes:

    - This function requires NVIDIA GPUs with TMA support (compute capability
        9.0+).
    - The source tensor must be in GENERIC or GLOBAL address space (DRAM).
    - The destination tensor must be in SHARED address space (SRAM).
    - Both tensors must have the same data type.
    - This function is asynchronous, so you must call
        [`async_copy_wait_all()`](/mojo/stdlib/gpu/memory/async_copy_wait_all/)
        or
        [`async_copy_wait_group()`](/mojo/stdlib/gpu/memory/async_copy_wait_group/)
        to ensure the copy has completed before using the data.
    - K-major layout is particularly beneficial for matrix multiplication
        operations where the inner dimension (K) is accessed contiguously.
    """
    alias dst_layout = dst.layout

    alias src_layout = src.layout
    alias src_shape0 = src_layout.shape[0].value()
    alias src_shape1 = src_layout.shape[1].value()

    alias desc_layout = _tma_desc_tile_layout[
        dtype,
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
    alias simd_size = simd_width_of[dtype]()
    # single warp group
    alias thread_layout = Layout.row_major(
        128 * simd_size // desc_shape1, desc_shape1 // simd_size
    )

    @parameter
    for tile_id in range(num_tiles):
        src_tile = src.tile[desc_shape0, desc_shape1](0, tile_id)
        dst_tile = LayoutTensor[
            dtype, desc_layout, address_space = gpu_memory.AddressSpace.SHARED
        ](dst.ptr + tile_id * desc_size)

        copy_dram_to_sram_async[
            thread_layout, swizzle=True, eviction_policy=eviction_policy
        ](
            dst_tile.vectorize[1, simd_size](),
            src_tile.vectorize[1, simd_size](),
        )


@always_inline("nodebug")
fn cp_async_mn_major[
    dtype: DType,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
](
    dst: LayoutTensor[
        dtype, _, address_space = gpu_memory.AddressSpace.SHARED, *_, **_
    ],
    src: LayoutTensor[
        dtype, _, address_space = gpu_memory.AddressSpace.GENERIC, *_, **_
    ],
):
    """Asynchronously copy data from DRAM to SRAM using TMA (Tensor Memory
    Accelerator) with MN-major layout.

    This function performs an asynchronous copy operation from global memory
    (DRAM) to shared memory (SRAM) using NVIDIA's Tensor Memory Accelerator
    (TMA) hardware. It optimizes for MN-major memory access patterns, which is
    particularly beneficial for tensor operations where the outer dimensions (M,
    N) are accessed contiguously.

    The function automatically determines the optimal tile size and thread
    distribution based on the tensor shapes and hardware capabilities,
    leveraging TMA's efficient memory transfer mechanisms.

    Constraints:
        - Requires NVIDIA GPUs with TMA support (compute capability 9.0+).
        - Source tensor must be in `GENERIC` or `GLOBAL` address space.
        - Destination tensor must be in `SHARED` address space.
        - Both tensors must have the same data type.
        - Source and destination tensors must be 2D.

    Parameters:
        dtype: The data type of the tensor elements.
        eviction_policy: The cache eviction policy to use. Default is `CacheEviction.EVICT_NORMAL`.

    Args:
        dst: The destination tensor, which must be in shared memory (SRAM).
        src: The source tensor, which must be in global or generic memory
            (DRAM).

    Performance:

    - Uses TMA hardware acceleration for optimal memory transfer performance.
    - Optimizes for MN-major access patterns, which can significantly improve
        performance for certain tensor operations where outer dimensions are accessed
        contiguously.
    - Performs asynchronous transfers, allowing computation to overlap with memory operations.
    - Automatically determines optimal tile sizes based on tensor dimensions.
    - Uses hardware-accelerated swizzling to reduce shared memory bank conflicts.

    Notes:

    - This function requires NVIDIA GPUs with TMA support (compute capability 9.0+).
    - The source tensor must be in `GENERIC` or `GLOBAL` address space (DRAM).
    - The destination tensor must be in `SHARED` address space (SRAM).
    - Both tensors must have the same data type.
    - This function is asynchronous, so you must call
        [`async_copy_wait_all()`](/mojo/stdlib/gpu/memory/async_copy_wait_all/)
        or
        [`async_copy_wait_group()`](/mojo/stdlib/gpu/memory/async_copy_wait_group/)
        to ensure the copy has completed before using the data.
    - MN-major layout is particularly beneficial for operations where the outer
        dimensions are accessed contiguously, such as certain convolution operations.
    """
    alias dst_layout = dst.layout

    alias src_layout = src.layout
    alias src_shape0 = src_layout.shape[0].value()
    alias src_shape1 = src_layout.shape[1].value()

    alias desc_layout = _tma_desc_tile_layout[
        dtype,
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

    alias simd_size = simd_width_of[dtype]()
    alias thread_layout_per_warp = Layout.row_major(
        gpu_memory.WARP_SIZE * simd_size // desc_shape1,
        desc_shape1 // simd_size,
    )

    warp_id = thread_idx.x // UInt(gpu_memory.WARP_SIZE)

    @parameter
    for tile_id_per_warp in range(num_tiles_per_warp):
        tile_id = warp_id + UInt(tile_id_per_warp) * num_warps
        tile_coord0, tile_coord1 = divmod(tile_id, UInt(num_tiles1))
        src_tile = src.tile[desc_shape0, desc_shape1](
            Int(tile_coord0), Int(tile_coord1)
        )
        dst_tile = LayoutTensor[
            dtype, desc_layout, address_space = gpu_memory.AddressSpace.SHARED
        ](dst.ptr + tile_id * UInt(desc_size))

        copy_dram_to_sram_async[
            thread_layout_per_warp,
            swizzle=True,
            eviction_policy=eviction_policy,
        ](
            dst_tile.vectorize[1, simd_size](),
            src_tile.vectorize[1, simd_size](),
        )


@always_inline("nodebug")
fn copy_dram_to_sram[
    thread_layout: Layout,
    swizzle: OptionalReg[Swizzle] = None,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
    block_dim_count: Int = 1,
](dst: LayoutTensor, src_iter: LayoutTensorIter, bound: Int):
    """Synchronously copy data from DRAM to SRAM using a unified thread layout
    for AMD GPUs.

    This is a convenience wrapper around the more general `copy_dram_to_sram()`
    function that uses the same layout for both source and destination tensors.
    It's specifically designed for AMD GPUs where the buffer_load intrinsic
    requires the original base tensor.

    Parameters:
        thread_layout: Layout defining how threads are organized for both source
            and destination. This determines how the workload is distributed
            among threads.
        swizzle: Optional swizzling function to rearrange the destination
            indices, which can improve memory access patterns and reduce bank
            conflicts.
        num_threads: Total number of threads participating in the copy
            operation. Defaults to the size of thread_layout.
        thread_scope: Scope at which thread operations are performed (`BLOCK` or
            `WARP`). Defaults to `BLOCK`, where all threads in a block
            participate.
        block_dim_count: The number of dimensions in the thread block.

    Args:
        dst: The destination tensor, which must be in shared memory (SRAM).
        src_iter: The source tensor iterator, which must be in global or generic
            memory (DRAM).
        bound: The bound of the source tensor iterator.

    Performance:

    - Simplifies API usage when the same thread layout is appropriate for both
        source and destination tensors.
    - Optimized for AMD GPUs using buffer_load intrinsics for efficient memory
        transfers.
    - Distributes the copy workload across multiple threads for parallel
        execution.

    Notes:

    - This function is only supported on AMD GPUs.
    - The source tensor must be in GENERIC or GLOBAL address space (DRAM).
    - The destination tensor must be in SHARED address space (SRAM).
    - Both tensors must have the same data type.
    """
    copy_dram_to_sram[
        src_thread_layout=thread_layout,
        dst_thread_layout=thread_layout,
        swizzle=swizzle,
        num_threads=num_threads,
        block_dim_count=block_dim_count,
        thread_scope=thread_scope,
    ](dst, src_iter, bound)


@always_inline("nodebug")
fn copy_dram_to_sram[
    thread_layout: Layout,
    swizzle: OptionalReg[Swizzle] = None,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
    block_dim_count: Int = 1,
](dst: LayoutTensor, src: LayoutTensor):
    """Synchronously copy data from DRAM to SRAM using a unified thread layout.

    This is a convenience wrapper around the more general `copy_dram_to_sram()`
    function that uses the same layout for both source and destination tensors.
    It simplifies the API for the common case where the same thread distribution
    pattern works well for both tensors.

    Parameters:
        thread_layout: Layout defining how threads are organized for both source
            and destination. This determines how the workload is distributed
            among threads.
        swizzle: Optional swizzling function to rearrange the destination
            indices, which can improve memory access patterns and reduce bank
            conflicts.
        num_threads: Total number of threads participating in the copy
            operation. Defaults to the size of `thread_layout`.
        thread_scope: Scope at which thread operations are performed
                (`BLOCK` or `WARP`). Defaults to `ThreadScope.BLOCK`, where all
                threads in a block participate.
        block_dim_count: The number of dimensions in the thread block.

    Args:
        dst: The destination tensor, which must be in shared memory (SRAM).
        src: The source tensor, which must be in global or generic memory
            (DRAM).

    Performance:

    - Simplifies API usage when the same thread layout is appropriate for both
        source and destination tensors.
    - Distributes the copy workload across multiple threads for parallel
        execution.
    - Supports vectorized loads and stores for better memory throughput.
    - Can use swizzling to optimize memory access patterns and reduce bank
        conflicts.

    Notes:

    - The source tensor must be in `GENERIC` or `GLOBAL` address space (DRAM).
    - The destination tensor must be in `SHARED` address space (SRAM).
    - Both tensors must have the same data type.
    - This function is synchronous, meaning all threads must complete their
        copy operations before proceeding.
    """
    copy_dram_to_sram[
        src_thread_layout=thread_layout,
        dst_thread_layout=thread_layout,
        swizzle=swizzle,
        num_threads=num_threads,
        block_dim_count=block_dim_count,
        thread_scope=thread_scope,
    ](dst, src)


@always_inline("nodebug")
fn copy_dram_to_sram_async[
    src_thread_layout: Layout,
    dst_thread_layout: Layout,
    swizzle: Bool = False,
    fill: Fill = Fill.NONE,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    num_threads: Int = src_thread_layout.size(),
    block_dim_count: Int = 1,
](dst: LayoutTensor, src: LayoutTensor):
    """Asynchronously copy data from DRAM (global memory) to SRAM (shared
    memory) in a GPU context.

    This function performs an asynchronous copy operation from global memory
    (DRAM) to shared memory (SRAM) in a GPU context, using NVIDIA's cp.async
    hardware mechanism. It distributes the workload across multiple threads and
    allows computation to overlap with memory transfers for improved
    performance.

    Constraints:
        - Requires NVIDIA GPUs with cp.async support (compute capability 8.0+).
        - Source tensor must be in `GENERIC` or `GLOBAL` address space.
        - Destination tensor must be in `SHARED` address space.
        - Both tensors must have the same data type.
        - Element size must be 4, 8, or 16 bytes.

    Parameters:
        src_thread_layout: Layout defining how threads are organized for the
            source tensor. This determines how the workload is distributed among
            threads.
        dst_thread_layout: Layout defining how threads are organized for the
            destination tensor.
        swizzle: Whether to apply swizzling to the destination indices to
            reduce bank conflicts. Defaults to False.
        fill: Fill policy for handling out-of-bounds accesses. Options
            include:
            - `Fill.NONE`: No special handling (default).
            - `Fill.ZERO`: Fill out-of-bounds elements with zeros.
        eviction_policy: Cache eviction policy for the source data. Options
            include:
            - `CacheEviction.EVICT_NORMAL`: Normal eviction (default).
            - `CacheEviction.EVICT_FIRST`: Evict data after first use.
            - `CacheEviction.EVICT_LAST`: Keep data in cache until last use.
        num_threads: Total number of threads participating in the copy operation.
                    Defaults to the size of src_thread_layout.
        block_dim_count: The number of dimensions in the thread block.

    Args:
        dst: The destination tensor, which must be in shared memory (SRAM).
        src: The source tensor, which must be in global or generic memory (DRAM).

    Performance:

    - Performs asynchronous transfers, allowing computation to overlap with
        memory operations.
    - Distributes the copy workload across multiple threads for parallel
        execution.
    - Can use swizzling to optimize memory access patterns and reduce bank
        conflicts.
    - Supports different cache eviction policies to optimize memory hierarchy
        usage.
    - For masked tensors, performs bounds checking to handle edge cases
        correctly.

    Notes:

    - This function requires NVIDIA GPUs with `cp.async` support (compute
        capability 8.0+).
    - The source tensor must be in GENERIC or GLOBAL address space (DRAM).
    - The destination tensor must be in SHARED address space (SRAM).
    - Both tensors must have the same data type.
    - This function is asynchronous, so you must call
        [`async_copy_wait_all()`](/mojo/stdlib/gpu/memory/async_copy_wait_all/)
        or
        [`async_copy_wait_group()`](/mojo/stdlib/gpu/memory/async_copy_wait_group/)
        to ensure the copy has completed before using the data.
    - The maximum size of each element that can be copied is 16 bytes.
    """
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
        "src thread layout size ",
        String(src_thread_layout.size()),
        " does not match dst thread layout size ",
        String(dst_thread_layout.size()),
    ]()

    alias num_busy_threads = src_thread_layout.size()
    var worker_idx = _get_worker_idx[ThreadScope.BLOCK, block_dim_count]()

    # We know at compile time that only partial threads copy based on the size
    # of input tensors. Return if current thread doesn't have work.
    @parameter
    if num_threads > num_busy_threads:
        if worker_idx >= UInt(num_busy_threads):
            return

    alias row_size = dst.stride[0]()
    # See make_ldmatrix_swizzle in Swizzle.mojo for `conflict_ways`.
    # TODO: use the above when MOCO-1048 is fixed.
    alias bytes_32_banks = 128
    alias conflict_ways = min(
        8 * row_size * size_of[dst.dtype]() // bytes_32_banks, 8
    )
    constrained[
        (swizzle and (conflict_ways in (4, 8))) or not swizzle,
        "Only support swizzle for 4 or 8 ways conflict.",
    ]()

    constrained[
        (swizzle and row_size in (16, 32, 64, 128, 256, 512)) or not swizzle,
        (
            "Only support 2^4-2^9 elements per row in shared memory tile for"
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

    var src_fragments = src.distribute[src_thread_layout](worker_idx)
    var dst_fragments = dst.distribute[dst_thread_layout](worker_idx)

    var dst_frag_offset = dst_fragments.distance(dst.ptr) if swizzle else 0

    @parameter
    if not src_fragments.masked:
        dst_fragments.copy_from_async[
            swizzle=swizzle_option, eviction_policy=eviction_policy
        ](src_fragments, base_offset=Int(dst_frag_offset))
    else:
        var src_frag_offset = src_fragments.distance(src.ptr)

        # Stride between two rows
        alias static_row_stride = Scalar[src_fragments.linear_idx_type](
            src.layout.stride[0].value()
        )
        var row_stride = static_row_stride

        @parameter
        if src.layout.stride[0].value() == UNKNOWN_VALUE:
            row_stride = Scalar[src_fragments.linear_idx_type](
                src.runtime_layout.stride.value[0]
            )

        var src_idx_bound = (
            Scalar[src_fragments.linear_idx_type](src.dim[0]()) * row_stride
            - src_frag_offset
        )

        dst_fragments.copy_from_async[
            is_masked=True,
            swizzle=swizzle_option,
            eviction_policy=eviction_policy,
        ](
            src_fragments,
            src_idx_bound=src_idx_bound,
            base_offset=dst_frag_offset,
        )


@always_inline("nodebug")
fn copy_dram_to_sram_async[
    thread_layout: Layout,
    swizzle: Bool = False,
    masked: Bool = False,
    fill: Fill = Fill.NONE,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    num_threads: Int = thread_layout.size(),
    block_dim_count: Int = 1,
](dst: LayoutTensor, src: LayoutTensor):
    """
    Asynchronous copy from DRAM to SRAM with thread affinity mapping.

    This function performs an asynchronous memory transfer from DRAM (global
    memory) to SRAM (shared memory) using the specified thread layout for
    distribution.

    Parameters:
        thread_layout: The layout used to distribute work across threads.
        swizzle: Whether to apply memory access swizzling for better performance.
        masked: Whether the copy operation should use masking.
        fill: Fill policy for uninitialized memory regions.
        eviction_policy: Cache eviction policy to use during the transfer.
        num_threads: Number of threads to use for the operation, defaults to
            the size of `thread_layout`.
        block_dim_count: The number of dimensions in the thread block.

    Args:
        dst: Destination tensor in SRAM.
        src: Source tensor in DRAM.

    Notes:

    This is a convenience wrapper around the more general
    `copy_dram_to_sram_async()` function, using the same thread layout for
    both source and destination.
    """
    copy_dram_to_sram_async[
        src_thread_layout=thread_layout,
        dst_thread_layout=thread_layout,
        swizzle=swizzle,
        eviction_policy=eviction_policy,
        num_threads=num_threads,
        block_dim_count=block_dim_count,
    ](dst, src)


alias binary_op_type = fn[dtype: DType, width: Int] (
    lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]
) -> SIMD[dtype, width]
"""
Type alias for binary operations on SIMD vectors.

This type represents a function that takes two SIMD vectors of the same type and
width and returns a SIMD vector of the same type and width.

Args:
    dtype: The data type of the SIMD vector elements.
    width: The width of the SIMD vector.
    lhs: Left-hand side SIMD vector operand.
    rhs: Right-hand side SIMD vector operand.

Returns:
    A SIMD vector containing the result of the binary operation.
"""


@always_inline("nodebug")
fn copy_sram_to_dram[
    thread_layout: Layout,
    swizzle: OptionalReg[Swizzle] = None,
    num_threads: Int = thread_layout.size(),
    block_dim_count: Int = 1,
    binary_op: OptionalReg[binary_op_type] = None,
](dst: LayoutTensor, src: LayoutTensor):
    """Synchronously copy data from SRAM (shared memory) to DRAM (global
    memory).

    This function performs a synchronous memory transfer from SRAM (shared
    memory) to DRAM (global memory) using the specified thread layout for
    workload distribution. It supports optional swizzling for optimized memory
    access patterns and binary operations for combining data during the
    transfer.

    Constraints:
        - Source tensor must be in SHARED address space with a static layout.
        - Destination tensor must be in GENERIC or GLOBAL address space.
        - For type conversion, only FP32 to half-precision is supported.
        - For vectorized copy with type conversion, both tensors must have
          element layouts matching the SIMD width of the destination type.

    Parameters:
        thread_layout: Layout defining how threads are organized for both source
            and destination. This determines how the workload is distributed
            among threads.
        swizzle: Optional swizzling function to rearrange the source indices,
            which can improve memory access patterns and reduce bank conflicts.
        num_threads: Total number of threads participating in the copy
            operation. Defaults to the size of thread_layout.
        block_dim_count: The number of dimensions in the thread block.
        binary_op: Optional binary operation to apply during the copy, combining
            source data with existing destination data.

    Args:
        dst: The destination tensor, which must be in global or generic memory
            (DRAM).
        src: The source tensor, which must be in shared memory (SRAM).

    Performance:

    - Distributes the copy workload across multiple threads for parallel
        execution.
    - Supports vectorized loads and stores for better memory throughput.
    - Can use swizzling to optimize memory access patterns.
    - Supports binary operations to combine data during transfer (e.g., for
        reduction operations).

    Notes:

    - The source tensor must be in `SHARED` address space (SRAM).
    - The destination tensor must be in `GENERIC` or `GLOBAL` address space
        (DRAM).
    - Supports FP32 to half-precision downcast during copy if needed.
    - Handles masked tensors with proper bounds checking.
    - This function is synchronous, meaning all threads must complete their
        copy operations before proceeding.
    """
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
    var worker_idx = _get_worker_idx[ThreadScope.BLOCK, block_dim_count]()

    @parameter
    if num_threads > num_busy_threads:
        if worker_idx >= UInt(num_busy_threads):
            return

    var src_fragments = src.distribute[thread_layout](worker_idx)
    var dst_fragments = dst.distribute[thread_layout](worker_idx)

    # TODO: copy_from only allows static layout
    @parameter
    if src.dtype == dst.dtype and not swizzle and not dst.masked:
        dst_fragments.copy_from(src_fragments)
    else:
        constrained[
            src.dtype == dst.dtype
            or (src.dtype is DType.float32 and dst.dtype.is_half_float()),
            "Only support FP32 -> half precision downcast during copy.",
        ]()

        alias simd_size = simd_width_of[dst.dtype]()
        # TODO: generalize the copy to non-scalar case if possible.
        constrained[
            src.element_layout.size() == simd_size
            and dst.element_layout.size() == simd_size,
            "Only FP32 -> half precision downcast for vectorized copy.",
        ]()

        alias src_align = align_of[
            SIMD[src.dtype, simd_width_of[src.dtype]()]
        ]()
        alias dst_align = align_of[SIMD[dst.dtype, simd_size]]()

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
                    # in the unrolled loop. Hopefully compiler can eliminate the duplicated
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
            var dst_idx_bound = (dst.dim[0]() * stride - dst_frag_offset).cast[
                dst_fragments.linear_idx_type
            ]()

            @parameter
            for i in range(num_stores_per_thread):
                alias src_idx = src_fragments.layout(i)

                alias dst_uint_dtype = _get_unsigned_type(
                    dst_fragments.layout, dst_fragments.address_space
                )
                alias dst_static_idx = dst_fragments.layout(i)

                var dst_idx: Scalar[dst_fragments.linear_idx_type]

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
                    # in the unrolled loop. Hopefully compiler can eliminate the duplicated
                    # xor computation.
                    swizzled_idx = (
                        swizzle_fn(src_frag_offset + src_idx_base)
                        + src_idx_diff
                    )

                if dst_idx < dst_idx_bound:
                    var src_vec = (
                        (src.ptr)
                        .load[width=simd_size, alignment=src_align](
                            swizzled_idx
                        )
                        .cast[dst.dtype]()
                    )

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


@always_inline("nodebug")
fn copy_sram_to_local[
    src_warp_layout: Layout,
    axis: OptionalReg[Int] = None,
](dst: LayoutTensor, src: LayoutTensor):
    """Synchronously copy data from SRAM (shared memory) to local memory.

    This function performs a synchronous memory transfer from SRAM (shared
    memory) to local memory (registers) using the specified thread layout for
    workload distribution.

    Constraints:
        - The source tensor must be in SHARED address space (SRAM).
        - The destination tensor must be in LOCAL address space (registers).
        - Both tensors must have the same data type.

    Parameters:
        src_warp_layout: Layout defining how threads are organized for the
            source tensor. This determines how the workload is distributed among
            threads.
        axis: Optional parameter specifying which axis to distribute along.
            When provided, distribution happens along the specified axis.
            When None (default), distribution uses the standard layout pattern.

    Args:
        dst: The destination tensor, which must be in local memory (registers).
        src: The source tensor, which must be in shared memory (SRAM).

    Performance:

    - Distributes the copy workload across multiple threads for parallel
        execution.
    - Optimized for transferring data from shared memory to registers.
    - Supports optional axis-specific distribution for specialized access
        patterns.
    """
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


@always_inline("nodebug")
fn copy_local_to_dram[
    dst_thread_layout: Layout,
    num_threads: Int = dst_thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
    block_dim_count: Int = 1,
](dst: LayoutTensor, src: LayoutTensor):
    """Efficiently copy data from registers (LOCAL) to global memory (DRAM).

    This function implements a high-performance memory transfer operation from
    register memory to global memory. It distributes the copy operation across
    multiple threads for maximum throughput while handling bounds checking for
    safety.

    Constraints:
        - The source tensor must be in LOCAL address space (registers).
        - The destination tensor must be in GENERIC or GLOBAL address space (DRAM).
        - Both tensors must have compatible data types.

    Parameters:
        dst_thread_layout: The layout used to distribute the destination tensor
            across threads. This determines how the workload is divided among
            participating threads.
        num_threads: Total number of threads participating in the copy
            operation. Defaults to the size of thread_layout.
        thread_scope: Defines whether operations are performed at `BLOCK` or
            `WARP` level. `BLOCK` scope involves all threads in a thread block,
            while `WARP` scope restricts operations to threads within the same
            warp. Defaults to `ThreadScope.BLOCK`.
        block_dim_count: The number of dimensions in the thread block.

    Args:
        dst: The destination tensor in global memory (DRAM).
        src: The source tensor in register memory (LOCAL) to be copied.
    """
    _copy_local_to_dram_validate_args(dst, src)

    alias num_busy_threads = dst_thread_layout.size()
    var worker_idx = _get_worker_idx[thread_scope, block_dim_count]()

    @parameter
    if num_threads > num_busy_threads:
        if worker_idx >= UInt(num_busy_threads):
            return

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
        var dst_idx_bound = (dst.dim[0]() * stride - dst_frag_offset).cast[
            dst_fragments.linear_idx_type
        ]()

        alias num_stores_per_thread = dst_fragments.layout.size()

        @parameter
        for i in range(num_stores_per_thread):
            alias src_idx = src.layout(i)
            alias dst_uint_dtype = _get_unsigned_type(
                dst_fragments.layout, dst_fragments.address_space
            )
            alias dst_static_idx = dst_fragments.layout(i)

            var dst_idx: Scalar[dst_fragments.linear_idx_type]

            @parameter
            if dst_fragments.layout.all_dims_known():
                dst_idx = dst_static_idx
            else:
                dst_idx = dst_fragments.runtime_layout(i)

            if dst_idx < dst_idx_bound:
                var src_element = Element[
                    index_type = src.linear_idx_type
                ].load(
                    src.ptr.offset(src_idx),
                    src.runtime_element_layout,
                )
                alias dst_element_type = Element[
                    dst.dtype, dst.element_layout, dst.linear_idx_type
                ]
                dst_element_type(
                    rebind[dst_element_type.element_data_type](
                        src_element.element_data.cast[dst.dtype]()
                    )
                ).store(dst_fragments.ptr.offset(dst_idx))


@always_inline("nodebug")
fn _copy_local_to_dram_static_row_major[
    dst_type: DType
](
    src: LayoutTensor,
    dst_fragments: LayoutTensor,
    dst_frag_offset: Int32,
    buffer: AMDBufferResource,
):
    alias M = dst_fragments.shape[0]()
    alias N = dst_fragments.shape[1]()

    @parameter
    for i in range(M):

        @parameter
        for j in range(N):
            alias idx = Layout.col_major(M, N)([i, j])
            alias src_frag_idx = src.layout(idx)
            alias dst_frag_idx = Int32(dst_fragments.layout(idx))

            var src_element = Element[index_type = src.linear_idx_type].load(
                src.ptr.offset(src_frag_idx),
                src.runtime_element_layout,
            )

            buffer.store(
                dst_frag_offset + dst_frag_idx,
                src_element.element_data.cast[dst_type](),
            )


@always_inline("nodebug")
fn _copy_local_to_dram[
    dst_thread_layout: Layout,
    num_threads: Int = dst_thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
    block_dim_count: Int = 1,
](dst: LayoutTensor, src: LayoutTensor, buffer: AMDBufferResource):
    constrained[is_amd_gpu(), "This function is only supported on AMD GPUs."]()

    _copy_local_to_dram_validate_args(dst, src)

    alias num_busy_threads = dst_thread_layout.size()
    var worker_idx = _get_worker_idx[thread_scope, block_dim_count]()

    @parameter
    if num_threads > num_busy_threads:
        if worker_idx >= UInt(num_busy_threads):
            return

    var dst_fragments = dst.distribute[dst_thread_layout](worker_idx)

    var base_ptr = buffer.get_base_ptr()
    var offset = (Int(dst.ptr) - base_ptr) // size_of[dst.dtype]()
    var dst_frag_offset = dst_fragments.distance(dst.ptr) + offset

    alias dst_element_stride = dst_fragments.element_layout.stride[1].value()

    @parameter
    if dst_element_stride == 1 and dst_fragments.layout.all_dims_known():
        _copy_local_to_dram_static_row_major[dst.dtype](
            src,
            dst_fragments,
            Int32(dst_frag_offset),
            buffer,
        )
    else:
        alias num_stores_per_thread = dst_fragments.layout.size()

        @parameter
        for i in range(num_stores_per_thread):
            alias src_idx = src.layout(i)
            alias dst_static_idx = dst_fragments.layout(i)
            var dst_idx = dst_frag_offset

            @parameter
            if dst_fragments.layout.all_dims_known():
                dst_idx += dst_static_idx
            else:
                dst_idx += dst_fragments.runtime_layout(i)

            var src_element = Element[index_type = src.linear_idx_type].load(
                src.ptr.offset(src_idx),
                src.runtime_element_layout,
            )

            @parameter
            if dst_element_stride == 1:
                buffer.store(
                    Int32(dst_idx),
                    src_element.element_data.cast[dst.dtype](),
                )
            else:

                @parameter
                for i in range(dst_fragments.element_layout.size()):
                    alias element_offset = dst_fragments.element_layout(i)
                    var src = src_element.element_data[i].cast[dst.dtype]()
                    buffer.store(
                        Int32(dst_idx + element_offset),
                        src,
                    )


@always_inline("nodebug")
fn copy_local_to_dram[
    dst_thread_layout: Layout,
    num_threads: Int = dst_thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
    block_dim_count: Int = 1,
](dst: LayoutTensor, src: LayoutTensor, dst_base: LayoutTensor):
    """Efficiently copy data from registers (LOCAL) to global memory (DRAM) on
    AMD GPUs.

    This function implements an optimized memory transfer operation specifically
    for AMD GPU architectures. It utilizes the hardware's buffer_store intrinsic
    to efficiently transfer data from registers to global memory while handling
    bounds checking. The function distributes the copy operation across multiple
    threads for maximum throughput.

    Constraints:
        - Only supported on AMD GPUs.
        - Destination tensor must be in GLOBAL address space.
        - Source tensor must be in LOCAL address space.
        - Data types must match between source and destination tensors.

    Parameters:
        dst_thread_layout: The layout used to distribute the destination tensor
            across threads. This determines how the workload is divided among
            participating threads.
        num_threads: Total number of threads participating in the copy
            operation. Defaults to the size of thread_layout.
        thread_scope: Defines whether operations are performed at `BLOCK` or
            `WARP` level. `BLOCK` scope involves all threads in a thread block,
            while `WARP` scope restricts operations to threads within the same
            warp. Defaults to `ThreadScope.BLOCK`.
        block_dim_count: The number of dimensions in the thread block.

    Args:
        dst: The destination tensor in global memory (DRAM).
        src: The source tensor in register memory (LOCAL address space) to be
            copied.
        dst_base: The original global memory tensor from which dst is derived.
            This is used to construct the buffer descriptor required by AMD's
            `buffer_store` intrinsic.

    Notes:

    - This function is particularly useful for writing computed results from
        registers back to global memory with minimal latency.
    - The offset calculation is optimized for performance rather than
        flexibility.
    """
    constrained[is_amd_gpu(), "This function is only supported on AMD GPUs."]()
    var buffer = make_amd_buffer_resource(dst_base)

    _copy_local_to_dram[
        dst_thread_layout, num_threads, thread_scope, block_dim_count
    ](dst, src, buffer)


@always_inline("nodebug")
fn _copy_dram_to_local[
    src_thread_layout: Layout,
    num_threads: Int = src_thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
    block_dim_count: Int = 1,
    cache_policy: CacheOperation = CacheOperation.ALWAYS,
](
    dst: LayoutTensor,
    src: LayoutTensor,
    buffer: AMDBufferResource,
    offset: OptionalReg[UInt] = None,
):
    constrained[is_amd_gpu(), "This function is only supported on AMD GPUs."]()
    alias simd_width = src.element_layout.size()
    _copy_local_to_dram_validate_args(src, dst)

    alias num_busy_threads = src_thread_layout.size()
    var worker_idx = _get_worker_idx[thread_scope, block_dim_count]()

    @parameter
    if num_threads > num_busy_threads:
        if worker_idx >= UInt(num_busy_threads):
            return

    var src_fragments = src.distribute[src_thread_layout](worker_idx)

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

    @always_inline
    @parameter
    fn offset_helper(offset_val: UInt):
        var src_frag_offset = Int32(
            src_fragments.distance(src.ptr) + offset_val
        )

        # These loads need to be row-major for L1 cache performance
        @parameter
        for i in range(M):

            @parameter
            for j in range(N):
                alias dst_frag_idx = Layout.col_major(M, N)([i, j])
                alias src_frag_idx = Int32(src_fragments.layout([i, j]))
                dst[dst_frag_idx, 0] = rebind[dst.element_type](
                    buffer.load[
                        src.dtype, simd_width, cache_policy=cache_policy
                    ](
                        src_frag_offset,
                        scalar_offset=src_frag_idx,
                    )
                )

    if offset:
        offset_helper(offset.value())
    else:
        var base_ptr = buffer.get_base_ptr()
        offset_helper(
            UInt(UInt((Int(src.ptr) - base_ptr)) // UInt(size_of[src.dtype]()))
        )


@always_inline("nodebug")
fn copy_dram_to_local[
    src_thread_layout: Layout,
    num_threads: Int = src_thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
    block_dim_count: Int = 1,
    cache_policy: CacheOperation = CacheOperation.ALWAYS,
](
    dst: LayoutTensor,
    src: LayoutTensor,
    src_base: LayoutTensor,
    offset: OptionalReg[UInt] = None,
):
    """Efficiently copy data from global memory (DRAM) to registers for AMD GPUs.

    This function implements an optimized memory transfer operation specifically
    for AMD GPU architectures. It utilizes the hardware's buffer_load intrinsic
    to efficiently transfer data from global memory to registers while handling
    bounds checking. The function distributes the copy operation across multiple
    threads for maximum throughput.

    Constraints:
        - Only supported on AMD GPUs.
        - The destination element layout size must match the SIMD width.
        - Source fragments must be rank 2 with known dimensions.

    Parameters:
        src_thread_layout: The layout used to distribute the source tensor
            across threads. This determines how the workload is divided among
            participating threads.
        num_threads: Total number of threads participating in the copy
            operation. Defaults to the size of thread_layout.
        thread_scope: Defines whether operations are performed at `BLOCK` or
            `WARP` level. `BLOCK` scope involves all threads in a thread block,
            while `WARP` scope restricts operations to threads within the same
            warp. Defaults to `ThreadScope.BLOCK`.
        block_dim_count: The number of dimensions in the thread block.
        cache_policy: The cache policy to use for the copy operation.
            Defaults to `CacheOperation.ALWAYS`.

    Args:
        dst: The destination tensor in register memory (LOCAL address space).
        src: The source tensor in global memory (DRAM) to be copied.
        src_base: The original global memory tensor from which src is derived.
            This is used to construct the buffer struct required by AMD's
            `buffer_load` intrinsic.
        offset: The offset in the global memory.

    Notes:

    - The offset calculation method significantly impacts performance.
        Current implementation optimizes for throughput over flexibility.
    - This function is particularly useful for prefetching data into registers
        before performing computations, reducing memory access latency.
    """
    constrained[is_amd_gpu(), "This function is only supported on AMD GPUs."]()
    var buffer = make_amd_buffer_resource(src_base)

    _copy_dram_to_local[
        src_thread_layout,
        num_threads,
        thread_scope,
        block_dim_count,
        cache_policy,
    ](dst, src, buffer, offset)


@always_inline("nodebug")
fn _copy_dram_to_local[
    src_thread_layout: Layout,
    num_threads: Int = src_thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
    block_dim_count: Int = 1,
    cache_policy: CacheOperation = CacheOperation.ALWAYS,
](dst: LayoutTensor, src_iter: LayoutTensorIter, buffer: AMDBufferResource):
    constrained[is_amd_gpu(), "This function is only supported on AMD GPUs."]()
    var src_tensor = src_iter[].vectorize[
        dst.element_layout.shape[0].value(), dst.element_layout.shape[1].value()
    ]()

    _copy_dram_to_local[
        src_thread_layout,
        num_threads,
        thread_scope,
        block_dim_count,
        cache_policy,
    ](dst, src_tensor, buffer, UInt(src_iter.offset))


@always_inline("nodebug")
fn copy_dram_to_local[
    src_thread_layout: Layout,
    num_threads: Int = src_thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
    block_dim_count: Int = 1,
    cache_policy: CacheOperation = CacheOperation.ALWAYS,
](dst: LayoutTensor, src_iter: LayoutTensorIter, bounds: UInt32):
    """Efficiently copy data from global memory (DRAM) to registers for AMD GPUs.

    This function implements an optimized memory transfer operation specifically
    for AMD GPU architectures. It utilizes the hardware's buffer_load intrinsic
    to efficiently transfer data from global memory to registers while handling
    bounds checking. The function distributes the copy operation across multiple
    threads for maximum throughput.

    Parameters:
        src_thread_layout: The layout used to distribute the source tensor
            across threads. This determines how the workload is divided among
            participating threads.
        num_threads: Total number of threads participating in the copy
            operation. Defaults to the size of thread_layout.
        thread_scope: Defines whether operations are performed at `BLOCK` or
            `WARP` level. `BLOCK` scope involves all threads in a thread block,
            while `WARP` scope restricts operations to threads within the same
            warp. Defaults to `ThreadScope.BLOCK`.
        block_dim_count: The number of dimensions in the thread block.
        cache_policy: The cache policy to use for the copy operation.
            Defaults to `CacheOperation.ALWAYS`.

    Args:
        dst: The destination tensor in register memory (LOCAL address space).
        src_iter: The source tensor iterator.
        bounds: Bounds of the buffer, based on the ptr of the src_iter.

    Constraints:
        - Only supported on AMD GPUs.
        - The destination element layout size must match the SIMD width.
        - Source fragments must be rank 2 with known dimensions.

    Notes:

    - The offset calculation method significantly impacts performance.
        Current implementation optimizes for throughput over flexibility.
    - This function is particularly useful for prefetching data into registers
        before performing computations, reducing memory access latency.
    """
    var buffer = make_amd_buffer_resource(src_iter, Int(bounds))

    _copy_dram_to_local[
        src_thread_layout,
        num_threads,
        thread_scope,
        block_dim_count,
        cache_policy,
    ](dst, src_iter, buffer)


@always_inline("nodebug")
fn copy_dram_to_local[
    src_thread_layout: Layout,
    num_threads: Int = src_thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
    block_dim_count: Int = 1,
](dst: LayoutTensor, src: LayoutTensor):
    """Efficiently copy data from global memory (DRAM) to registers.

    This function implements an optimized memory transfer operation from
    global memory to register memory. It distributes the copy operation across
    multiple threads for maximum throughput while handling bounds checking for
    safety.

    Constraints:
        - The source tensor must be in GLOBAL address space (DRAM).
        - The destination tensor must be in LOCAL address space (registers).
        - Both tensors must have compatible data types.

    Parameters:
        src_thread_layout: The layout used to distribute the source tensor
            across threads. This determines how the workload is divided among
            participating threads.
        num_threads: Total number of threads participating in the copy
            operation. Defaults to the size of thread_layout.
        thread_scope: Defines whether operations are performed at `BLOCK` or
            `WARP` level. `BLOCK` scope involves all threads in a thread block,
            while `WARP` scope restricts operations to threads within the same
            warp. Defaults to `ThreadScope.BLOCK`.
        block_dim_count: The number of dimensions in the thread block.

    Args:
        dst: The destination tensor in register memory (LOCAL address space).
        src:  The source tensor in global memory (DRAM).
    """

    alias num_busy_threads = src_thread_layout.size()
    var worker_idx = _get_worker_idx[thread_scope, block_dim_count]()

    @parameter
    if num_threads > num_busy_threads:
        if worker_idx >= UInt(num_busy_threads):
            return

    var src_fragments = src.distribute[src_thread_layout](worker_idx)

    @parameter
    if not src_fragments.masked:
        dst.copy_from(src_fragments)
    else:
        var src_frag_offset = src_fragments.distance(src.ptr)
        alias static_stride = src.layout.stride[0].value()

        @parameter
        if src.layout.all_dims_known():
            stride = static_stride
        else:
            stride = src.runtime_layout.stride.value[0]
        var src_idx_bound = (src.dim[0]() * stride - src_frag_offset).cast[
            src_fragments.linear_idx_type
        ]()

        alias num_stores_per_thread = src_fragments.layout.size()

        @parameter
        for i in range(num_stores_per_thread):
            alias dst_idx = dst.layout(i)
            alias src_uint_dtype = _get_unsigned_type(
                src_fragments.layout, src_fragments.address_space
            )
            alias src_static_idx = src_fragments.layout(i)

            var src_idx: Scalar[src_fragments.linear_idx_type]

            @parameter
            if src_fragments.layout.all_dims_known():
                src_idx = src_static_idx
            else:
                src_idx = src_fragments.runtime_layout(i)

            if src_idx < src_idx_bound:
                var src_element = Element[
                    index_type = src.linear_idx_type
                ].load(
                    src_fragments.ptr.offset(src_idx),
                    src_fragments.runtime_element_layout,
                )
                alias dst_element_type = Element[
                    dst.dtype, dst.element_layout, dst.linear_idx_type
                ]
                dst_element_type(
                    rebind[dst_element_type.element_data_type](
                        src_element.element_data.cast[dst.dtype]()
                    )
                ).store(dst.ptr.offset(dst_idx))


@always_inline("nodebug")
fn copy_local_to_shared[
    thread_layout: Layout,
    swizzle: OptionalReg[Swizzle] = None,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
    block_dim_count: Int = 1,
    *,
    row_major: Bool = False,
](
    dst: LayoutTensor[*_, address_space = _GPUAddressSpace.SHARED, **_],
    src: LayoutTensor[*_, address_space = _GPUAddressSpace.LOCAL, **_],
):
    """Synchronously copy data from local memory (registers) to SRAM (shared
    memory).

    This function performs a synchronous copy operation from register memory to
    shared memory in a GPU context, distributing the workload across multiple
    threads for parallel execution. It's particularly useful for transferring
    processed data from registers to shared memory for inter-thread
    communication.

    Constraints:

        - Destination tensor must be in SHARED address space.
        - Source tensor must be in LOCAL address space.
        - For optimal performance, the thread layout should match the memory
          access patterns of the tensors.

    Parameters:
        thread_layout: Layout defining how threads are organized for the
            operation. This determines how the workload is distributed among
            threads.
        swizzle: Optional swizzling function to rearrange the destination
            indices, which can improve memory access patterns and reduce bank
            conflicts.
        num_threads: Total number of threads participating in the copy
            operation. Defaults to the size of thread_layout.
        thread_scope: Defines whether operations are performed at `BLOCK` or
            `WARP` level. `BLOCK` scope involves all threads in a thread block,
            while `WARP` scope restricts operations to threads within the same
            warp. Defaults to `ThreadScope.BLOCK`.
        block_dim_count: The number of dimensions in the thread block.
        row_major: Whether to use row-major ordering for the copy operation.
            This is particularly relevant when prefetching from DRAM to SRAM
            via registers on AMD GPUs. Defaults to False.

    Args:
        dst: The destination tensor, which must be in shared memory (SRAM).
        src: The source tensor, which must be in local memory (registers).

    Performance:

    - Distributes the copy workload across multiple threads for parallel execution.
    - Can use swizzling to optimize memory access patterns and reduce bank conflicts.
    - Optimized for transferring data from registers to shared memory.
    - On AMD GPUs, the `row_major` parameter can be used to match the memory
        access pattern used during prefetching from DRAM to registers.

    Notes:

    - The destination tensor must be in `SHARED` address space (SRAM).
    - The source tensor must be in `LOCAL` address space (registers).
    - This function is particularly useful in GPU kernels for sharing processed
        data between threads in the same block.
    - The `row_major` parameter is specifically designed for AMD GPUs when using
        a prefetching pattern from DRAM to SRAM via registers.
    """
    constrained[
        dst.address_space == _GPUAddressSpace.SHARED,
        "dst address space must be SHARED.",
    ]()

    constrained[
        src.address_space == _GPUAddressSpace.LOCAL,
        "src address space must be LOCAL.",
    ]()

    alias num_busy_threads = thread_layout.size()
    var worker_idx = _get_worker_idx[thread_scope, block_dim_count]()

    @parameter
    if num_threads > num_busy_threads:
        if worker_idx >= UInt(num_busy_threads):
            return

    constrained[
        src.dtype == dst.dtype
        or (src.dtype is DType.float32 and dst.dtype.is_half_float()),
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
            alias align_src = align_of[SIMD[src.dtype, src.element_size]]()
            alias align_dst = align_of[SIMD[dst.dtype, dst.element_size]]()
            var dst_frag_offset = dst_frag.distance(dst.ptr)

            @parameter
            for i in range(num_vecs):
                alias src_idx = src.layout(i)
                alias dst_idx = dst_frag.layout(i)
                alias dst_idx_base = dst_idx % swizzle_fn.size()
                alias dst_idx_diff = dst_idx - dst_idx_base
                var swizzled_idx = (
                    swizzle_fn(dst_frag_offset + dst_idx_base) + dst_idx_diff
                )
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
        alias M = product(dst_frag.layout.shape[0])
        alias N = product(dst_frag.layout.shape[1])

        constrained[
            dst_frag.layout.rank() == 2,
            "dst_frag must be rank 2.",
        ]()

        @parameter
        for i in range(M):

            @parameter
            for j in range(N):
                # The order here needs to match the order of the loads in copy_dram_to_local
                alias idx = Layout.col_major(M, N)([i, j])
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
    """Synchronously copy data between local memory (register) tensors with type
    conversion.

    This function performs a synchronous copy operation between register tensors
    in a GPU context, with support for converting from float32 to half-precision
    formats (bfloat16/float16). It's particularly optimized for specific tensor
    layouts commonly used in matrix multiplication operations.

    Constraints:
        - Destination tensor must be in `LOCAL` address space.
        - Source tensor must be in `LOCAL` address space.
        - Destination tensor must have a half-precision floating-point data type.
        - Source tensor must have float32 data type.
        - Both tensors must have the same total size.

    Args:
        dst: The destination tensor, which must be in local memory (registers)
            and have a half-precision floating-point data type (bfloat16 or
            float16).
        src: The source tensor, which must be in local memory (registers) and
            have float32 data type.

    Example:

    ```mojo
    from layout import LayoutTensor, Layout
    from layout.layout_tensor import copy_local_to_local
    from gpu.memory import AddressSpace

    fn kernel():
        ...
        var src_reg = LayoutTensor[DType.float32,
            Layout.row_major(16, 8),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ].stack_allocation().fill(1)

        var dst_reg = LayoutTensor[DType.bfloat16,
            Layout.row_major(16, 8),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ].stack_allocation()

        # Process data in float32 registers
        # ...

        # Convert and copy to bfloat16 registers
        copy_local_to_local(dst_reg, src_reg)
    ```

    Performance:

    - Optimized for specific 2D tensor layouts with contiguous inner dimensions.
    - Special fast path for 2D tensors with specific layouts used in matrix
        multiplication.
    - For MMA (Matrix Multiply-Accumulate) operations, efficiently handles the
        conversion between output fragments and input fragments with different
        layouts.
    - Falls back to element-wise copy for general cases.

    Notes:

    - Both source and destination tensors must be in `LOCAL` address space
        (registers).
    - This function currently only supports copying from float32 to half-precision formats.
    - For 2D tensors with stride[1] == 1, a specialized fast path is used that's optimized
        for matrix multiplication patterns.
    - This function is particularly useful in GPU kernels for converting between different
        precision formats while keeping data in registers.
    """
    constrained[
        dst.address_space == _GPUAddressSpace.LOCAL,
        "dst address space must be LOCAL.",
    ]()

    constrained[
        src.address_space == _GPUAddressSpace.LOCAL,
        "src address space must be LOCAL.",
    ]()

    constrained[
        dst.dtype.is_half_float() and src.dtype is DType.float32,
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
    dtype: DType,
    layout: Layout,
    origin: Origin[mut],
    /,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
    alignment: Int = align_of[dtype](),
    circular: Bool = False,
    axis: OptionalReg[Int] = None,
    layout_int_type: DType = _get_index_type(address_space),
    linear_idx_type: DType = _get_index_type(address_space),
    masked: Bool = False,
](Defaultable):
    """Iterator for traversing a memory buffer with a specific layout.

    `LayoutTensorIter` provides a way to iterate through memory according to a
    specific layout pattern, constructing layout tensors at each position. This
    enables efficient traversal of multi-dimensional data structures with custom
    memory layouts.

    Parameters:
        mut: Whether the iterator allows mutation of the underlying data.
        dtype: The data type of the tensor elements.
        layout: The memory layout pattern to follow during iteration.
        origin: Origin tracking for memory safety.
        address_space: The memory address space (`GLOBAL`, `SHARED`, etc.).
        alignment: Memory alignment requirement for the data.
        circular: Whether iteration wraps around at boundaries.
        axis: Optional axis for dimension-specific operations.
        layout_int_type: Integer type used for layout indices.
        linear_idx_type: Integer type used for indexing into memory.
        masked: Whether to apply bounds masking during iteration.

    Notes:

    The returned layout tensor is NOT vectorized. Users should explicitly vectorize
    if needed for performance-critical operations.
    """

    alias layout_uint_type = Scalar[
        _unsigned_integral_type_of[layout_int_type]()
    ]
    """The unsigned integer type used for layout, based on layout and address space."""

    alias linear_uint_type = Scalar[
        _unsigned_integral_type_of[linear_idx_type]()
    ]
    """The unsigned integer type used for indexing into memory."""

    var ptr: UnsafePointer[
        Scalar[dtype],
        address_space=address_space,
        mut=mut,
        origin=origin,
    ]
    """Pointer to the memory region being iterated, with appropriate type and memory attributes."""

    var offset: Self.linear_uint_type
    """Current offset from the base pointer, representing the iterator's position in memory."""

    var stride: Self.linear_uint_type
    """Step size between consecutive elements or blocks in memory during iteration."""

    var bound: Self.linear_uint_type
    """Upper bound of the memory region, limiting the iteration range."""

    alias RuntimeLayoutType = RuntimeLayout[
        layout, element_type=layout_int_type, linear_idx_type=linear_idx_type
    ]

    var runtime_layout: Self.RuntimeLayoutType
    """Runtime representation of the layout pattern used for mapping logical indices to memory locations."""

    var dimension_bound: Self.layout_uint_type
    """Boundary value for the current dimension when iterating along a specific axis."""

    var idx: Self.linear_uint_type
    """Current logical index position within the iteration sequence."""

    @always_inline
    fn __init__(out self):
        """Initialize an empty iterator.

        Creates a default iterator with zero values, typically used as a
        placeholder or default value.
        """

        @parameter
        if axis:
            constrained[
                not circular,
                "Circular use case is not supported if an axis is defined.",
            ]()

        self.ptr = {}
        self.offset = 0
        self.stride = 0
        self.bound = 0
        self.runtime_layout = {}
        self.dimension_bound = 0
        self.idx = 0

    @always_inline
    fn __init__(
        out self,
        ptr: UnsafePointer[
            Scalar[dtype],
            address_space=address_space,
            mut=mut,
            origin=origin,
        ],
        bound: Self.linear_uint_type,
        stride: Self.linear_uint_type = layout.size(),
        offset: Self.linear_uint_type = 0,
    ):
        """Initialize an iterator with a pointer and basic parameters.

        Creates an iterator for a memory region with the specified bounds and
        stride.

        Args:
            ptr: Pointer to the beginning of the memory region.
            bound: Upper bound of the memory region.
            stride: Step size between consecutive elements (defaults to layout
                size).
            offset: Initial offset from the base pointer.

        Constraints:
            The layout must have all dimensions known at compile time.
        """
        constrained[
            layout.all_dims_known(),
            "Cannot construct LayoutTensorIter with unknown layout.",
        ]()

        constrained[
            layout_int_type.is_signed() and linear_idx_type.is_signed(),
            "Layout integer type and linear index type must be signed.",
        ]()

        self.ptr = ptr
        self.bound = bound
        self.stride = stride
        self.runtime_layout = {}
        self.offset = offset
        self.dimension_bound = 0
        self.idx = 0

    @always_inline
    fn __init__(
        out self,
        ptr: UnsafePointer[
            Scalar[dtype],
            address_space=address_space,
            mut=mut,
            origin=origin,
        ],
        bound: Self.linear_uint_type,
        runtime_layout: RuntimeLayout[layout, **_],
        stride: Self.linear_uint_type = (
            layout.size() if layout.all_dims_known() else UNKNOWN_VALUE
        ),
        offset: Self.linear_uint_type = 0,
        dimension_bound: Self.layout_uint_type = 0,
        idx: Self.linear_uint_type = 0,
    ):
        """Initialize an iterator with a runtime layout.

        Creates an iterator with a runtime-determined layout, allowing for more
        flexible memory traversal patterns.

        Args:
            ptr: Pointer to the beginning of the memory region.
            bound: Upper bound of the memory region.
            runtime_layout: Layout determined at runtime.
            stride: Step size between consecutive elements.
            offset: Initial offset from the base pointer.
            dimension_bound: Bound for the specified dimension when using masked
                iteration.
            idx: Initial index position.

        Constraints:
            The runtime layout must have the same bitwidth as specified for the
            iterator. Circular iteration is not supported when an axis is
            defined.
        """

        constrained[
            runtime_layout.linear_idx_type == linear_idx_type,
            "Mismatch of index type for RuntimeLayout and LayoutTensorIter.",
        ]()

        constrained[
            runtime_layout.element_type == layout_int_type,
            (
                "Mismatch of dimension type for RuntimeLayout and"
                " LayoutTensorIter."
            ),
        ]()

        constrained[
            layout_int_type.is_signed() and linear_idx_type.is_signed(),
            "Layout integer type and linear index type must be signed.",
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
        self.runtime_layout = rebind[Self.RuntimeLayoutType](runtime_layout)
        self.dimension_bound = dimension_bound
        self.idx = idx

    alias LayoutTensorType = LayoutTensor[
        dtype,
        layout,
        origin,
        address_space=address_space,
        masked=masked,
        alignment=alignment,
        layout_int_type=layout_int_type,
        linear_idx_type=linear_idx_type,
    ]

    @always_inline
    fn get(self) -> Self.LayoutTensorType:
        """Get the layout tensor at the current iterator position.

        Returns a layout tensor representing the data at the current position
        of the iterator.

        Returns:
            A tensor view at the current iterator position with the
            same type, layout, and memory characteristics as specified by the
            output parameter.
        """
        # TODO: Use deref `[]` to be consistent with mojo feature.

        return Self.LayoutTensorType(
            self.ptr + Int(self.offset),
            self.runtime_layout,
        )

    @always_inline
    fn __getitem__(
        self,
    ) -> Self.LayoutTensorType:
        """Get the layout tensor at the current iterator position.

        Operator overload that returns a layout tensor representing the data
        at the current position of the iterator.

        Returns:
            A layout tensor at the current iterator position.
        """
        return self.get()

    @always_inline
    fn _clip_shape(self) -> Self.RuntimeLayoutType:
        """Clip the shape based on dimension bounds.

        Internal method that adjusts the shape of the layout tensor based on
        dimension bounds when using masked iteration.

        Returns:
            A new runtime layout with adjusted shape.
        """
        new_shape = self.runtime_layout.shape
        var cur_dim = new_shape.value[axis.value()]
        new_shape.value[axis.value()] = max(
            0, min(Int(Int(self.dimension_bound) - self.idx * cur_dim), cur_dim)
        )
        return Self.RuntimeLayoutType(new_shape, self.runtime_layout.stride)

    @always_inline
    fn __iadd__[T: Intable](mut self, rhs: T):
        """Increment the iterator by an integer value.

        Advances the iterator by the specified number of positions.

        Parameters:
            T: A type that can be converted to an integer.

        Args:
            rhs: The number of positions to advance.

        Notes:

        This function is unsafe. It omits bound checking for performance
        reasons. Caller must ensure the index doesn't go out-of-bounds.
        """
        self += Self.linear_uint_type(Int(rhs))

    @always_inline
    fn __iadd__(mut self, rhs: Self.linear_uint_type):
        """Increment the iterator by a uint value.

        Advances the iterator by the specified number of positions.

        Args:
            rhs: The number of positions to advance.

        Notes:

        This function is unsafe. It omits bound checking for performance
        reasons. Caller must ensure the index doesn't go out-of-bounds.
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
        """Increment the iterator by 1.

        Advances the iterator by a single position. This is equivalent to
        `iter += 1` but without the division operation, making it more
        efficient.
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
        """Return an iterator pointing to a position ahead by rhs steps.

        Creates a new iterator that points rhs positions ahead of the current
        one.

        Parameters:
            T: An integer-convertible type for the step size.

        Args:
            rhs: The number of positions to advance.

        Returns:
            A new iterator pointing to the advanced position.
        """
        var next_idx = Self.linear_uint_type(0)
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
    fn next(self, rhs: Self.linear_uint_type = 1) -> Self:
        """Return an iterator pointing to a position ahead by rhs steps.

        Creates a new iterator that points rhs positions ahead of the current
        one.

        Args:
            rhs: The number of positions to advance (defaults to 1).

        Returns:
            A new iterator pointing to the advanced position.
        """
        return self.next(Int(rhs))

    @always_inline
    fn next_unsafe(self, rhs: Self.linear_uint_type = 1) -> Self:
        """Return an iterator pointing to a position ahead by rhs steps (unsafe
        version).

        Creates a new iterator that points rhs positions ahead of the current
        one. This is an unsafe version that omits certain checks for
        performance.

        Args:
            rhs: The number of positions to advance (defaults to 1).

        Returns:
            A new iterator pointing to the advanced position.

        Constraints:
            Cannot be used with masked iterators.
            User must ensure rhs < bound / stride.
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

    alias ReshapeType[dst_layout: Layout] = LayoutTensorIter[
        dtype,
        dst_layout,
        origin,
        address_space=address_space,
        alignment=alignment,
        circular=circular,
        layout_int_type=layout_int_type,
        linear_idx_type=linear_idx_type,
        masked=masked,
    ]

    @always_inline
    fn reshape[dst_layout: Layout](self) -> Self.ReshapeType[dst_layout]:
        """Reshape the iterator to a new layout.

        This method creates a new iterator with a different layout while
        preserving the underlying data. The new layout must have the same total
        size as the original.

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

        return Self.ReshapeType[dst_layout](
            self.ptr,
            Int(self.bound),
            Self.ReshapeType[dst_layout].RuntimeLayoutType(),
            Int(self.stride),
            Int(self.offset),
            dimension_bound=Int(self.dimension_bound),
            idx=Int(self.idx),
        )

    alias BitcasType[
        new_type: DType,
        *,
        address_space: AddressSpace = Self.address_space,
        alignment: Int = Self.alignment,
    ] = LayoutTensorIter[
        new_type,
        layout,
        origin,
        address_space=address_space,
        alignment=alignment,
        circular = Self.circular,
        layout_int_type=layout_int_type,
        linear_idx_type=linear_idx_type,
        masked=masked,
    ]

    @always_inline
    fn bitcast[
        new_type: DType,
        *,
        target_address_space: AddressSpace = Self.address_space,
        target_alignment: Int = Self.alignment,
    ](self) -> Self.BitcasType[
        new_type, address_space=address_space, alignment=alignment
    ]:
        """Reinterpret the iterator's underlying pointer as a different data
        type.

        This method performs a bitcast operation, allowing you to view the same
        memory location as a different data type without copying or converting
        the data.

        Parameters:
            new_type: The target data type to cast to.
            target_address_space: The memory address space for the new
                iterator (defaults to current).
            target_alignment: Memory alignment requirement for the new
                iterator (defaults to current).

        Returns:
            A new LayoutTensorIter with the same layout but different data type.
        """
        return Self.BitcasType[
            new_type, address_space=address_space, alignment=alignment
        ](
            self.ptr.bitcast[Scalar[new_type]]().address_space_cast[
                address_space
            ](),
            Int(self.bound),
            self.runtime_layout,
            Int(self.stride),
            Int(self.offset),
            dimension_bound=Int(self.dimension_bound),
            idx=Int(self.idx),
        )
