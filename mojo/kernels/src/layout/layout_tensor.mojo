# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import Optional, OptionalReg
from math import ceildiv
from os import abort
from sys import alignof, prefetch, simdwidthof, sizeof
from sys.intrinsics import PrefetchOptions

from algorithm import vectorize
from builtin.int import int as _int
from gpu.id import ThreadIdx
from gpu.memory import Fill, async_copy
from layout.element import Element
from memory import UnsafePointer, memcpy, stack_allocation, memset_zero
from memory.reference import AddressSpace, _GPUAddressSpace

from utils import StaticIntTuple, StaticTuple
from utils.numerics import max_finite

from .int_tuple import (
    fill_like,
    flatten,
    idx2crd,
    product,
    to_int,
    depth,
)
from .layout import *
from .runtime_layout import RuntimeLayout
from .runtime_layout import coalesce as runtime_coalesce
from .runtime_layout import make_layout as make_runtime_layout
from .runtime_tuple import RuntimeTuple
from .swizzle import Swizzle, make_ldmatrix_swizzle
from .fillers import arange


fn _compute_distribute_layout[
    data_layout: Layout,
    threads_layout: Layout,
    axis: OptionalReg[Int] = None,
    /,
    __experimental_non_homogeneous_tile: Bool = False,
]() -> Layout:
    """Distribute thread_layout into self layout, if axis is provided
    distribute into threads_layout projected into this axis.
    """
    var thread_tile = LayoutList()

    @parameter
    if axis:
        var divided_layout = zipped_divide(
            data_layout, Layout(threads_layout.shape[axis.value()])
        )

        @parameter
        if not __experimental_non_homogeneous_tile:
            return divided_layout
        else:
            divided_layout.shape[1] = propagate_unknown(
                divided_layout.shape[1], data_layout.shape
            )
            # TODO: remove this after KERN-983 is fixed
            divided_layout.stride[1] = propagate_unknown(
                divided_layout.stride[1], data_layout.stride
            )
            return divided_layout

    else:
        for dim in threads_layout.shape:
            thread_tile.append(Layout(dim))

        var divided_layout = zipped_divide(data_layout, thread_tile)

        @parameter
        if not __experimental_non_homogeneous_tile:
            return divided_layout
        else:
            divided_layout.shape[1] = propagate_unknown(
                divided_layout.shape[1], data_layout.shape
            )
            # TODO: remove this after KERN-983 is fixed
            divided_layout.stride[1] = propagate_unknown(
                divided_layout.stride[1], data_layout.stride
            )
            return divided_layout


# Returns an IntTuple with all ones except axis same as input t, when
# submode_axis is provided the projection happens on the submode only.
fn _project_on_axis[
    axis: Int, submode_axis: OptionalReg[Int] = None
](t: IntTuple) -> IntTuple:
    if not submode_axis:
        var p_t = fill_like(t, 0)
        p_t[axis] = fill_like(t[axis], 1)
        return p_t
    var p_t = fill_like(t, 1)
    p_t[axis] = fill_like(t[axis], 0)
    p_t[axis][submode_axis.value()] = 1
    return p_t


fn _get_index_type(layout: Layout, address_space: AddressSpace) -> DType:
    if layout.cosize() < _int(max_finite[DType.int32]()):
        return DType.int32
    elif address_space in (
        _GPUAddressSpace.SHARED,
        _GPUAddressSpace.CONSTANT,
        _GPUAddressSpace.PARAM,
    ):
        return DType.int32
    else:
        return DType.index


alias _swizzle_signature = fn[type: DType] (Scalar[type]) -> Scalar[type]


# Returns the size of variadic integer parameters.
#
fn __get_len[*var_int: Int]() -> Int:
    return __mlir_op.`pop.variadic.size`(var_int)


# Returns the size of the slice in layout dim.
#
fn _get_slice_size(layout: Layout, slc: Slice, dim: Int) -> Int:
    var start: Int
    var end: Int
    start, end, _ = slc.indices(to_int(layout.shape[dim]))
    return end - start


# Returns true if n isn't in `tuple`.
#
fn _not_in_tuple[n: Int, size: Int, tuple: StaticIntTuple[size]]() -> Bool:
    @parameter
    for i in range(size):

        @parameter
        if tuple[i] == n:
            return False
    return True


@register_passable("trivial")
struct LayoutTensor[
    dtype: DType,
    layout: Layout,
    rank: Int = layout.rank(),
    /,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
    element_layout: Layout = Layout(1, 1),
    __experimental_non_homogeneous_tile: Bool = False,
](CollectionElement, CollectionElementNew, Stringable, Formattable):
    """This is a Tensor type that has a specified memory layout and rank. The
    following example demonstrate a LayoutTensor of float32 with a row major
    layout of shape (5, 4).

        alias f32 = DType.float32
        var tensor_5x4 = LayoutTensor[f32, Layout.row_major(5,4)].stack_allocation()

    Parameters:
        dtype: The data type of the underlying pointer.
        layout: The memory layout of the Tensor.
        rank: The rank of the Tensor.
        address_space: The address space of the underlying pointer.
        element_layout: The memory layout of each element in the Tensor.
        __experimental_non_homogeneous_tile: An experimental feature for dynamic tile sizes.
    """

    alias index_type: DType = _get_index_type(layout, address_space)

    var ptr: UnsafePointer[Scalar[dtype], address_space]

    var runtime_layout: RuntimeLayout[layout]

    var runtime_element_layout: RuntimeLayout[element_layout]

    alias element_size = element_layout.size()
    alias element_type = SIMD[dtype, Self.element_size]

    # An offset of the global coords.
    var org_coords_offset: StaticIntTuple[rank]
    # The stride of the global coords.
    var org_coords_stride: StaticIntTuple[rank]

    # ===------------------------------------------------------------------=== #
    # Life cycle methods
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn __init__(
        inout self,
        ptr: UnsafePointer[Scalar[dtype], address_space],
        /,
        *,
        org_coords_offset: StaticIntTuple[rank] = StaticIntTuple[rank](0),
        org_coords_stride: StaticIntTuple[rank] = StaticIntTuple[rank](1),
    ):
        """Create a LayoutTensor with an UnsafePointer. Expect layout to be
        fully static.

        Args:
            ptr: The UnsafePointer pointing to the underlying data.
            org_coords_offset: The coordinate offset with respect to the global
                               coordinates of the pointer.
            org_coords_stride: The coordinate stride with respect to the global
                               coordinates of the pointer.
        """

        constrained[layout.all_dims_known(), "Layout must be fully static"]()
        self.ptr = ptr
        self.runtime_layout = RuntimeLayout[layout]()
        self.runtime_element_layout = RuntimeLayout[element_layout]()
        self.org_coords_offset = org_coords_offset
        self.org_coords_stride = org_coords_stride

    @always_inline
    fn __init__(
        inout self,
        ptr: UnsafePointer[Scalar[dtype], address_space],
        runtime_layout: RuntimeLayout[layout],
        /,
        *,
        org_coords_offset: StaticIntTuple[rank] = StaticIntTuple[rank](0),
        org_coords_stride: StaticIntTuple[rank] = StaticIntTuple[rank](1),
    ):
        """Create a LayoutTensor with an UnsafePointer. Expect element layout
        to be fully static.

        Args:
            ptr: The UnsafePointer pointing to the underlying data.
            runtime_layout: The runtime layout of the LayoutTensor.
            org_coords_offset: The coordinate offset with respect to the global
                               coordinates of the pointer.
            org_coords_stride: The coordinate stride with respect to the global
                               coordinates of the pointer.
        """

        constrained[
            element_layout.all_dims_known(), "Layout must be fully static"
        ]()
        self.ptr = ptr
        self.runtime_layout = runtime_layout
        self.runtime_element_layout = RuntimeLayout[element_layout]()
        self.org_coords_offset = org_coords_offset
        self.org_coords_stride = org_coords_stride

    @always_inline
    fn __init__(
        inout self,
        ptr: UnsafePointer[Scalar[dtype], address_space],
        runtime_layout: RuntimeLayout[layout],
        element_runtime_layout: RuntimeLayout[element_layout],
        /,
        *,
        org_coords_offset: StaticIntTuple[rank] = StaticIntTuple[rank](0),
        org_coords_stride: StaticIntTuple[rank] = StaticIntTuple[rank](1),
    ):
        """Create a LayoutTensor with an UnsafePointer, a runtime layout of the
        Tensor, the runtime layout of each element.

        Args:
            ptr: The UnsafePointer pointing to the underlying data.
            runtime_layout: The runtime layout of the LayoutTensor.
            element_runtime_layout: The runtime layout of each element.
            org_coords_offset: The coordinate offset with respect to the global
                               coordinates of the pointer.
            org_coords_stride: The coordinate stride with respect to the global
                               coordinates of the pointer.
        """
        self.ptr = ptr
        self.runtime_layout = runtime_layout
        self.runtime_element_layout = element_runtime_layout
        self.org_coords_offset = org_coords_offset
        self.org_coords_stride = org_coords_stride

    fn __init__(inout self, *, other: Self):
        """Explicitly copy the other LayoutTensor.

        Args:
            other: The LayoutTensor to copy.
        """
        self = other

    @always_inline
    fn bitcast[
        new_type: DType,
        /,
        address_space: AddressSpace = Self.address_space,
        element_layout: Layout = Self.element_layout,
    ](self) -> LayoutTensor[
        new_type,
        layout,
        address_space=address_space,
        element_layout=element_layout,
    ]:
        """Bitcast the underlying pointer to a new data type.

        Parameters:
            new_type: The new data type it is casting to.
            address_space: The address space of the returned LayoutTensor.
            element_layout: The element layout of the returned LayoutTensor.
        """
        return LayoutTensor[
            new_type,
            layout,
            address_space=address_space,
            element_layout=element_layout,
        ](self.ptr.bitcast[new_type, address_space=address_space]())

    @always_inline
    fn _offset(self, m: Int, n: Int) -> Int:
        return Self.stride[0]() * m + Self.stride[1]() * n

    @always_inline
    fn __elementwise_unary[
        func: fn (Self.element_type) capturing -> (Self.element_type),
        inplace: Bool = False,
    ](self) -> Self:
        constrained[
            layout.all_dims_known(),
            (
                "__elmentwise_unary must operates on tensors of statically know"
                " layouts"
            ),
        ]()

        var res_tensor = self if inplace else Self.stack_allocation()

        @parameter
        for i in range(self.layout.size()):
            alias idx = self.layout(i)
            res_tensor.ptr.store(
                idx, func(self.ptr.load[width = Self.element_size](idx))
            )
        return res_tensor

    @always_inline
    fn __elementwise_binary_with_broadcast[
        func: fn (Self.element_type, Self.element_type) capturing -> (
            Self.element_type
        ),
        other_layout: Layout,
        inplace: Bool = False,
    ](
        self,
        other: LayoutTensor[
            dtype,
            other_layout,
            _,
            address_space=address_space,
            element_layout=element_layout,
        ],
    ) -> Self:
        @parameter
        if rank == other.rank:

            @parameter
            for axis in range(rank):
                constrained[
                    other.shape[axis]() == self.shape[axis](),
                    (
                        "__elementwise_binary_with_broadcast requires shape to"
                        " be the same for tensors of the same rank"
                    ),
                ]()

        constrained[
            layout.all_dims_known(),
            (
                "__elementwise_binary_with_broadcast must operates on tensors"
                " of statically know layouts"
            ),
        ]()
        constrained[
            other.rank <= rank,
            (
                "__elementwise_binary_with_broadcast must operates on tensor of"
                " equal of lower rank"
            ),
        ]()

        # TODO(KERN-812): Support numpy like broadcasting and relax rank-2
        # constrain.
        constrained[
            rank == 2 or rank == other.rank,
            "Only supports rank-2 tensor, or same rank",
        ]()

        var res_tensor = self if inplace else Self.stack_allocation()

        @parameter
        if other.rank == 1:
            constrained[
                other.shape[0]() == self.shape[0](),
                (
                    "__elementwise_binary_with_broadcast 1d tensor operand must"
                    " have a dim that matches the tensors"
                ),
            ]()

            @parameter
            for i in range(self.layout.size()):
                alias other_size = other.layout.size()

                alias lhs_idx = self.layout(i)
                alias rhs_idx = other.layout(i % other_size)

                res_tensor.ptr.store(
                    lhs_idx,
                    func(
                        self.ptr.load[width = Self.element_size](lhs_idx),
                        other.ptr.load[width = Self.element_size](rhs_idx),
                    ),
                )
            return res_tensor

        @parameter
        for i in range(self.layout.size()):
            alias idx = self.layout(i)
            res_tensor.ptr.store(
                idx,
                func(
                    self.ptr.load[width = Self.element_size](idx),
                    other.ptr.load[width = Self.element_size](idx),
                ),
            )
        return res_tensor

    @always_inline
    fn __add__(self, other: Scalar[dtype]) -> Self:
        """Add the LayoutTensor with a scalar value. The scalar value will be
        broadcasted to the entire tensor.

        Args:
            other: The scalar value.
        """

        @parameter
        fn add_val(val: Self.element_type) -> Self.element_type:
            return Self.element_type(other) + val

        return self.__elementwise_unary[add_val]()

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

        _ = self.__elementwise_unary[add_val, inplace=True]()

    @always_inline
    fn __add__[
        other_layout: Layout
    ](
        self,
        other: LayoutTensor[
            dtype,
            other_layout,
            _,
            address_space=address_space,
            element_layout=element_layout,
        ],
    ) -> Self:
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

        return self.__elementwise_binary_with_broadcast[add_val](other)

    @always_inline
    fn __iadd__[
        other_layout: Layout
    ](
        self,
        other: LayoutTensor[
            dtype,
            other_layout,
            _,
            address_space=address_space,
            element_layout=element_layout,
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

        _ = self.__elementwise_binary_with_broadcast[add_val, inplace=True](
            other
        )

    @always_inline
    fn __mul__(self, other: Scalar[dtype]) -> Self:
        """Multiply the LayoutTensor with a scalar value. The scalar value will
        be broadcasted to the entire tensor.

        Args:
            other: The scalar value.
        """

        @parameter
        fn mul_val(val: Self.element_type) -> Self.element_type:
            return Self.element_type(other) * val

        return self.__elementwise_unary[mul_val]()

    @always_inline
    fn __mul__[
        other_layout: Layout
    ](
        self,
        other: LayoutTensor[
            dtype,
            other_layout,
            _,
            address_space=address_space,
            element_layout=element_layout,
        ],
    ) -> Self:
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

        return self.__elementwise_binary_with_broadcast[mul_val](other)

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

        _ = self.__elementwise_unary[mul_val, inplace=True]()

    @always_inline
    fn __imul__[
        other_layout: Layout
    ](
        self,
        other: LayoutTensor[
            dtype,
            other_layout,
            _,
            address_space=address_space,
            element_layout=element_layout,
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

        _ = self.__elementwise_binary_with_broadcast[mul_val, inplace=True](
            other
        )

    @always_inline
    fn __sub__(self, other: Scalar[dtype]) -> Self:
        """Subtract the LayoutTensor with a scalar value. The scalar value will be
        broadcasted to the entire tensor.

        Args:
            other: The scalar value.
        """

        @parameter
        fn sub_val(val: Self.element_type) -> Self.element_type:
            return val - Self.element_type(other)

        return self.__elementwise_unary[sub_val]()

    @always_inline
    fn __sub__[
        other_layout: Layout,
    ](
        self,
        other: LayoutTensor[
            dtype,
            other_layout,
            _,
            address_space=address_space,
            element_layout=element_layout,
        ],
    ) -> Self:
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

        return self.__elementwise_binary_with_broadcast[sub_val](other)

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

        _ = self.__elementwise_unary[sub_val, inplace=True]()

    @always_inline
    fn __isub__[
        other_layout: Layout
    ](
        self,
        other: LayoutTensor[
            dtype,
            other_layout,
            _,
            address_space=address_space,
            element_layout=element_layout,
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

        _ = self.__elementwise_binary_with_broadcast[sub_val, inplace=True](
            other
        )

    @always_inline
    fn __truediv__(self, other: Scalar[dtype]) -> Self:
        """Truediv the LayoutTensor with a scalar value. The scalar value will be
        broadcasted to the entire tensor.

        Args:
            other: The scalar value.
        """

        @parameter
        fn div_val(val: Self.element_type) -> Self.element_type:
            return val / Self.element_type(other)

        return self.__elementwise_unary[div_val]()

    @always_inline
    fn __truediv__[
        other_layout: Layout
    ](
        self,
        other: LayoutTensor[
            dtype,
            other_layout,
            _,
            address_space=address_space,
            element_layout=element_layout,
        ],
    ) -> Self:
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

        return self.__elementwise_binary_with_broadcast[div_val](other)

    @always_inline
    fn __getitem__(self, *dims: Int) -> Self.element_type:
        """Get the element of the tensor with a specified index. Note that the
        size of index has to match the rank of the tensor.

        Args:
            dims: The indexes that specify which element to retrieve.
        """

        var strides = self.runtime_layout.stride.value
        var offset = Self._getOffset(strides, dims)

        return (
            Element[dtype, Self.element_layout]
            .load(self.ptr.offset(offset), self.runtime_element_layout)
            .element_data
        )

    @always_inline
    fn __setitem__(self, d0: Int, val: Self.element_type):
        """Set the element of the tensor with a specified index and value.

        Args:
            d0: The first dimensional index.
            val: The value writing to the tensor.
        """

        var strides = self.runtime_layout.stride.value
        var offset = Self._getOffset(strides, VariadicList[Int](d0))

        Element[dtype, Self.element_layout](
            val, self.runtime_element_layout
        ).store(self.ptr.offset(offset))

    @always_inline
    fn __setitem__(self, d0: Int, d1: Int, val: Self.element_type):
        """Set the element of the tensor with a specified index and value.

        Args:
            d0: The first dimensional index.
            d1: The second dimensional index.
            val: The value writing to the tensor.
        """

        var strides = self.runtime_layout.stride.value
        var offset = Self._getOffset(strides, VariadicList[Int](d0, d1))

        Element[dtype, Self.element_layout](
            val, self.runtime_element_layout
        ).store(self.ptr.offset(offset))

    @always_inline
    fn __setitem__(self, d0: Int, d1: Int, d2: Int, val: Self.element_type):
        """Set the element of the tensor with a specified index and value.

        Args:
            d0: The first dimensional index.
            d1: The second dimensional index.
            d2: The third dimensional index.
            val: The value writing to the tensor.
        """

        var strides = self.runtime_layout.stride.value
        var offset = Self._getOffset(strides, VariadicList[Int](d0, d1, d2))

        Element[dtype, Self.element_layout](
            val, self.runtime_element_layout
        ).store(self.ptr.offset(offset))

    @always_inline
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
        var offset = Self._getOffset(strides, VariadicList[Int](d0, d1, d2, d3))

        Element[dtype, Self.element_layout](
            val, self.runtime_element_layout
        ).store(self.ptr.offset(offset))

    @always_inline
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

    @always_inline
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

    @always_inline
    fn store[width: Int](self, m: Int, n: Int, val: SIMD[dtype, width]):
        """Store a value to a specified location.

        Parameters:
            width: The simd width of the stored value.

        Args:
            m: The m dimensional index to the tensor.
            n: The n dimensional index to the tensor.
            val: The value to be stored.
        """

        return self.ptr.store[width=width](self._offset(m, n), val)

    @always_inline
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
        return self.ptr.store[width=width, alignment=alignment](
            self._offset(m, n), val
        )

    @staticmethod
    @always_inline("nodebug")
    fn stack_allocation[*, alignment: Int = alignof[dtype]()]() -> Self:
        """Allocates stack memory for a LayoutTensor. Expects layout to be
        fully static.

        Parameters:
            alignment: The memory alignment of the underlying pointer.
        """

        constrained[layout.all_dims_known(), "Requires fully static layout"]()
        var ptr = stack_allocation[
            layout.size(),
            dtype,
            alignment=alignment,
            address_space=address_space,
        ]()

        return ptr

    @staticmethod
    @always_inline("nodebug")
    fn _toStatic[t: IntTuple]() -> StaticIntTuple[len(t)]:
        var st = StaticIntTuple[len(t)]()

        @parameter
        for i in range(len(t)):
            st[i] = to_int(t[i])
        return st

    @staticmethod
    @always_inline("nodebug")
    fn _getOffset[
        rank: Int
    ](stride: StaticIntTuple[rank], vals: VariadicList[Int]) -> Int:
        var offset = 0

        @parameter
        for i in range(rank):
            offset += vals[i] * stride[i]
        return offset

    @staticmethod
    @always_inline("nodebug")
    fn _getOffset[
        rank_1: Int, rank_2: Int
    ](stride: StaticIntTuple[rank_1], vals: StaticIntTuple[rank_2]) -> Int:
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

        alias shape = Self._toStatic[layout.shape]()
        return shape[idx]

    @always_inline
    @staticmethod
    fn stride[idx: Int]() -> Int:
        """Returns the stride of the tensor given a index.

        Parameters:
            idx: The index to the stride of the tensor.
        """

        alias stride = Self._toStatic[layout.stride]()
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
    ) -> LayoutTensor[
        dtype,
        coalesce(layout),
        address_space=address_space,
        element_layout = self.element_layout,
    ] as out:
        """Returns a LayoutTensor with a coalesced Layout."""

        return __type_of(out)(self.ptr)

    @staticmethod
    fn _compute_tile_layout[*tile_sizes: Int]() -> Layout:
        alias tiles = Self._divide_tiles[tile_sizes]()

        @parameter
        if __experimental_non_homogeneous_tile:
            # override the tile shapes with original unknown dim
            return Self._prop_unknown_shape[0, tiles, layout]()

        return tiles

    @staticmethod
    fn _divide_tiles[*tile_sizes: Int]() -> Layout:
        alias tiler = MakeTileLayoutList[tile_sizes]()
        return zipped_divide(layout, tiler)

    @staticmethod
    @always_inline
    fn _prop_unknown_shape[idx: Int, src: Layout, target: Layout]() -> Layout:
        """Propagate all unknown dim from target to a new layout."""
        var new_shape = src.shape
        new_shape[idx] = propagate_unknown(src.shape[idx], target.shape)
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
    ](self, *tile_coords: Int) -> LayoutTensor[
        dtype,
        Self._compute_tile_layout[tile_sizes]()[0],
        address_space=address_space,
        __experimental_non_homogeneous_tile = self.__experimental_non_homogeneous_tile,
    ] as result:
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

        alias num_tiles = __get_len[tile_sizes]()

        # need to calculate this again because __tiled_layout[1] is required for the offset calculation
        alias __tiled_layout = Self._compute_tile_layout[tile_sizes]()

        constrained[
            __tiled_layout[1].rank() == num_tiles,
            "Number of tiles should match the rank",
        ]()

        # Static layout tiling
        # TODO: Consider merge the two cases in away that won't slowdown the fully static layout.
        @parameter
        if result.layout.all_dims_known():
            var offset = 0

            @parameter
            for i in range(num_tiles):
                alias stride = to_int(__tiled_layout[1].stride[i])
                offset += tile_coords[i] * stride

            # Update offset to account for tile coords.
            var org_coords_offset = self.org_coords_offset

            @parameter
            for i in range(rank):
                org_coords_offset[i] += tile_sizes[i] * tile_coords[i]

            return __type_of(result)(
                self.ptr.offset(offset),
                org_coords_offset=rebind[StaticIntTuple[result.layout.rank()]](
                    org_coords_offset
                ),
            )

        else:
            # Dynamic layout, use strides
            var offset = 0

            @parameter
            if __experimental_non_homogeneous_tile:
                dynamic_shape = RuntimeTuple[result.layout.shape](
                    self._clamp_tile[tile_sizes](tile_coords)
                )
            else:
                dynamic_shape = RuntimeTuple[result.layout.shape]()

            var dynamic_stride = RuntimeTuple[result.layout.stride]()

            @parameter
            for i in range(num_tiles):
                var stride = self.runtime_layout.stride.value[i] * tile_sizes[i]
                dynamic_stride.value[i] = self.runtime_layout.stride.value[i]
                offset += tile_coords[i] * stride

            return __type_of(result)(
                self.ptr.offset(offset),
                RuntimeLayout(dynamic_shape, dynamic_stride),
            )

    @always_inline
    fn _clamp_tile[
        *tile_sizes: Int
    ](self, tile_coords: StaticIntTuple[rank]) -> StaticIntTuple[rank]:
        var tile_shape = StaticIntTuple[rank]()
        var runtime_shape = self.runtime_layout.shape

        @parameter
        for i in range(__get_len[tile_sizes]()):
            var cur_dim = runtime_shape[i].get_int() - (
                tile_coords[i] * tile_sizes[i]
            )
            tile_shape[i] = min(tile_sizes[i], cur_dim)

        return tile_shape

    @always_inline
    fn tiled_iterator[
        *tile_sizes: Int,
        axis: Int = 0,
    ](self, *tile_coords: Int) -> LayoutTensorIter[
        dtype,
        Self._compute_tile_layout[tile_sizes]()[0],
        address_space,
        circular=False,
    ] as result:
        """Returns the tiled iterator of the LayoutTensor.

        Parameters:
            tile_sizes: Tile sizes of each tile the iterator will iterate through.
            axis: Axis of the LayoutTensor the iterator will iterate through.

        Args:
            tile_coords: The tile coordinate that the iterator will point to.
        """

        alias tiles_rank = __get_len[tile_sizes]()
        alias __tiled_layout = Self._compute_tile_layout[tile_sizes]()
        constrained[
            __tiled_layout[1].rank() == tiles_rank,
            "Number of tiles should match the rank",
        ]()

        constrained[
            layout.shape[axis].is_value(),
            "The layout in the input axis can't be a tuple",
        ]()

        var ptr_offset = 0

        @parameter
        if layout.all_dims_known():

            @parameter
            for i in range(tiles_rank):
                alias stride = to_int(__tiled_layout[1].stride[i])
                ptr_offset += tile_coords[i] * stride

            # fmt: off
            alias bound = layout.shape[axis].value() * layout.stride[axis].value()
            alias stride = __tiled_layout[1].stride[axis].value()
            # fmt: on

            return __type_of(result)(
                self.ptr + ptr_offset, bound, stride=stride
            )

        else:
            var runtime_shape = RuntimeTuple[result.layout.shape]()
            var runtime_stride = RuntimeTuple[result.layout.stride]()

            @parameter
            for i in range(tiles_rank):
                var stride = self.runtime_layout.stride.value[i] * tile_sizes[i]
                runtime_stride.value[i] = self.runtime_layout.stride.value[i]
                ptr_offset += tile_coords[i] * stride

            var axis_dim = self.runtime_layout.shape.value[axis]
            var axis_stride = self.runtime_layout.stride.value[axis]
            var iter_bound = axis_dim * axis_stride
            var iter_stride = tile_sizes[axis] * axis_stride

            return __type_of(result)(
                self.ptr + ptr_offset,
                iter_bound,
                stride=iter_stride,
                offset=0,
                runtime_layout=RuntimeLayout(runtime_shape, runtime_stride),
            )

    @always_inline
    fn split[
        count: Int,
        axis: Int = 0,
    ](self) -> StaticTuple[
        LayoutTensor[
            dtype,
            Self._compute_tile_layout[
                layout.shape[axis].value() // count, axis
            ]()[0],
            address_space=address_space,
            element_layout=element_layout,
        ],
        count,
    ] as result:
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
                address_space=address_space,
                element_layout=element_layout,
            ](self.ptr.offset(i * tile_size * stride))

        return tiles

    @always_inline
    fn split[
        axis: Int = 0,
    ](self, count: Int, idx: Int) -> LayoutTensor[
        dtype,
        layout.make_shape_unknown[axis](),
        address_space=address_space,
        element_layout=element_layout,
    ] as result:
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

        var runtime_shape = RuntimeTuple[result.layout.shape]()

        @parameter
        for i in range(flatten_rank):

            @parameter
            if i == axis_in_flatten_tuple:
                runtime_shape.value[i] = axis_dim // count
            else:
                runtime_shape.value[i] = self.runtime_layout.shape.value[i]

        return __type_of(result)(
            self.ptr + idx * (axis_dim // count) * axis_stride,
            RuntimeLayout[result.layout](
                runtime_shape,
                rebind[RuntimeTuple[result.layout.stride]](
                    self.runtime_layout.stride
                ),
            ),
        )

    @always_inline
    fn _clamp_distribute_shape[
        thread_layout: Layout
    ](self, thread_id: UInt) -> StaticIntTuple[rank]:
        constrained[
            len(flatten(thread_layout.shape)) <= 2
            and len(flatten(thread_layout.stride)) <= 2,
            "Only supporting rank-2 or less thread layout for dynamic tile.",
        ]()

        # clamp staticinttuple using thread_id and thread_layout
        var tile_shape = StaticIntTuple[rank]()
        var runtime_shape = self.runtime_layout.shape
        alias thread_shape = thread_layout.shape
        alias thread_stride = thread_layout.stride

        # this would only work for rank-2 thread layout, need to extend this
        # to support thread layout such as Layout((2, 2), 2)
        @parameter
        for i in range(rank):
            alias thread_stride_i = to_int(thread_stride[i])
            alias thread_shape_i = to_int(thread_shape[i])
            var tile_idx = (thread_id // thread_stride_i) % thread_shape_i
            var runtime_dim = runtime_shape[i].get_int()
            var tile_shape_i = ceildiv(runtime_dim, thread_shape_i)
            var bound_i = (tile_shape_i - 1) * thread_shape_i + tile_idx
            tile_shape[i] = tile_shape_i if bound_i < runtime_dim else (
                runtime_dim // thread_shape_i
            )

        return tile_shape

    @always_inline
    fn distribute[
        threads_layout: Layout,
        axis: OptionalReg[Int] = None,
        swizzle: OptionalReg[Swizzle] = None,
        submode_axis: OptionalReg[Int] = None,
    ](self, thread_id: UInt) -> LayoutTensor[
        dtype,
        _compute_distribute_layout[
            layout,
            threads_layout,
            axis,
            __experimental_non_homogeneous_tile = self.__experimental_non_homogeneous_tile,
        ]()[1],
        address_space=address_space,
        element_layout=element_layout,
        __experimental_non_homogeneous_tile = self.__experimental_non_homogeneous_tile,
    ] as result:
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
            __experimental_non_homogeneous_tile = self.__experimental_non_homogeneous_tile,
        ]()

        alias coalesce_thread_layout = coalesce(threads_layout, keep_rank=True)

        alias res_rank = result.layout.rank()

        # Update org_coords offset and stride according to thread_id.
        var org_coords_offset = StaticIntTuple[res_rank]()
        var org_coords_stride = StaticIntTuple[res_rank]()

        @parameter
        for i in range(res_rank):
            alias stride_i: UInt = to_int(
                flatten(coalesce_thread_layout.stride)[axis.value()]
            ) if axis else to_int(flatten(coalesce_thread_layout.stride)[i])
            alias shape_i: UInt = to_int(
                flatten(coalesce_thread_layout.shape)[axis.value()]
            ) if axis else to_int(flatten(coalesce_thread_layout.shape)[i])
            var thread_corrrds_i: UInt = (thread_id // stride_i) % shape_i
            org_coords_offset[i] = thread_corrrds_i + self.org_coords_offset[i]
            org_coords_stride[i] = self.org_coords_stride[i] * shape_i

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
                alias fragments_stride_i: UInt = to_int(
                    fragments_layout_stride[i]
                ).value
                alias shape_i: UInt = to_int(thread_projected_shape[i]).value
                alias stride_i: UInt = to_int(thread_projected_stride[i]).value
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

            return __type_of(result)(
                self.ptr.offset(int(swizzled_offset)),
                org_coords_offset=rebind[StaticIntTuple[result.layout.rank()]](
                    org_coords_offset
                ),
                org_coords_stride=rebind[StaticIntTuple[result.layout.rank()]](
                    org_coords_stride
                ),
            )

        else:

            @parameter
            if not __experimental_non_homogeneous_tile:
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
            if __experimental_non_homogeneous_tile:
                runtime_shape = RuntimeTuple[result.layout.shape](
                    self._clamp_distribute_shape[threads_layout](thread_id)
                )
            else:
                runtime_shape = RuntimeTuple[result.layout.shape]()

            var runtime_stride = RuntimeTuple[result.layout.stride]()

            @parameter
            for i in range(runtime_shape.scalar_length):
                alias thread_shape_i = threads_layout[i].size()
                runtime_stride.value[i] = (
                    self.runtime_layout.stride.value[i] * thread_shape_i
                )

            @parameter
            for i in range(len(flatten(Self.layout.stride))):
                var fragments_stride_i = self.runtime_layout.stride.value[i]
                alias shape_i: UInt = to_int(thread_projected_shape[i]).value
                alias stride_i: UInt = to_int(thread_projected_stride[i]).value
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
                    self.ptr.offset(int(swizzled_offset)),
                    RuntimeLayout(runtime_shape, runtime_stride),
                    org_coords_offset=rebind[
                        StaticIntTuple[result.layout.rank()]
                    ](org_coords_offset),
                    org_coords_stride=rebind[
                        StaticIntTuple[result.layout.rank()]
                    ](org_coords_stride),
                )
            else:
                return __type_of(result)(
                    self.ptr.offset(int(swizzled_offset)),
                    RuntimeLayout(runtime_shape, runtime_stride),
                    self.runtime_element_layout,
                    org_coords_offset=rebind[
                        StaticIntTuple[result.layout.rank()]
                    ](org_coords_offset),
                    org_coords_stride=rebind[
                        StaticIntTuple[result.layout.rank()]
                    ](org_coords_stride),
                )

    # Returns the original coordiantes a specific tensor element at `idx`.
    @always_inline
    fn element_coords[idx: Int](self) -> StaticIntTuple[rank]:
        constrained[
            layout.known_shape(),
            "element_coords only support layouts of know shape",
        ]()
        alias layout_coords = Layout(Self.layout.shape)
        alias coords = Self._toStatic[layout_coords.idx2crd(idx)]()
        return (
            self.org_coords_offset
            + rebind[StaticIntTuple[rank]](coords) * self.org_coords_stride
        )

    @always_inline
    fn vectorize[
        *vector_shape: Int
    ](self) -> LayoutTensor[
        dtype,
        coalesce(Self._compute_tile_layout[vector_shape]()[1], keep_rank=True),
        address_space=address_space,
        element_layout = Self._divide_tiles[vector_shape]()[0],
        __experimental_non_homogeneous_tile = self.__experimental_non_homogeneous_tile,
    ] as result:
        # Update element stride to account for vector shapes.
        var org_coords_stride = StaticIntTuple[rank]()

        @parameter
        @always_inline
        fn __check_vector_shape[*vec_shape: Int]():
            @parameter
            for i in range(__get_len[vec_shape]()):
                alias shape_i = to_int(self.layout.shape[i])

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

        @parameter
        if __experimental_non_homogeneous_tile:
            __check_vector_shape[vector_shape]()

        @parameter
        for i in range(rank):
            org_coords_stride[i] = vector_shape[i]

        @parameter
        if layout.all_dims_known():
            return __type_of(result)(
                self.ptr,
                org_coords_offset=rebind[StaticIntTuple[result.layout.rank()]](
                    self.org_coords_offset
                ),
                org_coords_stride=rebind[StaticIntTuple[result.layout.rank()]](
                    org_coords_stride
                ),
            )
        else:
            constrained[
                coalesce(result.element_layout).known_shape(),
                "Result element layout should have known shape",
            ]()
            var runtime_shape = RuntimeTuple[result.layout.shape]()
            var runtime_stride = RuntimeTuple[result.layout.stride]()

            var runtime_element_layout_shape = RuntimeTuple[
                result.element_layout.shape
            ]()
            var runtime_element_layout_stride = RuntimeTuple[
                result.element_layout.stride
            ](self.runtime_layout.stride.value)

            @parameter
            for i in range(runtime_shape.scalar_length):
                runtime_shape.value[i] = (
                    self.runtime_layout.shape.value[i] // vector_shape[i]
                )
                runtime_stride.value[i] = (
                    self.runtime_layout.stride.value[i] * vector_shape[i]
                )

            return __type_of(result)(
                self.ptr,
                RuntimeLayout(runtime_shape, runtime_stride),
                rebind[RuntimeLayout[result.element_layout]](
                    RuntimeLayout(
                        runtime_element_layout_shape,
                        runtime_element_layout_stride,
                    )
                ),
                org_coords_offset=rebind[StaticIntTuple[result.layout.rank()]](
                    self.org_coords_offset
                ),
            )

    @staticmethod
    fn __compute_slice_layout(d0_slice: Slice, d1_slice: Slice) -> Layout:
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
    fn __compute_slice_layout(
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
    fn __compute_slice_layout(slice_0: Slice, slice_0_axis: Int) -> Layout:
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
    ](self) -> LayoutTensor[
        dtype,
        Self.__compute_slice_layout(
            d0_slice,
            d1_slice,
        ),
        address_space=address_space,
        element_layout=element_layout,
    ] as result:
        constrained[
            d0_slice.step == 1 and d1_slice.step == 1,
            "Slice should have no gaps",
        ]()
        alias stride_m = to_int(result.layout.stride[0])
        alias stride_n = to_int(result.layout.stride[1])

        alias d0_slice_start = d0_slice.start.or_else(0)
        alias d1_slice_start = d1_slice.start.or_else(0)

        var offset = d0_slice_start * stride_m + d1_slice_start * stride_n

        return __type_of(result)(self.ptr.offset(offset))

    @always_inline
    fn slice[
        d0_slice: Slice,
        d1_slice: Slice,
        slice_indices: StaticIntTuple[2],
        __offset_dims: Int = Self.rank - 2,
    ](
        self,
        offsets: StaticIntTuple[__offset_dims],
    ) -> LayoutTensor[
        dtype,
        Self.__compute_slice_layout(
            d0_slice, d1_slice, slice_indices[0], slice_indices[1]
        ),
        address_space=address_space,
        element_layout=element_layout,
    ] as result:
        constrained[
            d0_slice.step == 1 and d1_slice.step == 1,
            "Slice should have no gaps",
        ]()
        constrained[
            slice_indices[0] < slice_indices[1],
            "Slice indices should be ordered",
        ]()
        alias stride_0 = to_int(result.layout.stride[0])
        alias stride_1 = to_int(result.layout.stride[1])

        alias d0_slice_start = d0_slice.start.or_else(0)
        alias d1_slice_start = d1_slice.start.or_else(0)

        var slice_offset = d0_slice_start * stride_0 + d1_slice_start * stride_1

        var idx = 0

        @parameter
        for i in range(Self.rank):
            alias stride_i = to_int(Self.layout.stride[i])

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
        slice_indices: StaticIntTuple[1],
        __offset_dims: Int = Self.rank - 1,
    ](
        self,
        offsets: StaticIntTuple[__offset_dims],
    ) -> LayoutTensor[
        dtype,
        Self.__compute_slice_layout(d0_slice, slice_indices[0]),
        address_space=address_space,
        element_layout=element_layout,
    ] as result:
        constrained[
            d0_slice.step == 1,
            "Slice should have no gaps",
        ]()

        alias stride_0 = to_int(result.layout.stride[0])

        alias d0_slice_start = d0_slice.start.or_else(0)

        var slice_offset = d0_slice_start * stride_0

        var idx = 0

        @parameter
        for i in range(Self.rank):
            alias stride_i = to_int(Self.layout.stride[i])

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
    ](self) -> LayoutTensor[
        dtype,
        composition(
            layout,
            Layout(IntTuple(N, M), IntTuple(M, 1)),
        ),
        address_space=address_space,
        element_layout=element_layout,
    ] as result:
        return __type_of(result)(self.ptr)

    @always_inline
    fn reshape[
        dst_layout: Layout,
    ](self) -> LayoutTensor[
        dtype,
        dst_layout,
        address_space=address_space,
        element_layout=element_layout,
    ] as result:
        return __type_of(result)(self.ptr)

    @always_inline
    fn composition[
        rhs_layout: Layout,
        dst_layout: Layout = composition(layout, rhs_layout),
    ](self) -> LayoutTensor[
        dtype,
        dst_layout,
        address_space=address_space,
        element_layout=element_layout,
    ] as result:
        return __type_of(result)(self.ptr)

    @always_inline
    fn distance(
        self, addr: UnsafePointer[Scalar[dtype], address_space, *_]
    ) -> UInt:
        """Returns the distance from the input address."""

        return UInt(int(self.ptr) - int(addr)) // sizeof[dtype]()

    @always_inline
    fn distance[
        _layout: Layout  # see MOCO-1089
    ](
        self, src: LayoutTensor[dtype, _layout, address_space=address_space]
    ) -> UInt:
        """Returns the distance from the input address."""

        return UInt(int(self.ptr) - int(src.ptr)) // sizeof[dtype]()

    # Returns the linear index of an elem_i 0 ... size(layout).
    #
    @always_inline
    fn __get_element_idx[elem_i: Int](self) -> Int:
        alias element_size = int(self.element_size)

        @parameter
        if layout.all_dims_known():
            alias idx = make_layout(element_layout, layout)(
                elem_i * element_size
            )
            return idx
        else:
            var idx = make_runtime_layout(
                self.runtime_element_layout, self.runtime_layout
            )(elem_i * element_size)
            return idx

    @always_inline
    fn copy_from[
        dst_coords_bound: OptionalReg[StaticIntTuple[rank]] = None,
        src_coords_bound: OptionalReg[StaticIntTuple[rank]] = None,
    ](self, other: LayoutTensor):
        alias other_layout = other.layout

        alias dst_element_size = int(self.element_size)
        alias src_element_size = int(other.element_size)

        alias dst_size = layout.size()
        alias src_size = other_layout.size()

        @parameter
        if not __experimental_non_homogeneous_tile:
            constrained[
                layout.known_shape() and other_layout.known_shape(),
                "copy_from must move data of statically known shape",
            ]()

            constrained[
                dst_size == src_size,
                "copy_from should move data of the same size",
            ]()
        else:
            alias is_rank2 = self.rank == 2 and other.rank == 2
            constrained[
                is_rank2,
                (
                    "non homogeneous tile copy is only available for rank-2"
                    " tensor."
                ),
            ]()

        constrained[
            dst_element_size == src_element_size, "copy_from should move"
        ]()

        alias has_copy_bounds = dst_coords_bound or src_coords_bound

        @parameter
        @always_inline
        fn __is_in_bound[
            rank: Int
        ](coords: StaticIntTuple[rank], bounds: StaticIntTuple[rank]) -> Bool:
            var in_bound = True

            @parameter
            for dim in range(rank):
                in_bound &= coords[dim] < bounds[dim]
            return in_bound

        @parameter
        @always_inline
        fn __compute_element_bound[
            element_layout: Layout
        ](
            coords: StaticIntTuple[rank], bounds: StaticIntTuple[rank]
        ) -> StaticIntTuple[rank]:
            var element_bound = StaticIntTuple[rank]()

            @parameter
            for dim in range(rank):
                alias dim_size = to_int(element_layout.shape[dim])
                element_bound[dim] = (
                    min(dim_size, bounds[dim] - coords[dim]) if coords[dim]
                    < bounds[dim] else 0
                )
            return element_bound

        @parameter
        @always_inline
        fn __load_element[i: Int]() -> Element[dtype, other.element_layout]:
            var src_idx = other.__get_element_idx[i]()

            @parameter
            if src_element_size != 1 and src_coords_bound.__bool__():
                var element_bounds = __compute_element_bound[
                    other.element_layout
                ](
                    rebind[StaticIntTuple[rank]](other.element_coords[i]()),
                    src_coords_bound.value(),
                )
                return Element[dtype, other.element_layout].masked_load[
                    other.address_space
                ](
                    rebind[UnsafePointer[Scalar[dtype], other.address_space]](
                        other.ptr
                    ).offset(src_idx),
                    element_bounds,
                    other.runtime_element_layout,
                )

            return Element[dtype, other.element_layout].load[
                other.address_space
            ](
                rebind[UnsafePointer[Scalar[dtype], other.address_space]](
                    other.ptr
                ).offset(src_idx),
                other.runtime_element_layout,
            )

        @parameter
        @always_inline
        fn __store_element[
            i: Int
        ](src_element: Element[dtype, other.element_layout]):
            var dst_idx = self.__get_element_idx[i]()

            @parameter
            if dst_element_size != 1 and dst_coords_bound.__bool__():
                var element_bounds = __compute_element_bound[
                    self.element_layout
                ](self.element_coords[i](), dst_coords_bound.value())
                Element[dtype, self.element_layout](
                    rebind[
                        Element[dtype, self.element_layout].element_data_type
                    ](src_element.element_data)
                ).masked_store(self.ptr.offset(dst_idx), element_bounds)
            else:
                Element[dtype, self.element_layout](
                    rebind[
                        Element[dtype, self.element_layout].element_data_type
                    ](src_element.element_data)
                ).store(self.ptr.offset(dst_idx))

        @parameter
        if not __experimental_non_homogeneous_tile:

            @parameter
            for i in range(dst_size):

                @parameter
                if has_copy_bounds:

                    @parameter
                    if src_coords_bound.__bool__() and dst_element_size == 1:
                        if not __is_in_bound(
                            self.element_coords[i](), dst_coords_bound.value()
                        ):
                            continue

                    @parameter
                    if dst_coords_bound.__bool__() and src_element_size == 1:
                        if not __is_in_bound(
                            other.element_coords[i](),
                            rebind[StaticIntTuple[other.rank]](
                                dst_coords_bound.value()
                            ),
                        ):
                            continue

                __store_element[i](__load_element[i]())

        else:
            var d0 = min(other.dim(0), self.dim(0))
            var d1 = min(other.dim(1), self.dim(1))

            var dst_layout = RuntimeLayout[self.layout](
                RuntimeTuple[self.layout.shape](d0, d1),
                self.runtime_layout.stride,
            )
            var src_layout = RuntimeLayout[other.layout](
                RuntimeTuple[other.layout.shape](d0, d1),
                other.runtime_layout.stride,
            )

            var dst_element_size = self.runtime_element_layout.size()
            var src_element_size = other.runtime_element_layout.size()

            for i in range(d0 * d1):
                var dst_idx = make_runtime_layout(
                    self.runtime_element_layout, dst_layout
                )(i * dst_element_size)

                var src_idx = make_runtime_layout(
                    other.runtime_element_layout, src_layout
                )(i * src_element_size)

                var src_element = Element[dtype, other.element_layout].load[
                    other.address_space
                ](
                    rebind[UnsafePointer[Scalar[dtype], other.address_space]](
                        other.ptr
                    ).offset(src_idx),
                    other.runtime_element_layout,
                )

                alias dst_element_type = Element[dtype, self.element_layout]
                dst_element_type(
                    rebind[dst_element_type.element_data_type](
                        src_element.element_data
                    )
                ).store(self.ptr.offset(dst_idx))

    @always_inline
    fn copy_from(
        self,
        other: LayoutTensor,
        offset: Int,
        rows: Int,
        cols: Int,
    ):
        alias other_layout = other.layout

        alias dst_size = layout.size()
        alias src_size = other_layout.size()

        alias dst_element_size = int(self.element_size)
        alias src_element_size = int(other.element_size)

        constrained[
            layout.known_shape() and other_layout.known_shape(),
            "copy_from must move data of statically known shape",
        ]()

        constrained[
            dst_size == src_size, "copy_from should move data of the same size"
        ]()

        constrained[
            dst_element_size == src_element_size, "copy_from should move"
        ]()

        @parameter
        for i in range(dst_size):
            alias src_idx = other_layout(i)
            alias dst_static_idx = self.layout(i)

            var dst_idx = 0

            @parameter
            if self.layout.all_dims_known():
                dst_idx = dst_static_idx
            else:
                dst_idx = self.runtime_layout(i)

            if offset + dst_idx < rows * cols:
                var src_element = Element[dtype, other.element_layout].load[
                    other.address_space
                ](
                    rebind[UnsafePointer[Scalar[dtype], other.address_space]](
                        other.ptr
                    ).offset(src_idx),
                    other.runtime_element_layout,
                )
                alias dst_element_type = Element[dtype, self.element_layout]
                dst_element_type(
                    rebind[dst_element_type.element_data_type](
                        src_element.element_data
                    )
                ).store(self.ptr.offset(dst_idx))

    @always_inline
    fn copy_from_async[
        src_layout: Layout,
        src_addr_space: AddressSpace,
        src_element_layout: Layout,
        *,
        masked: Bool = False,
        swizzle: OptionalReg[Swizzle] = None,
        fill: Fill = Fill.NONE,
    ](
        self,
        src: LayoutTensor[
            dtype,
            src_layout,
            address_space=src_addr_space,
            element_layout=src_element_layout,
        ],
        src_idx_bound: Int = UNKNOWN_VALUE,
        base_offset: Int = UNKNOWN_VALUE,
    ):
        constrained[
            self.address_space == _GPUAddressSpace.SHARED,
            "Async is only supported for destinations in shared memory",
        ]()

        alias dst_size = layout.size()
        alias src_size = src_layout.size()
        constrained[
            dst_size == src_size,
            "copy_from_async should move data of the same size",
        ]()

        alias dst_element_size = int(self.element_size)
        alias src_element_size = int(src.element_size)
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
            src_layout.all_dims_known() and src.element_layout.all_dims_known()
        )

        var dst_ptr = self.ptr.bitcast[
            address_space = _GPUAddressSpace.SHARED
        ]()
        var src_ptr = src.ptr.bitcast[address_space = _GPUAddressSpace.GLOBAL]()

        # Coalesce element layouts to simplify vectorization condition.
        alias coalesce_src_element_layout = coalesce(src_element_layout)
        alias coalesce_dst_element_layout = coalesce(self.element_layout)

        @parameter
        if (
            src_element_layout.all_dims_known()
            and coalesce_src_element_layout.rank() == 1
            and coalesce_src_element_layout.stride[0] == 1
            and coalesce_dst_element_layout.rank() == 1
            and coalesce_dst_element_layout.stride[0] == 1
        ):
            # Swizzle does `^bits`. Every 2^bits rows is xor-ed with the same binary number.
            # For vectors within 2^bits, we swizzle each of them.
            alias num_vecs_per_swizzle = ceildiv(
                swizzle.value().size(),
                layout.stride[0].value() // self.element_size,
            ) if swizzle.__bool__() and layout.size() > 1 else 1

            @parameter
            for j in range(num_vecs_per_swizzle):
                alias dst_offset = layout(j)
                var swizzled_offset = dst_offset

                # Swizzle each vector within the first 2^bits rows.
                @parameter
                if swizzle:
                    alias swizzle_fn = swizzle.value()
                    swizzled_offset = (
                        swizzle_fn(
                            (base_offset + dst_offset) // self.element_size
                        )
                        * self.element_size
                        - base_offset
                    )

                # Group vectors that are more than (multiple of) 2^bits rows part.
                # The vector's idx is swizzled_offset + distance to the first vector
                # in the group, which is within the first 2^bits rows.
                @parameter
                for i in range(j, dst_size, num_vecs_per_swizzle):
                    var src_idx = 0

                    # only get index like this when it is vectorized form
                    alias src_static_idx = src_layout(i)
                    alias dst_distance = layout(i) - layout(j)

                    @parameter
                    if src_dims_known:
                        src_idx = src_static_idx
                    else:
                        src_idx = src.runtime_layout(i)

                    var dst_idx = swizzled_offset + dst_distance

                    @parameter
                    if masked:
                        var src_copy_size = element_size_bytes if src_idx < src_idx_bound else 0
                        async_copy[element_size_bytes, fill=fill](
                            src_ptr + src_idx, dst_ptr + dst_idx, src_copy_size
                        )
                    else:
                        async_copy[element_size_bytes, fill=fill](
                            src_ptr + src_idx, dst_ptr + dst_idx
                        )

        # Async copy should only be used for 16B vector for bypassing L1.
        # Scalar path is only for kernel tests.
        else:
            constrained[not swizzle, "Should not swizzle scalar copy."]()

            @parameter
            for i in range(dst_size * dst_element_size):
                var src_idx = 0

                alias src_static_idx = make_layout(
                    src.element_layout, src_layout
                )(i)
                alias dst_idx = make_layout(self.element_layout, self.layout)(i)

                @parameter
                if src_dims_known:
                    src_idx = src_static_idx
                else:
                    src_idx = make_runtime_layout(
                        src.runtime_element_layout, src.runtime_layout
                    )(i)

                async_copy[4, fill=fill](src_ptr + src_idx, dst_ptr + dst_idx)

    @always_inline
    fn copy_from_async_masked_src[
        src_layout: Layout,
        src_addr_space: AddressSpace,
        src_element_layout: Layout,
        *,
        fill: Fill = Fill.NONE,
        swizzle: OptionalReg[Swizzle] = None,
    ](
        self,
        src: LayoutTensor[
            dtype,
            src_layout,
            address_space=src_addr_space,
            element_layout=src_element_layout,
        ],
        offset: Int,
        rows: Int,
        cols: Int,
        base_offset: Int = UNKNOWN_VALUE,
    ):
        constrained[
            self.address_space == _GPUAddressSpace.SHARED,
            "Async is only supported for destinations in shared memory",
        ]()

        alias dst_size = layout.size()
        alias src_size = src_layout.size()
        constrained[
            dst_size == src_size,
            "copy_from_async should move data of the same size",
        ]()

        alias dst_element_size = int(self.element_size)
        alias src_element_size = int(src.element_size)
        constrained[
            dst_element_size == src_element_size,
            "copy_from_async should move data of the same element size",
        ]()

        # Share memory must always have static layout.
        alias dst_dims_known = (
            self.layout.all_dims_known()
            and self.element_layout.all_dims_known()
        )
        constrained[dst_dims_known, "dst tensor must have static layout"]()

        # Eligibility for 4, 8, 16 bytes async load.
        alias element_size_bytes = sizeof[dtype]() * src_element_size
        constrained[
            element_size_bytes == 4
            or element_size_bytes == 8
            or element_size_bytes == 16,
            "copy_from_async only allows 4, 8, 16 bytes element",
        ]()

        var dst_ptr = self.ptr.bitcast[
            address_space = _GPUAddressSpace.SHARED
        ]()
        var src_ptr = src.ptr.bitcast[address_space = _GPUAddressSpace.GLOBAL]()

        # Coalesce element layouts to simplify vectorization condition.
        alias coalesce_src_element_layout = coalesce(src_element_layout)
        alias coalesce_dst_element_layout = coalesce(self.element_layout)

        @parameter
        if (
            src_element_layout.all_dims_known()
            and coalesce_src_element_layout.rank() == 1
            and coalesce_src_element_layout.stride[0] == 1
            and coalesce_dst_element_layout.rank() == 1
            and coalesce_dst_element_layout.stride[0] == 1
        ):
            alias num_vecs_per_swizzle = ceildiv(
                swizzle.value().size(),
                layout.stride[0].value() // self.element_size,
            ) if swizzle.__bool__() and layout.size() > 1 else 1

            @parameter
            for j in range(num_vecs_per_swizzle):
                alias dst_offset = layout(j)
                var swizzled_offset = dst_offset

                # Swizzle each vector within the first 2^bits rows.
                @parameter
                if swizzle:
                    alias swizzle_fn = swizzle.value()
                    swizzled_offset = (
                        swizzle_fn(
                            (base_offset + dst_offset) // self.element_size
                        )
                        * self.element_size
                        - base_offset
                    )

                @parameter
                for i in range(j, dst_size, num_vecs_per_swizzle):
                    alias src_static_idx = src_layout(i)
                    var src_idx = 0

                    @parameter
                    if src_layout.all_dims_known():
                        src_idx = src_static_idx
                    else:
                        src_idx = src.runtime_layout(i)

                    alias dst_distance = layout(i) - layout(j)

                    var dst_idx = swizzled_offset + dst_distance

                    if offset + src_idx < rows * cols:
                        async_copy[element_size_bytes, fill=fill](
                            src_ptr + src_idx, dst_ptr + dst_idx
                        )
        else:

            @parameter
            for i in range(dst_size * dst_element_size):
                var src_idx = 0

                alias src_static_idx = make_layout(
                    src.element_layout, src_layout
                )(i)
                alias dst_idx = make_layout(self.element_layout, self.layout)(i)

                @parameter
                if src_layout.all_dims_known():
                    src_idx = src_static_idx
                else:
                    src_idx = make_runtime_layout(
                        src.runtime_element_layout, src.runtime_layout
                    )(i)

                if offset + src_idx < rows * cols:
                    async_copy[4, fill=fill](
                        src_ptr + src_idx, dst_ptr + dst_idx
                    )

    @always_inline
    fn fill(self, val: Scalar[dtype]) -> Self:
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
        return String.format_sequence(self)

    fn format_to(self, inout writer: Formatter):
        """Format 2D tensor in 2D, otherwise print all values in column major
        coordinate order."""

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
        if is_2d_print(layout) or is_2d_print(coalesce(layout)):
            var m_dim = self.runtime_layout.shape[0].value[0]
            var n_dim = self.runtime_layout.shape[1].value[0]
            for m in range(m_dim):
                for n in range(n_dim):
                    writer.write(self[m, n], " ")
                if m < m_dim - 1:
                    writer.write("\n")
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


fn stack_allocation_like[
    layout: Layout,
    dtype: DType,
    *,
    address_space: AddressSpace,
    target_address_space: AddressSpace = AddressSpace.GENERIC,
](
    in_tensor: LayoutTensor[dtype, layout, address_space=address_space]
) -> LayoutTensor[dtype, layout, address_space=target_address_space]:
    return LayoutTensor[
        dtype, layout, address_space=target_address_space
    ].stack_allocation()


# Synchronous copy from DRAM -> SRAM, this requires w/r thread affinity mapping.
#
@always_inline
fn copy_dram_to_sram[
    src_layout: Layout,
    dst_layout: Layout,
    dtype: DType,
    src_thread_layout: Layout,
    dst_thread_layout: Layout,
    src_element_layout: Layout,
    dst_element_layout: Layout,
    swizzle: OptionalReg[Swizzle] = None,
](
    dst: LayoutTensor[
        dtype,
        dst_layout,
        address_space = _GPUAddressSpace.SHARED,
        element_layout=dst_element_layout,
    ],
    src: LayoutTensor[
        dtype,
        src_layout,
        address_space = _GPUAddressSpace.GENERIC,
        element_layout=src_element_layout,
    ],
):
    var src_fragments = src.distribute[src_thread_layout](ThreadIdx.x())
    var dst_fragments = dst.distribute[dst_thread_layout, swizzle=swizzle](
        ThreadIdx.x()
    )
    dst_fragments.copy_from(src_fragments)


# Synchronous copy from DRAM -> SRAM, this requires w/r thread affinity mapping.
#
@always_inline
fn copy_dram_to_sram[
    src_layout: Layout,
    dst_layout: Layout,
    dtype: DType,
    thread_layout: Layout,
    src_element_layout: Layout,
    dst_element_layout: Layout,
    swizzle: OptionalReg[Swizzle] = None,
](
    dst: LayoutTensor[
        dtype,
        dst_layout,
        address_space = _GPUAddressSpace.SHARED,
        element_layout=dst_element_layout,
    ],
    src: LayoutTensor[
        dtype,
        src_layout,
        address_space = _GPUAddressSpace.GENERIC,
        element_layout=src_element_layout,
    ],
):
    copy_dram_to_sram[
        src_layout,
        dst_layout,
        dtype,
        thread_layout,
        thread_layout,
        src_element_layout,
        dst_element_layout,
        swizzle,
    ](dst, src)


# Asynchronous copy from DRAM -> SRAM, this requires w/r thread affinity mapping.
#
@always_inline
fn copy_dram_to_sram_async[
    src_layout: Layout,
    dst_layout: Layout,
    dtype: DType,
    src_thread_layout: Layout,
    dst_thread_layout: Layout,
    src_element_layout: Layout,
    dst_element_layout: Layout,
    swizzle: Bool = False,
    masked: Bool = False,
](
    dst: LayoutTensor[
        dtype,
        dst_layout,
        address_space = _GPUAddressSpace.SHARED,
        element_layout=dst_element_layout,
    ],
    src: LayoutTensor[
        dtype,
        src_layout,
        address_space = _GPUAddressSpace.GENERIC,
        element_layout=src_element_layout,
    ],
    num_rows: Int = UNKNOWN_VALUE,
):
    alias row_size = dst.stride[0]()
    # See make_ldmatrix_swizzle in Swizzle.mojo for `conflict_ways`.
    # TODO: use the above when MOCO-1048 is fixed.
    alias bytes_32_banks = 128
    alias conflict_ways = min(
        8 * row_size * sizeof[dtype]() // bytes_32_banks, 8
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
        OptionalReg[Swizzle](make_ldmatrix_swizzle[dtype, row_size]())
    )

    var src_fragments = src.distribute[src_thread_layout](ThreadIdx.x())
    var dst_fragments = dst.distribute[dst_thread_layout](ThreadIdx.x())

    var dst_frag_offset = dst_fragments.distance(dst.ptr) if swizzle else 0

    @parameter
    if not masked:
        dst_fragments.copy_from_async[swizzle=swizzle_option](
            src_fragments, base_offset=dst_frag_offset
        )
    else:
        constrained[
            src_layout.stride[1].value() == src.element_size
            and src_layout.rank() == 2,
            "Only support masking rows and 2D row major layout.",
        ]()
        var src_frag_offset = src_fragments.distance(src.ptr)
        alias stride = src_layout.stride[0].value()
        var src_idx_bound = num_rows * stride - src_frag_offset
        dst_fragments.copy_from_async[masked=True, swizzle=swizzle_option](
            src_fragments,
            src_idx_bound=src_idx_bound,
            base_offset=dst_frag_offset,
        )


# Asynchronous copy from DRAM -> SRAM, this requires w/r thread affinity mapping.
#
@always_inline
fn copy_dram_to_sram_async[
    src_layout: Layout,
    dst_layout: Layout,
    dtype: DType,
    thread_layout: Layout,
    src_element_layout: Layout,
    dst_element_layout: Layout,
    swizzle: Bool = False,
    masked: Bool = False,
](
    dst: LayoutTensor[
        dtype,
        dst_layout,
        address_space = _GPUAddressSpace.SHARED,
        element_layout=dst_element_layout,
    ],
    src: LayoutTensor[
        dtype,
        src_layout,
        address_space = _GPUAddressSpace.GENERIC,
        element_layout=src_element_layout,
    ],
    num_rows: Int = UNKNOWN_VALUE,
):
    copy_dram_to_sram_async[
        src_layout,
        dst_layout,
        dtype,
        thread_layout,
        thread_layout,
        src_element_layout,
        dst_element_layout,
        swizzle,
        masked,
    ](dst, src, num_rows)


@always_inline
fn copy_sram_to_dram[
    src_layout: Layout,
    dst_layout: Layout,
    src_type: DType,
    dst_type: DType,
    thread_layout: Layout,
    src_element_layout: Layout,
    dst_element_layout: Layout,
    swizzle: OptionalReg[Swizzle] = None,
](
    dst: LayoutTensor[
        dst_type,
        dst_layout,
        address_space = _GPUAddressSpace.GENERIC,
        element_layout=dst_element_layout,
    ],
    src: LayoutTensor[
        src_type,
        src_layout,
        address_space = _GPUAddressSpace.SHARED,
        element_layout=src_element_layout,
    ],
):
    constrained[
        src_layout.all_dims_known(), "Shared memory must have static layout"
    ]()

    var src_fragments = src.distribute[thread_layout](ThreadIdx.x())
    var dst_fragments = dst.distribute[thread_layout](ThreadIdx.x())

    # TODO: copy_from only allows static layout
    @parameter
    if src_type == dst_type:
        dst_fragments.copy_from(src_fragments)
    else:
        constrained[
            src_type == DType.float32 and dst_type.is_half_float(),
            "Only support FP32 -> half precision downcast during copy.",
        ]()

        alias simd_size = simdwidthof[dst_type]()
        # TODO: generalize the copy to non-scalar case if possible.
        constrained[
            src_element_layout.size() == simd_size
            and dst_element_layout.size() == simd_size,
            "Only FP32 -> half precision downcast for vectorized copy.",
        ]()

        alias num_stores_per_thread = dst_fragments.layout.size()
        alias src_align = alignof[SIMD[src_type, simdwidthof[src_type]()]]()
        alias dst_align = alignof[SIMD[dst_type, simd_size]]()

        var src_frag_offset = src_fragments.distance(src.ptr)

        @parameter
        for i in range(num_stores_per_thread):
            var dst_idx = 0

            alias src_idx = src_fragments.layout(i)
            alias dst_static_idx = dst_fragments.layout(i)

            @parameter
            if dst_fragments.layout.all_dims_known():
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
                    swizzle_fn(src_frag_offset + src_idx_base) + src_idx_diff
                )

            var src_vec = (src.ptr + swizzled_idx).load[
                width=simd_size, alignment=src_align
            ]()
            (dst_fragments.ptr + dst_idx).store[alignment=dst_align](
                src_vec.cast[dst_type]()
            )


@always_inline
fn copy_sram_to_dram[
    src_layout: Layout,
    dst_layout: Layout,
    src_type: DType,
    dst_type: DType,
    thread_layout: Layout,
    src_element_layout: Layout,
    dst_element_layout: Layout,
    swizzle: OptionalReg[Swizzle] = None,
](
    dst: LayoutTensor[
        dst_type,
        dst_layout,
        address_space = _GPUAddressSpace.GENERIC,
        element_layout=dst_element_layout,
    ],
    src: LayoutTensor[
        src_type,
        src_layout,
        address_space = _GPUAddressSpace.SHARED,
        element_layout=src_element_layout,
    ],
    offset: Int,
    rows: Int,
    cols: Int,
):
    var src_fragments = src.distribute[thread_layout](ThreadIdx.x())
    var dst_fragments = dst.distribute[thread_layout](ThreadIdx.x())
    var thread_offset = offset + dst_fragments.distance(dst.ptr)

    @parameter
    if src_type == dst_type:
        dst_fragments.copy_from(src_fragments)
    else:
        constrained[
            src_type == DType.float32 and dst_type.is_half_float(),
            "Only support FP32 -> half precision downcast during copy.",
        ]()

        alias simd_size = simdwidthof[dst_type]()
        # TODO: generalize the copy to non-scalar case if possible.
        constrained[
            src_element_layout.size() == simd_size
            and dst_element_layout.size() == simd_size,
            "Only FP32 -> half precision downcast for vectorized copy.",
        ]()

        alias num_stores_per_thread = dst_fragments.layout.size()
        alias src_align = alignof[SIMD[src_type, simdwidthof[src_type]()]]()
        alias dst_align = alignof[SIMD[dst_type, simd_size]]()

        var src_frag_offset = src_fragments.distance(src.ptr)

        @parameter
        for i in range(num_stores_per_thread):
            alias src_idx = src_fragments.layout(i)
            alias dst_static_idx = dst_fragments.layout(i)

            var dst_idx = 0

            @parameter
            if dst_layout.all_dims_known():
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
                    swizzle_fn(src_frag_offset + src_idx_base) + src_idx_diff
                )

            if thread_offset + dst_idx < rows * cols:
                var src_vec = (src.ptr + swizzled_idx).load[
                    width=simd_size, alignment=src_align
                ]()
                dst_fragments.ptr.store[alignment=dst_align](
                    dst_idx, src_vec.cast[dst_type]()
                )


# Copy from SRAM to local memory.
#
@always_inline
fn copy_sram_to_local[
    src_layout: Layout,
    dst_layout: Layout,
    dtype: DType,
    src_warp_layout: Layout,
    src_element_layout: Layout,
    dst_element_layout: Layout,
    axis: OptionalReg[Int] = None,
](
    dst: LayoutTensor[
        dtype,
        dst_layout,
        address_space = _GPUAddressSpace.GENERIC,
        element_layout=dst_element_layout,
    ],
    src: LayoutTensor[
        dtype,
        src_layout,
        address_space = _GPUAddressSpace.SHARED,
        element_layout=src_element_layout,
    ],
):
    @parameter
    if axis:
        var src_fragments = src.distribute[
            src_warp_layout, axis = axis.value()
        ](ThreadIdx.x())
        dst.copy_from(src_fragments)
    else:
        var src_fragments = src.distribute[src_warp_layout](ThreadIdx.x())
        dst.copy_from(src_fragments)


# Copy local memory to DRAM, thread affinity is needed only for dst fragments.
#
@always_inline
fn copy_local_to_dram[
    src_layout: Layout,
    dst_layout: Layout,
    dtype: DType,
    dst_thread_layout: Layout,
    src_element_layout: Layout,
    dst_element_layout: Layout,
    src_addr_space: AddressSpace,
](
    dst: LayoutTensor[
        dtype,
        dst_layout,
        address_space = _GPUAddressSpace.GENERIC,
        element_layout=dst_element_layout,
    ],
    src: LayoutTensor[
        dtype,
        src_layout,
        address_space=src_addr_space,
        element_layout=src_element_layout,
    ],
):
    var dst_fragments = dst.distribute[dst_thread_layout](ThreadIdx.x())
    dst_fragments.copy_from(src)


# Copy local memory to DRAM, thread affinity is needed only for dst fragments.
#
@always_inline
fn copy_local_to_dram[
    src_layout: Layout,
    dst_layout: Layout,
    dtype: DType,
    dst_thread_layout: Layout,
    src_element_layout: Layout,
    dst_element_layout: Layout,
    src_addr_space: AddressSpace,
](
    dst: LayoutTensor[
        dtype,
        dst_layout,
        address_space = _GPUAddressSpace.GENERIC,
        element_layout=dst_element_layout,
    ],
    src: LayoutTensor[
        dtype,
        src_layout,
        address_space=src_addr_space,
        element_layout=src_element_layout,
    ],
    offset: Int,
    rows: Int,
    cols: Int,
):
    var dst_fragments = dst.distribute[dst_thread_layout](ThreadIdx.x())
    var thread_offset = dst_fragments.distance(dst.ptr) + offset
    dst_fragments.copy_from(src, thread_offset, rows, cols)


@always_inline
fn copy_local_to_sram[
    src_layout: Layout,
    dst_layout: Layout,
    src_type: DType,
    dst_type: DType,
    thread_layout: Layout,
    src_element_layout: Layout,
    dst_element_layout: Layout,
    src_addr_space: AddressSpace,
    swizzle: OptionalReg[Swizzle] = None,
](
    dst: LayoutTensor[
        dst_type,
        dst_layout,
        address_space = _GPUAddressSpace.SHARED,
        element_layout=dst_element_layout,
    ],
    src: LayoutTensor[
        src_type,
        src_layout,
        address_space=src_addr_space,
        element_layout=src_element_layout,
    ],
):
    var dst_frag = dst.distribute[thread_layout](ThreadIdx.x())

    @parameter
    if src_type == dst_type:

        @parameter
        if swizzle:
            alias swizzle_fn = swizzle.value()
            alias num_vecs = src_layout.size()
            alias align = alignof[SIMD[src_type, src.element_size]]()

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
                    width = src.element_size, alignment=align
                ](src_idx)
                dst.ptr.store[alignment=align](
                    swizzled_idx, src_vec.cast[dst_type]()
                )

        else:
            dst_frag.copy_from(src)
    else:
        constrained[
            src_type == DType.float32 and dst_type.is_half_float(),
            "Only support FP32 -> half precision downcast during copy.",
        ]()

        constrained[
            src.element_size == dst.element_size,
            "src and dst element size mismatch.",
        ]()

        alias num_stores_per_thread = dst_frag.layout.size()
        alias elem_size = src.element_size

        @parameter
        for i in range(num_stores_per_thread):
            alias dst_idx = dst_frag.layout(i)

            dst_frag.ptr.store[
                alignment = alignof[SIMD[dst_type, src.element_size]](),
            ](dst_idx, src.aligned_load[elem_size](i, 0).cast[dst_type]())


@always_inline
fn copy_local_to_local[
    dst_type: DType,
    src_type: DType,
    dst_layout: Layout,
    src_layout: Layout,
    dst_element_layout: Layout,
    src_addr_space: AddressSpace,
](
    dst: LayoutTensor[
        dst_type,
        dst_layout,
        address_space = _GPUAddressSpace.LOCAL,
        element_layout=dst_element_layout,
    ],
    src: LayoutTensor[src_type, src_layout, address_space=src_addr_space],
):
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
        alias num_mmas = src_layout.shape[0].value()
        alias src_frag_size = src_layout.shape[1].value()
        alias a_frag_layout = composition(
            src_layout,
            make_layout(Layout.row_major(num_mmas // 2, 2), src_layout[1]),
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
                    dst_type
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
    type: DType,
    layout: Layout,
    address_space: AddressSpace = AddressSpace.GENERIC,
    circular: Bool = False,
]:
    """Iterate through a memory buffer and construct layout tensor.

    The returned layout tensor is NOT vectorized. User should explicitly vectorize.
    """

    var ptr: UnsafePointer[Scalar[type], address_space]
    var offset: Int
    var stride: Int
    var bound: Int
    var runtime_layout: RuntimeLayout[layout]

    @always_inline
    fn __init__(inout self):
        """Empty iterator, used as default value."""
        self.ptr = UnsafePointer[Scalar[type], address_space]()
        self.offset = 0
        self.stride = 0
        self.bound = 0
        self.runtime_layout = RuntimeLayout[layout]()

    @always_inline
    fn __init__(
        inout self,
        ptr: UnsafePointer[Scalar[type], address_space],
        bound: Int,
        stride: Int = layout.size() if layout.all_dims_known() else UNKNOWN_VALUE,
        offset: Int = 0,
        runtime_layout: RuntimeLayout[layout] = RuntimeLayout[layout](),
    ):
        self.ptr = ptr
        self.offset = offset
        self.stride = (
            runtime_layout.size() if stride == UNKNOWN_VALUE else stride
        )
        self.bound = bound
        self.runtime_layout = runtime_layout

    @always_inline
    fn get(self) -> LayoutTensor[type, layout, address_space=address_space]:
        """Return the layout tensor at current iterator."""
        # TODO: Use deref `[]` to be consistent with mojo feature.

        return LayoutTensor[type, layout, address_space=address_space](
            self.ptr + self.offset, self.runtime_layout
        )

    @always_inline
    fn __getitem__(
        self,
    ) -> LayoutTensor[type, layout, address_space=address_space]:
        """Return the layout tensor at current iterator."""
        return self.get()

    @always_inline
    fn __iadd__[T: Intable](inout self, rhs: T):
        """Increment the iterator.

        This function is unsafe. It omits bound checking for performance reasons.
        Caller must make sure index doesn't go out-of-bound.
        """

        self.offset += int(rhs) * self.stride

        @parameter
        if circular:
            self.offset = self.offset % self.bound

    @always_inline
    fn _incr(inout self):
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
    fn __iadd__(inout self, rhs: UInt):
        """Increment the iterator.

        This function is unsafe. It omits bound checking for performance reasons.
        Caller must make sure index doesn't go out-of-bound.
        """
        self += int(rhs)

    @always_inline
    fn next[T: Intable](self, rhs: T) -> Self:
        """Return an iterator pointing to the next `rhs` layout tensor."""

        var next_offset = self.offset + int(rhs) * self.stride

        @parameter
        if circular:
            next_offset = next_offset % self.bound

        return LayoutTensorIter[
            type, layout, address_space=address_space, circular=circular
        ](
            self.ptr,
            self.bound,
            stride=self.stride,
            offset=next_offset,
            runtime_layout=self.runtime_layout,
        )

    @always_inline
    fn next(self, rhs: UInt = 1) -> Self:
        return self.next(int(rhs))

    @always_inline
    fn next_unsafe(self, rhs: UInt = 1) -> Self:
        """Return an iterator pointing to the next `rhs` layout tensor.
        This is the unsafe version and user must ensure rhs < bound / stride.
        """

        var next_offset = self.offset + int(rhs) * self.stride

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
            runtime_layout=self.runtime_layout,
        )

    @always_inline
    fn reshape[
        dst_layout: Layout,
    ](self) -> LayoutTensorIter[type, dst_layout, address_space, circular]:
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

        return LayoutTensorIter[type, dst_layout, address_space, circular](
            self.ptr,
            self.bound,
            self.stride,
            self.offset,
            RuntimeLayout[dst_layout](),
        )

    @always_inline
    fn bitcast[
        new_type: DType, *, address_space: AddressSpace = Self.address_space
    ](self) -> LayoutTensorIter[new_type, layout, address_space, Self.circular]:
        return LayoutTensorIter[new_type, layout, address_space, Self.circular](
            self.ptr.bitcast[new_type, address_space](),
            self.bound,
            self.stride,
            self.offset,
            self.runtime_layout,
        )
