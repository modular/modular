# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import Optional, OptionalReg
from math import align_up, ceildiv
from os import abort
from sys import (
    alignof,
    prefetch,
    bitwidthof,
    simdwidthof,
    sizeof,
    is_nvidia_gpu,
)
from sys.intrinsics import PrefetchOptions

from algorithm import vectorize
from builtin.int import int as _int
from gpu.id import ThreadIdx
from gpu.memory import Fill, CacheEviction, async_copy, async_copy
from layout.element import Element, MemoryElement
from memory import UnsafePointer, memcpy, memset_zero, stack_allocation
from memory.pointer import AddressSpace, _GPUAddressSpace

from utils import IndexList, StaticTuple
from utils.numerics import max_finite

from .fillers import arange
from .int_tuple import depth, fill_like, flatten, idx2crd, product, to_int
from .layout import *
from .runtime_layout import RuntimeLayout
from .runtime_layout import coalesce as runtime_coalesce
from .runtime_layout import make_layout as make_runtime_layout
from .runtime_tuple import RuntimeTuple
from .swizzle import Swizzle, make_ldmatrix_swizzle


fn _compute_distribute_layout[
    data_layout: Layout,
    threads_layout: Layout,
    axis: OptionalReg[Int] = None,
]() -> Layout:
    """Distribute thread_layout into self layout, if axis is provided
    distribute into threads_layout projected into this axis.
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


fn _get_index_type(address_space: AddressSpace) -> DType:
    if address_space in (
        _GPUAddressSpace.SHARED,
        _GPUAddressSpace.CONSTANT,
        _GPUAddressSpace.PARAM,
    ):
        return DType.int32
    else:
        return DType.index


fn _get_index_type(layout: Layout, address_space: AddressSpace) -> DType:
    if layout.all_dims_known() and layout.cosize() < _int(
        max_finite[DType.int32]()
    ):
        return DType.int32
    else:
        return _get_index_type(address_space)


fn _get_unsigned_type(layout: Layout, address_space: AddressSpace) -> DType:
    if layout.all_dims_known() and layout.cosize() < _int(
        max_finite[DType.uint32]()
    ):
        return DType.uint32
    else:
        var dtype = _get_index_type(address_space)
        return DType.uint32 if dtype is DType.int32 else DType.index


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
fn _not_in_tuple[n: Int, size: Int, tuple: IndexList[size]]() -> Bool:
    @parameter
    for i in range(size):

        @parameter
        if tuple[i] == n:
            return False
    return True


# Returns whether the resulting tiled layout with the specified tile sizes
# requires masked access or not.
#
fn _tile_is_masked[layout: Layout, *tile_sizes: Int]() -> Bool:
    if not layout.all_dims_known():
        return True

    @parameter
    for axis in range(layout.rank()):
        alias dim = to_int(layout.shape[axis])
        if dim % tile_sizes[axis] != 0:
            return True
    return False


fn _distribute_is_masked[
    layout: Layout, threads_layout: Layout, axis: OptionalReg[Int] = None
]() -> Bool:
    # TODO: relax this constrain
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
        alias layout_dim = to_int(layout.shape[i])
        alias thread_dim = to_int(threads_layout.shape[i])

        @parameter
        if layout_dim % thread_dim != 0:
            return True

    return False


@register_passable("trivial")
struct LayoutTensor[
    dtype: DType,
    layout: Layout,
    rank: Int = layout.rank(),
    /,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
    element_layout: Layout = Layout(1, 1),
    layout_bitwidth: Int = bitwidthof[_get_index_type(address_space)](),
    masked: Bool = False,
](CollectionElement, CollectionElementNew, Stringable, Writable):
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
        layout_bitwidth: The bitwidth of each dimension of runtime layout.
        masked: If true the tensor is masked and runtime layouts determine the shape.
    """

    alias index_type: DType = _get_index_type(layout, address_space)

    var ptr: UnsafePointer[Scalar[dtype], address_space]

    var runtime_layout: RuntimeLayout[layout, bitwidth=layout_bitwidth]

    var runtime_element_layout: RuntimeLayout[element_layout]

    alias element_size = element_layout.size()
    alias element_type = SIMD[dtype, Self.element_size]

    # ===------------------------------------------------------------------=== #
    # Life cycle methods
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn __init__(out self, ptr: UnsafePointer[Scalar[dtype], address_space]):
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
        inout self,
        ptr: UnsafePointer[Scalar[dtype], address_space],
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
        inout self,
        ptr: UnsafePointer[Scalar[dtype], address_space],
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

    fn __init__(out self, *, other: Self):
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
    ] as result:
        """Bitcast the underlying pointer to a new data type.

        Parameters:
            new_type: The new data type it is casting to.
            address_space: The address space of the returned LayoutTensor.
            element_layout: The element layout of the returned LayoutTensor.
        """
        return __type_of(result)(
            self.ptr.bitcast[Scalar[new_type], address_space=address_space](),
            rebind[RuntimeLayout[layout, bitwidth = result.layout_bitwidth]](
                self.runtime_layout
            ),
        )

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
            layout_bitwidth=layout_bitwidth,
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
            layout_bitwidth=layout_bitwidth,
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
            layout_bitwidth=layout_bitwidth,
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
            layout_bitwidth=layout_bitwidth,
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
            layout_bitwidth=layout_bitwidth,
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

        return self.ptr.store(self._offset(m, n), val)

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
        return self.ptr.store[alignment=alignment](self._offset(m, n), val)

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
    fn _toStatic[t: IntTuple]() -> IndexList[len(t)]:
        var st = IndexList[len(t)]()

        @parameter
        for i in range(len(t)):
            st[i] = to_int(t[i])
        return st

    @staticmethod
    @always_inline("nodebug")
    fn _getOffset[
        rank: Int
    ](stride: IndexList[rank, **_], vals: VariadicList[Int]) -> Int:
        var offset = 0

        @parameter
        for i in range(rank):
            offset += vals[i] * stride[i]
        return offset

    @staticmethod
    @always_inline("nodebug")
    fn _getOffset[
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
        Self._compute_tile_layout[*tile_sizes]()[0],
        address_space=address_space,
        masked = masked or _tile_is_masked[layout, *tile_sizes](),
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

        alias num_tiles = __get_len[*tile_sizes]()

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
                alias stride = to_int(__tiled_layout[1].stride[i])
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
    ](self, *tile_coords: Int) -> LayoutTensorIter[
        dtype,
        Self._compute_tile_layout[*tile_sizes]()[0],
        address_space=address_space,
        circular=False,
        axis=axis,
        layout_bitwidth = Self.layout_bitwidth,
        masked = masked or _tile_is_masked[layout, *tile_sizes](),
    ] as result:
        """Returns the tiled iterator of the LayoutTensor.

        Parameters:
            tile_sizes: Tile sizes of each tile the iterator will iterate through.
            axis: Axis of the LayoutTensor the iterator will iterate through.

        Args:
            tile_coords: The tile coordinate that the iterator will point to.
        """

        alias tiles_rank = __get_len[*tile_sizes]()
        alias __tiled_layout = Self._compute_tile_layout[*tile_sizes]()
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
                alias stride = to_int(__tiled_layout[1].stride[i])
                ptr_offset += tile_coords[i] * stride

            # fmt: off
            alias bound = layout.shape[axis].value() * layout.stride[axis].value()
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
                    dimension_bound=self.dim(axis),
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
        alignment: Int = 1,
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
    ](self, thread_id: UInt) -> IndexList[rank]:
        constrained[
            len(flatten(thread_layout.shape)) <= 2
            and len(flatten(thread_layout.stride)) <= 2,
            "Only supporting rank-2 or less thread layout for dynamic tile.",
        ]()

        # clamp IndexList using thread_id and thread_layout
        var tile_shape = IndexList[rank]()
        alias thread_shape = thread_layout.shape
        alias thread_stride = thread_layout.stride

        # this would only work for rank-2 thread layout, need to extend this
        # to support thread layout such as Layout((2, 2), 2)
        @parameter
        for i in range(rank):
            alias thread_stride_i = to_int(thread_stride[i])
            alias thread_shape_i = to_int(thread_shape[i])
            var tile_idx = (thread_id // thread_stride_i) % thread_shape_i
            var tile_shape_i = ceildiv(self.dim(i), thread_shape_i)
            var bound_i = int((tile_shape_i - 1) * thread_shape_i + tile_idx)
            tile_shape[i] = min(self.dim(i) - bound_i, tile_shape_i)

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
        ]()[1],
        address_space=address_space,
        element_layout=element_layout,
        masked = masked
        or _distribute_is_masked[layout, threads_layout, axis](),
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

            @parameter
            if result.masked:
                return __type_of(result)(
                    self.ptr.offset(int(swizzled_offset)),
                    RuntimeLayout(runtime_shape, runtime_stride),
                )
            else:
                return __type_of(result)(
                    self.ptr.offset(int(swizzled_offset)),
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
                )
            else:
                return __type_of(result)(
                    self.ptr.offset(int(swizzled_offset)),
                    RuntimeLayout(runtime_shape, runtime_stride),
                    self.runtime_element_layout,
                )

    @always_inline
    fn vectorize[
        *vector_shape: Int
    ](self) -> LayoutTensor[
        dtype,
        coalesce(Self._compute_tile_layout[*vector_shape]()[1], keep_rank=True),
        address_space=address_space,
        element_layout = Self._divide_tiles[*vector_shape]()[0],
        masked=masked,
    ] as result:
        @parameter
        @always_inline
        fn __check_vector_shape[*vec_shape: Int]():
            @parameter
            for i in range(__get_len[*vec_shape]()):
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
            d0_slice.step.or_else(1) == 1 and d1_slice.step.or_else(1) == 1,
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
        slice_indices: IndexList[2],
        __offset_dims: Int = Self.rank - 2,
    ](
        self,
        offsets: IndexList[__offset_dims],
    ) -> LayoutTensor[
        dtype,
        Self.__compute_slice_layout(
            d0_slice, d1_slice, slice_indices[0], slice_indices[1]
        ),
        address_space=address_space,
        element_layout=element_layout,
    ] as result:
        constrained[
            d0_slice.step.or_else(1) == 1 and d1_slice.step.or_else(1) == 1,
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
        slice_indices: IndexList[1],
        __offset_dims: Int = Self.rank - 1,
    ](
        self,
        offsets: IndexList[__offset_dims],
    ) -> LayoutTensor[
        dtype,
        Self.__compute_slice_layout(d0_slice, slice_indices[0]),
        address_space=address_space,
        element_layout=element_layout,
    ] as result:
        constrained[
            d0_slice.step.or_else(1) == 1,
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
    fn copy_from(self, other: LayoutTensor):
        alias other_layout = other.layout

        alias dst_element_size = int(self.element_size)
        alias src_element_size = int(other.element_size)

        alias dst_size = layout.size()
        alias src_size = other_layout.size()

        constrained[
            layout.known_shape() and other_layout.known_shape(),
            "copy_from must move data of statically known shape",
        ]()

        constrained[
            dst_size == src_size,
            "copy_from should move data of the same size",
        ]()

        constrained[
            dst_element_size == src_element_size, "copy_from should move"
        ]()

        @parameter
        for i in range(dst_size):
            src_idx = other.__get_element_idx[i]()
            dst_idx = self.__get_element_idx[i]()

            src_element = MemoryElement(
                other.ptr.offset(src_idx), other.runtime_element_layout
            )

            dst_element = MemoryElement(
                self.ptr.offset(dst_idx), self.runtime_element_layout
            )

            dst_element.transfer(src_element)

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
                var src_element = Element[dtype, other.element_layout].load(
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
        masked: Bool = False,
        swizzle: OptionalReg[Swizzle] = None,
        fill: Fill = Fill.NONE,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        src: LayoutTensor,
        src_idx_bound: Int = UNKNOWN_VALUE,
        base_offset: Int = UNKNOWN_VALUE,
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
            src.layout.all_dims_known() and src.element_layout.all_dims_known()
        )

        var dst_ptr = self.ptr.bitcast[
            address_space = _GPUAddressSpace.SHARED
        ]()
        var src_ptr = src.ptr.bitcast[address_space = _GPUAddressSpace.GLOBAL]()

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
                    alias src_static_idx = src.layout(i)
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
                        async_copy[element_size_bytes, fill = Fill.ZERO](
                            src_ptr.bitcast[Scalar[dtype]]() + src_idx,
                            dst_ptr + dst_idx,
                            src_copy_size,
                        )
                    else:
                        async_copy[
                            element_size_bytes,
                            eviction_policy=eviction_policy,
                        ](
                            src_ptr.bitcast[Scalar[dtype]]() + src_idx,
                            dst_ptr + dst_idx,
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
                    src_idx = make_runtime_layout(
                        src.runtime_element_layout, src.runtime_layout
                    )(i)

                async_copy[4, eviction_policy=eviction_policy](
                    src_ptr.bitcast[Scalar[dtype]]() + src_idx,
                    dst_ptr + dst_idx,
                )

    @always_inline
    fn copy_from_async_masked_src[
        src_layout: Layout,
        src_addr_space: AddressSpace,
        src_element_layout: Layout,
        *,
        fill: Fill = Fill.NONE,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
        swizzle: OptionalReg[Swizzle] = None,
    ](
        self,
        src: LayoutTensor[
            dtype,
            src_layout,
            address_space=src_addr_space,
            element_layout=src_element_layout, **_,
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
                        async_copy[
                            element_size_bytes,
                            eviction_policy=eviction_policy,
                        ](
                            src_ptr + src_idx,
                            dst_ptr + dst_idx,
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
                    async_copy[4, eviction_policy=eviction_policy](
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
        return String.write(self)

    fn write_to[W: Writer](self, inout writer: W):
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
    in_tensor: LayoutTensor[dtype, layout, address_space=address_space, **_]
) -> LayoutTensor[
    dtype, layout, address_space=target_address_space, masked = in_tensor.masked
] as result:
    return __type_of(result).stack_allocation()


# Synchronous copy from DRAM -> SRAM, this requires w/r thread affinity mapping.
#
@always_inline
fn copy_dram_to_sram[
    src_thread_layout: Layout,
    dst_thread_layout: Layout = src_thread_layout,
    swizzle: OptionalReg[Swizzle] = None,
](dst: LayoutTensor, src: LayoutTensor):
    constrained[
        dst.dtype == src.dtype, "src dtype and dst dtype must be the same."
    ]()

    constrained[
        src.address_space == _GPUAddressSpace.GENERIC,
        "src address space must be GENERIC.",
    ]()

    constrained[
        dst.address_space == _GPUAddressSpace.SHARED,
        "dst address space must be SHARED.",
    ]()

    var src_fragments = src.distribute[src_thread_layout](ThreadIdx.x())
    var dst_fragments = dst.distribute[dst_thread_layout, swizzle=swizzle](
        ThreadIdx.x()
    )
    dst_fragments.copy_from(src_fragments)


# Synchronous copy from DRAM -> SRAM, this requires w/r thread affinity mapping.
#
@always_inline
fn copy_dram_to_sram[
    thread_layout: Layout,
    swizzle: OptionalReg[Swizzle] = None,
](dst: LayoutTensor, src: LayoutTensor):
    copy_dram_to_sram[
        src_thread_layout=thread_layout,
        dst_thread_layout=thread_layout,
        swizzle=swizzle,
    ](dst, src)


# Asynchronous copy from DRAM -> SRAM, this requires w/r thread affinity mapping.
#
@always_inline
fn copy_dram_to_sram_async[
    src_thread_layout: Layout,
    dst_thread_layout: Layout,
    swizzle: Bool = False,
    masked: Bool = False,
    fill: Fill = Fill.NONE,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
](dst: LayoutTensor, src: LayoutTensor, num_rows: Int = UNKNOWN_VALUE):
    constrained[
        src.address_space == _GPUAddressSpace.GENERIC,
        "src address space must be GENERIC.",
    ]()

    constrained[
        dst.address_space == _GPUAddressSpace.SHARED,
        "dst address space must be SHARED.",
    ]()

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
        OptionalReg[Swizzle](make_ldmatrix_swizzle[dst.dtype, row_size]())
    )

    var src_fragments = src.distribute[src_thread_layout](ThreadIdx.x())
    var dst_fragments = dst.distribute[dst_thread_layout](ThreadIdx.x())

    var dst_frag_offset = dst_fragments.distance(dst.ptr) if swizzle else 0

    @parameter
    if not masked:
        dst_fragments.copy_from_async[
            swizzle=swizzle_option, eviction_policy=eviction_policy
        ](src_fragments, base_offset=dst_frag_offset)
    else:
        constrained[
            src.layout.stride[1].value() == src.element_size
            and src.layout.rank() == 2,
            "Only support masking rows and 2D row major layout.",
        ]()
        var src_frag_offset = src_fragments.distance(src.ptr)
        alias stride = src.layout.stride[0].value()
        var src_idx_bound = num_rows * stride - src_frag_offset
        dst_fragments.copy_from_async[
            masked=True,
            swizzle=swizzle_option,
            eviction_policy=eviction_policy,
        ](
            src_fragments,
            src_idx_bound=src_idx_bound,
            base_offset=dst_frag_offset,
        )


# Asynchronous copy from DRAM -> SRAM, this requires w/r thread affinity mapping.
#
@always_inline
fn copy_dram_to_sram_async[
    thread_layout: Layout,
    swizzle: Bool = False,
    masked: Bool = False,
    fill: Fill = Fill.NONE,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
](dst: LayoutTensor, src: LayoutTensor, num_rows: Int = UNKNOWN_VALUE):
    copy_dram_to_sram_async[
        src_thread_layout=thread_layout,
        dst_thread_layout=thread_layout,
        swizzle=swizzle,
        masked=masked,
        eviction_policy=eviction_policy,
    ](dst, src, num_rows)


@always_inline
fn copy_sram_to_dram[
    thread_layout: Layout,
    swizzle: OptionalReg[Swizzle] = None,
](dst: LayoutTensor, src: LayoutTensor):
    constrained[
        dst.address_space == _GPUAddressSpace.GENERIC,
        "dst address space must be GENERIC.",
    ]()

    constrained[
        src.address_space == _GPUAddressSpace.SHARED,
        "src address space must be SHARED.",
    ]()

    constrained[
        src.layout.all_dims_known(), "Shared memory must have static layout"
    ]()

    var src_fragments = src.distribute[thread_layout](ThreadIdx.x())
    var dst_fragments = dst.distribute[thread_layout](ThreadIdx.x())

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
            var src_vec = src.ptr.load[width=simd_size, alignment=src_align](
                swizzled_idx
            )
            dst_fragments.ptr.store[alignment=dst_align](
                dst_idx, src_vec.cast[dst.dtype]()
            )


@always_inline
fn copy_sram_to_dram[
    thread_layout: Layout,
    swizzle: OptionalReg[Swizzle] = None,
](dst: LayoutTensor, src: LayoutTensor, offset: Int, rows: Int, cols: Int):
    constrained[
        dst.address_space == _GPUAddressSpace.GENERIC,
        "dst address space must be GENERIC.",
    ]()

    constrained[
        src.address_space == _GPUAddressSpace.SHARED,
        "src address space must be SHARED.",
    ]()

    var src_fragments = src.distribute[thread_layout](ThreadIdx.x())
    var dst_fragments = dst.distribute[thread_layout](ThreadIdx.x())
    var thread_offset = offset + dst_fragments.distance(dst.ptr)

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

        alias num_stores_per_thread = dst_fragments.layout.size()
        alias src_align = alignof[SIMD[src.dtype, simdwidthof[src.dtype]()]]()
        alias dst_align = alignof[SIMD[dst.dtype, simd_size]]()

        var src_frag_offset = src_fragments.distance(src.ptr)

        @parameter
        for i in range(num_stores_per_thread):
            alias src_idx = src_fragments.layout(i)
            alias dst_static_idx = dst_fragments.layout(i)

            var dst_idx = 0

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
                    swizzle_fn(src_frag_offset + src_idx_base) + src_idx_diff
                )

            if thread_offset + dst_idx < rows * cols:
                var src_vec = (src.ptr).load[
                    width=simd_size, alignment=src_align
                ](swizzled_idx)
                dst_fragments.ptr.store[alignment=dst_align](
                    dst_idx, src_vec.cast[dst.dtype]()
                )


# Copy from SRAM to local memory.
#
@always_inline
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
        ](ThreadIdx.x())
        dst.copy_from(src_fragments)
    else:
        var src_fragments = src.distribute[src_warp_layout](ThreadIdx.x())
        dst.copy_from(src_fragments)


# Copy local memory to DRAM, thread affinity is needed only for dst fragments.
#
@always_inline
fn copy_local_to_dram[
    dst_thread_layout: Layout,
](dst: LayoutTensor, src: LayoutTensor):
    constrained[
        dst.dtype == src.dtype, "dst dtype must be the same as src dtype."
    ]()

    constrained[
        src.address_space == _GPUAddressSpace.LOCAL,
        "src address space must be LOCAL.",
    ]()

    constrained[
        dst.address_space == _GPUAddressSpace.GENERIC,
        "dst address space must be GENERIC.",
    ]()

    var dst_fragments = dst.distribute[dst_thread_layout](ThreadIdx.x())
    dst_fragments.copy_from(src)


# Copy local memory to DRAM, thread affinity is needed only for dst fragments.
#
@always_inline
fn copy_local_to_dram[
    dst_thread_layout: Layout,
](dst: LayoutTensor, src: LayoutTensor, offset: Int, rows: Int, cols: Int):
    constrained[
        dst.dtype == src.dtype, "dst dtype must be the same as src dtype."
    ]()

    constrained[
        src.address_space == _GPUAddressSpace.LOCAL,
        "src address space must be LOCAL.",
    ]()

    constrained[
        dst.address_space == _GPUAddressSpace.GENERIC,
        "dst address space must be GENERIC.",
    ]()

    var dst_fragments = dst.distribute[dst_thread_layout](ThreadIdx.x())
    var thread_offset = dst_fragments.distance(dst.ptr) + offset
    dst_fragments.copy_from(src, thread_offset, rows, cols)


@always_inline
fn copy_local_to_sram[
    thread_layout: Layout,
    swizzle: OptionalReg[Swizzle] = None,
](dst: LayoutTensor, src: LayoutTensor):
    constrained[
        dst.address_space == _GPUAddressSpace.SHARED,
        "dst address space must be SHARED.",
    ]()

    constrained[
        src.address_space == _GPUAddressSpace.LOCAL,
        "src address space must be LOCAL.",
    ]()

    var dst_frag = dst.distribute[thread_layout](ThreadIdx.x())

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

    elif src.dtype == dst.dtype:
        dst_frag.copy_from(src)

    else:
        alias num_stores_per_thread = dst_frag.layout.size()
        alias elem_size = src.element_size

        @parameter
        for i in range(num_stores_per_thread):
            alias dst_idx = dst_frag.layout(i)

            dst_frag.ptr.store[
                alignment = alignof[SIMD[dst.dtype, src.element_size]](),
            ](dst_idx, src.aligned_load[elem_size](i, 0).cast[dst.dtype]())


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
    type: DType,
    layout: Layout,
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

    var ptr: UnsafePointer[Scalar[type], address_space, alignment]
    var offset: Scalar[_get_unsigned_type(layout, address_space)]
    var stride: Scalar[_get_unsigned_type(layout, address_space)]
    var bound: Scalar[_get_unsigned_type(layout, address_space)]
    var runtime_layout: RuntimeLayout[layout, bitwidth=layout_bitwidth]
    var dimension_bound: Scalar[_get_unsigned_type(layout, address_space)]
    var idx: Scalar[_get_unsigned_type(layout, address_space)]

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
        inout self,
        ptr: __type_of(self.ptr),
        bound: __type_of(self.offset),
        stride: __type_of(self.stride) = layout.size(),
        offset: __type_of(self.offset) = 0,
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
        inout self,
        ptr: __type_of(self.ptr),
        bound: __type_of(self.offset),
        runtime_layout: RuntimeLayout[layout, **_],
        stride: __type_of(
            self.stride
        ) = layout.size() if layout.all_dims_known() else UNKNOWN_VALUE,
        offset: __type_of(self.offset) = 0,
        dimension_bound: __type_of(self.offset) = 0,
        idx: __type_of(self.offset) = 0,
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
    ) -> LayoutTensor[
        type, layout, address_space=address_space, masked=masked
    ] as result:
        """Return the layout tensor at current iterator."""
        # TODO: Use deref `[]` to be consistent with mojo feature.

        return __type_of(result)(
            self.ptr + int(self.offset),
            rebind[RuntimeLayout[layout, bitwidth = result.layout_bitwidth]](
                self.runtime_layout
            ),
        )

    @always_inline
    fn __getitem__(
        self,
    ) -> LayoutTensor[type, layout, address_space=address_space, masked=masked]:
        """Return the layout tensor at current iterator."""
        return self.get()

    @always_inline
    fn _clip_shape(self) -> RuntimeLayout[layout, bitwidth=layout_bitwidth]:
        new_shape = self.runtime_layout.shape
        var cur_dim = new_shape.value[axis.value()]
        new_shape.value[axis.value()] = max(
            0, min(int(self.dimension_bound - self.idx * cur_dim), cur_dim)
        )
        return RuntimeLayout(new_shape, self.runtime_layout.stride)

    @always_inline
    fn __iadd__[T: Intable](inout self, rhs: T):
        """Increment the iterator.

        This function is unsafe. It omits bound checking for performance reasons.
        Caller must make sure index doesn't go out-of-bound.
        """
        self.offset += int(rhs) * self.stride

        @parameter
        if axis:
            self.idx += int(rhs)

        @parameter
        if masked:
            self.runtime_layout = self._clip_shape()

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

        var next_idx = 0
        var next_offset = self.offset + int(rhs) * self.stride

        @parameter
        if axis:
            next_idx = int(self.idx) + int(rhs)

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
            offset=int(next_offset),
            runtime_layout=runtime_layout,
            dimension_bound=self.dimension_bound,
            idx=next_idx,
        )

    @always_inline
    fn next(self, rhs: UInt = 1) -> Self:
        return self.next(int(rhs))

    @always_inline
    fn next_unsafe(self, rhs: UInt = 1) -> Self:
        """Return an iterator pointing to the next `rhs` layout tensor.
        This is the unsafe version and user must ensure rhs < bound / stride.
        """
        constrained[
            not masked, "Cannot use unsafe increment for masked iterator."
        ]()

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
            offset=int(next_offset),
        )

    @always_inline
    fn reshape[
        dst_layout: Layout,
    ](self) -> LayoutTensorIter[
        type,
        dst_layout,
        address_space=address_space,
        alignment=alignment,
        circular=circular,
        layout_bitwidth=layout_bitwidth,
        masked=masked,
    ] as result:
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
            int(self.bound),
            RuntimeLayout[dst_layout, bitwidth=layout_bitwidth](),
            int(self.stride),
            int(self.offset),
            dimension_bound=int(self.dimension_bound),
            idx=int(self.idx),
        )

    @always_inline
    fn bitcast[
        new_type: DType,
        *,
        address_space: AddressSpace = Self.address_space,
        alignment: Int = Self.alignment,
    ](self) -> LayoutTensorIter[
        new_type,
        layout,
        address_space=address_space,
        alignment=alignment,
        circular = Self.circular,
        layout_bitwidth=layout_bitwidth,
        masked=masked,
    ] as result:
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
            self.ptr.bitcast[
                Scalar[new_type],
                address_space=address_space,
                alignment=alignment,
            ](),
            int(self.bound),
            self.runtime_layout,
            int(self.stride),
            int(self.offset),
            dimension_bound=int(self.dimension_bound),
            idx=int(self.idx),
        )
