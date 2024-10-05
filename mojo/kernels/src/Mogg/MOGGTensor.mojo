# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# A temporary home for the experimental tensor type.

from collections import OptionalReg
from math import fma
from sys.info import simdwidthof
from sys.intrinsics import strided_load

from algorithm.functional import elementwise, vectorize
from buffer import NDBuffer
from buffer.dimlist import DimList
from memory import UnsafePointer
from memory.unsafe import bitcast
from MOGGIntList import IntList
from register import *

from utils import IndexList, unroll
from os.atomic import Atomic


@value
@register_passable("trivial")
struct InnerStride:
    var val: Int
    alias Broadcast = InnerStride(0)
    alias Contiguous = InnerStride(1)
    alias Strided = InnerStride(2)

    @staticmethod
    fn from_stride(inner_stride: Int) -> Self:
        if inner_stride == 0:
            return InnerStride.Broadcast
        elif inner_stride == 1:
            return InnerStride.Contiguous
        else:
            return InnerStride.Strided

    @always_inline
    fn __eq__(self, other: InnerStride) -> Bool:
        return self.val == other.val


@value
@register_passable
struct UnsafeRefCounter[type: DType]:
    """
    Implements an atomic which can be sharable as a copyable reference.

    By design the counter memory must be managed manually, this allows us to
    copy this around by ref and have multiple references to the same pointer
    under the hood.
    """

    var _underlying_value: UnsafePointer[Scalar[type]]

    fn deallocate(owned self):
        self._underlying_value.free()

    fn increment(self) -> Scalar[type]:
        return Atomic[type]._fetch_add(self._underlying_value, 1)

    fn decrement(self) -> Scalar[type]:
        return Atomic[type]._fetch_add(self._underlying_value, -1)

    fn _value(inout self) -> Scalar[type]:
        return Atomic[type]._fetch_add(self._underlying_value, 0)


@always_inline
fn _get_start_indices_of_nth_subvolume[
    subvolume_rank: Int, static_shape: DimList
](n: Int, shape: IntList[static_shape]) -> IntList[shape._size_but_unknown]:
    """
    Converts from a flat 1D index into an ND index for the given shape. The
    `subvolume_rank` parameter is used to skip some of the ND dimension so we
    can calculate an index over a subset of the shape. I.E subvolume_rank=1
    will calcualte over (N, B, C, 0) with the last dimension being 0.
    """
    var out = IntList[shape._size_but_unknown].empty(shape.length)
    var curr_index = n

    @parameter
    if shape.has_static_length():
        alias rank = shape._length

        @always_inline
        @parameter
        fn compute_shape[idx: Int]():
            alias i = rank - 1 - idx - subvolume_rank
            out[i] = curr_index % shape[i]
            curr_index //= shape[i]

        unroll[compute_shape, rank - subvolume_rank]()
    else:
        var rank = len(out)
        for i in reversed(range(rank - subvolume_rank)):
            out[i] = curr_index % shape[i]
            curr_index //= shape[i]

    return out^


fn _static_strides_from_shape[static_shape: DimList]() -> DimList:
    @parameter
    if len(static_shape) == 0:
        return DimList()
    else:
        # Just ensure it is using stack memory for now
        # TODO: Calculate the static strides.
        return DimList.create_unknown[len(static_shape)]()


struct Tensor[
    type: DType,
    static_shape: DimList = DimList(),
    static_strides: DimList = _static_strides_from_shape[static_shape](),
    _internal_in_lambda: OptionalReg[
        fn[_w: Int, _v: DimList] (IntList[_v]) capturing -> SIMD[type, _w]
    ] = None,
    _internal_out_lambda: OptionalReg[
        fn[_w: Int, _v: DimList] (IntList[_v], SIMD[type, _w]) capturing -> None
    ] = None,
    _OWNED_MEMORY: Bool = True,
]:
    alias static_rank = OptionalReg[Int](None) if len(
        static_shape
    ) == 0 else len(static_shape)

    var data: UnsafePointer[Scalar[type]]
    var shape: IntList[static_shape]
    var strides: IntList[static_strides]
    var dyn_rank: Int
    var storage_ref_count: UnsafeRefCounter[DType.index]

    # Empty strides...
    @always_inline
    fn __init__(
        inout self,
        ptr: UnsafePointer[Scalar[type]],
        shape: IntList,
        ref_count_ptr: UnsafePointer[Scalar[DType.index]],
    ):
        self.data = ptr
        self.shape = IntList[static_shape](shape)
        self.strides = IntList[static_strides](shape)
        self.dyn_rank = len(shape)

        # If the strides are already fully static we don't need to set them.
        @parameter
        if not IntList[static_strides].is_fully_static():
            var stride = 1
            # Walk backwards to compute the fully contiguous strides.
            for i in reversed(range(len(self.shape))):
                self.strides[i] = stride
                stride *= self.shape[i]

        # Increment the refcount if we own the memory otherwise create an
        # empty counter.
        @parameter
        if Self._OWNED_MEMORY:
            self.storage_ref_count = UnsafeRefCounter[DType.index](
                ref_count_ptr
            )
            _ = self.storage_ref_count.increment()
        else:
            self.storage_ref_count = UnsafeRefCounter[DType.index](
                UnsafePointer[Scalar[DType.index]]()
            )

    @always_inline
    fn __init__(
        inout self,
        ptr: UnsafePointer[Scalar[type]],
        shape: IntList,
        strides: IntList,
        ref_count_ptr: UnsafePointer[Scalar[DType.index]],
    ):
        self.data = ptr
        self.shape = IntList[static_shape](shape)
        self.strides = IntList[static_strides](strides)
        self.dyn_rank = len(shape)

        # Increment the refcount if we own the memory otherwise create an
        # empty counter.
        @parameter
        if Self._OWNED_MEMORY:
            self.storage_ref_count = UnsafeRefCounter[DType.index](
                ref_count_ptr
            )
            _ = self.storage_ref_count.increment()
        else:
            self.storage_ref_count = UnsafeRefCounter[DType.index](
                UnsafePointer[Scalar[DType.index]]()
            )

    @mogg_tensor_copy_constructor()
    @no_inline
    fn __copyinit__(inout self, existing: Self):
        @parameter
        if Self._OWNED_MEMORY:
            self.storage_ref_count = existing.storage_ref_count
            _ = self.storage_ref_count.increment()
        else:
            # Create an empty refcount if the memory isn't owned.
            self.storage_ref_count = UnsafeRefCounter[DType.index](
                UnsafePointer[Scalar[DType.index]]()
            )

        self.data = existing.data
        self.shape = existing.shape
        self.strides = existing.strides
        self.dyn_rank = existing.dyn_rank

    @staticmethod
    fn same_rank_param() -> DimList:
        return DimList.create_unknown[len(Self.static_shape)]()

    @always_inline
    fn refcount(self) -> UnsafePointer[SIMD[DType.index, 1]]:
        return self.storage_ref_count._underlying_value

    @mogg_tensor_deconstructor()
    fn __del__(owned self):
        @parameter
        if Self._OWNED_MEMORY:
            var res = self.storage_ref_count.decrement()
            if res == 1:
                # Clean up the managed refcount itself
                self.storage_ref_count.deallocate()

                # TODO: Invoke parameterized deconstructor.
                self.data.free()

    @mogg_enable_fusion()
    @no_inline
    fn enable_fusion(self):
        pass

    @staticmethod
    @no_inline
    fn _nop():
        pass

    @mogg_input_fusion_hook()
    @no_inline
    fn _input_fusion_hook(self):
        @always_inline
        @parameter
        fn _default_lambda[
            _w: Int, _v: DimList
        ](i: IntList[_v]) -> SIMD[type, _w]:
            return rebind[SIMD[type, _w]](self._simd_load_internal[_w](i))

        # This should return a rebind but until that works we just call this nop.
        Tensor[
            Self.type, Self.static_shape, Self.static_strides, _default_lambda
        ]._nop()

    @mogg_output_fusion_hook()
    @no_inline
    fn _output_fusion_hook(inout self):
        @always_inline
        @parameter
        fn _default_output[
            _w: Int, _v: DimList
        ](i: IntList[_v], value: SIMD[type, _w]):
            self._simd_store_internal(i, rebind[SIMD[type, _w]](value))

        # This should return a rebind but until that works we just call this nop.
        Tensor[
            Self.type,
            Self.static_shape,
            Self.static_strides,
            None,
            _default_output,
        ]._nop()

    @always_inline
    fn nelems(self) -> Int:
        return self.shape.nelems()

    @always_inline
    fn rank(self) -> Int:
        @parameter
        if Self.static_rank:
            return Self.static_rank.value()
        return self.dyn_rank

    @always_inline
    fn _compute_flat_index(
        self,
        index: IntList,
    ) -> Int:
        var flat_index: Int = 0

        @parameter
        if Self.static_rank:

            @always_inline
            @parameter
            fn body[idx: Int]():
                flat_index = fma(index[idx], self.strides[idx], flat_index)

            unroll[body, Self.static_rank.value()]()
        else:
            # Dynamic case for dynamic ranks.
            for idx in range(self.dyn_rank):
                flat_index = fma(index[idx], self.strides[idx], flat_index)
        return flat_index

    @always_inline
    fn store(
        inout self, index: IndexList[Self.static_rank.value()], value: SIMD
    ):
        self.store(
            IntList[DimList.create_unknown[Self.static_rank.value()]()](index),
            value,
        )

    @always_inline
    fn store(inout self, index: IntList, value: SIMD):
        # Nop function to preserve symbol.
        self._output_fusion_hook()

        var val = rebind[SIMD[type, value.size]](value)

        @parameter
        if Self._internal_out_lambda:
            alias func = Self._internal_out_lambda.value()
            func[val.size, index.static_values](index, val)
        else:
            self._simd_store_internal(index, val)

    @always_inline
    fn store(inout self, index: Int, value: SIMD):
        constrained[
            self.static_rank.value() == 1,
            (
                "Single int access to kernels only allowed on tensors"
                " statically known to be 1D"
            ),
        ]()
        var as_nd = self.get_nd_indices()
        as_nd[0] = index
        self.store(as_nd, value)

    @always_inline
    fn _simd_store_internal(inout self, index: IntList, val: SIMD):
        var flat_index = self._compute_flat_index(index)
        var value = rebind[SIMD[type, val.size]](val)
        self.data.store(flat_index, value)

    @always_inline
    fn get_nd_indices(self) -> IntList[Self.same_rank_param()]:
        return IntList[Self.same_rank_param()].zeros(self.dyn_rank)

    @always_inline
    fn simd_load[simd_width: Int](self, index: Int) -> SIMD[type, simd_width]:
        constrained[
            self.static_rank.value() == 1,
            (
                "Single int access to kernels only allowed on tensors"
                " statically known to be 1D"
            ),
        ]()
        var as_nd = self.get_nd_indices()
        as_nd[0] = index
        return self.simd_load[simd_width](as_nd)

    @always_inline
    fn simd_load[
        simd_width: Int,
    ](self, index: IndexList[Self.static_rank.value()]) -> SIMD[
        type, simd_width
    ]:
        return self.simd_load[simd_width](
            IntList[DimList.create_unknown[Self.static_rank.value()]()](index)
        )

    @always_inline
    fn simd_load[
        simd_width: Int
    ](self, index: IntList) -> SIMD[type, simd_width]:
        # Nop function to preserve symbols from DCE.
        self._input_fusion_hook()

        @parameter
        if Self._internal_in_lambda:
            alias func = Self._internal_in_lambda.value()
            return func[simd_width, index.static_values](index)
        else:
            return self._simd_load_internal[simd_width](index)

    @always_inline
    fn _simd_load_internal[
        simd_width: Int
    ](self, index: IntList) -> SIMD[type, simd_width]:
        var flat_index = self._compute_flat_index(index)
        var stride = self.strides[self.rank() - 1]

        @parameter
        @always_inline
        fn _load[
            stride_type: InnerStride
        ](stride: Int) -> SIMD[type, simd_width]:
            @parameter
            if stride_type == InnerStride.Broadcast:
                return self.data.load(flat_index)
            elif stride_type == InnerStride.Contiguous:

                @parameter
                if type is DType.bool:
                    var v = self.data.bitcast[DType.uint8]().load[
                        width=simd_width
                    ](flat_index)
                    return v.cast[type]()
                else:
                    return self.data.load[width=simd_width](flat_index)

            else:

                @parameter
                if type is DType.bool:
                    var v = strided_load[simd_width](
                        self.data.bitcast[DType.uint8]().offset(flat_index),
                        stride,
                    )
                    return v.cast[type]()
                else:
                    return strided_load[simd_width](
                        self.data.offset(flat_index), stride
                    )

        @parameter
        if (
            self.static_rank.__bool__()
            and self.static_strides.at[
                self.static_rank.value() - 1
            ]().__bool__()
        ):  # we know the exact type of load at compile time
            # TODO(#33183): should not need __bool__ since Dim is Boolable
            alias inner_stride_static = self.static_strides.at[
                self.static_rank.value() - 1
            ]().get()
            return _load[InnerStride.from_stride(inner_stride_static)](
                inner_stride_static
            )
        else:  # we need to dispatch on different types of loads depending on runtime stride
            if self.strides[self.rank() - 1] == 0:
                return _load[InnerStride.Broadcast](stride)
            elif stride > 1:
                return _load[InnerStride.Strided](stride)
            return _load[InnerStride.Contiguous](stride)

    @always_inline
    fn to_buffer[
        rank: Int
    ](self) -> NDBuffer[Self.type, rank, Self.static_shape]:
        var shape = IndexList[rank]()
        var strides = IndexList[rank]()

        @parameter
        for i in range(rank):
            shape[i] = self.shape[i]
            strides[i] = self.strides[i]

        var buffer = NDBuffer[Self.type, rank, Self.static_shape](
            self.data, shape, strides
        )
        return buffer

    # Stand in for elementwise while we experiment with the new tensor api
    @mogg_elementwise_hook()
    @no_inline
    fn for_each[
        func: fn[_width: Int] (IntList) capturing -> SIMD[type, _width],
    ](inout self):
        alias simd_width = simdwidthof[Self.type]()

        @parameter
        if not Self.static_rank:
            self._for_each_dynamic_rank[simd_width, func]()
        else:

            @parameter
            fn elementwise_fn_wrapper[
                width: Int, rank: Int
            ](coords_static: IndexList[rank]) capturing:
                alias dims = DimList.create_unknown[Self.static_rank.value()]()
                var coords = IntList[dims](
                    rebind[IndexList[Self.static_rank.value()]](coords_static)
                )
                var val = func[width](coords)
                self.store(coords, val)

            elementwise[elementwise_fn_wrapper, simd_width](
                rebind[IndexList[Self.static_rank.value()]](
                    self.shape.to_static_tuple()
                )
            )

    fn _for_each_dynamic_rank[
        simd_width: Int,
        func: fn[_width: Int] (IntList) capturing -> SIMD[type, _width],
    ](inout self):
        var rank = len(self.shape)
        var total_size: Int = self.shape.nelems()
        var inner_loop = self.shape[len(self.shape) - 1]
        var outer_loop = total_size // inner_loop

        for outer_i in range(outer_loop):
            var indices = _get_start_indices_of_nth_subvolume[1](
                outer_i, self.shape
            )

            @always_inline
            @parameter
            fn func_wrapper[width: Int](idx: Int):
                # The inner most dimension is vectorized, so we set it
                # to the index offset.
                indices[rank - 1] = idx
                var out = func[width](indices)
                self.store(indices, out)

            # We vectorize over the innermost dimension.
            vectorize[func_wrapper, simd_width](inner_loop)
