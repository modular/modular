# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# A temporary home for the experimental tensor type.

from algorithm.functional import vectorize_unroll
from math import fma
from memory.buffer import NDBuffer
from MOGGIntList import IntList
from sys.intrinsics import strided_load
from utils._optional import Optional
from utils._annotations import *


from memory.unsafe import Pointer, bitcast


@value
@register_passable
struct UnsafeRefCounter[type: DType]:
    """
    Implements an atomic which can be sharable as a copyable reference.

    By design the counter memory must be managed manually, this allows us to
    copy this around by ref and have multiple references to the same pointer
    under the hood.
    """

    var _underlying_value: Pointer[SIMD[type, 1]]

    fn deallocate(owned self):
        self._underlying_value.free()

    fn increment(self) -> SIMD[type, 1]:
        return Atomic[type]._fetch_add(self._underlying_value, 1)

    fn decrement(self) -> SIMD[type, 1]:
        return Atomic[type]._fetch_add(self._underlying_value, -1)

    fn _value(inout self) -> SIMD[type, 1]:
        return self._underlying_value.load()


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
            out._unsafe_set_dim(i, curr_index % shape[i])
            curr_index //= shape[i]

        unroll[rank - subvolume_rank, compute_shape]()
    else:
        let rank = len(out)
        for i in range(rank - subvolume_rank - 1, -1, -1):
            out._unsafe_set_dim(i, curr_index % shape[i])
            curr_index //= shape[i]

    return out ^


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
    MOGG_input_lambda: Optional[
        fn[
            _w: Int, _t: DType, _v: DimList
        ] (IntList[_v]) capturing -> SIMD[_t, _w]
    ] = None,
    MOGG_output_lambda: Optional[
        fn[
            _w: Int, _t: DType, _v: DimList
        ] (IntList[_v], SIMD[_t, _w]) capturing -> None
    ] = None,
    _OWNED_MEMORY: Bool = True,
]:
    alias static_rank = -1 if len(static_shape) == 0 else len(static_shape)

    var data: DTypePointer[type]
    var shape: IntList[static_shape]
    var strides: IntList[static_strides]
    var dyn_rank: Int
    var storage_ref_count: UnsafeRefCounter[DType.index]

    # Empty strides...
    @always_inline
    fn __init__(
        inout self,
        ptr: DTypePointer[type],
        shape: IntList,
        ref_count_ptr: Pointer[Scalar[DType.index]],
    ):
        self.data = ptr
        self.shape = IntList[static_shape](shape)
        self.strides = IntList[static_strides](shape)
        self.dyn_rank = len(shape)

        var stride = 1

        # Walk backwards to compute the fully contiguous strides.
        for i in range(len(self.shape) - 1, -1, -1):
            self.strides._unsafe_set_dim(i, stride)
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
                Pointer[Scalar[DType.index]]()
            )

    @always_inline
    fn __init__(
        inout self,
        ptr: DTypePointer[type],
        shape: IntList,
        strides: IntList,
        ref_count_ptr: Pointer[Scalar[DType.index]],
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
                Pointer[Scalar[DType.index]]()
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
                Pointer[Scalar[DType.index]]()
            )

        self.data = existing.data
        self.shape = existing.shape
        self.strides = existing.strides
        self.dyn_rank = existing.dyn_rank

    @staticmethod
    fn same_rank_param() -> DimList:
        return DimList.create_unknown[len(Self.static_shape)]()

    @always_inline
    fn refcount(self) -> Pointer[SIMD[DType.index, 1]]:
        return self.storage_ref_count._underlying_value

    @mogg_tensor_deconstructor()
    fn __del__(owned self):
        @parameter
        if Self._OWNED_MEMORY:
            let res = self.storage_ref_count.decrement()
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
            _w: Int, _t: DType, _v: DimList
        ](i: IntList[_v]) -> SIMD[_t, _w]:
            return rebind[SIMD[_t, _w]](
                self._simd_load_internal[_w, i.static_values](i)
            )

        # This should return a rebind but until that works we just call this nop.
        Tensor[
            Self.type, Self.static_shape, Self.static_strides, _default_lambda
        ]._nop()

    @mogg_output_fusion_hook()
    @no_inline
    fn _output_fusion_hook(self):
        @always_inline
        @parameter
        fn _default_output[
            _w: Int, _t: DType, _v: DimList
        ](i: IntList[_v], value: SIMD[_t, _w]):
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
        if Self.static_rank != -1:
            return Self.static_rank
        return self.dyn_rank

    @always_inline
    fn _compute_flat_index(
        self,
        index: IntList,
    ) -> Int:
        var flat_index: Int = 0

        @parameter
        if Self.static_rank != -1:

            @always_inline
            @parameter
            fn body[idx: Int]():
                flat_index = fma(index[idx], self.strides[idx], flat_index)

            unroll[Self.static_rank, body]()
        else:
            # Dynamic case for dynamic ranks.
            for idx in range(self.dyn_rank):
                flat_index = fma(index[idx], self.strides[idx], flat_index)
        return flat_index

    @always_inline
    fn store(self, index: IntList, value: SIMD):
        # Nop function to preserve symbol.
        self._output_fusion_hook()

        let val = rebind[SIMD[type, value.size]](value)

        @parameter
        if Self.MOGG_output_lambda:
            alias func = Self.MOGG_output_lambda.value()
            func[val.size, type, index.static_values](index, val)
        else:
            self._simd_store_internal(index, val)

    @always_inline
    fn _simd_store_internal(self, index: IntList, val: SIMD):
        let flat_index = self._compute_flat_index(index)
        let value = rebind[SIMD[type, val.size]](val)
        self.data.simd_store[val.size](flat_index, value)

    @always_inline
    fn simd_load[
        simd_width: Int
    ](self, index: IntList) -> SIMD[type, simd_width]:
        # Nop function to preserve symbols from DCE.
        self._input_fusion_hook()

        @parameter
        if Self.MOGG_input_lambda:
            alias func = Self.MOGG_input_lambda.value()
            return func[simd_width, type, index.static_values](index)
        else:
            return self._simd_load_internal[simd_width](index)

    @always_inline
    fn _simd_load_internal[
        simd_width: Int
    ](self, index: IntList) -> SIMD[type, simd_width]:
        let flat_index = self._compute_flat_index(index)
        let stride = self.strides[self.rank() - 1]

        if self.strides[self.rank() - 1] == 0:
            return self.data.load(flat_index)
        elif stride > 1:

            @parameter
            if type == DType.bool:
                let v = strided_load[DType.uint8, simd_width](
                    self.data.bitcast[DType.uint8]().offset(flat_index), stride
                )
                return v.cast[type]()
            else:
                return strided_load[type, simd_width](
                    self.data.offset(flat_index), stride
                )

        @parameter
        if type == DType.bool:
            let v = self.data.bitcast[DType.uint8]().simd_load[simd_width](
                flat_index
            )
            return v.cast[type]()
        else:
            return self.data.simd_load[simd_width](flat_index)

    @always_inline
    fn to_buffer[
        rank: Int
    ](self) -> NDBuffer[rank, Self.static_shape, Self.type]:
        var shape = StaticIntTuple[rank]()
        var strides = StaticIntTuple[rank]()

        @unroll
        for i in range(rank):
            shape[i] = self.shape[i]
            strides[i] = self.strides[i]

        let buffer = NDBuffer[rank, Self.static_shape, Self.type](
            self.data, shape, strides
        )
        return buffer

    # Stand in for elementwise while we experiment with the new tensor api
    @mogg_elementwise_hook()
    @no_inline
    fn for_each[
        simd_width: Int,
        func: fn[_width: Int, _t: DType] (IntList) capturing -> SIMD[
            _t, _width
        ],
    ](inout self):
        let rank = len(self.shape)
        let total_size: Int = self.shape.nelems()
        let inner_loop = self.shape[len(self.shape) - 1]
        let outer_loop = total_size // inner_loop

        for outer_i in range(outer_loop):
            var indices = _get_start_indices_of_nth_subvolume[1](
                outer_i, self.shape
            )

            @always_inline
            @parameter
            fn func_wrapper[width: Int](idx: Int):
                # The inner most dimension is vectorized, so we set it
                # to the index offset.
                indices._unsafe_set_dim(rank - 1, idx)
                let out = func[width, Self.type, indices.static_values](indices)
                self.store(indices, out)

            # We vectorize over the innermost dimension.
            vectorize_unroll[simd_width, 1, func_wrapper](inner_loop)

            # We have to extend the lifetime of the indices as the above parameter
            # capture and use does not extend the lifetime of the object.
            _ = indices
