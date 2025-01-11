# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import InlineArray, OptionalReg
from gpu.host._compile import _get_gpu_target
from gpu.host.info import is_cpu
from math import ceil, fma
from sys import alignof, simdwidthof
from sys.info import is_nvidia_gpu
from sys.intrinsics import strided_load, strided_store

import algorithm
from bit import is_power_of_two
from buffer import DimList, NDBuffer
from compiler_internal.directives import (
    StaticTensorSpec,
    __mogg_intrinsic_attr,
    specsof,
)
from memory import UnsafePointer
from memory.pointer import _GPUAddressSpace
from runtime.asyncrt import MojoCallContextPtr
from tensor_internal import RuntimeTensorSpec, TensorSpec

from buffer import NDBuffer, DimList
from buffer.dimlist import _make_partially_static_index_list
from utils import IndexList
from register import register_internal_override
from .indexing import _dot_prod, _row_major_strides, _slice_to_tuple
from .tensor_like import TensorLike


# ===----------------------------------------------------------------------=== #
# Load / Store Helper primitives
# ===----------------------------------------------------------------------=== #


@parameter
@always_inline
fn gcd_pow2[a: Int, b: Int]() -> Int:
    # alignments should always be powers of 2
    constrained[
        is_power_of_two(a) and is_power_of_two(b),
        "a and b must be powers of 2",
    ]()
    return min(a, b)


# TODO(GEX-1523): Consider moving these and other methods implementation into
# non-class member functions.
#
# Note: these methods are forced inline in the graph compiler. We keep the
# inlining at the whims of the automatic inliner for now since we want to
# predictably introspect and manipulate these particular functions.
#
# They are set to be inlined further down graph compiler stack.
@register_internal_override("simd_store_into_managed_tensor_slice", 1)
@no_inline
fn simd_store_into_managed_tensor_slice[
    type: DType,
    rank: Int,
    simd_width: Int,
    alignment: Int = 1,
    address_space: AddressSpace = AddressSpace.GENERIC,
    static_strides: DimList = DimList.create_unknown[rank](),
](
    tensor: ManagedTensorSlice[type, rank],
    indices: IndexList[rank],
    value: SIMD[type, simd_width],
):
    var flat_index = tensor._compute_offset[address_space, static_strides](
        indices
    )

    # Store alignment cannot exceed the data type's alignment.
    alias max_alignment = gcd_pow2[alignment, alignof[type]()]()

    alias static_stride = static_strides.at[rank - 1]()

    # Stride = 1
    @parameter
    @always_inline
    fn store_stride1():
        @parameter
        if type is DType.bool:
            var v = value.cast[DType.uint8]()
            tensor._ptr.bitcast[Scalar[DType.uint8]]().store(flat_index, v)
        else:
            tensor._ptr.store[alignment=max_alignment](flat_index, value)

    # Stride > 1
    @parameter
    @always_inline
    fn store_strided(stride: Int):
        @parameter
        if type is DType.bool:
            var v = value.cast[DType.uint8]()
            strided_store(
                v,
                tensor._ptr.bitcast[Scalar[DType.uint8]]().offset(flat_index),
                stride,
            )
        else:
            return strided_store(value, tensor._ptr.offset(flat_index), stride)

    @parameter
    if static_stride.is_dynamic():
        var stride = tensor._runtime_strides[rank - 1]
        # Dynamic stride
        if stride == 0:
            tensor._ptr.store[alignment=max_alignment](0, value)
        elif stride == 1:
            store_stride1()
        else:
            store_strided(stride)
    else:
        # static stride
        @parameter
        if static_stride.get() == 0:
            tensor._ptr.store[alignment=max_alignment](0, value)
        elif static_stride.get() == 1:
            store_stride1()
        else:
            store_strided(static_stride.get())


@register_internal_override("simd_load_from_managed_tensor_slice", 1)
@no_inline
fn simd_load_from_managed_tensor_slice[
    type: DType,
    rank: Int,
    simd_width: Int,
    alignment: Int = 1,
    address_space: AddressSpace = AddressSpace.GENERIC,
    static_strides: DimList = DimList.create_unknown[rank](),
](tensor: ManagedTensorSlice[type, rank], indices: IndexList[rank]) -> SIMD[
    type, simd_width
]:
    var flat_index = tensor._compute_offset[address_space, static_strides](
        indices
    )
    alias static_stride = static_strides.at[rank - 1]()

    # Load alignment cannot exceed the data type's alignment.
    alias max_alignment = gcd_pow2[alignment, alignof[type]()]()

    # Stride = 1
    @parameter
    @always_inline
    fn load_stride1() -> SIMD[type, simd_width]:
        @parameter
        if type is DType.bool:
            var v = tensor._ptr.bitcast[Scalar[DType.uint8]]().load[
                width=simd_width
            ](flat_index)
            return v.cast[type]()
        else:
            return tensor._ptr.load[width=simd_width, alignment=max_alignment](
                flat_index
            )

    # Stride > 1
    @parameter
    @always_inline
    fn load_strided(stride: Int) -> SIMD[type, simd_width]:
        @parameter
        if type is DType.bool:
            var v = strided_load[simd_width](
                tensor._ptr.bitcast[Scalar[DType.uint8]]().offset(flat_index),
                stride,
            )
            return v.cast[type]()
        else:
            return strided_load[simd_width](
                tensor._ptr.offset(flat_index), stride
            )

    @parameter
    if static_stride.is_dynamic():
        var stride = tensor._runtime_strides[rank - 1]
        # Dynamic stride
        if stride == 0:
            return tensor._ptr.load(flat_index)
        elif stride == 1:
            return load_stride1()
        else:
            return load_strided(stride)
    else:
        # Static stride
        @parameter
        if static_stride.get() == 0:
            return tensor._ptr.load(flat_index)
        elif static_stride.get() == 1:
            return load_stride1()
        else:
            return load_strided(static_stride.get())


# ===----------------------------------------------------------------------=== #
# Input / output fusion primitives
# ===----------------------------------------------------------------------=== #


# This is a special version of `specsof(self)`
# Returns the static tensor spec of self, but without I/O lambdas.
# This function is called only inside the lambda implementations.
# Using specsof wouldn't work because the lambda would recursively depend on itself:
# kgen.param.declare_region lambda_fn = {
#   kgen.param.declare specs = build_tensor_specs(..., lambda_fn, ...) // cycle
# }
@__mogg_intrinsic_attr("mogg.get_tensor_specs_without_lambdas")
@no_inline
fn _get_tensor_specs_without_lambdas[
    type: DType, rank: Int
]() -> StaticTensorSpec[type, rank]:
    return StaticTensorSpec[type, rank]()


# Helper functions used in SliceMOGGDPSFunc to ensure the input lambda isn't DCE
@no_inline
fn _extract_input_lambda[
    type: DType, rank: Int, T: StaticTensorSpec[type, rank].in_lambda_t
]():
    pass


# Helper functions used in SliceMOGGDPSFunc to ensure the output lambda isn't DCE
@no_inline
fn _extract_output_lambda[
    type: DType, rank: Int, T: StaticTensorSpec[type, rank].out_lambda_t
]():
    pass


# Helper function used in SliceMOGGDPSFunc to generate the body of the input lambda
@__mogg_intrinsic_attr("mogg.dps_input_fusion_hook")
@no_inline
fn _input_fusion_hook_impl[
    type: DType, rank: Int
](tensor: ManagedTensorSlice[type, rank]):
    @always_inline
    @parameter
    fn _input_lambda[_w: Int](i: IndexList[rank]) -> SIMD[type, _w]:
        alias static_specs = _get_tensor_specs_without_lambdas[type, rank]()

        # We use these methods to help with fusion passes which manipulates
        # calls. It is helpful to have a registered function.
        return rebind[SIMD[type, _w]](
            simd_load_from_managed_tensor_slice[
                simd_width=_w,
                alignment = static_specs.alignment,
                address_space = static_specs.address_space,
                static_strides = static_specs.strides,
            ](tensor, i)
        )

    _extract_input_lambda[type, rank, _input_lambda]()


# Helper function used in SliceMOGGDPSFunc to generate the body of the output lambda
@__mogg_intrinsic_attr("mogg.dps_output_fusion_hook")
@no_inline
fn _output_fusion_hook_impl[
    type: DType, rank: Int
](tensor: ManagedTensorSlice[type, rank]):
    @always_inline
    @parameter
    fn _output_lambda[_w: Int](i: IndexList[rank], v: SIMD[type, _w]):
        alias static_specs = _get_tensor_specs_without_lambdas[type, rank]()

        # We use these methods to help with fusion passes which manipulates
        # calls. It is helpful to have a registered function.
        simd_store_into_managed_tensor_slice[
            simd_width=_w,
            alignment = static_specs.alignment,
            address_space = static_specs.address_space,
            static_strides = static_specs.strides,
        ](tensor, i, rebind[SIMD[type, _w]](v))

    _extract_output_lambda[type, rank, _output_lambda]()


# ===----------------------------------------------------------------------=== #
# ManagedTensorSlice class
# ===----------------------------------------------------------------------=== #


@value
@register_passable("trivial")
struct ManagedTensorSlice[
    type: DType,
    rank: Int,
](CollectionElement, TensorLike):
    """ManagedTensorSlice is like TensorSlice but it does not effect the life
    of the underlying allocated pointer. Unlike TensorSlice, when the object
    lifetime ends it does not affect the lifetime of the underlying pointer.
    Conversly, if a ManagedTensorSlice is created, it will not extend the life
    of the underlying pointer.

    Therefore, the user must take care to keep the ptr alive until
    ManagedTensorSlice's last use. This class is useful for writing kernels
    where memory is managed by an external runtime like in MAX's inference
    stack.
    """

    var _ptr: UnsafePointer[Scalar[type]]
    var _spec: RuntimeTensorSpec[type, rank]
    var _runtime_strides: IndexList[rank]

    fn __init__(
        out self,
        ptr: UnsafePointer[Scalar[type]],
        slices: InlineArray[Slice, rank],
        slicer_spec: RuntimeTensorSpec[type, rank],
    ):
        @parameter
        @always_inline
        fn start_fn(slice: Slice) -> Int:
            return slice.start.value()

        @parameter
        @always_inline
        fn stop_fn(slice: Slice) -> Int:
            return slice.end.value()

        @parameter
        @always_inline
        fn step_fn(slice: Slice) -> Int:
            return slice.step.or_else(1)

        var start = _slice_to_tuple[start_fn](slices)
        var stop = _slice_to_tuple[stop_fn](slices)
        var step = _slice_to_tuple[step_fn](slices)

        var adjusted_shape = IndexList[rank]()
        for i in range(rank):
            adjusted_shape[i] = Int(ceil((stop[i] - start[i]) / step[i]))
        var slice_spec = RuntimeTensorSpec[type](adjusted_shape)

        var slicer_strides = _row_major_strides(slicer_spec)
        var start_offset = _dot_prod(start, slicer_strides)

        var strides = IndexList[rank]()

        @parameter
        for i in range(rank):
            strides[i] = step[i] * slicer_strides[i]

        self = Self(ptr.offset(start_offset), slice_spec, strides)

    fn __init__(
        out self,
        ptr: UnsafePointer[Scalar[type]],
        shape: IndexList[rank],
    ):
        self._ptr = ptr
        self._spec = RuntimeTensorSpec[type, rank](shape)
        self._runtime_strides = _row_major_strides(self._spec)

    fn __init__(
        out self,
        ptr: UnsafePointer[Scalar[type]],
        shape: IndexList[rank],
        strides: IndexList[rank],
    ):
        self = Self(
            ptr,
            RuntimeTensorSpec[type, rank](shape),
            strides,
        )

    @implicit
    fn __init__(out self, ndbuffer: NDBuffer[type, rank]):
        """Initializes a ManagedTensorSlice from an NDBuffer.

        Note that forwarding of static shape, strides, and lambdas won't work.
        """
        self = Self(ndbuffer.data, ndbuffer.get_shape())

    @always_inline
    fn __getitem__(self, indices: IndexList[rank]) -> Scalar[type]:
        """Gets the value at the specified indices.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        var offset = _dot_prod(indices, self.strides())
        return self._ptr[offset]

    @always_inline
    fn __getitem__(self, *indices: Int) -> Scalar[type]:
        """Gets the value at the specified indices.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        debug_assert(
            len(indices) == rank, "mismatch between requested index and rank"
        )
        return self[indices]

    @always_inline
    fn __setitem__(self, *indices: Int, val: Scalar[type]):
        """Stores the value at the specified indices.

        Args:
          indices: The indices of the value to store.
          val: The value to store.

        """
        debug_assert(
            len(indices) == rank, "mismatch between requested index and rank"
        )
        self[indices] = val

    @always_inline
    fn __setitem__(self, indices: IndexList[rank], val: Scalar[type]):
        """Stores the value at the specified indices.

        Args:
          indices: The indices of the value to store.
          val: The value to store.

        """
        var offset = _dot_prod(indices, self.strides())
        self._ptr[offset] = val

    fn spec(self) -> TensorSpec:
        return self._spec.get_tensor_spec()

    @always_inline
    fn shape(self) -> IndexList[rank]:
        alias static_shape = specsof[type, rank]("self").shape
        return _make_partially_static_index_list[rank, static_shape](
            self._spec.shape
        )

    @always_inline
    fn dim_size(self, index: Int) -> Int:
        return self.shape()[index]

    @always_inline
    fn dim_size[index: Int](self) -> Int:
        alias static_shape = specsof[type, rank]("self").shape

        @parameter
        if static_shape.at[index]().is_dynamic():
            return self._spec.shape[index]
        else:
            return static_shape.get[index]()

    @always_inline
    fn strides(self) -> IndexList[rank]:
        alias static_shape = specsof[type, rank]("self").strides
        return _make_partially_static_index_list[rank, static_shape](
            self._runtime_strides
        )

    @always_inline
    fn stride_length(self, index: Int) -> Int:
        return self.strides()[index]

    @always_inline
    fn stride_length[index: Int](self) -> Int:
        alias static_strides = specsof[type, rank]("self").strides

        @parameter
        if static_strides.at[index]().is_dynamic():
            return self._runtime_strides[index]
        else:
            return static_strides.get[index]()

    @always_inline
    fn size(self) -> Int:
        """Computes the tensor slice's number of elements.

        Returns:
            The total number of elements in the ManagedTensorSlice.
        """
        var product: Int = 1

        @parameter
        for i in range(rank):
            product *= self.dim_size[i]()

        return product

    @always_inline
    fn unsafe_ptr[__type: DType = type](self) -> UnsafePointer[Scalar[__type]]:
        return rebind[UnsafePointer[Scalar[__type]]](self._ptr)

    @always_inline
    fn load[
        width: Int,
        # Necessary to make it simpler on the call site.
        _rank: Int,
    ](self, index: IndexList[_rank]) -> SIMD[type, width]:
        constrained[_rank == rank]()
        var ridx = rebind[IndexList[rank]](index)
        alias static_specs = specsof[type, rank]("self")
        return simd_load_from_managed_tensor_slice[
            simd_width=width,
            alignment = static_specs.alignment,
            address_space = static_specs.address_space,
            static_strides = static_specs.strides,
        ](self, ridx)

    @always_inline
    fn _fused_load[
        width: Int,
        # Necessary to make it simpler on the call site.
        _rank: Int,
    ](self, index: IndexList[_rank]) -> SIMD[type, width]:
        constrained[_rank == rank]()
        var ridx = rebind[IndexList[rank]](index)

        alias in_lambda = specsof[type, rank]("self").in_lambda
        alias alignment = specsof[type, rank]("self").alignment
        alias address_space = specsof[type, rank]("self").address_space
        alias strides = specsof[type, rank]("self").strides

        @parameter
        if in_lambda:
            alias in_fn = in_lambda.value()
            return in_fn[width](ridx)
        else:
            return simd_load_from_managed_tensor_slice[
                simd_width=width,
                alignment=alignment,
                address_space=address_space,
                static_strides=strides,
            ](self, ridx)

    @always_inline
    fn _compute_offset[
        address_space: AddressSpace, static_strides: DimList
    ](self, index: IndexList[rank]) -> Int:
        @parameter
        if rank == 0:
            return 0

        # Special case for NVidia GPU on shared memory.
        # We can do the offset computation in int32 instead.
        @parameter
        if is_nvidia_gpu() and address_space in (
            _GPUAddressSpace.SHARED,
            _GPUAddressSpace.LOCAL,
            _GPUAddressSpace.CONSTANT,
        ):
            var offset: Int32 = 0

            @parameter
            for i in range(rank):

                @parameter
                if static_strides.at[i]().is_dynamic():
                    offset = fma(
                        Int32(index[i]), Int32(self._runtime_strides[i]), offset
                    )
                else:
                    offset = fma(
                        Int32(index[i]), Int32(static_strides.get[i]()), offset
                    )
            return Int(offset)

        var offset = 0

        @parameter
        for i in range(rank):

            @parameter
            if static_strides.at[i]().is_dynamic():
                offset = fma(index[i], self._runtime_strides[i], offset)
            else:
                offset = fma(index[i], static_strides.get[i](), offset)

        return offset

    @always_inline
    fn store[
        width: Int,
        # Necessary to make it simpler on the call site.
        _rank: Int,
    ](self, index: IndexList[_rank], val: SIMD[type, width]):
        constrained[_rank == rank]()
        var ridx = rebind[IndexList[rank]](index)

        alias static_specs = specsof[type, rank]("self")
        simd_store_into_managed_tensor_slice[
            simd_width=width,
            alignment = static_specs.alignment,
            address_space = static_specs.address_space,
            static_strides = static_specs.strides,
        ](self, ridx, val)

    @always_inline
    fn _fused_store[
        width: Int,
        # Necessary to make it simpler on the call site.
        _rank: Int,
    ](self, index: IndexList[_rank], val: SIMD[type, width]):
        constrained[_rank == rank]()
        var ridx = rebind[IndexList[rank]](index)

        alias out_lambda = specsof[type, rank]("self").out_lambda
        alias alignment = specsof[type, rank]("self").alignment
        alias address_space = specsof[type, rank]("self").address_space
        alias strides = specsof[type, rank]("self").strides

        @parameter
        if out_lambda:
            alias out_fn = out_lambda.value()
            out_fn[width](ridx, val)
        else:
            simd_store_into_managed_tensor_slice[
                simd_width=width,
                alignment=alignment,
                address_space=address_space,
                static_strides=strides,
            ](self, ridx, val)


# ===----------------------------------------------------------------------=== #
# ForEach / view copy primitives
# ===----------------------------------------------------------------------=== #


fn get_kernel_simd_width[type: DType, target: StringLiteral]() -> Int:
    return simdwidthof[type]() if is_cpu[target]() else simdwidthof[
        type, target = _get_gpu_target()
    ]()


# This version of the function supports CPU only. For GPU, use the one with the
# MojoCallContextPtr.
@__mogg_intrinsic_attr("mogg.for_each")
@no_inline
fn foreach[
    type: DType,
    rank: Int, //,
    func: fn[width: Int] (IndexList[rank]) capturing -> SIMD[type, width],
    synchronous: Bool = False,
    target: StringLiteral = "cpu",
    simd_width: Int = get_kernel_simd_width[type, target](),
](tensor: ManagedTensorSlice[type, rank]):
    @parameter
    @always_inline
    fn elementwise_fn_wrapper[
        width: Int, rank: Int
    ](index: IndexList[rank]) capturing:
        var val = func[width](rebind[IndexList[tensor.rank]](index))
        tensor._fused_store(index, val)

    algorithm.functional.elementwise[
        elementwise_fn_wrapper,
        simd_width,
        use_blocking_impl=synchronous,
        target=target,
    ](tensor.shape())


@__mogg_intrinsic_attr("mogg.for_each")
@no_inline
fn foreach[
    type: DType,
    rank: Int, //,
    func: fn[width: Int] (IndexList[rank]) capturing -> SIMD[type, width],
    synchronous: Bool = False,
    target: StringLiteral = "cpu",
    simd_width: Int = get_kernel_simd_width[type, target](),
](tensor: ManagedTensorSlice[type, rank], ctx: MojoCallContextPtr):
    @parameter
    @always_inline
    fn elementwise_fn_wrapper[
        width: Int, rank: Int
    ](index: IndexList[rank]) capturing:
        var val = func[width](rebind[IndexList[tensor.rank]](index))
        tensor._fused_store(index, val)

    algorithm.functional.elementwise[
        elementwise_fn_wrapper,
        simd_width,
        use_blocking_impl=synchronous,
        target=target,
    ](tensor.shape(), ctx)


# TensorCopy intrinsic used by view kernels.
# z is a kernel output, and x a view of the input.
@no_inline
fn view_copy_impl[
    synchronous: Bool,
    target: StringLiteral,
    type: DType,
    rank: Int,
    view_strides: DimList = DimList.create_unknown[rank](),
](
    z: ManagedTensorSlice[type, rank],
    x: ManagedTensorSlice[type, rank],
    ctx: MojoCallContextPtr,
):
    @parameter
    @always_inline
    fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.type, width]:
        return simd_load_from_managed_tensor_slice[
            simd_width=width, static_strides=view_strides
        ](x, idx)

    foreach[func, synchronous, target](z, ctx)
