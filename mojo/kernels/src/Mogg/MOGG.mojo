# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Activations import relu, gelu, sigmoid
from Buffer import NDBuffer
from DType import DType
from Functional import (
    _elementwise_impl,
    unroll,
    vectorize,
    vectorize_unroll,
    async_parallelize,
)
from Reductions import _reduce_generator
from Intrinsics import strided_load
from Index import Index, StaticIntTuple
from Memory import memset_zero
from IO import print
from List import Dim, DimList, VariadicList
from LLCL import Runtime, OutputChainPtr, OwningOutputChainPtr
from Math import (
    add,
    div,
    ceil,
    erf,
    exp,
    equal,
    floor,
    greater,
    greater_equal,
    isinf,
    isnan,
    pow,
    max,
    min,
    mul,
    not_equal,
    rsqrt,
    select,
    sqrt,
    sub,
    tanh,
    fma,
    abs,
    log1p,
)
from Matmul import matmul_parallel_async
from BatchedMatmul import (
    batched_matmul_parallel_async,
    get_trace_information as get_trace_information_batched_matmul,
)
from MatmulUtils import GemmShape, get_trace_information
from Pointer import Pointer, DTypePointer
from Range import range
from SIMD import SIMD
from TargetInfo import simdwidthof
from Tracing import Trace, TraceLevel
from TypeUtilities import rebind
from String import String
from Slice import slice_as_view
from MatrixSolve import matrix_solve as _matrix_solve
from Index import Index


# Prevent these functions from being DCE'd by explicitly exporting them.
@export
fn MOGGExport():
    alias _indices = TensorIndicesTypeDef
    alias _out_chain = OutputChainPtrDef
    alias _simd_typedef = SimdTypeDef
    alias _index_typedef = IndexTypeDef
    alias _dtype_float32 = DTypeFloat32TypeDef
    alias _dtype_float64 = DTypeFloat64TypeDef
    alias _dtype_si8 = DTypeInt8TypeDef
    alias _dtype_si16 = DTypeInt16TypeDef
    alias _dtype_si32 = DTypeInt32TypeDef
    alias _dtype_si64 = DTypeInt64TypeDef
    alias _dtype_ui32 = DTypeUInt32TypeDef
    alias _dtype_ui16 = DTypeUInt16TypeDef
    alias _dtype_ui8 = DTypeUInt8TypeDef
    alias _dtype_bool = DTypeBoolTypeDef
    alias _to_buffer = to_buffer
    alias _to_shape = to_shape
    alias _abs = abs_wrapped
    alias _add = add
    alias _cast = cast
    alias _ceil = ceil
    alias _concat = mogg_concat
    alias _div = div
    alias _erf = erf
    alias _exp = exp
    alias _equal = equal
    alias _floor = floor
    alias _gather = _gather_with_lambdas
    alias _gelu = gelu
    alias _greater = greater
    alias _greater_equal = greater_equal
    alias _isinf = isinf
    alias _isnan = isnan
    alias _log1p = log1p
    alias _pow = pow_wrapped
    alias _load_scalar = load_scalar
    alias _mean = mean
    alias _matrix_solve = matrix_solve
    alias _matmul = matmul
    alias _batched_matmul = batched_matmul
    alias _mogg_max = mogg_max
    alias _mogg_min = mogg_min
    alias _mul = mul
    alias _not_equal = not_equal
    alias _rsqrt = rsqrt
    alias _select = select
    alias _sigmoid = sigmoid
    alias _sqrt = sqrt
    alias _sub = sub
    alias _tanh = tanh
    alias _relu = relu
    alias _reshape = reshape
    alias _broadcast = broadcast_to_tensor
    alias _slice = slice
    alias _simd_load = simd_load
    alias _simd_store = simd_store
    alias _simd_load_1D = simd_load_1D
    alias _simd_load_splat = simd_load_splat
    alias _simd_load_maybe_splat = simd_load_maybe_splat
    alias _simd_load_strided = simd_load_strided
    alias _simd_target = get_target_simd
    alias _simd_width_to_int = simd_width_to_int
    alias _sum = sum
    alias _splat = splat
    alias _transpose = transpose
    alias _elementwise = elementwise_wrapper
    alias _print_shape_info = print_buffer_info
    alias _mark_output_chain_ready = mark_output_chain_ready

    alias _test_many_ranks_and_types = test_many_ranks_and_types
    alias _test_one_rank_many_tensor = test_one_rank_many_tensor
    alias _test_3D_in_out_lambda = test_3D_in_out_lambda


# ===----------------------------------------------------------------------===#
# Nop functions to expose different types to the compiler.
# ===----------------------------------------------------------------------===#


fn DTypeFloat32TypeDef(ty: DType.type) -> DType.type:
    return DType.float32.value


fn DTypeFloat64TypeDef(ty: DType.type) -> DType.type:
    return DType.float64.value


fn DTypeInt8TypeDef(ty: DType.type) -> DType.type:
    return DType.int8.value


fn DTypeInt16TypeDef(ty: DType.type) -> DType.type:
    return DType.int16.value


fn DTypeInt32TypeDef(ty: DType.type) -> DType.type:
    return DType.int32.value


fn DTypeUInt32TypeDef(ty: DType.type) -> DType.type:
    return DType.uint32.value


fn DTypeInt64TypeDef(ty: DType.type) -> DType.type:
    return DType.int64.value


fn DTypeUInt8TypeDef(ty: DType.type) -> DType.type:
    return DType.uint8.value


fn DTypeUInt16TypeDef(ty: DType.type) -> DType.type:
    return DType.uint16.value


fn DTypeBoolTypeDef(ty: DType.type) -> DType.type:
    return DType.bool.value


fn IndexTypeDef(ty: Int) -> Int:
    return ty


fn OutputChainPtrDef(ty: OutputChainPtr) -> OutputChainPtr:
    return ty


fn SimdTypeDef[
    type: DType, width: Int
](ty: SIMD[type, width]) -> SIMD[type, width]:
    return ty


fn TensorIndicesTypeDef[
    rank: Int
](ty: StaticIntTuple[rank]) -> StaticIntTuple[rank]:
    return ty


# ===----------------------------------------------------------------------===#
# Basic generated kernel building blocks
# ===----------------------------------------------------------------------===#


@always_inline
fn to_buffer[
    type: DType, rank: Int
](
    data: __mlir_type[`!pop.pointer<scalar<`, type.value, `>>`],
    shape: __mlir_type.`!pop.pointer<index>`,
) -> NDBuffer[rank, DimList.create_unknown[rank](), type]:
    let shape_ptr = Pointer(shape)
    var shape_tuple = StaticIntTuple[rank]()

    var stride_tuple = StaticIntTuple[rank]()
    var stride: Int = 1

    @always_inline
    @parameter
    fn body[idx: Int]():
        # Start from the back so we can accumulate the strides.
        let i = rank - 1 - idx
        shape_tuple[i] = shape_ptr.load(i)
        stride_tuple[i] = stride
        stride *= shape_tuple[i]

    unroll[rank, body]()

    return NDBuffer[rank, DimList.create_unknown[rank](), type](
        DTypePointer[type](data), shape_tuple, type, stride_tuple
    )


@always_inline
fn to_shape[
    rank: Int
](shape: __mlir_type.`!pop.pointer<index>`,) -> StaticIntTuple[rank]:
    let shape_ptr = Pointer(shape)
    var shape_tuple = StaticIntTuple[rank]()

    @always_inline
    @parameter
    fn body[idx: Int]():
        shape_tuple[idx] = shape_ptr.load(idx)

    unroll[rank, body]()

    return shape_tuple


@always_inline
fn elementwise_wrapper[
    trace_description: StringLiteral,
    simd_width: Int,
    type: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
    func: fn[width: Int, rank: Int] (StaticIntTuple[rank]) capturing -> None,
](
    buffer: NDBuffer[rank, DimList.create_unknown[rank](), type],
    out_chain: OutputChainPtr,
):
    alias unroll_factor: Int = 1

    @always_inline
    @parameter
    fn description_fn() -> String:
        let name_str = String("name=") + trace_description
        let shape_str = String("shape=") + String("x").join(buffer.get_shape())

        let unroll_factor_str = String("unroll_factor=") + unroll_factor
        let vector_width_str = String("vector_width=") + simd_width

        let res = String(";").join(
            name_str, shape_str, unroll_factor_str, vector_width_str
        )

        return res

    out_chain.trace[TraceLevel.OP, description_fn]("mojo.elementwise")

    _elementwise_impl[
        rank, simd_width, unroll_factor, single_thread_blocking_override, func
    ](
        buffer.dynamic_shape,
        out_chain,
    )


# ===----------------------------------------------------------------------===#
# Simd load/store helper functions
# ===----------------------------------------------------------------------===#


@always_inline
fn _compute_flat_index[
    type: DType, rank: Int, iters: Int
](
    buffer: NDBuffer[rank, DimList.create_unknown[rank](), type],
    index: StaticIntTuple[rank],
) -> Int:
    var flat_index: Int = 0

    @always_inline
    @parameter
    fn body[idx: Int]():
        flat_index = fma(index[idx], buffer.dynamic_stride[idx], flat_index)

    unroll[iters, body]()
    return flat_index


# If we know the tensor is 1D then we can avoid the stride calculation. If
# the stride is 0 then we just splat the value. Hopefully LLVM is able to hoist
# this `if` as it should be a constant.
@always_inline
fn simd_load_1D[
    type: DType, simd_width: Int, rank: Int
](
    buffer: NDBuffer[rank, DimList.create_unknown[rank](), type],
    index: StaticIntTuple[rank],
) -> SIMD[type, simd_width]:
    let stride = buffer.dynamic_stride[rank - 1]
    if stride == 0:
        let scalar = load_scalar[type, rank](buffer)
        return splat[type, simd_width](scalar)

    let i = stride * index[rank - 1]
    return _simd_load_internal[simd_width, type, rank](buffer, i)


# If we know the tensor is actually a scalar tensor we can avoid all indexing
# calculation. It's broken into the two parts (load followed by splat) so we can
# hoist the load from the lambda body.
@always_inline
fn load_scalar[
    type: DType, rank: Int
](buffer: NDBuffer[rank, DimList.create_unknown[rank](), type]) -> SIMD[
    type, 1
]:
    @parameter
    if type == DType.bool:
        let v = buffer.data.bitcast[DType.uint8]().load(0)
        return v.cast[type]()
    else:
        return buffer.data.load(0)


@always_inline
fn _simd_load_internal[
    simd_width: Int, type: DType, rank: Int
](
    buffer: NDBuffer[rank, DimList.create_unknown[rank](), type], index: Int
) -> SIMD[type, simd_width]:
    @parameter
    if type == DType.bool:
        let v = buffer.data.bitcast[DType.uint8]().simd_load[simd_width](index)
        return v.cast[type]()
    else:
        return buffer.data.simd_load[simd_width](index)


@always_inline
fn splat[
    type: DType,
    simd_width: Int,
](val: SIMD[type, 1]) -> SIMD[type, simd_width]:
    return SIMD[type, simd_width].splat(val)


# Load a tensor which might splat along the last dimension.
@always_inline
fn simd_load_maybe_splat[
    type: DType, simd_width: Int, rank: Int
](
    buffer: NDBuffer[rank, DimList.create_unknown[rank](), type],
    index: StaticIntTuple[rank],
) -> SIMD[type, simd_width]:
    let flat_index = _compute_flat_index[type, rank, rank](buffer, index)

    if buffer.dynamic_stride[rank - 1] == 0:
        return buffer.data.load(flat_index)

    return _simd_load_internal[simd_width, type, rank](buffer, flat_index)


# Load a tensor which does a splat along the last dimension.
@always_inline
fn simd_load_splat[
    type: DType, simd_width: Int, rank: Int
](
    buffer: NDBuffer[rank, DimList.create_unknown[rank](), type],
    index: StaticIntTuple[rank],
) -> SIMD[type, simd_width]:
    # Last dimension will be 0 for splats so don't compute last dim.
    let flat_index = _compute_flat_index[type, rank, rank - 1](buffer, index)

    return buffer.data.load(flat_index)


@always_inline
fn simd_load[
    type: DType, simd_width: Int, rank: Int
](
    buffer: NDBuffer[rank, DimList.create_unknown[rank](), type],
    index: StaticIntTuple[rank],
) -> SIMD[type, simd_width]:
    let flat_index = _compute_flat_index[type, rank, rank](buffer, index)
    return _simd_load_internal[simd_width, type, rank](buffer, flat_index)


@always_inline
fn simd_load_strided[
    type: DType, simd_width: Int, rank: Int
](
    buffer: NDBuffer[rank, DimList.create_unknown[rank](), type],
    index: StaticIntTuple[rank],
) -> SIMD[type, simd_width]:
    let flat_index = _compute_flat_index[type, rank, rank](buffer, index)

    let stride = buffer.dynamic_stride[rank - 1]

    # We aren't loading from something of stride == 1 or stride == 0 then
    # we have to use a gather load unfortunately.
    if stride > 1:

        @parameter
        if type == DType.bool:
            let v = strided_load[DType.uint8, simd_width](
                buffer.data.bitcast[DType.uint8]().offset(flat_index), stride
            )
            return v.cast[type]()
        else:
            return strided_load[type, simd_width](
                buffer.data.offset(flat_index), stride
            )
    else:
        return _simd_load_internal[simd_width, type, rank](buffer, flat_index)


@always_inline
fn simd_store[
    type: DType, simd_width: Int, rank: Int
](
    buffer: NDBuffer[rank, DimList.create_unknown[rank](), type],
    index: StaticIntTuple[rank],
    val: SIMD[type, simd_width],
):
    let flat_index = _compute_flat_index[type, rank, rank](buffer, index)

    # We have to cast bools into their runtime storage type.
    @parameter
    if type == DType.bool:
        let v = val.cast[DType.uint8]()
        buffer.data.bitcast[DType.uint8]().simd_store[simd_width](flat_index, v)
    else:
        buffer.data.simd_store[simd_width](flat_index, val)


# ===----------------------------------------------------------------------===#
# Broadcast
# ===----------------------------------------------------------------------===#


@always_inline
fn broadcast_to_tensor[
    type: DType,
    target_type: DType,
    original_rank: Int,
    target_rank: Int,
    output_rank: Int,
    single_thread_blocking_override: Bool,
](
    original: NDBuffer[
        original_rank, DimList.create_unknown[original_rank](), type
    ],
    target: NDBuffer[
        target_rank, DimList.create_unknown[target_rank](), target_type
    ],
) -> NDBuffer[output_rank, DimList.create_unknown[output_rank](), type]:
    var shape = StaticIntTuple[output_rank]()
    var stride = StaticIntTuple[output_rank]()

    # The offset from where the implicit new dimensions end. I.E broadcasting
    # <1, 1> to <40,40,40,40> the two dimensions at the start are new
    # dimensions and then the two ones are broadcasted.
    var offset: Int = 0

    # New dimensions are always broadcast.
    @always_inline
    @parameter
    fn add_new_dims[i: Int]():
        @parameter
        if target_rank >= original_rank:
            shape[i] = target.dim(i)
            stride[i] = 0
        else:
            shape[i] = original.dim(i)
            stride[i] = original.stride(i)

    # Broadcast in dimensions the original started with.
    @always_inline
    @parameter
    fn broadcast_dim[small_index: Int]():
        # We are traversing as if they are the same size.
        let big_index = small_index + offset

        # Switch the indexes depending on which is bigger.
        var orig_index = small_index
        var target_index = big_index

        @parameter
        if target_rank < original_rank:
            orig_index = big_index
            target_index = small_index

        # If the dims are the same use the stride of the original.
        if original.dim(orig_index) == target.dim(target_index):
            stride[big_index] = original.stride(orig_index)
            shape[big_index] = original.dim(orig_index)
        elif original.dim(orig_index) == 1:
            # If they don't match and original is 1 then we broadcast.
            stride[big_index] = 0
            shape[big_index] = target.dim(target_index)
        else:
            # The target is the one being broadcast here so we don't need to
            # change the strides but we do need to restore the old shape.
            shape[big_index] = original.dim(orig_index)
            stride[big_index] = original.stride(orig_index)

    # Broadcast by appending new dimensions to the front if the sizes are not
    # the same then broadcast each of remaining dimensions. We represent this
    # using the unroll construct to help codegeneration.
    @parameter
    if target_rank < original_rank:
        unroll[original_rank - target_rank, add_new_dims]()
        offset = original_rank - target_rank
        unroll[target_rank, broadcast_dim]()
    else:
        unroll[target_rank - original_rank, add_new_dims]()
        offset = target_rank - original_rank
        unroll[original_rank, broadcast_dim]()

    # Create a view of the original data with the new shape and strides.
    var out = NDBuffer[
        output_rank, DimList.create_unknown[output_rank](), type
    ](
        original.data,
        shape,
        original.dynamic_dtype,
        stride,
    )

    return out


# When we have many SIMD types in one kernel we need to use the `min` of them.
# This involves applying parameter expressions to this result which must be
# `mlir.index` typed so we need to return as `mlir.index` and then cast to int.
fn get_target_simd[type: DType]() -> __mlir_type.index:
    return simdwidthof[type]().value


fn simd_width_to_int[simd_width: __mlir_type.index]() -> Int:
    return Int(simd_width)


# ===----------------------------------------------------------------------===#
# Abs wrapper op
# ===----------------------------------------------------------------------===#


# Call abs, needed as it has multiple overloads which can't be aliased
@always_inline
fn abs_wrapped[
    type: DType, simd_width: Int
](value: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return abs(value)


# ===----------------------------------------------------------------------===#
# Cast op
# ===----------------------------------------------------------------------===#


# Cast a SIMD value to a new SIMD value of different type.
@always_inline
fn cast[
    type: DType, new_type: DType, simd_width: Int
](value: SIMD[type, simd_width],) -> SIMD[new_type, simd_width]:
    return value.cast[new_type]()


@always_inline
fn mogg_concat[
    type: DType,
    rank: Int,
    simd_width: Int,
    single_thread_blocking_override: Bool,
    axis_type: DType,
](
    output: NDBuffer[rank, DimList.create_unknown[rank](), type],
    axis: NDBuffer[1, DimList.create_unknown[1](), axis_type],
    out_chain: OutputChainPtr,
    *variadic_ins: NDBuffer[rank, DimList.create_unknown[rank](), type],
):
    let axis_int = axis[0].to_int()
    let inputs = VariadicList(variadic_ins)

    @parameter
    @always_inline
    fn concat_lambda[
        simd_width: Int, rank: Int
    ](out_index: StaticIntTuple[rank]):

        # Concating [:, 10, :], [:, 20, :], [:, 30, :] results in shape
        # [:, 60, :] so when the target dim is:
        #   0 >= target_dim < 10: We are loading from first input.
        #   10 >= target_dim < 20: We are loading from second input.
        #   20 >= target_dim < 30: We are loading from third input.
        # The output will always be storing to the full index but we load from
        # an offset.

        var target_dim = out_index[axis_int]

        # Iterate through the inputs to find the one we should be storing to.
        for i in range(inputs.__len__()):
            let input = inputs[i]

            # This is the input we should be loading/storing.
            if target_dim < input.dynamic_shape[axis_int]:
                var in_index = out_index
                in_index[axis_int] = target_dim
                let load = simd_load[type, simd_width, rank](
                    rebind[
                        NDBuffer[rank, DimList.create_unknown[rank](), type]
                    ](input),
                    in_index,
                )
                simd_store[type, simd_width, rank](
                    rebind[
                        NDBuffer[rank, DimList.create_unknown[rank](), type]
                    ](output),
                    out_index,
                    load,
                )
                return
            else:
                # Keep looking...
                target_dim -= input.dynamic_shape[axis_int]

    alias unroll_factor = 1

    # We need to check it's safe to simd_load from each input.
    var inputs_simd_aligned = True
    for i in range(inputs.__len__()):
        if inputs[i].dynamic_shape[rank - 1] % simd_width != 0:
            inputs_simd_aligned = False

    # If we are concat'ing along the last dimension we can do a simd load.
    if axis_int == rank - 1 and inputs_simd_aligned:
        _elementwise_impl[
            rank,
            simd_width,
            unroll_factor,
            single_thread_blocking_override,
            concat_lambda,
        ](
            output.dynamic_shape,
            out_chain,
        )
    else:
        # Otherwise we must run scalar.
        _elementwise_impl[
            rank,
            1,
            unroll_factor,
            single_thread_blocking_override,
            concat_lambda,
        ](
            output.dynamic_shape,
            out_chain,
        )

    # If we aren't using the trivial kernel we actually still have to wait.
    # The variadics fall off the stack when captured by the lambda.
    @parameter
    if not single_thread_blocking_override:
        out_chain.wait()


# ===----------------------------------------------------------------------===#
# Pow wrapper op
# ===----------------------------------------------------------------------===#


# Call pow, needed as it has multiple overloads which can't be aliased
@always_inline
fn pow_wrapped[
    type: DType, power_type: DType, simd_width: Int
](value: SIMD[type, simd_width], power: SIMD[power_type, simd_width]) -> SIMD[
    type, simd_width
]:
    return pow[type, power_type, simd_width](value, power)


# ===----------------------------------------------------------------------===#
# Max & min ops
# ===----------------------------------------------------------------------===#

# These need wrappers as we can't take an alias of the ambigious overload.


@always_inline
fn mogg_max[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    type, simd_width
]:
    return max(x, y)


@always_inline
fn mogg_min[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    type, simd_width
]:
    return min(x, y)


# ===----------------------------------------------------------------------===#
# Mean op
# ===----------------------------------------------------------------------===#


# Cast a SIMD value to a new SIMD value of different type.
@always_inline
fn mean[
    type: DType,
    index_type: DType,
    rank: Int,
    simd_width: Int,
    single_thread_blocking_override: Bool,
    input_0_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    input_1_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
](
    input_buffer: NDBuffer[rank, DimList.create_unknown[rank](), type],
    axis_buffer: NDBuffer[1, DimList.create_unknown[1](), index_type],
    output_buffer: NDBuffer[rank, DimList.create_unknown[rank](), type],
    out_chain: OutputChainPtr,
):
    out_chain.trace[TraceLevel.OP]("mogg.mean")

    # Only one reduce dimension supported currently, it must be deduced from
    # the attached input lambda rather than read directly.
    let zero_idx = StaticIntTuple[1]()
    var reduce_dim = input_1_fn[index_type, 1, 1](zero_idx).to_int()
    if reduce_dim < 0:
        reduce_dim = rank + reduce_dim

    @always_inline
    fn reduce_impl[
        ty: DType, width: Int
    ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
        return v1 + v2

    # For floats apply the reciprocal as a multiply.
    @parameter
    if type == DType.float32 or type == DType.float64 or type == DType.float16:
        # Apply mean division before storing to the output lambda.
        let reciprocal = 1.0 / input_buffer.dim(reduce_dim)

        @always_inline
        @parameter
        fn wrapped_output_mul[
            type: DType, width: Int, rank: Int
        ](indices: StaticIntTuple[rank], value: SIMD[type, width]):
            let mean_val = value * reciprocal
            output_0_fn[type, width, rank](indices, mean_val)

        _reduce_generator[
            type,
            rank,
            simdwidthof[type](),
            single_thread_blocking_override,
            input_0_fn,
            wrapped_output_mul,
            reduce_impl,
        ](input_buffer, 0, reduce_dim, out_chain)

    else:
        # For ints just a normal divide.
        let dim_size = input_buffer.dim(reduce_dim)

        @always_inline
        @parameter
        fn wrapped_output_div[
            type: DType, width: Int, rank: Int
        ](indices: StaticIntTuple[rank], value: SIMD[type, width]):
            let mean_val = value / dim_size
            output_0_fn[type, width, rank](indices, mean_val)

        _reduce_generator[
            type,
            rank,
            simdwidthof[type](),
            single_thread_blocking_override,
            input_0_fn,
            wrapped_output_div,
            reduce_impl,
        ](input_buffer, 0, reduce_dim, out_chain)


# ===----------------------------------------------------------------------===#
# Sum op
# ===----------------------------------------------------------------------===#


# Cast a SIMD value to a new SIMD value of different type.
@always_inline
fn sum[
    type: DType,
    index_type: DType,
    rank: Int,
    simd_width: Int,
    single_thread_blocking_override: Bool,
    input_0_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    input_1_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
](
    input_buffer: NDBuffer[rank, DimList.create_unknown[rank](), type],
    axis_buffer: NDBuffer[1, DimList.create_unknown[1](), index_type],
    output_buffer: NDBuffer[rank, DimList.create_unknown[rank](), type],
    out_chain: OutputChainPtr,
):
    out_chain.trace[TraceLevel.OP]("mogg.sum")

    # Only one reduce dimension supported currently, it must be deduced from
    # the attached input lambda rather than read directly.
    let zero_idx = StaticIntTuple[1]()
    var reduce_dim = input_1_fn[index_type, 1, 1](zero_idx).to_int()
    if reduce_dim < 0:
        reduce_dim = rank + reduce_dim

    @always_inline
    fn reduce_impl[
        ty: DType, width: Int
    ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
        return v1 + v2

    _reduce_generator[
        type,
        rank,
        simdwidthof[type](),
        single_thread_blocking_override,
        input_0_fn,
        output_0_fn,
        reduce_impl,
    ](input_buffer, 0, reduce_dim, out_chain)


# Slice op
# ===----------------------------------------------------------------------===#

# Wrapper for slice here to include the `single_thread_blocking_override`.
@always_inline
fn slice[
    type: DType,
    index_type: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
](
    tensor: NDBuffer[rank, DimList.create_unknown[rank](), type],
    starts: NDBuffer[1, DimList.create_unknown[1](), index_type],
    ends: NDBuffer[1, DimList.create_unknown[1](), index_type],
    steps: NDBuffer[1, DimList.create_unknown[1](), index_type],
) -> NDBuffer[rank, DimList.create_unknown[rank](), type]:
    return slice_as_view(tensor, starts, ends, steps)


# ===----------------------------------------------------------------------===#
# Reshape op
# ===----------------------------------------------------------------------===#

# Reshape assumes inputs are contiguous. It should always be fused last and
# a non-contiguous tensor cannot be fused *into* this as input.
@always_inline
fn reshape[
    rank: Int,
    output_rank: Int,
    type: DType,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    new_shape: StaticIntTuple[output_rank],
) -> NDBuffer[output_rank, DimList.create_unknown[output_rank](), type]:

    var stride_tuple = StaticIntTuple[output_rank]()
    var stride: Int = 1

    # Create contiguous strides.
    @always_inline
    @parameter
    fn body[idx: Int]():
        # Start from the back so we can accumulate the strides.
        let i = output_rank - 1 - idx
        stride_tuple[i] = stride
        stride *= new_shape[i]

    unroll[output_rank, body]()

    # Return the a view with the new shape.
    return NDBuffer[output_rank, DimList.create_unknown[output_rank](), type](
        input.data, new_shape, input.dynamic_dtype, stride_tuple
    )


# ===----------------------------------------------------------------------===#
# Transpose op
# ===----------------------------------------------------------------------===#


@always_inline
fn transpose[
    rank: Int,
    type: DType,
    int_type: DType,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    perms: NDBuffer[1, DimList.create_unknown[1](), int_type],
) -> NDBuffer[rank, DimList.create_unknown[rank](), type]:
    var new_shape = StaticIntTuple[rank]()
    var new_stride = StaticIntTuple[rank]()

    @always_inline
    @parameter
    fn body[i: Int]():
        let dim = perms[i].to_int()
        new_shape[i] = input.dynamic_shape[dim]
        new_stride[i] = input.dynamic_stride[dim]

    unroll[rank, body]()

    # Create the transposed view.
    return NDBuffer[rank, DimList.create_unknown[rank](), type](
        input.data, new_shape, input.dynamic_dtype, new_stride
    )


# ===----------------------------------------------------------------------===#
# Gather
# ===----------------------------------------------------------------------===#


@always_inline
fn _gather_with_lambdas[
    type: DType,
    in_rank: Int,
    indices_type: DType,
    indices_rank: Int,
    axis_type: DType,
    output_rank: Int,
    simd_width: Int,
    single_thread_blocking_override: Bool,
    input_0_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    input_1_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    input_2_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
](
    input: NDBuffer[in_rank, DimList.create_unknown[in_rank](), type],
    indices: NDBuffer[
        indices_rank, DimList.create_unknown[indices_rank](), indices_type
    ],
    axis_buf: NDBuffer[1, DimList.create_unknown[1](), axis_type],
    output: NDBuffer[output_rank, DimList.create_unknown[output_rank](), type],
    out_chain: OutputChainPtr,
):
    # Look through the lambda to pull the index out.
    let zero_idx = StaticIntTuple[1]()
    let axis = input_2_fn[axis_type, 1, 1](zero_idx).to_int()

    @parameter
    @always_inline
    fn gather_lambda[simd_width: Int, rank: Int](idx: StaticIntTuple[rank]):
        # Get the gather indices.
        var indices_index = StaticIntTuple[indices_rank]()

        # The gather can be in the form of:
        # InTensor = <10, 20, 30, 40>
        # Indices = <3, 6, 9>
        # Axis = 1
        # In this case we are inserting 3 new dimensions so the output shape is:
        # InTensor<10, 3, 6, 9, 30, 40>
        #

        # So the index that we access the indices tensor at should be the sub
        # indices over the input within the range of axis -> indices rank.
        # Accessing the output at indices <1, 2, 3, 8, 4, 6>
        # Would access the indices at <2, 3, 8>

        # Get the indices of the index.
        @always_inline
        @parameter
        fn indices_get[unrolled_i: Int]():
            indices_index[unrolled_i] = idx[unrolled_i + axis]

        unroll[indices_rank, indices_get]()

        # The index we are gathering.
        let data_index = input_1_fn[indices_type, 1, indices_rank](
            indices_index
        ).to_int()

        # Update the indices with the new data index.
        var data_indices = StaticIntTuple[in_rank]()

        let skip_factor = indices_rank - 1

        # Build the indices for the input. We have replaced in index in 'axis'
        # with an index from the indices tensor.
        @always_inline
        @parameter
        fn input_indices_get[unrolled_i: Int]():
            indices_index[unrolled_i] = idx[unrolled_i + axis]
            if unrolled_i == axis:
                data_indices[unrolled_i] = data_index
            elif unrolled_i > axis:
                # Skip over any extra indices dimensions. These are essentially new dimensions.
                data_indices[unrolled_i] = idx[unrolled_i + skip_factor]
            else:
                data_indices[unrolled_i] = idx[unrolled_i]

        unroll[in_rank, input_indices_get]()

        # Load the the data.
        let data = input_0_fn[type, simd_width, in_rank](data_indices)

        # Store it to the original index.
        output_0_fn[type, simd_width, rank](idx, data)

    alias unroll_factor = 1

    # If we are gathering on the last dimension then we have to be scalar.
    if axis == in_rank - 1:
        _elementwise_impl[
            output_rank,
            1,
            unroll_factor,
            single_thread_blocking_override,
            gather_lambda,
        ](
            output.dynamic_shape,
            out_chain,
        )
    else:
        _elementwise_impl[
            output_rank,
            simd_width,
            unroll_factor,
            single_thread_blocking_override,
            gather_lambda,
        ](
            output.dynamic_shape,
            out_chain,
        )


# ===----------------------------------------------------------------------===#
# MOGG matmul
# ===----------------------------------------------------------------------===#


@always_inline
fn soft_fusion_run_wrapper[
    single_thread_blocking_override: Bool,
    func: fn (OutputChainPtr) capturing -> None,
](out_chain: OutputChainPtr):
    """Runs func with the async behaviour expected by single_thread_blocking_override.

    If single_thread_blocking_override is true, we want to run the func
    synchronously without signaling the out_chain since the out_chain represents
    the out_chain for the entire soft-fused kernel.

    Else if single_thread_blocking_override is false, run the kernel asynchronously
    and signal the out_chain when done.
    """

    @parameter
    if single_thread_blocking_override:
        let new_chain = OwningOutputChainPtr(out_chain.get_runtime())
        func(new_chain.borrow())
        new_chain.wait()
    else:
        func(out_chain)


# TODO: move into kernel lib
@always_inline
fn small_matmul[
    type: DType,
    simd_width: Int,
    epilogue_wrapper: fn[type: DType, width: Int] (
        StaticIntTuple[2], SIMD[type, width]
    ) capturing -> None,
](
    a: NDBuffer[2, DimList.create_unknown[2](), type],
    b: NDBuffer[2, DimList.create_unknown[2](), type],
    c: NDBuffer[2, DimList.create_unknown[2](), type],
):
    let M = a.dim[0]()
    let N = b.dim[1]()
    let K = a.dim[1]()

    alias unroll_factor = 2  # don't unroll too much since this is for tiny shapes

    @parameter
    @always_inline
    fn normal_update[
        inner_type: DType, width: Int
    ](coords: StaticIntTuple[2], val: SIMD[inner_type, width]):
        c.simd_store[width](
            Index(coords[0], coords[1]), rebind[SIMD[type, width]](val)
        )

    @parameter
    @always_inline
    fn last_update[
        type: DType, width: Int
    ](coords: StaticIntTuple[2], val: SIMD[type, width]):
        epilogue_wrapper[type, width](coords, val)

    @always_inline
    @parameter
    fn accum_out_row[
        output_func: fn[type: DType, width: Int] (
            StaticIntTuple[2], SIMD[type, width]
        ) capturing -> None,
    ](m: Int, k: Int):
        let a_val = a[m, k]

        @always_inline
        @parameter
        fn _wrapper[simd_width: Int](n: Int):
            output_func[type, simd_width](
                Index(m, n),
                c.simd_load[simd_width](m, n)
                + a_val * b.simd_load[simd_width](k, n),
            )

        vectorize_unroll[simd_width, unroll_factor, _wrapper](N)

    for m in range(M):
        memset_zero(c.data + m * N, N)
        for k in range(K - 1):
            accum_out_row[normal_update](m, k)
        accum_out_row[last_update](m, K - 1)
    return


@always_inline
fn matmul[
    type: DType,
    transpose_in_1: Bool,  # matches name of MO attribute
    single_thread_blocking_override: Bool,
    output_0_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
](
    a: NDBuffer[2, DimList.create_unknown[2](), type],
    b: NDBuffer[2, DimList.create_unknown[2](), type],
    c: NDBuffer[2, DimList.create_unknown[2](), type],
    out_chain: OutputChainPtr,
):
    alias transpose_a = False
    alias transpose_b = transpose_in_1
    alias b_packed = False
    alias simd_width = simdwidthof[type]()

    @parameter
    @always_inline
    fn epilogue_wrapper[
        type: DType, width: Int
    ](coords: StaticIntTuple[2], val: SIMD[type, width]):
        output_0_fn[type, width, 2](coords, val)

    @parameter
    if (
        single_thread_blocking_override
        and not transpose_a
        and not transpose_b
        and not b_packed
    ):
        return small_matmul[type, simd_width, epilogue_wrapper](a, b, c)

    @always_inline
    @parameter
    fn description_fn() -> String:
        return get_trace_information(
            "dynamic_tile",
            GemmShape.get[
                transpose_a,
                transpose_b,
            ](c, a, b),
            transpose_a,
            transpose_b,
            b_packed,
        )

    out_chain.trace[TraceLevel.OP, description_fn]("mojo.mogg.matmul")

    @parameter
    @always_inline
    fn func(chain: OutputChainPtr):
        matmul_parallel_async[
            type, transpose_a, transpose_b, b_packed, True, epilogue_wrapper
        ](
            c,
            a,
            b,
            chain,
        )

    soft_fusion_run_wrapper[single_thread_blocking_override, func](out_chain)


# ===----------------------------------------------------------------------===#
# MOGG batched matmul
# ===----------------------------------------------------------------------===#


@always_inline
fn batched_matmul[
    rank: Int,
    type: DType,
    single_thread_blocking_override: Bool,
](
    a: NDBuffer[rank, DimList.create_unknown[rank](), type],
    b: NDBuffer[rank, DimList.create_unknown[rank](), type],
    c: NDBuffer[rank, DimList.create_unknown[rank](), type],
    out_chain: OutputChainPtr,
):
    alias adj_a = False
    alias adj_b = False

    alias simd_width = simdwidthof[type]()

    let B = a.dim[0]()
    let M = a.dim[1]()
    let N = b.dim[2]()
    let K = a.dim[2]()

    @parameter
    if single_thread_blocking_override and rank == 3:
        for batch in range(B):
            memset_zero(c.data + batch * M * N, M * N)
            for m in range(M):
                for k in range(K):
                    let a_val = a[batch, m, k]

                    @always_inline
                    @parameter
                    fn compute_fn[simd_width: Int](n: Int):
                        c.simd_store[simd_width](
                            StaticIntTuple[rank](batch, m, n),
                            c.simd_load[simd_width](batch, m, n)
                            + a_val * b.simd_load[simd_width](batch, k, n),
                        )

                    alias unroll_factor = 2

                    vectorize_unroll[simd_width, unroll_factor, compute_fn](N)
        return

    @always_inline
    @parameter
    fn description_fn() -> String:
        return get_trace_information_batched_matmul[rank](
            "dynamic_tile",
            a.get_shape(),
            b.get_shape(),
            c.get_shape(),
            adj_a,
            adj_b,
        )

    out_chain.trace[TraceLevel.OP, description_fn]("mojo.mogg.batched_matmul")

    @parameter
    @always_inline
    fn func(chain: OutputChainPtr):
        return batched_matmul_parallel_async[
            rank,
            type,
            adj_a,
            adj_b,
        ](c, a, b, chain)

    soft_fusion_run_wrapper[single_thread_blocking_override, func](out_chain)


# ===----------------------------------------------------------------------===#
# MOGG matrix solve
# ===----------------------------------------------------------------------===#


@always_inline
fn matrix_solve[
    type: DType,
    x_rank: Int,
    a_rank: Int,
    b_rank: Int,
    single_thread_blocking_override: Bool,
](
    a: NDBuffer[a_rank, DimList.create_unknown[a_rank](), type],
    b: NDBuffer[b_rank, DimList.create_unknown[b_rank](), type],
    x: NDBuffer[x_rank, DimList.create_unknown[x_rank](), type],
    out_chain: OutputChainPtr,
):
    @parameter
    @always_inline
    fn func(chain: OutputChainPtr):
        return _matrix_solve[type, x_rank, a_rank, b_rank](a, b, x, chain)

    soft_fusion_run_wrapper[single_thread_blocking_override, func](out_chain)


# ===----------------------------------------------------------------------===#
# Helpers
# ===----------------------------------------------------------------------===#


# Helper function to mark the output chain as ready in tests.
fn mark_output_chain_ready(out_chain: OutputChainPtr):
    out_chain.mark_ready()


# Helper function to query buffer shapes for tests.
fn print_buffer_info[
    type: DType, rank: Int
](buffer: NDBuffer[rank, DimList.create_unknown[rank](), type]):
    print("Rank:", rank)
    print("Shape:", buffer.dynamic_shape)
    print("Strides:", buffer.dynamic_stride)


# ===----------------------------------------------------------------------===#
# Special targets just for generation tests
# ===----------------------------------------------------------------------===#


fn test_many_ranks_and_types[
    type1: DType,
    rank1: Int,
    type2: DType,
    rank2: Int,
    type3: DType,
    rank3: Int,
    type4: DType,
    rank4: Int,
    type5: DType,
    rank5: Int,
](
    tensor1: NDBuffer[rank1, DimList.create_unknown[rank1](), type1],
    tensor2: NDBuffer[rank2, DimList.create_unknown[rank2](), type2],
    tensor3: NDBuffer[rank3, DimList.create_unknown[rank3](), type3],
    tensor4: NDBuffer[rank4, DimList.create_unknown[rank4](), type4],
    tensor5: NDBuffer[rank5, DimList.create_unknown[rank5](), type5],
) -> NDBuffer[rank1, DimList.create_unknown[rank1](), type1]:
    """
    Used as a test target to ensure parameter deduction works when there are
    many to deduce and also used to check errors.
    """
    return tensor1


fn test_one_rank_many_tensor[
    type: DType, rank: Int
](
    tensor1: NDBuffer[rank, DimList.create_unknown[rank](), type],
    tensor2: NDBuffer[rank, DimList.create_unknown[rank](), type],
    tensor3: NDBuffer[rank, DimList.create_unknown[rank](), type],
    tensor4: NDBuffer[rank, DimList.create_unknown[rank](), type],
    tensor5: NDBuffer[rank, DimList.create_unknown[rank](), type],
) -> NDBuffer[rank, DimList.create_unknown[rank](), type]:
    """
    Used as a test target to ensure we can deduce type and rank when used by
    many arguments.
    """
    return tensor1


fn test_3D_in_out_lambda[
    type: DType,
    simd_width: Int,
    input_0_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
](
    tensor1: NDBuffer[3, DimList.create_unknown[3](), type],
    output: NDBuffer[3, DimList.create_unknown[3](), type],
    out_chain: OutputChainPtr,
) -> NDBuffer[3, DimList.create_unknown[3](), type]:
    """
    Used as a target to test passing input and output lambdas.
    """

    for x in range(0, tensor1.dim[0]()):
        for y in range(0, tensor1.dim[1]()):

            @always_inline
            @parameter
            fn func_wrapper[simd_width: Int](idx: Int):
                let indices = StaticIntTuple[3](x, y, idx)
                let result = input_0_fn[type, simd_width, 3](indices)
                output_0_fn[type, simd_width, 3](indices, result)

            vectorize[
                simd_width,
                func_wrapper,
            ](tensor1.dim[2]())

    out_chain.mark_ready()
    return output
