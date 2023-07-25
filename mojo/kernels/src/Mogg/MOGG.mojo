# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Activations import relu, gelu, sigmoid
from Assert import assert_param, debug_assert
from Buffer import NDBuffer
from Concat import concat as _concat
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
from OptionalParam import OptionalParamInt
from LLCL import Runtime, OutputChainPtr, OwningOutputChainPtr
from Math import (
    add,
    ceil,
    div,
    div_ceil,
    erf,
    exp,
    equal,
    floor,
    greater,
    greater_equal,
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
from Limits import isinf, min_or_neginf, max_or_inf
from Matmul import matmul_parallel_sync, pack_b_ndbuffer
from BatchedMatmul import (
    batched_matmul_parallel_sync,
    get_trace_information as get_trace_information_batched_matmul,
)
from MatmulUtils import (
    GemmShape,
    get_trace_information,
    is_critical_stride,
    _get_tile_n_k_ND,
    search_mm_config,
)

from Pointer import Pointer, DTypePointer
from Range import range
from SIMD import SIMD
from TargetInfo import simdwidthof
from Tracing import Trace, TraceLevel
from TypeUtilities import rebind
from Softmax import softmax as _softmax, logsoftmax as _logsoftmax
from Split import split as _split
from String import String
from Slice import slice_as_view
from MatrixSolve import matrix_solve as _matrix_solve
from Index import Index
from GatherScatter import scatter_nd as _scatter_nd
from Where import where, where_shape


# Prevent these functions from being DCE'd by explicitly exporting them.
@export
fn MOGGExport():
    alias _indices = TensorIndicesTypeDef
    alias _out_chain = OutputChainPtrDef
    alias _simd_typedef = SimdTypeDef
    alias _index_typedef = IndexTypeDef
    alias _dtype_bfloat16 = DTypeBFloat16TypeDef
    alias _dtype_float16 = DTypeFloat16TypeDef
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
    alias _concat = concat
    alias _concat_shape = concat_shape
    alias _div = div
    alias _erf = erf
    alias _exp = exp
    alias _equal = equal
    alias _floor = floor
    alias _gather = _gather_with_lambdas
    alias _gather_shape = gather_shape
    alias _gelu = gelu
    alias _pack_matmul_b_shape_func = pack_matmul_b_shape_func
    alias _pad_shape = pad_shape
    alias _greater = greater
    alias _greater_equal = greater_equal
    alias _isinf = isinf
    alias _isnan = isnan
    alias _log1p = log1p
    alias _logsoftmax = logsoftmax
    alias _pack_b = pack_b_ndbuffer
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
    alias _broadcast_to_shape = broadcast_to_shape
    alias _broadcast_to_tensor = broadcast_to_tensor
    alias _scatter_nd = scatter_nd
    alias _slice = slice
    alias _simd_load = simd_load
    alias _simd_store = simd_store
    alias _simd_load_1D = simd_load_1D
    alias _simd_load_splat = simd_load_splat
    alias _simd_load_maybe_splat = simd_load_maybe_splat
    alias _simd_load_strided = simd_load_strided
    alias _simd_target = get_target_simd
    alias _simd_width_to_int = simd_width_to_int
    alias _softmax = softmax
    alias _split_ith_output_shape = split_ith_output_shape
    alias _split = split
    alias _reduce_shape = reduce_shape
    alias _reduce_add = reduce_add
    alias _reduce_max = reduce_max
    alias _reduce_min = reduce_min
    alias _reduce_mul = reduce_mul
    alias _splat = splat
    alias _transpose = transpose
    alias _transpose_shape = transpose_shape
    alias _elementwise = elementwise_wrapper
    alias _get_int_from_shape = get_int_from_shape
    alias _tensor_to_shape = tensor_to_shape
    alias _print_shape_info = print_buffer_info
    alias _mark_output_chain_ready = mark_output_chain_ready
    alias _where = where
    alias _where_shape = where_shape

    alias _test_many_ranks_and_types = test_many_ranks_and_types
    alias _test_one_rank_many_tensor = test_one_rank_many_tensor
    alias _test_3D_in_out_lambda = test_3D_in_out_lambda


# ===----------------------------------------------------------------------===#
# Nop functions to expose different types to the compiler.
# ===----------------------------------------------------------------------===#


fn DTypeBFloat16TypeDef(ty: DType.type) -> DType.type:
    return DType.bfloat16.value


fn DTypeFloat16TypeDef(ty: DType.type) -> DType.type:
    return DType.float16.value


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


# Convert a tensor into a shape.
@always_inline
fn tensor_to_shape[
    type: DType,
    rank: Int,
](tensor: NDBuffer[1, DimList.create_unknown[1](), type],) -> StaticIntTuple[
    rank
]:
    var out = StaticIntTuple[rank]()
    for i in range(rank):
        out[i] = tensor[i].to_int()

    return out


# Extract a value from a shape.
@always_inline
fn get_int_from_shape[
    param_index: Int, rank: Int
](shape: StaticIntTuple[rank]) -> Int:
    return shape[param_index]


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
    @always_inline
    @parameter
    fn description_fn() -> String:
        let name_str = String("name=") + trace_description
        let shape_str = String("shape=") + String("x").join(buffer.get_shape())

        let vector_width_str = String("vector_width=") + simd_width

        let res = String(";").join(name_str, shape_str, vector_width_str)

        return res

    out_chain.trace[TraceLevel.OP, description_fn]("mojo.elementwise")

    _elementwise_impl[rank, simd_width, single_thread_blocking_override, func](
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
    original_rank: Int,
    target_rank: Int,
    output_rank: Int,
    single_thread_blocking_override: Bool,
](
    original: NDBuffer[
        original_rank, DimList.create_unknown[original_rank](), type
    ],
    target: StaticIntTuple[target_rank],
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
            shape[i] = target[i]
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
        if original.dim(orig_index) == target[target_index]:
            stride[big_index] = original.stride(orig_index)
            shape[big_index] = original.dim(orig_index)
        elif original.dim(orig_index) == 1:
            # If they don't match and original is 1 then we broadcast.
            stride[big_index] = 0
            shape[big_index] = target[target_index]
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


@always_inline
fn broadcast_to_shape[
    input_rank: Int,
    output_rank: Int,
    input_type: DType,
    target_shape_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[
        input_rank, DimList.create_unknown[input_rank](), input_type
    ],
    target_shape_buf: NDBuffer[
        1, DimList.create_unknown[1](), target_shape_type
    ],
) -> StaticIntTuple[output_rank]:

    # TODO(#17512)
    debug_assert(
        output_rank == target_shape_buf.dim(0),
        "output rank must match target shape",
    )
    debug_assert(
        input_rank <= output_rank,
        "input rank must not exceed output rank",
    )

    # move the output shape from buffer into a static int tuple
    var output_shape = StaticIntTuple[output_rank]()
    for axis in range(output_rank):
        output_shape[axis] = target_shape_buf[axis].to_int()

    # Validate the compatibility between input and output shapes
    # NOTE we don't need to check the padded dims
    for i in range(input_rank):
        let input_axis = input_rank - i - 1
        let output_axis = output_rank - i - 1
        let input_dim = input_buf.dim(input_axis)
        let output_dim = output_shape[output_axis]
        debug_assert(
            input_dim == 1 or input_dim == output_dim,
            "input dim must be either 1 or equal to corresponding output dim",
        )
    return output_shape


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


# ===----------------------------------------------------------------------===#
# Concat op
# ===----------------------------------------------------------------------===#


@always_inline
fn concat[
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
    var axis_int = axis[0].to_int()
    if axis_int < 0:
        axis_int = axis_int + rank

    @parameter
    @always_inline
    fn func(out_chain: OutputChainPtr):
        # NOTE: Synchronous, so stack allocated variadic list is safe
        _concat[rank, type](
            output, axis_int, VariadicList(variadic_ins), out_chain
        )

    soft_fusion_run_wrapper[single_thread_blocking_override, func](out_chain)


@always_inline
fn concat_shape[
    rank: Int,
    input_type: DType,
    axis_type: DType,
    single_thread_blocking_override: Bool,
](
    axis_buf: NDBuffer[1, DimList.create_unknown[1](), axis_type],
    *input_bufs: NDBuffer[rank, DimList.create_unknown[rank](), input_type],
) -> StaticIntTuple[rank]:

    # extract hyper parameters
    var axis = axis_buf[0].to_int()
    if axis < 0:
        axis += rank
    # TODO(#17512)
    debug_assert(
        0 <= axis and axis < rank,
        "normalized split axis must be within range [0, rank)",
    )

    @always_inline
    fn shape_equal_ignore_axis(
        s1: StaticIntTuple[rank], s2: StaticIntTuple[rank]
    ) -> Bool:
        for i in range(rank):
            if i != axis and s1[i] != s2[i]:
                return False
        return True

    var concat_axis_dim_sum = 0
    for i in range(VariadicList(input_bufs).__len__()):
        concat_axis_dim_sum += input_bufs[i].dim(axis)
        # TODO(#17512)
        debug_assert(
            shape_equal_ignore_axis(
                input_bufs[0].get_shape(), input_bufs[i].get_shape()
            ),
            "input shapes must be equal except for at the concat axis",
        )

    # compute and return the output shape
    var output_shape = input_bufs[0].get_shape()
    output_shape[axis] = concat_axis_dim_sum
    return output_shape


# ===----------------------------------------------------------------------===#
# Split op
# ===----------------------------------------------------------------------===#


# Not targetted yet because MOGG assumes single output
@always_inline
fn split[
    type: DType,
    rank: Int,
    simd_width: Int,
    single_thread_blocking_override: Bool,
    axis_type: DType,
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    axis: NDBuffer[1, DimList.create_unknown[1](), axis_type],
    out_chain: OutputChainPtr,
    *variadic_outs: NDBuffer[rank, DimList.create_unknown[rank](), type],
):
    var axis_int = axis[0].to_int()
    if axis_int < 0:
        axis_int = axis_int + rank

    @parameter
    @always_inline
    fn func(out_chain: OutputChainPtr):
        # NOTE: Synchronous, so stack allocated variadic list is safe
        _split[type, rank](
            input, axis_int, VariadicList(variadic_outs), out_chain
        )

    soft_fusion_run_wrapper[single_thread_blocking_override, func](out_chain)


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

# These need wrappers as we can't take an alias of the ambiguous overload.


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


@always_inline
fn mean[
    type: DType,
    index_type: DType,
    rank: Int,
    simd_width: Int,
    single_thread_blocking_override: Bool,
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
    let reduce_dim = input_1_fn[index_type, 1, 1](0).to_int()

    # TODO (#17421): Remove and add back input_0_fn to MOGG signature so that it
    # can be fused
    @parameter
    @always_inline
    fn input_0_fn[
        _type: DType, width: Int, _rank: Int
    ](coords: StaticIntTuple[_rank]) -> SIMD[_type, width]:
        return rebind[SIMD[_type, width]](
            input_buffer.simd_load[width](rebind[StaticIntTuple[rank]](coords))
        )

    @always_inline
    fn reduce_impl[
        ty: DType, width: Int
    ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
        return v1 + v2

    # For floats apply the reciprocal as a multiply.
    @parameter
    if type.is_floating_point():
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
# Pad op
# ===----------------------------------------------------------------------===#


@always_inline
fn pad_shape[
    input_rank: Int,
    input_type: DType,
    paddings_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[
        input_rank, DimList.create_unknown[input_rank](), input_type
    ],
    paddings_buf: NDBuffer[2, DimList.create_unknown[2](), paddings_type],
) -> StaticIntTuple[input_rank]:

    # TODO(#17512)
    debug_assert(
        paddings_buf.dim(0) == input_rank and paddings_buf.dim(1) == 2,
        "paddings shape must be (input_rank, 2)",
    )

    # compute and return the output shape
    var output_shape = StaticIntTuple[input_rank]()
    for axis in range(input_rank):
        let pre_pad = paddings_buf[axis, 0].to_int()
        let post_pad = paddings_buf[axis, 1].to_int()
        output_shape[axis] = pre_pad + input_buf.dim(axis) + post_pad

    return output_shape


# ===----------------------------------------------------------------------===#
# Reduction ops
# ===----------------------------------------------------------------------===#


@always_inline
fn reduce_add[
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
    out_chain.trace[TraceLevel.OP]("mogg.reduce_add")

    # Only one reduce dimension supported currently, it must be deduced from
    # the attached input lambda rather than read directly.
    let reduce_dim = input_1_fn[index_type, 1, 1](0).to_int()

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


@always_inline
fn reduce_max[
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
    out_chain.trace[TraceLevel.OP]("mogg.reduce_max")

    # Only one reduce dimension supported currently, it must be deduced from
    # the attached input lambda rather than read directly.
    let reduce_dim = input_1_fn[index_type, 1, 1](0).to_int()

    @always_inline
    fn reduce_impl[
        ty: DType, width: Int
    ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
        return max(v1, v2)

    _reduce_generator[
        type,
        rank,
        simdwidthof[type](),
        single_thread_blocking_override,
        input_0_fn,
        output_0_fn,
        reduce_impl,
    ](input_buffer, min_or_neginf[type](), reduce_dim, out_chain)


@always_inline
fn reduce_min[
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
    out_chain.trace[TraceLevel.OP]("mogg.reduce_min")

    # Only one reduce dimension supported currently, it must be deduced from
    # the attached input lambda rather than read directly.
    let reduce_dim = input_1_fn[index_type, 1, 1](0).to_int()

    @always_inline
    fn reduce_impl[
        ty: DType, width: Int
    ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
        return min(v1, v2)

    _reduce_generator[
        type,
        rank,
        simdwidthof[type](),
        single_thread_blocking_override,
        input_0_fn,
        output_0_fn,
        reduce_impl,
    ](input_buffer, max_or_inf[type](), reduce_dim, out_chain)


@always_inline
fn reduce_mul[
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
    out_chain.trace[TraceLevel.OP]("mogg.reduce_mul")

    # Only one reduce dimension supported currently, it must be deduced from
    # the attached input lambda rather than read directly.
    let reduce_dim = input_1_fn[index_type, 1, 1](0).to_int()

    @always_inline
    fn reduce_impl[
        ty: DType, width: Int
    ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
        return v1 * v2

    _reduce_generator[
        type,
        rank,
        simdwidthof[type](),
        single_thread_blocking_override,
        input_0_fn,
        output_0_fn,
        reduce_impl,
    ](input_buffer, 1, reduce_dim, out_chain)


@always_inline
fn reduce_shape[
    rank: Int,
    input_type: DType,
    axis_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[rank, DimList.create_unknown[rank](), input_type],
    axis_buf: NDBuffer[1, DimList.create_unknown[1](), axis_type],
) -> StaticIntTuple[rank]:

    # extract hyper parameter
    var axis = axis_buf[0].to_int()
    if axis < 0:
        axis += rank
    # TODO(#17512)
    debug_assert(
        0 <= axis and axis < rank,
        "normalized axis must be within range [0, rank)",
    )

    # compute and return the output shape
    var output_shape = input_buf.get_shape()
    output_shape[axis] = 1
    return output_shape


# ===----------------------------------------------------------------------===#
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


@always_inline
fn transpose_shape[
    rank: Int,
    type: DType,
    int_type: DType,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    perms: NDBuffer[1, DimList.create_unknown[1](), int_type],
) -> StaticIntTuple[rank]:
    debug_assert(
        perms.dim(0) == rank,
        (
            "Size of transpose permutation vector doesn't match expected output"
            " shape"
        ),
    )

    @unroll
    for i in range(rank):
        let perm = perms[i].to_int()
        # TODO(17512)
        debug_assert(perm >= 0, "Transpose permutation is less than zero")
        debug_assert(perm < rank, "Transpose permutation is out of range")

    let out = transpose[rank, type, int_type, single_thread_blocking_override](
        input, perms
    ).dynamic_shape

    # TODO(17512)
    debug_assert(
        out.flattened_length() == input.dynamic_shape.flattened_length(),
        "Dynamic transpose has changed the number of elements",
    )
    return out


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
    let axis = OptionalParamInt[Dim()](input_2_fn[axis_type, 1, 1](0).to_int())

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
            indices_index[unrolled_i] = idx[unrolled_i + axis.get()]

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
            indices_index[unrolled_i] = idx[unrolled_i + axis.get()]
            if unrolled_i == axis.get():
                data_indices[unrolled_i] = data_index
            elif unrolled_i > axis.get():
                # Skip over any extra indices dimensions. These are essentially new dimensions.
                data_indices[unrolled_i] = idx[unrolled_i + skip_factor]
            else:
                data_indices[unrolled_i] = idx[unrolled_i]

        unroll[in_rank, input_indices_get]()

        # Load the the data.
        let data = input_0_fn[type, simd_width, in_rank](data_indices)

        # Store it to the original index.
        output_0_fn[type, simd_width, rank](idx, data)

    # If we are gathering on the last dimension then we have to be scalar.
    if axis.get() == in_rank - 1:
        _elementwise_impl[
            output_rank,
            1,
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
            single_thread_blocking_override,
            gather_lambda,
        ](
            output.dynamic_shape,
            out_chain,
        )


@always_inline
fn gather_shape[
    input_rank: Int,
    indices_rank: Int,
    output_rank: Int,
    input_type: DType,
    indices_type: DType,
    axis_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[
        input_rank, DimList.create_unknown[input_rank](), input_type
    ],
    indices_buf: NDBuffer[
        indices_rank, DimList.create_unknown[indices_rank](), indices_type
    ],
    axis_buf: NDBuffer[1, DimList.create_unknown[1](), axis_type],
) -> StaticIntTuple[output_rank]:
    assert_param[
        output_rank == input_rank + indices_rank - 1,
        "output rank must equal (input_rank + indices_rank - 1)",
    ]()

    # extract hyper parameter
    var axis = axis_buf[0].to_int()
    if axis < 0:
        axis += input_rank
    # TODO(#17512)
    debug_assert(
        0 <= axis and axis < input_rank,
        "normalized axis must be within range [0, input_rank)",
    )

    # compute and return the output shape
    var output_shape = StaticIntTuple[output_rank]()
    var next_out_dim = 0

    let input_shape = input_buf.get_shape()
    for i in range(axis):
        output_shape[next_out_dim] = input_shape[i]
        next_out_dim += 1

    let indices_shape = indices_buf.get_shape()
    for i in range(indices_rank):
        output_shape[next_out_dim] = indices_shape[i]
        next_out_dim += 1

    for i in range(axis + 1, input_rank):
        output_shape[next_out_dim] = input_shape[i]
        next_out_dim += 1

    return output_shape


# ===----------------------------------------------------------------------===#
# MOGG matmul
# ===----------------------------------------------------------------------===#

# TODO(16425): Unify with existing shim.
@always_inline
fn pack_matmul_b_shape_func[
    type: DType, transpose_in_0: Bool, single_thread_blocking_override: Bool
](b_input: NDBuffer[2, DimList.create_unknown[2](), type],) -> StaticIntTuple[
    2
]:
    """Sets in shape_ref the shape required by `pack_b`'s `b_packed_ref`
    argument.

    If transpose_b is True, this returns the un-transposed shape, since pack_b
    will un-transpose `b_ref` as part of the packing layout transformation."""

    var output = StaticIntTuple[2]()

    let k = b_input.dim(1) if transpose_in_0 else b_input.dim(0)
    var tile_n_k = StaticIntTuple[2]()

    if is_critical_stride(k):
        alias config = search_mm_config[type, True, True]()
        tile_n_k = _get_tile_n_k_ND[config, transpose_in_0, type](b_input)
    else:
        alias config2 = search_mm_config[type, True, False]()
        tile_n_k = _get_tile_n_k_ND[config2, transpose_in_0, type](b_input)

    @parameter
    if transpose_in_0:
        output[0] = b_input.dim(1)
        output[1] = b_input.dim(0)
    else:
        output[0] = b_input.dim(0)
        output[1] = b_input.dim(1)

    output[0] = div_ceil(output[0], tile_n_k[1]) * tile_n_k[1]
    output[1] = div_ceil(output[1], tile_n_k[0]) * tile_n_k[0]

    return output


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


@always_inline
fn matmul[
    type: DType,
    transpose_in_1: Bool,  # matches name of MO attribute
    packed_in_1: Bool,
    single_thread_blocking_override: Bool,
    lambdas_have_fusion: Bool,
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
    alias b_packed = packed_in_1

    assert_param[
        not (b_packed and transpose_b),
        (
            "transpose_b and b_packed cannot both be true because pre-packing"
            " transposes B"
        ),
    ]()

    @parameter
    @always_inline
    fn epilogue_wrapper[
        type: DType, width: Int
    ](coords: StaticIntTuple[2], val: SIMD[type, width]):
        output_0_fn[type, width, 2](coords, val)

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

    matmul_parallel_sync[
        type,  # a_type
        type,  # b_type
        type,  # c_type
        transpose_a,
        transpose_b,
        b_packed,
        lambdas_have_fusion,
        epilogue_wrapper,
        single_thread_blocking_override,
    ](
        c,
        a,
        b,
        out_chain,
    )


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

    return batched_matmul_parallel_sync[
        rank, type, adj_a, adj_b, single_thread_blocking_override
    ](c, a, b, out_chain)


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
# MOGG scatter_nd
# ===----------------------------------------------------------------------===#


fn scatter_nd[
    output_rank: Int,
    updates_rank: Int,
    indices_rank: Int,
    type: DType,
    single_thread_blocking_override: Bool,
](
    updates: NDBuffer[
        updates_rank, DimList.create_unknown[updates_rank](), type
    ],
    indices: NDBuffer[
        indices_rank, DimList.create_unknown[indices_rank](), DType.int32
    ],
    output: NDBuffer[output_rank, DimList.create_unknown[output_rank](), type],
    out_chain: OutputChainPtr,
):
    @parameter
    @always_inline
    fn func(chain: OutputChainPtr):
        return _scatter_nd[type, updates_rank, indices_rank, output_rank](
            updates, indices, output, chain
        )

    soft_fusion_run_wrapper[single_thread_blocking_override, func](out_chain)


# Define a wrapper in MOGG.mojo so that softmax kernel in stdlib takes static shapes
fn softmax[
    rank: Int,
    type: DType,
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    output: NDBuffer[rank, DimList.create_unknown[rank](), type],
    out_chain: OutputChainPtr,
):
    _softmax[
        type,
        simdwidthof[type](),
        rank,
        DimList.create_unknown[rank](),
    ](input, output, rank - 1, out_chain)


# Define a wrapper in MOGG.mojo so that softmax kernel in stdlib takes static shapes
fn logsoftmax[
    rank: Int,
    type: DType,
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    output: NDBuffer[rank, DimList.create_unknown[rank](), type],
    out_chain: OutputChainPtr,
):
    _logsoftmax[
        type,
        simdwidthof[type](),
        rank,
        DimList.create_unknown[rank](),
    ](input, output, rank - 1, out_chain)


# ===----------------------------------------------------------------------===#
# MOGG split
# ===----------------------------------------------------------------------===#


@always_inline
fn split_ith_output_shape[
    output_idx: Int,
    rank: Int,
    input_type: DType,
    split_size_type: DType,
    axis_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[rank, DimList.create_unknown[rank](), input_type],
    split_sizes_buf: NDBuffer[1, DimList.create_unknown[1](), split_size_type],
    split_axis_buf: NDBuffer[1, DimList.create_unknown[1](), axis_type],
) -> StaticIntTuple[rank]:

    # extract relevant hyper parameters
    # TODO(#17512)
    debug_assert(
        0 <= output_idx and output_idx < split_sizes_buf.size(),
        "output index must be within range [0, len(split_sizes))",
    )
    let output_split_size = split_sizes_buf[output_idx].to_int()

    var split_axis = split_axis_buf[0].to_int()
    if split_axis < 0:
        split_axis += rank
    # TODO(#17512)
    debug_assert(
        0 <= split_axis and split_axis < rank,
        "normalized split axis must be within range [0, rank)",
    )

    var split_sizes_sum = 0
    for i in range(split_sizes_buf.dim(0)):
        split_sizes_sum += split_sizes_buf[i].to_int()
    # TODO(#17512)
    debug_assert(
        split_sizes_sum == input_buf.dim(split_axis),
        "sum of split sizes must be equal to input dimension at split axis",
    )

    # compute and return the output shape
    var output_shape = input_buf.get_shape()
    output_shape[split_axis] = output_split_size
    return output_shape


# ===----------------------------------------------------------------------===#
# Helpers
# ===----------------------------------------------------------------------===#


# Helper function to mark the output chain as ready in tests.
@always_inline
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
