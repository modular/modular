# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import (
    abs,
    add,
    ceil,
    cos,
    div,
    div_ceil,
    equal,
    erf,
    exp,
    floor,
    fma,
    greater,
    greater_equal,
    isnan,
    log,
    logical_and,
    logical_not,
    logical_xor,
    log1p,
    max,
    min,
    mod,
    mul,
    not_equal,
    pow,
    rsqrt,
    round,
    roundeven,
    select,
    sin,
    sqrt,
    sub,
    tanh,
    trunc,
)
from math.limit import isinf, max_or_inf, min_or_neginf
from sys.info import simdwidthof
from sys.intrinsics import strided_load

from Activations import gelu, relu, sigmoid
from ArgNonzero import arg_nonzero, arg_nonzero_shape
from algorithm import argmax as _argmax
from algorithm import argmin as _argmin
from algorithm import (
    async_parallelize,
    reduce_shape,
    unroll,
    vectorize,
    vectorize_unroll,
)
from algorithm.functional import _elementwise_impl
from algorithm.reduction import _reduce_generator
from Arange import arange, arange_shape
from BatchedMatmul import batched_matmul as _batched_matmul
from BatchedMatmul import (
    get_trace_information as get_trace_information_batched_matmul,
)
from Concat import concat as _concat
from Concat import concat_shape
from Conv import (
    ConvInfo,
    ConvInfoStatic,
    conv_2d_nhwc_direct,
    conv_shape,
    pack_conv_filter,
    pack_conv_filter_shape,
)
from Cumsum import cumsum as _cumsum
from GatherScatter import gather as _gather
from GatherScatter import gather_shape
from GatherScatter import gather_nd as _gather_nd
from GatherScatter import gather_reduce
from GatherScatter import (
    scatter_elements,
    scatter_elements_shape as scatter_shape,
    scatter_nd as _scatter_nd,
    scatter_nd_generator,
)
from Matmul import (
    pack_b_ndbuffer,
    pack_matmul_b_shape_func,
    pack_transposed_b_ndbuffer,
)
from Matmul import matmul as _matmul

from MatmulUtils import (
    GemmShape,
    get_trace_information,
    search_mm_config,
)
from MatrixBandPart import matrix_band_part
from MatrixSolve import matrix_solve
from memory import memset_zero
from memory.buffer import NDBuffer
from memory.unsafe import DTypePointer, Pointer
from MOGGDecorators import *
from NonMaxSuppression import (
    non_max_suppression,
    non_max_suppression_shape_func,
)
from Normalization import layer_norm
from Pad import pad as _pad
from Pad import pad_shape
from Pool import avg_pool, max_pool, pool_shape
from runtime.llcl import OutputChainPtr
from runtime.tracing import Trace, TraceLevel
from Resize import (
    resize_nearest_neighbor,
    RoundMode,
    CoordinateTransformationMode,
)
from Slice import slice_as_view, slice_shape
from Softmax import logsoftmax as _logsoftmax
from Softmax import softmax as _softmax
from Split import split as _split
from TopK import top_k as _top_k
from TopK import top_k_shape

from utils.index import Index, StaticIntTuple, product
from utils.list import Dim, DimList, VariadicList
from utils.optional_param import OptionalParamInt


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
    alias _arg_max = argmax_wrapped
    alias _arg_min = argmin_wrapped
    alias _abs = abs_wrapped
    alias _add = add
    alias _avg_pool_shape = pool_shape
    alias _avg_pool = avg_pool
    alias _broadcast_shape = broadcast_shape
    alias _cast = cast
    alias _ceil = ceil
    alias _concat = concat
    alias _concat_shape = concat_shape
    alias _conv_shape = conv_shape
    alias _cumsum = cumsum
    alias _conv = conv
    alias _cos = cos
    alias _div = div
    alias _erf = erf
    alias _exp = exp
    alias _equal = equal
    alias _floor = floor
    alias _gather_shape = gather_shape
    alias _gather_nd = gather_nd
    alias _gather = gather
    alias _gelu = gelu
    alias _pack_matmul_b_shape_func = pack_matmul_b_shape_func
    alias _pack_conv_filter_shape = pack_conv_filter_shape
    alias _pad = pad
    alias _pad_shape = pad_shape
    alias _greater = greater
    alias _greater_equal = greater_equal
    alias _isinf = isinf
    alias _isnan = isnan
    alias _logical_and = logical_and
    alias _logical_not = logical_not
    alias _logical_xor = logical_xor
    alias _log = log
    alias _log1p = log1p
    alias _logsoftmax = logsoftmax
    alias _pack_b_ndbuffer = pack_b_ndbuffer
    alias _pack_transposed_b_ndbuffer = pack_transposed_b_ndbuffer
    alias _pack_conv_filter = pack_conv_filter
    alias _pow = pow_wrapped
    alias _max_pool_shape = pool_shape
    alias _max_pool = max_pool
    alias _mean = mean
    alias _matrix_solve = matrix_solve
    alias _matrix_band_part = matrix_band_part
    alias _matmul = matmul
    alias _negative = negative
    alias _non_maximum_suppression = non_maximum_suppression
    alias _non_maximum_suppression_shape_func = non_maximum_suppression_shape_func
    alias _batched_matmul = batched_matmul
    alias _mogg_max = mogg_max
    alias _mogg_min = mogg_min
    alias _mod = mod
    alias _mul = mul
    alias _not_equal = not_equal
    alias _rsqrt = rsqrt
    alias _select = select
    alias _sigmoid = sigmoid
    alias _sin = sin
    alias _sqrt = sqrt_wrapped
    alias _sub = sub
    alias _tanh = tanh
    alias _arange = arange
    alias _arange_shape = arange_shape
    alias _relu = relu
    alias _reshape = reshape
    alias _reshape_shape = reshape_shape
    alias _calculate_squeeze_shape = calculate_squeeze_shape
    alias _calculate_unsqueeze_shape = calculate_unsqueeze_shape
    alias _broadcast_to_shape = broadcast_to_shape
    alias _broadcast_to_tensor = broadcast_to_tensor
    alias _scatter_nd = scatter_nd
    alias _scatter_nd_add = scatter_nd_add
    alias _scatter_nd_max = scatter_nd_max
    alias _scatter_nd_min = scatter_nd_min
    alias _scatter_nd_mul = scatter_nd_mul
    alias _scatter = scatter
    alias _scatter_shape = scatter_shape
    alias _scatter_add = scatter_add
    alias _scatter_max = scatter_max
    alias _scatter_min = scatter_min
    alias _scatter_mul = scatter_mul
    alias _slice = slice
    alias _simd_load = simd_load
    alias _simd_store = simd_store
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
    alias _resize_nearest = resize_nearest
    alias _resize_shape = resize_shape
    alias _round = round
    alias _roundeven = roundeven
    alias _slice_shape = slice_shape
    alias _transpose = transpose
    alias _transpose_shape = transpose_shape
    alias _trunc = trunc
    alias _elementwise = elementwise_wrapper
    alias _get_int_from_shape = get_int_from_shape
    alias _tensor_to_shape = tensor_to_shape
    alias _print_shape_info = print_buffer_info
    alias _mark_output_chain_ready = mark_output_chain_ready
    alias _arg_nonzero = arg_nonzero
    alias _arg_nonzero_shape = arg_nonzero_shape
    alias _top_k = top_k
    alias _bottom_k = bottom_k
    alias _top_k_shape = top_k_shape


# ===----------------------------------------------------------------------===#
# Nop functions to expose different types to the compiler.
# ===----------------------------------------------------------------------===#


@mogg_register("bfloat16")
fn DTypeBFloat16TypeDef(ty: DType.type) -> DType.type:
    return DType.bfloat16.value


@mogg_register("float16")
fn DTypeFloat16TypeDef(ty: DType.type) -> DType.type:
    return DType.float16.value


@mogg_register("float32")
fn DTypeFloat32TypeDef(ty: DType.type) -> DType.type:
    return DType.float32.value


@mogg_register("float64")
fn DTypeFloat64TypeDef(ty: DType.type) -> DType.type:
    return DType.float64.value


@mogg_register("int8")
fn DTypeInt8TypeDef(ty: DType.type) -> DType.type:
    return DType.int8.value


@mogg_register("int16")
fn DTypeInt16TypeDef(ty: DType.type) -> DType.type:
    return DType.int16.value


@mogg_register("int32")
fn DTypeInt32TypeDef(ty: DType.type) -> DType.type:
    return DType.int32.value


@mogg_register("uint32")
fn DTypeUInt32TypeDef(ty: DType.type) -> DType.type:
    return DType.uint32.value


@mogg_register("int64")
fn DTypeInt64TypeDef(ty: DType.type) -> DType.type:
    return DType.int64.value


@mogg_register("uint8")
fn DTypeUInt8TypeDef(ty: DType.type) -> DType.type:
    return DType.uint8.value


@mogg_register("uint16")
fn DTypeUInt16TypeDef(ty: DType.type) -> DType.type:
    return DType.uint16.value


@mogg_register("bool")
fn DTypeBoolTypeDef(ty: DType.type) -> DType.type:
    return DType.bool.value


@mogg_register("index")
fn IndexTypeDef(ty: Int) -> Int:
    return ty


@mogg_register("outChain")
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
    data: __mlir_type[`!kgen.pointer<scalar<`, type.value, `>>`],
    shape: __mlir_type.`!kgen.pointer<index>`,
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
        DTypePointer[type](data), shape_tuple, stride_tuple
    )


@always_inline
fn to_shape[
    rank: Int
](shape: __mlir_type.`!kgen.pointer<index>`) -> StaticIntTuple[rank]:
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
](tensor: NDBuffer[1, DimList.create_unknown[1](), type]) -> StaticIntTuple[
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

        let info = String(";").join(name_str, shape_str, vector_width_str)

        return (
            info
            + String(";single_thread_blocking_override=")
            + single_thread_blocking_override
        )

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


@mogg_register("simd_load")
@always_inline
fn simd_load[
    type: DType, simd_width: Int, rank: Int
](
    buffer: NDBuffer[rank, DimList.create_unknown[rank](), type],
    index: StaticIntTuple[rank],
) -> SIMD[type, simd_width]:
    let flat_index = _compute_flat_index[type, rank, rank](buffer, index)
    let stride = buffer.dynamic_stride[rank - 1]

    if buffer.dynamic_stride[rank - 1] == 0:
        return buffer.data.load(flat_index)
    elif stride > 1:

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
    return _simd_load_internal[simd_width, type, rank](buffer, flat_index)


@mogg_register("simd_store")
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
fn broadcast_shape[
    lhs_type: DType,
    rhs_type: DType,
    out_type: DType,
](
    lhs_buf: NDBuffer[1, DimList.create_unknown[1](), lhs_type],
    rhs_buf: NDBuffer[1, DimList.create_unknown[1](), rhs_type],
    out_buf: NDBuffer[1, DimList.create_unknown[1](), out_type],
    out_chain: OutputChainPtr,
):
    let lhs_size = lhs_buf.size()
    let rhs_size = rhs_buf.size()
    if lhs_size > rhs_size:
        return broadcast_shape_impl(rhs_buf, lhs_buf, out_buf, out_chain)
    return broadcast_shape_impl(lhs_buf, rhs_buf, out_buf, out_chain)


@always_inline
fn broadcast_shape_impl[
    lhs_type: DType,
    rhs_type: DType,
    out_type: DType,
](
    lhs_buf: NDBuffer[1, DimList.create_unknown[1](), lhs_type],
    rhs_buf: NDBuffer[1, DimList.create_unknown[1](), rhs_type],
    out_buf: NDBuffer[1, DimList.create_unknown[1](), out_type],
    out_chain: OutputChainPtr,
):
    # Ensure lhs is always the smaller shape
    let lhs_rank = lhs_buf.size()
    let rhs_rank = rhs_buf.size()
    debug_assert(lhs_rank <= rhs_rank, "lhs shape must be the smaller one")

    # lhs_buf =      [l0, l1, ...]
    # rhs_buf = [..., r0, r1, ...]
    # out_buf = [..., o0, o1, ...]
    let size_diff = rhs_rank - lhs_rank
    for i in range(size_diff):
        out_buf[i] = rhs_buf[i].cast[out_type]()

    for lhs_idx in range(lhs_rank):
        let rhs_idx = lhs_idx + size_diff
        let lhs_dim = lhs_buf[lhs_idx].to_int()
        let rhs_dim = rhs_buf[rhs_idx].to_int()
        if lhs_dim == rhs_dim:
            out_buf[rhs_idx] = rhs_buf[rhs_idx].cast[out_type]()

        elif lhs_dim != 1 and rhs_dim != 1:
            debug_assert(
                rhs_dim == 1, "one of the differing dimensions must be 1"
            )

        elif lhs_dim != 1:
            out_buf[rhs_idx] = lhs_buf[lhs_idx].cast[out_type]()

        elif rhs_dim != 1:
            out_buf[rhs_idx] = rhs_buf[rhs_idx].cast[out_type]()

    out_chain.mark_ready()


@mogg_register("mo.static.broadcast_to")
@mogg_view_op
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
@mogg_register("mo.abs")
@mogg_elementwise
@always_inline
fn abs_wrapped[
    type: DType, simd_width: Int
](value: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return abs(value)


# ===----------------------------------------------------------------------===#
# ArgMax wrapper op
# ===----------------------------------------------------------------------===#


# Call argmax, needed as it has multiple overloads which can't be aliased
@always_inline
fn argmax_wrapped[
    type: DType,
    out_type: DType,
    axis_type: DType,
    rank: Int,
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    axis_buf: NDBuffer[1, DimList.create_unknown[1](), axis_type],
    output: NDBuffer[rank, DimList.create_unknown[rank](), out_type],
    out_chain: OutputChainPtr,
):
    return _argmax(input, axis_buf, output, out_chain)


# ===----------------------------------------------------------------------===#
# ArgMin wrapper op
# ===----------------------------------------------------------------------===#


# Call argmin, needed as it has multiple overloads which can't be aliased
@always_inline
fn argmin_wrapped[
    type: DType,
    out_type: DType,
    axis_type: DType,
    rank: Int,
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    axis_buf: NDBuffer[1, DimList.create_unknown[1](), axis_type],
    output: NDBuffer[rank, DimList.create_unknown[rank](), out_type],
    out_chain: OutputChainPtr,
):
    return _argmin(input, axis_buf, output, out_chain)


# ===----------------------------------------------------------------------===#
# Cast op
# ===----------------------------------------------------------------------===#


# Cast a SIMD value to a new SIMD value of different type.
@mogg_register("mo.cast")
@mogg_elementwise
@always_inline
fn cast[
    type: DType, new_type: DType, simd_width: Int
](value: SIMD[type, simd_width]) -> SIMD[new_type, simd_width]:
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
    let inputs = VariadicList(variadic_ins)

    _concat[rank, type, single_thread_blocking_override](
        output, axis_int, inputs, out_chain
    )


# ===----------------------------------------------------------------------===#
# Cumsum op
# ===----------------------------------------------------------------------===#


@always_inline
fn cumsum[
    type: DType,
    axis_type: DType,
    rank: Int,
    exclusive: Int,
    reverse: Int,
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    axis: NDBuffer[1, DimList.create_unknown[1](), axis_type],
    output: NDBuffer[rank, DimList.create_unknown[rank](), type],
    out_chain: OutputChainPtr,
):
    var axis_int = axis[0].to_int()
    if axis_int < 0:
        axis_int = axis_int + rank
    debug_assert(
        axis_int >= 0 and axis_int < rank, "axis must be between 0 and rank - 1"
    )
    _cumsum[rank, type, exclusive, reverse](output, input, axis_int)
    out_chain.mark_ready()


# ===----------------------------------------------------------------------===#
# Split op
# ===----------------------------------------------------------------------===#


# Not targetted yet because MOGG assumes single output
@always_inline
fn split[
    type: DType,
    rank: Int,
    simd_width: Int,
    axis_type: DType,
    split_sizes_type: DType,
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    split_sizes: NDBuffer[1, DimList.create_unknown[1](), split_sizes_type],
    axis: NDBuffer[1, DimList.create_unknown[1](), axis_type],
    out_chain: OutputChainPtr,
    *variadic_outs: NDBuffer[rank, DimList.create_unknown[rank](), type],
):
    var axis_int = axis[0].to_int()
    if axis_int < 0:
        axis_int = axis_int + rank

    # NOTE: Synchronous, so stack allocated variadic list is safe
    _split[type, rank](input, axis_int, VariadicList(variadic_outs), out_chain)


# ===----------------------------------------------------------------------===#
# Pow wrapper op
# ===----------------------------------------------------------------------===#


# Call pow, needed as it has multiple overloads which can't be aliased
@mogg_register("mo.pow")
@mogg_elementwise
@always_inline
fn pow_wrapped[
    type: DType, power_type: DType, simd_width: Int
](value: SIMD[type, simd_width], power: SIMD[power_type, simd_width]) -> SIMD[
    type, simd_width
]:
    return pow[type, power_type, simd_width](value, power)


# ===----------------------------------------------------------------------===#
# Sqrt wrapper op
# ===----------------------------------------------------------------------===#


# Call sqrt, needed as it has multiple overloads which can't be aliased
@mogg_register("mo.sqrt")
@mogg_elementwise
@always_inline
fn sqrt_wrapped[
    type: DType, simd_width: Int
](value: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return sqrt(value)


# ===----------------------------------------------------------------------===#
# Max & min ops
# ===----------------------------------------------------------------------===#

# These need wrappers as we can't take an alias of the ambiguous overload.


@mogg_register("mo.max")
@mogg_elementwise
@always_inline
fn mogg_max[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    type, simd_width
]:
    return max(x, y)


@mogg_register("mo.min")
@mogg_elementwise
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
    input_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
](
    input_shape: StaticIntTuple[rank],
    axis_buffer: NDBuffer[1, DimList.create_unknown[1](), index_type],
    output_shape: StaticIntTuple[rank],
    out_chain: OutputChainPtr,
):
    out_chain.trace[TraceLevel.OP]("mogg.mean")

    # Only one reduce dimension supported currently, it must be deduced from
    # the attached input lambda rather than read directly.
    let reduce_dim = axis_buffer[0].to_int()

    @always_inline
    fn reduce_impl[
        ty: DType, width: Int
    ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
        return v1 + v2

    @always_inline
    fn input_0_fn_wrapper[
        _type: DType, width: Int, rank: Int
    ](idx: StaticIntTuple[rank]) -> SIMD[_type, width]:
        return rebind[SIMD[_type, width]](input_0_fn[width, rank](idx))

    # For floats apply the reciprocal as a multiply.
    @parameter
    if type.is_floating_point():
        # Apply mean division before storing to the output lambda.
        let reciprocal = 1.0 / input_shape[reduce_dim]

        @always_inline
        @parameter
        fn wrapped_output_mul[
            _type: DType, width: Int, rank: Int
        ](indices: StaticIntTuple[rank], value: SIMD[_type, width]):
            let mean_val = value * reciprocal
            output_0_fn[width, rank](
                indices, rebind[SIMD[type, width]](mean_val)
            )

        _reduce_generator[
            type,
            rank,
            simdwidthof[type](),
            single_thread_blocking_override,
            input_0_fn_wrapper,
            wrapped_output_mul,
            reduce_impl,
        ](input_shape, 0, reduce_dim, out_chain)

    else:
        # For ints just a normal divide.
        let dim_size = input_shape[reduce_dim]

        @always_inline
        @parameter
        fn wrapped_output_div[
            _type: DType, width: Int, rank: Int
        ](indices: StaticIntTuple[rank], value: SIMD[_type, width]):
            let mean_val = value / dim_size
            output_0_fn[width, rank](
                indices, rebind[SIMD[type, width]](mean_val)
            )

        _reduce_generator[
            type,
            rank,
            simdwidthof[type](),
            single_thread_blocking_override,
            input_0_fn_wrapper,
            wrapped_output_div,
            reduce_impl,
        ](input_shape, 0, reduce_dim, out_chain)


# ===----------------------------------------------------------------------===#
# Negative op
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn negative[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Negates a SIMD vector.
    Parameters:
      type: DType of the input SIMD vector.
      simd_width: Width of the input SIMD vector.

    Args:
      x: SIMD vector to negate.

    Returns:
      Negative of x (i.e., x multiplied by -1).
    """
    return -x


# ===----------------------------------------------------------------------===#
# Pad op
# ===----------------------------------------------------------------------===#


@always_inline
fn pad[
    rank: Int,
    type: DType,
    paddings_type: DType,
    constant_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[rank, DimList.create_unknown[rank](), type],
    paddings_buf: NDBuffer[2, DimList.create_unknown[2](), paddings_type],
    constant_buf: NDBuffer[1, DimList.create_unknown[1](), constant_type],
    output_buf: NDBuffer[rank, DimList.create_unknown[rank](), type],
    out_chain: OutputChainPtr,
):
    let paddings_ptr = paddings_buf.data
    let constant_simd = constant_buf[0]
    _pad(output_buf, input_buf, paddings_ptr, constant_simd)

    @parameter
    if not single_thread_blocking_override:
        out_chain.mark_ready()


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
    input_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
](
    input_shape: StaticIntTuple[rank],
    axis_buffer: NDBuffer[1, DimList.create_unknown[rank](), index_type],
    output_shape: StaticIntTuple[rank],
    out_chain: OutputChainPtr,
):
    out_chain.trace[TraceLevel.OP]("mogg.reduce_add")

    # Only one reduce dimension supported currently, it must be deduced from
    # the attached input lambda rather than read directly.
    let reduce_dim = axis_buffer[0].to_int()

    @always_inline
    fn input_0_fn_wrapper[
        _type: DType, width: Int, rank: Int
    ](idx: StaticIntTuple[rank]) -> SIMD[_type, width]:
        return rebind[SIMD[_type, width]](input_0_fn[width, rank](idx))

    @always_inline
    fn output_0_fn_wrapper[
        _type: DType, width: Int, rank: Int
    ](indices: StaticIntTuple[rank], value: SIMD[_type, width]):
        output_0_fn[width, rank](indices, rebind[SIMD[type, width]](value))

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
        input_0_fn_wrapper,
        output_0_fn_wrapper,
        reduce_impl,
    ](input_shape, 0, reduce_dim, out_chain)


@always_inline
fn reduce_max[
    type: DType,
    index_type: DType,
    rank: Int,
    simd_width: Int,
    single_thread_blocking_override: Bool,
    input_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
](
    input_shape: StaticIntTuple[rank],
    axis_buffer: NDBuffer[1, DimList.create_unknown[rank](), index_type],
    output_shape: StaticIntTuple[rank],
    out_chain: OutputChainPtr,
):
    out_chain.trace[TraceLevel.OP]("mogg.reduce_max")

    # Only one reduce dimension supported currently, it must be deduced from
    # the attached input lambda rather than read directly.
    let reduce_dim = axis_buffer[0].to_int()

    @always_inline
    fn input_0_fn_wrapper[
        _type: DType, width: Int, rank: Int
    ](idx: StaticIntTuple[rank]) -> SIMD[_type, width]:
        return rebind[SIMD[_type, width]](input_0_fn[width, rank](idx))

    @always_inline
    fn output_0_fn_wrapper[
        _type: DType, width: Int, rank: Int
    ](indices: StaticIntTuple[rank], value: SIMD[_type, width]):
        output_0_fn[width, rank](indices, rebind[SIMD[type, width]](value))

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
        input_0_fn_wrapper,
        output_0_fn_wrapper,
        reduce_impl,
    ](input_shape, min_or_neginf[type](), reduce_dim, out_chain)


@always_inline
fn reduce_min[
    type: DType,
    index_type: DType,
    rank: Int,
    simd_width: Int,
    single_thread_blocking_override: Bool,
    input_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
](
    input_shape: StaticIntTuple[rank],
    axis_buffer: NDBuffer[1, DimList.create_unknown[rank](), index_type],
    output_shape: StaticIntTuple[rank],
    out_chain: OutputChainPtr,
):
    out_chain.trace[TraceLevel.OP]("mogg.reduce_min")

    # Only one reduce dimension supported currently, it must be deduced from
    # the attached input lambda rather than read directly.
    let reduce_dim = axis_buffer[0].to_int()

    @always_inline
    fn input_0_fn_wrapper[
        _type: DType, width: Int, rank: Int
    ](idx: StaticIntTuple[rank]) -> SIMD[_type, width]:
        return rebind[SIMD[_type, width]](input_0_fn[width, rank](idx))

    @always_inline
    fn output_0_fn_wrapper[
        _type: DType, width: Int, rank: Int
    ](indices: StaticIntTuple[rank], value: SIMD[_type, width]):
        output_0_fn[width, rank](indices, rebind[SIMD[type, width]](value))

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
        input_0_fn_wrapper,
        output_0_fn_wrapper,
        reduce_impl,
    ](input_shape, max_or_inf[type](), reduce_dim, out_chain)


@always_inline
fn reduce_mul[
    type: DType,
    index_type: DType,
    rank: Int,
    simd_width: Int,
    single_thread_blocking_override: Bool,
    input_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
](
    input_shape: StaticIntTuple[rank],
    axis: NDBuffer[1, DimList.create_unknown[rank](), index_type],
    output_shape: StaticIntTuple[rank],
    out_chain: OutputChainPtr,
):
    out_chain.trace[TraceLevel.OP]("mogg.reduce_mul")

    # Only one reduce dimension supported currently, it must be deduced from
    # the attached input lambda rather than read directly.
    let reduce_dim = axis[0].to_int()

    @always_inline
    fn input_0_fn_wrapper[
        _type: DType, width: Int, rank: Int
    ](idx: StaticIntTuple[rank]) -> SIMD[_type, width]:
        return rebind[SIMD[_type, width]](input_0_fn[width, rank](idx))

    @always_inline
    fn output_0_fn_wrapper[
        _type: DType, width: Int, rank: Int
    ](indices: StaticIntTuple[rank], value: SIMD[_type, width]):
        output_0_fn[width, rank](indices, rebind[SIMD[type, width]](value))

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
        input_0_fn_wrapper,
        output_0_fn_wrapper,
        reduce_impl,
    ](input_shape, 1, reduce_dim, out_chain)


# ===----------------------------------------------------------------------===#
# Slice op
# ===----------------------------------------------------------------------===#


# Wrapper for slice here to include the `single_thread_blocking_override`.
@mogg_register("mo.slice")
@mogg_view_op
@always_inline
fn slice[
    type: DType,
    start_type: DType,
    end_type: DType,
    step_type: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
](
    tensor: NDBuffer[rank, DimList.create_unknown[rank](), type],
    starts: NDBuffer[1, DimList.create_unknown[1](), start_type],
    ends: NDBuffer[1, DimList.create_unknown[1](), end_type],
    steps: NDBuffer[1, DimList.create_unknown[1](), step_type],
) -> NDBuffer[rank, DimList.create_unknown[rank](), type]:
    return slice_as_view(tensor, starts, ends, steps)


# ===----------------------------------------------------------------------===#
# Reshape op
# ===----------------------------------------------------------------------===#


# Reshape assumes inputs are contiguous. It should always be fused last and
# a non-contiguous tensor cannot be fused *into* this as input.
@mogg_register("mo.static.reshape")
@mogg_view_op
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
        input.data, new_shape, stride_tuple
    )


@always_inline
fn reshape_shape[
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

    # move the target shape from buffer into a static int tuple; also check and
    # record if there's any to-be-inferred dimension (-1).
    var target_shape = StaticIntTuple[output_rank]()
    var to_be_inferred_axis = -1
    var non_negative_dim_prodcut = 1
    for axis in range(output_rank):
        let target_dim = target_shape_buf[axis].to_int()
        target_shape[axis] = target_dim
        if target_dim == -1:
            # TODO(#17512)
            debug_assert(
                to_be_inferred_axis == -1,
                "only one -1 is allowed in target shape",
            )
            to_be_inferred_axis = axis
        else:
            # TODO(#17512)
            debug_assert(
                target_dim >= 0,
                "only -1 is allowed as a negative value in target shape",
            )
            non_negative_dim_prodcut *= target_dim

    let input_num_elems = input_buf.num_elements()
    var output_num_elems = non_negative_dim_prodcut
    # Infer a dimension as the remaining elements, if needed.
    if to_be_inferred_axis != -1:
        # TODO(#17512)
        debug_assert(
            non_negative_dim_prodcut != 0,
            (
                "concrete dimensions must not contain 0 if there's a"
                " to-be-inferred dimension"
            ),
        )
        debug_assert(
            input_num_elems % non_negative_dim_prodcut == 0,
            "to-be-inferred dimension must be an integer",
        )
        target_shape[to_be_inferred_axis] = (
            input_num_elems // non_negative_dim_prodcut
        )
        output_num_elems = input_num_elems

    # TODO(#17512)
    debug_assert(
        output_num_elems == input_num_elems,
        "output and input number of elements must match",
    )

    return target_shape


# ===----------------------------------------------------------------------===#
# SqueezeShape
# ===----------------------------------------------------------------------===#


@always_inline
fn calculate_squeeze_shape[
    type: DType, indices_type: DType, single_thread_blocking_override: Bool
](
    input_shape: NDBuffer[1, DimList.create_unknown[1](), type],
    remove_indices: NDBuffer[1, DimList.create_unknown[1](), indices_type],
    output_shape: NDBuffer[1, DimList.create_unknown[1](), type],
):
    # remove_indices may not be sorted so our strategy is to use -1 to
    # represent removed dimensions in a copied version of our input shape buffer
    let num_input_dims = input_shape.dynamic_shape[0]
    let num_remove_indices = remove_indices.dynamic_shape[0]
    let final_rank = num_input_dims - num_remove_indices

    debug_assert(
        final_rank == output_shape.dynamic_shape[0],
        "Incorrect output shape.",
    )

    alias MAX_VECTOR_LIMIT = 12
    debug_assert(
        num_input_dims <= MAX_VECTOR_LIMIT,
        "Only support shape vectors up to rank-12.",
    )
    var input_shape_copy = StaticIntTuple[MAX_VECTOR_LIMIT]()
    for i in range(num_input_dims):
        input_shape_copy[i] = input_shape[i].to_int()

    # Mark every squeezed dimension as -1 in our copy of the shape tensor
    for remove_index_index in range(num_remove_indices):
        let remove_index = remove_indices[remove_index_index].to_int()
        let remove_index_normalize = remove_index + num_input_dims * (
            remove_indices[remove_index_index] < 0
        ).to_int()

        debug_assert(
            remove_index_normalize >= 0 and remove_index_normalize < final_rank,
            (
                "Remove indices must be between [-r, r-1] where r is the final"
                " output rank."
            ),
        )
        debug_assert(
            output_shape[remove_index_normalize] != -1,
            "Multiple indices point to the same dimension.",
        )
        debug_assert(
            output_shape[remove_index_normalize] == 1,
            "Attempting to unsqueeze a dimension that is not 1.",
        )
        input_shape_copy[remove_index_normalize] = -1

    # # Copy over the non -1 dimensions
    var output_shape_index = 0
    for input_shape_index in range(num_input_dims):
        if input_shape_copy[input_shape_index] == -1:
            continue
        output_shape[output_shape_index] = input_shape_copy[input_shape_index]
        output_shape_index += 1


# ===----------------------------------------------------------------------===#
# UnsqueezeShape op
# ===----------------------------------------------------------------------===#


@always_inline
fn calculate_unsqueeze_shape[
    type: DType, indices_type: DType, single_thread_blocking_override: Bool
](
    input_shape: NDBuffer[1, DimList.create_unknown[1](), type],
    padding_indices: NDBuffer[1, DimList.create_unknown[1](), indices_type],
    output_shape: NDBuffer[1, DimList.create_unknown[1](), type],
):
    # padding_indices_buf may not be sorted so our strategy is to use -1 to
    # represent uninitialized dimensions, add the padding dimensions, and copy
    # over the remaining dimensions later.
    let num_input_dims = input_shape.dynamic_shape[0]
    let num_padding_indices = padding_indices.dynamic_shape[0]
    let final_rank = num_input_dims + num_padding_indices
    debug_assert(
        final_rank == output_shape.dynamic_shape[0],
        "Incorrect output shape.",
    )
    for output_index in range(final_rank):
        output_shape[output_index] = -1

    for padding_index_index in range(num_padding_indices):
        let padding_index = padding_indices[padding_index_index].to_int()
        let padding_index_normalize = padding_index + final_rank * (
            padding_indices[padding_index_index] < 0
        ).to_int()

        debug_assert(
            padding_index_normalize >= 0
            and padding_index_normalize < final_rank,
            (
                "Padding indices must be between [-r, r-1] where r is the final"
                " output rank."
            ),
        )
        debug_assert(
            output_shape[padding_index_normalize] == -1,
            (
                "Duplicate padding indices point to the same dimension in the"
                " final output shape."
            ),
        )
        output_shape[padding_index_normalize] = 1

    # Copy over the remaining shapes
    var orig_shape_index = 0
    for output_shape_index in range(final_rank):
        if output_shape[output_shape_index] != -1:
            continue
        output_shape[output_shape_index] = input_shape[orig_shape_index]
        orig_shape_index += 1


# ===----------------------------------------------------------------------===#
# Transpose op
# ===----------------------------------------------------------------------===#


@mogg_register("mo.transpose")
@mogg_view_op
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
        input.data, new_shape, new_stride
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

    # NOTE this assumes `transpose` can handle input with null data pointer
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


# TODO(#20442): Remove with generic fusion.
@mogg_register("mo.gather_sum")
@always_inline
@export
fn mogg_gather_sum[
    output_rank: Int,
    input_rank: Int,
    indices_rank: Int,
    type: DType,
](
    input: NDBuffer[input_rank, DimList.create_unknown[input_rank](), type],
    indices: NDBuffer[
        indices_rank,
        DimList.create_unknown[indices_rank](),
        DType.int32,
    ],
    output: NDBuffer[output_rank, DimList.create_unknown[output_rank](), type],
    out_chain: OutputChainPtr,
):
    gather_reduce[
        output_rank,
        DimList.create_unknown[output_rank](),
        input_rank,
        DimList.create_unknown[input_rank](),
        indices_rank,
        DimList.create_unknown[indices_rank](),
        type,
        0,
        1,
        simdwidthof[type](),
        add,
    ](output, input, indices, 0, out_chain)


@always_inline
fn gather[
    type: DType,
    in_rank: Int,
    indices_type: DType,
    indices_rank: Int,
    axis_type: DType,
    output_rank: Int,
    simd_width: Int,
    single_thread_blocking_override: Bool,
    input_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
](
    input_shape: StaticIntTuple[in_rank],
    indices: NDBuffer[
        indices_rank,
        DimList.create_unknown[indices_rank](),
        indices_type,
    ],
    axis_buffer: NDBuffer[1, DimList.create_unknown[1](), axis_type],
    output_shape: StaticIntTuple[output_rank],
    out_chain: OutputChainPtr,
):
    # Look through the lambda to pull the index out.
    alias axis_static = Dim()
    let axis = axis_buffer[0].to_int()

    let axis_normalized = OptionalParamInt[axis_static](
        axis + in_rank if axis < 0 else axis
    )

    @parameter
    @always_inline
    fn no_prefetch[
        input_rank: Int, indices_rank: Int
    ](
        input_cooords: StaticIntTuple[input_rank],
        indices_coords: StaticIntTuple[indices_rank],
    ):
        pass

    # TODO: This is disabled as if we make this a shape without a spec we have
    # nothing to deduce `indices_type` from.
    @parameter
    @always_inline
    fn load_indices[
        width: Int, _rank: Int
    ](coords: StaticIntTuple[_rank]) -> SIMD[indices_type, width]:
        return indices.simd_load[width](
            rebind[StaticIntTuple[indices_rank]](coords)
        )

    _gather[
        type,
        in_rank,
        indices_type,
        indices_rank,
        output_rank,
        simd_width,
        single_thread_blocking_override,
        input_0_fn,
        load_indices,
        output_0_fn,
        no_prefetch,
        axis_static,
    ](
        axis_normalized,
        input_shape,
        indices.dynamic_shape,
        output_shape,
        out_chain,
    )


# ===----------------------------------------------------------------------===#
# MOGG matmul
# ===----------------------------------------------------------------------===#


@always_inline
fn matmul[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_in_1: Bool,  # matches name of MO attribute
    packed_in_1: Bool,
    single_thread_blocking_override: Bool,
    lambdas_have_fusion: Bool,
    output_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[c_type, width]
    ) capturing -> None,
](
    a: NDBuffer[2, DimList.create_unknown[2](), a_type],
    b: NDBuffer[2, DimList.create_unknown[2](), b_type],
    c: NDBuffer[2, DimList.create_unknown[2](), c_type],
    out_chain: OutputChainPtr,
):
    alias transpose_a = False
    alias transpose_b = transpose_in_1
    alias b_packed = packed_in_1

    constrained[
        not (b_packed and transpose_b),
        (
            "transpose_b and b_packed cannot both be true because pre-packing"
            " transposes B"
        ),
    ]()

    @parameter
    @always_inline
    fn epilogue_wrapper[
        _type: DType, width: Int
    ](coords: StaticIntTuple[2], val: SIMD[_type, width]):
        output_0_fn[width, 2](coords, rebind[SIMD[c_type, width]](val))

    @always_inline
    @parameter
    fn description_fn() -> String:
        let info = get_trace_information(
            "dynamic_tile",
            GemmShape.get[
                transpose_a,
                transpose_b,
            ](c, a, b),
            transpose_a,
            transpose_b,
            b_packed,
        )
        return (
            info
            + String(";single_thread_blocking_override=")
            + single_thread_blocking_override
        )

    out_chain.trace[TraceLevel.OP, description_fn]("mojo.mogg.matmul")

    _matmul[
        a_type,
        b_type,
        c_type,
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
    transpose_in_1: Bool,
    single_thread_blocking_override: Bool,
    lambdas_have_fusion: Bool,
    output_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
](
    a: NDBuffer[rank, DimList.create_unknown[rank](), type],
    b: NDBuffer[rank, DimList.create_unknown[rank](), type],
    c: NDBuffer[rank, DimList.create_unknown[rank](), type],
    out_chain: OutputChainPtr,
):
    alias adj_a = False
    alias adj_b = transpose_in_1

    @parameter
    @always_inline
    fn epilogue_wrapper[
        _type: DType, width: Int, rank: Int
    ](coords: StaticIntTuple[rank], val: SIMD[_type, width]):
        output_0_fn[width, rank](coords, rebind[SIMD[type, width]](val))

    @always_inline
    @parameter
    fn description_fn() -> String:
        let info = get_trace_information_batched_matmul[rank](
            "dynamic_tile",
            a.get_shape(),
            b.get_shape(),
            c.get_shape(),
            adj_a,
            adj_b,
        )
        return (
            info
            + String(";single_thread_blocking_override=")
            + single_thread_blocking_override
        )

    out_chain.trace[TraceLevel.OP, description_fn]("mojo.mogg.batched_matmul")

    return _batched_matmul[
        rank,
        type,
        adj_a,
        adj_b,
        lambdas_have_fusion,
        epilogue_wrapper,
        single_thread_blocking_override,
    ](c, a, b, out_chain)


# ===----------------------------------------------------------------------===#
# MOGG scatter
# ===----------------------------------------------------------------------===#


@always_inline
fn scatter[
    rank: Int,
    input_type: DType,
    indices_type: DType,
    axis_type: DType,
](
    input: NDBuffer[
        rank,
        DimList.create_unknown[rank](),
        input_type,
    ],
    updates: NDBuffer[rank, DimList.create_unknown[rank](), input_type],
    indices: NDBuffer[
        rank,
        DimList.create_unknown[rank](),
        indices_type,
    ],
    axis: NDBuffer[1, DimList.create_unknown[1](), axis_type],
    output: NDBuffer[rank, DimList.create_unknown[rank](), input_type],
    out_chain: OutputChainPtr,
):
    @always_inline
    fn reduce_func[
        type: DType, width: Int
    ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
        return rhs  # always return the latest update element

    let axis_int = axis[0].to_int()
    return scatter_elements[reduce_func](
        input, indices, updates, axis_int, output, out_chain
    )


@always_inline
fn scatter_add[
    rank: Int,
    input_type: DType,
    indices_type: DType,
    axis_type: DType,
](
    input: NDBuffer[
        rank,
        DimList.create_unknown[rank](),
        input_type,
    ],
    updates: NDBuffer[rank, DimList.create_unknown[rank](), input_type],
    indices: NDBuffer[
        rank,
        DimList.create_unknown[rank](),
        indices_type,
    ],
    axis: NDBuffer[1, DimList.create_unknown[1](), axis_type],
    output: NDBuffer[rank, DimList.create_unknown[rank](), input_type],
    out_chain: OutputChainPtr,
):
    @always_inline
    fn reduce_func[
        type: DType, width: Int
    ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
        return lhs + rhs

    let axis_int = axis[0].to_int()
    return scatter_elements[reduce_func](
        input, indices, updates, axis_int, output, out_chain
    )


@always_inline
fn scatter_max[
    rank: Int,
    input_type: DType,
    indices_type: DType,
    axis_type: DType,
](
    input: NDBuffer[
        rank,
        DimList.create_unknown[rank](),
        input_type,
    ],
    updates: NDBuffer[rank, DimList.create_unknown[rank](), input_type],
    indices: NDBuffer[
        rank,
        DimList.create_unknown[rank](),
        indices_type,
    ],
    axis: NDBuffer[1, DimList.create_unknown[1](), axis_type],
    output: NDBuffer[rank, DimList.create_unknown[rank](), input_type],
    out_chain: OutputChainPtr,
):
    @always_inline
    fn reduce_func[
        type: DType, width: Int
    ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
        return lhs.max(rhs)

    let axis_int = axis[0].to_int()
    return scatter_elements[reduce_func](
        input, indices, updates, axis_int, output, out_chain
    )


@always_inline
fn scatter_min[
    rank: Int,
    input_type: DType,
    indices_type: DType,
    axis_type: DType,
](
    input: NDBuffer[
        rank,
        DimList.create_unknown[rank](),
        input_type,
    ],
    updates: NDBuffer[rank, DimList.create_unknown[rank](), input_type],
    indices: NDBuffer[
        rank,
        DimList.create_unknown[rank](),
        indices_type,
    ],
    axis: NDBuffer[1, DimList.create_unknown[1](), axis_type],
    output: NDBuffer[rank, DimList.create_unknown[rank](), input_type],
    out_chain: OutputChainPtr,
):
    @always_inline
    fn reduce_func[
        type: DType, width: Int
    ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
        return lhs.min(rhs)

    let axis_int = axis[0].to_int()
    return scatter_elements[reduce_func](
        input, indices, updates, axis_int, output, out_chain
    )


@always_inline
fn scatter_mul[
    rank: Int,
    input_type: DType,
    indices_type: DType,
    axis_type: DType,
](
    input: NDBuffer[
        rank,
        DimList.create_unknown[rank](),
        input_type,
    ],
    updates: NDBuffer[rank, DimList.create_unknown[rank](), input_type],
    indices: NDBuffer[
        rank,
        DimList.create_unknown[rank](),
        indices_type,
    ],
    axis: NDBuffer[1, DimList.create_unknown[1](), axis_type],
    output: NDBuffer[rank, DimList.create_unknown[rank](), input_type],
    out_chain: OutputChainPtr,
):
    @always_inline
    fn reduce_func[
        type: DType, width: Int
    ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
        return lhs * rhs

    let axis_int = axis[0].to_int()
    return scatter_elements[reduce_func](
        input, indices, updates, axis_int, output, out_chain
    )


# ===----------------------------------------------------------------------===#
# MOGG scatter_nd
# ===----------------------------------------------------------------------===#


@always_inline
fn scatter_nd[
    output_rank: Int,
    updates_rank: Int,
    indices_rank: Int,
    output_type: DType,
    indices_type: DType,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[
        output_rank,
        DimList.create_unknown[output_rank](),
        output_type,
    ],
    updates: NDBuffer[
        updates_rank, DimList.create_unknown[updates_rank](), output_type
    ],
    indices: NDBuffer[
        indices_rank,
        DimList.create_unknown[indices_rank](),
        indices_type,
    ],
    output: NDBuffer[
        output_rank, DimList.create_unknown[output_rank](), output_type
    ],
    out_chain: OutputChainPtr,
):
    return _scatter_nd[
        output_type,
        indices_type,
        updates_rank,
        indices_rank,
        output_rank,
        single_thread_blocking_override,
    ](input, updates, indices, output, out_chain)


@always_inline
fn scatter_nd_add[
    output_type: DType,
    indices_type: DType,
    updates_rank: Int,
    indices_rank: Int,
    output_rank: Int,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[
        output_rank,
        DimList.create_unknown[output_rank](),
        output_type,
    ],
    updates: NDBuffer[
        updates_rank, DimList.create_unknown[updates_rank](), output_type
    ],
    indices: NDBuffer[
        indices_rank, DimList.create_unknown[indices_rank](), indices_type
    ],
    output: NDBuffer[
        output_rank, DimList.create_unknown[output_rank](), output_type
    ],
    out_chain: OutputChainPtr,
):
    @always_inline
    fn reduce_func(
        lhs: SIMD[output_type, 1], rhs: SIMD[output_type, 1]
    ) -> SIMD[output_type, 1]:
        return lhs + rhs

    scatter_nd_generator[
        output_type,
        indices_type,
        updates_rank,
        indices_rank,
        output_rank,
        single_thread_blocking_override,
        reduce_func,
    ](input, updates, indices, output, out_chain)


@always_inline
fn scatter_nd_max[
    output_type: DType,
    indices_type: DType,
    updates_rank: Int,
    indices_rank: Int,
    output_rank: Int,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[
        output_rank,
        DimList.create_unknown[output_rank](),
        output_type,
    ],
    updates: NDBuffer[
        updates_rank, DimList.create_unknown[updates_rank](), output_type
    ],
    indices: NDBuffer[
        indices_rank, DimList.create_unknown[indices_rank](), indices_type
    ],
    output: NDBuffer[
        output_rank, DimList.create_unknown[output_rank](), output_type
    ],
    out_chain: OutputChainPtr,
):
    @always_inline
    fn reduce_func(
        lhs: SIMD[output_type, 1], rhs: SIMD[output_type, 1]
    ) -> SIMD[output_type, 1]:
        return lhs.max(rhs)

    scatter_nd_generator[
        output_type,
        indices_type,
        updates_rank,
        indices_rank,
        output_rank,
        single_thread_blocking_override,
        reduce_func,
    ](input, updates, indices, output, out_chain)


@always_inline
fn scatter_nd_min[
    output_type: DType,
    indices_type: DType,
    updates_rank: Int,
    indices_rank: Int,
    output_rank: Int,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[
        output_rank,
        DimList.create_unknown[output_rank](),
        output_type,
    ],
    updates: NDBuffer[
        updates_rank, DimList.create_unknown[updates_rank](), output_type
    ],
    indices: NDBuffer[
        indices_rank, DimList.create_unknown[indices_rank](), indices_type
    ],
    output: NDBuffer[
        output_rank, DimList.create_unknown[output_rank](), output_type
    ],
    out_chain: OutputChainPtr,
):
    @always_inline
    fn reduce_func(
        lhs: SIMD[output_type, 1], rhs: SIMD[output_type, 1]
    ) -> SIMD[output_type, 1]:
        return lhs.min(rhs)

    scatter_nd_generator[
        output_type,
        indices_type,
        updates_rank,
        indices_rank,
        output_rank,
        single_thread_blocking_override,
        reduce_func,
    ](input, updates, indices, output, out_chain)


@always_inline
fn scatter_nd_mul[
    output_type: DType,
    indices_type: DType,
    updates_rank: Int,
    indices_rank: Int,
    output_rank: Int,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[
        output_rank,
        DimList.create_unknown[output_rank](),
        output_type,
    ],
    updates: NDBuffer[
        updates_rank, DimList.create_unknown[updates_rank](), output_type
    ],
    indices: NDBuffer[
        indices_rank, DimList.create_unknown[indices_rank](), indices_type
    ],
    output: NDBuffer[
        output_rank, DimList.create_unknown[output_rank](), output_type
    ],
    out_chain: OutputChainPtr,
):
    @always_inline
    fn reduce_func(
        lhs: SIMD[output_type, 1], rhs: SIMD[output_type, 1]
    ) -> SIMD[output_type, 1]:
        return lhs * rhs

    scatter_nd_generator[
        output_type,
        indices_type,
        updates_rank,
        indices_rank,
        output_rank,
        single_thread_blocking_override,
        reduce_func,
    ](input, updates, indices, output, out_chain)


# Define a wrapper in MOGG.mojo so that softmax kernel in stdlib takes static shapes
fn softmax[
    rank: Int,
    type: DType,
    input_0_fn: fn[_simd_width: Int, _rank: Int] (
        StaticIntTuple[_rank]
    ) capturing -> SIMD[type, _simd_width],
](
    shape: StaticIntTuple[rank],
    output: NDBuffer[rank, DimList.create_unknown[rank](), type],
    out_chain: OutputChainPtr,
):
    _softmax[
        type,
        simdwidthof[type](),
        rank,
        DimList.create_unknown[rank](),
        input_0_fn,
    ](shape, output, rank - 1, out_chain)


# Define a wrapper in MOGG.mojo so that softmax kernel in stdlib takes static shapes
@mogg_register("mo.logsoftmax")
@always_inline
fn logsoftmax[
    rank: Int,
    type: DType,
    input_0_fn: fn[_simd_width: Int, _rank: Int] (
        StaticIntTuple[_rank]
    ) capturing -> SIMD[type, _simd_width],
](
    shape: StaticIntTuple[rank],
    output: NDBuffer[rank, DimList.create_unknown[rank](), type],
    out_chain: OutputChainPtr,
):
    _logsoftmax[
        type,
        simdwidthof[type](),
        rank,
        DimList.create_unknown[rank](),
        input_0_fn,
    ](shape, output, rank - 1, out_chain)


# ===----------------------------------------------------------------------===#
# MOGG non_maximum_suppression
# ===----------------------------------------------------------------------===#


fn non_maximum_suppression[
    type: DType
](
    boxes: NDBuffer[3, DimList.create_unknown[3](), type],
    scores: NDBuffer[3, DimList.create_unknown[3](), type],
    max_output_boxes_per_class: NDBuffer[1, DimList(1), DType.int64],
    iou_threshold: NDBuffer[1, DimList(1), DType.float32],
    score_threshold: NDBuffer[1, DimList(1), DType.float32],
    output: NDBuffer[2, DimList.create_unknown[2](), DType.int64],
    out_chain: OutputChainPtr,
):
    let max_output_boxes_int = max_output_boxes_per_class[0].to_int()
    let iou_threshold_float = iou_threshold[0]
    let score_threshold_float = score_threshold[0]

    non_max_suppression[type](
        boxes,
        scores,
        output,
        max_output_boxes_int,
        iou_threshold_float,
        score_threshold_float,
    )
    mark_output_chain_ready(out_chain)


fn non_maximum_suppression_shape_func[
    type: DType, single_thread_blocking_override: Bool
](
    boxes: NDBuffer[3, DimList.create_unknown[3](), type],
    scores: NDBuffer[3, DimList.create_unknown[3](), type],
    max_output_boxes_per_class: NDBuffer[1, DimList(1), DType.int64],
    iou_threshold: NDBuffer[1, DimList(1), DType.float32],
    score_threshold: NDBuffer[1, DimList(1), DType.float32],
) -> StaticIntTuple[2]:
    let max_output_boxes_int = max_output_boxes_per_class[0].to_int()
    let iou_threshold_float = iou_threshold[0]
    let score_threshold_float = score_threshold[0]

    return non_max_suppression_shape_func[type](
        boxes,
        scores,
        max_output_boxes_int,
        iou_threshold_float,
        score_threshold_float,
    )


# ===----------------------------------------------------------------------===#
# MOGG resize
# ===----------------------------------------------------------------------===#


fn resize_nearest[
    coordinate_transform_mode: Int,
    round_mode: Int,
    rank: Int,
    inpType: DType,
    sizeType: DType,
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), inpType],
    size: NDBuffer[1, DimList(rank), sizeType],
    output: NDBuffer[rank, DimList.create_unknown[rank](), inpType],
    out_chain: OutputChainPtr,
):
    resize_nearest_neighbor[
        coordinate_transform_mode, round_mode, rank, inpType
    ](input, output, out_chain)


fn resize_shape[
    rank: Int,
    inpType: DType,
    sizeType: DType,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), inpType],
    size: NDBuffer[1, DimList(rank), sizeType],
) -> StaticIntTuple[rank]:
    var shape = StaticIntTuple[rank]()

    @unroll
    for i in range(rank):
        shape[i] = size[i].to_int()
    return shape


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


@always_inline
fn conv[
    filter_rank: Int,
    strides_rank: Int,
    dilation_rank: Int,
    padding_rank: Int,
    input_type: DType,
    filter_type: DType,
    strides_type: DType,
    dilation_type: DType,
    padding_type: DType,
    output_type: DType,
    filter_packed: Bool,
    lambdas_have_fusion: Bool,
    output_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[output_type, width]
    ) capturing -> None,
](
    input: NDBuffer[4, DimList.create_unknown[4](), input_type],
    filter: NDBuffer[
        filter_rank, DimList.create_unknown[filter_rank](), filter_type
    ],
    strides: NDBuffer[
        strides_rank, DimList.create_unknown[strides_rank](), strides_type
    ],
    dilation: NDBuffer[
        dilation_rank, DimList.create_unknown[dilation_rank](), dilation_type
    ],
    paddings: NDBuffer[
        padding_rank, DimList.create_unknown[padding_rank](), padding_type
    ],
    output: NDBuffer[4, DimList.create_unknown[4](), output_type],
    out_chain: OutputChainPtr,
):
    """Including this function in MOGG.mojo since it is intended to be a temporary
    wrapper around the Stdlib conv. Currently the strides and dilation are NDBuffers,
    but eventually they will be StaticIntTuple parameters (along with padding).
    """
    constrained[
        strides_type.is_integral() and dilation_type.is_integral(),
        "stride and dilation must have integral type",
    ]()

    if strides.size() != 2:
        return out_chain.mark_error("2 values expected in strides input")

    if dilation.size() != 2:
        return out_chain.mark_error("2 values expected in dilation input")

    if paddings.size() != 4:
        return out_chain.mark_error("4 values expected in paddings input")

    let strides_flat = strides.flatten()
    let dilation_flat = dilation.flatten()
    let paddings_flat = paddings.flatten()

    let strides_tuple = Index(
        strides_flat[0].to_int(), strides_flat[1].to_int()
    )
    let dilation_tuple = Index(
        dilation_flat[0].to_int(), dilation_flat[1].to_int()
    )
    if dilation_tuple != Index(1, 1):
        return out_chain.mark_error("Non-unit dilation is not supported yet.")

    let pad_h_tuple = Index(
        paddings_flat[0].to_int(), paddings_flat[1].to_int()
    )
    let pad_w_tuple = Index(
        paddings_flat[2].to_int(), paddings_flat[3].to_int()
    )

    # TODO: eventually padding, strides and dilation will be passed in as
    # parameters here when they are constant in the graph
    alias conv_info_static = ConvInfoStatic(
        DimList.create_unknown[2](),
        DimList.create_unknown[2](),
        DimList.create_unknown[2](),
        DimList.create_unknown[2](),
    )
    let conv_info = ConvInfo[conv_info_static](
        pad_h_tuple, pad_w_tuple, strides_tuple, dilation_tuple
    )

    # Specialize the function to take 4D coordiantes.
    # The bias is broadcasted to the same shape as output and
    # accessed by the 4D coordinates.
    @parameter
    @always_inline
    fn epilogue_wrapper[
        _type: DType, width: Int
    ](coords: StaticIntTuple[4], val: SIMD[_type, width]):
        output_0_fn[width, 4](coords, rebind[SIMD[output_type, width]](val))

    conv_2d_nhwc_direct[
        filter_rank,
        filter_packed,
        conv_info_static,
        lambdas_have_fusion,
        epilogue_wrapper,
    ](input, filter, output, conv_info, out_chain)


# ===----------------------------------------------------------------------===#
# MOGG layer_norm
# ===----------------------------------------------------------------------===#

# input, gamma, beta, eps


@mogg_register("mo.layer_norm")
@export
fn mogg_layer_norm[
    type: DType, rank: Int
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    gamma: NDBuffer[1, DimList.create_unknown[1](), type],
    beta: NDBuffer[1, DimList.create_unknown[1](), type],
    epsilon: NDBuffer[1, DimList.create_unknown[1](), type],
    output: NDBuffer[rank, DimList.create_unknown[rank](), type],
    out_chain: OutputChainPtr,
):
    out_chain.trace[TraceLevel.OP]("mojo.layer_norm")
    let eps = epsilon[0]

    alias simd_width = simdwidthof[type]()

    let last_dim = input.dynamic_shape[rank - 1]
    let prod_all_but_last_dim = input.dynamic_shape.flattened_length() // last_dim
    let flat_shape = StaticIntTuple[2](prod_all_but_last_dim, last_dim)

    let input_buf = reshape[rank, 2, type, True](input, flat_shape)
    let output_buf = reshape[rank, 2, type, True](output, flat_shape)

    let num_workers = min(
        out_chain.get_runtime().parallelism_level(), prod_all_but_last_dim
    )
    let chunk_size = div_ceil(prod_all_but_last_dim, num_workers)

    @parameter
    fn task_func(thread_id: Int):
        let num_rows = min(
            chunk_size, prod_all_but_last_dim - thread_id * chunk_size
        )
        let row_idx = thread_id * chunk_size
        let thread_starting_coord = StaticIntTuple[2](row_idx, 0)
        let per_thread_dims = DimList(num_rows, last_dim)
        let output_buf_view = NDBuffer[2, DimList.create_unknown[2](), type](
            output_buf._offset(thread_starting_coord), per_thread_dims
        )
        let input_buf_view = NDBuffer[2, DimList.create_unknown[2](), type](
            input_buf._offset(thread_starting_coord), per_thread_dims
        )

        @parameter
        fn noop[
            return_type: DType, simd_width: Int
        ](idx: Int, row: Int) -> SIMD[return_type, simd_width]:
            let coord = StaticIntTuple[2](row, idx)
            let input_val = input_buf_view.simd_load[simd_width](coord)
            return input_val.cast[return_type]()

        layer_norm[simd_width, type, noop](output_buf_view, gamma, beta, eps)

    async_parallelize[task_func](out_chain, num_workers)


@mogg_register("mo.layer_norm_skip")
@export
fn mogg_layer_norm_skip[
    type: DType, rank: Int
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    gamma: NDBuffer[1, DimList.create_unknown[1](), type],
    beta: NDBuffer[1, DimList.create_unknown[1](), type],
    epsilon: NDBuffer[1, DimList.create_unknown[1](), type],
    skip: NDBuffer[rank, DimList.create_unknown[rank](), type],
    output: NDBuffer[rank, DimList.create_unknown[rank](), type],
    out_chain: OutputChainPtr,
):
    out_chain.trace[TraceLevel.OP]("mojo.layer_norm")
    let eps = epsilon[0]

    alias simd_width = simdwidthof[type]()

    let last_dim = input.dynamic_shape[rank - 1]
    let prod_all_but_last_dim = input.dynamic_shape.flattened_length() // last_dim
    let flat_shape = StaticIntTuple[2](prod_all_but_last_dim, last_dim)

    let input_buf = reshape[rank, 2, type, True](input, flat_shape)
    let output_buf = reshape[rank, 2, type, True](output, flat_shape)
    let skip_buf = reshape[rank, 2, type, True](skip, flat_shape)

    let num_workers = min(
        out_chain.get_runtime().parallelism_level(), prod_all_but_last_dim
    )
    let chunk_size = div_ceil(prod_all_but_last_dim, num_workers)

    @parameter
    fn task_func(thread_id: Int):
        let num_rows = min(
            chunk_size, prod_all_but_last_dim - thread_id * chunk_size
        )
        let row_idx = thread_id * chunk_size
        let thread_starting_coord = StaticIntTuple[2](row_idx, 0)
        let per_thread_dims = DimList(num_rows, last_dim)
        let output_buf_view = NDBuffer[2, DimList.create_unknown[2](), type](
            output_buf._offset(thread_starting_coord), per_thread_dims
        )
        let input_buf_view = NDBuffer[2, DimList.create_unknown[2](), type](
            input_buf._offset(thread_starting_coord), per_thread_dims
        )
        let skip_buf_view = NDBuffer[2, DimList.create_unknown[2](), type](
            skip_buf._offset(thread_starting_coord), per_thread_dims
        )

        @parameter
        fn add_skip[
            return_type: DType, simd_width: Int
        ](idx: Int, row: Int) -> SIMD[return_type, simd_width]:
            let coord = StaticIntTuple[2](row, idx)
            let skip_val = skip_buf_view.simd_load[simd_width](coord)
            let input_val = input_buf_view.simd_load[simd_width](coord)
            return (input_val + skip_val).cast[return_type]()

        layer_norm[simd_width, type, add_skip](
            output_buf_view, gamma, beta, eps
        )

    async_parallelize[task_func](out_chain, num_workers)


# ===----------------------------------------------------------------------===#
# Helpers
# ===----------------------------------------------------------------------===#


# Helper function to mark the output chain as ready in tests.
@mogg_register("mark_output_chain_ready")
@always_inline
fn mark_output_chain_ready(out_chain: OutputChainPtr):
    out_chain.mark_ready()


# Helper function to query buffer shapes for tests.
@mogg_register("print_shape_info")
fn print_buffer_info[
    type: DType, rank: Int
](buffer: NDBuffer[rank, DimList.create_unknown[rank](), type]):
    print("Rank:", rank)
    print("Shape:", buffer.dynamic_shape)
    print("Strides:", buffer.dynamic_stride)


# ===----------------------------------------------------------------------===#
# TopK/BottomK
# ===----------------------------------------------------------------------===#


@always_inline
fn bottom_k[
    type: DType,
    rank: Int,
    axis_type: DType,
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    k_buf: NDBuffer[1, DimList.create_unknown[1](), axis_type],
    axis_buf: NDBuffer[1, DimList.create_unknown[1](), axis_type],
    sorted: NDBuffer[1, DimList.create_unknown[1](), DType.bool],
    out_vals: NDBuffer[rank, DimList.create_unknown[rank](), type],
    out_idxs: NDBuffer[rank, DimList.create_unknown[rank](), DType.int64],
    out_chain: OutputChainPtr,
):
    _top_k[rank, type](
        input,
        k_buf[0].to_int(),
        axis_buf[0].to_int(),
        False,
        rebind[NDBuffer[rank, DimList.create_unknown[rank](), type]](out_vals),
        out_idxs,
        out_chain,
        sorted[0],
    )


@always_inline
fn top_k[
    type: DType,
    rank: Int,
    axis_type: DType,
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    k_buf: NDBuffer[1, DimList.create_unknown[1](), axis_type],
    axis_buf: NDBuffer[1, DimList.create_unknown[1](), axis_type],
    sorted: NDBuffer[1, DimList.create_unknown[1](), DType.bool],
    out_vals: NDBuffer[rank, DimList.create_unknown[rank](), type],
    out_idxs: NDBuffer[rank, DimList.create_unknown[rank](), DType.int64],
    out_chain: OutputChainPtr,
):
    _top_k[rank, type](
        input,
        k_buf[0].to_int(),
        axis_buf[0].to_int(),
        True,
        rebind[NDBuffer[rank, DimList.create_unknown[rank](), type]](out_vals),
        out_idxs,
        out_chain,
        sorted[0],
    )


# ===----------------------------------------------------------------------===#
# GatherND
# ===----------------------------------------------------------------------===#


@always_inline
fn gather_nd[
    type: DType,
    indices_type: DType,
    data_rank: Int,
    indices_rank: Int,
    output_rank: Int,
    batch_dims: Int,
    single_thread_blocking_override: Bool,
](
    data: NDBuffer[data_rank, DimList.create_unknown[data_rank](), type],
    indices: NDBuffer[
        indices_rank, DimList.create_unknown[indices_rank](), indices_type
    ],
    output: NDBuffer[output_rank, DimList.create_unknown[output_rank](), type],
    out_chain: OutputChainPtr,
):
    _gather_nd[
        type, indices_type, data_rank, indices_rank, output_rank, batch_dims
    ](data, indices, output)
    if not single_thread_blocking_override:
        out_chain.mark_ready()
