# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import OptionalReg
from collections.vector import InlinedFixedVector
from math import ceildiv, cos, erf, exp, fma, log, log1p, rsqrt, sin, sqrt
from random import randn, seed
from sys import external_call
from sys.info import simdwidthof, sizeof
from sys.intrinsics import strided_load
from sys.param_env import is_defined

from algorithm import argmax as _argmax
from algorithm import argmin as _argmin
from algorithm import sync_parallelize, vectorize
from algorithm.functional import _elementwise_impl
from algorithm.reduction import (
    _get_nd_indices_from_flat_index,
    _reduce_generator,
    _reduce_generator_cpu,
)
from algorithm.reduction import mean as _mean
from buffer import NDBuffer
from buffer.list import Dim, DimList
from builtin.simd import Int64, UInt8, UInt64, _pow
from extensibility import Tensor as ExtensibilityTensor
from gpu.host._compile import _get_nvptx_target
from LinAlg.BatchedMatmul import batched_matmul as _batched_matmul
from LinAlg.BatchedMatmul import batched_matmul_shape
from LinAlg.BatchedMatmul import (
    get_trace_information as get_trace_information_batched_matmul,
)
from LinAlg.Matmul import matmul as _matmul
from LinAlg.MatmulPack import (
    pack_b_ndbuffer,
    pack_matmul_b_shape_func,
    pack_transposed_b_ndbuffer,
)
from LinAlg.MatmulUtils import GemmShape, get_trace_information
from LinAlg.MatrixBandPart import matrix_band_part
from LinAlg.MatrixSolve import matrix_solve, matrix_solve_shape
from memory import memset_zero
from memory.unsafe import DTypePointer, Pointer, bitcast
from MOGGIntList import IntList
from MOGGTensor import Tensor
from nn._optional_param import OptionalParamInt
from nn.activations import gelu, relu
from nn.arange import arange, arange_shape
from nn.arg_nonzero import arg_nonzero, arg_nonzero_shape
from nn.concat import concat as _concat
from nn.concat import concat_shape as concat_from_list_shape
from nn.concat import variadic_list_to_vector
from nn.conv import ConvInfoStatic, conv_nhwc_direct, conv_shape
from nn.conv import pack_filter as _pack_conv_filter
from nn.conv import pack_filter_shape as _pack_conv_filter_shape
from nn.conv_transpose import conv_transpose_shape
from nn.conv_transpose import conv_transposed as conv_transpose_impl
from nn.conv_transpose import pack_filter as _pack_conv_transpose_filter
from nn.conv_transpose import (
    pack_filter_shape as _pack_conv_transpose_filter_shape,
)
from nn.cumsum import cumsum as _cumsum
from nn.flash_attention import flash_attention as cpu_flash_attention
from nn.flash_attention import (
    flash_attention_split_kv as cpu_flash_attention_split_kv,
)
from nn.gather_scatter import Axis
from nn.gather_scatter import gather as _gather
from nn.gather_scatter import gather_nd as _gather_nd
from nn.gather_scatter import (
    gather_nd_shape,
    gather_reduce,
    gather_shape,
    normalize_neg_index,
    scatter_elements,
)
from nn.gather_scatter import scatter_elements_shape as scatter_shape
from nn.gather_scatter import scatter_nd as _scatter_nd
from nn.gather_scatter import scatter_nd_generator, scatter_nd_shape
from nn.index_tensor import index_tensor_1d as _index_tensor
from nn.math import add, ceil, div, floor, mod, mul, sub, tanh
from nn.mha import flash_attention as gpu_flash_attention
from nn.mha import fused_attention as cpu_fused_attention_impl
from nn.nms import non_max_suppression, non_max_suppression_shape_func
from nn.normalization import layer_norm as _layer_norm
from nn.normalization import layer_norm_shape
from nn.pad import pad_constant as _pad_constant
from nn.pad import pad_reflect as _pad_reflect
from nn.pad import pad_repeat as _pad_repeat
from nn.pad import pad_shape
from nn.pool import avg_pool as _avg_pool
from nn.pool import max_pool as _max_pool
from nn.pool import pool_shape, pool_shape_ceil
from nn.reshape import ndbuffer_reshape, reshape, reshape_shape
from nn.resize import CoordinateTransformationMode, RoundMode
from nn.resize import resize_linear as resize_linear_kernel
from nn.resize import resize_nearest_neighbor
from nn.roi_align import roi_align_nhwc
from nn.slice import slice_as_view, slice_shape
from nn.softmax import logsoftmax as _logsoftmax
from nn.softmax import softmax as _softmax
from nn.split import split as _split
from nn.tile import tile, tile_shape
from nn.topk import top_k as _top_k
from nn.topk import top_k_shape
from register import *
from runtime.llcl import MojoCallContextPtr, Runtime
from runtime.tracing import Trace, TraceLevel

from utils import StaticTuple
from utils.index import Index, StaticIntTuple, product
from utils.loop import unroll
from utils.numerics import isinf, isnan


# Prevent these functions from being DCE'd by explicitly exporting them.
@export
fn MOGGExport():
    alias _add = add
    alias _avg_pool_shape = pool_shape
    alias _avg_pool_shape_ceil = pool_shape_ceil
    alias _cast = cast
    alias _ceil = ceil
    alias _concat_from_list_shape = concat_from_list_shape
    alias _conv_shape = conv_shape
    alias _conv_transpose_shape = conv_transpose_shape
    alias _div = div
    alias _floor = floor
    alias _gather_shape = gather_shape
    alias _gather_nd_shape = gather_nd_shape
    alias _gelu = gelu
    alias _pack_matmul_b_shape_func = pack_matmul_b_shape_func
    alias _pad_shape = pad_shape
    alias _layer_norm_shape = layer_norm_shape
    alias _pack_b_ndbuffer = pack_b_ndbuffer
    alias _pack_transposed_b_ndbuffer = pack_transposed_b_ndbuffer
    alias _max_pool_shape = pool_shape
    alias _max_pool_shape_ceil = pool_shape_ceil
    alias _matrix_solve_shape = matrix_solve_shape
    alias _matrix_band_part = matrix_band_part
    alias _batched_matmul_shape = batched_matmul_shape
    alias _mul = mul
    alias _mod = mod
    alias _tanh = tanh
    alias _arange = arange
    alias _arange_shape = arange_shape
    alias _relu = relu
    alias _reshape = reshape
    alias _reshape_shape = reshape_shape
    alias _ndbuffer_reshape = ndbuffer_reshape
    alias _scatter_shape = scatter_shape
    alias _scatter_nd_shape = scatter_nd_shape
    alias _slice = slice
    alias _sub = sub
    alias _random_shape = random_shape
    alias _roi_align_shape = roi_align_shape
    alias _slice_shape = slice_shape
    alias _arg_nonzero = arg_nonzero
    alias _arg_nonzero_shape = arg_nonzero_shape
    alias _top_k_shape = top_k_shape
    alias _tile = tile
    alias _tile_shape = tile_shape


# ===----------------------------------------------------------------------===#
# Feature gate for EAP
# TODO: Replace with entitlements when available.
# ===----------------------------------------------------------------------===#

alias MODULAR_PRODUCTION = is_defined["MODULAR_PRODUCTION"]()


@always_inline
fn _guard_against_gpu_target[
    target: StringLiteral
](ctx: MojoCallContextPtr) -> Bool:
    @parameter
    if MODULAR_PRODUCTION and target == "cuda":
        ctx.set_to_error("GPU Support is not available in the current release")
        return True
    else:
        return False


# ===----------------------------------------------------------------------===#
# Data structures used in MOGG/MGP ABI
# ===----------------------------------------------------------------------===#


# NOTE the layout must match `CompiledKernelABI::Tensor`
struct ABI_Tensor:
    var dims: Pointer[Int]
    var data: UnsafePointer[NoneType]


# NOTE the layout must match `CompiledKernelABI::List`
struct ABI_List:
    var num_elems: Int
    var elements: UnsafePointer[NoneType]


# ===----------------------------------------------------------------------===#
# Nop functions to expose different types to the compiler.
# ===----------------------------------------------------------------------===#


@mogg_register("bfloat16")
@export
fn DTypeBFloat16TypeDef(ty: DType.type) -> DType.type:
    return DType.bfloat16.value


@mogg_register("float16")
@export
fn DTypeFloat16TypeDef(ty: DType.type) -> DType.type:
    return DType.float16.value


@mogg_register("float32")
@export
fn DTypeFloat32TypeDef(ty: DType.type) -> DType.type:
    return DType.float32.value


@mogg_register("float64")
@export
fn DTypeFloat64TypeDef(ty: DType.type) -> DType.type:
    return DType.float64.value


@mogg_register("int8")
@export
fn DTypeInt8TypeDef(ty: DType.type) -> DType.type:
    return DType.int8.value


@mogg_register("int16")
@export
fn DTypeInt16TypeDef(ty: DType.type) -> DType.type:
    return DType.int16.value


@mogg_register("int32")
@export
fn DTypeInt32TypeDef(ty: DType.type) -> DType.type:
    return DType.int32.value


@mogg_register("uint32")
@export
fn DTypeUInt32TypeDef(ty: DType.type) -> DType.type:
    return DType.uint32.value


@mogg_register("uint64")
@export
fn DTypeUInt64TypeDef(ty: DType.type) -> DType.type:
    return DType.uint64.value


@mogg_register("int64")
@export
fn DTypeInt64TypeDef(ty: DType.type) -> DType.type:
    return DType.int64.value


@mogg_register("uint8")
@export
fn DTypeUInt8TypeDef(ty: DType.type) -> DType.type:
    return DType.uint8.value


@mogg_register("uint16")
@export
fn DTypeUInt16TypeDef(ty: DType.type) -> DType.type:
    return DType.uint16.value


@mogg_register("bool")
@export
fn DTypeBoolTypeDef(ty: DType.type) -> DType.type:
    return DType.bool.value


@mogg_register("index")
@export
fn IndexTypeDef(ty: Int) -> Int:
    return ty


@mogg_register("mojoCallContext")
@export
fn MojoCallContextDef(ty: MojoCallContextPtr):
    pass


@mogg_register("simd")
fn SimdTypeDef[
    type: DType, width: Int
](ty: SIMD[type, width]) -> SIMD[type, width]:
    return ty


@mogg_register("indices")
fn TensorIndicesTypeDef[
    rank: Int
](ty: StaticIntTuple[rank]) -> StaticIntTuple[rank]:
    return ty


@mogg_register("dim_type")
@export
fn DimTypeDef(ty: Dim) -> Dim:
    return ty


# ===----------------------------------------------------------------------===#
# Hooks to help build static shapes.
# ===----------------------------------------------------------------------===#


@mogg_register("create_unknown_dim")
@export
fn create_unknown_dim() -> Dim:
    return Dim()


@mogg_register("create_known_dim")
@export
fn create_known_dim[known_val: Int]() -> Dim:
    return Dim(known_val)


# ===----------------------------------------------------------------------===#
# Basic generated kernel building blocks
# ===----------------------------------------------------------------------===#


@mogg_register("extensibility_tensor_to_ndbuffer")
@always_inline
fn extensibility_tensor_to_ndbuffer[
    type: DType, rank: Int
](tensor: ExtensibilityTensor[type, rank]) -> NDBuffer[type, rank]:
    return NDBuffer[type, rank](tensor.data, tensor.shape, tensor.strides)


@mogg_register("to_buffer")
@always_inline
fn to_buffer[
    type: DType, rank: Int
](data: DTypePointer[type], shape: Pointer[Int],) -> NDBuffer[type, rank]:
    var shape_ptr = shape
    var shape_tuple = StaticIntTuple[rank]()

    var stride_tuple = StaticIntTuple[rank]()
    var stride: Int = 1

    @always_inline
    @__copy_capture(shape_ptr)
    @parameter
    fn body[idx: Int]():
        # Start from the back so we can accumulate the strides.
        var i = rank - 1 - idx
        shape_tuple[i] = shape_ptr.load(i)
        stride_tuple[i] = stride
        stride *= shape_tuple[i]

    unroll[body, rank]()

    return NDBuffer[type, rank](data, shape_tuple, stride_tuple)


@mogg_register("to_shape")
@always_inline
fn to_shape[rank: Int](shape: Pointer[Int]) -> StaticIntTuple[rank]:
    var shape_ptr = shape
    var shape_tuple = StaticIntTuple[rank]()

    @always_inline
    @__copy_capture(shape_ptr)
    @parameter
    fn body[idx: Int]():
        shape_tuple[idx] = shape_ptr.load(idx)

    unroll[body, rank]()

    return shape_tuple


# Convert a tensor into a shape.
@mogg_register("tensor_to_shape")
@always_inline
fn tensor_to_shape[
    type: DType,
    rank: Int,
](tensor: NDBuffer[type, 1]) -> StaticIntTuple[rank]:
    var out = StaticIntTuple[rank]()
    for i in range(rank):
        out[i] = int(tensor[i])

    return out


# Extract a value from a shape.
@mogg_register("get_int_from_shape")
@always_inline
fn get_int_from_shape[
    param_index: Int, rank: Int
](shape: StaticIntTuple[rank]) -> Int:
    return shape[param_index]


@mogg_register("shape_to_ndbuffer")
@always_inline
fn shape_to_ndbuffer[
    shape_rank: Int, buf_rank: Int, type: DType
](shape: StaticIntTuple[shape_rank], buf: NDBuffer[type, buf_rank]):
    @parameter
    for i in range(shape_rank):
        buf[i] = shape[i]


@mogg_register("shape_to_extensibility_tensor")
@always_inline
fn shape_to_extensibility_tensor[
    shape_rank: Int, buf_rank: Int, type: DType
](
    shape: StaticIntTuple[shape_rank],
    inout tensor: ExtensibilityTensor[type, buf_rank],
):
    @parameter
    for i in range(shape_rank):
        tensor.store[1](i, shape[i])


@mogg_register("to_buffer_list")
@always_inline
fn to_buffer_list[
    type: DType, rank: Int
](
    raw_list_ptr: UnsafePointer[NoneType],
) -> InlinedFixedVector[
    NDBuffer[type, rank]
]:
    # Cast input list pointer
    var abi_list_ptr = raw_list_ptr.bitcast[ABI_List]()
    var elems_ptr = abi_list_ptr[].elements
    var abi_tensors_ptr = elems_ptr.bitcast[ABI_Tensor]()

    # Create output list
    var num_elements = abi_list_ptr[].num_elems
    var out_list = InlinedFixedVector[NDBuffer[type, rank]](num_elements)

    # Convert individual elements of the input list into NDBuffer, and
    # accumulate the results to output list.
    for i in range(num_elements):
        var abi_tensor_ptr = abi_tensors_ptr + i
        var dims = abi_tensor_ptr[].dims
        var data = abi_tensor_ptr[].data.bitcast[Scalar[type]]()
        var buffer = to_buffer[type, rank](data, dims)
        out_list.append(buffer)

    return out_list


@mogg_register("destruct_buffer_list")
@always_inline
fn destruct_buffer_list[
    type: DType, rank: Int
](owned list: InlinedFixedVector[NDBuffer[type, rank]]):
    # Must call destructor explicitly until `InlinedFixedVector` has `__del__`
    list._del_old()


# TODO(#27757): All calls with concrete body functions are as if annotated with
#               @mogg_register("mo.original_op")
@mogg_register("elementwise")
@always_inline
fn elementwise_wrapper[
    trace_description: StringLiteral,
    simd_width: Int,
    type: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
    func: fn[width: Int, rank: Int] (StaticIntTuple[rank]) capturing -> None,
    /,
    target: StringLiteral = "cpu",
](buffer: NDBuffer[type, rank], ctx: MojoCallContextPtr,):
    @always_inline
    @parameter
    fn description_fn() -> String:
        var name_str = String("name=") + trace_description
        var shape_str = String("shape=") + String("x").join(buffer.get_shape())

        var vector_width_str = String("vector_width=") + str(simd_width)

        var info = String(";").join(name_str, shape_str, vector_width_str)

        return (
            info
            + String(";single_thread_blocking_override=")
            + str(single_thread_blocking_override)
        )

    if _guard_against_gpu_target[target](ctx):
        return

    with Trace[TraceLevel.OP](
        "mojo.elementwise",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
    ) as t:
        _elementwise_impl[
            func,
            simd_width,
            rank,
            use_blocking_impl=single_thread_blocking_override,
            target=target,
        ](
            buffer.dynamic_shape,
        )


# ===----------------------------------------------------------------------===#
# Tensor API intrinsics
# ===----------------------------------------------------------------------===#


@mogg_register("to_tensor")
@export
@always_inline
fn to_tensor[
    type: DType,
    static_shape: DimList = DimList(),
    static_strides: DimList = DimList(),
](
    data: DTypePointer[type],
    raw_shape_ptr: Pointer[Int],
    length: Int,
) -> Tensor[type, static_shape, static_strides, _OWNED_MEMORY=False]:
    var shape_ptr = raw_shape_ptr

    var shape = IntList[static_shape].empty(length)
    var strides = IntList[static_strides].empty(length)

    var stride: Int = 1

    @parameter
    if shape.has_static_length():
        alias rank = len(static_shape)

        @always_inline
        @__copy_capture(shape_ptr)
        @parameter
        fn body[idx: Int]():
            # Start from the back so we can accumulate the strides.
            var i = rank - 1 - idx
            shape[i] = shape_ptr.load(i)
            strides[i] = stride
            stride *= shape[i]

        unroll[body, rank]()
    else:
        # Start from the back so we can accumulate the strides.
        for i in reversed(range(length)):
            shape[i] = shape_ptr.load(i)
            strides[i] = stride
            stride *= shape[i]

    return Tensor[type, static_shape, static_strides, _OWNED_MEMORY=False](
        data,
        shape,
        strides,
        UnsafePointer[Scalar[DType.index]](),
    )


@mogg_register("to_extensibility_tensor")
@export
@always_inline
fn to_extensibility_tensor[
    type: DType, rank: Int
](
    data: DTypePointer[type],
    shape: Pointer[Int],
) -> ExtensibilityTensor[
    type, rank
]:
    var shape_ptr = shape
    var shape_tuple = StaticIntTuple[rank]()
    var stride_tuple = StaticIntTuple[rank]()
    var stride: Int = 1

    @always_inline
    @__copy_capture(shape_ptr)
    @parameter
    fn body[idx: Int]():
        # Start from the back so we can accumulate the strides.
        var i = rank - 1 - idx
        shape_tuple[i] = shape_ptr.load(i)
        stride_tuple[i] = stride
        stride *= shape_tuple[i]

    unroll[body, rank]()

    return ExtensibilityTensor[type, rank](
        data,
        shape_tuple,
        stride_tuple,
    )


@mogg_register("get_shape_of_tensor")
@always_inline
@export
fn get_shape_of_tensor[
    rank: Int, type: DType
](tensor: ExtensibilityTensor[type, rank]) -> StaticIntTuple[rank]:
    var shape_tuple = StaticIntTuple[rank]()
    var tensor_shape = tensor.shape
    for i in range(rank):
        shape_tuple[i] = tensor_shape[i]
    return shape_tuple


@mogg_register("mogg.shape_from_kgen")
@always_inline
@export
fn get_static_shape(shape: IntList) -> StaticIntTuple[shape._safe_len]:
    return shape.stack_alloc_data


# ===----------------------------------------------------------------------===#
# Simd load/store helper functions
# ===----------------------------------------------------------------------===#


@always_inline
fn _compute_flat_index[
    type: DType, rank: Int, iters: Int, static_shape: DimList
](
    buffer: NDBuffer[type, rank, static_shape],
    index: StaticIntTuple[rank],
) -> Int:
    var flat_index: Int = 0

    @always_inline
    @parameter
    fn body[idx: Int]():
        flat_index = fma(index[idx], buffer.dynamic_stride[idx], flat_index)

    unroll[body, iters]()
    return flat_index


@always_inline
fn _simd_load_internal[
    simd_width: Int, type: DType, rank: Int, static_shape: DimList
](buffer: NDBuffer[type, rank, static_shape], index: Int) -> SIMD[
    type, simd_width
]:
    @parameter
    if type == DType.bool:
        var v = SIMD[size=simd_width].load(
            buffer.data.bitcast[DType.uint8](), index
        )
        return v.cast[type]()
    else:
        return SIMD[size=simd_width].load(buffer.data, index)


@mogg_register("simd_load")
@export
@always_inline
fn simd_load[
    type: DType, simd_width: Int, rank: Int, input_0_static_shape: DimList
](
    buffer: NDBuffer[type, rank, input_0_static_shape],
    index: StaticIntTuple[rank],
) -> SIMD[type, simd_width]:
    var flat_index = _compute_flat_index[
        type, rank, rank, input_0_static_shape
    ](buffer, index)
    var stride = buffer.dynamic_stride[rank - 1]

    if buffer.dynamic_stride[rank - 1] == 0:
        return Scalar.load(buffer.data, flat_index)
    elif stride > 1:

        @parameter
        if type == DType.bool:
            var v = strided_load[DType.uint8, simd_width](
                buffer.data.bitcast[DType.uint8]().offset(flat_index), stride
            )
            return v.cast[type]()
        else:
            return strided_load[type, simd_width](
                buffer.data.offset(flat_index), stride
            )
    return _simd_load_internal[simd_width, type, rank, input_0_static_shape](
        buffer, flat_index
    )


@mogg_register("simd_store")
@export
@always_inline
fn simd_store[
    type: DType, simd_width: Int, rank: Int
](
    buffer: NDBuffer[type, rank],
    index: StaticIntTuple[rank],
    val: SIMD[type, simd_width],
):
    var flat_index = _compute_flat_index[type, rank, rank](buffer, index)

    # We have to cast bools into their runtime storage type.
    @parameter
    if type == DType.bool:
        var v = val.cast[DType.uint8]()
        SIMD[size=simd_width].store(
            buffer.data.bitcast[DType.uint8](), flat_index, v
        )
    else:
        SIMD[size=simd_width].store(buffer.data, flat_index, val)


# ===----------------------------------------------------------------------===#
# Broadcast
# ===----------------------------------------------------------------------===#


@mogg_register("mo.broadcast_shape")
@always_inline
@export
fn broadcast_shape[
    lhs_type: DType,
    rhs_type: DType,
    out_type: DType,
](
    lhs_buf: NDBuffer[lhs_type, 1],
    rhs_buf: NDBuffer[rhs_type, 1],
    out_buf: NDBuffer[out_type, 1],
    ctx: MojoCallContextPtr,
):
    var lhs_size = lhs_buf.size()
    var rhs_size = rhs_buf.size()
    if lhs_size > rhs_size:
        return broadcast_shape_impl(rhs_buf, lhs_buf, out_buf)
    return broadcast_shape_impl(lhs_buf, rhs_buf, out_buf)


@always_inline
fn broadcast_shape_impl[
    lhs_type: DType,
    rhs_type: DType,
    out_type: DType,
](
    lhs_buf: NDBuffer[lhs_type, 1],
    rhs_buf: NDBuffer[rhs_type, 1],
    out_buf: NDBuffer[out_type, 1],
):
    # Ensure lhs is always the smaller shape
    var lhs_rank = lhs_buf.size()
    var rhs_rank = rhs_buf.size()
    debug_assert(lhs_rank <= rhs_rank, "lhs shape must be the smaller one")

    # lhs_buf =      [l0, l1, ...]
    # rhs_buf = [..., r0, r1, ...]
    # out_buf = [..., o0, o1, ...]
    var size_diff = rhs_rank - lhs_rank
    for i in range(size_diff):
        out_buf[i] = rhs_buf[i].cast[out_type]()

    for lhs_idx in range(lhs_rank):
        var rhs_idx = lhs_idx + size_diff
        var lhs_dim = int(lhs_buf[lhs_idx])
        var rhs_dim = int(rhs_buf[rhs_idx])
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


@mogg_register("broadcast_shape_shape")
@always_inline
@export
fn broadcast_shape_shape[
    lhs_type: DType,
    rhs_type: DType,
    single_thread_blocking_override: Bool,
](
    lhs_buf: NDBuffer[lhs_type, 1],
    rhs_buf: NDBuffer[rhs_type, 1],
    ctx: MojoCallContextPtr,
) -> StaticIntTuple[1]:
    var lhs_dim = lhs_buf.dim(0)
    var rhs_dim = rhs_buf.dim(0)
    return StaticIntTuple[1](max(lhs_dim, rhs_dim))


@mogg_register("mo.static.broadcast_to")
@mogg_view_op
@always_inline
fn broadcast_to_tensor[
    type: DType,
    original_rank: Int,
    target_rank: Int,
    output_rank: Int,
    single_thread_blocking_override: Bool,
    target: StringLiteral = "cpu",
](
    original: NDBuffer[type, original_rank],
    target_shape: StaticIntTuple[target_rank],
) -> NDBuffer[type, output_rank]:
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
            shape[i] = target_shape[i]
            stride[i] = 0
        else:
            shape[i] = original.dim(i)
            stride[i] = original.stride(i)

    # Broadcast in dimensions the original started with.
    @always_inline
    @parameter
    fn broadcast_dim[small_index: Int]():
        # We are traversing as if they are the same size.
        var big_index = small_index + offset

        # Switch the indexes depending on which is bigger.
        var orig_index = small_index
        var target_index = big_index

        @parameter
        if target_rank < original_rank:
            orig_index = big_index
            target_index = small_index

        # If the dims are the same use the stride of the original.
        if original.dim(orig_index) == target_shape[target_index]:
            stride[big_index] = original.stride(orig_index)
            shape[big_index] = original.dim(orig_index)
        elif original.dim(orig_index) == 1:
            # If they don't match and original is 1 then we broadcast.
            stride[big_index] = 0
            shape[big_index] = target_shape[target_index]
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
        unroll[add_new_dims, original_rank - target_rank]()
        offset = original_rank - target_rank
        unroll[broadcast_dim, target_rank]()
    else:
        unroll[add_new_dims, target_rank - original_rank]()
        offset = target_rank - original_rank
        unroll[broadcast_dim, original_rank]()

    # Create a view of the original data with the new shape and strides.
    var out = NDBuffer[type, output_rank](
        original.data,
        shape,
        stride,
    )

    return out


@mogg_register("broadcast_to_shape")
@always_inline
fn broadcast_to_shape[
    input_rank: Int,
    output_rank: Int,
    input_type: DType,
    target_shape_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[input_type, input_rank],
    target_shape_buf: NDBuffer[target_shape_type, 1],
) raises -> StaticIntTuple[output_rank]:
    if output_rank != target_shape_buf.dim(0):
        raise Error(
            "[broadcast_to] requires (len(target_shape) == output_rank)"
        )
    if input_rank > output_rank:
        raise Error("[broadcast_to] requires (input_rank <= output_rank)")

    # move the output shape from buffer into a static int tuple
    var output_shape = StaticIntTuple[output_rank]()

    for axis in range(output_rank):
        output_shape[axis] = int(target_shape_buf[axis])

    # Validate the compatibility between input and output shapes
    # NOTE we don't need to check the padded dims
    for i in range(input_rank):
        var input_axis = input_rank - i - 1
        var output_axis = output_rank - i - 1
        var input_dim = input_buf.dim(input_axis)
        var output_dim = output_shape[output_axis]
        if input_dim != 1 and input_dim != output_dim:
            raise Error(
                "[broadcast_to] input dim must be either 1 or equal to"
                " corresponding output dim starting from the rightmost dim"
            )
    return output_shape


# When we have many SIMD types in one kernel we need to use the `min` of them.
# This involves applying parameter expressions to this result which must be
# `mlir.index` typed so we need to return as `mlir.index` and then cast to int.
@mogg_register("simd_target")
fn get_target_simd[type: DType]() -> __mlir_type.index:
    return int(simdwidthof[type]()).value


@mogg_register("simd_target_cuda")
fn get_target_simd_cuda[type: DType]() -> __mlir_type.index:
    return int(simdwidthof[Scalar[type], target = _get_nvptx_target()]()).value


@mogg_register("simd_target_to_int")
fn simd_width_to_int[simd_width: __mlir_type.index]() -> Int:
    return Int(simd_width)


# ===----------------------------------------------------------------------===#
# Abs wrapper op
# ===----------------------------------------------------------------------===#


# Call abs, needed as it has multiple overloads which can't be aliased
@mogg_register("mo.abs")
@mogg_elementwise
@always_inline
@export
fn abs_wrapped[
    type: DType, simd_width: Int
](value: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return abs(value)


# ===----------------------------------------------------------------------===#
# ArgMax wrapper op
# ===----------------------------------------------------------------------===#


# Call argmax, needed as it has multiple overloads which can't be aliased
@mogg_register("mo.arg_max")
@always_inline
@export
fn argmax_wrapped[
    type: DType,
    input_0_static_shape: DimList,
    axis_type: DType,
    input_1_static_shape: DimList,
    out_type: DType,
    input_2_static_shape: DimList,
    rank: Int,
](
    input: NDBuffer[type, rank, input_0_static_shape],
    axis_buf: NDBuffer[axis_type, 1, input_1_static_shape],
    output: NDBuffer[out_type, rank, input_2_static_shape],
    ctx: MojoCallContextPtr,
) raises:
    _argmax(input, axis_buf, output)


# ===----------------------------------------------------------------------===#
# ArgMin wrapper op
# ===----------------------------------------------------------------------===#


# Call argmin, needed as it has multiple overloads which can't be aliased
@mogg_register("mo.arg_min")
@always_inline
@export
fn argmin_wrapped[
    type: DType,
    input_0_static_shape: DimList,
    axis_type: DType,
    input_1_static_shape: DimList,
    out_type: DType,
    input_2_static_shape: DimList,
    rank: Int,
](
    input: NDBuffer[type, rank, input_0_static_shape],
    axis_buf: NDBuffer[axis_type, 1, input_1_static_shape],
    output: NDBuffer[out_type, rank, input_2_static_shape],
    ctx: MojoCallContextPtr,
) raises:
    _argmin(input, axis_buf, output)


# ===----------------------------------------------------------------------===#
# Cast op
# ===----------------------------------------------------------------------===#


# Cast a SIMD value to a new SIMD value of different type.
@mogg_register("mo.cast")
@mogg_elementwise
@always_inline
@export
fn cast[
    type: DType, new_type: DType, simd_width: Int
](value: SIMD[type, simd_width]) -> SIMD[new_type, simd_width]:
    return value.cast[new_type]()


# ===----------------------------------------------------------------------===#
# Concat op
# ===----------------------------------------------------------------------===#


@mogg_register("mo.concat_from_list")
@always_inline
@export
fn concat_from_list[
    input_type: DType,
    input_rank: Int,
    simd_width: Int,
    single_thread_blocking_override: Bool,
    axis_type: DType,
](
    inputs: InlinedFixedVector[NDBuffer[input_type, input_rank]],
    axis: NDBuffer[axis_type, 1],
    output: NDBuffer[input_type, input_rank],
    ctx: MojoCallContextPtr,
) raises:
    with Trace[TraceLevel.OP]("mojo.concat_from_list") as t:
        _concat[input_rank, input_type, single_thread_blocking_override](
            output, int(normalize_neg_index(axis[0], input_rank)), inputs
        )


@mogg_register("mo.concat")
@always_inline
@export
fn concat[
    type: DType,
    rank: Int,
    simd_width: Int,
    single_thread_blocking_override: Bool,
    axis_type: DType,
    target: StringLiteral = "cpu",
](
    output: NDBuffer[type, rank],
    axis: NDBuffer[axis_type, 1],
    ctx: MojoCallContextPtr,
    *variadic_ins: NDBuffer[type, rank],
) raises:
    if _guard_against_gpu_target[target](ctx):
        return

    with Trace[TraceLevel.OP]("mojo.concat") as t:
        var ins = variadic_list_to_vector(variadic_ins)
        _concat[rank, type, single_thread_blocking_override, target](
            output,
            int(normalize_neg_index(axis[0], rank)),
            ins,
        )
        ins._del_old()


@mogg_register("concat_shape")
@always_inline
fn concat_shape[
    input_type: DType,
    input_rank: Int,
    simd_width: Int,
    single_thread_blocking_override: Bool,
    axis_type: DType,
](
    axis_buf: NDBuffer[axis_type, 1],
    *input_bufs: NDBuffer[input_type, input_rank],
) raises -> StaticIntTuple[input_rank]:
    # TODO we should refactor this with `concat_from_list_shape`, but this
    # variadic input version has more static info (we _always_ know how many
    # input buffers there'll be for each invocation), thereby yielding simpler
    # KGEN IR. In order to obtain the same simple IR via
    # `concat_from_list_shape`, we need the KGEN optimizations to
    # 1. perform "@unroll_if_possible" when the loop bounds are not parameters.
    # 2. either optimize away the conversion from variadic list to
    # InlinedFixedVector, or generalize `concat_from_list_shape` with some List
    # Trait so we can pass in variadic list directly.
    var axis = int(axis_buf[0])
    if axis < 0:
        axis += input_rank
    if axis < 0 or input_rank <= axis:
        raise ("[concat] normalized axis must be within range [0, input_rank)")

    @parameter
    @always_inline
    fn shape_equal_ignore_axis(
        s1: StaticIntTuple[input_rank], s2: StaticIntTuple[input_rank]
    ) -> Bool:
        for i in range(input_rank):
            if i != axis and s1[i] != s2[i]:
                return False
        return True

    var concat_axis_dim_sum = 0

    for i in range(input_bufs.__len__()):
        concat_axis_dim_sum += input_bufs[i].dim(axis)
        if not shape_equal_ignore_axis(
            input_bufs[0].get_shape(), input_bufs[i].get_shape()
        ):
            raise Error(
                "[concat] input shapes must match except at concat axis"
            )

    # compute and return the output shape
    var output_shape = input_bufs[0].get_shape()
    output_shape[axis] = concat_axis_dim_sum
    return output_shape


# ===----------------------------------------------------------------------===#
# avg_pool
# ===----------------------------------------------------------------------===#


@mogg_register("mo.avg_pool")
@always_inline
@export
fn avg_pool[
    type: DType,
    int_type: DType,
    count_boundary: Bool,
](
    input: NDBuffer[type, 4],
    filter: NDBuffer[int_type, 1],
    strides: NDBuffer[int_type, 1],
    dilations: NDBuffer[int_type, 1],
    paddings: NDBuffer[int_type, 1],
    output: NDBuffer[type, 4],
    ctx: MojoCallContextPtr,
):
    return _avg_pool[count_boundary=count_boundary](
        input, filter, strides, dilations, paddings, output
    )


# This handles avg_pool in the case where ceilMode = True. The default
# (ceilMode = False) case is handled by avg_pool above.
@mogg_register("mo.avg_pool_ceil_mode_true")
@always_inline
@export
fn avg_pool_ceil_mode_true[
    type: DType,
    int_type: DType,
    count_boundary: Bool,
](
    input: NDBuffer[type, 4],
    filter: NDBuffer[int_type, 1],
    strides: NDBuffer[int_type, 1],
    dilations: NDBuffer[int_type, 1],
    paddings: NDBuffer[int_type, 1],
    output: NDBuffer[type, 4],
    ctx: MojoCallContextPtr,
):
    return _avg_pool[count_boundary=count_boundary](
        input, filter, strides, dilations, paddings, output, True
    )


# ===----------------------------------------------------------------------===#
# max_pool
# ===----------------------------------------------------------------------===#


@mogg_register("mo.max_pool")
@always_inline
@export
fn max_pool[
    type: DType,
    int_type: DType,
](
    input: NDBuffer[type, 4],
    filter: NDBuffer[int_type, 1],
    strides: NDBuffer[int_type, 1],
    dilations: NDBuffer[int_type, 1],
    paddings: NDBuffer[int_type, 1],
    output: NDBuffer[type, 4],
    ctx: MojoCallContextPtr,
):
    return _max_pool(input, filter, strides, dilations, paddings, output)


# This handles max_pool in the case where ceilMode = True. The default
# (ceilMode = False) case is handled by max_pool above.
@mogg_register("mo.max_pool_ceil_mode_true")
@always_inline
@export
fn max_pool_ceil_mode_true[
    type: DType,
    int_type: DType,
](
    input: NDBuffer[type, 4],
    filter: NDBuffer[int_type, 1],
    strides: NDBuffer[int_type, 1],
    dilations: NDBuffer[int_type, 1],
    paddings: NDBuffer[int_type, 1],
    output: NDBuffer[type, 4],
    ctx: MojoCallContextPtr,
):
    return _max_pool(input, filter, strides, dilations, paddings, output, True)


# ===----------------------------------------------------------------------===#
# Cumsum op
# ===----------------------------------------------------------------------===#


@mogg_register("mo.cumsum")
@always_inline
@export
fn cumsum[
    type: DType,
    axis_type: DType,
    rank: Int,
    exclusive: Int,
    reverse: Int,
](
    input: NDBuffer[type, rank],
    axis: NDBuffer[axis_type, 1],
    output: NDBuffer[type, rank],
    ctx: MojoCallContextPtr,
):
    _cumsum[rank, type, exclusive == 1, reverse == 1](
        output, input, int(normalize_neg_index(axis[0], rank))
    )


# ===----------------------------------------------------------------------===#
# Split op
# ===----------------------------------------------------------------------===#


# Not targeted yet because MOGG assumes single output
@mogg_register("mo.split")
@always_inline
@export
fn split[
    type: DType,
    rank: Int,
    simd_width: Int,
    axis_type: DType,
    split_sizes_type: DType,
](
    input: NDBuffer[type, rank],
    split_sizes: NDBuffer[split_sizes_type, 1],
    axis: NDBuffer[axis_type, 1],
    ctx: MojoCallContextPtr,
    *variadic_outs: NDBuffer[type, rank],
) raises:
    # NOTE: Synchronous, so stack allocated variadic list is safe
    _split[type, rank](
        input, int(normalize_neg_index(axis[0], rank)), variadic_outs
    )


# ===----------------------------------------------------------------------===#
# Pow wrapper op
# ===----------------------------------------------------------------------===#


# Call pow, needed as it has multiple overloads which can't be aliased
@mogg_register("mo.pow")
@mogg_elementwise
@always_inline
@export
fn pow_wrapped[
    type: DType, power_type: DType, simd_width: Int
](value: SIMD[type, simd_width], power: SIMD[power_type, simd_width]) -> SIMD[
    type, simd_width
]:
    return _pow(value, power)


# ===----------------------------------------------------------------------===#
# Sqrt wrapper op
# ===----------------------------------------------------------------------===#


# Call sqrt, needed as it has multiple overloads which can't be aliased
@mogg_register("mo.sqrt")
@mogg_elementwise
@always_inline
@export
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
@export
fn mogg_max[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    type, simd_width
]:
    return max(x, y)


@mogg_register("mo.min")
@mogg_elementwise
@always_inline
@export
fn mogg_min[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    type, simd_width
]:
    return min(x, y)


# ===----------------------------------------------------------------------===#
# Mean op
# ===----------------------------------------------------------------------===#


@mogg_register("mo.mean")
@always_inline
@export
fn mean[
    type: DType,
    index_type: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
    input_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
    /,
    target: StringLiteral = "cpu",
](
    input_shape: StaticIntTuple[rank],
    axis_buffer: NDBuffer[index_type, 1],
    output_shape: StaticIntTuple[rank],
    ctx: MojoCallContextPtr,
) raises:
    if _guard_against_gpu_target[target](ctx):
        return

    var axis = axis_buffer[0]

    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            String("input_shape=") + String("x").join(input_shape),
            String("output_shape=") + String("x").join(output_shape),
        )

    with Trace[TraceLevel.OP](
        "mojo.mean", Trace[TraceLevel.OP]._get_detail_str[description_fn]()
    ) as t:
        _mean[
            type,
            input_0_fn,
            output_0_fn,
            target=target,
            single_thread_blocking_override=single_thread_blocking_override,
        ](
            input_shape,
            int(axis),
            output_shape,
        )


# ===----------------------------------------------------------------------===#
# Negative op
# ===----------------------------------------------------------------------===#


@mogg_register("mo.negative")
@always_inline("nodebug")
@export
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
# Pad .* op
# ===----------------------------------------------------------------------===#


@mogg_register("mo.pad.constant")
@always_inline
@export
fn pad_constant[
    rank: Int,
    type: DType,
    paddings_type: DType,
    constant_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[type, rank],
    paddings_buf: NDBuffer[paddings_type, 1],
    constant_buf: NDBuffer[constant_type, 1],
    output_buf: NDBuffer[type, rank],
    ctx: MojoCallContextPtr,
):
    var paddings_ptr = paddings_buf.data
    var constant_simd = constant_buf[0]

    _pad_constant(output_buf, input_buf, paddings_ptr, constant_simd)


@mogg_register("mo.pad.reflect")
@always_inline
@export
fn pad_reflect[
    rank: Int,
    type: DType,
    paddings_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[type, rank],
    paddings_buf: NDBuffer[paddings_type, 1],
    output_buf: NDBuffer[type, rank],
    ctx: MojoCallContextPtr,
):
    var paddings_ptr = paddings_buf.data

    _pad_reflect(output_buf, input_buf, paddings_ptr)


@mogg_register("mo.pad.repeat")
@always_inline
@export
fn pad_repeat[
    rank: Int,
    type: DType,
    paddings_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[type, rank],
    paddings_buf: NDBuffer[paddings_type, 1],
    output_buf: NDBuffer[type, rank],
    ctx: MojoCallContextPtr,
):
    var paddings_ptr = paddings_buf.data

    _pad_repeat(output_buf, input_buf, paddings_ptr)


# ===----------------------------------------------------------------------===#
# Reduction ops
# ===----------------------------------------------------------------------===#


@mogg_register("reduce_shape")
@always_inline("nodebug")
fn reduce_shape[
    input_rank: Int,
    input_type: DType,
    axis_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[input_type, input_rank],
    axis_buf: NDBuffer[axis_type, 1],
) raises -> StaticIntTuple[input_rank]:
    """
    Compute the output shape of a `pad` operation, and assert the inputs are
    compatible.

    Parameters:
        input_rank: Input_rank of the input tensor.
        input_type: Type of the input tensor.
        axis_type: Type of the axis tensor.
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.

    Args:
        input_buf: The input tensor.
        axis_buf: The axis tensor.

    Returns:
        The output shape.
    """

    # extract hyper parameter
    var axis = int(axis_buf[0])
    if axis < 0:
        axis += input_rank

    if axis < 0 or input_rank <= axis:
        raise Error(
            "[reduction] normalized axis must be within range [0, input_rank)"
        )

    # compute and return the output shape
    var output_shape = input_buf.get_shape()
    output_shape[axis] = 1
    return output_shape


@mogg_register("mo.reduce.add")
@always_inline
@export
fn reduce_add[
    type: DType,
    index_type: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
    input_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
    target: StringLiteral = "cpu",
](
    input_shape: StaticIntTuple[rank],
    axis_buffer: NDBuffer[index_type, 1],
    output_shape: StaticIntTuple[rank],
    ctx: MojoCallContextPtr,
) raises:
    if _guard_against_gpu_target[target](ctx):
        return

    @always_inline
    @parameter
    fn input_0_fn_wrapper[
        _type: DType, width: Int, rank: Int
    ](idx: StaticIntTuple[rank]) -> SIMD[_type, width]:
        return rebind[SIMD[_type, width]](input_0_fn[width, rank](idx))

    @always_inline
    @parameter
    fn output_0_fn_wrapper[
        _type: DType, width: Int, rank: Int
    ](indices: StaticIntTuple[rank], value: SIMD[_type, width]):
        output_0_fn[width, rank](indices, rebind[SIMD[type, width]](value))

    @always_inline
    @parameter
    fn reduce_impl[
        ty: DType, width: Int
    ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
        return v1 + v2

    var axis = axis_buffer[0]
    with Trace[TraceLevel.OP]("mojo.reduce_add") as t:
        _reduce_generator[
            input_0_fn_wrapper,
            output_0_fn_wrapper,
            reduce_impl,
            target=target,
            single_thread_blocking_override=single_thread_blocking_override,
        ](input_shape, Scalar[type](0), int(axis))


@mogg_register("mo.reduce.max")
@always_inline
@export
fn reduce_max[
    type: DType,
    index_type: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
    input_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
    target: StringLiteral = "cpu",
](
    input_shape: StaticIntTuple[rank],
    axis_buffer: NDBuffer[index_type, 1],
    output_shape: StaticIntTuple[rank],
    ctx: MojoCallContextPtr,
) raises:
    if _guard_against_gpu_target[target](ctx):
        return

    @always_inline
    @parameter
    fn input_0_fn_wrapper[
        _type: DType, width: Int, rank: Int
    ](idx: StaticIntTuple[rank]) -> SIMD[_type, width]:
        return rebind[SIMD[_type, width]](input_0_fn[width, rank](idx))

    @always_inline
    @parameter
    fn output_0_fn_wrapper[
        _type: DType, width: Int, rank: Int
    ](indices: StaticIntTuple[rank], value: SIMD[_type, width]):
        output_0_fn[width, rank](indices, rebind[SIMD[type, width]](value))

    @always_inline
    @parameter
    fn reduce_impl[
        ty: DType, width: Int
    ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
        return max(v1, v2)

    var axis = axis_buffer[0]
    with Trace[TraceLevel.OP]("mojo.reduce_max") as t:
        _reduce_generator[
            input_0_fn_wrapper,
            output_0_fn_wrapper,
            reduce_impl,
            target=target,
            single_thread_blocking_override=single_thread_blocking_override,
        ](input_shape, Scalar[type].MIN, int(axis))


@mogg_register("mo.reduce.min")
@always_inline
@export
fn reduce_min[
    type: DType,
    index_type: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
    input_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
    target: StringLiteral = "cpu",
](
    input_shape: StaticIntTuple[rank],
    axis_buffer: NDBuffer[index_type, 1],
    output_shape: StaticIntTuple[rank],
    ctx: MojoCallContextPtr,
) raises:
    if _guard_against_gpu_target[target](ctx):
        return

    @always_inline
    @parameter
    fn input_0_fn_wrapper[
        _type: DType, width: Int, rank: Int
    ](idx: StaticIntTuple[rank]) -> SIMD[_type, width]:
        return rebind[SIMD[_type, width]](input_0_fn[width, rank](idx))

    @always_inline
    @parameter
    fn output_0_fn_wrapper[
        _type: DType, width: Int, rank: Int
    ](indices: StaticIntTuple[rank], value: SIMD[_type, width]):
        output_0_fn[width, rank](indices, rebind[SIMD[type, width]](value))

    @always_inline
    @parameter
    fn reduce_impl[
        ty: DType, width: Int
    ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
        return min(v1, v2)

    var axis = axis_buffer[0]

    with Trace[TraceLevel.OP]("mojo.reduce_min") as t:
        _reduce_generator[
            input_0_fn_wrapper,
            output_0_fn_wrapper,
            reduce_impl,
            target=target,
            single_thread_blocking_override=single_thread_blocking_override,
        ](input_shape, Scalar[type].MAX, int(axis))


@mogg_register("mo.reduce.mul")
@always_inline
@export
fn reduce_mul[
    type: DType,
    index_type: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
    input_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
    target: StringLiteral = "cpu",
](
    input_shape: StaticIntTuple[rank],
    axis_buffer: NDBuffer[index_type, 1],
    output_shape: StaticIntTuple[rank],
    ctx: MojoCallContextPtr,
) raises:
    if _guard_against_gpu_target[target](ctx):
        return

    @always_inline
    @parameter
    fn input_0_fn_wrapper[
        _type: DType, width: Int, rank: Int
    ](idx: StaticIntTuple[rank]) -> SIMD[_type, width]:
        return rebind[SIMD[_type, width]](input_0_fn[width, rank](idx))

    @always_inline
    @parameter
    fn output_0_fn_wrapper[
        _type: DType, width: Int, rank: Int
    ](indices: StaticIntTuple[rank], value: SIMD[_type, width]):
        output_0_fn[width, rank](indices, rebind[SIMD[type, width]](value))

    @always_inline
    @parameter
    fn reduce_impl[
        ty: DType, width: Int
    ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
        return v1 * v2

    var axis = axis_buffer[0]

    with Trace[TraceLevel.OP]("mojo.reduce_mul") as t:
        _reduce_generator[
            input_0_fn_wrapper,
            output_0_fn_wrapper,
            reduce_impl,
            target=target,
            single_thread_blocking_override=single_thread_blocking_override,
        ](input_shape, Scalar[type](1), int(axis))


# ===----------------------------------------------------------------------===#
# Slice op
# ===----------------------------------------------------------------------===#


# Wrapper for slice here to include the `single_thread_blocking_override`.
@mogg_register("mo.slice")
@mogg_view_op
@always_inline
@export
fn slice[
    type: DType,
    start_type: DType,
    end_type: DType,
    step_type: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
    target: StringLiteral = "cpu",
](
    tensor: NDBuffer[type, rank],
    starts: NDBuffer[start_type, 1],
    ends: NDBuffer[end_type, 1],
    steps: NDBuffer[step_type, 1],
    ctx: MojoCallContextPtr,  # remove (#24946)
) -> NDBuffer[type, rank]:
    if _guard_against_gpu_target[target](ctx):
        return tensor

    return slice_as_view(tensor, starts, ends, steps)


# ===----------------------------------------------------------------------===#
# SqueezeShape
# ===----------------------------------------------------------------------===#


@mogg_register("mo.squeeze_shape")
@always_inline
@export
fn calculate_squeeze_shape[
    type: DType, indices_type: DType, single_thread_blocking_override: Bool
](
    input_shape: NDBuffer[type, 1],
    remove_indices: NDBuffer[indices_type, 1],
    output_shape: NDBuffer[type, 1],
):
    # remove_indices may not be sorted so our strategy is to use -1 to
    # represent removed dimensions in a copied version of our input shape buffer
    var num_input_dims = input_shape.dynamic_shape[0]
    var num_remove_indices = remove_indices.dynamic_shape[0]
    var final_rank = num_input_dims - num_remove_indices

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
        input_shape_copy[i] = int(input_shape[i])

    # Mark every squeezed dimension as -1 in our copy of the shape tensor
    for remove_index_index in range(num_remove_indices):
        var remove_index = int(remove_indices[remove_index_index])
        var remove_index_normalize = remove_index + num_input_dims * int(
            remove_indices[remove_index_index] < 0
        )

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


@mogg_register("squeeze_shape_shape")
@always_inline
@export
fn squeeze_shape_shape[
    type: DType, indices_type: DType, single_thread_blocking_override: Bool
](
    input_shape: NDBuffer[type, 1],
    remove_indices: NDBuffer[indices_type, 1],
) raises -> StaticIntTuple[1]:
    var out_dim = input_shape.dim(0) - remove_indices.dim(0)

    if out_dim < 0:
        raise Error(
            "[squeeze_shape] cannot remove more dimensions than there exists"
        )

    return StaticIntTuple[1](out_dim)


# ===----------------------------------------------------------------------===#
# UnsqueezeShape op
# ===----------------------------------------------------------------------===#


@mogg_register("mo.unsqueeze_shape")
@always_inline
@export
fn calculate_unsqueeze_shape[
    type: DType, indices_type: DType, single_thread_blocking_override: Bool
](
    input_shape: NDBuffer[type, 1],
    padding_indices: NDBuffer[indices_type, 1],
    output_shape: NDBuffer[type, 1],
):
    # padding_indices_buf may not be sorted so our strategy is to use -1 to
    # represent uninitialized dimensions, add the padding dimensions, and copy
    # over the remaining dimensions later.
    var num_input_dims = input_shape.dynamic_shape[0]
    var num_padding_indices = padding_indices.dynamic_shape[0]
    var final_rank = num_input_dims + num_padding_indices
    debug_assert(
        final_rank == output_shape.dynamic_shape[0],
        "Incorrect output shape.",
    )
    for output_index in range(final_rank):
        output_shape[output_index] = -1

    for padding_index_index in range(num_padding_indices):
        var padding_index = int(padding_indices[padding_index_index])
        var padding_index_normalize = padding_index + final_rank * int(
            padding_indices[padding_index_index] < 0
        )

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


@mogg_register("unsqueeze_shape_shape")
@always_inline
fn unsqueeze_shape_shape[
    type: DType, indices_type: DType, single_thread_blocking_override: Bool
](
    input_shape: NDBuffer[type, 1],
    padding_indices: NDBuffer[indices_type, 1],
) -> StaticIntTuple[1]:
    var out_dim = input_shape.dim(0) + padding_indices.dim(0)
    return StaticIntTuple[1](out_dim)


# ===----------------------------------------------------------------------===#
# Transpose op
# ===----------------------------------------------------------------------===#


@mogg_register("mo.transpose")
@mogg_view_op
@always_inline
@export
fn transpose[
    rank: Int,
    type: DType,
    int_type: DType,
    single_thread_blocking_override: Bool,
    target: StringLiteral = "cpu",
](
    input: NDBuffer[type, rank],
    perms: NDBuffer[int_type, 1],
    ctx: MojoCallContextPtr,
) -> NDBuffer[type, rank]:
    var new_shape = StaticIntTuple[rank]()
    var new_stride = StaticIntTuple[rank]()

    if _guard_against_gpu_target[target](ctx):
        return input

    @always_inline
    @parameter
    fn body[i: Int]():
        var dim = int(perms[i])
        new_shape[i] = input.dynamic_shape[dim]
        new_stride[i] = input.dynamic_stride[dim]

    unroll[body, rank]()

    # Create the transposed view.
    return NDBuffer[type, rank](input.data, new_shape, new_stride)


@mogg_register("transpose_shape")
@always_inline
fn transpose_shape[
    rank: Int,
    type: DType,
    int_type: DType,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[type, rank],
    perms: NDBuffer[int_type, 1],
) raises -> StaticIntTuple[rank]:
    if perms.dim(0) != rank:
        raise Error("[transpose] permutation size must match input rank")

    for i in range(rank):
        var perm = int(perms[i])
        if perm < 0 or rank <= perm:
            raise Error(
                "[transpose] each permutation must be within range [0, rank)"
            )

    # NOTE this assumes `transpose` can handle input with null data pointer
    var out = transpose[rank, type, int_type, single_thread_blocking_override](
        input, perms, MojoCallContextPtr()
    ).dynamic_shape

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
    input: NDBuffer[type, input_rank],
    indices: NDBuffer[DType.int32, indices_rank],
    output: NDBuffer[type, output_rank],
    ctx: MojoCallContextPtr,
):
    @always_inline
    @parameter
    fn description_fn() -> String:
        var output_shape_str = String("output_shape=") + String("x").join(
            output.get_shape()
        )
        var input_shape_str = String("input_shape=") + String("x").join(
            input.get_shape()
        )
        var indices_shape_str = String("indices_shape=") + String("x").join(
            indices.get_shape()
        )

        return String(";").join(
            output_shape_str, input_shape_str, indices_shape_str
        )

    with Trace[TraceLevel.OP](
        "mojo.gather_sum",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
    ) as t:
        gather_reduce[type, 0, 1, simdwidthof[type](), add](
            output, input, indices, 0
        )


@mogg_register("mo.gather")
@always_inline
@export
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
    target: StringLiteral = "cpu",
](
    input_shape: StaticIntTuple[in_rank],
    indices: NDBuffer[indices_type, indices_rank],
    axis_buffer: NDBuffer[axis_type, 1],
    output_shape: StaticIntTuple[output_rank],
    ctx: MojoCallContextPtr,
) raises:
    if _guard_against_gpu_target[target](ctx):
        return

    # TODO: This is disabled as if we make this a shape without a spec we have
    # nothing to deduce `indices_type` from.
    @parameter
    @always_inline
    fn load_indices[
        width: Int, _rank: Int
    ](coords: StaticIntTuple[_rank]) -> SIMD[indices_type, width]:
        return indices.load[width=width](
            rebind[StaticIntTuple[indices_rank]](coords)
        )

    # FIXME(#26008): async raising functions are temporarily disabled.
    _gather[
        type,
        indices_type,
        input_0_fn,
        load_indices,
        output_0_fn,
        target=target,
        single_thread_blocking_override=single_thread_blocking_override,
    ](
        Axis(axis_buffer[0], in_rank),
        input_shape,
        indices.dynamic_shape,
        output_shape,
    )


# ===----------------------------------------------------------------------===#
# MOGG matmul
# ===----------------------------------------------------------------------===#

from LinAlg.BatchedMatmul import (
    elementwise_epilogue_type as batched_matmul_elementwise_epilogue_type,
)

# TODO(#29765): remove import and allow Optional type to be inferred
from LinAlg.MatmulUtils import (
    elementwise_epilogue_type as matmul_elementwise_epilogue_type,
)


@mogg_register("mo.matmul")
@always_inline
@export
fn matmul[
    a_type: DType,
    input_0_static_shape: DimList,
    b_type: DType,
    input_1_static_shape: DimList,
    c_type: DType,
    input_2_static_shape: DimList,
    transpose_in_1: Bool,  # matches name of MO attribute
    packed_in_1: Bool,
    single_thread_blocking_override: Bool,
    lambdas_have_fusion: Bool,
    output_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[c_type, width]
    ) capturing -> None,
    /,
    target: StringLiteral = "cpu",
](
    a: NDBuffer[a_type, 2, input_0_static_shape],
    b: NDBuffer[b_type, 2, input_1_static_shape],
    c: NDBuffer[c_type, 2, input_2_static_shape],
    ctx: MojoCallContextPtr,
) raises:
    alias transpose_a = False
    alias transpose_b = transpose_in_1
    alias b_packed = packed_in_1

    if _guard_against_gpu_target[target](ctx):
        return

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
        var info = get_trace_information(
            "dynamic_tile",
            GemmShape.get[transpose_b](c, a, b),
            transpose_a,
            transpose_b,
            b_packed,
        )
        return (
            info
            + String(";single_thread_blocking_override=")
            + str(single_thread_blocking_override)
        )

    # TODO(#23049): Pipe info on whether using faster, saturated_vnni is ok
    with Trace[TraceLevel.OP](
        "mojo.matmul", Trace[TraceLevel.OP]._get_detail_str[description_fn]()
    ) as t:
        _matmul[
            a_type,
            input_0_static_shape,
            b_type,
            input_1_static_shape,
            c_type,
            input_2_static_shape,
            transpose_a,
            transpose_b,
            b_packed,
            OptionalReg[matmul_elementwise_epilogue_type](
                epilogue_wrapper
            ) if lambdas_have_fusion else None,
            saturated_vnni=False,
            single_thread_blocking_override=single_thread_blocking_override,
            target=target,
        ](
            c,
            a,
            b,
            ctx,
        )


# ===----------------------------------------------------------------------===#
# MOGG batched matmul
# ===----------------------------------------------------------------------===#


@mogg_register("mo.batch_matmul")
@always_inline
fn batched_matmul[
    rank: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_in_1: Bool,
    single_thread_blocking_override: Bool,
    lambdas_have_fusion: Bool,
    output_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[c_type, width]
    ) capturing -> None,
    target: StringLiteral = "cpu",
](
    a: NDBuffer[a_type, rank],
    b: NDBuffer[b_type, rank],
    c: NDBuffer[c_type, rank],
    ctx: MojoCallContextPtr,
):
    alias transpose_a = False
    alias transpose_b = transpose_in_1

    if _guard_against_gpu_target[target](ctx):
        return

    @parameter
    @always_inline
    fn epilogue_wrapper[
        _type: DType, width: Int, rank: Int
    ](coords: StaticIntTuple[rank], val: SIMD[_type, width]):
        output_0_fn[width, rank](coords, rebind[SIMD[c_type, width]](val))

    @always_inline
    @parameter
    fn description_fn() -> String:
        var info = get_trace_information_batched_matmul[rank](
            "dynamic_tile",
            a.get_shape(),
            b.get_shape(),
            c.get_shape(),
            transpose_a,
            transpose_b,
        )
        return (
            info
            + String(";single_thread_blocking_override=")
            + str(single_thread_blocking_override)
        )

    with Trace[TraceLevel.OP](
        "mojo.batched_matmul",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
    ) as t:
        _batched_matmul[
            rank,
            a_type,
            b_type,
            c_type,
            transpose_a,
            transpose_b,
            OptionalReg[batched_matmul_elementwise_epilogue_type](
                epilogue_wrapper
            ) if lambdas_have_fusion else None,
            saturated_vnni=False,
            single_thread_blocking_override=single_thread_blocking_override,
            target=target,
        ](c, a, b, context=ctx)


# ===----------------------------------------------------------------------===#
# MOGG scatter
# ===----------------------------------------------------------------------===#


@mogg_register("mo.scatter")
@always_inline
@export
fn scatter[
    rank: Int,
    input_type: DType,
    indices_type: DType,
    axis_type: DType,
](
    input: NDBuffer[input_type, rank],
    updates: NDBuffer[input_type, rank],
    indices: NDBuffer[indices_type, rank],
    axis: NDBuffer[axis_type, 1],
    output: NDBuffer[input_type, rank],
    ctx: MojoCallContextPtr,
) raises:
    @always_inline
    @parameter
    fn reduce_func[
        type: DType, width: Int
    ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
        return rhs  # always return the latest update element

    scatter_elements[reduce_func](
        input,
        indices,
        updates,
        int(normalize_neg_index(axis[0], rank)),
        output,
    )


@mogg_register("mo.scatter.add")
@always_inline
@export
fn scatter_add[
    rank: Int,
    input_type: DType,
    indices_type: DType,
    axis_type: DType,
](
    input: NDBuffer[input_type, rank],
    updates: NDBuffer[input_type, rank],
    indices: NDBuffer[indices_type, rank],
    axis: NDBuffer[axis_type, 1],
    output: NDBuffer[input_type, rank],
    ctx: MojoCallContextPtr,
) raises:
    @always_inline
    @parameter
    fn reduce_func[
        type: DType, width: Int
    ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
        return lhs + rhs

    scatter_elements[reduce_func](
        input,
        indices,
        updates,
        int(normalize_neg_index(axis[0], rank)),
        output,
    )


@mogg_register("mo.scatter.max")
@always_inline
@export
fn scatter_max[
    rank: Int,
    input_type: DType,
    indices_type: DType,
    axis_type: DType,
](
    input: NDBuffer[input_type, rank],
    updates: NDBuffer[input_type, rank],
    indices: NDBuffer[indices_type, rank],
    axis: NDBuffer[axis_type, 1],
    output: NDBuffer[input_type, rank],
    ctx: MojoCallContextPtr,
) raises:
    @always_inline
    @parameter
    fn reduce_func[
        type: DType, width: Int
    ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
        return lhs.max(rhs)

    scatter_elements[reduce_func](
        input,
        indices,
        updates,
        int(normalize_neg_index(axis[0], rank)),
        output,
    )


@mogg_register("mo.scatter.min")
@always_inline
@export
fn scatter_min[
    rank: Int,
    input_type: DType,
    indices_type: DType,
    axis_type: DType,
](
    input: NDBuffer[input_type, rank],
    updates: NDBuffer[input_type, rank],
    indices: NDBuffer[indices_type, rank],
    axis: NDBuffer[axis_type, 1],
    output: NDBuffer[input_type, rank],
    ctx: MojoCallContextPtr,
) raises:
    @always_inline
    @parameter
    fn reduce_func[
        type: DType, width: Int
    ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
        return lhs.min(rhs)

    scatter_elements[reduce_func](
        input,
        indices,
        updates,
        int(normalize_neg_index(axis[0], rank)),
        output,
    )


@mogg_register("mo.scatter.mul")
@always_inline
@export
fn scatter_mul[
    rank: Int,
    input_type: DType,
    indices_type: DType,
    axis_type: DType,
](
    input: NDBuffer[input_type, rank],
    updates: NDBuffer[input_type, rank],
    indices: NDBuffer[indices_type, rank],
    axis: NDBuffer[axis_type, 1],
    output: NDBuffer[input_type, rank],
    ctx: MojoCallContextPtr,
) raises:
    @always_inline
    @parameter
    fn reduce_func[
        type: DType, width: Int
    ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
        return lhs * rhs

    scatter_elements[reduce_func](
        input,
        indices,
        updates,
        int(normalize_neg_index(axis[0], rank)),
        output,
    )


# ===----------------------------------------------------------------------===#
# MOGG scatter_nd
# ===----------------------------------------------------------------------===#


@mogg_register("mo.scatter_nd")
@always_inline
@export
fn scatter_nd[
    output_rank: Int,
    updates_rank: Int,
    indices_rank: Int,
    output_type: DType,
    indices_type: DType,
    single_thread_blocking_override: Bool,
    target: StringLiteral = "cpu",
](
    input: NDBuffer[output_type, output_rank],
    updates: NDBuffer[output_type, updates_rank],
    indices: NDBuffer[indices_type, indices_rank],
    output: NDBuffer[output_type, output_rank],
    ctx: MojoCallContextPtr,
) raises:
    if _guard_against_gpu_target[target](ctx):
        return

    _scatter_nd[
        output_type,
        indices_type,
        output_rank,
        indices_rank,
        updates_rank,
        single_thread_blocking_override,
        target,
    ](input, indices, updates, output)


@mogg_register("mo.scatter_nd.add")
@always_inline
@export
fn scatter_nd_add[
    output_type: DType,
    indices_type: DType,
    updates_rank: Int,
    indices_rank: Int,
    output_rank: Int,
    single_thread_blocking_override: Bool,
    target: StringLiteral = "cpu",
](
    input: NDBuffer[output_type, output_rank],
    updates: NDBuffer[output_type, updates_rank],
    indices: NDBuffer[indices_type, indices_rank],
    output: NDBuffer[output_type, output_rank],
    ctx: MojoCallContextPtr,
) raises:
    if _guard_against_gpu_target[target](ctx):
        return

    @always_inline
    @parameter
    fn reduce_fn[
        type: DType, width: Int
    ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
        return lhs + rhs

    scatter_nd_generator[
        output_type,
        indices_type,
        output_rank,
        indices_rank,
        updates_rank,
        single_thread_blocking_override,
        target,
        reduce_fn=reduce_fn,
    ](input, indices, updates, output)


@mogg_register("mo.scatter_nd.max")
@always_inline
@export
fn scatter_nd_max[
    output_type: DType,
    indices_type: DType,
    updates_rank: Int,
    indices_rank: Int,
    output_rank: Int,
    single_thread_blocking_override: Bool,
    target: StringLiteral = "cpu",
](
    input: NDBuffer[output_type, output_rank],
    updates: NDBuffer[output_type, updates_rank],
    indices: NDBuffer[indices_type, indices_rank],
    output: NDBuffer[output_type, output_rank],
    ctx: MojoCallContextPtr,
) raises:
    if _guard_against_gpu_target[target](ctx):
        return

    @always_inline
    @parameter
    fn reduce_fn[
        type: DType, width: Int
    ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
        return lhs.max(rhs)

    scatter_nd_generator[
        output_type,
        indices_type,
        output_rank,
        indices_rank,
        updates_rank,
        single_thread_blocking_override,
        target,
        reduce_fn=reduce_fn,
    ](input, indices, updates, output)


@mogg_register("mo.scatter_nd.min")
@always_inline
@export
fn scatter_nd_min[
    output_type: DType,
    indices_type: DType,
    updates_rank: Int,
    indices_rank: Int,
    output_rank: Int,
    single_thread_blocking_override: Bool,
    target: StringLiteral = "cpu",
](
    input: NDBuffer[output_type, output_rank],
    updates: NDBuffer[output_type, updates_rank],
    indices: NDBuffer[indices_type, indices_rank],
    output: NDBuffer[output_type, output_rank],
    ctx: MojoCallContextPtr,
) raises:
    if _guard_against_gpu_target[target](ctx):
        return

    @always_inline
    @parameter
    fn reduce_fn[
        type: DType, width: Int
    ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
        return lhs.min(rhs)

    scatter_nd_generator[
        output_type,
        indices_type,
        output_rank,
        indices_rank,
        updates_rank,
        single_thread_blocking_override,
        target,
        reduce_fn=reduce_fn,
    ](input, indices, updates, output)


@mogg_register("mo.scatter_nd.mul")
@always_inline
@export
fn scatter_nd_mul[
    output_type: DType,
    indices_type: DType,
    updates_rank: Int,
    indices_rank: Int,
    output_rank: Int,
    single_thread_blocking_override: Bool,
    target: StringLiteral = "cpu",
](
    input: NDBuffer[output_type, output_rank],
    updates: NDBuffer[output_type, updates_rank],
    indices: NDBuffer[indices_type, indices_rank],
    output: NDBuffer[output_type, output_rank],
    ctx: MojoCallContextPtr,
) raises:
    if _guard_against_gpu_target[target](ctx):
        return

    @always_inline
    @parameter
    fn reduce_fn[
        type: DType, width: Int
    ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
        return lhs * rhs

    scatter_nd_generator[
        output_type,
        indices_type,
        output_rank,
        indices_rank,
        updates_rank,
        single_thread_blocking_override,
        target,
        reduce_fn=reduce_fn,
    ](input, indices, updates, output)


# Define a wrapper in MOGG.mojo so that softmax kernel in stdlib takes static shapes
@mogg_register("mo.softmax")
@always_inline
@export
fn softmax[
    rank: Int,
    type: DType,
    input_0_fn: fn[_simd_width: Int, _rank: Int] (
        StaticIntTuple[_rank]
    ) capturing -> SIMD[type, _simd_width],
    target: StringLiteral = "cpu",
](
    shape: StaticIntTuple[rank],
    output: NDBuffer[type, rank],
    ctx: MojoCallContextPtr,
) raises:
    if _guard_against_gpu_target[target](ctx):
        return

    _softmax[
        type,
        simdwidthof[type](),
        rank,
        DimList.create_unknown[rank](),
        input_0_fn,
        target,
    ](shape, output, rank - 1, context=ctx)


# Define a wrapper in MOGG.mojo so that softmax kernel in stdlib takes static shapes
@mogg_register("mo.logsoftmax")
@always_inline
@export
fn logsoftmax[
    rank: Int,
    type: DType,
    input_0_fn: fn[_simd_width: Int, _rank: Int] (
        StaticIntTuple[_rank]
    ) capturing -> SIMD[type, _simd_width],
](
    shape: StaticIntTuple[rank],
    output: NDBuffer[type, rank],
    ctx: MojoCallContextPtr,
) raises:
    _logsoftmax[
        type,
        simdwidthof[type](),
        rank,
        DimList.create_unknown[rank](),
        input_0_fn,
    ](shape, output, rank - 1)


# ===----------------------------------------------------------------------===#
# MOGG non_maximum_suppression
# ===----------------------------------------------------------------------===#


@mogg_register("mo.non_maximum_suppression")
@always_inline
fn non_maximum_suppression[
    type: DType
](
    boxes: NDBuffer[type, 3],
    scores: NDBuffer[type, 3],
    max_output_boxes_per_class: NDBuffer[DType.int64, 1, DimList(1)],
    iou_threshold: NDBuffer[DType.float32, 1, DimList(1)],
    score_threshold: NDBuffer[DType.float32, 1, DimList(1)],
    output: NDBuffer[DType.int64, 2],
    ctx: MojoCallContextPtr,
):
    var max_output_boxes_int = int(max_output_boxes_per_class[0])
    var iou_threshold_float = iou_threshold[0]
    var score_threshold_float = score_threshold[0]

    non_max_suppression[type](
        boxes,
        scores,
        output,
        max_output_boxes_int,
        iou_threshold_float,
        score_threshold_float,
    )


@mogg_register("non_maximum_suppression_shape")
@always_inline
fn non_maximum_suppression_shape_func[
    type: DType, single_thread_blocking_override: Bool
](
    boxes: NDBuffer[type, 3],
    scores: NDBuffer[type, 3],
    max_output_boxes_per_class: NDBuffer[DType.int64, 1, DimList(1)],
    iou_threshold: NDBuffer[DType.float32, 1, DimList(1)],
    score_threshold: NDBuffer[DType.float32, 1, DimList(1)],
) -> StaticIntTuple[2]:
    var max_output_boxes_int = int(max_output_boxes_per_class[0])
    var iou_threshold_float = iou_threshold[0]
    var score_threshold_float = score_threshold[0]

    return non_max_suppression_shape_func[type](
        boxes,
        scores,
        max_output_boxes_int,
        iou_threshold_float,
        score_threshold_float,
    )


# ===----------------------------------------------------------------------===#
# MOGG mo.random.normal
# ===----------------------------------------------------------------------===#

# TODO(31691): Correctly handle PRNG state with asynchronous runtime


@mogg_register("mo.random.normal")
@export
fn random_normal[
    type: DType,
    shapeType: DType,
    mean_var_type: DType,
    seed_type: DType,
    rank: Int,
](
    shape: NDBuffer[shapeType, 1, DimList(rank)],
    mean: NDBuffer[mean_var_type, 1, DimList(1)],
    variance: NDBuffer[mean_var_type, 1, DimList(1)],
    op_seed: NDBuffer[seed_type, 1, DimList(1)],
    output: NDBuffer[type, rank],
    ctx: MojoCallContextPtr,
):
    seed(int(op_seed[0]))
    var num_elements = 1
    for i in range(len(shape)):
        num_elements *= int(shape[i])
    randn[type](
        output.data,
        num_elements,
        mean[0].cast[DType.float64](),
        variance[0].cast[DType.float64](),
    )


@mogg_register("mo.static.random.normal")
@export
fn static_random_normal[
    type: DType,
    meanVarType: DType,
    seedType: DType,
    rank: Int,
](
    mean: NDBuffer[meanVarType, 1, DimList(1)],
    variance: NDBuffer[meanVarType, 1, DimList(1)],
    op_seed: NDBuffer[seedType, 1, DimList(1)],
    output: NDBuffer[type, rank],
    ctx: MojoCallContextPtr,
):
    seed(int(op_seed[0]))
    var num_elements = output.num_elements()
    randn(
        output.data,
        num_elements,
        mean[0].cast[DType.float64](),
        variance[0].cast[DType.float64](),
    )


@mogg_register("random_shape")
@always_inline
fn random_shape[
    shapeType: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
](shape: NDBuffer[shapeType, 1, DimList(rank)],) -> StaticIntTuple[rank]:
    var unrolledShape = StaticIntTuple[rank]()

    for i in range(rank):
        unrolledShape[i] = int(shape[i])
    return unrolledShape


# ===----------------------------------------------------------------------===#
# MOGG resize
# ===----------------------------------------------------------------------===#


@mogg_register("mo.resize.nearest")
@always_inline
@export
fn resize_nearest[
    coordinate_transform_mode: Int,
    round_mode: Int,
    rank: Int,
    inpType: DType,
    sizeType: DType,
](
    input: NDBuffer[inpType, rank],
    size: NDBuffer[sizeType, 1, DimList(rank)],
    output: NDBuffer[inpType, rank],
    ctx: MojoCallContextPtr,
):
    resize_nearest_neighbor[
        coordinate_transform_mode, round_mode, rank, inpType
    ](input, output)


@mogg_register("mo.resize.linear")
@always_inline
@export
fn resize_linear[
    coordinate_transform_mode: Int,
    antialias: Bool,
    rank: Int,
    inpType: DType,
    sizeType: DType,
](
    input: NDBuffer[inpType, rank],
    size: NDBuffer[sizeType, 1, DimList(rank)],
    output: NDBuffer[inpType, rank],
    ctx: MojoCallContextPtr,
):
    resize_linear_kernel[coordinate_transform_mode, antialias, rank, inpType](
        input, output
    )


@mogg_register("resize_shape")
@always_inline
fn resize_shape[
    rank: Int,
    inpType: DType,
    sizeType: DType,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[inpType, rank],
    size: NDBuffer[sizeType, 1, DimList(rank)],
) -> StaticIntTuple[rank]:
    var shape = StaticIntTuple[rank]()

    for i in range(rank):
        shape[i] = int(size[i])
    return shape


# ===----------------------------------------------------------------------===#
# MOGG ROI Align
# ===----------------------------------------------------------------------===#


@mogg_register("mo.roi_align")
@export
fn roi_align[
    type: DType, aligned: Bool, mode: StringLiteral
](
    input: NDBuffer[type, 4],
    rois: NDBuffer[type, 2],
    output_height: NDBuffer[DType.int64, 1],
    output_width: NDBuffer[DType.int64, 1],
    spatial_scale: NDBuffer[DType.float32, 1],
    sampling_ratio: NDBuffer[DType.float32, 1],
    output: NDBuffer[type, 4],
    ctx: MojoCallContextPtr,
):
    roi_align_nhwc[
        type,
        DimList.create_unknown[4](),
        DimList.create_unknown[2](),
        aligned,
        mode,
    ](
        output,
        input,
        rois,
        int(output_height[0]),
        int(output_width[0]),
        spatial_scale[0],
        sampling_ratio[0],
    )


@mogg_register("roi_align_shape")
@always_inline
fn roi_align_shape[
    inpTy: DType,
    roisTy: DType,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[inpTy, 4],
    rois: NDBuffer[roisTy, 2],
    output_height: NDBuffer[DType.int64, 1],
    output_width: NDBuffer[DType.int64, 1],
) -> StaticIntTuple[4]:
    var shape = StaticIntTuple[4]()

    # input shape is [N, H, W, C]
    # rois shape is [M, 5]
    # output shape is [M, output_height, output_width, C]
    shape[0] = rois.get_shape()[0]
    shape[1] = int(output_height[0])
    shape[2] = int(output_width[0])
    shape[3] = input.get_shape()[3]

    return shape


# ===----------------------------------------------------------------------===#
# MOGG split
# ===----------------------------------------------------------------------===#


@mogg_register("split_ith_output_shape")
@always_inline
fn split_ith_output_shape[
    output_idx: Int,
    rank: Int,
    input_type: DType,
    split_size_type: DType,
    axis_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[input_type, rank],
    split_sizes_buf: NDBuffer[split_size_type, 1],
    split_axis_buf: NDBuffer[axis_type, 1],
) raises -> StaticIntTuple[rank]:
    # extract relevant hyper parameters
    if output_idx < 0 or split_sizes_buf.size() <= output_idx:
        raise Error(
            "[split] output index must be within range [0, len(split_sizes))"
        )
    var output_split_size = int(split_sizes_buf[output_idx])

    var split_axis = int(split_axis_buf[0])
    if split_axis < 0:
        split_axis += rank
    if split_axis < 0 or rank <= split_axis:
        raise Error("[split] normalized axis must be within range [0, rank)")

    var split_sizes_sum = 0

    for i in range(split_sizes_buf.dim(0)):
        split_sizes_sum += int(split_sizes_buf[i])
    if split_sizes_sum != input_buf.dim(split_axis):
        raise Error(
            "[split] sum of split sizes must match input dimension at split"
            " axis"
        )

    # compute and return the output shape
    var output_shape = input_buf.get_shape()
    output_shape[split_axis] = output_split_size
    return output_shape


@mogg_register("mo.conv")
@export
fn conv[
    input_rank: Int,
    filter_rank: Int,
    strides_rank: Int,
    dilation_rank: Int,
    padding_rank: Int,
    input_type: DType,
    input_0_static_shape: DimList,
    filter_type: DType,
    input_1_static_shape: DimList,
    strides_type: DType,
    dilation_type: DType,
    padding_type: DType,
    num_groups_type: DType,
    output_type: DType,
    input_6_static_shape: DimList,
    filter_packed: Bool,
    lambdas_have_fusion: Bool,
    static_strides: DimList,
    static_dilations: DimList,
    static_padding: DimList,
    output_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[output_type, width]
    ) capturing -> None,
](
    input: NDBuffer[input_type, input_rank, input_0_static_shape],
    filter: NDBuffer[filter_type, filter_rank, input_1_static_shape],
    strides: NDBuffer[strides_type, strides_rank],
    dilation: NDBuffer[dilation_type, dilation_rank],
    paddings: NDBuffer[padding_type, padding_rank],
    num_groups: NDBuffer[num_groups_type, 1],
    # output and input have the same rank.
    output: NDBuffer[output_type, input_rank, input_6_static_shape],
    ctx: MojoCallContextPtr,
) raises:
    """Including this function in MOGG.mojo since it is intended to be a temporary
    wrapper around the Stdlib conv. Currently the strides and dilation are NDBuffers,
    but eventually they will be StaticIntTuple parameters (along with padding).
    """
    constrained[
        strides_type.is_integral() and dilation_type.is_integral(),
        "stride and dilation must have integral type",
    ]()

    if strides.size() != input_rank - 2:
        return ctx.set_to_error(
            "$(input_rank-2) values expected in conv strides"
        )

    if dilation.size() != input_rank - 2:
        return ctx.set_to_error(
            "$(input_rank-2) values expected in conv dilation"
        )

    if paddings.size() != 2 * (input_rank - 2):
        return ctx.set_to_error(
            "$(2*(input_rank-2)) value expected in conv paddings"
        )

    var stride_flat = strides.flatten()
    var dilation_flat = dilation.flatten()
    var padding_flat = paddings.flatten()

    var stride_tuple = StaticIntTuple[input_rank - 2](0)
    var dilation_tuple = StaticIntTuple[input_rank - 2](0)

    @parameter
    for i in range(input_rank - 2):
        stride_tuple[i] = int(stride_flat[i])
        dilation_tuple[i] = int(dilation_flat[i])

    if dilation_tuple != StaticIntTuple[input_rank - 2](1):
        return ctx.set_to_error("Non-unit dilation is not supported yet.")

    var pad_d_tuple = StaticIntTuple[2](0)
    var pad_h_tuple = StaticIntTuple[2](0)
    var pad_w_tuple = StaticIntTuple[2](0)

    @parameter
    if input_rank == 3:
        pad_w_tuple = Index(padding_flat[0], padding_flat[1])
    elif input_rank == 4:
        pad_h_tuple = Index(padding_flat[0], padding_flat[1])
        pad_w_tuple = Index(padding_flat[2], padding_flat[3])
    elif input_rank == 5:
        pad_d_tuple = Index(padding_flat[0], padding_flat[1])
        pad_h_tuple = Index(padding_flat[2], padding_flat[3])
        pad_w_tuple = Index(padding_flat[4], padding_flat[5])

    alias conv_attr = ConvInfoStatic[input_rank - 2](
        static_padding,
        static_strides,
        static_dilations,
        input_0_static_shape.at[input_rank - 1](),  # input C, NHWC
        input_1_static_shape.at[filter_rank - 2](),  # filter C, RSCF or FRSCf
    )

    # Specialize the function to take 4D coordiantes.
    # The bias is broadcasted to the same shape as output and
    # accessed by the 4D coordinates.
    @parameter
    @always_inline
    fn epilogue_wrapper[
        _type: DType, _rank: Int, _width: Int
    ](coords: StaticIntTuple[_rank], val: SIMD[_type, _width]):
        output_0_fn[_width, _rank](
            coords, rebind[SIMD[output_type, _width]](val)
        )

    @always_inline
    @__copy_capture(pad_h_tuple, pad_w_tuple)
    @parameter
    fn description_fn() -> String:
        var input_shape_str = String("input=") + String("x").join(
            input.dynamic_shape
        )
        var filter_shape_str = String("filter=") + String("x").join(
            filter.dynamic_shape
        )
        var output_shape_str = String("output=") + String("x").join(
            output.dynamic_shape
        )
        var group_str = String("group=") + str(int(num_groups[0]))
        var stride_str = String("stride=") + String("x").join(stride_tuple)
        var padding_h_str = String("padding_h=") + String("x").join(pad_h_tuple)
        var padding_w_str = String("padding_w=") + String("x").join(pad_w_tuple)

        return String(";").join(
            input_shape_str,
            filter_shape_str,
            output_shape_str,
            group_str,
            stride_str,
            padding_h_str,
            padding_w_str,
        )

    with Trace[TraceLevel.OP](
        "mojo.conv", Trace[TraceLevel.OP]._get_detail_str[description_fn]()
    ) as t:
        conv_nhwc_direct[
            input_rank,
            filter_rank,
            input_0_static_shape,  # input shape
            input_1_static_shape,  # filter shape
            input_6_static_shape,  # output shape
            input_type,
            filter_type,
            output_type,
            filter_packed,
            conv_attr,
            lambdas_have_fusion,
            epilogue_wrapper,
        ](
            input,
            filter,
            output,
            stride_tuple,
            dilation_tuple,
            pad_d_tuple,
            pad_h_tuple,
            pad_w_tuple,
            int(num_groups[0]),
        )


@mogg_register("mo.conv_transpose")
@always_inline
@export
fn conv_transpose[
    input_rank: Int,
    filter_rank: Int,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    strides_type: DType,
    dilation_type: DType,
    padding_type: DType,
    output_padding_type: DType,
    lambdas_have_fusion: Bool,
    filter_packed: Bool,
    output_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[output_type, width]
    ) capturing -> None,
](
    input: NDBuffer[input_type, input_rank],
    filter: NDBuffer[filter_type, filter_rank],
    strides: NDBuffer[strides_type, 1],
    dilation: NDBuffer[dilation_type, 1],
    paddings: NDBuffer[padding_type, 1],
    output_paddings: NDBuffer[output_padding_type, 1],
    output: NDBuffer[output_type, input_rank],
    ctx: MojoCallContextPtr,
):
    constrained[
        strides_type.is_integral()
        and dilation_type.is_integral()
        and output_padding_type.is_integral(),
        "stride, dilation and output_paddings must have integral type",
    ]()

    if strides.size() != input_rank - 2:
        return ctx.set_to_error(
            "$(input_rank-2) values expected in convTranspose stride"
        )

    if dilation.size() != input_rank - 2:
        return ctx.set_to_error(
            "$(input_rank-2) values expected in convTranspose dilation"
        )

    if output_paddings.size() != input_rank - 2:
        return ctx.set_to_error(
            "$(input_rank-2) values expected in convTranspose output paddings"
        )

    if paddings.size() != 2 * (input_rank - 2):
        return ctx.set_to_error(
            "$(2*(input_rank-2)) value expected in convTranspose paddings"
        )

    var stride_tuple = StaticIntTuple[input_rank - 2](0)
    var dilation_tuple = StaticIntTuple[input_rank - 2](0)

    @parameter
    for i in range(input_rank - 2):
        stride_tuple[i] = int(strides[i])
        dilation_tuple[i] = int(dilation[i])

    var pad_d = StaticIntTuple[2](0)
    var pad_h = StaticIntTuple[2](0)
    var pad_w = StaticIntTuple[2](0)

    @parameter
    if input_rank == 3:
        pad_w = Index(paddings[0], paddings[1])
    elif input_rank == 4:
        pad_h = Index(paddings[0], paddings[1])
        pad_w = Index(paddings[2], paddings[3])
    elif input_rank == 5:
        pad_d = Index(paddings[0], paddings[1])
        pad_h = Index(paddings[2], paddings[3])
        pad_w = Index(paddings[4], paddings[5])

    @always_inline
    @__copy_capture(stride_tuple, pad_h, pad_w)
    @parameter
    fn description_fn() -> String:
        var input_shape_str = String("input=") + String("x").join(
            input.dynamic_shape
        )
        var filter_shape_str = String("filter=") + String("x").join(
            filter.dynamic_shape
        )
        var output_shape_str = String("output=") + String("x").join(
            output.dynamic_shape
        )
        var group_str = String("group=") + str(1)
        var stride_str = String("stride=") + String("x").join(stride_tuple)
        var padding_d_str = String("padding_d=") + String("x").join(Index(0, 0))
        var padding_h_str = String("padding_h=") + String("x").join(pad_h)
        var padding_w_str = String("padding_w=") + String("x").join(pad_w)

        return String(";").join(
            input_shape_str,
            filter_shape_str,
            output_shape_str,
            group_str,
            stride_str,
            padding_d_str,
            padding_h_str,
            padding_w_str,
        )

    @parameter
    @always_inline
    fn epilogue_wrapper[
        _type: DType, _rank: Int, _width: Int
    ](coords: StaticIntTuple[_rank], val: SIMD[_type, _width]):
        output_0_fn[_width, _rank](
            coords, rebind[SIMD[output_type, _width]](val)
        )

    with Trace[TraceLevel.OP](
        "mojo.conv_transposed",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
    ) as t:
        try:
            conv_transpose_impl[
                input_rank,
                filter_rank,
                DimList.create_unknown[input_rank](),  # Input shape.
                DimList.create_unknown[filter_rank](),  # Filter shape.
                DimList.create_unknown[input_rank](),  # Output shape.
                input_type,
                filter_type,  # Filter type.
                output_type,  # Output type.
                filter_packed,
                lambdas_have_fusion,
                epilogue_wrapper,
            ](
                output,
                input,
                filter,
                stride_tuple,
                dilation_tuple,
                pad_d,
                pad_h,
                pad_w,
            )
        except e:
            ctx.set_to_error(e)


@mogg_register("mo.layer_norm")
@export
fn layer_norm[
    type: DType,
    rank: Int,
    input_0_fn: fn[_width: Int, _rank: Int] (
        StaticIntTuple[_rank]
    ) capturing -> SIMD[type, _width],
](
    shape: StaticIntTuple[rank],
    gamma: NDBuffer[type, 1],
    beta: NDBuffer[type, 1],
    epsilon: NDBuffer[type, 1],
    output: NDBuffer[type, rank],
    ctx: MojoCallContextPtr,
):
    _layer_norm[type, input_0_fn](shape, gamma, beta, epsilon, output)


# ===----------------------------------------------------------------------===#
# Helpers
# ===----------------------------------------------------------------------===#


# Helper function to query buffer shapes for tests.
@mogg_register("print_shape_info")
@export
fn print_buffer_info[type: DType, rank: Int](buffer: NDBuffer[type, rank]):
    print("Rank:", rank)
    print("Shape:", buffer.dynamic_shape)
    print("Strides:", buffer.dynamic_stride)


# Test helper to throw an error
@mogg_register("mo.test.return_error")
@always_inline
@export
fn return_error[
    type: DType, rank: Int
](input: NDBuffer[type, rank], ctx: MojoCallContextPtr):
    ctx.set_to_error("This is an error")


# ===----------------------------------------------------------------------===#
# TopK/BottomK
# ===----------------------------------------------------------------------===#


@mogg_register("mo.bottom_k")
@always_inline
@export
fn bottom_k[
    type: DType,
    rank: Int,
    axis_type: DType,
](
    input: NDBuffer[type, rank],
    k_buf: NDBuffer[axis_type, 1],
    axis_buf: NDBuffer[axis_type, 1],
    sorted: NDBuffer[DType.bool, 1],
    out_vals: NDBuffer[type, rank],
    out_idxs: NDBuffer[DType.int64, rank],
    ctx: MojoCallContextPtr,
):
    _top_k[rank, type](
        input,
        int(k_buf[0]),
        int(axis_buf[0]),
        False,
        rebind[NDBuffer[type, rank]](out_vals),
        out_idxs,
        sorted[0],
    )


@mogg_register("mo.top_k")
@always_inline
@export
fn top_k[
    type: DType,
    rank: Int,
    axis_type: DType,
](
    input: NDBuffer[type, rank],
    k_buf: NDBuffer[axis_type, 1],
    axis_buf: NDBuffer[axis_type, 1],
    sorted: NDBuffer[DType.bool, 1],
    out_vals: NDBuffer[type, rank],
    out_idxs: NDBuffer[DType.int64, rank],
    ctx: MojoCallContextPtr,
):
    _top_k[rank, type](
        input,
        int(k_buf[0]),
        int(axis_buf[0]),
        True,
        rebind[NDBuffer[type, rank]](out_vals),
        out_idxs,
        sorted[0],
    )


# ===----------------------------------------------------------------------===#
# GatherND
# ===----------------------------------------------------------------------===#


@mogg_register("mo.gather_nd")
@always_inline
@export
fn gather_nd[
    type: DType,
    indices_type: DType,
    data_rank: Int,
    indices_rank: Int,
    output_rank: Int,
    batch_dims: Int,
    single_thread_blocking_override: Bool,
](
    data: NDBuffer[type, data_rank],
    indices: NDBuffer[indices_type, indices_rank],
    output: NDBuffer[type, output_rank],
    ctx: MojoCallContextPtr,
):
    _gather_nd[
        type, indices_type, data_rank, indices_rank, output_rank, batch_dims
    ](data, indices, output)


# Note: this is not a "real" index_tensor op that covers all cases, but rather
# a stopgap measure for some important models (DLRM, CLIP-ViT, LLaMa2)
@mogg_register("index_tensor")
@always_inline
@export
fn index_tensor[
    type: DType,
    indices_type: DType,
    data_rank: Int,
    indices_rank: Int,
    output_rank: Int,
    batch_dims: Int,
](
    data: NDBuffer[type, data_rank],
    indices: NDBuffer[indices_type, indices_rank],
    output: NDBuffer[type, output_rank],
    ctx: MojoCallContextPtr,
):
    _index_tensor[
        type, indices_type, data_rank, indices_rank, output_rank, batch_dims
    ](data, indices, output)


# Wrappers that take `num_groups` as a parameter.
# This is required unti `mo.layout.transform` passes `num_groups` as a runtime
# value.
@mogg_register("layout_transform_QRSCF_to_FQRSCf")
@mogg_register("layout_transform_RSCF_to_FRSCf")
@always_inline
fn pack_conv_filter[
    filter_type: DType,
    rank: Int,
    num_groups: Int,
    input_1_static_shape: DimList,
](
    filter: NDBuffer[filter_type, rank],
    packed_filter: NDBuffer[filter_type, rank + 1, input_1_static_shape],
    ctx: MojoCallContextPtr,
):
    _pack_conv_filter(filter, packed_filter, num_groups)


@mogg_register("layout_transform_RSFC_to_FRSCf")
@mogg_register("layout_transform_QRSFC_to_FQRSCf")
@always_inline
fn pack_conv_transpose_filter[
    filter_type: DType,
    rank: Int,
](
    filter: NDBuffer[filter_type, rank],
    packed_filter: NDBuffer[filter_type, rank + 1],
    ctx: MojoCallContextPtr,
):
    # last param is num_groups which is currently not an available
    # arg for the MO level op
    _pack_conv_transpose_filter(filter, packed_filter, 1)


@mogg_register("pack_conv_filter_shape")
@always_inline
fn pack_conv_filter_shape[
    rank: Int,
    filter_type: DType,
    input_shape: DimList,
    filter_shape: DimList,
    output_shape: DimList,
    strides: DimList,
    dilations: DimList,
    paddings: DimList,
    num_groups: Int,
    single_thread_blocking_override: Bool,
](filter_buf: NDBuffer[filter_type, rank]) -> StaticIntTuple[rank + 1]:
    """
    Compute the output shape of convolution filter packing.

    Parameters:
        rank: Rank of the un-packed filter.
        filter_type: Type of the filter.
        input_shape: NHWC layout.
        filter_shape: Filter shape.
        output_shape: NHWC layout.
        strides: Should be rank 1 size 2.
        dilations: Should be rank 1 size 2.
        paddings: Should be rank 1 size 4.
        num_groups: The number of groups in the convolution.
        single_thread_blocking_override: If True, then reduction is run
          synchronously using a single thread.

    Args:
        filter_buf: The filter to be packed.

    Returns:
        The output shape.
    """

    return _pack_conv_filter_shape[
        filter_type,
        input_shape,
        filter_shape,
        output_shape,
        strides,
        dilations,
        paddings,
        num_groups,
        single_thread_blocking_override,
    ](filter_buf)


@mogg_register("pack_conv_transpose_filter_shape")
@always_inline
fn pack_conv_transpose_filter_shape[
    rank: Int,
    filter_type: DType,
    single_thread_blocking_override: Bool,
](filter_buf: NDBuffer[filter_type, rank]) -> StaticIntTuple[rank + 1]:
    return _pack_conv_transpose_filter_shape(filter_buf, 1)


# ===----------------------------------------------------------------------===#
# Elementwise Ops
# ===----------------------------------------------------------------------===#


@mogg_register("mo.cos")
fn wrapped_cos[
    type: DType, simd_width: Int
](arg: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return cos(arg)


@mogg_register("mo.erf")
@always_inline("nodebug")
fn wrapped_erf[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return erf(x)


@mogg_register("mo.exp")
@always_inline("nodebug")
fn wrapped_exp[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return exp(x)


@mogg_register("mo.equal")
@always_inline
fn equal[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    DType.bool, simd_width
]:
    return x == y


@mogg_register("mo.greater")
@always_inline("nodebug")
fn greater[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    DType.bool, simd_width
]:
    return x > y


@mogg_register("mo.greater_equal")
@always_inline
fn greater_equal[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    DType.bool, simd_width
]:
    return x >= y


@mogg_register("mo.not_equal")
@always_inline("nodebug")
fn not_equal[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    DType.bool, simd_width
]:
    return x != y


@mogg_register("mo.round")
@always_inline("nodebug")
fn wrapped_round[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return round(x)


@mogg_register("mo.roundeven")
@always_inline("nodebug")
fn roundeven[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return x.roundeven()


@mogg_register("mo.rsqrt")
@always_inline("nodebug")
fn wrapped_rsqrt[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return 1 / sqrt(x)


@mogg_register("mo.select")
@always_inline("nodebug")
fn select[
    type: DType, simd_width: Int
](
    cond: SIMD[DType.bool, simd_width],
    true_case: SIMD[type, simd_width],
    false_case: SIMD[type, simd_width],
) -> SIMD[type, simd_width]:
    return cond.select(true_case, false_case)


@mogg_register("mo.sin")
fn wrapped_sin[
    type: DType, simd_width: Int
](arg: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return sin(arg)


@mogg_register("mo.trunc")
@always_inline("nodebug")
fn wrapped_trunc[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return llvm_intrinsic["llvm.trunc", __type_of(x), has_side_effect=False](x)


@mogg_register("mo.log")
@always_inline("nodebug")
fn wrapped_log[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return log(x)


@mogg_register("mo.log1p")
@always_inline("nodebug")
fn wrapped_log1p[
    type: DType, simd_width: Int
](arg: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return log1p(arg)


@mogg_register("mo.is_nan")
@always_inline("nodebug")
fn wrapped_isnan[
    type: DType, simd_width: Int
](val: SIMD[type, simd_width]) -> SIMD[DType.bool, simd_width]:
    return isnan(val)


@mogg_register("mo.is_inf")
@always_inline("nodebug")
fn wrapped_isinf[
    type: DType, simd_width: Int
](val: SIMD[type, simd_width]) -> SIMD[DType.bool, simd_width]:
    return isinf(val)


@mogg_register("mo.and")
@always_inline
fn logical_and[
    simd_width: Int
](x: SIMD[DType.bool, simd_width], y: SIMD[DType.bool, simd_width]) -> SIMD[
    DType.bool, simd_width
]:
    return x & y


@mogg_register("mo.or")
@always_inline
fn logical_or[
    simd_width: Int
](x: SIMD[DType.bool, simd_width], y: SIMD[DType.bool, simd_width]) -> SIMD[
    DType.bool, simd_width
]:
    return x | y


@mogg_register("mo.not")
@always_inline
fn logical_not[
    simd_width: Int
](x: SIMD[DType.bool, simd_width]) -> SIMD[DType.bool, simd_width]:
    return ~x


@mogg_register("mo.xor")
@always_inline
fn logical_xor[
    simd_width: Int
](x: SIMD[DType.bool, simd_width], y: SIMD[DType.bool, simd_width]) -> SIMD[
    DType.bool, simd_width
]:
    return x ^ y


# ===----------------------------------------------------------------------===#
# Custom Ops
# ===----------------------------------------------------------------------===#


@mogg_register("reduce_min_and_max")
@always_inline
@export
fn reduce_min_and_max[
    type: DType,
    index_type: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
    input_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
    target: StringLiteral = "cpu",
](
    input_shape: StaticIntTuple[rank],
    axis_buffer: NDBuffer[index_type, 1],
    output_shape: StaticIntTuple[rank],
    ctx: MojoCallContextPtr,
) raises:
    """Given a tensor of shape [A, B, C, D] and reducing along dimension 'C'
    writes to a tensor of shape [A, B, 2, D] where [:, :, 0, :] contains
    the minimum reduction and [:, :, 1, :] contains the maximum reduction.
    """
    if _guard_against_gpu_target[target](ctx):
        return

    alias num_reductions = 2
    var axis = int(normalize_neg_index(axis_buffer[0], rank))

    @always_inline
    @parameter
    fn input_0_fn_wrapper[
        _type: DType, width: Int, rank: Int
    ](idx: StaticIntTuple[rank]) -> SIMD[_type, width]:
        return rebind[SIMD[_type, width]](input_0_fn[width, rank](idx))

    @always_inline
    @parameter
    fn output_0_fn_wrapper[
        _type: DType, width: Int, rank: Int
    ](
        indices: StaticIntTuple[rank],
        val: StaticTuple[SIMD[_type, width], num_reductions],
    ):
        # TODO: once we support multiple outputs, change this to route to
        # TODO: multiple output tensors.
        var indices_min = indices
        indices_min[axis] = 0
        output_0_fn[width, rank](indices_min, rebind[SIMD[type, width]](val[0]))

        var indices_max = indices
        indices_max[axis] = 1
        output_0_fn[width, rank](indices_max, rebind[SIMD[type, width]](val[1]))

    @always_inline
    @parameter
    fn reduce_fn[
        ty: DType,
        width: Int,
        reduction_idx: Int,
    ](left: SIMD[ty, width], right: SIMD[ty, width]) -> SIMD[ty, width]:
        constrained[reduction_idx < num_reductions, "reduction_idx OOB"]()

        @parameter
        if reduction_idx == 0:
            return min(left, right)
        else:
            return max(left, right)

    var init_min = Scalar[type].MAX
    var init_max = Scalar[type].MIN
    var init = StaticTuple[Scalar[type], num_reductions](init_min, init_max)

    with Trace[TraceLevel.OP]("reduce_min_and_max") as t:
        _reduce_generator[
            num_reductions,
            type,
            input_0_fn_wrapper,
            output_0_fn_wrapper,
            reduce_fn,
            single_thread_blocking_override=single_thread_blocking_override,
            target=target,
        ](
            input_shape,
            init=init,
            reduce_dim=axis,
        )


@mogg_register_shape_func("reduce_min_and_max")
@always_inline
@export
fn reduce_min_and_max_shape_func[
    type: DType,
    index_type: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
](
    data: NDBuffer[type, rank, DimList.create_unknown[rank]()],
    axis_buffer: NDBuffer[index_type, 1],
) -> StaticIntTuple[rank]:
    var new_shape = data.get_shape()
    var axis = int(normalize_neg_index(axis_buffer[0], rank))
    new_shape[axis] = 2
    return new_shape


# MHA Kernels:
@mogg_register("mo.gpu_multi_head_flash_attention")
@always_inline
@export
fn gpu_multi_head_flash_attention[
    rank: Int,
    input_0_static_shape: DimList,
    input_1_static_shape: DimList,
    input_2_static_shape: DimList,
    input_3_static_shape: DimList,
    q_type: DType,
    k_type: DType,
    v_type: DType,
    mask_type: DType,
    output_type: DType,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[q_type, rank, input_0_static_shape],
    k: NDBuffer[k_type, rank, input_1_static_shape],
    v: NDBuffer[v_type, rank, input_2_static_shape],
    mask: NDBuffer[mask_type, 3, input_3_static_shape],
    scale: NDBuffer[DType.float32, 1],
    output: NDBuffer[output_type, rank],
    ctx: MojoCallContextPtr,
) raises:
    if _guard_against_gpu_target[target](ctx):
        return

    constrained[target == "cuda", "only valid on CUDA GPUs"]()

    @always_inline
    @__copy_capture(q, k, v, mask, output)
    @parameter
    fn description_fn() -> String:
        var q_shape_str = String("q_shape=") + String("x").join(q.dynamic_shape)
        var k_shape_str = String("k_shape=") + String("x").join(k.dynamic_shape)
        var v_shape_str = String("v_shape=") + String("x").join(v.dynamic_shape)
        var mask_shape_str = String("mask_shape=") + String("x").join(
            mask.dynamic_shape
        )
        var out_shape_str = String("o_shape=") + String("x").join(
            output.dynamic_shape
        )

        return String(";").join(
            q_shape_str,
            k_shape_str,
            v_shape_str,
            mask_shape_str,
            out_shape_str,
        )

    with Trace[TraceLevel.OP](
        "mojo.flash_attention",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
    ) as t:
        gpu_flash_attention[
            rank,
            input_0_static_shape,
            input_1_static_shape,
            input_2_static_shape,
            input_3_static_shape,
            DimList.create_unknown[rank](),
            q_type,
            k_type,
            v_type,
            mask_type,
            output_type,
            True,
            target,
        ](output, q, k, v, mask, scale[0])


@mogg_register("no_mask_fused_attention_cpu")
@always_inline
@export
fn no_mask_fused_attention_cpu[
    rank: Int,
    input_0_static_shape: DimList,
    input_1_static_shape: DimList,
    input_2_static_shape: DimList,
    input_3_static_shape: DimList,
    output_type: DType,
    q_type: DType,
    k_type: DType,
    v_type: DType,
    scale_type: DType,
    single_thread_blocking_override: Bool,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[q_type, rank, input_0_static_shape],
    k: NDBuffer[k_type, rank, input_1_static_shape],
    v: NDBuffer[v_type, rank, input_2_static_shape],
    # TODO(28121): This should be rank 0, but only works with rank 1
    scale: NDBuffer[scale_type, 1, input_3_static_shape],
    output: NDBuffer[output_type, rank],
    ctx: MojoCallContextPtr,
) raises:
    # TODO:
    # - no attention mask
    # - no causaul mask

    # Dimension names:
    #     - (B)atch
    #     - Attention (H)ead
    #     - (S)equence
    #     - Embedding (D)imension
    #
    # layouts:
    # q -- BHSD
    # k -- BHDS (we assume transpose = true for now)
    # v -- BHSD
    # output: BHSD

    if _guard_against_gpu_target[target](ctx):
        return

    constrained[target == "cpu"]()

    # TODO: Unimplemented and not used
    alias mask_shape = DimList()
    alias mask_type = DType.float32
    var mask = NDBuffer[mask_type, rank, mask_shape]()
    var scale_f32 = scale[0].cast[DType.float32]()
    var causal_mask: Float32 = 0
    with Trace[TraceLevel.OP]("mojo.fused_attention") as t:
        cpu_fused_attention_impl[
            rank,
            input_0_static_shape,
            input_1_static_shape,
            input_2_static_shape,
            mask_shape,
            DimList.create_unknown[rank](),
            q_type,
            k_type,
            v_type,
            mask_type,
            output_type,
            transpose_k=False,
            add_attn_mask=False,
            add_causal_mask=False,
        ](output, q, k, v, mask, scale_f32, causal_mask)


@mogg_register("with_mask_fused_attention_cpu")
@always_inline
@export
fn with_mask_fused_attention_cpu[
    rank: Int,
    input_0_static_shape: DimList,
    input_1_static_shape: DimList,
    input_2_static_shape: DimList,
    input_3_static_shape: DimList,
    input_4_static_shape: DimList,
    output_type: DType,
    q_type: DType,
    k_type: DType,
    v_type: DType,
    attn_mask_type: DType,
    scale_type: DType,
    single_thread_blocking_override: Bool,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[q_type, rank, input_0_static_shape],
    k: NDBuffer[k_type, rank, input_1_static_shape],
    v: NDBuffer[v_type, rank, input_2_static_shape],
    attn_mask: NDBuffer[attn_mask_type, rank, input_3_static_shape],
    # TODO(28121): This should be rank 0, but only works with rank 1
    scale: NDBuffer[scale_type, 1, input_4_static_shape],
    output: NDBuffer[output_type, rank],
    ctx: MojoCallContextPtr,
) raises:
    # TODO:
    # - no attention mask
    # - no causaul mask

    # Dimension names:
    #     - (B)atch
    #     - Attention (H)ead
    #     - (S)equence
    #     - Embedding (D)imension
    #
    # layouts:
    # q -- BHSD
    # k -- BHDS (we assume transpose = true for now)
    # v -- BHSD
    # output: BHSD

    if _guard_against_gpu_target[target](ctx):
        return

    constrained[target == "cpu"]()

    # TODO: Unimplemented and not used
    var scale_f32 = scale[0].cast[DType.float32]()
    var causal_mask: Float32 = 0
    with Trace[TraceLevel.OP]("mojo.fused_attention") as t:
        cpu_fused_attention_impl[
            rank,
            input_0_static_shape,
            input_1_static_shape,
            input_2_static_shape,
            input_3_static_shape,
            DimList.create_unknown[rank](),
            q_type,
            k_type,
            v_type,
            attn_mask_type,
            output_type,
            transpose_k=False,
            add_attn_mask=True,
            add_causal_mask=False,
        ](output, q, k, v, attn_mask, scale_f32, causal_mask)


@mogg_register("no_mask_flash_attention_cpu")
@always_inline
@export
fn no_mask_flash_attention_cpu[
    type: DType,
    rank: Int,
    input_1_fn: fn[simd_width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, simd_width],
    input_2_fn: fn[simd_width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, simd_width],
    input_4_static_shape: DimList,
    single_thread_blocking_override: Bool,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[type, rank],
    input_1_shape: StaticIntTuple[rank],
    input_2_shape: StaticIntTuple[rank],
    # TODO(28121): This should be rank 0, but only works with rank 1
    scale: NDBuffer[type, 1],
    output: NDBuffer[type, rank, input_4_static_shape],
    ctx: MojoCallContextPtr,
) raises:
    if _guard_against_gpu_target[target](ctx):
        return

    constrained[target == "cpu"]()

    @always_inline
    @__copy_capture(q, input_1_shape, input_2_shape, output)
    @parameter
    fn description_fn() -> String:
        var q_shape_str = String("q_shape=") + String("x").join(q.dynamic_shape)
        var k_shape_str = String("k_shape=") + String("x").join(input_1_shape)
        var v_shape_str = String("v_shape=") + String("x").join(input_2_shape)
        var out_shape_str = String("o_shape=") + String("x").join(
            output.dynamic_shape
        )

        return String(";").join(
            q_shape_str,
            k_shape_str,
            v_shape_str,
            out_shape_str,
        )

    with Trace[TraceLevel.OP](
        "mojo.flash_attention_no_mask",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
    ) as t:

        @parameter
        @always_inline
        fn mask_fn[
            simd_width: Int, _rank: Int
        ](idx: StaticIntTuple[_rank]) -> SIMD[type, simd_width]:
            return SIMD[type, simd_width](0)

        cpu_flash_attention[
            type,
            rank,
            input_1_fn,
            input_2_fn,
            mask_fn,
            input_4_static_shape,
        ](
            q,
            input_1_shape,
            input_2_shape,
            output,
            scale[0].cast[DType.float32](),
        )


@mogg_register("with_mask_flash_attention_split_kv_cpu")
@always_inline
@export
fn with_mask_flash_attention_split_kv_cache_cpu[
    type: DType,
    rank: Int,
    input_1_fn: fn[simd_width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, simd_width],
    input_2_fn: fn[simd_width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, simd_width],
    input_3_fn: fn[simd_width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, simd_width],
    input_4_fn: fn[simd_width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, simd_width],
    input_5_fn: fn[simd_width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, simd_width],
    input_7_static_shape: DimList,
    single_thread_blocking_override: Bool,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[type, rank],
    input_1_shape: StaticIntTuple[rank],
    input_2_shape: StaticIntTuple[rank],
    input_3_shape: StaticIntTuple[rank + 1],
    input_4_shape: StaticIntTuple[rank + 1],
    input_5_shape: StaticIntTuple[rank],
    # TODO(28121): This should be rank 0, but only works with rank 1
    scale: NDBuffer[type, 1],
    output: NDBuffer[type, rank, input_7_static_shape],
    ctx: MojoCallContextPtr,
) raises:
    """A version of flash attention that takes current k and v arguments
    separately from the past KV cache arguments.
    This acts as a workaround to avoid materializing copies of the entire KV
    cache, for example in `mo.concat`.
    Instead, the KV cache is passed as input lambdas (`input_3_fn` and
    `input_4_fn`).
    The attention mask is passed in `input_5_fn`, and `input_7_static_shape` is
    the output shape.

    Arguments have the following shapes:
        q: BHSD
        input_1_fn (k): BHSD
        input_2_fn (v): BHSD
        input_3_fn (k_cache): 1BHSD
        input_4_fn (v_cache: 1BHSD
    """
    if _guard_against_gpu_target[target](ctx):
        return

    constrained[target == "cpu"]()

    @always_inline
    @__copy_capture(
        q,
        input_1_shape,
        input_2_shape,
        input_3_shape,
        input_4_shape,
        input_5_shape,
        output,
    )
    @parameter
    fn description_fn() -> String:
        var q_shape_str = String("q_shape=") + String("x").join(q.dynamic_shape)
        var k_shape_str = String("k_shape=") + String("x").join(input_1_shape)
        var v_shape_str = String("v_shape=") + String("x").join(input_2_shape)
        var k_cache_shape_str = String("k_cache_shape=") + String("x").join(
            input_3_shape
        )
        var v_cache_shape_str = String("v_cache_shape=") + String("x").join(
            input_4_shape
        )
        var mask_shape_str = String("mask_shape=") + String("x").join(
            input_5_shape
        )
        var out_shape_str = String("o_shape=") + String("x").join(
            output.dynamic_shape
        )

        return String(";").join(
            q_shape_str,
            k_shape_str,
            v_shape_str,
            k_cache_shape_str,
            v_cache_shape_str,
            mask_shape_str,
            out_shape_str,
        )

    with Trace[TraceLevel.OP](
        "mojo.flash_attention_split_kv",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
    ):
        cpu_flash_attention_split_kv[
            type,
            rank,
            input_1_fn,
            input_2_fn,
            input_3_fn,
            input_4_fn,
            input_5_fn,
            input_7_static_shape,
        ](
            q,
            input_1_shape,
            input_2_shape,
            input_3_shape,
            input_4_shape,
            output,
            scale[0].cast[DType.float32](),
        )


@mogg_register_shape_func("with_mask_flash_attention_split_kv_cpu")
@always_inline
@export
fn with_mask_flash_attention_split_kv_cpu_shape_func[
    type: DType, rank: Int, single_thread_blocking_override: Bool
](q: NDBuffer[type, rank]) -> StaticIntTuple[rank]:
    return q.get_shape()
    # TODO(#37702): The following shape function incurs a KV cache copy:
    # Take the `softmax(Q @ K^T) @ V` product's channel dimension from `V`.
    # var new_shape = q.get_shape()
    # new_shape[rank - 1] = v.get_shape()[rank]

    # return new_shape


@mogg_register("with_mask_flash_attention_cpu")
@always_inline
@export
fn with_mask_flash_attention_cpu[
    type: DType,
    rank: Int,
    input_1_fn: fn[simd_width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, simd_width],
    input_2_fn: fn[simd_width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, simd_width],
    input_3_fn: fn[simd_width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, simd_width],
    input_5_static_shape: DimList,
    single_thread_blocking_override: Bool,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[type, rank],
    input_1_shape: StaticIntTuple[rank],
    input_2_shape: StaticIntTuple[rank],
    input_3_shape: StaticIntTuple[rank],
    # TODO(28121): This should be rank 0, but only works with rank 1
    scale: NDBuffer[type, 1],
    output: NDBuffer[type, rank, input_5_static_shape],
    ctx: MojoCallContextPtr,
) raises:
    if _guard_against_gpu_target[target](ctx):
        return

    constrained[target == "cpu"]()

    @always_inline
    @__copy_capture(q, input_1_shape, input_2_shape, input_3_shape, output)
    @parameter
    fn description_fn() -> String:
        var q_shape_str = String("q_shape=") + String("x").join(q.dynamic_shape)
        var k_shape_str = String("k_shape=") + String("x").join(input_1_shape)
        var v_shape_str = String("v_shape=") + String("x").join(input_2_shape)
        var mask_shape_str = String("mask_shape=") + String("x").join(
            input_3_shape
        )
        var out_shape_str = String("o_shape=") + String("x").join(
            output.dynamic_shape
        )

        return String(";").join(
            q_shape_str,
            k_shape_str,
            v_shape_str,
            mask_shape_str,
            out_shape_str,
        )

    with Trace[TraceLevel.OP](
        "mojo.flash_attention",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
    ) as t:
        cpu_flash_attention[
            type,
            rank,
            input_1_fn,
            input_2_fn,
            input_3_fn,
            input_5_static_shape,
        ](
            q,
            input_1_shape,
            input_2_shape,
            output,
            scale[0].cast[DType.float32](),
        )


# # ===----------------------------------------------------------------------===#
# # INT4/5/6 packing format and op implementations
# # ===----------------------------------------------------------------------===#

from quantization import (
    Q4sym,
    block_Q4_K,
    block_Q6_K,
    block_QK_K,
    ggml_q4_0_matmul_impl,
    q4_k_dequantize_impl,
    q6_k_dequantize_impl,
)
from quantization.qmatmul import matmul_qint4, matmul_qint4_pack_b
from quantization.qmatmul_k import (
    matmul_Q4_K,
    matmul_Q4_K_pack_b,
    matmul_Q6_K,
    matmul_Q6_K_pack_b,
)


# TODO: closures to make this simpler
######
# Q4 -- group size 8
######
@mogg_register("quantize_Q4symG8")
@always_inline
@export
fn quantize_Q4sym_g8[
    input_type: DType, rank: Int, single_thread_blocking_override: Bool
](
    input: NDBuffer[input_type, rank],
    output: NDBuffer[DType.uint8, rank],
    ctx: MojoCallContextPtr,
):
    Q4sym[8, input_type].quantize_and_write_to_tensor(
        input, output, input.dynamic_shape
    )


@mogg_register_shape_func("quantize_Q4symG8")
@always_inline
@export
fn quantize_Q4sym_g8_shape_func[
    type: DType, rank: Int, single_thread_blocking_override: Bool
](data: NDBuffer[type, rank],) -> StaticIntTuple[rank]:
    var new_shape = data.get_shape()
    new_shape[rank - 1] = (
        ceildiv(new_shape[rank - 1], 8) * sizeof[Q4sym[8, type]]()
    )
    return new_shape


@mogg_register("dequantize_Q4symG8")
@always_inline
@export
fn dequantize_Q4sym_g8[
    output_type: DType, rank: Int, single_thread_blocking_override: Bool
](
    input: NDBuffer[DType.uint8, rank],
    output: NDBuffer[output_type, rank],
    ctx: MojoCallContextPtr,
):
    Q4sym[8, output_type].dequantize_and_write_to_tensor(
        input, output, output.dynamic_shape
    )


######
# Q4 -- group size 16
######
@mogg_register("quantize_Q4symG16")
@always_inline
@export
fn quantize_Q4sym_g16[
    input_type: DType, rank: Int, single_thread_blocking_override: Bool
](
    input: NDBuffer[input_type, rank],
    output: NDBuffer[DType.uint8, rank],
    ctx: MojoCallContextPtr,
):
    Q4sym[16, input_type].quantize_and_write_to_tensor(
        input, output, input.dynamic_shape
    )


@mogg_register_shape_func("quantize_Q4symG16")
@always_inline
@export
fn quantize_Q4sym_g16_shape_func[
    type: DType, rank: Int, single_thread_blocking_override: Bool
](data: NDBuffer[type, rank],) -> StaticIntTuple[rank]:
    var new_shape = data.get_shape()
    new_shape[rank - 1] = (
        ceildiv(new_shape[rank - 1], 16) * sizeof[Q4sym[16, type]]()
    )
    return new_shape


@mogg_register("dequantize_Q4symG16")
@always_inline
@export
fn dequantize_Q4sym_g16[
    output_type: DType, rank: Int, single_thread_blocking_override: Bool
](
    input: NDBuffer[DType.uint8, rank],
    output: NDBuffer[output_type, rank],
    ctx: MojoCallContextPtr,
):
    Q4sym[16, output_type].dequantize_and_write_to_tensor(
        input, output, output.dynamic_shape
    )


######
# Q4 -- group size 32
######
@mogg_register("quantize_Q4symG32")
@always_inline
@export
fn quantize_Q4sym_g32[
    input_type: DType, rank: Int, single_thread_blocking_override: Bool
](
    input: NDBuffer[input_type, rank],
    output: NDBuffer[DType.uint8, rank],
    ctx: MojoCallContextPtr,
):
    Q4sym[32, input_type].quantize_and_write_to_tensor(
        input, output, input.dynamic_shape
    )


@mogg_register_shape_func("quantize_Q4symG32")
@always_inline
@export
fn quantize_Q4sym_g32_shape_func[
    type: DType, rank: Int, single_thread_blocking_override: Bool
](data: NDBuffer[type, rank],) -> StaticIntTuple[rank]:
    var new_shape = data.get_shape()
    new_shape[rank - 1] = (
        ceildiv(new_shape[rank - 1], 32) * sizeof[Q4sym[32, type]]()
    )
    return new_shape


@mogg_register("dequantize_Q4symG32")
@always_inline
@export
fn dequantize_Q4sym_g32[
    output_type: DType, rank: Int, single_thread_blocking_override: Bool
](
    input: NDBuffer[DType.uint8, rank],
    output: NDBuffer[output_type, rank],
    ctx: MojoCallContextPtr,
):
    Q4sym[32, output_type].dequantize_and_write_to_tensor(
        input, output, output.dynamic_shape
    )


@mogg_register("qmatmul_Af32_BTQ4symG32_Cf32")
@always_inline
@export
fn qmatmul_Af32_BTQ4symG32_Cf32[
    type: DType
](
    a: NDBuffer[type, 2],
    b: NDBuffer[DType.uint8, 2],
    c: NDBuffer[type, 2],
    ctx: MojoCallContextPtr,
) raises:
    ggml_q4_0_matmul_impl[32](a, b, c)


@mogg_register_shape_func("qmatmul_Af32_BTQ4symG32_Cf32")
@always_inline
@export
fn qmatmul_Af32_BTQ4symG32_Cf32_shape_func[
    type: DType,
    single_thread_blocking_override: Bool,
](a: NDBuffer[type, 2], b: NDBuffer[DType.uint8, 2],) -> StaticIntTuple[2]:
    return StaticIntTuple[2](a.dim[0](), b.dim[0]())


@mogg_register("mo.linalg.solve")
@always_inline
fn mogg_matrix_solve[
    type: DType,
    x_rank: Int,
    a_rank: Int,
    b_rank: Int,
    single_thread_blocking_override: Bool,
](
    a: NDBuffer[type, a_rank],
    b: NDBuffer[type, b_rank],
    x: NDBuffer[type, x_rank],
    ctx: MojoCallContextPtr,
) raises:
    matrix_solve[type, x_rank, a_rank, b_rank, single_thread_blocking_override](
        a, b, x
    )


# NOTE we don't inline this because `SymbolicizeFallbackShapeFunctions` pass
# needs to pattern match for this call to figure out where mojo raises happen
# inside the shape functions.
@mogg_register("set_ctx_error_and_destruct_error")
@no_inline
@export
fn set_ctx_error_and_destruct_error(
    ctx: MojoCallContextPtr, owned error: Error
):
    ctx.set_to_error(error)
    # mojo lowering will insert destructor call for `error`


@mogg_register("pytorch_operator_custom_test")
@export
fn pytorch_test_custom[
    type: DType,
    rank: Int,
](data: NDBuffer[type, rank], out: NDBuffer[type, rank]):
    print("hello")


######
# Q4_0
######


@mogg_register_override("vroom_q4_0_matmul", 1)
@always_inline
@export
fn vroom_q4_0_matmul(
    a: NDBuffer[DType.float32, 2],
    b: NDBuffer[DType.uint8, 2],
    c: NDBuffer[DType.float32, 2],
    ctx: MojoCallContextPtr,
) raises:
    constrained[
        (a.type == c.type) and a.type.is_floating_point(),
        "expected float inputs and outputs",
    ]()
    constrained[b.type == DType.uint8, "expected uint8 input b"]()

    with Trace[TraceLevel.OP]("mojo.vroom_q4_0_matmul"):
        matmul_qint4[32](a, b, c)


@mogg_register_shape_func("vroom_q4_0_matmul")
@always_inline
@export
fn vroom_q4_0_matmul_shape_func[
    single_thread_blocking_override: Bool
](a: NDBuffer[DType.float32, 2], b: NDBuffer[DType.uint8, 2]) -> StaticIntTuple[
    2
]:
    constrained[
        a.type.is_floating_point(), "expected float inputs and outputs"
    ]()
    constrained[b.type == DType.uint8, "expected uint8 input b"]()
    constrained[a.rank == b.rank == 2, "expected rank to be 2"]()

    return StaticIntTuple[2](a.dim[0](), b.dim[0]())


@mogg_register_override("vroom_q4_0_repack_weights", 1)
@always_inline
@export
fn vroom_q4_0_repack_weights(
    b: NDBuffer[DType.uint8, 2],
    b_packed: NDBuffer[DType.uint8, 2],
    ctx: MojoCallContextPtr,
) raises:
    matmul_qint4_pack_b[32](b, b_packed)


@mogg_register_shape_func("vroom_q4_0_repack_weights")
@always_inline
@export
fn vroom_q4_0_repack_weights_shape_func[
    single_thread_blocking_override: Bool
](b: NDBuffer[DType.uint8, 2]) -> StaticIntTuple[2]:
    return b.get_shape()


@mogg_register_override("ggml_q4_0_matmul", 1)
@always_inline
@export
fn ggml_q4_0_matmul[
    float_dtype: DType
](
    a: NDBuffer[float_dtype, 2],
    b: NDBuffer[DType.uint8, 2],
    c: NDBuffer[float_dtype, 2],
    ctx: MojoCallContextPtr,
) raises:
    constrained[
        (a.type == c.type) and a.type.is_floating_point(),
        "expected float inputs and outputs",
    ]()
    constrained[b.type == DType.uint8, "expected uint8 input b"]()

    with Trace[TraceLevel.OP]("mojo.ggml_q4_0_matmul"):
        ggml_q4_0_matmul_impl[group_size=32, type=float_dtype](a, b, c)


@mogg_register_shape_func("ggml_q4_0_matmul")
@always_inline
@export
fn ggml_q4_0_matmul_shape_func[
    float_dtype: DType, single_thread_blocking_override: Bool
](a: NDBuffer[float_dtype, 2], b: NDBuffer[DType.uint8, 2]) -> StaticIntTuple[
    2
]:
    constrained[
        a.type.is_floating_point(), "expected float inputs and outputs"
    ]()
    constrained[b.type == DType.uint8, "expected uint8 input b"]()
    constrained[a.rank == b.rank == 2, "expected rank to be 2"]()

    return StaticIntTuple[2](a.dim[0](), b.dim[0]())


@mogg_register_override("ggml_q4_0_dequantize", 1)
@always_inline
@export
fn ggml_q4_0_dequantize(
    input: NDBuffer[DType.uint8, 2],
    output: NDBuffer[DType.float32, 2],
    ctx: MojoCallContextPtr,
) raises:
    with Trace[TraceLevel.OP]("mojo.ggml_q4_0_dequantize"):
        Q4sym[group_size=32].dequantize_and_write_to_tensor[rank=2](
            input, output, output.dynamic_shape
        )


@mogg_register_shape_func("ggml_q4_0_dequantize")
@always_inline
@export
fn ggml_q4_0_dequantize_shape_func[
    single_thread_blocking_override: Bool
](input: NDBuffer[DType.uint8, 2]) -> StaticIntTuple[2]:
    constrained[input.type == DType.uint8, "expected uint8 input"]()

    alias block_nbytes = sizeof[Q4sym[group_size=32]]()
    alias quants_per_block = 32

    var num_block_per_batch = (
        input.size() // input.dynamic_shape[0]
    ) // block_nbytes

    return (input.dynamic_shape[0], quants_per_block * num_block_per_batch)


######
# Q4_K
######


@mogg_register_override("vroom_q4_k_matmul", 1)
@always_inline
@export
fn vroom_q4_k_matmul(
    a: NDBuffer[DType.float32, 2],
    b: NDBuffer[DType.uint8, 2],
    c: NDBuffer[DType.float32, 2],
    ctx: MojoCallContextPtr,
) raises:
    with Trace[TraceLevel.OP]("mojo.vroom_q4_k_matmul"):
        matmul_Q4_K(a, b, c)


@mogg_register_shape_func("vroom_q4_k_matmul")
@always_inline
@export
fn vroom_q4_k_matmul_shape_func[
    single_thread_blocking_override: Bool
](a: NDBuffer[DType.float32, 2], b: NDBuffer[DType.uint8, 2]) -> StaticIntTuple[
    2
]:
    return StaticIntTuple[2](a.dim[0](), b.dim[0]())


@mogg_register_override("vroom_q4_k_repack_weights", 1)
@always_inline
@export
fn vroom_q4_k_repack_weights(
    b: NDBuffer[DType.uint8, 2],
    b_packed: NDBuffer[DType.uint8, 2],
    ctx: MojoCallContextPtr,
) raises:
    matmul_Q4_K_pack_b(b, b_packed)


@mogg_register_shape_func("vroom_q4_k_repack_weights")
@always_inline
@export
fn vroom_q4_k_repack_weights_shape_func[
    single_thread_blocking_override: Bool
](b: NDBuffer[DType.uint8, 2]) -> StaticIntTuple[2]:
    return b.get_shape()


@mogg_register_override("ggml_q4_k_dequantize", 1)
@always_inline
@export
fn ggml_q4_k_dequantize(
    input: NDBuffer[DType.uint8, 2],
    output: NDBuffer[DType.float32, 2],
    ctx: MojoCallContextPtr,
) raises:
    with Trace[TraceLevel.OP]("mojo.ggml_q4_k_dequantize"):
        q4_k_dequantize_impl(input, output)


@mogg_register_shape_func("ggml_q4_k_dequantize")
@always_inline
@export
fn ggml_q4_k_dequantize_shape_func[
    single_thread_blocking_override: Bool
](input: NDBuffer[DType.uint8, 2]) -> StaticIntTuple[2]:
    alias block_nbytes = sizeof[block_Q4_K]()
    alias elements_per_block = block_QK_K.quantized_k

    var num_block_per_batch = (
        input.size() // input.dynamic_shape[0]
    ) // block_nbytes

    return (input.dynamic_shape[0], elements_per_block * num_block_per_batch)


######
# Q6_K
######


@mogg_register_override("vroom_q6_k_matmul", 1)
@always_inline
@export
fn vroom_q6_k_matmul(
    a: NDBuffer[DType.float32, 2],
    b: NDBuffer[DType.uint8, 2],
    c: NDBuffer[DType.float32, 2],
    ctx: MojoCallContextPtr,
) raises:
    with Trace[TraceLevel.OP]("mojo.vroom_q6_k_matmul"):
        matmul_Q6_K(a, b, c)


@mogg_register_shape_func("vroom_q6_k_matmul")
@always_inline
@export
fn vroom_q6_k_matmul_shape_func[
    single_thread_blocking_override: Bool
](a: NDBuffer[DType.float32, 2], b: NDBuffer[DType.uint8, 2]) -> StaticIntTuple[
    2
]:
    return StaticIntTuple[2](a.dim[0](), b.dim[0]())


@mogg_register_override("vroom_q6_k_repack_weights", 1)
@always_inline
@export
fn vroom_q6_k_repack_weights(
    b: NDBuffer[DType.uint8, 2],
    b_packed: NDBuffer[DType.uint8, 2],
    ctx: MojoCallContextPtr,
) raises:
    matmul_Q6_K_pack_b(b, b_packed)


@mogg_register_shape_func("vroom_q6_k_repack_weights")
@always_inline
@export
fn vroom_q6_k_repack_weights_shape_func[
    single_thread_blocking_override: Bool
](b: NDBuffer[DType.uint8, 2]) -> StaticIntTuple[2]:
    return b.get_shape()


@mogg_register_override("ggml_q6_k_dequantize", 1)
@always_inline
@export
fn ggml_q6_k_dequantize(
    input: NDBuffer[DType.uint8, 2],
    output: NDBuffer[DType.float32, 2],
    ctx: MojoCallContextPtr,
) raises:
    with Trace[TraceLevel.OP]("mojo.ggml_q6_k_dequantize"):
        q6_k_dequantize_impl(input, output, output.dynamic_shape)


@mogg_register_shape_func("ggml_q6_k_dequantize")
@always_inline
@export
fn ggml_q6_k_dequantize_shape_func[
    single_thread_blocking_override: Bool
](input: NDBuffer[DType.uint8, 2]) -> StaticIntTuple[2]:
    alias block_nbytes = sizeof[block_Q6_K]()
    alias elements_per_block = block_QK_K.quantized_k

    var num_block_per_batch = (
        input.size() // input.dynamic_shape[0]
    ) // block_nbytes

    return (input.dynamic_shape[0], elements_per_block * num_block_per_batch)
