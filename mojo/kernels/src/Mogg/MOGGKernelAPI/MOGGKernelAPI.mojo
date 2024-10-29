# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import OptionalReg
from math import (
    ceil,
    cos,
    erf,
    exp,
    floor,
    fma,
    iota,
    isqrt,
    log,
    log1p,
    sin,
    sqrt,
    tanh,
)
from nn.concat import concat, _concat_cpu
from random import randn, seed
from sys import llvm_intrinsic
from sys.info import simdwidthof
from nn.split import split
import compiler_internal as compiler
from collections.vector import InlinedFixedVector
from sys.info import bitwidthof, simdwidthof, sizeof

# ===----------------------------------------------------------------------===#
# Kernel imports
# ===----------------------------------------------------------------------===#
from algorithm import max as reduce_max
from algorithm import mean
from algorithm import min as reduce_min
from algorithm import product, sum
from algorithm.reduction import _reduce_generator, _reduce_generator_cpu
from utils import StaticTuple

# ===----------------------------------------------------------------------===#
# General imports
# ===----------------------------------------------------------------------===#
from buffer import NDBuffer
from buffer.dimlist import DimList, Dim
from builtin.simd import _pow
from linalg.bmm import batched_matmul, batched_matmul_shape
from linalg.bmm import (
    elementwise_epilogue_type as batched_matmul_elementwise_epilogue_type,
)
from linalg.matmul import matmul
from linalg.matrix_band_part import matrix_band_part
from linalg.matrix_solve import matrix_solve, matrix_solve_shape
from linalg.utils import (
    elementwise_epilogue_type as matmul_elementwise_epilogue_type,
)
from nn import arg_nonzero
from nn.argmaxmin import argmax, argmin
from nn.activations import gelu, relu
from nn.arange import arange, arange_shape
from nn.conv import ConvInfoStatic, conv_nhwc_direct, conv_shape
from nn.conv_transpose import conv_transpose_shape, conv_transposed
from nn.cumsum import cumsum
from nn.flash_attention import flash_attention as nn_flash_attention
from nn.flash_attention import flash_attention_split_kv
from nn.gather_scatter import (
    Axis,
    gather,
    gather_nd,
    gather_reduce,
    gather_shape,
    normalize_neg_index,
    scatter_elements,
    scatter_elements_shape,
    scatter_nd,
    scatter_nd_generator,
)
from nn.mha import flash_attention, fused_attention
from nn.nms import non_max_suppression, non_max_suppression_shape_func
from nn.normalization import layer_norm, rms_norm
from nn.pad import pad_constant, pad_reflect, pad_repeat, pad_shape
from nn.pool import avg_pool, max_pool, pool_shape, pool_shape_ceil
from nn.reshape import reshape, reshape_shape
from nn.resize import resize_linear, resize_nearest_neighbor
from nn.roi_align import roi_align_nhwc
from nn.slice import slice_as_view, slice_dim_as_view, slice_shape
from nn.softmax import logsoftmax, softmax
from nn.tile import tile, tile_shape
from nn.topk import top_k, top_k_shape_impl
from runtime.asyncrt import MojoCallContextPtr
from runtime.tracing import Trace, TraceLevel, trace_arg
from tensor_utils_internal import ManagedTensorSlice, foreach
from memory import UnsafePointer
from utils import IndexList, StaticTuple
from utils.index import Index
from utils.numerics import isinf, isnan
from compiler_internal import StaticTensorSpec

from register import mogg_register_override
from nn.conv import pack_filter as _pack_conv_filter
from nn.conv_transpose import pack_filter as _pack_conv_transpose_filter

from quantization import (
    Q4sym,
    block_Q4_K,
    block_Q6_K,
    block_QK_K,
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

# ===----------------------------------------------------------------------===#
# Nop functions to expose different types to the compiler.
# ===----------------------------------------------------------------------===#


@mogg_register_override("bfloat16", 1)
fn DTypeBFloat16TypeDef(ty: DType.type) -> DType.type:
    return DType.bfloat16.value


@mogg_register_override("float16", 1)
fn DTypeFloat16TypeDef(ty: DType.type) -> DType.type:
    return DType.float16.value


@mogg_register_override("float32", 1)
fn DTypeFloat32TypeDef(ty: DType.type) -> DType.type:
    return DType.float32.value


@mogg_register_override("float64", 1)
fn DTypeFloat64TypeDef(ty: DType.type) -> DType.type:
    return DType.float64.value


@mogg_register_override("int8", 1)
fn DTypeInt8TypeDef(ty: DType.type) -> DType.type:
    return DType.int8.value


@mogg_register_override("int16", 1)
fn DTypeInt16TypeDef(ty: DType.type) -> DType.type:
    return DType.int16.value


@mogg_register_override("int32", 1)
fn DTypeInt32TypeDef(ty: DType.type) -> DType.type:
    return DType.int32.value


@mogg_register_override("uint32", 1)
fn DTypeUInt32TypeDef(ty: DType.type) -> DType.type:
    return DType.uint32.value


@mogg_register_override("uint64", 1)
fn DTypeUInt64TypeDef(ty: DType.type) -> DType.type:
    return DType.uint64.value


@mogg_register_override("int64", 1)
fn DTypeInt64TypeDef(ty: DType.type) -> DType.type:
    return DType.int64.value


@mogg_register_override("uint8", 1)
fn DTypeUInt8TypeDef(ty: DType.type) -> DType.type:
    return DType.uint8.value


@mogg_register_override("uint16", 1)
fn DTypeUInt16TypeDef(ty: DType.type) -> DType.type:
    return DType.uint16.value


@mogg_register_override("bool", 1)
fn DTypeBoolTypeDef(ty: DType.type) -> DType.type:
    return DType.bool.value


@mogg_register_override("index", 1)
fn IndexTypeDef(ty: Int) -> Int:
    return ty


@mogg_register_override("mojoCallContext", 1)
fn MojoCallContextDef(ty: MojoCallContextPtr):
    pass


@mogg_register_override("simd", 1)
fn SimdTypeDef[
    type: DType, width: Int
](ty: SIMD[type, width]) -> SIMD[type, width]:
    return ty


@mogg_register_override("indices", 1)
fn TensorIndicesTypeDef[rank: Int](ty: IndexList[rank]) -> IndexList[rank]:
    return ty


@mogg_register_override("dim_type", 1)
fn DimTypeDef(ty: Dim) -> Dim:
    return ty


# ===----------------------------------------------------------------------===#
# Hooks to help build static shapes.
# ===----------------------------------------------------------------------===#


@mogg_register_override("create_unknown_dim", 1)
fn create_unknown_dim() -> Dim:
    return Dim()


@mogg_register_override("create_known_dim", 1)
fn create_known_dim[known_val: Int]() -> Dim:
    return Dim(known_val)


# ===----------------------------------------------------------------------===#
# Additional expected primitives
# ===----------------------------------------------------------------------===#


@mogg_register_override("get_address_space", 1)
fn get_address_space() -> AddressSpace:
    return AddressSpace.GENERIC


# Build the StaticTensorSpec parameter for the DPS kernels
@mogg_register_override("build_static_tensor_specs", 1)
@export
fn build_static_tensor_specs[
    type: DType, rank: Int
](shape: DimList, strides: DimList) -> StaticTensorSpec[type, rank]:
    return StaticTensorSpec(
        shape,
        strides,
        OptionalReg[StaticTensorSpec[type, rank].in_lambda_t](None),
        OptionalReg[StaticTensorSpec[type, rank].out_lambda_t](None),
    )


# Used by the graph compiler to construct tensors from MGP repr. of tensor
@mogg_register_override("to_managed_tensor_slice", 1)
@export
@always_inline
fn to_managed_tensor_slice[
    type: DType, rank: Int
](
    data: UnsafePointer[Scalar[type]],
    shape: UnsafePointer[Int],
) -> ManagedTensorSlice[type, rank]:
    var shape_ptr = shape
    var shape_tuple = IndexList[rank]()

    var stride_tuple = IndexList[rank]()
    var stride: Int = 1

    @parameter
    for i in reversed(range(rank)):
        # Start from the back so we can accumulate the strides.
        shape_tuple[i] = shape_ptr[i]
        stride_tuple[i] = stride
        stride *= shape_tuple[i]

    return ManagedTensorSlice[type, rank](data, shape_tuple, stride_tuple)


# Extract a value from a shape.
@mogg_register_override("get_scalar_from_ndbuffer", 1)
@always_inline
fn get_scalar_from_ndbuffer[
    dtype: DType
](tensor: NDBuffer[dtype, 1]) -> Scalar[dtype]:
    # Assumes that tensor is on the host!
    return tensor[0]


# Wrappers that take `num_groups` as a parameter.
# This is required unti `mo.layout.transform` passes `num_groups` as a runtime
# value.
@mogg_register_override("layout_transform_QRSCF_to_FQRSCf", 1)
@mogg_register_override("layout_transform_RSCF_to_FRSCf", 1)
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


@mogg_register_override("layout_transform_RSFC_to_FRSCf", 1)
@mogg_register_override("layout_transform_QRSFC_to_FQRSCf", 1)
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


@mogg_register_override("get_int_from_shape", 1)
@always_inline
fn get_int_from_shape[
    param_index: Int, rank: Int
](shape: IndexList[rank]) -> Int:
    return shape[param_index]


# ===----------------------------------------------------------------------===#
# Helpers
# ===----------------------------------------------------------------------===#

# TODO(GRA-914): Properly support scalars.
alias ScalarTensor = ManagedTensorSlice[rank=1]


# Used by the graph compiler -- which right now does not support static shape
@mogg_register_override("managed_tensor_slice_to_ndbuffer", 1)
@always_inline
fn managed_tensor_slice_to_ndbuffer_primitive[
    type: DType, rank: Int
](tensor: ManagedTensorSlice[type, rank]) -> NDBuffer[type, rank]:
    return managed_tensor_slice_to_ndbuffer(tensor)


@always_inline
fn managed_tensor_slice_to_ndbuffer[
    type: DType,
    rank: Int,
    static_shape: DimList = DimList.create_unknown[rank](),
](tensor: ManagedTensorSlice[type, rank]) -> NDBuffer[type, rank, static_shape]:
    return NDBuffer[type, rank, static_shape](
        tensor._ptr, tensor.get_static_spec().shape, tensor._strides
    )


@always_inline("nodebug")
fn reduce_shape[
    input_rank: Int,
    input_type: DType,
](
    input_buf: ManagedTensorSlice[input_type, input_rank],
    axis0: ScalarTensor,
) raises -> IndexList[input_rank]:
    """
    Compute the output shape of a `reduce` operation, and assert the inputs are
    compatible.

    Parameters:
        input_rank: Input_rank of the input tensor.
        input_type: Type of the input tensor.

    Args:
        input_buf: The input tensor.
        axis0: The axis tensor.

    Returns:
        The output shape.
    """

    var axis_scalar = axis0._ptr.load(0)
    var axis = int(normalize_neg_index(axis_scalar, input_rank))

    if axis < 0 or input_rank <= axis:
        raise Error(
            "[reduction] normalized axis must be within range [0, input_rank)"
        )

    # compute and return the output shape
    var output_shape = input_buf.get_static_spec().shape
    output_shape[axis] = 1
    return output_shape


# ===----------------------------------------------------------------------===#
# Data structures used in MOGG/MGP ABI
# ===----------------------------------------------------------------------===#


# NOTE the layout must match `CompiledKernelABI::Tensor`
struct ABI_Tensor:
    var dims: UnsafePointer[Int]
    var data: UnsafePointer[NoneType]


# NOTE the layout must match `CompiledKernelABI::List`
struct ABI_List:
    var num_elems: Int
    var elements: UnsafePointer[NoneType]


# ===----------------------------------------------------------------------===#
# Elementwise Kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.range")
@compiler.elementwise
struct Range:
    @staticmethod
    fn execute[
        type: DType,
        synchronous: Bool,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice[type=type, rank=1],
        start: ScalarTensor[type=type],
        stop: ScalarTensor[type=type],
        step: ScalarTensor[type=type],
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[1]) -> SIMD[type, width]:
            return start[0] + step[0] * (iota[type, width](idx[0]))

        foreach[func, synchronous, target](output, ctx)

    @staticmethod
    fn shape[
        type: DType
    ](
        start: ScalarTensor[type=type],
        stop: ScalarTensor[type=type],
        step: ScalarTensor[type=type],
    ) raises -> IndexList[1]:
        return arange_shape[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(start),
            managed_tensor_slice_to_ndbuffer(stop),
            managed_tensor_slice_to_ndbuffer(step),
        )


# ===----------------------------------------------------------------------===#
# Binary Elementwise Kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.add")
@compiler.elementwise
struct Add:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        z: ManagedTensorSlice,
        x: ManagedTensorSlice,
        y: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.type, width]:
            var lhs = rebind[SIMD[z.type, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[z.type, width]](y._fused_load[width](idx))
            return lhs + rhs

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.sub")
@compiler.elementwise
struct Sub:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        z: ManagedTensorSlice,
        x: ManagedTensorSlice,
        y: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.type, width]:
            var lhs = rebind[SIMD[z.type, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[z.type, width]](y._fused_load[width](idx))
            return lhs - rhs

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.mul")
@compiler.elementwise
struct Mul:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        z: ManagedTensorSlice,
        x: ManagedTensorSlice,
        y: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.type, width]:
            var lhs = rebind[SIMD[z.type, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[z.type, width]](y._fused_load[width](idx))
            return lhs * rhs

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.div")
@compiler.elementwise
struct Div:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        z: ManagedTensorSlice,
        x: ManagedTensorSlice,
        y: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.type, width]:
            var lhs = rebind[SIMD[z.type, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[z.type, width]](y._fused_load[width](idx))
            return lhs / rhs

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.mod")
@compiler.elementwise
struct Mod:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        z: ManagedTensorSlice,
        x: ManagedTensorSlice,
        y: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.type, width]:
            var lhs = rebind[SIMD[z.type, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[z.type, width]](y._fused_load[width](idx))
            return lhs % rhs

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.equal")
@compiler.elementwise
struct Equal:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        z: ManagedTensorSlice,
        x: ManagedTensorSlice,
        y: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.type, width]:
            var lhs = rebind[SIMD[x.type, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[x.type, width]](y._fused_load[width](idx))
            return rebind[SIMD[z.type, width]](lhs == rhs)

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.greater")
@compiler.elementwise
struct Greater:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        z: ManagedTensorSlice,
        x: ManagedTensorSlice,
        y: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.type, width]:
            var lhs = rebind[SIMD[x.type, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[x.type, width]](y._fused_load[width](idx))
            return rebind[SIMD[z.type, width]](lhs > rhs)

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.greater_equal")
@compiler.elementwise
struct GreaterEqual:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        z: ManagedTensorSlice,
        x: ManagedTensorSlice,
        y: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.type, width]:
            var lhs = rebind[SIMD[x.type, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[x.type, width]](y._fused_load[width](idx))
            return rebind[SIMD[z.type, width]](lhs >= rhs)

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.not_equal")
@compiler.elementwise
struct NotEqual:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        z: ManagedTensorSlice,
        x: ManagedTensorSlice,
        y: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.type, width]:
            var lhs = rebind[SIMD[x.type, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[x.type, width]](y._fused_load[width](idx))
            return rebind[SIMD[z.type, width]](lhs != rhs)

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.and")
@compiler.elementwise
struct And:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        z: ManagedTensorSlice,
        x: ManagedTensorSlice,
        y: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.type, width]:
            var lhs = rebind[SIMD[DType.bool, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[DType.bool, width]](y._fused_load[width](idx))
            return rebind[SIMD[z.type, width]](lhs & rhs)

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.or")
@compiler.elementwise
struct Or:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        z: ManagedTensorSlice,
        x: ManagedTensorSlice,
        y: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.type, width]:
            var lhs = rebind[SIMD[DType.bool, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[DType.bool, width]](y._fused_load[width](idx))
            return rebind[SIMD[z.type, width]](lhs | rhs)

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.xor")
@compiler.elementwise
struct Xor:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        z: ManagedTensorSlice,
        x: ManagedTensorSlice,
        y: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.type, width]:
            var lhs = rebind[SIMD[DType.bool, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[DType.bool, width]](y._fused_load[width](idx))
            return rebind[SIMD[z.type, width]](lhs ^ rhs)

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.pow")
@compiler.elementwise
struct Pow:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        z: ManagedTensorSlice,
        x: ManagedTensorSlice,
        y: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.type, width]:
            var lhs = rebind[SIMD[z.type, width]](x._fused_load[width](idx))
            var rhs = y._fused_load[width](idx)
            return _pow(lhs, rhs)

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.max")
@compiler.elementwise
struct Max:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        z: ManagedTensorSlice,
        x: ManagedTensorSlice,
        y: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.type, width]:
            var lhs = rebind[SIMD[z.type, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[z.type, width]](y._fused_load[width](idx))
            return max(lhs, rhs)

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.min")
@compiler.elementwise
struct Min:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        z: ManagedTensorSlice,
        x: ManagedTensorSlice,
        y: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.type, width]:
            var lhs = rebind[SIMD[z.type, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[z.type, width]](y._fused_load[width](idx))
            return min(lhs, rhs)

        foreach[func, synchronous, target](z, ctx)


# ===----------------------------------------------------------------------===#
# Unary Elementwise Kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.cast")
@compiler.elementwise
struct Cast:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](
                x._fused_load[width](idx).cast[y.type]()
            )

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.negative")
@compiler.elementwise
struct Negative:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](-x._fused_load[width](idx))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.relu")
@compiler.elementwise
struct ReLU:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](relu(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.gelu")
@compiler.elementwise
struct GeLU:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](gelu(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.ceil")
@compiler.elementwise
struct Ceil:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](ceil(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.floor")
@compiler.elementwise
struct Floor:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](floor(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.tanh")
@compiler.elementwise
struct Tanh:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](tanh(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.cos")
@compiler.elementwise
struct Cos:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](cos(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.sin")
@compiler.elementwise
struct Sin:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](sin(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.erf")
@compiler.elementwise
struct Erf:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](erf(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.exp")
@compiler.elementwise
struct Exp:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](exp(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.round")
@compiler.elementwise
struct Round:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](round(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.roundeven")
@compiler.elementwise
struct RoundEven:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](
                x._fused_load[width](idx).roundeven()
            )

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.sqrt")
@compiler.elementwise
struct Sqrt:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](sqrt(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.isqrt")
@compiler.elementwise
struct Isqrt:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](isqrt(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.select")
@compiler.elementwise
struct Select:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        out: ManagedTensorSlice,
        condition: ManagedTensorSlice,
        true_case: ManagedTensorSlice,
        false_case: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[out.rank]) -> SIMD[out.type, width]:
            var cond = condition._fused_load[width](idx)
            var tc = rebind[SIMD[out.type, width]](
                true_case._fused_load[width](idx)
            )
            var fc = rebind[SIMD[out.type, width]](
                false_case._fused_load[width](idx)
            )
            return cond.select(tc, fc)

        foreach[func, synchronous, target](out, ctx)


@compiler.register("mo.trunc")
@compiler.elementwise
struct Trunc:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            var val = x._fused_load[width](idx)
            return rebind[SIMD[y.type, width]](
                llvm_intrinsic[
                    "llvm.trunc", __type_of(val), has_side_effect=False
                ](val)
            )

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.log")
@compiler.elementwise
struct Log:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](log(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.log1p")
@compiler.elementwise
struct Log1p:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](log1p(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.is_nan")
@compiler.elementwise
struct IsNan:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](isnan(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.is_inf")
@compiler.elementwise
struct IsInf:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](isinf(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.not")
@compiler.elementwise
struct Not:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            var val = rebind[SIMD[DType.bool, width]](x._fused_load[width](idx))
            return rebind[SIMD[y.type, width]](~val)

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.abs")
@compiler.elementwise
struct Abs:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](abs(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


# ===----------------------------------------------------------------------===#
# ScatterND kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.scatter_nd")
struct ScatterND:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[output.type, output.rank],
        updates: ManagedTensorSlice[output.type, *_],
        indices: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ) raises:
        # Existing implementations do not require static shape information
        var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)
        scatter_nd[
            output_ndbuffer.type,
            indices_ndbuffer.type,
            output_ndbuffer.rank,
            indices_ndbuffer.rank,
            updates_ndbuffer.rank,
            synchronous,
            target,
        ](
            input_ndbuffer,
            indices_ndbuffer,
            updates_ndbuffer,
            output_ndbuffer,
            context=ctx,
        )


@compiler.register("mo.scatter_nd.add")
struct ScatterNDAdd:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[output.type, output.rank],
        updates: ManagedTensorSlice[output.type, *_],
        indices: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ) raises:
        # Existing implementations do not require static shape information
        var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)

        @always_inline
        @parameter
        fn reduce_fn[
            type: DType, width: Int
        ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
            return lhs + rhs

        scatter_nd_generator[
            output_ndbuffer.type,
            indices_ndbuffer.type,
            output_ndbuffer.rank,
            indices_ndbuffer.rank,
            updates_ndbuffer.rank,
            synchronous,
            target,
            reduce_fn=reduce_fn,
        ](
            input_ndbuffer,
            indices_ndbuffer,
            updates_ndbuffer,
            output_ndbuffer,
            context=ctx,
        )


@compiler.register("mo.scatter_nd.mul")
struct ScatterNDMul:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[output.type, output.rank],
        updates: ManagedTensorSlice[output.type, *_],
        indices: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ) raises:
        # Existing implementations do not require static shape information
        var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)

        @always_inline
        @parameter
        fn reduce_fn[
            type: DType, width: Int
        ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
            return lhs * rhs

        scatter_nd_generator[
            output_ndbuffer.type,
            indices_ndbuffer.type,
            output_ndbuffer.rank,
            indices_ndbuffer.rank,
            updates_ndbuffer.rank,
            synchronous,
            target,
            reduce_fn=reduce_fn,
        ](
            input_ndbuffer,
            indices_ndbuffer,
            updates_ndbuffer,
            output_ndbuffer,
            context=ctx,
        )


@compiler.register("mo.scatter_nd.min")
struct ScatterNDMin:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[output.type, output.rank],
        updates: ManagedTensorSlice[output.type, *_],
        indices: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ) raises:
        # Existing implementations do not require static shape information
        var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)

        @always_inline
        @parameter
        fn reduce_fn[
            type: DType, width: Int
        ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
            return min(lhs, rhs)

        scatter_nd_generator[
            output_ndbuffer.type,
            indices_ndbuffer.type,
            output_ndbuffer.rank,
            indices_ndbuffer.rank,
            updates_ndbuffer.rank,
            synchronous,
            target,
            reduce_fn=reduce_fn,
        ](
            input_ndbuffer,
            indices_ndbuffer,
            updates_ndbuffer,
            output_ndbuffer,
            context=ctx,
        )


@compiler.register("mo.scatter_nd.max")
struct ScatterNDMax:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[output.type, output.rank],
        updates: ManagedTensorSlice[output.type, *_],
        indices: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ) raises:
        # Existing implementations do not require static shape information
        var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)

        @always_inline
        @parameter
        fn reduce_fn[
            type: DType, width: Int
        ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
            return max(lhs, rhs)

        scatter_nd_generator[
            output_ndbuffer.type,
            indices_ndbuffer.type,
            output_ndbuffer.rank,
            indices_ndbuffer.rank,
            updates_ndbuffer.rank,
            synchronous,
            target,
            reduce_fn=reduce_fn,
        ](
            input_ndbuffer,
            indices_ndbuffer,
            updates_ndbuffer,
            output_ndbuffer,
            context=ctx,
        )


# ===----------------------------------------------------------------------===#
# Scatter kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.scatter")
struct Scatter:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[output.type, output.rank],
        updates: ManagedTensorSlice[output.type, output.rank],
        indices: ManagedTensorSlice[rank = output.rank],
        axis: ManagedTensorSlice[rank=1],
        ctx: MojoCallContextPtr,
    ) raises:
        var scalar_axis = managed_tensor_slice_to_ndbuffer(axis)[0]

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
            int(normalize_neg_index(scalar_axis, output.rank)),
            output,
        )

    @staticmethod
    fn shape(
        input: ManagedTensorSlice,
        updates: ManagedTensorSlice[input.type, input.rank],
        indices: ManagedTensorSlice[rank = input.rank],
        axis: ManagedTensorSlice[rank=1],
    ) raises -> IndexList[input.rank]:
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)
        var axis_ndbuffer = managed_tensor_slice_to_ndbuffer(axis)
        return scatter_elements_shape[
            input.rank,
            input.type,
            indices.type,
            axis.type,
            single_thread_blocking_override=True,
        ](input_ndbuffer, updates_ndbuffer, indices_ndbuffer, axis_ndbuffer)


@compiler.register("mo.scatter.add")
struct ScatterAdd:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[output.type, output.rank],
        updates: ManagedTensorSlice[output.type, output.rank],
        indices: ManagedTensorSlice[rank = output.rank],
        axis: ManagedTensorSlice[rank=1],
        ctx: MojoCallContextPtr,
    ) raises:
        var scalar_axis = managed_tensor_slice_to_ndbuffer(axis)[0]

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
            int(normalize_neg_index(scalar_axis, output.rank)),
            output,
        )

    @staticmethod
    fn shape(
        input: ManagedTensorSlice,
        updates: ManagedTensorSlice[input.type, input.rank],
        indices: ManagedTensorSlice[rank = input.rank],
        axis: ManagedTensorSlice[rank=1],
    ) raises -> IndexList[input.rank]:
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)
        var axis_ndbuffer = managed_tensor_slice_to_ndbuffer(axis)
        return scatter_elements_shape[
            input.rank,
            input.type,
            indices.type,
            axis.type,
            single_thread_blocking_override=True,
        ](input_ndbuffer, updates_ndbuffer, indices_ndbuffer, axis_ndbuffer)


@compiler.register("mo.scatter.max")
struct ScatterMax:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[output.type, output.rank],
        updates: ManagedTensorSlice[output.type, output.rank],
        indices: ManagedTensorSlice[rank = output.rank],
        axis: ManagedTensorSlice[rank=1],
        ctx: MojoCallContextPtr,
    ) raises:
        var scalar_axis = managed_tensor_slice_to_ndbuffer(axis)[0]

        @always_inline
        @parameter
        fn reduce_func[
            type: DType, width: Int
        ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
            return max(lhs, rhs)

        scatter_elements[reduce_func](
            input,
            indices,
            updates,
            int(normalize_neg_index(scalar_axis, output.rank)),
            output,
        )

    @staticmethod
    fn shape(
        input: ManagedTensorSlice,
        updates: ManagedTensorSlice[input.type, input.rank],
        indices: ManagedTensorSlice[rank = input.rank],
        axis: ManagedTensorSlice[rank=1],
    ) raises -> IndexList[input.rank]:
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)
        var axis_ndbuffer = managed_tensor_slice_to_ndbuffer(axis)
        return scatter_elements_shape[
            input.rank,
            input.type,
            indices.type,
            axis.type,
            single_thread_blocking_override=True,
        ](input_ndbuffer, updates_ndbuffer, indices_ndbuffer, axis_ndbuffer)


@compiler.register("mo.scatter.min")
struct ScatterMin:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[output.type, output.rank],
        updates: ManagedTensorSlice[output.type, output.rank],
        indices: ManagedTensorSlice[rank = output.rank],
        axis: ManagedTensorSlice[rank=1],
        ctx: MojoCallContextPtr,
    ) raises:
        var scalar_axis = managed_tensor_slice_to_ndbuffer(axis)[0]

        @always_inline
        @parameter
        fn reduce_func[
            type: DType, width: Int
        ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
            return min(lhs, rhs)

        scatter_elements[reduce_func](
            input,
            indices,
            updates,
            int(normalize_neg_index(scalar_axis, output.rank)),
            output,
        )

    @staticmethod
    fn shape(
        input: ManagedTensorSlice,
        updates: ManagedTensorSlice[input.type, input.rank],
        indices: ManagedTensorSlice[rank = input.rank],
        axis: ManagedTensorSlice[rank=1],
    ) raises -> IndexList[input.rank]:
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)
        var axis_ndbuffer = managed_tensor_slice_to_ndbuffer(axis)
        return scatter_elements_shape[
            input.rank,
            input.type,
            indices.type,
            axis.type,
            single_thread_blocking_override=True,
        ](input_ndbuffer, updates_ndbuffer, indices_ndbuffer, axis_ndbuffer)


@compiler.register("mo.scatter.mul")
struct ScatterMul:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[output.type, output.rank],
        updates: ManagedTensorSlice[output.type, output.rank],
        indices: ManagedTensorSlice[rank = output.rank],
        axis: ManagedTensorSlice[rank=1],
        ctx: MojoCallContextPtr,
    ) raises:
        # Existing implementations do not require static shape information
        var scalar_axis = managed_tensor_slice_to_ndbuffer(axis)[0]

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
            int(normalize_neg_index(scalar_axis, output.rank)),
            output,
        )

    @staticmethod
    fn shape(
        input: ManagedTensorSlice,
        updates: ManagedTensorSlice[input.type, input.rank],
        indices: ManagedTensorSlice[rank = input.rank],
        axis: ManagedTensorSlice[rank=1],
    ) raises -> IndexList[input.rank]:
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)
        var axis_ndbuffer = managed_tensor_slice_to_ndbuffer(axis)
        return scatter_elements_shape[
            input.rank,
            input.type,
            indices.type,
            axis.type,
            single_thread_blocking_override=True,
        ](input_ndbuffer, updates_ndbuffer, indices_ndbuffer, axis_ndbuffer)


# ===----------------------------------------------------------------------===#
# View kernels
# ===----------------------------------------------------------------------===#


# TensorCopy intrinsic used by view kernels.
# z is a kernel output, and x a view of the input.
@no_inline
fn view_copy_impl[
    synchronous: Bool, target: StringLiteral, type: DType, rank: Int
](z: ManagedTensorSlice[type, rank], x: ManagedTensorSlice[type, rank]):
    @parameter
    @always_inline
    fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.type, width]:
        return x._simd_load_internal[width](idx)

    foreach[func](z)


@compiler.register("mo.static.broadcast_to")
@compiler.view_kernel
struct BroadcastTo:
    @staticmethod
    fn build_view[
        type: DType,
        in_rank: Int,
        out_rank: Int,
    ](
        x: ManagedTensorSlice[type, in_rank],
        output_shape: IndexList[out_rank],
    ) -> ManagedTensorSlice[type, out_rank]:
        var new_strides = IndexList[out_rank]()
        alias delta = out_rank - in_rank

        @parameter
        for i in range(out_rank):
            if i < delta:
                new_strides[i] = 0
            elif x.dim_size(i - delta) <= 1:
                new_strides[i] = 0
            else:
                new_strides[i] = x._strides[i - delta]

        return ManagedTensorSlice[type, out_rank](
            x._ptr, output_shape, new_strides
        )

    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
        type: DType,
        in_rank: Int,
        out_rank: Int,
    ](
        z: ManagedTensorSlice[type, out_rank],
        x: ManagedTensorSlice[type, in_rank],
        output_shape: IndexList[out_rank],
    ):
        # We need the extra output_shape argument.
        # Using `z.shape` instead will prevent the compiler from fusing the kernels.
        var x_view = Self.build_view(x, output_shape)
        view_copy_impl[synchronous, target](z, x_view)


@compiler.register("mo.static.reshape")
@compiler.view_kernel
struct StaticReshape:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
        type: DType,
        output_rank: Int,
    ](
        output: ManagedTensorSlice[type=type, rank=output_rank],
        input: ManagedTensorSlice[type=type],
        shape: IndexList[output_rank],
    ):
        var view_buffer = reshape(
            managed_tensor_slice_to_ndbuffer(input), shape
        )
        var view_tensor = ManagedTensorSlice[type, output_rank](
            view_buffer.data, shape, view_buffer.get_strides()
        )
        view_copy_impl[synchronous, target](output, view_tensor)


@compiler.register("mo.reshape")
struct Reshape:
    # The `execute` method should never be used in the graph compiler.
    @staticmethod
    fn execute():
        pass

    @staticmethod
    fn shape[
        output_rank: Int
    ](
        input: ManagedTensorSlice, shape: ManagedTensorSlice[rank=1]
    ) raises -> IndexList[output_rank]:
        return reshape_shape[
            output_rank=output_rank, single_thread_blocking_override=True
        ](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(shape),
        )


@compiler.register("mo.transpose")
@compiler.view_kernel
struct Transpose:
    @staticmethod
    fn transpose_in_place(
        input: ManagedTensorSlice,
        permutations: ManagedTensorSlice[rank=1],
    ) -> ManagedTensorSlice[type = input.type, rank = input.rank]:
        var new_shape = IndexList[input.rank]()
        var new_stride = IndexList[input.rank]()

        @parameter
        for i in range(input.rank):
            var dim = int(permutations[i])
            new_shape[i] = input.spec().shape[dim]
            new_stride[i] = input._strides[dim]

        return ManagedTensorSlice[type = input.type, rank = input.rank](
            input._ptr, new_shape, new_stride
        )

    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
        type: DType,
        rank: Int,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        permutations: ManagedTensorSlice[rank=1],
    ):
        view_copy_impl[synchronous, target](
            output, Self.transpose_in_place(input, permutations)
        )

    # TODO(GRA-1033) Make it possible to have multiple raises.
    @no_inline
    @staticmethod
    fn shape_impl(
        input: ManagedTensorSlice,
        permutations: ManagedTensorSlice[rank=1],
    ) raises -> IndexList[input.rank]:
        if permutations.spec().shape[0] != input.rank:
            raise Error("[transpose] permutation size must match input rank")

        @parameter
        for i in range(input.rank):
            var perm = int(permutations[i])
            if perm < 0 or input.rank <= perm:
                raise Error(
                    "[transpose] each permutation must be within range [0,"
                    " rank)"
                )

        var view_tensor = Self.transpose_in_place(input, permutations)
        var out = IndexList[input.rank]()

        @parameter
        for i in range(input.rank):
            out[i] = view_tensor.spec().shape[i]

        return out

    @staticmethod
    fn shape(
        input: ManagedTensorSlice,
        permutations: ManagedTensorSlice[rank=1],
    ) raises -> IndexList[input.rank]:
        return Self.shape_impl(input, permutations)


@compiler.register("mo.slice")
@compiler.view_kernel
struct Slice:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
        type: DType,
        rank: Int,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        starts: ManagedTensorSlice[rank=1],
        stops: ManagedTensorSlice[rank=1],
        steps: ManagedTensorSlice[rank=1],
    ):
        var view_buffer = slice_as_view(
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(starts),
            managed_tensor_slice_to_ndbuffer(stops),
            managed_tensor_slice_to_ndbuffer(steps),
        )
        var view_tensor = ManagedTensorSlice[type, rank](
            view_buffer.data,
            view_buffer.get_shape(),
            view_buffer.get_strides(),
        )
        view_copy_impl[synchronous, target](output, view_tensor)

    @staticmethod
    fn shape(
        input: ManagedTensorSlice,
        starts: ManagedTensorSlice[rank=1],
        stops: ManagedTensorSlice[rank=1],
        steps: ManagedTensorSlice[rank=1],
    ) raises -> IndexList[input.rank]:
        return slice_shape[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(starts),
            managed_tensor_slice_to_ndbuffer(stops),
            managed_tensor_slice_to_ndbuffer(steps),
        )


@compiler.register("mo.mutable.store.slice", num_dps_outputs=0)
struct MutableStoreSlice:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
        type: DType,
        rank: Int,
    ](
        to_buffer: ManagedTensorSlice[type=type, rank=rank],
        from_slice: ManagedTensorSlice[type=type, rank=rank],
        starts: ManagedTensorSlice[rank=1],
        stops: ManagedTensorSlice[rank=1],
        steps: ManagedTensorSlice[rank=1],
    ):
        var to_buffer_ndb_view = slice_as_view(
            managed_tensor_slice_to_ndbuffer(to_buffer),
            managed_tensor_slice_to_ndbuffer(starts),
            managed_tensor_slice_to_ndbuffer(stops),
            managed_tensor_slice_to_ndbuffer(steps),
        )
        var to_buffer_mts_view = ManagedTensorSlice[type, rank](
            to_buffer_ndb_view.data,
            to_buffer_ndb_view.get_shape(),
            to_buffer_ndb_view.get_strides(),
        )
        view_copy_impl[synchronous, target](to_buffer_mts_view, from_slice)

    # No shape function as it currently just routes to mo.slice's (done in
    # legalize-rmo-operators) Can have a proper shape function once the whole
    # GC stack has moved to the new kernel API
    #
    # TODO(GRA-1178): Generic support for in-place kernels where the shape is
    # being enforced on one of the inputs (e.g. on from_slice for this kernel)


@compiler.register("mo.slice_dim")
@compiler.view_kernel
struct SliceDim:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
        type: DType,
        rank: Int,
        axis: Int,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        starts: ScalarTensor,
        stops: ScalarTensor,
        steps: ScalarTensor,
    ):
        var view_buffer = slice_dim_as_view[dim=axis](
            managed_tensor_slice_to_ndbuffer(input),
            int(starts[0]),
            int(stops[0]),
            int(steps[0]),
        )
        var view_tensor = ManagedTensorSlice[type, rank](
            view_buffer.data,
            view_buffer.get_shape(),
            view_buffer.get_strides(),
        )
        view_copy_impl[synchronous, target](output, view_tensor)


# ===----------------------------------------------------------------------===#
# Data dependent kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.arg_max")
struct ArgMax:
    @staticmethod
    fn execute(
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[rank = output.rank],
        axis: ManagedTensorSlice[rank=1],
    ) raises:
        alias output_shape = compiler.specsof[output.type, output.rank](
            "output"
        ).shape
        alias axis_shape = compiler.specsof[axis.type, axis.rank]("axis").shape
        alias input_shape = compiler.specsof[input.type, input.rank](
            "input"
        ).shape

        var output_ndbuffer = managed_tensor_slice_to_ndbuffer[
            static_shape=output_shape
        ](output)
        var axis_ndbuffer = managed_tensor_slice_to_ndbuffer[
            static_shape=axis_shape
        ](axis)
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer[
            static_shape=input_shape
        ](input)

        argmax(input_ndbuffer, axis_ndbuffer, output_ndbuffer)


@compiler.register("mo.arg_min")
struct ArgMin:
    @staticmethod
    fn execute(
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[rank = output.rank],
        axis: ManagedTensorSlice[rank=1],
    ) raises:
        alias output_shape = compiler.specsof[output.type, output.rank](
            "output"
        ).shape
        alias axis_shape = compiler.specsof[axis.type, axis.rank]("axis").shape
        alias input_shape = compiler.specsof[input.type, input.rank](
            "input"
        ).shape

        var output_ndbuffer = managed_tensor_slice_to_ndbuffer[
            static_shape=output_shape
        ](output)
        var axis_ndbuffer = managed_tensor_slice_to_ndbuffer[
            static_shape=axis_shape
        ](axis)
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer[
            static_shape=input_shape
        ](input)

        argmin(input_ndbuffer, axis_ndbuffer, output_ndbuffer)


@compiler.register("mo.arg_nonzero")
struct ArgNonZero:
    @staticmethod
    fn execute(
        output_buffer: ManagedTensorSlice[rank=2],
        input_buffer: ManagedTensorSlice,
    ):
        var out_ndbuffer = managed_tensor_slice_to_ndbuffer(output_buffer)
        var in_ndbuffer = managed_tensor_slice_to_ndbuffer(input_buffer)

        arg_nonzero.arg_nonzero(in_ndbuffer, out_ndbuffer)

    @staticmethod
    fn shape(input_buffer: ManagedTensorSlice) -> IndexList[2]:
        return arg_nonzero.arg_nonzero_shape[
            single_thread_blocking_override=True
        ](managed_tensor_slice_to_ndbuffer(input_buffer))


@compiler.register("mo.mean")
struct Mean:
    @staticmethod
    fn execute[
        synchronous: Bool, target: StringLiteral
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[type = output.type, rank = output.rank],
        axis: ScalarTensor,
        ctx: MojoCallContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[input.type, width]:
            return input._fused_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        fn output_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank], val: SIMD[output.type, width]):
            output._fused_store[width=width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.type, width]](val),
            )

        var axis_val = int(axis[0])

        mean[
            output.type,
            input_fn,
            output_fn,
            single_thread_blocking_override=synchronous,
            target=target,
        ](input._spec.shape, axis_val, output._spec.shape, ctx)

    @staticmethod
    fn shape[
        input_rank: Int,
        input_type: DType,
    ](
        input: ManagedTensorSlice[input_type, input_rank],
        axis: ScalarTensor,
    ) raises -> IndexList[input_rank]:
        return reduce_shape[input_rank, input_type](input, axis)


@compiler.register("mo.reduce.add")
struct ReduceAdd:
    @staticmethod
    fn execute[
        synchronous: Bool, target: StringLiteral
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[type = output.type, rank = output.rank],
        axis: ScalarTensor,
        ctx: MojoCallContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[input.type, width]:
            return input._fused_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        fn output_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank], val: SIMD[output.type, width]):
            output._fused_store[width=width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.type, width]](val),
            )

        var axis_val = int(axis[0])

        sum[
            output.type,
            input_fn,
            output_fn,
            single_thread_blocking_override=synchronous,
            target=target,
        ](input._spec.shape, axis_val, ctx)

    @staticmethod
    fn shape[
        input_rank: Int,
        input_type: DType,
    ](
        input: ManagedTensorSlice[input_type, input_rank],
        axis: ScalarTensor,
    ) raises -> IndexList[input_rank]:
        return reduce_shape[input_rank, input_type](input, axis)


@compiler.register("mo.reduce.mul")
struct ReduceMul:
    @staticmethod
    fn execute[
        synchronous: Bool, target: StringLiteral
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[type = output.type, rank = output.rank],
        axis: ScalarTensor,
        ctx: MojoCallContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[input.type, width]:
            return input._fused_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        fn output_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank], val: SIMD[output.type, width]):
            output._fused_store[width=width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.type, width]](val),
            )

        var axis_val = int(axis[0])

        product[
            output.type,
            input_fn,
            output_fn,
            single_thread_blocking_override=synchronous,
            target=target,
        ](input._spec.shape, axis_val, ctx)

    @staticmethod
    fn shape[
        input_rank: Int,
        input_type: DType,
    ](
        input: ManagedTensorSlice[input_type, input_rank],
        axis: ScalarTensor,
    ) raises -> IndexList[input_rank]:
        return reduce_shape[input_rank, input_type](input, axis)


@compiler.register("mo.reduce.max")
struct ReduceMax:
    @staticmethod
    fn execute[
        synchronous: Bool, target: StringLiteral
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[type = output.type, rank = output.rank],
        axis: ScalarTensor,
        ctx: MojoCallContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[input.type, width]:
            return input._fused_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        fn output_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank], val: SIMD[output.type, width]):
            output._fused_store[width=width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.type, width]](val),
            )

        var axis_val = int(axis[0])

        reduce_max[
            output.type,
            input_fn,
            output_fn,
            single_thread_blocking_override=synchronous,
            target=target,
        ](input._spec.shape, axis_val, ctx)

    @staticmethod
    fn shape[
        input_rank: Int,
        input_type: DType,
    ](
        input: ManagedTensorSlice[input_type, input_rank],
        axis: ScalarTensor,
    ) raises -> IndexList[input_rank]:
        return reduce_shape[input_rank, input_type](input, axis)


@compiler.register("mo.reduce.min")
struct ReduceMin:
    @staticmethod
    fn execute[
        synchronous: Bool, target: StringLiteral
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[type = output.type, rank = output.rank],
        axis: ScalarTensor,
        ctx: MojoCallContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[input.type, width]:
            return input._fused_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        fn output_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank], val: SIMD[output.type, width]):
            output._fused_store[width=width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.type, width]](val),
            )

        var axis_val = int(axis[0])

        reduce_min[
            output.type,
            input_fn,
            output_fn,
            single_thread_blocking_override=synchronous,
            target=target,
        ](input._spec.shape, axis_val, ctx)

    @staticmethod
    fn shape[
        input_rank: Int,
        input_type: DType,
    ](
        input: ManagedTensorSlice[input_type, input_rank],
        axis: ScalarTensor,
    ) raises -> IndexList[input_rank]:
        return reduce_shape[input_rank, input_type](input, axis)


@compiler.register("reduce_min_and_max")
struct ReduceMinMax:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
        type: DType,
        rank: Int,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        axis0: ScalarTensor,
        ctx: MojoCallContextPtr,
    ) raises:
        """Given a tensor of shape [A, B, C, D] and reducing along dimension 'C'
        writes to a tensor of shape [A, B, 2, D] where [:, :, 0, :] contains
        the minimum reduction and [:, :, 1, :] contains the maximum reduction.
        """

        alias num_reductions = 2
        var axis = int(normalize_neg_index(axis0[0], rank))

        @parameter
        @always_inline
        fn input_0_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[input.type, width]:
            return input._fused_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        fn output_0_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank], val: SIMD[output.type, width]):
            output._fused_store[width=width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.type, width]](val),
            )

        @always_inline
        @parameter
        fn input_0_fn_wrapper[
            _type: DType, width: Int, rank: Int
        ](idx: IndexList[rank]) -> SIMD[_type, width]:
            return rebind[SIMD[_type, width]](input_0_fn[width, rank](idx))

        @always_inline
        @parameter
        fn output_0_fn_wrapper[
            _type: DType, width: Int, rank: Int
        ](
            indices: IndexList[rank],
            val: StaticTuple[SIMD[_type, width], num_reductions],
        ):
            # TODO: once we support multiple outputs, change this to route to
            # TODO: multiple output tensors.
            var indices_min = indices
            indices_min[axis] = 0
            output_0_fn[width, rank](
                indices_min, rebind[SIMD[type, width]](val[0])
            )

            var indices_max = indices
            indices_max[axis] = 1
            output_0_fn[width, rank](
                indices_max, rebind[SIMD[type, width]](val[1])
            )

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

        with Trace[TraceLevel.OP, target=target]("reduce_min_and_max"):
            _reduce_generator[
                num_reductions,
                type,
                input_0_fn_wrapper,
                output_0_fn_wrapper,
                reduce_fn,
                single_thread_blocking_override=synchronous,
                target=target,
            ](
                input.get_static_spec().shape,
                init=init,
                reduce_dim=axis,
                context=ctx,
            )
        _ = axis

    @staticmethod
    fn shape(
        input: ManagedTensorSlice, axis0: ScalarTensor
    ) -> IndexList[input.rank]:
        var new_shape = input.get_static_spec().shape
        var axis = int(normalize_neg_index(axis0[0], input.rank))
        new_shape[axis] = 2

        return new_shape


# ===----------------------------------------------------------------------===#
# Pooling kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.avg_pool")
struct AvgPool:
    @staticmethod
    fn execute[
        count_boundary: Bool,
        type: DType,
        int_type: DType,
    ](
        output: ManagedTensorSlice[type, 4],
        input: ManagedTensorSlice[type, 4],
        filter: ManagedTensorSlice[int_type, 1],
        strides: ManagedTensorSlice[int_type, 1],
        dilations: ManagedTensorSlice[int_type, 1],
        paddings: ManagedTensorSlice[int_type, 1],
    ):
        avg_pool[count_boundary=count_boundary](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(filter),
            managed_tensor_slice_to_ndbuffer(strides),
            managed_tensor_slice_to_ndbuffer(dilations),
            managed_tensor_slice_to_ndbuffer(paddings),
            managed_tensor_slice_to_ndbuffer(output),
            False,
        )

    @staticmethod
    fn shape[
        type: DType,
        int_type: DType,
    ](
        input: ManagedTensorSlice[type, 4],
        filter: ManagedTensorSlice[int_type, 1],
        strides: ManagedTensorSlice[int_type, 1],
        dilations: ManagedTensorSlice[int_type, 1],
        paddings: ManagedTensorSlice[int_type, 1],
    ) raises -> IndexList[input.rank]:
        return pool_shape[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(filter),
            managed_tensor_slice_to_ndbuffer(strides),
            managed_tensor_slice_to_ndbuffer(dilations),
            managed_tensor_slice_to_ndbuffer(paddings),
        )


@compiler.register("mo.avg_pool_ceil_mode_true")
struct AvgPoolCeilModeTrue:
    @staticmethod
    fn execute[
        count_boundary: Bool,
        type: DType,
        int_type: DType,
    ](
        output: ManagedTensorSlice[type, 4],
        input: ManagedTensorSlice[type, 4],
        filter: ManagedTensorSlice[int_type, 1],
        strides: ManagedTensorSlice[int_type, 1],
        dilations: ManagedTensorSlice[int_type, 1],
        paddings: ManagedTensorSlice[int_type, 1],
    ):
        avg_pool[count_boundary=count_boundary](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(filter),
            managed_tensor_slice_to_ndbuffer(strides),
            managed_tensor_slice_to_ndbuffer(dilations),
            managed_tensor_slice_to_ndbuffer(paddings),
            managed_tensor_slice_to_ndbuffer(output),
            True,
        )

    @staticmethod
    fn shape[
        type: DType,
        int_type: DType,
    ](
        input: ManagedTensorSlice[type, 4],
        filter: ManagedTensorSlice[int_type, 1],
        strides: ManagedTensorSlice[int_type, 1],
        dilations: ManagedTensorSlice[int_type, 1],
        paddings: ManagedTensorSlice[int_type, 1],
        ctx: MojoCallContextPtr,
    ) raises -> IndexList[input.rank]:
        return pool_shape_ceil[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(filter),
            managed_tensor_slice_to_ndbuffer(strides),
            managed_tensor_slice_to_ndbuffer(dilations),
            managed_tensor_slice_to_ndbuffer(paddings),
        )


@compiler.register("mo.max_pool")
struct MaxPool:
    @staticmethod
    fn execute[
        type: DType,
        int_type: DType,
    ](
        output: ManagedTensorSlice[type, 4],
        input: ManagedTensorSlice[type, 4],
        filter: ManagedTensorSlice[int_type, 1],
        strides: ManagedTensorSlice[int_type, 1],
        dilations: ManagedTensorSlice[int_type, 1],
        paddings: ManagedTensorSlice[int_type, 1],
    ):
        max_pool(
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(filter),
            managed_tensor_slice_to_ndbuffer(strides),
            managed_tensor_slice_to_ndbuffer(dilations),
            managed_tensor_slice_to_ndbuffer(paddings),
            managed_tensor_slice_to_ndbuffer(output),
            False,
        )

    @staticmethod
    fn shape[
        type: DType,
        int_type: DType,
    ](
        input: ManagedTensorSlice[type, 4],
        filter: ManagedTensorSlice[int_type, 1],
        strides: ManagedTensorSlice[int_type, 1],
        dilations: ManagedTensorSlice[int_type, 1],
        paddings: ManagedTensorSlice[int_type, 1],
    ) raises -> IndexList[input.rank]:
        return pool_shape[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(filter),
            managed_tensor_slice_to_ndbuffer(strides),
            managed_tensor_slice_to_ndbuffer(dilations),
            managed_tensor_slice_to_ndbuffer(paddings),
        )


@compiler.register("mo.max_pool_ceil_mode_true")
struct MaxPoolCeilModeTrue:
    @staticmethod
    fn execute[
        type: DType,
        int_type: DType,
    ](
        output: ManagedTensorSlice[type, 4],
        input: ManagedTensorSlice[type, 4],
        filter: ManagedTensorSlice[int_type, 1],
        strides: ManagedTensorSlice[int_type, 1],
        dilations: ManagedTensorSlice[int_type, 1],
        paddings: ManagedTensorSlice[int_type, 1],
    ):
        max_pool(
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(filter),
            managed_tensor_slice_to_ndbuffer(strides),
            managed_tensor_slice_to_ndbuffer(dilations),
            managed_tensor_slice_to_ndbuffer(paddings),
            managed_tensor_slice_to_ndbuffer(output),
            True,
        )

    @staticmethod
    fn shape[
        type: DType,
        int_type: DType,
    ](
        input: ManagedTensorSlice[type, 4],
        filter: ManagedTensorSlice[int_type, 1],
        strides: ManagedTensorSlice[int_type, 1],
        dilations: ManagedTensorSlice[int_type, 1],
        paddings: ManagedTensorSlice[int_type, 1],
    ) raises -> IndexList[input.rank]:
        return pool_shape_ceil[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(filter),
            managed_tensor_slice_to_ndbuffer(strides),
            managed_tensor_slice_to_ndbuffer(dilations),
            managed_tensor_slice_to_ndbuffer(paddings),
        )


# ===----------------------------------------------------------------------===#
# Padding kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.pad.constant")
struct PadConstant:
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        padding: ManagedTensorSlice[rank=1],
        constant: ManagedTensorSlice[rank=1],
    ):
        var paddings_ptr = padding._ptr
        var constant_simd = constant._ptr.load(0)
        var input_buf = managed_tensor_slice_to_ndbuffer(input)
        var output_buf = managed_tensor_slice_to_ndbuffer(output)
        pad_constant(output_buf, input_buf, paddings_ptr, constant_simd)

    @staticmethod
    fn shape[
        type: DType,
        rank: Int,
    ](
        input: ManagedTensorSlice[type=type, rank=rank],
        padding: ManagedTensorSlice[rank=1],
    ) raises -> IndexList[rank]:
        return pad_shape[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(padding),
        )


@compiler.register("mo.pad.repeat")
struct PadRepeat:
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        padding: ManagedTensorSlice[rank=1],
    ):
        var paddings_ptr = padding._ptr
        var input_buf = managed_tensor_slice_to_ndbuffer(input)
        var output_buf = managed_tensor_slice_to_ndbuffer(output)
        pad_repeat(output_buf, input_buf, paddings_ptr)

    @staticmethod
    fn shape[
        type: DType,
        rank: Int,
    ](
        input: ManagedTensorSlice[type=type, rank=rank],
        padding: ManagedTensorSlice[rank=1],
    ) raises -> IndexList[rank]:
        return pad_shape[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(padding),
        )


@compiler.register("mo.pad.reflect")
struct PadReflect:
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        padding: ManagedTensorSlice[rank=1],
    ):
        var paddings_ptr = padding._ptr
        var input_buf = managed_tensor_slice_to_ndbuffer(input)
        var output_buf = managed_tensor_slice_to_ndbuffer(output)
        pad_reflect(output_buf, input_buf, paddings_ptr)

    @staticmethod
    fn shape[
        type: DType,
        rank: Int,
    ](
        input: ManagedTensorSlice[type=type, rank=rank],
        padding: ManagedTensorSlice[rank=1],
    ) raises -> IndexList[rank]:
        return pad_shape[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(padding),
        )


# ===----------------------------------------------------------------------===#
# Gather kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.gather_nd")
struct GatherND:
    @staticmethod
    fn execute[
        batchDims: Int,
        synchronous: Bool,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice,
        data: ManagedTensorSlice[output.type, *_],
        indices: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ) raises:
        # Existing implementations do not require static shape information
        var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)
        var data_ndbuffer = managed_tensor_slice_to_ndbuffer(data)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)

        gather_nd[batch_dims=batchDims](
            data_ndbuffer, indices_ndbuffer, output_ndbuffer, ctx
        )


@compiler.register("mo.gather")
struct Gather:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[output.type, *_],
        indices: ManagedTensorSlice,
        axis: ScalarTensor,
        ctx: MojoCallContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[output.type, width]:
            return input.load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        fn indices_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[indices.type, width]:
            return indices.load[width=width](
                rebind[IndexList[indices.rank]](coords)
            )

        @parameter
        @always_inline
        fn output_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank], val: SIMD[output.type, width]):
            output.store[width=width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.type, width]](val),
            )

        var axis_val = axis._ptr.load(0)

        gather[
            type = output.type,
            indices_type = indices.type,
            input_fn=input_fn,
            indices_fn=indices_fn,
            output_fn=output_fn,
            target=target,
            single_thread_blocking_override=synchronous,
        ](
            Axis(axis_val, input.rank),
            input._spec.shape,
            indices._spec.shape,
            output._spec.shape,
            context=ctx,
        )

    @staticmethod
    fn shape[
        output_rank: Int,
    ](
        input: ManagedTensorSlice,
        indices: ManagedTensorSlice,
        axis: ScalarTensor,
    ) raises -> IndexList[output_rank]:
        return gather_shape[
            output_rank=output_rank,
            single_thread_blocking_override=True,
        ](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(indices),
            managed_tensor_slice_to_ndbuffer(axis),
        )


@compiler.register("mo.gather_sum")
struct GatherSum:
    @staticmethod
    fn execute(
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[output.type, *_],
        indices: ManagedTensorSlice[DType.int32, *_],
    ) raises:
        # Existing implementations do not require static shape information
        var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)

        fn add[
            type: DType, simd_width: Int
        ](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
            type, simd_width
        ]:
            return x + y

        gather_reduce[output.type, 0, 1, simdwidthof[output.type](), add](
            output_ndbuffer, input_ndbuffer, indices_ndbuffer, 0
        )


# ===----------------------------------------------------------------------===#
# Normalization kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.layer_norm")
struct LayerNorm:
    @compiler.enable_fusion_for("input", "gamma")
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        gamma: ManagedTensorSlice[type=type, rank=1],
        beta: ManagedTensorSlice[type=type, rank=1],
        epsilon: ScalarTensor[type=type],
        ctx: MojoCallContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[type, width]:
            return input._fused_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        fn gamma_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[type, width]:
            return gamma._fused_load[width=width](rebind[IndexList[1]](coords))

        var beta_buf = managed_tensor_slice_to_ndbuffer(beta)
        var epsilon_val = epsilon._ptr.load(0)
        var output_buf = managed_tensor_slice_to_ndbuffer(output)

        layer_norm[type, rank, input_fn, gamma_fn, target=target,](
            input._spec.shape,
            gamma._spec.shape,
            beta_buf,
            epsilon_val,
            output_buf,
            ctx,
        )

    @staticmethod
    fn shape[
        type: DType,
        rank: Int,
    ](
        input: ManagedTensorSlice[type=type, rank=rank],
        gamma: ManagedTensorSlice[type=type, rank=1],
        beta: ManagedTensorSlice[type=type, rank=1],
        epsilon: ScalarTensor[type=type],
    ) -> IndexList[rank]:
        return input._spec.shape


@compiler.register("rms_norm")
struct RMSNorm:
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        gamma: ManagedTensorSlice[type=type, rank=1],
        epsilon: ScalarTensor[type=type],
        ctx: MojoCallContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[type, width]:
            return input.load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        var gamma_buf = managed_tensor_slice_to_ndbuffer(gamma)
        var epsilon_val = epsilon._ptr.load(0)
        var output_buf = managed_tensor_slice_to_ndbuffer(output)

        rms_norm[type, rank, input_fn, target=target](
            input._spec.shape, gamma_buf, epsilon_val, output_buf, ctx
        )

    @staticmethod
    fn shape[
        type: DType,
        rank: Int,
    ](
        input: ManagedTensorSlice[type=type, rank=rank],
        gamma: ManagedTensorSlice[type=type, rank=1],
        epsilon: ScalarTensor[type=type],
    ) -> IndexList[rank]:
        return input._spec.shape


# ===----------------------------------------------------------------------===#
# TopK kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.bottom_k", num_dps_outputs=2)
struct BottomK:
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
    ](
        values: ManagedTensorSlice[type=type, rank=rank],
        indices: ManagedTensorSlice[type = DType.int64, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        k: ScalarTensor,
        axis: ScalarTensor,
        sorted: ScalarTensor[type = DType.bool],
    ):
        top_k(
            managed_tensor_slice_to_ndbuffer(input),
            int(k[0]),
            int(axis[0]),
            False,
            managed_tensor_slice_to_ndbuffer(values),
            managed_tensor_slice_to_ndbuffer(indices),
            sorted[0],
        )

    @staticmethod
    fn shape[
        axis_type: DType
    ](
        input: ManagedTensorSlice,
        k: ScalarTensor[axis_type],
        axis: ScalarTensor[axis_type],
        sorted: ScalarTensor[type = DType.bool],
    ) raises -> IndexList[input.rank]:
        return top_k_shape_impl[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(k),
            managed_tensor_slice_to_ndbuffer(axis),
        )


@compiler.register("mo.top_k", num_dps_outputs=2)
struct TopK:
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
    ](
        values: ManagedTensorSlice[type=type, rank=rank],
        indices: ManagedTensorSlice[type = DType.int64, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        k: ScalarTensor,
        axis: ScalarTensor,
        sorted: ScalarTensor[type = DType.bool],
    ):
        top_k(
            managed_tensor_slice_to_ndbuffer(input),
            int(k[0]),
            int(axis[0]),
            True,
            managed_tensor_slice_to_ndbuffer(values),
            managed_tensor_slice_to_ndbuffer(indices),
            sorted[0],
        )

    @staticmethod
    fn shape[
        axis_type: DType
    ](
        input: ManagedTensorSlice,
        k: ScalarTensor[axis_type],
        axis: ScalarTensor[axis_type],
        sorted: ScalarTensor[type = DType.bool],
    ) raises -> IndexList[input.rank]:
        return top_k_shape_impl[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(k),
            managed_tensor_slice_to_ndbuffer(axis),
        )


# ===----------------------------------------------------------------------===#
# Non maximum suppression kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.non_maximum_suppression")
struct NonMaximumSupression:
    @staticmethod
    fn execute[
        type: DType
    ](
        output: ManagedTensorSlice[type = DType.int64, rank=2],
        boxes: ManagedTensorSlice[type=type, rank=3],
        scores: ManagedTensorSlice[type, rank=3],
        max_output_boxes_per_class: ScalarTensor[DType.int64],
        iou_threshold: ScalarTensor[DType.float32],
        score_threshold: ScalarTensor[DType.float32],
    ):
        var max_output_boxes_int = int(max_output_boxes_per_class[0])
        var iou_threshold_float = iou_threshold[0]
        var score_threshold_float = score_threshold[0]

        non_max_suppression(
            managed_tensor_slice_to_ndbuffer(boxes),
            managed_tensor_slice_to_ndbuffer(scores),
            managed_tensor_slice_to_ndbuffer(output),
            max_output_boxes_int,
            iou_threshold_float,
            score_threshold_float,
        )

    @staticmethod
    fn shape[
        type: DType
    ](
        boxes: ManagedTensorSlice[type=type, rank=3],
        scores: ManagedTensorSlice[type=type, rank=3],
        max_output_boxes_per_class: ScalarTensor[DType.int64],
        iou_threshold: ScalarTensor[DType.float32],
        score_threshold: ScalarTensor[DType.float32],
    ) -> IndexList[2]:
        var max_output_boxes_int = int(max_output_boxes_per_class[0])
        var iou_threshold_float = iou_threshold[0]
        var score_threshold_float = score_threshold[0]

        return non_max_suppression_shape_func(
            managed_tensor_slice_to_ndbuffer(boxes),
            managed_tensor_slice_to_ndbuffer(scores),
            max_output_boxes_int,
            iou_threshold_float,
            score_threshold_float,
        )


# ===----------------------------------------------------------------------===#
# Linalg kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.matmul")
struct Matmul:
    @compiler.enable_fusion_for("c")
    @staticmethod
    fn execute[
        transpose_b: Bool,
        packed_b: Bool,
        lambdas_have_fusion: Bool,
        synchronous: Bool,
        target: StringLiteral = "cpu",
    ](
        c: ManagedTensorSlice[rank=2],
        a: ManagedTensorSlice[rank=2],
        b: ManagedTensorSlice[rank=2],
        ctx: MojoCallContextPtr,
    ) raises:
        constrained[
            not (packed_b and transpose_b),
            (
                "transpose_b and b_packed cannot both be true because"
                " pre-packing transposes B"
            ),
        ]()

        alias transposed_a = False

        alias a_shape = compiler.specsof[a.type, a.rank]("a").shape
        alias b_shape = compiler.specsof[b.type, b.rank]("b").shape
        alias c_shape = compiler.specsof[c.type, c.rank]("c").shape

        var a_buffer = managed_tensor_slice_to_ndbuffer[static_shape=a_shape](a)
        var b_buffer = managed_tensor_slice_to_ndbuffer[static_shape=b_shape](b)
        var c_buffer = managed_tensor_slice_to_ndbuffer[static_shape=c_shape](c)

        alias out_lambda = compiler.specsof[c.type, c.rank]("c").out_lambda

        @parameter
        @always_inline
        fn output_fn[
            _type: DType, _width: Int, *, alignment: Int = 1
        ](coords: IndexList[2], val: SIMD[_type, _width]):
            c._fused_store[width=_width](
                coords,
                rebind[SIMD[c.type, _width]](val),
            )

        matmul[
            transposed_a,
            transpose_b,
            packed_b,
            OptionalReg[matmul_elementwise_epilogue_type](
                output_fn
            ) if lambdas_have_fusion else None,
            saturated_vnni=False,
            single_thread_blocking_override=synchronous,
            target=target,
        ](c_buffer, a_buffer, b_buffer, ctx)


@compiler.register("mo.batch_matmul")
struct BatchMatmul:
    @compiler.enable_fusion_for("c")
    @staticmethod
    fn execute[
        lambdas_have_fusion: Bool,
        rank: Int,
        transpose_b: Bool,
        synchronous: Bool,
        target: StringLiteral = "cpu",
    ](
        c: ManagedTensorSlice[rank=rank],
        a: ManagedTensorSlice[rank=rank],
        b: ManagedTensorSlice[rank=rank],
        ctx: MojoCallContextPtr,
    ) raises:
        alias transpose_a = False

        var a_buffer = managed_tensor_slice_to_ndbuffer(a)
        var b_buffer = managed_tensor_slice_to_ndbuffer(b)
        var c_buffer = managed_tensor_slice_to_ndbuffer(c)

        alias out_lambda = compiler.specsof[c.type, c.rank]("c").out_lambda

        @parameter
        @always_inline
        fn output_fn[
            _type: DType, _width: Int, _rank: Int, *, alignment: Int = 1
        ](coords: IndexList[_rank], val: SIMD[_type, _width]):
            c._fused_store[width=_width](
                rebind[IndexList[c.rank]](coords),
                rebind[SIMD[c.type, _width]](val),
            )

        batched_matmul[
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            elementwise_epilogue_fn = OptionalReg[
                batched_matmul_elementwise_epilogue_type
            ](output_fn) if lambdas_have_fusion else None,
            saturated_vnni=False,
            single_thread_blocking_override=synchronous,
            target=target,
        ](c_buffer, a_buffer, b_buffer, context=ctx)

    @staticmethod
    fn shape[
        rank: Int,
        a_type: DType,
        b_type: DType,
    ](
        a: ManagedTensorSlice[a_type, rank],
        b: ManagedTensorSlice[b_type, rank],
    ) raises -> IndexList[rank]:
        var a_buffer = managed_tensor_slice_to_ndbuffer(a)
        var b_buffer = managed_tensor_slice_to_ndbuffer(b)
        return batched_matmul_shape[single_thread_blocking_override=True](
            a_buffer, b_buffer
        )


@compiler.register("mo.linalg.solve")
struct LinalgSolve:
    @staticmethod
    fn execute[
        synchronous: Bool,
        type: DType,
    ](
        x: ManagedTensorSlice[type=type],
        a: ManagedTensorSlice[type=type],
        b: ManagedTensorSlice[type=type],
    ) raises:
        matrix_solve[single_thread_blocking_override=synchronous](
            managed_tensor_slice_to_ndbuffer(a),
            managed_tensor_slice_to_ndbuffer(b),
            managed_tensor_slice_to_ndbuffer(x),
        )

    @staticmethod
    fn shape[
        type: DType,
        rank: Int,
    ](
        a: ManagedTensorSlice[type=type, rank=rank],
        b: ManagedTensorSlice[type=type, rank=rank],
    ) raises -> IndexList[a.rank]:
        return matrix_solve_shape[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(a),
            managed_tensor_slice_to_ndbuffer(b),
        )


@compiler.register("mo.linalg.band_part")
struct LinalgBandPart:
    @compiler.enable_fusion_for("input")
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
        type: DType,
        int_type: DType,
        rank: Int,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        num_lower: ManagedTensorSlice[type=int_type, rank=1],
        num_upper: ManagedTensorSlice[type=int_type, rank=1],
        exclude: ManagedTensorSlice[rank=1],
        ctx: MojoCallContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[output.type, width]:
            return input._fused_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        var num_lower_buf = managed_tensor_slice_to_ndbuffer(num_lower)
        var num_upper_buf = managed_tensor_slice_to_ndbuffer(num_upper)
        var exclude_buf = managed_tensor_slice_to_ndbuffer(exclude)
        var output_buf = managed_tensor_slice_to_ndbuffer(output)

        matrix_band_part[
            input_0_fn=input_fn,
            simd_width = simdwidthof[type](),
            single_thread_blocking_override=synchronous,
            target=target,
        ](
            input.get_static_spec().shape,
            num_lower_buf,
            num_upper_buf,
            exclude_buf,
            output_buf,
            ctx,
        )


# ===----------------------------------------------------------------------===#
# Resize kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.resize.nearest")
struct ResizeNearest:
    @staticmethod
    fn execute[
        coordinate_transform_mode: Int,
        round_mode: Int,
        rank: Int,
        type: DType,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        size: ManagedTensorSlice[rank=1],
    ):
        resize_nearest_neighbor[coordinate_transform_mode, round_mode](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(output),
        )

    @staticmethod
    fn shape[
        rank: Int
    ](
        input: ManagedTensorSlice[rank=rank],
        size: ManagedTensorSlice[rank=1],
    ) -> IndexList[rank]:
        var shape = IndexList[rank]()
        for i in range(rank):
            shape[i] = int(size[i])

        return shape


@compiler.register("mo.resize.linear")
struct ResizeLinear:
    @staticmethod
    fn execute[
        coordinate_transform_mode: Int,
        antialias: Bool,
        rank: Int,
        type: DType,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        size: ManagedTensorSlice[rank=1],
    ):
        resize_linear[coordinate_transform_mode, antialias](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(output),
        )

    @staticmethod
    fn shape[
        rank: Int
    ](
        input: ManagedTensorSlice[rank=rank],
        size: ManagedTensorSlice[rank=1],
    ) -> IndexList[rank]:
        var shape = IndexList[rank]()
        for i in range(rank):
            shape[i] = int(size[i])

        return shape


# ===----------------------------------------------------------------------===#
# ROI align kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.roi_align")
struct ROIAlign:
    @staticmethod
    fn execute[
        aligned: Bool,
        mode: StringLiteral,
        type: DType,
    ](
        output: ManagedTensorSlice[type=type, rank=4],
        input: ManagedTensorSlice[type=type, rank=4],
        rois: ManagedTensorSlice[type=type, rank=2],
        output_height: ScalarTensor[DType.int64],
        output_width: ScalarTensor[DType.int64],
        spatial_scale: ScalarTensor,
        sampling_ratio: ScalarTensor,
    ):
        roi_align_nhwc[aligned, mode](
            managed_tensor_slice_to_ndbuffer(output),
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(rois),
            int(output_height[0]),
            int(output_width[0]),
            spatial_scale[0],
            sampling_ratio[0],
        )

    @staticmethod
    fn shape(
        input: ManagedTensorSlice[rank=4],
        rois: ManagedTensorSlice[rank=2],
        output_height: ScalarTensor[DType.int64],
        output_width: ScalarTensor[DType.int64],
        spatial_scale: ScalarTensor,
        sampling_ratio: ScalarTensor,
    ) -> IndexList[4]:
        var shape = IndexList[4]()
        # input shape is [N, H, W, C]
        # rois shape is [M, 5]
        # output shape is [M, output_height, output_width, C]
        shape[0] = rois.spec().shape[0]
        shape[1] = int(output_height[0])
        shape[2] = int(output_width[0])
        shape[3] = input.spec().shape[3]

        return shape


# ===----------------------------------------------------------------------===#
# Tile kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.tile")
struct Tile:
    @staticmethod
    fn execute[
        type: DType, rank: Int
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        repeats: ManagedTensorSlice,
    ) raises:
        tile(
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(repeats),
            managed_tensor_slice_to_ndbuffer(output),
        )

    @staticmethod
    fn shape(
        input: ManagedTensorSlice,
        repeats: ManagedTensorSlice[rank=1],
    ) raises -> IndexList[input.rank]:
        return tile_shape[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(repeats),
        )


# ===----------------------------------------------------------------------===#
# Random kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.random.normal")
struct RandomNormal:
    @staticmethod
    fn execute[
        mean_var_type: DType
    ](
        output: ManagedTensorSlice,
        shape: ManagedTensorSlice[rank=1],
        mean: ScalarTensor,
        variance: ScalarTensor,
        seed_value: ScalarTensor,
    ):
        seed(int(seed_value[0]))
        var num_elements = 1
        # TODO: Add __len__ support in ManagedTensorSlice.
        for i in range(shape.spec().shape[0]):
            num_elements *= int(shape[i])
        randn(
            output._ptr,
            num_elements,
            mean[0].cast[DType.float64](),
            variance[0].cast[DType.float64](),
        )

    @staticmethod
    fn shape[
        output_rank: Int
    ](shape: ManagedTensorSlice[rank=1]) -> IndexList[output_rank]:
        var unrolled_shape = IndexList[output_rank]()
        for i in range(output_rank):
            unrolled_shape[i] = int(shape[i])

        return unrolled_shape


@compiler.register("mo.static.random.normal")
struct StaticRandomNormal:
    @staticmethod
    fn execute[
        mean_var_type: DType
    ](
        output: ManagedTensorSlice,
        mean: ScalarTensor,
        variance: ScalarTensor,
        seed_value: ScalarTensor,
    ):
        seed(int(seed_value[0]))
        var num_elements = output.spec().shape.num_elements()
        randn(
            output._ptr,
            num_elements,
            mean[0].cast[DType.float64](),
            variance[0].cast[DType.float64](),
        )


@compiler.register("mo.softmax")
struct Softmax:
    @staticmethod
    fn execute[
        target: StringLiteral
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[output.type, output.rank],
        ctx: MojoCallContextPtr,
    ) raises:
        # shape should be the same between the two inputs
        alias static_shape = compiler.specsof[output.type, output.rank](
            "output"
        ).shape
        output_ndbuffer = managed_tensor_slice_to_ndbuffer[
            static_shape=static_shape
        ](output)

        # For adapting input fusion lambda required by call
        @parameter
        @always_inline
        fn input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[output.type, width]:
            @parameter
            if compiler.specsof[output.type, output.rank]("input").in_lambda:
                return input._fused_load[width=width](
                    rebind[IndexList[input.rank]](coords)
                )
            else:
                return input.load[width=width](
                    rebind[IndexList[input.rank]](coords)
                )

        softmax[
            output.type,
            simdwidthof[output.type](),
            output.rank,
            static_shape,
            input_fn,
            target,
        ](
            output.get_static_spec().shape,
            output_ndbuffer,
            output.rank - 1,
            context=ctx,
        )


@compiler.register("mo.logsoftmax")
struct LogSoftmax:
    @staticmethod
    fn execute[
        target: StringLiteral
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[output.type, output.rank],
    ) raises:
        # shape should be the same between the two inputs
        alias static_shape = compiler.specsof[output.type, output.rank](
            "output"
        ).shape
        output_ndbuffer = managed_tensor_slice_to_ndbuffer[
            static_shape=static_shape
        ](output)

        # For adapting input fusion lambda required by call
        @parameter
        @always_inline
        fn input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[output.type, width]:
            @parameter
            if compiler.specsof[output.type, output.rank]("input").in_lambda:
                return input._fused_load[width=width](
                    rebind[IndexList[input.rank]](coords)
                )
            else:
                return input.load[width=width](
                    rebind[IndexList[input.rank]](coords)
                )

        logsoftmax[
            output.type,
            simdwidthof[output.type](),
            output.rank,
            static_shape,
            input_fn,
        ](output.get_static_spec().shape, output_ndbuffer, output.rank - 1)


# ===----------------------------------------------------------------------===#
# Cumsum kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.cumsum")
struct CumSum:
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
        exclusive: Int,
        reverse: Int,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        axis: ScalarTensor,
        ctx: MojoCallContextPtr,
    ):
        var output_buf = managed_tensor_slice_to_ndbuffer(output)
        var input_buf = managed_tensor_slice_to_ndbuffer(input)
        var axis_val = axis._ptr.load(0)

        cumsum[rank, type, exclusive, reverse](
            output_buf, input_buf, int(normalize_neg_index(axis_val, rank))
        )


# ===----------------------------------------------------------------------===#
# Concat kernels
# ===----------------------------------------------------------------------===#


fn concat_shape_impl[
    type: DType, rank: Int, size: Int
](
    axis_buf: ManagedTensorSlice[rank=1],
    inputs: StaticTuple[ManagedTensorSlice[type, rank], size],
) raises -> IndexList[rank]:
    var axis_val = axis_buf._ptr.load(0)
    var axis = int(normalize_neg_index(axis_val, rank))
    if axis < 0 or rank <= axis:
        raise ("[concat] normalized axis must be within range [0, rank)")

    @parameter
    @always_inline
    fn shape_equal_ignore_axis(
        s1: IndexList[rank], s2: IndexList[rank]
    ) -> Bool:
        for i in range(rank):
            if i != axis and s1[i] != s2[i]:
                return False
        return True

    var concat_axis_dim_sum = 0
    for i in range(len(inputs)):
        concat_axis_dim_sum += inputs[i].dim_size(axis)
        if not shape_equal_ignore_axis(
            inputs[0].get_static_spec().shape,
            inputs[i].get_static_spec().shape,
        ):
            raise Error(
                "[concat] input shapes must match except at concat axis"
            )

    # compute and return the output shape
    var output_shape = inputs[0].get_static_spec().shape
    output_shape[axis] = concat_axis_dim_sum
    return output_shape


@compiler.register("mo.concat")
struct Concat:
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
        synchronous: Bool,
        target: StringLiteral,
        # TODO(GRA-1116): Support input fusion for concat
        # lambdas_have_fusion: Bool,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        axis: ManagedTensorSlice[rank=1],
        inputs: StaticTuple[ManagedTensorSlice[type, rank], *_],
        ctx: MojoCallContextPtr,
    ) raises:
        var output_buf = managed_tensor_slice_to_ndbuffer(output)
        var axis_val = axis._ptr.load(0)
        var input_bufs = StaticTuple[NDBuffer[type, rank], inputs.size]()

        @parameter
        for i in range(inputs.size):
            input_bufs[i] = managed_tensor_slice_to_ndbuffer(inputs[i])

        alias fusion = None
        concat[rank, type, synchronous, target, fusion](
            output_buf,
            int(normalize_neg_index(axis_val, rank)),
            input_bufs,
            context=ctx,
        )

    @staticmethod
    fn shape[
        type: DType,
        rank: Int,
        synchronous: Bool,
    ](
        axis_buf: ManagedTensorSlice[rank=1],
        inputs: StaticTuple[ManagedTensorSlice[type, rank], *_],
    ) raises -> IndexList[rank]:
        return concat_shape_impl(axis_buf, inputs)


# Helper method used by compiler to reconcile MGP list with type Mojo expects.
@mogg_register_override("to_managed_tensor_slice_list", 1)
@always_inline
fn to_managed_tensor_slice_list[
    type: DType, rank: Int
](
    raw_list_ptr: UnsafePointer[NoneType],
) -> InlinedFixedVector[
    ManagedTensorSlice[type, rank]
]:
    # Cast input list Unsafepointer
    var abi_list_ptr = raw_list_ptr.bitcast[ABI_List]()
    var elems_ptr = abi_list_ptr[].elements
    var abi_tensors_ptr = elems_ptr.bitcast[ABI_Tensor]()

    # Create output list
    var num_elements = abi_list_ptr[].num_elems
    var out_list = InlinedFixedVector[ManagedTensorSlice[type, rank]](
        num_elements
    )

    # Convert individual elements of the input list into NDBuffer, and
    # accumulate the results to output list.
    for i in range(num_elements):
        var abi_tensor_ptr = abi_tensors_ptr + i
        var dims = abi_tensor_ptr[].dims
        var data = abi_tensor_ptr[].data.bitcast[Scalar[type]]()
        var buffer = to_managed_tensor_slice[type, rank](data, dims)
        out_list.append(buffer)

    return InlinedFixedVector(out_list)


# NOTE: there are a lot of similarities between this and the shape func
# for mo.concat.
fn concat_from_list_shape_impl[
    type: DType, rank: Int
](
    axis_buf: ManagedTensorSlice[rank=1],
    inputs: InlinedFixedVector[ManagedTensorSlice[type, rank]],
) raises -> IndexList[rank]:
    var axis_val = axis_buf._ptr.load(0)
    var axis = int(normalize_neg_index(axis_val, rank))
    if axis < 0 or rank <= axis:
        raise ("[concat] normalized axis must be within range [0, rank)")

    @parameter
    @always_inline
    fn shape_equal_ignore_axis(
        s1: IndexList[rank], s2: IndexList[rank]
    ) -> Bool:
        for i in range(rank):
            if i != axis and s1[i] != s2[i]:
                return False
        return True

    var concat_axis_dim_sum = 0
    for i in range(len(inputs)):
        concat_axis_dim_sum += inputs[i].dim_size(axis)
        if not shape_equal_ignore_axis(
            inputs[0].get_static_spec().shape,
            inputs[i].get_static_spec().shape,
        ):
            raise Error(
                "[concat] input shapes must match except at concat axis"
            )

    # compute and return the output shape
    var output_shape = inputs[0].get_static_spec().shape
    output_shape[axis] = concat_axis_dim_sum
    return output_shape


@compiler.register("mo.concat_from_list")
struct ConcatFromList:
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
        synchronous: Bool,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        inputs: InlinedFixedVector[ManagedTensorSlice[type, rank]],
        axis: ManagedTensorSlice[rank=1],
        ctx: MojoCallContextPtr,
    ) raises:
        var output_buf = managed_tensor_slice_to_ndbuffer(output)

        # TODO: convert underlying kernel to accept lists of ManagedTensorSlice
        var input_as_ndbuffer = InlinedFixedVector[NDBuffer[type, rank]](
            inputs.current_size
        )
        for i in range(inputs.current_size):
            input_as_ndbuffer.append(
                managed_tensor_slice_to_ndbuffer(inputs[i])
            )

        var axis_val = axis[0]
        _concat_cpu[rank, type, None, synchronous](
            output_buf,
            int(normalize_neg_index(axis_val, rank)),
            input_as_ndbuffer,
        )

    @staticmethod
    fn shape[
        type: DType,
        rank: Int,
        synchronous: Bool,
    ](
        inputs: InlinedFixedVector[ManagedTensorSlice[type, rank]],
        axis_buf: ManagedTensorSlice[rank=1],
    ) raises -> IndexList[rank]:
        return concat_from_list_shape_impl(axis_buf, inputs)


# ===----------------------------------------------------------------------===#
# Split kernels
# ===----------------------------------------------------------------------===#


# TODO(GRA-1127): the shape function for split is special and there is special
# handling in the graph compiler to make things work.
@compiler.register("mo.split")
struct Split:
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
        synchronous: Bool,
        target: StringLiteral,
    ](
        output: StaticTuple[ManagedTensorSlice[type, rank], *_],
        input: ManagedTensorSlice[type, rank],
        split_sizes: ManagedTensorSlice[rank=1],
        axis: ManagedTensorSlice[rank=1],
        ctx: MojoCallContextPtr,
    ) raises:
        var input_buf = managed_tensor_slice_to_ndbuffer(input)
        var axis_val = axis._ptr.load(0)
        var output_bufs = StaticTuple[NDBuffer[type, rank], output.size]()

        @parameter
        for i in range(output.size):
            output_bufs[i] = managed_tensor_slice_to_ndbuffer(output[i])

        split[type, rank](
            input_buf, int(normalize_neg_index(axis_val, rank)), output_bufs
        )


# ===----------------------------------------------------------------------===#
# Convolution kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.conv")
struct Conv:
    @staticmethod
    fn execute[
        filter_packed: Bool,
        lambdas_have_fusion: Bool,
        static_strides: DimList,
        static_dilations: DimList,
        static_padding: DimList,
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[rank = output.rank],
        filter: ManagedTensorSlice,
        strides: ManagedTensorSlice,
        dilation: ManagedTensorSlice,
        paddings: ManagedTensorSlice,
        num_groups: ScalarTensor,
    ) raises:
        @parameter
        @always_inline
        fn output_fn[
            _type: DType, _rank: Int, _width: Int
        ](coords: IndexList[_rank], val: SIMD[_type, _width]):
            output.store[width=_width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.type, _width]](val),
            )

        alias input_static_shape = compiler.specsof[input.type, input.rank](
            "input"
        ).shape
        alias filter_static_shape = compiler.specsof[filter.type, filter.rank](
            "filter"
        ).shape
        alias output_static_shape = compiler.specsof[output.type, output.rank](
            "output"
        ).shape

        constrained[
            strides.type.is_integral() and dilation.type.is_integral(),
            "stride and dilation must have integral type",
        ]()

        if strides.size() != input.rank - 2:
            raise Error("$(input_rank-2) values expected in conv strides")

        if dilation.size() != input.rank - 2:
            raise Error("$(input_rank-2) values expected in conv dilation")

        if paddings.size() != 2 * (input.rank - 2):
            raise Error("$(2*(input_rank-2)) value expected in conv paddings")

        var stride_tuple = IndexList[input.rank - 2](0)
        var dilation_tuple = IndexList[input.rank - 2](0)

        @parameter
        for i in range(input.rank - 2):
            stride_tuple[i] = int(strides._ptr[i])
            dilation_tuple[i] = int(dilation._ptr[i])

        if dilation_tuple != IndexList[input.rank - 2](1):
            raise Error("Non-unit dilation is not supported yet.")

        var pad_d_tuple = IndexList[2](0)
        var pad_h_tuple = IndexList[2](0)
        var pad_w_tuple = IndexList[2](0)

        @parameter
        if input.rank == 3:
            pad_w_tuple = Index(paddings._ptr[0], paddings._ptr[1])
        elif input.rank == 4:
            pad_h_tuple = Index(paddings._ptr[0], paddings._ptr[1])
            pad_w_tuple = Index(paddings._ptr[2], paddings._ptr[3])
        elif input.rank == 5:
            pad_d_tuple = Index(paddings._ptr[0], paddings._ptr[1])
            pad_h_tuple = Index(paddings._ptr[2], paddings._ptr[3])
            pad_w_tuple = Index(paddings._ptr[4], paddings._ptr[5])

        alias conv_attr = ConvInfoStatic[input.rank - 2](
            static_padding,
            static_strides,
            static_dilations,
            input_static_shape.at[input.rank - 1](),  # input C, NHWC
            filter_static_shape.at[
                filter.rank - 2
            ](),  # filter C, RSCF or FRSCf
        )

        var input_buf = managed_tensor_slice_to_ndbuffer[
            static_shape=input_static_shape
        ](input)
        var filter_buf = managed_tensor_slice_to_ndbuffer[
            static_shape=filter_static_shape
        ](filter)
        var output_buf = managed_tensor_slice_to_ndbuffer[
            static_shape=output_static_shape
        ](output)

        conv_nhwc_direct[
            input.rank,
            filter.rank,
            input_static_shape,  # input shape
            filter_static_shape,  # filter shape
            output_static_shape,  # output shape
            input.type,
            filter.type,
            output.type,
            filter_packed,
            conv_attr,
            lambdas_have_fusion,
            output_fn,
        ](
            input_buf,
            filter_buf,
            output_buf,
            stride_tuple,
            dilation_tuple,
            pad_d_tuple,
            pad_h_tuple,
            pad_w_tuple,
            int(num_groups._ptr[0]),
        )

    @staticmethod
    fn shape[
        type: DType
    ](
        input: ManagedTensorSlice,
        filter: ManagedTensorSlice,
        strides: ManagedTensorSlice[rank=1],
        dilations: ManagedTensorSlice[rank=1],
        paddings: ManagedTensorSlice[rank=1],
        num_groups: ManagedTensorSlice[rank=1],
    ) raises -> IndexList[input.rank]:
        return conv_shape[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(filter),
            managed_tensor_slice_to_ndbuffer(strides),
            managed_tensor_slice_to_ndbuffer(dilations),
            managed_tensor_slice_to_ndbuffer(paddings),
            managed_tensor_slice_to_ndbuffer(num_groups),
        )


@compiler.register("mo.conv_transpose")
struct ConvTranspose:
    @staticmethod
    fn execute[
        filter_packed: Bool,
        lambdas_have_fusion: Bool,
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[rank = output.rank],
        filter: ManagedTensorSlice,
        strides: ManagedTensorSlice[rank=1],
        dilation: ManagedTensorSlice[rank=1],
        paddings: ManagedTensorSlice[rank=1],
        output_paddings: ManagedTensorSlice[rank=1],
    ) raises:
        constrained[
            strides.type.is_integral()
            and dilation.type.is_integral()
            and output_paddings.type.is_integral()
        ]()

        alias input_static_shape = compiler.specsof[input.type, input.rank](
            "input"
        ).shape
        alias filter_static_shape = compiler.specsof[filter.type, filter.rank](
            "filter"
        ).shape
        alias output_static_shape = compiler.specsof[output.type, output.rank](
            "output"
        ).shape

        if strides.size() != input.rank - 2:
            raise Error(
                "$(input_rank-2) values expected in convTranspose stride"
            )

        if dilation.size() != input.rank - 2:
            raise Error(
                "$(input_rank-2) values expected in convTranspose dilation"
            )

        if output_paddings.size() != input.rank - 2:
            raise Error(
                "$(input_rank-2) values expected in convTranspose output"
                " paddings"
            )

        if paddings.size() != 2 * (input.rank - 2):
            raise Error(
                "$(2*(input_rank-2)) value expected in convTranspose paddings"
            )

        var stride_tuple = IndexList[input.rank - 2](0)
        var dilation_tuple = IndexList[input.rank - 2](0)

        @parameter
        for i in range(input.rank - 2):
            stride_tuple[i] = int(strides._ptr[i])
            dilation_tuple[i] = int(dilation._ptr[i])

        var pad_d = IndexList[2](0)
        var pad_h = IndexList[2](0)
        var pad_w = IndexList[2](0)

        @parameter
        if input.rank == 3:
            pad_w = Index(paddings[0], paddings[1])
        elif input.rank == 4:
            pad_h = Index(paddings[0], paddings[1])
            pad_w = Index(paddings[2], paddings[3])
        elif input.rank == 5:
            pad_d = Index(paddings[0], paddings[1])
            pad_h = Index(paddings[2], paddings[3])
            pad_w = Index(paddings[4], paddings[5])

        @parameter
        @always_inline
        fn output_fn[
            _type: DType, _rank: Int, _width: Int
        ](coords: IndexList[_rank], val: SIMD[_type, _width]):
            output._fused_store[width=_width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.type, _width]](val),
            )

        var input_buf = managed_tensor_slice_to_ndbuffer[
            static_shape=input_static_shape
        ](input)
        var filter_buf = managed_tensor_slice_to_ndbuffer[
            static_shape=filter_static_shape
        ](filter)
        var output_buf = managed_tensor_slice_to_ndbuffer[
            static_shape=output_static_shape
        ](output)

        conv_transposed[
            input.rank,
            filter.rank,
            input_static_shape,  # Input shape.
            filter_static_shape,  # Filter shape.
            output_static_shape,  # Output shape.
            input.type,
            filter.type,  # Filter type.
            output.type,  # Output type.
            filter_packed,
            lambdas_have_fusion,
            output_fn,
        ](
            output_buf,
            input_buf,
            filter_buf,
            stride_tuple,
            dilation_tuple,
            pad_d,
            pad_h,
            pad_w,
        )

    @staticmethod
    fn shape[
        type: DType
    ](
        input: ManagedTensorSlice[type],
        filter: ManagedTensorSlice[type],
        strides: ManagedTensorSlice[rank=1],
        dilations: ManagedTensorSlice[rank=1],
        paddings: ManagedTensorSlice[rank=1],
        output_paddings: ManagedTensorSlice[rank=1],
    ) raises -> IndexList[input.rank]:
        return conv_transpose_shape[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(filter),
            managed_tensor_slice_to_ndbuffer(strides),
            managed_tensor_slice_to_ndbuffer(dilations),
            managed_tensor_slice_to_ndbuffer(paddings),
            managed_tensor_slice_to_ndbuffer(output_paddings),
        )


# ===----------------------------------------------------------------------===#
# Attention kernels
# ===----------------------------------------------------------------------===#


@compiler.register("masked_flash_attention_gpu")
struct MaskedFlashAttentionGPU:
    @staticmethod
    fn execute[
        target: StringLiteral, rank: Int
    ](
        output: ManagedTensorSlice[rank=rank],
        q: ManagedTensorSlice[rank=rank],
        k: ManagedTensorSlice[rank=rank],
        v: ManagedTensorSlice[rank=rank],
        mask: ManagedTensorSlice,
        scale: ScalarTensor[type = DType.float32],
        ctx: MojoCallContextPtr,
    ) raises:
        """`masked_flash_attention_gpu` is a hand-fused operator which does
        something analogous to the following list of operations.

        **Step 0:
        Transpose:
        query_processed = transpose(query) # BSHD --> BHSD
        key_processed = transpose(key)     # BSHD --> BHDS
        value_processed = transpose(value) # BSHD --> BHSD

        **Step 1:
        attentionMatrix = query_processed @ key_processed

        **Step 2:
        norm = broadcast_to(normScalarTensor, shape_of(attentionMatrix))

        **Step 3:
        # Normalize and apply masking
        attentionMatrixNorm = attentionMatrix * scale

        # Note attention_mask is HSS and auto-broadcasts
        attentionMatrixNormMasked = attentionMatrixNorm + attention_mask

        **Step 4:
        # Apply softmax and reproject result
        attentionMatrixSoftMax = softmax(attentionMatrixNormMasked)
        answer = attentionMatrixSoftMax @ value_processed
        answer = transpose(answer) # BHSD --> BSHD

        Compared to the CPU patterns the notable differences are:
        1. The mask is rank 3 and is of shape BSS
        2. The transposes are part of the kernel itself

        Finally, this pattern supports grouped attention patterns. That is if we
        have G groups, then let h = H / G. Key and value are allowed to be BShD
        in these scenarios. Both key and value must be BShD if one is. If this is
        true the following is equivalently run before Step 0:

        ** Step -1:
        key = concat(key, ...) # concat BShD --> BSHD
        value = concat(value, ...) # concat BShD --> BSHD

        The underlying fusion follows ideas taken from the 2022 FlashAttention paper
        by Tri Dao et al.
        """
        constrained["cuda" in target, "only valid on CUDA GPUs"]()

        flash_attention[
            add_attn_mask=True, target=target, use_tensor_core=True
        ](
            managed_tensor_slice_to_ndbuffer(output),
            managed_tensor_slice_to_ndbuffer(q),
            managed_tensor_slice_to_ndbuffer(k),
            managed_tensor_slice_to_ndbuffer(v),
            managed_tensor_slice_to_ndbuffer(mask),
            scale[0],
            context=ctx,
        )


@compiler.register("no_mask_fused_attention_cpu")
struct NoMaskFusedAttentionCPU:
    @staticmethod
    fn execute[
        rank: Int
    ](
        output: ManagedTensorSlice[rank=rank],
        q: ManagedTensorSlice[rank=rank],
        k: ManagedTensorSlice[rank=rank],
        v: ManagedTensorSlice[rank=rank],
        scale: ScalarTensor[type = DType.float32],
    ) raises:
        alias q_shape = compiler.specsof[q.type, q.rank]("q").shape
        alias k_shape = compiler.specsof[k.type, q.rank]("k").shape
        alias v_shape = compiler.specsof[v.type, q.rank]("v").shape
        alias output_shape = compiler.specsof[output.type, output.rank](
            "output"
        ).shape

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

        alias mask_shape = DimList()
        alias mask_type = DType.float32
        var mask = NDBuffer[mask_type, rank, mask_shape]()
        var scale_f32 = scale[0].cast[DType.float32]()
        var causal_mask: Float32 = 0

        fused_attention[
            rank,
            q_shape,
            k_shape,
            v_shape,
            mask_shape,
            output_shape,
            q.type,
            k.type,
            v.type,
            mask.type,
            output.type,
            transpose_k=False,
            add_attn_mask=False,
            add_causal_mask=False,
        ](
            managed_tensor_slice_to_ndbuffer[static_shape=output_shape](output),
            managed_tensor_slice_to_ndbuffer[static_shape=q_shape](q),
            managed_tensor_slice_to_ndbuffer[static_shape=k_shape](k),
            managed_tensor_slice_to_ndbuffer[static_shape=v_shape](v),
            mask,
            scale_f32,
            causal_mask,
        )


@compiler.register("with_mask_fused_attention_cpu")
struct WithMAskFusedAttentionCPU:
    @staticmethod
    fn execute[
        rank: Int
    ](
        output: ManagedTensorSlice[rank=rank],
        q: ManagedTensorSlice[rank=rank],
        k: ManagedTensorSlice[rank=rank],
        v: ManagedTensorSlice[rank=rank],
        mask: ManagedTensorSlice[rank=rank],
        scale: ScalarTensor[type = DType.float32],
    ) raises:
        alias q_shape = compiler.specsof[q.type, q.rank]("q").shape
        alias k_shape = compiler.specsof[k.type, q.rank]("k").shape
        alias v_shape = compiler.specsof[v.type, q.rank]("v").shape
        alias mask_shape = compiler.specsof[mask.type, mask.rank]("mask").shape
        alias output_shape = compiler.specsof[output.type, output.rank](
            "output"
        ).shape

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
        # TODO: Unimplemented and not used
        var scale_f32 = scale[0].cast[DType.float32]()
        var causal_mask: Float32 = 0

        fused_attention[
            rank,
            q_shape,
            k_shape,
            v_shape,
            mask_shape,
            output_shape,
            q.type,
            k.type,
            v.type,
            mask.type,
            output.type,
            transpose_k=False,
            add_attn_mask=True,
            add_causal_mask=False,
        ](
            managed_tensor_slice_to_ndbuffer[static_shape=output_shape](output),
            managed_tensor_slice_to_ndbuffer[static_shape=q_shape](q),
            managed_tensor_slice_to_ndbuffer[static_shape=k_shape](k),
            managed_tensor_slice_to_ndbuffer[static_shape=v_shape](v),
            managed_tensor_slice_to_ndbuffer[static_shape=mask_shape](mask),
            scale_f32,
            causal_mask,
        )


@compiler.register("no_mask_flash_attention_cpu")
struct NoMaskFlashAttentionCPU:
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        q: ManagedTensorSlice[type=type, rank=rank],
        k: ManagedTensorSlice[type=type, rank=rank],
        v: ManagedTensorSlice[type=type, rank=rank],
        scale: ScalarTensor[type = DType.float32],
    ) raises:
        alias output_shape = compiler.specsof[output.type, output.rank](
            "output"
        ).shape

        @parameter
        @always_inline
        fn k_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[k.type, width]:
            return k.load[width=width](rebind[IndexList[k.rank]](coords))

        @parameter
        @always_inline
        fn v_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[v.type, width]:
            return v.load[width=width](rebind[IndexList[v.rank]](coords))

        @parameter
        @always_inline
        fn mask_fn[
            width: Int, _rank: Int
        ](idx: IndexList[_rank]) -> SIMD[type, width]:
            return SIMD[type, width](0)

        nn_flash_attention[k_input_fn, v_input_fn, mask_fn,](
            managed_tensor_slice_to_ndbuffer(q),
            k.get_static_spec().shape,
            v.get_static_spec().shape,
            IndexList[0](),
            managed_tensor_slice_to_ndbuffer[static_shape=output_shape](output),
            scale[0].cast[DType.float32](),
        )


@compiler.register("with_mask_flash_attention_split_kv_cpu")
struct WithMaskFlashAttentionSplitKVCPU:
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        q: ManagedTensorSlice[type=type, rank=rank],
        k: ManagedTensorSlice[type=type, rank=rank],
        v: ManagedTensorSlice[type=type, rank=rank],
        k_cache: ManagedTensorSlice[type=type, rank = rank + 1],
        v_cache: ManagedTensorSlice[type=type, rank = rank + 1],
        mask: ManagedTensorSlice[type=type],
        scale: ScalarTensor[type = DType.float32],
    ) raises:
        alias output_shape = compiler.specsof[output.type, output.rank](
            "output"
        ).shape

        @parameter
        @always_inline
        fn k_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[k.type, width]:
            return k.load[width=width](rebind[IndexList[k.rank]](coords))

        @parameter
        @always_inline
        fn v_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[v.type, width]:
            return v.load[width=width](rebind[IndexList[v.rank]](coords))

        @parameter
        @always_inline
        fn k_cache_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[k_cache.type, width]:
            return k_cache.load[width=width](
                rebind[IndexList[k_cache.rank]](coords)
            )

        @parameter
        @always_inline
        fn v_cache_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[v_cache.type, width]:
            return v_cache.load[width=width](
                rebind[IndexList[v_cache.rank]](coords)
            )

        @parameter
        @always_inline
        fn mask_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[mask.type, width]:
            return mask.load[width=width](rebind[IndexList[mask.rank]](coords))

        flash_attention_split_kv[
            k_input_fn,
            v_input_fn,
            k_cache_input_fn,
            v_cache_input_fn,
            mask_input_fn,
        ](
            managed_tensor_slice_to_ndbuffer(q),
            k.get_static_spec().shape,
            v.get_static_spec().shape,
            k_cache.get_static_spec().shape,
            v_cache.get_static_spec().shape,
            mask.get_static_spec().shape,
            managed_tensor_slice_to_ndbuffer[static_shape=output_shape](output),
            scale[0].cast[DType.float32](),
        )

    @staticmethod
    fn shape(q: ManagedTensorSlice) -> IndexList[q.rank]:
        return q.get_static_spec().shape


@compiler.register("with_mask_flash_attention_cpu")
struct WithMaskFlashAttentionCPU:
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        q: ManagedTensorSlice[type=type, rank=rank],
        k: ManagedTensorSlice[type=type, rank=rank],
        v: ManagedTensorSlice[type=type, rank=rank],
        mask: ManagedTensorSlice[type=type],
        scale: ScalarTensor[type = DType.float32],
    ) raises:
        alias output_shape = compiler.specsof[output.type, output.rank](
            "output"
        ).shape

        @parameter
        @always_inline
        fn k_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[k.type, width]:
            return k.load[width=width](rebind[IndexList[k.rank]](coords))

        @parameter
        @always_inline
        fn v_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[v.type, width]:
            return v.load[width=width](rebind[IndexList[v.rank]](coords))

        @parameter
        @always_inline
        fn mask_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[mask.type, width]:
            return mask.load[width=width](rebind[IndexList[mask.rank]](coords))

        nn_flash_attention[k_input_fn, v_input_fn, mask_input_fn,](
            managed_tensor_slice_to_ndbuffer(q),
            k.get_static_spec().shape,
            v.get_static_spec().shape,
            mask.get_static_spec().shape,
            managed_tensor_slice_to_ndbuffer[static_shape=output_shape](output),
            scale[0].cast[DType.float32](),
        )


# ===----------------------------------------------------------------------===#
# Quantization for CPU
# ===----------------------------------------------------------------------===#

######
# Q4_0
######


@compiler.register("ggml_q4_0_dequantize")
struct GGMLQ40Dequantize:
    @staticmethod
    @always_inline
    fn execute(
        output: ManagedTensorSlice[DType.float32, 2],
        input: ManagedTensorSlice[DType.uint8, 2],
    ) raises:
        with Trace[TraceLevel.OP, target="cpu"]("ggml_q4_0_dequantize"):
            Q4sym[group_size=32].dequantize_and_write_to_tensor(
                managed_tensor_slice_to_ndbuffer(input),
                managed_tensor_slice_to_ndbuffer(output),
                output.get_static_spec().shape,
            )

    @staticmethod
    @always_inline
    fn shape(input: ManagedTensorSlice[DType.uint8, 2]) -> IndexList[2]:
        alias block_nbytes = sizeof[Q4sym[group_size=32]]()
        alias quants_per_block = 32
        var num_block_per_batch = (
            input.size() // input.dim_size(0)
        ) // block_nbytes
        return (input.dim_size(0), quants_per_block * num_block_per_batch)


@compiler.register("vroom_q4_0_matmul")
struct VroomQ40Matmul:
    @staticmethod
    @always_inline
    fn execute(
        c: ManagedTensorSlice[DType.float32, 2],
        a: ManagedTensorSlice[DType.float32, 2],
        b: ManagedTensorSlice[DType.uint8, 2],
    ) raises:
        with Trace[TraceLevel.OP, target="cpu"]("vroom_q4_0_matmul"):
            matmul_qint4[32](
                managed_tensor_slice_to_ndbuffer(a),
                managed_tensor_slice_to_ndbuffer(b),
                managed_tensor_slice_to_ndbuffer(c),
            )

    @staticmethod
    @always_inline
    fn shape(
        a: ManagedTensorSlice[DType.float32, 2],
        b: ManagedTensorSlice[DType.uint8, 2],
    ) -> IndexList[2]:
        return IndexList[2](a.dim_size(0), b.dim_size(0))


@compiler.register("vroom_q4_0_repack_weights")
struct VroomQ40RepackWeights:
    @staticmethod
    @always_inline
    fn execute(
        b_packed: ManagedTensorSlice[DType.uint8, 2],
        b: ManagedTensorSlice[DType.uint8, 2],
    ) raises:
        with Trace[TraceLevel.OP, target="cpu"]("vroom_q4_0_matmul"):
            matmul_qint4_pack_b[32](
                managed_tensor_slice_to_ndbuffer(b),
                managed_tensor_slice_to_ndbuffer(b_packed),
            )

    @staticmethod
    @always_inline
    fn shape(
        a: ManagedTensorSlice[DType.float32, 2],
        b: ManagedTensorSlice[DType.uint8, 2],
    ) -> IndexList[2]:
        return b.get_static_spec().shape


######
# Q4_K
######


@compiler.register("ggml_q4_k_dequantize")
struct GGMLQ4KDequantize:
    @staticmethod
    @always_inline
    fn execute(
        output: ManagedTensorSlice[DType.float32, 2],
        input: ManagedTensorSlice[DType.uint8, 2],
    ) raises:
        with Trace[TraceLevel.OP, target="cpu"]("ggml_q4_k_dequantize"):
            q4_k_dequantize_impl(
                managed_tensor_slice_to_ndbuffer(input),
                managed_tensor_slice_to_ndbuffer(output),
            )

    @staticmethod
    @always_inline
    fn shape(input: ManagedTensorSlice[DType.uint8, 2]) -> IndexList[2]:
        alias block_nbytes = sizeof[block_Q4_K]()
        alias elements_per_block = block_QK_K.quantized_k

        var num_block_per_batch = (
            input.size() // input.dim_size(0)
        ) // block_nbytes

        return (input.dim_size(0), elements_per_block * num_block_per_batch)


@compiler.register("vroom_q4_k_matmul")
struct VroomQ4KMatmul:
    @staticmethod
    @always_inline
    fn execute(
        c: ManagedTensorSlice[DType.float32, 2],
        a: ManagedTensorSlice[DType.float32, 2],
        b: ManagedTensorSlice[DType.uint8, 2],
    ) raises:
        with Trace[TraceLevel.OP, target="cpu"]("vroom_q4_k_matmul"):
            matmul_Q4_K(
                managed_tensor_slice_to_ndbuffer(a),
                managed_tensor_slice_to_ndbuffer(b),
                managed_tensor_slice_to_ndbuffer(c),
            )

    @staticmethod
    @always_inline
    fn shape(
        a: ManagedTensorSlice[DType.float32, 2],
        b: ManagedTensorSlice[DType.uint8, 2],
    ) -> IndexList[2]:
        return IndexList[2](a.dim_size(0), b.dim_size(0))


@compiler.register("vroom_q4_k_repack_weights")
struct VroomQ4KRepackWeights:
    @staticmethod
    @always_inline
    fn execute(
        b_packed: ManagedTensorSlice[DType.uint8, 2],
        b: ManagedTensorSlice[DType.uint8, 2],
    ) raises:
        with Trace[TraceLevel.OP, target="cpu"]("vroom_q4_k_repack_weights"):
            matmul_Q4_K_pack_b(
                managed_tensor_slice_to_ndbuffer(b),
                managed_tensor_slice_to_ndbuffer(b_packed),
            )

    @staticmethod
    @always_inline
    fn shape(
        a: ManagedTensorSlice[DType.float32, 2],
        b: ManagedTensorSlice[DType.uint8, 2],
    ) -> IndexList[2]:
        return b.get_static_spec().shape


######
# Q6_K
######


@compiler.register("ggml_q6_k_dequantize")
struct GGMLQ6KDequantize:
    @staticmethod
    @always_inline
    fn execute(
        output: ManagedTensorSlice[DType.float32, 2],
        input: ManagedTensorSlice[DType.uint8, 2],
    ) raises:
        with Trace[TraceLevel.OP, target="cpu"]("ggml_q6_k_dequantize"):
            q6_k_dequantize_impl(
                managed_tensor_slice_to_ndbuffer(input),
                managed_tensor_slice_to_ndbuffer(output),
                output.get_static_spec().shape,
            )

    @staticmethod
    @always_inline
    fn shape(input: ManagedTensorSlice[DType.uint8, 2]) -> IndexList[2]:
        alias block_nbytes = sizeof[block_Q6_K]()
        alias elements_per_block = block_QK_K.quantized_k

        var num_block_per_batch = (
            input.size() // input.dim_size(0)
        ) // block_nbytes

        return (input.dim_size(0), elements_per_block * num_block_per_batch)


@compiler.register("vroom_q6_k_matmul")
struct VroomQ6KMatmul:
    @staticmethod
    @always_inline
    fn execute(
        c: ManagedTensorSlice[DType.float32, 2],
        a: ManagedTensorSlice[DType.float32, 2],
        b: ManagedTensorSlice[DType.uint8, 2],
    ) raises:
        with Trace[TraceLevel.OP, target="cpu"]("vroom_q6_k_matmul"):
            matmul_Q6_K(
                managed_tensor_slice_to_ndbuffer(a),
                managed_tensor_slice_to_ndbuffer(b),
                managed_tensor_slice_to_ndbuffer(c),
            )

    @staticmethod
    @always_inline
    fn shape(
        a: ManagedTensorSlice[DType.float32, 2],
        b: ManagedTensorSlice[DType.uint8, 2],
    ) -> IndexList[2]:
        return IndexList[2](a.dim_size(0), b.dim_size(0))


@compiler.register("vroom_q6_k_repack_weights")
struct VroomQ6KRepackWeights:
    @staticmethod
    @always_inline
    fn execute(
        b_packed: ManagedTensorSlice[DType.uint8, 2],
        b: ManagedTensorSlice[DType.uint8, 2],
    ) raises:
        with Trace[TraceLevel.OP, target="cpu"]("vroom_q6_k_repack_weights"):
            matmul_Q6_K_pack_b(
                managed_tensor_slice_to_ndbuffer(b),
                managed_tensor_slice_to_ndbuffer(b_packed),
            )

    @staticmethod
    @always_inline
    fn shape(
        a: ManagedTensorSlice[DType.float32, 2],
        b: ManagedTensorSlice[DType.uint8, 2],
    ) -> IndexList[2]:
        return b.get_static_spec().shape
