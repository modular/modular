# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# ===-----------------------------------------------------------------------===#
# General imports
# ===-----------------------------------------------------------------------===#

from collections import Optional, OptionalReg
from collections.vector import InlinedFixedVector
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
from random import randn, seed
from sys import external_call, llvm_intrinsic
from sys.info import simdwidthof, sizeof

import compiler_internal as compiler

# ===-----------------------------------------------------------------------===#
# Kernel imports
# ===-----------------------------------------------------------------------===#
from algorithm import max as reduce_max
from algorithm import mean
from algorithm import min as reduce_min
from algorithm import product, sum
from algorithm.reduction import _reduce_generator, _reduce_generator_cpu
from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from builtin.simd import _pow
from compiler_internal import StaticTensorSpec
from gpu.all_reduce import MAX_GPUS, Signal, all_reduce
from gpu.host import DeviceBuffer, DeviceContext
from gpu.host.info import is_cpu, is_gpu, is_valid_target
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
    KVCollectionT,
    PagedKVCacheCollection,
)
from linalg.bmm import batched_matmul, batched_matmul_shape
from linalg.bmm import (
    elementwise_epilogue_type as batched_matmul_elementwise_epilogue_type,
)
from linalg.dual_gemm import swishGLU
from linalg.matmul import matmul
from linalg.matrix_band_part import matrix_band_part
from linalg.matrix_solve import matrix_solve, matrix_solve_shape
from linalg.packing import _pack_b_ndbuffer_impl, pack_matmul_b_shape_func
from linalg.utils import (
    elementwise_epilogue_type as matmul_elementwise_epilogue_type,
)
from memory import AddressSpace, UnsafePointer
from nn import arg_nonzero
from nn.activations import gelu, relu
from nn.arange import arange, arange_shape
from nn.argmaxmin import argmax, argmin
from nn.argmaxmin_gpu import argmax_gpu, argmin_gpu
from nn.concat import _concat_cpu, concat, fused_concat
from nn.conv import ConvInfoStatic, conv_gpu, conv_nhwc_direct, conv_shape
from nn.conv import pack_filter as _pack_conv_filter
from nn.conv import pack_filter_shape as pack_filter_shape_conv
from nn.conv_transpose import conv_transpose_shape, conv_transposed
from nn.conv_transpose import pack_filter as _pack_conv_transpose_filter
from nn.conv_transpose import (
    pack_filter_shape as pack_filter_shape_conv_transpose,
)
from nn.cumsum import cumsum
from nn.flash_attention import flash_attention as nn_flash_attention
from nn.flash_attention import flash_attention_split_kv
from nn.fused_qk_rope import fused_qk_rope_ragged
from nn.gather_scatter import (
    Axis,
    gather,
    gather_nd,
    gather_nd_shape,
    gather_reduce,
    gather_shape,
    normalize_neg_index,
    scatter_elements,
    scatter_elements_shape,
    scatter_nd,
    scatter_nd_generator,
    scatter_nd_shape,
)
from nn.index_tensor import index_tensor
from nn.kv_cache import (
    generic_flash_attention_kv_cache_causal_alibi_mask_continuous_batch,
    generic_flash_attention_kv_cache_causal_mask_continuous_batch,
    generic_flash_attention_kv_cache_continuous_batch,
    generic_fused_qk_rope_bshd_continuous_batch,
    generic_fused_qkv_matmul_kv_cache_bshd_continuous_batch,
    generic_get_continuous_cache,
    generic_get_paged_cache,
    print_kv_cache_cont_batch_generic_cpu,
    print_kv_cache_cont_batch_generic_gpu,
    print_kv_cache_paged_generic_cpu,
    print_kv_cache_paged_generic_gpu,
    rms_norm_kv_cache_ragged_continuous_batching,
)
from nn.kv_cache_ragged import (
    generic_cross_attention_kv_cache_null_mask_cont_batch_ragged,
    generic_flash_attention_kv_cache_alibi_mask_cont_batch_ragged,
    generic_flash_attention_kv_cache_causal_mask_cont_batch_ragged,
    generic_flash_attention_kv_cache_causal_mask_paged_ragged,
    generic_flash_attention_kv_cache_null_mask_cont_batch_ragged,
    generic_fused_qk_rope_bshd_continous_batch_ragged,
    generic_fused_qk_rope_bshd_paged_ragged,
    generic_fused_qkv_matmul_kv_cache_cont_batch_ragged,
    generic_fused_qkv_matmul_kv_cache_paged_ragged,
    generic_fused_qkv_matmul_kv_cache_paged_ragged_bias,
    kv_matmul_ragged_continuous_batching,
)
from nn.mha import flash_attention
from nn.nms import non_max_suppression, non_max_suppression_shape_func
from nn.normalization import layer_norm, rms_norm
from nn.pad import pad_constant, pad_reflect, pad_repeat, pad_shape
from nn.pool import avg_pool, max_pool, pool_shape, pool_shape_ceil
from nn.reshape import reshape, reshape_shape
from nn.resize import resize_linear, resize_nearest_neighbor
from nn.roi_align import roi_align_nhwc
from nn.slice import (
    copy_to_slice,
    slice_as_view,
    slice_dim_as_view,
    slice_shape,
)
from nn.softmax import logsoftmax, softmax
from nn.split import split
from nn.tile import tile, tile_shape
from nn.topk import top_k
from nn.topk import top_k_fused_sampling_cpu as _topk_fused_sampling_cpu
from nn.topk import top_k_shape_impl
from nn.topk import topk_fused_sampling_gpu as _topk_fused_sampling_gpu
from quantization import (
    Q4sym,
    block_Q4_K,
    block_Q6_K,
    block_QK_K,
    q4_k_dequantize_impl,
    q6_k_dequantize_impl,
)
from quantization.qmatmul import matmul_qint4, matmul_qint4_pack_b
from quantization.qmatmul_gpu import (
    gpu_qint4_repack_GPTQ,
    gpu_qint4_repack_Q4_0,
    matmul_gpu_qint4,
)
from quantization.qmatmul_k import (
    matmul_Q4_K,
    matmul_Q4_K_pack_b,
    matmul_Q6_K,
    matmul_Q6_K_pack_b,
)
from register import register_internal
from runtime.asyncrt import DeviceContextPtr, MojoCallContextPtr
from runtime.tracing import Trace, TraceLevel, trace_arg
from tensor_internal import (
    DynamicTensor,
    ManagedTensorSlice,
    IODynamicTensor,
    VariadicTensors,
    _input_fusion_hook_impl,
    _output_fusion_hook_impl,
    foreach,
    simd_load_from_managed_tensor_slice,
    simd_store_into_managed_tensor_slice,
    view_copy_impl,
    IOSpec,
    IOUnknown,
)

from utils import IndexList, StaticTuple
from utils.index import Index
from utils.loop import unroll
from utils.numerics import isinf, isnan
from utils.static_tuple import _create_array, _set_array_elem

# ===-----------------------------------------------------------------------===#
# Nop functions to expose different types to the compiler.
# ===-----------------------------------------------------------------------===#


@register_internal("bfloat16")
fn DTypeBFloat16TypeDef(ty: DType.type) -> DType.type:
    return DType.bfloat16.value


@register_internal("float16")
fn DTypeFloat16TypeDef(ty: DType.type) -> DType.type:
    return DType.float16.value


@register_internal("float32")
fn DTypeFloat32TypeDef(ty: DType.type) -> DType.type:
    return DType.float32.value


@register_internal("float64")
fn DTypeFloat64TypeDef(ty: DType.type) -> DType.type:
    return DType.float64.value


@register_internal("int8")
fn DTypeInt8TypeDef(ty: DType.type) -> DType.type:
    return DType.int8.value


@register_internal("int16")
fn DTypeInt16TypeDef(ty: DType.type) -> DType.type:
    return DType.int16.value


@register_internal("int32")
fn DTypeInt32TypeDef(ty: DType.type) -> DType.type:
    return DType.int32.value


@register_internal("uint32")
fn DTypeUInt32TypeDef(ty: DType.type) -> DType.type:
    return DType.uint32.value


@register_internal("uint64")
fn DTypeUInt64TypeDef(ty: DType.type) -> DType.type:
    return DType.uint64.value


@register_internal("int64")
fn DTypeInt64TypeDef(ty: DType.type) -> DType.type:
    return DType.int64.value


@register_internal("uint8")
fn DTypeUInt8TypeDef(ty: DType.type) -> DType.type:
    return DType.uint8.value


@register_internal("uint16")
fn DTypeUInt16TypeDef(ty: DType.type) -> DType.type:
    return DType.uint16.value


@register_internal("bool")
fn DTypeBoolTypeDef(ty: DType.type) -> DType.type:
    return DType.bool.value


@register_internal("index")
fn IndexTypeDef(ty: Int) -> Int:
    return ty


@register_internal("mojoCallContext")
fn MojoCallContextDef(ty: MojoCallContextPtr):
    pass


@register_internal("deviceContext")
fn DeviceContextDef(ty: DeviceContextPtr):
    pass


@register_internal("simd")
fn SimdTypeDef[
    type: DType, width: Int
](ty: SIMD[type, width]) -> SIMD[type, width]:
    return ty


@register_internal("indices")
fn TensorIndicesTypeDef[rank: Int](ty: IndexList[rank]) -> IndexList[rank]:
    return ty


@register_internal("dim_type")
fn DimTypeDef(ty: Dim) -> Dim:
    return ty


# ===-----------------------------------------------------------------------===#
# Hooks to help build static shapes.
# ===-----------------------------------------------------------------------===#


@register_internal("create_unknown_dim")
fn create_unknown_dim() -> Dim:
    return Dim()


@register_internal("create_known_dim")
fn create_known_dim[known_val: Int]() -> Dim:
    return Dim(known_val)


@register_internal("reshape_contiguous_managed_tensor_slice")
@always_inline
fn reshape_contiguous_buffer[
    type: DType, old_rank: Int, new_rank: Int
](
    buffer: ManagedTensorSlice[
        type,
        old_rank,
        io_spec=IOUnknown,
        static_spec = StaticTensorSpec[type, old_rank](),
    ],
    shape: IndexList[new_rank],
) -> DynamicTensor[type, new_rank].Type:
    return DynamicTensor[type, new_rank].Type(buffer._ptr, shape)


# ===----------------------------------------------------------------------===#
# Additional expected primitives
# ===-----------------------------------------------------------------------===#


@register_internal("get_address_space")
fn get_address_space() -> AddressSpace:
    return AddressSpace.GENERIC


# Build the StaticTensorSpec parameter for the DPS kernels
@register_internal("build_static_tensor_specs")
fn build_static_tensor_specs[
    type: DType,
    rank: Int,
](
    shape: DimList,
    strides: DimList,
    alignment: Int,
    address_space: AddressSpace,
    exclusive: Bool,
) -> StaticTensorSpec[type, rank]:
    alias SpecType = StaticTensorSpec[type, rank]

    return SpecType(
        shape,
        strides,
        alignment,
        address_space,
        exclusive,
        None,
        None,
    )


# Build the tuple of StaticTensorSpecs for DPS kernels
@register_internal("build_static_tensor_specs_tuple")
fn build_static_tensor_specs_tuple[
    type: DType,
    rank: Int,
    size: Int,
](
    array_of_specs: VariadicList[StaticTensorSpec[type, rank]],
    out result: StaticTuple[StaticTensorSpec[type, rank], size],
):
    return __type_of(result)(array_of_specs)


# Rebuild the StaticTensorSpec parameter for the DPS kernels with different strides
@register_internal("rebuild_static_tensor_specs_with_strides")
fn rebuild_static_tensor_specs_with_strides[
    type: DType,
    rank: Int,
](
    spec: StaticTensorSpec[type, rank],
    strides: DimList,
) -> StaticTensorSpec[
    type, rank
]:
    alias SpecType = StaticTensorSpec[type, rank]

    return SpecType(
        spec.shape,
        strides,
        spec.alignment,
        spec.address_space,
        spec.exclusive,
        spec.in_lambda,
        spec.out_lambda,
    )


# TODO: this should take IOSpec as a param -- will require graph compiler changes
# Used by the graph compiler to construct tensors from MGP repr. of tensor
@register_internal("to_managed_tensor_slice")
@always_inline
fn to_managed_tensor_slice[
    type: DType, rank: Int
](
    data: UnsafePointer[Scalar[type]],
    shape: UnsafePointer[Int],
) -> DynamicTensor[type, rank].Type:
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

    return DynamicTensor[type, rank].Type(data, shape_tuple, stride_tuple)


@always_inline
fn _to_managed_tensor_slice_index_list_shape[
    type: DType, rank: Int
](
    data: UnsafePointer[Scalar[type]],
    shape_tuple: IndexList[rank],
) -> DynamicTensor[type, rank].Type:
    var stride_tuple = IndexList[rank]()
    var stride: Int = 1

    @parameter
    for i in reversed(range(rank)):
        # Start from the back so we can accumulate the strides.
        stride_tuple[i] = stride
        stride *= shape_tuple[i]

    return DynamicTensor[type, rank].Type(data, shape_tuple, stride_tuple)


# Extract a value from a shape.
@register_internal("get_scalar_from_ndbuffer")
@always_inline
fn get_scalar_from_ndbuffer[
    dtype: DType
](tensor: NDBuffer[dtype, 1]) -> Scalar[dtype]:
    # Assumes that tensor is on the host!
    return tensor[0]


# Extract a value from a managed tensor slice.
@register_internal("get_scalar_from_managed_tensor_slice")
@always_inline
fn get_scalar_from_managed_tensor_slice[
    dtype: DType,
](
    tensor: ManagedTensorSlice[
        dtype, 1, io_spec=IOUnknown, static_spec = StaticTensorSpec[dtype, 1]()
    ]
) -> Scalar[dtype]:
    # Assumes that tensor is on the host!
    # This is used instead of [0] since __getitem__ for `ManagedTesnorSlice`
    # does not work with `register_internal` out of the box.
    return tensor.load[width=1](IndexList[1](0))


@register_internal("get_int_from_shape")
@always_inline
fn get_int_from_shape[
    param_index: Int, rank: Int
](shape: IndexList[rank]) -> Int:
    return shape[param_index]


# ===-----------------------------------------------------------------------===#
# Helpers
# ===-----------------------------------------------------------------------===#

# TODO(GEX-1449): Replace remaining uses of ScalarTensor with Scalar
alias ScalarTensor = ManagedTensorSlice[rank=1]


# Used by the graph compiler -- which right now does not support static spec
@register_internal("managed_tensor_slice_to_ndbuffer")
@always_inline
fn managed_tensor_slice_to_ndbuffer_primitive[
    type: DType,
    rank: Int,
](tensor: ManagedTensorSlice[type, rank]) -> NDBuffer[type, rank]:
    return NDBuffer[type, rank](
        tensor._ptr, tensor._spec.shape, tensor._runtime_strides
    )


@always_inline
fn managed_tensor_slice_to_ndbuffer[
    spec: StaticTensorSpec
](
    tensor: ManagedTensorSlice[spec.type, spec.rank, static_spec=spec, *_]
) -> NDBuffer[
    spec.type,
    spec.rank,
    spec.shape,
    spec.strides,
    alignment = spec.alignment,
    address_space = spec.address_space,
    exclusive = spec.exclusive,
]:
    var ptr = tensor._ptr.address_space_cast[spec.address_space]()
    return NDBuffer[
        spec.type,
        spec.rank,
        spec.shape,
        spec.strides,
        alignment = spec.alignment,
        address_space = spec.address_space,
        exclusive = spec.exclusive,
    ](ptr, tensor.shape(), tensor._runtime_strides)


@always_inline("nodebug")
fn reduce_shape[
    input_rank: Int,
    input_type: DType,
](
    input_buf: ManagedTensorSlice[input_type, input_rank],
    axis0: Scalar,
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

    var axis = Int(normalize_neg_index(axis0, input_rank))

    if axis < 0 or input_rank <= axis:
        raise Error(
            "[reduction] normalized axis must be within range [0, input_rank)"
        )

    # compute and return the output shape
    var output_shape = input_buf.shape()
    output_shape[axis] = 1
    return output_shape


# ===----------------------------------------------------------------------===#
# Helpers for Affine Fusion
# ===----------------------------------------------------------------------===#


@register_internal("split_dim_indices")
@always_inline
fn split_dim_indices[
    rank: Int, axis: Int
](indices: IndexList[rank], new_shape_dim: Int64) -> IndexList[rank + 1]:
    var out = IndexList[rank + 1]()

    # This op is transforming the INDICES of an access into a reshaped tensor.
    # Consider the tensor is [40, 30, 2] and we reshape it to [5, 8, 30, 2].
    # If we are accessing the index [21, 16, 1] in the original shape then to
    # preserve the reshape we would need to transform the indices into [2, 5, 16, 1].
    # Or [21 // 8, 21 % 8, ...old dims...].
    # In this case, the axis = 0 and the new_shape_dim = 8.

    @parameter
    for i in range(rank + 1):

        @parameter
        if i == axis:
            out[i] = indices[axis] // Int(new_shape_dim)
        elif i == axis + 1:
            out[i] = indices[axis] % Int(new_shape_dim)
        elif i < axis:
            out[i] = indices[i]
        elif i > axis:
            out[i] = indices[i - 1]

    return out


@register_internal("merge_dim_indices")
@always_inline
fn merge_dim_indices[
    rank: Int, axis: Int
](indices: IndexList[rank], old_shape_dim: Int64) -> IndexList[rank - 1]:
    var out = IndexList[rank - 1]()

    # This op is transforming the INDICES of an access into a reshaped tensor.
    # Consider the tensor is [5, 8, 30, 2] and we reshape it to [40, 30, 2].
    # If we are accessing the index [2, 5, 16, 1] in the original shape then to
    # preserve the reshape we would need to transform the indices into [21, 16, 1].
    # Or [2 * 8 + 5, 16, 1].
    # In this case, the axis = 0 and the old_shape_dim = 8.

    @parameter
    for i in range(rank - 1):

        @parameter
        if i == axis:
            out[i] = fma(indices[i], Int(old_shape_dim), indices[i + 1])
        elif i < axis:
            out[i] = indices[i]
        elif i > axis:
            out[i] = indices[i + 1]

    return out


@register_internal("insert_index")
@always_inline
fn insert_index[
    rank: Int, axis: Int, value: Int
](indices: IndexList[rank]) -> IndexList[rank + 1]:
    var out = IndexList[rank + 1]()

    @always_inline
    @parameter
    fn add_dim[i: Int]():
        @parameter
        if i < axis:
            out[i] = indices[i]
        elif i > axis:
            out[i] = indices[i - 1]
        else:
            out[i] = value

    unroll[add_dim, rank + 1]()
    return out


# TODO(MOCO-1413): remove this need to keep imported exported funcs alive.
@export
fn export():
    alias _simd_load_from_managed_tensor_slice = simd_load_from_managed_tensor_slice
    alias _simd_store_into_managed_tensor_slice = simd_store_into_managed_tensor_slice
    alias __input_fusion_hook_impl = _input_fusion_hook_impl
    alias __output_fusion_hook_impl = _output_fusion_hook_impl


# ===-----------------------------------------------------------------------===#
# Elementwise Kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.range")
@compiler.elementwise
struct Range:
    @staticmethod
    fn execute[
        type: DType,
        target: StringLiteral,
        _synchronous: Bool,
    ](
        output: ManagedTensorSlice[type=type, rank=1],
        start: ManagedTensorSlice[type=type, rank=1],
        stop: ManagedTensorSlice[type=type, rank=1],
        step: ManagedTensorSlice[type=type, rank=1],
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[1]) -> SIMD[type, width]:
            return start[0] + step[0] * (iota[type, width](idx[0]))

        foreach[func, target=target, _synchronous=_synchronous](output, ctx)

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


# ===-----------------------------------------------------------------------===#
# Binary Elementwise Kernels
# ===-----------------------------------------------------------------------===#


# useful for testing --> identity op that simply copies input into output
@compiler.register("copy")
@compiler.elementwise
struct Copy:
    @staticmethod
    fn execute[
        type: DType, rank: Int
    ](
        output: ManagedTensorSlice[type, rank],
        input: ManagedTensorSlice[type, rank],
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[rank]) -> SIMD[type, width]:
            return input._fused_load[width](idx)

        foreach[func](output, ctx)


@compiler.register("mo.add")
@compiler.elementwise
struct Add:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
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

        foreach[func, target=target, _synchronous=_synchronous](z, ctx)


@compiler.register("mo.sub")
@compiler.elementwise
struct Sub:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
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

        foreach[func, target=target, _synchronous=_synchronous](z, ctx)


@compiler.register("mo.mul")
@compiler.elementwise
struct Mul:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
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

        foreach[func, target=target, _synchronous=_synchronous](z, ctx)


@compiler.register("mo.div")
@compiler.elementwise
struct Div:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
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

        foreach[func, target=target, _synchronous=_synchronous](z, ctx)


@compiler.register("mo.mod")
@compiler.elementwise
struct Mod:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
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

        foreach[func, target=target, _synchronous=_synchronous](z, ctx)


@compiler.register("mo.equal")
@compiler.elementwise
struct Equal:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
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

        foreach[func, target=target, _synchronous=_synchronous](z, ctx)


@compiler.register("mo.greater")
@compiler.elementwise
struct Greater:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
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

        foreach[func, target=target, _synchronous=_synchronous](z, ctx)


@compiler.register("mo.greater_equal")
@compiler.elementwise
struct GreaterEqual:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
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

        foreach[func, target=target, _synchronous=_synchronous](z, ctx)


@compiler.register("mo.not_equal")
@compiler.elementwise
struct NotEqual:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
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

        foreach[func, target=target, _synchronous=_synchronous](z, ctx)


@compiler.register("mo.and")
@compiler.elementwise
struct And:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
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

        foreach[func, target=target, _synchronous=_synchronous](z, ctx)


@compiler.register("mo.or")
@compiler.elementwise
struct Or:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
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

        foreach[func, target=target, _synchronous=_synchronous](z, ctx)


@compiler.register("mo.xor")
@compiler.elementwise
struct Xor:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
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

        foreach[func, target=target, _synchronous=_synchronous](z, ctx)


@compiler.register("mo.pow")
@compiler.elementwise
struct Pow:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
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

        foreach[func, target=target, _synchronous=_synchronous](z, ctx)


@compiler.register("mo.max")
@compiler.elementwise
struct Max:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
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

        foreach[func, target=target, _synchronous=_synchronous](z, ctx)


@compiler.register("mo.min")
@compiler.elementwise
struct Min:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
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

        foreach[func, target=target, _synchronous=_synchronous](z, ctx)


# ===-----------------------------------------------------------------------===#
# Unary Elementwise Kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.cast")
@compiler.elementwise
struct Cast:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](
                x._fused_load[width](idx).cast[y.type]()
            )

        foreach[func, target=target, _synchronous=_synchronous](y, ctx)


@compiler.register("mo.negative")
@compiler.elementwise
struct Negative:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](-x._fused_load[width](idx))

        foreach[func, target=target, _synchronous=_synchronous](y, ctx)


@compiler.register("mo.relu")
@compiler.elementwise
struct ReLU:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](relu(x._fused_load[width](idx)))

        foreach[func, target=target, _synchronous=_synchronous](y, ctx)


@compiler.register("mo.gelu")
@compiler.elementwise
struct GeLU:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](gelu(x._fused_load[width](idx)))

        foreach[func, target=target, _synchronous=_synchronous](y, ctx)


@compiler.register("mo.ceil")
@compiler.elementwise
struct Ceil:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](ceil(x._fused_load[width](idx)))

        foreach[func, target=target, _synchronous=_synchronous](y, ctx)


@compiler.register("mo.floor")
@compiler.elementwise
struct Floor:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](floor(x._fused_load[width](idx)))

        foreach[func, target=target, _synchronous=_synchronous](y, ctx)


@compiler.register("mo.tanh")
@compiler.elementwise
struct Tanh:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](tanh(x._fused_load[width](idx)))

        foreach[func, target=target, _synchronous=_synchronous](y, ctx)


@compiler.register("mo.cos")
@compiler.elementwise
struct Cos:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](cos(x._fused_load[width](idx)))

        foreach[func, target=target, _synchronous=_synchronous](y, ctx)


@compiler.register("mo.sin")
@compiler.elementwise
struct Sin:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](sin(x._fused_load[width](idx)))

        foreach[func, target=target, _synchronous=_synchronous](y, ctx)


@compiler.register("mo.erf")
@compiler.elementwise
struct Erf:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](erf(x._fused_load[width](idx)))

        foreach[func, target=target, _synchronous=_synchronous](y, ctx)


@compiler.register("mo.exp")
@compiler.elementwise
struct Exp:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](exp(x._fused_load[width](idx)))

        foreach[func, target=target, _synchronous=_synchronous](y, ctx)


@compiler.register("mo.round")
@compiler.elementwise
struct Round:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](round(x._fused_load[width](idx)))

        foreach[func, target=target, _synchronous=_synchronous](y, ctx)


@compiler.register("mo.roundeven")
@compiler.elementwise
struct RoundEven:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](
                x._fused_load[width](idx).roundeven()
            )

        foreach[func, target=target, _synchronous=_synchronous](y, ctx)


@compiler.register("mo.sqrt")
@compiler.elementwise
struct Sqrt:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](sqrt(x._fused_load[width](idx)))

        foreach[func, target=target, _synchronous=_synchronous](y, ctx)


@compiler.register("mo.isqrt")
@compiler.elementwise
struct Isqrt:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](isqrt(x._fused_load[width](idx)))

        foreach[func, target=target, _synchronous=_synchronous](y, ctx)


@compiler.register("mo.select")
@compiler.elementwise
struct Select:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
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

        foreach[func, target=target, _synchronous=_synchronous](out, ctx)


@compiler.register("mo.trunc")
@compiler.elementwise
struct Trunc:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
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

        foreach[func, target=target, _synchronous=_synchronous](y, ctx)


@compiler.register("mo.log")
@compiler.elementwise
struct Log:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](log(x._fused_load[width](idx)))

        foreach[func, target=target, _synchronous=_synchronous](y, ctx)


@compiler.register("mo.log1p")
@compiler.elementwise
struct Log1p:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](log1p(x._fused_load[width](idx)))

        foreach[func, target=target, _synchronous=_synchronous](y, ctx)


@compiler.register("mo.is_nan")
@compiler.elementwise
struct IsNan:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](isnan(x._fused_load[width](idx)))

        foreach[func, target=target, _synchronous=_synchronous](y, ctx)


@compiler.register("mo.is_inf")
@compiler.elementwise
struct IsInf:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](isinf(x._fused_load[width](idx)))

        foreach[func, target=target, _synchronous=_synchronous](y, ctx)


@compiler.register("mo.not")
@compiler.elementwise
struct Not:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            var val = rebind[SIMD[DType.bool, width]](x._fused_load[width](idx))
            return rebind[SIMD[y.type, width]](~val)

        foreach[func, target=target, _synchronous=_synchronous](y, ctx)


@compiler.register("mo.abs")
@compiler.elementwise
struct Abs:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](abs(x._fused_load[width](idx)))

        foreach[func, target=target, _synchronous=_synchronous](y, ctx)


@compiler.register("mo.squeeze_shape")
@compiler.elementwise
struct SqueezeShape:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
        type: DType,
        indices_type: DType,
    ](
        output_shape: ManagedTensorSlice[type, 1],
        input_shape: ManagedTensorSlice[type, 1],
        remove_indices: ManagedTensorSlice[indices_type, 1],
    ):
        # remove_indices may not be sorted so our strategy is to use -1 to
        # represent removed dimensions in a copied version of our input shape buffer
        var num_input_dims = input_shape.dim_size[0]()
        var num_remove_indices = remove_indices.dim_size[0]()
        var final_rank = num_input_dims - num_remove_indices

        debug_assert(
            final_rank == output_shape.dim_size[0](),
            "Incorrect output shape.",
        )

        alias MAX_VECTOR_LIMIT = 12
        debug_assert(
            num_input_dims <= MAX_VECTOR_LIMIT,
            "Only support shape vectors up to rank-12.",
        )
        var input_shape_copy = IndexList[MAX_VECTOR_LIMIT]()
        for i in range(num_input_dims):
            input_shape_copy[i] = Int(input_shape[i])

        # Mark every squeezed dimension as -1 in our copy of the shape tensor
        for remove_index_index in range(num_remove_indices):
            var remove_index = Int(remove_indices[remove_index_index])
            var remove_index_normalize = remove_index + num_input_dims * Int(
                remove_indices[remove_index_index] < 0
            )
            input_shape_copy[remove_index_normalize] = -1

        # # Copy over the non -1 dimensions
        var output_shape_index = 0
        for input_shape_index in range(num_input_dims):
            if input_shape_copy[input_shape_index] == -1:
                continue
            output_shape[output_shape_index] = input_shape_copy[
                input_shape_index
            ]
            output_shape_index += 1

    @staticmethod
    fn shape[
        type: DType, indices_type: DType
    ](
        input_shape: ManagedTensorSlice[type, 1],
        remove_indices: ManagedTensorSlice[indices_type, 1],
    ) raises -> IndexList[1]:
        var out_dim = input_shape.dim_size[0]() - remove_indices.dim_size[0]()

        if out_dim < 0:
            raise Error(
                "[squeeze_shape] cannot remove more dimensions than there"
                " exists"
            )

        return IndexList[1](out_dim)


@compiler.register("mo.unsqueeze_shape")
@compiler.elementwise
struct UnsqueezeShape:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
        type: DType,
        indices_type: DType,
    ](
        output_shape: ManagedTensorSlice[type, 1],
        input_shape: ManagedTensorSlice[type, 1],
        padding_indices: ManagedTensorSlice[indices_type, 1],
    ):
        # represent uninitialized dimensions, add the padding dimensions, and copy
        # over the remaining dimensions later.
        var num_input_dims = input_shape.dim_size[0]()
        var num_padding_indices = padding_indices.dim_size[0]()
        var final_rank = num_input_dims + num_padding_indices
        debug_assert(
            final_rank == output_shape.dim_size[0](),
            "Incorrect output shape.",
        )
        for output_index in range(final_rank):
            output_shape[output_index] = -1

        for padding_index_index in range(num_padding_indices):
            var padding_index = Int(padding_indices[padding_index_index])
            var padding_index_normalize = padding_index + final_rank * Int(
                padding_indices[padding_index_index] < 0
            )

            debug_assert(
                padding_index_normalize >= 0
                and padding_index_normalize < final_rank,
                (
                    "Padding indices must be between [-r, r-1] where r is the"
                    " final output rank."
                ),
            )
            debug_assert(
                output_shape[padding_index_normalize] == -1,
                (
                    "Duplicate padding indices point to the same dimension in"
                    " the final output shape."
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

    @staticmethod
    fn shape[
        type: DType, indices_type: DType
    ](
        input_shape: ManagedTensorSlice[type, 1],
        remove_indices: ManagedTensorSlice[indices_type, 1],
    ) -> IndexList[1]:
        var out_dim = input_shape.dim_size[0]() + remove_indices.dim_size[0]()
        return IndexList[1](out_dim)


# ===-----------------------------------------------------------------------===#
# ScatterND kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.scatter_nd")
struct ScatterND:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
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
            _synchronous,
            target,
        ](
            input_ndbuffer,
            indices_ndbuffer,
            updates_ndbuffer,
            output_ndbuffer,
            context=ctx,
        )

    @staticmethod
    fn shape[
        _synchronous: Bool,
    ](
        input: ManagedTensorSlice,
        updates: ManagedTensorSlice[input.type, *_],
        indices: ManagedTensorSlice,
    ) raises -> IndexList[input.rank]:
        return scatter_nd_shape[single_thread_blocking_override=_synchronous](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(updates),
            managed_tensor_slice_to_ndbuffer(indices),
        )


@compiler.register("mo.scatter_nd.add")
struct ScatterNDAdd:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
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
            _synchronous,
            target,
            reduce_fn=reduce_fn,
            trace_description="scatter_nd.add",
        ](
            input_ndbuffer,
            indices_ndbuffer,
            updates_ndbuffer,
            output_ndbuffer,
            context=ctx,
        )

    @staticmethod
    fn shape[
        _synchronous: Bool,
    ](
        input: ManagedTensorSlice,
        updates: ManagedTensorSlice[input.type, *_],
        indices: ManagedTensorSlice,
    ) raises -> IndexList[input.rank]:
        return scatter_nd_shape[single_thread_blocking_override=_synchronous](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(updates),
            managed_tensor_slice_to_ndbuffer(indices),
        )


@compiler.register("mo.scatter_nd.mul")
struct ScatterNDMul:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
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
            _synchronous,
            target,
            reduce_fn=reduce_fn,
            trace_description="scatter_nd.mul",
        ](
            input_ndbuffer,
            indices_ndbuffer,
            updates_ndbuffer,
            output_ndbuffer,
            context=ctx,
        )

    @staticmethod
    fn shape[
        _synchronous: Bool,
    ](
        input: ManagedTensorSlice,
        updates: ManagedTensorSlice[input.type, *_],
        indices: ManagedTensorSlice,
    ) raises -> IndexList[input.rank]:
        return scatter_nd_shape[single_thread_blocking_override=_synchronous](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(updates),
            managed_tensor_slice_to_ndbuffer(indices),
        )


@compiler.register("mo.scatter_nd.min")
struct ScatterNDMin:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
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
            _synchronous,
            target,
            reduce_fn=reduce_fn,
            trace_description="scatter_nd.min",
        ](
            input_ndbuffer,
            indices_ndbuffer,
            updates_ndbuffer,
            output_ndbuffer,
            context=ctx,
        )

    @staticmethod
    fn shape[
        _synchronous: Bool,
    ](
        input: ManagedTensorSlice,
        updates: ManagedTensorSlice[input.type, *_],
        indices: ManagedTensorSlice,
    ) raises -> IndexList[input.rank]:
        return scatter_nd_shape[single_thread_blocking_override=_synchronous](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(updates),
            managed_tensor_slice_to_ndbuffer(indices),
        )


@compiler.register("mo.scatter_nd.max")
struct ScatterNDMax:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
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
            _synchronous,
            target,
            reduce_fn=reduce_fn,
            trace_description="scatter_nd.max",
        ](
            input_ndbuffer,
            indices_ndbuffer,
            updates_ndbuffer,
            output_ndbuffer,
            context=ctx,
        )

    @staticmethod
    fn shape[
        _synchronous: Bool,
    ](
        input: ManagedTensorSlice,
        updates: ManagedTensorSlice[input.type, *_],
        indices: ManagedTensorSlice,
    ) raises -> IndexList[input.rank]:
        return scatter_nd_shape[single_thread_blocking_override=_synchronous](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(updates),
            managed_tensor_slice_to_ndbuffer(indices),
        )


# ===-----------------------------------------------------------------------===#
# Scatter kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.scatter")
struct Scatter:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
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
            Int(normalize_neg_index(scalar_axis, output.rank)),
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
        target: StringLiteral,
        _synchronous: Bool,
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
            Int(normalize_neg_index(scalar_axis, output.rank)),
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
        target: StringLiteral,
        _synchronous: Bool,
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
            Int(normalize_neg_index(scalar_axis, output.rank)),
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
        target: StringLiteral,
        _synchronous: Bool,
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
            Int(normalize_neg_index(scalar_axis, output.rank)),
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
        target: StringLiteral,
        _synchronous: Bool,
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
            Int(normalize_neg_index(scalar_axis, output.rank)),
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


# ===-----------------------------------------------------------------------===#
# View kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.broadcast_to")
struct BroadcastTo:
    # The `execute` method should never be used in the graph compiler.
    # We expect `mo.broadcast_to` to always simplify to `mo.static.broadcast_to`
    #
    # Sometimes with a call to the below shape function.
    @staticmethod
    fn execute() raises:
        raise Error("Should never be called!")

    @staticmethod
    fn shape_impl[
        input_rank: Int, output_rank: Int
    ](
        input: ManagedTensorSlice[rank=input_rank],
        shape: ManagedTensorSlice[rank=1],
    ) raises -> IndexList[output_rank]:
        if output_rank != shape.dim_size[0]():
            raise Error(
                "[broadcast_to] requires (len(target_shape) == output_rank)"
            )
        if input_rank > output_rank:
            raise Error("[broadcast_to] requires (input_rank <= output_rank)")

        # move the output shape from buffer into a static int tuple
        var output_shape = IndexList[output_rank]()

        for axis in range(output_rank):
            output_shape[axis] = Int(shape[axis])

        # Validate the compatibility between input and output shapes
        # NOTE we don't need to check the padded dims
        for i in range(input_rank):
            var input_axis = input_rank - i - 1
            var output_axis = output_rank - i - 1
            var input_dim = input.dim_size(input_axis)
            var output_dim = output_shape[output_axis]
            if input_dim != 1 and input_dim != output_dim:
                raise Error(
                    "[broadcast_to] input dim must be either 1 or equal to"
                    " corresponding output dim starting from the rightmost dim"
                )
        return output_shape

    @staticmethod
    fn shape[
        input_rank: Int, output_rank: Int
    ](
        input: ManagedTensorSlice[rank=input_rank],
        shape: ManagedTensorSlice[rank=1],
    ) raises -> IndexList[output_rank]:
        return BroadcastTo.shape_impl[output_rank=output_rank](input, shape)


@compiler.register("mo.broadcast_shape")
struct BroadcastShape:
    @always_inline
    @staticmethod
    fn broadcast_shape_impl(
        out_buf: ManagedTensorSlice[rank=1],
        lhs_buf: ManagedTensorSlice[rank=1],
        rhs_buf: ManagedTensorSlice[rank=1],
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
            out_buf[i] = rhs_buf[i].cast[out_buf.type]()

        for lhs_idx in range(lhs_rank):
            var rhs_idx = lhs_idx + size_diff
            var lhs_dim = Int(lhs_buf[lhs_idx])
            var rhs_dim = Int(rhs_buf[rhs_idx])
            if lhs_dim == rhs_dim:
                out_buf[rhs_idx] = rhs_buf[rhs_idx].cast[out_buf.type]()

            elif lhs_dim != 1 and rhs_dim != 1:
                debug_assert(
                    rhs_dim == 1, "one of the differing dimensions must be 1"
                )

            elif lhs_dim != 1:
                out_buf[rhs_idx] = lhs_buf[lhs_idx].cast[out_buf.type]()

            elif rhs_dim != 1:
                out_buf[rhs_idx] = rhs_buf[rhs_idx].cast[out_buf.type]()

    # The `execute` method should never be used in the graph compiler.
    # We expect `mo.broadcast_to` to always simplify to `mo.static.broadcast_to`
    #
    # Sometimes with a call to the below shape function.
    @staticmethod
    fn execute(
        out_buf: ManagedTensorSlice[rank=1],
        lhs_buf: ManagedTensorSlice[rank=1],
        rhs_buf: ManagedTensorSlice[rank=1],
    ):
        var lhs_size = lhs_buf.size()
        var rhs_size = rhs_buf.size()
        if lhs_size > rhs_size:
            return BroadcastShape.broadcast_shape_impl(
                out_buf, rhs_buf, lhs_buf
            )
        return BroadcastShape.broadcast_shape_impl(out_buf, lhs_buf, rhs_buf)

    @staticmethod
    fn shape(
        lhs_buf: ManagedTensorSlice[rank=1], rhs_buf: ManagedTensorSlice[rank=1]
    ) raises -> IndexList[1]:
        var lhs_dim = lhs_buf.dim_size[0]()
        var rhs_dim = rhs_buf.dim_size[0]()
        return IndexList[1](max(lhs_dim, rhs_dim))


fn tuple_to_dimlist[size: Int](tuple: StaticTuple[Dim, size]) -> DimList:
    @parameter
    if size == 1:
        return DimList(VariadicList[Dim](tuple[0]))
    elif size == 2:
        return DimList(VariadicList[Dim](tuple[0], tuple[1]))
    elif size == 3:
        return DimList(VariadicList[Dim](tuple[0], tuple[1], tuple[2]))
    elif size == 4:
        return DimList(
            VariadicList[Dim](tuple[0], tuple[1], tuple[2], tuple[3])
        )
    elif size == 5:
        return DimList(
            VariadicList[Dim](tuple[0], tuple[1], tuple[2], tuple[3], tuple[4])
        )

    return DimList.create_unknown[size]()


@compiler.register("mo.static.broadcast_to")
@compiler.view_kernel
struct StaticBroadcastTo:
    @always_inline
    @staticmethod
    fn build_view[
        type: DType,
        in_rank: Int,
        out_rank: Int,
        io_spec: IOSpec,
    ](
        x: ManagedTensorSlice[type, in_rank, io_spec=io_spec],
        output_shape: IndexList[out_rank],
    ) -> IODynamicTensor[type, out_rank, io_spec=io_spec].Type:
        var new_strides = IndexList[out_rank]()
        alias delta = out_rank - in_rank

        @parameter
        for i in range(out_rank):

            @parameter
            if i < delta:
                new_strides[i] = 0
            else:
                if x.dim_size[i - delta]() <= 1:
                    new_strides[i] = 0
                else:
                    new_strides[i] = x.stride_length[i - delta]()

        return IODynamicTensor[type, out_rank, io_spec].Type(
            x._ptr, output_shape, new_strides
        )

    @staticmethod
    fn get_view_strides[
        out_rank: Int,
        in_rank: Int,
    ](input_shape: DimList, input_strides: DimList) -> DimList:
        var new_strides = StaticTuple[Dim, out_rank]()
        alias delta = out_rank - in_rank

        @parameter
        for i in range(out_rank):

            @parameter
            if i < delta:
                new_strides[i] = 0
            else:
                if input_shape.at[i - delta]().is_dynamic():
                    new_strides[i] = Dim()
                elif input_shape.get[i - delta]() <= 1:
                    new_strides[i] = 0
                else:
                    new_strides[i] = input_strides.at[i - delta]()

        return tuple_to_dimlist(new_strides)

    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
        type: DType,
        in_rank: Int,
        out_rank: Int,
    ](
        z: ManagedTensorSlice[type, out_rank],
        x: ManagedTensorSlice[type, in_rank],
        output_shape: IndexList[out_rank],
        ctx: MojoCallContextPtr,
    ):
        # We need the extra output_shape argument.
        # Using `z.shape` instead will prevent the compiler from fusing the kernels.

        alias view_strides = Self.get_view_strides[z.rank, x.rank](
            x._static_shape, x._static_strides
        )

        var x_view = Self.build_view(x, output_shape)
        view_copy_impl[
            view_strides=view_strides,
            trace_name="static.broadcast_to",
            target=target,
            _synchronous=_synchronous,
        ](z, x_view, ctx)


@compiler.register("mo.static.reshape")
@compiler.view_kernel
struct StaticReshape:
    @staticmethod
    fn get_view_strides[
        out_rank: Int,
    ](out_shape: DimList) -> DimList:
        # reshape is a bit special as we assume the input is always contigous.
        # So it will be the same with the output.
        var new_strides = StaticTuple[Dim, out_rank]()

        var stride = Dim(1)

        @parameter
        for i in reversed(range(out_rank)):
            # Start from the back so we can accumulate the strides.
            new_strides[i] = stride
            stride *= out_shape.at[i]()

        return tuple_to_dimlist(new_strides)

    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
        type: DType,
        output_rank: Int,
    ](
        output: ManagedTensorSlice[type=type, rank=output_rank],
        input: ManagedTensorSlice[type=type],
        shape: IndexList[output_rank],
        ctx: MojoCallContextPtr,
    ):
        var view_buffer = reshape(
            managed_tensor_slice_to_ndbuffer(input),
            shape,
        )
        var view_tensor = IODynamicTensor[
            type, output_rank, input.io_spec
        ].Type(view_buffer.data, shape, view_buffer.get_strides())
        alias view_strides = Self.get_view_strides[output.rank](
            output._static_shape
        )
        view_copy_impl[
            view_strides=view_strides,
            trace_name="static.reshape",
            target=target,
            _synchronous=_synchronous,
        ](output, view_tensor, ctx)


@compiler.register("mo.reshape")
struct Reshape:
    # The `execute` method should never be used in the graph compiler.
    # We expect `mo.reshape` to always simplify to `mo.static.reshape`
    #
    # Sometimes with a call to the below shape function.
    @staticmethod
    fn execute() raises:
        raise Error("Should never be called!")

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
    @always_inline
    @staticmethod
    fn transpose_in_place(
        input: ManagedTensorSlice,
        permutations: ManagedTensorSlice[rank=1],
        out result: IODynamicTensor[
            type = input.type, rank = input.rank, io_spec = input.io_spec
        ].Type,
    ):
        var new_shape = IndexList[input.rank]()
        var new_stride = IndexList[input.rank]()

        @parameter
        for i in range(input.rank):
            var dim = Int(permutations[i])
            new_shape[i] = input.dim_size(dim)
            new_stride[i] = input.stride_length(dim)

        return __type_of(result)(input._ptr, new_shape, new_stride)

    @staticmethod
    fn get_view_strides[
        permutations: DimList, rank: Int
    ](input_strides: DimList) -> DimList:
        var new_strides = StaticTuple[Dim, rank]()

        @parameter
        for i in range(rank):
            alias perm = permutations.at[i]()

            @parameter
            if perm.is_dynamic():
                new_strides[i] = Dim()
            else:
                new_strides[i] = input_strides.at[Int(perm)]()

        return tuple_to_dimlist(new_strides)

    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
        static_permutations: DimList,
        type: DType,
        rank: Int,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        permutations: ManagedTensorSlice[rank=1],
        ctx: MojoCallContextPtr,
    ):
        alias view_strides = Self.get_view_strides[static_permutations, rank](
            input._static_strides
        )

        view_copy_impl[
            view_strides=view_strides,
            trace_name="transpose",
            target=target,
            _synchronous=_synchronous,
        ](output, Self.transpose_in_place(input, permutations), ctx)

    # TODO(GEX-1033) Make it possible to have multiple raises.
    @no_inline
    @staticmethod
    fn shape_impl(
        input: ManagedTensorSlice,
        permutations: ManagedTensorSlice[rank=1],
    ) raises -> IndexList[input.rank]:
        if permutations.dim_size[0]() != input.rank:
            raise Error("[transpose] permutation size must match input rank")

        @parameter
        for i in range(input.rank):
            var perm = Int(permutations[i])
            if perm < 0 or input.rank <= perm:
                raise Error(
                    "[transpose] each permutation must be within range [0,"
                    " rank)"
                )

        var view_tensor = Self.transpose_in_place(input, permutations)
        var out = IndexList[input.rank]()

        @parameter
        for i in range(input.rank):
            out[i] = view_tensor.dim_size[i]()

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
    fn get_view_strides[
        rank: Int
    ](input_strides: DimList, steps: DimList) -> DimList:
        var new_strides = StaticTuple[Dim, rank]()

        @parameter
        for i in range(rank):
            new_strides[i] = input_strides.at[i]() * steps.at[i]()

        return tuple_to_dimlist(new_strides)

    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
        static_steps: DimList,
        type: DType,
        rank: Int,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        starts: ManagedTensorSlice[rank=1],
        stops: ManagedTensorSlice[rank=1],
        steps: ManagedTensorSlice[rank=1],
        ctx: MojoCallContextPtr,
    ):
        var view_buffer = slice_as_view(
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(starts),
            managed_tensor_slice_to_ndbuffer(stops),
            managed_tensor_slice_to_ndbuffer(steps),
        )
        var view_tensor = IODynamicTensor[
            type, rank, io_spec = input.io_spec
        ].Type(
            view_buffer.data,
            view_buffer.get_shape(),
            view_buffer.get_strides(),
        )

        alias view_strides = Self.get_view_strides[rank](
            input._static_strides, static_steps
        )
        view_copy_impl[
            view_strides=view_strides,
            trace_name="slice",
            target=target,
            _synchronous=_synchronous,
        ](output, view_tensor, ctx)

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
        target: StringLiteral,
        _synchronous: Bool,
        type: DType,
        rank: Int,
    ](
        to_buffer: ManagedTensorSlice[type=type, rank=rank],
        in_slice: ManagedTensorSlice[type=type, rank=rank],
        starts: ManagedTensorSlice[rank=1],
        stops: ManagedTensorSlice[rank=1],
        steps: ManagedTensorSlice[rank=1],
        ctx: MojoCallContextPtr,
    ) raises:
        copy_to_slice[target=target](
            managed_tensor_slice_to_ndbuffer(to_buffer),
            managed_tensor_slice_to_ndbuffer(in_slice),
            managed_tensor_slice_to_ndbuffer(starts),
            managed_tensor_slice_to_ndbuffer(stops),
            managed_tensor_slice_to_ndbuffer(steps),
            ctx,
        )

    # No shape function as we just directly embed the logic to check the shape
    # of the 'slice' operand of the MO op directly in the kernel.


@compiler.register("mo.slice_dim")
@compiler.view_kernel
struct SliceDim:
    @staticmethod
    fn get_view_strides[
        rank: Int,
        axis: Int,
    ](input_strides: DimList, step: Dim) -> DimList:
        var new_strides = StaticTuple[Dim, rank]()

        @parameter
        for i in range(rank):
            if i == axis:
                new_strides[i] = input_strides.at[i]() * step
            else:
                new_strides[i] = input_strides.at[i]()

        return tuple_to_dimlist(new_strides)

    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
        type: DType,
        rank: Int,
        axis: Int,
        static_step: DimList,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        starts: Scalar,
        stops: Scalar,
        steps: Scalar,
        ctx: MojoCallContextPtr,
    ):
        var view_buffer = slice_dim_as_view[dim=axis](
            managed_tensor_slice_to_ndbuffer(input),
            Int(starts),
            Int(stops),
            Int(steps),
        )
        var view_tensor = IODynamicTensor[
            type, rank, io_spec = input.io_spec
        ].Type(
            view_buffer.data,
            view_buffer.get_shape(),
            view_buffer.get_strides(),
        )
        alias view_strides = Self.get_view_strides[rank, axis](
            input._static_strides, static_step.at[0]()
        )
        view_copy_impl[
            view_strides=view_strides,
            trace_name="slice_dim",
            target=target,
            _synchronous=_synchronous,
        ](output, view_tensor, ctx)


# ===-----------------------------------------------------------------------===#
# Data dependent kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.arg_max")
struct ArgMax:
    @staticmethod
    fn execute[
        target: StringLiteral, rank: Int
    ](
        output: ManagedTensorSlice[rank=rank],
        input: ManagedTensorSlice[rank=rank],
        axis: ManagedTensorSlice[rank=1],
        ctx: MojoCallContextPtr,
    ) raises:
        var axis_val = Int(normalize_neg_index(axis[0], rank))

        with Trace[TraceLevel.OP, target=target]("argmax"):

            @parameter
            if target == "cpu":
                var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)
                var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)

                argmax(input_ndbuffer, axis_val, output_ndbuffer)
            else:
                if axis_val != rank - 1:
                    raise Error("axis other than -1 not supported on GPU")

                # Has no static shape info
                var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)
                var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)

                # TODO(KERN-1045): Add support for taking advantage of static_shapes
                var cuda_ctx = ctx.get_device_context()
                argmax_gpu(
                    cuda_ctx,
                    input_ndbuffer,
                    output_ndbuffer,
                )


@compiler.register("mo.arg_min")
struct ArgMin:
    @staticmethod
    fn execute[
        target: StringLiteral, rank: Int
    ](
        output: ManagedTensorSlice[rank=rank],
        input: ManagedTensorSlice[rank=rank],
        axis: ManagedTensorSlice[rank=1],
        ctx: MojoCallContextPtr,
    ) raises:
        var axis_val = Int(normalize_neg_index(axis[0], rank))

        with Trace[TraceLevel.OP, target=target]("argmin"):

            @parameter
            if target == "cpu":
                var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)
                var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)

                argmin(input_ndbuffer, axis_val, output_ndbuffer)
            else:
                if axis_val != rank - 1:
                    raise Error("axis other than -1 not supported on GPU")

                # Has no static shape info
                var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)
                var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)

                # TODO(KERN-1045): Add support for taking advantage of static_shapes
                var cuda_ctx = ctx.get_device_context()
                argmin_gpu(
                    cuda_ctx,
                    input_ndbuffer,
                    output_ndbuffer,
                )


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
    @compiler.enable_fusion_for("input", "output")
    @staticmethod
    fn execute[
        _synchronous: Bool, target: StringLiteral
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[type = output.type, rank = output.rank],
        axis: Scalar,
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

        var axis_val = Int(axis)

        mean[
            output.type,
            input_fn,
            output_fn,
            single_thread_blocking_override=_synchronous,
            target=target,
        ](input.shape(), axis_val, output.shape(), ctx)

    @staticmethod
    fn shape[
        input_rank: Int,
        input_type: DType,
    ](
        input: ManagedTensorSlice[input_type, input_rank],
        axis: Scalar,
    ) raises -> IndexList[input_rank]:
        return reduce_shape[input_rank, input_type](input, axis)


@compiler.register("mo.reduce.add")
struct ReduceAdd:
    @compiler.enable_fusion_for("input", "output")
    @staticmethod
    fn execute[
        _synchronous: Bool, target: StringLiteral
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[type = output.type, rank = output.rank],
        axis: Scalar,
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

        var axis_val = Int(axis)

        with Trace[TraceLevel.OP, target=target]("mo.reduce.add"):
            sum[
                output.type,
                input_fn,
                output_fn,
                single_thread_blocking_override=_synchronous,
                target=target,
            ](input.shape(), axis_val, ctx)

    @staticmethod
    fn shape[
        input_rank: Int,
        input_type: DType,
    ](
        input: ManagedTensorSlice[input_type, input_rank],
        axis: Scalar,
    ) raises -> IndexList[input_rank]:
        return reduce_shape[input_rank, input_type](input, axis)


@compiler.register("mo.reduce.mul")
struct ReduceMul:
    @compiler.enable_fusion_for("input", "output")
    @staticmethod
    fn execute[
        _synchronous: Bool, target: StringLiteral
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[type = output.type, rank = output.rank],
        axis: Scalar,
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

        var axis_val = Int(axis)

        with Trace[TraceLevel.OP, target=target]("mo.reduce.mul"):
            product[
                output.type,
                input_fn,
                output_fn,
                single_thread_blocking_override=_synchronous,
                target=target,
            ](input.shape(), axis_val, ctx)

    @staticmethod
    fn shape[
        input_rank: Int,
        input_type: DType,
    ](
        input: ManagedTensorSlice[input_type, input_rank],
        axis: Scalar,
    ) raises -> IndexList[input_rank]:
        return reduce_shape[input_rank, input_type](input, axis)


@compiler.register("mo.reduce.max")
struct ReduceMax:
    @compiler.enable_fusion_for("input", "output")
    @staticmethod
    fn execute[
        _synchronous: Bool, target: StringLiteral
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[type = output.type, rank = output.rank],
        axis: Scalar,
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

        var axis_val = Int(axis)

        with Trace[TraceLevel.OP, target=target]("mo.reduce.max"):
            reduce_max[
                output.type,
                input_fn,
                output_fn,
                single_thread_blocking_override=_synchronous,
                target=target,
            ](input.shape(), axis_val, ctx)

    @staticmethod
    fn shape[
        input_rank: Int,
        input_type: DType,
    ](
        input: ManagedTensorSlice[input_type, input_rank],
        axis: Scalar,
    ) raises -> IndexList[input_rank]:
        return reduce_shape[input_rank, input_type](input, axis)


@compiler.register("mo.reduce.min")
struct ReduceMin:
    @compiler.enable_fusion_for("input", "output")
    @staticmethod
    fn execute[
        _synchronous: Bool, target: StringLiteral
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[type = output.type, rank = output.rank],
        axis: Scalar,
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

        var axis_val = Int(axis)

        with Trace[TraceLevel.OP, target=target]("mo.reduce.min"):
            reduce_min[
                output.type,
                input_fn,
                output_fn,
                single_thread_blocking_override=_synchronous,
                target=target,
            ](input.shape(), axis_val, ctx)

    @staticmethod
    fn shape[
        input_rank: Int,
        input_type: DType,
    ](
        input: ManagedTensorSlice[input_type, input_rank],
        axis: Scalar,
    ) raises -> IndexList[input_rank]:
        return reduce_shape[input_rank, input_type](input, axis)


@compiler.register("reduce_min_and_max")
struct ReduceMinMax:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
        type: DType,
        rank: Int,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        axis0: Scalar,
        ctx: MojoCallContextPtr,
    ) raises:
        """Given a tensor of shape [A, B, C, D] and reducing along dimension 'C'
        writes to a tensor of shape [A, B, 2, D] where [:, :, 0, :] contains
        the minimum reduction and [:, :, 1, :] contains the maximum reduction.
        """

        alias num_reductions = 2
        var axis = Int(normalize_neg_index(axis0, rank))

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
                single_thread_blocking_override=_synchronous,
                target=target,
            ](
                input.shape(),
                init=init,
                reduce_dim=axis,
                context=ctx,
            )
        _ = axis

    @staticmethod
    fn shape(input: ManagedTensorSlice, axis0: Scalar) -> IndexList[input.rank]:
        var new_shape = input.shape()
        var axis = Int(normalize_neg_index(axis0, input.rank))
        new_shape[axis] = 2

        return new_shape


# ===-----------------------------------------------------------------------===#
# Pooling kernels
# ===-----------------------------------------------------------------------===#


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


# ===-----------------------------------------------------------------------===#
# Padding kernels
# ===-----------------------------------------------------------------------===#


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


# ===-----------------------------------------------------------------------===#
# Gather kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.gather_nd")
struct GatherND:
    @staticmethod
    fn execute[
        batchDims: Int,
        target: StringLiteral,
        _synchronous: Bool,
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

        with Trace[TraceLevel.OP, target=target]("mo.gather_nd"):
            gather_nd[batch_dims=batchDims, target=target](
                data_ndbuffer, indices_ndbuffer, output_ndbuffer, ctx
            )

    @staticmethod
    fn shape[
        batch_dims: Int, output_rank: Int, _synchronous: Bool
    ](
        data: ManagedTensorSlice,
        indices: ManagedTensorSlice,
    ) raises -> IndexList[output_rank]:
        return gather_nd_shape[
            batch_dims=batch_dims,
            output_rank=output_rank,
            single_thread_blocking_override=_synchronous,
        ](
            managed_tensor_slice_to_ndbuffer(data),
            managed_tensor_slice_to_ndbuffer(indices),
        )


@compiler.register("mo.gather")
struct Gather:
    @compiler.enable_fusion_for("input", "output")
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[output.type, *_],
        indices: ManagedTensorSlice,
        axis: Scalar,
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

        @parameter
        @always_inline
        fn indices_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[indices.type, width]:
            return indices._fused_load[width=width](
                rebind[IndexList[indices.rank]](coords)
            )

        @parameter
        @always_inline
        fn output_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank], val: SIMD[output.type, width]):
            output._fused_store[width=width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.type, width]](val),
            )

        with Trace[TraceLevel.OP, target=target]("mo.gather"):
            gather[
                type = output.type,
                indices_type = indices.type,
                input_fn=input_fn,
                indices_fn=indices_fn,
                output_fn=output_fn,
                target=target,
                single_thread_blocking_override=_synchronous,
            ](
                Axis(axis, input.rank),
                input.shape(),
                indices.shape(),
                output.shape(),
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


# ===-----------------------------------------------------------------------===#
# Normalization kernels
# ===-----------------------------------------------------------------------===#


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
        epsilon: Scalar[type=type],
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
        var output_buf = managed_tensor_slice_to_ndbuffer(output)

        layer_norm[type, rank, input_fn, gamma_fn, target=target,](
            input.shape(),
            gamma.shape(),
            beta_buf,
            epsilon,
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
        epsilon: Scalar[type=type],
    ) -> IndexList[rank]:
        return input.shape()


@compiler.register("rms_norm")
struct RMSNorm:
    @compiler.enable_fusion_for("input")
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        gamma: ManagedTensorSlice[type=type, rank=1],
        epsilon: Scalar[type=type],
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

        var gamma_buf = managed_tensor_slice_to_ndbuffer(gamma)
        var output_buf = managed_tensor_slice_to_ndbuffer(output)

        rms_norm[type, rank, input_fn, target=target](
            input.shape(), gamma_buf, epsilon, output_buf, ctx
        )

    @staticmethod
    fn shape[
        type: DType,
        rank: Int,
    ](
        input: ManagedTensorSlice[type=type, rank=rank],
        gamma: ManagedTensorSlice[type=type, rank=1],
        epsilon: Scalar[type=type],
    ) -> IndexList[rank]:
        return input.shape()


# ===-----------------------------------------------------------------------===#
# TopK kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.bottom_k", num_dps_outputs=2)
struct BottomK:
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
        target: StringLiteral,
    ](
        values: ManagedTensorSlice[type=type, rank=rank],
        indices: ManagedTensorSlice[type = DType.int64, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        k: Scalar,
        axis: Scalar,
        sorted: Scalar[type = DType.bool],
        ctx: MojoCallContextPtr,
    ) raises:
        top_k[largest=False, target=target](
            managed_tensor_slice_to_ndbuffer(input),
            Int(k),
            Int(axis),
            managed_tensor_slice_to_ndbuffer(values),
            managed_tensor_slice_to_ndbuffer(indices),
            sorted,
            ctx,
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
        target: StringLiteral,
    ](
        values: ManagedTensorSlice[type=type, rank=rank],
        indices: ManagedTensorSlice[type = DType.int64, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        k: Scalar,
        axis: Scalar,
        sorted: Scalar[type = DType.bool],
        ctx: MojoCallContextPtr,
    ) raises:
        top_k[largest=True, target=target](
            managed_tensor_slice_to_ndbuffer(input),
            Int(k),
            Int(axis),
            managed_tensor_slice_to_ndbuffer(values),
            managed_tensor_slice_to_ndbuffer(indices),
            sorted,
            ctx,
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


# ===-----------------------------------------------------------------------===#
# Non maximum suppression kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.non_maximum_suppression")
struct NonMaximumSupression:
    @staticmethod
    fn execute[
        type: DType
    ](
        output: ManagedTensorSlice[type = DType.int64, rank=2],
        boxes: ManagedTensorSlice[type=type, rank=3],
        scores: ManagedTensorSlice[type, rank=3],
        max_output_boxes_per_class: Scalar[DType.int64],
        iou_threshold: Scalar[DType.float32],
        score_threshold: Scalar[DType.float32],
    ):
        var max_output_boxes_int = Int(max_output_boxes_per_class)
        var iou_threshold_float = iou_threshold
        var score_threshold_float = score_threshold

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
        max_output_boxes_per_class: Scalar[DType.int64],
        iou_threshold: Scalar[DType.float32],
        score_threshold: Scalar[DType.float32],
    ) -> IndexList[2]:
        var max_output_boxes_int = Int(max_output_boxes_per_class)
        var iou_threshold_float = iou_threshold
        var score_threshold_float = score_threshold

        return non_max_suppression_shape_func(
            managed_tensor_slice_to_ndbuffer(boxes),
            managed_tensor_slice_to_ndbuffer(scores),
            max_output_boxes_int,
            iou_threshold_float,
            score_threshold_float,
        )


# ===-----------------------------------------------------------------------===#
# Linalg kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.matmul")
struct Matmul:
    @compiler.enable_fusion_for("c")
    @staticmethod
    fn execute[
        transpose_b: Bool,
        packed_b: Bool,
        lambdas_have_fusion: Bool,
        target: StringLiteral,
        _synchronous: Bool,
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

        var a_buffer = managed_tensor_slice_to_ndbuffer(a)
        var b_buffer = managed_tensor_slice_to_ndbuffer(b)
        var c_buffer = managed_tensor_slice_to_ndbuffer(c)

        @parameter
        @always_inline
        fn output_fn[
            _type: DType, _width: Int, *, alignment: Int = 1
        ](coords: IndexList[2], val: SIMD[_type, _width]):
            c._fused_store[width=_width, element_alignment=alignment](
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
            single_thread_blocking_override=_synchronous,
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
        target: StringLiteral,
        _synchronous: Bool,
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

        @parameter
        @always_inline
        fn output_fn[
            _type: DType, _width: Int, _rank: Int, *, alignment: Int = 1
        ](coords: IndexList[_rank], val: SIMD[_type, _width]):
            c._fused_store[width=_width, element_alignment=alignment](
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
            single_thread_blocking_override=_synchronous,
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
        _synchronous: Bool,
        type: DType,
    ](
        x: ManagedTensorSlice[type=type],
        a: ManagedTensorSlice[type=type],
        b: ManagedTensorSlice[type=type],
    ) raises:
        matrix_solve[single_thread_blocking_override=_synchronous](
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
        target: StringLiteral,
        _synchronous: Bool,
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
            single_thread_blocking_override=_synchronous,
            target=target,
        ](
            input.shape(),
            num_lower_buf,
            num_upper_buf,
            exclude_buf,
            output_buf,
            ctx,
        )


# ===-----------------------------------------------------------------------===#
# Resize kernels
# ===-----------------------------------------------------------------------===#


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
            shape[i] = Int(size[i])

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
            shape[i] = Int(size[i])

        return shape


# ===-----------------------------------------------------------------------===#
# ROI align kernels
# ===-----------------------------------------------------------------------===#


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
        output_height: Scalar[DType.int64],
        output_width: Scalar[DType.int64],
        spatial_scale: Scalar,
        sampling_ratio: Scalar,
    ):
        roi_align_nhwc[aligned, mode](
            managed_tensor_slice_to_ndbuffer(output),
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(rois),
            Int(output_height),
            Int(output_width),
            spatial_scale,
            sampling_ratio,
        )

    @staticmethod
    fn shape(
        input: ManagedTensorSlice[rank=4],
        rois: ManagedTensorSlice[rank=2],
        output_height: Scalar[DType.int64],
        output_width: Scalar[DType.int64],
        spatial_scale: Scalar,
        sampling_ratio: Scalar,
    ) -> IndexList[4]:
        var shape = IndexList[4]()
        # input shape is [N, H, W, C]
        # rois shape is [M, 5]
        # output shape is [M, output_height, output_width, C]
        shape[0] = rois.dim_size[0]()
        shape[1] = Int(output_height)
        shape[2] = Int(output_width)
        shape[3] = input.dim_size[3]()

        return shape


# ===-----------------------------------------------------------------------===#
# Tile kernels
# ===-----------------------------------------------------------------------===#


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


# ===-----------------------------------------------------------------------===#
# Random kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.random.normal")
struct RandomNormal:
    @staticmethod
    fn execute[
        mean_var_type: DType
    ](
        output: ManagedTensorSlice,
        shape: ManagedTensorSlice[rank=1],
        mean: Scalar,
        variance: Scalar,
        seed_value: Scalar,
    ):
        seed(Int(seed_value))
        var num_elements = 1
        # TODO: Add __len__ support in ManagedTensorSlice.
        for i in range(shape.dim_size[0]()):
            num_elements *= Int(shape[i])
        randn(
            output._ptr,
            num_elements,
            mean.cast[DType.float64](),
            variance.cast[DType.float64](),
        )

    @staticmethod
    fn shape[
        output_rank: Int
    ](shape: ManagedTensorSlice[rank=1]) -> IndexList[output_rank]:
        var unrolled_shape = IndexList[output_rank]()
        for i in range(output_rank):
            unrolled_shape[i] = Int(shape[i])

        return unrolled_shape


@compiler.register("mo.static.random.normal")
struct StaticRandomNormal:
    @staticmethod
    fn execute[
        mean_var_type: DType
    ](
        output: ManagedTensorSlice,
        mean: Scalar,
        variance: Scalar,
        seed_value: Scalar,
    ):
        seed(Int(seed_value))
        var num_elements = output.size()
        randn(
            output._ptr,
            num_elements,
            mean.cast[DType.float64](),
            variance.cast[DType.float64](),
        )


@compiler.register("mo.softmax")
struct Softmax:
    @compiler.enable_fusion_for("input")
    @staticmethod
    fn execute[
        target: StringLiteral
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[output.type, output.rank],
        ctx: MojoCallContextPtr,
    ) raises:
        # shape should be the same between the two inputs
        var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)

        # For adapting input fusion lambda required by call
        @parameter
        @always_inline
        fn input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[output.type, width]:
            return input._fused_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        softmax[
            output.type,
            simdwidthof[output.type](),
            output.rank,
            output_ndbuffer.shape,
            input_fn,
            target,
        ](
            output.shape(),
            output_ndbuffer,
            output.rank - 1,
            context=ctx,
        )


@compiler.register("mo.logsoftmax")
struct LogSoftmax:
    @compiler.enable_fusion_for("input")
    @staticmethod
    fn execute[
        target: StringLiteral
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[output.type, output.rank],
    ) raises:
        # shape should be the same between the two inputs
        var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)

        # For adapting input fusion lambda required by call
        @parameter
        @always_inline
        fn input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[output.type, width]:
            return input._fused_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        logsoftmax[
            output.type,
            simdwidthof[output.type](),
            output.rank,
            output_ndbuffer.shape,
            input_fn,
        ](output.shape(), output_ndbuffer, output.rank - 1)


# ===-----------------------------------------------------------------------===#
# Cumsum kernels
# ===-----------------------------------------------------------------------===#


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
        axis: Scalar,
        ctx: MojoCallContextPtr,
    ):
        var output_buf = managed_tensor_slice_to_ndbuffer(output)
        var input_buf = managed_tensor_slice_to_ndbuffer(input)

        cumsum[rank, type, exclusive, reverse](
            output_buf, input_buf, Int(normalize_neg_index(axis, rank))
        )


# ===-----------------------------------------------------------------------===#
# Concat kernels
# ===-----------------------------------------------------------------------===#


fn concat_shape_impl[
    type: DType, rank: Int, size: Int, io_spec: IOSpec
](
    axis_buf: ManagedTensorSlice[rank=1],
    inputs: VariadicTensors[type, rank, size, io_spec=io_spec],
) raises -> IndexList[rank]:
    var axis_val = axis_buf._ptr.load(0)
    var axis = Int(normalize_neg_index(axis_val, rank))
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

    @parameter
    for i in range(inputs.size):
        concat_axis_dim_sum += inputs[i].dim_size(axis)
        if not shape_equal_ignore_axis(
            inputs[0].shape(),
            inputs[i].shape(),
        ):
            raise Error(
                "[concat] input shapes must match except at concat axis"
            )

    # compute and return the output shape
    var output_shape = inputs[0].shape()
    output_shape[axis] = concat_axis_dim_sum
    return output_shape


@compiler.register("mo.concat")
struct Concat:
    # TODO(GEX-1591): Re-enable fusion of variadic operands
    @compiler.enable_fusion_for("output")
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
        target: StringLiteral,
        _synchronous: Bool,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        axis: ManagedTensorSlice[rank=1],
        inputs: VariadicTensors[type, rank, io_spec=IOUnknown, *_],
        ctx: MojoCallContextPtr,
    ) raises:
        var output_buf = managed_tensor_slice_to_ndbuffer(output)
        var axis_val = axis._ptr.load(0)

        var input_shapes = StaticTuple[IndexList[rank], inputs.size]()

        @parameter
        for i in range(inputs.size):
            input_shapes[i] = inputs[i].shape()

        @always_inline
        @parameter
        fn inputs_lambda[
            input_index: Int,
            width: Int,
            _rank: Int,
        ](indices: IndexList[_rank]) -> SIMD[type, width]:
            constrained[
                input_index < inputs.size, "tensor index out of bounds"
            ]()
            return inputs[input_index]._fused_load[width=width](
                rebind[IndexList[rank]](indices)
            )

        @always_inline
        @parameter
        fn epilogue_wrapper[
            _type: DType, _rank: Int, width: Int, *, alignment: Int = 1
        ](indices: IndexList[_rank], value: SIMD[_type, width]):
            output._fused_store[width=width, element_alignment=alignment](
                rebind[IndexList[output.rank]](indices),
                rebind[SIMD[output.type, width]](value),
            )

        fused_concat[
            type,
            rank,
            _synchronous,
            inputs_lambda,
            epilogue_wrapper,
            target,
        ](
            Int(normalize_neg_index(axis_val, rank)),
            input_shapes,
            output_buf,
            ctx,
        )

    @staticmethod
    fn shape[
        type: DType,
        rank: Int,
        _synchronous: Bool,
    ](
        axis_buf: ManagedTensorSlice[rank=1],
        inputs: VariadicTensors[type, rank, *_],
    ) raises -> IndexList[rank]:
        return concat_shape_impl(axis_buf, inputs)


# Helper method used by compiler to reconcile MGP list with type Mojo expects.
# TODO(GEX-1789): Specialized on just Input tensors atm which is fine since
# this is only used by an input list of tensor for mo.concat_from_list
@register_internal("to_managed_tensor_slice_list")
@always_inline
fn to_managed_tensor_slice_list[
    type: DType, rank: Int
](
    raw_list_ptr: UnsafePointer[NoneType],
) -> InlinedFixedVector[
    DynamicTensor[type, rank].Type
]:
    var num_elements = external_call["MGP_RT_ListSize", Int64](
        raw_list_ptr
    ).__int__()

    var data_ptrs = InlinedFixedVector[UnsafePointer[NoneType], 0](num_elements)
    var dim_values = InlinedFixedVector[Int64, 0](num_elements * rank)

    data_ptrs.current_size = num_elements
    dim_values.current_size = num_elements * rank

    # Collect the data pointers and dimensions of each element from the list.
    external_call["MGP_RT_ListPopulate", NoneType](
        raw_list_ptr, data_ptrs.dynamic_data, dim_values.dynamic_data
    )

    # TODO: revist the use of unknown here
    # Create output list
    var out_list = InlinedFixedVector[DynamicTensor[type, rank].Type](
        num_elements
    )

    # Convert individual elements of the input list into NDBuffer, and
    # accumulate the results to output list.
    for i in range(num_elements):
        var data = data_ptrs[i].bitcast[Scalar[type]]()

        var dims = IndexList[rank]()

        @parameter
        for dim in range(rank):
            dims[dim] = dim_values[dim + i * rank].__int__()

        var buffer = _to_managed_tensor_slice_index_list_shape[type, rank](
            data, dims
        )
        out_list.append(buffer)

    return out_list^


# NOTE: there are a lot of similarities between this and the shape func
# for mo.concat.
fn concat_from_list_shape_impl[
    type: DType, rank: Int
](
    axis_buf: ManagedTensorSlice[rank=1],
    inputs: InlinedFixedVector[DynamicTensor[type, rank].Type],
) raises -> IndexList[rank]:
    var axis_val = axis_buf._ptr.load(0)
    var axis = Int(normalize_neg_index(axis_val, rank))
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
            inputs[0].shape(),
            inputs[i].shape(),
        ):
            raise Error(
                "[concat] input shapes must match except at concat axis"
            )

    # compute and return the output shape
    var output_shape = inputs[0].shape()
    output_shape[axis] = concat_axis_dim_sum
    return output_shape


@compiler.register("mo.concat_from_list")
struct ConcatFromList:
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
        target: StringLiteral,
        _synchronous: Bool,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        inputs: InlinedFixedVector[DynamicTensor[type, rank].Type],
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

        _concat_cpu[rank, type, None, _synchronous](
            output_buf,
            Int(normalize_neg_index(axis[0], rank)),
            input_as_ndbuffer,
        )

    @staticmethod
    fn shape[
        type: DType,
        rank: Int,
        _synchronous: Bool,
    ](
        inputs: InlinedFixedVector[DynamicTensor[type, rank].Type],
        axis_buf: ManagedTensorSlice[rank=1],
    ) raises -> IndexList[rank]:
        return concat_from_list_shape_impl(axis_buf, inputs)


# ===-----------------------------------------------------------------------===#
# Split kernels
# ===-----------------------------------------------------------------------===#


# The shape function for split is special and there is special
# handling in the graph compiler to make things work.
@compiler.register("mo.split")
struct Split:
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
        _synchronous: Bool,
        target: StringLiteral,
    ](
        output: VariadicTensors[type, rank, *_],
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
            input_buf, Int(normalize_neg_index(axis_val, rank)), output_bufs
        )


# In practice this is how it's done. The graph compiler has additional logic
# to properly dispatch this function.
@compiler.register("split_ith_output_shape")
struct SplitOutputShapeHelper:
    @staticmethod
    fn execute() raises:
        raise Error("Should not be called directly.")

    @staticmethod
    @always_inline
    fn shape[
        rank: Int,
        input_type: DType,
        split_size_type: DType,
        axis_type: DType,
        _synchronous: Bool,
    ](
        input_buf: ManagedTensorSlice[input_type, rank],
        split_sizes_buf: ManagedTensorSlice[split_size_type, 1],
        split_axis_buf: ManagedTensorSlice[axis_type, 1],
        output_idx: Int,
    ) raises -> IndexList[rank]:
        # extract relevant hyper parameters
        if output_idx < 0 or split_sizes_buf.size() <= output_idx:
            raise Error(
                "[split] output index must be within range [0,"
                " len(split_sizes))"
            )
        var output_split_size = Int(split_sizes_buf[output_idx])

        var split_axis = Int(split_axis_buf[0])
        if split_axis < 0:
            split_axis += rank
        if split_axis < 0 or rank <= split_axis:
            raise Error(
                "[split] normalized axis must be within range [0, rank)"
            )

        var split_sizes_sum = 0

        for i in range(split_sizes_buf.dim_size[0]()):
            split_sizes_sum += Int(split_sizes_buf[i])
        if split_sizes_sum != input_buf.dim_size(split_axis):
            raise Error(
                "[split] sum of split sizes must match input dimension at split"
                " axis"
            )

        # compute and return the output shape
        var output_shape = input_buf.shape()
        output_shape[split_axis] = output_split_size
        return output_shape


# ===-----------------------------------------------------------------------===#
# Convolution kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.conv")
struct Conv:
    @compiler.enable_fusion_for("output")
    @staticmethod
    fn execute[
        filter_packed: Bool,
        lambdas_have_fusion: Bool,
        static_strides: DimList,
        static_dilations: DimList,
        static_padding: DimList,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[rank = output.rank],
        filter: ManagedTensorSlice,
        strides: ManagedTensorSlice,
        dilation: ManagedTensorSlice,
        paddings: ManagedTensorSlice,
        num_groups: Scalar,
        ctx: MojoCallContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn output_fn[
            _type: DType, _rank: Int, _width: Int
        ](coords: IndexList[_rank], val: SIMD[_type, _width]):
            output._fused_store[width=_width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.type, _width]](val),
            )

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
            stride_tuple[i] = Int(strides._ptr[i])
            dilation_tuple[i] = Int(dilation._ptr[i])

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
            input._static_shape.at[input.rank - 1](),  # input C, NHWC
            filter._static_shape.at[
                filter.rank - 2
            ](),  # filter C, RSCF or FRSCf
        )

        var input_buf = managed_tensor_slice_to_ndbuffer(input)
        var filter_buf = managed_tensor_slice_to_ndbuffer(filter)
        var output_buf = managed_tensor_slice_to_ndbuffer(output)

        @parameter
        if is_cpu[target]():
            conv_nhwc_direct[
                input.rank,
                filter.rank,
                input._static_shape,  # input shape
                filter._static_shape,  # filter shape
                output._static_shape,  # output shape
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
                Int(num_groups),
            )
        else:
            constrained[
                input.rank == 4 and filter.rank == 4,
                "only rank 4 tensor is supported on cuda gpu",
            ]()
            constrained[
                filter_packed == False,
                "only unpacked filter is supported on cuda gpu",
            ]()

            constrained[
                lambdas_have_fusion == False, "lambda fusion isnt supported"
            ]()

            var cuda_ctx = ctx.get_device_context()
            conv_gpu[
                input.rank,
                filter.rank,
                input._static_shape,  # input shape
                filter._static_shape,  # filter shape
                output._static_shape,  # output shape
                input.type,
                filter.type,
                output.type,
            ](
                input_buf,
                filter_buf,
                output_buf,
                IndexList[2](stride_tuple[0], stride_tuple[1]),
                IndexList[2](dilation_tuple[0], dilation_tuple[1]),
                IndexList[2](pad_h_tuple[0], pad_w_tuple[0]),
                Int(num_groups),
                cuda_ctx,
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
    @compiler.enable_fusion_for("output")
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
            stride_tuple[i] = Int(strides._ptr[i])
            dilation_tuple[i] = Int(dilation._ptr[i])

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

        var input_buf = managed_tensor_slice_to_ndbuffer(input)
        var filter_buf = managed_tensor_slice_to_ndbuffer(filter)
        var output_buf = managed_tensor_slice_to_ndbuffer(output)

        conv_transposed[
            input.rank,
            filter.rank,
            input._static_shape,  # Input shape.
            filter._static_shape,  # Filter shape.
            output._static_shape,  # Output shape.
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


# ===-----------------------------------------------------------------------===#
# Attention kernels
# ===-----------------------------------------------------------------------===#


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
        scale: Scalar[type = DType.float32],
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
        norm = broadcast_to(normScalar, shape_of(attentionMatrix))

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
        constrained[is_gpu[target](), "only valid on GPUs"]()

        var output_buffer = managed_tensor_slice_to_ndbuffer(output)
        var q_buffer = managed_tensor_slice_to_ndbuffer(q)
        var k_buffer = managed_tensor_slice_to_ndbuffer(k)
        var v_buffer = managed_tensor_slice_to_ndbuffer(v)
        var mask_buffer = managed_tensor_slice_to_ndbuffer(mask)

        flash_attention[add_attn_mask=True](
            output_buffer,
            q_buffer,
            k_buffer,
            v_buffer,
            mask_buffer,
            scale,
            context=ctx,
        )


@compiler.register("no_mask_flash_attention_cpu")
struct NoMaskFlashAttentionCPU:
    @compiler.enable_fusion_for("k", "v")
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        q: ManagedTensorSlice[type=type, rank=rank],
        k: ManagedTensorSlice[type=type, rank=rank],
        v: ManagedTensorSlice[type=type, rank=rank],
        scale: Scalar[type = DType.float32],
    ) raises:
        @parameter
        @always_inline
        fn k_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[k.type, width]:
            return k._fused_load[width=width](rebind[IndexList[k.rank]](coords))

        @parameter
        @always_inline
        fn v_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[v.type, width]:
            return v._fused_load[width=width](rebind[IndexList[v.rank]](coords))

        @parameter
        @always_inline
        fn mask_input_fn[
            width: Int, _rank: Int
        ](idx: IndexList[_rank]) -> SIMD[type, width]:
            return SIMD[type, width](0)

        nn_flash_attention[k_input_fn, v_input_fn, mask_input_fn](
            managed_tensor_slice_to_ndbuffer(q),
            k.shape(),
            v.shape(),
            IndexList[0](),
            managed_tensor_slice_to_ndbuffer(output),
            scale.cast[DType.float32](),
        )


@compiler.register("with_mask_flash_attention_split_kv_cpu")
struct WithMaskFlashAttentionSplitKVCPU:
    @compiler.enable_fusion_for("k", "v", "k_cache", "v_cache", "mask")
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
        scale: Scalar[type = DType.float32],
    ) raises:
        @parameter
        @always_inline
        fn k_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[k.type, width]:
            return k._fused_load[width=width](rebind[IndexList[k.rank]](coords))

        @parameter
        @always_inline
        fn v_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[v.type, width]:
            return v._fused_load[width=width](rebind[IndexList[v.rank]](coords))

        @parameter
        @always_inline
        fn k_cache_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[k_cache.type, width]:
            return k_cache._fused_load[width=width](
                rebind[IndexList[k_cache.rank]](coords)
            )

        @parameter
        @always_inline
        fn v_cache_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[v_cache.type, width]:
            return v_cache._fused_load[width=width](
                rebind[IndexList[v_cache.rank]](coords)
            )

        @parameter
        @always_inline
        fn mask_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[mask.type, width]:
            return mask._fused_load[width=width](
                rebind[IndexList[mask.rank]](coords)
            )

        flash_attention_split_kv[
            k_input_fn,
            v_input_fn,
            k_cache_input_fn,
            v_cache_input_fn,
            mask_input_fn,
        ](
            managed_tensor_slice_to_ndbuffer(q),
            k.shape(),
            v.shape(),
            k_cache.shape(),
            v_cache.shape(),
            mask.shape(),
            managed_tensor_slice_to_ndbuffer(output),
            scale.cast[DType.float32](),
        )

    @staticmethod
    fn shape(q: ManagedTensorSlice) -> IndexList[q.rank]:
        return q.shape()


@compiler.register("with_mask_flash_attention_cpu")
struct WithMaskFlashAttentionCPU:
    @compiler.enable_fusion_for("k", "v", "mask")
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
        scale: Scalar[type = DType.float32],
    ) raises:
        @parameter
        @always_inline
        fn k_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[k.type, width]:
            return k._fused_load[width=width](rebind[IndexList[k.rank]](coords))

        @parameter
        @always_inline
        fn v_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[v.type, width]:
            return v._fused_load[width=width](rebind[IndexList[v.rank]](coords))

        @parameter
        @always_inline
        fn mask_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[mask.type, width]:
            return mask._fused_load[width=width](
                rebind[IndexList[mask.rank]](coords)
            )

        nn_flash_attention[k_input_fn, v_input_fn, mask_input_fn](
            managed_tensor_slice_to_ndbuffer(q),
            k.shape(),
            v.shape(),
            mask.shape(),
            managed_tensor_slice_to_ndbuffer(output),
            scale.cast[DType.float32](),
        )


# ===-----------------------------------------------------------------------===#
# Quantization for CPU
# ===-----------------------------------------------------------------------===#

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
                output.shape(),
            )

    @staticmethod
    @always_inline
    fn shape(input: ManagedTensorSlice[DType.uint8, 2]) -> IndexList[2]:
        alias block_nbytes = sizeof[Q4sym[group_size=32]]()
        alias quants_per_block = 32
        var num_block_per_batch = (
            input.size() // input.dim_size[0]()
        ) // block_nbytes
        return (input.dim_size[0](), quants_per_block * num_block_per_batch)


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
        return IndexList[2](a.dim_size[0](), b.dim_size[0]())


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
        return b.shape()


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
            input.size() // input.dim_size[0]()
        ) // block_nbytes

        return (
            input.dim_size[0](),
            elements_per_block * num_block_per_batch,
        )


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
        return IndexList[2](a.dim_size[0](), b.dim_size[0]())


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
        return b.shape()


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
                output.shape(),
            )

    @staticmethod
    @always_inline
    fn shape(input: ManagedTensorSlice[DType.uint8, 2]) -> IndexList[2]:
        alias block_nbytes = sizeof[block_Q6_K]()
        alias elements_per_block = block_QK_K.quantized_k

        var num_block_per_batch = (
            input.size() // input.dim_size[0]()
        ) // block_nbytes

        return (
            input.dim_size[0](),
            elements_per_block * num_block_per_batch,
        )


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
        return IndexList[2](a.dim_size[0](), b.dim_size[0]())


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
        return b.shape()


######
# 4-bit quant GPU implementation
######


@compiler.register("qmatmul_b4_g32")
struct QMatmulGPU_b4_g32:
    @staticmethod
    @always_inline
    fn execute[
        target: StringLiteral,
    ](
        c: ManagedTensorSlice[DType.bfloat16, 2],
        a: ManagedTensorSlice[DType.bfloat16, 2],
        b: ManagedTensorSlice[DType.uint8, 2],
        ctx: MojoCallContextPtr,
    ) raises:
        constrained[is_gpu[target](), "only valid on GPUs"]()

        with Trace[TraceLevel.OP, target=target]("qmatmul_b4_g32"):
            matmul_gpu_qint4[32, target](
                managed_tensor_slice_to_ndbuffer(c),
                managed_tensor_slice_to_ndbuffer(a),
                managed_tensor_slice_to_ndbuffer(b),
                ctx,
            )

    @staticmethod
    @always_inline
    fn shape(
        a: ManagedTensorSlice[DType.float32, 2],
        b: ManagedTensorSlice[DType.uint8, 2],
    ) -> IndexList[2]:
        return IndexList[2](a.dim_size[0](), b.dim_size[0]())


@compiler.register("qmatmul_b4_g128")
struct QMatmulGPU_b4_g128:
    @staticmethod
    @always_inline
    fn execute[
        target: StringLiteral,
    ](
        c: ManagedTensorSlice[DType.bfloat16, 2],
        a: ManagedTensorSlice[DType.bfloat16, 2],
        b: ManagedTensorSlice[DType.uint8, 2],
        ctx: MojoCallContextPtr,
    ) raises:
        constrained[is_gpu[target](), "only valid on GPUs"]()

        with Trace[TraceLevel.OP, target=target]("qmatmul_b4_g128"):
            matmul_gpu_qint4[128, target](
                managed_tensor_slice_to_ndbuffer(c),
                managed_tensor_slice_to_ndbuffer(a),
                managed_tensor_slice_to_ndbuffer(b),
                ctx,
            )

    @staticmethod
    @always_inline
    fn shape(
        a: ManagedTensorSlice[DType.float32, 2],
        b: ManagedTensorSlice[DType.uint8, 2],
    ) -> IndexList[2]:
        return IndexList[2](a.dim_size[0](), b.dim_size[0]())


@compiler.register("GGUF_gpu_repack_q4_0")
struct QMatmulGPURepackGGUF:
    @staticmethod
    @always_inline
    fn execute[
        target: StringLiteral,
    ](
        b_packed: ManagedTensorSlice[DType.uint8, 2],
        b: ManagedTensorSlice[DType.uint8, 2],
        ctx: MojoCallContextPtr,
    ) raises:
        constrained[is_gpu[target](), "only valid on GPUs"]()

        with Trace[TraceLevel.OP, target=target]("GGUF_gpu_repack_q4_0"):
            gpu_qint4_repack_Q4_0[target](
                managed_tensor_slice_to_ndbuffer(b),
                managed_tensor_slice_to_ndbuffer(b_packed),
                ctx,
            )

    @staticmethod
    @always_inline
    fn shape(
        b: ManagedTensorSlice[DType.uint8, 2],
    ) -> IndexList[2]:
        return b.shape()


@compiler.register("GPTQ_gpu_repack_b4_g128")
struct QMatmulGPURepackGPTQ_b4_g128:
    @staticmethod
    @always_inline
    fn execute[
        target: StringLiteral,
    ](
        b_packed: ManagedTensorSlice[DType.uint8, 2],
        b: ManagedTensorSlice[DType.uint8, 2],
        ctx: MojoCallContextPtr,
    ) raises:
        constrained[is_gpu[target](), "only valid on GPUs"]()

        with Trace[TraceLevel.OP, target=target]("GPTQ_gpu_repack_b4_g128"):
            gpu_qint4_repack_GPTQ[128, target](
                managed_tensor_slice_to_ndbuffer(b),
                managed_tensor_slice_to_ndbuffer(b_packed),
                ctx=ctx,
            )

    @staticmethod
    @always_inline
    fn shape(
        b: ManagedTensorSlice[DType.uint8, 2],
    ) -> IndexList[2]:
        return IndexList[2](b.dim_size[1](), b.dim_size[0]())


@compiler.register("GPTQ_gpu_repack_b4_g128_desc_act")
struct QMatmulGPURepackGPTQ_b4_g128_desc_act:
    @staticmethod
    @always_inline
    fn execute[
        target: StringLiteral,
    ](
        b_packed: ManagedTensorSlice[DType.uint8, 2],
        b: ManagedTensorSlice[DType.uint8, 2],
        perm_idx: ManagedTensorSlice[DType.int32, 1],
        ctx: MojoCallContextPtr,
    ) raises:
        constrained[is_gpu[target](), "only valid on GPUs"]()

        with Trace[TraceLevel.OP, target=target](
            "GPTQ_gpu_repack_b4_g128_desc_act"
        ):
            gpu_qint4_repack_GPTQ[128, target](
                managed_tensor_slice_to_ndbuffer(b),
                managed_tensor_slice_to_ndbuffer(b_packed),
                rebind[NDBuffer[DType.int32, 1]](
                    managed_tensor_slice_to_ndbuffer(perm_idx)
                ),
                ctx=ctx,
            )

    @staticmethod
    @always_inline
    fn shape(
        b: ManagedTensorSlice[DType.uint8, 2],
        perm_idx: ManagedTensorSlice[DType.int32, 1],
    ) -> IndexList[2]:
        return IndexList[2](b.dim_size(1), b.dim_size(0))


# ===----------------------------------------------------------------------===#
# KV Cache
# ===-----------------------------------------------------------------------===#


# ===-----------------------------------------------------------------------===#
# Fused QKV matmul
#
# Expected kernel name format:
# mo.fused_qkv_matmul.<padded/ragged>.<continuous_batching/paged>
# ===-----------------------------------------------------------------------===#


@always_inline
fn generic_fused_qkv_matmul_kv_cache_cont_batch_ragged_kernel_api[
    target: StringLiteral,
    type: DType,
](
    output: ManagedTensorSlice[type, 2],
    hidden_state: ManagedTensorSlice[type, 2],
    input_row_offsets: ManagedTensorSlice[DType.uint32, 1],
    weight: ManagedTensorSlice[type, 2],
    kv_collection: ContinuousBatchingKVCacheCollection,
    layer_idx: Scalar[DType.uint32],
    ctx: MojoCallContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,).
            The value at each index is the start_idx of the corresponding batch in hidden_state.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The historical KVCache for keys and values. The KVCache for
            this layer is retrieved via layer_idx.
        layer_idx: The index of the layer being executed. Used to retrieve the KVCache
            for the given layer from kv_collection.
        ctx: The call context pointer, passed by the graph compiler.
    """
    generic_fused_qkv_matmul_kv_cache_cont_batch_ragged[target=target](
        managed_tensor_slice_to_ndbuffer(hidden_state),
        managed_tensor_slice_to_ndbuffer(input_row_offsets),
        managed_tensor_slice_to_ndbuffer(weight),
        kv_collection,
        layer_idx,
        managed_tensor_slice_to_ndbuffer(output),
        ctx,
    )


@compiler.register("mo.fused_qkv_matmul.ragged.continuous_batching")
struct Struct_fused_qkv_matmul_ragged_continuous_batching:
    @always_inline
    @staticmethod
    fn execute[
        type: DType, num_heads: Int, head_dim: Int, //, target: StringLiteral
    ](
        output: ManagedTensorSlice[type, 2],
        hidden_state: ManagedTensorSlice[type, 2],
        input_row_offsets: ManagedTensorSlice[DType.uint32, 1],
        weight: ManagedTensorSlice[type, 2],
        kv_collection: ContinuousBatchingKVCacheCollection[
            type,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
        ],
        layer_idx: Scalar[DType.uint32],
        ctx: MojoCallContextPtr,
    ) raises:
        generic_fused_qkv_matmul_kv_cache_cont_batch_ragged_kernel_api[target](
            output,
            hidden_state,
            input_row_offsets,
            weight,
            kv_collection,
            layer_idx,
            ctx,
        )


@always_inline
fn generic_fused_qkv_matmul_kv_cache_bshd_continuous_batch_kernel_api[
    target: StringLiteral,
    type: DType,
](
    output: ManagedTensorSlice[type, 3],
    hidden_state: ManagedTensorSlice[type, 3],
    weight: ManagedTensorSlice[type, 2],
    kv_collection: ContinuousBatchingKVCacheCollection,
    layer_idx: Scalar[DType.uint32],
    ctx: MojoCallContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The historical KVCache for keys and values. The KVCache for
            this layer is retrieved via layer_idx.
        layer_idx: The index of the layer being executed. Used to retrieve the KVCache
            for the given layer from kv_collection.
        ctx: The call context pointer, passed by the graph compiler.
    """
    generic_fused_qkv_matmul_kv_cache_bshd_continuous_batch[target=target](
        managed_tensor_slice_to_ndbuffer(hidden_state),
        managed_tensor_slice_to_ndbuffer(weight),
        kv_collection,
        layer_idx,
        managed_tensor_slice_to_ndbuffer(output),
        ctx,
    )


@compiler.register("mo.fused_qkv_matmul.padded.continuous_batching")
struct Struct_fused_qkv_matmul_padded_continuous_batching:
    @always_inline
    @staticmethod
    fn execute[
        type: DType, num_heads: Int, head_dim: Int, //, target: StringLiteral
    ](
        output: ManagedTensorSlice[type, 3],
        hidden_state: ManagedTensorSlice[type, 3],
        weight: ManagedTensorSlice[type, 2],
        kv_collection: ContinuousBatchingKVCacheCollection[
            type,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
        ],
        layer_idx: Scalar[DType.uint32],
        ctx: MojoCallContextPtr,
    ) raises:
        generic_fused_qkv_matmul_kv_cache_bshd_continuous_batch_kernel_api[
            target
        ](output, hidden_state, weight, kv_collection, layer_idx, ctx)


@always_inline
fn generic_fused_qkv_matmul_kv_cache_paged_ragged_kernel_api[
    type: DType,
    weight_type: DType,
    target: StringLiteral,
    group_size: OptionalReg[Int] = None,
    has_zp: OptionalReg[Bool] = None,
](
    hidden_state: ManagedTensorSlice[type, 2],
    input_row_offsets: ManagedTensorSlice[DType.uint32, 1],
    weight: ManagedTensorSlice[weight_type, 2],
    kv_collection: PagedKVCacheCollection[
        type,
        *_,
    ],
    layer_idx: Scalar[DType.uint32],
    output: ManagedTensorSlice[type, 2],
    ctx: MojoCallContextPtr,
) raises:
    generic_fused_qkv_matmul_kv_cache_paged_ragged[
        target=target,
        group_size=group_size,
        has_zp=has_zp,
    ](
        managed_tensor_slice_to_ndbuffer(hidden_state),
        managed_tensor_slice_to_ndbuffer(input_row_offsets),
        managed_tensor_slice_to_ndbuffer(weight),
        kv_collection,
        layer_idx,
        managed_tensor_slice_to_ndbuffer(output),
        ctx,
    )


@always_inline
fn generic_fused_qkv_matmul_kv_cache_paged_ragged_kernel_api_bias[
    type: DType,
    weight_type: DType,
    target: StringLiteral,
    group_size: OptionalReg[Int] = None,
    has_zp: OptionalReg[Bool] = None,
](
    hidden_state: ManagedTensorSlice[type, 2],
    input_row_offsets: ManagedTensorSlice[DType.uint32, 1],
    weight: ManagedTensorSlice[weight_type, 2],
    kv_collection: PagedKVCacheCollection[
        type,
        *_,
    ],
    layer_idx: Scalar[DType.uint32],
    output: ManagedTensorSlice[type, 2],
    bias: ManagedTensorSlice[type, 1],
    ctx: MojoCallContextPtr,
) raises:
    generic_fused_qkv_matmul_kv_cache_paged_ragged_bias[
        target=target,
        group_size=group_size,
        has_zp=has_zp,
    ](
        managed_tensor_slice_to_ndbuffer(hidden_state),
        managed_tensor_slice_to_ndbuffer(input_row_offsets),
        managed_tensor_slice_to_ndbuffer(weight),
        kv_collection,
        layer_idx,
        managed_tensor_slice_to_ndbuffer(output),
        managed_tensor_slice_to_ndbuffer(bias),
        ctx,
    )


@compiler.register("mo.fused_qkv_matmul.ragged.paged")
struct Struct_fused_qkv_matmul_padded_ragged:
    @always_inline
    @staticmethod
    fn execute[
        type: DType,
        num_heads: Int,
        head_dim: Int,
        page_size: Int, //,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice[type, 2],
        hidden_state: ManagedTensorSlice[type, 2],
        input_row_offsets: ManagedTensorSlice[DType.uint32, 1],
        weight: ManagedTensorSlice[type, 2],
        kv_collection: PagedKVCacheCollection[
            type,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
            page_size,
        ],
        layer_idx: Scalar[DType.uint32],
        ctx: MojoCallContextPtr,
    ) raises:
        return generic_fused_qkv_matmul_kv_cache_paged_ragged_kernel_api[
            target=target
        ](
            hidden_state,
            input_row_offsets,
            weight,
            kv_collection,
            layer_idx,
            output,
            ctx,
        )


@compiler.register("mo.fused_qkv_matmul.ragged.paged.quantized")
struct Struct_fused_qkv_matmul_padded_ragged_quantized:
    @always_inline
    @staticmethod
    fn execute[
        type: DType,
        weight_type: DType,
        num_heads: Int,
        head_dim: Int,
        group_size: Int,
        has_zp_int: Int,
        page_size: Int, //,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice[type, 2],
        hidden_state: ManagedTensorSlice[type, 2],
        input_row_offsets: ManagedTensorSlice[DType.uint32, 1],
        weight: ManagedTensorSlice[weight_type, 2],
        kv_collection: PagedKVCacheCollection[
            type,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
            page_size,
        ],
        layer_idx: Scalar[DType.uint32],
        ctx: MojoCallContextPtr,
    ) raises:
        # In the group-wise quantization scheme, every `group_size` quantized weights
        # share the same scale. If `has_zp_int` is non-zero, there is also a group-wise
        # zero point that need to be substracted from the quantized weights.
        alias has_zp = True if has_zp_int == 1 else False

        return generic_fused_qkv_matmul_kv_cache_paged_ragged_kernel_api[
            target=target,
            group_size=group_size,
            has_zp=has_zp,
        ](
            hidden_state,
            input_row_offsets,
            weight,
            kv_collection,
            layer_idx,
            output,
            ctx,
        )


@compiler.register("mo.fused_qkv_matmul.ragged.paged.bias")
struct Struct_fused_qkv_matmul_padded_ragged_bias:
    @always_inline
    @staticmethod
    fn execute[
        type: DType,
        num_heads: Int,
        head_dim: Int,
        page_size: Int, //,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice[type, 2],
        hidden_state: ManagedTensorSlice[type, 2],
        input_row_offsets: ManagedTensorSlice[DType.uint32, 1],
        weight: ManagedTensorSlice[type, 2],
        kv_collection: PagedKVCacheCollection[
            type,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
            page_size,
        ],
        layer_idx: Scalar[DType.uint32],
        bias: ManagedTensorSlice[type, 1],
        ctx: MojoCallContextPtr,
    ) raises:
        return generic_fused_qkv_matmul_kv_cache_paged_ragged_kernel_api_bias[
            target=target
        ](
            hidden_state,
            input_row_offsets,
            weight,
            kv_collection,
            layer_idx,
            output,
            bias,
            ctx,
        )


@compiler.register("mo.fused_qkv_matmul.ragged.paged.bias.quantized")
struct Struct_fused_qkv_matmul_padded_ragged_bias_quantized:
    @always_inline
    @staticmethod
    fn execute[
        type: DType,
        weight_type: DType,
        num_heads: Int,
        head_dim: Int,
        group_size: Int,
        has_zp_int: Int,
        page_size: Int, //,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice[type, 2],
        hidden_state: ManagedTensorSlice[type, 2],
        input_row_offsets: ManagedTensorSlice[DType.uint32, 1],
        weight: ManagedTensorSlice[weight_type, 2],
        kv_collection: PagedKVCacheCollection[
            type,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
            page_size,
        ],
        layer_idx: Scalar[DType.uint32],
        bias: ManagedTensorSlice[type, 1],
        ctx: MojoCallContextPtr,
    ) raises:
        # In the group-wise quantization scheme, every `group_size` quantized weights
        # share the same scale. If `has_zp_int` is non-zero, there is also a group-wise
        # zero point that need to be substracted from the quantized weights.
        alias has_zp = True if has_zp_int == 1 else False

        return generic_fused_qkv_matmul_kv_cache_paged_ragged_kernel_api_bias[
            target=target,
            group_size=group_size,
            has_zp=has_zp,
        ](
            hidden_state,
            input_row_offsets,
            weight,
            kv_collection,
            layer_idx,
            output,
            bias,
            ctx,
        )


# ===-----------------------------------------------------------------------===#
# Fused QK RoPE

# Expected kernel name format:
# mo.fused_qk_rope.<padded/ragged>.<continuous_batching/paged>
# ===-----------------------------------------------------------------------===#


@always_inline
fn generic_fused_qk_rope_bshd_continuous_batch_kernel_api[
    type: DType, //,
    *,
    interleaved: Bool,
    target: StringLiteral,
](
    output: ManagedTensorSlice[type, 4],
    q_proj: ManagedTensorSlice[type, 4],
    kv_collection: ContinuousBatchingKVCacheCollection,
    freqs_cis: ManagedTensorSlice[type, 2],
    layer_idx: Scalar[DType.uint32],
    ctx: MojoCallContextPtr,
):
    """Performs a fused RoPE projection for Q and K projections.

    We have a manually fused QKV projection with mo.opaque types in our Llama model.
    Due to a limitation in custom op definitions, we can't declare both a tensor
    and opaque type as output from a custom kernel. This requires us to only note
    Q_proj as an output from the QKV projection. If we immediately follow the
    QKV proj kernel with a RoPE kernel applied to K, we'll get a race condition
    because the graph compiler doesn't know about the dependency between these
    kernels in the graph definition. Here we fuse the RoPE kernel applied to
    Q_proj with K_proj, so K_proj RoPE is only excuted after QKV completes.
    """
    generic_fused_qk_rope_bshd_continuous_batch[
        interleaved=interleaved, target=target
    ](
        managed_tensor_slice_to_ndbuffer(q_proj),
        kv_collection,
        managed_tensor_slice_to_ndbuffer(freqs_cis),
        layer_idx,
        managed_tensor_slice_to_ndbuffer(output),
        ctx,
    )


@compiler.register("mo.fused_qk_rope.padded.continuous_batching")
struct Struct_fused_qk_rope_padded_continuous_batching[interleaved: Bool]:
    @always_inline
    @staticmethod
    fn execute[
        type: DType, num_heads: Int, head_dim: Int, //, target: StringLiteral
    ](
        output: ManagedTensorSlice[type, 4],
        q_proj: ManagedTensorSlice[type, 4],
        kv_collection: ContinuousBatchingKVCacheCollection[
            type,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
        ],
        freqs_cis: ManagedTensorSlice[type, 2],
        layer_idx: Scalar[DType.uint32],
        ctx: MojoCallContextPtr,
    ) raises:
        generic_fused_qk_rope_bshd_continuous_batch_kernel_api[
            interleaved=interleaved, target=target
        ](
            output,
            q_proj,
            kv_collection,
            freqs_cis,
            layer_idx,
            ctx,
        )


# ===-----------------------------------------------------------------------===#
# Fused QK Rope Ragged
# ===-----------------------------------------------------------------------===#


@always_inline
fn generic_fused_qk_rope_bshd_continuous_batch_ragged_kernel_api[
    type: DType, //, *, interleaved: Bool, target: StringLiteral
](
    output: ManagedTensorSlice[type, 3],
    q_proj: ManagedTensorSlice[type, 3],
    input_row_offsets: ManagedTensorSlice[DType.uint32, 1],
    kv_collection: ContinuousBatchingKVCacheCollection,
    freqs_cis: ManagedTensorSlice[type, 2],
    layer_idx: Scalar[DType.uint32],
    ctx: MojoCallContextPtr,
):
    generic_fused_qk_rope_bshd_continous_batch_ragged[
        interleaved=interleaved, target=target
    ](
        managed_tensor_slice_to_ndbuffer(q_proj),
        managed_tensor_slice_to_ndbuffer(input_row_offsets),
        kv_collection,
        managed_tensor_slice_to_ndbuffer(freqs_cis),
        layer_idx,
        managed_tensor_slice_to_ndbuffer(output),
        ctx,
    )


@compiler.register("mo.fused_qk_rope.ragged.continuous_batching")
struct Struct_fused_qk_rope_bshd_continuous_batch_ragged[interleaved: Bool]:
    @always_inline
    @staticmethod
    fn execute[
        type: DType, num_heads: Int, head_dim: Int, //, target: StringLiteral
    ](
        output: ManagedTensorSlice[type, 3],
        q_proj: ManagedTensorSlice[type, 3],
        input_row_offsets: ManagedTensorSlice[DType.uint32, 1],
        kv_collection: ContinuousBatchingKVCacheCollection[
            type,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
        ],
        freqs_cis: ManagedTensorSlice[type, 2],
        layer_idx: Scalar[DType.uint32],
        ctx: MojoCallContextPtr,
    ) raises:
        generic_fused_qk_rope_bshd_continuous_batch_ragged_kernel_api[
            interleaved=interleaved, target=target
        ](
            output,
            q_proj,
            input_row_offsets,
            kv_collection,
            freqs_cis,
            layer_idx,
            ctx,
        )


@always_inline
fn generic_fused_qk_rope_bshd_paged_ragged_kernel_api[
    type: DType, //,
    *,
    interleaved: Bool,
    target: StringLiteral,
](
    q_proj: ManagedTensorSlice[type, 3],
    input_row_offsets: ManagedTensorSlice[DType.uint32, 1],
    kv_collection: PagedKVCacheCollection[
        type,
        *_,
    ],
    freqs_cis: ManagedTensorSlice[type, 2],
    layer_idx: Scalar[DType.uint32],
    output: ManagedTensorSlice[type, 3],
    context: MojoCallContextPtr = MojoCallContextPtr(),
):
    generic_fused_qk_rope_bshd_paged_ragged[
        interleaved=interleaved, target=target
    ](
        managed_tensor_slice_to_ndbuffer(q_proj),
        managed_tensor_slice_to_ndbuffer(input_row_offsets),
        kv_collection,
        managed_tensor_slice_to_ndbuffer(freqs_cis),
        layer_idx,
        managed_tensor_slice_to_ndbuffer(output),
        context,
    )


@compiler.register("mo.fused_qk_rope.ragged.paged")
struct Struct_fused_qk_rope_ragged_paged[interleaved: Bool]:
    @always_inline
    @staticmethod
    fn execute[
        type: DType,
        num_heads: Int,
        head_dim: Int,
        page_size: Int, //,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice[type, 3],
        q_proj: ManagedTensorSlice[type, 3],
        input_row_offsets: ManagedTensorSlice[DType.uint32, 1],
        kv_collection: PagedKVCacheCollection[
            type,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
            page_size,
        ],
        freqs_cis: ManagedTensorSlice[type, 2],
        layer_idx: Scalar[DType.uint32],
        context: MojoCallContextPtr = MojoCallContextPtr(),
    ):
        generic_fused_qk_rope_bshd_paged_ragged_kernel_api[
            interleaved=interleaved, target=target
        ](
            q_proj,
            input_row_offsets,
            kv_collection,
            freqs_cis,
            layer_idx,
            output,
            context,
        )


# ===-----------------------------------------------------------------------===#
# MHA
#
# Expected kernel name format:
# mo.mha.<padded/ragged>.<continuous_batching/paged>.<MASK_TYPE>.<POS_TYPE>
# ===-----------------------------------------------------------------------===#


@always_inline
fn generic_flash_attention_kv_cache_continuous_batch_kernel_api[
    target: StringLiteral, type: DType
](
    output: ManagedTensorSlice[type, 4],
    q: ManagedTensorSlice[type, 4],
    kv_collection: ContinuousBatchingKVCacheCollection,
    layer_idx: Scalar[DType.uint32],
    mask: ManagedTensorSlice[type],
    valid_lengths: ManagedTensorSlice[DType.uint32, 1],
    scale: Scalar[DType.float32],
    context: MojoCallContextPtr,
) raises:
    generic_flash_attention_kv_cache_continuous_batch[target](
        managed_tensor_slice_to_ndbuffer(q),
        kv_collection,
        layer_idx,
        managed_tensor_slice_to_ndbuffer(mask),
        managed_tensor_slice_to_ndbuffer(valid_lengths),
        scale,
        managed_tensor_slice_to_ndbuffer(output),
        context,
    )


@compiler.register("mo.mha.padded.continuous_batching.tensor_mask.no_pos")
struct Struct_mha_padded_continuous_batching_tensor_mask_no_pos:
    @always_inline
    @staticmethod
    fn execute[
        type: DType, num_heads: Int, head_dim: Int, //, target: StringLiteral
    ](
        output: ManagedTensorSlice[type, 4],
        q: ManagedTensorSlice[type, 4],
        kv_collection: ContinuousBatchingKVCacheCollection[
            type,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
        ],
        layer_idx: Scalar[DType.uint32],
        mask: ManagedTensorSlice[type],
        valid_lengths: ManagedTensorSlice[DType.uint32, 1],
        scale: Scalar[DType.float32],
        context: MojoCallContextPtr,
    ) raises:
        generic_flash_attention_kv_cache_continuous_batch_kernel_api[target](
            output,
            q,
            kv_collection,
            layer_idx,
            mask,
            valid_lengths,
            scale,
            context,
        )


@always_inline
fn generic_flash_attention_kv_cache_causal_mask_continuous_batch_kernel_api[
    target: StringLiteral, type: DType
](
    output: ManagedTensorSlice[type, 4],
    q: ManagedTensorSlice[type, 4],
    kv_collection: ContinuousBatchingKVCacheCollection,
    layer_idx: Scalar[DType.uint32],
    valid_lengths: ManagedTensorSlice[DType.uint32, 1],
    scale: Scalar[DType.float32],
    context: MojoCallContextPtr,
) raises:
    generic_flash_attention_kv_cache_causal_mask_continuous_batch[target](
        managed_tensor_slice_to_ndbuffer(q),
        kv_collection,
        layer_idx,
        managed_tensor_slice_to_ndbuffer(valid_lengths),
        scale,
        managed_tensor_slice_to_ndbuffer(output),
        context,
    )


@compiler.register("mo.mha.padded.continuous_batching.causal_mask.no_pos")
struct Struct_mha_padded_continuous_batching_causal_mask_no_pos:
    @always_inline
    @staticmethod
    fn execute[
        type: DType, num_heads: Int, head_dim: Int, //, target: StringLiteral
    ](
        output: ManagedTensorSlice[type, 4],
        q: ManagedTensorSlice[type, 4],
        kv_collection: ContinuousBatchingKVCacheCollection[
            type,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
        ],
        layer_idx: Scalar[DType.uint32],
        valid_lengths: ManagedTensorSlice[DType.uint32, 1],
        scale: Scalar[DType.float32],
        context: MojoCallContextPtr,
    ) raises:
        generic_flash_attention_kv_cache_causal_mask_continuous_batch_kernel_api[
            target
        ](
            output, q, kv_collection, layer_idx, valid_lengths, scale, context
        )


@always_inline
fn generic_flash_attention_kv_cache_causal_mask_cont_batch_ragged_kernel_api[
    type: DType, //,
    target: StringLiteral,
](
    q: ManagedTensorSlice[type, 3],
    input_row_offsets: ManagedTensorSlice[DType.uint32, 1],
    kv_collection: ContinuousBatchingKVCacheCollection,
    layer_idx: Scalar[DType.uint32],
    scale: Scalar[DType.float32],
    output: ManagedTensorSlice[type, 3],
    context: MojoCallContextPtr,
) raises:
    generic_flash_attention_kv_cache_causal_mask_cont_batch_ragged[target](
        managed_tensor_slice_to_ndbuffer(q),
        managed_tensor_slice_to_ndbuffer(input_row_offsets),
        kv_collection,
        layer_idx,
        scale,
        managed_tensor_slice_to_ndbuffer(output),
        context,
    )


@always_inline
fn generic_flash_attention_kv_cache_alibi_mask_cont_batch_ragged_kernel_api[
    type: DType, //,
    target: StringLiteral,
](
    q: ManagedTensorSlice[type, 3],
    input_row_offsets: ManagedTensorSlice[DType.uint32, 1],
    kv_collection: ContinuousBatchingKVCacheCollection,
    layer_idx: Scalar[DType.uint32],
    scale: Scalar[DType.float32],
    output: ManagedTensorSlice[type, 3],
    context: MojoCallContextPtr,
) raises:
    generic_flash_attention_kv_cache_alibi_mask_cont_batch_ragged[target](
        managed_tensor_slice_to_ndbuffer(q),
        managed_tensor_slice_to_ndbuffer(input_row_offsets),
        kv_collection,
        layer_idx,
        scale,
        managed_tensor_slice_to_ndbuffer(output),
        context,
    )


@compiler.register("mo.mha.ragged.continuous_batching.causal_mask.no_pos")
struct Struct_mha_ragged_continuous_batching_causal_mask_no_pos:
    @always_inline
    @staticmethod
    fn execute[
        type: DType, num_heads: Int, head_dim: Int, //, target: StringLiteral
    ](
        output: ManagedTensorSlice[type, 3],
        q: ManagedTensorSlice[type, 3],
        input_row_offsets: ManagedTensorSlice[DType.uint32, 1],
        kv_collection: ContinuousBatchingKVCacheCollection[
            type,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
        ],
        layer_idx: Scalar[DType.uint32],
        scale: Scalar[DType.float32],
        context: MojoCallContextPtr,
    ) raises:
        generic_flash_attention_kv_cache_causal_mask_cont_batch_ragged_kernel_api[
            target
        ](
            q,
            input_row_offsets,
            kv_collection,
            layer_idx,
            scale,
            output,
            context,
        )


@compiler.register("mo.mha.ragged.continuous_batching.causal_mask.alibi_pos")
struct Struct_mha_ragged_continuous_batching_causal_mask_alibi_pos:
    @always_inline
    @staticmethod
    fn execute[
        type: DType, num_heads: Int, head_dim: Int, //, target: StringLiteral
    ](
        output: ManagedTensorSlice[type, 3],
        q: ManagedTensorSlice[type, 3],
        input_row_offsets: ManagedTensorSlice[DType.uint32, 1],
        kv_collection: ContinuousBatchingKVCacheCollection[
            type,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
        ],
        layer_idx: Scalar[DType.uint32],
        scale: Scalar[DType.float32],
        context: MojoCallContextPtr,
    ) raises:
        generic_flash_attention_kv_cache_alibi_mask_cont_batch_ragged_kernel_api[
            target
        ](
            q,
            input_row_offsets,
            kv_collection,
            layer_idx,
            scale,
            output,
            context,
        )


@always_inline
fn generic_flash_attention_kv_cache_causal_mask_paged_ragged_kernel_api[
    type: DType,
    target: StringLiteral,
](
    q: ManagedTensorSlice[type, 3],
    input_row_offsets: ManagedTensorSlice[DType.uint32, 1],
    kv_collection: PagedKVCacheCollection[type, *_],
    layer_idx: Scalar[DType.uint32],
    scale: Scalar[DType.float32],
    output: ManagedTensorSlice[type, 3],
    context: MojoCallContextPtr,
) raises:
    generic_flash_attention_kv_cache_causal_mask_paged_ragged[target=target](
        managed_tensor_slice_to_ndbuffer(q),
        managed_tensor_slice_to_ndbuffer(input_row_offsets),
        kv_collection,
        layer_idx,
        scale,
        managed_tensor_slice_to_ndbuffer(output),
        context,
    )


@compiler.register("mo.mha.ragged.paged.causal_mask.no_pos")
struct Struct_mha_ragged_paged_causal_mask_no_pos:
    @staticmethod
    @always_inline
    fn execute[
        type: DType,
        num_heads: Int,
        head_dim: Int,
        page_size: Int, //,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice[type, 3],
        q: ManagedTensorSlice[type, 3],
        input_row_offsets: ManagedTensorSlice[DType.uint32, 1],
        kv_collection: PagedKVCacheCollection[
            type,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
            page_size,
        ],
        layer_idx: Scalar[DType.uint32],
        scale: Scalar[DType.float32],
        context: MojoCallContextPtr,
    ) raises:
        generic_flash_attention_kv_cache_causal_mask_paged_ragged_kernel_api[
            target=target
        ](
            q,
            input_row_offsets,
            kv_collection,
            layer_idx,
            scale,
            output,
            context,
        )


# ===-----------------------------------------------------------------------===#
# Cross attention
#
# Expected kernel name format:
# mo.cross_attention.<padded/ragged>.<continuous_batching/paged>.<MASK_TYPE>.<POS_TYPE>
# ===-----------------------------------------------------------------------===#


@always_inline
fn generic_cross_attention_kv_cache_null_mask_cont_batch_ragged_kernel_api[
    type: DType, //, target: StringLiteral
](
    output: ManagedTensorSlice[type, 3],
    q: ManagedTensorSlice[type, 3],
    q_input_row_offsets: ManagedTensorSlice[DType.uint32, 1],
    q_max_seq_len: ManagedTensorSlice[DType.uint32, 1],
    kv_input_row_offsets: ManagedTensorSlice[DType.uint32, 1],
    kv_collection: ContinuousBatchingKVCacheCollection,
    layer_idx: UInt32,
    scale: Float32,
    context: MojoCallContextPtr,
) raises:
    generic_cross_attention_kv_cache_null_mask_cont_batch_ragged[target=target](
        managed_tensor_slice_to_ndbuffer(q),
        managed_tensor_slice_to_ndbuffer(q_input_row_offsets),
        managed_tensor_slice_to_ndbuffer(q_max_seq_len),
        managed_tensor_slice_to_ndbuffer(kv_input_row_offsets),
        kv_collection,
        layer_idx,
        scale,
        managed_tensor_slice_to_ndbuffer(output),
        context,
    )


@compiler.register(
    "mo.cross_attention.ragged.continuous_batching.null_mask.no_pos"
)
struct Struct_cross_attention_ragged_continuous_batching_null_mask_no_pos:
    @always_inline
    @staticmethod
    fn execute[
        type: DType, num_heads: Int, head_dim: Int, //, target: StringLiteral
    ](
        output: ManagedTensorSlice[type, 3],
        q: ManagedTensorSlice[type, 3],
        q_input_row_offsets: ManagedTensorSlice[DType.uint32, 1],
        q_max_seq_len: ManagedTensorSlice[DType.uint32, 1],
        kv_input_row_offsets: ManagedTensorSlice[DType.uint32, 1],
        kv_collection: ContinuousBatchingKVCacheCollection[
            type, KVCacheStaticParams(num_heads=num_heads, head_size=head_dim)
        ],
        layer_idx: UInt32,
        scale: Float32,
        context: MojoCallContextPtr,
    ) raises:
        generic_cross_attention_kv_cache_null_mask_cont_batch_ragged_kernel_api[
            target=target
        ](
            output,
            q,
            q_input_row_offsets,
            q_max_seq_len,
            kv_input_row_offsets,
            kv_collection,
            layer_idx,
            scale,
            context,
        )


# ===-----------------------------------------------------------------------===#
# KV Collection Constructors (Ctor)
#
# Expected kernel name format:
# mo.kv_collection_ctor.<continuous_batching/paged>
# ===-----------------------------------------------------------------------===#


@always_inline
fn generic_get_continuous_cache_kernel_api[
    type: DType,
    kv_params: KVCacheStaticParams,
](
    blocks: ManagedTensorSlice[type, 6],
    cache_lengths: ManagedTensorSlice[DType.uint32, 1],
    lookup_table: ManagedTensorSlice[DType.uint32, 1],
    max_lengths: ManagedTensorSlice[DType.uint32, 2],
) -> ContinuousBatchingKVCacheCollection[
    type,
    kv_params,
]:
    return generic_get_continuous_cache[type, kv_params](
        managed_tensor_slice_to_ndbuffer(blocks),
        managed_tensor_slice_to_ndbuffer(cache_lengths),
        managed_tensor_slice_to_ndbuffer(lookup_table),
        managed_tensor_slice_to_ndbuffer(max_lengths),
    )


@compiler.register(
    "mo.kv_collection_ctor.continuous_batching", num_dps_outputs=0
)
struct Struct_kv_collection_ctor_continuous_batching:
    @always_inline
    @staticmethod
    fn execute[
        type: DType, num_heads: Int, head_dim: Int, target: StringLiteral
    ](
        blocks: ManagedTensorSlice[type, 6],
        cache_lengths: ManagedTensorSlice[DType.uint32, 1],
        lookup_table: ManagedTensorSlice[DType.uint32, 1],
        max_lengths: ManagedTensorSlice[DType.uint32, 2],
    ) -> ContinuousBatchingKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads, head_dim),
    ]:
        return generic_get_continuous_cache[
            kv_params = KVCacheStaticParams(num_heads, head_dim)
        ](
            managed_tensor_slice_to_ndbuffer(blocks),
            managed_tensor_slice_to_ndbuffer(cache_lengths),
            managed_tensor_slice_to_ndbuffer(lookup_table),
            managed_tensor_slice_to_ndbuffer(max_lengths),
        )


# ===-----------------------------------------------------------------------===#
# LayoutTransforms
# ===-----------------------------------------------------------------------===#


# TODO(GEX-1492): use filter_rank+1 instead of packed_filter_rank
fn layout_transform_conv_transpose_filter_common[
    type: DType,
    filter_rank: Int,
    packed_filter_rank: Int,
](
    packed_filter: ManagedTensorSlice[type, packed_filter_rank],
    filter: ManagedTensorSlice[type, filter_rank],
):
    constrained[filter_rank + 1 == packed_filter_rank]()
    # last param is num_groups which is currently not an available
    # arg for the MO level op
    _pack_conv_transpose_filter(
        managed_tensor_slice_to_ndbuffer(filter),
        managed_tensor_slice_to_ndbuffer(packed_filter),
        1,
    )


@compiler.register("layout_transform_RSFC_to_FRSCf")
struct LayoutTransformRSFC2FRSCf:
    @always_inline
    @staticmethod
    fn execute[
        type: DType, filter_rank: Int, packed_filter_rank: Int
    ](
        packed_filter: ManagedTensorSlice[type, packed_filter_rank],
        filter: ManagedTensorSlice[type, filter_rank],
    ):
        layout_transform_conv_transpose_filter_common(packed_filter, filter)


@compiler.register("layout_transform_QRSFC_to_FQRSCf")
struct LayoutTransformQRSFC2FQRSCf:
    @always_inline
    @staticmethod
    fn execute[
        type: DType, filter_rank: Int, packed_filter_rank: Int
    ](
        packed_filter: ManagedTensorSlice[type, packed_filter_rank],
        filter: ManagedTensorSlice[type, filter_rank],
    ):
        layout_transform_conv_transpose_filter_common(packed_filter, filter)


@compiler.register("pack_conv_filter_shape")
struct PackConvFilterShape:
    @always_inline
    @staticmethod
    fn execute() raises:
        raise Error("Only meant to be used for shape function!")

    @always_inline
    @staticmethod
    fn shape[
        rank: Int,
        filter_type: DType,
        input_shape: DimList,
        filter_shape: DimList,
        output_shape: DimList,
        strides: DimList,
        dilations: DimList,
        paddings: DimList,
        num_groups: Int,
        _synchronous: Bool,
    ](filter_buf: ManagedTensorSlice[filter_type, rank]) -> IndexList[rank + 1]:
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
            _synchronous: If True, then reduction is run sync with 1 thread.

        Args:
            filter_buf: The filter to be packed.

        Returns:
            The output shape.
        """

        return pack_filter_shape_conv[
            filter_type,
            input_shape,
            filter_shape,
            output_shape,
            strides,
            dilations,
            paddings,
            num_groups,
            _synchronous,
        ](managed_tensor_slice_to_ndbuffer(filter_buf))


@compiler.register("pack_conv_transpose_filter_shape")
struct PackConvTransposeFilterShape:
    @always_inline
    @staticmethod
    fn execute() raises:
        raise Error("Only meant to be used for shape function!")

    @always_inline
    @staticmethod
    fn shape[
        rank: Int,
        filter_type: DType,
        _synchronous: Bool,
    ](filter_buf: NDBuffer[filter_type, rank]) -> IndexList[rank + 1]:
        return pack_filter_shape_conv_transpose(filter_buf, 1)


# Wrapper that take `num_groups` as a parameter.
# This is required unti `mo.layout.transform` passes `num_groups` as a runtime
# value.
fn layout_transform_conv_filter_common[
    type: DType, filter_rank: Int, packed_rank: Int, num_groups: Int
](
    packed_filter: ManagedTensorSlice[type, packed_rank],
    filter: ManagedTensorSlice[type, filter_rank],
):
    constrained[packed_rank == filter_rank + 1]()

    # last param is num_groups which is currently not an available
    # arg for the MO level op
    _pack_conv_filter(
        managed_tensor_slice_to_ndbuffer(filter),
        managed_tensor_slice_to_ndbuffer(packed_filter),
        num_groups,
    )


@compiler.register("layout_transform_QRSCF_to_FQRSCf")
struct LayoutTransformQRSCF2FQRSCf:
    @always_inline
    @staticmethod
    fn execute[
        type: DType, filter_rank: Int, packed_rank: Int, num_groups: Int
    ](
        packed_filter: ManagedTensorSlice[type, packed_rank],
        filter: ManagedTensorSlice[type, filter_rank],
    ):
        layout_transform_conv_filter_common[num_groups=num_groups](
            packed_filter, filter
        )


@compiler.register("layout_transform_RSCF_to_FRSCf")
struct LayoutTransformRSCF2FRSCf:
    @always_inline
    @staticmethod
    fn execute[
        type: DType, filter_rank: Int, packed_rank: Int, num_groups: Int
    ](
        packed_filter: ManagedTensorSlice[type, packed_rank],
        filter: ManagedTensorSlice[type, filter_rank],
    ):
        layout_transform_conv_filter_common[num_groups=num_groups](
            packed_filter, filter
        )


@compiler.register("layout_transform_KN_to_KNkni")
struct LayoutTransformMatmulKN2KNkni:
    @always_inline
    @staticmethod
    fn execute[
        a_type: DType,
        a_shape: DimList,
        b_type: DType,
        b_shape: DimList,
        c_type: DType,
        c_shape: DimList,
    ](
        output_buffer: ManagedTensorSlice[b_type, 2],
        b_input: ManagedTensorSlice[b_type, 2],
    ) raises:
        # NOTE `get_kernel_type` expects `m == 0` for dynamic M.
        var kernel_type_m = 0

        @parameter
        if a_shape.at[0]().has_value():
            kernel_type_m = a_shape.at[0]().get()
        _pack_b_ndbuffer_impl[
            a_type,
            a_shape,
            b_type,
            b_shape,
            c_type,
            c_shape,
            transposed=False,
        ](
            managed_tensor_slice_to_ndbuffer(b_input),
            managed_tensor_slice_to_ndbuffer(output_buffer),
            kernel_type_m,
        )


@compiler.register("layout_transform_NK_to_KNkni")
struct LayoutTransformMatmulNK2KNkni:
    @always_inline
    @staticmethod
    fn execute[
        a_type: DType,
        a_shape: DimList,
        b_type: DType,
        b_shape: DimList,
        c_type: DType,
        c_shape: DimList,
    ](
        output_buffer: ManagedTensorSlice[b_type, 2],
        b_input: ManagedTensorSlice[b_type, 2],
    ) raises:
        # NOTE `get_kernel_type` expects `m == 0` for dynamic M.
        var kernel_type_m = 0

        @parameter
        if a_shape.at[0]().has_value():
            kernel_type_m = a_shape.at[0]().get()
        _pack_b_ndbuffer_impl[
            a_type,
            a_shape,
            b_type,
            b_shape,
            c_type,
            c_shape,
            transposed=True,
        ](
            managed_tensor_slice_to_ndbuffer(b_input),
            managed_tensor_slice_to_ndbuffer(output_buffer),
            kernel_type_m,
        )


@compiler.register("pack_matmul_b_shape_func")
struct PackMatmulBShapeFunc:
    @always_inline
    @staticmethod
    fn execute() raises:
        raise Error("Only meant to be used for shape function!")

    @always_inline
    @staticmethod
    fn shape[
        a_type: DType,
        a_shape: DimList,
        b_type: DType,
        b_shape: DimList,
        c_type: DType,
        c_shape: DimList,
        transpose_in_0: Bool,
        _synchronous: Bool,
    ](b_input: ManagedTensorSlice[b_type, 2]) -> IndexList[2]:
        return pack_matmul_b_shape_func[
            a_type,
            a_shape,
            b_type,
            b_shape,
            c_type,
            c_shape,
            transpose_in_0,
            _synchronous,
        ](managed_tensor_slice_to_ndbuffer(b_input))


# ===-----------------------------------------------------------------------===#
# RMSNorm
#
# Expected kernel name format:
# mo.rms_norm_kv_cache.<padded/ragged>.<continuous_batching/paged>
# ===-----------------------------------------------------------------------===#


@compiler.register(
    "mo.rms_norm_kv_cache.ragged.continuous_batching",
    num_dps_outputs=0,
)
struct Struct_rms_norm_kv_cache_ragged_continuous_batching:
    @always_inline
    @staticmethod
    fn execute[
        type: DType, num_heads: Int, head_dim: Int, //, target: StringLiteral
    ](
        kv_collection: ContinuousBatchingKVCacheCollection[
            type,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
        ],
        gamma: ManagedTensorSlice[type, 1],
        epsilon: Scalar[type],
        layer_idx: Scalar[DType.uint32],
        total_seq_len: Scalar[DType.uint32],
        input_row_offsets: ManagedTensorSlice[DType.uint32, 1],
        context: MojoCallContextPtr,
    ) raises:
        rms_norm_kv_cache_ragged_continuous_batching[target=target](
            kv_collection,
            managed_tensor_slice_to_ndbuffer(gamma),
            epsilon,
            layer_idx,
            total_seq_len,
            managed_tensor_slice_to_ndbuffer(input_row_offsets),
            context,
        )


# ===-----------------------------------------------------------------------===#
# Print KV Cache
#
# Expected kernel name format:
# mo.print_kv_cache.<continuous_batching/paged>
# ===-----------------------------------------------------------------------===#


fn print_kv_cache_cont_batch_generic_kernel_api[
    type: DType, //, target: StringLiteral
](
    valid_lengths: ManagedTensorSlice[DType.uint32, 1],
    kv_collection: ContinuousBatchingKVCacheCollection[type, _],
    layer_idx: Scalar[DType.uint32],
    is_print_compact: ManagedTensorSlice[DType.bool, 1],
    context: MojoCallContextPtr,
) raises:
    @parameter
    if is_gpu[target]():
        print_kv_cache_cont_batch_generic_gpu[target](
            managed_tensor_slice_to_ndbuffer(valid_lengths),
            kv_collection,
            layer_idx,
            is_print_compact[0],
            context,
        )
    elif is_cpu[target]():
        print_kv_cache_cont_batch_generic_cpu[target](
            managed_tensor_slice_to_ndbuffer(valid_lengths),
            kv_collection,
            layer_idx,
            is_print_compact[0],
            context,
        )


fn print_kv_cache_paged_generic_kernel_api[
    type: DType, //,
    target: StringLiteral,
](
    valid_lengths: ManagedTensorSlice[DType.uint32, 1],
    kv_collection: PagedKVCacheCollection[type, *_],
    layer_idx: Scalar[DType.uint32],
    is_print_compact: ManagedTensorSlice[DType.bool, 1],
    context: MojoCallContextPtr,
) raises:
    @parameter
    if is_gpu[target]():
        print_kv_cache_paged_generic_gpu[target](
            managed_tensor_slice_to_ndbuffer(valid_lengths),
            kv_collection,
            layer_idx,
            True,
            context,
        )
    elif is_cpu[target]():
        print_kv_cache_paged_generic_cpu[target](
            managed_tensor_slice_to_ndbuffer(valid_lengths),
            kv_collection,
            layer_idx,
            is_print_compact[0],
            context,
        )


@compiler.register("mo.print_kv_cache.paged", num_dps_outputs=0)
struct Struct_print_kv_cache_paged:
    @always_inline
    @staticmethod
    fn execute[
        type: DType,
        num_heads: Int,
        head_dim: Int,
        page_size: Int, //,
        target: StringLiteral,
    ](
        valid_lengths: ManagedTensorSlice[DType.uint32, 1],
        kv_collection: PagedKVCacheCollection[
            type,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
            page_size,
        ],
        layer_idx: Scalar[DType.uint32],
        is_print_compact: ManagedTensorSlice[DType.bool, 1],
        context: MojoCallContextPtr,
    ) raises:
        print_kv_cache_paged_generic_kernel_api[target](
            valid_lengths,
            kv_collection,
            layer_idx,
            is_print_compact,
            context,
        )


@compiler.register("mo.print_kv_cache.continuous_batching", num_dps_outputs=0)
struct Struct_print_kv_cache_continuous_batching:
    @always_inline
    @staticmethod
    fn execute[
        type: DType, num_heads: Int, head_dim: Int, //, target: StringLiteral
    ](
        valid_lengths: ManagedTensorSlice[DType.uint32, 1],
        kv_collection: ContinuousBatchingKVCacheCollection[
            type,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
        ],
        layer_idx: Scalar[DType.uint32],
        is_print_compact: ManagedTensorSlice[DType.bool, 1],
        context: MojoCallContextPtr,
    ) raises:
        print_kv_cache_cont_batch_generic_kernel_api[target](
            valid_lengths,
            kv_collection,
            layer_idx,
            is_print_compact,
            context,
        )


# ===-----------------------------------------------------------------------===#
# KV Collection Constructors (Ctor)
#
# Expected kernel name format:
# mo.kv_collection_ctor.<continuous_batching/paged>
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.kv_collection_ctor.paged", num_dps_outputs=0)
struct Struct_kv_collection_ctor_paged:
    @always_inline
    @staticmethod
    fn execute[
        type: DType,
        num_heads: Int,
        head_dim: Int,
        page_size: Int,
        target: StringLiteral,
    ](
        blocks: ManagedTensorSlice[type, 6],
        cache_lengths: ManagedTensorSlice[DType.uint32, 1],
        lookup_table: ManagedTensorSlice[DType.uint32, 2],
        max_lengths: ManagedTensorSlice[DType.uint32, 2],
    ) -> PagedKVCacheCollection[
        type, KVCacheStaticParams(num_heads, head_dim), page_size
    ]:
        return generic_get_paged_cache[
            kv_params = KVCacheStaticParams(num_heads, head_dim),
            page_size=page_size,
        ](
            managed_tensor_slice_to_ndbuffer(blocks),
            managed_tensor_slice_to_ndbuffer(cache_lengths),
            managed_tensor_slice_to_ndbuffer(lookup_table),
            managed_tensor_slice_to_ndbuffer(max_lengths),
        )


# ===-----------------------------------------------------------------------===#
# Matmul KV cache
#
# Expected kernel name format:
# mo.kv_matmul.ragged.<continuous_batching/paged>
# ===-----------------------------------------------------------------------===#


@compiler.register(
    "mo.kv_matmul.ragged.continuous_batching",
    num_dps_outputs=0,
)
struct Struct_kv_matmul_ragged_continuous_batching:
    @staticmethod
    @always_inline
    fn execute[
        type: DType, num_heads: Int, head_dim: Int, //, target: StringLiteral
    ](
        hidden_state: ManagedTensorSlice[type, 2],
        input_row_offsets: ManagedTensorSlice[DType.uint32, 1],
        weight: ManagedTensorSlice[type, 2],
        kv_collection: ContinuousBatchingKVCacheCollection[
            type,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
        ],
        layer_idx: Scalar[DType.uint32],
        ctx: MojoCallContextPtr,
    ) raises:
        kv_matmul_ragged_continuous_batching[target=target](
            managed_tensor_slice_to_ndbuffer(hidden_state),
            managed_tensor_slice_to_ndbuffer(input_row_offsets),
            managed_tensor_slice_to_ndbuffer(weight),
            kv_collection,
            layer_idx,
            ctx,
        )


# ===-----------------------------------------------------------------------===#
# Misc Operations
# ===-----------------------------------------------------------------------===#


@compiler.register("topk_fused_sampling")
struct Struct_topk_fused_sampling:
    @staticmethod
    @always_inline
    fn execute[
        type: DType,
        rank: Int,
        out_idx_type: DType,
        target: StringLiteral,
    ](
        out_idxs: ManagedTensorSlice[out_idx_type, rank],
        K: Scalar,
        input: ManagedTensorSlice[type, rank],
        ctx: MojoCallContextPtr,
    ) raises:
        constrained[is_valid_target[target](), "not a valid target"]()

        var input_buf = managed_tensor_slice_to_ndbuffer(input)
        var out_idxs_buf = managed_tensor_slice_to_ndbuffer(out_idxs)
        with Trace[TraceLevel.OP, target=target]("topk_fused_sampling"):

            @parameter
            if is_cpu[target]():
                # When top_k == 1, argmax is equivalent to our topk_fused_sampling with k == 1
                # However, switching to just using our topk_fused_sampling leads to a -37% perf
                # drop in q4_k benchmarking for llama 3.
                if K == 1:
                    argmax(input_buf, rank - 1, out_idxs_buf)
                    return
                _topk_fused_sampling_cpu(Int(K), input_buf, out_idxs_buf)
            else:
                var cuda_ctx = ctx.get_device_context()
                _topk_fused_sampling_gpu(
                    cuda_ctx,
                    Int(K),
                    input_buf,
                    out_idxs_buf,
                )


@compiler.register("swishGLU")
struct Struct_swishGLU:
    @staticmethod
    @always_inline
    fn execute[
        target: StringLiteral,
    ](
        c: ManagedTensorSlice[rank=2],
        a: ManagedTensorSlice[rank=2],
        b0: ManagedTensorSlice[rank=2],
        b1: ManagedTensorSlice[b0.type, 2],
        ctx: MojoCallContextPtr,
    ) raises:
        swishGLU[target=target](
            managed_tensor_slice_to_ndbuffer(a),
            managed_tensor_slice_to_ndbuffer(b0),
            managed_tensor_slice_to_ndbuffer(b1),
            managed_tensor_slice_to_ndbuffer(c),
            ctx,
        )


@compiler.register("mo.distributed.allreduce.sum")
struct DistributedAllReduceSum:
    @staticmethod
    @always_inline
    fn execute[
        type: DType,
        rank: Int,
        target: StringLiteral,
    ](
        outputs: VariadicTensors[type, rank, io_spec=IOUnknown, *_],
        inputs: VariadicTensors[type, rank, io_spec=IOUnknown, *_],
        dev_ctxs: StaticTuple[DeviceContextPtr, *_],
    ) raises:
        # Stub for now
        outputs[0][0] = inputs[0][0]
        print(
            "Hello! You should not run this kernel: `DistributedAllReduceSum`"
        )


fn _check_signal_buffer_size(
    signal_buffer: ManagedTensorSlice[DType.uint8, rank=1]
) raises:
    if signal_buffer.size() < sizeof[Signal]():
        raise Error(
            "expected signal buffer to be at least ",
            sizeof[Signal](),
            " bytes, but got ",
            signal_buffer.size(),
        )


@compiler.register("mo.distributed.allreduce.2gpu.sum")
struct DistributedAllReduceSum2Devices:
    alias num_devices = 2

    @staticmethod
    @always_inline
    fn execute[
        type: DType,
        rank: Int,
        target: StringLiteral,
    ](
        outputs: VariadicTensors[
            type, rank, size = Self.num_devices, io_spec=IOUnknown
        ],
        signal_buffer0: ManagedTensorSlice[DType.uint8, rank=1],
        signal_buffer1: ManagedTensorSlice[DType.uint8, rank=1],
        inputs: VariadicTensors[
            type, rank, size = Self.num_devices, io_spec=IOUnknown
        ],
        dev_ctx0: DeviceContextPtr,
        dev_ctx1: DeviceContextPtr,
        ctx: MojoCallContextPtr,
    ) raises:
        _check_signal_buffer_size(signal_buffer0)
        _check_signal_buffer_size(signal_buffer1)

        dev_ctxs = List[DeviceContext](dev_ctx0[], dev_ctx1[])

        var out_bufs = StaticTuple[NDBuffer[type, rank], outputs.size]()

        @parameter
        for i in range(outputs.size):
            out_bufs[i] = managed_tensor_slice_to_ndbuffer(outputs[i])

        var in_bufs = StaticTuple[NDBuffer[type, rank], inputs.size]()

        @parameter
        for i in range(inputs.size):
            in_bufs[i] = managed_tensor_slice_to_ndbuffer(inputs[i])

        var rank_sigs = StaticTuple[UnsafePointer[Signal], MAX_GPUS]()
        rank_sigs[0] = signal_buffer0._ptr.bitcast[Signal]()
        rank_sigs[1] = signal_buffer1._ptr.bitcast[Signal]()

        all_reduce[ngpus = Self.num_devices](
            dev_ctxs, in_bufs, out_bufs, rank_sigs
        )


@compiler.register("mo.distributed.allreduce.4gpu.sum")
struct DistributedAllReduceSum4Devices:
    alias num_devices = 4

    @staticmethod
    @always_inline
    fn execute[
        type: DType,
        rank: Int,
        target: StringLiteral,
    ](
        outputs: VariadicTensors[
            type, rank, size = Self.num_devices, io_spec=IOUnknown
        ],
        signal_buffer0: ManagedTensorSlice[DType.uint8, rank=1],
        signal_buffer1: ManagedTensorSlice[DType.uint8, rank=1],
        signal_buffer2: ManagedTensorSlice[DType.uint8, rank=1],
        signal_buffer3: ManagedTensorSlice[DType.uint8, rank=1],
        inputs: VariadicTensors[
            type, rank, size = Self.num_devices, io_spec=IOUnknown
        ],
        dev_ctx0: DeviceContextPtr,
        dev_ctx1: DeviceContextPtr,
        dev_ctx2: DeviceContextPtr,
        dev_ctx3: DeviceContextPtr,
        ctx: MojoCallContextPtr,
    ) raises:
        _check_signal_buffer_size(signal_buffer0)
        _check_signal_buffer_size(signal_buffer1)
        _check_signal_buffer_size(signal_buffer2)
        _check_signal_buffer_size(signal_buffer3)

        dev_ctxs = List[DeviceContext](
            dev_ctx0[], dev_ctx1[], dev_ctx2[], dev_ctx3[]
        )

        var out_bufs = StaticTuple[NDBuffer[type, rank], outputs.size]()

        @parameter
        for i in range(outputs.size):
            out_bufs[i] = managed_tensor_slice_to_ndbuffer(outputs[i])

        var in_bufs = StaticTuple[NDBuffer[type, rank], inputs.size]()

        @parameter
        for i in range(inputs.size):
            in_bufs[i] = managed_tensor_slice_to_ndbuffer(inputs[i])

        var rank_sigs = StaticTuple[UnsafePointer[Signal], MAX_GPUS]()
        rank_sigs[0] = signal_buffer0._ptr.bitcast[Signal]()
        rank_sigs[1] = signal_buffer1._ptr.bitcast[Signal]()
        rank_sigs[2] = signal_buffer2._ptr.bitcast[Signal]()
        rank_sigs[3] = signal_buffer3._ptr.bitcast[Signal]()

        all_reduce[ngpus = Self.num_devices](
            dev_ctxs, in_bufs, out_bufs, rank_sigs
        )


# Note: this is not a "real" index_tensor op that covers all cases, but rather
# a stopgap measure for some important models (DLRM, CLIP-ViT, LLaMa2)
@compiler.register("index_tensor")
struct IndexTensor:
    @staticmethod
    fn execute[
        type: DType,
        indices_type: DType,
        data_rank: Int,
        indices_rank: Int,
        output_rank: Int,
        batch_dims: Int,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice[type, output_rank],
        data: ManagedTensorSlice[type, data_rank],
        indices: ManagedTensorSlice[indices_type, indices_rank],
        ctx: MojoCallContextPtr,
    ):
        index_tensor[
            type,
            indices_type,
            data_rank,
            indices_rank,
            output_rank,
            batch_dims,
            target=target,
        ](
            managed_tensor_slice_to_ndbuffer(data),
            managed_tensor_slice_to_ndbuffer(indices),
            managed_tensor_slice_to_ndbuffer(output),
            ctx,
        )
