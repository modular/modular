# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import OptionalReg
from collections.vector import InlinedFixedVector
from math import ceil as _ceil
from math import cos, erf, exp
from math import floor as _floor
from math import fma, isqrt, log, log1p, sin, sqrt
from math import tanh as _tanh
from os import abort
from random import randn, seed
from sys import alignof, external_call, llvm_intrinsic
from sys.info import bitwidthof, simdwidthof, sizeof
from sys.intrinsics import strided_load
from sys.param_env import is_defined

from algorithm import sync_parallelize, vectorize
from algorithm.functional import elementwise
from algorithm.reduction import _reduce_generator, _reduce_generator_cpu
from algorithm.reduction import mean as _mean
from bit import is_power_of_two
from buffer import NDBuffer
from buffer.buffer import _compute_ndbuffer_offset
from buffer.dimlist import Dim, DimList, _make_tuple
from builtin.simd import Int64, UInt8, UInt64, _pow
from compiler_internal import StaticTensorSpec
from gpu.host._compile import _get_gpu_target
from linalg.bmm import batched_matmul as _batched_matmul
from linalg.bmm import batched_matmul_shape
from linalg.dual_gemm import swishGLU
from linalg.matmul import matmul as _matmul
from linalg.matrix_band_part import matrix_band_part
from linalg.matrix_solve import matrix_solve, matrix_solve_shape
from linalg.packing import (
    pack_b_ndbuffer,
    pack_matmul_b_shape_func,
    pack_transposed_b_ndbuffer,
)
from linalg.utils import GemmShape
from memory import AddressSpace, UnsafePointer, memcpy, memset_zero
from memory.unsafe import bitcast
from MOGGIntList import IntList
from MOGGKernelAPI import (
    managed_tensor_slice_to_ndbuffer as managed_tensor_slice_to_ndbuffer_impl,
)
from nn._optional_param import OptionalParamInt
from nn.activations import gelu, relu
from nn.arange import arange, arange_shape
from nn.arg_nonzero import arg_nonzero, arg_nonzero_shape
from nn.argmaxmin import argmax as _argmax
from nn.argmaxmin import argmin as _argmin
from nn.argmaxmin_gpu import argmax_gpu as _argmax_gpu
from nn.argmaxmin_gpu import argmin_gpu as _argmin_gpu
from nn.concat import _concat_cpu
from nn.concat import concat as _concat
from nn.concat import concat_shape as concat_from_list_shape
from nn.concat import test_concat_fusion
from nn.conv import ConvInfoStatic, conv_gpu, conv_nhwc_direct, conv_shape
from nn.conv import pack_filter as _pack_conv_filter
from nn.conv import pack_filter_shape as _pack_conv_filter_shape
from nn.conv_transpose import conv_transpose_shape
from nn.conv_transpose import conv_transposed as conv_transpose_impl
from nn.conv_transpose import pack_filter as _pack_conv_transpose_filter
from nn.conv_transpose import (
    pack_filter_shape as _pack_conv_transpose_filter_shape,
)
from nn.cumsum import cumsum as _cumsum
from nn.flash_attention import flash_attention as nn_flash_attention
from nn.flash_attention import flash_attention_split_kv
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
from nn.index_tensor import index_tensor as _index_tensor
from nn.kv_cache import (
    contiguous_kv_cache_collection_h1_d16_bshd,
    contiguous_kv_cache_collection_h6_d48_bshd,
    contiguous_kv_cache_collection_h8_d32_bshd,
    contiguous_kv_cache_collection_h8_d64_bshd,
    contiguous_kv_cache_collection_h8_d128_bshd,
    continuous_batching_kv_cache_collection_h1_d16_bshd,
    continuous_batching_kv_cache_collection_h8_d32_bshd,
    continuous_batching_kv_cache_collection_h8_d64_bshd,
    continuous_batching_kv_cache_collection_h8_d128_bshd,
    continuous_batching_kv_cache_collection_h8_d512_bshd,
    continuous_batching_kv_cache_collection_h16_d128_bshd,
    continuous_batching_kv_cache_collection_h32_d128_bshd,
    paged_kv_cache_collection_h1_d16_bshd,
    paged_kv_cache_collection_h6_d48_bshd,
    paged_kv_cache_collection_h8_d32_bshd,
    paged_kv_cache_collection_h8_d64_bshd,
    paged_kv_cache_collection_h8_d128_bshd,
    paged_kv_cache_collection_h8_d512_bshd,
    paged_kv_cache_collection_h32_d128_bshd,
    flash_attention_kv_cache_h1_d16_bshd,
    flash_attention_kv_cache_h1_d16_bshd_continuous_batch,
    flash_attention_kv_cache_h1_d16_causal_mask_continuous_batch,
    flash_attention_kv_cache_h1_d16_causal_alibi_mask_continuous_batch,
    flash_attention_kv_cache_h6_d48_bshd,
    flash_attention_kv_cache_h8_d32_bshd,
    flash_attention_kv_cache_h8_d32_bshd_continuous_batch,
    flash_attention_kv_cache_h8_d32_causal_mask_continuous_batch,
    flash_attention_kv_cache_h8_d32_causal_alibi_mask_continuous_batch,
    flash_attention_kv_cache_h8_d64_bshd,
    flash_attention_kv_cache_h8_d64_bshd_continuous_batch,
    flash_attention_kv_cache_h8_d64_causal_mask_continuous_batch,
    flash_attention_kv_cache_h8_d64_causal_alibi_mask_continuous_batch,
    flash_attention_kv_cache_h8_d128_bshd,
    flash_attention_kv_cache_h8_d128_bshd_continuous_batch,
    flash_attention_kv_cache_h16_d128_bshd_continuous_batch,
    flash_attention_kv_cache_h8_d128_causal_mask_continuous_batch,
    flash_attention_kv_cache_h8_d128_causal_alibi_mask_continuous_batch,
    flash_attention_kv_cache_h8_d512_bshd_continuous_batch,
    flash_attention_kv_cache_h32_d128_bshd_continuous_batch,
    flash_attention_kv_cache_h32_d128_causal_mask_continuous_batch,
    flash_attention_kv_cache_h32_d128_causal_alibi_mask_continuous_batch,
    fused_qk_rope_h1_d16_bshd,
    fused_qk_rope_h1_d16_bshd_continuous_batch,
    fused_qk_rope_h6_d48_bshd,
    fused_qk_rope_h8_d32_bshd,
    fused_qk_rope_h8_d32_bshd_continuous_batch,
    fused_qk_rope_h8_d64_bshd,
    fused_qk_rope_h8_d64_bshd_continuous_batch,
    fused_qk_rope_h8_d128_bshd,
    fused_qk_rope_h8_d128_bshd_continuous_batch,
    fused_qk_rope_h32_d128_bshd_continuous_batch,
    fused_qkv_matmul_kv_cache_h1_d16_bshd,
    fused_qkv_matmul_kv_cache_h1_d16_bshd_continuous_batch,
    fused_qkv_matmul_kv_cache_h6_d48_bshd,
    fused_qkv_matmul_kv_cache_h8_d32_bshd,
    fused_qkv_matmul_kv_cache_h8_d32_bshd_continuous_batch,
    fused_qkv_matmul_kv_cache_h8_d64_bshd,
    fused_qkv_matmul_kv_cache_h8_d64_bshd_continuous_batch,
    fused_qkv_matmul_kv_cache_h8_d128_bshd,
    fused_qkv_matmul_kv_cache_h8_d128_bshd_continuous_batch,
    fused_qkv_matmul_kv_cache_h8_d512_bshd_continuous_batch,
    fused_qkv_matmul_kv_cache_h16_d128_bshd_continuous_batch,
    fused_qkv_matmul_kv_cache_h32_d128_bshd_continuous_batch,
    kv_cache_length_h1_d16_bshd_bf16,
    kv_cache_length_h1_d16_bshd_bf16_continuous_batch,
    kv_cache_length_h1_d16_bshd_f32,
    kv_cache_length_h1_d16_bshd_f32_continuous_batch,
    kv_cache_length_h6_d48_bshd_f32,
    kv_cache_length_h8_d32_bshd_bf16,
    kv_cache_length_h8_d32_bshd_bf16_continuous_batch,
    kv_cache_length_h8_d32_bshd_f32,
    kv_cache_length_h8_d32_bshd_f32_continuous_batch,
    kv_cache_length_h8_d64_bshd_bf16,
    kv_cache_length_h8_d64_bshd_bf16_continuous_batch,
    kv_cache_length_h8_d64_bshd_f32,
    kv_cache_length_h8_d64_bshd_f32_continuous_batch,
    kv_cache_length_h8_d128_bshd_bf16,
    kv_cache_length_h8_d128_bshd_bf16_continuous_batch,
    kv_cache_length_h8_d128_bshd_f32,
    kv_cache_length_h8_d128_bshd_f32_continuous_batch,
    kv_cache_length_h32_d128_bshd_bf16_continuous_batch,
)
from nn.kv_cache_ragged import (
    flash_attention_kv_cache_h1_d16_cont_batch_ragged,
    flash_attention_kv_cache_h8_d64_cont_batch_ragged,
    flash_attention_kv_cache_h8_d128_cont_batch_ragged,
    flash_attention_kv_cache_h8_d512_cont_batch_ragged,
    flash_attention_kv_cache_h32_d128_cont_batch_ragged,
    flash_attention_kv_cache_h1_d16_bshd_paged_ragged,
    flash_attention_kv_cache_h6_d48_bshd_paged_ragged,
    flash_attention_kv_cache_h8_d32_bshd_paged_ragged,
    flash_attention_kv_cache_h8_d64_bshd_paged_ragged,
    flash_attention_kv_cache_h8_d128_bshd_paged_ragged,
    flash_attention_kv_cache_h8_d512_bshd_paged_ragged,
    flash_attention_kv_cache_h32_d128_bshd_paged_ragged,
    fused_qk_rope_h1_d16_bshd_continuous_batch_ragged,
    fused_qk_rope_h1_d16_bshd_ragged,
    fused_qk_rope_h6_d48_bshd_ragged,
    fused_qk_rope_h8_d32_bshd_continuous_batch_ragged,
    fused_qk_rope_h8_d32_bshd_ragged,
    fused_qk_rope_h8_d64_bshd_continuous_batch_ragged,
    fused_qk_rope_h8_d64_bshd_ragged,
    fused_qk_rope_h8_d128_bshd_continuous_batch_ragged,
    fused_qk_rope_h8_d128_bshd_ragged,
    fused_qk_rope_h8_d512_bshd_continuous_batch_ragged,
    fused_qk_rope_h8_d512_bshd_ragged,
    fused_qk_rope_h32_d128_bshd_continuous_batch_ragged,
    fused_qk_rope_h1_d16_bshd_paged_ragged,
    fused_qk_rope_h6_d48_bshd_paged_ragged,
    fused_qk_rope_h8_d32_bshd_paged_ragged,
    fused_qk_rope_h8_d64_bshd_paged_ragged,
    fused_qk_rope_h8_d128_bshd_paged_ragged,
    fused_qk_rope_h8_d512_bshd_paged_ragged,
    fused_qk_rope_h32_d128_bshd_paged_ragged,
    fused_qkv_matmul_kv_cache_h1_d16_cont_batch_ragged,
    fused_qkv_matmul_kv_cache_h8_d64_cont_batch_ragged,
    fused_qkv_matmul_kv_cache_h8_d128_cont_batch_ragged,
    fused_qkv_matmul_kv_cache_h8_d512_cont_batch_ragged,
    fused_qkv_matmul_kv_cache_h32_d128_cont_batch_ragged,
    fused_qkv_matmul_kv_cache_h1_d16_bshd_paged_ragged,
    fused_qkv_matmul_kv_cache_h6_d48_bshd_paged_ragged,
    fused_qkv_matmul_kv_cache_h8_d32_bshd_paged_ragged,
    fused_qkv_matmul_kv_cache_h8_d64_bshd_paged_ragged,
    fused_qkv_matmul_kv_cache_h8_d128_bshd_paged_ragged,
    fused_qkv_matmul_kv_cache_h8_d512_bshd_paged_ragged,
    fused_qkv_matmul_kv_cache_h32_d128_bshd_paged_ragged,
    matmul_kv_cache_h8_d128_cont_batch_ragged,
)
from nn.mha import flash_attention
from nn.mha import fused_attention as cpu_fused_attention_impl
from nn.nms import non_max_suppression, non_max_suppression_shape_func
from nn.normalization import (
    layer_norm,
    layer_norm_shape,
    rms_norm,
    rms_norm_shape,
)
from nn.pad import pad_constant as _pad_constant
from nn.pad import pad_reflect as _pad_reflect
from nn.pad import pad_repeat as _pad_repeat
from nn.pad import pad_shape
from nn.pool import avg_pool as _avg_pool
from nn.pool import avg_pool_gpu as _avg_pool_gpu
from nn.pool import max_pool as _max_pool
from nn.pool import max_pool_gpu as _max_pool_gpu
from nn.pool import pool_shape, pool_shape_ceil
from nn.reshape import ndbuffer_reshape, reshape, reshape_shape
from nn.resize import CoordinateTransformationMode, RoundMode
from nn.resize import resize_linear as resize_linear_kernel
from nn.resize import resize_nearest_neighbor
from nn.roi_align import roi_align_nhwc
from nn.slice import (
    copy_to_slice,
    slice_as_view,
    slice_dim_as_view,
    slice_shape,
)
from nn.softmax import logsoftmax as _logsoftmax
from nn.softmax import softmax as _softmax
from nn.split import split as _split
from nn.tile import tile, tile_shape
from nn.topk import top_k as _top_k
from nn.topk import top_k_fused_sampling as _topk_fused_sampling
from nn.topk import top_k_shape
from nn.topk_gpu import topk_fused_sampling_gpu as _topk_fused_sampling_gpu
from nn.topk_gpu import topk_gpu as _topk_gpu
from nn.toppminp import min_p_sampling as _min_p_sampling
from nn.toppminp import top_p_sampling as _top_p_sampling
from nn.toppminp_gpu import min_p_sampling_gpu as _min_p_sampling_gpu
from nn.toppminp_gpu import top_p_sampling_gpu as _top_p_sampling_gpu
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
from register import *
from runtime.asyncrt import MojoCallContextPtr
from runtime.tracing import Trace, TraceLevel, trace_arg
from tensor_utils_internal import ManagedTensorSlice

from utils import StaticTuple
from utils.index import Index, IndexList, product
from utils.loop import unroll
from utils.numerics import isinf, isnan


# Prevent these functions from being DCE'd by explicitly exporting them.
@export
fn MOGGExport():
    alias _add = add
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
    alias _layer_norm = layer_norm
    alias _layer_norm_shape = layer_norm_shape
    alias _rms_norm = rms_norm
    alias _rms_norm_shape = rms_norm_shape
    alias _pack_b_ndbuffer = pack_b_ndbuffer
    alias _pack_transposed_b_ndbuffer = pack_transposed_b_ndbuffer
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
    alias _sub = sub
    alias _random_shape = random_shape
    alias _roi_align_shape = roi_align_shape
    alias _slice_shape = slice_shape
    alias _arg_nonzero = arg_nonzero
    alias _arg_nonzero_shape = arg_nonzero_shape
    alias _top_k_shape = top_k_shape
    alias _tile = tile
    alias _tile_shape = tile_shape
    alias _swishGLU = swishGLU

    # kv-cache
    alias _kv_cache_length_h8_d128_bshd_bf16 = kv_cache_length_h8_d128_bshd_bf16
    alias _kv_cache_length_h6_d48_bshd_f32 = kv_cache_length_h6_d48_bshd_f32
    alias _kv_cache_length_h8_d128_bshd_f32 = kv_cache_length_h8_d128_bshd_f32
    alias _kv_cache_length_h1_d16_bshd_f32 = kv_cache_length_h1_d16_bshd_f32
    alias _kv_cache_length_h1_d16_bshd_bf16 = kv_cache_length_h1_d16_bshd_bf16
    alias _kv_cache_length_h8_d32_bshd_f32 = kv_cache_length_h8_d32_bshd_f32
    alias _kv_cache_length_h8_d32_bshd_bf16 = kv_cache_length_h8_d32_bshd_bf16
    alias _kv_cache_length_h8_d64_bshd_f32 = kv_cache_length_h8_d64_bshd_f32
    alias _kv_cache_length_h8_d64_bshd_bf16 = kv_cache_length_h8_d64_bshd_bf16
    alias _kv_cache_length_h8_d128_bshd_f32_continuous_batch = kv_cache_length_h8_d128_bshd_f32_continuous_batch
    alias _kv_cache_length_h8_d128_bshd_bf16_continuous_batch = kv_cache_length_h8_d128_bshd_bf16_continuous_batch
    alias _kv_cache_length_h32_d128_bshd_bf16_continuous_batch = kv_cache_length_h32_d128_bshd_bf16_continuous_batch
    alias _kv_cache_length_h8_d32_bshd_f32_continuous_batch = kv_cache_length_h8_d32_bshd_f32_continuous_batch
    alias _kv_cache_length_h8_d32_bshd_bf16_continuous_batch = kv_cache_length_h8_d32_bshd_bf16_continuous_batch
    alias _kv_cache_length_h8_d64_bshd_f32_continuous_batch = kv_cache_length_h8_d64_bshd_f32_continuous_batch
    alias _kv_cache_length_h8_d64_bshd_bf16_continuous_batch = kv_cache_length_h8_d64_bshd_bf16_continuous_batch
    alias _kv_cache_length_h1_d16_bshd_f32_continuous_batch = kv_cache_length_h1_d16_bshd_f32_continuous_batch
    alias _kv_cache_length_h1_d16_bshd_bf16_continuous_batch = kv_cache_length_h1_d16_bshd_bf16_continuous_batch
    alias _fused_qkv_matmul_kv_cache_h6_d48_bshd = fused_qkv_matmul_kv_cache_h6_d48_bshd
    alias _fused_qkv_matmul_kv_cache_h8_d128_bshd = fused_qkv_matmul_kv_cache_h8_d128_bshd
    alias _fused_qkv_matmul_kv_cache_h1_d16_bshd = fused_qkv_matmul_kv_cache_h1_d16_bshd
    alias _fused_qkv_matmul_kv_cache_h8_d32_bshd = fused_qkv_matmul_kv_cache_h8_d32_bshd
    alias _fused_qkv_matmul_kv_cache_h8_d64_bshd = fused_qkv_matmul_kv_cache_h8_d64_bshd
    alias _fused_qkv_matmul_kv_cache_h8_d128_bshd_continuous_batch = fused_qkv_matmul_kv_cache_h8_d128_bshd_continuous_batch
    alias _fused_qkv_matmul_kv_cache_h8_d32_bshd_continuous_batch = fused_qkv_matmul_kv_cache_h8_d32_bshd_continuous_batch
    alias _fused_qkv_matmul_kv_cache_h8_d64_bshd_continuous_batch = fused_qkv_matmul_kv_cache_h8_d64_bshd_continuous_batch
    alias _fused_qkv_matmul_kv_cache_h8_d512_bshd_continuous_batch = fused_qkv_matmul_kv_cache_h8_d512_bshd_continuous_batch
    alias _fused_qkv_matmul_kv_cache_h1_d16_bshd_continuous_batch = fused_qkv_matmul_kv_cache_h1_d16_bshd_continuous_batch
    alias _fused_qkv_matmul_kv_cache_h16_d128_bshd_continuous_batch = fused_qkv_matmul_kv_cache_h16_d128_bshd_continuous_batch
    alias _fused_qkv_matmul_kv_cache_h32_d128_bshd_continuous_batch = fused_qkv_matmul_kv_cache_h32_d128_bshd_continuous_batch
    alias _fused_qkv_matmul_kv_cache_h1_d16_bshd_paged_ragged = fused_qkv_matmul_kv_cache_h1_d16_bshd_paged_ragged
    alias _fused_qkv_matmul_kv_cache_h6_d48_bshd_paged_ragged = fused_qkv_matmul_kv_cache_h6_d48_bshd_paged_ragged
    alias _fused_qkv_matmul_kv_cache_h8_d32_bshd_paged_ragged = fused_qkv_matmul_kv_cache_h8_d32_bshd_paged_ragged
    alias _fused_qkv_matmul_kv_cache_h8_d64_bshd_paged_ragged = fused_qkv_matmul_kv_cache_h8_d64_bshd_paged_ragged
    alias _fused_qkv_matmul_kv_cache_h8_d128_bshd_paged_ragged = fused_qkv_matmul_kv_cache_h8_d128_bshd_paged_ragged
    alias _fused_qkv_matmul_kv_cache_h8_d512_bshd_paged_ragged = fused_qkv_matmul_kv_cache_h8_d512_bshd_paged_ragged
    alias _fused_qkv_matmul_kv_cache_h32_d128_bshd_paged_ragged = fused_qkv_matmul_kv_cache_h32_d128_bshd_paged_ragged
    alias _fused_qk_rope_h6_d48_bshd = fused_qk_rope_h6_d48_bshd
    alias _fused_qk_rope_h8_d128_bshd = fused_qk_rope_h8_d128_bshd
    alias _fused_qk_rope_h1_d16_bshd = fused_qk_rope_h1_d16_bshd
    alias _fused_qk_rope_h1_d16_bshd_continuous_batch = fused_qk_rope_h1_d16_bshd_continuous_batch
    alias _fused_qk_rope_h8_d32_bshd = fused_qk_rope_h8_d32_bshd
    alias _fused_qk_rope_h8_d128_bshd_continuous_batch = fused_qk_rope_h8_d128_bshd_continuous_batch
    alias _fused_qk_rope_h32_d128_bshd_continuous_batch = fused_qk_rope_h32_d128_bshd_continuous_batch
    alias _fused_qk_rope_h8_d32_bshd_continuous_batch = fused_qk_rope_h8_d32_bshd_continuous_batch
    alias _fused_qk_rope_h8_d64_bshd_continuous_batch = fused_qk_rope_h8_d64_bshd_continuous_batch
    alias _fused_qk_rope_h1_d16_bshd_paged_ragged = fused_qk_rope_h1_d16_bshd_paged_ragged
    alias _fused_qk_rope_h6_d48_bshd_paged_ragged = fused_qk_rope_h6_d48_bshd_paged_ragged
    alias _fused_qk_rope_h8_d32_bshd_paged_ragged = fused_qk_rope_h8_d32_bshd_paged_ragged
    alias _fused_qk_rope_h8_d64_bshd_paged_ragged = fused_qk_rope_h8_d64_bshd_paged_ragged
    alias _fused_qk_rope_h8_d128_bshd_paged_ragged = fused_qk_rope_h8_d128_bshd_paged_ragged
    alias _fused_qk_rope_h8_d512_bshd_paged_ragged = fused_qk_rope_h8_d512_bshd_paged_ragged
    alias _fused_qk_rope_h32_d128_bshd_paged_ragged = fused_qk_rope_h32_d128_bshd_paged_ragged
    alias _flash_attention_kv_cache_h6_d48_bshd = flash_attention_kv_cache_h6_d48_bshd
    alias _flash_attention_kv_cache_h8_d128_bshd = flash_attention_kv_cache_h8_d128_bshd
    alias _flash_attention_kv_cache_h1_d16_bshd = flash_attention_kv_cache_h1_d16_bshd
    alias _flash_attention_kv_cache_h1_d16_bshd_continuous_batch = flash_attention_kv_cache_h1_d16_bshd_continuous_batch
    alias _flash_attention_kv_cache_h8_d32_bshd = flash_attention_kv_cache_h8_d32_bshd
    alias _flash_attention_kv_cache_h8_d64_bshd = flash_attention_kv_cache_h8_d64_bshd
    alias _flash_attention_kv_cache_h8_d128_bshd_continuous_batch = flash_attention_kv_cache_h8_d128_bshd_continuous_batch
    alias _flash_attention_kv_cache_h16_d128_bshd_continuous_batch = flash_attention_kv_cache_h16_d128_bshd_continuous_batch
    alias _flash_attention_kv_cache_h8_d32_bshd_continuous_batch = flash_attention_kv_cache_h8_d32_bshd_continuous_batch
    alias _flash_attention_kv_cache_h8_d64_bshd_continuous_batch = flash_attention_kv_cache_h8_d64_bshd_continuous_batch
    alias _flash_attention_kv_cache_h8_d512_bshd_continuous_batch = flash_attention_kv_cache_h8_d512_bshd_continuous_batch
    alias _flash_attention_kv_cache_h32_d128_bshd_continuous_batch = flash_attention_kv_cache_h32_d128_bshd_continuous_batch
    alias _flash_attention_kv_cache_h8_d128_causal_mask_continuous_batch = flash_attention_kv_cache_h8_d128_causal_mask_continuous_batch
    alias _flash_attention_kv_cache_h32_d128_causal_mask_continuous_batch = flash_attention_kv_cache_h32_d128_causal_mask_continuous_batch
    alias _flash_attention_kv_cache_h8_d32_causal_mask_continuous_batch = flash_attention_kv_cache_h8_d32_causal_mask_continuous_batch
    alias _flash_attention_kv_cache_h8_d64_causal_mask_continuous_batch = flash_attention_kv_cache_h8_d64_causal_mask_continuous_batch
    alias _flash_attention_kv_cache_h1_d16_causal_mask_continuous_batch = flash_attention_kv_cache_h1_d16_causal_mask_continuous_batch
    alias _flash_attention_kv_cache_h8_d128_causal_alibi_mask_continuous_batch = flash_attention_kv_cache_h8_d128_causal_alibi_mask_continuous_batch
    alias _flash_attention_kv_cache_h32_d128_causal_alibi_mask_continuous_batch = flash_attention_kv_cache_h32_d128_causal_alibi_mask_continuous_batch
    alias _flash_attention_kv_cache_h8_d32_causal_alibi_mask_continuous_batch = flash_attention_kv_cache_h8_d32_causal_alibi_mask_continuous_batch
    alias _flash_attention_kv_cache_h8_d64_causal_alibi_mask_continuous_batch = flash_attention_kv_cache_h8_d64_causal_alibi_mask_continuous_batch
    alias _flash_attention_kv_cache_h1_d16_causal_alibi_mask_continuous_batch = flash_attention_kv_cache_h1_d16_causal_alibi_mask_continuous_batch
    alias _contiguous_kv_cache_collection_h6_d48_bshd = contiguous_kv_cache_collection_h6_d48_bshd
    alias _contiguous_kv_cache_collection_h8_d128_bshd = contiguous_kv_cache_collection_h8_d128_bshd
    alias _contiguous_kv_cache_collection_h1_d16_bshd = contiguous_kv_cache_collection_h1_d16_bshd
    alias _contiguous_kv_cache_collection_h8_d32_bshd = contiguous_kv_cache_collection_h8_d32_bshd
    alias _contiguous_kv_cache_collection_h8_d64_bshd = contiguous_kv_cache_collection_h8_d64_bshd
    alias _continuous_batching_kv_cache_collection_h8_d32_bshd = continuous_batching_kv_cache_collection_h8_d32_bshd
    alias _continuous_batching_kv_cache_collection_h8_d64_bshd = continuous_batching_kv_cache_collection_h8_d64_bshd
    alias _continuous_batching_kv_cache_collection_h8_d128_bshd = continuous_batching_kv_cache_collection_h8_d128_bshd
    alias _continuous_batching_kv_cache_collection_h8_d512_bshd = continuous_batching_kv_cache_collection_h8_d512_bshd
    alias _continuous_batching_kv_cache_collection_h16_d128_bshd = continuous_batching_kv_cache_collection_h16_d128_bshd
    alias _continuous_batching_kv_cache_collection_h32_d128_bshd = continuous_batching_kv_cache_collection_h32_d128_bshd
    alias _continuous_batching_kv_cache_collection_h1_d16_bshd = continuous_batching_kv_cache_collection_h1_d16_bshd
    alias _paged_kv_cache_collection_h1_d16_bshd = paged_kv_cache_collection_h1_d16_bshd
    alias _paged_kv_cache_collection_h6_d48_bshd = paged_kv_cache_collection_h6_d48_bshd
    alias _paged_kv_cache_collection_h8_d32_bshd = paged_kv_cache_collection_h8_d32_bshd
    alias _paged_kv_cache_collection_h8_d64_bshd = paged_kv_cache_collection_h8_d64_bshd
    alias _paged_kv_cache_collection_h8_d128_bshd = paged_kv_cache_collection_h8_d128_bshd
    alias _paged_kv_cache_collection_h8_d512_bshd = paged_kv_cache_collection_h8_d512_bshd
    alias _paged_kv_cache_collection_h32_d128_bshd = paged_kv_cache_collection_h32_d128_bshd
    alias _fused_qkv_matmul_kv_cache_h8_d128_cont_batch_ragged = fused_qkv_matmul_kv_cache_h8_d128_cont_batch_ragged
    alias _fused_qkv_matmul_kv_cache_h8_d512_cont_batch_ragged = fused_qkv_matmul_kv_cache_h8_d512_cont_batch_ragged
    alias _fused_qkv_matmul_kv_cache_h32_d128_cont_batch_ragged = fused_qkv_matmul_kv_cache_h32_d128_cont_batch_ragged
    alias _fused_qkv_matmul_kv_cache_h8_d64_cont_batch_ragged = fused_qkv_matmul_kv_cache_h8_d64_cont_batch_ragged
    alias _fused_qkv_matmul_kv_cache_h1_d16_cont_batch_ragged = fused_qkv_matmul_kv_cache_h1_d16_cont_batch_ragged
    alias _fused_qk_rope_h6_d48_bshd_ragged = fused_qk_rope_h6_d48_bshd_ragged
    alias _fused_qk_rope_h8_d128_bshd_ragged = fused_qk_rope_h8_d128_bshd_ragged
    alias _fused_qk_rope_h8_d512_bshd_ragged = fused_qk_rope_h8_d512_bshd_ragged
    alias _fused_qk_rope_h1_d16_bshd_ragged = fused_qk_rope_h1_d16_bshd_ragged
    alias _fused_qk_rope_h8_d32_bshd_ragged = fused_qk_rope_h8_d32_bshd_ragged
    alias _fused_qk_rope_h8_d64_bshd_ragged = fused_qk_rope_h8_d64_bshd_ragged
    alias _fused_qk_rope_h8_d128_bshd_continuous_batch_ragged = fused_qk_rope_h8_d128_bshd_continuous_batch_ragged
    alias _fused_qk_rope_h8_d512_bshd_continuous_batch_ragged = fused_qk_rope_h8_d512_bshd_continuous_batch_ragged
    alias _fused_qk_rope_h32_d128_bshd_continuous_batch_ragged = fused_qk_rope_h32_d128_bshd_continuous_batch_ragged
    alias _fused_qk_rope_h1_d16_bshd_continuous_batch_ragged = fused_qk_rope_h1_d16_bshd_continuous_batch_ragged
    alias _fused_qk_rope_h8_d32_bshd_continuous_batch_ragged = fused_qk_rope_h8_d32_bshd_continuous_batch_ragged
    alias _fused_qk_rope_h8_d64_bshd_continuous_batch_ragged = fused_qk_rope_h8_d64_bshd_continuous_batch_ragged
    alias _flash_attention_kv_cache_h1_d16_cont_batch_ragged = flash_attention_kv_cache_h1_d16_cont_batch_ragged
    alias _flash_attention_kv_cache_h8_d64_cont_batch_ragged = flash_attention_kv_cache_h8_d64_cont_batch_ragged
    alias _flash_attention_kv_cache_h8_d128_cont_batch_ragged = flash_attention_kv_cache_h8_d128_cont_batch_ragged
    alias _flash_attention_kv_cache_h8_d512_cont_batch_ragged = flash_attention_kv_cache_h8_d512_cont_batch_ragged
    alias _flash_attention_kv_cache_h32_d128_cont_batch_ragged = flash_attention_kv_cache_h32_d128_cont_batch_ragged
    alias _flash_attention_kv_cache_h1_d16_bshd_paged_ragged = flash_attention_kv_cache_h1_d16_bshd_paged_ragged
    alias _flash_attention_kv_cache_h6_d48_bshd_paged_ragged = flash_attention_kv_cache_h6_d48_bshd_paged_ragged
    alias _flash_attention_kv_cache_h8_d32_bshd_paged_ragged = flash_attention_kv_cache_h8_d32_bshd_paged_ragged
    alias _flash_attention_kv_cache_h8_d64_bshd_paged_ragged = flash_attention_kv_cache_h8_d64_bshd_paged_ragged
    alias _flash_attention_kv_cache_h8_d128_bshd_paged_ragged = flash_attention_kv_cache_h8_d128_bshd_paged_ragged
    alias _flash_attention_kv_cache_h8_d512_bshd_paged_ragged = flash_attention_kv_cache_h8_d512_bshd_paged_ragged
    alias _flash_attention_kv_cache_h32_d128_bshd_paged_ragged = flash_attention_kv_cache_h32_d128_bshd_paged_ragged
    alias _matmul_kv_cache_h8_d128_cont_batch_ragged = matmul_kv_cache_h8_d128_cont_batch_ragged


# ===----------------------------------------------------------------------===#
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


# ===-----------------------------------------------------------------------===#
# Basic generated kernel building blocks
# ===-----------------------------------------------------------------------===#


# This function is used in some MLIR tests, so we need to define it here to
# register it as a kernel. However the _real_ implementation is in MOGGKernelAPI.mojo,
# we forward to that definition here.
@register_internal("managed_tensor_slice_to_ndbuffer")
@always_inline
fn managed_tensor_slice_to_ndbuffer[
    type: DType, rank: Int
](tensor: ManagedTensorSlice[type, rank]) -> NDBuffer[type, rank]:
    return managed_tensor_slice_to_ndbuffer_impl(tensor)


@register_internal("to_buffer")
@always_inline
fn to_buffer[
    type: DType, rank: Int
](
    data: UnsafePointer[Scalar[type]],
    shape: UnsafePointer[Int],
) -> NDBuffer[
    type, rank
]:
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

    return NDBuffer[type, rank](data, shape_tuple, stride_tuple)


@always_inline
fn _to_buffer_index_list_shape[
    type: DType, rank: Int
](
    data: UnsafePointer[Scalar[type]],
    shape_tuple: IndexList[rank],
) -> NDBuffer[
    type, rank
]:
    var stride_tuple = IndexList[rank]()
    var stride: Int = 1

    @parameter
    for i in reversed(range(rank)):
        stride_tuple[i] = stride
        stride *= shape_tuple[i]

    return NDBuffer[type, rank](data, shape_tuple, stride_tuple)


@register_internal("to_shape")
@always_inline
fn to_shape[rank: Int](shape: UnsafePointer[Int]) -> IndexList[rank]:
    var shape_ptr = shape
    var shape_tuple = IndexList[rank]()

    @parameter
    for i in range(rank):
        shape_tuple[i] = shape_ptr[i]

    return shape_tuple


# Convert a tensor into a shape.
@register_internal("tensor_to_shape")
@always_inline
fn tensor_to_shape[
    type: DType,
    rank: Int,
](tensor: NDBuffer[type, 1]) -> IndexList[rank]:
    var out = IndexList[rank]()

    @parameter
    for i in range(rank):
        out[i] = int(tensor[i])

    return out


# Extract a value from a shape.
@register_internal("get_scalar_from_ndbuffer")
@always_inline
fn get_scalar_from_ndbuffer[
    dtype: DType
](tensor: NDBuffer[dtype, 1]) -> Scalar[dtype]:
    # Assumes that tensor is on the host!
    return tensor[0]


@register_internal("shape_to_ndbuffer")
@always_inline
fn shape_to_ndbuffer[
    shape_rank: Int, buf_rank: Int, type: DType
](shape: IndexList[shape_rank], buf: NDBuffer[type, buf_rank]):
    @parameter
    for i in range(shape_rank):
        buf[i] = shape[i]


@register_internal("shape_to_managed_tensor_slice")
@always_inline
fn shape_to_managed_tensor_slice[
    shape_rank: Int, buf_rank: Int, type: DType
](
    shape: IndexList[shape_rank],
    mut tensor: ManagedTensorSlice[type, buf_rank],
):
    @parameter
    for i in range(shape_rank):
        tensor.store[width=1](IndexList[1](i), shape[i])


@register_internal("to_buffer_list")
@always_inline
fn to_buffer_list[
    type: DType, rank: Int
](
    raw_list_ptr: UnsafePointer[NoneType],
) -> InlinedFixedVector[
    NDBuffer[type, rank]
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

    # Create output list
    var out_list = InlinedFixedVector[NDBuffer[type, rank]](num_elements)

    # Convert individual elements of the input list into NDBuffer, and
    # accumulate the results to output list.
    for i in range(num_elements):
        var data = data_ptrs[i].bitcast[Scalar[type]]()

        var dims = IndexList[rank]()

        @parameter
        for dim in range(rank):
            dims[dim] = dim_values[dim + i * rank].__int__()

        var buffer = _to_buffer_index_list_shape[type, rank](data, dims)
        out_list.append(buffer)

    return InlinedFixedVector(out_list)


@register_internal("destruct_buffer_list")
@always_inline
fn destruct_buffer_list[
    type: DType, rank: Int
](owned list: InlinedFixedVector[NDBuffer[type, rank]]):
    # TODO: remove this now that `InlinedFixedVector` removed `del_old`
    pass


# TODO(#27757): All calls with concrete body functions are as if annotated with
#               @register_internal("mo.original_op")
@register_internal("elementwise")
@always_inline
fn elementwise_wrapper[
    trace_description: StringLiteral,
    simd_width: Int,
    type: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
    func: fn[width: Int, rank: Int, element_alignment: Int] (
        IndexList[rank]
    ) capturing -> None,
    /,
    shape: DimList,
    target: StringLiteral = "cpu",
](buffer: NDBuffer[type, rank, shape], ctx: MojoCallContextPtr):
    elementwise[
        func[element_alignment=1],
        simd_width=simd_width,
        use_blocking_impl=single_thread_blocking_override,
        target=target,
        trace_description=trace_description,
    ](
        _make_tuple[buffer.rank](
            buffer.shape
        ).canonicalize() if buffer.shape.all_known[
            buffer.rank
        ]() else buffer.get_shape(),
        context=ctx,
    )


@register_internal("get_address_space")
fn get_address_space() -> AddressSpace:
    return AddressSpace.GENERIC


# ===-----------------------------------------------------------------------===#
# Tensor API intrinsics
# ===-----------------------------------------------------------------------===#


@register_internal("to_managed_tensor_slice")
@always_inline
fn to_managed_tensor_slice[
    type: DType, rank: Int
](
    data: UnsafePointer[Scalar[type]],
    shape: UnsafePointer[Int],
) -> ManagedTensorSlice[type, rank]:
    var shape_tuple = IndexList[rank]()

    @parameter
    for i in reversed(range(rank)):
        shape_tuple[i] = shape[i]

    return ManagedTensorSlice[type, rank](
        data,
        shape_tuple,
    )


@register_internal("shape_from_kgen")
@always_inline
fn get_static_shape(shape: IntList) -> IndexList[shape._safe_len]:
    return shape.stack_alloc_data


# ===-----------------------------------------------------------------------===#
# Simd load/store helper functions
# ===-----------------------------------------------------------------------===#


@always_inline
fn _simd_load_internal[
    simd_width: Int
](buffer: NDBuffer, index: Int) -> SIMD[buffer.type, simd_width]:
    @parameter
    if buffer.type is DType.bool:
        var v = buffer.data.bitcast[Scalar[DType.uint8]]().load[
            width=simd_width
        ](index)
        return v.cast[buffer.type]()
    return buffer.data.load[width=simd_width](index)


@register_internal("simd_load")
@always_inline
fn simd_load[
    simd_width: Int
](
    buffer: NDBuffer,
    index: IndexList[buffer.rank],
) -> SIMD[
    buffer.type, simd_width
]:
    var flat_index = _compute_ndbuffer_offset(buffer, index)

    if buffer.is_contiguous():
        return _simd_load_internal[simd_width](buffer, flat_index)

    var stride = buffer.stride[buffer.rank - 1]()
    if stride == 0:
        return buffer.data.load(flat_index)

    if buffer.type is DType.bool:
        var v = strided_load[simd_width](
            buffer.data.bitcast[Scalar[DType.uint8]]().offset(flat_index),
            stride,
        )
        return v.cast[buffer.type]()
    return strided_load[simd_width](buffer.data.offset(flat_index), stride)


@register_internal("simd_store")
@always_inline
fn simd_store[
    simd_width: Int, element_alignment: Int
](
    buffer: NDBuffer,
    index: IndexList[buffer.rank],
    val: SIMD[buffer.type, simd_width],
):
    var flat_index = _compute_ndbuffer_offset(buffer, index)

    @parameter
    fn gcd_pow2[a: Int, b: Int]() -> Int:
        # alignments should always be powers of 2
        constrained[
            is_power_of_two(a) and is_power_of_two(b),
            "a and b must be powers of 2",
        ]()
        return min(a, b)

    # We have to cast bools into their runtime storage type.
    @parameter
    if buffer.type is DType.bool:
        buffer.data.bitcast[Scalar[DType.uint8]]().store(
            flat_index, val.cast[DType.uint8]()
        )
    else:
        buffer.data.store[
            alignment = gcd_pow2[
                buffer.alignment, element_alignment * alignof[buffer.type]()
            ](),
        ](flat_index, val)


# ===-----------------------------------------------------------------------===#
# Broadcast
# ===-----------------------------------------------------------------------===#


@register_internal("mo.broadcast_shape")
@always_inline
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


@register_internal_shape_func("mo.broadcast_shape")
@always_inline
fn broadcast_shape_shape[
    lhs_type: DType,
    rhs_type: DType,
    single_thread_blocking_override: Bool,
](
    lhs_buf: NDBuffer[lhs_type, 1],
    rhs_buf: NDBuffer[rhs_type, 1],
    ctx: MojoCallContextPtr,
) -> IndexList[1]:
    var lhs_dim = lhs_buf.dim(0)
    var rhs_dim = rhs_buf.dim(0)
    return IndexList[1](max(lhs_dim, rhs_dim))


@register_internal("mo.static.broadcast_to")
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
    target_shape: IndexList[target_rank],
) -> NDBuffer[type, output_rank]:
    var shape = IndexList[output_rank]()
    var stride = IndexList[output_rank]()

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


@register_internal("reshape_contiguous_buffer")
@always_inline
fn reshape_contiguous_buffer[
    type: DType, old_rank: Int, new_rank: Int
](buffer: NDBuffer[type, old_rank], shape: IndexList[new_rank]) -> NDBuffer[
    type, new_rank
]:
    # Reshape a contigious buffer, as we know the strides don't have to change.
    var stride_tuple = IndexList[new_rank]()
    var stride: Int = 1

    @parameter
    for i in reversed(range(new_rank)):
        # Start from the back so we can accumulate the strides.
        stride_tuple[i] = stride
        stride *= shape[i]

    return NDBuffer[type, new_rank](buffer.data, shape, stride_tuple)


@register_internal("split_dim_indices")
@mogg_view_op
@always_inline
fn split_dim_indices[
    rank: Int, axis: Int
](indices: IndexList[rank], new_shape_dim: Int) -> IndexList[rank + 1]:
    var out = IndexList[rank + 1]()

    # This op is transforming the INDICES of an access into a reshaped tensor.
    # Consider the tensor is [40, 30, 2] and we reshape it to [5, 8, 30, 2].
    # If we are accessing the index [21, 16, 1] in the original shape then to
    # preserve the reshape we would need to transform the indices into [2, 5, 16, 1].
    # Or [21 // 8, 21 % 8, ...old dims...].

    @parameter
    for i in range(rank + 1):

        @parameter
        if i == axis:
            out[i] = indices[axis] // new_shape_dim
        elif i == axis + 1:
            out[i] = indices[axis] % new_shape_dim
        elif i < axis:
            out[i] = indices[i]
        elif i > axis:
            out[i] = indices[i - 1]

    return out


@register_internal("insert_index")
@mogg_view_op
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


@register_internal("translate_gather_indices")
@always_inline
fn translate_gather_indices[
    indices_dtype: DType,
    indices_rank: Int,
    input_rank: Int,
](
    indices_buffer: NDBuffer[indices_dtype, indices_rank],
    axis: Scalar,
    out_coords: IndexList[input_rank + indices_rank - 1],
    input_shape: IndexList[input_rank],
) -> IndexList[input_rank]:
    # From a `output_indices` computed from a gather operation using
    # `indices_buffer` and `axis`, find the correponding input indices.

    var normalized_axis = int(normalize_neg_index(axis, input_rank))
    # out_coords consists of 3 chunks:
    #   out_coords[0:axis] = input coords[0:axis]
    #   out_coords[axis:axis+indices_rank] = indices_coords
    #   out_coords[axis + indices_rank:] = input_coords[axis + 1:]
    # and input_coords[axis] = indices[indices_coords]
    # Get the gather indices.
    var indices_index = IndexList[indices_rank]()

    # Get the indices of the index.
    @parameter
    for i in range(indices_rank):
        indices_index[i] = out_coords[i + normalized_axis]

    # The index we are gathering.
    var data_index = indices_buffer.load(indices_index)

    # Update the indices with the new data index.
    var data_indices = IndexList[input_rank]()

    alias skip_factor = indices_rank - 1

    # Build the indices for the input. We have replaced in index in 'axis'
    # with an index from the indices tensor.
    @parameter
    for i in range(input_rank):
        if i == normalized_axis:
            data_indices[i] = int(
                normalize_neg_index(data_index, input_shape[normalized_axis])
            )
        elif i > normalized_axis:
            # Skip over any extra indices dimensions. These are essentially new dimensions.
            data_indices[i] = out_coords[i + skip_factor]
        else:
            data_indices[i] = out_coords[i]

    return data_indices


@register_internal_shape_func("mo.broadcast_to")
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
) raises -> IndexList[output_rank]:
    if output_rank != target_shape_buf.dim(0):
        raise Error(
            "[broadcast_to] requires (len(target_shape) == output_rank)"
        )
    if input_rank > output_rank:
        raise Error("[broadcast_to] requires (input_rank <= output_rank)")

    # move the output shape from buffer into a static int tuple
    var output_shape = IndexList[output_rank]()

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
@register_internal("simd_target_cpu")
fn get_target_simd[type: DType]() -> __mlir_type.index:
    return int(simdwidthof[type]()).value


@register_internal("simd_target_gpu")
fn get_target_simd_gpu[type: DType]() -> __mlir_type.index:
    return int(simdwidthof[Scalar[type], target = _get_gpu_target()]()).value


@register_internal("simd_target_to_int")
fn simd_width_to_int[simd_width: __mlir_type.index]() -> Int:
    return Int(simd_width)


# ===-----------------------------------------------------------------------===#
# Abs wrapper op
# ===-----------------------------------------------------------------------===#


# Call abs, needed as it has multiple overloads which can't be aliased
@register_internal("mo.abs")
@mogg_elementwise
@always_inline
fn abs_wrapped[
    type: DType, simd_width: Int
](value: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return abs(value)


# ===-----------------------------------------------------------------------===#
# ArgMax wrapper op
# ===-----------------------------------------------------------------------===#


# Call argmax, needed as it has multiple overloads which can't be aliased
@register_internal("mo.arg_max")
@always_inline
fn argmax_wrapped[
    type: DType,
    input_0_static_shape: DimList,
    out_type: DType,
    input_2_static_shape: DimList,
    rank: Int,
    target: StringLiteral = "cpu",
](
    input: NDBuffer[type, rank, input_0_static_shape],
    axis: Scalar,
    output: NDBuffer[out_type, rank, input_2_static_shape],
    ctx: MojoCallContextPtr,
) raises:
    constrained[target == "cpu" or "cuda" in target, "not a valid target"]()

    with Trace[TraceLevel.OP, target=target]("argmax"):

        @parameter
        if target == "cpu":
            _argmax(input, int(axis), output)
        else:
            var axis = int(normalize_neg_index(axis, rank))
            if axis != rank - 1:
                raise Error("axis other than -1 not supported on GPU")

            # TODO(KERN-1045): Add support for taking advantage of static_shapes
            var cuda_ctx = ctx.get_device_context()
            _argmax_gpu(
                cuda_ctx,
                rebind[NDBuffer[type, rank]](input),
                rebind[NDBuffer[out_type, rank]](output),
            )


# ===-----------------------------------------------------------------------===#
# ArgMin wrapper op
# ===-----------------------------------------------------------------------===#


# Call argmin, needed as it has multiple overloads which can't be aliased
@register_internal("mo.arg_min")
@always_inline
fn argmin_wrapped[
    type: DType,
    input_0_static_shape: DimList,
    out_type: DType,
    input_2_static_shape: DimList,
    rank: Int,
    target: StringLiteral = "cpu",
](
    input: NDBuffer[type, rank, input_0_static_shape],
    axis: Scalar,
    output: NDBuffer[out_type, rank, input_2_static_shape],
    ctx: MojoCallContextPtr,
) raises:
    constrained[target == "cpu" or "cuda" in target, "not a valid target"]()

    with Trace[TraceLevel.OP, target=target]("argmin"):

        @parameter
        if target == "cpu":
            _argmin(input, int(axis), output)
        else:
            var axis = int(normalize_neg_index(axis, rank))
            if axis != rank - 1:
                raise Error("axis other than -1 not supported on GPU")

            # TODO(KERN-1045): Add support for taking advantage of static_shapes
            var cuda_ctx = ctx.get_device_context()
            _argmin_gpu(
                cuda_ctx,
                rebind[NDBuffer[type, rank]](input),
                rebind[NDBuffer[out_type, rank]](output),
            )


# ===-----------------------------------------------------------------------===#
# Cast op
# ===-----------------------------------------------------------------------===#


# Cast a SIMD value to a new SIMD value of different type.
@register_internal("mo.cast")
@mogg_elementwise
@always_inline
fn cast[
    type: DType, new_type: DType, simd_width: Int
](value: SIMD[type, simd_width]) -> SIMD[new_type, simd_width]:
    return value.cast[new_type]()


# ===-----------------------------------------------------------------------===#
# Concat op
# ===-----------------------------------------------------------------------===#

from nn.concat import (
    elementwise_epilogue_type as concat_elementwise_epilogue_type,
)


@register_internal("mo.concat_from_list")
@always_inline
fn concat_from_list[
    input_type: DType,
    input_rank: Int,
    simd_width: Int,
    single_thread_blocking_override: Bool,
](
    inputs: InlinedFixedVector[NDBuffer[input_type, input_rank]],
    axis: Scalar,
    output: NDBuffer[input_type, input_rank],
    ctx: MojoCallContextPtr,
) raises:
    _concat_cpu[input_rank, input_type, None, single_thread_blocking_override](
        output,
        int(normalize_neg_index(axis, input_rank)),
        inputs,
    )


@register_internal("mo.concat")
@always_inline
fn concat[
    type: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
    # A tuple of callbacks, one per input.
    input_1_fn_tuple: StaticTuple[
        fn[
            width: Int, rank: Int
        ] (IndexList[rank]) capturing -> SIMD[type, width], *_
    ],
    output_0_fn: fn[width: Int, rank: Int, element_alignment: Int] (
        IndexList[rank], SIMD[type, width]
    ) capturing -> None,
    target: StringLiteral = "cpu",
](
    axis: Scalar,
    input_shapes: StaticTuple[IndexList[rank], input_1_fn_tuple.size],
    # TODO: we should probably take output shape here.
    output: NDBuffer[type, rank],
    ctx: MojoCallContextPtr,
) raises:
    var normalized_axis = int(normalize_neg_index(axis, rank))

    @always_inline
    @parameter
    fn epilogue_wrapper[
        _type: DType, _rank: Int, width: Int, *, alignment: Int = 1
    ](indices: IndexList[_rank], value: SIMD[_type, width]):
        output_0_fn[width, rank, alignment](
            rebind[IndexList[rank]](indices),
            rebind[SIMD[type, width]](value),
        )

    return test_concat_fusion[
        type,
        rank,
        single_thread_blocking_override,
        input_1_fn_tuple,
        epilogue_wrapper,
        target,
    ](normalized_axis, input_shapes, output, ctx)


@register_internal_shape_func("mo.concat")
@always_inline
fn concat_shape[
    input_type: DType,
    input_rank: Int,
    simd_width: Int,
    single_thread_blocking_override: Bool,
](
    axis0: Scalar,
    *input_bufs: NDBuffer[input_type, input_rank],
) raises -> IndexList[input_rank]:
    # TODO we should refactor this with `concat_from_list_shape`, but this
    # variadic input version has more static info (we _always_ know how many
    # input buffers there'll be for each invocation), thereby yielding simpler
    # KGEN IR. In order to obtain the same simple IR via
    # `concat_from_list_shape`, we need the KGEN optimizations to
    # 1. perform "@unroll_if_possible" when the loop bounds are not parameters.
    # 2. either optimize away the conversion from variadic list to
    # InlinedFixedVector, or generalize `concat_from_list_shape` with some List
    # Trait so we can pass in variadic list directly.
    var axis = int(normalize_neg_index(axis0, input_rank))
    if axis < 0 or input_rank <= axis:
        raise ("[concat] normalized axis must be within range [0, input_rank)")

    @parameter
    @always_inline
    fn shape_equal_ignore_axis(
        s1: IndexList[input_rank], s2: IndexList[input_rank]
    ) -> Bool:
        for i in range(input_rank):
            if i != axis and s1[i] != s2[i]:
                return False
        return True

    var concat_axis_dim_sum = 0

    for i in range(len(input_bufs)):
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


# ===-----------------------------------------------------------------------===#
# avg_pool
# ===-----------------------------------------------------------------------===#


@register_internal("mo.avg_pool")
@always_inline
fn avg_pool[
    type: DType,
    int_type: DType,
    count_boundary: Bool,
    target: StringLiteral = "cpu",
](
    input: NDBuffer[type, 4],
    filter: NDBuffer[int_type, 1],
    strides: NDBuffer[int_type, 1],
    dilations: NDBuffer[int_type, 1],
    paddings: NDBuffer[int_type, 1],
    output: NDBuffer[type, 4],
    ctx: MojoCallContextPtr,
) raises:
    constrained[target == "cpu" or "cuda" in target, "not a valid target"]()

    with Trace[TraceLevel.OP, target=target]("avg_pool"):

        @parameter
        if target == "cpu":
            _avg_pool[count_boundary=count_boundary](
                input, filter, strides, dilations, paddings, output
            )
        else:
            var cuda_ctx = ctx.get_device_context()
            _avg_pool_gpu[count_boundary=count_boundary](
                cuda_ctx, input, filter, strides, dilations, paddings, output
            )


# This handles avg_pool in the case where ceilMode = True. The default
# (ceilMode = False) case is handled by avg_pool above.
@register_internal("mo.avg_pool_ceil_mode_true")
@always_inline
fn avg_pool_ceil_mode_true[
    type: DType,
    int_type: DType,
    count_boundary: Bool,
    target: StringLiteral = "cpu",
](
    input: NDBuffer[type, 4],
    filter: NDBuffer[int_type, 1],
    strides: NDBuffer[int_type, 1],
    dilations: NDBuffer[int_type, 1],
    paddings: NDBuffer[int_type, 1],
    output: NDBuffer[type, 4],
    ctx: MojoCallContextPtr,
) raises:
    constrained[target == "cpu" or "cuda" in target, "not a valid target"]()

    with Trace[TraceLevel.OP, target=target]("avg_pool_ceil_mode_true"):

        @parameter
        if target == "cpu":
            _avg_pool[count_boundary=count_boundary](
                input, filter, strides, dilations, paddings, output, True
            )
        else:
            var cuda_ctx = ctx.get_device_context()
            _avg_pool_gpu[count_boundary=count_boundary](
                cuda_ctx,
                input,
                filter,
                strides,
                dilations,
                paddings,
                output,
                True,
            )


# ===-----------------------------------------------------------------------===#
# max_pool
# ===-----------------------------------------------------------------------===#


@register_internal("mo.max_pool")
@always_inline
fn max_pool[
    type: DType,
    int_type: DType,
    target: StringLiteral = "cpu",
](
    input: NDBuffer[type, 4],
    filter: NDBuffer[int_type, 1],
    strides: NDBuffer[int_type, 1],
    dilations: NDBuffer[int_type, 1],
    paddings: NDBuffer[int_type, 1],
    output: NDBuffer[type, 4],
    ctx: MojoCallContextPtr,
) raises:
    constrained[target == "cpu" or "cuda" in target, "not a valid target"]()

    with Trace[TraceLevel.OP, target=target]("max_pool"):

        @parameter
        if target == "cpu":
            _max_pool(input, filter, strides, dilations, paddings, output)
        else:
            var cuda_ctx = ctx.get_device_context()
            _max_pool_gpu(
                cuda_ctx, input, filter, strides, dilations, paddings, output
            )


# This handles max_pool in the case where ceilMode = True. The default
# (ceilMode = False) case is handled by max_pool above.
@register_internal("mo.max_pool_ceil_mode_true")
@always_inline
fn max_pool_ceil_mode_true[
    type: DType,
    int_type: DType,
    target: StringLiteral = "cpu",
](
    input: NDBuffer[type, 4],
    filter: NDBuffer[int_type, 1],
    strides: NDBuffer[int_type, 1],
    dilations: NDBuffer[int_type, 1],
    paddings: NDBuffer[int_type, 1],
    output: NDBuffer[type, 4],
    ctx: MojoCallContextPtr,
) raises:
    constrained[target == "cpu" or "cuda" in target, "not a valid target"]()

    with Trace[TraceLevel.OP, target=target]("max_pool_ceil_mode_true"):

        @parameter
        if target == "cpu":
            _max_pool(input, filter, strides, dilations, paddings, output, True)
        else:
            var cuda_ctx = ctx.get_device_context()
            _max_pool_gpu(
                cuda_ctx,
                input,
                filter,
                strides,
                dilations,
                paddings,
                output,
                True,
            )


# ===-----------------------------------------------------------------------===#
# Cumsum op
# ===-----------------------------------------------------------------------===#


@register_internal("mo.cumsum")
@always_inline
fn cumsum[
    type: DType,
    rank: Int,
    exclusive: Int,
    reverse: Int,
](
    input: NDBuffer[type, rank],
    axis: Scalar,
    output: NDBuffer[type, rank],
    ctx: MojoCallContextPtr,
):
    _cumsum[rank, type, exclusive == 1, reverse == 1](
        output, input, int(normalize_neg_index(axis, rank))
    )


# ===-----------------------------------------------------------------------===#
# Split op
# ===-----------------------------------------------------------------------===#


# Not targeted yet because MOGG assumes single output
@register_internal("mo.split")
@always_inline
fn split[
    type: DType,
    rank: Int,
    simd_width: Int,
    split_sizes_type: DType,
](
    input: NDBuffer[type, rank],
    split_sizes: NDBuffer[split_sizes_type, 1],
    axis: Scalar,
    ctx: MojoCallContextPtr,
    *variadic_outs: NDBuffer[type, rank],
) raises:
    # NOTE: Synchronous, so stack allocated variadic list is safe
    _split[type, rank](
        input, int(normalize_neg_index(axis, rank)), variadic_outs
    )


# ===-----------------------------------------------------------------------===#
# Pow wrapper op
# ===-----------------------------------------------------------------------===#


# Call pow, needed as it has multiple overloads which can't be aliased
@register_internal("mo.pow")
@mogg_elementwise
@always_inline
fn pow_wrapped[
    type: DType, power_type: DType, simd_width: Int
](value: SIMD[type, simd_width], power: SIMD[power_type, simd_width]) -> SIMD[
    type, simd_width
]:
    return _pow(value, power)


# ===-----------------------------------------------------------------------===#
# Sqrt wrapper op
# ===-----------------------------------------------------------------------===#


# Call sqrt, needed as it has multiple overloads which can't be aliased
@register_internal("mo.sqrt")
@mogg_elementwise
@always_inline
fn sqrt_wrapped[
    type: DType, simd_width: Int
](value: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return sqrt(value)


# ===-----------------------------------------------------------------------===#
# Max & min ops
# ===-----------------------------------------------------------------------===#

# These need wrappers as we can't take an alias of the ambiguous overload.


@register_internal("mo.max")
@mogg_elementwise
@always_inline
fn mogg_max[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    type, simd_width
]:
    return max(x, y)


@register_internal("mo.min")
@mogg_elementwise
@always_inline
fn mogg_min[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    type, simd_width
]:
    return min(x, y)


# ===-----------------------------------------------------------------------===#
# Mean op
# ===-----------------------------------------------------------------------===#


@register_internal("mo.mean")
@always_inline
fn mean[
    type: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
    input_0_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        type, width
    ],
    output_0_fn: fn[width: Int, rank: Int, element_alignment: Int] (
        IndexList[rank], SIMD[type, width]
    ) capturing -> None,
    /,
    target: StringLiteral = "cpu",
](
    input_shape: IndexList[rank],
    axis: Scalar,
    output_shape: IndexList[rank],
    ctx: MojoCallContextPtr,
) raises:
    _mean[
        type,
        input_0_fn,
        output_0_fn[element_alignment=1],
        target=target,
        single_thread_blocking_override=single_thread_blocking_override,
    ](input_shape, int(axis), output_shape, context=ctx)


# ===-----------------------------------------------------------------------===#
# Negative op
# ===-----------------------------------------------------------------------===#


@register_internal("mo.negative")
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


# ===-----------------------------------------------------------------------===#
# Pad .* op
# ===-----------------------------------------------------------------------===#


@register_internal("mo.pad.constant")
@always_inline
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


@register_internal("mo.pad.reflect")
@always_inline
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


@register_internal("mo.pad.repeat")
@always_inline
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


# ===-----------------------------------------------------------------------===#
# Reduction ops
# ===-----------------------------------------------------------------------===#


@register_internal_shape_func("mo.arg_max")
@register_internal_shape_func("mo.arg_min")
@register_internal_shape_func("mo.mean")
@register_internal_shape_func("mo.reduce.add")
@register_internal_shape_func("mo.reduce.max")
@register_internal_shape_func("mo.reduce.min")
@register_internal_shape_func("mo.reduce.mul")
@always_inline("nodebug")
fn reduce_shape[
    input_rank: Int,
    input_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[input_type, input_rank],
    axis0: Scalar,
) raises -> IndexList[input_rank]:
    """
    Compute the output shape of a `reduce` operation, and assert the inputs are
    compatible.

    Parameters:
        input_rank: Input_rank of the input tensor.
        input_type: Type of the input tensor.
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.

    Args:
        input_buf: The input tensor.
        axis0: The axis tensor.

    Returns:
        The output shape.
    """

    var axis = int(normalize_neg_index(axis0, input_rank))

    if axis < 0 or input_rank <= axis:
        raise Error(
            "[reduction] normalized axis must be within range [0, input_rank)"
        )

    # compute and return the output shape
    var output_shape = input_buf.get_shape()
    output_shape[axis] = 1
    return output_shape


@register_internal("mo.reduce.add")
@always_inline
fn reduce_add[
    type: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
    input_0_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        type, width
    ],
    output_0_fn: fn[width: Int, rank: Int, element_alignment: Int] (
        IndexList[rank], SIMD[type, width]
    ) capturing -> None,
    target: StringLiteral = "cpu",
](
    input_shape: IndexList[rank],
    axis: Scalar,
    output_shape: IndexList[rank],
    ctx: MojoCallContextPtr,
) raises:
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
    ](indices: IndexList[rank], value: SIMD[_type, width]):
        output_0_fn[width, rank, element_alignment=1](
            indices, rebind[SIMD[type, width]](value)
        )

    @always_inline
    @parameter
    fn reduce_impl[
        ty: DType, width: Int
    ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
        return v1 + v2

    with Trace[TraceLevel.OP, target=target]("reduce_add"):
        _reduce_generator[
            input_0_fn_wrapper,
            output_0_fn_wrapper,
            reduce_impl,
            target=target,
            single_thread_blocking_override=single_thread_blocking_override,
        ](input_shape, Scalar[type](0), int(axis), context=ctx)


@register_internal("mo.reduce.max")
@always_inline
fn reduce_max[
    type: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
    input_0_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        type, width
    ],
    output_0_fn: fn[width: Int, rank: Int, element_alignment: Int] (
        IndexList[rank], SIMD[type, width]
    ) capturing -> None,
    target: StringLiteral = "cpu",
](
    input_shape: IndexList[rank],
    axis: Scalar,
    output_shape: IndexList[rank],
    ctx: MojoCallContextPtr,
) raises:
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
    ](indices: IndexList[rank], value: SIMD[_type, width]):
        output_0_fn[width, rank, element_alignment=1](
            indices, rebind[SIMD[type, width]](value)
        )

    @always_inline
    @parameter
    fn reduce_impl[
        ty: DType, width: Int
    ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
        return max(v1, v2)

    with Trace[TraceLevel.OP, target=target]("reduce_max"):
        _reduce_generator[
            input_0_fn_wrapper,
            output_0_fn_wrapper,
            reduce_impl,
            target=target,
            single_thread_blocking_override=single_thread_blocking_override,
        ](input_shape, Scalar[type].MIN, int(axis), context=ctx)


@register_internal("mo.reduce.min")
@always_inline
fn reduce_min[
    type: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
    input_0_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        type, width
    ],
    output_0_fn: fn[width: Int, rank: Int, element_alignment: Int] (
        IndexList[rank], SIMD[type, width]
    ) capturing -> None,
    target: StringLiteral = "cpu",
](
    input_shape: IndexList[rank],
    axis: Scalar,
    output_shape: IndexList[rank],
    ctx: MojoCallContextPtr,
) raises:
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
    ](indices: IndexList[rank], value: SIMD[_type, width]):
        output_0_fn[width, rank, element_alignment=1](
            indices, rebind[SIMD[type, width]](value)
        )

    @always_inline
    @parameter
    fn reduce_impl[
        ty: DType, width: Int
    ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
        return min(v1, v2)

    with Trace[TraceLevel.OP, target=target]("reduce_min"):
        _reduce_generator[
            input_0_fn_wrapper,
            output_0_fn_wrapper,
            reduce_impl,
            target=target,
            single_thread_blocking_override=single_thread_blocking_override,
        ](input_shape, Scalar[type].MAX, int(axis), context=ctx)


@register_internal("mo.reduce.mul")
@always_inline
fn reduce_mul[
    type: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
    input_0_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        type, width
    ],
    output_0_fn: fn[width: Int, rank: Int, element_alignment: Int] (
        IndexList[rank], SIMD[type, width]
    ) capturing -> None,
    target: StringLiteral = "cpu",
](
    input_shape: IndexList[rank],
    axis: Scalar,
    output_shape: IndexList[rank],
    ctx: MojoCallContextPtr,
) raises:
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
    ](indices: IndexList[rank], value: SIMD[_type, width]):
        output_0_fn[width, rank, element_alignment=1](
            indices, rebind[SIMD[type, width]](value)
        )

    @always_inline
    @parameter
    fn reduce_impl[
        ty: DType, width: Int
    ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
        return v1 * v2

    with Trace[TraceLevel.OP, target=target]("reduce_mul"):
        _reduce_generator[
            input_0_fn_wrapper,
            output_0_fn_wrapper,
            reduce_impl,
            target=target,
            single_thread_blocking_override=single_thread_blocking_override,
        ](input_shape, Scalar[type](1), int(axis), context=ctx)


# ===-----------------------------------------------------------------------===#
# Slice op
# ===-----------------------------------------------------------------------===#


# Wrapper for slice here to include the `single_thread_blocking_override`.
@register_internal("mo.slice")
@mogg_view_op
@always_inline
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
    return slice_as_view(tensor, starts, ends, steps)


@register_internal("mo.mutable.store.slice")
@always_inline
@export
fn store_slice[
    type: DType,
    start_type: DType,
    end_type: DType,
    step_type: DType,
    in_rank: Int,
    single_thread_blocking_override: Bool,
    target: StringLiteral = "cpu",
](
    buffer: NDBuffer[type, in_rank],
    in_slice: NDBuffer[type, in_rank],
    start: NDBuffer[start_type, 1],
    end: NDBuffer[end_type, 1],
    step: NDBuffer[step_type, 1],
    ctx: MojoCallContextPtr,
) raises:
    copy_to_slice[target=target](buffer, in_slice, start, end, step, ctx)


@register_internal("mo.slice_dim")
@mogg_view_op
@always_inline
fn slice_dim[
    type: DType,
    rank: Int,
    dim: Int,
    single_thread_blocking_override: Bool,
    target: StringLiteral = "cpu",
](
    tensor: NDBuffer[type, rank],
    start: Scalar,
    end: Scalar,
    step: Scalar,
    ctx: MojoCallContextPtr,  # remove (#24946)
) -> NDBuffer[type, rank]:
    return slice_dim_as_view[type, rank, dim](
        tensor, int(start), int(end), int(step)
    )


# ===-----------------------------------------------------------------------===#
# SqueezeShape
# ===-----------------------------------------------------------------------===#


@register_internal("mo.squeeze_shape")
@always_inline
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
    var input_shape_copy = IndexList[MAX_VECTOR_LIMIT]()
    for i in range(num_input_dims):
        input_shape_copy[i] = int(input_shape[i])

    # Mark every squeezed dimension as -1 in our copy of the shape tensor
    for remove_index_index in range(num_remove_indices):
        var remove_index = int(remove_indices[remove_index_index])
        var remove_index_normalize = remove_index + num_input_dims * int(
            remove_indices[remove_index_index] < 0
        )
        input_shape_copy[remove_index_normalize] = -1

    # # Copy over the non -1 dimensions
    var output_shape_index = 0
    for input_shape_index in range(num_input_dims):
        if input_shape_copy[input_shape_index] == -1:
            continue
        output_shape[output_shape_index] = input_shape_copy[input_shape_index]
        output_shape_index += 1


@register_internal_shape_func("mo.squeeze_shape")
@always_inline
fn squeeze_shape_shape[
    type: DType, indices_type: DType, single_thread_blocking_override: Bool
](
    input_shape: NDBuffer[type, 1],
    remove_indices: NDBuffer[indices_type, 1],
) raises -> IndexList[1]:
    var out_dim = input_shape.dim(0) - remove_indices.dim(0)

    if out_dim < 0:
        raise Error(
            "[squeeze_shape] cannot remove more dimensions than there exists"
        )

    return IndexList[1](out_dim)


# ===-----------------------------------------------------------------------===#
# UnsqueezeShape op
# ===-----------------------------------------------------------------------===#


@register_internal("mo.unsqueeze_shape")
@always_inline
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


@register_internal_shape_func("mo.unsqueeze_shape")
@always_inline
fn unsqueeze_shape_shape[
    type: DType, indices_type: DType, single_thread_blocking_override: Bool
](
    input_shape: NDBuffer[type, 1],
    padding_indices: NDBuffer[indices_type, 1],
) -> IndexList[1]:
    var out_dim = input_shape.dim(0) + padding_indices.dim(0)
    return IndexList[1](out_dim)


# ===-----------------------------------------------------------------------===#
# Transpose op
# ===-----------------------------------------------------------------------===#


@register_internal("mo.transpose")
@mogg_view_op
@always_inline
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
    var new_shape = IndexList[rank]()
    var new_stride = IndexList[rank]()

    @parameter
    for i in range(rank):
        var dim = int(perms[i])
        new_shape[i] = input.dim(dim)
        new_stride[i] = input.stride(dim)

    # Create the transposed view.
    return NDBuffer[type, rank](input.data, new_shape, new_stride)


@register_internal_shape_func("mo.transpose")
@always_inline
fn transpose_shape[
    rank: Int,
    type: DType,
    int_type: DType,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[type, rank],
    perms: NDBuffer[int_type, 1],
) raises -> IndexList[rank]:
    if perms.dim(0) != rank:
        raise Error("[transpose] permutation size must match input rank")

    for i in range(rank):
        var perm = int(perms[i])
        if perm < 0 or rank <= perm:
            raise Error(
                "[transpose] each permutation must be within range [0, rank)"
            )

    # NOTE this assumes `transpose` can handle input with null data pointer
    return transpose[rank, type, int_type, single_thread_blocking_override](
        input, perms, MojoCallContextPtr()
    ).get_shape()


# ===-----------------------------------------------------------------------===#
# Gather
# ===-----------------------------------------------------------------------===#


# TODO(#20442): Remove with generic fusion.
@register_internal("mo.gather_sum")
@always_inline
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
        return String(";").join(
            trace_arg("output", output),
            trace_arg("input", input),
            trace_arg("indices", indices),
        )

    with Trace[TraceLevel.OP, target="cpu"](
        "gather_sum",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
    ):
        gather_reduce[type, 0, 1, simdwidthof[type](), add](
            output, input, indices, 0
        )


@register_internal("mo.gather")
@always_inline
fn gather[
    type: DType,
    in_rank: Int,
    indices_type: DType,
    indices_rank: Int,
    output_rank: Int,
    simd_width: Int,
    single_thread_blocking_override: Bool,
    input_0_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        type, width
    ],
    output_0_fn: fn[width: Int, rank: Int, element_alignment: Int] (
        IndexList[rank], SIMD[type, width]
    ) capturing -> None,
    target: StringLiteral = "cpu",
](
    input_shape: IndexList[in_rank],
    indices: NDBuffer[indices_type, indices_rank],
    axis: Scalar,
    output_shape: IndexList[output_rank],
    ctx: MojoCallContextPtr,
) raises:
    # TODO: This is disabled as if we make this a shape without a spec we have
    # nothing to deduce `indices_type` from.
    @parameter
    @always_inline
    fn load_indices[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[indices_type, width]:
        return indices.load[width=width](
            rebind[IndexList[indices_rank]](coords)
        )

    # FIXME(#26008): async raising functions are temporarily disabled.
    _gather[
        type=type,
        indices_type=indices_type,
        input_fn=input_0_fn,
        indices_fn=load_indices,
        output_fn = output_0_fn[element_alignment=1],
        target=target,
        single_thread_blocking_override=single_thread_blocking_override,
    ](
        Axis(axis, in_rank),
        input_shape,
        indices.dynamic_shape,
        output_shape,
        context=ctx,
    )


# ===-----------------------------------------------------------------------===#
# MOGG matmul
# ===-----------------------------------------------------------------------===#

from linalg.bmm import (
    elementwise_epilogue_type as batched_matmul_elementwise_epilogue_type,
)

# TODO(#29765): remove import and allow Optional type to be inferred
from linalg.utils import (
    elementwise_epilogue_type as matmul_elementwise_epilogue_type,
)


@register_internal("mo.matmul")
@always_inline
fn matmul[
    a_type: DType,
    input_0_static_shape: DimList,
    alignment_0: Int,
    b_type: DType,
    input_1_static_shape: DimList,
    alignment_1: Int,
    c_type: DType,
    input_2_static_shape: DimList,
    alignment_2: Int,
    transpose_b: Bool,  # matches name of MO attribute
    packed_b: Bool,
    single_thread_blocking_override: Bool,
    lambdas_have_fusion: Bool,
    output_0_fn: fn[width: Int, rank: Int, element_alignment: Int] (
        IndexList[rank], SIMD[c_type, width]
    ) capturing -> None,
    /,
    trace_description: StringLiteral,
    target: StringLiteral = "cpu",
](
    a: NDBuffer[a_type, 2, input_0_static_shape],
    b: NDBuffer[b_type, 2, input_1_static_shape],
    c: NDBuffer[c_type, 2, input_2_static_shape],
    ctx: MojoCallContextPtr,
) raises:
    alias transpose_a = False

    constrained[
        not (packed_b and transpose_b),
        (
            "transpose_b and packed_b cannot both be true because pre-packing"
            " transposes B"
        ),
    ]()

    @parameter
    @always_inline
    fn epilogue_wrapper[
        _type: DType, width: Int, *, alignment: Int = 1
    ](coords: IndexList[2], val: SIMD[_type, width]):
        output_0_fn[width, 2, alignment](
            coords, rebind[SIMD[c_type, width]](val)
        )

    _matmul[
        transpose_a,
        transpose_b,
        packed_b,
        OptionalReg[matmul_elementwise_epilogue_type](
            epilogue_wrapper
        ) if lambdas_have_fusion else None,
        saturated_vnni=False,
        single_thread_blocking_override=single_thread_blocking_override,
        target=target,
        trace_description=trace_description,
    ](c, a, b, ctx)


# ===-----------------------------------------------------------------------===#
# MOGG batched matmul
# ===-----------------------------------------------------------------------===#


@register_internal("mo.batch_matmul")
@always_inline
fn batched_matmul[
    rank: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool,
    single_thread_blocking_override: Bool,
    lambdas_have_fusion: Bool,
    output_0_fn: fn[width: Int, rank: Int, element_alignment: Int] (
        IndexList[rank], SIMD[c_type, width]
    ) capturing -> None,
    target: StringLiteral = "cpu",
](
    a: NDBuffer[a_type, rank],
    b: NDBuffer[b_type, rank],
    c: NDBuffer[c_type, rank],
    ctx: MojoCallContextPtr,
) raises:
    alias transpose_a = False

    @parameter
    @always_inline
    fn epilogue_wrapper[
        _type: DType,
        width: Int,
        rank: Int,
        *,
        alignment: Int = 1,
    ](coords: IndexList[rank], val: SIMD[_type, width],):
        output_0_fn[width, rank, alignment](
            coords.canonicalize(), rebind[SIMD[c_type, width]](val)
        )

    _batched_matmul[
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        elementwise_epilogue_fn = OptionalReg[
            batched_matmul_elementwise_epilogue_type
        ](epilogue_wrapper) if lambdas_have_fusion else None,
        saturated_vnni=False,
        single_thread_blocking_override=single_thread_blocking_override,
        target=target,
    ](c, a, b, context=ctx)


# ===-----------------------------------------------------------------------===#
# MOGG scatter
# ===-----------------------------------------------------------------------===#


@register_internal("mo.scatter")
@always_inline
fn scatter[
    rank: Int,
    input_type: DType,
    indices_type: DType,
](
    input: NDBuffer[input_type, rank],
    updates: NDBuffer[input_type, rank],
    indices: NDBuffer[indices_type, rank],
    axis: Scalar,
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
        int(normalize_neg_index(axis, rank)),
        output,
    )


@register_internal("mo.scatter.add")
@always_inline
fn scatter_add[
    rank: Int,
    input_type: DType,
    indices_type: DType,
](
    input: NDBuffer[input_type, rank],
    updates: NDBuffer[input_type, rank],
    indices: NDBuffer[indices_type, rank],
    axis: Scalar,
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
        int(normalize_neg_index(axis, rank)),
        output,
    )


@register_internal("mo.scatter.max")
@always_inline
fn scatter_max[
    rank: Int,
    input_type: DType,
    indices_type: DType,
](
    input: NDBuffer[input_type, rank],
    updates: NDBuffer[input_type, rank],
    indices: NDBuffer[indices_type, rank],
    axis: Scalar,
    output: NDBuffer[input_type, rank],
    ctx: MojoCallContextPtr,
) raises:
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
        int(normalize_neg_index(axis, rank)),
        output,
    )


@register_internal("mo.scatter.min")
@always_inline
fn scatter_min[
    rank: Int,
    input_type: DType,
    indices_type: DType,
](
    input: NDBuffer[input_type, rank],
    updates: NDBuffer[input_type, rank],
    indices: NDBuffer[indices_type, rank],
    axis: Scalar,
    output: NDBuffer[input_type, rank],
    ctx: MojoCallContextPtr,
) raises:
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
        int(normalize_neg_index(axis, rank)),
        output,
    )


@register_internal("mo.scatter.mul")
@always_inline
fn scatter_mul[
    rank: Int,
    input_type: DType,
    indices_type: DType,
](
    input: NDBuffer[input_type, rank],
    updates: NDBuffer[input_type, rank],
    indices: NDBuffer[indices_type, rank],
    axis: Scalar,
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
        int(normalize_neg_index(axis, rank)),
        output,
    )


# ===-----------------------------------------------------------------------===#
# MOGG scatter_nd
# ===-----------------------------------------------------------------------===#


@register_internal("mo.scatter_nd")
@always_inline
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
    _scatter_nd[
        output_type,
        indices_type,
        output_rank,
        indices_rank,
        updates_rank,
        single_thread_blocking_override,
        target,
    ](input, indices, updates, output, context=ctx)


@register_internal("mo.scatter_nd.add")
@always_inline
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
    ](input, indices, updates, output, context=ctx)


@register_internal("mo.scatter_nd.max")
@always_inline
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
    @always_inline
    @parameter
    fn reduce_fn[
        type: DType, width: Int
    ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
        return max(lhs, rhs)

    scatter_nd_generator[
        output_type,
        indices_type,
        output_rank,
        indices_rank,
        updates_rank,
        single_thread_blocking_override,
        target,
        reduce_fn=reduce_fn,
    ](input, indices, updates, output, context=ctx)


@register_internal("mo.scatter_nd.min")
@always_inline
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
    @always_inline
    @parameter
    fn reduce_fn[
        type: DType, width: Int
    ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
        return min(lhs, rhs)

    scatter_nd_generator[
        output_type,
        indices_type,
        output_rank,
        indices_rank,
        updates_rank,
        single_thread_blocking_override,
        target,
        reduce_fn=reduce_fn,
    ](input, indices, updates, output, context=ctx)


@register_internal("mo.scatter_nd.mul")
@always_inline
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
    ](input, indices, updates, output, context=ctx)


# Define a wrapper in MOGG.mojo so that softmax kernel in stdlib takes static shapes
@register_internal("mo.softmax")
@always_inline
fn softmax[
    rank: Int,
    type: DType,
    input_0_fn: fn[_simd_width: Int, _rank: Int] (
        IndexList[_rank]
    ) capturing -> SIMD[type, _simd_width],
    target: StringLiteral = "cpu",
](
    shape: IndexList[rank],
    output: NDBuffer[type, rank],
    ctx: MojoCallContextPtr,
) raises:
    _softmax[
        type,
        simdwidthof[type](),
        rank,
        DimList.create_unknown[rank](),
        input_0_fn,
        target,
    ](shape, output, rank - 1, context=ctx)


# Define a wrapper in MOGG.mojo so that softmax kernel in stdlib takes static shapes
@register_internal("mo.logsoftmax")
@always_inline
fn logsoftmax[
    rank: Int,
    type: DType,
    input_0_fn: fn[_simd_width: Int, _rank: Int] (
        IndexList[_rank]
    ) capturing -> SIMD[type, _simd_width],
](
    shape: IndexList[rank],
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


# ===-----------------------------------------------------------------------===#
# MOGG non_maximum_suppression
# ===-----------------------------------------------------------------------===#


@register_internal("mo.non_maximum_suppression")
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


@register_internal_shape_func("mo.non_maximum_suppression")
@always_inline
fn non_maximum_suppression_shape_func[
    type: DType, single_thread_blocking_override: Bool
](
    boxes: NDBuffer[type, 3],
    scores: NDBuffer[type, 3],
    max_output_boxes_per_class: NDBuffer[DType.int64, 1, DimList(1)],
    iou_threshold: NDBuffer[DType.float32, 1, DimList(1)],
    score_threshold: NDBuffer[DType.float32, 1, DimList(1)],
) -> IndexList[2]:
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


# ===-----------------------------------------------------------------------===#
# MOGG mo.random.normal
# ===-----------------------------------------------------------------------===#

# TODO(31691): Correctly handle PRNG state with asynchronous runtime


@register_internal("mo.random.normal")
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


@register_internal("mo.static.random.normal")
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


@register_internal_shape_func("mo.random.normal")
@always_inline
fn random_shape[
    shapeType: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
](shape: NDBuffer[shapeType, 1, DimList(rank)],) -> IndexList[rank]:
    var unrolledShape = IndexList[rank]()

    for i in range(rank):
        unrolledShape[i] = int(shape[i])
    return unrolledShape


# ===-----------------------------------------------------------------------===#
# MOGG resize
# ===-----------------------------------------------------------------------===#


@register_internal("mo.resize.nearest")
@always_inline
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


@register_internal("mo.resize.linear")
@always_inline
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


@register_internal_shape_func("mo.resize.nearest")
@always_inline
fn resize_nearest_shape[
    rank: Int,
    inpType: DType,
    sizeType: DType,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[inpType, rank],
    size: NDBuffer[sizeType, 1, DimList(rank)],
) -> IndexList[rank]:
    var shape = IndexList[rank]()

    for i in range(rank):
        shape[i] = int(size[i])
    return shape


@register_internal_shape_func("mo.resize.linear")
@always_inline
fn resize_linear_shape[
    rank: Int,
    inpType: DType,
    sizeType: DType,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[inpType, rank],
    size: NDBuffer[sizeType, 1, DimList(rank)],
) -> IndexList[rank]:
    var shape = IndexList[rank]()

    for i in range(rank):
        shape[i] = int(size[i])
    return shape


# ===-----------------------------------------------------------------------===#
# MOGG ROI Align
# ===-----------------------------------------------------------------------===#


@register_internal("mo.roi_align")
fn roi_align[
    type: DType, aligned: Bool, mode: StringLiteral
](
    input: NDBuffer[type, 4, *_],
    rois: NDBuffer[type, 2, *_],
    output_height: Int64,
    output_width: Int64,
    spatial_scale: Scalar,
    sampling_ratio: Scalar,
    output: NDBuffer[type, 4],
    ctx: MojoCallContextPtr,
):
    roi_align_nhwc[aligned, mode](
        output,
        input,
        rois,
        int(output_height),
        int(output_width),
        spatial_scale,
        sampling_ratio,
    )


@register_internal_shape_func("mo.roi_align")
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
) -> IndexList[4]:
    var shape = IndexList[4]()

    # input shape is [N, H, W, C]
    # rois shape is [M, 5]
    # output shape is [M, output_height, output_width, C]
    shape[0] = rois.get_shape()[0]
    shape[1] = int(output_height[0])
    shape[2] = int(output_width[0])
    shape[3] = input.get_shape()[3]

    return shape


# ===-----------------------------------------------------------------------===#
# MOGG split
# ===-----------------------------------------------------------------------===#


@register_internal("split_ith_output_shape")
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
) raises -> IndexList[rank]:
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


@register_internal("mo.conv")
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
    output_type: DType,
    input_6_static_shape: DimList,
    filter_packed: Bool,
    lambdas_have_fusion: Bool,
    static_strides: DimList,
    static_dilations: DimList,
    static_padding: DimList,
    output_0_fn: fn[width: Int, rank: Int, element_alignment: Int] (
        IndexList[rank], SIMD[output_type, width]
    ) capturing -> None,
    target: StringLiteral = "cpu",
](
    input: NDBuffer[input_type, input_rank, input_0_static_shape],
    filter: NDBuffer[filter_type, filter_rank, input_1_static_shape],
    strides: NDBuffer[strides_type, strides_rank],
    dilation: NDBuffer[dilation_type, dilation_rank],
    paddings: NDBuffer[padding_type, padding_rank],
    num_groups: Scalar,
    # output and input have the same rank.
    output: NDBuffer[output_type, input_rank, input_6_static_shape],
    ctx: MojoCallContextPtr,
) raises:
    """Including this function in MOGG.mojo since it is intended to be a temporary
    wrapper around the Stdlib conv. Currently the strides and dilation are NDBuffers,
    but eventually they will be IndexList parameters (along with padding).
    """
    constrained[target == "cpu" or "cuda" in target, "not a valid target"]()
    constrained[
        strides_type.is_integral() and dilation_type.is_integral(),
        "stride and dilation must have integral type",
    ]()

    if strides.size() != input_rank - 2:
        raise Error("$(input_rank-2) values expected in conv strides")

    if dilation.size() != input_rank - 2:
        raise Error("$(input_rank-2) values expected in conv dilation")

    if paddings.size() != 2 * (input_rank - 2):
        raise Error("$(2*(input_rank-2)) value expected in conv paddings")

    var stride_flat = strides.flatten()
    var dilation_flat = dilation.flatten()
    var padding_flat = paddings.flatten()

    var stride_tuple = IndexList[input_rank - 2](0)
    var dilation_tuple = IndexList[input_rank - 2](0)

    @parameter
    for i in range(input_rank - 2):
        stride_tuple[i] = int(stride_flat[i])
        dilation_tuple[i] = int(dilation_flat[i])

    if dilation_tuple != IndexList[input_rank - 2](1):
        raise Error("Non-unit dilation is not supported yet.")

    var pad_d_tuple = IndexList[2](0)
    var pad_h_tuple = IndexList[2](0)
    var pad_w_tuple = IndexList[2](0)

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

    # Specialize the function to take 4D coordinates.
    # The bias is broadcasted to the same shape as output and
    # accessed by the 4D coordinates.
    @parameter
    @always_inline
    fn epilogue_wrapper[
        _type: DType, _rank: Int, _width: Int
    ](coords: IndexList[_rank], val: SIMD[_type, _width]):
        output_0_fn[_width, _rank, element_alignment=1](
            coords, rebind[SIMD[output_type, _width]](val)
        )

    @parameter
    if target == "cpu":
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
    else:
        constrained[
            input_rank == 4 and filter_rank == 4,
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
            input_rank,
            filter_rank,
            input_0_static_shape,  # input shape
            input_1_static_shape,  # filter shape
            input_6_static_shape,  # output shape
            input_type,
            filter_type,
            output_type,
        ](
            input,
            filter,
            output,
            IndexList[2](stride_tuple[0], stride_tuple[1]),
            IndexList[2](dilation_tuple[0], dilation_tuple[1]),
            IndexList[2](pad_h_tuple[0], pad_w_tuple[0]),
            int(num_groups[0]),
            cuda_ctx,
        )


@register_internal("mo.conv_transpose")
@always_inline
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
    output_0_fn: fn[width: Int, rank: Int, element_alignment: Int] (
        IndexList[rank], SIMD[output_type, width]
    ) capturing -> None,
](
    input: NDBuffer[input_type, input_rank],
    filter: NDBuffer[filter_type, filter_rank],
    strides: NDBuffer[strides_type, 1],
    dilation: NDBuffer[dilation_type, 1],
    paddings: NDBuffer[padding_type, 1],
    output_paddings: NDBuffer[output_padding_type, 1],
    output: NDBuffer[output_type, input_rank],
) raises:
    constrained[
        strides_type.is_integral()
        and dilation_type.is_integral()
        and output_padding_type.is_integral(),
        "stride, dilation and output_paddings must have integral type",
    ]()

    if strides.size() != input_rank - 2:
        raise Error("$(input_rank-2) values expected in convTranspose stride")

    if dilation.size() != input_rank - 2:
        raise Error("$(input_rank-2) values expected in convTranspose dilation")

    if output_paddings.size() != input_rank - 2:
        raise Error(
            "$(input_rank-2) values expected in convTranspose output paddings"
        )

    if paddings.size() != 2 * (input_rank - 2):
        raise Error(
            "$(2*(input_rank-2)) value expected in convTranspose paddings"
        )

    var stride_tuple = IndexList[input_rank - 2](0)
    var dilation_tuple = IndexList[input_rank - 2](0)

    @parameter
    for i in range(input_rank - 2):
        stride_tuple[i] = int(strides[i])
        dilation_tuple[i] = int(dilation[i])

    var pad_d = IndexList[2](0)
    var pad_h = IndexList[2](0)
    var pad_w = IndexList[2](0)

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

    @parameter
    @always_inline
    fn epilogue_wrapper[
        _type: DType, _rank: Int, _width: Int
    ](coords: IndexList[_rank], val: SIMD[_type, _width]):
        output_0_fn[_width, _rank, element_alignment=1](
            coords, rebind[SIMD[output_type, _width]](val)
        )

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


# ===-----------------------------------------------------------------------===#
# Helpers
# ===-----------------------------------------------------------------------===#


# Helper function to query buffer shapes for tests.
@register_internal("print_shape_info")
fn print_buffer_info[type: DType, rank: Int](buffer: NDBuffer[type, rank]):
    print("Rank:", rank)
    print("Shape:", buffer.get_shape())
    print("Strides:", buffer.get_strides())


# Test helper to throw an error
@register_internal("mo.test.return_error")
@always_inline
fn return_error[
    type: DType, rank: Int
](input: NDBuffer[type, rank], ctx: MojoCallContextPtr) raises:
    raise Error("This is an error")


@register_internal("mo.test.failing_constraint")
@always_inline
fn kernel_with_failing_constraint[
    type: DType, rank: Int
](input: NDBuffer[type, rank], ctx: MojoCallContextPtr):
    constrained[
        1 == 2,
        "Expected constraint failure for error message testing",
    ]()


@register_internal("mo.test.abort")
@always_inline
fn test_abort[
    type: DType, rank: Int
](input: NDBuffer[type, rank], ctx: MojoCallContextPtr) raises:
    abort()


# ===-----------------------------------------------------------------------===#
# TopK/BottomK
# ===-----------------------------------------------------------------------===#


@register_internal("mo.bottom_k")
@always_inline
fn bottom_k[
    type: DType,
    rank: Int,
    out_idxs_type: DType = DType.int64,
    target: StringLiteral = "cpu",
](
    input: NDBuffer[type, rank],
    k_buf: Scalar,
    axis: Scalar,
    sorted: NDBuffer[DType.bool, 1],
    out_vals: NDBuffer[type, rank],
    out_idxs: NDBuffer[out_idxs_type, rank],
    ctx: MojoCallContextPtr,
) raises:
    constrained[target == "cpu" or "cuda" in target, "not a valid target"]()

    with Trace[TraceLevel.OP, target=target]("bottom_k"):

        @parameter
        if target == "cpu":
            constrained[
                out_idxs_type == DType.int64,
                "out_idxs_type must be int64 for cpu",
            ]()
            _top_k[rank, type](
                input,
                int(k_buf),
                int(axis),
                False,
                rebind[NDBuffer[type, rank]](out_vals),
                rebind[NDBuffer[DType.int64, rank]](out_idxs),
                sorted[0],
            )

        else:
            var axis = int(normalize_neg_index(axis, rank))
            if axis != rank - 1:
                raise Error("axis other than -1 not supported on GPU")
            if not sorted[0]:
                print(
                    "Warning: Unsorted top-k is not supported on GPU. Falling"
                    " back to sorted top-k."
                )

            var cuda_ctx = ctx.get_device_context()
            _topk_gpu[
                type,
                rank,
                out_idxs_type,
                sampling=False,
                largest=False,
            ](
                cuda_ctx,
                int(k_buf),
                input,
                out_vals,
                out_idxs,
            )


@register_internal("mo.top_k")
@always_inline
fn top_k[
    type: DType,
    rank: Int,
    out_idxs_type: DType = DType.int64,
    target: StringLiteral = "cpu",
](
    input: NDBuffer[type, rank],
    k_buf: Scalar,
    axis: Scalar,
    sorted: NDBuffer[DType.bool, 1],
    out_vals: NDBuffer[type, rank],
    out_idxs: NDBuffer[out_idxs_type, rank],
    ctx: MojoCallContextPtr,
) raises:
    constrained[target == "cpu" or "cuda" in target, "not a valid target"]()

    with Trace[TraceLevel.OP, target=target]("top_k"):

        @parameter
        if target == "cpu":
            constrained[
                out_idxs_type == DType.int64,
                "out_idxs_type must be int64 for cpu",
            ]()
            _top_k[rank, type](
                input,
                int(k_buf),
                int(axis),
                True,
                rebind[NDBuffer[type, rank]](out_vals),
                rebind[NDBuffer[DType.int64, rank]](out_idxs),
                sorted[0],
            )

        else:
            var axis = int(normalize_neg_index(axis, rank))
            if axis != rank - 1:
                raise Error("axis other than -1 not supported on GPU")
            if not sorted[0]:
                print(
                    "Warning: Unsorted top-k is not supported on GPU. Falling"
                    " back to sorted top-k."
                )

            var cuda_ctx = ctx.get_device_context()
            _topk_gpu[type, rank, out_idxs_type, sampling=False, largest=True,](
                cuda_ctx,
                int(k_buf),
                input,
                out_vals,
                out_idxs,
            )


# ===-----------------------------------------------------------------------===#
# GatherND
# ===-----------------------------------------------------------------------===#


@register_internal("mo.gather_nd")
@always_inline
fn gather_nd[
    type: DType,
    indices_type: DType,
    data_rank: Int,
    indices_rank: Int,
    output_rank: Int,
    batchDims: Int,
    single_thread_blocking_override: Bool,
    target: StringLiteral = "cpu",
](
    data: NDBuffer[type, data_rank],
    indices: NDBuffer[indices_type, indices_rank],
    output: NDBuffer[type, output_rank],
    ctx: MojoCallContextPtr,
):
    _gather_nd[
        type,
        indices_type,
        data_rank,
        indices_rank,
        output_rank,
        batchDims,
        target=target,
    ](data, indices, output, ctx)


# Note: this is not a "real" index_tensor op that covers all cases, but rather
# a stopgap measure for some important models (DLRM, CLIP-ViT, LLaMa2)
@register_internal("index_tensor")
@always_inline
fn index_tensor[
    type: DType,
    indices_type: DType,
    data_rank: Int,
    indices_rank: Int,
    output_rank: Int,
    batch_dims: Int,
    target: StringLiteral = "cpu",
](
    data: NDBuffer[type, data_rank],
    indices: NDBuffer[indices_type, indices_rank],
    output: NDBuffer[type, output_rank],
    ctx: MojoCallContextPtr,
):
    _index_tensor[
        type,
        indices_type,
        data_rank,
        indices_rank,
        output_rank,
        batch_dims,
        target=target,
    ](data, indices, output, ctx)


# Wrappers that take `num_groups` as a parameter.
# This is required unti `mo.layout.transform` passes `num_groups` as a runtime
# value.
@register_internal("layout_transform_QRSCF_to_FQRSCf")
@register_internal("layout_transform_RSCF_to_FRSCf")
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


@register_internal("layout_transform_RSFC_to_FRSCf")
@register_internal("layout_transform_QRSFC_to_FQRSCf")
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


@register_internal("pack_conv_filter_shape")
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
](filter_buf: NDBuffer[filter_type, rank]) -> IndexList[rank + 1]:
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


@register_internal("pack_conv_transpose_filter_shape")
@always_inline
fn pack_conv_transpose_filter_shape[
    rank: Int,
    filter_type: DType,
    single_thread_blocking_override: Bool,
](filter_buf: NDBuffer[filter_type, rank]) -> IndexList[rank + 1]:
    return _pack_conv_transpose_filter_shape(filter_buf, 1)


# ===-----------------------------------------------------------------------===#
# MOGG distributed.allreduce.sum
# ===-----------------------------------------------------------------------===#


@register_internal("mo.distributed.allreduce.sum")
@always_inline
fn allreduce_sum[
    type: DType,
    rank: Int,
    target: StringLiteral = "cpu",
](
    inputs: StaticTuple[NDBuffer[type, rank, *_], *_],
    outputs: StaticTuple[NDBuffer[type, rank, *_], *_],
    ctx: MojoCallContextPtr,
) raises:
    pass


# ===-----------------------------------------------------------------------===#
# Elementwise Ops
# ===-----------------------------------------------------------------------===#


@register_internal("mo.cos")
fn wrapped_cos[
    type: DType, simd_width: Int
](arg: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return cos(arg)


@register_internal("mo.erf")
@always_inline("nodebug")
fn wrapped_erf[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return erf(x)


@register_internal("mo.exp")
@always_inline("nodebug")
fn wrapped_exp[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return exp(x)


@register_internal("mo.equal")
@always_inline
fn equal[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    DType.bool, simd_width
]:
    return x == y


@register_internal("mo.greater")
@always_inline("nodebug")
fn greater[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    DType.bool, simd_width
]:
    return x > y


@register_internal("mo.greater_equal")
@always_inline
fn greater_equal[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    DType.bool, simd_width
]:
    return x >= y


@register_internal("mo.not_equal")
@always_inline("nodebug")
fn not_equal[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    DType.bool, simd_width
]:
    return x != y


@register_internal("mo.round")
@always_inline("nodebug")
fn wrapped_round[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return round(x)


@register_internal("mo.roundeven")
@always_inline("nodebug")
fn roundeven[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return x.roundeven()


@register_internal("mo.isqrt")
@always_inline("nodebug")
fn wrapped_isqrt[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return isqrt(x)


@register_internal("mo.select")
@always_inline("nodebug")
fn select[
    type: DType, simd_width: Int
](
    cond: SIMD[DType.bool, simd_width],
    true_case: SIMD[type, simd_width],
    false_case: SIMD[type, simd_width],
) -> SIMD[type, simd_width]:
    return cond.select(true_case, false_case)


@register_internal("mo.sin")
fn wrapped_sin[
    type: DType, simd_width: Int
](arg: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return sin(arg)


@register_internal("mo.trunc")
@always_inline("nodebug")
fn wrapped_trunc[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return llvm_intrinsic["llvm.trunc", __type_of(x), has_side_effect=False](x)


@register_internal("mo.log")
@always_inline("nodebug")
fn wrapped_log[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return log(x)


@register_internal("mo.log1p")
@always_inline("nodebug")
fn wrapped_log1p[
    type: DType, simd_width: Int
](arg: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return log1p(arg)


@register_internal("mo.is_nan")
@always_inline("nodebug")
fn wrapped_isnan[
    type: DType, simd_width: Int
](val: SIMD[type, simd_width]) -> SIMD[DType.bool, simd_width]:
    return isnan(val)


@register_internal("mo.is_inf")
@always_inline("nodebug")
fn wrapped_isinf[
    type: DType, simd_width: Int
](val: SIMD[type, simd_width]) -> SIMD[DType.bool, simd_width]:
    return isinf(val)


@register_internal("mo.and")
@always_inline
fn logical_and[
    simd_width: Int
](x: SIMD[DType.bool, simd_width], y: SIMD[DType.bool, simd_width]) -> SIMD[
    DType.bool, simd_width
]:
    return x & y


@register_internal("mo.or")
@always_inline
fn logical_or[
    simd_width: Int
](x: SIMD[DType.bool, simd_width], y: SIMD[DType.bool, simd_width]) -> SIMD[
    DType.bool, simd_width
]:
    return x | y


@register_internal("mo.not")
@always_inline
fn logical_not[
    simd_width: Int
](x: SIMD[DType.bool, simd_width]) -> SIMD[DType.bool, simd_width]:
    return ~x


@register_internal("mo.xor")
@always_inline
fn logical_xor[
    simd_width: Int
](x: SIMD[DType.bool, simd_width], y: SIMD[DType.bool, simd_width]) -> SIMD[
    DType.bool, simd_width
]:
    return x ^ y


# ===-----------------------------------------------------------------------===#
# Custom Ops
# ===-----------------------------------------------------------------------===#


@register_internal("top_p_sampling")
@always_inline
fn top_p_sampling[
    type: DType,
    rank: Int,
    out_idx_type: DType,
    target: StringLiteral = "cpu",
](
    top_ps: NDBuffer[type, 1],
    input: NDBuffer[type, rank],
    out_idxs: NDBuffer[out_idx_type, rank],
    temperature: Scalar[type],  # should be default or no?
    ctx: MojoCallContextPtr,
) raises:
    constrained[target == "cpu" or "cuda" in target, "not a valid target"]()

    with Trace[TraceLevel.OP, target=target]("top_p_sampling"):

        @parameter
        if target == "cpu":
            _top_p_sampling(top_ps, input, out_idxs, temperature)
        else:
            var cuda_ctx = ctx.get_device_context()
            _top_p_sampling_gpu(
                cuda_ctx,
                top_ps,
                input,
                out_idxs,
                temperature,
            )


@register_internal("min_p_sampling")
@always_inline
fn min_p_sampling[
    type: DType,
    rank: Int,
    out_idx_type: DType,
    target: StringLiteral = "cpu",
](
    min_ps: NDBuffer[type, 1],
    input: NDBuffer[type, rank],
    out_idxs: NDBuffer[out_idx_type, rank],
    temperature: Scalar[type],  # should be default or no?
    ctx: MojoCallContextPtr,
) raises:
    constrained[target == "cpu" or "cuda" in target, "not a valid target"]()

    with Trace[TraceLevel.OP, target=target]("min_p_sampling"):

        @parameter
        if target == "cpu":
            _min_p_sampling(min_ps, input, out_idxs, temperature)
        else:
            var cuda_ctx = ctx.get_device_context()
            _min_p_sampling_gpu(
                cuda_ctx,
                min_ps,
                input,
                out_idxs,
                temperature,
            )


@register_internal("topk_fused_sampling")
@always_inline
fn topk_fused_sampling[
    type: DType,
    rank: Int,
    out_idx_type: DType,
    target: StringLiteral = "cpu",
](
    K: Scalar,
    input: NDBuffer[type, rank],
    out_idxs: NDBuffer[out_idx_type, rank],
    ctx: MojoCallContextPtr,
) raises:
    constrained[target == "cpu" or "cuda" in target, "not a valid target"]()

    with Trace[TraceLevel.OP, target=target]("topk_fused_sampling"):

        @parameter
        if target == "cpu":
            _topk_fused_sampling(int(K), input, out_idxs)
        else:
            var cuda_ctx = ctx.get_device_context()
            _topk_fused_sampling_gpu(
                cuda_ctx,
                int(K),
                input,
                out_idxs,
            )


@register_internal("reduce_min_and_max")
@always_inline
fn reduce_min_and_max[
    type: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
    input_0_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        type, width
    ],
    output_0_fn: fn[width: Int, rank: Int, element_alignment: Int] (
        IndexList[rank], SIMD[type, width]
    ) capturing -> None,
    target: StringLiteral = "cpu",
](
    input_shape: IndexList[rank],
    axis0: Scalar,
    output_shape: IndexList[rank],
    ctx: MojoCallContextPtr,
) raises:
    """Given a tensor of shape [A, B, C, D] and reducing along dimension 'C'
    writes to a tensor of shape [A, B, 2, D] where [:, :, 0, :] contains
    the minimum reduction and [:, :, 1, :] contains the maximum reduction.
    """

    alias num_reductions = 2
    var axis = int(normalize_neg_index(axis0, rank))

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
        output_0_fn[width, rank, element_alignment=1](
            indices_min, rebind[SIMD[type, width]](val[0])
        )

        var indices_max = indices
        indices_max[axis] = 1
        output_0_fn[width, rank, element_alignment=1](
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
            single_thread_blocking_override=single_thread_blocking_override,
            target=target,
        ](input_shape, init=init, reduce_dim=axis, context=ctx)
    _ = axis


@register_internal_shape_func("reduce_min_and_max")
@always_inline
fn reduce_min_and_max_shape_func[
    type: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
](
    data: NDBuffer[type, rank, DimList.create_unknown[rank]()],
    axis0: Scalar,
) -> IndexList[rank]:
    var new_shape = data.get_shape()
    var axis = int(normalize_neg_index(axis0, rank))
    new_shape[axis] = 2
    return new_shape


# MHA Kernels:
@register_internal("masked_flash_attention_gpu")
@always_inline
fn masked_flash_attention_gpu[
    rank: Int,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[_, rank, *_],
    k: NDBuffer[_, rank, *_],
    v: NDBuffer[_, rank, *_],
    mask: NDBuffer,
    scale: Scalar[DType.float32],
    output: NDBuffer[_, rank, *_],
    ctx: MojoCallContextPtr,
) raises:
    """`masked_flash_attention_gpu` is a hand-fused operator which does something
    analogous to the following list of operations.

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

    constrained["cuda" in target, "only valid on CUDA GPUs"]()

    flash_attention[add_attn_mask=True](
        output, q, k, v, mask, scale[0], context=ctx
    )


@register_internal("no_mask_fused_attention_cpu")
@always_inline
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

    constrained[target == "cpu"]()

    # TODO: Unimplemented and not used
    alias mask_shape = DimList()
    alias mask_type = DType.float32
    var mask = NDBuffer[mask_type, rank, mask_shape]()
    var scale_f32 = scale[0].cast[DType.float32]()
    var causal_mask: Float32 = 0
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


@register_internal("with_mask_fused_attention_cpu")
@always_inline
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

    constrained[target == "cpu"]()

    # TODO: Unimplemented and not used
    var scale_f32 = scale[0].cast[DType.float32]()
    var causal_mask: Float32 = 0
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


@register_internal("no_mask_flash_attention_cpu")
@always_inline
fn no_mask_flash_attention_cpu[
    type: DType,
    rank: Int,
    input_1_fn: fn[simd_width: Int, rank: Int] (
        IndexList[rank]
    ) capturing -> SIMD[type, simd_width],
    input_2_fn: fn[simd_width: Int, rank: Int] (
        IndexList[rank]
    ) capturing -> SIMD[type, simd_width],
    input_4_static_shape: DimList,
    single_thread_blocking_override: Bool,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[type, rank],
    input_1_shape: IndexList[rank],
    input_2_shape: IndexList[rank],
    scale: Scalar[type],
    output: NDBuffer[type, rank, input_4_static_shape],
    ctx: MojoCallContextPtr,
) raises:
    constrained[target == "cpu"]()

    @parameter
    @always_inline
    fn mask_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[type, simd_width]:
        return SIMD[type, simd_width](0)

    nn_flash_attention[input_1_fn, input_2_fn, mask_fn](
        q,
        input_1_shape,
        input_2_shape,
        IndexList[0](),
        output,
        scale[0].cast[DType.float32](),
    )


@register_internal("with_mask_flash_attention_split_kv_cpu")
@always_inline
fn with_mask_flash_attention_split_kv_cache_cpu[
    type: DType,
    rank: Int,
    mask_rank: Int,
    input_1_fn: fn[simd_width: Int, rank: Int] (
        IndexList[rank]
    ) capturing -> SIMD[type, simd_width],
    input_2_fn: fn[simd_width: Int, rank: Int] (
        IndexList[rank]
    ) capturing -> SIMD[type, simd_width],
    input_3_fn: fn[simd_width: Int, rank: Int] (
        IndexList[rank]
    ) capturing -> SIMD[type, simd_width],
    input_4_fn: fn[simd_width: Int, rank: Int] (
        IndexList[rank]
    ) capturing -> SIMD[type, simd_width],
    input_5_fn: fn[simd_width: Int, rank: Int] (
        IndexList[rank]
    ) capturing -> SIMD[type, simd_width],
    input_7_static_shape: DimList,
    single_thread_blocking_override: Bool,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[type, rank, *_],
    input_1_shape: IndexList[rank],
    input_2_shape: IndexList[rank],
    input_3_shape: IndexList[rank + 1],
    input_4_shape: IndexList[rank + 1],
    input_5_shape: IndexList[mask_rank],
    scale: Scalar[type],
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
        q: BSHD
        input_1_fn (k): BSHD
        input_2_fn (v): BSHD
        input_3_fn (k_cache): 1BHSD
        input_4_fn (v_cache): 1BHSD
    """

    constrained[target == "cpu"]()

    flash_attention_split_kv[
        input_1_fn,
        input_2_fn,
        input_3_fn,
        input_4_fn,
        input_5_fn,
    ](
        q,
        input_1_shape,
        input_2_shape,
        input_3_shape,
        input_4_shape,
        input_5_shape,
        output,
        scale[0].cast[DType.float32](),
    )


@register_internal_shape_func("with_mask_flash_attention_split_kv_cpu")
@always_inline
fn with_mask_flash_attention_split_kv_cpu_shape_func[
    type: DType, rank: Int, single_thread_blocking_override: Bool
](q: NDBuffer[type, rank]) -> IndexList[rank]:
    return q.get_shape()


@register_internal("with_mask_flash_attention_cpu")
@always_inline
fn with_mask_flash_attention_cpu[
    type: DType,
    rank: Int,
    mask_rank: Int,
    input_1_fn: fn[simd_width: Int, rank: Int] (
        IndexList[rank]
    ) capturing -> SIMD[type, simd_width],
    input_2_fn: fn[simd_width: Int, rank: Int] (
        IndexList[rank]
    ) capturing -> SIMD[type, simd_width],
    input_3_fn: fn[simd_width: Int, rank: Int] (
        IndexList[rank]
    ) capturing -> SIMD[type, simd_width],
    input_5_static_shape: DimList,
    single_thread_blocking_override: Bool,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[type, rank],
    input_1_shape: IndexList[rank],
    input_2_shape: IndexList[rank],
    input_3_shape: IndexList[mask_rank],
    scale: Scalar[type],
    output: NDBuffer[type, rank, input_5_static_shape],
    ctx: MojoCallContextPtr,
) raises:
    constrained[target == "cpu"]()

    nn_flash_attention[input_1_fn, input_2_fn, input_3_fn](
        q,
        input_1_shape,
        input_2_shape,
        input_3_shape,
        output,
        scale[0].cast[DType.float32](),
    )


@register_internal("mo.linalg.solve")
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


@register_internal("pytorch_operator_custom_test")
fn pytorch_test_custom[
    type: DType,
    rank: Int,
](data: NDBuffer[type, rank], out: NDBuffer[type, rank]):
    print("hello")


######
# Q4_0
######


@register_internal("vroom_q4_0_matmul")
@always_inline
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
    constrained[b.type is DType.uint8, "expected uint8 input b"]()

    with Trace[TraceLevel.OP, target="cpu"]("vroom_q4_0_matmul"):
        matmul_qint4[32](a, b, c)


@register_internal_shape_func("vroom_q4_0_matmul")
@always_inline
fn vroom_q4_0_matmul_shape_func[
    single_thread_blocking_override: Bool
](a: NDBuffer[DType.float32, 2], b: NDBuffer[DType.uint8, 2]) -> IndexList[2]:
    constrained[
        a.type.is_floating_point(), "expected float inputs and outputs"
    ]()
    constrained[b.type is DType.uint8, "expected uint8 input b"]()
    constrained[a.rank == b.rank == 2, "expected rank to be 2"]()

    return IndexList[2](a.dim[0](), b.dim[0]())


@register_internal("vroom_q4_0_repack_weights")
@always_inline
fn vroom_q4_0_repack_weights(
    b: NDBuffer[DType.uint8, 2],
    b_packed: NDBuffer[DType.uint8, 2],
    ctx: MojoCallContextPtr,
) raises:
    matmul_qint4_pack_b[32](b, b_packed)


@register_internal_shape_func("vroom_q4_0_repack_weights")
@always_inline
fn vroom_q4_0_repack_weights_shape_func[
    single_thread_blocking_override: Bool
](b: NDBuffer[DType.uint8, 2]) -> IndexList[2]:
    return b.get_shape()


@register_internal("ggml_q4_0_dequantize")
@always_inline
fn ggml_q4_0_dequantize(
    input: NDBuffer[DType.uint8, 2],
    output: NDBuffer[DType.float32, 2],
    ctx: MojoCallContextPtr,
) raises:
    with Trace[TraceLevel.OP, target="cpu"]("ggml_q4_0_dequantize"):
        Q4sym[group_size=32].dequantize_and_write_to_tensor(
            input, output, output.get_shape()
        )


@register_internal_shape_func("ggml_q4_0_dequantize")
@always_inline
fn ggml_q4_0_dequantize_shape_func[
    single_thread_blocking_override: Bool
](input: NDBuffer[DType.uint8, 2]) -> IndexList[2]:
    constrained[input.type is DType.uint8, "expected uint8 input"]()

    alias block_nbytes = sizeof[Q4sym[group_size=32]]()
    alias quants_per_block = 32

    var num_block_per_batch = (input.size() // input.dim[0]()) // block_nbytes

    return (input.dim[0](), quants_per_block * num_block_per_batch)


######
# Q4_K
######


@register_internal("vroom_q4_k_matmul")
@always_inline
fn vroom_q4_k_matmul(
    a: NDBuffer[DType.float32, 2],
    b: NDBuffer[DType.uint8, 2],
    c: NDBuffer[DType.float32, 2],
    ctx: MojoCallContextPtr,
) raises:
    with Trace[TraceLevel.OP, target="cpu"]("vroom_q4_k_matmul"):
        matmul_Q4_K(a, b, c)


@register_internal_shape_func("vroom_q4_k_matmul")
@always_inline
fn vroom_q4_k_matmul_shape_func[
    single_thread_blocking_override: Bool
](a: NDBuffer[DType.float32, 2], b: NDBuffer[DType.uint8, 2]) -> IndexList[2]:
    return IndexList[2](a.dim[0](), b.dim[0]())


@register_internal("vroom_q4_k_repack_weights")
@always_inline
fn vroom_q4_k_repack_weights(
    b: NDBuffer[DType.uint8, 2],
    b_packed: NDBuffer[DType.uint8, 2],
    ctx: MojoCallContextPtr,
) raises:
    matmul_Q4_K_pack_b(b, b_packed)


@register_internal_shape_func("vroom_q4_k_repack_weights")
@always_inline
fn vroom_q4_k_repack_weights_shape_func[
    single_thread_blocking_override: Bool
](b: NDBuffer[DType.uint8, 2]) -> IndexList[2]:
    return b.get_shape()


@register_internal("ggml_q4_k_dequantize")
@always_inline
fn ggml_q4_k_dequantize(
    input: NDBuffer[DType.uint8, 2],
    output: NDBuffer[DType.float32, 2],
    ctx: MojoCallContextPtr,
) raises:
    with Trace[TraceLevel.OP, target="cpu"]("ggml_q4_k_dequantize"):
        q4_k_dequantize_impl(input, output)


@register_internal_shape_func("ggml_q4_k_dequantize")
@always_inline
fn ggml_q4_k_dequantize_shape_func[
    single_thread_blocking_override: Bool
](input: NDBuffer[DType.uint8, 2]) -> IndexList[2]:
    alias block_nbytes = sizeof[block_Q4_K]()
    alias elements_per_block = block_QK_K.quantized_k

    var num_block_per_batch = (
        input.size() // input.dynamic_shape[0]
    ) // block_nbytes

    return (input.dynamic_shape[0], elements_per_block * num_block_per_batch)


######
# Q6_K
######


@register_internal("vroom_q6_k_matmul")
@always_inline
fn vroom_q6_k_matmul(
    a: NDBuffer[DType.float32, 2],
    b: NDBuffer[DType.uint8, 2],
    c: NDBuffer[DType.float32, 2],
    ctx: MojoCallContextPtr,
) raises:
    with Trace[TraceLevel.OP, target="cpu"]("vroom_q6_k_matmul"):
        matmul_Q6_K(a, b, c)


@register_internal_shape_func("vroom_q6_k_matmul")
@always_inline
fn vroom_q6_k_matmul_shape_func[
    single_thread_blocking_override: Bool
](a: NDBuffer[DType.float32, 2], b: NDBuffer[DType.uint8, 2]) -> IndexList[2]:
    return IndexList[2](a.dim[0](), b.dim[0]())


@register_internal("vroom_q6_k_repack_weights")
@always_inline
fn vroom_q6_k_repack_weights(
    b: NDBuffer[DType.uint8, 2],
    b_packed: NDBuffer[DType.uint8, 2],
    ctx: MojoCallContextPtr,
) raises:
    matmul_Q6_K_pack_b(b, b_packed)


@register_internal_shape_func("vroom_q6_k_repack_weights")
@always_inline
fn vroom_q6_k_repack_weights_shape_func[
    single_thread_blocking_override: Bool
](b: NDBuffer[DType.uint8, 2]) -> IndexList[2]:
    return b.get_shape()


@register_internal("ggml_q6_k_dequantize")
@always_inline
fn ggml_q6_k_dequantize(
    input: NDBuffer[DType.uint8, 2],
    output: NDBuffer[DType.float32, 2],
    ctx: MojoCallContextPtr,
) raises:
    with Trace[TraceLevel.OP, target="cpu"]("ggml_q6_k_dequantize"):
        q6_k_dequantize_impl(input, output, output.get_shape())


@register_internal_shape_func("ggml_q6_k_dequantize")
@always_inline
fn ggml_q6_k_dequantize_shape_func[
    single_thread_blocking_override: Bool
](input: NDBuffer[DType.uint8, 2]) -> IndexList[2]:
    alias block_nbytes = sizeof[block_Q6_K]()
    alias elements_per_block = block_QK_K.quantized_k

    var num_block_per_batch = (
        input.size() // input.dynamic_shape[0]
    ) // block_nbytes

    return (input.dynamic_shape[0], elements_per_block * num_block_per_batch)


######
# 4-bit quant GPU implementation
######


@register_internal("qmatmul_b4_g32")
@always_inline
fn qmatmul_b4_g32[
    input_0_static_shape: DimList,
    input_1_static_shape: DimList,
    output_0_static_shape: DimList,
    target: StringLiteral = "cpu",
](
    a: NDBuffer[DType.bfloat16, 2, input_0_static_shape],
    b: NDBuffer[DType.uint8, 2, input_1_static_shape],
    c: NDBuffer[DType.bfloat16, 2, output_0_static_shape],
    ctx: MojoCallContextPtr,
) raises:
    constrained["cuda" in target, "only valid on CUDA GPUs"]()

    with Trace[TraceLevel.OP, target=target]("qmatmul_b4_g32"):
        matmul_gpu_qint4[32, target](c, a, b, ctx)


@register_internal_shape_func("qmatmul_b4_g32")
@always_inline
fn qmatmul_b4_g32_shape_func[
    single_thread_blocking_override: Bool
](a: NDBuffer[DType.bfloat16, 2], b: NDBuffer[DType.uint8, 2]) -> IndexList[2]:
    constrained[
        a.type.is_floating_point(), "expected float inputs and outputs"
    ]()
    constrained[b.type is DType.uint8, "expected uint8 input b"]()
    constrained[a.rank == b.rank == 2, "expected rank to be 2"]()

    return IndexList[2](a.dim[0](), b.dim[0]())


@register_internal("qmatmul_b4_g128")
@always_inline
fn qmatmul_b4_g128[
    input_0_static_shape: DimList,
    input_1_static_shape: DimList,
    output_0_static_shape: DimList,
    target: StringLiteral = "cpu",
](
    a: NDBuffer[DType.bfloat16, 2, input_0_static_shape],
    b: NDBuffer[DType.uint8, 2, input_1_static_shape],
    c: NDBuffer[DType.bfloat16, 2, output_0_static_shape],
    ctx: MojoCallContextPtr,
) raises:
    constrained["cuda" in target, "only valid on CUDA GPUs"]()

    with Trace[TraceLevel.OP, target=target]("qmatmul_b4_g128"):
        matmul_gpu_qint4[128, target](c, a, b, ctx)


@register_internal_shape_func("qmatmul_b4_g128")
@always_inline
fn qmatmul_b4_g128_shape_func[
    single_thread_blocking_override: Bool
](a: NDBuffer[DType.bfloat16, 2], b: NDBuffer[DType.uint8, 2]) -> IndexList[2]:
    constrained[
        a.type.is_floating_point(), "expected float inputs and outputs"
    ]()
    constrained[b.type is DType.uint8, "expected uint8 input b"]()
    constrained[a.rank == b.rank == 2, "expected rank to be 2"]()

    return IndexList[2](a.dim[0](), b.dim[0]())


@register_internal_override("GGUF_gpu_repack_q4_0", 1)
@always_inline
fn GGUF_gpu_repack_q4_0[
    input_0_static_shape: DimList,
    target: StringLiteral = "cpu",
](
    b: NDBuffer[DType.uint8, 2, input_0_static_shape],
    b_packed: NDBuffer[DType.uint8, 2, input_0_static_shape],
    ctx: MojoCallContextPtr,
) raises:
    constrained["cuda" in target, "only valid on CUDA GPUs"]()

    with Trace[TraceLevel.OP, target=target]("GGUF_gpu_repack_q4_0"):
        gpu_qint4_repack_Q4_0[target](b, b_packed, ctx)


@register_internal_shape_func("GGUF_gpu_repack_q4_0")
@always_inline
fn GGUF_gpu_repack_q4_0_shape_func[
    single_thread_blocking_override: Bool
](b: NDBuffer[DType.uint8, 2]) -> IndexList[2]:
    return b.get_shape()


@register_internal_override("GPTQ_gpu_repack_b4_g128", 1)
@always_inline
fn GPTQ_gpu_repack_b4_g128[
    input_0_static_shape: DimList,
    input_1_static_shape: DimList,
    target: StringLiteral = "cpu",
](
    b: NDBuffer[DType.uint8, 2, input_0_static_shape],
    b_packed: NDBuffer[DType.uint8, 2, input_1_static_shape],
    ctx: MojoCallContextPtr,
) raises:
    constrained["cuda" in target, "only valid on CUDA GPUs"]()

    with Trace[TraceLevel.OP, target=target]("GPTQ_gpu_repack_b4_g128"):
        gpu_qint4_repack_GPTQ[128, target](b, b_packed, ctx)


@register_internal_shape_func("GPTQ_gpu_repack_b4_g128")
@always_inline
fn GPTQ_gpu_repack_b4_g128_shape_func[
    single_thread_blocking_override: Bool
](b: NDBuffer[DType.uint8, 2]) -> IndexList[2]:
    return IndexList[2](b.dim[1](), b.dim[0]())


# ===-----------------------------------------------------------------------===#
# Basic elementwise primitives
# ===-----------------------------------------------------------------------===#


@register_internal("mo.mod")
@always_inline
fn mod[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    type, simd_width
]:
    """Performs elementwise modulo operation of two SIMD vectors.

    Parameters:
        type: DType of the input SIMD vectors.
        simd_width: Width of the input SIMD vectors.

    Args:
        x: The numerator of the operation.
        y: The denominator of the operation.

    Returns:
        Elementwise remainder of x divided by y.
    """
    return x % y


@register_internal("mo.mul")
@always_inline
fn mul[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    type, simd_width
]:
    """Performs elementwise multiplication of two SIMD vectors.

    Parameters:
        type: DType of the input SIMD vectors.
        simd_width: Width of the input SIMD vectors.

    Args:
        x: First SIMD vector to multiply.
        y: Second SIMD vector to multiply.

    Returns:
        Elementwise multiplication of x and y.
    """
    return x * y


@register_internal("mo.sub")
@always_inline
fn sub[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    type, simd_width
]:
    """Performs elementwise subtraction of two SIMD vectors.

    Parameters:
        type: DType of the input SIMD vectors.
        simd_width: Width of the input SIMD vectors.

    Args:
        x: SIMD vector which y will be subtracted from.
        y: SIMD vector to subtract from x.

    Returns:
        Elementwise subtraction of x and y.
    """
    return x - y


@register_internal("mo.add")
@always_inline
fn add[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    type, simd_width
]:
    """Performs elementwise addition of two SIMD vectors.

    Parameters:
        type: DType of the input SIMD vectors.
        simd_width: Width of the input SIMD vectors.

    Args:
        x: First SIMD vector to add.
        y: Second SIMD vector to add.

    Returns:
        Elementwise addition of x and y.
    """
    return x + y


@register_internal("mo.div")
@always_inline
fn div[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    type, simd_width
]:
    """Performs elementwise division of two SIMD vectors.

    Parameters:
        type: DType of the input SIMD vectors.
        simd_width: Width of the input SIMD vectors.

    Args:
        x: SIMD vector containing the dividends.
        y: SIMD vector containing the quotients.

    Returns:
        Elementwise division of SIMD vector x by SIMD vector y (this is x / y).
    """
    return x / y


# ===----------------------------------------------------------------------=== #
# ceil
# ===----------------------------------------------------------------------=== #


@register_internal("mo.ceil")
@always_inline
fn ceil[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Compute the Ceil Op.

    Parameters:
        type: DType used for the computation.
        simd_width: SIMD width used for the computation.

    Args:
        x : The value to compute the Ceil operation on.

    Returns:
        The result of the Ceil operation.
    """
    return _ceil(x)


# ===----------------------------------------------------------------------=== #
# floor
# ===----------------------------------------------------------------------=== #


@register_internal("mo.floor")
@always_inline
fn floor[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Compute the Floor Op.

    Parameters:
        type: DType used for the computation.
        simd_width: SIMD width used for the computation.

    Args:
        x : The value to compute the Floor operation on.

    Returns:
        The result of the Floor operation.
    """
    return _floor(x)


# ===----------------------------------------------------------------------=== #
# tanh
# ===----------------------------------------------------------------------=== #


@register_internal("mo.tanh")
@always_inline
fn tanh[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Compute the Tanh Op.

    Parameters:
        type: DType used for the computation.
        simd_width: SIMD width used for the computation.

    Args:
        x : The value to compute the Tanh operation on.

    Returns:
        The result of the Tanh operation.
    """
    return _tanh(x)


# useful for testing --> identity op that simply copies input into output
@register_internal("copy")
@always_inline
fn identity[
    rank: Int,
    input_type: DType,
](
    input: NDBuffer[input_type, rank],
    output: NDBuffer[input_type, rank],
    ctx: MojoCallContextPtr,
) raises:
    memcpy(output.data, input.data, len(input))


@register_internal_shape_func("mo.avg_pool")
@always_inline
fn avg_pool_shape[
    input_rank: Int,
    input_type: DType,
    filter_type: DType,
    strides_type: DType,
    dilations_type: DType,
    paddings_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[input_type, input_rank],
    filter_buf: NDBuffer[filter_type, 1],
    strides_buf: NDBuffer[strides_type, 1],
    dilations_buf: NDBuffer[dilations_type, 1],
    paddings_buf: NDBuffer[paddings_type, 1],
) raises -> IndexList[input_rank]:
    return pool_shape[
        input_rank,
        input_type,
        filter_type,
        strides_type,
        dilations_type,
        paddings_type,
        single_thread_blocking_override,
    ](input_buf, filter_buf, strides_buf, dilations_buf, paddings_buf)


@register_internal_shape_func("mo.avg_pool_ceil_mode_true")
@always_inline
fn avg_pool_ceil_mode_true_shape[
    input_rank: Int,
    input_type: DType,
    filter_type: DType,
    strides_type: DType,
    dilations_type: DType,
    paddings_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[input_type, input_rank],
    filter_buf: NDBuffer[filter_type, 1],
    strides_buf: NDBuffer[strides_type, 1],
    dilations_buf: NDBuffer[dilations_type, 1],
    paddings_buf: NDBuffer[paddings_type, 1],
) raises -> IndexList[input_rank]:
    return pool_shape_ceil[
        input_rank,
        input_type,
        filter_type,
        strides_type,
        dilations_type,
        paddings_type,
        single_thread_blocking_override,
    ](input_buf, filter_buf, strides_buf, dilations_buf, paddings_buf)


@register_internal_shape_func("mo.max_pool")
@always_inline
fn max_pool_shape[
    input_rank: Int,
    input_type: DType,
    filter_type: DType,
    strides_type: DType,
    dilations_type: DType,
    paddings_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[input_type, input_rank],
    filter_buf: NDBuffer[filter_type, 1],
    strides_buf: NDBuffer[strides_type, 1],
    dilations_buf: NDBuffer[dilations_type, 1],
    paddings_buf: NDBuffer[paddings_type, 1],
) raises -> IndexList[input_rank]:
    return pool_shape[
        input_rank,
        input_type,
        filter_type,
        strides_type,
        dilations_type,
        paddings_type,
        single_thread_blocking_override,
    ](input_buf, filter_buf, strides_buf, dilations_buf, paddings_buf)


@register_internal_shape_func("mo.max_pool_ceil_mode_true")
@always_inline
fn max_pool_ceil_mode_true_shape[
    input_rank: Int,
    input_type: DType,
    filter_type: DType,
    strides_type: DType,
    dilations_type: DType,
    paddings_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[input_type, input_rank],
    filter_buf: NDBuffer[filter_type, 1],
    strides_buf: NDBuffer[strides_type, 1],
    dilations_buf: NDBuffer[dilations_type, 1],
    paddings_buf: NDBuffer[paddings_type, 1],
) raises -> IndexList[input_rank]:
    return pool_shape_ceil[
        input_rank,
        input_type,
        filter_type,
        strides_type,
        dilations_type,
        paddings_type,
        single_thread_blocking_override,
    ](input_buf, filter_buf, strides_buf, dilations_buf, paddings_buf)


@register_internal_shape_func("mo.pad.constant")
@always_inline
fn pad_constant_shape[
    input_rank: Int,
    input_type: DType,
    paddings_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[input_type, input_rank],
    paddings_buf: NDBuffer[paddings_type, 1],
) raises -> IndexList[input_rank]:
    return pad_shape[
        single_thread_blocking_override=single_thread_blocking_override
    ](input_buf, paddings_buf)


@register_internal_shape_func("mo.pad.repeat")
@always_inline
fn pad_repeat_shape[
    input_rank: Int,
    input_type: DType,
    paddings_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[input_type, input_rank],
    paddings_buf: NDBuffer[paddings_type, 1],
) raises -> IndexList[input_rank]:
    return pad_shape[
        single_thread_blocking_override=single_thread_blocking_override
    ](input_buf, paddings_buf)


@register_internal_shape_func("mo.pad.reflect")
@always_inline
fn pad_reflect_shape[
    input_rank: Int,
    input_type: DType,
    paddings_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[input_type, input_rank],
    paddings_buf: NDBuffer[paddings_type, 1],
) raises -> IndexList[input_rank]:
    return pad_shape[
        single_thread_blocking_override=single_thread_blocking_override
    ](input_buf, paddings_buf)
