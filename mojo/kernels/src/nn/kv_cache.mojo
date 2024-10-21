# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from collections import InlineArray, Optional, OptionalReg
from math import gcd, isqrt
from os import abort
from sys.info import _current_target, simdwidthof
from sys.intrinsics import _type_is_eq

from algorithm.functional import elementwise
from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceBuffer, DeviceContext, Stream
from gpu.host._compile import _get_nvptx_target
from kv_cache.types import (
    ContiguousKVCache,
    ContiguousKVCacheCollection,
    ContinuousBatchingKVCache,
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
    KVCacheT,
    KVCollectionT,
)
from linalg import transpose
from linalg.matmul import _matmul_cpu, elementwise_epilogue_type
from linalg.matmul_gpu import _matmul_gpu
from memory import UnsafePointer, memcpy
from nn.flash_attention import (
    flash_attention_kv_cache as flash_attention_kv_cache_cpu,
)
from nn.fused_qk_rope import fused_qk_rope
from nn.mha import flash_attention as gpu_flash_attention
from nn.mha_mask import NullMask, CausalMask
from register import mogg_register
from runtime.asyncrt import MojoCallContextPtr
from runtime.tracing import Trace, TraceLevel

from utils import Index, IndexList
from utils.numerics import isnan, min_finite


@mogg_register("kv_cache_length_h8_d128_bshd_bf16")
@export
fn kv_cache_length_h8_d128_bshd_bf16[
    target: StringLiteral = "cpu"
](
    kv_collection: ContiguousKVCacheCollection[
        DType.bfloat16,
        KVCacheStaticParams(num_heads=8, head_size=128),
    ],
    output: NDBuffer[DType.uint32, 1],
    ctx: MojoCallContextPtr,
) raises:
    return _kv_cache_length[target=target](kv_collection, output, ctx)


@mogg_register("kv_cache_length_h6_d48_bshd_f32")
@export
fn kv_cache_length_h6_d48_bshd_f32[
    target: StringLiteral = "cpu"
](
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(num_heads=6, head_size=48),
    ],
    output: NDBuffer[DType.uint32, 1],
    ctx: MojoCallContextPtr,
) raises:
    return _kv_cache_length[target=target](kv_collection, output, ctx)


@mogg_register("kv_cache_length_h8_d128_bshd_f32")
@export
fn kv_cache_length_h8_d128_bshd_f32[
    target: StringLiteral = "cpu"
](
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(num_heads=8, head_size=128),
    ],
    output: NDBuffer[DType.uint32, 1],
    ctx: MojoCallContextPtr,
) raises:
    return _kv_cache_length[target=target](kv_collection, output, ctx)


@mogg_register("kv_cache_length_h1_d16_bshd_f32")
@export
fn kv_cache_length_h1_d16_bshd_f32[
    target: StringLiteral = "cpu"
](
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(num_heads=1, head_size=16),
    ],
    output: NDBuffer[DType.uint32, 1],
    ctx: MojoCallContextPtr,
) raises:
    return _kv_cache_length[target=target](kv_collection, output, ctx)


@mogg_register("kv_cache_length_h1_d16_bshd_bf16")
@export
fn kv_cache_length_h1_d16_bshd_bf16[
    target: StringLiteral = "cpu"
](
    kv_collection: ContiguousKVCacheCollection[
        DType.bfloat16,
        KVCacheStaticParams(num_heads=1, head_size=16),
    ],
    output: NDBuffer[DType.uint32, 1],
    ctx: MojoCallContextPtr,
) raises:
    return _kv_cache_length[target=target](kv_collection, output, ctx)


@mogg_register("kv_cache_length_h8_d32_bshd_bf16")
@export
fn kv_cache_length_h8_d32_bshd_bf16[
    target: StringLiteral = "cpu"
](
    kv_collection: ContiguousKVCacheCollection[
        DType.bfloat16,
        KVCacheStaticParams(num_heads=8, head_size=32),
    ],
    output: NDBuffer[DType.uint32, 1],
    ctx: MojoCallContextPtr,
) raises:
    return _kv_cache_length[target=target](kv_collection, output, ctx)


@mogg_register("kv_cache_length_h8_d32_bshd_f32")
@export
fn kv_cache_length_h8_d32_bshd_f32[
    target: StringLiteral = "cpu"
](
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(num_heads=8, head_size=32),
    ],
    output: NDBuffer[DType.uint32, 1],
    ctx: MojoCallContextPtr,
) raises:
    return _kv_cache_length[target=target](kv_collection, output, ctx)


@mogg_register("kv_cache_length_h8_d64_bshd_bf16")
@export
fn kv_cache_length_h8_d64_bshd_bf16[
    target: StringLiteral = "cpu"
](
    kv_collection: ContiguousKVCacheCollection[
        DType.bfloat16,
        KVCacheStaticParams(num_heads=8, head_size=64),
    ],
    output: NDBuffer[DType.uint32, 1],
    ctx: MojoCallContextPtr,
) raises:
    return _kv_cache_length[target=target](kv_collection, output, ctx)


@mogg_register("kv_cache_length_h8_d64_bshd_f32")
@export
fn kv_cache_length_h8_d64_bshd_f32[
    target: StringLiteral = "cpu"
](
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(num_heads=8, head_size=64),
    ],
    output: NDBuffer[DType.uint32, 1],
    ctx: MojoCallContextPtr,
) raises:
    return _kv_cache_length[target=target](kv_collection, output, ctx)


@mogg_register("kv_cache_length_h8_d128_bshd_bf16_continuous_batch")
@export
fn kv_cache_length_h8_d128_bshd_bf16_continuous_batch[
    target: StringLiteral = "cpu"
](
    kv_collection: ContinuousBatchingKVCacheCollection[
        DType.bfloat16,
        KVCacheStaticParams(num_heads=8, head_size=128),
    ],
    output: NDBuffer[DType.uint32, 1],
    ctx: MojoCallContextPtr,
) raises:
    return _kv_cache_length[target=target](kv_collection, output, ctx)


@mogg_register("kv_cache_length_h8_d128_bshd_f32_continuous_batch")
@export
fn kv_cache_length_h8_d128_bshd_f32_continuous_batch[
    target: StringLiteral = "cpu"
](
    kv_collection: ContinuousBatchingKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(num_heads=8, head_size=128),
    ],
    output: NDBuffer[DType.uint32, 1],
    ctx: MojoCallContextPtr,
) raises:
    return _kv_cache_length[target=target](kv_collection, output, ctx)


@mogg_register("kv_cache_length_h1_d16_bshd_bf16_continuous_batch")
@export
fn kv_cache_length_h1_d16_bshd_bf16_continuous_batch[
    target: StringLiteral = "cpu"
](
    kv_collection: ContinuousBatchingKVCacheCollection[
        DType.bfloat16,
        KVCacheStaticParams(num_heads=1, head_size=16),
    ],
    output: NDBuffer[DType.uint32, 1],
    ctx: MojoCallContextPtr,
) raises:
    return _kv_cache_length[target=target](kv_collection, output, ctx)


@mogg_register("kv_cache_length_h1_d16_bshd_f32_continuous_batch")
@export
fn kv_cache_length_h1_d16_bshd_f32_continuous_batch[
    target: StringLiteral = "cpu"
](
    kv_collection: ContinuousBatchingKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(num_heads=1, head_size=16),
    ],
    output: NDBuffer[DType.uint32, 1],
    ctx: MojoCallContextPtr,
) raises:
    return _kv_cache_length[target=target](kv_collection, output, ctx)


@mogg_register("kv_cache_length_h8_d32_bshd_bf16_continuous_batch")
@export
fn kv_cache_length_h8_d32_bshd_bf16_continuous_batch[
    target: StringLiteral = "cpu"
](
    kv_collection: ContinuousBatchingKVCacheCollection[
        DType.bfloat16,
        KVCacheStaticParams(num_heads=8, head_size=32),
    ],
    output: NDBuffer[DType.uint32, 1],
    ctx: MojoCallContextPtr,
) raises:
    return _kv_cache_length[target=target](kv_collection, output, ctx)


@mogg_register("kv_cache_length_h8_d32_bshd_f32_continuous_batch")
@export
fn kv_cache_length_h8_d32_bshd_f32_continuous_batch[
    target: StringLiteral = "cpu"
](
    kv_collection: ContinuousBatchingKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(num_heads=8, head_size=32),
    ],
    output: NDBuffer[DType.uint32, 1],
    ctx: MojoCallContextPtr,
) raises:
    return _kv_cache_length[target=target](kv_collection, output, ctx)


@mogg_register("kv_cache_length_h8_d64_bshd_bf16_continuous_batch")
@export
fn kv_cache_length_h8_d64_bshd_bf16_continuous_batch[
    target: StringLiteral = "cpu"
](
    kv_collection: ContinuousBatchingKVCacheCollection[
        DType.bfloat16,
        KVCacheStaticParams(num_heads=8, head_size=64),
    ],
    output: NDBuffer[DType.uint32, 1],
    ctx: MojoCallContextPtr,
) raises:
    return _kv_cache_length[target=target](kv_collection, output, ctx)


@mogg_register("kv_cache_length_h8_d64_bshd_f32_continuous_batch")
@export
fn kv_cache_length_h8_d64_bshd_f32_continuous_batch[
    target: StringLiteral = "cpu"
](
    kv_collection: ContinuousBatchingKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(num_heads=8, head_size=64),
    ],
    output: NDBuffer[DType.uint32, 1],
    ctx: MojoCallContextPtr,
) raises:
    return _kv_cache_length[target=target](kv_collection, output, ctx)


@always_inline
fn _kv_cache_length[
    collection_t: KVCollectionT, //, target: StringLiteral
](
    kv_collection: collection_t,
    output: NDBuffer[DType.uint32, 1],
    ctx: MojoCallContextPtr,
) raises:
    """Returns the size of the cache in a ContiguousKVCacheCollection mo.opaque object.
    """

    @parameter
    if target != "cpu":
        var dev_ctx = ctx.get_device_context()
        var size = output.dim[0]()
        var dst = DeviceBuffer(dev_ctx, output.data, size, owning=False)
        var src = DeviceBuffer(
            dev_ctx, kv_collection.cache_length_nd().data, size, owning=False
        )
        dev_ctx.enqueue_copy_device_to_device(dst, src)
    else:
        memcpy(
            output.data,
            kv_collection.cache_length_nd().data,
            output.dim[0](),
        )


@mogg_register("fused_qkv_matmul_kv_cache_h6_d48_bshd")
@export
fn fused_qkv_matmul_kv_cache_h6_d48_bshd[
    type: DType,
    hidden_state_shape: DimList,
    weight_shape: DimList,
    output_shape: DimList,
    target: StringLiteral = "cpu",
](
    hidden_state: NDBuffer[type, 3, hidden_state_shape],
    weight: NDBuffer[type, 2, weight_shape],
    kv_collection: ContiguousKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=6, head_size=48),
    ],
    layer_idx: UInt32,
    output: NDBuffer[type, 3, output_shape],
    ctx: MojoCallContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The historical KVCache for keys and values. The KVCache for
            this layer is retrieved via layer_idx.
        layer_idx: The index of the layer being executed. Used to retrieve the KVCache
            for the given layer from kv_collection.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
        ctx: The call context pointer, passed by the graph compiler.
    """
    with Trace[TraceLevel.OP, target=target](
        "fused_qkv_matmul_kv_cache_h6_d48_bshd"
    ):
        return _fused_qkv_matmul_kv_cache[
            kv_collection.CacheType, target=target
        ](hidden_state, weight, kv_collection, layer_idx, output, ctx)


@mogg_register("fused_qkv_matmul_kv_cache_h8_d128_bshd")
@export
fn fused_qkv_matmul_kv_cache_h8_d128_bshd[
    type: DType,
    hidden_state_shape: DimList,
    weight_shape: DimList,
    output_shape: DimList,
    target: StringLiteral = "cpu",
](
    hidden_state: NDBuffer[type, 3, hidden_state_shape],
    weight: NDBuffer[type, 2, weight_shape],
    kv_collection: ContiguousKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=8, head_size=128),
    ],
    layer_idx: UInt32,
    output: NDBuffer[type, 3, output_shape],
    ctx: MojoCallContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The historical KVCache for keys and values. The KVCache for
            this layer is retrieved via layer_idx.
        layer_idx: The index of the layer being executed. Used to retrieve the KVCache
            for the given layer from kv_collection.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
        ctx: The call context pointer, passed by the graph compiler.
    """
    with Trace[TraceLevel.OP, target=target](
        "fused_qkv_matmul_kv_cache_h8_d128_bshd"
    ):
        return _fused_qkv_matmul_kv_cache[
            kv_collection.CacheType, target=target
        ](hidden_state, weight, kv_collection, layer_idx, output, ctx)


@mogg_register("fused_qkv_matmul_kv_cache_h1_d16_bshd")
@export
fn fused_qkv_matmul_kv_cache_h1_d16_bshd[
    type: DType,
    hidden_state_shape: DimList,
    weight_shape: DimList,
    output_shape: DimList,
    target: StringLiteral = "cpu",
](
    hidden_state: NDBuffer[type, 3, hidden_state_shape],
    weight: NDBuffer[type, 2, weight_shape],
    kv_collection: ContiguousKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=1, head_size=16),
    ],
    layer_idx: UInt32,
    output: NDBuffer[type, 3, output_shape],
    ctx: MojoCallContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The historical KVCache for keys and values. The KVCache for
            this layer is retrieved via layer_idx.
        layer_idx: The index of the layer being executed. Used to retrieve the KVCache
            for the given layer from kv_collection.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
        ctx: The call context pointer, passed by the graph compiler.
    """
    with Trace[TraceLevel.OP, target=target](
        "fused_qkv_matmul_kv_cache_h1_d16_bshd"
    ):
        return _fused_qkv_matmul_kv_cache[
            kv_collection.CacheType, target=target
        ](hidden_state, weight, kv_collection, layer_idx, output, ctx)


alias embed_fn_type = fn[type: DType, width: Int] (
    IndexList[4], SIMD[type, width]
) capturing -> SIMD[type, width]


@mogg_register("fused_qkv_matmul_kv_cache_h8_d32_bshd")
@export
fn fused_qkv_matmul_kv_cache_h8_d32_bshd[
    type: DType,
    hidden_state_shape: DimList,
    weight_shape: DimList,
    output_shape: DimList,
    target: StringLiteral = "cpu",
](
    hidden_state: NDBuffer[type, 3, hidden_state_shape],
    weight: NDBuffer[type, 2, weight_shape],
    kv_collection: ContiguousKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=8, head_size=32),
    ],
    layer_idx: UInt32,
    output: NDBuffer[type, 3, output_shape],
    ctx: MojoCallContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The historical KVCache for keys and values. The KVCache for
            this layer is retrieved via layer_idx.
        layer_idx: The index of the layer being executed. Used to retrieve the KVCache
            for the given layer from kv_collection.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
        ctx: The call context pointer, passed by the graph compiler.
    """
    with Trace[TraceLevel.OP, target=target](
        "fused_qkv_matmul_kv_cache_h8_d32_bshd"
    ):
        return _fused_qkv_matmul_kv_cache[
            kv_collection.CacheType, target=target
        ](hidden_state, weight, kv_collection, layer_idx, output, ctx)


@mogg_register("fused_qkv_matmul_kv_cache_h8_d64_bshd")
@export
fn fused_qkv_matmul_kv_cache_h8_d64_bshd[
    type: DType,
    hidden_state_shape: DimList,
    weight_shape: DimList,
    output_shape: DimList,
    target: StringLiteral = "cpu",
](
    hidden_state: NDBuffer[type, 3, hidden_state_shape],
    weight: NDBuffer[type, 2, weight_shape],
    kv_collection: ContinuousBatchingKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=8, head_size=64),
    ],
    layer_idx: UInt32,
    output: NDBuffer[type, 3, output_shape],
    ctx: MojoCallContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The historical KVCache for keys and values. The KVCache for
            this layer is retrieved via layer_idx.
        layer_idx: The index of the layer being executed. Used to retrieve the KVCache
            for the given layer from kv_collection.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
        ctx: The call context pointer, passed by the graph compiler.
    """
    with Trace[TraceLevel.OP, target=target](
        "fused_qkv_matmul_kv_cache_h8_d64_bshd"
    ):
        return _fused_qkv_matmul_kv_cache[
            kv_collection.CacheType, target=target
        ](hidden_state, weight, kv_collection, layer_idx, output, ctx)


@mogg_register("fused_qkv_matmul_kv_cache_h8_d128_bshd_continuous_batch")
@export
fn fused_qkv_matmul_kv_cache_h8_d128_bshd_continuous_batch[
    type: DType,
    hidden_state_shape: DimList,
    weight_shape: DimList,
    output_shape: DimList,
    target: StringLiteral = "cpu",
](
    hidden_state: NDBuffer[type, 3, hidden_state_shape],
    weight: NDBuffer[type, 2, weight_shape],
    kv_collection: ContinuousBatchingKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=8, head_size=128),
    ],
    layer_idx: UInt32,
    output: NDBuffer[type, 3, output_shape],
    ctx: MojoCallContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The historical KVCache for keys and values. The KVCache for
            this layer is retrieved via layer_idx.
        layer_idx: The index of the layer being executed. Used to retrieve the KVCache
            for the given layer from kv_collection.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
        ctx: The call context pointer, passed by the graph compiler.
    """
    with Trace[TraceLevel.OP, target=target](
        "fused_qkv_matmul_kv_cache_h8_d128_bshd_continuous_batch"
    ):
        return _fused_qkv_matmul_kv_cache[
            kv_collection.CacheType, target=target
        ](hidden_state, weight, kv_collection, layer_idx, output, ctx)


@mogg_register("fused_qkv_matmul_kv_cache_h1_d16_bshd_continuous_batch")
@export
fn fused_qkv_matmul_kv_cache_h1_d16_bshd_continuous_batch[
    type: DType,
    hidden_state_shape: DimList,
    weight_shape: DimList,
    output_shape: DimList,
    target: StringLiteral = "cpu",
](
    hidden_state: NDBuffer[type, 3, hidden_state_shape],
    weight: NDBuffer[type, 2, weight_shape],
    kv_collection: ContinuousBatchingKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=1, head_size=16),
    ],
    layer_idx: UInt32,
    output: NDBuffer[type, 3, output_shape],
    ctx: MojoCallContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The historical KVCache for keys and values. The KVCache for
            this layer is retrieved via layer_idx.
        layer_idx: The index of the layer being executed. Used to retrieve the KVCache
            for the given layer from kv_collection.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
        ctx: The call context pointer, passed by the graph compiler.
    """
    with Trace[TraceLevel.OP, target=target](
        "fused_qkv_matmul_kv_cache_h1_d16_bshd_continuous_batch"
    ):
        return _fused_qkv_matmul_kv_cache[
            kv_collection.CacheType, target=target
        ](hidden_state, weight, kv_collection, layer_idx, output, ctx)


@mogg_register("fused_qkv_matmul_kv_cache_h8_d32_bshd_continuous_batch")
@export
fn fused_qkv_matmul_kv_cache_h8_d32_bshd_continuous_batch[
    type: DType,
    hidden_state_shape: DimList,
    weight_shape: DimList,
    output_shape: DimList,
    target: StringLiteral = "cpu",
](
    hidden_state: NDBuffer[type, 3, hidden_state_shape],
    weight: NDBuffer[type, 2, weight_shape],
    kv_collection: ContiguousKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=8, head_size=32),
    ],
    layer_idx: UInt32,
    output: NDBuffer[type, 3, output_shape],
    ctx: MojoCallContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The historical KVCache for keys and values. The KVCache for
            this layer is retrieved via layer_idx.
        layer_idx: The index of the layer being executed. Used to retrieve the KVCache
            for the given layer from kv_collection.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
        ctx: The call context pointer, passed by the graph compiler.
    """
    with Trace[TraceLevel.OP, target=target](
        "fused_qkv_matmul_kv_cache_h8_d32_bshd_continuous_batch"
    ):
        return _fused_qkv_matmul_kv_cache[
            kv_collection.CacheType, target=target
        ](hidden_state, weight, kv_collection, layer_idx, output, ctx)


@mogg_register("fused_qkv_matmul_kv_cache_h8_d64_bshd_continuous_batch")
@export
fn fused_qkv_matmul_kv_cache_h8_d64_bshd_continuous_batch[
    type: DType,
    hidden_state_shape: DimList,
    weight_shape: DimList,
    output_shape: DimList,
    target: StringLiteral = "cpu",
](
    hidden_state: NDBuffer[type, 3, hidden_state_shape],
    weight: NDBuffer[type, 2, weight_shape],
    kv_collection: ContinuousBatchingKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=8, head_size=64),
    ],
    layer_idx: UInt32,
    output: NDBuffer[type, 3, output_shape],
    ctx: MojoCallContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The historical KVCache for keys and values. The KVCache for
            this layer is retrieved via layer_idx.
        layer_idx: The index of the layer being executed. Used to retrieve the KVCache
            for the given layer from kv_collection.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
        ctx: The call context pointer, passed by the graph compiler.
    """
    with Trace[TraceLevel.OP, target=target](
        "fused_qkv_matmul_kv_cache_h8_d64_bshd_continuous_batch"
    ):
        return _fused_qkv_matmul_kv_cache[
            kv_collection.CacheType, target=target
        ](hidden_state, weight, kv_collection, layer_idx, output, ctx)


@always_inline
fn _fused_qkv_matmul_kv_cache[
    type: DType,
    hidden_state_shape: DimList,
    weight_shape: DimList,
    output_shape: DimList,
    collection_t: KVCollectionT, //,
    cache_t: KVCacheT,
    *,
    target: StringLiteral,
](
    hidden_state: NDBuffer[type, 3, hidden_state_shape],
    weight: NDBuffer[type, 2, weight_shape],
    kv_collection: collection_t,
    layer_idx: UInt32,
    output: NDBuffer[type, 3, output_shape],
    context: MojoCallContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The historical KVCache for keys and values. The KVCache for
            this layer is retrieved via layer_idx.
        layer_idx: The index of the layer being executed. Used to retrieve the KVCache
            for the given layer from kv_collection.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
        context: The call context pointer, passed by the graph compiler.
    """
    var cuda_ctx: Optional[DeviceContext] = None

    @parameter
    if target != "cpu":
        cuda_ctx = context.get_device_context()

    return _fused_qkv_matmul_kv_cache_impl[cache_t, target=target](
        hidden_state, weight, kv_collection, layer_idx, output, cuda_ctx
    )


@always_inline
fn _fused_qkv_matmul_kv_cache_impl[
    type: DType,
    hidden_state_shape: DimList,
    weight_shape: DimList,
    output_shape: DimList,
    collection_t: KVCollectionT, //,
    cache_t: KVCacheT,
    *,
    target: StringLiteral,
    q_embed_fn: OptionalReg[embed_fn_type] = None,
    k_embed_fn: OptionalReg[embed_fn_type] = None,
](
    hidden_state: NDBuffer[type, 3, hidden_state_shape],
    weight: NDBuffer[type, 2, weight_shape],
    kv_collection: collection_t,
    layer_idx: UInt32,
    output: NDBuffer[type, 3, output_shape],
    context: Optional[DeviceContext],
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The historical KVCache for keys and values. The KVCache for
            this layer is retrieved via layer_idx.
        layer_idx: The index of the layer being executed. Used to retrieve the KVCache
            for the given layer from kv_collection.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
        context: The DeviceContext. This is unused if target == "cpu".
    """
    alias kv_params = cache_t.get_kv_params()
    alias N = weight_shape.get[0]()
    alias K = weight_shape.get[1]()

    var BS: UInt = hidden_state.dim[0]()
    var SEQ_LEN: UInt = hidden_state.dim[1]()

    var q_dim = output.dim[2]()
    var k_dim = kv_params.head_size * kv_params.num_heads
    var qk_offset = q_dim + k_dim

    var k_cache = kv_collection.get_key_cache[cache_t](int(layer_idx))
    var v_cache = kv_collection.get_value_cache[cache_t](int(layer_idx))

    @parameter
    @__copy_capture(output, q_dim, qk_offset, SEQ_LEN)
    fn write_to_cache_common[
        type_: DType, width: Int, cache_t_: KVCacheT, *, alignment: Int = 1
    ](
        k_cache: cache_t_,
        v_cache: cache_t_,
        idx: IndexList[2],
        val: SIMD[type_, width],
    ):
        b_idx, t_idx = divmod(UInt(idx[0]), SEQ_LEN)
        if idx[1] < q_dim:
            output.store[width=width, alignment=alignment](
                Index(int(b_idx), int(t_idx), idx[1]),
                rebind[SIMD[type, width]](val),
            )
            return

        var h_idx: UInt
        var hd_idx: UInt
        var cache: cache_t_
        var output_val = val
        if idx[1] < qk_offset:
            cache = k_cache
            h_idx, hd_idx = divmod(UInt(idx[1]) - q_dim, kv_params.head_size)

        else:
            cache = v_cache
            h_idx, hd_idx = divmod(
                UInt(idx[1]) - qk_offset, kv_params.head_size
            )

        var valid_len = cache.cache_length(b_idx)
        var cache_t_idx = t_idx + valid_len
        cache.store(
            b_idx,
            h_idx,
            cache_t_idx,
            hd_idx,
            rebind[SIMD[type, width]](output_val),
        )

    # TODO this is necessary due to traits not having a notion of being register_passable
    # remove this forking after MOCO-1205 (or after we get rid of mo.opaque)
    @parameter
    if _type_is_eq[cache_t, ContiguousKVCache[type, kv_params]]():
        # cast to a register passable type so the function closure works on GPU
        var k_cache_reg = rebind[ContiguousKVCache[type, kv_params]](k_cache)
        var v_cache_reg = rebind[ContiguousKVCache[type, kv_params]](v_cache)

        @parameter
        @__copy_capture(k_cache_reg, v_cache_reg)
        fn write_to_cache_contig[
            type_: DType, width: Int, *, alignment: Int = 1
        ](idx: IndexList[2], val: SIMD[type_, width]):
            write_to_cache_common[alignment=alignment,](
                k_cache_reg, v_cache_reg, idx, val
            )

        _matmul_common[
            target=target, elementwise_lambda_fn=write_to_cache_contig
        ](hidden_state, weight, context)
    elif _type_is_eq[cache_t, ContinuousBatchingKVCache[type, kv_params]]():
        # cast to a register passable type so the function closure works on GPU
        var k_cache_reg = rebind[ContinuousBatchingKVCache[type, kv_params]](
            k_cache
        )
        var v_cache_reg = rebind[ContinuousBatchingKVCache[type, kv_params]](
            v_cache
        )

        @parameter
        @__copy_capture(k_cache_reg, v_cache_reg)
        fn write_to_cache_continuous[
            type_: DType, width: Int, *, alignment: Int = 1
        ](idx: IndexList[2], val: SIMD[type_, width]):
            write_to_cache_common[alignment=alignment,](
                k_cache_reg, v_cache_reg, idx, val
            )

        _matmul_common[
            target=target, elementwise_lambda_fn=write_to_cache_continuous
        ](hidden_state, weight, context)


@always_inline
fn _matmul_common[
    type: DType, //,
    *,
    target: StringLiteral,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    hidden_state: NDBuffer[type, 3, _],
    weight: NDBuffer[type, 2, _],
    context: Optional[DeviceContext],
) raises:
    var BS = hidden_state.dim[0]()
    var SEQ_LEN = hidden_state.dim[1]()
    alias N = weight.shape.get[0]()
    alias K = weight.shape.get[1]()

    var hidden_state_2d = NDBuffer[
        type, 2, DimList(Dim(), hidden_state.shape.get[2]())
    ](
        hidden_state.data,
        IndexList[2](BS * SEQ_LEN, K),
    )

    var c_nd: NDBuffer[type, 2, DimList(Dim(), N)]

    @parameter
    if target == "cpu":
        var c_ptr = UnsafePointer[Scalar[type]].alloc(BS * SEQ_LEN * N)

        c_nd = NDBuffer[type, 2, DimList(Dim(), N)](
            c_ptr,
            IndexList[2](BS * SEQ_LEN, N),
        )
    else:
        c_nd = NDBuffer[type, 2, DimList(Dim(), N)](
            UnsafePointer[Scalar[type]](),
            IndexList[2](BS * SEQ_LEN, N),
        )

    # TODO unify with other matmul
    @parameter
    if target == "cpu":
        var kernel_type_m = hidden_state_2d.shape.at[0]().or_else(0)

        _matmul_cpu[
            transpose_b=True,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ](c_nd, hidden_state_2d, weight, kernel_type_m)
        c_nd.data.free()

    else:
        _matmul_gpu[
            elementwise_lambda_fn=elementwise_lambda_fn,
            use_tensor_core=True,
            transpose_b=True,
            target=target,
        ](c_nd, hidden_state_2d, weight, context.value())


@mogg_register("fused_qk_rope_h6_d48_bshd")
fn fused_qk_rope_h6_d48_bshd[
    type: DType, //,
    *,
    target: StringLiteral,
](
    q_proj: NDBuffer[type, 4, *_],
    kv_collection: ContiguousKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=6, head_size=48),
    ],
    freqs_cis: NDBuffer[type, 2, *_],
    layer_idx: UInt32,
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr = MojoCallContextPtr(),
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
    # Pass device context only on GPU.
    var dev_ctx = Optional[
        DeviceContext
    ]() if target == "cpu" else context.get_device_context()
    with Trace[TraceLevel.OP, target=target]("fused_qk_rope_h6_d48_bshd"):
        fused_qk_rope[kv_collection.CacheType, target=target](
            q_proj, kv_collection, freqs_cis, layer_idx, output, dev_ctx
        )


@mogg_register("fused_qk_rope_h8_d128_bshd")
fn fused_qk_rope_h8_d128_bshd[
    type: DType, //,
    *,
    target: StringLiteral,
](
    q_proj: NDBuffer[type, 4, *_],
    kv_collection: ContiguousKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=8, head_size=128),
    ],
    freqs_cis: NDBuffer[type, 2, *_],
    layer_idx: UInt32,
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr = MojoCallContextPtr(),
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
    # Pass device context only on GPU.
    var dev_ctx = Optional[
        DeviceContext
    ]() if target == "cpu" else context.get_device_context()
    with Trace[TraceLevel.OP, target=target]("fused_qk_rope_h8_d128_bshd"):
        fused_qk_rope[kv_collection.CacheType, target=target](
            q_proj, kv_collection, freqs_cis, layer_idx, output, dev_ctx
        )


@mogg_register("fused_qk_rope_h1_d16_bshd")
fn fused_qk_rope_h1_d16_bshd[
    type: DType, //,
    *,
    target: StringLiteral,
](
    q_proj: NDBuffer[type, 4, *_],
    kv_collection: ContiguousKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=1, head_size=16),
    ],
    freqs_cis: NDBuffer[type, 2, *_],
    layer_idx: UInt32,
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr = MojoCallContextPtr(),
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
    # Pass device context only on GPU.
    var dev_ctx = Optional[
        DeviceContext
    ]() if target == "cpu" else context.get_device_context()
    with Trace[TraceLevel.OP, target=target]("fused_qk_rope_h1_d16_bshd"):
        fused_qk_rope[kv_collection.CacheType, target=target](
            q_proj, kv_collection, freqs_cis, layer_idx, output, dev_ctx
        )


@mogg_register("fused_qk_rope_h8_d32_bshd")
fn fused_qk_rope_h8_d32_bshd[
    type: DType, //,
    *,
    target: StringLiteral,
](
    q_proj: NDBuffer[type, 4, *_],
    kv_collection: ContiguousKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=8, head_size=32),
    ],
    freqs_cis: NDBuffer[type, 2, *_],
    layer_idx: UInt32,
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr = MojoCallContextPtr(),
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
    # Pass device context only on GPU.
    var dev_ctx = Optional[
        DeviceContext
    ]() if target == "cpu" else context.get_device_context()
    with Trace[TraceLevel.OP, target=target]("fused_qk_rope_h8_d32_bshd"):
        fused_qk_rope[kv_collection.CacheType, target=target](
            q_proj, kv_collection, freqs_cis, layer_idx, output, dev_ctx
        )


@mogg_register("fused_qk_rope_h8_d64_bshd")
fn fused_qk_rope_h8_d64_bshd[
    type: DType, //,
    *,
    target: StringLiteral,
](
    q_proj: NDBuffer[type, 4, *_],
    kv_collection: ContiguousKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=8, head_size=64),
    ],
    freqs_cis: NDBuffer[type, 2, *_],
    layer_idx: UInt32,
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr = MojoCallContextPtr(),
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
    # Pass device context only on GPU.
    var dev_ctx = Optional[
        DeviceContext
    ]() if target == "cpu" else context.get_device_context()
    with Trace[TraceLevel.OP, target=target]("fused_qk_rope_h8_d64_bshd"):
        fused_qk_rope[kv_collection.CacheType, target=target](
            q_proj, kv_collection, freqs_cis, layer_idx, output, dev_ctx
        )


@mogg_register("fused_qk_rope_h8_d128_bshd_continuous_batch")
fn fused_qk_rope_h8_d128_bshd_continuous_batch[
    type: DType, //,
    *,
    target: StringLiteral,
](
    q_proj: NDBuffer[type, 4, *_],
    kv_collection: ContinuousBatchingKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=8, head_size=128),
    ],
    freqs_cis: NDBuffer[type, 2, *_],
    layer_idx: UInt32,
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr = MojoCallContextPtr(),
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
    # Pass device context only on GPU.
    var dev_ctx = Optional[
        DeviceContext
    ]() if target == "cpu" else context.get_device_context()
    with Trace[TraceLevel.OP, target=target](
        "fused_qk_rope_h8_d128_bshd_continuous_batch"
    ):
        fused_qk_rope[kv_collection.CacheType, target=target](
            q_proj, kv_collection, freqs_cis, layer_idx, output, dev_ctx
        )


@mogg_register("fused_qk_rope_h1_d16_bshd_continuous_batch")
fn fused_qk_rope_h1_d16_bshd_continuous_batch[
    type: DType, //,
    *,
    target: StringLiteral,
](
    q_proj: NDBuffer[type, 4, *_],
    kv_collection: ContinuousBatchingKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=1, head_size=16),
    ],
    freqs_cis: NDBuffer[type, 2, *_],
    layer_idx: UInt32,
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr = MojoCallContextPtr(),
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
    # Pass device context only on GPU.
    var dev_ctx = Optional[
        DeviceContext
    ]() if target == "cpu" else context.get_device_context()
    with Trace[TraceLevel.OP, target=target](
        "fused_qk_rope_h1_d16_bshd_continuous_batch"
    ):
        fused_qk_rope[kv_collection.CacheType, target=target](
            q_proj, kv_collection, freqs_cis, layer_idx, output, dev_ctx
        )


@mogg_register("fused_qk_rope_h8_d32_bshd_continuous_batch")
fn fused_qk_rope_h8_d32_bshd_continuous_batch[
    type: DType, //,
    *,
    target: StringLiteral,
](
    q_proj: NDBuffer[type, 4, *_],
    kv_collection: ContinuousBatchingKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=8, head_size=32),
    ],
    freqs_cis: NDBuffer[type, 2, *_],
    layer_idx: UInt32,
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr = MojoCallContextPtr(),
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
    # Pass device context only on GPU.
    var dev_ctx = Optional[
        DeviceContext
    ]() if target == "cpu" else context.get_device_context()
    with Trace[TraceLevel.OP, target=target](
        "fused_qk_rope_h8_d32_bshd_continuous_batch"
    ):
        fused_qk_rope[kv_collection.CacheType, target=target](
            q_proj, kv_collection, freqs_cis, layer_idx, output, dev_ctx
        )


@mogg_register("fused_qk_rope_h8_d64_bshd_continuous_batch")
fn fused_qk_rope_h8_d64_bshd_continuous_batch[
    type: DType, //,
    *,
    target: StringLiteral,
](
    q_proj: NDBuffer[type, 4, *_],
    kv_collection: ContinuousBatchingKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=8, head_size=64),
    ],
    freqs_cis: NDBuffer[type, 2, *_],
    layer_idx: UInt32,
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr = MojoCallContextPtr(),
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
    # Pass device context only on GPU.
    var dev_ctx = Optional[
        DeviceContext
    ]() if target == "cpu" else context.get_device_context()
    with Trace[TraceLevel.OP, target=target](
        "fused_qk_rope_h8_d64_bshd_continuous_batch"
    ):
        fused_qk_rope[kv_collection.CacheType, target=target](
            q_proj, kv_collection, freqs_cis, layer_idx, output, dev_ctx
        )


@mogg_register("flash_attention_kv_cache_h6_d48_bshd")
@export
fn flash_attention_kv_cache_h6_d48_bshd[
    type: DType, //,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[type, 4, *_],
    kv_collection: ContiguousKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=6, head_size=48),
    ],
    layer_idx: UInt32,
    mask: NDBuffer[type, *_],
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr,
) raises:
    with Trace[TraceLevel.OP, target=target](
        "flash_attention_kv_cache_h6_d48_bshd"
    ):
        return _flash_attention_kv_cache[
            kv_collection.CacheType, target=target
        ](
            q,
            kv_collection,
            layer_idx,
            mask,
            valid_lengths,
            scale,
            output,
            context,
        )


@mogg_register("flash_attention_kv_cache_h8_d128_bshd")
@export
fn flash_attention_kv_cache_h8_d128_bshd[
    type: DType, //,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[type, 4, *_],
    kv_collection: ContiguousKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=8, head_size=128),
    ],
    layer_idx: UInt32,
    mask: NDBuffer[type, *_],
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr,
) raises:
    with Trace[TraceLevel.OP, target=target](
        "flash_attention_kv_cache_h8_d128_bshd"
    ):
        return _flash_attention_kv_cache[
            kv_collection.CacheType, target=target
        ](
            q,
            kv_collection,
            layer_idx,
            mask,
            valid_lengths,
            scale,
            output,
            context,
        )


@mogg_register("flash_attention_kv_cache_h1_d16_bshd")
@export
fn flash_attention_kv_cache_h1_d16_bshd[
    type: DType, //,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[type, 4, *_],
    kv_collection: ContiguousKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=1, head_size=16),
    ],
    layer_idx: UInt32,
    mask: NDBuffer[type, *_],
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr,
) raises:
    with Trace[TraceLevel.OP, target=target](
        "flash_attention_kv_cache_h1_d16_bshd"
    ):
        return _flash_attention_kv_cache[
            kv_collection.CacheType, target=target
        ](
            q,
            kv_collection,
            layer_idx,
            mask,
            valid_lengths,
            scale,
            output,
            context,
        )


@mogg_register("flash_attention_kv_cache_h8_d32_bshd")
@export
fn flash_attention_kv_cache_h8_d32_bshd[
    type: DType, //,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[type, 4, *_],
    kv_collection: ContiguousKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=8, head_size=32),
    ],
    layer_idx: UInt32,
    mask: NDBuffer[type, *_],
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr,
) raises:
    with Trace[TraceLevel.OP, target=target](
        "flash_attention_kv_cache_h8_d32_bshd"
    ):
        return _flash_attention_kv_cache[
            kv_collection.CacheType, target=target
        ](
            q,
            kv_collection,
            layer_idx,
            mask,
            valid_lengths,
            scale,
            output,
            context,
        )


@mogg_register("flash_attention_kv_cache_h8_d64_bshd")
@export
fn flash_attention_kv_cache_h8_d64_bshd[
    type: DType, //,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[type, 4, *_],
    kv_collection: ContiguousKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=8, head_size=64),
    ],
    layer_idx: UInt32,
    mask: NDBuffer[type, *_],
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr,
) raises:
    with Trace[TraceLevel.OP, target=target](
        "flash_attention_kv_cache_h8_d64_bshd"
    ):
        return _flash_attention_kv_cache[
            kv_collection.CacheType, target=target
        ](
            q,
            kv_collection,
            layer_idx,
            mask,
            valid_lengths,
            scale,
            output,
            context,
        )


@mogg_register("flash_attention_kv_cache_h8_d128_bshd_continuous_batch")
@export
fn flash_attention_kv_cache_h8_d128_bshd_continuous_batch[
    type: DType, //,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[type, 4, *_],
    kv_collection: ContinuousBatchingKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=8, head_size=128),
    ],
    layer_idx: UInt32,
    mask: NDBuffer[type, *_],
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr,
) raises:
    with Trace[TraceLevel.OP, target=target](
        "flash_attention_kv_cache_h8_d128_bshd_continuous_batch"
    ):
        return _flash_attention_kv_cache[
            kv_collection.CacheType, target=target
        ](
            q,
            kv_collection,
            layer_idx,
            mask,
            valid_lengths,
            scale,
            output,
            context,
        )


@mogg_register("flash_attention_kv_cache_h1_d16_bshd_continuous_batch")
@export
fn flash_attention_kv_cache_h1_d16_bshd_continuous_batch[
    type: DType, //,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[type, 4, *_],
    kv_collection: ContinuousBatchingKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=1, head_size=16),
    ],
    layer_idx: UInt32,
    mask: NDBuffer[type, *_],
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr,
) raises:
    with Trace[TraceLevel.OP, target=target](
        "flash_attention_kv_cache_h1_d16_bshd_continuous_batch"
    ):
        return _flash_attention_kv_cache[
            kv_collection.CacheType, target=target
        ](
            q,
            kv_collection,
            layer_idx,
            mask,
            valid_lengths,
            scale,
            output,
            context,
        )


@mogg_register("flash_attention_kv_cache_h8_d32_bshd_continuous_batch")
@export
fn flash_attention_kv_cache_h8_d32_bshd_continuous_batch[
    type: DType, //,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[type, 4, *_],
    kv_collection: ContinuousBatchingKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=8, head_size=32),
    ],
    layer_idx: UInt32,
    mask: NDBuffer[type, *_],
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr,
) raises:
    with Trace[TraceLevel.OP, target=target](
        "flash_attention_kv_cache_h8_d32_bshd_continuous_batch"
    ):
        return _flash_attention_kv_cache[
            kv_collection.CacheType, target=target
        ](
            q,
            kv_collection,
            layer_idx,
            mask,
            valid_lengths,
            scale,
            output,
            context,
        )


@mogg_register("flash_attention_kv_cache_h8_d64_bshd_continuous_batch")
@export
fn flash_attention_kv_cache_h8_d64_bshd_continuous_batch[
    type: DType, //,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[type, 4, *_],
    kv_collection: ContinuousBatchingKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=8, head_size=64),
    ],
    layer_idx: UInt32,
    mask: NDBuffer[type, *_],
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr,
) raises:
    with Trace[TraceLevel.OP, target=target](
        "flash_attention_kv_cache_h8_d64_bshd_continuous_batch"
    ):
        return _flash_attention_kv_cache[
            kv_collection.CacheType, target=target
        ](
            q,
            kv_collection,
            layer_idx,
            mask,
            valid_lengths,
            scale,
            output,
            context,
        )


@always_inline
fn _flash_attention_kv_cache[
    type: DType,
    collection_t: KVCollectionT, //,
    cache_t: KVCacheT,
    target: StringLiteral,
](
    q: NDBuffer[type, 4, *_],
    kv_collection: collection_t,
    layer_idx: UInt32,
    mask: NDBuffer[type, *_],
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr,
) raises:
    """Performs flash attention using k and v caches from ContiguousKVCache custom types.

    Args:
        q: NDBuffer with shape (batch_size, num_heads, seq_len, head_size).
        kv_collection: The Collection object storing out KVCache entries for this layer
        layer_idx: The current layer, used to retrieve kv_cache objects from kv_colleciton
        mask: The attention mask to apply to the score matrix.
        valid_lengths: The unpadded lengths of the sequences contained in q
        scale: The scaled factor in scaled-dot product attention. Usually isqrt(head_size).
        output: The Pre-allocated output buffer to write results to. Has shape:
            (batch_size, num_heads, seq_len, head_size).
        context: Pointer containing the runtime context for the target device.
    """
    var cuda_ctx: Optional[DeviceContext] = None

    @parameter
    if target != "cpu":
        cuda_ctx = context.get_device_context()

    _flash_attention_kv_cache_impl[cache_t, target=target](
        q,
        kv_collection,
        layer_idx,
        mask,
        valid_lengths,
        scale,
        output,
        cuda_ctx,
    )


@always_inline
fn _flash_attention_kv_cache_impl[
    type: DType,
    collection_t: KVCollectionT, //,
    cache_t: KVCacheT,
    target: StringLiteral,
](
    q: NDBuffer[type, 4, *_],
    kv_collection: collection_t,
    layer_idx: UInt32,
    mask: NDBuffer[type, *_],
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[type, 4, *_],
    context: Optional[DeviceContext],
) raises:
    """Performs flash attention using k and v caches from ContiguousKVCache custom types.

    Args:
        q: NDBuffer with shape (batch_size, num_heads, seq_len, head_size).
        kv_collection: The Collection object storing out KVCache entries for this layer
        layer_idx: The current layer, used to retrieve kv_cache objects from kv_colleciton
        mask: The attention mask to apply to the score matrix.
        valid_lengths: The unpadded lengths of the sequences contained in q
        scale: The scaled factor in scaled-dot product attention. Usually isqrt(head_size).
        output: The Pre-allocated output buffer to write results to. Has shape:
            (batch_size, num_heads, seq_len, head_size).
        context: CUDA DeviceContext. This is not used if target == "cpu"
    """

    var layer_idx_cast = int(layer_idx)
    var k = kv_collection.get_key_cache[cache_t](layer_idx_cast)
    var v = kv_collection.get_value_cache[cache_t](layer_idx_cast)

    @parameter
    if target == "cpu":
        return flash_attention_kv_cache_cpu(q, k, v, mask, scale, output)
    else:
        return _flash_attention_kv_cache_gpu[target=target](
            q, k, v, mask, valid_lengths, scale, output, context.value()
        )


@mogg_register("flash_attention_kv_cache_h8_d128_causal_mask_continuous_batch")
@export
fn flash_attention_kv_cache_h8_d128_causal_mask_continuous_batch[
    type: DType, //,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[type, 4, *_],
    k: ContinuousBatchingKVCache[
        type,
        KVCacheStaticParams(num_heads=8, head_size=128),
    ],
    v: ContinuousBatchingKVCache[type, k.kv_params],
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr,
) raises:
    with Trace[TraceLevel.OP, target=target](
        "flash_attention_kv_cache_h8_d128_causal_mask_continuous_batch"
    ):
        return _flash_attention_kv_cache_causal_mask[target=target](
            q, k, v, valid_lengths, scale, output, context
        )


@mogg_register("flash_attention_kv_cache_h8_d32_causal_mask_continuous_batch")
@export
fn flash_attention_kv_cache_h8_d32_causal_mask_continuous_batch[
    type: DType, //,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[type, 4, *_],
    k: ContinuousBatchingKVCache[
        type,
        KVCacheStaticParams(num_heads=8, head_size=32),
    ],
    v: ContinuousBatchingKVCache[type, k.kv_params],
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr,
) raises:
    with Trace[TraceLevel.OP, target=target](
        "flash_attention_kv_cache_h8_d32_causal_mask_continuous_batch"
    ):
        return _flash_attention_kv_cache_causal_mask[target=target](
            q, k, v, valid_lengths, scale, output, context
        )


@mogg_register("flash_attention_kv_cache_h8_d64_causal_mask_continuous_batch")
@export
fn flash_attention_kv_cache_h8_d64_causal_mask_continuous_batch[
    type: DType, //,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[type, 4, *_],
    k: ContinuousBatchingKVCache[
        type,
        KVCacheStaticParams(num_heads=8, head_size=64),
    ],
    v: ContinuousBatchingKVCache[type, k.kv_params],
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr,
) raises:
    with Trace[TraceLevel.OP, target=target](
        "flash_attention_kv_cache_h8_d64_causal_mask_continuous_batch"
    ):
        return _flash_attention_kv_cache_causal_mask[target=target](
            q, k, v, valid_lengths, scale, output, context
        )


@mogg_register("flash_attention_kv_cache_h1_d16_causal_mask_continuous_batch")
@export
fn flash_attention_kv_cache_h1_d16_causal_mask_continuous_batch[
    type: DType, //,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[type, 4, *_],
    k: ContinuousBatchingKVCache[
        type,
        KVCacheStaticParams(num_heads=1, head_size=16),
    ],
    v: ContinuousBatchingKVCache[type, k.kv_params],
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr,
) raises:
    with Trace[TraceLevel.OP, target=target](
        "flash_attention_kv_cache_h1_d16_causal_mask_continuous_batch"
    ):
        return _flash_attention_kv_cache_causal_mask[target=target](
            q, k, v, valid_lengths, scale, output, context
        )


@always_inline
fn _flash_attention_kv_cache_causal_mask[
    type: DType,
    cache_t: KVCacheT, //,
    target: StringLiteral,
](
    q: NDBuffer[type, 4, *_],
    k: cache_t,
    v: cache_t,
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr,
) raises:
    """Performs flash attention using k and v caches from ContiguousKVCache/ContinuousBatchingKVCache custom types, with the causal mask materialized inside the kernel.

    Args:
        q: NDBuffer with shape (batch_size, num_heads, seq_len, head_size).
        k: ContiguousKVCache type with logical shape (batch_size, num_heads, max_seq_len, head_size).
        v: ContiguousKVCache type with logical shape (batch_size, num_heads, max_seq_len, head_size).
        valid_lengths: The unpadded lengths of the sequences contained in q
        scale: The scaled factor in scaled-dot product attention. Usually isqrt(head_size).
        output: The Pre-allocated output buffer to write results to. Has shape:
            (batch_size, num_heads, seq_len, head_size).
        context: Pointer containing the runtime context for the target device.
    """
    var cuda_ctx: Optional[DeviceContext] = None

    @parameter
    if target != "cpu":
        cuda_ctx = context.get_device_context()

    _flash_attention_kv_cache_causal_mask_impl[target=target](
        q, k, v, valid_lengths, scale, output, cuda_ctx
    )


@always_inline
fn _flash_attention_kv_cache_causal_mask_impl[
    type: DType,
    cache_t: KVCacheT,
    target: StringLiteral,
](
    q: NDBuffer[type, 4, *_],
    k: cache_t,
    v: cache_t,
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[type, 4, *_],
    context: Optional[DeviceContext],
) raises:
    """Performs flash attention using k and v caches from ContiguousKVCache/ContinuousBatchingKVCache custom types, with the causal mask materialized inside the kernel.

    Args:
        q: NDBuffer with shape (batch_size, num_heads, seq_len, head_size).
        k: ContiguousKVCache type with logical shape (batch_size, num_heads, max_seq_len, head_size).
        v: ContiguousKVCache type with logical shape (batch_size, num_heads, max_seq_len, head_size).
        valid_lengths: The unpadded lengths of the sequences contained in q
        scale: The scaled factor in scaled-dot product attention. Usually isqrt(head_size).
        output: The Pre-allocated output buffer to write results to. Has shape:
            (batch_size, num_heads, seq_len, head_size).
        context: CUDA DeviceContext. This is not used if target == "cpu"
    """

    @parameter
    if target == "cpu":
        return flash_attention_kv_cache_cpu(
            q, k, v, CausalMask(), scale, output
        )
    else:
        return _flash_attention_kv_cache_causal_mask_gpu[target=target](
            q, k, v, valid_lengths, scale, output, context.value()
        )


# TODO: Change this as needed when plumbed with pipelines.
#       This is a copy of _flash_attention_kv_cache_gpu with the difference that
#       it calls gpu_flash_attention with the option to use mask tensor and
#       passing CausalMask().
@always_inline
fn _flash_attention_kv_cache_causal_mask_gpu[
    type: DType, cache_t: KVCacheT, //, *, target: StringLiteral
](
    q: NDBuffer[type, 4, *_],
    k: cache_t,
    v: cache_t,
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[type, 4, *_],
    context: DeviceContext,
) raises:
    var mask_nd = NDBuffer[type, 4, DimList.create_unknown[4]()](
        UnsafePointer[Scalar[type]](), IndexList[4]()
    )

    # GPU flash attention kernel gets the cache length from the k tensor shape
    # TODO remove this an instead pass in explicit KVCache lengths to the GPU kernel.
    # KERN-725
    gpu_flash_attention[
        add_attn_mask=False,
        use_tensor_core=True,
        target=target,
    ](
        output,
        q,
        k,
        v,
        mask_nd,
        CausalMask(),
        valid_lengths,
        scale,
        context,
    )


@always_inline
fn _flash_attention_kv_cache_gpu[
    type: DType, cache_t: KVCacheT, //, *, target: StringLiteral
](
    q: NDBuffer[type, 4, *_],
    k: cache_t,
    v: cache_t,
    mask: NDBuffer[type, *_],
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[type, 4, *_],
    context: DeviceContext,
) raises:
    alias wrapped_mask_rank = mask.rank if mask.rank == 4 else 3
    var mask_nd: NDBuffer[
        type,
        wrapped_mask_rank,
        DimList.create_unknown[wrapped_mask_rank](),
    ]

    @parameter
    if mask.rank == 2:
        mask_nd = NDBuffer[
            type,
            wrapped_mask_rank,
            DimList.create_unknown[wrapped_mask_rank](),
        ](
            mask.data,
            IndexList[wrapped_mask_rank](
                q.dim[0](), mask.dim[0](), mask.dim[1]()
            ),
        )
    else:
        mask_nd = rebind[
            NDBuffer[
                type,
                wrapped_mask_rank,
                DimList.create_unknown[wrapped_mask_rank](),
            ]
        ](mask)

    # GPU flash attention kernel gets the cache length from the k tensor shape
    # TODO remove this an instead pass in explicit KVCache lengths to the GPU kernel.
    # KERN-725
    gpu_flash_attention[
        add_attn_mask=True,
        use_tensor_core=True,
        target=target,
    ](
        output,
        q,
        k,
        v,
        mask_nd,
        NullMask(),
        valid_lengths,
        scale,
        context,
    )


fn _contiguous_kv_cache_collection[
    type: DType, //, kv_params: KVCacheStaticParams
](
    key_cache: NDBuffer[type, 5],
    value_cache: NDBuffer[type, 5],
    cache_lengths: NDBuffer[DType.uint32, 1],
    is_cache_empty: NDBuffer[DType.bool, 1],
    # Note that num_layers and batch_size are scalars.
    num_layers: NDBuffer[DType.int32, 1],
    batch_size: NDBuffer[DType.int32, 1],
) -> ContiguousKVCacheCollection[type, kv_params] as result:
    # Marshal NDBuffers into arguments expected by the
    # ContiguousKVCacheCollection constructor.

    seq_ids_list = List[Int]()
    for _ in range(int(batch_size[0])):
        # seq_ids are only used in Mojo for this type,
        # but this op is only used by Python.
        # Just fill it with dummy values.
        seq_ids_list.append(-1)

    return __type_of(result)(
        key_cache,
        value_cache,
        cache_lengths,
        is_cache_empty[0],
        seq_ids_list,
        int(num_layers[0]),
        int(batch_size[0]),
    )


fn _continuous_batch_kv_cache_collection[
    type: DType, //, kv_params: KVCacheStaticParams
](
    blocks: NDBuffer[type, 6],
    cache_lengths: NDBuffer[DType.uint32, 1],
    lookup_table: NDBuffer[DType.uint32, 1],
    is_cache_empty: NDBuffer[DType.bool, 1],
) -> ContinuousBatchingKVCacheCollection[type, kv_params] as result:
    # Marshal NDBuffers into arguments expected by the
    # ContiguousKVCacheCollection constructor.
    batch_size = lookup_table.dim[0]()
    seq_ids_list = List[Int]()
    for _ in range(batch_size):
        # seq_ids are only used in Mojo for this type,
        # but this op is only used by Python.
        # Just fill it with dummy values.
        seq_ids_list.append(-1)

    return __type_of(result)(
        blocks,
        cache_lengths,
        lookup_table,
        is_cache_empty[0],
        seq_ids_list,
    )


# Boilerplate: stub out interface for every combination of KV cache parameters.
alias kv_params_h1_d16_bshd = KVCacheStaticParams(num_heads=1, head_size=16)
alias kv_params_h6_d48_bshd = KVCacheStaticParams(num_heads=6, head_size=48)
alias kv_params_h8_d128_bshd = KVCacheStaticParams(num_heads=8, head_size=128)
alias kv_params_h8_d32_bshd = KVCacheStaticParams(num_heads=8, head_size=32)
alias kv_params_h8_d64_bshd = KVCacheStaticParams(num_heads=8, head_size=64)


@mogg_register("contiguous_kv_cache_collection_h6_d48_bshd")
@export
fn contiguous_kv_cache_collection_h6_d48_bshd[
    type: DType, //, target: StringLiteral
](
    key_cache: NDBuffer[type, 5],
    value_cache: NDBuffer[type, 5],
    cache_lengths: NDBuffer[DType.uint32, 1],
    is_cache_empty: NDBuffer[DType.bool, 1],
    num_layers: NDBuffer[DType.int32, 1],
    batch_size: NDBuffer[DType.int32, 1],
) -> ContiguousKVCacheCollection[
    type,
    kv_params_h6_d48_bshd,
]:
    return _contiguous_kv_cache_collection[kv_params_h6_d48_bshd](
        key_cache,
        value_cache,
        cache_lengths,
        is_cache_empty,
        num_layers,
        batch_size,
    )


@mogg_register("contiguous_kv_cache_collection_h8_d128_bshd")
@export
fn contiguous_kv_cache_collection_h8_d128_bshd[
    type: DType, //, target: StringLiteral
](
    key_cache: NDBuffer[type, 5],
    value_cache: NDBuffer[type, 5],
    cache_lengths: NDBuffer[DType.uint32, 1],
    is_cache_empty: NDBuffer[DType.bool, 1],
    num_layers: NDBuffer[DType.int32, 1],
    batch_size: NDBuffer[DType.int32, 1],
) -> ContiguousKVCacheCollection[
    type,
    kv_params_h8_d128_bshd,
]:
    return _contiguous_kv_cache_collection[kv_params_h8_d128_bshd](
        key_cache,
        value_cache,
        cache_lengths,
        is_cache_empty,
        num_layers,
        batch_size,
    )


@mogg_register("contiguous_kv_cache_collection_h1_d16_bshd")
@export
fn contiguous_kv_cache_collection_h1_d16_bshd[
    type: DType, //, target: StringLiteral
](
    key_cache: NDBuffer[type, 5],
    value_cache: NDBuffer[type, 5],
    cache_lengths: NDBuffer[DType.uint32, 1],
    is_cache_empty: NDBuffer[DType.bool, 1],
    num_layers: NDBuffer[DType.int32, 1],
    batch_size: NDBuffer[DType.int32, 1],
) -> ContiguousKVCacheCollection[
    type,
    kv_params_h1_d16_bshd,
]:
    return _contiguous_kv_cache_collection[kv_params_h1_d16_bshd](
        key_cache,
        value_cache,
        cache_lengths,
        is_cache_empty,
        num_layers,
        batch_size,
    )


@mogg_register("contiguous_kv_cache_collection_h8_d32_bshd")
@export
fn contiguous_kv_cache_collection_h8_d32_bshd[
    type: DType, //, target: StringLiteral
](
    key_cache: NDBuffer[type, 5],
    value_cache: NDBuffer[type, 5],
    cache_lengths: NDBuffer[DType.uint32, 1],
    is_cache_empty: NDBuffer[DType.bool, 1],
    num_layers: NDBuffer[DType.int32, 1],
    batch_size: NDBuffer[DType.int32, 1],
) -> ContiguousKVCacheCollection[
    type,
    kv_params_h8_d32_bshd,
]:
    return _contiguous_kv_cache_collection[kv_params_h8_d32_bshd](
        key_cache,
        value_cache,
        cache_lengths,
        is_cache_empty,
        num_layers,
        batch_size,
    )


@mogg_register("contiguous_kv_cache_collection_h8_d64_bshd")
@export
fn contiguous_kv_cache_collection_h8_d64_bshd[
    type: DType, //, target: StringLiteral
](
    key_cache: NDBuffer[type, 5],
    value_cache: NDBuffer[type, 5],
    cache_lengths: NDBuffer[DType.uint32, 1],
    is_cache_empty: NDBuffer[DType.bool, 1],
    num_layers: NDBuffer[DType.int32, 1],
    batch_size: NDBuffer[DType.int32, 1],
) -> ContiguousKVCacheCollection[
    type,
    kv_params_h8_d64_bshd,
]:
    return _contiguous_kv_cache_collection[kv_params_h8_d64_bshd](
        key_cache,
        value_cache,
        cache_lengths,
        is_cache_empty,
        num_layers,
        batch_size,
    )


@mogg_register("continuous_batching_kv_cache_collection_h8_d128_bshd")
@export
fn continuous_batching_kv_cache_collection_h8_d128_bshd[
    type: DType, //, target: StringLiteral
](
    blocks: NDBuffer[type, 6],
    cache_lengths: NDBuffer[DType.uint32, 1],
    lookup_table: NDBuffer[DType.uint32, 1],
    is_cache_empty: NDBuffer[DType.bool, 1],
) -> ContinuousBatchingKVCacheCollection[
    type,
    kv_params_h8_d128_bshd,
]:
    return _continuous_batch_kv_cache_collection[kv_params_h8_d128_bshd](
        blocks,
        cache_lengths,
        lookup_table,
        is_cache_empty,
    )


@mogg_register("continuous_batching_kv_cache_collection_h8_d32_bshd")
@export
fn continuous_batching_kv_cache_collection_h8_d32_bshd[
    type: DType, //, target: StringLiteral
](
    blocks: NDBuffer[type, 6],
    cache_lengths: NDBuffer[DType.uint32, 1],
    lookup_table: NDBuffer[DType.uint32, 1],
    is_cache_empty: NDBuffer[DType.bool, 1],
) -> ContinuousBatchingKVCacheCollection[
    type,
    kv_params_h8_d32_bshd,
]:
    return _continuous_batch_kv_cache_collection[kv_params_h8_d32_bshd](
        blocks,
        cache_lengths,
        lookup_table,
        is_cache_empty,
    )


@mogg_register("continuous_batching_kv_cache_collection_h8_d64_bshd")
@export
fn continuous_batching_kv_cache_collection_h8_d64_bshd[
    type: DType, //, target: StringLiteral
](
    blocks: NDBuffer[type, 6],
    cache_lengths: NDBuffer[DType.uint32, 1],
    lookup_table: NDBuffer[DType.uint32, 1],
    is_cache_empty: NDBuffer[DType.bool, 1],
) -> ContinuousBatchingKVCacheCollection[
    type,
    kv_params_h8_d64_bshd,
]:
    return _continuous_batch_kv_cache_collection[kv_params_h8_d64_bshd](
        blocks,
        cache_lengths,
        lookup_table,
        is_cache_empty,
    )


@mogg_register("continuous_batching_kv_cache_collection_h1_d16_bshd")
@export
fn continuous_batching_kv_cache_collection_h1_d16_bshd[
    type: DType, //, target: StringLiteral
](
    blocks: NDBuffer[type, 6],
    cache_lengths: NDBuffer[DType.uint32, 1],
    lookup_table: NDBuffer[DType.uint32, 1],
    is_cache_empty: NDBuffer[DType.bool, 1],
) -> ContinuousBatchingKVCacheCollection[
    type,
    kv_params_h1_d16_bshd,
]:
    return _continuous_batch_kv_cache_collection[kv_params_h1_d16_bshd](
        blocks,
        cache_lengths,
        lookup_table,
        is_cache_empty,
    )
