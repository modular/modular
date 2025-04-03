# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from collections import InlineArray, Optional, OptionalReg
from collections.string import StaticString
from math import gcd, isqrt
from sys.info import _current_target, simdwidthof

from algorithm.functional import elementwise
from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceContext
from gpu.host.info import is_cpu, is_gpu
from kv_cache.types import (
    ContinuousBatchingKVCache,
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
    KVCacheT,
    KVCollectionT,
    PagedKVCache,
    PagedKVCacheCollection,
)
from linalg.matmul import elementwise_epilogue_type, matmul
from memory import UnsafePointer, memcpy
from nn._ragged_utils import get_batch_from_row_offsets
from nn.flash_attention import (
    flash_attention_kv_cache as flash_attention_kv_cache_cpu,
)
from nn.fused_qk_rope import fused_qk_rope
from nn.mha import flash_attention as gpu_flash_attention
from nn.mha_mask import CausalMask, NullMask, MaterializedMask
from nn.mha_score_mod import AlibiScoreMod, IdentityScoreMod
from nn.normalization import _rms_norm_impl
from register import register_internal
from runtime.asyncrt import DeviceContextPtr
from runtime.tracing import Trace, TraceLevel, trace_arg

from utils import Index, IndexList

# ===-----------------------------------------------------------------------===#
# Fused QKV matmul (padded)
# ===-----------------------------------------------------------------------===#


@always_inline
fn generic_fused_qkv_matmul_kv_cache_bshd_continuous_batch[
    type: DType,
    target: StaticString = "cpu",
](
    hidden_state: NDBuffer[type, 3, _, _],
    weight: NDBuffer[type, 2, _, _],
    kv_collection: ContinuousBatchingKVCacheCollection,
    layer_idx: UInt32,
    output: NDBuffer[mut=True, type, 3, _, _],
    ctx: DeviceContextPtr,
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

    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            trace_arg("output", output),
            trace_arg("hidden_state", hidden_state),
            trace_arg("weight", weight),
            "layer_idx=" + String(layer_idx),
            "num_heads=" + String(kv_collection.kv_params.num_heads),
            "head_size=" + String(kv_collection.kv_params.head_size),
        )

    with Trace[TraceLevel.OP, target=target](
        "mo.fused_qkv_matmul.padded.continuous_batching.nhead_"
        + String(kv_collection.kv_params.num_heads)
        + ".hdim_"
        + String(kv_collection.kv_params.head_size),
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
    ):
        return _fused_qkv_matmul_kv_cache[
            kv_collection.CacheType, target=target
        ](hidden_state, weight, kv_collection, layer_idx, output, ctx)


@always_inline
fn _fused_qkv_matmul_kv_cache[
    type: DType,
    collection_t: KVCollectionT, //,
    cache_t: KVCacheT,
    *,
    target: StaticString,
](
    hidden_state: NDBuffer[type, 3, _, _],
    weight: NDBuffer[type, 2, _, _],
    kv_collection: collection_t,
    layer_idx: UInt32,
    output: NDBuffer[mut=True, type, 3, _, _],
    context: DeviceContextPtr,
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
    if is_gpu[target]():
        cuda_ctx = context.get_device_context()

    return _fused_qkv_matmul_kv_cache_impl[target=target](
        hidden_state, weight, kv_collection, layer_idx, output, cuda_ctx
    )


alias embed_fn_type = fn[type: DType, width: Int] (
    IndexList[4], SIMD[type, width]
) capturing -> SIMD[type, width]


@always_inline
fn _fused_qkv_matmul_kv_cache_impl[
    type: DType,
    hidden_state_shape: DimList,
    weight_shape: DimList,
    output_shape: DimList,
    collection_t: KVCollectionT, //,
    *,
    target: StaticString,
    q_embed_fn: OptionalReg[embed_fn_type] = None,
    k_embed_fn: OptionalReg[embed_fn_type] = None,
](
    hidden_state: NDBuffer[type, 3, _, hidden_state_shape],
    weight: NDBuffer[type, 2, _, weight_shape],
    kv_collection: collection_t,
    layer_idx: UInt32,
    output: NDBuffer[mut=True, type, 3, _, output_shape],
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
        context: The DeviceContext. This is unused if is_cpu[target]().
    """
    alias cache_t = collection_t.CacheType
    alias cache_type = cache_t.type

    constrained[
        cache_type == type,
        "Expected cache type "
        + String(cache_type)
        + " to match input type "
        + String(type),
    ]()

    alias kv_params = cache_t.kv_params
    alias N = weight_shape.get[0]()
    alias K = weight_shape.get[1]()

    var SEQ_LEN: UInt = hidden_state.dim[1]()

    var q_dim = output.dim[2]()
    var k_dim = kv_params.head_size * kv_params.num_heads
    var qk_offset = q_dim + k_dim

    var k_cache = kv_collection.get_key_cache(Int(layer_idx))
    var v_cache = kv_collection.get_value_cache(Int(layer_idx))

    @parameter
    @__copy_capture(q_dim, qk_offset, SEQ_LEN, k_cache, v_cache)
    fn write_to_cache[
        type_: DType, width: Int, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[type_, width]):
        b_idx, t_idx = divmod(UInt(idx[0]), SEQ_LEN)
        if idx[1] < q_dim:
            output.store[width=width, alignment=alignment](
                Index(Int(b_idx), Int(t_idx), idx[1]),
                rebind[SIMD[type, width]](val),
            )
            return

        var h_idx: UInt
        var hd_idx: UInt
        var cache: cache_t
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
            rebind[SIMD[cache_type, width]](output_val),
        )

    _matmul_common[target=target, elementwise_lambda_fn=write_to_cache](
        hidden_state, weight, context
    )


@always_inline
fn _matmul_common[
    type: DType, //,
    *,
    target: StaticString,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    hidden_state: NDBuffer[type, 3, _, _],
    weight: NDBuffer[type, 2, _, _],
    context: Optional[DeviceContext],
) raises:
    var BS = hidden_state.dim[0]()
    var SEQ_LEN = hidden_state.dim[1]()
    alias N = weight.shape.get[0]()
    alias K = weight.shape.get[1]()

    var hidden_state_2d = NDBuffer[
        type, 2, MutableAnyOrigin, DimList(Dim(), hidden_state.shape.get[2]())
    ](
        hidden_state.data,
        IndexList[2](BS * SEQ_LEN, K),
    )

    var c_nd: NDBuffer[type, 2, MutableAnyOrigin, DimList(Dim(), N)]

    @parameter
    if is_cpu[target]():
        var c_ptr = UnsafePointer[Scalar[type]].alloc(BS * SEQ_LEN * N)

        c_nd = NDBuffer[type, 2, MutableAnyOrigin, DimList(Dim(), N)](
            c_ptr,
            IndexList[2](BS * SEQ_LEN, N),
        )
    else:
        c_nd = NDBuffer[type, 2, MutableAnyOrigin, DimList(Dim(), N)](
            UnsafePointer[Scalar[type]](),
            IndexList[2](BS * SEQ_LEN, N),
        )

    matmul[
        transpose_b=True,
        target=target,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ](c_nd, hidden_state_2d, weight, context)

    @parameter
    if is_cpu[target]():
        c_nd.data.free()


# ===-----------------------------------------------------------------------===#
# Fused QK RoPE (padded)
# ===-----------------------------------------------------------------------===#


@always_inline
fn generic_fused_qk_rope_bshd_continuous_batch[
    type: DType, //,
    *,
    interleaved: Bool,
    target: StaticString,
](
    q_proj: NDBuffer[type, 4, *_],
    kv_collection: ContinuousBatchingKVCacheCollection,
    freqs_cis: NDBuffer[type, 2, *_],
    layer_idx: UInt32,
    output: NDBuffer[mut=True, type, 4, *_],
    context: DeviceContextPtr = DeviceContextPtr(),
) raises:
    """Performs a fused RoPE projection for Q and K projections.

    We have a manually fused QKV projection with mo.opaque types in our Llama model.
    Due to a limitation in custom op definitions, we can't declare both a tensor
    and opaque type as output from a custom kernel. This requires us to only note
    Q_proj as an output from the QKV projection. If we immediately follow the
    QKV proj kernel with a RoPE kernel applied to K, we'll get a race condition
    because the graph compiler doesn't know about the dependency between these
    kernels in the graph definition. Here we fuse the RoPE kernel applied to
    Q_proj with K_proj, so K_proj RoPE is only executed after QKV completes.
    """

    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            trace_arg("output", output),
            trace_arg("q_proj", q_proj),
            trace_arg("freqs_cis", freqs_cis),
            "layer_idx=" + String(layer_idx),
            "num_heads=" + String(kv_collection.kv_params.num_heads),
            "head_size=" + String(kv_collection.kv_params.head_size),
            "interleaved=" + String(interleaved),
        )

    # Pass device context only on GPU.
    var dev_ctx = Optional[DeviceContext]() if is_cpu[
        target
    ]() else context.get_device_context()
    with Trace[TraceLevel.OP, target=target](
        "mo.fused_qk_rope.padded.continuous_batching.nhead_"
        + String(kv_collection.kv_params.num_heads)
        + ".hdim_"
        + String(kv_collection.kv_params.head_size),
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
    ):
        fused_qk_rope[
            kv_collection.CacheType, interleaved=interleaved, target=target
        ](
            q_proj,
            kv_collection,
            freqs_cis,
            layer_idx,
            output,
            dev_ctx,
        )


# ===-----------------------------------------------------------------------===#
# MHA (padded)
# ===-----------------------------------------------------------------------===#


@always_inline
fn generic_flash_attention_kv_cache_continuous_batch[
    target: StaticString, type: DType
](
    q: NDBuffer[type, 4, *_],
    kv_collection: ContinuousBatchingKVCacheCollection,
    layer_idx: UInt32,
    mask: NDBuffer[type, *_],
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[mut=True, type, 4, *_],
    context: DeviceContextPtr,
) raises:
    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            trace_arg("q", q),
            trace_arg("mask", mask),
            trace_arg("valid_lengths", valid_lengths),
            "scale=" + String(scale),
            "layer_idx=" + String(layer_idx),
            "num_heads=" + String(kv_collection.kv_params.num_heads),
            "head_size=" + String(kv_collection.kv_params.head_size),
        )

    with Trace[TraceLevel.OP, target=target](
        "mo.mha.padded.continunous_batching.tensor_mask.no_pos.nhead_"
        + String(kv_collection.kv_params.num_heads)
        + ".hdim_"
        + String(kv_collection.kv_params.head_size),
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
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
    target: StaticString,
](
    q: NDBuffer[type, 4, *_],
    kv_collection: collection_t,
    layer_idx: UInt32,
    mask: NDBuffer[type, *_],
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[mut=True, type, 4, *_],
    context: DeviceContextPtr,
) raises:
    """Performs flash attention using k and v caches from KVCacheT custom types.

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
    if is_gpu[target]():
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
    target: StaticString,
](
    q: NDBuffer[type, 4, *_],
    kv_collection: collection_t,
    layer_idx: UInt32,
    mask: NDBuffer[type, *_],
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[mut=True, type, 4, *_],
    context: Optional[DeviceContext],
) raises:
    """Performs flash attention using k and v caches from KVCacheT custom custom types.

    Args:
        q: NDBuffer with shape (batch_size, num_heads, seq_len, head_size).
        kv_collection: The Collection object storing out KVCache entries for this layer
        layer_idx: The current layer, used to retrieve kv_cache objects from kv_colleciton
        mask: The attention mask to apply to the score matrix.
        valid_lengths: The unpadded lengths of the sequences contained in q
        scale: The scaled factor in scaled-dot product attention. Usually isqrt(head_size).
        output: The Pre-allocated output buffer to write results to. Has shape:
            (batch_size, num_heads, seq_len, head_size).
        context: CUDA DeviceContext. This is not used if is_cpu[target]()
    """

    var layer_idx_cast = Int(layer_idx)
    var k = kv_collection.get_key_cache(layer_idx_cast)
    var v = kv_collection.get_value_cache(layer_idx_cast)

    @parameter
    if is_cpu[target]():
        return flash_attention_kv_cache_cpu(q, k, v, mask, scale, output)
    else:
        return _flash_attention_kv_cache_gpu[target=target](
            q, k, v, mask, valid_lengths, scale, output, context.value()
        )


@always_inline
fn generic_flash_attention_kv_cache_causal_mask_continuous_batch[
    target: StaticString, type: DType
](
    q: NDBuffer[type, 4, *_],
    kv_collection: ContinuousBatchingKVCacheCollection,
    layer_idx: UInt32,
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[mut=True, type, 4, *_],
    context: DeviceContextPtr,
) raises:
    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            trace_arg("q", q),
            trace_arg("valid_lengths", valid_lengths),
            "scale=" + String(scale),
            "layer_idx=" + String(layer_idx),
            "num_heads=" + String(kv_collection.kv_params.num_heads),
            "head_size=" + String(kv_collection.kv_params.head_size),
        )

    with Trace[TraceLevel.OP, target=target](
        "mo.mha.padded.continuous_batching.causal_mask.no_pos.nhead_"
        + String(kv_collection.kv_params.num_heads)
        + ".hdim_"
        + String(kv_collection.kv_params.head_size),
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
    ):
        return _flash_attention_kv_cache_causal_mask[
            kv_collection.CacheType, target=target
        ](q, kv_collection, layer_idx, valid_lengths, scale, output, context)


@always_inline
fn _flash_attention_kv_cache_causal_mask[
    type: DType,
    collection_t: KVCollectionT, //,
    cache_t: KVCacheT,
    target: StaticString,
](
    q: NDBuffer[type, 4, *_],
    kv_collection: collection_t,
    layer_idx: UInt32,
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[mut=True, type, 4, *_],
    context: DeviceContextPtr,
) raises:
    """Performs flash attention using k and v caches from KVCacheT custom types, with the causal mask materialized inside the kernel.

    Args:
        q: NDBuffer with shape (batch_size, num_heads, seq_len, head_size).
        kv_collection: The Collection object storing out KVCache entries for this layer
        layer_idx: The current layer, used to retrieve kv_cache objects from kv_colleciton
        valid_lengths: The unpadded lengths of the sequences contained in q
        scale: The scaled factor in scaled-dot product attention. Usually isqrt(head_size).
        output: The Pre-allocated output buffer to write results to. Has shape:
            (batch_size, num_heads, seq_len, head_size).
        context: Pointer containing the runtime context for the target device.
    """
    var cuda_ctx: Optional[DeviceContext] = None

    @parameter
    if is_gpu[target]():
        cuda_ctx = context.get_device_context()

    _flash_attention_kv_cache_causal_mask_impl[cache_t, target=target](
        q, kv_collection, layer_idx, valid_lengths, scale, output, cuda_ctx
    )


@always_inline
fn _flash_attention_kv_cache_causal_mask_impl[
    type: DType,
    collection_t: KVCollectionT, //,
    cache_t: KVCacheT,
    target: StaticString,
](
    q: NDBuffer[type, 4, *_],
    kv_collection: collection_t,
    layer_idx: UInt32,
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[mut=True, type, 4, *_],
    context: Optional[DeviceContext],
) raises:
    """Performs flash attention using k and v caches from KVCacheT custom types, with the causal mask materialized inside the kernel.

    Args:
        q: NDBuffer with shape (batch_size, num_heads, seq_len, head_size).
        kv_collection: The Collection object storing out KVCache entries for this layer
        layer_idx: The current layer, used to retrieve kv_cache objects from kv_colleciton
        valid_lengths: The unpadded lengths of the sequences contained in q
        scale: The scaled factor in scaled-dot product attention. Usually isqrt(head_size).
        output: The Pre-allocated output buffer to write results to. Has shape:
            (batch_size, num_heads, seq_len, head_size).
        context: CUDA DeviceContext. This is not used if is_cpu[target]()
    """
    var layer_idx_cast = Int(layer_idx)
    var k = kv_collection.get_key_cache(layer_idx_cast)
    var v = kv_collection.get_value_cache(layer_idx_cast)

    @parameter
    if is_cpu[target]():
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
    type: DType, cache_t: KVCacheT, //, *, target: StaticString
](
    q: NDBuffer[type, 4, *_],
    k: cache_t,
    v: cache_t,
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[mut=True, type, 4, *_],
    context: DeviceContext,
) raises:
    # GPU flash attention kernel gets the cache length from the k tensor shape
    # TODO remove this and instead pass in explicit KVCache lengths to the GPU kernel.
    # KERN-725
    gpu_flash_attention(
        output,
        q,
        k,
        v,
        CausalMask(),
        IdentityScoreMod(),
        valid_lengths,
        scale,
        context,
    )


@always_inline
fn _flash_attention_kv_cache_gpu[
    type: DType, cache_t: KVCacheT, //, *, target: StaticString
](
    q: NDBuffer[type, 4, *_],
    k: cache_t,
    v: cache_t,
    mask: NDBuffer[type, *_],
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[mut=True, type, 4, *_],
    context: DeviceContext,
) raises:
    alias wrapped_mask_rank = mask.rank if mask.rank == 4 else 3
    var mask_nd: NDBuffer[
        type,
        wrapped_mask_rank,
        MutableAnyOrigin,
        DimList.create_unknown[wrapped_mask_rank](),
    ]

    @parameter
    if mask.rank == 2:
        mask_nd = NDBuffer[
            type,
            wrapped_mask_rank,
            MutableAnyOrigin,
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
                MutableAnyOrigin,
                DimList.create_unknown[wrapped_mask_rank](),
            ]
        ](mask)

    # GPU flash attention kernel gets the cache length from the k tensor shape
    # TODO remove this an instead pass in explicit KVCache lengths to the GPU kernel.
    # KERN-725
    gpu_flash_attention(
        output,
        q,
        k,
        v,
        MaterializedMask(mask_nd, start_pos=k.cache_lengths_nd()),
        IdentityScoreMod(),
        valid_lengths,
        scale,
        context,
    )


@always_inline
fn generic_flash_attention_kv_cache_causal_alibi_mask_continuous_batch[
    target: StaticString, type: DType
](
    q: NDBuffer[type, 4, *_],
    kv_collection: ContinuousBatchingKVCacheCollection,
    layer_idx: UInt32,
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[mut=True, type, 4, *_],
    context: DeviceContextPtr,
) raises:
    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            trace_arg("q", q),
            trace_arg("valid_lengths", valid_lengths),
            "scale=" + String(scale),
            "layer_idx=" + String(layer_idx),
            "num_heads=" + String(kv_collection.kv_params.num_heads),
            "head_size=" + String(kv_collection.kv_params.head_size),
        )

    with Trace[TraceLevel.OP, target=target](
        "mo.mha.padded.continuous_batching.causal_mask.alibi_pos.nhead_"
        + String(kv_collection.kv_params.num_heads)
        + ".hdim_"
        + String(kv_collection.kv_params.head_size),
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
    ):
        return _flash_attention_kv_cache_causal_alibi_mask[
            kv_collection.CacheType, target=target
        ](q, kv_collection, layer_idx, valid_lengths, scale, output, context)


@always_inline
fn _flash_attention_kv_cache_causal_alibi_mask[
    type: DType,
    collection_t: KVCollectionT, //,
    cache_t: KVCacheT,
    target: StaticString,
](
    q: NDBuffer[type, 4, *_],
    kv_collection: collection_t,
    layer_idx: UInt32,
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[mut=True, type, 4, *_],
    context: DeviceContextPtr,
) raises:
    """Performs flash attention using k and v caches from KVCacheT
    custom types, with the causal mask and alibi mask materialized inside the kernel.

    Args:
        q: NDBuffer with shape (batch_size, num_heads, seq_len, head_size).
        kv_collection: The Collection object storing out KVCache entries for this layer
        layer_idx: The current layer, used to retrieve kv_cache objects from kv_colleciton
        valid_lengths: The unpadded lengths of the sequences contained in q
        scale: The scaled factor in scaled-dot product attention. Usually isqrt(head_size).
        output: The Pre-allocated output buffer to write results to. Has shape:
            (batch_size, num_heads, seq_len, head_size).
        context: Pointer containing the runtime context for the target device.
    """
    var cuda_ctx: Optional[DeviceContext] = None

    @parameter
    if is_gpu[target]():
        cuda_ctx = context.get_device_context()

    _flash_attention_kv_cache_causal_alibi_mask_impl[cache_t, target=target](
        q, kv_collection, layer_idx, valid_lengths, scale, output, cuda_ctx
    )


@always_inline
fn _flash_attention_kv_cache_causal_alibi_mask_impl[
    type: DType,
    collection_t: KVCollectionT, //,
    cache_t: KVCacheT,
    target: StaticString,
](
    q: NDBuffer[type, 4, *_],
    kv_collection: collection_t,
    layer_idx: UInt32,
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[mut=True, type, 4, *_],
    context: Optional[DeviceContext],
) raises:
    """Performs flash attention using k and v caches from KVCacheT
    custom types, with the causal mask and alibi mask materialized inside the kernel.

    Args:
        q: NDBuffer with shape (batch_size, num_heads, seq_len, head_size).
        kv_collection: The Collection object storing out KVCache entries for this layer
        layer_idx: The current layer, used to retrieve kv_cache objects from kv_colleciton
        valid_lengths: The unpadded lengths of the sequences contained in q
        scale: The scaled factor in scaled-dot product attention. Usually isqrt(head_size).
        output: The Pre-allocated output buffer to write results to. Has shape:
            (batch_size, num_heads, seq_len, head_size).
        context: CUDA DeviceContext. This is not used if is_cpu[target]()
    """
    var layer_idx_cast = Int(layer_idx)
    var k = kv_collection.get_key_cache(layer_idx_cast)
    var v = kv_collection.get_value_cache(layer_idx_cast)

    @parameter
    if is_cpu[target]():
        return flash_attention_kv_cache_cpu(
            q, k, v, CausalMask(), scale, output
        )
    else:
        return _flash_attention_kv_cache_causal_alibi_mask_gpu[target=target](
            q, k, v, valid_lengths, scale, output, context.value()
        )


# TODO: Change this as needed when plumbed with pipelines.
#       This is a copy of _flash_attention_kv_cache_gpu with the difference that
#       it calls gpu_flash_attention with the option to use mask tensor and
#       passing CausalMask() and AlibiScoreMod().
@always_inline
fn _flash_attention_kv_cache_causal_alibi_mask_gpu[
    type: DType, cache_t: KVCacheT, //, *, target: StaticString
](
    q: NDBuffer[type, 4, *_],
    k: cache_t,
    v: cache_t,
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[mut=True, type, 4, *_],
    context: DeviceContext,
) raises:
    var mask_nd = NDBuffer[
        type, 4, MutableAnyOrigin, DimList.create_unknown[4]()
    ](UnsafePointer[Scalar[type]](), IndexList[4]())

    # This assumes that, the q tensor is static in the 1 dim.
    alias num_q_heads = Int(q.shape.at[1]())

    # GPU flash attention kernel gets the cache length from the k tensor shape
    # TODO remove this an instead pass in explicit KVCache lengths to the GPU kernel.
    # KERN-725
    gpu_flash_attention[use_score_mod=True](
        output,
        q,
        k,
        v,
        MaterializedMask(mask_nd, start_pos=k.cache_lengths_nd()),
        AlibiScoreMod[num_q_heads](),
        valid_lengths,
        scale,
        context,
    )


# ===-----------------------------------------------------------------------===#
# RMSNorm
# ===-----------------------------------------------------------------------===#


def rms_norm_kv_cache_ragged_continuous_batching[
    type: DType, num_heads: Int, head_dim: Int, //, target: StaticString
](
    kv_collection: ContinuousBatchingKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
    ],
    gamma: NDBuffer[type, 1, *_],
    epsilon: Scalar[type],
    layer_idx: UInt32,
    total_seq_len: UInt32,
    input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    context: DeviceContextPtr,
):
    """Performs RMSNorm in place on new entries in the key cache.

    This is done by first creating the ragged tensor weight_shape
    (total_seq_len, num_heads, head_dim) of the new token tensor.
    To do this we need to pass in `total_seq_len` on host.
    Then, using `input_row_offsets` we find the corresponding batch and token
    index, and use that together with the static head and channel indices to
    store to/load from the key cache.
    This uses the input/output lambdas on the RMSNorm kernel.

    This function could apply RMSNorm to a subset of dimensions in each head,
    determined by the size of the gamma tensor. In this case, it operates on a
    ragged tensor view of the key cache with shape (total_seq_len, num_heads,
    rms_norm_cols), where rms_norm_cols is the length of gamma and must be <=
    head_size.
    """
    # Rank of ragged tensors of shape (total_seq_len, num_heads, head_dim).
    alias rank = 3
    var k_cache = kv_collection.get_key_cache(Int(layer_idx))
    var kv_params = k_cache.kv_params
    alias rms_norm_cols = gamma.shape.get[0]()

    constrained[gamma.shape.has_value[0](), "Need static shape for gamma"]()
    constrained[
        rms_norm_cols <= kv_collection.kv_params.head_size,
        "Size of gamma must be smaller or equal to head size",
    ]()

    var shape = IndexList[rank](
        Int(total_seq_len), kv_params.num_heads, rms_norm_cols
    )

    @always_inline
    @parameter
    @__copy_capture(k_cache, input_row_offsets)
    fn key_cache_input_fn[
        width: Int, rank_: Int
    ](idx: IndexList[rank_]) -> SIMD[type, width]:
        constrained[
            rank_ == rank,
            "rms_norm_key_cache input lambda index should have rank 3",
        ]()

        var global_token_idx = idx[0]
        var batch_idx = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )
        var token_idx = Int(global_token_idx - input_row_offsets[batch_idx])

        var cache_length = k_cache.cache_length(batch_idx)
        var cache_token_idx = token_idx + cache_length

        return k_cache.load[width=width](
            bs=batch_idx,
            tok_idx=cache_token_idx,
            head_idx=idx[1],
            head_dim_idx=idx[2],
        )

    @always_inline
    @parameter
    @__copy_capture(k_cache)
    fn key_cache_output_fn[
        width: Int
    ](idx: IndexList[rank], val: SIMD[type, width]) -> None:
        var global_token_idx = idx[0]
        var batch_idx = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )
        var token_idx = Int(global_token_idx - input_row_offsets[batch_idx])

        var cache_length = k_cache.cache_length(batch_idx)
        var cache_token_idx = token_idx + cache_length

        k_cache.store(
            bs=batch_idx,
            tok_idx=cache_token_idx,
            head_idx=idx[1],
            head_dim_idx=idx[2],
            val=val,
        )

    with Trace[TraceLevel.OP](
        "rms_norm_kv_cache_ragged_continuous_batching_nhead_"
        + String(kv_collection.kv_params.num_heads)
        + ".hdim_"
        + String(kv_collection.kv_params.head_size),
    ):
        _rms_norm_impl[
            type, rank, key_cache_input_fn, key_cache_output_fn, target=target
        ](shape, gamma, epsilon, context)


def rms_norm_kv_cache_ragged_paged[
    type: DType, num_heads: Int, head_dim: Int, //, target: StaticString
](
    kv_collection: PagedKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
    ],
    gamma: NDBuffer[type, 1, *_],
    epsilon: Scalar[type],
    layer_idx: UInt32,
    total_seq_len: UInt32,
    input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    context: DeviceContextPtr,
):
    """Performs RMSNorm in place on new entries in the key cache.

    This is done by first creating the ragged tensor weight_shape
    (total_seq_len, num_heads, head_dim) of the new token tensor.
    To do this we need to pass in `total_seq_len` on host.
    Then, using `input_row_offsets` we find the corresponding batch and token
    index, and use that together with the static head and channel indices to
    store to/load from the key cache.
    This uses the input/output lambdas on the RMSNorm kernel.

    This function could apply RMSNorm to a subset of dimensions in each head,
    determined by the size of the gamma tensor. In this case, it operates on a
    ragged tensor view of the key cache with shape (total_seq_len, num_heads,
    rms_norm_cols), where rms_norm_cols is the length of gamma and must be <=
    head_size.
    """
    # Rank of ragged tensors of shape (total_seq_len, num_heads, head_dim).
    alias rank = 3
    var k_cache = kv_collection.get_key_cache(Int(layer_idx))
    var kv_params = k_cache.kv_params
    alias rms_norm_cols = gamma.shape.get[0]()

    constrained[gamma.shape.has_value[0](), "Need static shape for gamma"]()
    constrained[
        rms_norm_cols <= kv_collection.kv_params.head_size,
        "Length of gamma must be smaller or equal to head size",
    ]()

    var shape = IndexList[rank](
        Int(total_seq_len), kv_params.num_heads, rms_norm_cols
    )

    @always_inline
    @parameter
    @__copy_capture(k_cache, input_row_offsets)
    fn key_cache_input_fn[
        width: Int, rank_: Int
    ](idx: IndexList[rank_]) -> SIMD[type, width]:
        constrained[
            rank_ == rank,
            "rms_norm_key_cache input lambda index should have rank 3",
        ]()

        var global_token_idx = idx[0]
        var batch_idx = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )
        var token_idx = Int(global_token_idx - input_row_offsets[batch_idx])

        var cache_length = k_cache.cache_length(batch_idx)
        var cache_token_idx = token_idx + cache_length

        return k_cache.load[width=width](
            bs=batch_idx,
            tok_idx=cache_token_idx,
            head_idx=idx[1],
            head_dim_idx=idx[2],
        )

    @always_inline
    @parameter
    @__copy_capture(k_cache)
    fn key_cache_output_fn[
        width: Int
    ](idx: IndexList[rank], val: SIMD[type, width]) -> None:
        var global_token_idx = idx[0]
        var batch_idx = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )
        var token_idx = Int(global_token_idx - input_row_offsets[batch_idx])

        var cache_length = k_cache.cache_length(batch_idx)
        var cache_token_idx = token_idx + cache_length

        k_cache.store(
            bs=batch_idx,
            tok_idx=cache_token_idx,
            head_idx=idx[1],
            head_dim_idx=idx[2],
            val=val,
        )

    with Trace[TraceLevel.OP](
        "rms_norm_kv_cache_ragged_paged_nhead_"
        + String(kv_collection.kv_params.num_heads)
        + ".hdim_"
        + String(kv_collection.kv_params.head_size),
    ):
        _rms_norm_impl[
            type, rank, key_cache_input_fn, key_cache_output_fn, target=target
        ](shape, gamma, epsilon, context)


# ===-----------------------------------------------------------------------===#
# Print KV Cache
# ===-----------------------------------------------------------------------===#


def _print_cache[
    collection_t: KVCollectionT,
    *,
](
    cache: collection_t.CacheType,
    kv_collection: collection_t,
    valid_lengths: NDBuffer[DType.uint32, 1],
    is_print_compact: Bool,
) -> None:
    """Prints a cache buffer, abbreviating output with ellipses."""
    alias kv_params = collection_t.CacheType.kv_params

    # Only abbreviate output when `is_print_compact` is set.
    var num_to_print: Int = 7 if is_print_compact else Int.MAX
    for b_idx in range(valid_lengths.dim[0]()):
        var total_cache_length = Int(
            valid_lengths[b_idx] + cache.cache_length(b_idx)
        )
        for t_idx in range(min(num_to_print, total_cache_length)):
            for h in range(kv_params.num_heads):
                for hd in range(
                    min(
                        num_to_print,
                        Int(kv_params.head_size),
                    )
                ):
                    print(
                        cache.load[width=1](
                            Int(b_idx),
                            Int(h),
                            Int(t_idx),
                            Int(hd),
                        ),
                        end=", ",
                    )
                if kv_params.head_size > num_to_print:
                    print("...", end=", ")
            if total_cache_length > num_to_print:
                print("\n...", end=",")
            print()


def print_kv_cache_cont_batch_generic_cpu[
    target: StaticString, type: DType, kv_params: KVCacheStaticParams
](
    valid_lengths: NDBuffer[DType.uint32, 1],
    kv_collection: ContinuousBatchingKVCacheCollection[type, kv_params],
    layer_idx: UInt32,
    is_print_compact: Bool,
    context: DeviceContextPtr,
):
    var k_cache = kv_collection.get_key_cache(Int(layer_idx))
    var v_cache = kv_collection.get_value_cache(Int(layer_idx))

    print("K:")
    _print_cache[__type_of(kv_collection)](
        k_cache,
        kv_collection,
        valid_lengths,
        is_print_compact,
    )

    print("V:")
    _print_cache[__type_of(kv_collection)](
        v_cache,
        kv_collection,
        valid_lengths,
        is_print_compact,
    )


def print_kv_cache_paged_generic_cpu[
    target: StaticString,
    type: DType,
    kv_params: KVCacheStaticParams,
    page_size: Int,
](
    valid_lengths: NDBuffer[DType.uint32, 1],
    kv_collection: PagedKVCacheCollection[type, kv_params, page_size],
    layer_idx: UInt32,
    is_print_compact: Bool,
    context: DeviceContextPtr,
):
    var k_cache = kv_collection.get_key_cache(Int(layer_idx))
    var v_cache = kv_collection.get_value_cache(Int(layer_idx))

    print("K:")
    _print_cache[__type_of(kv_collection)](
        k_cache,
        kv_collection,
        valid_lengths,
        is_print_compact,
    )

    print("V:")
    _print_cache[__type_of(kv_collection)](
        v_cache,
        kv_collection,
        valid_lengths,
        is_print_compact,
    )


def print_kv_cache_cont_batch_generic_gpu[
    target: StaticString, type: DType, kv_params: KVCacheStaticParams
](
    valid_lengths: NDBuffer[DType.uint32, 1],
    kv_collection: ContinuousBatchingKVCacheCollection[type, kv_params],
    layer_idx: UInt32,
    is_print_compact: Bool,
    context: DeviceContextPtr,
):
    var blocks_ptr = UnsafePointer[Scalar[type]].alloc(
        kv_collection.blocks.num_elements()
    )
    var blocks_host_nd = __type_of(kv_collection.blocks)(
        blocks_ptr, kv_collection.blocks.dynamic_shape
    )
    var dev_ctx = context.get_device_context()
    dev_ctx.enqueue_copy(
        blocks_host_nd.data,
        kv_collection.blocks.data,
        kv_collection.blocks.num_elements(),
    )

    var cache_lengths_ptr = UnsafePointer[UInt32].alloc(
        kv_collection.cache_lengths.num_elements()
    )
    var cache_lengths_host_nd = __type_of(kv_collection.cache_lengths)(
        cache_lengths_ptr, kv_collection.cache_lengths.dynamic_shape
    )
    dev_ctx.enqueue_copy(
        cache_lengths_host_nd.data,
        kv_collection.cache_lengths.data,
        kv_collection.cache_lengths.num_elements(),
    )

    var lookup_table_ptr = UnsafePointer[UInt32].alloc(
        kv_collection.lookup_table.num_elements()
    )
    var lookup_table_host_nd = __type_of(kv_collection.lookup_table)(
        lookup_table_ptr, kv_collection.lookup_table.dynamic_shape
    )
    dev_ctx.enqueue_copy(
        lookup_table_host_nd.data,
        kv_collection.lookup_table.data,
        kv_collection.lookup_table.num_elements(),
    )

    var host_kv_collection = __type_of(kv_collection)(
        blocks_host_nd,
        cache_lengths_host_nd,
        lookup_table_host_nd,
        kv_collection.max_seq_length,
        kv_collection.max_cache_length,
    )

    var valid_lengths_host_ptr = UnsafePointer[UInt32].alloc(
        valid_lengths.num_elements()
    )
    var valid_lengths_host_nd = __type_of(valid_lengths)(
        valid_lengths_host_ptr, valid_lengths.dynamic_shape
    )
    dev_ctx.enqueue_copy(
        valid_lengths_host_nd.data,
        valid_lengths.data,
        valid_lengths.num_elements(),
    )

    var k_cache = host_kv_collection.get_key_cache(Int(layer_idx))
    var v_cache = host_kv_collection.get_value_cache(Int(layer_idx))

    # Bring host buffers in sync with device buffers.
    dev_ctx.synchronize()

    print("K:")
    _print_cache[__type_of(kv_collection)](
        k_cache,
        host_kv_collection,
        valid_lengths_host_nd,
        is_print_compact,
    )

    print("V:")
    _print_cache[__type_of(kv_collection)](
        v_cache,
        host_kv_collection,
        valid_lengths_host_nd,
        is_print_compact,
    )

    blocks_host_nd.data.free()
    cache_lengths_host_nd.data.free()
    lookup_table_host_nd.data.free()
    valid_lengths_host_nd.data.free()


def print_kv_cache_paged_generic_gpu[
    target: StaticString,
    type: DType,
    kv_params: KVCacheStaticParams,
    page_size: Int,
](
    valid_lengths: NDBuffer[DType.uint32, 1],
    kv_collection: PagedKVCacheCollection[type, kv_params, page_size],
    layer_idx: UInt32,
    is_print_compact: Bool,
    context: DeviceContextPtr,
):
    var blocks_ptr = UnsafePointer[Scalar[type]].alloc(
        kv_collection.blocks.num_elements()
    )
    var blocks_host_nd = __type_of(kv_collection.blocks)(
        blocks_ptr, kv_collection.blocks.dynamic_shape
    )
    var dev_ctx = context.get_device_context()
    dev_ctx.enqueue_copy(
        blocks_host_nd.data,
        kv_collection.blocks.data,
        kv_collection.blocks.num_elements(),
    )
    var cache_lengths_ptr = UnsafePointer[UInt32].alloc(
        kv_collection.cache_lengths.num_elements()
    )
    var cache_lengths_host_nd = __type_of(kv_collection.cache_lengths)(
        cache_lengths_ptr, kv_collection.cache_lengths.dynamic_shape
    )
    dev_ctx.enqueue_copy(
        cache_lengths_host_nd.data,
        kv_collection.cache_lengths.data,
        kv_collection.cache_lengths.num_elements(),
    )
    var lookup_table_ptr = UnsafePointer[UInt32].alloc(
        kv_collection.lookup_table.num_elements()
    )
    var lookup_table_host_nd = __type_of(kv_collection.lookup_table)(
        lookup_table_ptr, kv_collection.lookup_table.dynamic_shape
    )
    dev_ctx.enqueue_copy(
        lookup_table_host_nd.data,
        kv_collection.lookup_table.data,
        kv_collection.lookup_table.num_elements(),
    )
    var host_kv_collection = __type_of(kv_collection)(
        blocks_host_nd,
        cache_lengths_host_nd,
        lookup_table_host_nd,
        kv_collection.max_seq_length,
        kv_collection.max_cache_length,
    )
    var valid_lengths_host_ptr = UnsafePointer[UInt32].alloc(
        valid_lengths.num_elements()
    )
    var valid_lengths_host_nd = __type_of(valid_lengths)(
        valid_lengths_host_ptr, valid_lengths.dynamic_shape
    )
    dev_ctx.enqueue_copy(
        valid_lengths_host_nd.data,
        valid_lengths.data,
        valid_lengths.num_elements(),
    )

    var k_cache = host_kv_collection.get_key_cache(Int(layer_idx))
    var v_cache = host_kv_collection.get_value_cache(Int(layer_idx))

    # Bring host buffers in sync with device buffers.
    dev_ctx.synchronize()

    print("K:")
    _print_cache[__type_of(kv_collection)](
        k_cache,
        host_kv_collection,
        valid_lengths_host_nd,
        is_print_compact,
    )

    print("V:")
    _print_cache[__type_of(kv_collection)](
        v_cache,
        host_kv_collection,
        valid_lengths_host_nd,
        is_print_compact,
    )

    blocks_host_nd.data.free()
    cache_lengths_host_nd.data.free()
    lookup_table_host_nd.data.free()
    valid_lengths_host_nd.data.free()


# ===-----------------------------------------------------------------------===#
# KV Collection Constructors (Ctor)
# ===-----------------------------------------------------------------------===#


fn _continuous_batch_kv_cache_collection[
    type: DType, //, kv_params: KVCacheStaticParams
](
    blocks: NDBuffer[type, 6],
    cache_lengths: NDBuffer[DType.uint32, 1],
    lookup_table: NDBuffer[DType.uint32, 1],
    max_lengths: NDBuffer[DType.uint32, 2],
    out result: ContinuousBatchingKVCacheCollection[type, kv_params],
):
    # Marshal NDBuffers into arguments expected by the
    # ContinuousKVCacheCollection constructor.
    return __type_of(result)(
        blocks=blocks,
        cache_lengths=cache_lengths,
        lookup_table=lookup_table,
        max_seq_length=max_lengths[Index(0, 0)],
        max_cache_length=max_lengths[Index(0, 1)],
    )


@always_inline
fn generic_get_continuous_cache[
    type: DType, kv_params: KVCacheStaticParams
](
    blocks: NDBuffer[type, 6],
    cache_lengths: NDBuffer[DType.uint32, 1],
    lookup_table: NDBuffer[DType.uint32, 1],
    max_lengths: NDBuffer[DType.uint32, 2],
) -> ContinuousBatchingKVCacheCollection[
    type,
    kv_params,
]:
    return _continuous_batch_kv_cache_collection[kv_params](
        blocks, cache_lengths, lookup_table, max_lengths
    )


fn generic_get_paged_cache[
    type: DType,
    kv_params: KVCacheStaticParams,
    page_size: Int,
](
    blocks: NDBuffer[type, 6],
    cache_lengths: NDBuffer[DType.uint32, 1],
    lookup_table: NDBuffer[DType.uint32, 2],
    max_lengths: NDBuffer[DType.uint32, 2],
    out result: PagedKVCacheCollection[type, kv_params, page_size],
):
    return __type_of(result)(
        blocks=blocks,
        cache_lengths=cache_lengths,
        lookup_table=lookup_table,
        max_seq_length=max_lengths[Index(0, 0)],
        max_cache_length=max_lengths[Index(0, 1)],
    )
