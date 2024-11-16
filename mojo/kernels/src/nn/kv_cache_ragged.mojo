# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from buffer import NDBuffer, DimList, Dim
from memory import UnsafePointer
from collections import Optional, OptionalReg
from gpu.host import DeviceContext
from kv_cache.types import (
    ContiguousKVCache,
    ContinuousBatchingKVCache,
    ContiguousKVCacheCollection,
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
    KVCacheT,
    KVCollectionT,
)
from linalg.matmul import matmul, elementwise_epilogue_type
from sys.intrinsics import _type_is_eq
from utils.index import IndexList, Index
from runtime.asyncrt import MojoCallContextPtr
from runtime.tracing import Trace, TraceLevel
from register import register_internal
from nn.fused_qk_rope import fused_qk_rope_ragged
from nn.mha import flash_attention as gpu_flash_attention
from nn.mha_mask import CausalMask
from nn.mha_score_mod import IdentityScoreMod
from nn.flash_attention import (
    flash_attention_kv_cache as flash_attention_kv_cache_cpu,
)
from nn._ragged_utils import get_batch_from_row_offsets

# ===----------------------------------------------------------------------===#
# Fused QKV Matmul
# ===----------------------------------------------------------------------===#


@register_internal("fused_qkv_matmul_kv_cache_h8_d128_cont_batch_ragged")
fn fused_qkv_matmul_kv_cache_h8_d128_cont_batch_ragged[
    type: DType, //,
    target: StringLiteral = "cpu",
](
    hidden_state: NDBuffer[type, 2, _],
    input_row_offset: NDBuffer[DType.uint32, 1, *_],
    weight: NDBuffer[type, 2, _],
    kv_collection: ContinuousBatchingKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=8, head_size=128),
    ],
    layer_idx: UInt32,
    output: NDBuffer[type, 2, _],
    ctx: MojoCallContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offset: Tensor with shape (batch_size + 1,).
            The value at each index is the start_idx of the corresponding batch in hidden_state.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The object storing the KVCache for this layer.
        layer_idx: The current layer, used to retrieve the KVCache object from kv_collection.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
            Shape: (sum(seq_lens), num_heads * head_size).
        ctx: The call context pointer, passed by the graph compiler.
    """
    with Trace[TraceLevel.OP, target=target](
        "fused_qkv_matmul_kv_cache_h8_d128_cont_batch_ragged"
    ):
        return _fused_qkv_matmul_kv_cache_ragged[
            kv_collection.CacheType, target=target
        ](
            hidden_state,
            input_row_offset,
            weight,
            kv_collection,
            layer_idx,
            output,
            ctx,
        )


@register_internal("fused_qkv_matmul_kv_cache_h8_d64_cont_batch_ragged")
fn fused_qkv_matmul_kv_cache_h8_d64_cont_batch_ragged[
    type: DType, //,
    target: StringLiteral = "cpu",
](
    hidden_state: NDBuffer[type, 2, _],
    input_row_offset: NDBuffer[DType.uint32, 1, *_],
    weight: NDBuffer[type, 2, _],
    kv_collection: ContinuousBatchingKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=8, head_size=64),
    ],
    layer_idx: UInt32,
    output: NDBuffer[type, 2, _],
    ctx: MojoCallContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offset: Tensor with shape (batch_size + 1,).
            The value at each index is the start_idx of the corresponding batch in hidden_state.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The object storing the KVCache for this layer.
        layer_idx: The current layer, used to retrieve the KVCache object from kv_collection.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
            Shape: (sum(seq_lens), num_heads * head_size).
        ctx: The call context pointer, passed by the graph compiler.
    """
    with Trace[TraceLevel.OP, target=target](
        "fused_qkv_matmul_kv_cache_h8_d64_cont_batch_ragged"
    ):
        return _fused_qkv_matmul_kv_cache_ragged[
            kv_collection.CacheType, target=target
        ](
            hidden_state,
            input_row_offset,
            weight,
            kv_collection,
            layer_idx,
            output,
            ctx,
        )


@register_internal("fused_qkv_matmul_kv_cache_h1_d16_cont_batch_ragged")
fn fused_qkv_matmul_kv_cache_h1_d16_cont_batch_ragged[
    type: DType, //,
    target: StringLiteral = "cpu",
](
    hidden_state: NDBuffer[type, 2, _],
    input_row_offset: NDBuffer[DType.uint32, 1, *_],
    weight: NDBuffer[type, 2, _],
    kv_collection: ContinuousBatchingKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=1, head_size=16),
    ],
    layer_idx: UInt32,
    output: NDBuffer[type, 2, _],
    ctx: MojoCallContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offset: Tensor with shape (batch_size + 1,).
            The value at each index is the start_idx of the corresponding batch in hidden_state.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The object storing the KVCache for this layer.
        layer_idx: The current layer, used to retrieve the KVCache object from kv_collection.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
            Shape: (sum(seq_lens), num_heads * head_size).
        ctx: The call context pointer, passed by the graph compiler.
    """
    with Trace[TraceLevel.OP, target=target](
        "fused_qkv_matmul_kv_cache_h1_d16_cont_batch_ragged"
    ):
        return _fused_qkv_matmul_kv_cache_ragged[
            kv_collection.CacheType, target=target
        ](
            hidden_state,
            input_row_offset,
            weight,
            kv_collection,
            layer_idx,
            output,
            ctx,
        )


@always_inline
fn _fused_qkv_matmul_kv_cache_ragged[
    type: DType,
    collection_t: KVCollectionT, //,
    cache_t: KVCacheT,
    *,
    target: StringLiteral,
](
    hidden_state: NDBuffer[type, 2, _],
    input_row_offset: NDBuffer[DType.uint32, 1, *_],
    weight: NDBuffer[type, 2, _],
    kv_collection: collection_t,
    layer_idx: UInt32,
    output: NDBuffer[type, 2, _],
    context: MojoCallContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        input_row_offset: Tensor with shape (batch_size + 1,).
            The value at each index is the start_idx of the corresponding batch in hidden_state.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The object storing the KVCache for this layer.
        layer_idx: The current layer, used to retrieve the KVCache object from kv_collection.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
        context: The call context pointer, passed by the graph compiler.
    """
    var cuda_ctx: Optional[DeviceContext] = None
    var layer_idx_cast = int(layer_idx)
    var k_cache = kv_collection.get_key_cache[cache_t](layer_idx_cast)
    var v_cache = kv_collection.get_value_cache[cache_t](layer_idx_cast)

    @parameter
    if target != "cpu":
        cuda_ctx = context.get_device_context()

    return _fused_qkv_matmul_kv_cache_ragged_impl[target=target](
        hidden_state,
        input_row_offset,
        weight,
        k_cache,
        v_cache,
        output,
        cuda_ctx,
    )


@always_inline
fn _fused_qkv_matmul_kv_cache_ragged_impl[
    type: DType,
    cache_t: KVCacheT, //,
    *,
    target: StringLiteral,
](
    hidden_state: NDBuffer[type, 2, _],
    input_row_offset: NDBuffer[DType.uint32, 1, *_],
    weight: NDBuffer[type, 2, _],
    k_cache: cache_t,
    v_cache: cache_t,
    output: NDBuffer[type, 2, *_],
    context: Optional[DeviceContext],
) raises:
    """Performs a fused QKV matmul on ragged tensors. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offset: Tensor with shape (batch_size + 1,)
            denoting the start of each sequence along the seq_len dimension.
        weight: Tensor with shape (num_heads * head_size, (num_heads + 2 * num_kv_heads) * head_size).
        k_cache: The historical ContiguousKVCache for keys, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        v_cache: The historical ContiguousKVCache for values, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
            Shape is (sum(seq_lens), num_heads * head_size)
        context: The DeviceContext. This is unused if target == "cpu".
    """
    alias kv_type = cache_t.get_type()
    alias kv_params = cache_t.get_kv_params()
    alias N = weight.shape.get[0]()
    alias K = weight.shape.get[1]()

    constrained[kv_type == type, "Mismatch in type between Q and KV tensors"]()

    var q_dim = output.dim[1]()
    var k_dim = kv_params.head_size * kv_params.num_heads
    var qk_offset = q_dim + k_dim
    var batch_size = input_row_offset.dim[0]() - 1

    @parameter
    @__copy_capture(output, input_row_offset, q_dim, qk_offset, batch_size)
    fn write_to_cache_common[
        type_: DType, width: Int, cache_t_: KVCacheT, *, alignment: Int = 1
    ](
        k_cache: cache_t_,
        v_cache: cache_t_,
        idx: IndexList[2],
        val: SIMD[type_, width],
    ):
        if idx[1] < q_dim:
            output.store[width=width, alignment=alignment](
                idx,
                rebind[SIMD[type, width]](val),
            )
            return

        global_token_idx = idx[0]

        var batch_idx: Int = get_batch_from_row_offsets(
            input_row_offset, global_token_idx
        )

        token_idx = int(global_token_idx - input_row_offset[batch_idx])

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

        var cache_length = cache.cache_length(batch_idx)
        var cache_token_idx = token_idx + cache_length
        cache.store(
            batch_idx,
            h_idx,
            cache_token_idx,
            hd_idx,
            rebind[SIMD[kv_type, width]](output_val),
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
    hidden_state: NDBuffer[type, 2, _],
    weight: NDBuffer[type, 2, _],
    context: Optional[DeviceContext],
) raises:
    var TOTAL_SEQ_LEN = hidden_state.dim[0]()
    alias N = weight.shape.get[0]()
    alias K = weight.shape.get[1]()
    var c_nd: NDBuffer[type, 2, DimList(Dim(), N)]

    @parameter
    if target == "cpu":
        # The CPU matmul codepath uses the C buffer as a workspace
        # even if an epilogue is provided, here we just allocate
        # something to ensure we don't segfault.
        var c_ptr = UnsafePointer[Scalar[type]].alloc(TOTAL_SEQ_LEN * N)

        c_nd = __type_of(c_nd)(
            c_ptr,
            IndexList[2](TOTAL_SEQ_LEN, N),
        )
    else:
        c_nd = __type_of(c_nd)(
            UnsafePointer[Scalar[type]](),
            IndexList[2](TOTAL_SEQ_LEN, N),
        )

    matmul[
        target=target,
        transpose_b=True,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ](c_nd, hidden_state, weight, context)

    @parameter
    if target == "cpu":
        c_nd.data.free()


# ===----------------------------------------------------------------------===#
# Fused QK Rope
# ===----------------------------------------------------------------------===#


@register_internal("fused_qk_rope_h6_d48_bshd_ragged")
fn fused_qk_rope_h6_d48_bshd_ragged[
    type: DType, //,
    *,
    target: StringLiteral,
](
    q_proj: NDBuffer[type, 3, *_],
    input_row_offset: NDBuffer[DType.uint32, 1, *_],
    kv_collection: ContiguousKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=6, head_size=48),
    ],
    freqs_cis: NDBuffer[type, 2, *_],
    layer_idx: UInt32,
    output: NDBuffer[type, 3, *_],
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
        "fused_qk_rope_h6_d48_bshd_ragged"
    ):
        fused_qk_rope_ragged[kv_collection.CacheType, target=target](
            q_proj,
            input_row_offset,
            kv_collection,
            freqs_cis,
            layer_idx,
            output,
            dev_ctx,
        )


@register_internal("fused_qk_rope_h8_d128_bshd_ragged")
fn fused_qk_rope_h8_d128_bshd_ragged[
    type: DType, //,
    *,
    target: StringLiteral,
](
    q_proj: NDBuffer[type, 3, *_],
    input_row_offset: NDBuffer[DType.uint32, 1, *_],
    kv_collection: ContiguousKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=8, head_size=128),
    ],
    freqs_cis: NDBuffer[type, 2, *_],
    layer_idx: UInt32,
    output: NDBuffer[type, 3, *_],
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
        "fused_qk_rope_h8_d128_bshd_ragged"
    ):
        fused_qk_rope_ragged[kv_collection.CacheType, target=target](
            q_proj,
            input_row_offset,
            kv_collection,
            freqs_cis,
            layer_idx,
            output,
            dev_ctx,
        )


@register_internal("fused_qk_rope_h1_d16_bshd_ragged")
fn fused_qk_rope_h1_d16_bshd_ragged[
    type: DType, //,
    *,
    target: StringLiteral,
](
    q_proj: NDBuffer[type, 3, *_],
    input_row_offset: NDBuffer[DType.uint32, 1, *_],
    kv_collection: ContiguousKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=1, head_size=16),
    ],
    freqs_cis: NDBuffer[type, 2, *_],
    layer_idx: UInt32,
    output: NDBuffer[type, 3, *_],
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
        "fused_qk_rope_h1_d16_bshd_ragged"
    ):
        fused_qk_rope_ragged[kv_collection.CacheType, target=target](
            q_proj,
            input_row_offset,
            kv_collection,
            freqs_cis,
            layer_idx,
            output,
            dev_ctx,
        )


@register_internal("fused_qk_rope_h8_d32_bshd_ragged")
fn fused_qk_rope_h8_d32_bshd_ragged[
    type: DType, //,
    *,
    target: StringLiteral,
](
    q_proj: NDBuffer[type, 3, *_],
    input_row_offset: NDBuffer[DType.uint32, 1, *_],
    kv_collection: ContiguousKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=8, head_size=32),
    ],
    freqs_cis: NDBuffer[type, 2, *_],
    layer_idx: UInt32,
    output: NDBuffer[type, 3, *_],
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
        "fused_qk_rope_h8_d32_bshd_ragged"
    ):
        fused_qk_rope_ragged[kv_collection.CacheType, target=target](
            q_proj,
            input_row_offset,
            kv_collection,
            freqs_cis,
            layer_idx,
            output,
            dev_ctx,
        )


@register_internal("fused_qk_rope_h8_d64_bshd_ragged")
fn fused_qk_rope_h8_d64_bshd_ragged[
    type: DType, //,
    *,
    target: StringLiteral,
](
    q_proj: NDBuffer[type, 3, *_],
    input_row_offset: NDBuffer[DType.uint32, 1, *_],
    kv_collection: ContiguousKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=8, head_size=64),
    ],
    freqs_cis: NDBuffer[type, 2, *_],
    layer_idx: UInt32,
    output: NDBuffer[type, 3, *_],
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
        "fused_qk_rope_h8_d64_bshd_ragged"
    ):
        fused_qk_rope_ragged[kv_collection.CacheType, target=target](
            q_proj,
            input_row_offset,
            kv_collection,
            freqs_cis,
            layer_idx,
            output,
            dev_ctx,
        )


@register_internal("fused_qk_rope_h8_d128_bshd_continuous_batch_ragged")
fn fused_qk_rope_h8_d128_bshd_continuous_batch_ragged[
    type: DType, //,
    *,
    target: StringLiteral,
](
    q_proj: NDBuffer[type, 3, *_],
    input_row_offset: NDBuffer[DType.uint32, 1, *_],
    kv_collection: ContinuousBatchingKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=8, head_size=128),
    ],
    freqs_cis: NDBuffer[type, 2, *_],
    layer_idx: UInt32,
    output: NDBuffer[type, 3, *_],
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
        "fused_qk_rope_h8_d128_bshd_continuous_batch_ragged"
    ):
        fused_qk_rope_ragged[kv_collection.CacheType, target=target](
            q_proj,
            input_row_offset,
            kv_collection,
            freqs_cis,
            layer_idx,
            output,
            dev_ctx,
        )


@register_internal("fused_qk_rope_h1_d16_bshd_continuous_batch_ragged")
fn fused_qk_rope_h1_d16_bshd_continuous_batch_ragged[
    type: DType, //,
    *,
    target: StringLiteral,
](
    q_proj: NDBuffer[type, 3, *_],
    input_row_offset: NDBuffer[DType.uint32, 1, *_],
    kv_collection: ContinuousBatchingKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=1, head_size=16),
    ],
    freqs_cis: NDBuffer[type, 2, *_],
    layer_idx: UInt32,
    output: NDBuffer[type, 3, *_],
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
        "fused_qk_rope_h1_d16_bshd_continuous_batch_ragged"
    ):
        fused_qk_rope_ragged[kv_collection.CacheType, target=target](
            q_proj,
            input_row_offset,
            kv_collection,
            freqs_cis,
            layer_idx,
            output,
            dev_ctx,
        )


@register_internal("fused_qk_rope_h8_d32_bshd_continuous_batch_ragged")
fn fused_qk_rope_h8_d32_bshd_continuous_batch_ragged[
    type: DType, //,
    *,
    target: StringLiteral,
](
    q_proj: NDBuffer[type, 3, *_],
    input_row_offset: NDBuffer[DType.uint32, 1, *_],
    kv_collection: ContinuousBatchingKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=8, head_size=32),
    ],
    freqs_cis: NDBuffer[type, 2, *_],
    layer_idx: UInt32,
    output: NDBuffer[type, 3, *_],
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
        "fused_qk_rope_h8_d32_bshd_continuous_batch_ragged"
    ):
        fused_qk_rope_ragged[kv_collection.CacheType, target=target](
            q_proj,
            input_row_offset,
            kv_collection,
            freqs_cis,
            layer_idx,
            output,
            dev_ctx,
        )


@register_internal("fused_qk_rope_h8_d64_bshd_continuous_batch_ragged")
fn fused_qk_rope_h8_d64_bshd_continuous_batch_ragged[
    type: DType, //,
    *,
    target: StringLiteral,
](
    q_proj: NDBuffer[type, 3, *_],
    input_row_offset: NDBuffer[DType.uint32, 1, *_],
    kv_collection: ContinuousBatchingKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=8, head_size=64),
    ],
    freqs_cis: NDBuffer[type, 2, *_],
    layer_idx: UInt32,
    output: NDBuffer[type, 3, *_],
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
        "fused_qk_rope_h8_d64_bshd_continuous_batch_ragged"
    ):
        fused_qk_rope_ragged[kv_collection.CacheType, target=target](
            q_proj,
            input_row_offset,
            kv_collection,
            freqs_cis,
            layer_idx,
            output,
            dev_ctx,
        )


# ===----------------------------------------------------------------------===#
#   Flash Attention
# ===----------------------------------------------------------------------===#


@register_internal("flash_attention_kv_cache_h1_d16_cont_batch_ragged")
fn flash_attention_kv_cache_h1_d16_cont_batch_ragged[
    type: DType, //,
    target: StringLiteral,
](
    q: NDBuffer[type, 3, *_],
    input_row_offset: NDBuffer[DType.uint32, 1, *_],
    kv_collection: ContinuousBatchingKVCacheCollection[
        type, KVCacheStaticParams(num_heads=1, head_size=16)
    ],
    layer_idx: UInt32,
    scale: Float32,
    output: NDBuffer[type, 3, *_],
    context: MojoCallContextPtr,
) raises:
    with Trace[TraceLevel.OP, target=target](
        "flash_attention_kv_cache_h1_d16_cont_batch_ragged"
    ):
        return _flash_attention_kv_cache_ragged[
            kv_collection.CacheType, target=target
        ](q, input_row_offset, kv_collection, layer_idx, scale, output, context)


@register_internal("flash_attention_kv_cache_h8_d64_cont_batch_ragged")
fn flash_attention_kv_cache_h8_d64_cont_batch_ragged[
    type: DType, //,
    target: StringLiteral,
](
    q: NDBuffer[type, 3, *_],
    input_row_offset: NDBuffer[DType.uint32, 1, *_],
    kv_collection: ContinuousBatchingKVCacheCollection[
        type, KVCacheStaticParams(num_heads=6, head_size=64)
    ],
    layer_idx: UInt32,
    scale: Float32,
    output: NDBuffer[type, 3, *_],
    context: MojoCallContextPtr,
) raises:
    with Trace[TraceLevel.OP, target=target](
        "flash_attention_kv_cache_h8_d64_cont_batch_ragged"
    ):
        return _flash_attention_kv_cache_ragged[
            kv_collection.CacheType, target=target
        ](q, input_row_offset, kv_collection, layer_idx, scale, output, context)


@register_internal("flash_attention_kv_cache_h8_d128_cont_batch_ragged")
fn flash_attention_kv_cache_h8_d128_cont_batch_ragged[
    type: DType, //,
    target: StringLiteral,
](
    q: NDBuffer[type, 3, *_],
    input_row_offset: NDBuffer[DType.uint32, 1, *_],
    kv_collection: ContinuousBatchingKVCacheCollection[
        type, KVCacheStaticParams(num_heads=8, head_size=128)
    ],
    layer_idx: UInt32,
    scale: Float32,
    output: NDBuffer[type, 3, *_],
    context: MojoCallContextPtr,
) raises:
    with Trace[TraceLevel.OP, target=target](
        "flash_attention_kv_cache_h8_d128_cont_batch_ragged"
    ):
        return _flash_attention_kv_cache_ragged[
            kv_collection.CacheType, target=target
        ](q, input_row_offset, kv_collection, layer_idx, scale, output, context)


@always_inline
fn _flash_attention_kv_cache_ragged[
    type: DType,
    collection_t: KVCollectionT, //,
    cache_t: KVCacheT,
    target: StringLiteral,
](
    q: NDBuffer[type, 3, *_],
    input_row_offset: NDBuffer[DType.uint32, 1, *_],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    output: NDBuffer[type, 3, *_],
    context: MojoCallContextPtr,
) raises:
    """Performs flash attention using k and v caches from ContiguousKVCache custom types.

    Args:
        q: NDBuffer with shape (batch_size, num_heads, seq_len, head_size).
        input_row_offset: The start and end position of each entry in the batch.
        kv_collection: The Collection object storing out KVCache entries for this layer
        layer_idx: The current layer, used to retrieve kv_cache objects from kv_colleciton
        scale: The scaled factor in scaled-dot product attention. Usually isqrt(head_size).
        output: The Pre-allocated output buffer to write results to. Has shape:
            (batch_size, num_heads, seq_len, head_size).
        context: Pointer containing the runtime context for the target device.
    """
    var cuda_ctx: Optional[DeviceContext] = None

    @parameter
    if target != "cpu":
        cuda_ctx = context.get_device_context()

    _flash_attention_kv_cache_ragged_impl[cache_t, target=target](
        q,
        input_row_offset,
        kv_collection,
        layer_idx,
        scale,
        output,
        cuda_ctx,
    )


@always_inline
fn _flash_attention_kv_cache_ragged_impl[
    type: DType,
    collection_t: KVCollectionT, //,
    cache_t: KVCacheT,
    target: StringLiteral,
](
    q: NDBuffer[type, 3, *_],
    input_row_offset: NDBuffer[DType.uint32, 1, *_],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    output: NDBuffer[type, 3, *_],
    context: Optional[DeviceContext],
) raises:
    """Performs flash attention using k and v caches from ContiguousKVCache custom types.

    Args:
        q: NDBuffer with shape (sum(seq_lens in batch), num_heads, head_size).
        input_row_offset: The start and end position of each entry in the batch.
        kv_collection: The Collection object storing out KVCache entries for this layer
        layer_idx: The current layer, used to retrieve kv_cache objects from kv_colleciton
        scale: The scaled factor in scaled-dot product attention. Usually isqrt(head_size).
        output: The Pre-allocated output buffer to write results to. Has shape:
            (sum(seq_lens in batch), num_heads, head_size).
        context: CUDA DeviceContext. This is not used if target == "cpu"
    """

    var layer_idx_cast = int(layer_idx)
    var k = kv_collection.get_key_cache[cache_t](layer_idx_cast)
    var v = kv_collection.get_value_cache[cache_t](layer_idx_cast)

    @parameter
    if target == "cpu":
        return flash_attention_kv_cache_cpu(
            q, input_row_offset, k, v, CausalMask(), scale, output
        )
    else:
        return _flash_attention_kv_cache_ragged_gpu[target=target](
            q, input_row_offset, k, v, scale, output, context.value()
        )


@always_inline
fn _flash_attention_kv_cache_ragged_gpu[
    type: DType, cache_t: KVCacheT, //, *, target: StringLiteral
](
    q: NDBuffer[type, 3, *_],
    input_row_offset: NDBuffer[DType.uint32, 1, *_],
    k: cache_t,
    v: cache_t,
    scale: Float32,
    output: NDBuffer[type, 3, *_],
    context: DeviceContext,
) raises:
    var dummy_mask = NDBuffer[
        type,
        4,
        DimList.create_unknown[4](),
    ]()

    gpu_flash_attention[add_attn_mask=False, ragged=True](
        output,
        q,
        k,
        v,
        dummy_mask,
        CausalMask(),
        IdentityScoreMod(),
        input_row_offset,
        scale,
        context,
    )
