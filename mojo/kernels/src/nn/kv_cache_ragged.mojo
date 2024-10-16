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
    KVCacheStaticParams,
    KVCacheT,
)
from linalg.matmul import _matmul_cpu, elementwise_epilogue_type
from linalg.matmul_gpu import _matmul_gpu
from sys.intrinsics import _type_is_eq
from utils.index import IndexList, Index
from runtime.asyncrt import MojoCallContextPtr
from runtime.tracing import Trace, TraceLevel
from register import mogg_register


@mogg_register("fused_qkv_matmul_kv_cache_h8_d128_cont_batch_packed")
@export
fn fused_qkv_matmul_kv_cache_h8_d128_cont_batch_packed[
    type: DType, //,
    target: StringLiteral = "cpu",
](
    hidden_state: NDBuffer[type, 2, _],
    weight: NDBuffer[type, 2, _],
    prefix_sum: NDBuffer[DType.uint32, 1, *_],
    k_cache: ContinuousBatchingKVCache[
        type,
        KVCacheStaticParams(num_heads=8, head_size=128),
    ],
    v_cache: ContinuousBatchingKVCache[type, k_cache.kv_params],
    output: NDBuffer[type, 2, _],
    ctx: MojoCallContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        prefix_sum: Tensor with shape (batch_size + 1,).
            The value at each index is the start_idx of the corresponding batch in hidden_state.
        k_cache: The historical ContinuousBatchingKVCache for keys, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        v_cache: The historical ContinuousBatchingKVCache for values, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
            Shape: (sum(seq_lens), num_heads * head_size).
        ctx: The call context pointer, passed by the graph compiler.
    """
    with Trace[TraceLevel.OP, target=target](
        "fused_qkv_matmul_kv_cache_h8_d128_cont_batch_packed"
    ):
        return _fused_qkv_matmul_kv_cache_packed[target=target](
            hidden_state, weight, prefix_sum, k_cache, v_cache, output, ctx
        )


@mogg_register("fused_qkv_matmul_kv_cache_h8_d64_cont_batch_packed")
@export
fn fused_qkv_matmul_kv_cache_h8_d64_cont_batch_packed[
    type: DType, //,
    target: StringLiteral = "cpu",
](
    hidden_state: NDBuffer[type, 2, _],
    weight: NDBuffer[type, 2, _],
    prefix_sum: NDBuffer[DType.uint32, 1, *_],
    k_cache: ContinuousBatchingKVCache[
        type,
        KVCacheStaticParams(num_heads=8, head_size=64),
    ],
    v_cache: ContinuousBatchingKVCache[type, k_cache.kv_params],
    output: NDBuffer[type, 2, _],
    ctx: MojoCallContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        prefix_sum: Tensor with shape (batch_size + 1,).
            The value at each index is the start_idx of the corresponding batch in hidden_state.
        k_cache: The historical ContinuousBatchingKVCache for keys, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        v_cache: The historical ContinuousBatchingKVCache for values, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
            Shape: (sum(seq_lens), num_heads * head_size).
        ctx: The call context pointer, passed by the graph compiler.
    """
    with Trace[TraceLevel.OP, target=target](
        "fused_qkv_matmul_kv_cache_h8_d64_cont_batch_packed"
    ):
        return _fused_qkv_matmul_kv_cache_packed[target=target](
            hidden_state, weight, prefix_sum, k_cache, v_cache, output, ctx
        )


@mogg_register("fused_qkv_matmul_kv_cache_h1_d16_cont_batch_packed")
@export
fn fused_qkv_matmul_kv_cache_h1_d16_cont_batch_packed[
    type: DType, //,
    target: StringLiteral = "cpu",
](
    hidden_state: NDBuffer[type, 2, _],
    weight: NDBuffer[type, 2, _],
    prefix_sum: NDBuffer[DType.uint32, 1, *_],
    k_cache: ContinuousBatchingKVCache[
        type,
        KVCacheStaticParams(num_heads=1, head_size=16),
    ],
    v_cache: ContinuousBatchingKVCache[type, k_cache.kv_params],
    output: NDBuffer[type, 2, _],
    ctx: MojoCallContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        prefix_sum: Tensor with shape (batch_size + 1,).
            The value at each index is the start_idx of the corresponding batch in hidden_state.
        k_cache: The historical ContinuousBatchingKVCache for keys, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        v_cache: The historical ContinuousBatchingKVCache for values, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
            Shape: (sum(seq_lens), num_heads * head_size).
        ctx: The call context pointer, passed by the graph compiler.
    """
    with Trace[TraceLevel.OP, target=target](
        "fused_qkv_matmul_kv_cache_h1_d16_cont_batch_packed"
    ):
        return _fused_qkv_matmul_kv_cache_packed[target=target](
            hidden_state, weight, prefix_sum, k_cache, v_cache, output, ctx
        )


@always_inline
fn _fused_qkv_matmul_kv_cache_packed[
    type: DType,
    cache_t: KVCacheT, //,
    *,
    target: StringLiteral,
](
    hidden_state: NDBuffer[type, 2, _],
    weight: NDBuffer[type, 2, _],
    prefix_sum: NDBuffer[DType.uint32, 1, *_],
    k_cache: cache_t,
    v_cache: cache_t,
    output: NDBuffer[type, 2, _],
    context: MojoCallContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        prefix_sum: Tensor with shape (batch_size + 1,).
            The value at each index is the start_idx of the corresponding batch in hidden_state.
        k_cache: The historical ContiguousKVCache for keys, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        v_cache: The historical ContiguousKVCache for values, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
        context: The call context pointer, passed by the graph compiler.
    """
    var cuda_ctx: Optional[DeviceContext] = None

    @parameter
    if target != "cpu":
        cuda_ctx = context.get_device_context()

    return _fused_qkv_matmul_kv_cache_packed_impl[target=target](
        hidden_state, weight, prefix_sum, k_cache, v_cache, output, cuda_ctx
    )


@always_inline
fn _fused_qkv_matmul_kv_cache_packed_impl[
    type: DType,
    cache_t: KVCacheT, //,
    *,
    target: StringLiteral,
](
    hidden_state: NDBuffer[type, 2, _],
    weight: NDBuffer[type, 2, _],
    prefix_sum: NDBuffer[DType.uint32, 1, *_],
    k_cache: cache_t,
    v_cache: cache_t,
    output: NDBuffer[type, 2, *_],
    context: Optional[DeviceContext],
) raises:
    """Performs a fused QKV matmul on packed tensors. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, (num_heads + 2 * num_kv_heads) * head_size).
        prefix_sum: Tensor with shape (batch_size + 1,)
            denoting the start of each sequence along the packed dimension.
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
    var batch_size = prefix_sum.dim[0]() - 1

    @parameter
    @__copy_capture(output, prefix_sum, q_dim, qk_offset, batch_size)
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

        # loop through prefix_sum until we get to our sequence
        # NOTE: this is O(N), we could do a binary search, but I'm not sure
        # how that would perform vs a linear search on GPU
        var batch_idx: Int = -1
        for i in range(batch_size):
            if prefix_sum[i + 1] <= global_token_idx:
                continue

            batch_idx = i
            break

        token_idx = int(global_token_idx - prefix_sum[batch_idx])

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

    @parameter
    if target == "cpu":
        var kernel_type_m = hidden_state.shape.at[0]().or_else(0)

        _matmul_cpu[
            transpose_b=True,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ](c_nd, hidden_state, weight, kernel_type_m)
        c_nd.data.free()

    else:
        _matmul_gpu[
            elementwise_lambda_fn=elementwise_lambda_fn,
            use_tensor_core=True,
            transpose_b=True,
            target=target,
        ](c_nd, hidden_state, weight, context.value())
