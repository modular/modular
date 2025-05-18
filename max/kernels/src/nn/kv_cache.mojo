# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
from collections import InlineArray, Optional, OptionalReg
from collections.string import StaticString
from math import gcd, isqrt
from sys.info import _current_target, simdwidthof
from sys.intrinsics import _type_is_eq

from algorithm.functional import elementwise
from buffer import Dim, DimList, NDBuffer
from compiler_internal import StaticTensorSpec
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
from layout.layout import Layout
from layout.layout_tensor import LayoutTensor
from linalg.matmul import elementwise_epilogue_type, matmul
from memory import UnsafePointer, memcpy
from nn._ragged_utils import get_batch_from_row_offsets
from nn.flash_attention import (
    flash_attention_kv_cache as flash_attention_kv_cache_cpu,
)
from nn.fused_qk_rope import fused_qk_rope
from nn.mha import flash_attention as gpu_flash_attention
from nn.mha_mask import MaterializedMask, MHAMask
from nn.mha_score_mod import IdentityScoreMod, ScoreModTrait
from nn.mha_utils import (
    dispatch_mask_and_score_mod,
    dispatch_materialized_mask_and_score_mod,
)
from nn.normalization import _rms_norm_impl
from register import register_internal
from runtime.asyncrt import DeviceContextPtr
from runtime.tracing import Trace, TraceLevel, trace_arg
from tensor_internal import ManagedTensorSlice, trace_slice_arg

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
    @always_inline
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
fn generic_flash_attention_kv_cache_padded[
    collection_t: KVCollectionT,
    type: DType, //,
    *,
    target: StaticString,
    mask_str: StaticString,
    score_mod_str: StaticString,
    local_window_size: Int = -1,
    num_heads: Int = -1,
](
    q: NDBuffer[type, 4, *_],
    kv_collection: collection_t,
    layer_idx: UInt32,
    valid_lengths: ManagedTensorSlice[type = DType.uint32, rank=1],
    scale: Float32,
    output: NDBuffer[mut=True, type, 4, *_],
    context: DeviceContextPtr,
) raises:
    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            trace_arg("q", q),
            trace_slice_arg("valid_lengths", valid_lengths),
            "scale=" + String(scale),
            "layer_idx=" + String(layer_idx),
            "num_heads=" + String(collection_t.kv_params.num_heads),
            "head_size=" + String(collection_t.kv_params.head_size),
        )

    with Trace[TraceLevel.OP, target=target](
        "mo.mha.padded."
        + collection_t.name_str
        + "."
        + mask_str
        + "."
        + score_mod_str
        + ".nhead_"
        + String(collection_t.kv_params.num_heads)
        + ".hdim_"
        + String(collection_t.kv_params.head_size),
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
    ):
        return _flash_attention_dispatch[
            target=target,
            mask_str=mask_str,
            score_mod_str=score_mod_str,
            local_window_size=local_window_size,
        ](
            q,
            kv_collection,
            layer_idx,
            valid_lengths,
            scale,
            output,
            context,
        )


@always_inline
fn generic_flash_attention_kv_cache_padded_materialized_mask[
    collection_t: KVCollectionT,
    type: DType, //,
    *,
    target: StaticString,
    score_mod_str: StaticString,
    local_window_size: Int = -1,
    num_heads: Int = -1,
](
    q: NDBuffer[type, 4, *_],
    kv_collection: collection_t,
    layer_idx: UInt32,
    mask: NDBuffer[type, *_],
    valid_lengths: ManagedTensorSlice[type = DType.uint32, rank=1],
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
            trace_slice_arg("valid_lengths", valid_lengths),
            "scale=" + String(scale),
            "layer_idx=" + String(layer_idx),
            "num_heads=" + String(collection_t.kv_params.num_heads),
            "head_size=" + String(collection_t.kv_params.head_size),
        )

    with Trace[TraceLevel.OP, target=target](
        "mo.mha.padded.continuous_batching.tensor_mask."
        + score_mod_str
        + ".nhead_"
        + String(collection_t.kv_params.num_heads)
        + ".hdim_"
        + String(collection_t.kv_params.head_size),
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
    ):
        return _flash_attention_dispatch_materialized_mask[
            target=target,
            score_mod_str=score_mod_str,
            local_window_size=local_window_size,
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


fn _flash_attention_dispatch[
    type: DType,
    collection_t: KVCollectionT, //,
    *,
    target: StaticString,
    mask_str: StaticString,
    score_mod_str: StaticString,
    local_window_size: Int = -1,
](
    q: NDBuffer[type, 4, *_],
    kv_cache: collection_t,
    layer_idx: UInt32,
    valid_lengths: ManagedTensorSlice[type = DType.uint32, rank=1],
    scale: Float32,
    output: NDBuffer[mut=True, type, 4, *_],
    context: DeviceContextPtr,
) raises:
    var k = kv_cache.get_key_cache(Int(layer_idx))
    var v = kv_cache.get_value_cache(Int(layer_idx))

    @parameter
    @__copy_capture(k, v)
    fn _dispatch_flash_attention[
        mask_t: MHAMask, score_mod_t: ScoreModTrait
    ](mask: mask_t, score_mod: score_mod_t) raises:
        @parameter
        if is_cpu[target]():
            return flash_attention_kv_cache_cpu(q, k, v, mask, scale, output)
        else:
            alias use_score_mod = not _type_is_eq[
                score_mod_t, IdentityScoreMod
            ]()
            gpu_flash_attention[use_score_mod=use_score_mod](
                output,
                q,
                k,
                v,
                mask,
                score_mod,
                valid_lengths,
                scale,
                context.get_device_context(),
            )

    return dispatch_mask_and_score_mod[
        mask_str, score_mod_str, _dispatch_flash_attention
    ]()


fn _flash_attention_dispatch_materialized_mask[
    type: DType,
    collection_t: KVCollectionT, //,
    *,
    target: StaticString,
    score_mod_str: String,
    local_window_size: Int = -1,
](
    q: NDBuffer[type, 4, *_],
    kv_cache: collection_t,
    layer_idx: UInt32,
    mask_nd: NDBuffer[type, *_],
    valid_lengths: ManagedTensorSlice[type = DType.uint32, rank=1],
    scale: Float32,
    output: NDBuffer[mut=True, type, 4, *_],
    context: DeviceContextPtr,
) raises:
    var k = kv_cache.get_key_cache(Int(layer_idx))
    var v = kv_cache.get_value_cache(Int(layer_idx))

    @parameter
    fn _dispatch_flash_attention[
        mask_t: MHAMask, score_mod_t: ScoreModTrait
    ](mask: mask_t, score_mod: score_mod_t) raises:
        @parameter
        if is_cpu[target]():
            return flash_attention_kv_cache_cpu(q, k, v, mask, scale, output)
        else:
            alias use_score_mod = not _type_is_eq[
                score_mod_t, IdentityScoreMod
            ]()
            gpu_flash_attention[use_score_mod=use_score_mod](
                output,
                q,
                k,
                v,
                mask,
                score_mod,
                valid_lengths,
                scale,
                context.get_device_context(),
            )

    return dispatch_materialized_mask_and_score_mod[
        score_mod_str,
        _dispatch_flash_attention,
        collection_t.kv_params.num_heads,
    ](mask_nd)


# ===-----------------------------------------------------------------------===#
# RMSNorm
# ===-----------------------------------------------------------------------===#


def rms_norm_kv_cache_ragged_continuous_batching[
    type: DType,
    num_heads: Int,
    head_dim: Int, //,
    target: StaticString,
    multiply_before_cast: Bool,
](
    kv_collection: ContinuousBatchingKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
    ],
    gamma: NDBuffer[type, 1, *_],
    epsilon: Scalar[type],
    weight_offset: Scalar[type],
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

    `weight_offset` is a constant offset argument added to the learned weights
    at runtime. Here, we don't use any offset, so we pass in a zero scalar.

    `multiply_before_cast` is a boolean parameter that determines whether to
    multiply the normalized values by the gamma tensor before casting to the
    output type or not. We set it to `True` by default.
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
            type,
            rank,
            key_cache_input_fn,
            key_cache_output_fn,
            target=target,
            multiply_before_cast=multiply_before_cast,
        ](shape, gamma, epsilon, weight_offset, context)


def rms_norm_kv_cache_ragged_paged[
    type: DType,
    num_heads: Int,
    head_dim: Int, //,
    target: StaticString,
    multiply_before_cast: Bool,
](
    kv_collection: PagedKVCacheCollection[
        type,
        KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
    ],
    gamma: NDBuffer[type, 1, *_],
    epsilon: Scalar[type],
    weight_offset: Scalar[type],
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

    `weight_offset` is a constant offset argument added to the learned weights
    at runtime. Here, we don't use any offset, so we pass in a zero scalar.

    `multiply_before_cast` is a boolean parameter that determines whether to
    multiply the normalized values by the gamma tensor before casting to the
    output type or not. We set it to `True` by default.
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
            type,
            rank,
            key_cache_input_fn,
            key_cache_output_fn,
            target=target,
            multiply_before_cast=multiply_before_cast,
        ](shape, gamma, epsilon, weight_offset, context)


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
    assert_write_mode: WRITE_MODE = WRITE_MODE_REG,
](
    valid_lengths: NDBuffer[DType.uint32, 1],
    kv_collection: PagedKVCacheCollection[
        type, kv_params, page_size, assert_write_mode
    ],
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
    assert_write_mode: WRITE_MODE = WRITE_MODE_REG,
](
    valid_lengths: NDBuffer[DType.uint32, 1],
    kv_collection: PagedKVCacheCollection[
        type, kv_params, page_size, assert_write_mode
    ],
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
) -> ContinuousBatchingKVCacheCollection[type, kv_params]:
    return _continuous_batch_kv_cache_collection[kv_params](
        blocks, cache_lengths, lookup_table, max_lengths
    )


fn generic_get_paged_cache[
    type: DType, kv_params: KVCacheStaticParams, page_size: Int
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


@always_inline
fn managed_tensor_slice_to_ndbuffer[
    spec: StaticTensorSpec, //
](tensor: ManagedTensorSlice[static_spec=spec]) -> NDBuffer[
    spec.type,
    spec.rank,
    MutableAnyOrigin,
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
        _,
        spec.shape,
        spec.strides,
        alignment = spec.alignment,
        address_space = spec.address_space,
        exclusive = spec.exclusive,
    ](ptr, tensor.shape(), tensor._runtime_strides)
