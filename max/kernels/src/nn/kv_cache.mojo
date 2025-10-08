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
from collections import OptionalReg
from sys.intrinsics import _type_is_eq

from algorithm.functional import unswitch
from buffer import Dim, DimList, NDBuffer
from compiler_internal import StaticTensorSpec
from gpu.host import DeviceContext
from gpu.host.info import is_cpu, is_gpu
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
    KVCacheT,
    KVCollectionT,
    PagedKVCacheCollection,
)
from layout import UNKNOWN_VALUE, Layout, LayoutTensor, RuntimeLayout
from linalg.matmul import elementwise_epilogue_type, matmul
from nn._ragged_utils import get_batch_from_row_offsets
from nn.flash_attention import (
    flash_attention_kv_cache as flash_attention_kv_cache_cpu,
)
from nn.fused_qk_rope import fused_qk_rope
from nn.mha import flash_attention as gpu_flash_attention
from nn.mha_mask import MHAMask
from nn.mha_score_mod import IdentityScoreMod, ScoreModTrait
from nn.mha_utils import (
    dispatch_mask_and_score_mod,
    dispatch_materialized_mask_and_score_mod,
)
from nn.normalization import _rms_norm_impl
from runtime.asyncrt import DeviceContextPtr
from runtime.tracing import Trace, TraceLevel, get_safe_task_id, trace_arg
from tensor_internal import ManagedTensorSlice, trace_slice_arg

from utils import Index, IndexList
from tensor_internal import InputTensor
from tensor_internal.managed_tensor_slice import (
    _MutableInputTensor as MutableInputTensor,
)

# ===-----------------------------------------------------------------------===#
# Fused QKV matmul (padded)
# ===-----------------------------------------------------------------------===#


@always_inline
fn generic_fused_qkv_matmul_kv_cache_bshd_continuous_batch[
    dtype: DType,
    target: StaticString = "cpu",
](
    hidden_state: NDBuffer[dtype, 3, _, _],
    weight: NDBuffer[dtype, 2, _, _],
    kv_collection: ContinuousBatchingKVCacheCollection,
    layer_idx: UInt32,
    output: NDBuffer[mut=True, dtype, 3, _, _],
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
        task_id=get_safe_task_id(ctx),
    ):
        return _fused_qkv_matmul_kv_cache[
            kv_collection.CacheType, target=target
        ](hidden_state, weight, kv_collection, layer_idx, output, ctx)


@always_inline
fn _fused_qkv_matmul_kv_cache[
    dtype: DType,
    collection_t: KVCollectionT, //,
    cache_t: KVCacheT,
    *,
    target: StaticString,
](
    hidden_state: NDBuffer[dtype, 3, _, _],
    weight: NDBuffer[dtype, 2, _, _],
    kv_collection: collection_t,
    layer_idx: UInt32,
    output: NDBuffer[mut=True, dtype, 3, _, _],
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


alias embed_fn_type = fn[dtype: DType, width: Int] (
    IndexList[4], SIMD[dtype, width]
) capturing -> SIMD[dtype, width]


@always_inline
fn _fused_qkv_matmul_kv_cache_impl[
    dtype: DType,
    hidden_state_shape: DimList,
    weight_shape: DimList,
    output_shape: DimList,
    collection_t: KVCollectionT, //,
    *,
    target: StaticString,
    q_embed_fn: OptionalReg[embed_fn_type] = None,
    k_embed_fn: OptionalReg[embed_fn_type] = None,
](
    hidden_state: NDBuffer[dtype, 3, _, hidden_state_shape],
    weight: NDBuffer[dtype, 2, _, weight_shape],
    kv_collection: collection_t,
    layer_idx: UInt32,
    output: NDBuffer[mut=True, dtype, 3, _, output_shape],
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
    alias cache_dtype = cache_t.dtype

    constrained[
        cache_dtype == dtype,
        "Expected cache dtype ",
        String(cache_dtype),
        " to match input dtype ",
        String(dtype),
    ]()

    alias kv_params = cache_t.kv_params
    alias N = weight_shape.get[0]()
    alias K = weight_shape.get[1]()

    var SEQ_LEN: UInt = UInt(hidden_state.dim[1]())

    var q_dim = output.dim[2]()
    var k_dim = kv_params.head_size * kv_params.num_heads
    var qk_offset = q_dim + k_dim

    var k_cache = kv_collection.get_key_cache(Int(layer_idx))
    var v_cache = kv_collection.get_value_cache(Int(layer_idx))

    @parameter
    @__copy_capture(q_dim, qk_offset, SEQ_LEN, k_cache, v_cache)
    @always_inline
    fn write_to_cache[
        dtype_: DType, width: Int, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype_, width]):
        var b_idx, t_idx = divmod(UInt(idx[0]), SEQ_LEN)
        if idx[1] < q_dim:
            output.store[width=width, alignment=alignment](
                Index(Int(b_idx), Int(t_idx), idx[1]),
                rebind[SIMD[dtype, width]](val),
            )
            return

        var h_idx: UInt
        var hd_idx: UInt
        var cache: cache_t
        var output_val = val
        if idx[1] < qk_offset:
            cache = k_cache
            h_idx, hd_idx = divmod(
                UInt(idx[1]) - UInt(q_dim), kv_params.head_size
            )

        else:
            cache = v_cache
            h_idx, hd_idx = divmod(
                UInt(idx[1]) - UInt(qk_offset), kv_params.head_size
            )

        var valid_len = cache.cache_length(b_idx)
        var cache_t_idx = t_idx + UInt(valid_len)
        cache.store(
            b_idx,
            h_idx,
            cache_t_idx,
            hd_idx,
            rebind[SIMD[cache_dtype, width]](output_val),
        )

    _matmul_common[target=target, elementwise_lambda_fn=write_to_cache](
        hidden_state, weight, context
    )


@always_inline
fn _matmul_common[
    dtype: DType, //,
    *,
    target: StaticString,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    hidden_state: NDBuffer[dtype, 3, _, _],
    weight: NDBuffer[dtype, 2, _, _],
    context: Optional[DeviceContext],
) raises:
    var BS = hidden_state.dim[0]()
    var SEQ_LEN = hidden_state.dim[1]()
    alias N = weight.shape.get[0]()
    alias K = weight.shape.get[1]()

    var hidden_state_2d = NDBuffer[
        dtype, 2, MutableAnyOrigin, DimList(Dim(), hidden_state.shape.get[2]())
    ](
        hidden_state.data,
        IndexList[2](BS * SEQ_LEN, K),
    )

    var c_nd: NDBuffer[dtype, 2, MutableAnyOrigin, DimList(Dim(), N)]

    @parameter
    if is_cpu[target]():
        var c_ptr = UnsafePointer[Scalar[dtype]].alloc(BS * SEQ_LEN * N)

        c_nd = NDBuffer[dtype, 2, MutableAnyOrigin, DimList(Dim(), N)](
            c_ptr,
            IndexList[2](BS * SEQ_LEN, N),
        )
    else:
        c_nd = NDBuffer[dtype, 2, MutableAnyOrigin, DimList(Dim(), N)](
            UnsafePointer[Scalar[dtype]](),
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
    dtype: DType, //,
    *,
    interleaved: Bool,
    target: StaticString,
](
    q_proj: NDBuffer[dtype, 4, *_],
    kv_collection: ContinuousBatchingKVCacheCollection,
    freqs_cis: NDBuffer[dtype, 2, *_],
    layer_idx: UInt32,
    output: NDBuffer[mut=True, dtype, 4, *_],
    context: DeviceContextPtr = DeviceContextPtr(),
) raises:
    """Performs a fused RoPE projection for Q and K projections.

    We have a manually fused QKV projection with mo.opaque dtypes in our Llama model.
    Due to a limitation in custom op definitions, we can't declare both a tensor
    and opaque dtype as output from a custom kernel. This requires us to only note
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
        task_id=get_safe_task_id(context),
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
    dtype: DType, //,
    *,
    target: StaticString,
    mask_str: StaticString,
    score_mod_str: StaticString,
    local_window_size: Int = -1,
    num_heads: Int = -1,
](
    q: NDBuffer[dtype, 4, *_],
    kv_collection: collection_t,
    layer_idx: UInt32,
    valid_lengths: ManagedTensorSlice[dtype = DType.uint32, rank=1],
    scale: Float32,
    output: NDBuffer[mut=True, dtype, 4, *_],
    context: DeviceContextPtr,
    sink_weights: OptionalReg[NDBuffer[dtype, 1, MutableAnyOrigin]] = None,
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
        task_id=get_safe_task_id(context),
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
            sink_weights,
        )


@always_inline
fn generic_flash_attention_kv_cache_padded_materialized_mask[
    collection_t: KVCollectionT,
    dtype: DType, //,
    *,
    target: StaticString,
    score_mod_str: StaticString,
    local_window_size: Int = -1,
    num_heads: Int = -1,
](
    q: NDBuffer[dtype, 4, *_],
    kv_collection: collection_t,
    layer_idx: UInt32,
    mask: NDBuffer[dtype, *_],
    valid_lengths: ManagedTensorSlice[dtype = DType.uint32, rank=1],
    scale: Float32,
    output: NDBuffer[mut=True, dtype, 4, *_],
    context: DeviceContextPtr,
    sink_weights: OptionalReg[NDBuffer[dtype, 1, MutableAnyOrigin]] = None,
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
            sink_weights,
        )


fn _flash_attention_dispatch[
    dtype: DType,
    collection_t: KVCollectionT, //,
    *,
    target: StaticString,
    mask_str: StaticString,
    score_mod_str: StaticString,
    local_window_size: Int = -1,
](
    q: NDBuffer[dtype, 4, *_],
    kv_cache: collection_t,
    layer_idx: UInt32,
    valid_lengths: ManagedTensorSlice[dtype = DType.uint32, rank=1],
    scale: Float32,
    output: NDBuffer[mut=True, dtype, 4, *_],
    context: DeviceContextPtr,
    sink_weights: OptionalReg[NDBuffer[dtype, 1, MutableAnyOrigin]] = None,
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
            alias q_layout = Layout.row_major[q.rank](q.shape)
            alias output_layout = Layout.row_major[output.rank](output.shape)
            var sink_weights_lt: OptionalReg[
                LayoutTensor[
                    dtype,
                    Layout.row_major(UNKNOWN_VALUE),
                    MutableAnyOrigin,
                ]
            ] = None
            if sink_weights:
                var sw = sink_weights.value()
                sink_weights_lt = sink_weights_lt.T(
                    sw.data,
                    RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
                        IndexList[1](len(sw))
                    ),
                )
            return flash_attention_kv_cache_cpu(
                LayoutTensor[q.type, q_layout](
                    q.data,
                    RuntimeLayout[q_layout].row_major(
                        q.dynamic_shape.canonicalize()
                    ),
                ),
                k,
                v,
                mask,
                scale,
                LayoutTensor[output.type, output_layout](
                    output.data,
                    RuntimeLayout[output_layout].row_major(
                        output.dynamic_shape.canonicalize()
                    ),
                ),
                sink_weights_lt,
            )
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
    dtype: DType,
    collection_t: KVCollectionT, //,
    *,
    target: StaticString,
    score_mod_str: String,
    local_window_size: Int = -1,
](
    q: NDBuffer[dtype, 4, *_],
    kv_cache: collection_t,
    layer_idx: UInt32,
    mask_nd: NDBuffer[dtype, *_],
    valid_lengths: ManagedTensorSlice[dtype = DType.uint32, rank=1],
    scale: Float32,
    output: NDBuffer[mut=True, dtype, 4, *_],
    context: DeviceContextPtr,
    sink_weights: OptionalReg[NDBuffer[dtype, 1, MutableAnyOrigin]] = None,
) raises:
    var k = kv_cache.get_key_cache(Int(layer_idx))
    var v = kv_cache.get_value_cache(Int(layer_idx))

    @parameter
    fn _dispatch_flash_attention[
        mask_t: MHAMask, score_mod_t: ScoreModTrait
    ](mask: mask_t, score_mod: score_mod_t) raises:
        @always_inline
        @parameter
        fn call_flash_attention[sink: Bool]() raises:
            @parameter
            if is_cpu[target]():
                alias q_layout = Layout.row_major[q.rank](q.shape)
                alias output_layout = Layout.row_major[output.rank](
                    output.shape
                )
                var sink_weights_lt: OptionalReg[
                    LayoutTensor[
                        dtype,
                        Layout.row_major(UNKNOWN_VALUE),
                        MutableAnyOrigin,
                    ]
                ] = None
                if sink_weights:
                    var sw = sink_weights.value()
                    sink_weights_lt = sink_weights_lt.T(
                        sw.data,
                        RuntimeLayout[
                            Layout.row_major(UNKNOWN_VALUE)
                        ].row_major(IndexList[1](len(sw))),
                    )
                return flash_attention_kv_cache_cpu(
                    LayoutTensor[q.type, q_layout](
                        q.data,
                        RuntimeLayout[q_layout].row_major(
                            q.dynamic_shape.canonicalize()
                        ),
                    ),
                    k,
                    v,
                    mask,
                    scale,
                    LayoutTensor[output.type, output_layout](
                        output.data,
                        RuntimeLayout[output_layout].row_major(
                            output.dynamic_shape.canonicalize()
                        ),
                    ),
                    sink_weights_lt,
                )
            else:
                alias use_score_mod = not _type_is_eq[
                    score_mod_t, IdentityScoreMod
                ]()
                gpu_flash_attention[use_score_mod=use_score_mod, sink=sink](
                    output,
                    q,
                    k,
                    v,
                    mask,
                    score_mod,
                    valid_lengths,
                    scale,
                    context.get_device_context(),
                    sink_weights=sink_weights,
                )

        unswitch[call_flash_attention](Bool(sink_weights))

    return dispatch_materialized_mask_and_score_mod[
        score_mod_str,
        _dispatch_flash_attention,
        collection_t.kv_params.num_heads,
    ](
        LayoutTensor[
            mask_nd.type,
            Layout.row_major[mask_nd.rank](mask_nd.shape),
            MutableAnyOrigin,
        ](
            mask_nd.data,
            RuntimeLayout[
                Layout.row_major[mask_nd.rank](mask_nd.shape)
            ].row_major(mask_nd.dynamic_shape.canonicalize()),
        )
    )


# ===-----------------------------------------------------------------------===#
# RMSNorm
# ===-----------------------------------------------------------------------===#


def rms_norm_kv_cache_ragged_continuous_batching[
    dtype: DType,
    num_heads: Int,
    head_dim: Int, //,
    target: StaticString,
    multiply_before_cast: Bool,
    per_head_norm: Bool,
](
    kv_collection: ContinuousBatchingKVCacheCollection[
        dtype,
        KVCacheStaticParams(
            num_heads=UInt(num_heads), head_size=UInt(head_dim)
        ),
    ],
    gamma: NDBuffer[dtype, 1, *_],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
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
    output dtype or not. We set it to `True` by default.
    """
    # Rank of ragged tensors of shape (total_seq_len, num_heads, head_dim).
    alias rank = 3 if per_head_norm else 2
    var k_cache = kv_collection.get_key_cache(Int(layer_idx))
    var kv_params = k_cache.kv_params
    alias rms_norm_cols = gamma.shape.get[0]()

    constrained[gamma.shape.has_value[0](), "Need static shape for gamma"]()
    constrained[
        rms_norm_cols <= kv_collection.kv_params.head_size or not per_head_norm,
        "Length of gamma must be smaller or equal to head size",
    ]()

    var shape = IndexList[rank]()
    shape[0] = Int(total_seq_len)

    @parameter
    if per_head_norm:
        shape[1] = Int(kv_params.num_heads)
        shape[2] = Int(rms_norm_cols)
    else:
        shape[1] = Int(rms_norm_cols)

    @always_inline
    @parameter
    @__copy_capture(k_cache, input_row_offsets)
    fn key_cache_input_fn[
        width: Int, rank_: Int
    ](idx: IndexList[rank_]) -> SIMD[dtype, width]:
        constrained[
            rank_ == rank,
            "rms_norm_key_cache input lambda index should have rank "
            + String(rank),
        ]()

        var global_token_idx = idx[0]
        var batch_idx = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )
        var token_idx = Int(global_token_idx - input_row_offsets[batch_idx])

        var cache_length = k_cache.cache_length(batch_idx)
        var cache_token_idx = token_idx + cache_length

        var head_idx: Int
        var head_dim_idx: Int

        @parameter
        if per_head_norm:
            head_idx = idx[1]
            head_dim_idx = idx[2]
        else:
            head_idx = idx[1] // head_dim
            head_dim_idx = idx[1] % head_dim

        return k_cache.load[width=width](
            bs=batch_idx,
            tok_idx=cache_token_idx,
            head_idx=head_idx,
            head_dim_idx=head_dim_idx,
        )

    @always_inline
    @parameter
    @__copy_capture(k_cache)
    fn key_cache_output_fn[
        width: Int, alignment: Int
    ](idx: IndexList[rank], val: SIMD[dtype, width]) -> None:
        var global_token_idx = idx[0]
        var batch_idx = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )
        var token_idx = Int(global_token_idx - input_row_offsets[batch_idx])

        var cache_length = k_cache.cache_length(batch_idx)
        var cache_token_idx = token_idx + cache_length

        var head_idx: Int
        var head_dim_idx: Int

        @parameter
        if per_head_norm:
            head_idx = idx[1]
            head_dim_idx = idx[2]
        else:
            head_idx = idx[1] // head_dim
            head_dim_idx = idx[1] % head_dim

        k_cache.store(
            bs=batch_idx,
            tok_idx=cache_token_idx,
            head_idx=head_idx,
            head_dim_idx=head_dim_idx,
            val=val,
        )

    with Trace[TraceLevel.OP, target=target](
        "rms_norm_kv_cache_ragged_continuous_batching_nhead_"
        + String(kv_collection.kv_params.num_heads)
        + ".hdim_"
        + String(kv_collection.kv_params.head_size),
        task_id=get_safe_task_id(context),
    ):
        _rms_norm_impl[
            dtype,
            rank,
            key_cache_input_fn,
            key_cache_output_fn,
            target=target,
            multiply_before_cast=multiply_before_cast,
        ](
            shape,
            LayoutTensor[
                gamma.type,
                Layout.row_major(UNKNOWN_VALUE),
                address_space = gamma.address_space,
            ](
                gamma.data,
                RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
                    gamma.dynamic_shape.canonicalize()
                ),
            ),
            epsilon,
            weight_offset,
            context,
        )


def rms_norm_kv_cache_ragged_paged[
    dtype: DType,
    num_heads: Int,
    head_dim: Int, //,
    target: StaticString,
    multiply_before_cast: Bool,
    per_head_norm: Bool,
](
    kv_collection: PagedKVCacheCollection[
        dtype,
        KVCacheStaticParams(
            num_heads=UInt(num_heads), head_size=UInt(head_dim)
        ),
    ],
    gamma: NDBuffer[dtype, 1, *_],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
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
    output dtype or not. We set it to `True` by default.
    """
    # Rank of ragged tensors of shape (total_seq_len, num_heads, head_dim).
    alias rank = 3 if per_head_norm else 2
    var k_cache = kv_collection.get_key_cache(Int(layer_idx))
    var kv_params = k_cache.kv_params
    alias rms_norm_cols = gamma.shape.get[0]()

    constrained[gamma.shape.has_value[0](), "Need static shape for gamma"]()
    constrained[
        rms_norm_cols <= kv_collection.kv_params.head_size or not per_head_norm,
        "Length of gamma must be smaller or equal to head size",
    ]()

    var shape = IndexList[rank]()
    shape[0] = Int(total_seq_len)

    @parameter
    if per_head_norm:
        shape[1] = Int(kv_params.num_heads)
        shape[2] = Int(rms_norm_cols)
    else:
        shape[1] = Int(rms_norm_cols)

    @always_inline
    @parameter
    @__copy_capture(k_cache, input_row_offsets)
    fn key_cache_input_fn[
        width: Int, rank_: Int
    ](idx: IndexList[rank_]) -> SIMD[dtype, width]:
        constrained[
            rank_ == rank,
            "rms_norm_key_cache input lambda index should have rank "
            + String(rank),
        ]()

        var global_token_idx = idx[0]
        var batch_idx = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )
        var token_idx = Int(global_token_idx - input_row_offsets[batch_idx])

        var cache_length = k_cache.cache_length(batch_idx)
        var cache_token_idx = token_idx + cache_length

        var head_idx: Int
        var head_dim_idx: Int

        @parameter
        if per_head_norm:
            head_idx = idx[1]
            head_dim_idx = idx[2]
        else:
            head_idx = idx[1] // head_dim
            head_dim_idx = idx[1] % head_dim

        return k_cache.load[width=width](
            bs=batch_idx,
            tok_idx=cache_token_idx,
            head_idx=head_idx,
            head_dim_idx=head_dim_idx,
        )

    @always_inline
    @parameter
    @__copy_capture(k_cache)
    fn key_cache_output_fn[
        width: Int, alignment: Int
    ](idx: IndexList[rank], val: SIMD[dtype, width]) -> None:
        var global_token_idx = idx[0]
        var batch_idx = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )
        var token_idx = Int(global_token_idx - input_row_offsets[batch_idx])

        var cache_length = k_cache.cache_length(batch_idx)
        var cache_token_idx = token_idx + cache_length

        var head_idx: Int
        var head_dim_idx: Int

        @parameter
        if per_head_norm:
            head_idx = idx[1]
            head_dim_idx = idx[2]
        else:
            head_idx = idx[1] // head_dim
            head_dim_idx = idx[1] % head_dim
        k_cache.store(
            bs=batch_idx,
            tok_idx=cache_token_idx,
            head_idx=head_idx,
            head_dim_idx=head_dim_idx,
            val=val,
        )

    with Trace[TraceLevel.OP, target=target](
        "rms_norm_kv_cache_ragged_paged_nhead_"
        + String(kv_collection.kv_params.num_heads)
        + ".hdim_"
        + String(kv_collection.kv_params.head_size),
        task_id=get_safe_task_id(context),
    ):
        _rms_norm_impl[
            dtype,
            rank,
            key_cache_input_fn,
            key_cache_output_fn,
            target=target,
            multiply_before_cast=multiply_before_cast,
        ](
            shape,
            LayoutTensor[
                gamma.type,
                Layout.row_major(UNKNOWN_VALUE),
                address_space = gamma.address_space,
            ](
                gamma.data,
                RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
                    gamma.dynamic_shape.canonicalize()
                ),
            ),
            epsilon,
            weight_offset,
            context,
        )


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
                if kv_params.head_size > UInt(num_to_print):
                    print("...", end=", ")
            if total_cache_length > num_to_print:
                print("\n...", end=",")
            print()


def print_kv_cache_cont_batch_generic_cpu[
    target: StaticString, dtype: DType, kv_params: KVCacheStaticParams
](
    valid_lengths: NDBuffer[DType.uint32, 1],
    kv_collection: ContinuousBatchingKVCacheCollection[dtype, kv_params],
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
    dtype: DType,
    kv_params: KVCacheStaticParams,
    page_size: Int,
](
    valid_lengths: NDBuffer[DType.uint32, 1],
    kv_collection: PagedKVCacheCollection[dtype, kv_params, page_size],
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
    target: StaticString, dtype: DType, kv_params: KVCacheStaticParams
](
    valid_lengths: NDBuffer[DType.uint32, 1],
    kv_collection: ContinuousBatchingKVCacheCollection[dtype, kv_params],
    layer_idx: UInt32,
    is_print_compact: Bool,
    context: DeviceContextPtr,
):
    var blocks_ptr = UnsafePointer[Scalar[dtype]].alloc(
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
    dtype: DType,
    kv_params: KVCacheStaticParams,
    page_size: Int,
](
    valid_lengths: NDBuffer[DType.uint32, 1],
    kv_collection: PagedKVCacheCollection[dtype, kv_params, page_size],
    layer_idx: UInt32,
    is_print_compact: Bool,
    context: DeviceContextPtr,
):
    var blocks_ptr = UnsafePointer[Scalar[dtype]].alloc(
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
    dtype: DType, //, kv_params: KVCacheStaticParams
](
    blocks: NDBuffer[dtype, 6],
    cache_lengths: NDBuffer[DType.uint32, 1],
    lookup_table: NDBuffer[DType.uint32, 1],
    max_lengths: NDBuffer[DType.uint32, 2],
    out result: ContinuousBatchingKVCacheCollection[dtype, kv_params],
):
    # Marshal NDBuffers into arguments expected by the
    # ContinuousKVCacheCollection constructor.
    return {
        blocks = blocks,
        cache_lengths = cache_lengths,
        lookup_table = lookup_table,
        max_seq_length = max_lengths[Index(0, 0)],
        max_cache_length = max_lengths[Index(0, 1)],
    }


@always_inline
fn generic_get_continuous_cache[
    dtype: DType, kv_params: KVCacheStaticParams
](
    blocks: NDBuffer[dtype, 6],
    cache_lengths: NDBuffer[DType.uint32, 1],
    lookup_table: NDBuffer[DType.uint32, 1],
    max_lengths: NDBuffer[DType.uint32, 2],
) -> ContinuousBatchingKVCacheCollection[dtype, kv_params]:
    return _continuous_batch_kv_cache_collection[kv_params](
        blocks, cache_lengths, lookup_table, max_lengths
    )


fn generic_get_paged_cache[
    dtype: DType
](
    blocks: MutableInputTensor[dtype=dtype, rank=6],
    cache_lengths: InputTensor[dtype = DType.uint32, rank=1],
    lookup_table: InputTensor[dtype = DType.uint32, rank=2],
    max_lengths: InputTensor[dtype = DType.uint32, rank=2],
    out result: PagedKVCacheCollection[
        dtype,
        KVCacheStaticParams(
            UInt(blocks.static_spec.shape.get[4]()),
            UInt(blocks.static_spec.shape.get[5]()),
        ),
        blocks.static_spec.shape.get[3](),
    ],
):
    alias page_size = blocks.static_spec.shape.get[3]()
    alias head_dim = blocks.static_spec.shape.get[5]()
    alias num_heads = blocks.static_spec.shape.get[4]()
    return generic_get_paged_cache[
        dtype, KVCacheStaticParams(UInt(num_heads), UInt(head_dim)), page_size
    ](
        managed_tensor_slice_to_ndbuffer(blocks),
        managed_tensor_slice_to_ndbuffer(cache_lengths),
        managed_tensor_slice_to_ndbuffer(lookup_table),
        managed_tensor_slice_to_ndbuffer(max_lengths),
    )


fn generic_get_paged_cache[
    dtype: DType, kv_params: KVCacheStaticParams, page_size: Int
](
    blocks: NDBuffer[dtype, 6],
    cache_lengths: NDBuffer[DType.uint32, 1],
    lookup_table: NDBuffer[DType.uint32, 2],
    max_lengths: NDBuffer[DType.uint32, 2],
    out result: PagedKVCacheCollection[dtype, kv_params, page_size],
):
    return {
        blocks = blocks,
        cache_lengths = cache_lengths,
        lookup_table = lookup_table,
        max_seq_length = max_lengths[Index(0, 0)],
        max_cache_length = max_lengths[Index(0, 1)],
    }


@always_inline
fn managed_tensor_slice_to_ndbuffer[
    spec: StaticTensorSpec, //
](tensor: ManagedTensorSlice[static_spec=spec]) -> NDBuffer[
    spec.dtype,
    spec.rank,
    MutableAnyOrigin,
    spec.shape,
    spec.strides,
    alignment2 = spec.alignment,
    address_space = spec.address_space,
    exclusive = spec.exclusive,
]:
    var ptr = tensor._ptr.address_space_cast[spec.address_space]()
    return {ptr, tensor.shape(), tensor._runtime_strides}
