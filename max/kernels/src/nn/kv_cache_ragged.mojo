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
from sys.info import _current_target, simd_width_of
from sys.intrinsics import _type_is_eq

from algorithm.functional import elementwise, unswitch
from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceContext, get_gpu_target
from gpu.host.info import is_cpu, is_gpu
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
    KVCacheT,
    KVCollectionT,
    PagedKVCache,
    PagedKVCacheCollection,
)
from layout import LayoutTensor, Layout, RuntimeLayout, IntTuple, UNKNOWN_VALUE
from linalg.grouped_matmul import grouped_matmul
from linalg.matmul import elementwise_epilogue_type, matmul
from linalg.fp8_quantization import (
    naive_blockwise_scaled_fp8_matmul,
    quantize_dynamic_scaled_fp8,
)
from nn._ragged_utils import get_batch_from_row_offsets
from nn.flash_attention import (
    flash_attention_kv_cache as flash_attention_kv_cache_cpu,
)
from nn.fused_qk_rope import fused_qk_rope_ragged
from nn.mha import flash_attention as gpu_flash_attention
from nn.mha_mask import MHAMask
from nn.mha_score_mod import IdentityScoreMod, ScoreModTrait
from nn.mha_utils import dispatch_mask_and_score_mod
from nn.mla import (
    _k_cache_to_buffer,
    flare_mla_decoding,
    flare_mla_prefill,
    mla_prefill_plan,
)
from quantization.qmatmul import matmul_qint4
from quantization.qmatmul_gpu import matmul_gpu_qint4_impl
from quantization.qmatmul_k import matmul_Q4_K, matmul_Q6_K
from runtime.asyncrt import DeviceContextPtr
from runtime.tracing import Trace, TraceLevel, trace_arg
from tensor_internal import ManagedTensorSlice, trace_slice_arg

from utils.index import IndexList

# ===-----------------------------------------------------------------------===#
# Fused QKV matmul (ragged)
# ===-----------------------------------------------------------------------===#


@always_inline
fn generic_fused_qkv_matmul_kv_cache_cont_batch_ragged[
    dtype: DType, //,
    target: StaticString = "cpu",
](
    hidden_state: NDBuffer[dtype, 2, _, _],
    input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    weight: NDBuffer[dtype, 2, _, _],
    kv_collection: ContinuousBatchingKVCacheCollection,
    layer_idx: UInt32,
    output: NDBuffer[mut=True, dtype, 2, _, _],
    ctx: DeviceContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,).
            The value at each index is the start_idx of the corresponding batch in hidden_state.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The object storing the KVCache for this layer.
        layer_idx: The current layer, used to retrieve the KVCache object from kv_collection.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
            Shape: (sum(seq_lens), num_heads * head_size).
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
        "mo.fused_qkv_matmul.ragged.continuous_batching.nhead_"
        + String(kv_collection.kv_params.num_heads)
        + ".hdim_"
        + String(kv_collection.kv_params.head_size),
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=Int(ctx.get_device_context().id()),
    ):
        return _fused_qkv_matmul_kv_cache_ragged[
            kv_collection.CacheType, target=target
        ](
            hidden_state,
            input_row_offsets,
            weight,
            kv_collection,
            layer_idx,
            output,
            ctx,
        )


@always_inline
fn generic_fused_qkv_matmul_kv_cache_paged_ragged[
    dtype: DType,
    weight_dtype: DType,
    target: StaticString = "cpu",
    group_size: OptionalReg[Int] = None,
    has_zp: OptionalReg[Bool] = None,
](
    hidden_state: NDBuffer[dtype, 2, _, _],
    input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    weight: NDBuffer[weight_dtype, 2, _, _],
    kv_collection: PagedKVCacheCollection,
    layer_idx: UInt32,
    output: NDBuffer[mut=True, dtype, 2, _, _],
    ctx: DeviceContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,).
            The value at each index is the start_idx of the corresponding batch in hidden_state.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The object storing the KVCache for this layer.
        layer_idx: The current layer, used to retrieve the KVCache object from kv_collection.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
            Shape: (sum(seq_lens), num_heads * head_size).
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

    alias name = "mo.fused_qkv_matmul.ragged.paged.nhead_" + String(
        kv_collection.kv_params.num_heads
    ) + ".hdim_" + String(kv_collection.kv_params.head_size)
    with Trace[TraceLevel.OP, target=target](
        name,
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=Int(ctx.get_device_context().id()),
    ):
        return _fused_qkv_matmul_kv_cache_ragged[
            kv_collection.CacheType,
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


@always_inline
fn generic_fused_qkv_matmul_kv_cache_paged_ragged_bias[
    dtype: DType,
    weight_dtype: DType,
    target: StaticString = "cpu",
    group_size: OptionalReg[Int] = None,
    has_zp: OptionalReg[Bool] = None,
](
    hidden_state: NDBuffer[dtype, 2, _, _],
    input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    weight: NDBuffer[weight_dtype, 2, _, _],
    kv_collection: PagedKVCacheCollection,
    layer_idx: UInt32,
    output: NDBuffer[mut=True, dtype, 2, _, _],
    bias: NDBuffer[dtype, 1],
    ctx: DeviceContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,).
            The value at each index is the start_idx of the corresponding batch in hidden_state.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The object storing the KVCache for this layer.
        layer_idx: The current layer, used to retrieve the KVCache object from kv_collection.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
            Shape: (sum(seq_lens), num_heads * head_size).
        bias: Bias to be added to the QKV Tensor. Tensor is concatenated q + k + v. Rank 1.
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

    alias name = "mo.fused_qkv_matmul.ragged.paged.bias.nhead_" + String(
        kv_collection.kv_params.num_heads
    ) + ".hdim_" + String(kv_collection.kv_params.head_size)
    with Trace[TraceLevel.OP, target=target](
        name,
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=Int(ctx.get_device_context().id()),
    ):
        return _fused_qkv_matmul_kv_cache_ragged_bias[
            kv_collection.CacheType,
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


@always_inline
fn generic_fused_qkv_matmul_kv_cache_paged_ragged_scale[
    dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    scale_dtype: DType,
    target: StaticString = "cpu",
](
    hidden_state: NDBuffer[dtype, 2, _, _],
    input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    weight: NDBuffer[weight_dtype, 2, _, _],
    input_scale: NDBuffer[scale_dtype, 2, _, _],
    weight_scale: NDBuffer[scale_dtype, 2, _, _],
    kv_collection: PagedKVCacheCollection,
    layer_idx: UInt32,
    output: NDBuffer[mut=True, output_dtype, 2, _, _],
    ctx: DeviceContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,).
            The value at each index is the start_idx of the corresponding batch
            in hidden_state.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads *
            head_size).
        input_scale: Scale to be multiplied to the input Tensor.
        weight_scale: Scale to be multiplied to the weight Tensor.
        kv_collection: The object storing the KVCache for this layer.
        layer_idx: The current layer, used to retrieve the KVCache object from
            kv_collection.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
            Shape: (sum(seq_lens), num_heads * head_size).
        ctx: The call context pointer, passed by the graph compiler.
    """

    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            trace_arg("output", output),
            trace_arg("hidden_state", hidden_state),
            trace_arg("weight", weight),
            trace_arg("input_scale", input_scale),
            trace_arg("weight_scale", weight_scale),
            "layer_idx=" + String(layer_idx),
            "num_heads=" + String(kv_collection.kv_params.num_heads),
            "head_size=" + String(kv_collection.kv_params.head_size),
        )

    alias name = "mo.fused_qkv_matmul.ragged.paged.scale.nhead_" + String(
        kv_collection.kv_params.num_heads
    ) + ".hdim_" + String(kv_collection.kv_params.head_size)
    with Trace[TraceLevel.OP, target=target](
        name,
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=Int(ctx.get_device_context().id()),
    ):
        return _fused_qkv_matmul_kv_cache_ragged_scale[
            kv_collection.CacheType,
            target=target,
        ](
            hidden_state,
            input_row_offsets,
            weight,
            input_scale,
            weight_scale,
            kv_collection,
            layer_idx,
            output,
            ctx,
        )


@always_inline
fn _fused_qkv_matmul_kv_cache_ragged[
    dtype: DType,
    weight_dtype: DType,
    collection_t: KVCollectionT, //,
    cache_t: KVCacheT,
    *,
    target: StaticString,
    group_size: OptionalReg[Int] = None,
    has_zp: OptionalReg[Bool] = None,
](
    hidden_state: NDBuffer[dtype, 2, _, _],
    input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    weight: NDBuffer[weight_dtype, 2, _, _],
    kv_collection: collection_t,
    layer_idx: UInt32,
    output: NDBuffer[mut=True, dtype, 2, _, _],
    context: DeviceContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,).
            The value at each index is the start_idx of the corresponding batch in hidden_state.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The object storing the KVCache for this layer.
        layer_idx: The current layer, used to retrieve the KVCache object from kv_collection.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
        context: The call context pointer, passed by the graph compiler.
    """
    var cuda_ctx: Optional[DeviceContext] = None
    var layer_idx_cast = Int(layer_idx)
    var k_cache = kv_collection.get_key_cache(layer_idx_cast)
    var v_cache = kv_collection.get_value_cache(layer_idx_cast)

    @parameter
    if is_gpu[target]():
        cuda_ctx = context.get_device_context()

    return _fused_qkv_matmul_kv_cache_ragged_impl[
        target=target,
        group_size=group_size,
        has_zp=has_zp,
    ](
        hidden_state,
        input_row_offsets,
        weight,
        k_cache,
        v_cache,
        output,
        cuda_ctx,
    )


@always_inline
fn _fused_qkv_matmul_kv_cache_ragged_bias[
    dtype: DType,
    weight_dtype: DType,
    collection_t: KVCollectionT, //,
    cache_t: KVCacheT,
    *,
    target: StaticString,
    group_size: OptionalReg[Int] = None,
    has_zp: OptionalReg[Bool] = None,
](
    hidden_state: NDBuffer[dtype, 2, _, _],
    input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    weight: NDBuffer[weight_dtype, 2, _, _],
    kv_collection: collection_t,
    layer_idx: UInt32,
    output: NDBuffer[mut=True, dtype, 2, _, _],
    bias: NDBuffer[dtype, 1],
    context: DeviceContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,).
            The value at each index is the start_idx of the corresponding batch in hidden_state.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The object storing the KVCache for this layer.
        layer_idx: The current layer, used to retrieve the KVCache object from kv_collection.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
        bias: Bias to be added to the QKV Tensor. Tensor is concatenated q + k + v. Rank 1.
        context: The call context pointer, passed by the graph compiler.
    """
    var cuda_ctx: Optional[DeviceContext] = None
    var layer_idx_cast = Int(layer_idx)
    var k_cache = kv_collection.get_key_cache(layer_idx_cast)
    var v_cache = kv_collection.get_value_cache(layer_idx_cast)

    @parameter
    if is_gpu[target]():
        cuda_ctx = context.get_device_context()

    return _fused_qkv_matmul_kv_cache_ragged_impl_bias[
        target=target,
        group_size=group_size,
        has_zp=has_zp,
    ](
        hidden_state,
        input_row_offsets,
        weight,
        k_cache,
        v_cache,
        output,
        bias,
        cuda_ctx,
    )


@always_inline
fn _fused_qkv_matmul_kv_cache_ragged_scale[
    dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    scale_dtype: DType,
    collection_t: KVCollectionT, //,
    cache_t: KVCacheT,
    *,
    target: StaticString,
](
    hidden_state: NDBuffer[dtype, 2, _, _],
    input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    weight: NDBuffer[weight_dtype, 2, _, _],
    input_scale: NDBuffer[scale_dtype, 2, _, _],
    weight_scale: NDBuffer[scale_dtype, 2, _, _],
    kv_collection: collection_t,
    layer_idx: UInt32,
    output: NDBuffer[mut=True, output_dtype, 2, _, _],
    context: DeviceContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads *
            head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,).
            The value at each index is the start_idx of the corresponding batch
            in hidden_state.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads *
            head_size).
        input_scale: Scale to be multiplied to the input Tensor.
        weight_scale: Scale to be multiplied to the weight Tensor.
        kv_collection: The object storing the KVCache for this layer.
        layer_idx: The current layer, used to retrieve the KVCache object
            from kv_collection.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
        context: The call context pointer, passed by the graph compiler.
    """
    var cuda_ctx: Optional[DeviceContext] = None
    var layer_idx_cast = Int(layer_idx)
    var k_cache = kv_collection.get_key_cache(layer_idx_cast)
    var v_cache = kv_collection.get_value_cache(layer_idx_cast)

    @parameter
    if is_gpu[target]():
        cuda_ctx = context.get_device_context()

    return _fused_qkv_matmul_kv_cache_ragged_impl_scale[target=target,](
        hidden_state,
        input_row_offsets,
        weight,
        input_scale,
        weight_scale,
        k_cache,
        v_cache,
        output,
        cuda_ctx,
    )


@always_inline
fn _fused_qkv_matmul_kv_cache_ragged_impl[
    dtype: DType,
    weight_dtype: DType,
    cache_t: KVCacheT, //,
    *,
    target: StaticString,
    group_size: OptionalReg[Int] = None,
    has_zp: OptionalReg[Bool] = None,
](
    hidden_state: NDBuffer[dtype, 2, _, _],
    input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    weight: NDBuffer[weight_dtype, 2, _, _],
    k_cache: cache_t,
    v_cache: cache_t,
    output: NDBuffer[mut=True, dtype, 2, *_],
    context: Optional[DeviceContext],
) raises:
    """Performs a fused QKV matmul on ragged tensors. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,)
            denoting the start of each sequence along the seq_len dimension.
        weight: Tensor with shape (num_heads * head_size, (num_heads + 2 * num_kv_heads) * head_size).
        k_cache: The historical KVCacheT for keys, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        v_cache: The historical KVCacheT for values, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
            Shape is (sum(seq_lens), num_heads * head_size)
        context: The DeviceContext. This is unused if is_cpu[target]().
    """
    alias kv_type = cache_t.dtype
    alias kv_params = cache_t.kv_params
    alias N = weight.shape.get[0]()
    alias K = weight.shape.get[1]()

    constrained[
        kv_type == dtype, "Mismatch in dtype between Q and KV tensors"
    ]()

    var q_dim = output.dim[1]()
    var k_dim = kv_params.head_size * kv_params.num_heads
    var qk_offset = q_dim + k_dim
    var batch_size = input_row_offsets.dim[0]() - 1

    if batch_size == 0:
        return

    @parameter
    @__copy_capture(q_dim, qk_offset, batch_size)
    @always_inline
    fn write_to_cache[
        _dtype: DType, width: Int, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[_dtype, width]):
        if idx[1] < q_dim:
            output.store[width=width, alignment=alignment](
                idx,
                rebind[SIMD[dtype, width]](val),
            )
            return

        global_token_idx = idx[0]

        var batch_idx: Int = get_batch_from_row_offsets(
            LayoutTensor[
                DType.uint32, Layout.row_major[1](input_row_offsets.shape)
            ](
                input_row_offsets.data,
                RuntimeLayout[
                    Layout.row_major[1](input_row_offsets.shape)
                ].row_major(input_row_offsets.get_shape().canonicalize()),
            ),
            global_token_idx,
        )

        token_idx = Int(global_token_idx - input_row_offsets[batch_idx])

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

        var cache_length = cache.cache_length(batch_idx)
        var cache_token_idx = token_idx + cache_length
        cache.store(
            batch_idx,
            h_idx,
            cache_token_idx,
            hd_idx,
            rebind[SIMD[kv_type, width]](output_val),
        )

    @parameter
    if group_size:
        constrained[
            not has_zp.value(), "Zero point is not supported for quantization."
        ]()
        constrained[
            weight_dtype is DType.uint8,
            "Expect GPTQ weights in an uint8 tensor.",
        ]()
        var new_weight = rebind[
            NDBuffer[
                DType.uint8,
                weight.rank,
                weight.origin,
                weight.shape,
                weight.strides,
            ]
        ](weight)

        _qmatmul_common[
            group_size = group_size.value(),
            target=target,
            elementwise_lambda_fn=write_to_cache,
        ](hidden_state, new_weight, context)

    else:
        constrained[
            weight_dtype == dtype,
            "Mismatch in dtype between weight and QKV tensors",
        ]()
        var new_weight = rebind[
            NDBuffer[
                dtype, weight.rank, weight.origin, weight.shape, weight.strides
            ]
        ](weight)

        _matmul_common[target=target, elementwise_lambda_fn=write_to_cache](
            hidden_state, new_weight, context
        )


@always_inline
fn _fused_qkv_matmul_kv_cache_ragged_impl_bias[
    dtype: DType,
    weight_dtype: DType,
    cache_t: KVCacheT, //,
    *,
    target: StaticString,
    group_size: OptionalReg[Int] = None,
    has_zp: OptionalReg[Bool] = None,
](
    hidden_state: NDBuffer[dtype, 2, _, _],
    input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    weight: NDBuffer[weight_dtype, 2, _, _],
    k_cache: cache_t,
    v_cache: cache_t,
    output: NDBuffer[mut=True, dtype, 2, *_],
    bias: NDBuffer[dtype, 1],
    context: Optional[DeviceContext],
) raises:
    """Performs a fused QKV matmul on ragged tensors. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,)
            denoting the start of each sequence along the seq_len dimension.
        weight: Tensor with shape (num_heads * head_size, (num_heads + 2 * num_kv_heads) * head_size).
        k_cache: The historical KVCacheT for keys, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        v_cache: The historical KVCacheT for values, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
            Shape is (sum(seq_lens), num_heads * head_size)
        bias: Bias to be added to the QKV Tensor. Tensor is concatenated q + k + v. Rank 1.
        context: The DeviceContext. This is unused if is_cpu[target]().
    """
    alias kv_type = cache_t.dtype
    alias kv_params = cache_t.kv_params
    alias N = weight.shape.get[0]()
    alias K = weight.shape.get[1]()

    constrained[
        kv_type == dtype, "Mismatch in dtype between Q and KV tensors"
    ]()

    var q_dim = output.dim[1]()
    var k_dim = kv_params.head_size * kv_params.num_heads
    var qk_offset = q_dim + k_dim
    var batch_size = input_row_offsets.dim[0]() - 1

    if batch_size == 0:
        return

    @parameter
    @__copy_capture(q_dim, qk_offset, batch_size)
    @always_inline
    fn write_to_cache[
        _dtype: DType, width: Int, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[_dtype, width]):
        var output_val = val + rebind[SIMD[_dtype, width]](
            bias.load[width=width, alignment=alignment](idx[1])
        )
        if idx[1] < q_dim:
            output.store[width=width, alignment=alignment](
                idx,
                rebind[SIMD[dtype, width]](output_val),
            )
            return

        global_token_idx = idx[0]

        var batch_idx: Int = get_batch_from_row_offsets(
            LayoutTensor[
                DType.uint32, Layout.row_major[1](input_row_offsets.shape)
            ](
                input_row_offsets.data,
                RuntimeLayout[
                    Layout.row_major[1](input_row_offsets.shape)
                ].row_major(input_row_offsets.get_shape().canonicalize()),
            ),
            global_token_idx,
        )

        token_idx = Int(global_token_idx - input_row_offsets[batch_idx])

        var h_idx: UInt
        var hd_idx: UInt
        var cache: cache_t
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

        var cache_length = cache.cache_length(batch_idx)
        var cache_token_idx = token_idx + cache_length
        cache.store(
            batch_idx,
            h_idx,
            cache_token_idx,
            hd_idx,
            rebind[SIMD[kv_type, width]](output_val),
        )

    @parameter
    if group_size:
        constrained[
            not has_zp.value(), "Zero point is not supported for quantization."
        ]()
        constrained[
            weight_dtype is DType.uint8,
            "Expect GPTQ weights to be a 'uint8' tensor.",
        ]()
        var new_weight = rebind[
            NDBuffer[
                DType.uint8,
                weight.rank,
                weight.origin,
                weight.shape,
                weight.strides,
            ]
        ](weight)

        _qmatmul_common[
            group_size = group_size.value(),
            target=target,
            elementwise_lambda_fn=write_to_cache,
        ](hidden_state, new_weight, context)

    else:
        constrained[
            weight_dtype == dtype,
            "Mismatch in dtype between weight and QKV tensors",
        ]()
        var new_weight = rebind[
            NDBuffer[
                dtype, weight.rank, weight.origin, weight.shape, weight.strides
            ]
        ](weight)

        _matmul_common[target=target, elementwise_lambda_fn=write_to_cache](
            hidden_state, new_weight, context
        )


@always_inline
fn _fused_qkv_matmul_kv_cache_ragged_impl_scale[
    dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    scale_dtype: DType,
    cache_t: KVCacheT, //,
    *,
    target: StaticString,
](
    hidden_state: NDBuffer[dtype, 2, _, _],
    input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    weight: NDBuffer[weight_dtype, 2, _, _],
    input_scale: NDBuffer[scale_dtype, 2, _, _],
    weight_scale: NDBuffer[scale_dtype, 2, _, _],
    k_cache: cache_t,
    v_cache: cache_t,
    output: NDBuffer[mut=True, output_dtype, 2, *_],
    context: Optional[DeviceContext],
) raises:
    """Performs a fused QKV matmul on ragged tensors. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,)
            denoting the start of each sequence along the seq_len dimension.
        weight: Tensor with shape (num_heads * head_size, (num_heads + 2 *
            num_kv_heads) * head_size).
        input_scale: Scale to be multiplied to the input Tensor.
        weight_scale: Scale to be multiplied to the weight Tensor.
        k_cache: The historical KVCacheT for keys, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        v_cache: The historical KVCacheT for values, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
            Shape is (sum(seq_lens), num_heads * head_size)
        context: The DeviceContext. This is unused if is_cpu[target]().
    """
    alias kv_type = cache_t.dtype
    alias kv_params = cache_t.kv_params
    alias N = weight.shape.get[0]()
    alias K = weight.shape.get[1]()

    var q_dim = output.dim[1]()
    var k_dim = kv_params.head_size * kv_params.num_heads
    var qk_offset = q_dim + k_dim
    var batch_size = input_row_offsets.dim[0]() - 1

    if batch_size == 0:
        return

    # Here we decide the quantization scheme for the QKV Tensor.
    alias use_per_tensor = (
        input_scale.shape.get[0]() == 1
        and input_scale.shape.get[1]() == 1
        and weight_scale.shape.get[0]() == 1
        and weight_scale.shape.get[1]() == 1
    )
    alias use_per_channel = (
        input_scale.shape.get[0]() == 1
        and weight_scale.shape.get[1]() == 1
        and not use_per_tensor
    )

    constrained[
        use_per_tensor or use_per_channel, "Invalid quantization scheme"
    ]()

    @parameter
    @__copy_capture(input_scale, weight_scale, q_dim, qk_offset, batch_size)
    @always_inline
    fn write_to_cache[
        dtype: DType, width: Int, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]):
        var output_val: SIMD[dtype, width]

        @parameter
        if use_per_tensor:
            var scale_a = input_scale[0, 0].cast[dtype]()
            var scale_b = weight_scale[0, 0].cast[dtype]()
            output_val = val * (scale_a * scale_b)
        else:
            var scale_a = input_scale.load[width=1](0, idx[0]).cast[dtype]()
            var scale_b = weight_scale.load[width=width](idx[1], 0).cast[
                dtype
            ]()
            output_val = val * (scale_a * scale_b)

        if idx[1] < q_dim:
            output.store[width=width, alignment=alignment](
                idx,
                rebind[SIMD[output_dtype, width]](
                    output_val.cast[output_dtype]()
                ),
            )
            return

        global_token_idx = idx[0]

        var batch_idx: Int = get_batch_from_row_offsets(
            LayoutTensor[
                DType.uint32, Layout.row_major[1](input_row_offsets.shape)
            ](
                input_row_offsets.data,
                RuntimeLayout[
                    Layout.row_major[1](input_row_offsets.shape)
                ].row_major(input_row_offsets.get_shape().canonicalize()),
            ),
            global_token_idx,
        )

        token_idx = Int(global_token_idx - input_row_offsets[batch_idx])

        var h_idx: UInt
        var hd_idx: UInt
        var cache: cache_t
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

        var cache_length = cache.cache_length(batch_idx)
        var cache_token_idx = token_idx + cache_length
        cache.store(
            batch_idx,
            h_idx,
            cache_token_idx,
            hd_idx,
            rebind[SIMD[kv_type, width]](output_val.cast[kv_type]()),
        )

    constrained[
        weight_dtype == dtype,
        "Mismatch in dtype between weight and QKV tensors",
    ]()
    var new_weight = rebind[
        NDBuffer[
            dtype, weight.rank, weight.origin, weight.shape, weight.strides
        ]
    ](weight)

    _matmul_common[
        target=target,
        elementwise_lambda_fn=write_to_cache,
        output_dtype = DType.float32,
    ](hidden_state, new_weight, context)


@always_inline
fn _matmul_common[
    dtype: DType, //,
    *,
    target: StaticString,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    output_dtype: DType = dtype,
](
    hidden_state: NDBuffer[dtype, 2, _, _],
    weight: NDBuffer[dtype, 2, _, _],
    context: Optional[DeviceContext],
) raises:
    var TOTAL_SEQ_LEN = hidden_state.dim[0]()
    alias N = weight.shape.get[0]()
    alias K = weight.shape.get[1]()
    var c_nd: NDBuffer[output_dtype, 2, MutableAnyOrigin, DimList(Dim(), N)]

    @parameter
    if is_cpu[target]():
        # The CPU matmul codepath uses the C buffer as a workspace
        # even if an epilogue is provided, here we just allocate
        # something to ensure we don't segfault.
        var c_ptr = UnsafePointer[Scalar[output_dtype]].alloc(TOTAL_SEQ_LEN * N)

        c_nd = {c_ptr, IndexList[2](TOTAL_SEQ_LEN, N)}
    else:
        c_nd = {
            UnsafePointer[Scalar[output_dtype]](),
            IndexList[2](TOTAL_SEQ_LEN, N),
        }

    matmul[
        target=target,
        transpose_b=True,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ](c_nd, hidden_state, weight, context)

    @parameter
    if is_cpu[target]():
        c_nd.data.free()


@always_inline
fn _qmatmul_common[
    dtype: DType, //,
    *,
    group_size: Int,
    target: StaticString,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    hidden_state: NDBuffer[dtype, 2, *_],
    weight: NDBuffer[DType.uint8, 2, _, _],
    context: Optional[DeviceContext],
) raises:
    constrained[is_gpu[target](), "GPTQ quantization only works on GPU."]()

    var TOTAL_SEQ_LEN = hidden_state.dim[0]()
    alias N = weight.shape.get[0]()
    var c_nd: NDBuffer[dtype, 2, MutableAnyOrigin, DimList(Dim(), N)]

    c_nd = {
        UnsafePointer[Scalar[dtype]](),
        IndexList[2](TOTAL_SEQ_LEN, N),
    }

    matmul_gpu_qint4_impl[
        target=target,
        group_size=group_size,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ](c_nd, hidden_state, weight, context)


@always_inline
fn _matmul_blockwise_scaled_fp8_common[
    output_dtype: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType, //,
    *,
    target: StaticString,
    scales_granularity_mnk: IndexList[3],
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    hidden_state: NDBuffer[a_type, 2, *_],
    weight: NDBuffer[b_type, 2, _, _],
    input_scale: NDBuffer[a_scales_type, 2, _, _],
    weight_scale: NDBuffer[b_scales_type, 2, _, _],
    context: DeviceContext,
) raises:
    constrained[
        is_gpu[target](), "Blockwise scaled fp8 matmul only works on GPU."
    ]()

    var TOTAL_SEQ_LEN = hidden_state.dim[0]()
    alias N = weight.shape.get[0]()
    var c_nd: NDBuffer[output_dtype, 2, MutableAnyOrigin, DimList(Dim(), N)]

    c_nd = {
        UnsafePointer[Scalar[output_dtype]](),
        IndexList[2](TOTAL_SEQ_LEN, N),
    }

    naive_blockwise_scaled_fp8_matmul[
        transpose_b=True,
        elementwise_lambda_fn=elementwise_lambda_fn,
        scales_granularity_mnk=scales_granularity_mnk,
    ](c_nd, hidden_state, weight, input_scale, weight_scale, context)


# ===-----------------------------------------------------------------------===#
# Unfused KV cache matmul (ragged)
# ===-----------------------------------------------------------------------===#


fn kv_matmul_ragged_paged[
    dtype: DType,
    params: KVCacheStaticParams,
    page_size: Int, //,
    target: StaticString,
](
    hidden_state: NDBuffer[dtype, 2, _, _],
    input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    weight: NDBuffer[dtype, 2, _, _],
    kv_collection: PagedKVCacheCollection[
        dtype,
        params,
        page_size,
    ],
    layer_idx: UInt32,
    ctx: DeviceContextPtr,
) raises:
    """Performs a matmul, writing the output into a mutable ContinuousBatchingKVCacheCollection object.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,)
            denoting the start of each sequence along the seq_len dimension.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The historical KVCache for keys and values. The KVCache for
            this layer is retrieved via layer_idx.
        layer_idx: The index of the layer being executed. Used to retrieve the KVCache
            for the given layer from kv_collection.
        ctx: The call context pointer, passed by the graph compiler.
    """

    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            trace_arg("weight", weight),
            "layer_idx=" + String(layer_idx),
            "num_heads=" + String(kv_collection.kv_params.num_heads),
            "head_size=" + String(kv_collection.kv_params.head_size),
        )

    with Trace[TraceLevel.OP, target=target](
        "mo.kv_matmul.ragged.paged.nhead_"
        + String(kv_collection.kv_params.num_heads)
        + ".hdim_"
        + String(kv_collection.kv_params.head_size),
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=Int(ctx.get_device_context().id()),
    ):
        return _matmul_kv_cache_ragged[target=target](
            hidden_state,
            input_row_offsets,
            weight,
            kv_collection,
            layer_idx,
            ctx,
        )


@always_inline
fn _matmul_kv_cache_ragged[
    dtype: DType, //, *, target: StaticString
](
    hidden_state: NDBuffer[dtype, 2, _, _],
    input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    weight: NDBuffer[dtype, 2, _, _],
    kv_collection: PagedKVCacheCollection,
    layer_idx: UInt32,
    context: DeviceContextPtr,
) raises:
    """Helper for performing matmul with custom ContinuousBatchingKVCacheCollection dtypes.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,)
            denoting the start of each sequence along the seq_len dimension.
        weight: Tensor with shape (num_heads * head_size, 2 * num_kv_heads * head_size)
        kv_collection: The historical KVCache for keys and values. The KVCache for
            this layer is retrieved via layer_idx.
        layer_idx: The index of the layer being executed. Used to retrieve the KVCache
            for the given layer from kv_collection.
        context: Pointer containing the runtime context for the target device.
    """
    var cuda_ctx: Optional[DeviceContext] = None
    layer_idx_cast = Int(layer_idx)
    k_cache = kv_collection.get_key_cache(layer_idx_cast)
    v_cache = kv_collection.get_value_cache(layer_idx_cast)

    @parameter
    if is_gpu[target]():
        cuda_ctx = context.get_device_context()

    _matmul_kv_cache_ragged_impl[target=target](
        hidden_state,
        input_row_offsets,
        weight,
        k_cache,
        v_cache,
        cuda_ctx,
    )


@always_inline
fn _matmul_kv_cache_ragged_impl[
    dtype: DType,
    cache_t: KVCacheT, //,
    *,
    target: StaticString,
](
    hidden_state: NDBuffer[dtype, 2, _, _],
    input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    weight: NDBuffer[dtype, 2, _, _],
    k_cache: cache_t,
    v_cache: cache_t,
    ctx: Optional[DeviceContext],
) raises:
    """Helper for performing matmul with custom KVCacheT dtypes.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,)
            denoting the start of each sequence along the seq_len dimension.
        weight: Tensor with shape (num_heads * head_size, 2 * num_kv_heads * head_size)
        k_cache: The historical KVCacheT for keys, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        v_cache: The historical KVCacheT for values, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        ctx: Pointer containing the runtime context for the target device.
    """
    if hidden_state.num_elements() == 0:
        # Nothing to do.
        return

    alias kv_params = cache_t.kv_params
    alias N: UInt = UInt(weight.shape.get[0]())
    alias K: UInt = UInt(weight.shape.get[1]())

    batch_size = input_row_offsets.dim[0]() - 1

    # Set the matmul_common output lambda to write to K cache for the first N
    # elements and V cache for the next N.
    k_offset = kv_params.head_size * kv_params.num_heads

    @parameter
    @__copy_capture(input_row_offsets, k_offset, batch_size)
    @always_inline
    fn write_to_cache_common[
        dtype: DType, cache_t: KVCacheT, width: Int
    ](
        k_cache: cache_t,
        v_cache: cache_t,
        idx: IndexList[2],
        val: SIMD[dtype, width],
    ):
        alias kv_type = cache_t.dtype

        constrained[
            kv_type == dtype,
            "Mismatch in dtype between hidden state and KV tensors",
        ]()

        # Token index in the "ragged" combined sequence dimension.
        global_token_idx = idx[0]

        batch_idx = get_batch_from_row_offsets(
            LayoutTensor[
                DType.uint32, Layout.row_major[1](input_row_offsets.shape)
            ](
                input_row_offsets.data,
                RuntimeLayout[
                    Layout.row_major[1](input_row_offsets.shape)
                ].row_major(input_row_offsets.get_shape().canonicalize()),
            ),
            global_token_idx,
        )
        token_idx = Int(global_token_idx - input_row_offsets[batch_idx])

        if idx[1] < k_offset:
            # Write this element to the K cache.
            cache = k_cache
            h_idx, hd_idx = divmod(UInt(idx[1]), kv_params.head_size)
        else:
            # Otherwise, write this element to the V cache.
            cache = v_cache
            h_idx, hd_idx = divmod(UInt(idx[1]) - k_offset, kv_params.head_size)

        cache_length = cache.cache_length(batch_idx)
        cache_token_idx = token_idx + cache_length
        cache.store(
            batch_idx,
            h_idx,
            cache_token_idx,
            hd_idx,
            rebind[SIMD[kv_type, width]](val),
        )

    # Cast to a register passable dtype so the function closure works on GPU.
    k_cache_reg = rebind[cache_t](k_cache)
    v_cache_reg = rebind[cache_t](v_cache)

    @parameter
    @__copy_capture(k_cache_reg, v_cache_reg)
    @always_inline
    fn write_to_cache_continuous[
        dtype: DType, width: Int, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]):
        write_to_cache_common(k_cache_reg, v_cache_reg, idx, val)

    _matmul_common[
        target=target, elementwise_lambda_fn=write_to_cache_continuous
    ](hidden_state, weight, ctx)


# ===-----------------------------------------------------------------------===#
# Unfused K cache matmul (ragged)
# ===-----------------------------------------------------------------------===#


fn k_matmul_ragged_paged[
    dtype: DType,
    params: KVCacheStaticParams,
    page_size: Int, //,
    target: StaticString,
](
    hidden_state: NDBuffer[dtype, 2, *_],
    input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    weight: NDBuffer[dtype, 2, *_],
    kv_collection: PagedKVCacheCollection[
        dtype,
        params,
        page_size,
    ],
    layer_idx: UInt32,
    ctx: DeviceContextPtr,
) raises:
    """Performs a matmul, writing the output into a mutable PagedKVCacheCollection object.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,)
            denoting the start of each sequence along the seq_len dimension.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The historical KVCache for keys and values. The KVCache for
            this layer is retrieved via layer_idx.
        layer_idx: The index of the layer being executed. Used to retrieve the KVCache
            for the given layer from kv_collection.
        ctx: The call context pointer, passed by the graph compiler.
    """

    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            trace_arg("weight", weight),
            "layer_idx=" + String(layer_idx),
        )

    with Trace[TraceLevel.OP, target=target](
        "mo.k_matmul.ragged.paged.nhead_"
        + String(kv_collection.kv_params.num_heads)
        + ".hdim_"
        + String(kv_collection.kv_params.head_size),
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=Int(ctx.get_device_context().id()),
    ):
        return _matmul_k_cache_ragged[target=target](
            hidden_state,
            input_row_offsets,
            weight,
            kv_collection,
            layer_idx,
            ctx,
        )


@always_inline
fn _matmul_k_cache_ragged[
    dtype: DType, //,
    *,
    target: StaticString,
](
    hidden_state: NDBuffer[dtype, 2, _, _],
    input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    weight: NDBuffer[dtype, 2, _, _],
    kv_collection: PagedKVCacheCollection,
    layer_idx: UInt32,
    context: DeviceContextPtr,
) raises:
    """Helper for performing matmul with custom PagedKVCacheCollection dtypes.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,)
            denoting the start of each sequence along the seq_len dimension.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size)
        kv_collection: The historical KVCache for keys and values. The KVCache for
            this layer is retrieved via layer_idx.
        layer_idx: The index of the layer being executed. Used to retrieve the KVCache
            for the given layer from kv_collection.
        context: Pointer containing the runtime context for the target device.
    """
    var cuda_ctx: Optional[DeviceContext] = None
    layer_idx_cast = Int(layer_idx)
    k_cache = kv_collection.get_key_cache(layer_idx_cast)

    @parameter
    if is_gpu[target]():
        cuda_ctx = context.get_device_context()

    _matmul_k_cache_ragged_impl[target=target](
        hidden_state,
        input_row_offsets,
        weight,
        k_cache,
        cuda_ctx,
    )


@always_inline
fn _matmul_k_cache_ragged_impl[
    dtype: DType,
    cache_t: KVCacheT, //,
    *,
    target: StaticString,
](
    hidden_state: NDBuffer[dtype, 2, _, _],
    input_row_offsets: NDBuffer[DType.uint32, 1, _, _],
    weight: NDBuffer[dtype, 2, _, _],
    k_cache: cache_t,
    ctx: Optional[DeviceContext],
) raises:
    """Helper for performing matmul with custom KVCacheT dtypes.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,)
            denoting the start of each sequence along the seq_len dimension.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size)
        k_cache: The historical KVCacheT for keys, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        ctx: Pointer containing the runtime context for the target device.
    """
    if hidden_state.num_elements() == 0:
        # Nothing to do.
        return

    alias kv_params = cache_t.kv_params
    alias N: UInt = UInt(weight.shape.get[0]())
    alias K: UInt = UInt(weight.shape.get[1]())

    batch_size = input_row_offsets.dim[0]() - 1

    @parameter
    @__copy_capture(batch_size)
    @always_inline
    fn write_to_cache[
        dtype: DType, width: Int, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width],):
        alias kv_type = cache_t.dtype

        constrained[
            kv_type == dtype,
            "Mismatch in dtype between hidden state and KV tensors",
        ]()

        # Token index in the "ragged" combined sequence dimension.
        global_token_idx = idx[0]

        batch_idx = get_batch_from_row_offsets(
            LayoutTensor[
                DType.uint32, Layout.row_major[1](input_row_offsets.shape)
            ](
                input_row_offsets.data,
                RuntimeLayout[
                    Layout.row_major[1](input_row_offsets.shape)
                ].row_major(input_row_offsets.get_shape().canonicalize()),
            ),
            global_token_idx,
        )
        token_idx = Int(global_token_idx - input_row_offsets[batch_idx])

        h_idx, hd_idx = divmod(UInt(idx[1]), kv_params.head_size)

        cache_length = k_cache.cache_length(batch_idx)
        cache_token_idx = token_idx + cache_length
        k_cache.store(
            batch_idx,
            h_idx,
            cache_token_idx,
            hd_idx,
            rebind[SIMD[kv_type, width]](val),
        )

    _matmul_common[target=target, elementwise_lambda_fn=write_to_cache](
        hidden_state, weight, ctx
    )


fn k_matmul_ragged_paged_scale[
    dtype: DType,
    weight_dtype: DType,
    scale_dtype: DType,
    target: StaticString,
    scales_granularity_mnk: IndexList[3],
](
    hidden_state: NDBuffer[dtype, 2, _, _],
    input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    weight: NDBuffer[weight_dtype, 2, _, _],
    input_scale: NDBuffer[scale_dtype, 2, _, _],
    weight_scale: NDBuffer[scale_dtype, 2, _, _],
    kv_collection: PagedKVCacheCollection,
    layer_idx: UInt32,
    ctx: DeviceContextPtr,
) raises:
    """Performs a matmul, writing the output into a mutable
    PagedKVCacheCollection object.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,)
            denoting the start of each sequence along the seq_len dimension.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        input_scale: Scale to be multiplied to the input Tensor.
        weight_scale: Scale to be multiplied to the weight Tensor.
        kv_collection: The historical KVCache for keys and values. The KVCache for
            this layer is retrieved via layer_idx.
        layer_idx: The index of the layer being executed. Used to retrieve the KVCache
            for the given layer from kv_collection.
        ctx: The call context pointer, passed by the graph compiler.
    """

    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            trace_arg("hidden_state", hidden_state),
            trace_arg("weight", weight),
            trace_arg("input_scale", input_scale),
            trace_arg("weight_scale", weight_scale),
            "layer_idx=" + String(layer_idx),
        )

    with Trace[TraceLevel.OP, target=target](
        "mo.k_matmul.ragged.paged.scale.nhead_"
        + String(kv_collection.kv_params.num_heads)
        + ".hdim_"
        + String(kv_collection.kv_params.head_size),
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=Int(ctx.get_device_context().id()),
    ):
        constrained[
            is_gpu[target](), "Blockwise scaled fp8 matmul only works on GPU."
        ]()
        var layer_idx_cast = Int(layer_idx)
        var k_cache = kv_collection.get_key_cache(layer_idx_cast)

        return _matmul_k_cache_ragged_scale_impl[
            target=target,
            scales_granularity_mnk=scales_granularity_mnk,
        ](
            hidden_state,
            input_row_offsets,
            weight,
            input_scale,
            weight_scale,
            k_cache,
            ctx.get_device_context(),
        )


@always_inline
fn _matmul_k_cache_ragged_scale_impl[
    dtype: DType,
    weight_dtype: DType,
    scale_dtype: DType, //,
    cache_t: KVCacheT,
    *,
    target: StaticString,
    scales_granularity_mnk: IndexList[3],
](
    hidden_state: NDBuffer[dtype, 2, _, _],
    input_row_offsets: NDBuffer[DType.uint32, 1, _, _],
    weight: NDBuffer[weight_dtype, 2, _, _],
    input_scale: NDBuffer[scale_dtype, 2, _, _],
    weight_scale: NDBuffer[scale_dtype, 2, _, _],
    k_cache: cache_t,
    ctx: DeviceContext,
) raises:
    """Helper for performing matmul with custom KVCacheT dtypes.

    Currently assumes block size scaling.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,)
            denoting the start of each sequence along the seq_len dimension.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size)
        input_scale: Scale to be multiplied to the input Tensor.
        weight_scale: Scale to be multiplied to the weight Tensor.
        k_cache: The historical KVCacheT for keys, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        ctx: Pointer containing the runtime context for the target device.
    """
    if hidden_state.num_elements() == 0:
        # Nothing to do.
        return

    alias kv_params = cache_t.kv_params
    alias N: UInt = UInt(weight.shape.get[0]())
    alias K: UInt = UInt(weight.shape.get[1]())

    var batch_size = input_row_offsets.dim[0]() - 1

    @parameter
    @__copy_capture(input_scale, weight_scale, batch_size)
    @always_inline
    fn write_to_cache[
        dtype: DType, width: Int, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width],):
        alias kv_type = cache_t.dtype

        constrained[
            kv_type == dtype,
            "Mismatch in dtype between hidden state and KV tensors",
        ]()

        # Token index in the "ragged" combined sequence dimension.
        var global_token_idx = idx[0]

        var batch_idx = get_batch_from_row_offsets(
            LayoutTensor[
                DType.uint32, Layout.row_major[1](input_row_offsets.shape)
            ](
                input_row_offsets.data,
                RuntimeLayout[
                    Layout.row_major[1](input_row_offsets.shape)
                ].row_major(input_row_offsets.get_shape().canonicalize()),
            ),
            global_token_idx,
        )
        var token_idx = Int(global_token_idx - input_row_offsets[batch_idx])

        var h_idx, hd_idx = divmod(UInt(idx[1]), kv_params.head_size)

        var cache_length = k_cache.cache_length(batch_idx)
        var cache_token_idx = token_idx + cache_length
        k_cache.store(
            batch_idx,
            h_idx,
            cache_token_idx,
            hd_idx,
            rebind[SIMD[kv_type, width]](val),
        )

    constrained[
        weight_dtype == dtype,
        "Mismatch in dtype between weight and QKV tensors",
    ]()
    var new_weight = rebind[
        NDBuffer[
            dtype, weight.rank, weight.origin, weight.shape, weight.strides
        ]
    ](weight)
    _matmul_blockwise_scaled_fp8_common[
        output_dtype = cache_t.dtype,
        target=target,
        elementwise_lambda_fn=write_to_cache,
        scales_granularity_mnk=scales_granularity_mnk,
    ](hidden_state, new_weight, input_scale, weight_scale, ctx)


# ===-----------------------------------------------------------------------===#
# Unfused gguf quantized QKV cache matmul (ragged)
# ===-----------------------------------------------------------------------===#


fn unfused_qkv_matmul_ragged_paged_gguf_quantized[
    dtype: DType,
    params: KVCacheStaticParams,
    page_size: Int, //,
    quantization_encoding_q: StaticString,
    quantization_encoding_k: StaticString,
    quantization_encoding_v: StaticString,
](
    hidden_state: NDBuffer[DType.float32, 2, _, _],
    input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    q_weight: NDBuffer[DType.uint8, 2, _, _],
    k_weight: NDBuffer[DType.uint8, 2, _, _],
    v_weight: NDBuffer[DType.uint8, 2, _, _],
    kv_collection: PagedKVCacheCollection[
        dtype,
        params,
        page_size,
    ],
    layer_idx: UInt32,
    output: NDBuffer[mut=True, DType.float32, 2, _, _],
    ctx: DeviceContextPtr,
) raises:
    """Performs a quantized matmul, writing the output into a mutable PagedKVCacheCollection object.

    Unlike the un-quantized version (kv_matmul_ragged_continuous_batching), this
    implementation does not concat the q, k, and v weights together. Instead, it
    performs three matmuls. This allows the q, k, and v weights to have different
    quantization encodings.

    This is only supported on CPU.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,)
            denoting the start of each sequence along the seq_len dimension.
        q_weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        k_weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        v_weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The Collection object storing KVCache entries.
        layer_idx: The index of the layer being executed. Used to retrieve the KVCache
            for the given layer from kv_collection.
        output: Tensor with shape (sum(seq_lens), num_kv_heads * head_size).
            This is the output buffer for the Q matmul.
        ctx: The call context pointer, passed by the graph compiler.
    """

    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            trace_arg("q_weight", q_weight),
            trace_arg("k_weight", k_weight),
            trace_arg("v_weight", v_weight),
            "layer_idx=" + String(layer_idx),
            "num_heads=" + String(kv_collection.kv_params.num_heads),
            "head_size=" + String(kv_collection.kv_params.head_size),
            "quantization_encoding_q=" + quantization_encoding_q,
            "quantization_encoding_k=" + quantization_encoding_k,
            "quantization_encoding_v=" + quantization_encoding_v,
        )

    with Trace[TraceLevel.OP, target = StaticString("cpu")](
        "mo.kv_matmul.ragged.paged.nhead_"
        + String(kv_collection.kv_params.num_heads)
        + ".hdim_"
        + String(kv_collection.kv_params.head_size)
        + ".quantization_encoding_q="
        + quantization_encoding_q
        + ".quantization_encoding_k="
        + quantization_encoding_k
        + ".quantization_encoding_v="
        + quantization_encoding_v,
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
    ):
        return _unfused_qkv_matmul_ragged_paged_gguf_quantized_impl[
            quantization_encoding_q,
            quantization_encoding_k,
            quantization_encoding_v,
        ](
            hidden_state,
            input_row_offsets,
            q_weight,
            k_weight,
            v_weight,
            kv_collection,
            layer_idx,
            output,
            ctx,
        )


@always_inline
fn _unfused_qkv_matmul_ragged_paged_gguf_quantized_impl[
    quantization_encoding_q: StaticString,
    quantization_encoding_k: StaticString,
    quantization_encoding_v: StaticString,
](
    hidden_state: NDBuffer[DType.float32, 2, _, _],
    input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    q_weight: NDBuffer[DType.uint8, 2, _, _],
    k_weight: NDBuffer[DType.uint8, 2, _, _],
    v_weight: NDBuffer[DType.uint8, 2, _, _],
    kv_collection: PagedKVCacheCollection,
    layer_idx: UInt32,
    output: NDBuffer[mut=True, DType.float32, 2, _, _],
    context: DeviceContextPtr,
) raises:
    layer_idx_cast = Int(layer_idx)
    k_cache = kv_collection.get_key_cache(layer_idx_cast)
    v_cache = kv_collection.get_value_cache(layer_idx_cast)

    alias cache_t = PagedKVCache[
        DType.float32, kv_collection.kv_params, kv_collection.page_size
    ]
    k_cache_reg = rebind[cache_t](k_cache)
    v_cache_reg = rebind[cache_t](v_cache)

    _matmul_kv_cache_ragged_gguf_quantized_impl[
        cache_t,
        quantization_encoding_q,
        quantization_encoding_k,
        quantization_encoding_v,
    ](
        hidden_state,
        input_row_offsets,
        q_weight,
        k_weight,
        v_weight,
        k_cache_reg,
        v_cache_reg,
        output,
    )


@always_inline
fn _matmul_kv_cache_ragged_gguf_quantized_impl[
    cache_t: KVCacheT,
    quantization_encoding_q: StaticString,
    quantization_encoding_k: StaticString,
    quantization_encoding_v: StaticString,
](
    hidden_state: NDBuffer[DType.float32, 2, _, _],
    input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    q_weight: NDBuffer[DType.uint8, 2, _, _],
    k_weight: NDBuffer[DType.uint8, 2, _, _],
    v_weight: NDBuffer[DType.uint8, 2, _, _],
    k_cache: cache_t,
    v_cache: cache_t,
    output: NDBuffer[mut=True, DType.float32, 2, _, _],
) raises:
    """Helper for performing quantized matmul with custom KVCacheT dtypes.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_kv_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,)
            denoting the start of each sequence along the seq_len dimension.
        q_weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size)
        k_weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size)
        v_weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size)
        k_cache: The Collection object storing KVCache K entries.
        v_cache: The Collection object storing KVCache V entries.
        output: Tensor with shape (sum(seq_lens), num_kv_heads * head_size).
            This is the output buffer for the Q matmul.
    """
    if hidden_state.num_elements() == 0:
        # Nothing to do.
        return

    # K matmul with epilogue
    _qmatmul_k_or_v_cache_ragged_gguf_quantized_impl[
        cache_t, quantization_encoding_k
    ](hidden_state, input_row_offsets, k_weight, k_cache)

    # V matmul with epilogue
    _qmatmul_k_or_v_cache_ragged_gguf_quantized_impl[
        cache_t, quantization_encoding_v
    ](hidden_state, input_row_offsets, v_weight, v_cache)

    # Q matmul without epilogue which writes to output buffer
    _qmatmul_gguf_quantized_common[quantization_encoding_q](
        hidden_state, q_weight, output
    )


@always_inline
fn _qmatmul_k_or_v_cache_ragged_gguf_quantized_impl[
    cache_t: KVCacheT,
    quantization_encoding: StaticString,
](
    hidden_state: NDBuffer[DType.float32, 2, _, _],
    input_row_offsets: NDBuffer[DType.uint32, 1, _, _],
    k_or_v_weight: NDBuffer[DType.uint8, 2, _, _],
    k_or_v_cache: cache_t,
) raises:
    alias kv_params = cache_t.kv_params
    alias N: UInt = UInt(k_or_v_weight.shape.get[0]())
    alias K: UInt = UInt(k_or_v_weight.shape.get[1]())

    batch_size = input_row_offsets.dim[0]() - 1

    @parameter
    @__copy_capture(input_row_offsets, batch_size)
    @always_inline
    fn write_to_cache_common[
        dtype: DType, cache_t: KVCacheT, width: Int
    ](k_or_v_cache: cache_t, idx: IndexList[2], val: SIMD[dtype, width],):
        alias k_or_v_type = cache_t.dtype

        constrained[
            k_or_v_type == dtype,
            "Mismatch in dtype between hidden state and KV tensors",
        ]()

        # Token index in the "ragged" combined sequence dimension.
        global_token_idx = idx[0]

        batch_idx = get_batch_from_row_offsets(
            LayoutTensor[
                DType.uint32, Layout.row_major[1](input_row_offsets.shape)
            ](
                input_row_offsets.data,
                RuntimeLayout[
                    Layout.row_major[1](input_row_offsets.shape)
                ].row_major(input_row_offsets.get_shape().canonicalize()),
            ),
            global_token_idx,
        )
        token_idx = Int(global_token_idx - input_row_offsets[batch_idx])

        # Write this element to the K or V cache.
        cache = k_or_v_cache
        h_idx, hd_idx = divmod(UInt(idx[1]), kv_params.head_size)

        cache_length = cache.cache_length(batch_idx)
        cache_token_idx = token_idx + cache_length

        cache.store(
            batch_idx,
            h_idx,
            cache_token_idx,
            hd_idx,
            rebind[SIMD[k_or_v_type, width]](val),
        )

    @parameter
    @__copy_capture(k_or_v_cache)
    fn write_to_k_or_v_cache_continuous[
        dtype: DType, width: Int, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]):
        write_to_cache_common(k_or_v_cache, idx, val)

    _qmatmul_gguf_quantized_alloc_output[
        quantization_encoding,
        elementwise_lambda_fn=write_to_k_or_v_cache_continuous,
    ](hidden_state, k_or_v_weight)


@always_inline
fn _qmatmul_gguf_quantized_alloc_output[
    quantization_encoding: StaticString,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    hidden_state: NDBuffer[DType.float32, 2, _, _],
    weight: NDBuffer[DType.uint8, 2, _, _],
) raises:
    var TOTAL_SEQ_LEN = hidden_state.dim[0]()
    alias N = weight.shape.get[0]()
    alias K = weight.shape.get[1]()
    var c_nd: NDBuffer[DType.float32, 2, MutableAnyOrigin, DimList(Dim(), N)]

    # The CPU matmul codepath uses the C buffer as a workspace
    # even if an epilogue is provided, here we just allocate
    # something to ensure we don't segfault.
    var c_ptr = UnsafePointer[Float32].alloc(TOTAL_SEQ_LEN * N)

    c_nd = {c_ptr, IndexList[2](TOTAL_SEQ_LEN, N)}

    _qmatmul_gguf_quantized_common[
        quantization_encoding, elementwise_lambda_fn
    ](hidden_state, weight, c_nd)

    c_nd.data.free()


@always_inline
fn _qmatmul_gguf_quantized_common[
    quantization_encoding: StaticString,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    hidden_state: NDBuffer[DType.float32, 2, _, _],
    weight: NDBuffer[DType.uint8, 2, _, _],
    output: NDBuffer[mut=True, DType.float32, 2, _, _],
) raises:
    @parameter
    if quantization_encoding == "q4_0":
        matmul_qint4[32, elementwise_lambda_fn=elementwise_lambda_fn](
            hidden_state,
            weight,
            output,
        )
    elif quantization_encoding == "q4_k":
        matmul_Q4_K[elementwise_lambda_fn=elementwise_lambda_fn](
            hidden_state,
            weight,
            output,
        )
    elif quantization_encoding == "q6_k":
        matmul_Q6_K[elementwise_lambda_fn=elementwise_lambda_fn](
            hidden_state,
            weight,
            output,
        )
    else:
        raise Error(
            "Unsupported quantization encoding: ", quantization_encoding
        )


# ===-----------------------------------------------------------------------===#
# Fused QK RoPE (ragged)
# ===-----------------------------------------------------------------------===#


@always_inline
fn generic_fused_qk_rope_bshd_continuous_batch_ragged[
    dtype: DType,
    freq_dtype: DType, //,
    *,
    interleaved: Bool,
    has_position_ids: Bool,
    target: StaticString,
    mrope_section: Optional[IntTuple] = None,
](
    q_proj: NDBuffer[dtype, 3, *_],
    input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    kv_collection: ContinuousBatchingKVCacheCollection,
    freqs_cis: NDBuffer[freq_dtype, 2, *_],
    position_ids: NDBuffer[DType.uint32, 2, *_],
    layer_idx: UInt32,
    output: NDBuffer[mut=True, dtype, 3, *_],
    context: DeviceContextPtr,
) raises:
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
        "mo.fused_qk_rope.ragged.continuous_batching.nhead_"
        + String(kv_collection.kv_params.num_heads)
        + ".hdim_"
        + String(kv_collection.kv_params.head_size),
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=Int(context.get_device_context().id()),
    ):

        @parameter
        if has_position_ids:
            fused_qk_rope_ragged[
                kv_collection.CacheType,
                interleaved=interleaved,
                target=target,
                mrope_section=mrope_section,
            ](
                LayoutTensor[
                    q_proj.dtype, Layout.row_major[q_proj.rank](q_proj.shape)
                ](
                    q_proj.data,
                    RuntimeLayout[
                        Layout.row_major[q_proj.rank](q_proj.shape)
                    ].row_major(q_proj.get_shape().canonicalize()),
                ),
                LayoutTensor[
                    input_row_offsets.dtype,
                    Layout.row_major[input_row_offsets.rank](
                        input_row_offsets.shape
                    ),
                ](
                    input_row_offsets.data,
                    RuntimeLayout[
                        Layout.row_major[input_row_offsets.rank](
                            input_row_offsets.shape
                        )
                    ].row_major(input_row_offsets.get_shape().canonicalize()),
                ),
                kv_collection,
                LayoutTensor[
                    freqs_cis.type,
                    Layout.row_major[freqs_cis.rank](freqs_cis.shape),
                ](
                    freqs_cis.data,
                    RuntimeLayout[
                        Layout.row_major[freqs_cis.rank](freqs_cis.shape)
                    ].row_major(freqs_cis.get_shape().canonicalize()),
                ),
                LayoutTensor[
                    position_ids.dtype, Layout.row_major[2](), MutableAnyOrigin
                ](
                    position_ids.data,
                    RuntimeLayout[Layout.row_major[2]()].row_major(
                        position_ids.get_shape().canonicalize()
                    ),
                ),
                layer_idx,
                LayoutTensor[
                    output.dtype, Layout.row_major[output.rank](output.shape)
                ](
                    output.data,
                    RuntimeLayout[
                        Layout.row_major[output.rank](output.shape)
                    ].row_major(output.get_shape().canonicalize()),
                ),
                dev_ctx,
            )
        else:
            fused_qk_rope_ragged[
                kv_collection.CacheType, interleaved=interleaved, target=target
            ](
                LayoutTensor[
                    q_proj.dtype, Layout.row_major[q_proj.rank](q_proj.shape)
                ](
                    q_proj.data,
                    RuntimeLayout[
                        Layout.row_major[q_proj.rank](q_proj.shape)
                    ].row_major(q_proj.get_shape().canonicalize()),
                ),
                LayoutTensor[
                    input_row_offsets.dtype,
                    Layout.row_major[input_row_offsets.rank](
                        input_row_offsets.shape
                    ),
                ](
                    input_row_offsets.data,
                    RuntimeLayout[
                        Layout.row_major[input_row_offsets.rank](
                            input_row_offsets.shape
                        )
                    ].row_major(input_row_offsets.get_shape().canonicalize()),
                ),
                kv_collection,
                LayoutTensor[
                    freqs_cis.type,
                    Layout.row_major[freqs_cis.rank](freqs_cis.shape),
                ](
                    freqs_cis.data,
                    RuntimeLayout[
                        Layout.row_major[freqs_cis.rank](freqs_cis.shape)
                    ].row_major(freqs_cis.get_shape().canonicalize()),
                ),
                None,
                layer_idx,
                LayoutTensor[
                    output.dtype, Layout.row_major[output.rank](output.shape)
                ](
                    output.data,
                    RuntimeLayout[
                        Layout.row_major[output.rank](output.shape)
                    ].row_major(output.get_shape().canonicalize()),
                ),
                dev_ctx,
            )


@always_inline
fn generic_fused_qk_rope_bshd_paged_ragged[
    dtype: DType,
    freq_dtype: DType, //,
    *,
    interleaved: Bool,
    has_position_ids: Bool,
    target: StaticString,
    mrope_section: Optional[IntTuple] = None,
](
    q_proj: NDBuffer[dtype, 3, *_],
    input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    kv_collection: PagedKVCacheCollection,
    freqs_cis: NDBuffer[freq_dtype, 2, *_],
    position_ids: NDBuffer[DType.uint32, 2, *_],
    layer_idx: UInt32,
    output: NDBuffer[mut=True, dtype, 3, *_],
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

    alias name = "mo.fused_qk_rope.ragged.paged.nhead_" + String(
        kv_collection.kv_params.num_heads
    ) + ".hdim_" + String(kv_collection.kv_params.head_size)
    with Trace[TraceLevel.OP, target=target](
        name,
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=Int(context.get_device_context().id()),
    ):

        @parameter
        if has_position_ids:
            fused_qk_rope_ragged[
                kv_collection.CacheType,
                interleaved=interleaved,
                target=target,
                mrope_section=mrope_section,
            ](
                LayoutTensor[
                    q_proj.dtype, Layout.row_major[q_proj.rank](q_proj.shape)
                ](
                    q_proj.data,
                    RuntimeLayout[
                        Layout.row_major[q_proj.rank](q_proj.shape)
                    ].row_major(q_proj.get_shape().canonicalize()),
                ),
                LayoutTensor[
                    input_row_offsets.dtype,
                    Layout.row_major[input_row_offsets.rank](
                        input_row_offsets.shape
                    ),
                ](
                    input_row_offsets.data,
                    RuntimeLayout[
                        Layout.row_major[input_row_offsets.rank](
                            input_row_offsets.shape
                        )
                    ].row_major(input_row_offsets.get_shape().canonicalize()),
                ),
                kv_collection,
                LayoutTensor[
                    freqs_cis.type,
                    Layout.row_major[freqs_cis.rank](freqs_cis.shape),
                ](
                    freqs_cis.data,
                    RuntimeLayout[
                        Layout.row_major[freqs_cis.rank](freqs_cis.shape)
                    ].row_major(freqs_cis.get_shape().canonicalize()),
                ),
                LayoutTensor[
                    position_ids.dtype, Layout.row_major[2](), MutableAnyOrigin
                ](
                    position_ids.data,
                    RuntimeLayout[Layout.row_major[2]()].row_major(
                        position_ids.get_shape().canonicalize()
                    ),
                ),
                layer_idx,
                LayoutTensor[
                    output.dtype, Layout.row_major[output.rank](output.shape)
                ](
                    output.data,
                    RuntimeLayout[
                        Layout.row_major[output.rank](output.shape)
                    ].row_major(output.get_shape().canonicalize()),
                ),
                dev_ctx,
            )
        else:
            fused_qk_rope_ragged[
                kv_collection.CacheType, interleaved=interleaved, target=target
            ](
                LayoutTensor[
                    q_proj.dtype, Layout.row_major[q_proj.rank](q_proj.shape)
                ](
                    q_proj.data,
                    RuntimeLayout[
                        Layout.row_major[q_proj.rank](q_proj.shape)
                    ].row_major(q_proj.get_shape().canonicalize()),
                ),
                LayoutTensor[
                    input_row_offsets.dtype,
                    Layout.row_major[input_row_offsets.rank](
                        input_row_offsets.shape
                    ),
                ](
                    input_row_offsets.data,
                    RuntimeLayout[
                        Layout.row_major[input_row_offsets.rank](
                            input_row_offsets.shape
                        )
                    ].row_major(input_row_offsets.get_shape().canonicalize()),
                ),
                kv_collection,
                LayoutTensor[
                    freqs_cis.type,
                    Layout.row_major[freqs_cis.rank](freqs_cis.shape),
                ](
                    freqs_cis.data,
                    RuntimeLayout[
                        Layout.row_major[freqs_cis.rank](freqs_cis.shape)
                    ].row_major(freqs_cis.get_shape().canonicalize()),
                ),
                None,
                layer_idx,
                LayoutTensor[
                    output.dtype, Layout.row_major[output.rank](output.shape)
                ](
                    output.data,
                    RuntimeLayout[
                        Layout.row_major[output.rank](output.shape)
                    ].row_major(output.get_shape().canonicalize()),
                ),
                dev_ctx,
            )


# ===-----------------------------------------------------------------------===#
# MHA (ragged)
# ===-----------------------------------------------------------------------===#


@always_inline
fn generic_flash_attention_kv_cache_ragged[
    collection_t: KVCollectionT,
    dtype: DType, //,
    *,
    target: StaticString,
    mask_str: StaticString,
    score_mod_str: StaticString,
    local_window_size: Int = -1,
](
    q: NDBuffer[dtype, 3, *_],
    input_row_offsets: ManagedTensorSlice[dtype = DType.uint32, rank=1],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    output: NDBuffer[mut=True, dtype, 3, *_],
    context: DeviceContextPtr,
) raises:
    @always_inline
    @parameter
    fn description_fn() -> String:
        var desc_parts = List[String]()
        desc_parts.append(trace_arg("q", q))
        desc_parts.append("scale=" + String(scale))
        desc_parts.append("layer_idx=" + String(layer_idx))
        desc_parts.append(
            "num_heads=" + String(collection_t.kv_params.num_heads)
        )
        desc_parts.append(
            "head_size=" + String(collection_t.kv_params.head_size)
        )
        desc_parts.append("local_window_size=" + String(local_window_size))
        desc_parts.append("sink=False")
        return String(";").join(desc_parts)

    alias name = "mo.mha.ragged." + collection_t.name_str + "." + mask_str + "." + score_mod_str + ".nhead_" + String(
        collection_t.kv_params.num_heads
    ) + ".hdim_" + String(
        collection_t.kv_params.head_size
    )

    with Trace[TraceLevel.OP, target=target](
        name,
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=Int(context.get_device_context().id()),
    ):
        return _flash_attention_dispatch[
            target=target,
            mask_str=mask_str,
            score_mod_str=score_mod_str,
            local_window_size=local_window_size,
        ](
            q,
            input_row_offsets,
            kv_collection,
            layer_idx,
            scale,
            output,
            context,
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
    q: NDBuffer[dtype, 3, *_],
    input_row_offsets: ManagedTensorSlice[dtype = DType.uint32, rank=1],
    kv_cache: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    output: NDBuffer[mut=True, dtype, 3, *_],
    context: DeviceContextPtr,
    sink_weights: OptionalReg[NDBuffer[dtype, 1, MutableAnyOrigin]] = None,
) raises:
    var k = kv_cache.get_key_cache(Int(layer_idx))
    var v = kv_cache.get_value_cache(Int(layer_idx))

    var has_inputs = q.dim[0]() > 0
    if not has_inputs:
        # no-op if there are no inputs
        return

    @parameter
    @__copy_capture(k, v)
    fn _dispatch_flash_attention[
        mask_t: MHAMask, score_mod_t: ScoreModTrait
    ](mask: mask_t, score_mod: score_mod_t) raises:
        @parameter
        fn call_flash_attention[sink: Bool]() raises:
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

            @parameter
            if is_cpu[target]():
                alias q_layout = Layout.row_major[q.rank](q.shape)
                alias output_layout = Layout.row_major[output.rank](
                    output.shape
                )
                return flash_attention_kv_cache_cpu(
                    LayoutTensor[q.type, q_layout](
                        q.data,
                        RuntimeLayout[q_layout].row_major(
                            q.dynamic_shape.canonicalize()
                        ),
                    ),
                    input_row_offsets.to_layout_tensor(),
                    input_row_offsets.to_layout_tensor(),
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
                gpu_flash_attention[
                    use_score_mod=use_score_mod, ragged=True, sink=sink
                ](
                    LayoutTensor[
                        output.type,
                        Layout(
                            IntTuple(output.shape), IntTuple(output.strides)
                        ),
                    ](
                        output.data,
                        RuntimeLayout[
                            Layout(
                                IntTuple(output.shape), IntTuple(output.strides)
                            )
                        ](
                            output.get_shape().canonicalize(),
                            output.get_strides().canonicalize(),
                        ),
                    ),
                    LayoutTensor[
                        q.type,
                        Layout(IntTuple(q.shape), IntTuple(q.strides)),
                    ](
                        q.data,
                        RuntimeLayout[
                            Layout(IntTuple(q.shape), IntTuple(q.strides))
                        ](
                            q.get_shape().canonicalize(),
                            q.get_strides().canonicalize(),
                        ),
                    ),
                    k,
                    v,
                    mask,
                    score_mod,
                    input_row_offsets,
                    scale,
                    context.get_device_context(),
                    sink_weights=sink_weights_lt,
                )

        unswitch[call_flash_attention](Bool(sink_weights))

    return dispatch_mask_and_score_mod[
        mask_str,
        score_mod_str,
        _dispatch_flash_attention,
        local_window_size,
        collection_t.kv_params.num_heads,
    ]()


@always_inline
fn generic_flash_attention_kv_cache_ragged_sink[
    collection_t: KVCollectionT,
    dtype: DType, //,
    *,
    target: StaticString,
    mask_str: StaticString,
    score_mod_str: StaticString,
    local_window_size: Int = -1,
](
    q: NDBuffer[dtype, 3, *_],
    input_row_offsets: ManagedTensorSlice[dtype = DType.uint32, rank=1],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    output: NDBuffer[mut=True, dtype, 3, *_],
    context: DeviceContextPtr,
    sink_weights: NDBuffer[dtype, 1, MutableAnyOrigin],
) raises:
    @always_inline
    @parameter
    fn description_fn() -> String:
        var desc_parts = List[String]()
        desc_parts.append(trace_arg("q", q))
        desc_parts.append("scale=" + String(scale))
        desc_parts.append("layer_idx=" + String(layer_idx))
        desc_parts.append(
            "num_heads=" + String(collection_t.kv_params.num_heads)
        )
        desc_parts.append(
            "head_size=" + String(collection_t.kv_params.head_size)
        )
        desc_parts.append("local_window_size=" + String(local_window_size))
        desc_parts.append("sink=True")
        return String(";").join(desc_parts)

    alias name = "mo.mha.ragged." + collection_t.name_str + "." + mask_str + "." + score_mod_str + ".nhead_" + String(
        collection_t.kv_params.num_heads
    ) + ".hdim_" + String(
        collection_t.kv_params.head_size
    )

    with Trace[TraceLevel.OP, target=target](
        name,
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=Int(context.get_device_context().id()),
    ):
        return _flash_attention_dispatch[
            target=target,
            mask_str=mask_str,
            score_mod_str=score_mod_str,
            local_window_size=local_window_size,
        ](
            q,
            input_row_offsets,
            kv_collection,
            layer_idx,
            scale,
            output,
            context,
            sink_weights,
        )


# ===-----------------------------------------------------------------------===#
# MLA (ragged)
# ===-----------------------------------------------------------------------===#


@always_inline
fn generic_flare_mla_decode_kv_cache_ragged[
    collection_t: KVCollectionT,
    dtype: DType, //,
    mask_str: StaticString,
    score_mod_str: StaticString,
    target: StaticString,
    local_window_size: Int = -1,
](
    q: NDBuffer[dtype, 3, *_],
    input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    output: NDBuffer[mut=True, dtype, 3, *_],
    context: DeviceContextPtr,
) raises:
    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            trace_arg("q", q),
            "scale=" + String(scale),
            "layer_idx=" + String(layer_idx),
            "num_heads=" + String(collection_t.kv_params.num_heads),
            "head_size=" + String(collection_t.kv_params.head_size),
        )

    with Trace[TraceLevel.OP, target=target](
        "mo.mla.decode.ragged."
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
        task_id=Int(context.get_device_context().id()),
    ):
        return _flare_mla_decode_kv_cache_ragged[
            target=target,
            mask_str=mask_str,
            score_mod_str=score_mod_str,
            local_window_size=local_window_size,
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
fn _flare_mla_decode_kv_cache_ragged[
    dtype: DType,
    collection_t: KVCollectionT, //,
    mask_str: StaticString,
    score_mod_str: StaticString,
    target: StaticString,
    local_window_size: Int = -1,
](
    q: NDBuffer[dtype, 3, *_],
    input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    output: NDBuffer[mut=True, dtype, 3, *_],
    context: DeviceContextPtr,
) raises:
    """Performs flash attention using k and v caches from KVCacheT custom dtypes.

    Args:
        q: NDBuffer with shape (batch_size, num_heads, seq_len, head_size).
        input_row_offsets: The start and end position of each Q entry in the batch.
        kv_collection: The Collection object storing out KVCache entries for this layer
        layer_idx: The current layer, used to retrieve kv_cache objects from kv_collection
        scale: The scaled factor in scaled-dot product attention. Usually rsqrt(head_size).
        output: The Pre-allocated output buffer to write results to. Has shape:
            (batch_size, num_heads, seq_len, head_size).
        context: Pointer containing the runtime context for the target device.
    """
    constrained[is_gpu[target](), "MLA is only supported on GPU"]()

    var layer_idx_cast = Int(layer_idx)
    var k = kv_collection.get_key_cache(layer_idx_cast)

    @parameter
    @always_inline
    @__copy_capture(k)
    fn _dispatch_mla[
        mask_t: MHAMask, score_mod_t: ScoreModTrait
    ](mask: mask_t, score_mod: score_mod_t) raises:
        flare_mla_decoding[ragged=True](
            output,
            q,
            k,
            mask,
            score_mod,
            input_row_offsets,
            scale,
            context.get_device_context(),
        )

    dispatch_mask_and_score_mod[
        mask_str,
        score_mod_str,
        _dispatch_mla,
        local_window_size,
        collection_t.kv_params.num_heads,
    ]()


@always_inline
fn generic_flare_mla_prefill_kv_cache_ragged[
    collection_t: KVCollectionT,
    dtype: DType, //,
    softmax_type: DType,
    write_softmax_info: Bool,
    use_cascade_attention: Bool,
    mask_str: StaticString,
    score_mod_str: StaticString,
    target: StaticString,
    local_window_size: Int = -1,
](
    q: NDBuffer[dtype, 3, *_],
    k: NDBuffer[dtype, 3, *_],
    v: NDBuffer[dtype, 3, *_],
    buffer_row_offsets: NDBuffer[DType.uint32, 1, *_],
    cache_offsets: NDBuffer[DType.uint32, 1, *_],
    input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    output: NDBuffer[mut=True, dtype, 3, *_],
    softmax_info: NDBuffer[mut=True, softmax_type, 3, MutableAnyOrigin],
    context: DeviceContextPtr,
    prev_output: OptionalReg[NDBuffer[dtype, 3, MutableAnyOrigin]] = None,
    prev_softmax_info: OptionalReg[
        NDBuffer[softmax_type, 3, MutableAnyOrigin]
    ] = None,
) raises:
    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            trace_arg("q", q),
            trace_arg("k", k),
            trace_arg("v", v),
            trace_arg("buffer_row_offsets", buffer_row_offsets),
            trace_arg("cache_offsets", cache_offsets),
            trace_arg("input_row_offsets", input_row_offsets),
            "scale=" + String(scale),
            "layer_idx=" + String(layer_idx),
            "num_heads=" + String(collection_t.kv_params.num_heads),
            "head_size=" + String(collection_t.kv_params.head_size),
        )

    with Trace[TraceLevel.OP, target=target](
        "mo.mla.prefill.ragged."
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
        task_id=Int(context.get_device_context().id()),
    ):
        return _flare_mla_prefill_kv_cache_ragged[
            write_softmax_info=write_softmax_info,
            use_cascade_attention=use_cascade_attention,
            mask_str=mask_str,
            score_mod_str=score_mod_str,
            target=target,
            local_window_size=local_window_size,
        ](
            q,
            k,
            v,
            buffer_row_offsets,
            cache_offsets,
            input_row_offsets,
            kv_collection,
            layer_idx,
            scale,
            output,
            softmax_info,
            context,
            prev_output,
            prev_softmax_info,
        )


@always_inline
fn _flare_mla_prefill_kv_cache_ragged[
    dtype: DType,
    collection_t: KVCollectionT, //,
    softmax_type: DType,
    mask_str: StaticString,
    score_mod_str: StaticString,
    write_softmax_info: Bool,
    use_cascade_attention: Bool,
    target: StaticString,
    local_window_size: Int = -1,
](
    q: NDBuffer[dtype, 3, *_],
    k: NDBuffer[dtype, 3, *_],
    v: NDBuffer[dtype, 3, *_],
    buffer_row_offsets: NDBuffer[DType.uint32, 1, *_],
    cache_offsets: NDBuffer[DType.uint32, 1, *_],
    input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    output: NDBuffer[mut=True, dtype, 3, *_],
    softmax_info: NDBuffer[mut=True, softmax_type, 3, MutableAnyOrigin],
    context: DeviceContextPtr,
    prev_output: OptionalReg[NDBuffer[dtype, 3, MutableAnyOrigin]] = None,
    prev_softmax_info: OptionalReg[
        NDBuffer[softmax_type, 3, MutableAnyOrigin]
    ] = None,
) raises:
    """Performs MLA prefill.

    Args:
        q: NDBuffer with shape (total_seq_len, num_heads, q_head_size).
        k: NDBuffer with shape (total_seq_len, num_heads, kv_head_size).
        v: NDBuffer with shape (total_seq_len, num_heads, kv_head_size).
        buffer_row_offsets: The start and end position of each K entry in the ragged K/V tensor.
        cache_offsets: The start position of each K entry in the PagedKVCacheCollection.
        input_row_offsets: The start and end position of each Q entry in the batch.
        kv_collection: The Collection object storing out KVCache entries for this layer
        layer_idx: The current layer, used to retrieve kv_cache objects from kv_collection
        scale: The scaled factor in scaled-dot product attention. Usually rsqrt(head_size).
        output: The Pre-allocated output buffer to write results to. Has shape:
            (total_seq_len, num_heads, kv_head_size).
        softmax_info: NDBuffer with shape (total_seq_len, num_heads, 2).
        context: Pointer containing the runtime context for the target device.
        prev_output: Optional tensor that stores the temporal results for the previous
            prefill iteration.
        prev_softmax_info: Optional tensor that stores the temporal softmax info for the
            previous prefill iteration.
    """
    constrained[is_gpu[target](), "MLA is only supported on GPU"]()

    var layer_idx_cast = Int(layer_idx)
    var k_rope = kv_collection.get_key_cache(layer_idx_cast)

    @parameter
    @__copy_capture(k_rope)
    fn _mla_dispatch[
        mask_t: MHAMask, score_mod_t: ScoreModTrait
    ](mask: mask_t, score_mod: score_mod_t) raises:
        flare_mla_prefill[
            write_softmax_info=write_softmax_info,
            use_cascade_attention=use_cascade_attention,
        ](
            output,
            q,
            k,
            v,
            k_rope,
            mask,
            score_mod,
            input_row_offsets,
            buffer_row_offsets,
            scale,
            context.get_device_context(),
            softmax_info=OptionalReg[
                NDBuffer[softmax_type, 3, MutableAnyOrigin]
            ](softmax_info),
            cache_offsets=OptionalReg[
                NDBuffer[DType.uint32, 1, MutableAnyOrigin]
            ](cache_offsets),
            prev_output=prev_output,
            prev_softmax_info=prev_softmax_info,
        )

    dispatch_mask_and_score_mod[
        mask_str,
        score_mod_str,
        _mla_dispatch,
        local_window_size,
        collection_t.kv_params.num_heads,
    ]()


@always_inline
fn generic_flare_mla_prefill_ragged_paged_plan[
    target: StaticString
](
    input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    kv_collection: PagedKVCacheCollection,
    layer_idx: UInt32,
    buffer_token_size: UInt32,
    buffer_row_offsets: NDBuffer[mut=True, DType.uint32, 2, *_],
    cache_offsets: NDBuffer[mut=True, DType.uint32, 2, *_],
    buffer_lengths: NDBuffer[mut=True, DType.int32, 1, *_],
    context: DeviceContextPtr,
) raises:
    constrained[is_gpu[target](), "Planning MLA is only supported on GPU"]()

    var cuda_ctx = context.get_device_context()

    var layer_idx_cast = Int(layer_idx)

    var k = kv_collection.get_key_cache(layer_idx_cast)

    with Trace[TraceLevel.OP, target=target](
        "mo.mla.prefill.ragged.paged.plan",
        task_id=Int(context.get_device_context().id()),
    ):
        mla_prefill_plan(
            buffer_row_offsets,
            cache_offsets,
            buffer_lengths,
            input_row_offsets,
            k,
            buffer_token_size,
            cuda_ctx,
        )


@always_inline
fn generic_flare_mla_decompress_k_cache_ragged_paged[
    target: StaticString, dtype: DType
](
    buffer_row_offsets_1d: NDBuffer[DType.uint32, 1, *_],
    cache_offsets_1d: NDBuffer[DType.uint32, 1, *_],
    buffer_length: Int32,
    weight: NDBuffer[dtype, 2, *_],
    kv_collection: PagedKVCacheCollection,
    layer_idx: UInt32,
    k_latent_buffer: NDBuffer[mut=True, dtype, 2, *_],
    k_buffer: NDBuffer[mut=True, dtype, 2, *_],
    context: DeviceContextPtr,
) raises:
    constrained[is_gpu[target](), "MLA is only supported on GPU"]()
    var cuda_ctx = context.get_device_context()

    var buffer_length_int = Int(buffer_length)
    var layer_idx_cast = Int(layer_idx)
    var k = kv_collection.get_key_cache(layer_idx_cast)

    _k_cache_to_buffer(
        buffer_row_offsets_1d,
        cache_offsets_1d,
        k,
        buffer_length_int,
        k_latent_buffer,
        cuda_ctx,
    )

    # rebind k_latent_buffer with dynamic dim
    alias latent_last_dim = k_latent_buffer.shape.get[1]()
    alias k_latent_shape = DimList(
        VariadicList[Dim](Dim(), Dim(latent_last_dim))
    )
    var k_latent_dynamic_shape = IndexList[2](
        buffer_length_int, latent_last_dim
    )

    var k_latent_buffer_dynamic = NDBuffer[
        dtype,
        2,
        k_latent_buffer.origin,
        k_latent_shape,
        k_latent_buffer.strides,
    ](k_latent_buffer.data, k_latent_dynamic_shape)

    # rebind k_buffer with dynamic dim
    alias k_last_dim = k_buffer.shape.get[1]()
    alias k_shape = DimList(VariadicList[Dim](Dim(), Dim(k_last_dim)))
    var k_dynamic_shape = IndexList[2](buffer_length_int, k_last_dim)

    var k_buffer_dynamic = NDBuffer[
        dtype, 2, k_buffer.origin, k_shape, k_buffer.strides
    ](k_buffer.data, k_dynamic_shape)

    matmul[
        target=target,
        transpose_b=True,
    ](k_buffer_dynamic, k_latent_buffer_dynamic, weight, Optional(cuda_ctx))


# ===-----------------------------------------------------------------------===#
# Cross attention (ragged)
# ===-----------------------------------------------------------------------===#


fn _cross_attention_dispatch[
    dtype: DType,
    collection_t: KVCollectionT, //,
    *,
    target: StaticString,
    mask_str: StaticString,
    score_mod_str: StaticString,
    local_window_size: Int = -1,
](
    q: NDBuffer[dtype, 3, *_],
    q_input_row_offsets: ManagedTensorSlice[dtype = DType.uint32, rank=1],
    q_max_seq_len: UInt32,
    kv_input_row_offsets: NDBuffer[DType.uint32, 1],
    kv_cache: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    output: NDBuffer[mut=True, dtype, 3, *_],
    context: DeviceContextPtr,
    sink_weights: OptionalReg[NDBuffer[dtype, 1, MutableAnyOrigin]] = None,
) raises:
    var k = kv_cache.get_key_cache(Int(layer_idx))
    var v = kv_cache.get_value_cache(Int(layer_idx))

    @parameter
    @__copy_capture(
        q, k, v, output, context, q_input_row_offsets, kv_input_row_offsets
    )
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
                q_input_row_offsets.to_layout_tensor(),
                # Use KV offsets for cross attention.
                LayoutTensor[
                    kv_input_row_offsets.type, Layout.row_major(UNKNOWN_VALUE)
                ](
                    kv_input_row_offsets.data,
                    RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
                        IndexList[1](len(kv_input_row_offsets))
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
            gpu_flash_attention[
                use_score_mod=use_score_mod, ragged=True, sink=False
            ](
                LayoutTensor[
                    output.type,
                    Layout(IntTuple(output.shape), IntTuple(output.strides)),
                ](
                    output.data,
                    RuntimeLayout[
                        Layout(IntTuple(output.shape), IntTuple(output.strides))
                    ](
                        output.get_shape().canonicalize(),
                        output.get_strides().canonicalize(),
                    ),
                ),
                LayoutTensor[
                    q.type,
                    Layout(IntTuple(q.shape), IntTuple(q.strides)),
                ](
                    q.data,
                    RuntimeLayout[
                        Layout(IntTuple(q.shape), IntTuple(q.strides))
                    ](
                        q.get_shape().canonicalize(),
                        q.get_strides().canonicalize(),
                    ),
                ),
                k,
                v,
                mask,
                IdentityScoreMod(),
                q_input_row_offsets,
                scale,
                context.get_device_context(),
                Int(q_max_seq_len),
                LayoutTensor[
                    kv_input_row_offsets.type,
                    Layout.row_major(UNKNOWN_VALUE),
                    MutableAnyOrigin,
                ](
                    kv_input_row_offsets.data,
                    RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
                        IndexList[1](len(kv_input_row_offsets))
                    ),
                ),
                None,
            )

    return dispatch_mask_and_score_mod[
        mask_str,
        score_mod_str,
        _dispatch_flash_attention,
        local_window_size,
        collection_t.kv_params.num_heads,
    ]()


@always_inline
fn generic_cross_attention_kv_cache[
    collection_t: KVCollectionT,
    dtype: DType, //,
    target: StaticString,
    mask_str: StaticString,
    score_mod_str: StaticString,
    local_window_size: Int = -1,
](
    q: NDBuffer[mut=True, dtype, 3, *_],
    q_input_row_offsets: ManagedTensorSlice[dtype = DType.uint32, rank=1],
    q_max_seq_len: NDBuffer[DType.uint32, 1, *_],
    kv_input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    output: NDBuffer[mut=True, dtype, 3, *_],
    context: DeviceContextPtr,
    sink_weights: OptionalReg[NDBuffer[dtype, 1, MutableAnyOrigin]] = None,
) raises:
    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            trace_arg("output", output),
            trace_arg("q", q),
            trace_slice_arg("q_input_row_offsets", q_input_row_offsets),
            trace_arg("kv_input_row_offsets", kv_input_row_offsets),
            "layer_idx=" + String(layer_idx),
            "num_heads=" + String(collection_t.kv_params.num_heads),
            "head_size=" + String(collection_t.kv_params.head_size),
        )

    with Trace[TraceLevel.OP, target=target](
        "mo.cross_attention.ragged."
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
        task_id=Int(context.get_device_context().id()),
    ):
        return _cross_attention_dispatch[
            target=target,
            mask_str=mask_str,
            score_mod_str=score_mod_str,
            local_window_size=local_window_size,
        ](
            q,
            q_input_row_offsets,
            q_max_seq_len[0],
            kv_input_row_offsets,
            kv_collection,
            layer_idx,
            scale,
            output,
            context,
            sink_weights,
        )


# TODO: Remove this when we're no longer using NDBuffers.
@always_inline
fn valid_length_managed_tensor_slice_to_ndbuffer(
    tensor: ManagedTensorSlice[dtype = DType.uint32, rank=1]
) -> NDBuffer[DType.uint32, 1, MutableAnyOrigin]:
    var ptr = tensor._ptr.address_space_cast[AddressSpace.GENERIC]()
    return NDBuffer[DType.uint32, 1, MutableAnyOrigin](
        ptr, tensor.shape(), tensor._runtime_strides
    )


# ===-----------------------------------------------------------------------===#
# KV cache ragged radd dispatch
# ===-----------------------------------------------------------------------===#


fn generic_kv_cache_radd_dispatch[
    dtype: DType,
    collection_t: KVCollectionT, //,
    target: StaticString,
](
    a: NDBuffer[dtype, 2, _, _],
    cache: collection_t,
    input_row_offsets: NDBuffer[DType.uint32, 1, _, _],
    batch_offset: UInt32,
    layer_idx: UInt32,
    ctx: Optional[DeviceContext],
) raises:
    alias hidden_size = collection_t.kv_params.head_size * collection_t.kv_params.num_heads

    constrained[
        dtype == collection_t.dtype,
        "Mismatch in dtype between computation and KV tensors",
    ]()
    constrained[
        a.shape.at[1]().has_value(),
        "Input tensor must have known shape in last dim",
    ]()
    constrained[
        a.shape.get[1]() == hidden_size * 2,
        "Mismatch in hidden size between input "
        + String(a.shape.get[1]())
        + " and KV tensors "
        + String(hidden_size),
    ]()

    var layer_idx_cast = Int(layer_idx)
    var k_cache = cache.get_key_cache(layer_idx_cast)
    var v_cache = cache.get_value_cache(layer_idx_cast)

    @parameter
    @__copy_capture(k_cache, v_cache, input_row_offsets)
    fn do_radd[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        constrained[rank == 2, "Rank must be 2"]()

        # we could be slicing the batch, so we need to add the offset to get the actual index in the flattened batch
        var corrected_token_idx = idx[0] + input_row_offsets[0]
        var batch_idx = get_batch_from_row_offsets(
            LayoutTensor[
                DType.uint32, Layout.row_major[1](input_row_offsets.shape)
            ](
                input_row_offsets.data,
                RuntimeLayout[
                    Layout.row_major[1](input_row_offsets.shape)
                ].row_major(input_row_offsets.get_shape().canonicalize()),
            ),
            Int(corrected_token_idx),
        )

        # we also need to add the batch offset to get the actual index in the flattened batch
        var corrected_batch_idx = batch_idx + batch_offset
        var tok_idx = Int(corrected_token_idx - input_row_offsets[batch_idx])

        var cache: collection_t.CacheType
        var corrected_dim: UInt
        if idx[1] < hidden_size:
            cache = k_cache
            corrected_dim = UInt(idx[1])
        else:
            cache = v_cache
            corrected_dim = UInt(idx[1] - hidden_size)

        var h_idx: UInt
        var hd_idx: UInt
        h_idx, hd_idx = divmod(
            UInt(corrected_dim), collection_t.kv_params.head_size
        )

        var cache_length = cache.cache_length(Int(corrected_batch_idx))
        var cache_token_idx = Int(tok_idx) + cache_length

        var old_val = cache.load[width=width](
            Int(corrected_batch_idx), h_idx, cache_token_idx, hd_idx
        )
        var a_val = rebind[__type_of(old_val)](a.load[width=width](idx))

        cache.store(
            Int(corrected_batch_idx),
            h_idx,
            cache_token_idx,
            hd_idx,
            a_val + old_val,
        )

    @parameter
    if is_gpu[target]():
        debug_assert(ctx is not None, "ctx is None")
        alias compile_target = get_gpu_target()
        alias simd_width = simd_width_of[dtype, target=compile_target]()

        elementwise[do_radd, simd_width, target=target](
            a.dynamic_shape, ctx.value()
        )
    else:
        alias compile_target = _current_target()
        alias simd_width = simd_width_of[dtype, target=compile_target]()

        elementwise[do_radd, simd_width, target=target](a.dynamic_shape)


fn kv_cache_store_ragged[
    cache_t: KVCacheT,
    input_row_offsets_layout: Layout, //,
    target: StaticString,
    input_fn: fn[width: Int, alignment: Int] (
        idx: IndexList[3]
    ) capturing -> SIMD[cache_t.dtype, width],
](
    cache: cache_t,
    input_shape: IndexList[3],
    input_row_offsets: LayoutTensor[
        DType.uint32, input_row_offsets_layout, MutableAnyOrigin
    ],
    context: Optional[DeviceContext],
) raises:
    constrained[
        input_row_offsets.layout.rank() == 1,
        (
            "Expected input_row_offsets to be a 1D tensor of shape `(batch_size"
            " + 1,)`"
        ),
    ]()

    @parameter
    @__copy_capture(cache)
    fn write_to_cache[
        width: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank]):
        var loaded_val = input_fn[width=width, alignment=alignment](
            rebind[IndexList[3]](idx)
        )
        var batch_idx = get_batch_from_row_offsets(input_row_offsets, idx[0])
        var token_idx = Int(idx[0] - input_row_offsets[batch_idx])
        var h_idx = idx[1]
        var hd_idx = idx[2]
        var cache_length = cache.cache_length(batch_idx)
        var cache_token_idx = token_idx + cache_length
        cache.store(
            batch_idx,
            h_idx,
            cache_token_idx,
            hd_idx,
            loaded_val,
        )

    @parameter
    if is_gpu[target]():
        debug_assert(context is not None, "ctx is None")
        alias compile_target = get_gpu_target()
        alias simd_width = simd_width_of[cache_t.dtype, target=compile_target]()

        elementwise[write_to_cache, simd_width, target=target](
            input_shape, context.value()
        )
    else:
        alias compile_target = _current_target()
        alias simd_width = simd_width_of[cache_t.dtype, target=compile_target]()

        elementwise[write_to_cache, simd_width, target=target](input_shape)
