# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
from sys.intrinsics import _type_is_eq

from algorithm.functional import unswitch
from gpu.host import DeviceContext, DeviceBuffer
from gpu.host.info import is_cpu, is_gpu
from collections import OptionalReg
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
    KVCacheT,
    KVCollectionT,
    PagedKVCacheCollection,
)
from layout import UNKNOWN_VALUE, Layout, LayoutTensor, RuntimeLayout
from layout._coord import Coord, Idx, coord_to_index_list
from layout._layout import row_major
from layout._tile_tensor import TileTensor
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

from utils import Index, IndexList
from tensor import InputTensor
from tensor.managed_tensor_slice import (
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
    hidden_state: LayoutTensor[
        dtype, address_space = AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    kv_collection: ContinuousBatchingKVCacheCollection,
    layer_idx: UInt32,
    valid_lengths: LayoutTensor[
        DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ],
    output: LayoutTensor[mut=True, dtype, ...],
    ctx: DeviceContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Only positions within valid_lengths are written to the KV cache.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The historical KVCache for keys and values. The KVCache for
            this layer is retrieved via layer_idx.
        layer_idx: The index of the layer being executed. Used to retrieve the KVCache
            for the given layer from kv_collection.
        valid_lengths: Tensor of shape [batch] containing the valid length for each
            sequence. K and V are only written to cache for positions within these lengths.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
        ctx: The call context pointer, passed by the graph compiler.
    """

    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("output", output.runtime_layout.shape.value),
                    trace_arg(
                        "hidden_state", hidden_state.runtime_layout.shape.value
                    ),
                    trace_arg("weight", weight.runtime_layout.shape.value),
                    trace_arg(
                        "valid_lengths",
                        valid_lengths.runtime_layout.shape.value,
                    ),
                    "layer_idx=" + String(layer_idx),
                    "num_heads=" + String(kv_collection.kv_params.num_heads),
                    "head_size=" + String(kv_collection.kv_params.head_size),
                ]
            )
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
        ](
            hidden_state,
            weight,
            kv_collection,
            layer_idx,
            valid_lengths,
            output,
            ctx,
        )


@always_inline
fn generic_fused_qkv_matmul_kv_cache_bshd_paged[
    dtype: DType,
    target: StaticString = "cpu",
](
    hidden_state: LayoutTensor[
        dtype, address_space = AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    kv_collection: PagedKVCacheCollection,
    layer_idx: UInt32,
    valid_lengths: LayoutTensor[
        DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ],
    output: LayoutTensor[mut=True, dtype, ...],
    ctx: DeviceContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Only positions within valid_lengths are written to the KV cache.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The historical KVCache for keys and values. The KVCache for
            this layer is retrieved via layer_idx.
        layer_idx: The index of the layer being executed. Used to retrieve the KVCache
            for the given layer from kv_collection.
        valid_lengths: Tensor of shape [batch] containing the valid length for each
            sequence. K and V are only written to cache for positions within these lengths.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
        ctx: The call context pointer, passed by the graph compiler.
    """

    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("output", output.runtime_layout.shape.value),
                    trace_arg(
                        "hidden_state", hidden_state.runtime_layout.shape.value
                    ),
                    trace_arg("weight", weight.runtime_layout.shape.value),
                    trace_arg(
                        "valid_lengths",
                        valid_lengths.runtime_layout.shape.value,
                    ),
                    "layer_idx=" + String(layer_idx),
                    "num_heads=" + String(kv_collection.kv_params.num_heads),
                    "head_size=" + String(kv_collection.kv_params.head_size),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=target](
        "mo.fused_qkv_matmul.padded.paged.nhead_"
        + String(kv_collection.kv_params.num_heads)
        + ".hdim_"
        + String(kv_collection.kv_params.head_size),
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=get_safe_task_id(ctx),
    ):
        return _fused_qkv_matmul_kv_cache[
            kv_collection.CacheType, target=target
        ](
            hidden_state,
            weight,
            kv_collection,
            layer_idx,
            valid_lengths,
            output,
            ctx,
        )


@always_inline
fn _fused_qkv_matmul_kv_cache[
    dtype: DType,
    collection_t: KVCollectionT,
    //,
    cache_t: KVCacheT,
    *,
    target: StaticString,
](
    hidden_state: LayoutTensor[
        dtype, address_space = AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    kv_collection: collection_t,
    layer_idx: UInt32,
    valid_lengths: LayoutTensor[
        DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ],
    output: LayoutTensor[mut=True, dtype, ...],
    context: DeviceContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Only positions within valid_lengths are written to the KV cache.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The historical KVCache for keys and values. The KVCache for
            this layer is retrieved via layer_idx.
        layer_idx: The index of the layer being executed. Used to retrieve the KVCache
            for the given layer from kv_collection.
        valid_lengths: Tensor of shape [batch] containing the valid length for each
            sequence. K and V are only written to cache for positions within these lengths.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
        context: The call context pointer, passed by the graph compiler.
    """
    var cuda_ctx: Optional[DeviceContext] = None

    comptime if is_gpu[target]():
        cuda_ctx = context.get_device_context()

    return _fused_qkv_matmul_kv_cache_impl[target=target](
        hidden_state,
        weight,
        kv_collection,
        layer_idx,
        valid_lengths,
        output,
        cuda_ctx,
    )


@always_inline
fn _fused_qkv_matmul_kv_cache_impl[
    dtype: DType,
    collection_t: KVCollectionT,
    //,
    *,
    target: StaticString,
](
    hidden_state: LayoutTensor[
        dtype, address_space = AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    kv_collection: collection_t,
    layer_idx: UInt32,
    valid_lengths: LayoutTensor[
        DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ],
    output: LayoutTensor[mut=True, dtype, ...],
    context: Optional[DeviceContext],
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Only positions within valid_lengths are written to the KV cache. Padded positions
    (where t_idx >= valid_lengths[b_idx]) are skipped for K and V writes.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The historical KVCache for keys and values. The KVCache for
            this layer is retrieved via layer_idx.
        layer_idx: The index of the layer being executed. Used to retrieve the KVCache
            for the given layer from kv_collection.
        valid_lengths: Tensor of shape [batch] containing the valid length for each
            sequence. K and V are only written to cache for positions within these lengths.
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
        context: The DeviceContext. This is unused if is_cpu[target]().
    """
    comptime cache_t = collection_t.CacheType
    comptime cache_dtype = cache_t.dtype

    comptime assert cache_dtype == dtype, (
        "Expected cache dtype "
        + String(cache_dtype)
        + " to match input dtype "
        + String(dtype)
    )

    comptime kv_params = cache_t.kv_params
    comptime N = Int(weight.layout.shape[0])
    comptime K = Int(weight.layout.shape[1])

    var SEQ_LEN = UInt(hidden_state.dim[1]())

    var q_dim = output.dim[2]()
    var k_dim = kv_params.head_size * kv_params.num_heads
    var qk_offset = q_dim + Int(k_dim)

    var k_cache = kv_collection.get_key_cache(Int(layer_idx))
    var v_cache = kv_collection.get_value_cache(Int(layer_idx))

    @parameter
    @__copy_capture(q_dim, qk_offset, SEQ_LEN, k_cache, v_cache, valid_lengths)
    @always_inline
    fn write_to_cache[
        dtype_: DType, width: Int, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype_, width]):
        var b_idx, t_idx = divmod(UInt(idx[0]), SEQ_LEN)
        if idx[1] < q_dim:
            output.store[width=width](
                Index(Int(b_idx), Int(t_idx), idx[1]),
                rebind[SIMD[dtype, width]](val),
            )
            return

        # Skip writing to cache for padded positions
        var valid_len_for_batch_vec = valid_lengths[Int(b_idx)]
        comptime assert valid_len_for_batch_vec.size == 1
        var valid_len_for_batch = UInt(valid_len_for_batch_vec[0])
        if t_idx >= valid_len_for_batch:
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

        var cache_len = cache.cache_length(Int(b_idx))
        var cache_t_idx = t_idx + UInt(cache_len)
        cache.store(
            Int(b_idx),
            Int(h_idx),
            Int(cache_t_idx),
            Int(hd_idx),
            rebind[SIMD[cache_dtype, width]](output_val),
        )

    _matmul_common[target=target, elementwise_lambda_fn=write_to_cache](
        hidden_state, weight, context
    )


@always_inline
fn _matmul_common[
    dtype: DType,
    //,
    *,
    target: StaticString,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    hidden_state: LayoutTensor[
        dtype, address_space = AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    context: Optional[DeviceContext],
) raises:
    var BS = hidden_state.dim[0]()
    var SEQ_LEN = hidden_state.dim[1]()
    comptime N = Int(weight.layout.shape[0])
    comptime K = Int(weight.layout.shape[1])

    comptime hidden_state_layout = Layout.row_major(
        UNKNOWN_VALUE, Int(hidden_state.layout.shape[2])
    )
    var hidden_state_2d = LayoutTensor[
        dtype,
        hidden_state_layout,
        MutAnyOrigin,
    ](
        hidden_state.ptr,
        RuntimeLayout[hidden_state_layout].row_major(
            IndexList[2](BS * SEQ_LEN, K)
        ),
    )

    comptime c_layout = Layout.row_major(UNKNOWN_VALUE, N)
    var c_nd: LayoutTensor[dtype, c_layout, MutAnyOrigin]

    comptime if is_cpu[target]():
        var c_ptr = alloc[Scalar[dtype]](BS * SEQ_LEN * N)

        c_nd = LayoutTensor[dtype, c_layout, MutAnyOrigin](
            c_ptr,
            RuntimeLayout[c_layout].row_major(IndexList[2](BS * SEQ_LEN, N)),
        )
    else:
        c_nd = LayoutTensor[dtype, c_layout, MutAnyOrigin](
            UnsafePointer[Scalar[dtype], MutExternalOrigin](),
            RuntimeLayout[c_layout].row_major(IndexList[2](BS * SEQ_LEN, N)),
        )

    matmul[
        transpose_b=True,
        target=target,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ](c_nd, hidden_state_2d, weight, context)

    comptime if is_cpu[target]():
        c_nd.ptr.free()


# ===-----------------------------------------------------------------------===#
# Fused QK RoPE (padded)
# ===-----------------------------------------------------------------------===#


@always_inline
fn generic_fused_qk_rope_bshd_continuous_batch[
    dtype: DType,
    //,
    *,
    interleaved: Bool,
    target: StaticString,
](
    q_proj: TileTensor[dtype, ...],
    kv_collection: ContinuousBatchingKVCacheCollection,
    freqs_cis: TileTensor[dtype, ...],
    layer_idx: UInt32,
    valid_lengths: TileTensor[DType.uint32, ...],
    output: TileTensor[mut=True, dtype, ...],
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

    Args:
        q_proj: Query projection tensor of shape [batch, seq_len, n_heads, head_dim].
        kv_collection: The continuous batching KV cache collection.
        freqs_cis: Frequency tensor for RoPE of shape [max_seq_len, head_dim].
        layer_idx: The layer index for accessing the correct cache.
        valid_lengths: Tensor of shape [batch] containing the valid length for each
            sequence. RoPE is only applied to positions within these lengths.
        output: Output tensor for Q with RoPE applied, same shape as q_proj.
        context: Device context pointer for execution.
    """

    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            Span(
                [
                    trace_arg(
                        "output",
                        coord_to_index_list(output.layout.shape_coord()),
                    ),
                    trace_arg(
                        "q_proj",
                        coord_to_index_list(q_proj.layout.shape_coord()),
                    ),
                    trace_arg(
                        "freqs_cis",
                        coord_to_index_list(freqs_cis.layout.shape_coord()),
                    ),
                    trace_arg(
                        "valid_lengths",
                        coord_to_index_list(valid_lengths.layout.shape_coord()),
                    ),
                    "layer_idx=" + String(layer_idx),
                    "num_heads=" + String(kv_collection.kv_params.num_heads),
                    "head_size=" + String(kv_collection.kv_params.head_size),
                    "interleaved=" + String(interleaved),
                ]
            )
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
            valid_lengths,
            output,
            dev_ctx,
        )


@always_inline
fn generic_fused_qk_rope_bshd_paged[
    dtype: DType,
    //,
    *,
    interleaved: Bool,
    target: StaticString,
](
    q_proj: TileTensor[dtype, ...],
    kv_collection: PagedKVCacheCollection,
    freqs_cis: TileTensor[dtype, ...],
    layer_idx: UInt32,
    valid_lengths: TileTensor[DType.uint32, ...],
    output: TileTensor[mut=True, dtype, ...],
    context: DeviceContextPtr = DeviceContextPtr(),
) raises:
    """Performs a fused RoPE projection for Q and K with paged KV cache.

    This is the paged equivalent of generic_fused_qk_rope_bshd_continuous_batch.
    It applies RoPE to both Q (returned) and K (in paged cache) to ensure
    proper dependency ordering after fused_qkv_padded_matmul.

    Args:
        q_proj: Query projection tensor of shape [batch, seq_len, n_heads, head_dim].
        kv_collection: The paged KV cache collection.
        freqs_cis: Frequency tensor for RoPE of shape [max_seq_len, head_dim].
        layer_idx: The layer index for accessing the correct cache.
        valid_lengths: Tensor of shape [batch] containing the valid length for each
            sequence. RoPE is only applied to positions within these lengths.
        output: Output tensor for Q with RoPE applied, same shape as q_proj.
        context: Device context pointer for execution.
    """

    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            Span(
                [
                    trace_arg(
                        "output",
                        coord_to_index_list(output.layout.shape_coord()),
                    ),
                    trace_arg(
                        "q_proj",
                        coord_to_index_list(q_proj.layout.shape_coord()),
                    ),
                    trace_arg(
                        "freqs_cis",
                        coord_to_index_list(freqs_cis.layout.shape_coord()),
                    ),
                    trace_arg(
                        "valid_lengths",
                        coord_to_index_list(valid_lengths.layout.shape_coord()),
                    ),
                    "layer_idx=" + String(layer_idx),
                    "num_heads=" + String(kv_collection.kv_params.num_heads),
                    "head_size=" + String(kv_collection.kv_params.head_size),
                    "interleaved=" + String(interleaved),
                ]
            )
        )

    # Pass device context only on GPU.
    var dev_ctx = Optional[DeviceContext]() if is_cpu[
        target
    ]() else context.get_device_context()
    with Trace[TraceLevel.OP, target=target](
        "mo.fused_qk_rope.padded.paged.nhead_"
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
            valid_lengths,
            output,
            dev_ctx,
        )


# ===-----------------------------------------------------------------------===#
# MHA (padded)
# ===-----------------------------------------------------------------------===#


@always_inline
fn generic_flash_attention_kv_cache_padded[
    collection_t: KVCollectionT,
    dtype: DType,
    //,
    *,
    target: StaticString,
    mask_str: StaticString,
    score_mod_str: StaticString,
    local_window_size: Int = -1,
    num_heads: Int = -1,
](
    q: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    kv_collection: collection_t,
    layer_idx: UInt32,
    valid_lengths: LayoutTensor[
        DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ],
    scale: Float32,
    output: LayoutTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    context: DeviceContextPtr,
    sink_weights: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
) raises:
    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("q", q.runtime_layout.shape.value),
                    trace_arg(
                        "valid_lengths",
                        valid_lengths.runtime_layout.shape.value,
                    ),
                    "scale=" + String(scale),
                    "layer_idx=" + String(layer_idx),
                    "num_heads=" + String(collection_t.kv_params.num_heads),
                    "head_size=" + String(collection_t.kv_params.head_size),
                ]
            )
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
    dtype: DType,
    //,
    *,
    target: StaticString,
    score_mod_str: StaticString,
    local_window_size: Int = -1,
    num_heads: Int = -1,
](
    q: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    kv_collection: collection_t,
    layer_idx: UInt32,
    mask: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    valid_lengths: LayoutTensor[
        DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ],
    scale: Float32,
    output: LayoutTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    context: DeviceContextPtr,
    sink_weights: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
) raises:
    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("q", q.runtime_layout.shape.value),
                    trace_arg("mask", mask.runtime_layout.shape.value),
                    trace_arg(
                        "valid_lengths",
                        valid_lengths.runtime_layout.shape.value,
                    ),
                    "scale=" + String(scale),
                    "layer_idx=" + String(layer_idx),
                    "num_heads=" + String(collection_t.kv_params.num_heads),
                    "head_size=" + String(collection_t.kv_params.head_size),
                ]
            )
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
    collection_t: KVCollectionT,
    //,
    *,
    target: StaticString,
    mask_str: StaticString,
    score_mod_str: StaticString,
    local_window_size: Int = -1,
](
    q: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    kv_cache: collection_t,
    layer_idx: UInt32,
    valid_lengths: LayoutTensor[
        DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ],
    scale: Float32,
    output: LayoutTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    context: DeviceContextPtr,
    sink_weights: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
) raises:
    var k = kv_cache.get_key_cache(Int(layer_idx))
    var v = kv_cache.get_value_cache(Int(layer_idx))

    @parameter
    @__copy_capture(k, v)
    fn _dispatch_flash_attention[
        mask_t: MHAMask, score_mod_t: ScoreModTrait
    ](mask: mask_t, score_mod: score_mod_t) raises:
        comptime if is_cpu[target]():
            return flash_attention_kv_cache_cpu(
                q, k, v, mask, scale, output, sink_weights
            )
        else:
            comptime use_score_mod = not _type_is_eq[
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
    collection_t: KVCollectionT,
    //,
    *,
    target: StaticString,
    score_mod_str: String,
    local_window_size: Int = -1,
](
    q: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    kv_cache: collection_t,
    layer_idx: UInt32,
    mask_nd: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    valid_lengths: LayoutTensor[
        DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ],
    scale: Float32,
    output: LayoutTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    context: DeviceContextPtr,
    sink_weights: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
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
            comptime if is_cpu[target]():
                return flash_attention_kv_cache_cpu(
                    q,
                    k,
                    v,
                    mask,
                    scale,
                    output,
                    sink_weights,
                )
            else:
                comptime use_score_mod = not _type_is_eq[
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
        Int(collection_t.kv_params.num_heads),
    ](
        LayoutTensor[mask_nd.dtype, mask_nd.layout, MutAnyOrigin](
            mask_nd.ptr,
            RuntimeLayout[mask_nd.layout].row_major(
                mask_nd.runtime_layout.shape.value.canonicalize()
            ),
        )
    )


# ===-----------------------------------------------------------------------===#
# RMSNorm
# ===-----------------------------------------------------------------------===#


def rms_norm_kv_cache_ragged_paged[
    dtype: DType,
    params: KVCacheStaticParams,
    page_size: Int,
    cache_dtype: DType,
    //,
    target: StaticString,
    multiply_before_cast: Bool,
    per_head_norm: Bool,
](
    kv_collection: PagedKVCacheCollection[
        cache_dtype,
        params,
        page_size,
    ],
    gamma: LayoutTensor[dtype, ...],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    layer_idx: UInt32,
    total_seq_len: UInt32,
    input_row_offsets: LayoutTensor[DType.uint32, ...],
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
    comptime rank = 3 if per_head_norm else 2
    var k_cache = kv_collection.get_key_cache(Int(layer_idx))
    var kv_params = k_cache.kv_params
    comptime rms_norm_cols = Int(gamma.layout.shape[0])

    comptime assert (
        gamma.layout.shape[0] != UNKNOWN_VALUE
    ), "Need static shape for gamma"
    comptime assert (
        rms_norm_cols <= Int(kv_collection.kv_params.head_size)
        or not per_head_norm
    ), "Length of gamma must be smaller or equal to head size"

    var shape = IndexList[rank]()
    shape[0] = Int(total_seq_len)

    comptime if per_head_norm:
        shape[1] = Int(kv_params.num_heads)
        shape[2] = rms_norm_cols
    else:
        shape[1] = rms_norm_cols

    @always_inline
    @parameter
    @__copy_capture(k_cache, input_row_offsets)
    fn key_cache_input_fn[
        width: Int, rank_: Int
    ](idx: IndexList[rank_]) -> SIMD[dtype, width]:
        comptime assert (
            rank_ == rank
        ), "rms_norm_key_cache input lambda index should have rank " + String(
            rank
        )

        var global_token_idx = idx[0]
        var batch_idx = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )
        var token_idx = Int(
            UInt32(global_token_idx) - input_row_offsets[batch_idx]
        )

        var cache_length = k_cache.cache_length(batch_idx)
        var cache_token_idx = token_idx + cache_length

        var head_idx: Int
        var head_dim_idx: Int

        comptime if per_head_norm:
            head_idx = idx[1]
            head_dim_idx = idx[2]
        else:
            head_idx = idx[1] // Int(params.head_size)
            head_dim_idx = idx[1] % Int(params.head_size)

        return k_cache.load[width=width](
            bs=batch_idx,
            tok_idx=cache_token_idx,
            head_idx=head_idx,
            head_dim_idx=head_dim_idx,
        ).cast[dtype]()

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
        var token_idx = Int(
            UInt32(global_token_idx) - input_row_offsets[batch_idx]
        )

        var cache_length = k_cache.cache_length(batch_idx)
        var cache_token_idx = token_idx + cache_length

        var head_idx: Int
        var head_dim_idx: Int

        comptime if per_head_norm:
            head_idx = idx[1]
            head_dim_idx = idx[2]
        else:
            head_idx = idx[1] // Int(params.head_size)
            head_dim_idx = idx[1] % Int(params.head_size)
        k_cache.store(
            bs=batch_idx,
            tok_idx=cache_token_idx,
            head_idx=head_idx,
            head_dim_idx=head_dim_idx,
            val=val.cast[cache_dtype](),
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
            TileTensor(gamma.ptr, row_major(Coord(Idx(gamma.size())))),
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
    valid_lengths: LayoutTensor[DType.uint32, ...],
    is_print_compact: Bool,
) -> None:
    """Prints a cache buffer, abbreviating output with ellipses."""
    comptime kv_params = collection_t.CacheType.kv_params

    # Only abbreviate output when `is_print_compact` is set.
    var num_to_print: Int = 7 if is_print_compact else Int.MAX
    for b_idx in range(valid_lengths.dim[0]()):
        var total_cache_length = Int(
            valid_lengths[b_idx] + UInt32(cache.cache_length(b_idx))
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
                            b_idx,
                            Int(h),
                            t_idx,
                            hd,
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
    valid_lengths: LayoutTensor[DType.uint32, ...],
    kv_collection: ContinuousBatchingKVCacheCollection[dtype, kv_params],
    layer_idx: UInt32,
    is_print_compact: Bool,
    context: DeviceContextPtr,
):
    var k_cache = kv_collection.get_key_cache(Int(layer_idx))
    var v_cache = kv_collection.get_value_cache(Int(layer_idx))

    print("K:")
    _print_cache[type_of(kv_collection)](
        k_cache,
        kv_collection,
        valid_lengths,
        is_print_compact,
    )

    print("V:")
    _print_cache[type_of(kv_collection)](
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
    valid_lengths: LayoutTensor[DType.uint32, ...],
    kv_collection: PagedKVCacheCollection[dtype, kv_params, page_size],
    layer_idx: UInt32,
    is_print_compact: Bool,
    context: DeviceContextPtr,
):
    var k_cache = kv_collection.get_key_cache(Int(layer_idx))
    var v_cache = kv_collection.get_value_cache(Int(layer_idx))

    print("K:")
    _print_cache[type_of(kv_collection)](
        k_cache,
        kv_collection,
        valid_lengths,
        is_print_compact,
    )

    print("V:")
    _print_cache[type_of(kv_collection)](
        v_cache,
        kv_collection,
        valid_lengths,
        is_print_compact,
    )


def print_kv_cache_cont_batch_generic_gpu[
    target: StaticString, dtype: DType, kv_params: KVCacheStaticParams
](
    valid_lengths: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    kv_collection: ContinuousBatchingKVCacheCollection[dtype, kv_params],
    layer_idx: UInt32,
    is_print_compact: Bool,
    context: DeviceContextPtr,
):
    var blocks_ptr = alloc[Scalar[dtype]](kv_collection.blocks.size())
    var blocks_host_nd = LayoutTensor[
        type_of(kv_collection.blocks).dtype, Layout.row_major[6](), MutAnyOrigin
    ](
        blocks_ptr,
        RuntimeLayout[Layout.row_major[6]()].row_major(
            kv_collection.blocks.runtime_layout.shape.value.canonicalize()
        ),
    )
    var dev_ctx = context.get_device_context()
    dev_ctx.enqueue_copy(
        blocks_host_nd.ptr,
        kv_collection.blocks.ptr,
        kv_collection.blocks.size(),
    )

    var cache_lengths_ptr = alloc[UInt32](kv_collection.cache_lengths.size())
    var cache_lengths_host_nd = type_of(kv_collection.cache_lengths)(
        cache_lengths_ptr,
        RuntimeLayout[type_of(kv_collection.cache_lengths).layout].row_major(
            kv_collection.cache_lengths.runtime_layout.shape.value.canonicalize(),
        ),
    )
    dev_ctx.enqueue_copy(
        cache_lengths_host_nd.ptr,
        kv_collection.cache_lengths.ptr,
        kv_collection.cache_lengths.size(),
    )

    var lookup_table_ptr = alloc[UInt32](kv_collection.lookup_table.size())
    var lookup_table_host_nd = type_of(kv_collection.lookup_table)(
        lookup_table_ptr,
        RuntimeLayout[type_of(kv_collection.lookup_table).layout].row_major(
            kv_collection.lookup_table.runtime_layout.shape.value.canonicalize(),
        ),
    )
    dev_ctx.enqueue_copy(
        lookup_table_host_nd.ptr,
        kv_collection.lookup_table.ptr,
        kv_collection.lookup_table.size(),
    )

    var host_kv_collection = type_of(kv_collection)(
        blocks_host_nd,
        cache_lengths_host_nd,
        lookup_table_host_nd,
        kv_collection.max_seq_length,
        kv_collection.max_cache_length,
    )

    var valid_lengths_host_ptr = alloc[UInt32](valid_lengths.size())
    var valid_lengths_host_nd = LayoutTensor[
        valid_lengths.dtype, valid_lengths.layout
    ](
        valid_lengths_host_ptr,
        RuntimeLayout[valid_lengths.layout].row_major(
            valid_lengths.runtime_layout.shape.value.canonicalize()
        ),
    )
    dev_ctx.enqueue_copy(
        valid_lengths_host_nd.ptr,
        valid_lengths.ptr,
        valid_lengths.size(),
    )

    var k_cache = host_kv_collection.get_key_cache(Int(layer_idx))
    var v_cache = host_kv_collection.get_value_cache(Int(layer_idx))

    # Bring host buffers in sync with device buffers.
    dev_ctx.synchronize()

    print("K:")
    _print_cache[type_of(kv_collection)](
        k_cache,
        host_kv_collection,
        valid_lengths_host_nd,
        is_print_compact,
    )

    print("V:")
    _print_cache[type_of(kv_collection)](
        v_cache,
        host_kv_collection,
        valid_lengths_host_nd,
        is_print_compact,
    )

    blocks_host_nd.ptr.free()
    cache_lengths_host_nd.ptr.free()
    lookup_table_host_nd.ptr.free()
    valid_lengths_host_nd.ptr.free()


def print_kv_cache_paged_generic_gpu[
    target: StaticString,
    dtype: DType,
    kv_params: KVCacheStaticParams,
    page_size: Int,
](
    valid_lengths: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    kv_collection: PagedKVCacheCollection[dtype, kv_params, page_size],
    layer_idx: UInt32,
    is_print_compact: Bool,
    context: DeviceContextPtr,
):
    var blocks_ptr = alloc[Scalar[dtype]](kv_collection.blocks.size())
    var blocks_host_nd = LayoutTensor[
        type_of(kv_collection.blocks).dtype, Layout.row_major[6](), MutAnyOrigin
    ](
        blocks_ptr,
        RuntimeLayout[Layout.row_major[6]()].row_major(
            kv_collection.blocks.runtime_layout.shape.value.canonicalize(),
        ),
    )
    var dev_ctx = context.get_device_context()
    dev_ctx.enqueue_copy(
        blocks_host_nd.ptr,
        kv_collection.blocks.ptr,
        kv_collection.blocks.size(),
    )
    var cache_lengths_ptr = alloc[UInt32](kv_collection.cache_lengths.size())
    var cache_lengths_host_nd = type_of(kv_collection.cache_lengths)(
        cache_lengths_ptr,
        RuntimeLayout[type_of(kv_collection.cache_lengths).layout].row_major(
            kv_collection.cache_lengths.runtime_layout.shape.value.canonicalize(),
        ),
    )
    dev_ctx.enqueue_copy(
        cache_lengths_host_nd.ptr,
        kv_collection.cache_lengths.ptr,
        kv_collection.cache_lengths.size(),
    )
    var lookup_table_ptr = alloc[UInt32](kv_collection.lookup_table.size())
    var lookup_table_host_nd = type_of(kv_collection.lookup_table)(
        lookup_table_ptr,
        RuntimeLayout[type_of(kv_collection.lookup_table).layout].row_major(
            kv_collection.lookup_table.runtime_layout.shape.value.canonicalize(),
        ),
    )
    dev_ctx.enqueue_copy(
        lookup_table_host_nd.ptr,
        kv_collection.lookup_table.ptr,
        kv_collection.lookup_table.size(),
    )
    var host_kv_collection = type_of(kv_collection)(
        blocks_host_nd,
        cache_lengths_host_nd,
        lookup_table_host_nd,
        kv_collection.max_seq_length,
        kv_collection.max_cache_length,
    )
    var valid_lengths_host_ptr = alloc[UInt32](valid_lengths.size())
    var valid_lengths_host_nd = LayoutTensor[
        valid_lengths.dtype, valid_lengths.layout
    ](
        valid_lengths_host_ptr,
        RuntimeLayout[valid_lengths.layout].row_major(
            valid_lengths.runtime_layout.shape.value.canonicalize()
        ),
    )
    dev_ctx.enqueue_copy(
        valid_lengths_host_nd.ptr,
        valid_lengths.ptr,
        valid_lengths.size(),
    )

    var k_cache = host_kv_collection.get_key_cache(Int(layer_idx))
    var v_cache = host_kv_collection.get_value_cache(Int(layer_idx))

    # Bring host buffers in sync with device buffers.
    dev_ctx.synchronize()

    print("K:")
    _print_cache[type_of(kv_collection)](
        k_cache,
        host_kv_collection,
        valid_lengths_host_nd,
        is_print_compact,
    )

    print("V:")
    _print_cache[type_of(kv_collection)](
        v_cache,
        host_kv_collection,
        valid_lengths_host_nd,
        is_print_compact,
    )

    blocks_host_nd.ptr.free()
    cache_lengths_host_nd.ptr.free()
    lookup_table_host_nd.ptr.free()
    valid_lengths_host_nd.ptr.free()


# ===-----------------------------------------------------------------------===#
# KV Collection Constructors (Ctor)
# ===-----------------------------------------------------------------------===#


fn _continuous_batch_kv_cache_collection[
    dtype: DType, //, kv_params: KVCacheStaticParams
](
    blocks: LayoutTensor[mut=True, dtype, Layout.row_major[6]()],
    cache_lengths: LayoutTensor[DType.uint32, Layout(UNKNOWN_VALUE)],
    lookup_table: LayoutTensor[DType.uint32, Layout(UNKNOWN_VALUE)],
    max_lengths: LayoutTensor[DType.uint32, Layout.row_major[2]()],
    out result: ContinuousBatchingKVCacheCollection[dtype, kv_params],
):
    # Marshal LayoutTensor into arguments expected by the
    # ContinuousKVCacheCollection constructor.
    return {
        blocks = blocks.as_any_origin(),
        cache_lengths = cache_lengths.get_immutable().as_any_origin(),
        lookup_table = lookup_table.get_immutable().as_any_origin(),
        max_seq_length = max_lengths[0, 0][0],
        max_cache_length = max_lengths[0, 1][0],
    }


@always_inline
fn generic_get_continuous_cache[
    dtype: DType, kv_params: KVCacheStaticParams
](
    blocks: LayoutTensor[mut=True, dtype, Layout.row_major[6]()],
    cache_lengths: LayoutTensor[DType.uint32, Layout(UNKNOWN_VALUE)],
    lookup_table: LayoutTensor[DType.uint32, Layout(UNKNOWN_VALUE)],
    max_lengths: LayoutTensor[DType.uint32, Layout.row_major[2]()],
) -> ContinuousBatchingKVCacheCollection[dtype, kv_params]:
    return _continuous_batch_kv_cache_collection[kv_params](
        blocks, cache_lengths, lookup_table, max_lengths
    )


fn generic_get_paged_cache[
    dtype: DType,
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
            blocks.static_spec.shape.get[1]() == 1,
        ),
        blocks.static_spec.shape.get[3](),
    ],
):
    comptime page_size = blocks.static_spec.shape.get[3]()
    comptime head_dim = blocks.static_spec.shape.get[5]()
    comptime num_heads = blocks.static_spec.shape.get[4]()
    comptime is_mla = blocks.static_spec.shape.get[1]() == 1
    return generic_get_paged_cache[
        dtype,
        KVCacheStaticParams(UInt(num_heads), UInt(head_dim), is_mla),
        page_size,
    ](
        LayoutTensor[blocks.dtype, Layout.row_major[6](), MutAnyOrigin](
            blocks.to_layout_tensor().ptr,
            RuntimeLayout[Layout.row_major[6]()].row_major(
                blocks.to_layout_tensor().runtime_layout.shape.value
            ),
        ),
        LayoutTensor[
            cache_lengths.dtype, Layout(UNKNOWN_VALUE), ImmutAnyOrigin
        ](
            cache_lengths.to_layout_tensor().ptr,
            RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(
                cache_lengths.to_layout_tensor().runtime_layout.shape.value
            ),
        ),
        LayoutTensor[lookup_table.dtype, Layout.row_major[2](), ImmutAnyOrigin](
            lookup_table.to_layout_tensor().ptr,
            RuntimeLayout[Layout.row_major[2]()].row_major(
                lookup_table.to_layout_tensor().runtime_layout.shape.value
            ),
        ),
        LayoutTensor[max_lengths.dtype, Layout.row_major[2](), ImmutAnyOrigin](
            max_lengths.to_layout_tensor().ptr,
            RuntimeLayout[Layout.row_major[2]()].row_major(
                max_lengths.to_layout_tensor().runtime_layout.shape.value
            ),
        ),
    )


fn generic_get_paged_cache[
    dtype: DType,
    kv_params: KVCacheStaticParams,
    page_size: Int,
](
    blocks: LayoutTensor[mut=True, dtype, Layout.row_major[6]()],
    cache_lengths: LayoutTensor[DType.uint32, Layout(UNKNOWN_VALUE)],
    lookup_table: LayoutTensor[DType.uint32, Layout.row_major[2]()],
    max_lengths: LayoutTensor[DType.uint32, Layout.row_major[2]()],
    out result: PagedKVCacheCollection[dtype, kv_params, page_size],
):
    return {
        blocks = blocks.as_any_origin(),
        cache_lengths = cache_lengths.get_immutable().as_any_origin(),
        lookup_table = lookup_table.get_immutable().as_any_origin(),
        max_seq_length = max_lengths[0, 0][0],
        max_cache_length = max_lengths[0, 1][0],
    }


# ===-----------------------------------------------------------------------===#
# GPU→CPU Page Copy for KV Cache Offloading
# ===-----------------------------------------------------------------------===#


fn copy_kv_pages_d2h[
    dtype: DType,
](
    device_kv_blocks: LayoutTensor[mut=True, dtype, Layout.row_major[6]()],
    host_kv_blocks: LayoutTensor[mut=True, dtype, Layout.row_major[6]()],
    src_page_ids: LayoutTensor[DType.int64, Layout.row_major[1]()],
    dst_page_ids: LayoutTensor[DType.int64, Layout.row_major[1]()],
    layer_idx: Int,
    ctx: DeviceContext,
) raises:
    """Copy selected pages for a single layer from device to host KV cache.

    This function performs true GPU→CPU async copy using enqueue_copy.
    It copies only the specified layer for each page, with separate source
    and destination page IDs to support independent page ID spaces.

    The 6D tensor layout is: [num_pages, kv_dim, num_layers, page_size, num_heads, head_dim]

    Args:
        device_kv_blocks: Source GPU KV cache blocks .
        host_kv_blocks: Destination CPU KV cache blocks.
        src_page_ids: Pointer to GPU page IDs.
        dst_page_ids: Pointer to CPU page IDs.
        layer_idx: Which layer to copy.
        ctx: Device context for GPU operations.
    """

    var kv_dim = device_kv_blocks.dim[1]()
    var page_size = device_kv_blocks.dim[3]()
    var num_heads = device_kv_blocks.dim[4]()
    var head_size = device_kv_blocks.dim[5]()
    var num_pages_to_copy = src_page_ids.dim[0]()

    var elements_per_layer_slice = page_size * num_heads * head_size

    for i in range(num_pages_to_copy):
        var src_page_id = Int(src_page_ids[i])
        var dst_page_id = Int(dst_page_ids[i])

        for kv_idx in range(kv_dim):
            var src_offset = device_kv_blocks._offset(
                IndexList[6](src_page_id, kv_idx, layer_idx, 0, 0, 0)
            )

            var dst_offset = host_kv_blocks._offset(
                IndexList[6](dst_page_id, kv_idx, layer_idx, 0, 0, 0)
            )

            var src_buf = DeviceBuffer[dtype](
                ctx,
                device_kv_blocks.ptr + src_offset,
                elements_per_layer_slice,
                owning=False,
            )

            ctx.enqueue_copy(host_kv_blocks.ptr + dst_offset, src_buf)
