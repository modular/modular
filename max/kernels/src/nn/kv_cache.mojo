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

from std.algorithm.functional import unswitch
from std.math import ceildiv, min
from std.math.uutils import udivmod
from std.sys.info import align_of, simd_width_of
from std.gpu import WARP_SIZE, barrier, block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer, get_gpu_target
from std.gpu.host.info import is_cpu, is_gpu
from std.gpu.memory import AddressSpace
from std.collections import OptionalReg
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
    KVCacheT,
    KVCollectionT,
    PagedKVCacheCollection,
)
from layout import (
    Coord,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TensorLayout,
    TileTensor,
    UNKNOWN_VALUE,
    coord_to_index_list,
    lt_to_tt,
    row_major,
    stack_allocation,
)
from linalg.matmul import elementwise_epilogue_type, matmul
from nn._ragged_utils import get_batch_from_row_offsets
from nn.attention.cpu.mha import (
    flash_attention_kv_cache as flash_attention_kv_cache_cpu,
)
from nn.fused_qk_rope import (
    fused_qk_rope,
    get_identity_rope_coeff,
    get_safetensors_idx,
    rope_value,
)
from nn.attention.gpu.mha import flash_attention as gpu_flash_attention
from nn.attention.mha_mask import MHAMask
from nn.attention.mha_utils import (
    dispatch_mask,
    dispatch_materialized_mask,
)
from nn.normalization import _rms_norm_impl, _rms_norm_warp_tiling_subkernel
from std.runtime.tracing import Trace, TraceLevel, get_safe_task_id, trace_arg
from std.utils.numerics import get_accum_type

from std.utils import Index, IndexList
from extensibility import InputTensor
from extensibility import (
    _MutableInputTensor as MutableInputTensor,
)

# ===-----------------------------------------------------------------------===#
# Fused QKV matmul (padded)
# ===-----------------------------------------------------------------------===#


@always_inline
def generic_fused_qkv_matmul_kv_cache_bshd_continuous_batch[
    dtype: DType,
    target: StaticString = "cpu",
](
    hidden_state: LayoutTensor[
        mut=False, dtype, address_space=AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[
        mut=False, dtype, address_space=AddressSpace.GENERIC, ...
    ],
    kv_collection: ContinuousBatchingKVCacheCollection,
    layer_idx: UInt32,
    valid_lengths: LayoutTensor[
        DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ],
    output: LayoutTensor[mut=True, dtype, ...],
    ctx: DeviceContext,
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
    def description_fn() -> String:
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
def generic_fused_qkv_matmul_kv_cache_bshd_paged[
    dtype: DType,
    target: StaticString = "cpu",
](
    hidden_state: LayoutTensor[
        mut=False, dtype, address_space=AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[
        mut=False, dtype, address_space=AddressSpace.GENERIC, ...
    ],
    kv_collection: PagedKVCacheCollection,
    layer_idx: UInt32,
    valid_lengths: LayoutTensor[
        DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ],
    output: LayoutTensor[mut=True, dtype, ...],
    ctx: DeviceContext,
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
    def description_fn() -> String:
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
def _fused_qkv_matmul_kv_cache[
    dtype: DType,
    collection_t: KVCollectionT,
    //,
    cache_t: KVCacheT,
    *,
    target: StaticString,
](
    hidden_state: LayoutTensor[
        mut=False, dtype, address_space=AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[
        mut=False, dtype, address_space=AddressSpace.GENERIC, ...
    ],
    kv_collection: collection_t,
    layer_idx: UInt32,
    valid_lengths: LayoutTensor[
        DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ],
    output: LayoutTensor[mut=True, dtype, ...],
    context: DeviceContext,
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
        cuda_ctx = context

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
def _fused_qkv_matmul_kv_cache_impl[
    dtype: DType,
    collection_t: KVCollectionT,
    //,
    *,
    target: StaticString,
](
    hidden_state: LayoutTensor[
        mut=False, dtype, address_space=AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[
        mut=False, dtype, address_space=AddressSpace.GENERIC, ...
    ],
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

    var SEQ_LEN: Int = hidden_state.dim[1]()

    var q_dim = output.dim[2]()
    var k_dim = kv_params.head_size * kv_params.num_heads
    var qk_offset = q_dim + k_dim

    var k_cache = kv_collection.get_key_cache(Int(layer_idx))
    var v_cache = kv_collection.get_value_cache(Int(layer_idx))

    @parameter
    @__copy_capture(q_dim, qk_offset, SEQ_LEN, k_cache, v_cache, valid_lengths)
    @always_inline
    def write_to_cache[
        dtype_: DType, width: SIMDSize, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype_, width]):
        var b_idx, t_idx = udivmod(idx[0], SEQ_LEN)
        if idx[1] < q_dim:
            output.store[width=width](
                Index(b_idx, t_idx, idx[1]),
                rebind[SIMD[dtype, width]](val),
            )
            return

        # Skip writing to cache for padded positions
        var valid_len_for_batch_vec = valid_lengths[b_idx]
        comptime assert valid_len_for_batch_vec.size == 1
        var valid_len_for_batch: UInt32 = valid_len_for_batch_vec[0]
        if t_idx >= Int(valid_len_for_batch):
            return

        var h_idx: Int
        var hd_idx: Int
        var cache: cache_t
        var output_val = val
        if idx[1] < qk_offset:
            cache = k_cache
            h_idx, hd_idx = udivmod(idx[1] - q_dim, kv_params.head_size)

        else:
            cache = v_cache
            h_idx, hd_idx = udivmod(idx[1] - qk_offset, kv_params.head_size)

        var cache_len = cache.cache_length(b_idx)
        var cache_t_idx = t_idx + cache_len
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
def _matmul_common[
    dtype: DType,
    //,
    *,
    target: StaticString,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    hidden_state: LayoutTensor[
        mut=False, dtype, address_space=AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[
        mut=False, dtype, address_space=AddressSpace.GENERIC, ...
    ],
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
        hidden_state.origin,
    ](
        hidden_state.ptr,
        RuntimeLayout[hidden_state_layout].row_major(
            IndexList[2](BS * SEQ_LEN, K)
        ),
    )

    comptime c_layout = Layout.row_major(UNKNOWN_VALUE, N)

    comptime if is_cpu[target]():
        var c_ptr = alloc[Scalar[dtype]](BS * SEQ_LEN * N)
        var c_nd = LayoutTensor[dtype, c_layout](
            c_ptr,
            RuntimeLayout[c_layout].row_major(IndexList[2](BS * SEQ_LEN, N)),
        )

        matmul[
            transpose_b=True,
            target=target,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ](lt_to_tt(c_nd), lt_to_tt(hidden_state_2d), lt_to_tt(weight), context)

        c_nd.ptr.free()
    else:
        # Allocate a device-local scratch for the matmul accumulator; the
        # epilogue lambda reads from it and scatters Q/K/V to the real
        # output and KV cache.
        var c_device_buffer = context.value().enqueue_create_buffer[dtype](
            BS * SEQ_LEN * N
        )
        var c_nd = LayoutTensor[dtype, c_layout](
            c_device_buffer.unsafe_ptr(),
            RuntimeLayout[c_layout].row_major(IndexList[2](BS * SEQ_LEN, N)),
        )

        matmul[
            transpose_b=True,
            target=target,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ](lt_to_tt(c_nd), lt_to_tt(hidden_state_2d), lt_to_tt(weight), context)


# ===-----------------------------------------------------------------------===#
# Fused QK RoPE (padded)
# ===-----------------------------------------------------------------------===#


@always_inline
def generic_fused_qk_rope_bshd_continuous_batch[
    dtype: DType,
    //,
    *,
    interleaved: Bool,
    target: StaticString,
](
    q_proj: TileTensor[mut=False, dtype, ...],
    kv_collection: ContinuousBatchingKVCacheCollection,
    freqs_cis: TileTensor[mut=False, dtype, ...],
    layer_idx: UInt32,
    valid_lengths: TileTensor[mut=False, DType.uint32, ...],
    output: TileTensor[mut=True, dtype, ...],
    context: DeviceContext,
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
    def description_fn() -> String:
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
            context,
        )


@always_inline
def generic_fused_qk_rope_bshd_paged[
    dtype: DType,
    //,
    *,
    interleaved: Bool,
    target: StaticString,
](
    q_proj: TileTensor[mut=False, dtype, ...],
    kv_collection: PagedKVCacheCollection,
    freqs_cis: TileTensor[mut=False, dtype, ...],
    layer_idx: UInt32,
    valid_lengths: TileTensor[mut=False, DType.uint32, ...],
    output: TileTensor[mut=True, dtype, ...],
    context: DeviceContext,
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
    def description_fn() -> String:
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
            context,
        )


# ===-----------------------------------------------------------------------===#
# MHA (padded)
# ===-----------------------------------------------------------------------===#


@always_inline
def generic_flash_attention_kv_cache_padded[
    collection_t: KVCollectionT,
    dtype: DType,
    //,
    *,
    target: StaticString,
    mask_str: StaticString,
    local_window_size: Int = -1,
    num_heads: Int = -1,
](
    q: LayoutTensor[mut=False, dtype, address_space=AddressSpace.GENERIC, ...],
    kv_collection: collection_t,
    layer_idx: UInt32,
    valid_lengths: LayoutTensor[
        DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ],
    scale: Float32,
    output: LayoutTensor[
        mut=True, dtype, address_space=AddressSpace.GENERIC, ...
    ],
    context: DeviceContext,
    sink_weights: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
) raises:
    @always_inline
    @parameter
    def description_fn() -> String:
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
def generic_flash_attention_kv_cache_padded_materialized_mask[
    collection_t: KVCollectionT,
    dtype: DType,
    //,
    *,
    target: StaticString,
    local_window_size: Int = -1,
    num_heads: Int = -1,
](
    q: LayoutTensor[mut=False, dtype, address_space=AddressSpace.GENERIC, ...],
    kv_collection: collection_t,
    layer_idx: UInt32,
    mask: LayoutTensor[
        mut=False, dtype, address_space=AddressSpace.GENERIC, ...
    ],
    valid_lengths: LayoutTensor[
        DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ],
    scale: Float32,
    output: LayoutTensor[
        mut=True, dtype, address_space=AddressSpace.GENERIC, ...
    ],
    context: DeviceContext,
    sink_weights: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
) raises:
    @always_inline
    @parameter
    def description_fn() -> String:
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
        "mo.mha.padded.continuous_batching.tensor_mask.nhead_"
        + String(collection_t.kv_params.num_heads)
        + ".hdim_"
        + String(collection_t.kv_params.head_size),
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
    ):
        return _flash_attention_dispatch_materialized_mask[
            target=target,
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


def _flash_attention_dispatch[
    dtype: DType,
    collection_t: KVCollectionT,
    q_origin: Origin[mut=False],
    output_origin: Origin[mut=True],
    //,
    *,
    target: StaticString,
    mask_str: StaticString,
    local_window_size: Int = -1,
](
    q: LayoutTensor[
        dtype, _, q_origin, address_space=AddressSpace.GENERIC, ...
    ],
    kv_cache: collection_t,
    layer_idx: UInt32,
    valid_lengths: LayoutTensor[
        mut=False, DType.uint32, Layout.row_major(UNKNOWN_VALUE), _
    ],
    scale: Float32,
    output: LayoutTensor[
        mut=True,
        dtype,
        _,
        output_origin,
        address_space=AddressSpace.GENERIC,
        ...,
    ],
    context: DeviceContext,
    sink_weights: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
) raises:
    var k = kv_cache.get_key_cache(Int(layer_idx))
    var v = kv_cache.get_value_cache(Int(layer_idx))

    @parameter
    @__copy_capture(k, v)
    def _dispatch_flash_attention[mask_t: MHAMask](mask: mask_t) raises:
        comptime if is_cpu[target]():
            return flash_attention_kv_cache_cpu(
                q, k, v, mask, scale, output, sink_weights
            )
        else:
            gpu_flash_attention[](
                output,
                q,
                k,
                v,
                mask,
                valid_lengths,
                scale,
                context,
            )

    return dispatch_mask[mask_str, _dispatch_flash_attention]()


def _flash_attention_dispatch_materialized_mask[
    dtype: DType,
    collection_t: KVCollectionT,
    //,
    *,
    target: StaticString,
    local_window_size: Int = -1,
](
    q: LayoutTensor[mut=False, dtype, address_space=AddressSpace.GENERIC, ...],
    kv_cache: collection_t,
    layer_idx: UInt32,
    mask_nd: LayoutTensor[
        mut=False, dtype, address_space=AddressSpace.GENERIC, ...
    ],
    valid_lengths: LayoutTensor[
        DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ],
    scale: Float32,
    output: LayoutTensor[
        mut=True, dtype, address_space=AddressSpace.GENERIC, ...
    ],
    context: DeviceContext,
    sink_weights: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
) raises:
    var k = kv_cache.get_key_cache(Int(layer_idx))
    var v = kv_cache.get_value_cache(Int(layer_idx))

    @parameter
    def _dispatch_flash_attention[mask_t: MHAMask](mask: mask_t) raises:
        @always_inline
        @parameter
        def call_flash_attention[sink: Bool]() raises:
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
                gpu_flash_attention[sink=sink](
                    output,
                    q,
                    k,
                    v,
                    mask,
                    valid_lengths,
                    scale,
                    context,
                    sink_weights=sink_weights,
                )

        unswitch[call_flash_attention](Bool(sink_weights))

    return dispatch_materialized_mask[_dispatch_flash_attention](
        LayoutTensor[mask_nd.dtype, mask_nd.layout, mask_nd.origin](
            mask_nd.ptr,
            RuntimeLayout[mask_nd.layout].row_major(
                mask_nd.runtime_layout.shape.value.canonicalize()
            ),
        )
    )


# ===-----------------------------------------------------------------------===#
# RMSNorm
# ===-----------------------------------------------------------------------===#


@__name(t"fused_qk_rms_norm_ragged_paged_gpu_{dtype}_{multiply_before_cast}")
def _fused_qk_rms_norm_ragged_paged_gpu[
    cache_t: KVCacheT,
    q_out_layout: TensorLayout,
    q_out_origin: Origin[mut=True],
    q_layout: TensorLayout,
    q_origin: Origin[mut=False],
    q_gamma_layout: TensorLayout,
    q_gamma_origin: Origin[mut=False],
    k_gamma_layout: TensorLayout,
    k_gamma_origin: Origin[mut=False],
    offsets_layout: TensorLayout,
    offsets_origin: Origin[mut=False],
    dtype: DType,
    //,
    simd_width: Int,
    warps_per_block: Int,
    multiply_before_cast: Bool,
](
    q_output: TileTensor[dtype, q_out_layout, q_out_origin],
    q_proj: TileTensor[dtype, q_layout, q_origin],
    k_cache: cache_t,
    q_gamma: TileTensor[dtype, q_gamma_layout, q_gamma_origin],
    k_gamma: TileTensor[dtype, k_gamma_layout, k_gamma_origin],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    total_seq_len: UInt32,
    input_row_offsets: TileTensor[DType.uint32, offsets_layout, offsets_origin],
    q_num_heads: Int,
    num_cols: Int,
):
    comptime assert q_output.flat_rank == 3, "q_output must have rank 3"
    comptime assert q_proj.flat_rank == 3, "q_proj must have rank 3"
    comptime assert q_gamma.flat_rank == 1, "q_gamma must have rank 1"
    comptime assert k_gamma.flat_rank == 1, "k_gamma must have rank 1"
    comptime assert (
        input_row_offsets.flat_rank == 1
    ), "input_row_offsets must be rank 1"

    comptime accum_type = get_accum_type[dtype]()
    var eps_accum = epsilon.cast[accum_type]()
    var weight_offset_accum = weight_offset.cast[accum_type]()

    var tid = thread_idx.x
    var combined_row = Int(block_idx.x)
    var q_rows = Int(total_seq_len) * q_num_heads
    var is_k = combined_row >= q_rows

    var global_token_idx: Int
    var head_idx: Int
    if is_k:
        comptime k_num_heads = cache_t.kv_params.num_heads
        var k_row = combined_row - q_rows
        global_token_idx = k_row // k_num_heads
        head_idx = k_row % k_num_heads
    else:
        global_token_idx = combined_row // q_num_heads
        head_idx = combined_row % q_num_heads

    var idx = tid * simd_width
    var vec_data = SIMD[accum_type, simd_width](0)
    var gamma_val = SIMD[dtype, simd_width](0)
    if idx < num_cols:
        if is_k:
            var batch_idx = get_batch_from_row_offsets(
                input_row_offsets, global_token_idx
            )
            var token_idx = Int(
                UInt32(global_token_idx) - input_row_offsets[batch_idx]
            )
            var cache_token_idx = token_idx + k_cache.cache_length(batch_idx)
            vec_data = k_cache.load[width=simd_width](
                bs=batch_idx,
                tok_idx=cache_token_idx,
                head_idx=head_idx,
                head_dim_idx=idx,
            ).cast[accum_type]()
            gamma_val = k_gamma.load[width=simd_width](Coord(idx))
        else:
            vec_data = q_proj.load[width=simd_width](
                Coord(Index(global_token_idx, head_idx, idx))
            ).cast[accum_type]()
            gamma_val = q_gamma.load[width=simd_width](Coord(idx))

    var norm_val = _rms_norm_warp_tiling_subkernel[
        warps_per_block, multiply_before_cast
    ](
        combined_row,
        idx,
        vec_data,
        gamma_val,
        eps_accum,
        weight_offset_accum,
        num_cols,
    )

    if idx < num_cols:
        if is_k:
            var batch_idx = get_batch_from_row_offsets(
                input_row_offsets, global_token_idx
            )
            var token_idx = Int(
                UInt32(global_token_idx) - input_row_offsets[batch_idx]
            )
            var cache_token_idx = token_idx + k_cache.cache_length(batch_idx)
            k_cache.store(
                bs=batch_idx,
                tok_idx=cache_token_idx,
                head_idx=head_idx,
                head_dim_idx=idx,
                val=norm_val.cast[cache_t.dtype](),
            )
        else:
            q_output.store[width=simd_width](
                Coord(Index(global_token_idx, head_idx, idx)), norm_val
            )


def fused_qk_rms_norm_ragged_paged[
    dtype: DType,
    params: KVCacheStaticParams,
    page_size: Int,
    cache_dtype: DType,
    //,
    target: StaticString,
    multiply_before_cast: Bool,
](
    q_proj: TileTensor[mut=False, dtype, ...],
    kv_collection: PagedKVCacheCollection[
        cache_dtype,
        params,
        page_size,
        ...,
    ],
    q_gamma: TileTensor[mut=False, dtype, ...],
    k_gamma: TileTensor[mut=False, dtype, ...],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    layer_idx: UInt32,
    input_row_offsets: TileTensor[mut=False, DType.uint32, ...],
    q_output: TileTensor[mut=True, dtype, ...],
    context: DeviceContext,
) raises:
    """Applies per-head RMSNorm to Q and new K-cache entries in one GPU launch.
    """
    comptime assert is_gpu[
        target
    ](), "fused_qk_rms_norm_ragged_paged is GPU-only"
    comptime assert q_proj.flat_rank == 3, "q_proj must be rank 3"
    comptime assert q_output.flat_rank == 3, "q_output must be rank 3"
    comptime assert q_gamma.flat_rank == 1, "q_gamma must be rank 1"
    comptime assert k_gamma.flat_rank == 1, "k_gamma must be rank 1"
    comptime assert (
        input_row_offsets.flat_rank == 1
    ), "input_row_offsets must be rank 1"
    comptime assert (
        cache_dtype == dtype
    ), "fused_qk_rms_norm_ragged_paged requires Q and K cache dtype to match"

    var k_cache = kv_collection.get_key_cache(Int(layer_idx))
    var q_num_heads = Int(q_proj.dim[1]())
    comptime rms_norm_cols = q_gamma.static_shape[0]
    comptime k_rms_norm_cols = k_gamma.static_shape[0]
    comptime assert rms_norm_cols != -1, "Need static shape for q_gamma"
    comptime assert k_rms_norm_cols != -1, "Need static shape for k_gamma"
    comptime assert (
        rms_norm_cols == k_rms_norm_cols
    ), "q_gamma and k_gamma must have the same static size"
    comptime assert (
        rms_norm_cols == params.head_size
    ), "fused QK RMSNorm requires full per-head normalization"

    var total_seq_len = UInt32(q_proj.dim[0]())
    if total_seq_len == 0:
        return

    var q_rows = Int(total_seq_len) * q_num_heads
    var k_rows = Int(total_seq_len) * params.num_heads
    var rows = q_rows + k_rows

    @always_inline
    @parameter
    def description_fn() -> String:
        return (
            trace_arg(
                "q_proj", coord_to_index_list(q_proj.layout.shape_coord())
            )
            + ";layer_idx="
            + String(layer_idx)
            + ";num_heads="
            + String(params.num_heads)
            + ";head_size="
            + String(params.head_size)
        )

    with Trace[TraceLevel.OP, target=target](
        "fused_qk_rms_norm_ragged_paged_nhead_"
        + String(params.num_heads)
        + ".hdim_"
        + String(params.head_size),
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=get_safe_task_id(context),
    ):
        comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
        comptime assert (
            rms_norm_cols % simd_width == 0
        ), "rms_norm_cols must be divisible by simd_width"
        comptime max_warps_per_block = (
            context.default_device_info.max_thread_block_size // WARP_SIZE
        )
        comptime warps_per_block = ceildiv(
            rms_norm_cols // simd_width, WARP_SIZE
        )
        comptime assert (
            warps_per_block <= max_warps_per_block
        ), "fused QK RMSNorm block size exceeds device max warps per block"
        var cols = Int(rms_norm_cols)
        comptime block_dim_value = WARP_SIZE * warps_per_block
        comptime kernel = _fused_qk_rms_norm_ragged_paged_gpu[
            cache_t=type_of(k_cache),
            q_out_layout=q_output.LayoutType,
            q_out_origin=q_output.origin,
            q_layout=q_proj.LayoutType,
            q_origin=q_proj.origin,
            q_gamma_layout=q_gamma.LayoutType,
            q_gamma_origin=q_gamma.origin,
            k_gamma_layout=k_gamma.LayoutType,
            k_gamma_origin=k_gamma.origin,
            offsets_layout=input_row_offsets.LayoutType,
            offsets_origin=input_row_offsets.origin,
            dtype=dtype,
            simd_width,
            warps_per_block,
            multiply_before_cast,
        ]
        context.enqueue_function[kernel](
            q_output,
            q_proj,
            k_cache,
            q_gamma,
            k_gamma,
            epsilon,
            weight_offset,
            total_seq_len,
            input_row_offsets,
            q_num_heads,
            cols,
            grid_dim=rows,
            block_dim=block_dim_value,
        )


# ===-----------------------------------------------------------------------===#
# Fused RMSNorm + RoPE
# ===-----------------------------------------------------------------------===#


@__name(
    t"fused_qk_rms_norm_rope_ragged_paged_gpu_{dtype}_{multiply_before_cast}_{interleaved}"
)
def _fused_qk_rms_norm_rope_ragged_paged_gpu[
    cache_t: KVCacheT,
    q_out_layout: TensorLayout,
    q_out_origin: Origin[mut=True],
    q_layout: TensorLayout,
    q_origin: Origin[mut=False],
    q_gamma_layout: TensorLayout,
    q_gamma_origin: Origin[mut=False],
    k_gamma_layout: TensorLayout,
    k_gamma_origin: Origin[mut=False],
    freqs_layout: TensorLayout,
    freqs_origin: Origin[mut=False],
    offsets_layout: TensorLayout,
    offsets_origin: Origin[mut=False],
    dtype: DType,
    freq_dtype: DType,
    //,
    simd_width: Int,
    warps_per_block: Int,
    multiply_before_cast: Bool,
    interleaved: Bool,
    has_nope_prefix: Bool,
    rope_dim: Int,
](
    q_output: TileTensor[dtype, q_out_layout, q_out_origin],
    q_proj: TileTensor[dtype, q_layout, q_origin],
    k_cache: cache_t,
    q_gamma: TileTensor[dtype, q_gamma_layout, q_gamma_origin],
    k_gamma: TileTensor[dtype, k_gamma_layout, k_gamma_origin],
    freqs_cis: TileTensor[freq_dtype, freqs_layout, freqs_origin],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    total_seq_len: UInt32,
    input_row_offsets: TileTensor[DType.uint32, offsets_layout, offsets_origin],
    q_num_heads: Int,
    num_cols: Int,
):
    comptime assert q_output.flat_rank == 3, "q_output must have rank 3"
    comptime assert q_proj.flat_rank == 3, "q_proj must have rank 3"
    comptime assert q_gamma.flat_rank == 1, "q_gamma must have rank 1"
    comptime assert k_gamma.flat_rank == 1, "k_gamma must have rank 1"
    comptime assert freqs_cis.flat_rank == 2, "freqs_cis must have rank 2"
    comptime assert (
        input_row_offsets.flat_rank == 1
    ), "input_row_offsets must be rank 1"

    comptime accum_type = get_accum_type[dtype]()
    var eps_accum = epsilon.cast[accum_type]()
    var weight_offset_accum = weight_offset.cast[accum_type]()

    comptime head_dim = q_gamma.static_shape[0]
    comptime assert head_dim != -1, "Need static shape for q_gamma"

    var tid = thread_idx.x
    var combined_row = Int(block_idx.x)
    var q_rows = Int(total_seq_len) * q_num_heads
    var is_k = combined_row >= q_rows

    var global_token_idx: Int
    var head_idx: Int
    if is_k:
        comptime k_num_heads = cache_t.kv_params.num_heads
        var k_row = combined_row - q_rows
        global_token_idx = k_row // k_num_heads
        head_idx = k_row % k_num_heads
    else:
        global_token_idx = combined_row // q_num_heads
        head_idx = combined_row % q_num_heads

    var idx = tid * simd_width
    var vec_data = SIMD[accum_type, simd_width](0)
    var gamma_val = SIMD[dtype, simd_width](0)
    if idx < num_cols:
        if is_k:
            var batch_idx = get_batch_from_row_offsets(
                input_row_offsets, global_token_idx
            )
            var token_idx = Int(
                UInt32(global_token_idx) - input_row_offsets[batch_idx]
            )
            var cache_token_idx = token_idx + k_cache.cache_length(batch_idx)
            vec_data = k_cache.load[width=simd_width](
                bs=batch_idx,
                tok_idx=cache_token_idx,
                head_idx=head_idx,
                head_dim_idx=idx,
            ).cast[accum_type]()
            gamma_val = k_gamma.load[width=simd_width](Coord(idx))
        else:
            vec_data = q_proj.load[width=simd_width](
                Coord(Index(global_token_idx, head_idx, idx))
            ).cast[accum_type]()
            gamma_val = q_gamma.load[width=simd_width](Coord(idx))

    var norm_val = _rms_norm_warp_tiling_subkernel[
        warps_per_block, multiply_before_cast
    ](
        combined_row,
        idx,
        vec_data,
        gamma_val,
        eps_accum,
        weight_offset_accum,
        num_cols,
    )

    # Non-interleaved RoPE pairs column j with j + rope_dim/2, which belongs to
    # a different thread's chunk, so the full normed row must be in shared memory
    # before any thread reads its partner. The alignment must cover the widest
    # vectorized store below; align_of[accum_type] alone (4 B for f32) is too
    # narrow and causes MISALIGNED_ADDRESS faults.
    comptime smem_align = align_of[SIMD[accum_type, simd_width]]()
    var s_norm = stack_allocation[
        accum_type,
        address_space=AddressSpace.SHARED,
        alignment=smem_align,
    ](row_major[head_dim]())
    if idx < num_cols:
        s_norm.store[width=simd_width](Coord(idx), norm_val.cast[accum_type]())
    barrier()

    if idx >= num_cols:
        return

    var batch_idx = get_batch_from_row_offsets(
        input_row_offsets, global_token_idx
    )
    var token_idx = Int(UInt32(global_token_idx) - input_row_offsets[batch_idx])
    var post_seq_idx = k_cache.cache_length(batch_idx) + token_idx

    comptime width_2 = simd_width // 2

    comptime if interleaved:
        var freq_val = freqs_cis.load[width=simd_width](
            Coord(Index(post_seq_idx, idx))
        )
        var val = s_norm.load[width=simd_width](Coord(idx))
        var res = rope_value(val, freq_val.cast[accum_type]()).cast[dtype]()
        if is_k:
            k_cache.store(
                bs=batch_idx,
                tok_idx=post_seq_idx,
                head_idx=head_idx,
                head_dim_idx=idx,
                val=res.cast[cache_t.dtype](),
            )
        else:
            q_output.store[width=simd_width](
                Coord(Index(global_token_idx, head_idx, idx)), res
            )
    else:
        # Non-interleaved (safetensors). With has_nope_prefix the roped region
        # is the prefix [0, rope_dim); the suffix [rope_dim, head_dim) is left
        # un-roped (only normed). Without it, the whole head is roped.
        comptime if has_nope_prefix:
            if idx >= rope_dim:
                var passthrough = s_norm.load[width=simd_width](
                    Coord(idx)
                ).cast[dtype]()
                if is_k:
                    k_cache.store(
                        bs=batch_idx,
                        tok_idx=post_seq_idx,
                        head_idx=head_idx,
                        head_dim_idx=idx,
                        val=passthrough.cast[cache_t.dtype](),
                    )
                else:
                    q_output.store[width=simd_width](
                        Coord(Index(global_token_idx, head_idx, idx)),
                        passthrough,
                    )
                return

        comptime split_size = rope_dim if has_nope_prefix else head_dim
        var freq_val = freqs_cis.load[width=simd_width](
            Coord(Index(post_seq_idx, idx))
        )

        h_re, h_im = get_safetensors_idx(idx, split_size)

        var val = rebind[SIMD[accum_type, simd_width]](
            s_norm.load[width=width_2](Coord(h_re)).interleave(
                s_norm.load[width=width_2](Coord(h_im))
            )
        )
        var res = rope_value(val, freq_val.cast[accum_type]()).cast[dtype]()
        # `deinterleave` yields `SIMD[dtype, simd_width / 2]`; let the stores
        # infer their width from the value (matches `rope_q_proj` /
        # `rope_k_cache`) rather than binding an explicit `width=width_2`, which
        # the comptime ternary in `store`'s default would fail to unify.
        output_re, output_im = res.deinterleave()

        if is_k:
            k_cache.store(
                bs=batch_idx,
                tok_idx=post_seq_idx,
                head_idx=head_idx,
                head_dim_idx=h_re,
                val=output_re.cast[cache_t.dtype](),
            )
            k_cache.store(
                bs=batch_idx,
                tok_idx=post_seq_idx,
                head_idx=head_idx,
                head_dim_idx=h_im,
                val=output_im.cast[cache_t.dtype](),
            )
        else:
            q_output.store(
                Coord(Index(global_token_idx, head_idx, h_re)), output_re
            )
            q_output.store(
                Coord(Index(global_token_idx, head_idx, h_im)), output_im
            )


def fused_qk_rms_norm_rope_ragged_paged[
    dtype: DType,
    freq_dtype: DType,
    params: KVCacheStaticParams,
    page_size: Int,
    cache_dtype: DType,
    //,
    target: StaticString,
    multiply_before_cast: Bool,
    interleaved: Bool,
](
    q_proj: TileTensor[mut=False, dtype, ...],
    kv_collection: PagedKVCacheCollection[
        cache_dtype,
        params,
        page_size,
        ...,
    ],
    q_gamma: TileTensor[mut=False, dtype, ...],
    k_gamma: TileTensor[mut=False, dtype, ...],
    freqs_cis: TileTensor[mut=False, freq_dtype, ...],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    layer_idx: UInt32,
    input_row_offsets: TileTensor[mut=False, DType.uint32, ...],
    q_output: TileTensor[mut=True, dtype, ...],
    context: DeviceContext,
) raises:
    """Fuses per-head RMSNorm and RoPE for Q and new K-cache entries.

    This applies per-head RMSNorm to Q and the newly written key-cache entries,
    then applies RoPE to the normalized values, in a single GPU launch. It is
    the fusion of `fused_qk_rms_norm_ragged_paged` and `fused_qk_rope_ragged`,
    saving one elementwise RoPE launch per QK group.

    The RoPE dimension is taken from `freqs_cis.static_shape[1]`. When it is
    smaller than the head dimension, RoPE is applied only to the prefix
    `[0, rope_dim)` of each head (non-interleaved layout) and the suffix is left
    un-roped, matching `fused_qk_rope_ragged`'s `has_nope_prefix` path.
    """
    comptime assert is_gpu[
        target
    ](), "fused_qk_rms_norm_rope_ragged_paged is GPU-only"
    comptime assert q_proj.flat_rank == 3, "q_proj must be rank 3"
    comptime assert q_output.flat_rank == 3, "q_output must be rank 3"
    comptime assert q_gamma.flat_rank == 1, "q_gamma must be rank 1"
    comptime assert k_gamma.flat_rank == 1, "k_gamma must be rank 1"
    comptime assert freqs_cis.flat_rank == 2, "freqs_cis must be rank 2"
    comptime assert (
        input_row_offsets.flat_rank == 1
    ), "input_row_offsets must be rank 1"
    comptime assert cache_dtype == dtype, (
        "fused_qk_rms_norm_rope_ragged_paged requires Q and K cache dtype to"
        " match"
    )

    var k_cache = kv_collection.get_key_cache(Int(layer_idx))
    var q_num_heads = Int(q_proj.dim[1]())
    comptime rms_norm_cols = q_gamma.static_shape[0]
    comptime k_rms_norm_cols = k_gamma.static_shape[0]
    comptime assert rms_norm_cols != -1, "Need static shape for q_gamma"
    comptime assert k_rms_norm_cols != -1, "Need static shape for k_gamma"
    comptime assert (
        rms_norm_cols == k_rms_norm_cols
    ), "q_gamma and k_gamma must have the same static size"
    comptime assert (
        rms_norm_cols == params.head_size
    ), "fused QK RMSNorm requires full per-head normalization"

    comptime rope_dim = Int(freqs_cis.static_shape[1])
    comptime assert rope_dim != -1, "Need static shape for freqs_cis"
    comptime unroped_dim = rms_norm_cols - rope_dim
    comptime has_nope = unroped_dim > 0
    comptime assert rope_dim <= rms_norm_cols, "rope_dim must be <= head_size"
    comptime has_nope_prefix = has_nope and not interleaved
    comptime if has_nope and not interleaved:
        comptime assert (
            rope_dim % 2 == 0
        ), "prefix partial RoPE rope_dim must be even for split layout"

    var total_seq_len = UInt32(q_proj.dim[0]())
    if total_seq_len == 0:
        return

    var q_rows = Int(total_seq_len) * q_num_heads
    var k_rows = Int(total_seq_len) * params.num_heads
    var rows = q_rows + k_rows

    @always_inline
    @parameter
    def description_fn() -> String:
        return (
            trace_arg(
                "q_proj", coord_to_index_list(q_proj.layout.shape_coord())
            )
            + ";layer_idx="
            + String(layer_idx)
            + ";num_heads="
            + String(params.num_heads)
            + ";head_size="
            + String(params.head_size)
            + ";rope_dim="
            + String(rope_dim)
        )

    with Trace[TraceLevel.OP, target=target](
        "fused_qk_rms_norm_rope_ragged_paged_nhead_"
        + String(params.num_heads)
        + ".hdim_"
        + String(params.head_size)
        + ".rope_"
        + String(rope_dim),
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=get_safe_task_id(context),
    ):
        comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
        comptime assert (
            rms_norm_cols % simd_width == 0
        ), "rms_norm_cols must be divisible by simd_width"
        comptime assert (
            rope_dim % simd_width == 0
        ), "rope_dim must be divisible by simd_width"
        comptime assert (
            simd_width % 2 == 0
        ), "simd_width must be even for the split RoPE layout"
        comptime max_warps_per_block = (
            context.default_device_info.max_thread_block_size // WARP_SIZE
        )
        comptime warps_per_block = ceildiv(
            rms_norm_cols // simd_width, WARP_SIZE
        )
        comptime assert (
            warps_per_block <= max_warps_per_block
        ), "fused QK RMSNorm+RoPE block size exceeds device max warps per block"
        var cols = Int(rms_norm_cols)
        comptime block_dim_value = WARP_SIZE * warps_per_block
        comptime kernel = _fused_qk_rms_norm_rope_ragged_paged_gpu[
            cache_t=type_of(k_cache),
            q_out_layout=q_output.LayoutType,
            q_out_origin=q_output.origin,
            q_layout=q_proj.LayoutType,
            q_origin=q_proj.origin,
            q_gamma_layout=q_gamma.LayoutType,
            q_gamma_origin=q_gamma.origin,
            k_gamma_layout=k_gamma.LayoutType,
            k_gamma_origin=k_gamma.origin,
            freqs_layout=freqs_cis.LayoutType,
            freqs_origin=freqs_cis.origin,
            offsets_layout=input_row_offsets.LayoutType,
            offsets_origin=input_row_offsets.origin,
            dtype=dtype,
            freq_dtype=freq_dtype,
            simd_width,
            warps_per_block,
            multiply_before_cast,
            interleaved,
            has_nope_prefix,
            rope_dim,
        ]
        context.enqueue_function[kernel](
            q_output,
            q_proj,
            k_cache,
            q_gamma,
            k_gamma,
            freqs_cis,
            epsilon,
            weight_offset,
            total_seq_len,
            input_row_offsets,
            q_num_heads,
            cols,
            grid_dim=rows,
            block_dim=block_dim_value,
        )


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
        ...,
    ],
    gamma: TileTensor[mut=False, dtype, ...],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    layer_idx: UInt32,
    total_seq_len: UInt32,
    input_row_offsets: TileTensor[mut=False, DType.uint32, ...],
    context: DeviceContext,
) raises:
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
    comptime assert gamma.flat_rank == 1, "gamma must be rank 1"
    comptime assert (
        input_row_offsets.flat_rank == 1
    ), "input_row_offsets must be rank 1"

    # Rank of ragged tensors of shape (total_seq_len, num_heads, head_dim).
    comptime rank = 3 if per_head_norm else 2
    var k_cache = kv_collection.get_key_cache(Int(layer_idx))
    var kv_params = k_cache.kv_params
    comptime rms_norm_cols = gamma.static_shape[0]

    comptime assert rms_norm_cols != -1, "Need static shape for gamma"
    comptime assert (
        rms_norm_cols <= kv_collection.kv_params.head_size or not per_head_norm
    ), "Length of gamma must be smaller or equal to head size"

    var shape = IndexList[rank]()
    shape[0] = Int(total_seq_len)

    comptime if per_head_norm:
        shape[1] = kv_params.num_heads
        shape[2] = rms_norm_cols
    else:
        shape[1] = rms_norm_cols

    @always_inline
    @parameter
    @__copy_capture(k_cache, input_row_offsets)
    def key_cache_input_fn[
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
            head_idx = idx[1] // params.head_size
            head_dim_idx = idx[1] % params.head_size

        return k_cache.load[width=width](
            bs=batch_idx,
            tok_idx=cache_token_idx,
            head_idx=head_idx,
            head_dim_idx=head_dim_idx,
        ).cast[dtype]()

    @always_inline
    @parameter
    @__copy_capture(k_cache)
    def key_cache_output_fn[
        width: SIMDSize, alignment: Int
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
            head_idx = idx[1] // params.head_size
            head_dim_idx = idx[1] % params.head_size
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
        # `_rms_norm_impl` migrated to a `Coord` boundary (softmax PR #88203).
        # The cache lambdas do runtime `idx[0]/idx[1]/idx[2]` subscripts, which
        # `Coord` cannot express, so they stay IndexList-form; wrap them to the
        # `Coord` interface here (`coord_to_index_list` recovers the runtime
        # IndexList the cache logic subscripts) and pass `Coord(shape)`.
        @parameter
        @always_inline
        def key_cache_input_fn_coord[
            width: Int
        ](coords: Coord) -> SIMD[dtype, width]:
            return key_cache_input_fn[width, rank](
                rebind[IndexList[rank]](coord_to_index_list(coords))
            )

        @parameter
        @always_inline
        def key_cache_output_fn_coord[
            width: SIMDSize, alignment: Int
        ](coords: Coord, val: SIMD[dtype, width]) -> None:
            key_cache_output_fn[width, alignment](
                rebind[IndexList[rank]](coord_to_index_list(coords)), val
            )

        _rms_norm_impl[
            dtype,
            rank,
            key_cache_input_fn_coord,
            key_cache_output_fn_coord,
            target=target,
            multiply_before_cast=multiply_before_cast,
        ](
            Coord(shape),
            gamma,
            epsilon,
            weight_offset,
            context,
        )


def rms_norm_value_cache_ragged_paged[
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
        ...,
    ],
    gamma: TileTensor[mut=False, dtype, ...],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    layer_idx: UInt32,
    total_seq_len: UInt32,
    input_row_offsets: TileTensor[mut=False, DType.uint32, ...],
    context: DeviceContext,
) raises:
    """Performs RMSNorm in place on new entries in the value cache.

    Same indexing and layout as ``rms_norm_kv_cache_ragged_paged`` on the key
    cache, but reads/writes the value cache tensor for ``layer_idx``.
    """
    comptime assert gamma.flat_rank == 1, "gamma must be rank 1"
    comptime assert (
        input_row_offsets.flat_rank == 1
    ), "input_row_offsets must be rank 1"

    comptime rank = 3 if per_head_norm else 2
    var v_cache = kv_collection.get_value_cache(Int(layer_idx))
    var kv_params = v_cache.kv_params
    comptime rms_norm_cols = gamma.static_shape[0]

    comptime assert rms_norm_cols != -1, "Need static shape for gamma"
    comptime assert (
        rms_norm_cols <= kv_collection.kv_params.head_size or not per_head_norm
    ), "Length of gamma must be smaller or equal to head size"

    var shape = IndexList[rank]()
    shape[0] = Int(total_seq_len)

    comptime if per_head_norm:
        shape[1] = kv_params.num_heads
        shape[2] = rms_norm_cols
    else:
        shape[1] = rms_norm_cols

    @always_inline
    @parameter
    @__copy_capture(v_cache, input_row_offsets)
    def value_cache_input_fn[
        width: Int, rank_: Int
    ](idx: IndexList[rank_]) -> SIMD[dtype, width]:
        comptime assert rank_ == rank, (
            "rms_norm_value_cache input lambda index should have rank "
            + String(rank)
        )

        var global_token_idx = idx[0]
        var batch_idx = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )
        var token_idx = Int(
            UInt32(global_token_idx) - input_row_offsets[batch_idx]
        )

        var cache_length = v_cache.cache_length(batch_idx)
        var cache_token_idx = token_idx + cache_length

        var head_idx: Int
        var head_dim_idx: Int

        comptime if per_head_norm:
            head_idx = idx[1]
            head_dim_idx = idx[2]
        else:
            head_idx = idx[1] // params.head_size
            head_dim_idx = idx[1] % params.head_size

        return v_cache.load[width=width](
            bs=batch_idx,
            tok_idx=cache_token_idx,
            head_idx=head_idx,
            head_dim_idx=head_dim_idx,
        ).cast[dtype]()

    @always_inline
    @parameter
    @__copy_capture(v_cache)
    def value_cache_output_fn[
        width: SIMDSize, alignment: Int
    ](idx: IndexList[rank], val: SIMD[dtype, width]) -> None:
        var global_token_idx = idx[0]
        var batch_idx = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )
        var token_idx = Int(
            UInt32(global_token_idx) - input_row_offsets[batch_idx]
        )

        var cache_length = v_cache.cache_length(batch_idx)
        var cache_token_idx = token_idx + cache_length

        var head_idx: Int
        var head_dim_idx: Int

        comptime if per_head_norm:
            head_idx = idx[1]
            head_dim_idx = idx[2]
        else:
            head_idx = idx[1] // params.head_size
            head_dim_idx = idx[1] % params.head_size
        v_cache.store(
            bs=batch_idx,
            tok_idx=cache_token_idx,
            head_idx=head_idx,
            head_dim_idx=head_dim_idx,
            val=val.cast[cache_dtype](),
        )

    with Trace[TraceLevel.OP, target=target](
        "rms_norm_value_cache_ragged_paged_nhead_"
        + String(kv_collection.kv_params.num_heads)
        + ".hdim_"
        + String(kv_collection.kv_params.head_size),
        task_id=get_safe_task_id(context),
    ):
        # See `rms_norm_key_cache_ragged_paged` above: cache lambdas stay
        # IndexList-form (runtime index subscripts) and are wrapped to the
        # `Coord` boundary `_rms_norm_impl` now expects.
        @parameter
        @always_inline
        def value_cache_input_fn_coord[
            width: Int
        ](coords: Coord) -> SIMD[dtype, width]:
            return value_cache_input_fn[width, rank](
                rebind[IndexList[rank]](coord_to_index_list(coords))
            )

        @parameter
        @always_inline
        def value_cache_output_fn_coord[
            width: SIMDSize, alignment: Int
        ](coords: Coord, val: SIMD[dtype, width]) -> None:
            value_cache_output_fn[width, alignment](
                rebind[IndexList[rank]](coord_to_index_list(coords)), val
            )

        _rms_norm_impl[
            dtype,
            rank,
            value_cache_input_fn_coord,
            value_cache_output_fn_coord,
            target=target,
            multiply_before_cast=multiply_before_cast,
        ](
            Coord(shape),
            gamma,
            epsilon,
            weight_offset,
            context,
        )


# ===-----------------------------------------------------------------------===#
# Print KV Cache
# ===-----------------------------------------------------------------------===#


# HACK: `cache` is a view into `kv_collection`'s `blocks`, so the two arguments
# share the collection's mutable `blocks_origin`. `_print_cache` only ever READS
# them (it prints), but the exclusivity checker can't prove that and rejects
# passing both. Disabling the nested-origin exclusivity check is safe here
# because this is a read-only debug helper, and it lets the (non-enqueued) print
# wrappers stay origin-generic (`...`) instead of pinning their args to
# any-origin.
@__unsafe_disable_nested_origin_exclusivity
def _print_cache[
    collection_t: KVCollectionT,
    *,
](
    cache: collection_t.CacheType,
    kv_collection: collection_t,
    valid_lengths: LayoutTensor[mut=False, DType.uint32, ...],
    is_print_compact: Bool,
) raises -> None:
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
                        kv_params.head_size,
                    )
                ):
                    print(
                        cache.load[width=1](
                            b_idx,
                            h,
                            t_idx,
                            hd,
                        ),
                        end=", ",
                    )
                if kv_params.head_size > num_to_print:
                    print("...", end=", ")
            if total_cache_length > num_to_print:
                print("\n...", end=",")
            print()


def print_kv_cache_cont_batch_generic_cpu[
    target: StaticString, dtype: DType, kv_params: KVCacheStaticParams
](
    valid_lengths: LayoutTensor[mut=False, DType.uint32, ...],
    kv_collection: ContinuousBatchingKVCacheCollection[dtype, kv_params, ...],
    layer_idx: UInt32,
    is_print_compact: Bool,
    context: DeviceContext,
) raises:
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
    valid_lengths: LayoutTensor[mut=False, DType.uint32, ...],
    kv_collection: PagedKVCacheCollection[
        dtype,
        kv_params,
        page_size,
        ...,
    ],
    layer_idx: UInt32,
    is_print_compact: Bool,
    context: DeviceContext,
) raises:
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
        mut=False, DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    kv_collection: ContinuousBatchingKVCacheCollection[dtype, kv_params, ...],
    layer_idx: UInt32,
    is_print_compact: Bool,
    context: DeviceContext,
) raises:
    # Create host TileTensor copies of device data. Each host copy re-origins the
    # device view's type onto its freshly-allocated host buffer via
    # `OriginCastType`; the host collection is then inferred from those copies.
    var dev_ctx = context

    var n_blocks = kv_collection.blocks.num_elements()
    var blocks_ptr = alloc[Scalar[dtype]](n_blocks)
    dev_ctx.enqueue_copy(blocks_ptr, kv_collection.blocks.ptr, n_blocks)
    var blocks_host = type_of(kv_collection.blocks).OriginCastType[_](
        ptr=blocks_ptr,
        layout=kv_collection.blocks.layout,
    )

    var n_cache_lengths = kv_collection.cache_lengths.num_elements()
    var cache_lengths_ptr = alloc[UInt32](n_cache_lengths)
    dev_ctx.enqueue_copy(
        cache_lengths_ptr, kv_collection.cache_lengths.ptr, n_cache_lengths
    )
    var cache_lengths_host = type_of(
        kv_collection.cache_lengths
    ).OriginCastType[mut=False, _](
        ptr=cache_lengths_ptr,
        layout=kv_collection.cache_lengths.layout,
    )

    var n_lookup_table = kv_collection.lookup_table.num_elements()
    var lookup_table_ptr = alloc[UInt32](n_lookup_table)
    dev_ctx.enqueue_copy(
        lookup_table_ptr, kv_collection.lookup_table.ptr, n_lookup_table
    )
    var lookup_table_host = type_of(kv_collection.lookup_table).OriginCastType[
        mut=False, _
    ](
        ptr=lookup_table_ptr,
        layout=kv_collection.lookup_table.layout,
    )

    var host_kv_collection = ContinuousBatchingKVCacheCollection[
        dtype, kv_params
    ](
        blocks_host,
        cache_lengths_host,
        lookup_table_host,
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
    _print_cache[type_of(host_kv_collection)](
        k_cache,
        host_kv_collection,
        valid_lengths_host_nd,
        is_print_compact,
    )

    print("V:")
    _print_cache[type_of(host_kv_collection)](
        v_cache,
        host_kv_collection,
        valid_lengths_host_nd,
        is_print_compact,
    )

    blocks_ptr.free()
    cache_lengths_ptr.free()
    lookup_table_ptr.free()
    valid_lengths_host_ptr.free()


def print_kv_cache_paged_generic_gpu[
    target: StaticString,
    dtype: DType,
    kv_params: KVCacheStaticParams,
    page_size: Int,
](
    valid_lengths: LayoutTensor[
        mut=False, DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    kv_collection: PagedKVCacheCollection[
        dtype,
        kv_params,
        page_size,
        ...,
    ],
    layer_idx: UInt32,
    is_print_compact: Bool,
    context: DeviceContext,
) raises:
    var dev_ctx = context

    var n_blocks = kv_collection.blocks.num_elements()
    var blocks_ptr = alloc[Scalar[dtype]](n_blocks)
    dev_ctx.enqueue_copy(blocks_ptr, kv_collection.blocks.ptr, n_blocks)
    var blocks_host = type_of(kv_collection.blocks).OriginCastType[_](
        ptr=blocks_ptr,
        layout=kv_collection.blocks.layout,
    )

    var n_cache_lengths = kv_collection.cache_lengths.num_elements()
    var cache_lengths_ptr = alloc[UInt32](n_cache_lengths)
    dev_ctx.enqueue_copy(
        cache_lengths_ptr, kv_collection.cache_lengths.ptr, n_cache_lengths
    )
    var cache_lengths_host = type_of(
        kv_collection.cache_lengths
    ).OriginCastType[mut=False, _](
        ptr=cache_lengths_ptr,
        layout=kv_collection.cache_lengths.layout,
    )

    var n_lookup_table = kv_collection.lookup_table.num_elements()
    var lookup_table_ptr = alloc[UInt32](n_lookup_table)
    dev_ctx.enqueue_copy(
        lookup_table_ptr, kv_collection.lookup_table.ptr, n_lookup_table
    )
    var lookup_table_host = type_of(kv_collection.lookup_table).OriginCastType[
        mut=False, _
    ](
        ptr=lookup_table_ptr,
        layout=kv_collection.lookup_table.layout,
    )

    # The host copies are `TileTensor`s (from `OriginCastType`), so this picks
    # the `TileTensor` constructor, whose `scales` default is a bare `None` that
    # cannot pin `scales_origin`. Thread the copies' origins explicitly and pin
    # `scales_origin` to the no-scales default. (Binding only the leading params
    # would infer everything for `LayoutTensor` inputs, but not here.)
    var host_kv_collection = PagedKVCacheCollection[
        dtype,
        kv_params,
        page_size,
        blocks_host.origin,
        cache_lengths_host.origin,
        lookup_table_host.origin,
        MutUntrackedOrigin,
    ](
        blocks_host,
        cache_lengths_host,
        lookup_table_host,
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
    _print_cache[type_of(host_kv_collection)](
        k_cache,
        host_kv_collection,
        valid_lengths_host_nd,
        is_print_compact,
    )

    print("V:")
    _print_cache[type_of(host_kv_collection)](
        v_cache,
        host_kv_collection,
        valid_lengths_host_nd,
        is_print_compact,
    )

    blocks_ptr.free()
    cache_lengths_ptr.free()
    lookup_table_ptr.free()
    valid_lengths_host_ptr.free()


# ===-----------------------------------------------------------------------===#
# KV Collection Constructors (Ctor)
# ===-----------------------------------------------------------------------===#


def _continuous_batch_kv_cache_collection[
    dtype: DType, //, kv_params: KVCacheStaticParams
](
    blocks: LayoutTensor[mut=True, dtype, Layout.row_major[6](), _],
    cache_lengths: LayoutTensor[
        mut=False, DType.uint32, Layout(UNKNOWN_VALUE), _
    ],
    lookup_table: LayoutTensor[
        mut=False, DType.uint32, Layout(UNKNOWN_VALUE), _
    ],
    max_prompt_length: LayoutTensor[
        mut=False, DType.uint32, Layout.row_major[1](), _
    ],
    max_cache_length: LayoutTensor[
        mut=False, DType.uint32, Layout.row_major[1](), _
    ],
    out result: ContinuousBatchingKVCacheCollection[
        dtype,
        kv_params,
        blocks.origin,
        cache_lengths.origin,
        lookup_table.origin,
    ],
):
    # Marshal LayoutTensor into arguments expected by the
    # ContinuousKVCacheCollection constructor. The collection carries the
    # input tensors' origins, so the borrow checker keeps the backing
    # buffers alive for as long as the collection (and any cache views
    # derived from it) are in use.
    return {
        blocks = blocks,
        cache_lengths = cache_lengths,
        lookup_table = lookup_table,
        max_seq_length = max_prompt_length[0][0],
        max_cache_length = max_cache_length[0][0],
    }


@always_inline
def generic_get_continuous_cache[
    dtype: DType, kv_params: KVCacheStaticParams
](
    blocks: LayoutTensor[mut=True, dtype, Layout.row_major[6](), _],
    cache_lengths: LayoutTensor[
        mut=False, DType.uint32, Layout(UNKNOWN_VALUE), _
    ],
    lookup_table: LayoutTensor[
        mut=False, DType.uint32, Layout(UNKNOWN_VALUE), _
    ],
    max_prompt_length: LayoutTensor[
        mut=False, DType.uint32, Layout.row_major[1](), _
    ],
    max_cache_length: LayoutTensor[
        mut=False, DType.uint32, Layout.row_major[1](), _
    ],
) -> ContinuousBatchingKVCacheCollection[
    dtype,
    kv_params,
    blocks.origin,
    cache_lengths.origin,
    lookup_table.origin,
]:
    return _continuous_batch_kv_cache_collection[kv_params](
        blocks,
        cache_lengths,
        lookup_table,
        max_prompt_length,
        max_cache_length,
    )


def generic_get_paged_cache[
    dtype: DType,
](
    blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
    cache_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
    lookup_table: InputTensor[dtype=DType.uint32, rank=2, ...],
    max_prompt_length: InputTensor[dtype=DType.uint32, rank=1, ...],
    max_cache_length: InputTensor[dtype=DType.uint32, rank=1, ...],
    out result: PagedKVCacheCollection[
        dtype,
        KVCacheStaticParams(
            Int(blocks.static_spec.shape_tuple[4]),
            Int(blocks.static_spec.shape_tuple[5]),
            Int(blocks.static_spec.shape_tuple[1]) == 1,
        ),
        Int(blocks.static_spec.shape_tuple[3]),
        # MOGG boundary: the device buffers are owned by the graph runtime
        # (kept alive externally), so the views built from `unsafe_ptr()`
        # below carry the untracked any-origins.
        # TODO: These should probably be UntrackedOrigin.
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutUntrackedOrigin,
    ],
):
    comptime page_size = Int(blocks.static_spec.shape_tuple[3])
    comptime head_dim = Int(blocks.static_spec.shape_tuple[5])
    comptime num_heads = Int(blocks.static_spec.shape_tuple[4])
    comptime is_mla = Int(blocks.static_spec.shape_tuple[1]) == 1
    return generic_get_paged_cache[
        dtype,
        KVCacheStaticParams(num_heads, head_dim, is_mla),
        page_size,
    ](
        LayoutTensor[blocks.dtype, Layout.row_major[6](), MutAnyOrigin](
            blocks.unsafe_ptr(),
            RuntimeLayout[Layout.row_major[6]()].row_major(blocks.shape()),
        ),
        LayoutTensor[
            cache_lengths.dtype, Layout(UNKNOWN_VALUE), ImmutAnyOrigin
        ](
            cache_lengths.unsafe_ptr(),
            RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(
                cache_lengths.shape()
            ),
        ),
        LayoutTensor[lookup_table.dtype, Layout.row_major[2](), ImmutAnyOrigin](
            lookup_table.unsafe_ptr(),
            RuntimeLayout[Layout.row_major[2]()].row_major(
                lookup_table.shape()
            ),
        ),
        LayoutTensor[
            max_prompt_length.dtype, Layout.row_major[1](), ImmutAnyOrigin
        ](
            max_prompt_length.unsafe_ptr(),
            RuntimeLayout[Layout.row_major[1]()].row_major(
                max_prompt_length.shape()
            ),
        ),
        LayoutTensor[
            max_cache_length.dtype, Layout.row_major[1](), ImmutAnyOrigin
        ](
            max_cache_length.unsafe_ptr(),
            RuntimeLayout[Layout.row_major[1]()].row_major(
                max_cache_length.shape()
            ),
        ),
    )


def generic_get_paged_cache[
    dtype: DType,
    kv_params: KVCacheStaticParams,
    page_size: Int,
](
    blocks: LayoutTensor[mut=True, dtype, Layout.row_major[6](), _],
    cache_lengths: LayoutTensor[
        mut=False, DType.uint32, Layout(UNKNOWN_VALUE), _
    ],
    lookup_table: LayoutTensor[
        mut=False, DType.uint32, Layout.row_major[2](), _
    ],
    max_prompt_length: LayoutTensor[
        mut=False, DType.uint32, Layout.row_major[1](), _
    ],
    max_cache_length: LayoutTensor[
        mut=False, DType.uint32, Layout.row_major[1](), _
    ],
    out result: PagedKVCacheCollection[
        dtype,
        kv_params,
        page_size,
        blocks.origin,
        cache_lengths.origin,
        lookup_table.origin,
        # No scales on this (non-quantized) path: matches the constructor's
        # default `scales=None`, whose untracked origin is MutUntrackedOrigin.
        MutUntrackedOrigin,
    ],
):
    # Thread the input tensors' origins into the collection so the borrow
    # checker keeps the backing buffers alive across the collection's use.
    return {
        blocks = blocks,
        cache_lengths = cache_lengths,
        lookup_table = lookup_table,
        max_seq_length = max_prompt_length[0][0],
        max_cache_length = max_cache_length[0][0],
    }


def generic_get_paged_cache_with_scales[
    dtype: DType,
    scale_dtype: DType,
    kv_params: KVCacheStaticParams,
    page_size: Int,
    quantization_granularity: Int,
](
    blocks: LayoutTensor[mut=True, dtype, Layout.row_major[6](), _],
    cache_lengths: LayoutTensor[
        mut=False, DType.uint32, Layout(UNKNOWN_VALUE), _
    ],
    lookup_table: LayoutTensor[
        mut=False, DType.uint32, Layout.row_major[2](), _
    ],
    max_prompt_length: LayoutTensor[
        mut=False, DType.uint32, Layout.row_major[1](), _
    ],
    max_cache_length: LayoutTensor[
        mut=False, DType.uint32, Layout.row_major[1](), _
    ],
    scales: LayoutTensor[mut=True, scale_dtype, Layout.row_major[6](), _],
    out result: PagedKVCacheCollection[
        dtype,
        kv_params,
        page_size,
        blocks.origin,
        cache_lengths.origin,
        lookup_table.origin,
        scales.origin,
        scale_dtype_=scale_dtype,
        quantization_granularity_=quantization_granularity,
    ],
):
    """Create a PagedKVCacheCollection with scales for MLA attention.

    Args:
        blocks: KV cache blocks tensor [num_blocks, kv_dim, num_layers, page_size, num_heads, head_dim].
        cache_lengths: Cache lengths per batch [batch_size].
        lookup_table: Page lookup table [batch_size, max_pages].
        max_prompt_length: Max prompt (query) length scalar tensor [1].
        max_cache_length: Max cache length scalar tensor [1].
        scales: Scales tensor [num_blocks, kv_dim, num_layers, page_size, num_heads, head_dim_granularity].
    """
    # Thread the input tensors' origins into the collection so the borrow
    # checker keeps the backing buffers alive across the collection's use.
    return {
        blocks = blocks,
        cache_lengths = cache_lengths,
        lookup_table = lookup_table,
        max_seq_length = max_prompt_length[0][0],
        max_cache_length = max_cache_length[0][0],
        scales = scales,
    }


# ===-----------------------------------------------------------------------===#
# GPU→CPU Page Copy for KV Cache Offloading
# ===-----------------------------------------------------------------------===#


def copy_kv_pages_d2h[
    dtype: DType,
](
    device_kv_blocks: LayoutTensor[mut=True, dtype, Layout.row_major[6](), _],
    host_kv_blocks: LayoutTensor[mut=True, dtype, Layout.row_major[6](), _],
    src_page_ids: LayoutTensor[DType.int64, Layout.row_major[1](), _],
    dst_page_ids: LayoutTensor[DType.int64, Layout.row_major[1](), _],
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
