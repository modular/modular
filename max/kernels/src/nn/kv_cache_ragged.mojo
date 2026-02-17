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

from sys.info import _current_target, simd_width_of
from sys.intrinsics import _type_is_eq

from algorithm.functional import elementwise, unswitch
from gpu.host import DeviceContext, get_gpu_target
from gpu.host.info import is_cpu, is_gpu
from collections import OptionalReg
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
    KVCacheT,
    KVCollectionT,
    PagedKVCache,
    PagedKVCacheCollection,
)
from layout import LayoutTensor, Layout, RuntimeLayout, IntTuple, UNKNOWN_VALUE
from layout._coord import Coord, CoordLike, Idx, coord_to_index_list
from layout._layout import row_major
from layout._tile_tensor import TileTensor
from linalg.grouped_matmul import grouped_matmul
from linalg.matmul import elementwise_epilogue_type, matmul
from linalg.fp8_quantization import blockwise_scaled_fp8_with_epilogue
from linalg.fp4_quantization import (
    block_scaled_matmul_with_epilogue as blockwise_scaled_fp4_with_epilogue,
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

from utils.index import IndexList

# ===-----------------------------------------------------------------------===#
# Fused QKV matmul (ragged)
# ===-----------------------------------------------------------------------===#


@always_inline
fn generic_fused_qkv_matmul_kv_cache_paged_ragged[
    dtype: DType,
    weight_dtype: DType,
    target: StaticString = "cpu",
    group_size: Optional[Int] = None,
    has_zp: Optional[Bool] = None,
](
    hidden_state: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    input_row_offsets: LayoutTensor[
        mut=False, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[
        mut=False, weight_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    kv_collection: PagedKVCacheCollection,
    layer_idx: UInt32,
    output: LayoutTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
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
            Span(
                [
                    trace_arg("output", output.runtime_layout.shape.value),
                    trace_arg(
                        "hidden_state", hidden_state.runtime_layout.shape.value
                    ),
                    trace_arg("weight", weight.runtime_layout.shape.value),
                    "layer_idx=" + String(layer_idx),
                    "num_heads=" + String(kv_collection.kv_params.num_heads),
                    "head_size=" + String(kv_collection.kv_params.head_size),
                ]
            )
        )

    comptime name = "mo.fused_qkv_matmul.ragged.paged.nhead_" + String(
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
    group_size: Optional[Int] = None,
    has_zp: Optional[Bool] = None,
](
    hidden_state: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    input_row_offsets: LayoutTensor[
        mut=False, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[
        mut=False, weight_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    kv_collection: PagedKVCacheCollection,
    layer_idx: UInt32,
    output: LayoutTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    bias: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
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
            Span(
                [
                    trace_arg("output", output.runtime_layout.shape.value),
                    trace_arg(
                        "hidden_state", hidden_state.runtime_layout.shape.value
                    ),
                    trace_arg("weight", weight.runtime_layout.shape.value),
                    "layer_idx=" + String(layer_idx),
                    "num_heads=" + String(kv_collection.kv_params.num_heads),
                    "head_size=" + String(kv_collection.kv_params.head_size),
                ]
            )
        )

    comptime name = "mo.fused_qkv_matmul.ragged.paged.bias.nhead_" + String(
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
    scales_granularity_mnk: IndexList[3],
    target: StaticString = "cpu",
](
    hidden_state: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    input_row_offsets: LayoutTensor[
        mut=False, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[
        mut=False, weight_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    input_scale: LayoutTensor[
        mut=False, scale_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    weight_scale: LayoutTensor[
        mut=False, scale_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    kv_collection: PagedKVCacheCollection,
    layer_idx: UInt32,
    output: LayoutTensor[
        mut=True, output_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    ctx: DeviceContextPtr,
    bias: OptionalReg[
        LayoutTensor[
            output_dtype,
            Layout.row_major(UNKNOWN_VALUE),
            ImmutAnyOrigin,
            address_space = AddressSpace.GENERIC,
        ]
    ] = None,
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
        bias: Optional bias vector concatenated as [q, k, v].
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
                        "input_scale", input_scale.runtime_layout.shape.value
                    ),
                    trace_arg(
                        "weight_scale", weight_scale.runtime_layout.shape.value
                    ),
                    "layer_idx=" + String(layer_idx),
                    "num_heads=" + String(kv_collection.kv_params.num_heads),
                    "head_size=" + String(kv_collection.kv_params.head_size),
                ]
            )
        )

    comptime name = "mo.fused_qkv_matmul.ragged.paged.scale.nhead_" + String(
        kv_collection.kv_params.num_heads
    ) + ".hdim_" + String(
        kv_collection.kv_params.head_size
    ) + ".m_scale_granularity_" + String(
        scales_granularity_mnk[0]
    ) + ".n_scale_granularity_" + String(
        scales_granularity_mnk[1]
    ) + ".k_scale_granularity_" + String(
        scales_granularity_mnk[2]
    )
    with Trace[TraceLevel.OP, target=target](
        name,
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=Int(ctx.get_device_context().id()),
    ):
        return _fused_qkv_matmul_kv_cache_ragged_scale[
            kv_collection.CacheType,
            scales_granularity_mnk=scales_granularity_mnk,
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
            bias,
        )


@always_inline
fn generic_fused_qkv_matmul_kv_cache_paged_ragged_scale_float4[
    dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    scale_dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    sfa_layout: Layout,
    sfb_layout: Layout,
    SF_VECTOR_SIZE: Int,
    target: StaticString = "cpu",
](
    hidden_state: LayoutTensor[mut=False, dtype, a_layout, MutAnyOrigin],
    input_row_offsets: LayoutTensor[
        mut=False, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[mut=False, weight_dtype, b_layout, MutAnyOrigin],
    input_scale: LayoutTensor[mut=False, scale_dtype, sfa_layout, MutAnyOrigin],
    weight_scale: LayoutTensor[
        mut=False, scale_dtype, sfb_layout, MutAnyOrigin
    ],
    tensor_sf: Float32,
    kv_collection: PagedKVCacheCollection,
    layer_idx: UInt32,
    output: LayoutTensor[
        mut=True, output_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    ctx: DeviceContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (sum(seq_lens), num_heads * head_size // 2).
        input_row_offsets: Tensor with shape (batch_size + 1,).
            The value at each index is the start_idx of the corresponding batch
            in hidden_state.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads *
            head_size // 2).
        input_scale: 5D blockwise scale tensor to be multiplied to the input Tensor.
        weight_scale: 5D blockwise scale tensor to the weight Tensor.
        tensor_sf: Per-tensor scaling factor.
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
            Span(
                [
                    trace_arg("output", output.runtime_layout.shape.value),
                    trace_arg(
                        "hidden_state", hidden_state.runtime_layout.shape.value
                    ),
                    trace_arg("weight", weight.runtime_layout.shape.value),
                    trace_arg(
                        "input_scale", input_scale.runtime_layout.shape.value
                    ),
                    trace_arg(
                        "weight_scale", weight_scale.runtime_layout.shape.value
                    ),
                    "tensor_sf=" + String(tensor_sf),
                    "layer_idx=" + String(layer_idx),
                    "num_heads=" + String(kv_collection.kv_params.num_heads),
                    "head_size=" + String(kv_collection.kv_params.head_size),
                ]
            )
        )

    comptime name = "mo.fused_qkv_matmul.ragged.paged.scale.nhead_" + String(
        kv_collection.kv_params.num_heads
    ) + ".hdim_" + String(kv_collection.kv_params.head_size)
    with Trace[TraceLevel.OP, target=target](
        name,
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=Int(ctx.get_device_context().id()),
    ):
        return _fused_qkv_matmul_kv_cache_ragged_scale_float4[
            kv_collection.CacheType,
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            target=target,
        ](
            hidden_state,
            input_row_offsets,
            weight,
            input_scale,
            weight_scale,
            tensor_sf,
            kv_collection,
            layer_idx,
            output,
            ctx,
        )


@always_inline
fn _fused_qkv_matmul_kv_cache_ragged[
    dtype: DType,
    weight_dtype: DType,
    collection_t: KVCollectionT,
    //,
    cache_t: KVCacheT,
    *,
    target: StaticString,
    group_size: Optional[Int] = None,
    has_zp: Optional[Bool] = None,
](
    hidden_state: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    input_row_offsets: LayoutTensor[
        mut=False, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[
        mut=False, weight_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    kv_collection: collection_t,
    layer_idx: UInt32,
    output: LayoutTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
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
    collection_t: KVCollectionT,
    //,
    cache_t: KVCacheT,
    *,
    target: StaticString,
    group_size: Optional[Int] = None,
    has_zp: Optional[Bool] = None,
](
    hidden_state: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    input_row_offsets: LayoutTensor[
        mut=False, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[
        mut=False, weight_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    kv_collection: collection_t,
    layer_idx: UInt32,
    output: LayoutTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    bias: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
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
    collection_t: KVCollectionT,
    //,
    cache_t: KVCacheT,
    scales_granularity_mnk: IndexList[3],
    *,
    target: StaticString,
](
    hidden_state: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    input_row_offsets: LayoutTensor[
        mut=False, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[
        mut=False, weight_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    input_scale: LayoutTensor[
        mut=False, scale_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    weight_scale: LayoutTensor[
        scale_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    kv_collection: collection_t,
    layer_idx: UInt32,
    output: LayoutTensor[
        mut=True, output_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    context: DeviceContextPtr,
    bias: OptionalReg[
        LayoutTensor[
            output_dtype,
            Layout.row_major(UNKNOWN_VALUE),
            ImmutAnyOrigin,
            address_space = AddressSpace.GENERIC,
        ]
    ] = None,
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
        bias: Optional bias vector concatenated as [q, k, v].
    """
    var cuda_ctx: Optional[DeviceContext] = None
    var layer_idx_cast = Int(layer_idx)
    var k_cache = kv_collection.get_key_cache(layer_idx_cast)
    var v_cache: OptionalReg[type_of(k_cache)] = None
    comptime kv_params = collection_t.kv_params

    @parameter
    if not kv_params.is_mla:
        v_cache = kv_collection.get_value_cache(layer_idx_cast)

    @parameter
    if is_gpu[target]():
        cuda_ctx = context.get_device_context()

    return _fused_qkv_matmul_kv_cache_ragged_impl_scale[
        scales_granularity_mnk=scales_granularity_mnk, target=target
    ](
        hidden_state,
        input_row_offsets,
        weight,
        input_scale,
        weight_scale,
        k_cache,
        v_cache,
        output,
        cuda_ctx,
        bias,
    )


@always_inline
fn _fused_qkv_matmul_kv_cache_ragged_scale_float4[
    dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    scale_dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    sfa_layout: Layout,
    sfb_layout: Layout,
    collection_t: KVCollectionT,
    //,
    cache_t: KVCacheT,
    SF_VECTOR_SIZE: Int,
    *,
    target: StaticString,
](
    hidden_state: LayoutTensor[mut=False, dtype, a_layout, MutAnyOrigin],
    input_row_offsets: LayoutTensor[
        mut=False, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[mut=False, weight_dtype, b_layout, MutAnyOrigin],
    input_scale: LayoutTensor[mut=False, scale_dtype, sfa_layout, MutAnyOrigin],
    weight_scale: LayoutTensor[
        mut=False, scale_dtype, sfb_layout, MutAnyOrigin
    ],
    tensor_sf: Float32,
    kv_collection: collection_t,
    layer_idx: UInt32,
    output: LayoutTensor[
        mut=True, output_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    context: DeviceContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads *
            head_size // 2).
        input_row_offsets: Tensor with shape (batch_size + 1,).
            The value at each index is the start_idx of the corresponding batch
            in hidden_state.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads *
            head_size // 2).
        input_scale: 5D blockwise scale tensor to be multiplied to the input Tensor.
        weight_scale: 5D blockwise scale tensor to the weight Tensor.
        tensor_sf: Per-tensor scaling factor.
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
    var v_cache: OptionalReg[type_of(k_cache)] = None
    comptime kv_params = collection_t.kv_params

    @parameter
    if not kv_params.is_mla:
        v_cache = kv_collection.get_value_cache(layer_idx_cast)

    @parameter
    if is_gpu[target]():
        cuda_ctx = context.get_device_context()

    return _fused_qkv_matmul_kv_cache_ragged_impl_scale_float4[
        SF_VECTOR_SIZE=SF_VECTOR_SIZE, target=target
    ](
        hidden_state,
        input_row_offsets,
        weight,
        input_scale,
        weight_scale,
        tensor_sf,
        k_cache,
        v_cache,
        output,
        cuda_ctx,
    )


@always_inline
fn _fused_qkv_matmul_kv_cache_ragged_impl[
    dtype: DType,
    weight_dtype: DType,
    cache_t: KVCacheT,
    //,
    *,
    target: StaticString,
    group_size: Optional[Int] = None,
    has_zp: Optional[Bool] = None,
](
    hidden_state: LayoutTensor[
        dtype, address_space = AddressSpace.GENERIC, ...
    ],
    input_row_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[
        weight_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    k_cache: cache_t,
    v_cache: cache_t,
    output: LayoutTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
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
    comptime kv_type = cache_t.dtype
    comptime kv_params = cache_t.kv_params

    comptime assert (
        kv_type == dtype
    ), "Mismatch in dtype between Q and KV tensors"

    var q_dim = output.dim[1]()
    var k_dim = kv_params.head_size * kv_params.num_heads
    var qk_offset = q_dim + Int(k_dim)
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
            output.store[width=width](
                idx,
                rebind[SIMD[dtype, width]](val),
            )
            return

        global_token_idx = idx[0]

        var batch_idx: Int = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )

        token_idx = Int(UInt32(global_token_idx) - input_row_offsets[batch_idx])

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
            Int(h_idx),
            cache_token_idx,
            Int(hd_idx),
            rebind[SIMD[kv_type, width]](output_val),
        )

    @parameter
    if group_size:
        comptime assert (
            not has_zp.value()
        ), "Zero point is not supported for quantization."
        comptime assert (
            weight_dtype == DType.uint8
        ), "Expect GPTQ weights in an uint8 tensor."

        _qmatmul_common[
            group_size = group_size.value(),
            target=target,
            elementwise_lambda_fn=write_to_cache,
        ](hidden_state, weight.bitcast[DType.uint8](), context)

    else:
        comptime assert (
            weight_dtype == dtype
        ), "Mismatch in dtype between weight and QKV tensors"

        _matmul_common[target=target, elementwise_lambda_fn=write_to_cache](
            hidden_state, weight.bitcast[dtype](), context
        )


@always_inline
fn _fused_qkv_matmul_kv_cache_ragged_impl_bias[
    dtype: DType,
    weight_dtype: DType,
    cache_t: KVCacheT,
    //,
    *,
    target: StaticString,
    group_size: Optional[Int] = None,
    has_zp: Optional[Bool] = None,
](
    hidden_state: LayoutTensor[
        dtype, address_space = AddressSpace.GENERIC, ...
    ],
    input_row_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[
        weight_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    k_cache: cache_t,
    v_cache: cache_t,
    output: LayoutTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    bias: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, ...],
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
    comptime kv_type = cache_t.dtype
    comptime kv_params = cache_t.kv_params

    comptime assert (
        kv_type == dtype
    ), "Mismatch in dtype between Q and KV tensors"

    var q_dim = output.dim[1]()
    var k_dim = kv_params.head_size * kv_params.num_heads
    var qk_offset = q_dim + Int(k_dim)
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
            bias.load[width=width](IndexList[1](idx[1]))
        )
        if idx[1] < q_dim:
            output.store[width=width](
                idx,
                rebind[SIMD[dtype, width]](output_val),
            )
            return

        global_token_idx = idx[0]

        var batch_idx: Int = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )

        token_idx = Int(UInt32(global_token_idx) - input_row_offsets[batch_idx])

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
            Int(h_idx),
            cache_token_idx,
            Int(hd_idx),
            rebind[SIMD[kv_type, width]](output_val),
        )

    @parameter
    if group_size:
        comptime assert (
            not has_zp.value()
        ), "Zero point is not supported for quantization."
        comptime assert (
            weight_dtype == DType.uint8
        ), "Expect GPTQ weights to be a 'uint8' tensor."

        _qmatmul_common[
            group_size = group_size.value(),
            target=target,
            elementwise_lambda_fn=write_to_cache,
        ](hidden_state, weight.bitcast[DType.uint8](), context)

    else:
        comptime assert (
            weight_dtype == dtype
        ), "Mismatch in dtype between weight and QKV tensors"

        _matmul_common[target=target, elementwise_lambda_fn=write_to_cache](
            hidden_state, weight.bitcast[dtype](), context
        )


@always_inline
fn _fused_qkv_matmul_kv_cache_ragged_impl_scale[
    dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    scale_dtype: DType,
    cache_t: KVCacheT,
    //,
    scales_granularity_mnk: IndexList[3],
    *,
    target: StaticString,
](
    hidden_state: LayoutTensor[
        dtype, address_space = AddressSpace.GENERIC, ...
    ],
    input_row_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[
        weight_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    input_scale: LayoutTensor[
        scale_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    weight_scale: LayoutTensor[
        scale_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    k_cache: cache_t,
    v_cache: OptionalReg[cache_t],
    output: LayoutTensor[
        mut=True, output_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    context: Optional[DeviceContext],
    bias: OptionalReg[
        LayoutTensor[
            mut=False,
            output_dtype,
            Layout.row_major(UNKNOWN_VALUE),
            MutAnyOrigin,
            address_space = AddressSpace.GENERIC,
        ]
    ] = None,
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
        bias: Optional bias vector concatenated as [q, k, v].
    """
    comptime kv_type = cache_t.dtype
    comptime kv_params = cache_t.kv_params

    var q_dim = output.dim[1]()
    var k_dim = kv_params.head_size * kv_params.num_heads
    var qk_offset = q_dim + Int(k_dim)
    var batch_size = input_row_offsets.dim[0]() - 1

    if batch_size == 0:
        return

    # Here we decide the quantization scheme for the QKV Tensor.
    comptime use_per_tensor = (
        scales_granularity_mnk[0] == -1
        and scales_granularity_mnk[1] == -1
        and scales_granularity_mnk[2] == -1
    )
    comptime use_per_channel = (
        scales_granularity_mnk[0] == 1
        and scales_granularity_mnk[1] == 1
        and scales_granularity_mnk[2] == -1
    )
    comptime use_block_wise = not (use_per_tensor or use_per_channel)

    @parameter
    @__copy_capture(
        input_scale, weight_scale, q_dim, qk_offset, batch_size, bias
    )
    @always_inline
    fn write_to_cache[
        dtype: DType, width: Int, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]):
        var output_val: SIMD[dtype, width]

        @parameter
        if use_per_tensor:
            var scale_a = input_scale[0, 0][0].cast[dtype]()
            var scale_b = weight_scale[0, 0][0].cast[dtype]()
            output_val = val * (scale_a * scale_b)
        elif use_per_channel:
            var scale_a = input_scale.load[width=1](0, idx[0]).cast[dtype]()
            var scale_b = weight_scale.load[width=width](idx[1], 0).cast[
                dtype
            ]()
            output_val = val * (scale_a * scale_b)
        else:
            # blockwise quantization, we need to use the blockwise_scaled_fp8_with_epilogue kernel
            output_val = val

        var output_val_out: SIMD[output_dtype, width] = rebind[
            SIMD[output_dtype, width]
        ](output_val.cast[output_dtype]())

        if bias:
            output_val_out += bias.value().load[width=width](
                IndexList[1](idx[1])
            )

        if idx[1] < q_dim:
            output.store[width=width](
                idx,
                output_val_out,
            )
            return

        global_token_idx = idx[0]

        var batch_idx: Int = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )

        token_idx = Int(UInt32(global_token_idx) - input_row_offsets[batch_idx])

        var h_idx: UInt
        var hd_idx: UInt
        var cache: cache_t

        @parameter
        if kv_params.is_mla:
            cache = k_cache
            h_idx = 0  # in MLA mode we only have one head
            hd_idx = UInt(idx[1]) - UInt(q_dim)

        else:
            if idx[1] < qk_offset:
                cache = k_cache
                h_idx, hd_idx = divmod(
                    UInt(idx[1]) - UInt(q_dim), kv_params.head_size
                )
            else:
                cache = v_cache.value()
                h_idx, hd_idx = divmod(
                    UInt(idx[1]) - UInt(qk_offset), kv_params.head_size
                )

        var cache_length = cache.cache_length(batch_idx)
        var cache_token_idx = token_idx + cache_length
        cache.store(
            batch_idx,
            Int(h_idx),
            cache_token_idx,
            Int(hd_idx),
            rebind[SIMD[kv_type, width]](output_val_out.cast[kv_type]()),
        )

    comptime assert (
        weight_dtype == dtype
    ), "Mismatch in dtype between weight and QKV tensors"

    @parameter
    if use_block_wise:
        comptime assert is_gpu[
            target
        ](), "Blockwise scaled fp8 matmul only works on GPU."

        _matmul_blockwise_scaled_fp8_common[
            output_dtype = cache_t.dtype,
            target=target,
            scales_granularity_mnk=scales_granularity_mnk,
            elementwise_lambda_fn=write_to_cache,
        ](hidden_state, weight, input_scale, weight_scale, context.value())
    else:
        _matmul_common[
            target=target,
            elementwise_lambda_fn=write_to_cache,
            output_dtype=output_dtype,
        ](hidden_state, weight.bitcast[dtype](), context)


@always_inline
fn _fused_qkv_matmul_kv_cache_ragged_impl_scale_float4[
    dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    scale_dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    sfa_layout: Layout,
    sfb_layout: Layout,
    cache_t: KVCacheT,
    //,
    SF_VECTOR_SIZE: Int,
    *,
    target: StaticString,
](
    hidden_state: LayoutTensor[dtype, a_layout, ImmutAnyOrigin],
    input_row_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[weight_dtype, b_layout, ImmutAnyOrigin],
    input_scale: LayoutTensor[scale_dtype, sfa_layout, ImmutAnyOrigin],
    weight_scale: LayoutTensor[scale_dtype, sfb_layout, ImmutAnyOrigin],
    tensor_sf: Float32,
    k_cache: cache_t,
    v_cache: OptionalReg[cache_t],
    output: LayoutTensor[
        mut=True, output_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    context: Optional[DeviceContext],
) raises:
    """Performs a fused QKV matmul on ragged tensors. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads *
            head_size // 2).
        input_row_offsets: Tensor with shape (batch_size + 1,).
            The value at each index is the start_idx of the corresponding batch
            in hidden_state.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads *
            head_size // 2).
        input_scale: 5D blockwise scale tensor to be multiplied to the input Tensor.
        weight_scale: 5D blockwise scale tensor to the weight Tensor.
        tensor_sf: Per-tensor scaling factor.
        k_cache: The historical KVCacheT for keys, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        v_cache: The historical KVCacheT for values, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
            Shape is (sum(seq_lens), num_heads * head_size)
        context: The DeviceContext. This is unused if is_cpu[target]().
    """
    comptime kv_type = cache_t.dtype
    comptime kv_params = cache_t.kv_params

    var q_dim = output.dim[1]()
    var k_dim = kv_params.head_size * kv_params.num_heads
    var qk_offset = q_dim + Int(k_dim)
    var batch_size = input_row_offsets.dim[0]() - 1

    if batch_size == 0:
        return

    @parameter
    @__copy_capture(input_scale, weight_scale, q_dim, qk_offset, batch_size)
    @always_inline
    fn write_to_cache[
        dtype: DType, width: Int, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]):
        # blockwise quantization, we need to use the blockwise_scaled_fp4_with_epilogue kernel
        var output_val_out: SIMD[output_dtype, width] = rebind[
            SIMD[output_dtype, width]
        ](val.cast[output_dtype]())

        if idx[1] < q_dim:
            output.store[width=width](
                idx,
                output_val_out,
            )
            return

        global_token_idx = idx[0]

        var batch_idx: Int = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )

        token_idx = Int(UInt32(global_token_idx) - input_row_offsets[batch_idx])

        var h_idx: UInt
        var hd_idx: UInt
        var cache: cache_t

        @parameter
        if kv_params.is_mla:
            cache = k_cache
            h_idx = 0  # in MLA mode we only have one head
            hd_idx = UInt(idx[1]) - UInt(q_dim)

        else:
            if idx[1] < qk_offset:
                cache = k_cache
                h_idx, hd_idx = divmod(
                    UInt(idx[1]) - UInt(q_dim), kv_params.head_size
                )
            else:
                cache = v_cache.value()
                h_idx, hd_idx = divmod(
                    UInt(idx[1]) - UInt(qk_offset), kv_params.head_size
                )

        var cache_length = cache.cache_length(batch_idx)
        var cache_token_idx = token_idx + cache_length
        cache.store(
            batch_idx,
            Int(h_idx),
            cache_token_idx,
            Int(hd_idx),
            rebind[SIMD[kv_type, width]](output_val_out.cast[kv_type]()),
        )

    comptime assert (
        weight_dtype == dtype
    ), "Mismatch in dtype between weight and QKV tensors"

    comptime assert is_gpu[
        target
    ](), "Blockwise scaled fp4 matmul only works on GPU."

    _matmul_blockwise_scaled_fp4_common[
        output_dtype = cache_t.dtype,
        target=target,
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        elementwise_lambda_fn=write_to_cache,
    ](
        hidden_state,
        weight,
        input_scale,
        weight_scale,
        tensor_sf,
        context.value(),
    )


@always_inline
fn _matmul_common[
    dtype: DType,
    //,
    *,
    target: StaticString,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    output_dtype: DType = dtype,
](
    hidden_state: LayoutTensor[
        dtype, address_space = AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    context: Optional[DeviceContext],
) raises:
    var TOTAL_SEQ_LEN = hidden_state.dim[0]()
    comptime N = Int(weight.layout.shape[0])
    var c_nd: LayoutTensor[
        output_dtype, Layout.row_major(UNKNOWN_VALUE, N), MutAnyOrigin
    ]

    @parameter
    if is_cpu[target]():
        # The CPU matmul codepath uses the C buffer as a workspace
        # even if an epilogue is provided, here we just allocate
        # something to ensure we don't segfault.
        var c_ptr = alloc[Scalar[output_dtype]](TOTAL_SEQ_LEN * N)

        c_nd = {
            c_ptr,
            RuntimeLayout[c_nd.layout].row_major(
                IndexList[2](TOTAL_SEQ_LEN, N)
            ),
        }
    else:
        c_nd = {
            UnsafePointer[Scalar[output_dtype], MutExternalOrigin](),
            RuntimeLayout[c_nd.layout].row_major(
                IndexList[2](TOTAL_SEQ_LEN, N)
            ),
        }

    matmul[
        target=target,
        transpose_b=True,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ](c_nd, hidden_state, weight, context)

    @parameter
    if is_cpu[target]():
        c_nd.ptr.free()


@always_inline
fn _qmatmul_common[
    dtype: DType,
    //,
    *,
    group_size: Int,
    target: StaticString,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    hidden_state: LayoutTensor[
        dtype, address_space = AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[
        DType.uint8, address_space = AddressSpace.GENERIC, ...
    ],
    context: Optional[DeviceContext],
) raises:
    comptime assert is_gpu[target](), "GPTQ quantization only works on GPU."

    var TOTAL_SEQ_LEN = hidden_state.dim[0]()
    comptime N = Int(weight.layout.shape[0])
    var c_nd: LayoutTensor[
        dtype, Layout.row_major(UNKNOWN_VALUE, N), MutAnyOrigin
    ]

    c_nd = {
        UnsafePointer[Scalar[dtype], MutAnyOrigin](),
        RuntimeLayout[c_nd.layout].row_major(IndexList[2](TOTAL_SEQ_LEN, N)),
    }

    matmul_gpu_qint4_impl[
        target=target,
        group_size=group_size,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ](
        c_nd,
        hidden_state,
        weight,
        context,
    )


@always_inline
fn _matmul_blockwise_scaled_fp8_common[
    output_dtype: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    //,
    *,
    target: StaticString,
    scales_granularity_mnk: IndexList[3],
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    hidden_state: LayoutTensor[
        a_type, address_space = AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[b_type, address_space = AddressSpace.GENERIC, ...],
    input_scale: LayoutTensor[
        a_scales_type, address_space = AddressSpace.GENERIC, ...
    ],
    weight_scale: LayoutTensor[
        b_scales_type, address_space = AddressSpace.GENERIC, ...
    ],
    context: DeviceContext,
) raises:
    comptime assert is_gpu[
        target
    ](), "Blockwise scaled fp8 matmul only works on GPU."

    var TOTAL_SEQ_LEN = hidden_state.dim[0]()
    comptime N = Int(weight.layout.shape[0])
    var c_nd: LayoutTensor[
        output_dtype, Layout.row_major(UNKNOWN_VALUE, N), MutAnyOrigin
    ]

    c_nd = {
        UnsafePointer[Scalar[output_dtype], MutAnyOrigin](),
        RuntimeLayout[c_nd.layout].row_major(IndexList[2](TOTAL_SEQ_LEN, N)),
    }

    blockwise_scaled_fp8_with_epilogue[
        transpose_b=True,
        elementwise_lambda_fn=elementwise_lambda_fn,
        scales_granularity_mnk=scales_granularity_mnk,
    ](c_nd, hidden_state, weight, input_scale, weight_scale, context)


@always_inline
fn _matmul_blockwise_scaled_fp4_common[
    output_dtype: DType,
    a_type: DType,
    b_type: DType,
    scales_dtype: DType,
    # c_layout: Layout,
    a_layout: Layout,
    b_layout: Layout,
    sfa_layout: Layout,
    sfb_layout: Layout,
    //,
    *,
    target: StaticString,
    SF_VECTOR_SIZE: Int = 16,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    hidden_state: LayoutTensor[a_type, a_layout, ImmutAnyOrigin],
    weight: LayoutTensor[b_type, b_layout, ImmutAnyOrigin],
    input_scale: LayoutTensor[scales_dtype, sfa_layout, ImmutAnyOrigin],
    weight_scale: LayoutTensor[scales_dtype, sfb_layout, ImmutAnyOrigin],
    tensor_sf: Float32,
    context: DeviceContext,
) raises:
    comptime assert is_gpu[
        target
    ](), "Blockwise scaled fp4 matmul only works on GPU."

    var TOTAL_SEQ_LEN = hidden_state.dim[0]()
    comptime N = Int(weight.layout.shape[0])
    var c_nd: LayoutTensor[
        output_dtype, Layout.row_major(UNKNOWN_VALUE, N), MutAnyOrigin
    ]

    c_nd = {
        UnsafePointer[Scalar[output_dtype], MutAnyOrigin](),
        RuntimeLayout[c_nd.layout].row_major(IndexList[2](TOTAL_SEQ_LEN, N)),
    }

    blockwise_scaled_fp4_with_epilogue[
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        transpose_b=True,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ](c_nd, hidden_state, weight, input_scale, weight_scale, tensor_sf, context)


# ===-----------------------------------------------------------------------===#
# Unfused KV cache matmul (ragged)
# ===-----------------------------------------------------------------------===#


fn kv_matmul_ragged_paged[
    dtype: DType,
    params: KVCacheStaticParams,
    page_size: Int,
    //,
    target: StaticString,
](
    hidden_state: LayoutTensor[
        dtype, address_space = AddressSpace.GENERIC, ...
    ],
    input_row_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, ...],
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
            Span(
                [
                    trace_arg("weight", weight.runtime_layout.shape.value),
                    "layer_idx=" + String(layer_idx),
                    "num_heads=" + String(kv_collection.kv_params.num_heads),
                    "head_size=" + String(kv_collection.kv_params.head_size),
                ]
            )
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
    hidden_state: LayoutTensor[
        dtype, address_space = AddressSpace.GENERIC, ...
    ],
    input_row_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, ...],
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
    cache_t: KVCacheT,
    //,
    *,
    target: StaticString,
](
    hidden_state: LayoutTensor[
        dtype, address_space = AddressSpace.GENERIC, ...
    ],
    input_row_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, ...],
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
    if hidden_state.size() == 0:
        # Nothing to do.
        return

    comptime kv_params = cache_t.kv_params

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
        comptime kv_type = cache_t.dtype

        comptime assert (
            kv_type == dtype
        ), "Mismatch in dtype between hidden state and KV tensors"

        # Token index in the "ragged" combined sequence dimension.
        global_token_idx = idx[0]

        batch_idx = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )
        token_idx = Int(UInt32(global_token_idx) - input_row_offsets[batch_idx])

        if idx[1] < Int(k_offset):
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
            Int(h_idx),
            cache_token_idx,
            Int(hd_idx),
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
    page_size: Int,
    //,
    target: StaticString,
](
    hidden_state: LayoutTensor[
        dtype, address_space = AddressSpace.GENERIC, ...
    ],
    input_row_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, ...],
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
            Span(
                [
                    trace_arg("weight", weight.runtime_layout.shape.value),
                    "layer_idx=" + String(layer_idx),
                ]
            )
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
    dtype: DType,
    //,
    *,
    target: StaticString,
](
    hidden_state: LayoutTensor[
        dtype, address_space = AddressSpace.GENERIC, ...
    ],
    input_row_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, ...],
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
    cache_t: KVCacheT,
    //,
    *,
    target: StaticString,
](
    hidden_state: LayoutTensor[
        dtype, address_space = AddressSpace.GENERIC, ...
    ],
    input_row_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, ...],
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
    if hidden_state.size() == 0:
        # Nothing to do.
        return

    comptime kv_params = cache_t.kv_params

    batch_size = input_row_offsets.dim[0]() - 1

    @parameter
    @__copy_capture(batch_size)
    @always_inline
    fn write_to_cache[
        dtype: DType, width: Int, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width],):
        comptime kv_type = cache_t.dtype

        comptime assert (
            kv_type == dtype
        ), "Mismatch in dtype between hidden state and KV tensors"

        # Token index in the "ragged" combined sequence dimension.
        global_token_idx = idx[0]

        batch_idx = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )
        token_idx = Int(UInt32(global_token_idx) - input_row_offsets[batch_idx])

        h_idx, hd_idx = divmod(UInt(idx[1]), kv_params.head_size)

        cache_length = k_cache.cache_length(batch_idx)
        cache_token_idx = token_idx + cache_length
        k_cache.store(
            batch_idx,
            Int(h_idx),
            cache_token_idx,
            Int(hd_idx),
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
    hidden_state: LayoutTensor[
        dtype, address_space = AddressSpace.GENERIC, ...
    ],
    input_row_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[
        weight_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    input_scale: LayoutTensor[
        scale_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    weight_scale: LayoutTensor[
        scale_dtype, address_space = AddressSpace.GENERIC, ...
    ],
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
            Span(
                [
                    trace_arg(
                        "hidden_state", hidden_state.runtime_layout.shape.value
                    ),
                    trace_arg("weight", weight.runtime_layout.shape.value),
                    trace_arg(
                        "input_scale", input_scale.runtime_layout.shape.value
                    ),
                    trace_arg(
                        "weight_scale", weight_scale.runtime_layout.shape.value
                    ),
                    "layer_idx=" + String(layer_idx),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=target](
        "mo.k_matmul.ragged.paged.scale.nhead_"
        + String(kv_collection.kv_params.num_heads)
        + ".hdim_"
        + String(kv_collection.kv_params.head_size),
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=Int(ctx.get_device_context().id()),
    ):
        comptime assert is_gpu[
            target
        ](), "Blockwise scaled fp8 matmul only works on GPU."
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
    scale_dtype: DType,
    //,
    cache_t: KVCacheT,
    *,
    target: StaticString,
    scales_granularity_mnk: IndexList[3],
](
    hidden_state: LayoutTensor[
        dtype, address_space = AddressSpace.GENERIC, ...
    ],
    input_row_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[
        weight_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    input_scale: LayoutTensor[
        scale_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    weight_scale: LayoutTensor[
        scale_dtype, address_space = AddressSpace.GENERIC, ...
    ],
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
    if hidden_state.size() == 0:
        # Nothing to do.
        return

    comptime kv_params = cache_t.kv_params

    var batch_size = input_row_offsets.dim[0]() - 1

    @parameter
    @__copy_capture(input_scale, weight_scale, batch_size)
    @always_inline
    fn write_to_cache[
        dtype: DType, width: Int, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width],):
        comptime kv_type = cache_t.dtype

        comptime assert (
            kv_type == dtype
        ), "Mismatch in dtype between hidden state and KV tensors"

        # Token index in the "ragged" combined sequence dimension.
        var global_token_idx = idx[0]

        var batch_idx = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )
        var token_idx = Int(
            UInt32(global_token_idx) - input_row_offsets[batch_idx]
        )

        var h_idx, hd_idx = divmod(UInt(idx[1]), kv_params.head_size)

        var cache_length = k_cache.cache_length(batch_idx)
        var cache_token_idx = token_idx + cache_length
        k_cache.store(
            batch_idx,
            Int(h_idx),
            cache_token_idx,
            Int(hd_idx),
            rebind[SIMD[kv_type, width]](val),
        )

    comptime assert (
        weight_dtype == dtype
    ), "Mismatch in dtype between weight and QKV tensors"
    _matmul_blockwise_scaled_fp8_common[
        output_dtype = cache_t.dtype,
        target=target,
        elementwise_lambda_fn=write_to_cache,
        scales_granularity_mnk=scales_granularity_mnk,
    ](hidden_state, weight.bitcast[dtype](), input_scale, weight_scale, ctx)


# ===-----------------------------------------------------------------------===#
# Unfused gguf quantized QKV cache matmul (ragged)
# ===-----------------------------------------------------------------------===#


fn unfused_qkv_matmul_ragged_paged_gguf_quantized[
    dtype: DType,
    params: KVCacheStaticParams,
    page_size: Int,
    //,
    quantization_encoding_q: StaticString,
    quantization_encoding_k: StaticString,
    quantization_encoding_v: StaticString,
](
    hidden_state: LayoutTensor[
        DType.float32, address_space = AddressSpace.GENERIC, ...
    ],
    input_row_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    q_weight: LayoutTensor[
        DType.uint8, address_space = AddressSpace.GENERIC, ...
    ],
    k_weight: LayoutTensor[
        DType.uint8, address_space = AddressSpace.GENERIC, ...
    ],
    v_weight: LayoutTensor[
        DType.uint8, address_space = AddressSpace.GENERIC, ...
    ],
    kv_collection: PagedKVCacheCollection[
        dtype,
        params,
        page_size,
    ],
    layer_idx: UInt32,
    output: LayoutTensor[
        mut=True, DType.float32, address_space = AddressSpace.GENERIC, ...
    ],
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
            Span(
                [
                    trace_arg("q_weight", q_weight.runtime_layout.shape.value),
                    trace_arg("k_weight", k_weight.runtime_layout.shape.value),
                    trace_arg("v_weight", v_weight.runtime_layout.shape.value),
                    "layer_idx=" + String(layer_idx),
                    "num_heads=" + String(kv_collection.kv_params.num_heads),
                    "head_size=" + String(kv_collection.kv_params.head_size),
                    "quantization_encoding_q=" + quantization_encoding_q,
                    "quantization_encoding_k=" + quantization_encoding_k,
                    "quantization_encoding_v=" + quantization_encoding_v,
                ]
            )
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
    hidden_state: LayoutTensor[
        DType.float32, address_space = AddressSpace.GENERIC, ...
    ],
    input_row_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    q_weight: LayoutTensor[
        DType.uint8, address_space = AddressSpace.GENERIC, ...
    ],
    k_weight: LayoutTensor[
        DType.uint8, address_space = AddressSpace.GENERIC, ...
    ],
    v_weight: LayoutTensor[
        DType.uint8, address_space = AddressSpace.GENERIC, ...
    ],
    kv_collection: PagedKVCacheCollection,
    layer_idx: UInt32,
    output: LayoutTensor[
        mut=True, DType.float32, address_space = AddressSpace.GENERIC, ...
    ],
    context: DeviceContextPtr,
) raises:
    layer_idx_cast = Int(layer_idx)
    k_cache = kv_collection.get_key_cache(layer_idx_cast)
    v_cache = kv_collection.get_value_cache(layer_idx_cast)

    comptime cache_t = PagedKVCache[
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
    hidden_state: LayoutTensor[
        DType.float32, address_space = AddressSpace.GENERIC, ...
    ],
    input_row_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    q_weight: LayoutTensor[
        DType.uint8, address_space = AddressSpace.GENERIC, ...
    ],
    k_weight: LayoutTensor[
        DType.uint8, address_space = AddressSpace.GENERIC, ...
    ],
    v_weight: LayoutTensor[
        DType.uint8, address_space = AddressSpace.GENERIC, ...
    ],
    k_cache: cache_t,
    v_cache: cache_t,
    output: LayoutTensor[
        mut=True, DType.float32, address_space = AddressSpace.GENERIC, ...
    ],
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
    if hidden_state.size() == 0:
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
    hidden_state: LayoutTensor[
        DType.float32, address_space = AddressSpace.GENERIC, ...
    ],
    input_row_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    k_or_v_weight: LayoutTensor[
        DType.uint8, address_space = AddressSpace.GENERIC, ...
    ],
    k_or_v_cache: cache_t,
) raises:
    comptime kv_params = cache_t.kv_params

    batch_size = input_row_offsets.dim[0]() - 1

    @parameter
    @__copy_capture(input_row_offsets, batch_size)
    @always_inline
    fn write_to_cache_common[
        dtype: DType, cache_t: KVCacheT, width: Int
    ](k_or_v_cache: cache_t, idx: IndexList[2], val: SIMD[dtype, width],):
        comptime k_or_v_type = cache_t.dtype

        comptime assert (
            k_or_v_type == dtype
        ), "Mismatch in dtype between hidden state and KV tensors"

        # Token index in the "ragged" combined sequence dimension.
        global_token_idx = idx[0]

        batch_idx = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )
        token_idx = Int(UInt32(global_token_idx) - input_row_offsets[batch_idx])

        # Write this element to the K or V cache.
        cache = k_or_v_cache
        h_idx, hd_idx = divmod(UInt(idx[1]), kv_params.head_size)

        cache_length = cache.cache_length(batch_idx)
        cache_token_idx = token_idx + cache_length

        cache.store(
            batch_idx,
            Int(h_idx),
            cache_token_idx,
            Int(hd_idx),
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
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    hidden_state: LayoutTensor[
        DType.float32, address_space = AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[
        DType.uint8, address_space = AddressSpace.GENERIC, ...
    ],
) raises:
    var TOTAL_SEQ_LEN = hidden_state.dim[0]()
    comptime N = Int(weight.layout.shape[0])
    var c_nd: LayoutTensor[
        DType.float32, Layout.row_major(UNKNOWN_VALUE, N), MutAnyOrigin
    ]

    # The CPU matmul codepath uses the C buffer as a workspace
    # even if an epilogue is provided, here we just allocate
    # something to ensure we don't segfault.
    var c_ptr = alloc[Float32](TOTAL_SEQ_LEN * N)

    c_nd = {
        c_ptr,
        RuntimeLayout[c_nd.layout].row_major(IndexList[2](TOTAL_SEQ_LEN, N)),
    }

    _qmatmul_gguf_quantized_common[
        quantization_encoding, elementwise_lambda_fn
    ](hidden_state, weight, c_nd)

    c_nd.ptr.free()


@always_inline
fn _qmatmul_gguf_quantized_common[
    quantization_encoding: StaticString,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    hidden_state: LayoutTensor[
        DType.float32, address_space = AddressSpace.GENERIC, ...
    ],
    weight: LayoutTensor[
        DType.uint8, address_space = AddressSpace.GENERIC, ...
    ],
    output: LayoutTensor[
        mut=True, DType.float32, address_space = AddressSpace.GENERIC, ...
    ],
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
fn generic_fused_qk_rope_bshd_paged_ragged[
    dtype: DType,
    freq_dtype: DType,
    //,
    *,
    interleaved: Bool,
    has_position_ids: Bool,
    target: StaticString,
    mrope_types: Variadic.TypesOfTrait[CoordLike] = Variadic.empty_of_trait[
        CoordLike
    ],
    mrope_section: Optional[Coord[*mrope_types]] = None,
](
    q_proj: TileTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    input_row_offsets: TileTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    kv_collection: PagedKVCacheCollection,
    freqs_cis: TileTensor[
        freq_dtype, address_space = AddressSpace.GENERIC, ...
    ],
    position_ids: TileTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    layer_idx: UInt32,
    output: TileTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
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

    comptime name = "mo.fused_qk_rope.ragged.paged.nhead_" + String(
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
                PositionIdsLayoutType = position_ids.LayoutType,
                mrope_types=mrope_types,
                mrope_section=mrope_section,
            ](
                q_proj,
                input_row_offsets,
                kv_collection,
                freqs_cis,
                TileTensor(
                    position_ids.ptr.mut_cast[True]().as_any_origin(),
                    position_ids.layout,
                ).as_immut(),
                layer_idx,
                output,
                dev_ctx,
            )
        else:
            fused_qk_rope_ragged[
                kv_collection.CacheType, interleaved=interleaved, target=target
            ](
                q_proj,
                input_row_offsets,
                kv_collection,
                freqs_cis,
                None,
                layer_idx,
                output,
                dev_ctx,
            )


# ===-----------------------------------------------------------------------===#
# MHA (ragged)
# ===-----------------------------------------------------------------------===#


@always_inline
fn generic_flash_attention_kv_cache_ragged[
    collection_t: KVCollectionT,
    dtype: DType,
    //,
    *,
    target: StaticString,
    mask_str: StaticString,
    score_mod_str: StaticString,
    local_window_size: Int = -1,
](
    q: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    input_row_offsets: LayoutTensor[
        DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    output: LayoutTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    context: DeviceContextPtr,
) raises:
    @always_inline
    @parameter
    fn description_fn() -> String:
        var desc_parts = List[String]()
        desc_parts.append(trace_arg("q", q.runtime_layout.shape.value))
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

    comptime name = "mo.mha.ragged." + collection_t.name_str + "." + mask_str + "." + score_mod_str + ".nhead_" + String(
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
    input_row_offsets: LayoutTensor[
        mut=False, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    kv_cache: collection_t,
    layer_idx: UInt32,
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
            @parameter
            if is_cpu[target]():
                return flash_attention_kv_cache_cpu(
                    q,
                    input_row_offsets,
                    input_row_offsets,
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
                gpu_flash_attention[
                    use_score_mod=use_score_mod, ragged=True, sink=sink
                ](
                    output,
                    q,
                    k,
                    v,
                    mask,
                    score_mod,
                    input_row_offsets,
                    scale,
                    context.get_device_context(),
                    sink_weights=sink_weights,
                )

        unswitch[call_flash_attention](Bool(sink_weights))

    return dispatch_mask_and_score_mod[
        mask_str,
        score_mod_str,
        _dispatch_flash_attention,
        local_window_size,
        Int(collection_t.kv_params.num_heads),
    ]()


@always_inline
fn generic_flash_attention_kv_cache_ragged_sink[
    collection_t: KVCollectionT,
    dtype: DType,
    //,
    *,
    target: StaticString,
    mask_str: StaticString,
    score_mod_str: StaticString,
    local_window_size: Int = -1,
](
    q: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    input_row_offsets: LayoutTensor[
        DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    output: LayoutTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    context: DeviceContextPtr,
    sink_weights: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
) raises:
    @always_inline
    @parameter
    fn description_fn() -> String:
        var desc_parts = List[String]()
        desc_parts.append(trace_arg("q", q.runtime_layout.shape.value))
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

    comptime name = "mo.mha.ragged." + collection_t.name_str + "." + mask_str + "." + score_mod_str + ".nhead_" + String(
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
            LayoutTensor[
                dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
            ](
                sink_weights.ptr,
                RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
                    sink_weights.runtime_layout.shape.value.canonicalize(),
                ),
            ),
        )


# ===-----------------------------------------------------------------------===#
# MLA (ragged)
# ===-----------------------------------------------------------------------===#


@always_inline
fn generic_flare_mla_decode_kv_cache_ragged[
    collection_t: KVCollectionT,
    dtype: DType,
    //,
    mask_str: StaticString,
    score_mod_str: StaticString,
    target: StaticString,
    local_window_size: Int = -1,
](
    q: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    input_row_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    output: LayoutTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    context: DeviceContextPtr,
) raises:
    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("q", q.runtime_layout.shape.value),
                    "scale=" + String(scale),
                    "layer_idx=" + String(layer_idx),
                    "num_heads=" + String(collection_t.kv_params.num_heads),
                    "head_size=" + String(collection_t.kv_params.head_size),
                ]
            )
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
    collection_t: KVCollectionT,
    //,
    mask_str: StaticString,
    score_mod_str: StaticString,
    target: StaticString,
    local_window_size: Int = -1,
](
    q: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    input_row_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    output: LayoutTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
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
    comptime assert is_gpu[target](), "MLA is only supported on GPU"

    var layer_idx_cast = Int(layer_idx)
    var k = kv_collection.get_key_cache(layer_idx_cast)

    @parameter
    @always_inline
    @__copy_capture(k)
    fn _dispatch_mla[
        mask_t: MHAMask, score_mod_t: ScoreModTrait
    ](mask: mask_t, score_mod: score_mod_t) raises:
        flare_mla_decoding[rank = q.rank, ragged=True](
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
        Int(collection_t.kv_params.num_heads),
    ]()


@always_inline
fn generic_flare_mla_prefill_kv_cache_ragged[
    collection_t: KVCollectionT,
    dtype: DType,
    //,
    mask_str: StaticString,
    score_mod_str: StaticString,
    target: StaticString,
    local_window_size: Int = -1,
](
    q: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    k: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    v: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    buffer_row_offsets: LayoutTensor[
        mut=False, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    cache_offsets: LayoutTensor[
        mut=True, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    input_row_offsets: LayoutTensor[
        mut=False, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    output: LayoutTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    context: DeviceContextPtr,
) raises:
    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("q", q.runtime_layout.shape.value),
                    trace_arg("k", k.runtime_layout.shape.value),
                    trace_arg("v", v.runtime_layout.shape.value),
                    trace_arg(
                        "buffer_row_offsets",
                        buffer_row_offsets.runtime_layout.shape.value,
                    ),
                    trace_arg(
                        "cache_offsets",
                        cache_offsets.runtime_layout.shape.value,
                    ),
                    trace_arg(
                        "input_row_offsets",
                        input_row_offsets.runtime_layout.shape.value,
                    ),
                    "scale=" + String(scale),
                    "layer_idx=" + String(layer_idx),
                    "num_heads=" + String(collection_t.kv_params.num_heads),
                    "head_size=" + String(collection_t.kv_params.head_size),
                ]
            )
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
            context,
        )


@always_inline
fn _flare_mla_prefill_kv_cache_ragged[
    dtype: DType,
    collection_t: KVCollectionT,
    //,
    mask_str: StaticString,
    score_mod_str: StaticString,
    target: StaticString,
    local_window_size: Int = -1,
](
    q: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    k: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    v: LayoutTensor[
        mut=False, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    buffer_row_offsets: LayoutTensor[
        mut=False, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    cache_offsets: LayoutTensor[
        mut=True, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    input_row_offsets: LayoutTensor[
        mut=False, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    kv_collection: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    output: LayoutTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    context: DeviceContextPtr,
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
        context: Pointer containing the runtime context for the target device.
    """
    comptime assert is_gpu[target](), "MLA is only supported on GPU"

    var layer_idx_cast = Int(layer_idx)
    var k_rope = kv_collection.get_key_cache(layer_idx_cast)

    @parameter
    @__copy_capture(k_rope)
    fn _mla_dispatch[
        mask_t: MHAMask, score_mod_t: ScoreModTrait
    ](mask: mask_t, score_mod: score_mod_t) raises:
        flare_mla_prefill[rank=3,](
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
            cache_offsets=LayoutTensor[
                cache_offsets.dtype,
                Layout.row_major(UNKNOWN_VALUE),
                MutAnyOrigin,
            ](
                cache_offsets.ptr,
                RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
                    cache_offsets.runtime_layout.shape.value.canonicalize()
                ),
            ),
        )

    dispatch_mask_and_score_mod[
        mask_str,
        score_mod_str,
        _mla_dispatch,
        local_window_size,
        Int(collection_t.kv_params.num_heads),
    ]()


@always_inline
fn generic_flare_mla_prefill_ragged_paged_plan[
    target: StaticString
](
    input_row_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    kv_collection: PagedKVCacheCollection,
    layer_idx: UInt32,
    buffer_token_size: UInt32,
    buffer_row_offsets: LayoutTensor[
        mut=True, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    cache_offsets: LayoutTensor[
        mut=True, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    buffer_lengths: LayoutTensor[
        mut=True, DType.int32, address_space = AddressSpace.GENERIC, ...
    ],
    context: DeviceContextPtr,
) raises:
    comptime assert is_gpu[target](), "Planning MLA is only supported on GPU"

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
    buffer_row_offsets_1d: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    cache_offsets_1d: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    buffer_length: Int32,
    weight: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    kv_collection: PagedKVCacheCollection,
    layer_idx: UInt32,
    k_latent_buffer: LayoutTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    k_buffer: LayoutTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    context: DeviceContextPtr,
) raises:
    comptime assert is_gpu[target](), "MLA is only supported on GPU"
    var cuda_ctx = context.get_device_context()

    var buffer_length_int = Int(buffer_length)
    var layer_idx_cast = Int(layer_idx)
    var k = kv_collection.get_key_cache(layer_idx_cast)

    _k_cache_to_buffer(
        buffer_row_offsets_1d,
        cache_offsets_1d,
        k,
        Int32(buffer_length_int),
        k_latent_buffer,
        cuda_ctx,
    )

    # rebind k_latent_buffer with dynamic dim
    comptime latent_last_dim = Int(k_latent_buffer.layout.shape[1])
    comptime k_latent_layout = Layout.row_major(UNKNOWN_VALUE, latent_last_dim)
    var k_latent_dynamic_shape = IndexList[2](
        buffer_length_int, latent_last_dim
    )

    var k_latent_buffer_dynamic = LayoutTensor[dtype, k_latent_layout](
        k_latent_buffer.ptr,
        RuntimeLayout[k_latent_layout].row_major(k_latent_dynamic_shape),
    )

    # rebind k_buffer with dynamic dim
    comptime k_last_dim = Int(k_buffer.layout.shape[1])
    comptime k_layout = Layout.row_major(UNKNOWN_VALUE, k_last_dim)
    var k_dynamic_shape = IndexList[2](buffer_length_int, k_last_dim)

    var k_buffer_dynamic = LayoutTensor[dtype, k_layout](
        k_buffer.ptr, RuntimeLayout[k_layout].row_major(k_dynamic_shape)
    )

    matmul[
        target=target,
        transpose_b=True,
    ](k_buffer_dynamic, k_latent_buffer_dynamic, weight, Optional(cuda_ctx))


# ===-----------------------------------------------------------------------===#
# Cross attention (ragged)
# ===-----------------------------------------------------------------------===#


fn _cross_attention_dispatch[
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
    q_input_row_offsets: LayoutTensor[
        mut=False, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    q_max_seq_len: UInt32,
    kv_input_row_offsets: LayoutTensor[
        mut=False, DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    kv_cache: collection_t,
    layer_idx: UInt32,
    scale: Float32,
    output: LayoutTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    context: DeviceContextPtr,
    sink_weights: OptionalReg[
        LayoutTensor[
            mut=False, dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
        ]
    ] = None,
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
            return flash_attention_kv_cache_cpu(
                q,
                q_input_row_offsets,
                # Use KV offsets for cross attention.
                kv_input_row_offsets,
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
            gpu_flash_attention[
                use_score_mod=use_score_mod, ragged=True, sink=False
            ](
                output,
                q,
                k,
                v,
                mask,
                IdentityScoreMod(),
                q_input_row_offsets,
                scale,
                context.get_device_context(),
                Int(q_max_seq_len),
                LayoutTensor[
                    kv_input_row_offsets.dtype,
                    Layout.row_major(UNKNOWN_VALUE),
                    ImmutAnyOrigin,
                ](
                    kv_input_row_offsets.ptr,
                    RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
                        kv_input_row_offsets.runtime_layout.shape.value.canonicalize()
                    ),
                ),
                None,
            )

    return dispatch_mask_and_score_mod[
        mask_str,
        score_mod_str,
        _dispatch_flash_attention,
        local_window_size,
        Int(collection_t.kv_params.num_heads),
    ]()


@always_inline
fn generic_cross_attention_kv_cache[
    collection_t: KVCollectionT,
    dtype: DType,
    //,
    target: StaticString,
    mask_str: StaticString,
    score_mod_str: StaticString,
    local_window_size: Int = -1,
](
    q: LayoutTensor[mut=True, dtype, address_space = AddressSpace.GENERIC, ...],
    q_input_row_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    q_max_seq_len: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    kv_input_row_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    kv_collection: collection_t,
    layer_idx: UInt32,
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
                    trace_arg("output", output.runtime_layout.shape.value),
                    trace_arg("q", q.runtime_layout.shape.value),
                    trace_arg(
                        "q_input_row_offsets",
                        q_input_row_offsets.runtime_layout.shape.value,
                    ),
                    trace_arg(
                        "kv_input_row_offsets",
                        kv_input_row_offsets.runtime_layout.shape.value,
                    ),
                    "layer_idx=" + String(layer_idx),
                    "num_heads=" + String(collection_t.kv_params.num_heads),
                    "head_size=" + String(collection_t.kv_params.head_size),
                ]
            )
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
            q_max_seq_len[0][0],
            kv_input_row_offsets,
            kv_collection,
            layer_idx,
            scale,
            output,
            context,
            sink_weights,
        )


# ===-----------------------------------------------------------------------===#
# KV cache ragged radd dispatch
# ===-----------------------------------------------------------------------===#


fn generic_kv_cache_radd_dispatch[
    dtype: DType,
    collection_t: KVCollectionT,
    //,
    target: StaticString,
](
    a: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    cache: collection_t,
    input_row_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    batch_offset: UInt32,
    layer_idx: UInt32,
    ctx: Optional[DeviceContext],
) raises:
    comptime hidden_size = collection_t.kv_params.head_size * collection_t.kv_params.num_heads

    comptime assert (
        dtype == collection_t.dtype
    ), "Mismatch in dtype between computation and KV tensors"
    comptime assert (
        a.layout.shape[1] != UNKNOWN_VALUE
    ), "Input tensor must have known shape in last dim"
    comptime assert Int(a.layout.shape[1]) == Int(hidden_size * 2), (
        "Mismatch in hidden size between input "
        + String(Int(a.layout.shape[1]))
        + " and KV tensors "
        + String(hidden_size)
    )

    var layer_idx_cast = Int(layer_idx)
    var k_cache = cache.get_key_cache(layer_idx_cast)
    var v_cache = cache.get_value_cache(layer_idx_cast)

    @parameter
    @__copy_capture(k_cache, v_cache, input_row_offsets)
    fn do_radd[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        comptime assert rank == 2, "Rank must be 2"

        # we could be slicing the batch, so we need to add the offset to get the actual index in the flattened batch
        var corrected_token_idx = UInt32(idx[0]) + input_row_offsets[0]
        var batch_idx = get_batch_from_row_offsets(
            input_row_offsets, Int(corrected_token_idx)
        )

        # we also need to add the batch offset to get the actual index in the flattened batch
        var corrected_batch_idx = UInt32(batch_idx) + batch_offset
        var tok_idx = Int(corrected_token_idx - input_row_offsets[batch_idx])

        var cache: collection_t.CacheType
        var corrected_dim: UInt
        if idx[1] < Int(hidden_size):
            cache = k_cache
            corrected_dim = UInt(idx[1])
        else:
            cache = v_cache
            corrected_dim = UInt(idx[1] - Int(hidden_size))

        var h_idx: UInt
        var hd_idx: UInt
        h_idx, hd_idx = divmod(corrected_dim, collection_t.kv_params.head_size)

        var cache_length = cache.cache_length(Int(corrected_batch_idx))
        var cache_token_idx = tok_idx + cache_length

        var old_val = cache.load[width=width](
            Int(corrected_batch_idx), Int(h_idx), cache_token_idx, Int(hd_idx)
        )
        var a_val = rebind[type_of(old_val)](a.load[width=width](idx))

        cache.store(
            Int(corrected_batch_idx),
            Int(h_idx),
            cache_token_idx,
            Int(hd_idx),
            a_val + old_val,
        )

    @parameter
    if is_gpu[target]():
        if ctx is None:
            raise Error("ctx is None")
        comptime compile_target = get_gpu_target()
        comptime simd_width = simd_width_of[dtype, target=compile_target]()

        elementwise[do_radd, simd_width, target=target](
            a.runtime_layout.shape.value.canonicalize(), ctx.value()
        )
    else:
        comptime compile_target = _current_target()
        comptime simd_width = simd_width_of[dtype, target=compile_target]()

        elementwise[do_radd, simd_width, target=target](
            a.runtime_layout.shape.value.canonicalize()
        )


fn kv_cache_store_ragged[
    cache_t: KVCacheT,
    //,
    target: StaticString,
    input_fn: fn[width: Int, alignment: Int](
        idx: IndexList[3]
    ) capturing -> SIMD[cache_t.dtype, width],
](
    cache: cache_t,
    input_shape: IndexList[3],
    input_row_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    context: Optional[DeviceContext],
) raises:
    comptime assert input_row_offsets.layout.rank() == 1, (
        "Expected input_row_offsets to be a 1D tensor of shape `(batch_size"
        " + 1,)`"
    )

    @parameter
    @__copy_capture(cache, input_row_offsets)
    fn write_to_cache[
        width: Int,
        rank: Int,
        alignment: Int = 1,
    ](idx: IndexList[rank]) capturing:
        var loaded_val = input_fn[width=width, alignment=alignment](
            rebind[IndexList[3]](idx)
        )
        var batch_idx = get_batch_from_row_offsets(input_row_offsets, idx[0])
        var token_idx = Int(UInt32(idx[0]) - input_row_offsets[batch_idx])
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
        if context is None:
            raise Error("ctx is None")
        comptime compile_target = get_gpu_target()
        comptime simd_width = simd_width_of[
            cache_t.dtype, target=compile_target
        ]()

        elementwise[write_to_cache, simd_width, target=target](
            input_shape, context.value()
        )
    else:
        comptime compile_target = _current_target()
        comptime simd_width = simd_width_of[
            cache_t.dtype, target=compile_target
        ]()

        elementwise[write_to_cache, simd_width, target=target](input_shape)


fn kv_cache_store_padded[
    cache_t: KVCacheT,
    //,
    target: StaticString,
    input_fn: fn[width: Int, alignment: Int](
        idx: IndexList[4]
    ) capturing -> SIMD[cache_t.dtype, width],
](
    cache: cache_t,
    input_shape: IndexList[4],
    valid_lengths: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    context: Optional[DeviceContext],
) raises:
    comptime assert (
        valid_lengths.layout.rank() == 1
    ), "Expected valid_lengths to be a 1D tensor of shape `(batch_size,)`"

    @parameter
    @__copy_capture(cache, valid_lengths)
    @always_inline
    fn write_to_cache[
        width: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank]) capturing:
        var batch_idx = idx[0]
        var token_idx = idx[1]
        var valid_len = Int(valid_lengths[batch_idx])
        if token_idx >= valid_len:
            return
        var loaded_val = input_fn[width=width, alignment=alignment](
            rebind[IndexList[4]](idx)
        )
        var h_idx = idx[2]
        var hd_idx = idx[3]
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
        if context is None:
            raise Error("ctx is None")
        comptime compile_target = get_gpu_target()
        comptime simd_width = simd_width_of[
            cache_t.dtype, target=compile_target
        ]()

        elementwise[write_to_cache, simd_width, target=target](
            input_shape, context.value()
        )
    else:
        comptime compile_target = _current_target()
        comptime simd_width = simd_width_of[
            cache_t.dtype, target=compile_target
        ]()

        elementwise[write_to_cache, simd_width, target=target](input_shape)


# ===-----------------------------------------------------------------------===#
# KV cache ragged 2M iadd dispatch
# ===-----------------------------------------------------------------------===#


fn kv_cache_2m_iadd_dispatch[
    dtype: DType,
    collection_t: KVCollectionT,
    //,
    target: StaticString,
](
    kv: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    cache: collection_t,
    input_row_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    lora_end_idx: LayoutTensor[
        DType.int64, address_space = AddressSpace.GENERIC, ...
    ],
    batch_seq_len: LayoutTensor[
        DType.int64, address_space = AddressSpace.GENERIC, ...
    ],
    layer_idx: UInt32,
    ctx: Optional[DeviceContext],
) raises:
    """
    In-place add to paged KV cache with concatenated K/V layout. This kernel is
    only used for LoRA.

    Performs an in-place addition of new key-value projections to paged KV cache.
    The input tensor `a` uses a "2m" layout where keys and values are concatenated:
    rows [0, m) contain keys and rows [m, 2m) contain values, where m is the number
    of tokens. We use the `lora_end_idx` to index into the K or V tensor.
    We call this value `m` since this value will be a subset of the
    total tokens in the batch. We write tokens to K as [0, m) and V as [m, 2m).
    """
    comptime hidden_size = collection_t.kv_params.head_size * collection_t.kv_params.num_heads
    var kv_shape = kv.runtime_layout.shape.value.canonicalize()
    comptime assert (
        dtype == collection_t.dtype
    ), "Mismatch in dtype between computation and KV tensors"
    comptime assert (
        kv.layout.shape[1] != UNKNOWN_VALUE
    ), "Input tensor must have known shape in last dim"
    comptime assert Int(kv.layout.shape[1]) == Int(hidden_size), (
        "Mismatch in hidden size between input "
        + String(Int(kv.layout.shape[1]))
        + " and KV tensors "
        + String(hidden_size)
    )

    var layer_idx_cast = Int(layer_idx)
    var k_cache = cache.get_key_cache(layer_idx_cast)
    var v_cache = cache.get_value_cache(layer_idx_cast)
    var m = Int(lora_end_idx[0])
    var M = Int(batch_seq_len[0])

    # [2m, N]
    var elementwise_shape = IndexList[2](2 * m, kv_shape[1])

    @parameter
    @__copy_capture(kv, k_cache, v_cache, input_row_offsets, m, M)
    fn iadd[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        comptime assert rank == 2, "Rank must be 2"

        var cache: collection_t.CacheType
        var row_idx: Int

        if idx[0] < m:
            cache = k_cache
            row_idx = idx[0]
        else:
            cache = v_cache
            row_idx = idx[0] - m

        var batch_idx = get_batch_from_row_offsets(input_row_offsets, row_idx)
        var tok_idx = Int(UInt32(row_idx) - input_row_offsets[batch_idx])

        var h_idx: UInt
        var hd_idx: UInt
        h_idx, hd_idx = divmod(UInt(idx[1]), collection_t.kv_params.head_size)

        var cache_length = cache.cache_length(batch_idx)
        var cache_token_idx = tok_idx + cache_length

        var old_val = cache.load[width=width](
            batch_idx, Int(h_idx), cache_token_idx, Int(hd_idx)
        )
        var a_val = rebind[type_of(old_val)](
            kv.load[width=width](idx[0], idx[1])
        )

        cache.store(
            batch_idx,
            Int(h_idx),
            cache_token_idx,
            Int(hd_idx),
            a_val + old_val,
        )

    @parameter
    if is_gpu[target]():
        if ctx is None:
            raise Error("ctx is None")
        with Trace[TraceLevel.OP, target=target](
            "kv-cache-2m-iadd",
            task_id=Int(ctx.value().id()),
        ):
            comptime compile_target = get_gpu_target()
            comptime simd_width = simd_width_of[dtype, target=compile_target]()

            elementwise[iadd, simd_width, target=target](
                elementwise_shape, ctx.value()
            )
    else:
        comptime compile_target = _current_target()
        comptime simd_width = simd_width_of[dtype, target=compile_target]()

        elementwise[iadd, simd_width, target=target](elementwise_shape)
