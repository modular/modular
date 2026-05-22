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

from std.memory import UnsafePointer

from kv_cache.types import KVCacheT
from nn.attention.mha_operand import MHAOperand
from nn.attention.mha_utils import MHAConfig, OptionallyStaticInt
from nn.attention.mha_mask import MHAMask
from std.gpu.host import DeviceContext
from layout import TileTensor
from std.gpu.memory import AddressSpace
from .mla_prefill_generic import mla_sm100_prefill_generic
from .mla_prefill_blockscale import mla_sm100_prefill_blockscale
from .mla_prefill_sparse import MLASparseConfig, mla_prefill_sparse


@always_inline
def mla_sm100_prefill[
    output_type: DType,
    q_type: DType,
    KVType: MHAOperand,
    KRopeType: MHAOperand,
    MaskType: MHAMask,
    MaxPromptLenType: OptionallyStaticInt,
    //,
    config: MHAConfig,
    group: Int,
    q_depth: Int,
    cache_depth: Int,
    _ndbuffer_mha_operand: Bool,
    blockwise_scale: Int = 0,
](
    output: TileTensor[output_type, address_space=AddressSpace.GENERIC, ...],
    q: TileTensor[q_type, address_space=AddressSpace.GENERIC, ...],
    k: KVType,
    v: KVType,
    k_rope: KRopeType,
    mask_functor: MaskType,
    valid_length: TileTensor[
        DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    max_prompt_len: MaxPromptLenType,
    scale: Float32,
    batch_size: Int,
    ctx: DeviceContext,
) raises:
    comptime assert (
        output_type == DType.bfloat16
    ), "Only support bfloat16 output for SM100 MLA prefill"

    comptime if blockwise_scale == 0 and (
        KRopeType.dtype == KVType.dtype == q.dtype
    ):
        comptime assert (
            blockwise_scale == 0
        ), "blockwise_scale is not supported for generic MLA prefill"
        mla_sm100_prefill_generic[
            config=config,
            group=Int(group),
            q_depth=q_depth,
            cache_depth=cache_depth,
            _ndbuffer_mha_operand=_ndbuffer_mha_operand,
        ](
            output,
            q,
            k,
            rebind[type_of(k)](v),
            k_rope,
            mask_functor,
            valid_length,
            max_prompt_len,
            scale,
            batch_size,
            ctx,
        )
    else:
        mla_sm100_prefill_blockscale[
            config=config,
            group=Int(group),
            q_depth=q_depth,
            cache_depth=cache_depth,
            _ndbuffer_mha_operand=_ndbuffer_mha_operand,
            blockwise_scale=blockwise_scale,
        ](
            output,
            q,
            k,
            rebind[type_of(k)](v),
            k_rope,
            mask_functor,
            valid_length,
            max_prompt_len,
            scale,
            batch_size,
            ctx,
        )


@always_inline
def mla_sm100_prefill_sparse[
    output_type: DType,
    q_type: DType,
    cache_t: KVCacheT,
    //,
    num_q_heads: Int,
    qk_depth: Int,
    v_depth: Int,
    indices_stride: Int,
](
    output: TileTensor[output_type, address_space=AddressSpace.GENERIC, ...],
    q: TileTensor[q_type, address_space=AddressSpace.GENERIC, ...],
    kv_cache: cache_t,
    indices: TileTensor[DType.uint32, address_space=AddressSpace.GENERIC, ...],
    topk_lengths: TileTensor[
        DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    attn_sink_ptr: UnsafePointer[Float32, ImmutAnyOrigin],
    scale: Float32,
    ctx: DeviceContext,
) raises:
    """Sparse MLA prefill (DSv3.2 absorbed shape, BF16, SM100).

    Thin wrapper around ``mla_prefill_sparse`` that builds the
    ``MLASparseConfig`` from the passed dimensions so callers don't have to
    reach into the kernel's config type. The kernel itself hardcodes the
    DSv3.2 absorbed/latent shape (``qk_depth=576``, ``v_depth=512``,
    ``num_q_heads=128``, ``num_kv_heads=1``) and asserts on those values.

    Parameters:
        output_type: Output element type (must be the same width as
            ``q_type``; the kernel asserts this).
        q_type: Query element type (BF16 in the supported DSv3.2 shape).
        cache_t: KV cache type (typically a paged MLA cache obtained from
            ``kv_collection.get_key_cache(layer_idx)``).
        num_q_heads: Number of query heads (must be 128 for the DSv3.2
            absorbed shape).
        qk_depth: Per-head Q/K depth (must be 576 = ``kv_lora_rank(512) +
            qk_rope_head_dim(64)``).
        v_depth: Per-head V depth (must be 512 = ``kv_lora_rank``).
        indices_stride: Per-query indices buffer stride (= the indexer's
            ``index_topk``). Also used as the runtime ``indices_stride`` to
            the kernel.

    Args:
        output: Output tile tensor with shape
            ``[total_q_tokens, num_q_heads, v_depth]``.
        q: Query tile tensor with shape
            ``[total_q_tokens, num_q_heads, qk_depth]``.
        kv_cache: Paged MLA KV cache for the current layer.
        indices: Per-query gather4 indices, encoded as
            ``Int32(physical_block_id * page_size + token_offset_within_page)``
            (reinterpreted via the ``uint32`` tile-tensor view; ``-1``-bit-pattern
            sentinels are masked out by the kernel's k-valid producer).
        topk_lengths: Per-query effective top-k count (``[total_q_tokens]``).
        attn_sink_ptr: Optional attention sink (one ``Float32`` per query head).
            Pass a null pointer to skip the sink term in the softmax epilogue.
        scale: Softmax scale (``1 / sqrt(qk_nope_head_dim + qk_rope_head_dim) *
            mscale^2``; for DSv3.2 with mscale=1, ``1 / sqrt(192)``).
        ctx: GPU device context.
    """
    comptime config = MLASparseConfig[q_type](
        num_q_heads=num_q_heads,
        num_kv_heads=1,
        qk_depth=qk_depth,
        v_depth=v_depth,
        indices_stride=indices_stride,
        group=num_q_heads,
    )
    mla_prefill_sparse[
        config=config,
        group=num_q_heads,
        q_depth=qk_depth,
    ](
        output,
        q,
        kv_cache,
        indices,
        topk_lengths,
        attn_sink_ptr,
        scale,
        Int32(indices_stride),
        ctx,
    )
