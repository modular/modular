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
from std.collections import OptionalReg
from std.math import ceildiv, clamp, gcd
from std.sys import size_of
from std.gpu.host import DeviceBuffer, DeviceContext, FuncAttribute
from std.gpu.memory import AddressSpace
from std.gpu.primitives.grid_controls import pdl_launch_attributes, PDLLevel
from layout import (
    Coord,
    Idx,
    Layout,
    LayoutTensor,
    TileTensor,
    row_major,
)
from layout.tma_async import (
    TMATensorTile,
    _gather4_box_width,
)
from std.gpu.host.nvidia.tma import TensorMapL2Promotion, TensorMapSwizzle
from std.logger import Logger
from std.memory import bitcast

from nn.attention.gpu.nvidia.sm90.attention import (
    NonNullPointer,
    NullPointer,
    OptionalPointer,
)
from nn.attention.mha_mask import MHAMask
from nn.attention.mha_operand import MHAOperand, KVCacheMHAOperand
from nn.attention.mha_utils import (
    MHAConfig,
)
from nn.attention.gpu.nvidia.sm90.attention import KVTMATile
from std.utils.numerics import get_accum_type
from std.utils.index import Index, IndexList

comptime logger = Logger()

# Q-sequence fold bounds for speculative decoding dispatch.
# When q_max_seq_len is in [MIN_FOLD_Q, MAX_FOLD_Q], the kernel folds q-tokens
# into the head dimension. MAX_FOLD_Q is the largest supported q_seq for fold
# (1 actual + 7 spec tokens). The fold loop iterates over
# range(MIN_FOLD_Q, MAX_FOLD_Q + 1).
comptime MIN_FOLD_Q = 2
comptime MAX_FOLD_Q = 8

# TODO: Remove once stdlib's SwitchedFunction2 supports `raises`.
# The stdlib unswitch 2-predicate overload uses SwitchedFunction2 which
# is `def[sw0: Bool, sw1: Bool]() capturing[_] -> None` (no raises).
# Our sparse dispatch needs raises, so we define a local raises variant.
comptime _SwitchedFunction2Raises = def[
    sw0: Bool, sw1: Bool
]() raises capturing[_] -> None


@always_inline
def _unswitch_raises[
    switched_func: _SwitchedFunction2Raises
](dynamic_switch_a: Bool, dynamic_switch_b: Bool) raises:
    if dynamic_switch_a:

        @always_inline
        @parameter
        def switched_a_true[static_switch: Bool]() raises:
            switched_func[True, static_switch]()

        unswitch[switched_a_true](dynamic_switch_b)
    else:

        @always_inline
        @parameter
        def switched_a_false[static_switch: Bool]() raises:
            switched_func[False, static_switch]()

        unswitch[switched_a_false](dynamic_switch_b)


@always_inline
def _get_partition_bucket[half_sms: Int, i: Int]() -> Int:
    """Return the i-th partition bucket value.

    The bucket list uses half_sms (sm_count // 2) as its last entry so it
    adapts to any GPU: B200 (148 SMs) -> 74, B300 (160 SMs) -> 80, etc.
    """
    # Fixed bucket values for num_partitions to reduce CUDA graph captures.
    # Instead of many distinct values (each requiring a separate capture),
    # we map to a small set of fixed buckets.  The last bucket is half_sms
    # so the cap adapts to the target GPU.
    comptime _NUM_PARTITIONS = {
        0: 1,
        1: 2,
        2: 4,
        3: 8,
        4: 9,
        5: 16,
        6: 18,
        7: 20,
        8: 32,
        9: 37,
        10: 64,
        11: 72,
        12: half_sms,
    }
    comptime _default = _NUM_PARTITIONS.get(len(_NUM_PARTITIONS) - 1, 0)
    comptime res = _NUM_PARTITIONS.get(i, _default)
    return res


def _bucket_num_partitions[half_sms: Int](num_partitions: Int) -> Int:
    """Map num_partitions to the smallest bucket value >= num_partitions.

    The bucket list uses half_sms (sm_count // 2) as its last entry so it
    adapts to any GPU: B200 (148 SMs) -> 74, B300 (160 SMs) -> 80, etc.
    """
    comptime _NUM_PARTITIONS = {
        0: 1,
        1: 2,
        2: 4,
        3: 8,
        4: 9,
        5: 16,
        6: 18,
        7: 20,
        8: 32,
        9: 37,
        10: 64,
        11: 72,
        12: half_sms,
    }
    comptime _default = _NUM_PARTITIONS.get(len(_NUM_PARTITIONS) - 1, 0)
    comptime for kv in _NUM_PARTITIONS.items():
        comptime v = kv.value
        if num_partitions <= v:
            return v
    return _default


# Number of bucket entries in the partition table (fixed at 13).
comptime _NUM_PARTITION_BUCKETS = 13


from nn.attention.gpu.nvidia.sm100.mla_decode_utils import (
    MLA_SM100_Decode_Config,
    QOTMATile,
    ScalesTMATile,
    tma_tile_qo,
    tma_tile_scales,
    MLA_Decode_Pack,
    num_matrix_view_rows_decode,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_kv_bf16 import (
    MLA_SM100_Decode_KV_BF16,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_kv_fp8 import (
    MLA_SM100_Decode_KV_FP8,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_qkv_fp8 import (
    MLA_SM100_Decode_QKV_FP8,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_qkv_fp8_layout_g import (
    MLA_SM100_Decode_QKV_FP8_Layout_G,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_qkv_fp8_per_token_scale_rope_aware import (
    MLA_SM100_Decode_QKV_FP8_PerTokenScale_RopeAware,
)

from nn.attention.gpu.nvidia.sm100.mla_decode_combine import (
    mla_decode_combine_partial_outputs,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_sparse import (
    MLA_SM100_Decode_Sparse,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_sparse_kv_fp8 import (
    MLA_SM100_Decode_Sparse_KV_FP8,
)


# ------------------------------------------------------------------------------
# Compute num_partitions heuristic (shared by dispatch and pre-compute op)
# ------------------------------------------------------------------------------
#
# Two separate functions: _compute_num_partitions_64 for single head group
# (64 heads, Kimi K2.5) and _compute_num_partitions_128 for multiple head
# groups (128 heads, DeepSeek V3/R1). The routing function
# _compute_num_partitions dispatches via comptime if on the head group count.
# ------------------------------------------------------------------------------


def _compute_num_partitions_64[
    num_heads: Int,
    is_fp8_kv: Bool = False,
    half_sms: Int = 74,
](
    batch_size: Int,
    effective_max_cache_len: Int,
    q_max_seq_len: Int,
    split_page_size: Int,
    sm_count: Int,
) -> Int:
    """Wave-aligned split count for single head group (e.g. Kimi K2.5, 64 heads).

    This is the EXACT logic from the original unified _compute_num_partitions
    when _head_groups == 1.  Do NOT modify without verifying byte-for-byte
    equivalence with the 64-head production path.

    Parameters:
        num_heads: Number of Q attention heads (compile-time).
        is_fp8_kv: Whether the KV cache is FP8 (compile-time).
        half_sms: sm_count // 2 — maximum split-K partitions (compile-time).

    Args:
        batch_size: Current batch size.
        effective_max_cache_len: Max KV cache length.
        q_max_seq_len: Max query sequence length (1 for decode).
        split_page_size: Page granularity for split-K (64 or 128).
        sm_count: Number of SMs on the target GPU.

    Returns:
        The number of split-K partitions.
    """
    var num_kv_cache_pages = ceildiv(effective_max_cache_len, split_page_size)

    # When fold is active (spec decoding with num_heads * q_max_seq_len <= 64),
    # the kernel packs all q_tokens into the M tile of a single CTA, so
    # ctas_per_partition should NOT multiply by q_max_seq_len. q > 1 implies
    # spec decoding; the fold path requires the M tile to fit ≤ 64.
    var fold_active = (q_max_seq_len > 1) and (num_heads * q_max_seq_len <= 64)
    var q_len_factor = 1 if fold_active else q_max_seq_len
    var ctas_per_partition = ceildiv(num_heads, 64) * q_len_factor * batch_size

    # Single head group: 85% fill threshold (floor * ctas * 20 >= sm * 17).
    var floor_target = sm_count // ctas_per_partition
    var ceil_target = ceildiv(sm_count, ctas_per_partition)
    var target_partitions: Int
    if (
        floor_target >= 1
        and floor_target * ctas_per_partition * 20 >= sm_count * 17
    ):
        target_partitions = floor_target
    else:
        target_partitions = ceil_target

    target_partitions = min(target_partitions, half_sms)

    # max_pages_per_split for single head group: 18 * 2 / 1 = 36
    comptime _head_groups = 1
    comptime _base_max_pages_per_split = 18
    comptime max_pages_per_split = _base_max_pages_per_split * 2 // _head_groups
    var min_partitions_for_work = ceildiv(
        num_kv_cache_pages, max_pages_per_split
    )

    # Single head group: use max(target_partitions, min_partitions_for_work).
    # wave_aligned and page_constrained (num_kv_cache_pages//min_pages_per_split)
    # are not used for single head group -- target_partitions dominates.
    #
    # Fold-active long-cache path: cap np at 8 only for bs≥16. At bs<16 the
    # cap starves CTAs (target_partitions floor is needed for SM utilization).
    # At bs≥16, combine grid overhead at np=16 dominates, and bs*8 ≈ SM count
    # keeps decode CTAs sufficient.
    var num_partitions: Int
    if fold_active and batch_size >= 16 and effective_max_cache_len >= 8192:
        # Long-cache + high-batch: combine grid overhead at np=16 dominates,
        # and bs*8 ≈ SM count keeps decode CTAs sufficient. Cap np at 9 so
        # bs=16 → 144 CTAs (vs 128 at np=8) — fills more SMs without hitting
        # the np=16 combine-overhead cliff.
        num_partitions = min(target_partitions, 9)
    else:
        # Default: maintain SM utilization (low batch needs higher np).
        num_partitions = max(target_partitions, min_partitions_for_work)

    # Clamp: allow np=1 for very short cache + large batch, or when single
    # head group fills >= 80% of SMs.
    comptime _np1_cache_threshold = 512 if is_fp8_kv else 256
    var min_partitions: Int
    if effective_max_cache_len <= _np1_cache_threshold and batch_size >= 64:
        min_partitions = 1
    elif ctas_per_partition * 5 >= sm_count * 4:
        min_partitions = 1
    elif num_kv_cache_pages >= 2:
        min_partitions = 2
    else:
        min_partitions = 1
    num_partitions = clamp(
        num_partitions, min_partitions, min(half_sms, num_kv_cache_pages)
    )

    num_partitions = _bucket_num_partitions[half_sms](num_partitions)
    return num_partitions


def _compute_num_partitions_128[
    num_heads: Int,
    is_fp8_kv: Bool = False,
    half_sms: Int = 74,
](
    batch_size: Int,
    effective_max_cache_len: Int,
    q_max_seq_len: Int,
    split_page_size: Int,
    sm_count: Int,
) -> Int:
    """Wave-aligned split count for multiple head groups (e.g. DeepSeek V3/R1,
    128 heads).

    Optimized to match FlashInfer's split counts for DeepSeek configurations:
    - max_pages_per_split = 32 (vs 18) to reduce min_partitions_for_work
    - Doubled min_pages_per_split to compensate for 2x CTA multiplier
    - np=1 allowed for bs >= 64 with short cache (eliminates combine kernel)
    - wave_quantum capped at target_partitions for bs=4-8

    Parameters:
        num_heads: Number of Q attention heads (compile-time).
        is_fp8_kv: Whether the KV cache is FP8 (compile-time).
        half_sms: sm_count // 2 — maximum split-K partitions (compile-time).

    Args:
        batch_size: Current batch size.
        effective_max_cache_len: Max KV cache length.
        q_max_seq_len: Max query sequence length (1 for decode).
        split_page_size: Page granularity for split-K (64 or 128).
        sm_count: Number of SMs on the target GPU.

    Returns:
        The number of split-K partitions.
    """
    var num_kv_cache_pages = ceildiv(effective_max_cache_len, split_page_size)

    var ctas_per_partition = ceildiv(num_heads, 64) * q_max_seq_len * batch_size
    var wave_quantum = min(
        sm_count // gcd(ctas_per_partition, sm_count), half_sms
    )

    # Multiple head groups: 90% fill threshold.
    var floor_target = sm_count // ctas_per_partition
    var ceil_target = ceildiv(sm_count, ctas_per_partition)
    var target_partitions: Int
    if (
        floor_target >= 1
        and floor_target * ctas_per_partition * 10 >= sm_count * 9
    ):
        target_partitions = floor_target
    else:
        target_partitions = ceil_target

    target_partitions = min(target_partitions, half_sms)

    if batch_size >= 16:
        wave_quantum = min(wave_quantum, max(target_partitions, 1))

    # Change 4: Cap wave_quantum for bs=4-8 to prevent over-splitting.
    # At these batch sizes, wave_quantum=37 can cause excessive splits
    # (e.g., bs=8/cl=65K: 37 splits vs FlashInfer's 18). Capping at
    # target_partitions keeps splits aligned with SM fill needs.
    if batch_size >= 4 and batch_size <= 8:
        wave_quantum = min(wave_quantum, target_partitions)

    # Change 1: max_pages_per_split = 32 (vs 18 in the 64-head path).
    # For DeepSeek bs=8/cl=65K: pages=512, min_partitions = ceil(512/32)=16
    # (was ceil(512/18)=29 with the old value).
    comptime _head_groups = ceildiv(num_heads, 64)
    comptime max_pages_per_split = 32
    var min_partitions_for_work = ceildiv(
        num_kv_cache_pages, max_pages_per_split
    )

    comptime _min_pps_large_batch = 16 if is_fp8_kv else 8
    comptime _min_pps_small_batch = 8 if is_fp8_kv else 4
    var min_pages_per_split: Int
    if batch_size >= 64:
        min_pages_per_split = _min_pps_large_batch
    elif batch_size <= 8 and is_fp8_kv:
        min_pages_per_split = 4
    else:
        min_pages_per_split = _min_pps_small_batch

    # Base thresholds are tuned for 2 head groups; apply 2/head_groups.
    min_pages_per_split = min_pages_per_split * 2 // _head_groups

    # Change 2: Double min_pages_per_split to compensate for 2x CTA
    # multiplier with 2 head groups. This halves the page_constrained
    # value, reducing np.
    min_pages_per_split = min_pages_per_split * 2

    var num_waves = max(1, ceildiv(min_partitions_for_work, wave_quantum))
    var wave_aligned = num_waves * wave_quantum

    # Multiple head groups: pick smaller of wave-aligned and
    # page-constrained, ensure at least target_partitions.
    var num_partitions = max(
        target_partitions,
        min(wave_aligned, num_kv_cache_pages // min_pages_per_split),
    )

    # Change 3: Allow np=1 at high batch with short cache.
    # For bs >= 64 and effective_max_cache_len <= 2176, the combine kernel
    # overhead (16K-32K CTAs) is catastrophic. Allowing np=1 eliminates it.
    comptime _np1_cache_threshold = 512 if is_fp8_kv else 256
    var min_partitions: Int
    if effective_max_cache_len <= _np1_cache_threshold and batch_size >= 64:
        min_partitions = 1
        # R2: Override target_partitions so it doesn't act as a floor that
        # prevents np=1. Recompute num_partitions with the lowered floor.
        target_partitions = 1
        num_partitions = max(
            target_partitions,
            min(wave_aligned, num_kv_cache_pages // min_pages_per_split),
        )
    elif batch_size >= 64 and effective_max_cache_len <= 2176:
        # DeepSeek-specific: eliminate combine kernel for high-batch
        # short-cache.
        min_partitions = 1
        # R2: Same fix — recompute with lowered floor.
        target_partitions = 1
        num_partitions = max(
            target_partitions,
            min(wave_aligned, num_kv_cache_pages // min_pages_per_split),
        )
    elif num_kv_cache_pages >= 2:
        min_partitions = 2
    else:
        min_partitions = 1

    # R1: When a single partition already fills the GPU
    # (ctas_per_partition >= sm_count), splitting just adds combine kernel
    # overhead without improving decode parallelism. Force np=1.
    #
    # For DeepSeek 128 heads: ctas_per_partition = 2 * batch_size.
    # ctas_per_partition >= sm_count when bs >= sm_count/2 (e.g., bs >= 74
    # on B200 with 148 SMs).
    #
    # The combine kernel grid at these batch sizes is catastrophic:
    #   bs=512/cl=4096 with np=2: 32768 combine CTAs (221 waves at wph=4)
    #   bs=256/cl=4096 with np=2: 16384 combine CTAs (110 waves at wph=4)
    # Forcing np=1 eliminates combine entirely. Each decode CTA processes
    # more pages but that cost is less than the combine overhead.
    if ctas_per_partition >= sm_count:
        num_partitions = 1
        min_partitions = 1

    num_partitions = clamp(
        num_partitions, min_partitions, min(half_sms, num_kv_cache_pages)
    )

    num_partitions = _bucket_num_partitions[half_sms](num_partitions)
    return num_partitions


def _compute_num_partitions[
    num_heads: Int,
    is_fp8_kv: Bool = False,
    half_sms: Int = 74,
](
    batch_size: Int,
    effective_max_cache_len: Int,
    q_max_seq_len: Int,
    split_page_size: Int,
    sm_count: Int,
) -> Int:
    """Routing function that dispatches to head-count-specific heuristics.

    Single head group (num_heads <= 64, e.g. Kimi K2.5) calls
    _compute_num_partitions_64.  Multiple head groups (num_heads > 64,
    e.g. DeepSeek V3/R1) calls _compute_num_partitions_128.

    Parameters:
        num_heads: Number of Q attention heads (compile-time).
        is_fp8_kv: Whether the KV cache is FP8 (compile-time).
        half_sms: sm_count // 2 — maximum split-K partitions (compile-time).

    Args:
        batch_size: Current batch size.
        effective_max_cache_len: Max KV cache length.
        q_max_seq_len: Max query sequence length (1 for decode).
        split_page_size: Page granularity for split-K (64 or 128).
        sm_count: Number of SMs on the target GPU.

    Returns:
        The number of split-K partitions.
    """
    comptime _head_groups = ceildiv(num_heads, 64)

    comptime if _head_groups == 1:
        return _compute_num_partitions_64[num_heads, is_fp8_kv, half_sms](
            batch_size,
            effective_max_cache_len,
            q_max_seq_len,
            split_page_size,
            sm_count,
        )
    else:
        return _compute_num_partitions_128[num_heads, is_fp8_kv, half_sms](
            batch_size,
            effective_max_cache_len,
            q_max_seq_len,
            split_page_size,
            sm_count,
        )


# ------------------------------------------------------------------------------
# Public pre-compute function for MOGG ops
# ------------------------------------------------------------------------------
def compute_mla_dispatch_scalars[
    num_heads: Int,
    _is_cache_length_accurate: Bool = False,
    is_fp8_kv: Bool = False,
    half_sms: Int = 74,
](
    batch_size: Int,
    max_cache_valid_length: Int,
    q_max_seq_len: Int,
    sm_count: Int,
) -> Tuple[Int, Int, Int]:
    """Pure computation of the packed 3-value MLA dispatch metadata.

    Returns ``(batch_size, q_max_seq_len, num_partitions)``.
    """
    var effective = max_cache_valid_length

    comptime if not _is_cache_length_accurate:
        effective += q_max_seq_len

    var split_page_size = 64 if (effective <= 512 and batch_size >= 32) else 128
    var num_partitions = _compute_num_partitions[
        num_heads, is_fp8_kv, half_sms
    ](batch_size, effective, q_max_seq_len, split_page_size, sm_count)

    return (batch_size, q_max_seq_len, num_partitions)


def compute_mla_dispatch_scalars_runtime(
    batch_size: Int,
    max_cache_valid_length: Int,
    q_max_seq_len: Int,
    num_heads: Int,
    is_fp8_kv: Bool,
    sm_count: Int,
) raises -> Tuple[Int, Int, Int]:
    if is_fp8_kv:
        if num_heads == 8:
            return compute_mla_dispatch_scalars[8, is_fp8_kv=True](
                batch_size,
                max_cache_valid_length,
                q_max_seq_len,
                sm_count,
            )
        if num_heads == 16:
            return compute_mla_dispatch_scalars[16, is_fp8_kv=True](
                batch_size,
                max_cache_valid_length,
                q_max_seq_len,
                sm_count,
            )
        if num_heads == 32:
            return compute_mla_dispatch_scalars[32, is_fp8_kv=True](
                batch_size,
                max_cache_valid_length,
                q_max_seq_len,
                sm_count,
            )
        if num_heads == 64:
            return compute_mla_dispatch_scalars[64, is_fp8_kv=True](
                batch_size,
                max_cache_valid_length,
                q_max_seq_len,
                sm_count,
            )
        if num_heads == 128:
            return compute_mla_dispatch_scalars[128, is_fp8_kv=True](
                batch_size,
                max_cache_valid_length,
                q_max_seq_len,
                sm_count,
            )
    else:
        if num_heads == 8:
            return compute_mla_dispatch_scalars[8](
                batch_size,
                max_cache_valid_length,
                q_max_seq_len,
                sm_count,
            )
        if num_heads == 16:
            return compute_mla_dispatch_scalars[16](
                batch_size,
                max_cache_valid_length,
                q_max_seq_len,
                sm_count,
            )
        if num_heads == 32:
            return compute_mla_dispatch_scalars[32](
                batch_size,
                max_cache_valid_length,
                q_max_seq_len,
                sm_count,
            )
        if num_heads == 64:
            return compute_mla_dispatch_scalars[64](
                batch_size,
                max_cache_valid_length,
                q_max_seq_len,
                sm_count,
            )
        if num_heads == 128:
            return compute_mla_dispatch_scalars[128](
                batch_size,
                max_cache_valid_length,
                q_max_seq_len,
                sm_count,
            )
    raise Error(
        "Unsupported MLA num_heads for direct dispatch metadata binding: "
        + String(num_heads)
    )


struct MLADispatchScalarArgs[
    num_heads: Int,
    _is_cache_length_accurate: Bool = False,
    is_fp8_kv: Bool = False,
]:
    """Pre-computed MLA decode args for the legacy (non-capturable) path.

    Owns a GPU buffer containing ``[batch_size, q_max_seq_len, num_partitions]``
    and caches the host-side ``batch_size``/``q_max_seq_len`` pair needed by
    ``mla_decode_sm100_dispatch``.

    Usage::

        var args = MLADispatchScalarArgs[num_heads=128](
            batch_size, max_cache_len, q_max_seq_len, ctx,
        )
        var gpu_lt = args.gpu_layout_tensor()
        mla_decode_sm100_dispatch[...](
            ..., gpu_lt,
            args.batch_size, args.q_max_seq_len, max_cache_len,
            ctx,
        )
        _ = args  # keepalive
    """

    comptime MLAScalarArgsLT = LayoutTensor[
        DType.int64, Layout.row_major(3), MutAnyOrigin
    ]

    var gpu_buf: DeviceBuffer[DType.int64]
    var batch_size: Int
    var q_max_seq_len: Int

    def __init__(
        out self,
        batch_size: Int,
        max_cache_len: Int,
        q_max_seq_len: Int,
        ctx: DeviceContext,
    ) raises:
        self.gpu_buf = ctx.enqueue_create_buffer[DType.int64](3)
        self.batch_size = batch_size
        self.q_max_seq_len = q_max_seq_len

        comptime sm_count = ctx.default_device_info.sm_count
        comptime _half_sms = sm_count // 2
        var scalars = compute_mla_dispatch_scalars[
            num_heads=Self.num_heads,
            _is_cache_length_accurate=Self._is_cache_length_accurate,
            is_fp8_kv=Self.is_fp8_kv,
            half_sms=_half_sms,
        ](batch_size, max_cache_len, q_max_seq_len, sm_count)

        var host_args = InlineArray[Int64, 3](uninitialized=True)
        host_args[0] = Int64(scalars[0])
        host_args[1] = Int64(scalars[1])
        host_args[2] = Int64(scalars[2])
        var output_buf = DeviceBuffer[DType.int64](
            ctx, self.gpu_buf.unsafe_ptr(), 3, owning=False
        )
        output_buf.enqueue_copy_from(
            UnsafePointer(to=host_args).bitcast[Scalar[DType.int64]]()
        )

    def gpu_layout_tensor(
        self,
    ) -> Self.MLAScalarArgsLT:
        return Self.MLAScalarArgsLT(
            rebind[UnsafePointer[Scalar[DType.int64], origin=MutAnyOrigin]](
                self.gpu_buf.unsafe_ptr()
            ),
        )


# ------------------------------------------------------------------------------
# MLA decoding implementation for SM100
# ------------------------------------------------------------------------------
def mla_decode_sm100_dispatch[
    q_type: DType,
    k_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    config: MHAConfig,
    depth: Int,
    num_heads: Int,
    group: Int = 1,
    *,
    ragged: Bool = False,
    _is_cache_length_accurate: Bool = False,
    decoding_warp_split_k: Bool = False,
    per_token_scale_rope_aware: Bool = False,
    sparse: Bool = False,
    # Sparse-only routing flag: when True, route to the BF16-rope sparse
    # kernel (split FP8 nope + BF16 rope, two TMAs). When False (default),
    # route to the all-FP8 sparse kernel (single 576-byte gather4 TMA).
    rope_aware_kv_sparse: Bool = False,
](
    q: TileTensor[q_type, address_space=AddressSpace.GENERIC, ...],
    k: k_t,
    output: TileTensor[
        mut=True, output_type, address_space=AddressSpace.GENERIC, ...
    ],
    scale: Float32,
    valid_length: TileTensor[
        DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    mask: mask_t,
    scalar_args_buf: TileTensor[
        DType.int64, address_space=AddressSpace.GENERIC, ...
    ],
    batch_size: Int,
    q_max_seq_len: Int,
    max_cache_valid_length: Int,
    ctx: DeviceContext,
    q_scale_ptr: OptionalReg[
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
    ] = None,
    d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    indices_stride: Int = 0,
    topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    attn_sink_ptr: OptionalReg[
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
    ] = None,
    # Extra KV parameters (forwarded to mla_decode_sm100_sink_split_k).
    extra_k: OptionalReg[k_t] = None,
    extra_d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    extra_indices_stride: Int = 0,
    extra_topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    extra_scales_ptr: OptionalReg[
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
    ] = None,
) raises:
    var scales_ptr = k.scales_raw_ptr()

    var effective_max_cache_len = max_cache_valid_length

    comptime if not _is_cache_length_accurate:
        effective_max_cache_len += q_max_seq_len

    # For sparse decode, the split-K factor must be driven by the actual
    # number of tokens each batch attends to (max_topk + max_extra_topk),
    # NOT the full cache length.  With topk << cache_len (e.g. topk=256
    # vs cache_len=163840), using the full cache length creates far too
    # many splits — each excess CTA just early-exits but still wastes
    # launch overhead and combine kernel grid size.
    #
    # indices_stride is the max topk across all batches: for fixed topk
    # it equals topk; for variable topk it equals max(topk_per_batch)
    # (the allocation stride).  Similarly extra_indices_stride is
    # max_extra_topk.
    var effective_split_len = effective_max_cache_len
    comptime if sparse:
        var max_topk = indices_stride
        var max_extra_topk = extra_indices_stride
        effective_split_len = max_topk + max_extra_topk

    # Sliding-window split-K shrink : cap the effective cache length at
    # window_size + q_max_seq_len so partition heuristics don't
    # over-split a region the kernel will skip anyway.
    comptime if mask_t.get_type_name() == "SlidingWindowCausalMask":
        # BM/BN_QK values are immaterial for window_size extraction; use the
        # MLA decode tile shape (BM=64, BN_QK=64) for consistency.
        comptime _sw_window_size: Int = Int(
            mask_t.mask_strategies[64, 64]()[0]._upper_triangular_window_size
        )
        var sw_cap: Int = _sw_window_size + q_max_seq_len
        # at high batch (bs>=64) with a small SW cap
        # (<=2048), shrinking the effective_split_len to sw_cap forces the
        # partition heuristic into np=1.
        if batch_size >= 64 and sw_cap <= 2048:
            sw_cap = 2048
        effective_split_len = min(effective_split_len, sw_cap)

    var use_small_split_pages = effective_split_len <= 512 and batch_size >= 32
    var split_page_size = 64 if use_small_split_pages else 128
    comptime sm_count = ctx.default_device_info.sm_count
    comptime _half_sms = sm_count // 2
    comptime _is_fp8_kv = (k_t.dtype == DType.float8_e4m3fn)
    var num_partitions = _compute_num_partitions[
        num_heads, _is_fp8_kv, _half_sms
    ](
        batch_size,
        effective_split_len,
        q_max_seq_len,
        split_page_size,
        sm_count,
    )

    # When sparse mode or sliding-window changes num_partitions, the GPU
    # scalar_args_buf (which was pre-computed from cache_len by the caller)
    # must be updated so the kernel reads the same value the host uses for
    # grid/buffers.
    comptime if sparse or (mask_t.get_type_name() == "SlidingWindowCausalMask"):
        var corrected_args = InlineArray[Int64, 3](uninitialized=True)
        corrected_args[0] = Int64(batch_size)
        corrected_args[1] = Int64(q_max_seq_len)
        corrected_args[2] = Int64(num_partitions)
        var scalar_buf = DeviceBuffer[DType.int64](
            ctx, scalar_args_buf.ptr, 3, owning=False
        )
        scalar_buf.enqueue_copy_from(
            UnsafePointer(to=corrected_args).bitcast[Scalar[DType.int64]]()
        )

    # =========================================================================
    # split_page_size routing: use finer split granularity for short cache
    # with moderate-to-large batch to improve split balance.
    #
    # When cache is short (effective_max_cache_len <= 512, i.e. <=4 pages at
    # page_size=128) and batch is large enough (>=32), splitting with
    # page_size=64 gives twice the page count, enabling better work
    # distribution across splits.
    # For example, bs=64/cl=256 gets 5 pages at page_size=64 (vs 3 at 128),
    # allowing np=2 with 2-3 pages per split instead of 1-2.
    # =========================================================================
    @parameter
    @always_inline
    def launch_impl[split_page_size_param: Int]() raises:
        _mla_decode_sm100_dispatch_impl[
            q_type=q_type,
            k_t=k_t,
            output_type=output_type,
            mask_t=mask_t,
            config=config,
            depth=depth,
            num_heads=num_heads,
            group=group,
            ragged=ragged,
            _is_cache_length_accurate=_is_cache_length_accurate,
            decoding_warp_split_k=decoding_warp_split_k,
            split_page_size=split_page_size_param,
            per_token_scale_rope_aware=per_token_scale_rope_aware,
            sparse=sparse,
            rope_aware_kv_sparse=rope_aware_kv_sparse,
        ](
            q,
            k,
            output,
            scale,
            valid_length,
            mask,
            scales_ptr,
            scalar_args_buf,
            batch_size,
            q_max_seq_len,
            num_partitions,
            effective_max_cache_len,
            ctx,
            q_scale_ptr,
            d_indices,
            indices_stride,
            topk_lengths,
            attn_sink_ptr,
            extra_k=extra_k,
            extra_d_indices=extra_d_indices,
            extra_indices_stride=extra_indices_stride,
            extra_topk_lengths=extra_topk_lengths,
            extra_scales_ptr=extra_scales_ptr,
        )

    comptime if k_t.page_size == 0 or k_t.page_size >= 128:
        # Non-paged (page_size=0) and page_size>=128 both satisfy the
        # `page_size % split_page_size == 0` assertion for both 64 and
        # 128 split granularities, so the full routing is emittable.
        if use_small_split_pages:
            launch_impl[64]()
        else:
            launch_impl[128]()
    elif k_t.page_size == 64:
        # page_size=64: only split_page_size=64 satisfies the divisibility
        # assertion. The 128 branch would fail instantiation.
        launch_impl[64]()
    else:
        # page_size < 64 (e.g. 16, 32): split_page_size = page_size so each
        # split aligns with a whole number of physical pages. Dispatch the
        # split_page_size at comptime so only the valid instantiation is
        # emitted.
        launch_impl[k_t.page_size]()


# ------------------------------------------------------------------------------
# Inner dispatch implementation parameterized on split_page_size
# ------------------------------------------------------------------------------
def _mla_decode_sm100_dispatch_impl[
    q_type: DType,
    k_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    config: MHAConfig,
    depth: Int,
    num_heads: Int,
    group: Int = 1,
    *,
    ragged: Bool = False,
    _is_cache_length_accurate: Bool = False,
    decoding_warp_split_k: Bool = False,
    split_page_size: Int = 128,
    per_token_scale_rope_aware: Bool = False,
    sparse: Bool = False,
    rope_aware_kv_sparse: Bool = False,
](
    q: TileTensor[q_type, address_space=AddressSpace.GENERIC, ...],
    k: k_t,
    output: TileTensor[
        mut=True, output_type, address_space=AddressSpace.GENERIC, ...
    ],
    scale: Float32,
    valid_length: TileTensor[
        DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    mask: mask_t,
    scales_ptr: UnsafePointer[Scalar[DType.float32], origin=MutAnyOrigin],
    scalar_args_buf: TileTensor[
        DType.int64, address_space=AddressSpace.GENERIC, ...
    ],
    batch_size: Int,
    q_max_seq_len: Int,
    num_partitions: Int,
    effective_max_cache_len: Int,
    ctx: DeviceContext,
    q_scale_ptr: OptionalReg[
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
    ] = None,
    d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    indices_stride: Int = 0,
    topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    attn_sink_ptr: OptionalReg[
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
    ] = None,
    # Extra KV parameters (forwarded to mla_decode_sm100_sink_split_k).
    extra_k: OptionalReg[k_t] = None,
    extra_d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    extra_indices_stride: Int = 0,
    extra_topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    extra_scales_ptr: OptionalReg[
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
    ] = None,
) raises:
    comptime hw_info = ctx.default_device_info
    comptime sm_count = hw_info.sm_count
    comptime _half_sms = sm_count // 2

    comptime AccumType = get_accum_type[output.dtype]()
    comptime v_depth = depth - 64
    comptime _is_fp8_kv = (k_t.dtype == DType.float8_e4m3fn)

    # Ensure KV cache page_size is evenly divisible by split_page_size.
    # If the KV cache page_size shrinks in the future, splits must not
    # straddle physical page boundaries.
    comptime assert (
        k_t.page_size % split_page_size == 0
    ), "KV cache page_size must be divisible by split_page_size"

    var block_z = batch_size * num_partitions

    if num_partitions > 1:
        comptime SplitAccumType = NonNullPointer[AccumType]
        # Create partial output buffer (same type as output - bfloat16)
        # Each split writes its partial attention result here
        # Note: Output dimension is v_depth (512), not depth (576)
        o_accum_split_data = ctx.enqueue_create_buffer[output_type](
            Int(
                num_partitions
                * batch_size
                * q_max_seq_len
                * num_heads
                * v_depth
            )
        )
        var o_accum_split = TileTensor(
            o_accum_split_data,
            row_major(
                Coord(
                    Idx(num_partitions),
                    Idx(batch_size),
                    Idx(q_max_seq_len),
                    Idx(Int(num_heads)),
                    Idx(Int(v_depth)),
                )
            ),
        )
        # Create LSE accumulator buffer (AccumType = float32 for numerical stability)
        var lse_accum_data = ctx.enqueue_create_buffer[AccumType](
            Int(num_partitions * batch_size * q_max_seq_len * num_heads)
        )
        var lse_accum_split = TileTensor(
            lse_accum_data,
            row_major(
                Coord(
                    Idx(num_partitions),
                    Idx(batch_size),
                    Idx(q_max_seq_len),
                    Idx(Int(num_heads)),
                )
            ),
        )
        var lse_accum_split_ptr: SplitAccumType = {
            lse_accum_split.to_device_buffer(ctx).unsafe_ptr()
        }

        # Get input_row_offsets pointer for combine kernel's ragged output writes.
        var input_row_offsets_ptr = rebind[
            UnsafePointer[Scalar[DType.uint32], origin=MutAnyOrigin]
        ](valid_length.ptr)

        # Inner function parameterized on has_attn_sink to specialize both
        # the decode kernel and combine kernel at compile time. The runtime
        # branch on attn_sink_ptr happens once (below) to select the right
        # compile-time specialization.
        @parameter
        def _launch_split_k_path[_has_attn_sink: Bool]() raises:
            # Launch main MLA decode kernel (writes partial results to accumulators)
            mla_decode_sm100_sink_split_k[
                q_type=q_type,
                k_t=k_t,
                output_type=output_type,
                mask_t=mask_t,
                config=config,
                depth=depth,
                num_heads=num_heads,
                SplitAccumType=SplitAccumType,
                group=group,
                ragged=ragged,
                _is_cache_length_accurate=_is_cache_length_accurate,
                decoding_warp_split_k=True,
                split_page_size=split_page_size,
                per_token_scale_rope_aware=per_token_scale_rope_aware,
                has_attn_sink=_has_attn_sink,
                sparse=sparse,
                rope_aware_kv_sparse=rope_aware_kv_sparse,
            ](
                q,
                k,
                o_accum_split,
                lse_accum_split_ptr,
                scale,
                batch_size,
                block_z,
                num_partitions,
                q_max_seq_len,
                valid_length,
                mask,
                scales_ptr,
                scalar_args_buf,
                ctx,
                q_scale_ptr,
                d_indices,
                indices_stride,
                topk_lengths,
                attn_sink_ptr,
                extra_k=extra_k,
                extra_d_indices=extra_d_indices,
                extra_indices_stride=extra_indices_stride,
                extra_topk_lengths=extra_topk_lengths,
                extra_scales_ptr=extra_scales_ptr,
                effective_max_cache_len=effective_max_cache_len,
            )

            # Dispatch to specialized kernel based on num_partitions for compile-time unrolling.
            # Supports up to sm_count//2 splits to allow higher SM utilization.
            @parameter
            def launch_combine[n_splits: Int, wph: Int]() raises:
                mla_decode_combine_partial_outputs[
                    output_type=output_type,
                    accum_type=AccumType,
                    head_dim=v_depth,
                    num_splits=n_splits,
                    ragged=ragged,
                    warps_per_head=wph,
                    has_attn_sink=_has_attn_sink,
                ](
                    o_accum_split,
                    lse_accum_split,
                    output,
                    input_row_offsets_ptr,
                    attn_sink_ptr,
                    batch_size,
                    q_max_seq_len,
                    Int(num_heads),
                    ctx,
                )

            @parameter
            def launch_combine_split_parallel[n_splits: Int]() raises:
                mla_decode_combine_partial_outputs[
                    output_type=output_type,
                    accum_type=AccumType,
                    head_dim=v_depth,
                    num_splits=n_splits,
                    ragged=ragged,
                    warps_per_head=1,
                    has_attn_sink=_has_attn_sink,
                    split_parallel=True,
                ](
                    o_accum_split,
                    lse_accum_split,
                    output,
                    input_row_offsets_ptr,
                    attn_sink_ptr,
                    batch_size,
                    q_max_seq_len,
                    Int(num_heads),
                    ctx,
                )

            @parameter
            def dispatch_combine[wph: Int]() raises:
                """Dispatch the combine kernel with the given warps_per_head,
                matching num_partitions to the correct compile-time bucket.

                Raises:
                    If the kernel dispatch fails.
                """
                comptime for _b in range(_NUM_PARTITION_BUCKETS):
                    comptime if _get_partition_bucket[_half_sms, _b]() >= 2:
                        if (
                            num_partitions
                            == _get_partition_bucket[_half_sms, _b]()
                        ):
                            launch_combine[
                                _get_partition_bucket[_half_sms, _b](), wph
                            ]()

            @parameter
            def dispatch_combine_split_parallel() raises:
                """Dispatch the split-parallel combine kernel, matching
                num_partitions to the correct compile-time bucket.

                Raises:
                    If the kernel dispatch fails.
                """
                comptime for _b in range(_NUM_PARTITION_BUCKETS):
                    comptime if _get_partition_bucket[_half_sms, _b]() >= 2:
                        if (
                            num_partitions
                            == _get_partition_bucket[_half_sms, _b]()
                        ):
                            launch_combine_split_parallel[
                                _get_partition_bucket[_half_sms, _b]()
                            ]()

            # Choose combine strategy based on split count and batch size.
            #
            # Split-parallel combine: 8 warps per CTA each process a range
            # of splits independently, then tree-reduce in shared memory.
            # This gives 8x memory stream parallelism and avoids the massive
            # compile-time unrolled loop of the original kernel. Best for
            # long KV cache (>= 16384 tokens) where the many splits benefit
            # from 8x parallel memory streams.
            #
            # Original combine: warps cooperate on head_dim within each split.
            # Better for moderate cache lengths where the per-split overhead
            # of split-parallel dominates.
            #
            # Decision: use split-parallel when cache_length >= 16384.
            # For shorter cache, use the original kernel with wph tuning.
            #
            # The original kernel's wph selection logic follows (unchanged):
            #
            # The combine grid is (batch_size, seq_len, ceildiv(num_heads, hpb))
            # where hpb = heads_per_block = 8 // wph. Each CTA processes hpb
            # heads, using wph warps per head. The total combine CTA count is:
            #   batch_size * seq_len * ceildiv(num_heads, 8 // wph)
            #
            # This is a heuristic based on the following observations and empirical
            # tuning for B200 with 148 SMs:
            #
            # For DeepSeek V3/R1 (num_heads=128, seq_len=1):
            #   wph=2: hpb=4, grid_z=32,  combine CTAs = bs * 32
            #   wph=4: hpb=2, grid_z=64,  combine CTAs = bs * 64
            #   wph=8: hpb=1, grid_z=128, combine CTAs = bs * 128
            #
            # The optimal wph depends on two factors:
            #
            # 1. Batch size (controls combine CTA count): large batch means more
            #    CTAs launched, so lower wph (fewer CTAs) reduces combine overhead.
            #
            # 2. Number of splits (work per CTA): with few splits (np <= 4), each
            #    CTA only reduces 2-4 partial results -- the work per CTA is tiny
            #    regardless of wph. In this case, wph=4 beats wph=2 because the
            #    extra warps reduce per-CTA latency via more parallel vector loads,
            #    and the CTA count difference (e.g., bs*64 vs bs*32) is secondary
            #    since each CTA finishes very quickly.
            #    With many splits (np > 4), the combine work per CTA is non-trivial
            #    and CTA count dominates, so lower wph is preferred.
            #
            # We use combine_ctas_base (the combine CTA count at wph=2, where
            # hpb=4) as the decision metric. This adapts to models with different
            # num_heads, unlike raw batch_size thresholds.
            #
            # For DeepSeek V3/R1 (num_heads=128, q_max_seq_len=1):
            #   combine_ctas_base = bs * 32
            #   combine_ctas_base >= 2048 <==> bs >= 64
            #   combine_ctas_base >= 512  <==> bs >= 16
            #
            # Decision matrix (empirically tuned for B200 with 148 SMs):
            #   cache_len >= 16384:                                 split-parallel
            #   BF16: ctas >= 4096 AND np <= 4 AND cache <= 1280:   wph=1
            #   ctas >= 2048 AND np > 4:                            wph=2
            #   ctas >= 512:                                        wph=4
            #   ctas < 512 (small grid):                            wph=8
            #
            # The wph=1 path is BF16-only. FP8 decode finishes ~2x faster
            # (half the KV bytes), leaving less PDL overlap for the combine
            # kernel. With wph=1 the combine kernel has too few warps per
            # head to sustain memory throughput. FP8 falls through to the
            # wph=4 path which provides better per-head parallelism.
            #   bs=128, cl=1024 FP8: wph=1 -> 37.3us, wph=4 -> 34.8us
            var combine_ctas_base = (
                batch_size * q_max_seq_len * ceildiv(num_heads, 4)
            )

            # Actual combine CTA count at wph=8 (highest CTA count candidate).
            # Used as a secondary guard to prevent excessive wave counts when
            # combine_ctas_base (computed at wph=2) falls below the primary
            # thresholds. This is especially important for models with fewer
            # heads (e.g., Kimi K2.5 with 64 heads) where ctas_base is half
            # of DeepSeek's but the actual wph=8 CTA count can still be large.
            #
            # For Kimi K2.5 (num_heads=64):
            #   wph=8: hpb=1, grid_z=64,  CTAs = bs * 64
            #   wph=4: hpb=2, grid_z=32,  CTAs = bs * 32
            #   wph=2: hpb=4, grid_z=16,  CTAs = bs * 16
            #
            #   bs=8: ctas_base=128 < 512 (old path -> wph=8, 512 CTAs, 3.5
            #         waves). With this guard: 8*64=512 > 296, -> wph=4,
            #         256 CTAs (1.7 waves).
            comptime _ctas_wph8 = ceildiv(num_heads, 1)  # hpb=1 at wph=8

            if effective_max_cache_len >= 16384 and batch_size <= 2:
                dispatch_combine_split_parallel()
            elif (
                combine_ctas_base >= 4096
                and num_partitions <= 4
                and effective_max_cache_len <= 1280
                and not _is_fp8_kv
            ):
                dispatch_combine[1]()
            elif combine_ctas_base >= 2048 and num_partitions > 4:
                dispatch_combine[2]()
            elif combine_ctas_base >= 512:
                dispatch_combine[4]()
            elif batch_size * _ctas_wph8 > sm_count * 2:
                dispatch_combine[4]()
            else:
                dispatch_combine[8]()

        # Runtime branch: specialize on has_attn_sink for both the decode
        # kernel and the combine kernel. When attn_sink_ptr is null, the
        # has_attn_sink=False path generates zero overhead.
        if attn_sink_ptr:
            _launch_split_k_path[True]()
        else:
            _launch_split_k_path[False]()
    else:
        comptime SplitAccumType = NullPointer[AccumType]
        var lse_accum_split_ptr: SplitAccumType = {}

        @parameter
        def _launch_no_split_path[_has_attn_sink: Bool]() raises:
            mla_decode_sm100_sink_split_k[
                q_type=q_type,
                k_t=k_t,
                output_type=output_type,
                mask_t=mask_t,
                config=config,
                depth=depth,
                num_heads=num_heads,
                SplitAccumType=SplitAccumType,
                group=group,
                ragged=ragged,
                _is_cache_length_accurate=_is_cache_length_accurate,
                decoding_warp_split_k=False,
                split_page_size=split_page_size,
                per_token_scale_rope_aware=per_token_scale_rope_aware,
                has_attn_sink=_has_attn_sink,
                sparse=sparse,
                rope_aware_kv_sparse=rope_aware_kv_sparse,
            ](
                q,
                k,
                output,
                lse_accum_split_ptr,
                scale,
                batch_size,
                block_z,
                num_partitions,
                q_max_seq_len,
                valid_length,
                mask,
                scales_ptr,
                scalar_args_buf,
                ctx,
                q_scale_ptr,
                d_indices,
                indices_stride,
                topk_lengths,
                attn_sink_ptr,
                extra_k=extra_k,
                extra_d_indices=extra_d_indices,
                extra_indices_stride=extra_indices_stride,
                extra_topk_lengths=extra_topk_lengths,
                extra_scales_ptr=extra_scales_ptr,
                effective_max_cache_len=effective_max_cache_len,
            )

        if attn_sink_ptr:
            _launch_no_split_path[True]()
        else:
            _launch_no_split_path[False]()


def mla_decode_sm100_sink_split_k[
    q_type: DType,
    k_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    *,
    config: MHAConfig,
    depth: Int,
    num_heads: Int,
    SplitAccumType: OptionalPointer,
    group: Int,
    ragged: Bool,
    _is_cache_length_accurate: Bool,
    decoding_warp_split_k: Bool,
    split_page_size: Int = 128,
    per_token_scale_rope_aware: Bool = False,
    has_attn_sink: Bool = False,
    sparse: Bool = False,
    # Sparse-only routing flag: when True, route to the BF16-rope sparse
    # kernel (split FP8 nope + BF16 rope, two TMAs). When False (default),
    # route to the all-FP8 sparse kernel (single 576-byte gather4 TMA).
    # Only meaningful when `sparse=True`. Ignored for dense paths.
    rope_aware_kv_sparse: Bool = False,
](
    q: TileTensor[q_type, address_space=AddressSpace.GENERIC, ...],
    k: k_t,
    output: TileTensor[mut=True, address_space=AddressSpace.GENERIC, ...],
    lse_accum_split_ptr: SplitAccumType,
    scale: Float32,
    batch_size: Int,
    block_z: Int,
    num_partitions: Int,
    q_max_seq_len: Int,
    valid_length: TileTensor[
        DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    mask: mask_t,
    scales_ptr: UnsafePointer[Scalar[DType.float32], origin=MutAnyOrigin],
    scalar_args_buf: TileTensor[
        DType.int64, address_space=AddressSpace.GENERIC, ...
    ],
    ctx: DeviceContext,
    q_scale_ptr: OptionalReg[
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
    ] = None,
    d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    indices_stride: Int = 0,
    topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    attn_sink_ptr: OptionalReg[
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
    ] = None,
    # Extra KV: separate always-attend cache. When extra_k is provided
    # (non-default), the sparse kernel appends extra_topk tokens after
    # the original topk tokens in a unified loop.
    extra_k: OptionalReg[k_t] = None,
    extra_d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    extra_indices_stride: Int = 0,
    extra_topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    extra_scales_ptr: OptionalReg[
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
    ] = None,
    # Effective max cache length.  Layout G structural eligibility uses
    # `num_heads * q_len <= BM_G(32)`.  Defaults to 0 so unrelated callers
    # (BF16, sparse, etc.) pass through to the BN_QK=64 branch unchanged.
    effective_max_cache_len: Int = 0,
) raises:
    comptime _scale_block_size = k_t.quantization_granularity if k_t.quantization_enabled else 0
    # Use native FP8 path when:
    # 1. KV is FP8 tensorwise (scale_block_size == 0)
    # 2. Q is also FP8 (q_type must match kv_type) — the pipeline provides FP8 Q
    # When Q is BF16, fall through to the old FP8 converter or BF16 path.
    comptime _native_fp8 = (
        k_t.dtype == DType.float8_e4m3fn
        and _scale_block_size == 0
        and q_type == DType.float8_e4m3fn
    )
    # Per-tensor rope-aware: split content (FP8 tensorwise) + rope (BF16) path
    comptime _per_token_scale_rope_aware = per_token_scale_rope_aware

    # For native FP8: Q is FP8 (1 byte) but swizzle_mode is the output
    # swizzle (SWIZZLE_128B for BF16). Using size_of[q_type]()=1 with
    # SWIZZLE_128B gives swizzle_elems=128, causing padded_q_depth=640
    # instead of 576. Use output_type size (2 for BF16) so
    # swizzle_elems=64 and padded dims are correct.
    comptime _dtype_size = size_of[output_type]() if _native_fp8 else size_of[
        q_type
    ]()
    comptime mla_config = MLA_SM100_Decode_Config(
        num_q_heads=num_heads,
        group=group,  # num_q_heads/h_k(1)
        depth=(depth - 64),  # 512
        q_depth=depth,  # 576
        dtype_size=_dtype_size,
        kv_type_size=size_of[k_t.dtype](),
        swizzle_mode=config.swizzle_mode,
        kv_mma_swizzle_mode=config.swizzle_mode,
        page_size=k_t.page_size,
        decoding_warp_split_k=decoding_warp_split_k,
        split_page_size=split_page_size,
        scale_block_size=_scale_block_size,
        native_fp8=_native_fp8,
        per_token_scale_rope_aware=_per_token_scale_rope_aware,
    )
    var num_rows_q = num_matrix_view_rows_decode(q)

    k_tma_op = k.create_tma_tile[
        BN=mla_config.BK_PV,  # tile_m =64
        depth=mla_config.q_depth,
        BK=mla_config.BK_QK,  # tile_n =576
        swizzle_mode=mla_config.kv_tma_swizzle_mode,
    ](ctx)
    o_ptr = rebind[UnsafePointer[Scalar[output_type], origin=MutAnyOrigin]](
        output.ptr
    )
    var num_rows_o = num_matrix_view_rows_decode(output)
    o_tma_op = tma_tile_qo[
        swizzle_mode=mla_config.swizzle_mode,
        BM=mla_config.out_rows,
        BK=mla_config.BN_PV // 4,
        depth=mla_config.depth,
    ](ctx, o_ptr, num_rows_o)

    # =========================================================================
    # Sparse routing: when sparse=True (comptime), use the gather4 sparse
    # kernel instead of the standard page-table path.
    # =========================================================================
    comptime if sparse:
        # Q TMA: BF16, BM x BK_QK(576), SWIZZLE_128B.
        q_ptr = rebind[UnsafePointer[Scalar[q_type], origin=MutAnyOrigin]](
            q.ptr
        )
        q_tma_sparse = tma_tile_qo[
            swizzle_mode=mla_config.swizzle_mode,
            BM=mla_config.BM,
            BK=mla_config.BK_QK,
            depth=mla_config.q_depth,
        ](ctx, q_ptr, num_rows_q)

        # Gate the all-FP8 KV sparse variant on the explicit caller flag.
        # `rope_aware_kv_sparse=True` (test-only) routes to the
        # BF16-rope sparse kernel; default `False` routes to the
        # all-FP8 path (single 576-byte gather4 TMA) here. This is a
        # comptime-only branch — production callers (MOGG/Python) get
        # the all-FP8 kernel transparently because they never set the
        # flag.
        comptime if not rope_aware_kv_sparse:
            # Single K gather4 TMA covering full 576-byte row
            # (INT64, SWIZZLE_NONE, tile_width=72 INT64 = 576 B).
            comptime _kv_tile_width = mla_config.padded_q_depth // 8
            k_gather4_tma = k.create_gather4_tma_tile[
                tile_width=_kv_tile_width,
                tile_stride=_kv_tile_width,
                swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
                tile_height=mla_config.BK_PV,
                tma_dtype=DType.int64,
                l2_promotion=TensorMapL2Promotion.L2_128B,
            ](ctx)

            var extra_k_val_fp8 = extra_k.or_else(k)
            extra_k_gather4_tma = extra_k_val_fp8.create_gather4_tma_tile[
                tile_width=_kv_tile_width,
                tile_stride=_kv_tile_width,
                swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
                tile_height=mla_config.BK_PV,
                tma_dtype=DType.int64,
                l2_promotion=TensorMapL2Promotion.L2_128B,
            ](ctx)
            var extra_kv_lut_val_fp8 = extra_k_val_fp8

            @parameter
            @always_inline
            def _launch_sparse_kv_fp8[
                _has_extra_kv: Bool, _has_variable_topk: Bool
            ]() raises:
                if ragged:
                    comptime ValidLengthType = NonNullPointer[DType.uint32]
                    var valid_len: ValidLengthType = {valid_length.ptr}
                    launch_mla_sm100_decode_sparse_kv_fp8[
                        q_type=q_type,
                        KVLUTType=k_t,
                        output_type=output_type,
                        SplitAccumType=SplitAccumType,
                        MaskType=mask_t,
                        config=mla_config,
                        ValidLengthType=ValidLengthType,
                        ragged=True,
                        _is_cache_length_accurate=_is_cache_length_accurate,
                        has_attn_sink=has_attn_sink,
                        has_extra_kv=_has_extra_kv,
                        has_variable_topk=_has_variable_topk,
                    ](
                        q_tma_sparse,
                        k_gather4_tma,
                        o_tma_op,
                        k,
                        lse_accum_split_ptr,
                        scale,
                        batch_size,
                        block_z,
                        num_partitions,
                        q_max_seq_len,
                        valid_len,
                        mask,
                        d_indices,
                        indices_stride,
                        topk_lengths,
                        scales_ptr,
                        attn_sink_ptr,
                        extra_k_gather4_tma,
                        extra_kv_lut_val_fp8,
                        extra_d_indices,
                        extra_topk_lengths,
                        extra_indices_stride,
                        extra_scales_ptr,
                        scalar_args_buf,
                        ctx,
                    )
                else:
                    comptime ValidLengthType = NullPointer[DType.uint32]
                    var valid_len: ValidLengthType = {}
                    launch_mla_sm100_decode_sparse_kv_fp8[
                        q_type=q_type,
                        KVLUTType=k_t,
                        output_type=output_type,
                        SplitAccumType=SplitAccumType,
                        MaskType=mask_t,
                        config=mla_config,
                        ValidLengthType=ValidLengthType,
                        ragged=False,
                        _is_cache_length_accurate=_is_cache_length_accurate,
                        has_attn_sink=has_attn_sink,
                        has_extra_kv=_has_extra_kv,
                        has_variable_topk=_has_variable_topk,
                    ](
                        q_tma_sparse,
                        k_gather4_tma,
                        o_tma_op,
                        k,
                        lse_accum_split_ptr,
                        scale,
                        batch_size,
                        block_z,
                        num_partitions,
                        q_max_seq_len,
                        valid_len,
                        mask,
                        d_indices,
                        indices_stride,
                        topk_lengths,
                        scales_ptr,
                        attn_sink_ptr,
                        extra_k_gather4_tma,
                        extra_kv_lut_val_fp8,
                        extra_d_indices,
                        extra_topk_lengths,
                        extra_indices_stride,
                        extra_scales_ptr,
                        scalar_args_buf,
                        ctx,
                    )

            _unswitch_raises[_launch_sparse_kv_fp8](
                extra_k is not None, Bool(topk_lengths)
            )
            return

        # K_nope gather4 TMA: INT64, SWIZZLE_NONE (linear SMEM layout).
        # tile_width = nope only (padded_depth / 8 = 64 INT64 elements).
        # tile_stride = full row (nope + rope) / 8 = 80 INT64 elements.
        comptime _nope_tile_width = mla_config.padded_depth // 8
        comptime _nope_tile_stride = (
            mla_config.padded_depth + mla_config.rope_depth * 2
        ) // 8
        k_nope_gather4_tma = k.create_gather4_tma_tile[
            tile_width=_nope_tile_width,
            tile_stride=_nope_tile_stride,
            swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
            tile_height=mla_config.BK_PV,
            tma_dtype=DType.int64,
            l2_promotion=TensorMapL2Promotion.L2_128B,
        ](ctx)

        # K_rope gather4 TMA: BF16, SWIZZLE_128B.
        # Row stride in BF16 elements = total_row_bytes / sizeof(bf16).
        comptime _rope_gather4_tile_width = (
            mla_config.padded_depth + mla_config.rope_depth * 2
        ) // 2
        k_rope_gather4_tma = k.create_rope_gather4_tma_tile[
            tile_width=_rope_gather4_tile_width,
            padded_depth=mla_config.padded_depth,
            swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
            tile_height=mla_config.BK_PV,
            l2_promotion=TensorMapL2Promotion.L2_128B,
        ](ctx)

        # Extra KV: create separate TMA descriptors from extra_k when provided.
        # When extra_k is None, we create dummy descriptors from k (they won't
        # be used since has_extra_kv=False eliminates all extra code paths).
        var extra_k_val = extra_k.or_else(k)
        extra_k_nope_gather4_tma = extra_k_val.create_gather4_tma_tile[
            tile_width=_nope_tile_width,
            tile_stride=_nope_tile_stride,
            swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
            tile_height=mla_config.BK_PV,
            tma_dtype=DType.int64,
            l2_promotion=TensorMapL2Promotion.L2_128B,
        ](ctx)
        extra_k_rope_gather4_tma = extra_k_val.create_rope_gather4_tma_tile[
            tile_width=_rope_gather4_tile_width,
            padded_depth=mla_config.padded_depth,
            swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
            tile_height=mla_config.BK_PV,
            l2_promotion=TensorMapL2Promotion.L2_128B,
        ](ctx)
        var extra_kv_lut_val = extra_k_val

        @parameter
        @always_inline
        def _launch_sparse[
            _has_extra_kv: Bool, _has_variable_topk: Bool
        ]() raises:
            if ragged:
                comptime ValidLengthType = NonNullPointer[DType.uint32]
                var valid_len: ValidLengthType = {valid_length.ptr}
                launch_mla_sm100_decode_sparse[
                    q_type=q_type,
                    KVLUTType=k_t,
                    output_type=output_type,
                    SplitAccumType=SplitAccumType,
                    MaskType=mask_t,
                    config=mla_config,
                    ValidLengthType=ValidLengthType,
                    ragged=True,
                    _is_cache_length_accurate=_is_cache_length_accurate,
                    has_attn_sink=has_attn_sink,
                    has_extra_kv=_has_extra_kv,
                    has_variable_topk=_has_variable_topk,
                ](
                    q_tma_sparse,
                    k_nope_gather4_tma,
                    k_rope_gather4_tma,
                    o_tma_op,
                    k,
                    lse_accum_split_ptr,
                    scale,
                    batch_size,
                    block_z,
                    num_partitions,
                    q_max_seq_len,
                    valid_len,
                    mask,
                    d_indices,
                    indices_stride,
                    topk_lengths,
                    scales_ptr,
                    attn_sink_ptr,
                    extra_k_nope_gather4_tma,
                    extra_k_rope_gather4_tma,
                    extra_kv_lut_val,
                    extra_d_indices,
                    extra_topk_lengths,
                    extra_indices_stride,
                    extra_scales_ptr,
                    scalar_args_buf,
                    ctx,
                )
            else:
                comptime ValidLengthType = NullPointer[DType.uint32]
                var valid_len: ValidLengthType = {}
                launch_mla_sm100_decode_sparse[
                    q_type=q_type,
                    KVLUTType=k_t,
                    output_type=output_type,
                    SplitAccumType=SplitAccumType,
                    MaskType=mask_t,
                    config=mla_config,
                    ValidLengthType=ValidLengthType,
                    ragged=False,
                    _is_cache_length_accurate=_is_cache_length_accurate,
                    has_attn_sink=has_attn_sink,
                    has_extra_kv=_has_extra_kv,
                    has_variable_topk=_has_variable_topk,
                ](
                    q_tma_sparse,
                    k_nope_gather4_tma,
                    k_rope_gather4_tma,
                    o_tma_op,
                    k,
                    lse_accum_split_ptr,
                    scale,
                    batch_size,
                    block_z,
                    num_partitions,
                    q_max_seq_len,
                    valid_len,
                    mask,
                    d_indices,
                    indices_stride,
                    topk_lengths,
                    scales_ptr,
                    attn_sink_ptr,
                    extra_k_nope_gather4_tma,
                    extra_k_rope_gather4_tma,
                    extra_kv_lut_val,
                    extra_d_indices,
                    extra_topk_lengths,
                    extra_indices_stride,
                    extra_scales_ptr,
                    scalar_args_buf,
                    ctx,
                )

        _unswitch_raises[_launch_sparse](
            extra_k is not None, Bool(topk_lengths)
        )
        return

    # Per-token-scale rope-aware: split content (FP8) + rope (BF16) with separate TMAs.
    # Q buffer layout: FP8 content (512 bytes) | BF16 rope (128 bytes) per row = 640 bytes/row.
    # K cache layout: FP8 content (512 bytes) | BF16 rope (128 bytes) per row = 640 bytes/row.
    # The KV cache 640 bytes/row layout is enforced by create_rope_tma_tile in kv_cache/types.mojo.
    comptime if _per_token_scale_rope_aware:
        # Q row stride in FP8 bytes: 512 FP8 content + 64 BF16 rope = 640 bytes.
        # The `depth` parameter in tma_tile_qo sets the row stride of the
        # LayoutTensor, which the TMA descriptor uses as the global memory
        # stride.  It must equal the full row width so that consecutive
        # rows (heads/tokens) are read correctly.
        comptime _q_row_bytes = mla_config.padded_depth + mla_config.rope_depth * 2  # 640
        # Same stride in BF16 units for the rope TMA.
        comptime _q_row_bf16 = _q_row_bytes // 2  # 320

        # Q_nope TMA: FP8 content, SWIZZLE_64B, BM x padded_depth (512)
        q_ptr_fp8_content = rebind[
            UnsafePointer[Scalar[DType.float8_e4m3fn], origin=MutAnyOrigin]
        ](q.ptr)
        q_nope_tma = tma_tile_qo[
            swizzle_mode=mla_config.content_swizzle_mode,  # SWIZZLE_64B
            BM=mla_config.BM,
            BK=mla_config.padded_depth,  # 512
            depth=_q_row_bytes,  # 640 (full row stride in FP8 bytes)
        ](ctx, q_ptr_fp8_content, num_rows_q)

        # Q_rope TMA: BF16 rope, SWIZZLE_128B, BM x rope_depth (64)
        # Rope starts at byte offset padded_depth (512) from Q row start.
        q_ptr_bf16_rope = rebind[
            UnsafePointer[Scalar[DType.bfloat16], origin=MutAnyOrigin]
        ](q.ptr + mla_config.padded_depth)
        q_rope_tma = tma_tile_qo[
            swizzle_mode=mla_config.rope_swizzle_mode,  # SWIZZLE_128B
            BM=mla_config.BM,
            BK=mla_config.rope_depth,  # 64
            depth=_q_row_bf16,  # 320 (full row stride in BF16 elements)
        ](ctx, q_ptr_bf16_rope, num_rows_q)

        # K_content TMA: FP8 content from KV cache, SWIZZLE_64B, BK_PV x padded_depth (512).
        # The KV cache has 640 bytes/row layout (512 FP8 content + 128 BF16 rope).
        # create_tma_tile reads only the first 512 bytes (FP8 content) per row.
        k_content_tma = k.create_tma_tile[
            BN=mla_config.BK_PV,  # 64
            depth=mla_config.padded_depth,  # 512
            BK=mla_config.padded_depth,  # 512
            swizzle_mode=mla_config.content_swizzle_mode,  # SWIZZLE_64B
        ](ctx)

        # K_rope TMA: BF16 rope from KV cache, SWIZZLE_128B, BK_PV x rope_depth (64).
        # The KV cache row layout is padded_depth FP8 bytes followed by
        # rope_depth BF16 elements.  create_rope_tma_tile offsets the base
        # pointer by padded_depth bytes and reinterprets as BF16.
        k_rope_tma = k.create_rope_tma_tile[
            BN=mla_config.BK_PV,  # 64
            BK=mla_config.rope_depth,  # 64
            padded_depth=mla_config.padded_depth,  # 512
            swizzle_mode=mla_config.rope_swizzle_mode,  # SWIZZLE_128B
        ](ctx)

        # Scales TMA: per-token float32 scales loaded via TMA.
        # The scales tensor is [total_blocks, page_size, 1, 1] in row-major,
        # indexed by row_idx (same paging as KV cache blocks).
        # We treat it as a flat [1, total_elements] 2D tensor for TMA.
        var _total_scale_elements = k.num_kv_rows()
        scale_tma = tma_tile_scales[BN_QK=mla_config.BN_QK](
            ctx, scales_ptr, _total_scale_elements
        )

        if ragged:
            comptime ValidLengthType = NonNullPointer[DType.uint32]
            var valid_len: ValidLengthType = {valid_length.ptr}
            launch_mla_sm100_decode_fp8_per_token_scale_rope_aware[
                q_type=q_type,
                KVLUTType=k_t,
                output_type=output_type,
                SplitAccumType=SplitAccumType,
                MaskType=mask_t,
                config=mla_config,
                ValidLengthType=ValidLengthType,
                ragged=True,
                _is_cache_length_accurate=_is_cache_length_accurate,
                has_per_token_scales=True,
            ](
                q_nope_tma,
                q_rope_tma,
                k_content_tma,
                k_rope_tma,
                scale_tma,
                o_tma_op,
                k,
                lse_accum_split_ptr,
                scale,
                batch_size,
                block_z,
                num_partitions,
                q_max_seq_len,
                valid_len,
                mask,
                q_scale_ptr,
                scalar_args_buf,
                ctx,
            )
        else:
            comptime ValidLengthType = NullPointer[DType.uint32]
            var valid_len: ValidLengthType = {}
            launch_mla_sm100_decode_fp8_per_token_scale_rope_aware[
                q_type=q_type,
                KVLUTType=k_t,
                output_type=output_type,
                SplitAccumType=SplitAccumType,
                MaskType=mask_t,
                config=mla_config,
                ValidLengthType=ValidLengthType,
                ragged=False,
                _is_cache_length_accurate=_is_cache_length_accurate,
                has_per_token_scales=True,
            ](
                q_nope_tma,
                q_rope_tma,
                k_content_tma,
                k_rope_tma,
                scale_tma,
                o_tma_op,
                k,
                lse_accum_split_ptr,
                scale,
                batch_size,
                block_z,
                num_partitions,
                q_max_seq_len,
                valid_len,
                mask,
                q_scale_ptr,
                scalar_args_buf,
                ctx,
            )
    elif _native_fp8:
        q_ptr_fp8 = rebind[
            UnsafePointer[Scalar[k_t.dtype], origin=MutAnyOrigin]
        ](q.ptr)
        q_tma_fp8 = tma_tile_qo[
            swizzle_mode=mla_config.kv_tma_swizzle_mode,  # SWIZZLE_64B
            BM=mla_config.BM,
            BK=mla_config.BK_QK,
            depth=mla_config.q_depth,
        ](ctx, q_ptr_fp8, num_rows_q)

        # Layout G config + BM=32 Q TMA tile. Built unconditionally so the
        # comptime branch inside `_launch_r` / `_launch_n` can reference
        # them; when the runtime gate misses, the unused branch is pruned.
        comptime mla_config_g = MLA_SM100_Decode_Config(
            num_q_heads=mla_config.num_q_heads,
            group=mla_config.group,
            depth=mla_config.depth,
            q_depth=mla_config.q_depth,
            dtype_size=mla_config.dtype_size,
            kv_type_size=size_of[k_t.dtype](),
            swizzle_mode=mla_config.swizzle_mode,
            kv_mma_swizzle_mode=mla_config.kv_mma_swizzle_mode,
            page_size=mla_config.page_size,
            decoding_warp_split_k=mla_config.decoding_warp_split_k,
            split_page_size=mla_config.split_page_size,
            scale_block_size=mla_config.scale_block_size,
            native_fp8=True,
            per_token_scale_rope_aware=mla_config.per_token_scale_rope_aware,
            decode_layout_g=True,
        )
        q_tma_fp8_g = tma_tile_qo[
            swizzle_mode=mla_config_g.kv_tma_swizzle_mode,  # SWIZZLE_64B
            BM=mla_config_g.BM,  # 32
            BK=mla_config_g.BK_QK,
            depth=mla_config_g.q_depth,
        ](ctx, q_ptr_fp8, num_rows_q)

        # Dispatch is routed at comptime by num_q_heads × q_max_seq_len.
        # q_max_seq_len > 1 implies spec decoding (1 actual + N spec ahead),
        # capped at MAX_FOLD_Q (1 actual + (MAX_FOLD_Q - 1) spec). For each q
        # in {MIN_FOLD_Q..MAX_FOLD_Q}: num_heads × q ≤ 32 → Layout-G fold
        # (BM=32), ≤ 64 → Layout-E fold (BM=64), > 64 → non-fold (kernel
        # handles q in grid dim). For q=1 (regular decode), num_heads ≤ 32 →
        # Layout-G non-fold, else → Layout-E non-fold.

        if ragged:
            comptime ValidLengthType = NonNullPointer[DType.uint32]
            var valid_len: ValidLengthType = {valid_length.ptr}

            @parameter
            @always_inline
            def _launch_r[
                _fold_q: Bool,
                _q_len_fold: Int,
                _layout_g: Bool = False,
            ]() raises:
                comptime if _layout_g:
                    launch_mla_sm100_decode_native_fp8_layout_g[
                        q_type=q_type,
                        KVLUTType=k_t,
                        output_type=output_type,
                        SplitAccumType=SplitAccumType,
                        MaskType=mask_t,
                        config_e=mla_config,
                        config_g=mla_config_g,
                        ValidLengthType=ValidLengthType,
                        ragged=True,
                        _is_cache_length_accurate=_is_cache_length_accurate,
                        fold_q=_fold_q,
                        q_len_fold=_q_len_fold,
                    ](
                        q_tma_fp8_g,
                        k_tma_op,
                        o_tma_op,
                        k,
                        lse_accum_split_ptr,
                        scale,
                        batch_size,
                        block_z,
                        num_partitions,
                        q_max_seq_len,
                        valid_len,
                        mask,
                        scales_ptr,
                        scalar_args_buf,
                        ctx,
                    )
                else:
                    launch_mla_sm100_decode_native_fp8[
                        q_type=q_type,
                        KVLUTType=k_t,
                        output_type=output_type,
                        SplitAccumType=SplitAccumType,
                        MaskType=mask_t,
                        config=mla_config,
                        ValidLengthType=ValidLengthType,
                        ragged=True,
                        _is_cache_length_accurate=_is_cache_length_accurate,
                        fold_q=_fold_q,
                        q_len_fold=_q_len_fold,
                    ](
                        q_tma_fp8,
                        k_tma_op,
                        o_tma_op,
                        k,
                        lse_accum_split_ptr,
                        scale,
                        batch_size,
                        block_z,
                        num_partitions,
                        q_max_seq_len,
                        valid_len,
                        mask,
                        scales_ptr,
                        scalar_args_buf,
                        ctx,
                    )

            # Spec decoding implied by q > 1; cap at MAX_FOLD_Q (1 actual +
            # (MAX_FOLD_Q - 1) spec).
            comptime for n in range(MIN_FOLD_Q, MAX_FOLD_Q + 1):
                comptime if mla_config.num_q_heads * n <= 32:
                    if q_max_seq_len == n:
                        _launch_r[True, n, True]()  # Layout-G fold (BM=32)
                        return
                elif mla_config.num_q_heads * n <= 64:
                    if q_max_seq_len == n:
                        _launch_r[True, n, False]()  # Layout-E fold (BM=64)
                        return
                else:
                    # num_heads * n > 64: kernel can't fold; fall through to
                    # non-fold (kernel handles q in grid dim).
                    if q_max_seq_len == n:
                        _launch_r[False, 1, False]()  # non-fold
                        return

            # q_max_seq_len == 1 (regular decode) or > MAX_FOLD_Q (shouldn't
            # happen, fallback).
            comptime if mla_config.num_q_heads <= 32:
                _launch_r[False, 1, True]()  # Layout-G non-fold
            else:
                _launch_r[False, 1, False]()  # Layout-E non-fold
            return
        else:
            comptime ValidLengthType = NullPointer[DType.uint32]
            var valid_len: ValidLengthType = {}

            @parameter
            @always_inline
            def _launch_n[
                _fold_q: Bool,
                _q_len_fold: Int,
                _layout_g: Bool = False,
            ]() raises:
                comptime if _layout_g:
                    launch_mla_sm100_decode_native_fp8_layout_g[
                        q_type=q_type,
                        KVLUTType=k_t,
                        output_type=output_type,
                        SplitAccumType=SplitAccumType,
                        MaskType=mask_t,
                        config_e=mla_config,
                        config_g=mla_config_g,
                        ValidLengthType=ValidLengthType,
                        ragged=False,
                        _is_cache_length_accurate=_is_cache_length_accurate,
                        fold_q=_fold_q,
                        q_len_fold=_q_len_fold,
                    ](
                        q_tma_fp8_g,
                        k_tma_op,
                        o_tma_op,
                        k,
                        lse_accum_split_ptr,
                        scale,
                        batch_size,
                        block_z,
                        num_partitions,
                        q_max_seq_len,
                        valid_len,
                        mask,
                        scales_ptr,
                        scalar_args_buf,
                        ctx,
                    )
                else:
                    launch_mla_sm100_decode_native_fp8[
                        q_type=q_type,
                        KVLUTType=k_t,
                        output_type=output_type,
                        SplitAccumType=SplitAccumType,
                        MaskType=mask_t,
                        config=mla_config,
                        ValidLengthType=ValidLengthType,
                        ragged=False,
                        _is_cache_length_accurate=_is_cache_length_accurate,
                        fold_q=_fold_q,
                        q_len_fold=_q_len_fold,
                    ](
                        q_tma_fp8,
                        k_tma_op,
                        o_tma_op,
                        k,
                        lse_accum_split_ptr,
                        scale,
                        batch_size,
                        block_z,
                        num_partitions,
                        q_max_seq_len,
                        valid_len,
                        mask,
                        scales_ptr,
                        scalar_args_buf,
                        ctx,
                    )

            # Spec decoding implied by q > 1; cap at MAX_FOLD_Q (1 actual +
            # (MAX_FOLD_Q - 1) spec).
            comptime for n in range(MIN_FOLD_Q, MAX_FOLD_Q + 1):
                comptime if mla_config.num_q_heads * n <= 32:
                    if q_max_seq_len == n:
                        _launch_n[True, n, True]()  # Layout-G fold (BM=32)
                        return
                elif mla_config.num_q_heads * n <= 64:
                    if q_max_seq_len == n:
                        _launch_n[True, n, False]()  # Layout-E fold (BM=64)
                        return
                else:
                    # num_heads * n > 64: kernel can't fold; fall through to
                    # non-fold (kernel handles q in grid dim).
                    if q_max_seq_len == n:
                        _launch_n[False, 1, False]()  # non-fold
                        return

            # q_max_seq_len == 1 (regular decode) or > MAX_FOLD_Q (shouldn't
            # happen, fallback).
            comptime if mla_config.num_q_heads <= 32:
                _launch_n[False, 1, True]()  # Layout-G non-fold
            else:
                _launch_n[False, 1, False]()  # Layout-E non-fold
            return
    else:
        # BF16 / old FP8 converter path: Q is BF16, create BF16 Q TMA.
        q_ptr = rebind[UnsafePointer[Scalar[q_type], origin=MutAnyOrigin]](
            q.ptr
        )
        q_tma_op = tma_tile_qo[
            swizzle_mode=mla_config.swizzle_mode,
            BM=mla_config.BM,
            BK=mla_config.BK_QK,
            depth=mla_config.q_depth,
        ](ctx, q_ptr, num_rows_q)

        if ragged:
            comptime ValidLengthType = NonNullPointer[DType.uint32]
            var valid_len: ValidLengthType = {valid_length.ptr}
            launch_mla_sm100_decode_enqueue_kernel[
                q_type=q_type,
                KVLUTType=k_t,
                output_type=output_type,
                SplitAccumType=SplitAccumType,
                MaskType=mask_t,
                config=mla_config,
                ValidLengthType=ValidLengthType,
                ragged=True,
                _is_cache_length_accurate=_is_cache_length_accurate,
            ](
                q_tma_op,
                k_tma_op,
                o_tma_op,
                k,
                lse_accum_split_ptr,
                scale,
                batch_size,
                block_z,
                num_partitions,
                q_max_seq_len,
                valid_len,
                mask,
                scales_ptr,
                scalar_args_buf,
                ctx,
            )
        else:
            comptime ValidLengthType = NullPointer[DType.uint32]
            var valid_len: ValidLengthType = {}
            launch_mla_sm100_decode_enqueue_kernel[
                q_type=q_type,
                KVLUTType=k_t,
                output_type=output_type,
                SplitAccumType=SplitAccumType,
                MaskType=mask_t,
                config=mla_config,
                ValidLengthType=ValidLengthType,
                ragged=False,
                _is_cache_length_accurate=_is_cache_length_accurate,
            ](
                q_tma_op,
                k_tma_op,
                o_tma_op,
                k,
                lse_accum_split_ptr,
                scale,
                batch_size,
                block_z,
                num_partitions,
                q_max_seq_len,
                valid_len,
                mask,
                scales_ptr,
                scalar_args_buf,
                ctx,
            )


@always_inline
def launch_mla_sm100_decode_enqueue_kernel[
    q_type: DType,
    KVLUTType: MHAOperand,
    output_type: DType,
    SplitAccumType: OptionalPointer,
    MaskType: MHAMask,
    config: MLA_SM100_Decode_Config,
    ValidLengthType: OptionalPointer,
    _is_cache_length_accurate: Bool = False,
    ragged: Bool = False,
](
    q_tma: QOTMATile[
        dtype=q_type,
        BM=config.BM,  # tile_m =64
        BK=config.BK_QK,  # tile_n =576
        swizzle_mode=config.swizzle_mode,
    ],
    k_tma: KVTMATile[
        dtype=KVLUTType.dtype,
        swizzle_mode=config.kv_tma_swizzle_mode,
        BN=config.BK_PV,  # tile_m =64
        BK=config.BK_QK,  # tile_n =576
    ],
    o_tma: QOTMATile[
        dtype=output_type,
        BM=config.out_rows,
        BK=config.BN_PV // 4,
        swizzle_mode=config.swizzle_mode,
    ],
    kv_lut: KVLUTType,
    lse_accum_split_ptr: SplitAccumType,
    scale: Float32,
    batch_size: Int,
    block_z: Int,
    num_partitions: Int,
    q_max_seq_len: Int,
    valid_len: ValidLengthType,
    mask: MaskType,
    scales_ptr: UnsafePointer[Scalar[DType.float32], origin=MutAnyOrigin],
    scalar_args_buf: TileTensor[
        DType.int64, address_space=AddressSpace.GENERIC, ...
    ],
    ctx: DeviceContext,
) raises:
    var mla_decode_pack = MLA_Decode_Pack[
        ValidLengthType=ValidLengthType,
        MaskType=MaskType,
        SplitAccumType=SplitAccumType,
    ](mask, valid_len, lse_accum_split_ptr)

    var block_x = ceildiv(config.num_q_heads, config.BM)
    var grid_dim = (block_x, q_max_seq_len, block_z)
    # bf16: 3 warp groups; fp8: 4 warp groups (adds fp8-to-bf16 convert WG)
    # - one for load/store/2xMMA
    # - one for compute softmax
    # - one for compute correction
    # - (fp8 only) one for fp8-to-bf16 conversion
    var block_dim = (config.num_threads, 1, 1)
    logger.info(
        "block_dim:",
        block_dim[0],
        block_dim[1],
        block_dim[2],
        "grid_dim:",
        grid_dim[0],
        grid_dim[1],
        grid_dim[2],
        "config.smem_used:",
        config.smem_used,
        "config.num_q_heads:",
        config.num_q_heads,
        "config.num_kv_heads:",
        config.num_kv_heads,
        "config.num_threads:",
        config.num_threads,
        "config.num_kv_stages:",
        config.num_kv_stages,
        "config.BM:",
        config.BM,
        "config.BN_QK:",
        config.BN_QK,
        "config.BK_QK:",
        config.BK_QK,
        "config.BK_PV:",
        config.BK_PV,
        "config.q_depth:",
        config.q_depth,
        "config.depth:",
        config.depth,
        "config.padded_depth:",
        config.padded_depth,
        "config.padded_q_depth:",
        config.padded_q_depth,
        "config.rope_depth:",
        config.rope_depth,
        "config.swizzle_mode:",
        config.swizzle_mode,
        "output_tile_width:",
        (config.BN_PV // 8) * (4 // size_of[output_type]()),
    )

    logger.info("------ Dispatching to SM100 MLA-SM100-DECODE ------")
    logger.info(
        "QK Type:",
        KVLUTType.dtype,
        "Q Depth:",
        config.q_depth,
        "Number of Q // KV Heads:",
        config.num_q_heads,
        "//",
        config.num_kv_heads,
        "Batch Size:",
        block_z,
        "Num Partitions:",
        num_partitions,
    )

    # Dispatch to BF16 or old FP8 converter kernel (not native FP8 — that has
    # its own launch function with FP8 Q TMA).
    # Route ALL FP8 KV (both tensorwise and blockwise) to the FP8 converter
    # kernel. When we reach this function, native FP8 has already been ruled
    # out (Q is BF16), so the converter kernel handles FP8->BF16 conversion.
    comptime _is_old_fp8 = KVLUTType.dtype == DType.float8_e4m3fn
    comptime kernel = MLA_SM100_Decode_KV_FP8[
        q_type=q_type,
        KVLUTType=KVLUTType,
        output_type=output_type,
        SplitAccumType=SplitAccumType,
        MaskType=MaskType,
        config=config,
        ValidLengthType=ValidLengthType,
        _is_cache_length_accurate=_is_cache_length_accurate,
        ragged=ragged,
    ].kernel if _is_old_fp8 else MLA_SM100_Decode_KV_BF16[
        q_type=q_type,
        KVLUTType=KVLUTType,
        output_type=output_type,
        SplitAccumType=SplitAccumType,
        MaskType=MaskType,
        config=config,
        ValidLengthType=ValidLengthType,
        _is_cache_length_accurate=_is_cache_length_accurate,
        ragged=ragged,
    ].kernel
    # Enable PDL (Programmatic Dependent Launch) for split-K mode to chain
    # the MLA decode kernel with the combine kernel, reducing host synchronization.
    comptime pdl_level = PDLLevel.OVERLAP_AT_END if config.decoding_warp_split_k else PDLLevel.OFF

    ctx.enqueue_function[kernel](
        q_tma,
        k_tma,
        o_tma,
        kv_lut,
        scale,
        mla_decode_pack,
        scales_ptr,
        scalar_args_buf,
        grid_dim=grid_dim,
        block_dim=block_dim,
        shared_mem_bytes=config.smem_used,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(config.smem_used)
        ),
        attributes=pdl_launch_attributes(pdl_level),
    )


@always_inline
def launch_mla_sm100_decode_native_fp8[
    q_type: DType,
    KVLUTType: MHAOperand,
    output_type: DType,
    SplitAccumType: OptionalPointer,
    MaskType: MHAMask,
    config: MLA_SM100_Decode_Config,
    ValidLengthType: OptionalPointer,
    _is_cache_length_accurate: Bool = False,
    ragged: Bool = False,
    # when True, the kernel packs
    # q_len_fold q_tokens into the BM=64 M tile and grid.y collapses to 1.
    fold_q: Bool = False,
    # comptime number of q_tokens to fold.  Must satisfy
    # `num_q_heads * q_len_fold <= BM` (caller-enforced) and `q_len_fold > 1`.
    q_len_fold: Int = 1,
](
    q_tma: QOTMATile[
        dtype=KVLUTType.dtype,  # FP8 Q TMA
        BM=config.BM,
        BK=config.BK_QK,
        swizzle_mode=config.kv_tma_swizzle_mode,  # SWIZZLE_64B
    ],
    k_tma: KVTMATile[
        dtype=KVLUTType.dtype,
        swizzle_mode=config.kv_tma_swizzle_mode,
        BN=config.BK_PV,
        BK=config.BK_QK,
    ],
    o_tma: QOTMATile[
        dtype=output_type,
        BM=config.out_rows,
        BK=config.BN_PV // 4,
        swizzle_mode=config.swizzle_mode,
    ],
    kv_lut: KVLUTType,
    lse_accum_split_ptr: SplitAccumType,
    scale: Float32,
    batch_size: Int,
    block_z: Int,
    num_partitions: Int,
    q_max_seq_len: Int,
    valid_len: ValidLengthType,
    mask: MaskType,
    scales_ptr: UnsafePointer[Scalar[DType.float32], origin=MutAnyOrigin],
    scalar_args_buf: TileTensor[
        DType.int64, address_space=AddressSpace.GENERIC, ...
    ],
    ctx: DeviceContext,
) raises:
    """Launch the native FP8 MLA decode kernel with FP8 Q TMA.

    This is a dedicated launch function for the native FP8 path because
    the Q TMA has FP8 dtype (SWIZZLE_64B) instead of BF16 (SWIZZLE_128B).

    Under `fold_q=True`, BM=64 packs `q_len_fold * num_q_heads` M-rows of
    the same batch element and grid.y collapses to 1 (all q_tokens live in
    one CTA).  This avoids spawning `q_max_seq_len` CTAs and lets the
    softmax/correction WGs amortize the QK setup.
    """
    var mla_decode_pack = MLA_Decode_Pack[
        ValidLengthType=ValidLengthType,
        MaskType=MaskType,
        SplitAccumType=SplitAccumType,
    ](mask, valid_len, lse_accum_split_ptr)
    var block_x = ceildiv(config.num_q_heads, config.BM)
    # fold collapses grid.y to 1 since BM packs all q_tokens.
    var grid_y = 1 if fold_q else q_max_seq_len
    var grid_dim = (block_x, grid_y, block_z)
    var block_dim = (config.num_threads, 1, 1)

    logger.info("------ Dispatching to SM100 Native FP8 MLA-DECODE ------")

    comptime kernel = MLA_SM100_Decode_QKV_FP8[
        q_type=q_type,
        KVLUTType=KVLUTType,
        output_type=output_type,
        SplitAccumType=SplitAccumType,
        MaskType=MaskType,
        config=config,
        ValidLengthType=ValidLengthType,
        _is_cache_length_accurate=_is_cache_length_accurate,
        ragged=ragged,
        fold_q=fold_q,
        q_len_fold=q_len_fold,
    ].kernel
    comptime pdl_level = PDLLevel.OVERLAP_AT_END if config.decoding_warp_split_k else PDLLevel.OFF
    ctx.enqueue_function[kernel](
        q_tma,
        k_tma,
        o_tma,
        kv_lut,
        scale,
        mla_decode_pack,
        scales_ptr,
        scalar_args_buf,
        grid_dim=grid_dim,
        block_dim=block_dim,
        shared_mem_bytes=config.smem_used,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(config.smem_used)
        ),
        attributes=pdl_launch_attributes(pdl_level),
    )


# Layout G launcher for the qkv_fp8 native-FP8 kernel (BM=32, MMA_M=32,
# 5-stage). Takes two configs because k_tma / o_tma are identical between
# Layout E and Layout G — `config_e` types those, while `config_g`
# (`decode_layout_g=True`) types the kernel struct and BM=32 Q TMA tile.
@always_inline
def launch_mla_sm100_decode_native_fp8_layout_g[
    q_type: DType,
    KVLUTType: MHAOperand,
    output_type: DType,
    SplitAccumType: OptionalPointer,
    MaskType: MHAMask,
    config_e: MLA_SM100_Decode_Config,
    config_g: MLA_SM100_Decode_Config,
    ValidLengthType: OptionalPointer,
    _is_cache_length_accurate: Bool = False,
    ragged: Bool = False,
    fold_q: Bool = False,
    q_len_fold: Int = 1,
](
    q_tma: QOTMATile[
        dtype=KVLUTType.dtype,
        BM=config_g.BM,
        BK=config_g.BK_QK,
        swizzle_mode=config_g.kv_tma_swizzle_mode,  # SWIZZLE_64B
    ],
    k_tma: KVTMATile[
        dtype=KVLUTType.dtype,
        swizzle_mode=config_e.kv_tma_swizzle_mode,
        BN=config_e.BK_PV,
        BK=config_e.BK_QK,
    ],
    o_tma: QOTMATile[
        dtype=output_type,
        BM=config_e.out_rows,
        BK=config_e.BN_PV // 4,
        swizzle_mode=config_e.swizzle_mode,
    ],
    kv_lut: KVLUTType,
    lse_accum_split_ptr: SplitAccumType,
    scale: Float32,
    batch_size: Int,
    block_z: Int,
    num_partitions: Int,
    q_max_seq_len: Int,
    valid_len: ValidLengthType,
    mask: MaskType,
    scales_ptr: UnsafePointer[Scalar[DType.float32], origin=MutAnyOrigin],
    scalar_args_buf: TileTensor[
        DType.int64, address_space=AddressSpace.GENERIC, ...
    ],
    ctx: DeviceContext,
) raises:
    """Launch the Layout G native FP8 MLA decode kernel (BM=32, 5-stage)."""
    comptime assert config_g.decode_layout_g, (
        "launch_mla_sm100_decode_native_fp8_layout_g requires"
        " config_g.decode_layout_g==True"
    )
    # Layout E and Layout G must agree on BN_QK/BK_QK/BK_PV/swizzle so the shared
    # k_tma / o_tma tiles flow through unchanged.
    comptime assert (
        config_e.BN_QK == config_g.BN_QK
    ), "Layout E/G BN_QK mismatch (k_tma reuse invariant)"
    comptime assert (
        config_e.BK_QK == config_g.BK_QK
    ), "Layout E/G BK_QK mismatch (k_tma reuse invariant)"
    comptime assert (
        config_e.BK_PV == config_g.BK_PV
    ), "Layout E/G BK_PV mismatch (k_tma reuse invariant)"
    var mla_decode_pack = MLA_Decode_Pack[
        ValidLengthType=ValidLengthType,
        MaskType=MaskType,
        SplitAccumType=SplitAccumType,
    ](mask, valid_len, lse_accum_split_ptr)
    var block_x = ceildiv(config_g.num_q_heads, config_g.BM)
    # fold collapses grid.y to 1 since BM packs all q_tokens.
    var grid_y = 1 if fold_q else q_max_seq_len
    var grid_dim = (block_x, grid_y, block_z)
    var block_dim = (config_g.num_threads, 1, 1)

    logger.info(
        "------ Dispatching to SM100 Native FP8 MLA-DECODE Layout G ------"
    )

    comptime kernel = MLA_SM100_Decode_QKV_FP8_Layout_G[
        q_type=q_type,
        KVLUTType=KVLUTType,
        output_type=output_type,
        SplitAccumType=SplitAccumType,
        MaskType=MaskType,
        config=config_g,
        ValidLengthType=ValidLengthType,
        _is_cache_length_accurate=_is_cache_length_accurate,
        ragged=ragged,
        fold_q=fold_q,
        q_len_fold=q_len_fold,
    ].kernel
    comptime pdl_level = PDLLevel.OVERLAP_AT_END if config_g.decoding_warp_split_k else PDLLevel.OFF
    ctx.enqueue_function[kernel](
        q_tma,
        k_tma,
        o_tma,
        kv_lut,
        scale,
        mla_decode_pack,
        scales_ptr,
        scalar_args_buf,
        grid_dim=grid_dim,
        block_dim=block_dim,
        shared_mem_bytes=config_g.smem_used,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(config_g.smem_used)
        ),
        attributes=pdl_launch_attributes(pdl_level),
    )


@always_inline
def launch_mla_sm100_decode_fp8_per_token_scale_rope_aware[
    q_type: DType,
    KVLUTType: MHAOperand,
    output_type: DType,
    SplitAccumType: OptionalPointer,
    MaskType: MHAMask,
    config: MLA_SM100_Decode_Config,
    ValidLengthType: OptionalPointer,
    _is_cache_length_accurate: Bool = False,
    ragged: Bool = False,
    has_per_token_scales: Bool = False,
](
    q_nope_tma: QOTMATile[
        dtype=DType.float8_e4m3fn,
        BM=config.BM,
        BK=config.padded_depth,  # 512
        swizzle_mode=config.content_swizzle_mode,  # SWIZZLE_64B
    ],
    q_rope_tma: QOTMATile[
        dtype=DType.bfloat16,
        BM=config.BM,
        BK=config.rope_depth,  # 64
        swizzle_mode=config.rope_swizzle_mode,  # SWIZZLE_128B
    ],
    k_content_tma: KVTMATile[
        dtype=KVLUTType.dtype,
        swizzle_mode=config.content_swizzle_mode,  # SWIZZLE_64B
        BN=config.BK_PV,  # 64
        BK=config.padded_depth,  # 512
    ],
    k_rope_tma: KVTMATile[
        dtype=DType.bfloat16,
        swizzle_mode=config.rope_swizzle_mode,  # SWIZZLE_128B
        BN=config.BK_PV,  # 64
        BK=config.rope_depth,  # 64
    ],
    scale_tma: ScalesTMATile[BN_QK=config.BN_QK],
    o_tma: QOTMATile[
        dtype=output_type,
        BM=config.out_rows,
        BK=config.BN_PV // 4,
        swizzle_mode=config.swizzle_mode,
    ],
    kv_lut: KVLUTType,
    lse_accum_split_ptr: SplitAccumType,
    scale: Float32,
    batch_size: Int,
    block_z: Int,
    num_partitions: Int,
    q_max_seq_len: Int,
    valid_len: ValidLengthType,
    mask: MaskType,
    q_scale_ptr: OptionalReg[
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
    ],
    scalar_args_buf: TileTensor[
        DType.int64, address_space=AddressSpace.GENERIC, ...
    ],
    ctx: DeviceContext,
) raises:
    """Launch the FP8 per-token-scale rope-aware MLA decode kernel with split content/rope TMAs.

    This is a dedicated launch function for the SnapMLA FP8 per-token-scale rope-aware path.
    Q and K are split into FP8 content (512 dims, SWIZZLE_64B) and BF16 rope
    (64 dims, SWIZZLE_128B), requiring 5 TMA descriptors (content, rope, scales, Q_nope, Q_rope).
    Per-token scales are loaded via TMA alongside content and rope.
    """
    var mla_decode_pack = MLA_Decode_Pack[
        ValidLengthType=ValidLengthType,
        MaskType=MaskType,
        SplitAccumType=SplitAccumType,
    ](mask, valid_len, lse_accum_split_ptr)
    var block_x = ceildiv(config.num_q_heads, config.BM)
    var grid_dim = (block_x, q_max_seq_len, block_z)
    var block_dim = (config.num_threads, 1, 1)

    logger.info(
        "------ Dispatching to SM100 FP8 PerTensor RopeAware MLA-DECODE ------"
    )

    comptime kernel = MLA_SM100_Decode_QKV_FP8_PerTokenScale_RopeAware[
        q_type=q_type,
        KVLUTType=KVLUTType,
        output_type=output_type,
        SplitAccumType=SplitAccumType,
        MaskType=MaskType,
        config=config,
        ValidLengthType=ValidLengthType,
        _is_cache_length_accurate=_is_cache_length_accurate,
        ragged=ragged,
        has_per_token_scales=has_per_token_scales,
    ].kernel
    comptime pdl_level = PDLLevel.OVERLAP_AT_END if config.decoding_warp_split_k else PDLLevel.OFF
    ctx.enqueue_function[kernel](
        q_nope_tma,
        q_rope_tma,
        k_content_tma,
        k_rope_tma,
        scale_tma,
        o_tma,
        kv_lut,
        scale,
        mla_decode_pack,
        q_scale_ptr,
        scalar_args_buf,
        grid_dim=grid_dim,
        block_dim=block_dim,
        shared_mem_bytes=config.smem_used,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(config.smem_used)
        ),
        attributes=pdl_launch_attributes(pdl_level),
    )


@always_inline
def launch_mla_sm100_decode_sparse[
    q_type: DType,
    KVLUTType: MHAOperand,
    output_type: DType,
    SplitAccumType: OptionalPointer,
    MaskType: MHAMask,
    config: MLA_SM100_Decode_Config,
    ValidLengthType: OptionalPointer,
    _is_cache_length_accurate: Bool = False,
    ragged: Bool = False,
    has_attn_sink: Bool = False,
    has_extra_kv: Bool = False,
    has_variable_topk: Bool = False,
](
    q_tma: QOTMATile[
        dtype=q_type,
        BM=config.BM,
        BK=config.BK_QK,
        swizzle_mode=config.swizzle_mode,
    ],
    # K_nope gather4 TMA: INT64, SWIZZLE_NONE (linear SMEM layout).
    # tile_width = padded_depth / 8 = 64 INT64 elements (nope only).
    k_nope_tma: TMATensorTile[
        DType.int64,
        2,
        tile_shape=IndexList[2](
            config.BK_PV,
            _gather4_box_width[
                DType.int64,
                config.padded_depth // 8,
                TensorMapSwizzle.SWIZZLE_NONE,
            ](),
        ),
        desc_shape=IndexList[2](
            1,
            _gather4_box_width[
                DType.int64,
                config.padded_depth // 8,
                TensorMapSwizzle.SWIZZLE_NONE,
            ](),
        ),
    ],
    # K_rope gather4 TMA: BF16, SWIZZLE_128B.
    k_rope_tma: TMATensorTile[
        DType.bfloat16,
        2,
        tile_shape=IndexList[2](
            config.BK_PV,
            _gather4_box_width[
                DType.bfloat16,
                (config.padded_depth + config.rope_depth * 2) // 2,
                TensorMapSwizzle.SWIZZLE_128B,
            ](),
        ),
        desc_shape=IndexList[2](
            1,
            _gather4_box_width[
                DType.bfloat16,
                (config.padded_depth + config.rope_depth * 2) // 2,
                TensorMapSwizzle.SWIZZLE_128B,
            ](),
        ),
    ],
    o_tma: QOTMATile[
        dtype=output_type,
        BM=config.out_rows,
        BK=config.BN_PV // 4,
        swizzle_mode=config.swizzle_mode,
    ],
    kv_lut: KVLUTType,
    lse_accum_split_ptr: SplitAccumType,
    scale: Float32,
    batch_size: Int,
    block_z: Int,
    num_partitions: Int,
    q_max_seq_len: Int,
    valid_len: ValidLengthType,
    mask: MaskType,
    d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
    indices_stride: Int,
    topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
    scales_ptr: UnsafePointer[Scalar[DType.float32], origin=MutAnyOrigin],
    attn_sink_ptr: OptionalReg[
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
    ],
    # Extra KV parameters (separate always-attend cache).
    extra_k_nope_tma: TMATensorTile[
        DType.int64,
        2,
        tile_shape=IndexList[2](
            config.BK_PV,
            _gather4_box_width[
                DType.int64,
                config.padded_depth // 8,
                TensorMapSwizzle.SWIZZLE_NONE,
            ](),
        ),
        desc_shape=IndexList[2](
            1,
            _gather4_box_width[
                DType.int64,
                config.padded_depth // 8,
                TensorMapSwizzle.SWIZZLE_NONE,
            ](),
        ),
    ],
    extra_k_rope_tma: TMATensorTile[
        DType.bfloat16,
        2,
        tile_shape=IndexList[2](
            config.BK_PV,
            _gather4_box_width[
                DType.bfloat16,
                (config.padded_depth + config.rope_depth * 2) // 2,
                TensorMapSwizzle.SWIZZLE_128B,
            ](),
        ),
        desc_shape=IndexList[2](
            1,
            _gather4_box_width[
                DType.bfloat16,
                (config.padded_depth + config.rope_depth * 2) // 2,
                TensorMapSwizzle.SWIZZLE_128B,
            ](),
        ),
    ],
    extra_kv_lut: KVLUTType,
    extra_d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
    extra_topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
    extra_indices_stride: Int,
    extra_scales_ptr: OptionalReg[
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
    ],
    scalar_args_buf: TileTensor[
        DType.int64, address_space=AddressSpace.GENERIC, ...
    ],
    ctx: DeviceContext,
) raises:
    """Launch the sparse MLA decode kernel with gather4 TMA descriptors.

    d_indices stores encoded values: physical_block * page_size + offset.
    The kernel uses kv_lut.get_tma_row() to convert each encoded index
    into a physical TMA row.

    topk_lengths: per-batch array of actual topk counts. When non-null,
    topk_lengths[batch_idx] gives the number of valid entries for that
    batch. indices_stride is the stride between batches in d_indices
    (i.e., the max topk / allocation size).

    attn_sink_ptr: per-head correction values [num_heads_q], float32.
    When non-null, adjusts softmax denominator to account for
    non-selected tokens in sparse attention.
    """
    var mla_decode_pack = MLA_Decode_Pack[
        ValidLengthType=ValidLengthType,
        MaskType=MaskType,
        SplitAccumType=SplitAccumType,
    ](mask, valid_len, lse_accum_split_ptr)
    var block_x = ceildiv(config.num_q_heads, config.BM)
    var grid_dim = (block_x, q_max_seq_len, block_z)
    var block_dim = (config.num_threads, 1, 1)

    logger.info("------ Dispatching to SM100 Sparse MLA-DECODE ------")

    comptime kernel = MLA_SM100_Decode_Sparse[
        q_type=q_type,
        KVLUTType=KVLUTType,
        output_type=output_type,
        SplitAccumType=SplitAccumType,
        MaskType=MaskType,
        config=config,
        ValidLengthType=ValidLengthType,
        _is_cache_length_accurate=_is_cache_length_accurate,
        ragged=ragged,
        has_attn_sink=has_attn_sink,
        has_extra_kv=has_extra_kv,
        has_variable_topk=has_variable_topk,
    ].kernel
    comptime pdl_level = PDLLevel.OVERLAP_AT_END if config.decoding_warp_split_k else PDLLevel.OFF
    # Sparse kernel needs extra SMEM beyond config.smem_used:
    # - 4 idx_bars barriers (4 * 8 = 32 bytes)
    # - ptr_tmem_addr (4 bytes, UInt32)
    # - idx_smem double-buffered (2 * BN_QK * sizeof(Int32) = 512 bytes)
    # Total extra: 548 bytes.
    comptime sparse_extra_smem = 4 * config.mbar_size + 4 + 2 * config.BN_QK * 4
    comptime sparse_smem_used = config.smem_used + sparse_extra_smem
    ctx.enqueue_function[kernel](
        q_tma,
        k_nope_tma,
        k_rope_tma,
        o_tma,
        kv_lut,
        scale,
        mla_decode_pack,
        d_indices,
        indices_stride,
        topk_lengths,
        scales_ptr,
        attn_sink_ptr,
        extra_k_nope_tma,
        extra_k_rope_tma,
        extra_kv_lut,
        extra_d_indices,
        extra_topk_lengths,
        extra_indices_stride,
        extra_scales_ptr,
        scalar_args_buf,
        grid_dim=grid_dim,
        block_dim=block_dim,
        shared_mem_bytes=sparse_smem_used,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(sparse_smem_used)
        ),
        attributes=pdl_launch_attributes(pdl_level),
    )


@always_inline
def launch_mla_sm100_decode_sparse_kv_fp8[
    q_type: DType,
    KVLUTType: MHAOperand,
    output_type: DType,
    SplitAccumType: OptionalPointer,
    MaskType: MHAMask,
    config: MLA_SM100_Decode_Config,
    ValidLengthType: OptionalPointer,
    _is_cache_length_accurate: Bool = False,
    ragged: Bool = False,
    has_attn_sink: Bool = False,
    has_extra_kv: Bool = False,
    has_variable_topk: Bool = False,
](
    q_tma: QOTMATile[
        dtype=q_type,
        BM=config.BM,
        BK=config.BK_QK,
        swizzle_mode=config.swizzle_mode,
    ],
    # Single K gather4 TMA: INT64, SWIZZLE_NONE, tile_width=72 INT64 (576 B).
    k_tma: TMATensorTile[
        DType.int64,
        2,
        tile_shape=IndexList[2](
            config.BK_PV,
            _gather4_box_width[
                DType.int64,
                config.padded_q_depth // 8,
                TensorMapSwizzle.SWIZZLE_NONE,
            ](),
        ),
        desc_shape=IndexList[2](
            1,
            _gather4_box_width[
                DType.int64,
                config.padded_q_depth // 8,
                TensorMapSwizzle.SWIZZLE_NONE,
            ](),
        ),
    ],
    o_tma: QOTMATile[
        dtype=output_type,
        BM=config.out_rows,
        BK=config.BN_PV // 4,
        swizzle_mode=config.swizzle_mode,
    ],
    kv_lut: KVLUTType,
    lse_accum_split_ptr: SplitAccumType,
    scale: Float32,
    batch_size: Int,
    block_z: Int,
    num_partitions: Int,
    q_max_seq_len: Int,
    valid_len: ValidLengthType,
    mask: MaskType,
    d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
    indices_stride: Int,
    topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
    scales_ptr: UnsafePointer[Scalar[DType.float32], origin=MutAnyOrigin],
    attn_sink_ptr: OptionalReg[
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
    ],
    # Extra KV parameters (separate always-attend cache).
    extra_k_tma: TMATensorTile[
        DType.int64,
        2,
        tile_shape=IndexList[2](
            config.BK_PV,
            _gather4_box_width[
                DType.int64,
                config.padded_q_depth // 8,
                TensorMapSwizzle.SWIZZLE_NONE,
            ](),
        ),
        desc_shape=IndexList[2](
            1,
            _gather4_box_width[
                DType.int64,
                config.padded_q_depth // 8,
                TensorMapSwizzle.SWIZZLE_NONE,
            ](),
        ),
    ],
    extra_kv_lut: KVLUTType,
    extra_d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
    extra_topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
    extra_indices_stride: Int,
    extra_scales_ptr: OptionalReg[
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
    ],
    scalar_args_buf: TileTensor[
        DType.int64, address_space=AddressSpace.GENERIC, ...
    ],
    ctx: DeviceContext,
) raises:
    """Launches the all-FP8 sparse MLA decode kernel.

    This sibling of `launch_mla_sm100_decode_sparse` uses a single 576-byte
    gather4 TMA covering the full nope+rope row as FP8 (vs the BF16-rope
    parent which uses two separate descriptors).
    """
    var mla_decode_pack = MLA_Decode_Pack[
        ValidLengthType=ValidLengthType,
        MaskType=MaskType,
        SplitAccumType=SplitAccumType,
    ](mask, valid_len, lse_accum_split_ptr)
    var block_x = ceildiv(config.num_q_heads, config.BM)
    var grid_dim = (block_x, q_max_seq_len, block_z)
    var block_dim = (config.num_threads, 1, 1)

    logger.info(
        "------ Dispatching to SM100 Sparse MLA-DECODE (all-FP8 KV) ------"
    )

    comptime kernel = MLA_SM100_Decode_Sparse_KV_FP8[
        q_type=q_type,
        KVLUTType=KVLUTType,
        output_type=output_type,
        SplitAccumType=SplitAccumType,
        MaskType=MaskType,
        config=config,
        ValidLengthType=ValidLengthType,
        _is_cache_length_accurate=_is_cache_length_accurate,
        ragged=ragged,
        has_attn_sink=has_attn_sink,
        has_extra_kv=has_extra_kv,
        has_variable_topk=has_variable_topk,
    ].kernel
    comptime pdl_level = PDLLevel.OVERLAP_AT_END if config.decoding_warp_split_k else PDLLevel.OFF
    # Same extra SMEM budget as the BF16-rope sparse kernel:
    # - 4 idx_bars barriers (4 * mbar_size bytes)
    # - ptr_tmem_addr (4 bytes, UInt32)
    # - idx_smem double-buffered (2 * BN_QK * sizeof(Int32) = 512 bytes)
    comptime sparse_extra_smem = 4 * config.mbar_size + 4 + 2 * config.BN_QK * 4
    comptime sparse_smem_used = config.smem_used + sparse_extra_smem

    ctx.enqueue_function[kernel](
        q_tma,
        k_tma,
        o_tma,
        kv_lut,
        scale,
        mla_decode_pack,
        d_indices,
        indices_stride,
        topk_lengths,
        scales_ptr,
        attn_sink_ptr,
        extra_k_tma,
        extra_kv_lut,
        extra_d_indices,
        extra_topk_lengths,
        extra_indices_stride,
        extra_scales_ptr,
        scalar_args_buf,
        grid_dim=grid_dim,
        block_dim=block_dim,
        shared_mem_bytes=sparse_smem_used,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(sparse_smem_used)
        ),
        attributes=pdl_launch_attributes(pdl_level),
    )
