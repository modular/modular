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

from std.os import getenv
from nn.attention.gpu.nvidia.sm100.mla_stamp import (
    MLA_STAMP,
    STAMP_SMEM_BYTES,
)
from std.collections import OptionalReg
from std.math import ceildiv, clamp, gcd
from std.sys import get_defined_int, size_of
from std.utils import StaticTuple
from max.gpu.host import DeviceBuffer, DeviceContext, Dim
from max.gpu.host.nvidia import (
    ClusterAttributeLaunchConfig,
    GridBlockAttributeLaunchConfig,
)
from max.gpu.primitives.grid_controls import pdl_launch_attributes, PDLLevel
from layout import (
    ComptimeInt,
    Idx,
    RowMajorLayout,
    TileTensor,
    coord,
    row_major,
)
from layout.tile_tensor import NullableTileTensor
from layout.tma_async import (
    TMATensorTile,
    _gather4_box_width,
)
from max.gpu.host.nvidia.tma import (
    TensorMapL2Promotion,
    TensorMapSwizzle,
)
from std.logger import Logger
from std.memory import bitcast

from nn.attention.gpu.nvidia.common import (
    NonNullPointer,
    NullPointer,
    OptionalPointer,
)
from nn.attention.mha_mask import MHAMask
from nn.attention.mha_operand import MHAOperand
from nn.attention.mha_utils import (
    MHAConfig,
    null_pointer,
    unread_pointer,
)
from nn.attention.gpu.nvidia.common import KVTMATile
from std.utils.numerics import get_accum_type, nan


comptime logger = Logger()

# Q-sequence fold bounds for speculative decoding dispatch.
# When q_max_seq_len is in [MIN_FOLD_Q, MAX_FOLD_Q], the kernel folds q-tokens
# into the head dimension. MAX_FOLD_Q is the largest supported q_seq for fold
# (1 actual + 7 spec tokens). The fold loop iterates over
# range(MIN_FOLD_Q, MAX_FOLD_Q + 1).
comptime MIN_FOLD_Q = 2
comptime MAX_FOLD_Q = 8


@inline(.always)
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
        2: 3,
        3: 4,
        4: 8,
        5: 9,
        6: 16,
        7: 18,
        8: 20,
        9: 32,
        10: 37,
        11: 64,
        12: 72,
        13: half_sms,
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
        2: 3,
        3: 4,
        4: 8,
        5: 9,
        6: 16,
        7: 18,
        8: 20,
        9: 32,
        10: 37,
        11: 64,
        12: 72,
        13: half_sms,
    }
    comptime _default = _NUM_PARTITIONS.get(len(_NUM_PARTITIONS) - 1, 0)
    comptime for kv in _NUM_PARTITIONS.items():
        comptime v = kv.value
        if num_partitions <= v:
            return v
    return _default


# Number of bucket entries in the partition table (fixed at 14).
comptime _NUM_PARTITION_BUCKETS = 14


def _unbucketed_split_error(num_partitions: Int) -> Error:
    """Builds the error for a split count with no compiled combine kernel.

    Falling through instead would skip the split-K reduction and leave the
    output at whatever it held, which is indistinguishable from a correct
    result and faster than one.

    Args:
        num_partitions: The split count that matched no bucket.

    Returns:
        The error to raise.
    """
    return Error(
        "MLA decode split-K: num_partitions=",
        num_partitions,
        (
            " has no compiled combine kernel (not in the bucket table), so the"
            " partial outputs would never be reduced. Route the split count"
            " through `_bucket_num_partitions` or add the value to the table."
        ),
    )


from nn.attention.gpu.nvidia.sm100.mla_decode_utils import (
    _MLA_PDL_ALWAYS,
    _MLA_SPARSE_CG2,
    _MLA_SPARSE_BN_QK,
    _MLA_SPARSE_CG2_BF16,
    _MLA_SPARSE_CG2_ROUTE,
    _MLA_SPARSE_DIRECT_GATHER_E,
    _MLA_NVFP4_PACKED_LINEAR,
    SparseRowLayout,
    MLA_SM100_Decode_Config,
    QOTMATile,
    ORaggedTMATile,
    ScalesTMATile,
    tma_tile_qo,
    tma_tile_o,
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
from nn.attention.gpu.nvidia.sm100.mla_decode_sparse_qkv_fp8 import (
    MLA_SM100_Decode_Sparse_QKV_FP8,
    sparse_qkv_fp8_carve_bytes,
)
from nn.attention.gpu.nvidia.sm100.nvfp4_kv_dequant import (
    mla_kv_is_nvfp4,
    nvfp4_mla_row_bytes,
    nvfp4_mla_scale_row_bytes,
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


# The part of a CTA's cost that does not scale with the tokens it gathers
# (prologue, pipeline fill, drain, epilogue), in gathered-token equivalents so
# that only its ratio to the per-token cost enters the choice below. Fitted;
# the selected split is insensitive to the exact value.
comptime _CTA_FIXED_COST_IN_TOKENS = 187

# The model charges a partial wave as a whole one, so a difference below this
# margin is noise. Splits it cannot tell apart go to the tie-break below
# rather than the cost.
comptime _COST_TIE_MARGIN_PERCENT = 2
comptime _LAYOUT_E_NP_MARGIN_PERCENT = 10


def _cost_optimal_partition_bucket[
    half_sms: Int
](
    ctas_per_partition: Int,
    split_len: Int,
    sm_count: Int,
    max_partitions: Int,
) -> Int:
    """Returns the cheapest split count the combine kernel can be launched for.

    Wall time for a decode launch is
    `ceil(total_ctas / sm_count) * (fixed_cost + work_per_split)`: splitting
    divides the work per split but multiplies the CTA count, and every extra
    wave re-pays the fixed cost. Sizing the split to fill one wave models
    neither term — measured at batch 16, the largest one-wave split is the
    worst reachable choice — so minimise the product directly over the (small)
    candidate set.

    The search runs over the partition buckets rather than over every integer
    because `dispatch_combine` matches `num_partitions` against that set
    exactly, with no fallback: a count outside it launches the decode and never
    reduces its partial outputs. Rounding an unconstrained optimum up to the
    next bucket is not the optimum of the reachable set — at the spec-decode
    verify shape the unconstrained optimum is 3, which rounds to the value the
    shipped policy already picks — so the constraint belongs inside the search.

    Parameters:
        half_sms: sm_count // 2 — the largest bucket value (compile-time).

    Args:
        ctas_per_partition: CTAs launched per split, i.e. head tiles *
            `grid_y` * `batch_size`.
        split_len: KV tokens each (batch, position) attends before splitting.
        sm_count: Number of SMs on the target GPU.
        max_partitions: Largest admissible split count.

    Returns:
        The cost-minimising number of split-K partitions.
    """
    var best_np = 1
    var best_cost = -1
    # Ascending, so taking a candidate only when it is materially cheaper keeps
    # the smallest split among those the model cannot distinguish.
    comptime for b in range(_NUM_PARTITION_BUCKETS):
        var np = _get_partition_bucket[half_sms, b]()
        if np <= max_partitions:
            var waves = ceildiv(ctas_per_partition * np, sm_count)
            var cost = waves * (
                _CTA_FIXED_COST_IN_TOKENS + ceildiv(split_len, np)
            )
            if best_cost < 0 or cost * 100 < best_cost * (
                100 - _COST_TIE_MARGIN_PERCENT
            ):
                best_cost = cost
                best_np = np
    return best_np


@fieldwise_init
struct _SplitCost(TrivialRegisterPassable):
    """Fitted wall-time model of one sparse decode kernel family, in ns.

    A launch of `np` splits costs, per wave `w` of `n_w` CTAs,
    `cta_fixed_ns + U * (unit_idle_ns + (unit_full_ns - unit_idle_ns) * n_w /
    sm_count)`, where `U` is the critical CTA's 64-key units, plus
    `split_fixed_ns + split_np_ns * np` once when `np > 1` (workspaces, the
    combine launch). `np = 1` launches a different kernel (no split
    epilogue, a deeper K/V ring), so it has its own `cta_fixed1_ns`,
    `unit_idle1_ns` and `unit_full1_ns`. A split that leaves a CTA fewer than
    `min_split_keys` keys is not considered.
    """

    var cta_fixed_ns: Int
    var split_fixed_ns: Int
    var split_np_ns: Int
    var unit_idle_ns: Int
    var unit_full_ns: Int
    var min_split_keys: Int
    var cta_fixed1_ns: Int
    var unit_idle1_ns: Int
    var unit_full1_ns: Int


# Layout G mode 1, direct gather, half-row softmax (the routed sparse decode
# for at most 16 rows), B200. Seeded from the P6 forced-np sweep (q = 6,
# top_k 2048 and 4096, B 8 and 32, np 1-8): per 64 keys 449 ns in a
# near-empty wave rising to 603 ns in a full one, 4.65 us per wave and
# 0.90 us per split launch; max residual 5.7%. The floor keeps two 64-key
# tiles per split.
# The split class is the seed. The `np = 1` class is refit after the
# non-split kernel moved to a 5-stage ring, to the P7b forced-np sweep (q = 6,
# top_k 2048 and 4096, B 8-64) plus np 1 against np 2 at q 2-8 where a second
# wave of the non-split kernel competes: every one of those 20 cells picks
# within 0.5% of its forced best (the seed: 11%). Lower fixed cost but dearer
# keys than the split class, so np = 1 now wins its second wave at top_k 2048
# (B 28-37 at q = 6) and top_k 512, and loses it at top_k 8192. Held out
# (17 cells, top_k 512-8192, q 3-8): every changed pick is 0.74x-0.99x of the
# seed's.
comptime _LAYOUT_G_DIRECT_HALF_ROWS_COST = _SplitCost(
    cta_fixed_ns=4650,
    split_fixed_ns=900,
    split_np_ns=0,
    unit_idle_ns=449,
    unit_full_ns=603,
    min_split_keys=128,
    cta_fixed1_ns=3500,
    unit_idle1_ns=475,
    unit_full1_ns=625,
)

# At top-k 2048 and a cache of 6144 to 32768 keys, np = 1 wins from one full
# wave (q = 6, B = 24) through B = 28 (168 CTAs) by 1.5% to 4.7%; the general
# record splits B = 25 to 27 in two. At a 4096-key cache two splits are 2%
# faster at B = 25 to 27, so the general record stays below 6144.
comptime _LAYOUT_G_DIRECT_2048_COST = _SplitCost(
    cta_fixed_ns=4650,
    split_fixed_ns=900,
    split_np_ns=0,
    unit_idle_ns=449,
    unit_full_ns=603,
    min_split_keys=128,
    cta_fixed1_ns=3400,
    unit_idle1_ns=475,
    unit_full1_ns=625,
)

# The other sparse decode families on B200, each fitted to its own forced-np
# sweep (top_k 2048, page 128, cache 32768, B 1-64, q 1-6, np 1-16; weighted
# least squares on relative error). A CTA covers `ceildiv(num_heads, 64)`
# heads of one q position in all four.
# Native FP8, Layout E (more than 32 heads).
comptime _SPARSE_E_FP8_COST = _SplitCost(
    cta_fixed_ns=6997,
    split_fixed_ns=203,
    split_np_ns=141,
    unit_idle_ns=625,
    unit_full_ns=742,
    min_split_keys=128,
    cta_fixed1_ns=25,
    unit_idle1_ns=805,
    unit_full1_ns=839,
)
# Native FP8, full-row Layout G (17 to 32 heads).
comptime _SPARSE_G_FULL_FP8_COST = _SplitCost(
    cta_fixed_ns=5184,
    split_fixed_ns=848,
    split_np_ns=160,
    unit_idle_ns=522,
    unit_full_ns=596,
    min_split_keys=128,
    cta_fixed1_ns=20,
    unit_idle1_ns=636,
    unit_full1_ns=703,
)
# BF16 Q over an FP8 cache (the kernel converts K/V to BF16).
comptime _SPARSE_BF16_Q_FP8_KV_COST = _SplitCost(
    cta_fixed_ns=5659,
    split_fixed_ns=605,
    split_np_ns=145,
    unit_idle_ns=1618,
    unit_full_ns=1809,
    min_split_keys=128,
    cta_fixed1_ns=53,
    unit_idle1_ns=1712,
    unit_full1_ns=2004,
)
# BF16 Q over a BF16 cache.
comptime _SPARSE_BF16_COST = _SplitCost(
    cta_fixed_ns=5487,
    split_fixed_ns=0,
    split_np_ns=212,
    unit_idle_ns=1448,
    unit_full_ns=1802,
    min_split_keys=128,
    cta_fixed1_ns=45,
    unit_idle1_ns=1450,
    unit_full1_ns=1918,
)


@fieldwise_init
struct _SparseLayoutEModels(TrivialRegisterPassable):
    var single64: _SplitCost
    var pair128: _SplitCost
    var single128: _SplitCost
    var validated: Int
    var automatic: Int


# B200 NVFP4 plain sparse decode: three rotated screening rounds,
# followed by fixed-choice confirmation. Automatic routing retains single64;
# the other fitted geometries support reproducible forced comparisons.
comptime _NO_LAYOUT_E_COST = _SplitCost(0, 0, 0, 0, 0, 0, 0, 0, 0)
comptime _NO_LAYOUT_E_MODELS = _SparseLayoutEModels(
    _NO_LAYOUT_E_COST, _NO_LAYOUT_E_COST, _NO_LAYOUT_E_COST, 0, 0
)
comptime _SPARSE_E_FP8_H64_MODELS = _NO_LAYOUT_E_MODELS
comptime _SPARSE_E_FP8_H128_MODELS = _NO_LAYOUT_E_MODELS
comptime _SPARSE_E_NVFP4_H64_SINGLE64_COST = _SplitCost(
    cta_fixed_ns=9401,
    split_fixed_ns=959,
    split_np_ns=120,
    unit_idle_ns=860,
    unit_full_ns=1288,
    min_split_keys=128,
    cta_fixed1_ns=6284,
    unit_idle1_ns=1027,
    unit_full1_ns=1127,
)
comptime _SPARSE_E_NVFP4_H64_SINGLE128_COST = _SplitCost(
    cta_fixed_ns=9307,
    split_fixed_ns=1275,
    split_np_ns=117,
    unit_idle_ns=1120,
    unit_full_ns=1677,
    min_split_keys=128,
    cta_fixed1_ns=6355,
    unit_idle1_ns=1342,
    unit_full1_ns=1593,
)
comptime _SPARSE_E_NVFP4_H64_MODELS = _SparseLayoutEModels(
    _SPARSE_E_NVFP4_H64_SINGLE64_COST,
    _NO_LAYOUT_E_COST,
    _SPARSE_E_NVFP4_H64_SINGLE128_COST,
    5,
    1,
)
comptime _SPARSE_E_NVFP4_H128_SINGLE64_COST = _SplitCost(
    cta_fixed_ns=9430,
    split_fixed_ns=1061,
    split_np_ns=129,
    unit_idle_ns=842,
    unit_full_ns=1214,
    min_split_keys=128,
    cta_fixed1_ns=6415,
    unit_idle1_ns=1024,
    unit_full1_ns=1111,
)
comptime _SPARSE_E_NVFP4_H128_PAIR128_COST = _SplitCost(
    cta_fixed_ns=8745,
    split_fixed_ns=804,
    split_np_ns=127,
    unit_idle_ns=1338,
    unit_full_ns=1826,
    min_split_keys=128,
    cta_fixed1_ns=5211,
    unit_idle1_ns=1560,
    unit_full1_ns=1740,
)
comptime _SPARSE_E_NVFP4_H128_SINGLE128_COST = _SplitCost(
    cta_fixed_ns=9546,
    split_fixed_ns=1395,
    split_np_ns=123,
    unit_idle_ns=1088,
    unit_full_ns=1578,
    min_split_keys=128,
    cta_fixed1_ns=6525,
    unit_idle1_ns=1322,
    unit_full1_ns=1545,
)
comptime _SPARSE_E_NVFP4_H128_MODELS = _SparseLayoutEModels(
    _SPARSE_E_NVFP4_H128_SINGLE64_COST,
    _SPARSE_E_NVFP4_H128_PAIR128_COST,
    _SPARSE_E_NVFP4_H128_SINGLE128_COST,
    7,
    1,
)


def _sparse_layout_e_models[
    nvfp4: Bool, heads: Int, prefill: Bool = False
]() -> _SparseLayoutEModels:
    # No Layout E records are fitted for prefill on decode, so it keeps the
    # incumbent launch.
    comptime if prefill:
        return _NO_LAYOUT_E_MODELS
    comptime if nvfp4:
        return (
            _SPARSE_E_NVFP4_H128_MODELS if heads
            == 128 else _SPARSE_E_NVFP4_H64_MODELS
        )
    return (
        _SPARSE_E_FP8_H128_MODELS if heads == 128 else _SPARSE_E_FP8_H64_MODELS
    )


def _sparse_variant_cost[
    models: _SparseLayoutEModels, variant: Int
]() -> _SplitCost:
    comptime if variant == 1:
        return models.pair128
    elif variant == 2:
        return models.single128
    return models.single64


def _sparse_cost_family[num_heads: Int, q_type: DType, kv_type: DType]() -> Int:
    """Returns which fitted sparse cost record `_sparse_partition_bucket`
    applies, or 0 when none does (half-row Layout G has its own route, and
    NVFP4 was not swept).
    """
    comptime if kv_type == DType.float8_e4m3fn:
        comptime if q_type == DType.float8_e4m3fn:
            comptime if _routes_half_row_layout_g[num_heads]():
                return 0
            return 1 if num_heads <= 32 else 2
        comptime if q_type == DType.bfloat16:
            return 3
    comptime if kv_type == DType.bfloat16 and q_type == DType.bfloat16:
        return 4
    return 0


def _sparse_partition_bucket[
    half_sms: Int, family: Int
](
    ctas_per_partition: Int,
    split_len: Int,
    page: Int,
    sm_count: Int,
    max_partitions: Int,
) -> Int:
    comptime if family == 1:
        return _config_cost_partition_bucket[half_sms, _SPARSE_G_FULL_FP8_COST](
            ctas_per_partition, split_len, page, sm_count, max_partitions
        )
    elif family == 2:
        return _config_cost_partition_bucket[half_sms, _SPARSE_E_FP8_COST](
            ctas_per_partition, split_len, page, sm_count, max_partitions
        )
    elif family == 3:
        return _config_cost_partition_bucket[
            half_sms, _SPARSE_BF16_Q_FP8_KV_COST
        ](ctas_per_partition, split_len, page, sm_count, max_partitions)
    elif family == 4:
        return _config_cost_partition_bucket[half_sms, _SPARSE_BF16_COST](
            ctas_per_partition, split_len, page, sm_count, max_partitions
        )
    # Family 0 is instantiated behind a runtime branch its caller never takes.
    return 1


def _split_cost_scaled[
    cost: _SplitCost, key_tile: Int = 64
](ctas_per_partition: Int, keys: Int, np: Int, sm_count: Int) -> Int:
    """Returns `cost`'s modelled launch time in ns times `sm_count`.

    Parameters:
        cost: The kernel family's fitted cost record.
        key_tile: The key tile the family is fitted at (compile-time).

    Args:
        ctas_per_partition: CTAs launched per split.
        keys: Keys the critical CTA walks.
        np: The split count.
        sm_count: Number of SMs on the target GPU.

    Returns:
        The modelled time, scaled by `sm_count` to keep the occupancy term
        integral.
    """
    comptime assert key_tile == 64 or key_tile == 128
    var units = ceildiv(keys, key_tile) * (key_tile // 64)
    var ctas = ctas_per_partition * np
    var waves = ceildiv(ctas, sm_count)
    if np > 1:
        return (
            waves * sm_count * (cost.cta_fixed_ns + units * cost.unit_idle_ns)
            + units * (cost.unit_full_ns - cost.unit_idle_ns) * ctas
            + sm_count * (cost.split_fixed_ns + cost.split_np_ns * np)
        )
    return (
        waves * sm_count * (cost.cta_fixed1_ns + units * cost.unit_idle1_ns)
        + units * (cost.unit_full1_ns - cost.unit_idle1_ns) * ctas
    )


def _config_cost_partition_bucket[
    half_sms: Int, cost: _SplitCost, key_tile: Int = 64, clip_keys: Bool = False
](
    ctas_per_partition: Int,
    split_len: Int,
    page: Int,
    sm_count: Int,
    max_partitions: Int,
) -> Int:
    """Returns the bucketed split count `cost` predicts is fastest.

    The kernel splits on whole split pages, so the critical CTA holds
    `ceildiv(ceildiv(split_len, page), np) * page` keys. Costs are compared
    in ns times `sm_count`, so the per-wave occupancy term stays integral.
    Candidates are the partition buckets in ascending order, and one is taken
    only when it is more than `_COST_TIE_MARGIN_PERCENT` cheaper, as in
    `_cost_optimal_partition_bucket`.

    Parameters:
        half_sms: sm_count // 2 -- the largest bucket value (compile-time).
        cost: The kernel family's fitted cost record.
        key_tile: The key tile the family is fitted at (compile-time).
        clip_keys: Whether a bucket's keys are clamped to `split_len`
            (compile-time), as the kernel's own tile loop clamps them.

    Args:
        ctas_per_partition: CTAs launched per split.
        split_len: Keys each CTA row attends before splitting.
        page: The split page size the kernel splits on.
        sm_count: Number of SMs on the target GPU.
        max_partitions: Largest admissible split count.

    Returns:
        The cost-minimising number of split-K partitions.
    """
    if split_len <= 0:
        return 1
    var pages = ceildiv(split_len, page)
    var best_np = 1
    var best_cost = -1
    comptime for b in range(_NUM_PARTITION_BUCKETS):
        var np = _get_partition_bucket[half_sms, b]()
        var keys = ceildiv(pages, np) * page
        comptime if clip_keys:
            keys = min(keys, split_len)
        if (
            np <= max_partitions
            and np <= pages
            and (np == 1 or keys >= cost.min_split_keys)
        ):
            var t = _split_cost_scaled[cost, key_tile](
                ctas_per_partition, keys, np, sm_count
            )
            if best_cost < 0 or t * 100 < best_cost * (
                100 - _COST_TIE_MARGIN_PERCENT
            ):
                best_cost = t
                best_np = np
    return best_np


def _one_wave_partition_bucket[
    half_sms: Int
](
    ctas_per_partition: Int,
    split_len: Int,
    page: Int,
    sm_count: Int,
    max_partitions: Int,
) -> Int:
    """Returns the largest split bucket whose grid fits in one wave, or 1.

    Each split keeps at least two 64-key tiles, and a bucket that would leave
    a split empty falls back to the smaller one with the same boundaries.

    Parameters:
        half_sms: sm_count // 2 -- the largest bucket value (compile-time).

    Args:
        ctas_per_partition: CTAs launched per split.
        split_len: Keys each CTA row attends before splitting.
        page: The split page size the kernel splits on.
        sm_count: Number of SMs on the target GPU.
        max_partitions: Largest admissible split count.

    Returns:
        The split count.
    """
    var pages = ceildiv(split_len, page)
    var best_np = 1
    comptime for b in range(_NUM_PARTITION_BUCKETS):
        var np = _get_partition_bucket[half_sms, b]()
        if (
            np <= max_partitions
            and np <= pages
            and ceildiv(pages, np) * page >= 128
            and ctas_per_partition * np <= sm_count
        ):
            best_np = np
    return _compact_partition_bucket[half_sms](best_np, split_len, page)


def _layout_g_partition_bucket[
    half_sms: Int
](
    ctas_per_partition: Int,
    split_len: Int,
    page: Int,
    sm_count: Int,
    max_partitions: Int,
    *,
    top_k: Int = 0,
    cache_len: Int = 0,
) -> Int:
    if (
        top_k == 2048
        and split_len == 2048
        and page == 128
        and sm_count == 148
        and cache_len >= 6144
    ):
        return _config_cost_partition_bucket[
            half_sms, _LAYOUT_G_DIRECT_2048_COST
        ](ctas_per_partition, split_len, page, sm_count, max_partitions)
    return _config_cost_partition_bucket[
        half_sms, _LAYOUT_G_DIRECT_HALF_ROWS_COST
    ](ctas_per_partition, split_len, page, sm_count, max_partitions)


def _compact_partition_bucket[
    half_sms: Int
](num_partitions: Int, split_len: Int, page: Int) -> Int:
    """Returns the smallest split bucket with the same partition boundaries.

    Split CTAs traverse `ceildiv(pages, np)` pages. A larger bucket with
    that same width adds empty partitions, without adding productive CTAs.
    Keep the split kernel rather than crossing to the non-split kernel.
    """
    if num_partitions <= 2 or split_len <= 0:
        return num_partitions
    var pages = ceildiv(split_len, page)
    var pages_per_split = ceildiv(pages, num_partitions)
    comptime for b in range(_NUM_PARTITION_BUCKETS):
        var np = _get_partition_bucket[half_sms, b]()
        if (
            np > 1
            and np < num_partitions
            and ceildiv(pages, np) == pages_per_split
        ):
            return np
    return num_partitions


# The h=128 FP8 CTA pair against the unpaired direct kernel, by the keys its
# critical CTA walks. In a forced-split sweep (top-k 2048, B 1 to 64, q 1 to
# 8, caches 128 to 32768, B200) the unpaired kernel is 3% to 11% faster
# (medians) at one to three 128-key tiles, and the pair 1% to 9% faster from
# four tiles up, in one wave or several: the pair's per-CTA fixed cost (about
# 1.4 us) outweighs its cheaper tiles until then.
comptime _SPARSE_PAIR_MAX_DIRECT_WALK = 384
# The same route for the BF16 form (`MLA_SPARSE_CG2_BF16=1`). The BF16 pair's
# crossover is earlier than FP8's: in the h=128 forced-split sweep (B 1 to
# 32, S 1 and 6, caches 64 to 1024) the pair is about even with the unpaired
# kernel at 64 keys and 0.90-0.93x at 128, so walks of 64 keys or fewer take
# the unpaired kernel. The SYNTHESIS estimate was 90-140 keys per CTA against
# FP8's 384.
comptime _SPARSE_BF16_PAIR_MAX_DIRECT_WALK = 64
# A token that sees at most this many keys runs one split on either kernel:
# the split's combine launch costs more than splitting four tiles saves
# (one split is 0.67x to 0.85x of the cost-model split at caches of 128 to
# 256 keys, and no slower at 512).
comptime _SPARSE_PAIR_NO_SPLIT_KEYS = 512


def _sparse_pair_form[q_type: DType, kv_type: DType, num_heads: Int]() -> Bool:
    """Returns whether these dtypes and head count select the h=128 CTA pair,
    before the cache's scaling and the launch's options are checked
    (`_sparse_pair_eligible`).
    """
    var fp8 = q_type == .float8_e4m3fn and kv_type == .float8_e4m3fn
    var bf16 = (
        _MLA_SPARSE_CG2_BF16 != 0
        and q_type == .bfloat16
        and kv_type == .bfloat16
    )
    return (
        _MLA_SPARSE_CG2 != 0
        and _MLA_SPARSE_BN_QK != 64
        and num_heads == 128
        and (fp8 or bf16)
    )


def _sparse_pair_eligible_form[
    q_type: DType,
    kv_type: DType,
    kv_quantized: Bool,
    kv_nvfp4: Bool,
    num_heads: Int,
    per_token_scale_rope_aware: Bool,
    fold_shared_index: Bool,
]() -> Bool:
    """Returns whether the sparse decode runs the h=128 CTA pair.

    That is the unit-scale Layout E FP8 form at 128 heads, unfolded, unless
    `-D MLA_SPARSE_CG2=0` disables the pair; with `-D MLA_SPARSE_CG2_BF16=1`
    also the BF16 form (BF16 Q over a BF16 cache).
    """
    return (
        (
            _sparse_pair_form[q_type, kv_type, num_heads]()
            or (
                kv_nvfp4
                and q_type == .float8_e4m3fn
                and num_heads == 128
                and _MLA_SPARSE_CG2 != 0
                and _MLA_SPARSE_BN_QK == 128
            )
        )
        and (not kv_quantized or kv_nvfp4)
        and not per_token_scale_rope_aware
        and not fold_shared_index
    )


def _sparse_pair_eligible[
    q_type: DType,
    k_t: MHAOperand,
    num_heads: Int,
    per_token_scale_rope_aware: Bool,
    fold_shared_index: Bool,
]() -> Bool:
    """Returns `_sparse_pair_eligible_form` for the KV operand `k_t`."""
    return _sparse_pair_eligible_form[
        q_type,
        k_t.dtype,
        k_t.quantization_enabled,
        mla_kv_is_nvfp4[k_t](),
        num_heads,
        per_token_scale_rope_aware,
        fold_shared_index,
    ]()


def _sparse_pair_keys(
    topk_stride: Int, extra_topk: Int, effective_max_cache_len: Int
) -> Int:
    """Returns the most keys any token of a sparse launch walks.

    The kernel clamps the top-k to the cache (`OffsetPosition`); a cache
    length of 0 means the caller passed none, which keeps the top-k.
    """
    var main = topk_stride if effective_max_cache_len <= 0 else min(
        topk_stride, effective_max_cache_len
    )
    return main + extra_topk


def _sparse_pair_route[
    walk_threshold: Int
](keys: Int, num_partitions: Int, page: Int) -> Bool:
    """Returns whether a pair-eligible launch keeps the pair.

    Parameters:
        walk_threshold: The critical CTA's walk above which the pair wins;
            `_SPARSE_PAIR_MAX_DIRECT_WALK` for FP8 and
            `_SPARSE_BF16_PAIR_MAX_DIRECT_WALK` for BF16.

    Args:
        keys: The most keys a token walks (`_sparse_pair_keys`).
        num_partitions: The launch's split count.
        page: The split page size the kernel splits on.

    Returns:
        True when the critical CTA walks more than `walk_threshold` keys.
    """
    var walk = min(ceildiv(ceildiv(keys, page), num_partitions) * page, keys)
    return walk > walk_threshold


def _sparse_pair_launches[
    eligible: Bool, walk_threshold: Int
](keys: Int, num_partitions: Int, page: Int) -> Bool:
    """Returns whether a sparse launch runs the CTA pair.

    Parameters:
        eligible: Whether the launch's form takes the pair
            (`_sparse_pair_eligible`).
        walk_threshold: The critical CTA's walk above which the pair wins.

    Args:
        keys: The most keys a token walks (`_sparse_pair_keys`).
        num_partitions: The launch's split count.
        page: The split page size the kernel splits on.

    Returns:
        True when the form is eligible and either `MLA_SPARSE_CG2_ROUTE=0`
        keeps every launch on the pair or `_sparse_pair_route` keeps this one.
    """
    comptime if not eligible:
        return False
    comptime if _MLA_SPARSE_CG2_ROUTE == 0:
        return True
    return _sparse_pair_route[walk_threshold](keys, num_partitions, page)


def _sparse_layout_e_variant_allowed[
    models: _SparseLayoutEModels, variant: Int, pair_available: Bool
](incumbent: Int, automatic: Int) -> Bool:
    comptime if variant == 1 and not pair_available:
        return False
    comptime if _MLA_SPARSE_BN_QK == 64:
        return variant == 0
    comptime if _MLA_SPARSE_CG2_ROUTE == 0 and pair_available and (
        models.validated & 2
    ) != 0:
        return variant == 1
    comptime if _MLA_SPARSE_BN_QK == 128:
        return variant != 0
    return variant == incumbent or (automatic & (1 << variant)) != 0


def _sparse_layout_e_incumbent_variant[
    legacy_pair: Bool, wide_forced: Bool, walk_threshold: Int
](keys: Int, np: Int, page: Int) -> Int:
    var variant = Int(
        _sparse_pair_launches[legacy_pair, walk_threshold](keys, np, page)
    )
    comptime if wide_forced:
        if variant == 0:
            variant = 2
    return variant


def _sparse_layout_e_cost_at_np[
    models: _SparseLayoutEModels,
    legacy_pair: Bool,
    wide_forced: Bool,
    pair_available: Bool,
    walk_threshold: Int,
](
    ctas: Int, keys: Int, page: Int, sm_count: Int, np: Int, automatic: Int
) -> Tuple[Int, Int, Int]:
    var variant = _sparse_layout_e_incumbent_variant[
        legacy_pair, wide_forced, walk_threshold
    ](keys, np, page)
    comptime if _MLA_SPARSE_BN_QK != 64 and _MLA_SPARSE_CG2_ROUTE == 0 and pair_available and (
        models.validated & 2
    ) != 0:
        variant = 1
    var critical_keys = min(keys, ceildiv(ceildiv(keys, page), np) * page)
    var best_variant = variant
    var best_cost = -1
    var min_keys = 0
    # Initialize from the incumbent so a tie does not change its grouping.
    comptime for v in range(3):
        comptime if (models.validated & (1 << v)) != 0:
            if variant == v:
                comptime cost = _sparse_variant_cost[models, v]()
                comptime assert (
                    cost.cta_fixed_ns
                    + cost.split_fixed_ns
                    + cost.split_np_ns
                    + cost.unit_idle_ns
                    + cost.unit_full_ns
                    > 0
                ) and (
                    cost.cta_fixed1_ns + cost.unit_idle1_ns + cost.unit_full1_ns
                    > 0
                )
                best_cost = _split_cost_scaled[cost, 64 if v == 0 else 128](
                    ctas, critical_keys, np, sm_count
                )
                min_keys = cost.min_split_keys
    if best_cost < 0:
        return (variant, -1, 0)
    comptime for v in range(3):
        comptime if (models.validated & (1 << v)) != 0:
            if _sparse_layout_e_variant_allowed[models, v, pair_available](
                variant, automatic
            ):
                comptime cost = _sparse_variant_cost[models, v]()
                var t = _split_cost_scaled[cost, 64 if v == 0 else 128](
                    ctas, critical_keys, np, sm_count
                )
                if t * 100 < best_cost * (100 - _COST_TIE_MARGIN_PERCENT):
                    best_cost = t
                    best_variant = v
                    min_keys = cost.min_split_keys
    return (best_variant, best_cost, min_keys)


def _select_sparse_layout_e_launch[
    q_type: DType,
    kv_type: DType,
    kv_quantized: Bool,
    kv_nvfp4: Bool,
    num_heads: Int,
    half_sms: Int,
    *,
    per_token_scale_rope_aware: Bool = False,
    fold_shared_index: Bool = False,
    has_extra_k: Bool = False,
    physical_indices: Bool = True,
    depth: Int = 576,
    fixed_topk: Bool = True,
    skip_masked_tiles: Bool = False,
    sliding_window_size: Int = -1,
    np_margin_percent: Int = _LAYOUT_E_NP_MARGIN_PERCENT,
    models: _SparseLayoutEModels = _sparse_layout_e_models[
        kv_nvfp4, num_heads, skip_masked_tiles
    ](),
](
    batch_size: Int,
    q_max_seq_len: Int,
    keys: Int,
    page: Int,
    sm_count: Int,
    incumbent_np: Int,
    fixed_np: Int = 0,
    effective_cache_len: Int = 0,
) -> Tuple[Int, Int]:
    """Selects a sparse Layout E split and variant from the same cost records.

    Variants are 0 single64, 1 pair128 and 2 single128. A positive fixed_np
    selects only the variant at that split, so explicit API overrides and
    capture keys use the same final geometry as default dispatch.
    """
    comptime assert 0 <= np_margin_percent and np_margin_percent < 100
    comptime legacy_pair = _sparse_pair_eligible_form[
        q_type,
        kv_type,
        kv_quantized,
        kv_nvfp4,
        num_heads,
        per_token_scale_rope_aware,
        fold_shared_index,
    ]()
    # BF16's pair is cheaper from about 128 keys per CTA, not FP8's 384:
    # the BF16 pair's fixed cost is lower and its per-key saving larger.
    comptime bf16_pair = (
        _MLA_SPARSE_CG2_BF16 != 0
        and q_type == .bfloat16
        and kv_type == .bfloat16
    )
    comptime walk_threshold = (
        _SPARSE_BF16_PAIR_MAX_DIRECT_WALK if bf16_pair else _SPARSE_PAIR_MAX_DIRECT_WALK
    )
    comptime wide_forced = (
        _MLA_SPARSE_BN_QK == 128
        and num_heads >= 64
        and q_type == .float8_e4m3fn
    )
    var np = fixed_np if fixed_np > 0 else incumbent_np
    var variant = _sparse_layout_e_incumbent_variant[
        legacy_pair, wide_forced, walk_threshold
    ](keys, np, page)
    comptime native_form = (
        q_type == .float8_e4m3fn
        and (kv_type == .float8_e4m3fn or kv_nvfp4)
        and (not kv_quantized or kv_nvfp4)
        and (num_heads == 64 or num_heads == 128)
        and depth == 576
        and fixed_topk
        and physical_indices
        and not per_token_scale_rope_aware
        and not fold_shared_index
        and not has_extra_k
        and sliding_window_size < 0
    )
    comptime if not native_form:
        return (np, variant)
    comptime if models.validated == 0:
        return (np, variant)
    var automatic = models.automatic
    # Initial matrix scope; widen only after validating the measured regions.
    # Format-specific regression fallbacks belong here, by workload property.
    if (
        batch_size < 1
        or batch_size > 64
        or q_max_seq_len < 1
        or q_max_seq_len > 8
        or page != 128
        or keys <= 0
        or keys > 2048
    ):
        return (np, variant)
    comptime pair_available = (
        num_heads == 128
        and _MLA_SPARSE_CG2 != 0
        and (not kv_nvfp4 or physical_indices)
    )
    var pages = ceildiv(keys, page)
    if np > pages:
        if fixed_np > 0 or pages > 2:
            return (np, variant)
        # Confirmed one/two-page defaults avoid empty CTAs. Larger regions
        # and explicit overrides retain their grid; empty-CTA parallelism
        # can outperform a useful-only split there.
        np = 1
        variant = _sparse_layout_e_incumbent_variant[
            legacy_pair, wide_forced, walk_threshold
        ](keys, np, page)
    var ctas = ceildiv(num_heads, 64) * batch_size * q_max_seq_len
    var best = _sparse_layout_e_cost_at_np[
        models, legacy_pair, wide_forced, pair_available, walk_threshold
    ](ctas, keys, page, sm_count, np, automatic)
    if best[1] < 0:
        return (np, variant)
    if fixed_np > 0:
        return (np, best[0])
    var best_np = np
    comptime for b in range(_NUM_PARTITION_BUCKETS):
        var candidate_np = _get_partition_bucket[half_sms, b]()
        if candidate_np <= pages:
            var choice = _sparse_layout_e_cost_at_np[
                models, legacy_pair, wide_forced, pair_available, walk_threshold
            ](ctas, keys, page, sm_count, candidate_np, automatic)
            var critical_keys = min(keys, ceildiv(pages, candidate_np) * page)
            if choice[1] >= 0 and (
                candidate_np == 1 or critical_keys >= choice[2]
            ):
                if choice[1] * 100 < best[1] * (100 - np_margin_percent):
                    best = choice
                    best_np = candidate_np
    return (best_np, best[0])


def _compute_num_partitions_64[
    num_heads: Int,
    is_fp8_kv: Bool = False,
    half_sms: Int = 74,
    # Read-once shared-index fold (KERN-3141): grid_y collapses to 1, so the
    # decode relies on split-K ALONE to fill the SMs. Relax the per-split page
    # floor for the folded shapes so the wave-aligned target_partitions
    # survives. Default False -> byte-identical to the production path.
    fold_shared_index: Bool = False,
](
    batch_size: Int,
    effective_max_cache_len: Int,
    q_max_seq_len: Int,
    split_page_size: Int,
    sm_count: Int,
    relax_split_floor: Bool = False,
    cost_optimal_split: Bool = False,
) -> Int:
    """Wave-aligned split count for single head group (e.g. Kimi K2.5, 64 heads).

    This is the EXACT logic from the original unified _compute_num_partitions
    when _head_groups == 1.  Do NOT modify without verifying byte-for-byte
    equivalence with the 64-head production path.

    Parameters:
        num_heads: Number of Q attention heads (compile-time).
        is_fp8_kv: Whether the KV cache is FP8 (compile-time).
        half_sms: sm_count // 2 — maximum split-K partitions (compile-time).
        fold_shared_index: Read-once shared-index fold (KERN-3141); relaxes the
            per-split page floor to 1 for folded shapes (compile-time).

    Args:
        batch_size: Current batch size.
        effective_max_cache_len: Max KV cache length.
        q_max_seq_len: Max query sequence length (1 for decode).
        split_page_size: Page granularity for split-K (64 or 128).
        sm_count: Number of SMs on the target GPU.
        relax_split_floor: When True, drop the per-split page floor to 1 so np
            tracks the effective KV page count (~one page per split). Set by the
            q_len=1 sparse FP8 split-K tuning (KERN-3217); the gating predicate
            lives at the dispatch call site. Default False retains the previous
            partition-selection logic (the production floor of 4).
        cost_optimal_split: When True, choose the split minimising modelled wall
            time (`_cost_optimal_partition_bucket`) instead of the wave target,
            page floor and bucket rounding below. Set only for the unfolded
            sparse spec-decode verify; the gating predicate lives at the
            dispatch call site, which alone sees whether the launch folds.

    Returns:
        The number of split-K partitions.
    """
    var num_kv_cache_pages = ceildiv(effective_max_cache_len, split_page_size)

    if cost_optimal_split:
        # The gate at the call site fires only for a launch that does not fold
        # q into the M tile, so grid_y is q_max_seq_len and every q position
        # costs its own CTA — which is exactly what the shape-keyed fold below
        # gets wrong for this launch.
        return _cost_optimal_partition_bucket[half_sms](
            ceildiv(num_heads, 64) * q_max_seq_len * batch_size,
            effective_max_cache_len,
            sm_count,
            min(half_sms, num_kv_cache_pages),
        )

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

    # Pages-per-CTA lower bound to avoid under-utilizing CTAs (per-CTA setup
    # + combine-grid overhead would dominate if each CTA processed too few
    # pages).
    comptime _min_pages_per_split = 4
    # Read-once shared-index fold (KERN-3141): grid_y=1 leaves split-K as the
    # only SM-fill lever, so drop the per-split page floor to 1 for the folded
    # shapes (matches FlashInfer's ~18-way split of the topk domain). Off ->
    # the production floor of 4, byte-identical.
    # Intentionally coarser than the launch-selection gate (_fold_ok also
    # excludes extra_kv/variable_topk/attn_sink): num_partitions is a tuning
    # knob, so a relaxed floor on a fallen-back unfolded launch is
    # correctness-neutral.
    # relax_split_floor additionally relaxes the floor for the unfolded q_len=1
    # sparse FP8 decode (KERN-3217), which is SM-underfilled at the floor of 4;
    # the caller has already clamped effective_max_cache_len to the kernel's
    # min(topk, cache+q) bound so np tracks the real page count (one per split).
    var _min_pages = 1 if (
        (fold_shared_index and fold_active) or relax_split_floor
    ) else _min_pages_per_split
    var max_np_for_min_pages = num_kv_cache_pages // _min_pages

    # Policy: honor wave-aligned target_partitions, but cap np DOWN if it
    # would leave too few pages per CTA (under-utilization). Long K-loops
    # per CTA are FINE — they amortize fixed costs better than splitting.
    var num_partitions: Int = min(target_partitions, max_np_for_min_pages)

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
    # Read-once shared-index fold (KERN-3141); relaxes the 64-head split floor.
    fold_shared_index: Bool = False,
](
    batch_size: Int,
    effective_max_cache_len: Int,
    q_max_seq_len: Int,
    split_page_size: Int,
    sm_count: Int,
    relax_split_floor: Bool = False,
    cost_optimal_split: Bool = False,
) -> Int:
    """Routing function that dispatches to head-count-specific heuristics.

    Single head group (num_heads <= 64, e.g. Kimi K2.5) calls
    _compute_num_partitions_64.  Multiple head groups (num_heads > 64,
    e.g. DeepSeek V3/R1) calls _compute_num_partitions_128.

    Parameters:
        num_heads: Number of Q attention heads (compile-time).
        is_fp8_kv: Whether the KV cache is FP8 (compile-time).
        half_sms: sm_count // 2 — maximum split-K partitions (compile-time).
        fold_shared_index: Read-once shared-index fold (KERN-3141); relaxes the
            per-split page floor for folded shapes (compile-time).

    Args:
        batch_size: Current batch size.
        effective_max_cache_len: Max KV cache length.
        q_max_seq_len: Max query sequence length (1 for decode).
        split_page_size: Page granularity for split-K (64 or 128).
        sm_count: Number of SMs on the target GPU.
        relax_split_floor: When True, drop the per-split page floor to 1 in the
            single-head-group heuristic so np tracks the effective KV page count
            (KERN-3217 q_len=1 sparse FP8 split-K tuning; the gating predicate
            lives at the dispatch call site). Default False retains the previous
            partition-selection logic (the production floor); ignored by the
            multi-head-group (128-head) path.
        cost_optimal_split: Choose the split minimising modelled wall time; see
            `_compute_num_partitions_64`. Ignored by the multi-head-group
            (128-head) path.

    Returns:
        The number of split-K partitions.
    """
    comptime _head_groups = ceildiv(num_heads, 64)

    comptime if _head_groups == 1:
        return _compute_num_partitions_64[
            num_heads, is_fp8_kv, half_sms, fold_shared_index
        ](
            batch_size,
            effective_max_cache_len,
            q_max_seq_len,
            split_page_size,
            sm_count,
            relax_split_floor=relax_split_floor,
            cost_optimal_split=cost_optimal_split,
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
    # Read-once shared-index fold (KERN-3141); relaxes the split-K page floor.
    fold_shared_index: Bool = False,
](
    batch_size: Int,
    max_cache_valid_length: Int,
    q_max_seq_len: Int,
    sm_count: Int,
) -> Tuple[Int, Int, Int]:
    """Pure computation of the packed MLA dispatch metadata.

    Returns ``(batch_size, q_max_seq_len, num_partitions)``.
    These three values are baked into the size-3 GPU buffer.
    ``effective_split_len`` is computed directly inside the MoGG ops from
    ``max_cache_length`` (``max_cache_valid_length + q_max_seq_len`` when
    ``_is_cache_length_accurate=False``, else ``max_cache_valid_length``),
    and no longer needs to be returned here.
    """
    var effective = max_cache_valid_length

    comptime if not _is_cache_length_accurate:
        effective += q_max_seq_len

    var split_page_size = 64 if (effective <= 512 and batch_size >= 32) else 128
    var num_partitions = _compute_num_partitions[
        num_heads, is_fp8_kv, half_sms, fold_shared_index
    ](batch_size, effective, q_max_seq_len, split_page_size, sm_count)

    return (batch_size, q_max_seq_len, num_partitions)


def _dispatch_scalars_for_heads[
    num_heads: Int
](
    batch_size: Int,
    max_cache_valid_length: Int,
    q_max_seq_len: Int,
    is_fp8_kv: Bool,
    sm_count: Int,
) -> Tuple[Int, Int, Int]:
    """Binds `is_fp8_kv` at comptime for one already-bound head count."""
    if is_fp8_kv:
        return compute_mla_dispatch_scalars[num_heads, is_fp8_kv=True](
            batch_size, max_cache_valid_length, q_max_seq_len, sm_count
        )
    return compute_mla_dispatch_scalars[num_heads](
        batch_size, max_cache_valid_length, q_max_seq_len, sm_count
    )


def compute_mla_dispatch_scalars_runtime(
    batch_size: Int,
    max_cache_valid_length: Int,
    q_max_seq_len: Int,
    num_heads: Int,
    is_fp8_kv: Bool,
    sm_count: Int,
) raises -> Tuple[Int, Int, Int]:
    """Binds a runtime Q-head count to a comptime dispatch instantiation.

    Every count here is one the kernel is measured correct at; 12, 24 and 48
    are Kimi K3 at TP8/TP4/TP2, and 96 is K3 whole at TP1. The list is not
    exhaustive — a count the kernel supports but that no model asks for yet is
    left out rather than paying its instantiation, so add an arm when a model
    needs one.
    """
    if num_heads == 8:
        return _dispatch_scalars_for_heads[8](
            batch_size,
            max_cache_valid_length,
            q_max_seq_len,
            is_fp8_kv,
            sm_count,
        )
    if num_heads == 12:
        return _dispatch_scalars_for_heads[12](
            batch_size,
            max_cache_valid_length,
            q_max_seq_len,
            is_fp8_kv,
            sm_count,
        )
    if num_heads == 16:
        return _dispatch_scalars_for_heads[16](
            batch_size,
            max_cache_valid_length,
            q_max_seq_len,
            is_fp8_kv,
            sm_count,
        )
    if num_heads == 24:
        return _dispatch_scalars_for_heads[24](
            batch_size,
            max_cache_valid_length,
            q_max_seq_len,
            is_fp8_kv,
            sm_count,
        )
    if num_heads == 32:
        return _dispatch_scalars_for_heads[32](
            batch_size,
            max_cache_valid_length,
            q_max_seq_len,
            is_fp8_kv,
            sm_count,
        )
    if num_heads == 48:
        return _dispatch_scalars_for_heads[48](
            batch_size,
            max_cache_valid_length,
            q_max_seq_len,
            is_fp8_kv,
            sm_count,
        )
    if num_heads == 64:
        return _dispatch_scalars_for_heads[64](
            batch_size,
            max_cache_valid_length,
            q_max_seq_len,
            is_fp8_kv,
            sm_count,
        )
    if num_heads == 96:
        return _dispatch_scalars_for_heads[96](
            batch_size,
            max_cache_valid_length,
            q_max_seq_len,
            is_fp8_kv,
            sm_count,
        )
    if num_heads == 128:
        return _dispatch_scalars_for_heads[128](
            batch_size,
            max_cache_valid_length,
            q_max_seq_len,
            is_fp8_kv,
            sm_count,
        )
    raise Error(
        String(
            t"Unsupported MLA num_heads for direct dispatch metadata binding: "
            t"{num_heads}"
        )
    )


# Cache forms of a sparse decode for `compute_mla_dispatch_key_runtime`.
# Plain is a cache without scales: unit-scale FP8 or BF16, by `is_fp8_kv`.
comptime MLA_SPARSE_KV_FORM_PLAIN = 0
# An FP8 cache with block scales.
comptime MLA_SPARSE_KV_FORM_SCALED_FP8 = 1
# A packed NVFP4 cache (an index row holds the value rows, then the scale rows).
comptime MLA_SPARSE_KV_FORM_NVFP4 = 2


def _sparse_graph_launch[
    num_heads: Int,
    q_type: DType,
    kv_type: DType,
    kv_quantized: Bool,
    kv_nvfp4: Bool,
    fixed_topk: Bool,
    half_sms: Int = 74,
    physical_indices: Bool = False,
    depth: Int = 0,
](
    batch_size: Int,
    max_cache_valid_length: Int,
    q_max_seq_len: Int,
    indices_stride: Int,
    page_size: Int,
    sm_count: Int,
) -> Tuple[Int, Int]:
    """Returns the split count and kernel a sparse graph op's decode launches.

    This is `mla_decode_sm100_dispatch`'s pick (`_mla_decode_split` and
    `_sparse_pair_launches`, the helpers the dispatch calls) for what the
    graph ops pass: plain decode or verify (no `skip_masked_tiles`), no
    extra KV, an inaccurate cache length, no sliding window and an unfolded
    launch. The graph ops do not forward their dispatch scalar on SM100, so
    the capture key must follow this launch rather than set it.

    Parameters:
        num_heads: Number of Q attention heads (compile-time).
        q_type: The Q dtype the graph op's kernel reads.
        kv_type: The KV operand's dtype.
        kv_quantized: Whether the KV operand carries scales.
        kv_nvfp4: Whether the KV operand is packed NVFP4.
        fixed_topk: Whether the op passes neither per-token top-k lengths
            nor an attention sink.
        half_sms: sm_count // 2 -- maximum split-K partitions (compile-time).
        physical_indices: Whether sparse rows are physical. NVFP4 always uses
            physical rows; plain FP8 defaults to logical rows.
        depth: Logical KV head dimension; zero (unknown) disables the 576-only
            Layout E records.

    Args:
        batch_size: Current batch size.
        max_cache_valid_length: Max valid KV-cache length across the batch.
        q_max_seq_len: Max query sequence length.
        indices_stride: The sparse index row stride.
        page_size: The KV cache page size.
        sm_count: Number of SMs on the target GPU.

    Returns:
        The split count and variant: 0 single64, 1 pair128, 2 single128.
    """
    var logical_topk = indices_stride // (2 if kv_nvfp4 else 1)
    var split = _mla_decode_split[
        q_type,
        kv_type,
        kv_quantized,
        kv_nvfp4,
        num_heads,
        half_sms,
        sparse=True,
        _is_cache_length_accurate=False,
        per_token_scale_rope_aware=False,
        fold_shared_index=False,
        has_extra_k=False,
        fixed_topk=fixed_topk,
        physical_indices=physical_indices,
        depth=depth,
        skip_masked_tiles=False,
    ](
        batch_size,
        q_max_seq_len,
        max_cache_valid_length,
        indices_stride,
        0,
        page_size,
        sm_count,
    )
    return _select_sparse_layout_e_launch[
        q_type,
        kv_type,
        kv_quantized,
        kv_nvfp4,
        num_heads,
        half_sms,
        fixed_topk=fixed_topk,
        physical_indices=physical_indices,
        depth=depth,
    ](
        batch_size,
        q_max_seq_len,
        _sparse_pair_keys(
            logical_topk, 0, max_cache_valid_length + q_max_seq_len
        ),
        _launched_split_page(split[1], page_size),
        sm_count,
        split[0],
        split[0],
        max_cache_valid_length + q_max_seq_len,
    )


def _sparse_graph_launch_fixed_impl[
    num_heads: Int,
    q_type: DType,
    kv_type: DType,
    kv_quantized: Bool,
    kv_nvfp4: Bool,
    physical_indices: Bool,
    depth: Int,
](
    fixed_topk: Bool,
    batch_size: Int,
    max_cache_valid_length: Int,
    q_max_seq_len: Int,
    indices_stride: Int,
    page_size: Int,
    sm_count: Int,
) -> Tuple[Int, Int]:
    """Returns `_sparse_graph_launch` with `fixed_topk` bound at runtime."""
    if fixed_topk:
        return _sparse_graph_launch[
            num_heads,
            q_type,
            kv_type,
            kv_quantized,
            kv_nvfp4,
            True,
            physical_indices=physical_indices,
            depth=depth,
        ](
            batch_size,
            max_cache_valid_length,
            q_max_seq_len,
            indices_stride,
            page_size,
            sm_count,
        )
    return _sparse_graph_launch[
        num_heads,
        q_type,
        kv_type,
        kv_quantized,
        kv_nvfp4,
        False,
        physical_indices=physical_indices,
        depth=depth,
    ](
        batch_size,
        max_cache_valid_length,
        q_max_seq_len,
        indices_stride,
        page_size,
        sm_count,
    )


def _sparse_graph_launch_fixed[
    num_heads: Int,
    q_type: DType,
    kv_type: DType,
    kv_quantized: Bool,
    kv_nvfp4: Bool,
](
    fixed_topk: Bool,
    batch_size: Int,
    max_cache_valid_length: Int,
    q_max_seq_len: Int,
    indices_stride: Int,
    page_size: Int,
    sm_count: Int,
    physical_indices: Bool,
    depth: Int,
) -> Tuple[Int, Int]:
    # Row form and depth reach only the Layout E records, which serve FP8 Q
    # over a unit-scale FP8 or packed NVFP4 cache at 64 or 128 heads; every
    # other form launches the same way whatever they are.
    comptime layout_e_form = (
        q_type == DType.float8_e4m3fn
        and ((kv_type == DType.float8_e4m3fn and not kv_quantized) or kv_nvfp4)
        and (num_heads == 64 or num_heads == 128)
    )
    comptime if not layout_e_form:
        return _sparse_graph_launch_fixed_impl[
            num_heads, q_type, kv_type, kv_quantized, kv_nvfp4, False, 0
        ](
            fixed_topk,
            batch_size,
            max_cache_valid_length,
            q_max_seq_len,
            indices_stride,
            page_size,
            sm_count,
        )
    comptime for form in range(4):
        comptime physical = (form & 1) != 0
        comptime native_depth = 576 if (form & 2) != 0 else 0
        if physical_indices == physical and (depth == 576) == (
            native_depth == 576
        ):
            return _sparse_graph_launch_fixed_impl[
                num_heads,
                q_type,
                kv_type,
                kv_quantized,
                kv_nvfp4,
                physical,
                native_depth,
            ](
                fixed_topk,
                batch_size,
                max_cache_valid_length,
                q_max_seq_len,
                indices_stride,
                page_size,
                sm_count,
            )
    return (0, 0)


def _sparse_graph_launch_cache_form[
    num_heads: Int
](
    batch_size: Int,
    max_cache_valid_length: Int,
    q_max_seq_len: Int,
    is_fp8_kv: Bool,
    sparse_indices_stride: Int,
    sm_count: Int,
    kv_form: Int,
    page_size: Int,
    fixed_topk: Bool,
    physical_indices: Bool,
    depth: Int,
) raises -> Tuple[Int, Int]:
    def launch_as[
        q_type: DType, kv_type: DType, kv_quantized: Bool, kv_nvfp4: Bool
    ](physical: Bool) raises {imm} -> Tuple[Int, Int]:
        return _sparse_graph_launch_fixed[
            num_heads, q_type, kv_type, kv_quantized, kv_nvfp4
        ](
            fixed_topk,
            batch_size,
            max_cache_valid_length,
            q_max_seq_len,
            sparse_indices_stride,
            page_size,
            sm_count,
            physical,
            depth,
        )

    if kv_form == MLA_SPARSE_KV_FORM_NVFP4:
        # NVFP4 always gathers physical rows.
        return launch_as[DType.float8_e4m3fn, DType.uint8, True, True](True)
    elif kv_form == MLA_SPARSE_KV_FORM_SCALED_FP8:
        if not is_fp8_kv:
            raise Error("a scaled FP8 sparse cache must be FP8")
        # The graph ops keep Q in BF16 over a scaled FP8 cache.
        return launch_as[DType.bfloat16, DType.float8_e4m3fn, True, False](
            physical_indices
        )
    elif kv_form != MLA_SPARSE_KV_FORM_PLAIN:
        raise Error("unknown sparse MLA cache form: ", kv_form)
    elif is_fp8_kv:
        # The graph ops stage Q as FP8 over a unit-scale FP8 cache.
        return launch_as[
            DType.float8_e4m3fn, DType.float8_e4m3fn, False, False
        ](physical_indices)
    return launch_as[DType.bfloat16, DType.bfloat16, False, False](
        physical_indices
    )


def compute_mla_dispatch_key_runtime(
    batch_size: Int,
    max_cache_valid_length: Int,
    q_max_seq_len: Int,
    num_heads: Int,
    is_fp8_kv: Bool,
    sparse_indices_stride: Int,
    sm_count: Int,
    *,
    kv_form: Int = MLA_SPARSE_KV_FORM_PLAIN,
    page_size: Int = 128,
    fixed_topk: Bool = True,
    physical_indices: Bool = False,
    depth: Int = 0,
) raises -> Tuple[Int, Int, Int, Int]:
    """Returns `(batch_size, q_max_seq_len, num_partitions, kernel_variant)`,
    the capture key of a decode dispatch.

    A sparse decode keys on the launch `_sparse_graph_launch` predicts:
    its split count and kernel variant (0 for 64-key single CTA, 1 for
    128-key pair, and 2 for 128-key single CTA). Dense dispatches use
    `compute_mla_dispatch_scalars_runtime` with variant 0.

    Device graph capture records one graph per key at the longest cache that
    maps to it and replays it for every shorter cache with that key, so a
    key that groups caches the dispatch launches differently replays the
    longest one's launch. That is correct, since a split or kernel captured
    for more keys still covers fewer, but slower: the pair where the short
    walk should go direct, or a wider split than the dispatch picks.

    The key follows the `-D` knobs this function is built with
    (`MLA_SPARSE_CG2*`, see mla_decode_utils.mojo), not the kernels'.

    Args:
        batch_size: Current batch size.
        max_cache_valid_length: Max valid KV-cache length across the batch.
        q_max_seq_len: Max query sequence length.
        num_heads: Number of Q attention heads.
        is_fp8_kv: Whether the KV cache stores FP8.
        sparse_indices_stride: The sparse index row stride; 0 for dense.
        sm_count: Number of SMs on the target GPU.
        kv_form: The sparse cache form (`MLA_SPARSE_KV_FORM_*`).
        page_size: The KV cache page size.
        fixed_topk: Whether the sparse decode passes neither per-token top-k
            lengths nor an attention sink.
        physical_indices: Whether sparse rows are physical. NVFP4 always uses
            physical rows; plain FP8 defaults to logical rows.
        depth: Logical KV head dimension; zero (unknown) disables the 576-only
            Layout E records.

    Returns:
        The dispatch metadata and the kernel variant.

    Raises:
        If `kv_form` is not a `MLA_SPARSE_KV_FORM_*` value, or names a
        scaled FP8 form over a cache that is not FP8.
    """
    if sparse_indices_stride > 0:
        comptime head_counts = (8, 16, 24, 32, 40, 48, 56, 64, 128)
        comptime for i in range(9):
            comptime heads = head_counts[i]
            if num_heads == heads:
                var launch = _sparse_graph_launch_cache_form[heads](
                    batch_size,
                    max_cache_valid_length,
                    q_max_seq_len,
                    is_fp8_kv,
                    sparse_indices_stride,
                    sm_count,
                    kv_form,
                    page_size,
                    fixed_topk,
                    physical_indices,
                    depth,
                )
                return (batch_size, q_max_seq_len, launch[0], launch[1])
    var scalars = compute_mla_dispatch_scalars_runtime(
        batch_size,
        max_cache_valid_length,
        q_max_seq_len,
        num_heads,
        is_fp8_kv,
        sm_count,
    )
    return (scalars[0], scalars[1], scalars[2], 0)


struct MLADispatchScalarArgs[
    num_heads: Int,
    _is_cache_length_accurate: Bool = False,
    is_fp8_kv: Bool = False,
    # Read-once shared-index fold (KERN-3141); relaxes the split-K page floor so
    # the pre-computed num_partitions matches what the folded launch expects.
    fold_shared_index: Bool = False,
]:
    """Pre-computed MLA decode args for the legacy (non-capturable) path.

    Owns a GPU buffer containing ``[batch_size, q_max_seq_len, num_partitions]``
    and caches the host-side ``batch_size``/``q_max_seq_len`` pair needed by
    ``mla_decode_sm100_dispatch``.

    Usage::

        var args = MLADispatchScalarArgs[num_heads=128](
            batch_size, max_cache_len, q_max_seq_len, ctx,
        )
        var gpu_tt = args.gpu_tile_tensor()
        mla_decode_sm100_dispatch[...](
            ..., gpu_tt,
            args.batch_size, args.q_max_seq_len, max_cache_len,
            ctx,
        )
        _ = args  # keepalive
    """

    var gpu_buf: DeviceBuffer[.int64]
    var batch_size: Int
    var q_max_seq_len: Int

    def __init__(
        out self,
        batch_size: Int,
        max_cache_len: Int,
        q_max_seq_len: Int,
        ctx: DeviceContext,
    ) raises:
        self.gpu_buf = ctx.enqueue_create_buffer[.int64](3)
        self.batch_size = batch_size
        self.q_max_seq_len = q_max_seq_len

        comptime sm_count = ctx.default_device_info.sm_count
        comptime _half_sms = sm_count // 2
        var scalars = compute_mla_dispatch_scalars[
            num_heads=Self.num_heads,
            _is_cache_length_accurate=Self._is_cache_length_accurate,
            is_fp8_kv=Self.is_fp8_kv,
            half_sms=_half_sms,
            fold_shared_index=Self.fold_shared_index,
        ](batch_size, max_cache_len, q_max_seq_len, sm_count)

        # Note: scalars[3] (effective_split_len) is only consumed by the
        # capturable-graph dispatcher path, not by the legacy GPU buffer.
        var host_args: Array[Int64, 3] = [
            Int64(scalars[0]),
            Int64(scalars[1]),
            Int64(scalars[2]),
        ]
        var output_buf = DeviceBuffer[.int64](
            ctx, self.gpu_buf.unsafe_ptr(), 3, owning=False
        )
        output_buf.enqueue_copy_from(
            Pointer(to=host_args).unsafe_bitcast[Int64]()
        )

    def gpu_tile_tensor(
        self,
    ) -> TileTensor[.int64, RowMajorLayout[ComptimeInt[3]], MutAnyOrigin]:
        return TileTensor(
            rebind[MutPointer[Int64, MutAnyOrigin]](self.gpu_buf.unsafe_ptr()),
            row_major((Idx[3],)),
        )


# ------------------------------------------------------------------------------
# MLA decoding implementation for SM100
# ------------------------------------------------------------------------------
def _sliding_window_size[mask_t: MHAMask]() -> Int:
    """Returns `mask_t`'s sliding window, or -1 when it has none."""
    comptime if mask_t.get_type_name() == "SlidingWindowCausalMask":
        return mask_t.sliding_window_size()
    return -1


def _launched_split_page(split_page_size: Int, kv_page_size: Int) -> Int:
    """Returns the page a decode launch splits on: `split_page_size` (64 or
    128) over pages of 128 or more, else the KV page itself, capped at 64.
    """
    if kv_page_size > 0 and kv_page_size < 128:
        return min(kv_page_size, 64)
    return split_page_size


def _mla_decode_split[
    q_type: DType,
    kv_type: DType,
    kv_quantized: Bool,
    kv_nvfp4: Bool,
    num_heads: Int,
    _half_sms: Int,
    *,
    sparse: Bool,
    _is_cache_length_accurate: Bool,
    per_token_scale_rope_aware: Bool,
    fold_shared_index: Bool,
    has_extra_k: Bool,
    fixed_topk: Bool,
    skip_masked_tiles: Bool,
    sliding_window_size: Int = -1,
    physical_indices: Bool = True,
    depth: Int = 576,
](
    batch_size: Int,
    q_max_seq_len: Int,
    max_cache_valid_length: Int,
    indices_stride: Int,
    extra_indices_stride: Int,
    kv_page_size: Int,
    sm_count: Int,
) -> Tuple[Int, Int]:
    """Returns the split count and split page size `mla_decode_sm100_dispatch`
    picks, before a caller's `num_partitions_in` overrides the split.

    A positive `MLA_SPARSE_NUM_PARTITIONS` forces an existing split bucket
    before an explicit caller override. The dispatch and sparse capture key
    (`compute_mla_dispatch_key_runtime`) both call this, so the key cannot
    drift from the launch.

    Parameters:
        q_type: The Q dtype the kernel reads.
        kv_type: The KV operand's dtype (`uint8` for packed NVFP4).
        kv_quantized: Whether the KV operand carries scales.
        kv_nvfp4: Whether the KV operand is packed NVFP4 (`mla_kv_is_nvfp4`).
        num_heads: Number of Q attention heads.
        _half_sms: sm_count // 2 -- maximum split-K partitions.
        sparse: Whether this is a sparse decode.
        _is_cache_length_accurate: Whether the cache length already counts
            the query tokens.
        per_token_scale_rope_aware: Whether the per-token-scale rope-aware
            form runs.
        fold_shared_index: Whether the read-once shared-index fold runs.
        has_extra_k: Whether extra KV is supplied.
        fixed_topk: Whether the launch has neither per-token top-k lengths
            nor an attention sink.
        skip_masked_tiles: Whether this is prefill on decode.
        sliding_window_size: The sliding window, or -1 for none.
        physical_indices: Whether sparse rows are physical, which the pair's
            routing reads.
        depth: Logical KV head dimension, which the fitted records are keyed
            on.

    Args:
        batch_size: Current batch size.
        q_max_seq_len: Max query sequence length.
        max_cache_valid_length: Max valid KV-cache length across the batch.
        indices_stride: The sparse index row stride (twice the top-k for
            NVFP4).
        extra_indices_stride: The extra KV's index row stride.
        kv_page_size: The KV operand's page size, 0 if not paged.
        sm_count: Number of SMs on the target GPU.

    Returns:
        The split count, and the split page size before the KV page caps it
        (`_launched_split_page`).
    """
    # An NVFP4 index row is the value rows then the scale rows, so the top-k is
    # half its stride.
    comptime _sparse_row_parts = 2 if kv_nvfp4 else 1
    var topk_stride = indices_stride // _sparse_row_parts
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
        effective_split_len = topk_stride + extra_indices_stride

    # Sliding-window split-K shrink : cap the effective cache length at
    # window_size + q_max_seq_len so partition heuristics don't
    # over-split a region the kernel will skip anyway.
    comptime if sliding_window_size >= 0:
        var sw_cap: Int = sliding_window_size + q_max_seq_len
        # at high batch (bs>=64) with a small SW cap
        # (<=2048), shrinking the effective_split_len to sw_cap forces the
        # partition heuristic into np=1.
        if batch_size >= 64 and sw_cap <= 2048:
            sw_cap = 2048
        effective_split_len = min(effective_split_len, sw_cap)

    var use_small_split_pages = effective_split_len <= 512 and batch_size >= 32
    var split_page_size = 64 if use_small_split_pages else 128
    comptime _is_fp8_kv = (kv_type == DType.float8_e4m3fn or kv_nvfp4)

    # q_len=1 sparse FP8 split-K tuning (KERN-3217): the single-token sparse
    # FP8 decode with no extra_kv / variable_topk / attn_sink is SM-underfilled
    # under the default per-split page floor of 4 (e.g. bs8 launches 4*8 = 32
    # CTAs on 148 SMs, ~0.2 waves). Scoped to the validated GLM-5.2 B200 regime:
    # q_len == 1, batch_size <= 8, configured top-k == 2048 (indices_stride is
    # the fixed top-k at this site; `not topk_lengths` confirms it is fixed, not
    # variable), and RAW cache length max_cache_valid_length in [1024, 2048]
    # (the raw dispatch arg, NOT effective_max_cache_len, so the guard bounds
    # the scope, not the clamp). In scope: relax the floor to 1 so np tracks the
    # effective KV page count (~one page per split), and clamp the split length
    # to the kernel's own bound min(topk, cache+q) so np never over-splits when
    # topk exceeds the live cache (the kernel clamps topk to actual_tokens =
    # cache_length + seq_len; see OffsetPosition in mla_decode_utils.mojo). This
    # is the unfolded-q1 analog of the fold_shared_index floor relax above.
    # Out of scope -- batch_size > 8, q_len > 1, top-k != 2048, cache outside
    # [1024, 2048], dense, BF16, and every unsupported sparse feature -- the
    # guard is false, so the floor stays and the split length is
    # effective_split_len, except that prefill on decode still clamps the
    # length (below). Layout E at up to 64 heads widens the scope
    # (`_relax_sparse_split_floor`).
    var _relax_q1_sparse_floor = (
        sparse
        and _is_fp8_kv
        and extra_indices_stride == 0
        and fixed_topk
        and _relax_sparse_split_floor[num_heads](
            batch_size, q_max_seq_len, topk_stride, max_cache_valid_length
        )
    )
    # Prefill on decode is never captured, so its split may track the cache:
    # no token sees more than cache + q keys, however wide the top-k stride.
    # Plain decode keeps the shape-pure split graph replay relies on.
    var _clamp_prefill = False
    comptime if sparse and skip_masked_tiles:
        _clamp_prefill = q_max_seq_len > MAX_FOLD_Q
    var _np_split_len = (
        min(
            effective_split_len, effective_max_cache_len
        ) if _relax_q1_sparse_floor
        or _clamp_prefill else effective_split_len
    )
    # Unfolded sparse spec-decode verify, where the shape-keyed fold below
    # reports a fold that only `fold_shared_index` actually performs, so the
    # split is sized for a grid `q_max_seq_len` times smaller than the one
    # launched. Sound here because sparse np comes from the top-k stride
    # rather than the cache length, so it stays a pure function of the shape
    # the caller already specialises on; the rest is scope.
    var _unfolded_verify = (
        sparse
        and _is_fp8_kv
        and not fold_shared_index
        and q_max_seq_len > 1
        and q_max_seq_len <= MAX_FOLD_Q
        and extra_indices_stride == 0
    )
    var _cost_optimal_split = (
        _unfolded_verify and num_heads * q_max_seq_len <= 64
    )
    # The routed sparse kernel's own cost record, where one is fitted: the
    # half-row Layout G kernel, in the verify regime above. Its record counts
    # the CTAs actually launched, so it needs no row cap: at 16 heads and
    # q = 6 the legacy split takes two at B = 16 to 20, 1.35x slower than one.
    comptime _layout_g_cost_route = (
        sparse
        and q_type == .float8_e4m3fn
        and _is_fp8_kv
        and (not kv_quantized or kv_nvfp4)
        and _routes_half_row_layout_g[num_heads]()
    )
    comptime _family = _sparse_cost_family[
        num_heads, q_type, kv_type
    ]() if sparse and not kv_quantized else 0
    # The 128-head Layout E FP8 record sizes its split from the top-k. Fed
    # the relaxed clamp's 1025 to 1152 keys (caches of 1024 to 1151) it picks
    # 9 one-page splits, where the top-k's 8 run 0.92x to 0.97x as long at
    # B = 5 to 8 and within round-to-round noise at B = 1 to 4 (q = 1, B200,
    # caches 1024 to 1088 measured). The BF16-Q record over an FP8 cache was
    # not measured without the clamp, so it keeps it.
    var _family_split_len = _np_split_len
    comptime if num_heads > 64 and _family == 2:
        if not _clamp_prefill:
            _family_split_len = effective_split_len
    # The page the kernel splits on (see `launch_impl` below).
    var cost_page = _launched_split_page(split_page_size, kv_page_size)
    var num_partitions: Int
    if _layout_g_cost_route and _unfolded_verify:
        comptime _top_k_record = (
            num_heads == 8 and fixed_topk and not has_extra_k
        )
        num_partitions = _layout_g_partition_bucket[_half_sms](
            ceildiv(num_heads, 32) * q_max_seq_len * batch_size,
            min(effective_split_len, effective_max_cache_len),
            cost_page,
            sm_count,
            _half_sms,
            top_k=topk_stride if _top_k_record else 0,
            cache_len=max_cache_valid_length,
        )
    elif _family != 0 and not fold_shared_index and extra_indices_stride == 0:
        num_partitions = _sparse_partition_bucket[_half_sms, _family](
            ceildiv(num_heads, 64) * q_max_seq_len * batch_size,
            _family_split_len,
            cost_page,
            sm_count,
            _half_sms,
        )
    else:
        num_partitions = _compute_num_partitions[
            num_heads, _is_fp8_kv, _half_sms, fold_shared_index
        ](
            batch_size,
            _np_split_len,
            q_max_seq_len,
            split_page_size,
            sm_count,
            relax_split_floor=_relax_q1_sparse_floor,
            cost_optimal_split=_cost_optimal_split,
        )

    # `_relax_q1_sparse_floor` holds only at top-k 2048.
    comptime if (
        q_type == .float8_e4m3fn
        and _is_fp8_kv
        and (not kv_quantized or kv_nvfp4)
        and _sparse_layout_g_mode[num_heads]() == 0
        and num_heads <= 64
        and not has_extra_k
    ):
        if _relax_q1_sparse_floor:
            num_partitions = _compact_partition_bucket[_half_sms](
                num_partitions, _np_split_len, cost_page
            )

    # Prefill on decode splits only as far as one wave holds: a second wave
    # costs a full CTA, while a split saves only part of one.
    comptime if sparse and skip_masked_tiles:
        num_partitions = _one_wave_partition_bucket[_half_sms](
            ceildiv(num_heads, 64) * q_max_seq_len * batch_size,
            _np_split_len,
            cost_page,
            sm_count,
            _half_sms,
        )

    # The BF16 pair's plain decode splits only as far as one wave holds, as
    # the masked route above does. Its fitted record trades waves for balance,
    # which at 96 to 192 CTAs buys a second half-wave and a combine launch
    # that outweigh the balance: (8, 6, 32768) reads 34.9 us at one split
    # against 44.7 at the record's three (five paired rounds), and (16, 6) is
    # 9% to 15% cheaper the same way. The short-walk clamp below still wins
    # over this.
    #
    # Only BF16: the FP8 pair was measured the other way. Across 12 underfilled
    # cells (batch 1-4, q 1-4) the record's 8-way split is 4.9% faster than
    # this rule's 16 (median of paired rounds, 5.5% and 6.3% on the two
    # tightest spreads), so FP8 keeps the record.
    comptime _bf16_pair_form = (
        _MLA_SPARSE_CG2_BF16 != 0
        and q_type == .bfloat16
        and kv_type == .bfloat16
        and _sparse_pair_eligible_form[
            q_type,
            kv_type,
            kv_quantized,
            kv_nvfp4,
            num_heads,
            per_token_scale_rope_aware,
            fold_shared_index,
        ]()
    )
    comptime if (
        sparse
        and not skip_masked_tiles
        and _MLA_SPARSE_CG2_ROUTE != 0
        and _bf16_pair_form
    ):
        num_partitions = _one_wave_partition_bucket[_half_sms](
            ceildiv(num_heads, 64) * q_max_seq_len * batch_size,
            _np_split_len,
            cost_page,
            sm_count,
            _half_sms,
        )

    # Short pair-eligible launches do not split (`_SPARSE_PAIR_NO_SPLIT_KEYS`).
    # This reads the cache length on plain decode too. The sparse graph ops
    # pass no split on SM100, so a capture bakes this one; their capture key
    # (`compute_mla_dispatch_key_runtime`) separates it and the route from
    # longer caches' launches, and each key is captured at the longest cache
    # that maps to it, so no key bakes one split for longer caches.
    comptime if (
        sparse
        and _MLA_SPARSE_CG2_ROUTE != 0
        and _sparse_pair_eligible_form[
            q_type,
            kv_type,
            kv_quantized,
            kv_nvfp4,
            num_heads,
            per_token_scale_rope_aware,
            fold_shared_index,
        ]()
    ):
        if (
            _sparse_pair_keys(
                topk_stride, extra_indices_stride, effective_max_cache_len
            )
            <= _SPARSE_PAIR_NO_SPLIT_KEYS
        ):
            num_partitions = 1
    comptime if sparse:
        var joint = _select_sparse_layout_e_launch[
            q_type,
            kv_type,
            kv_quantized,
            kv_nvfp4,
            num_heads,
            _half_sms,
            per_token_scale_rope_aware=per_token_scale_rope_aware,
            fold_shared_index=fold_shared_index,
            has_extra_k=has_extra_k,
            physical_indices=physical_indices,
            depth=depth,
            fixed_topk=fixed_topk,
            skip_masked_tiles=skip_masked_tiles,
            sliding_window_size=sliding_window_size,
        ](
            batch_size,
            q_max_seq_len,
            _sparse_pair_keys(
                topk_stride, extra_indices_stride, effective_max_cache_len
            ),
            cost_page,
            sm_count,
            num_partitions,
            0,
            effective_max_cache_len,
        )
        num_partitions = joint[0]

    # A split pays only while it fills the machine: past the first wave it adds
    # a whole CTA per extra split, and a bucket that leaves some splits empty
    # adds CTAs that do no work but still join the combine. The fitted records
    # trade balance for CTAs, which the BF16 pair's decode already overrode.
    # The forced-split sweep on B200 (top-k 2048, page 128, B 1-64, q 1-6,
    # caches 4096 and 32768) shows the blockwise-FP8 and NVFP4 forms need the
    # same fill: their heuristic oversplits h=128 cells by 10% to 25%
    # (B=4 q=2 and q=4, B=8 q=1 to q=4, B=16 q=1, B=32 q=1) and undersplits
    # h=64 B=4 q=4 and B=16 q=1 by 30%, while the one-wave bucket is within
    # noise of the best arm in every cell of either form. The FP8 unit records
    # already sit at the one-wave choice; raising their low-batch split costs
    # up to 8% (h=64 B=4 q=2, h=128 B=4 q=1), so they only take the cap.
    comptime _prod_fill_wave = num_heads >= 64 and (
        kv_nvfp4 or (kv_quantized and kv_type == DType.float8_e4m3fn)
    )
    # Below the top-k the live cache, not the top-k, bounds the walk, and the
    # existing split already tracks it; the one-wave rule is shape-pure only
    # once the cache covers the whole top-k.
    comptime if sparse and not skip_masked_tiles:
        if (
            effective_split_len >= 2048
            and effective_max_cache_len >= effective_split_len
        ):
            var production_ctas = (
                ceildiv(num_heads, 64) * q_max_seq_len * batch_size
            )
            var one_wave_np = _one_wave_partition_bucket[_half_sms](
                production_ctas,
                _np_split_len,
                cost_page,
                sm_count,
                _half_sms,
            )
            # NVFP4 B=1 q=6 loses 1.4% when the fill raises its 8-way split
            # to 16, so a B=1 NVFP4 launch keeps the smaller fitted split.
            if _prod_fill_wave and not (
                kv_nvfp4 and batch_size == 1 and one_wave_np > num_partitions
            ):
                num_partitions = one_wave_np
            else:
                if one_wave_np < num_partitions:
                    num_partitions = one_wave_np

    comptime forced_sparse_partitions = get_defined_int[
        "MLA_SPARSE_NUM_PARTITIONS", 0
    ]()
    comptime assert (
        forced_sparse_partitions >= 0
    ), "MLA_SPARSE_NUM_PARTITIONS must be nonnegative"
    comptime if sparse and forced_sparse_partitions > 0:
        num_partitions = forced_sparse_partitions
    return (num_partitions, split_page_size)


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
    # Read-once shared-index fold (KERN-3141): when the MTP-folded positions
    # share one identical topk list, gather it ONCE. Drives the sparse fp8
    # kernel's fold + the split-K floor relax. False -> unchanged baseline.
    fold_shared_index: Bool = False,
    # Whether `extra_k` is supplied. Comptime so callers that never pass it
    # (every graph op) do not build the extra-KV sparse kernels.
    has_extra_k: Bool = False,
    # Whether `d_indices` holds paged physical rows rather than token
    # positions (decoded in-kernel through `k`'s lookup table). Graph ops pass
    # token positions unless the indexer supplied physical rows; latent
    # sparse attention passes flat rows.
    physical_indices: Bool = True,
    # Presence of `attn_sink_ptr` / `topk_lengths` + `extra_topk_lengths`.
    AttnSinkPtrType: OptionalPointer = NullPointer[DType.float32],
    TopkLengthsPtrType: OptionalPointer = NullPointer[DType.int32],
    sparse_row: SparseRowLayout = SparseRowLayout(),
    skip_masked_tiles: Bool = False,
    fp8_scale_bf16: Bool = False,
    scale_block_size_override: Int = -1,
    prepared_bounds: Bool = False,
    # Presence of `lse_out` (inferred from it).
    LseOutPtrType: OptionalPointer = NullPointer[DType.float32],
](
    q: TileTensor[q_type, address_space=.GENERIC, ...],
    k: k_t,
    output: TileTensor[mut=True, output_type, address_space=.GENERIC, ...],
    scale: Float32,
    valid_length: TileTensor[.uint32, address_space=.GENERIC, ...],
    mask: mask_t,
    scalar_args_buf: NullableTileTensor[.int64, address_space=.GENERIC, ...],
    batch_size: Int,
    q_max_seq_len: Int,
    max_cache_valid_length: Int,
    ctx: DeviceContext,
    q_scale_ptr: OptionalReg[MutPointer[Float32, MutAnyOrigin]] = None,
    d_indices: OptionalReg[MutPointer[Int32, MutAnyOrigin]] = None,
    indices_stride: Int = 0,
    topk_lengths: TopkLengthsPtrType = null_pointer[TopkLengthsPtrType](),
    attn_sink_ptr: AttnSinkPtrType = null_pointer[AttnSinkPtrType](),
    # Extra KV parameters (forwarded to mla_decode_sm100_sink_split_k).
    extra_k: OptionalReg[k_t] = None,
    extra_d_indices: OptionalReg[MutPointer[Int32, MutAnyOrigin]] = None,
    extra_indices_stride: Int = 0,
    extra_topk_lengths: TopkLengthsPtrType = unread_pointer[
        TopkLengthsPtrType
    ](),
    extra_scales_ptr: OptionalReg[MutPointer[Float32, MutAnyOrigin]] = None,
    # Explicit host split override. Sparse QKV kernels read this launch pack
    # for their partition count; their scalar buffer supplies batch and Q size.
    num_partitions_in: Optional[Int] = None,
    scales_ptr_override: Optional[MutPointer[Float32, MutAnyOrigin]] = None,
    # Logical sparse indices for position-based causal masking; `None` keeps
    # the prior slot-count behavior. See mla_decode_utils.mojo.
    logical_indices: OptionalReg[MutPointer[Int32, MutAnyOrigin]] = None,
    # Positions between the token each sparse row was computed for and
    # the token that reads it.
    selection_lag: Int = 0,
    # Optional `[total_q_tokens, num_heads]` float32 output for each (token,
    # head)'s attention LSE in base 2 (`log2(sum(exp2(score * scale *
    # log2e)))`), written by the split-K combine or, unsplit, by the decode
    # kernel. A token with no keys gets `-inf` and an O of zero. Decode
    # context parallelism merges the ranks' partial attention with it.
    # Requires `skip_masked_tiles`, whose `topk_lengths` are per token.
    lse_out: LseOutPtrType = null_pointer[LseOutPtrType](),
    # With `lse_out`, an upper bound on any token's sparse top-k length that
    # sizes the split-K partitions in place of the index row stride (0 keeps
    # the stride). Correctness comes from the per-token lengths, so a loose
    # bound only costs performance.
    split_len_bound: Int = 0,
) raises:
    comptime if not LseOutPtrType.is_null:
        comptime assert sparse and skip_masked_tiles, (
            "the LSE output needs sparse decode with per-token lengths"
            " (skip_masked_tiles)"
        )
        comptime assert (
            AttnSinkPtrType.is_null
        ), "the LSE output does not support an attention sink"
        comptime assert (
            not fold_shared_index and not has_extra_k
        ), "the LSE output does not support the shared-index fold or extra KV"
        comptime assert (
            LseOutPtrType.dtype == DType.float32
        ), "the LSE output is float32"
    comptime assert not prepared_bounds or sparse
    comptime if not prepared_bounds:
        if not scalar_args_buf.ptr:
            raise Error("dynamic MLA dispatch requires device scalar metadata")
    comptime if skip_masked_tiles:
        if num_matrix_view_rows_decode(q) == 0:
            return

    if (extra_k is not None) != has_extra_k:
        raise Error(
            "mla_decode_sm100_dispatch: extra_k must be passed exactly when"
            " has_extra_k is True"
        )
    comptime assert scale_block_size_override >= -1
    comptime if scale_block_size_override > 0:
        comptime assert (
            sparse
            and q_type == .bfloat16
            and k_t.dtype == .float8_e4m3fn
            and not per_token_scale_rope_aware
        ), "external block scales require the sparse BF16 converter"
        comptime assert (
            scale_block_size_override % 16 == 0
            or scale_block_size_override >= depth
        )
    var scales_ptr = k.scales_raw_ptr()
    if case .Some(ptr) = scales_ptr_override:
        scales_ptr = ptr

    var effective_max_cache_len = max_cache_valid_length

    comptime if not _is_cache_length_accurate:
        effective_max_cache_len += q_max_seq_len

    comptime sm_count = ctx.default_device_info.sm_count
    comptime _half_sms = sm_count // 2
    # With the LSE output, a bound on the per-token lengths sizes the split
    # in place of the row stride (both halves of an NVFP4 row).
    var split_stride = indices_stride
    comptime if not LseOutPtrType.is_null:
        comptime _row_parts = 2 if mla_kv_is_nvfp4[k_t]() else 1
        if split_len_bound > 0:
            split_stride = min(indices_stride, split_len_bound * _row_parts)
    var _split = _mla_decode_split[
        q_type,
        k_t.dtype,
        k_t.quantization_enabled if (
            k_t.dtype != .float8_e4m3fn or scale_block_size_override < 0
        ) else scale_block_size_override
        > 0,
        mla_kv_is_nvfp4[k_t](),
        num_heads,
        _half_sms,
        sparse=sparse,
        _is_cache_length_accurate=_is_cache_length_accurate,
        per_token_scale_rope_aware=per_token_scale_rope_aware,
        fold_shared_index=fold_shared_index,
        has_extra_k=has_extra_k,
        fixed_topk=TopkLengthsPtrType.is_null and AttnSinkPtrType.is_null,
        skip_masked_tiles=skip_masked_tiles,
        sliding_window_size=_sliding_window_size[mask_t](),
        physical_indices=physical_indices,
        depth=depth,
    ](
        batch_size,
        q_max_seq_len,
        max_cache_valid_length,
        split_stride,
        extra_indices_stride,
        k_t.page_size,
        sm_count,
    )
    var num_partitions = _split[0]
    var use_small_split_pages = _split[1] == 64

    if case .Some(num_partitions_in) = num_partitions_in:
        num_partitions = num_partitions_in

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
    @inline(.always)
    def launch_impl[split_page_size_param: Int]() raises {imm}:
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
            fold_shared_index=fold_shared_index,
            has_extra_k=has_extra_k,
            physical_indices=physical_indices,
            sparse_row=sparse_row,
            skip_masked_tiles=skip_masked_tiles,
            fp8_scale_bf16=fp8_scale_bf16,
            scale_block_size_override=scale_block_size_override,
            prepared_bounds=prepared_bounds,
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
            logical_indices=logical_indices,
            selection_lag=selection_lag,
            lse_out=lse_out,
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
# Mirrors `FA4_WS_POISON` in `sm100/dispatch.mojo`: the split-K workspaces
# below are deliberately allocated without a fill, because an empty partition
# skips its O store while still writing `-inf` to its LSE slot, and
# `mla_decode_combine`'s scale check is what keeps the unwritten O slots from
# contributing. `mla_decode_combine.mojo`'s prefetch load therefore reads
# uninitialized memory by design and initcheck reports it. NaN-filling both
# workspaces initializes them (silencing the false report) and grades that
# contract: a dropped scale check or a missing `-inf` write propagates NaN into
# the output and trips the test's own assert. Off by default and
# comptime-pruned, so production codegen is untouched. Shares the FA4 flag so
# one `-D` covers every attention split-K workspace. See KERN-3535.
comptime MLA_WS_POISON: Bool = get_defined_int["FA4_WS_POISON", 0]() != 0


# Native-FP8 sparse decode routes on its softmax rows (`num_heads * fold_q`):
# at most 32 rows run the Layout G (MMA_M = 32) form of
# MLA_SM100_Decode_Sparse_QKV_FP8, more run Layout E.
def _sparse_layout_g_mode[rows: Int]() -> Int:
    """Returns the sparse native-FP8 form for `rows` softmax rows.

    0 is Layout E; 1 is Layout G.
    """
    return 1 if rows <= 32 else 0


def _relax_sparse_split_floor[
    num_heads: Int
](batch_size: Int, q_max_seq_len: Int, topk: Int, cache_len: Int) -> Bool:
    """Returns whether a plain sparse FP8 decode (no extra KV, variable top-k
    or sink) drops the split page floor to 1 (`relax_split_floor`).

    Parameters:
        num_heads: Number of Q attention heads.

    Args:
        batch_size: Current batch size.
        q_max_seq_len: Max query sequence length.
        topk: The fixed top-k (`indices_stride`).
        cache_len: The raw `max_cache_valid_length`.

    Returns:
        True when the relaxed floor applies.
    """
    if batch_size > 8 or topk != 2048 or cache_len < 1024:
        return False
    comptime if (_sparse_layout_g_mode[num_heads]() == 0 and num_heads <= 64):
        return q_max_seq_len <= 2
    return q_max_seq_len == 1 and cache_len <= 2048


def _routes_half_row_layout_g[rows: Int]() -> Bool:
    """Returns whether sparse native-FP8 decode with `rows` softmax rows runs
    the kernel `_LAYOUT_G_DIRECT_HALF_ROWS_COST` was fitted on: mode 1, the
    direct gather, 64-key tiles and the half-row softmax.
    """
    return _sparse_layout_g_mode[rows]() == 1 and rows <= 16


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
    # Read-once shared-index fold (KERN-3141); see mla_decode_sm100_dispatch.
    fold_shared_index: Bool = False,
    has_extra_k: Bool = False,
    # See mla_decode_sm100_dispatch.
    physical_indices: Bool = True,
    AttnSinkPtrType: OptionalPointer = NullPointer[DType.float32],
    TopkLengthsPtrType: OptionalPointer = NullPointer[DType.int32],
    sparse_row: SparseRowLayout = SparseRowLayout(),
    skip_masked_tiles: Bool = False,
    fp8_scale_bf16: Bool = False,
    scale_block_size_override: Int = -1,
    prepared_bounds: Bool = False,
    # Presence of `lse_out` (inferred from it).
    LseOutPtrType: OptionalPointer = NullPointer[DType.float32],
](
    q: TileTensor[q_type, address_space=.GENERIC, ...],
    k: k_t,
    output: TileTensor[mut=True, output_type, address_space=.GENERIC, ...],
    scale: Float32,
    valid_length: TileTensor[.uint32, address_space=.GENERIC, ...],
    mask: mask_t,
    scales_ptr: MutPointer[Float32, origin=MutAnyOrigin],
    scalar_args_buf: NullableTileTensor[.int64, address_space=.GENERIC, ...],
    batch_size: Int,
    q_max_seq_len: Int,
    num_partitions: Int,
    effective_max_cache_len: Int,
    ctx: DeviceContext,
    q_scale_ptr: OptionalReg[MutPointer[Float32, MutAnyOrigin]] = None,
    d_indices: OptionalReg[MutPointer[Int32, MutAnyOrigin]] = None,
    indices_stride: Int = 0,
    topk_lengths: TopkLengthsPtrType = null_pointer[TopkLengthsPtrType](),
    attn_sink_ptr: AttnSinkPtrType = null_pointer[AttnSinkPtrType](),
    # Extra KV parameters (forwarded to mla_decode_sm100_sink_split_k).
    extra_k: OptionalReg[k_t] = None,
    extra_d_indices: OptionalReg[MutPointer[Int32, MutAnyOrigin]] = None,
    extra_indices_stride: Int = 0,
    extra_topk_lengths: TopkLengthsPtrType = unread_pointer[
        TopkLengthsPtrType
    ](),
    extra_scales_ptr: OptionalReg[MutPointer[Float32, MutAnyOrigin]] = None,
    logical_indices: OptionalReg[MutPointer[Int32, MutAnyOrigin]] = None,
    # Positions between the token each sparse row was computed for and
    # the token that reads it.
    selection_lag: Int = 0,
    # See mla_decode_sm100_dispatch.
    lse_out: LseOutPtrType = null_pointer[LseOutPtrType](),
) raises:
    comptime hw_info = ctx.default_device_info
    comptime sm_count = hw_info.sm_count
    comptime _half_sms = sm_count // 2

    comptime AccumType = get_accum_type[output.dtype]()
    # The split-K output is the 512 latent (shared by both rows), not the row
    # width the model stores, so it stays 512 under NoPE.
    comptime v_depth = 512
    comptime _is_fp8_kv = (
        k_t.dtype == DType.float8_e4m3fn or mla_kv_is_nvfp4[k_t]()
    )

    # Ensure KV cache page_size is evenly divisible by split_page_size.
    # If the KV cache page_size shrinks in the future, splits must not
    # straddle physical page boundaries.
    comptime assert (
        k_t.page_size % split_page_size == 0
    ), "KV cache page_size must be divisible by split_page_size"

    var block_z = batch_size * num_partitions
    # The combine is a PDL dependent of the decode everywhere (main's
    # policy): the sparse split-K decode measured 2% to 10% faster on split
    # cells at 64 and 128 heads, and 3% to 9% for BF16 Q over FP8, and the
    # dense chain takes the same overlap.
    comptime _combine_pdl = sparse or _MLA_PDL_ALWAYS != 0

    if num_partitions > 1:
        comptime SplitAccumType = NonNullPointer[AccumType]
        # Create partial output buffer (same type as output - bfloat16)
        # Each split writes its partial attention result here
        # Note: Output dimension is v_depth (512), not depth (576)
        var o_accum_split_data = ctx.enqueue_create_buffer[output_type](
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
                num_partitions, batch_size, q_max_seq_len, num_heads, v_depth
            ),
        )
        # Create LSE accumulator buffer (AccumType = float32 for numerical stability)
        var lse_accum_data = ctx.enqueue_create_buffer[AccumType](
            Int(num_partitions * batch_size * q_max_seq_len * num_heads)
        )
        comptime if MLA_WS_POISON:
            ctx.enqueue_memset(o_accum_split_data, nan[output_type]())
            ctx.enqueue_memset(lse_accum_data, nan[AccumType]())
        var lse_accum_split = TileTensor(
            lse_accum_data,
            row_major(num_partitions, batch_size, q_max_seq_len, num_heads),
        )
        var lse_accum_split_ptr: SplitAccumType = {
            lse_accum_split.to_device_buffer(ctx)
            .unsafe_ptr()
            .as_imm()
            .as_unsafe_any_origin()
        }

        # Get input_row_offsets pointer for combine kernel's ragged output writes.
        var input_row_offsets_ptr = rebind[
            MutPointer[UInt32, origin=MutAnyOrigin]
        ](valid_length.ptr)

        def _launch_split_k_path() raises {imm}:
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
                sparse=sparse,
                fold_shared_index=fold_shared_index,
                has_extra_k=has_extra_k,
                physical_indices=physical_indices,
                sparse_row=sparse_row,
                skip_masked_tiles=skip_masked_tiles,
                fp8_scale_bf16=fp8_scale_bf16,
                scale_block_size_override=scale_block_size_override,
                prepared_bounds=prepared_bounds,
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
                logical_indices=logical_indices,
                selection_lag=selection_lag,
            )

            # Dispatch to specialized kernel based on num_partitions for compile-time unrolling.
            # Supports up to sm_count//2 splits to allow higher SM utilization.
            def launch_combine[n_splits: Int, wph: Int]() raises {imm}:
                mla_decode_combine_partial_outputs[
                    output_type=output_type,
                    accum_type=AccumType,
                    head_dim=v_depth,
                    num_splits=n_splits,
                    ragged=ragged,
                    warps_per_head=wph,
                    pdl=_combine_pdl,
                    early_loads=_combine_pdl,
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
                    lse_out,
                )

            def launch_combine_split_parallel[n_splits: Int]() raises {imm}:
                mla_decode_combine_partial_outputs[
                    output_type=output_type,
                    accum_type=AccumType,
                    head_dim=v_depth,
                    num_splits=n_splits,
                    ragged=ragged,
                    warps_per_head=1,
                    split_parallel=True,
                    pdl=_combine_pdl,
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

            def dispatch_combine[wph: Int]() raises {imm}:
                """Dispatch the combine kernel with the given warps_per_head,
                matching num_partitions to the correct compile-time bucket.

                Raises:
                    If the kernel dispatch fails, or if `num_partitions` is not
                    a compiled bucket.
                """
                var launched = False
                comptime for _b in range(_NUM_PARTITION_BUCKETS):
                    comptime if _get_partition_bucket[_half_sms, _b]() >= 2:
                        if (
                            num_partitions
                            == _get_partition_bucket[_half_sms, _b]()
                        ):
                            launch_combine[
                                _get_partition_bucket[_half_sms, _b](), wph
                            ]()
                            launched = True
                if not launched:
                    raise _unbucketed_split_error(num_partitions)

            def dispatch_combine_split_parallel() raises {imm}:
                """Dispatch the split-parallel combine kernel, matching
                num_partitions to the correct compile-time bucket.

                Raises:
                    If the kernel dispatch fails, or if `num_partitions` is not
                    a compiled bucket.
                """
                var launched = False
                comptime for _b in range(_NUM_PARTITION_BUCKETS):
                    comptime if _get_partition_bucket[_half_sms, _b]() >= 2:
                        if (
                            num_partitions
                            == _get_partition_bucket[_half_sms, _b]()
                        ):
                            launch_combine_split_parallel[
                                _get_partition_bucket[_half_sms, _b]()
                            ]()
                            launched = True
                if not launched:
                    raise _unbucketed_split_error(num_partitions)

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
            # Decision: dense uses split-parallel when cache_length >= 16384
            # and at most two sequences; sparse never does (below).
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
            #   dense, cache_len >= 16384, bs <= 2:                 split-parallel
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

            # Sparse never takes the split-parallel combine: at one or two
            # sequences over a 16K-32K cache it is 1.3x-1.8x slower than the
            # warp-per-head combine for 6-token decode at 64 and 128 heads,
            # and up to 2.5x for prompt chunks.
            if (
                not sparse
                and effective_max_cache_len >= 16384
                and batch_size <= 2
            ):
                dispatch_combine_split_parallel()
            elif (
                not _is_fp8_kv
                and combine_ctas_base >= 4096
                and num_partitions <= 4
                and effective_max_cache_len <= 1280
            ):
                # Comptime-gated so FP8 does not instantiate the wph=1 kernels.
                comptime if not _is_fp8_kv:
                    dispatch_combine[1]()
            elif combine_ctas_base >= 2048 and num_partitions > 4:
                dispatch_combine[2]()
            elif combine_ctas_base >= 512:
                dispatch_combine[4]()
            elif batch_size * _ctas_wph8 > sm_count * 2:
                dispatch_combine[4]()
            else:
                dispatch_combine[8]()

        _launch_split_k_path()
    else:
        # Unsplit, the LSE output (if any) takes the split-K workspace's slot,
        # which tells the kernel to write the final LSE per query token.
        comptime SplitAccumType = NullPointer[
            AccumType
        ] if LseOutPtrType.is_null else LseOutPtrType
        var lse_accum_split_ptr: SplitAccumType
        comptime if LseOutPtrType.is_null:
            lse_accum_split_ptr = rebind[SplitAccumType](
                NullPointer[AccumType]()
            )
        else:
            lse_accum_split_ptr = rebind[SplitAccumType](lse_out)

        def _launch_no_split_path() raises {imm}:
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
                sparse=sparse,
                fold_shared_index=fold_shared_index,
                has_extra_k=has_extra_k,
                physical_indices=physical_indices,
                sparse_row=sparse_row,
                skip_masked_tiles=skip_masked_tiles,
                fp8_scale_bf16=fp8_scale_bf16,
                scale_block_size_override=scale_block_size_override,
                prepared_bounds=prepared_bounds,
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
                logical_indices=logical_indices,
                selection_lag=selection_lag,
            )

        _launch_no_split_path()


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
    sparse: Bool = False,
    # Read-once shared-index MTP fold (KERN-3141); see mla_decode_sm100_dispatch.
    fold_shared_index: Bool = False,
    has_extra_k: Bool = False,
    # See mla_decode_sm100_dispatch.
    physical_indices: Bool = True,
    AttnSinkPtrType: OptionalPointer = NullPointer[DType.float32],
    TopkLengthsPtrType: OptionalPointer = NullPointer[DType.int32],
    sparse_row: SparseRowLayout = SparseRowLayout(),
    skip_masked_tiles: Bool = False,
    fp8_scale_bf16: Bool = False,
    scale_block_size_override: Int = -1,
    prepared_bounds: Bool = False,
](
    q: TileTensor[q_type, address_space=.GENERIC, ...],
    k: k_t,
    output: TileTensor[mut=True, address_space=.GENERIC, ...],
    lse_accum_split_ptr: SplitAccumType,
    scale: Float32,
    batch_size: Int,
    block_z: Int,
    num_partitions: Int,
    q_max_seq_len: Int,
    valid_length: TileTensor[.uint32, address_space=.GENERIC, ...],
    mask: mask_t,
    scales_ptr: MutPointer[Float32, origin=MutAnyOrigin],
    scalar_args_buf: NullableTileTensor[.int64, address_space=.GENERIC, ...],
    ctx: DeviceContext,
    q_scale_ptr: OptionalReg[MutPointer[Float32, MutAnyOrigin]] = None,
    d_indices: OptionalReg[MutPointer[Int32, MutAnyOrigin]] = None,
    indices_stride: Int = 0,
    topk_lengths: TopkLengthsPtrType = null_pointer[TopkLengthsPtrType](),
    attn_sink_ptr: AttnSinkPtrType = null_pointer[AttnSinkPtrType](),
    # Extra KV: separate always-attend cache. When extra_k is provided
    # (non-default), the sparse kernel appends extra_topk tokens after
    # the original topk tokens in a unified loop.
    extra_k: OptionalReg[k_t] = None,
    extra_d_indices: OptionalReg[Pointer[Int32, MutAnyOrigin]] = None,
    extra_indices_stride: Int = 0,
    extra_topk_lengths: TopkLengthsPtrType = unread_pointer[
        TopkLengthsPtrType
    ](),
    extra_scales_ptr: OptionalReg[Pointer[Float32, MutAnyOrigin]] = None,
    # Effective max cache length.  Layout G structural eligibility uses
    # `num_heads * q_len <= BM_G(32)`.  Defaults to 0 so unrelated callers
    # (BF16, sparse, etc.) pass through to the BN_QK=64 branch unchanged.
    effective_max_cache_len: Int = 0,
    # Logical sparse indices for position-based causal masking; `None` keeps
    # the prior slot-count behavior. See mla_decode_utils.mojo.
    logical_indices: OptionalReg[Pointer[Int32, MutAnyOrigin]] = None,
    # Positions between the token each sparse row was computed for and
    # the token that reads it.
    selection_lag: Int = 0,
) raises:
    # Packed NVFP4 KV is dequantized to FP8 in SMEM by the sparse kernel, so
    # its scales are not the blockwise-FP8 ones `_scale_block_size` tracks.
    comptime assert (
        _MLA_SPARSE_BN_QK == 0
        or _MLA_SPARSE_BN_QK == 64
        or _MLA_SPARSE_BN_QK == 128
    ), "MLA_SPARSE_BN_QK is 0, 64 or 128"
    comptime _nvfp4_kv = mla_kv_is_nvfp4[k_t]()
    comptime _scale_block_size = (
        scale_block_size_override if scale_block_size_override
        >= 0 else (
            k_t.quantization_granularity if k_t.quantization_enabled
            and not _nvfp4_kv else 0
        )
    )
    # Use native FP8 path when:
    # 1. KV is FP8 tensorwise (scale_block_size == 0), or packed NVFP4
    # 2. Q is also FP8 (q_type must match kv_type) — the pipeline provides FP8 Q
    # When Q is BF16, fall through to the old FP8 converter or BF16 path.
    comptime _native_fp8 = (
        (k_t.dtype == .float8_e4m3fn or _nvfp4_kv)
        and _scale_block_size == 0
        and q_type == .float8_e4m3fn
    )
    comptime assert not _nvfp4_kv or (
        _native_fp8 and sparse and not has_extra_k
    ), (
        "An NVFP4 KV cache needs FP8 Q and the sparse all-FP8 kernel, and has"
        " no extra-KV variant."
    )
    # Matching 16-bit Q and cache use the direct gather.
    comptime _bf16_sparse = (
        sparse
        and k_t.dtype == q_type
        and (q_type == .bfloat16 or q_type == .float16)
    )
    # BF16 Q over an FP8 cache (unit or blockwise scales) converts each
    # tile in place to BF16 and runs the BF16 form of the unified kernel.
    comptime _fp8_convert = (
        sparse and k_t.dtype == .float8_e4m3fn and q_type == .bfloat16
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
    # Comptime-local aliases so the nested `_launch_sparse_qkv_fp8_fold_sel`
    # closure can read these in its `comptime if` (nested closures capture
    # comptime locals, not the enclosing function's comptime params by bare
    # name).
    comptime _fold_shared_index = fold_shared_index
    comptime _sparse_row = sparse_row
    comptime _skip_masked_tiles = skip_masked_tiles
    comptime _has_attn_sink = not AttnSinkPtrType.is_null
    comptime _has_extra_k = has_extra_k
    comptime _physical_indices = physical_indices
    comptime _has_variable_topk = not TopkLengthsPtrType.is_null
    comptime _split_page_size = split_page_size
    comptime _sm_count = ctx.default_device_info.sm_count
    # Scoped to the sparse native-FP8 kernel ONLY: this same mla_config
    # object is also read by the dense native_fp8 branch below (`elif
    # _native_fp8:`) and by mla_config_g (Layout G); both keep their
    # existing single-buffer SW64-gather kernel bodies, so
    # native_fp8_unified_gather must be False there.
    comptime _native_fp8_unified_gather = sparse and (
        _native_fp8 or _bf16_sparse or _fp8_convert
    )
    # The NVFP4 form's linear gather slot: the tile's packed rows and
    # scales rather than a full FP8 stage. Full-size slots overrun the SMEM
    # carveout at 64 and 128 heads (the launcher's carveout assert fires,
    # so those head counts compile no decode form at all); the packed slot
    # fits them, and frees 14,336 B per slot at Layout G, where the config
    # spends it on a third gather stage. Behind `_MLA_NVFP4_PACKED_LINEAR`;
    # the knob-off build restores the full-size slot at both layouts
    # (Layout G's measured 2/3 rings, Layout E's failing carve). The linear
    # form uses 64 keys; 128-key NVFP4 tiles use the in-place ring instead.
    comptime _nvfp4_gather_stage = (
        64
        * (
            nvfp4_mla_row_bytes[depth]() + nvfp4_mla_scale_row_bytes[depth]()
        ) if _nvfp4_kv
        and _MLA_NVFP4_PACKED_LINEAR != 0 else 0
    )
    # V/output is the 512 latent. SMEM stays 576 so the trailing block P
    # aliases survives a NoPE row. Neither tracks the row the model stores.
    comptime _latent_depth = 512
    comptime _smem_q_depth = 576
    comptime mla_config = MLA_SM100_Decode_Config(
        num_q_heads=num_heads,
        group=group,  # num_q_heads/h_k(1)
        depth=_latent_depth,
        q_depth=_smem_q_depth,
        input_q_depth=depth,  # 576 with rotary, 512 NoPE
        dtype_size=_dtype_size,
        kv_type_size=size_of[k_t.dtype](),
        swizzle_mode=config.swizzle_mode,
        kv_mma_swizzle_mode=config.swizzle_mode,
        page_size=k_t.page_size,
        decoding_warp_split_k=decoding_warp_split_k,
        split_page_size=split_page_size,
        scale_block_size=_scale_block_size,
        fp8_scale_bf16=fp8_scale_bf16,
        native_fp8=_native_fp8,
        per_token_scale_rope_aware=_per_token_scale_rope_aware,
        native_fp8_unified_gather=_native_fp8_unified_gather,
        # A/B (`MLA_SPARSE_DIRECT_GATHER_E`): the Layout E FP8 form takes the
        # direct SW128 gather — Layout G's front end — instead of the linear
        # ring plus re-swizzle.
        direct_gather=(
            _bf16_sparse
            or (
                sparse
                and _native_fp8
                and not _nvfp4_kv
                and (
                    _MLA_SPARSE_DIRECT_GATHER_E != 0 or _MLA_SPARSE_BN_QK == 128
                )
            )
        ),
        kv_convert=_fp8_convert or (_nvfp4_kv and _MLA_SPARSE_BN_QK == 128),
        key_tile=(
            _MLA_SPARSE_BN_QK if sparse
            and _native_fp8
            and num_heads >= 64 else 0
        ),
        gather_stage_bytes=_nvfp4_gather_stage,
    )
    comptime _layout_e_models = _sparse_layout_e_models[
        _nvfp4_kv, num_heads, skip_masked_tiles
    ]()
    comptime _wide_available = (
        sparse
        and _native_fp8
        and num_heads >= 64
        and (
            _MLA_SPARSE_BN_QK == 128
            or (
                depth == 576
                and (num_heads == 64 or num_heads == 128)
                and (_layout_e_models.validated & 4) != 0
            )
        )
    )
    comptime mla_config_wide = MLA_SM100_Decode_Config(
        num_q_heads=num_heads,
        group=group,  # num_q_heads/h_k(1)
        depth=_latent_depth,
        q_depth=_smem_q_depth,
        input_q_depth=depth,  # 576 with rotary, 512 NoPE
        dtype_size=_dtype_size,
        kv_type_size=size_of[k_t.dtype](),
        swizzle_mode=config.swizzle_mode,
        kv_mma_swizzle_mode=config.swizzle_mode,
        page_size=k_t.page_size,
        decoding_warp_split_k=decoding_warp_split_k,
        split_page_size=split_page_size,
        scale_block_size=_scale_block_size,
        fp8_scale_bf16=fp8_scale_bf16,
        native_fp8=_native_fp8,
        per_token_scale_rope_aware=_per_token_scale_rope_aware,
        native_fp8_unified_gather=_native_fp8_unified_gather,
        direct_gather=_bf16_sparse
        or (sparse and _native_fp8 and not _nvfp4_kv),
        kv_convert=_fp8_convert or _nvfp4_kv,
        key_tile=(128 if sparse and _native_fp8 and num_heads >= 64 else 0),
        gather_stage_bytes=_nvfp4_gather_stage,
    )
    # Layout G (BM=MMA_M=32) form of the sparse native-FP8 kernel, in the
    # form `_sparse_layout_g_mode` routes these heads to. Every fold that
    # fits 32 rows takes the same form.
    comptime _sg_mode = _sparse_layout_g_mode[min(num_heads, 32)]()
    comptime _sg_direct = _sg_mode == 1
    comptime mla_config_sg = MLA_SM100_Decode_Config(
        num_q_heads=num_heads,
        group=group,
        depth=_latent_depth,
        q_depth=_smem_q_depth,
        input_q_depth=depth,
        dtype_size=_dtype_size,
        kv_type_size=size_of[k_t.dtype](),
        swizzle_mode=config.swizzle_mode,
        kv_mma_swizzle_mode=config.swizzle_mode,
        page_size=k_t.page_size,
        decoding_warp_split_k=decoding_warp_split_k,
        split_page_size=split_page_size,
        scale_block_size=_scale_block_size,
        fp8_scale_bf16=fp8_scale_bf16,
        native_fp8=_native_fp8,
        per_token_scale_rope_aware=_per_token_scale_rope_aware,
        decode_layout_g=True,
        native_fp8_unified_gather=_native_fp8_unified_gather,
        half_rows=True,
        direct_gather=(_sg_direct and not _fp8_convert and not _nvfp4_kv)
        or _bf16_sparse,
        kv_convert=_fp8_convert,
        gather_stage_bytes=_nvfp4_gather_stage,
    )
    # CTA pair (the default; `-D MLA_SPARSE_CG2=0` keeps these heads on the
    # unpaired direct kernel): the unit-scale Layout E FP8 form at 128 heads
    # runs as a (2, 1, 1) cluster whose leader issues cta_group::2 MMAs over
    # both CTAs' SMEM. See docs/plans/2026-10-19-sm100-sparse-mla-h128-pair.md.
    comptime _cg2 = _sparse_pair_eligible_form[
        q_type,
        k_t.dtype,
        k_t.quantization_enabled if (
            k_t.dtype != .float8_e4m3fn or scale_block_size_override < 0
        ) else scale_block_size_override
        > 0,
        _nvfp4_kv,
        num_heads,
        _per_token_scale_rope_aware,
        fold_shared_index,
    ]() or (
        _nvfp4_kv
        and _native_fp8
        and num_heads == 128
        and depth == 576
        and _MLA_SPARSE_CG2 != 0
        and _MLA_SPARSE_BN_QK != 64
        and physical_indices
        and not fold_shared_index
        and (_layout_e_models.validated & 2) != 0
    )
    comptime mla_config_cg2 = MLA_SM100_Decode_Config(
        num_q_heads=num_heads,
        group=group,
        depth=_latent_depth,
        q_depth=_smem_q_depth,
        input_q_depth=depth,
        dtype_size=_dtype_size,
        kv_type_size=size_of[k_t.dtype](),
        swizzle_mode=config.swizzle_mode,
        kv_mma_swizzle_mode=config.swizzle_mode,
        page_size=k_t.page_size,
        decoding_warp_split_k=decoding_warp_split_k,
        split_page_size=split_page_size,
        scale_block_size=_scale_block_size,
        fp8_scale_bf16=fp8_scale_bf16,
        native_fp8=_native_fp8,
        per_token_scale_rope_aware=False,
        native_fp8_unified_gather=_native_fp8_unified_gather,
        direct_gather=True,
        cta_group=2,
        gather_stage_bytes=_nvfp4_gather_stage,
    )
    var num_rows_q = num_matrix_view_rows_decode(q)
    # Cap only the grid. Scalar metadata and split workspaces retain the
    # padded bound required by graph replay and combine strides.
    var launch_q_max_seq_len = q_max_seq_len
    comptime if skip_masked_tiles and not fold_shared_index:
        launch_q_max_seq_len = min(q_max_seq_len, num_rows_q // num_heads)

    # `depth` and `BK` are the row as the model stores it (NoPE narrows both).
    var k_tma_op = k.create_tma_tile[
        BN=mla_config.BK_PV,  # tile_m =64
        depth=mla_config.input_q_depth,
        BK=mla_config.input_q_depth,
        swizzle_mode=mla_config.kv_tma_swizzle_mode,
    ](ctx)
    var o_ptr = rebind[MutPointer[Scalar[output_type], origin=MutAnyOrigin]](
        output.ptr
    )
    var num_rows_o = num_matrix_view_rows_decode(output)
    var o_tma_op = tma_tile_o[
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
        var q_ptr = rebind[MutPointer[Scalar[q_type], origin=MutAnyOrigin]](
            q.ptr
        )

        comptime assert (
            k_t.dtype != .bfloat16 and k_t.dtype != .float16
        ) or _bf16_sparse, "a sparse 16-bit cache needs matching Q"

        comptime assert _native_fp8 or _bf16_sparse or _fp8_convert, (
            "sparse decode takes FP8 Q over a tensorwise FP8 cache, matching"
            " 16-bit Q/cache, or BF16 Q over an FP8 cache"
        )
        var extra_k_val_fp8 = extra_k.or_else(k)
        var extra_kv_lut_val_fp8 = extra_k_val_fp8

        comptime if _native_fp8 or _bf16_sparse or _fp8_convert:
            # The launcher encodes the gather descriptors the kernel
            # reads (INT64 linear rows, or the cache dtype for the
            # direct gather), so a launch encodes only the two it uses.

            @inline(.always)
            def _launch_sparse_qkv_fp8[
                _fold_shared_index_val: Bool = False,
                _q_len_fold_val: Int = 1,
                _layout_g: Bool = False,
                _pair: Bool = True,
                _wide: Bool = False,
            ]() raises {imm}:
                comptime _cfg = (
                    mla_config_sg if _layout_g else (
                        mla_config_cg2 if _cg2
                        and _pair else (
                            mla_config_wide if _wide else mla_config
                        )
                    )
                )
                comptime if _cfg.cta_group == 2:
                    comptime assert (
                        _cfg.supported()
                    ), "unsupported paired sparse MLA config"
                var q_tma_sparse_fp8 = tma_tile_qo[
                    swizzle_mode=_cfg.q_swizzle(),
                    BM=_cfg.BM,
                    BK=_cfg.input_q_depth,
                    depth=_cfg.input_q_depth,
                ](ctx, q_ptr, num_rows_q)
                comptime if ragged:
                    comptime ValidLengthType = NonNullPointer[.uint32]
                    var valid_len: ValidLengthType = {valid_length}
                    launch_mla_sm100_decode_sparse_qkv_fp8[
                        q_type=q_type,
                        KVLUTType=k_t,
                        output_type=output_type,
                        SplitAccumType=SplitAccumType,
                        MaskType=mask_t,
                        config=mla_config,
                        config_g=_cfg,
                        ValidLengthType=ValidLengthType,
                        ragged=True,
                        _is_cache_length_accurate=_is_cache_length_accurate,
                        has_extra_kv=_has_extra_k,
                        physical_indices=_physical_indices,
                        fold_shared_index=_fold_shared_index_val,
                        q_len_fold=_q_len_fold_val,
                        sparse_row=_sparse_row,
                        skip_masked_tiles=_skip_masked_tiles,
                        prepared_bounds=prepared_bounds,
                    ](
                        q_tma_sparse_fp8,
                        o_tma_op,
                        k,
                        lse_accum_split_ptr,
                        scale,
                        batch_size,
                        block_z,
                        num_partitions,
                        launch_q_max_seq_len,
                        valid_len,
                        mask,
                        d_indices,
                        indices_stride,
                        topk_lengths,
                        scales_ptr,
                        attn_sink_ptr,
                        extra_kv_lut_val_fp8,
                        extra_d_indices,
                        extra_topk_lengths,
                        extra_indices_stride,
                        extra_scales_ptr,
                        scalar_args_buf,
                        ctx,
                        logical_indices=logical_indices,
                        selection_lag=selection_lag,
                        dispatch_cache_bound=effective_max_cache_len,
                        prepared_q_max_seq_len=q_max_seq_len,
                    )
                else:
                    comptime ValidLengthType = NullPointer[.uint32]
                    var valid_len: ValidLengthType = {}
                    launch_mla_sm100_decode_sparse_qkv_fp8[
                        q_type=q_type,
                        KVLUTType=k_t,
                        output_type=output_type,
                        SplitAccumType=SplitAccumType,
                        MaskType=mask_t,
                        config=mla_config,
                        config_g=_cfg,
                        ValidLengthType=ValidLengthType,
                        ragged=False,
                        _is_cache_length_accurate=_is_cache_length_accurate,
                        has_extra_kv=_has_extra_k,
                        physical_indices=_physical_indices,
                        fold_shared_index=_fold_shared_index_val,
                        q_len_fold=_q_len_fold_val,
                        sparse_row=_sparse_row,
                        skip_masked_tiles=_skip_masked_tiles,
                        prepared_bounds=prepared_bounds,
                    ](
                        q_tma_sparse_fp8,
                        o_tma_op,
                        k,
                        lse_accum_split_ptr,
                        scale,
                        batch_size,
                        block_z,
                        num_partitions,
                        launch_q_max_seq_len,
                        valid_len,
                        mask,
                        d_indices,
                        indices_stride,
                        topk_lengths,
                        scales_ptr,
                        attn_sink_ptr,
                        extra_kv_lut_val_fp8,
                        extra_d_indices,
                        extra_topk_lengths,
                        extra_indices_stride,
                        extra_scales_ptr,
                        scalar_args_buf,
                        ctx,
                        logical_indices=logical_indices,
                        selection_lag=selection_lag,
                        dispatch_cache_bound=effective_max_cache_len,
                        prepared_q_max_seq_len=q_max_seq_len,
                    )

            @inline(.always)
            def _launch_sparse_qkv_fp8_fold_sel() raises {imm}:
                comptime _fold_ok = (
                    _fold_shared_index
                    and (_native_fp8 or _fp8_convert)
                    and not _has_extra_k
                    and not _has_variable_topk
                    and not _has_attn_sink
                )
                comptime if _fold_ok:
                    comptime for n in range(MIN_FOLD_Q, MAX_FOLD_Q + 1):
                        comptime if (
                            _sparse_layout_g_mode[mla_config.num_q_heads * n]()
                            != 0
                        ):
                            if q_max_seq_len == n:
                                _launch_sparse_qkv_fp8[True, n, True]()
                                return
                        elif mla_config.num_q_heads * n <= mla_config.BM:
                            if q_max_seq_len == n:
                                _launch_sparse_qkv_fp8[True, n]()
                                return
                var selection = _select_sparse_layout_e_launch[
                    q_type,
                    k_t.dtype,
                    k_t.quantization_enabled if (
                        k_t.dtype != .float8_e4m3fn
                        or scale_block_size_override < 0
                    ) else scale_block_size_override
                    > 0,
                    _nvfp4_kv,
                    num_heads,
                    _sm_count // 2,
                    per_token_scale_rope_aware=_per_token_scale_rope_aware,
                    fold_shared_index=_fold_shared_index,
                    has_extra_k=_has_extra_k,
                    physical_indices=_physical_indices,
                    depth=depth,
                    fixed_topk=not _has_variable_topk and not _has_attn_sink,
                    skip_masked_tiles=_skip_masked_tiles,
                    sliding_window_size=_sliding_window_size[mask_t](),
                ](
                    batch_size,
                    q_max_seq_len,
                    _sparse_pair_keys(
                        indices_stride // (2 if _nvfp4_kv else 1),
                        extra_indices_stride,
                        effective_max_cache_len,
                    ),
                    _split_page_size,
                    _sm_count,
                    num_partitions,
                    num_partitions,
                    effective_max_cache_len,
                )
                comptime if _wide_available:
                    if selection[1] == 2:
                        _launch_sparse_qkv_fp8[_pair=False, _wide=True]()
                        return
                comptime if _cg2:
                    if selection[1] == 1:
                        _launch_sparse_qkv_fp8[_pair=True]()
                        return
                _launch_sparse_qkv_fp8[
                    _pair=False,
                    _layout_g=_sparse_layout_g_mode[mla_config.num_q_heads]()
                    != 0,
                ]()

            _launch_sparse_qkv_fp8_fold_sel()
            return

    # Per-token-scale rope-aware: split content (FP8) + rope (BF16) with separate TMAs.
    # Q buffer layout: FP8 content (512 bytes) | BF16 rope (128 bytes) per row = 640 bytes/row.
    # K cache layout: FP8 content (512 bytes) | BF16 rope (128 bytes) per row = 640 bytes/row.
    # The KV cache 640 bytes/row layout is enforced by create_rope_tma_tile in kv_cache/types.mojo.
    comptime if _per_token_scale_rope_aware:
        # Q row stride in FP8 bytes: 512 FP8 content + 64 BF16 rope = 640 bytes.
        # The `depth` parameter in tma_tile_qo sets the row stride of the
        # TileTensor, which the TMA descriptor uses as the global memory
        # stride.  It must equal the full row width so that consecutive
        # rows (heads/tokens) are read correctly.
        comptime _q_row_bytes = mla_config.padded_depth + mla_config.rope_depth * 2  # 640
        # Same stride in BF16 units for the rope TMA.
        comptime _q_row_bf16 = _q_row_bytes // 2  # 320

        # Q_nope TMA: FP8 content, SWIZZLE_64B, BM x padded_depth (512)
        var q_ptr_fp8_content = rebind[
            MutPointer[Float8_e4m3fn, origin=MutAnyOrigin]
        ](q.ptr)
        var q_nope_tma = tma_tile_qo[
            swizzle_mode=mla_config.content_swizzle_mode,  # SWIZZLE_64B
            BM=mla_config.BM,
            BK=mla_config.padded_depth,  # 512
            depth=_q_row_bytes,  # 640 (full row stride in FP8 bytes)
        ](ctx, q_ptr_fp8_content, num_rows_q)

        # Q_rope TMA: BF16 rope, SWIZZLE_128B, BM x rope_depth (64)
        # Rope starts at byte offset padded_depth (512) from Q row start.
        var q_ptr_bf16_rope = rebind[MutPointer[BFloat16, origin=MutAnyOrigin]](
            q.ptr.unsafe_offset(mla_config.padded_depth)
        )
        var q_rope_tma = tma_tile_qo[
            swizzle_mode=mla_config.rope_swizzle_mode,  # SWIZZLE_128B
            BM=mla_config.BM,
            BK=mla_config.rope_depth,  # 64
            depth=_q_row_bf16,  # 320 (full row stride in BF16 elements)
        ](ctx, q_ptr_bf16_rope, num_rows_q)

        # K_content TMA: FP8 content from KV cache, SWIZZLE_64B, BK_PV x padded_depth (512).
        # The KV cache has 640 bytes/row layout (512 FP8 content + 128 BF16 rope).
        # create_tma_tile reads only the first 512 bytes (FP8 content) per row.
        var k_content_tma = k.create_tma_tile[
            BN=mla_config.BK_PV,  # 64
            depth=mla_config.padded_depth,  # 512
            BK=mla_config.padded_depth,  # 512
            swizzle_mode=mla_config.content_swizzle_mode,  # SWIZZLE_64B
        ](ctx)

        # K_rope TMA: BF16 rope from KV cache, SWIZZLE_128B, BK_PV x rope_depth (64).
        # The KV cache row layout is padded_depth FP8 bytes followed by
        # rope_depth BF16 elements.  create_rope_tma_tile offsets the base
        # pointer by padded_depth bytes and reinterprets as BF16.
        var k_rope_tma = k.create_rope_tma_tile[
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
        var scale_tma = tma_tile_scales[BN_QK=mla_config.BN_QK](
            ctx, scales_ptr, _total_scale_elements
        )

        comptime if ragged:
            comptime ValidLengthType = NonNullPointer[.uint32]
            var valid_len: ValidLengthType = {valid_length}
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
                launch_q_max_seq_len,
                valid_len,
                mask,
                q_scale_ptr,
                scalar_args_buf.value(),
                ctx,
            )
        else:
            comptime ValidLengthType = NullPointer[.uint32]
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
                launch_q_max_seq_len,
                valid_len,
                mask,
                q_scale_ptr,
                scalar_args_buf.value(),
                ctx,
            )
    elif _native_fp8:
        var q_ptr_fp8 = rebind[
            MutPointer[Scalar[k_t.dtype], origin=MutAnyOrigin]
        ](q.ptr)
        var q_tma_fp8 = tma_tile_qo[
            swizzle_mode=mla_config.kv_tma_swizzle_mode,  # SWIZZLE_64B
            BM=mla_config.BM,
            BK=mla_config.input_q_depth,
            depth=mla_config.input_q_depth,
        ](ctx, q_ptr_fp8, num_rows_q)

        # Layout G config + BM=32 Q TMA tile. Built unconditionally so the
        # comptime branch inside `_launch_r` / `_launch_n` can reference
        # them; when the runtime gate misses, the unused branch is pruned.
        comptime mla_config_g = MLA_SM100_Decode_Config(
            num_q_heads=mla_config.num_q_heads,
            group=mla_config.group,
            depth=mla_config.depth,
            q_depth=mla_config.q_depth,
            input_q_depth=mla_config.input_q_depth,
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
        var q_tma_fp8_g = tma_tile_qo[
            swizzle_mode=mla_config_g.kv_tma_swizzle_mode,  # SWIZZLE_64B
            BM=mla_config_g.BM,  # 32
            BK=mla_config_g.input_q_depth,
            depth=mla_config_g.input_q_depth,
        ](ctx, q_ptr_fp8, num_rows_q)

        # Dispatch is routed at comptime by num_q_heads × q_max_seq_len.
        # q_max_seq_len > 1 implies spec decoding (1 actual + N spec ahead),
        # capped at MAX_FOLD_Q (1 actual + (MAX_FOLD_Q - 1) spec). For each q
        # in {MIN_FOLD_Q..MAX_FOLD_Q}: num_heads × q ≤ 32 → Layout-G fold
        # (BM=32), ≤ 64 → Layout-E fold (BM=64), > 64 → non-fold (kernel
        # handles q in grid dim). For q=1 (regular decode), num_heads ≤ 32 →
        # Layout-G non-fold, else → Layout-E non-fold.

        comptime if ragged:
            comptime ValidLengthType = NonNullPointer[.uint32]
            var valid_len: ValidLengthType = {valid_length}

            @inline(.always)
            def _launch_r[
                _fold_q: Bool,
                _q_len_fold: Int,
                _layout_g: Bool = False,
            ]() raises {imm}:
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
                        launch_q_max_seq_len,
                        valid_len,
                        mask,
                        scales_ptr,
                        scalar_args_buf.value(),
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
                        launch_q_max_seq_len,
                        valid_len,
                        mask,
                        scales_ptr,
                        scalar_args_buf.value(),
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
            comptime ValidLengthType = NullPointer[.uint32]
            var valid_len: ValidLengthType = {}

            @inline(.always)
            def _launch_n[
                _fold_q: Bool,
                _q_len_fold: Int,
                _layout_g: Bool = False,
            ]() raises {imm}:
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
                        launch_q_max_seq_len,
                        valid_len,
                        mask,
                        scales_ptr,
                        scalar_args_buf.value(),
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
                        launch_q_max_seq_len,
                        valid_len,
                        mask,
                        scales_ptr,
                        scalar_args_buf.value(),
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
        var q_ptr = rebind[MutPointer[Scalar[q_type], origin=MutAnyOrigin]](
            q.ptr
        )
        var q_tma_op = tma_tile_qo[
            swizzle_mode=mla_config.swizzle_mode,
            BM=mla_config.BM,
            BK=mla_config.input_q_depth,
            depth=mla_config.input_q_depth,
        ](ctx, q_ptr, num_rows_q)

        comptime if ragged:
            comptime ValidLengthType = NonNullPointer[.uint32]
            var valid_len: ValidLengthType = {valid_length}
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
                launch_q_max_seq_len,
                valid_len,
                mask,
                scales_ptr,
                scalar_args_buf.value(),
                ctx,
            )
        else:
            comptime ValidLengthType = NullPointer[.uint32]
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
                launch_q_max_seq_len,
                valid_len,
                mask,
                scales_ptr,
                scalar_args_buf.value(),
                ctx,
            )


@inline(.always)
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
        BK=config.input_q_depth,
        swizzle_mode=config.swizzle_mode,
    ],
    k_tma: KVTMATile[
        dtype=KVLUTType.dtype,
        swizzle_mode=config.kv_tma_swizzle_mode,
        BN=config.BK_PV,  # tile_m =64
        BK=config.input_q_depth,
    ],
    o_tma: ORaggedTMATile[
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
    scales_ptr: MutPointer[Float32, origin=MutAnyOrigin],
    scalar_args_buf: TileTensor[.int64, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    var mla_decode_pack = MLA_Decode_Pack[
        ValidLengthType=ValidLengthType,
        MaskType=MaskType,
        SplitAccumType=SplitAccumType,
    ](mask, valid_len, lse_accum_split_ptr, num_partitions)

    var block_x = ceildiv(config.num_q_heads, config.BM)
    var grid_dim = (block_x * q_max_seq_len, 1, block_z)
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
        Engine=scalar_args_buf.Engine,
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
        Engine=scalar_args_buf.Engine,
    ].kernel
    # Enable PDL (Programmatic Dependent Launch) for split-K mode to chain
    # the MLA decode kernel with the combine kernel, reducing host synchronization.
    comptime pdl_level = PDLLevel.OVERLAP_AT_END if (
        config.decoding_warp_split_k or _MLA_PDL_ALWAYS
    ) else PDLLevel.OFF

    ctx.enqueue_function[kernel](
        GridBlockAttributeLaunchConfig(
            grid_dim=grid_dim,
            block_dim=block_dim,
            shared_mem_bytes=config.smem_used,
            attributes=pdl_launch_attributes(pdl_level),
        ),
        q_tma,
        k_tma,
        o_tma,
        kv_lut,
        scale,
        mla_decode_pack,
        scales_ptr,
        scalar_args_buf,
    )


@inline(.always)
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
    # q_len_fold q_tokens into the BM=64 M tile and grid_y collapses to 1.
    fold_q: Bool = False,
    # comptime number of q_tokens to fold.  Must satisfy
    # `num_q_heads * q_len_fold <= BM` (caller-enforced) and `q_len_fold > 1`.
    q_len_fold: Int = 1,
](
    q_tma: QOTMATile[
        dtype=KVLUTType.dtype,  # FP8 Q TMA
        BM=config.BM,
        BK=config.input_q_depth,
        swizzle_mode=config.kv_tma_swizzle_mode,  # SWIZZLE_64B
    ],
    k_tma: KVTMATile[
        dtype=KVLUTType.dtype,
        swizzle_mode=config.kv_tma_swizzle_mode,
        BN=config.BK_PV,
        BK=config.input_q_depth,
    ],
    o_tma: ORaggedTMATile[
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
    scales_ptr: MutPointer[Float32, origin=MutAnyOrigin],
    scalar_args_buf: TileTensor[.int64, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    """Launch the native FP8 MLA decode kernel with FP8 Q TMA.

    This is a dedicated launch function for the native FP8 path because
    the Q TMA has FP8 dtype (SWIZZLE_64B) instead of BF16 (SWIZZLE_128B).

    Under `fold_q=True`, BM=64 packs `q_len_fold * num_q_heads` M-rows of
    the same batch element and grid_y collapses to 1 (all q_tokens live in
    one CTA).  This avoids spawning `q_max_seq_len` CTAs and lets the
    softmax/correction WGs amortize the QK setup.
    """
    var mla_decode_pack = MLA_Decode_Pack[
        ValidLengthType=ValidLengthType,
        MaskType=MaskType,
        SplitAccumType=SplitAccumType,
    ](mask, valid_len, lse_accum_split_ptr, num_partitions)
    var block_x = ceildiv(config.num_q_heads, config.BM)
    # fold collapses grid_y to 1 since BM packs all q_tokens.
    var grid_y = 1 if fold_q else q_max_seq_len
    var grid_dim = (block_x * grid_y, 1, block_z)
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
        Engine=scalar_args_buf.Engine,
    ].kernel
    comptime pdl_level = PDLLevel.OVERLAP_AT_END if (
        config.decoding_warp_split_k or _MLA_PDL_ALWAYS
    ) else PDLLevel.OFF
    ctx.enqueue_function[kernel](
        GridBlockAttributeLaunchConfig(
            grid_dim=grid_dim,
            block_dim=block_dim,
            shared_mem_bytes=config.smem_used,
            attributes=pdl_launch_attributes(pdl_level),
        ),
        q_tma,
        k_tma,
        o_tma,
        kv_lut,
        scale,
        mla_decode_pack,
        scales_ptr,
        scalar_args_buf,
    )


# Layout G launcher for the qkv_fp8 native-FP8 kernel (BM=32, MMA_M=32,
# 5-stage). Takes two configs because k_tma / o_tma are identical between
# Layout E and Layout G — `config_e` types those, while `config_g`
# (`decode_layout_g=True`) types the kernel struct and BM=32 Q TMA tile.
@inline(.always)
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
        BK=config_g.input_q_depth,
        swizzle_mode=config_g.kv_tma_swizzle_mode,  # SWIZZLE_64B
    ],
    k_tma: KVTMATile[
        dtype=KVLUTType.dtype,
        swizzle_mode=config_e.kv_tma_swizzle_mode,
        BN=config_e.BK_PV,
        BK=config_e.input_q_depth,
    ],
    o_tma: ORaggedTMATile[
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
    scales_ptr: MutPointer[Float32, origin=MutAnyOrigin],
    scalar_args_buf: TileTensor[.int64, address_space=.GENERIC, ...],
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
    ](mask, valid_len, lse_accum_split_ptr, num_partitions)
    var block_x = ceildiv(config_g.num_q_heads, config_g.BM)
    # fold collapses grid_y to 1 since BM packs all q_tokens.
    var grid_y = 1 if fold_q else q_max_seq_len
    var grid_dim = (block_x * grid_y, 1, block_z)
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
        Engine=scalar_args_buf.Engine,
    ].kernel
    comptime pdl_level = PDLLevel.OVERLAP_AT_END if (
        config_g.decoding_warp_split_k or _MLA_PDL_ALWAYS
    ) else PDLLevel.OFF
    ctx.enqueue_function[kernel](
        GridBlockAttributeLaunchConfig(
            grid_dim=grid_dim,
            block_dim=block_dim,
            shared_mem_bytes=config_g.smem_used,
            attributes=pdl_launch_attributes(pdl_level),
        ),
        q_tma,
        k_tma,
        o_tma,
        kv_lut,
        scale,
        mla_decode_pack,
        scales_ptr,
        scalar_args_buf,
    )


@inline(.always)
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
    o_tma: ORaggedTMATile[
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
    q_scale_ptr: OptionalReg[MutPointer[Float32, MutAnyOrigin]],
    scalar_args_buf: TileTensor[DType.int64, address_space=.GENERIC, ...],
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
    ](mask, valid_len, lse_accum_split_ptr, num_partitions)
    var block_x = ceildiv(config.num_q_heads, config.BM)
    var grid_dim = (block_x * q_max_seq_len, 1, block_z)
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
        Engine=scalar_args_buf.Engine,
    ].kernel
    comptime pdl_level = PDLLevel.OVERLAP_AT_END if (
        config.decoding_warp_split_k or _MLA_PDL_ALWAYS
    ) else PDLLevel.OFF
    ctx.enqueue_function[kernel](
        GridBlockAttributeLaunchConfig(
            grid_dim=grid_dim,
            block_dim=block_dim,
            shared_mem_bytes=config.smem_used,
            attributes=pdl_launch_attributes(pdl_level),
        ),
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
    )


@inline(.always)
def launch_mla_sm100_decode_sparse_qkv_fp8[
    q_type: DType,
    KVLUTType: MHAOperand,
    output_type: DType,
    SplitAccumType: OptionalPointer,
    MaskType: MHAMask,
    config: MLA_SM100_Decode_Config,
    ValidLengthType: OptionalPointer,
    _is_cache_length_accurate: Bool = False,
    ragged: Bool = False,
    AttnSinkPtrType: OptionalPointer = NullPointer[DType.float32],
    has_extra_kv: Bool = False,
    TopkLengthsPtrType: OptionalPointer = NullPointer[DType.int32],
    fold_shared_index: Bool = False,
    q_len_fold: Int = 1,
    physical_indices: Bool = True,
    sparse_row: SparseRowLayout = SparseRowLayout(),
    # Types the kernel and the Q tile. `config` keeps typing the INT64 gather
    # descriptor and o_tma, which Layout E and Layout G share; see the
    # asserts below.
    config_g: MLA_SM100_Decode_Config = config,
    skip_masked_tiles: Bool = False,
    prepared_bounds: Bool = False,
](
    q_tma: QOTMATile[
        dtype=q_type,
        BM=config_g.BM,
        BK=config_g.input_q_depth,
        swizzle_mode=config_g.q_swizzle(),
    ],
    o_tma: ORaggedTMATile[
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
    d_indices: OptionalReg[Pointer[Int32, MutAnyOrigin]],
    indices_stride: Int,
    topk_lengths: TopkLengthsPtrType,
    scales_ptr: Pointer[Float32, origin=MutAnyOrigin],
    attn_sink_ptr: AttnSinkPtrType,
    extra_kv_lut: KVLUTType,
    extra_d_indices: OptionalReg[Pointer[Int32, MutAnyOrigin]],
    extra_topk_lengths: TopkLengthsPtrType,
    extra_indices_stride: Int,
    extra_scales_ptr: OptionalReg[Pointer[Float32, MutAnyOrigin]],
    scalar_args_buf: NullableTileTensor[.int64, address_space=.GENERIC, ...],
    ctx: DeviceContext,
    # Logical sparse indices for position-based causal masking; `None` keeps
    # the prior slot-count behavior. See mla_decode_utils.mojo.
    logical_indices: OptionalReg[Pointer[Int32, MutAnyOrigin]] = None,
    # Positions between the token each sparse row was computed for and
    # the token that reads it.
    selection_lag: Int = 0,
    dispatch_cache_bound: Int = 0,
    prepared_q_max_seq_len: Int = 0,
) raises:
    """Launches the unified sparse MLA decode kernel.

    `MLA_SM100_Decode_Sparse_QKV_FP8` runs its MMAs in the Q dtype: FP8 Q
    and cache (`tcgen05.mma.kind::f8f6f4`), or BF16 Q and cache
    (`kind::f16`, always through the direct gather).
    """
    # Layout G may run 128-key tiles; the gather descriptor's box is one row
    # either way, so only the row width and the output tile must agree.
    comptime assert (
        (
            config.BN_QK == config_g.BN_QK
            or (config_g.decode_layout_g and config_g.BN_QK == 128)
            or (
                config_g.cta_group == 1
                and config_g.BN_QK == 128
                and config_g.BK_PV == config.BK_PV
            )
            or (
                config_g.cta_group == 2
                and config_g.BN_QK == 2 * config.BN_QK
                and config_g.BK_PV == config.BK_PV
            )
        )
        and (
            config.BK_PV == config_g.BK_PV
            or (config_g.decode_layout_g and config_g.BK_PV == 128)
        )
        and config.input_q_depth == config_g.input_q_depth
        and config.out_rows == config_g.out_rows
        and config.swizzle_mode == config_g.swizzle_mode
    ), "config and config_g must agree on the shared k_tma / o_tma tiles"
    # Split workspaces use the caller's padded Q bound even when the grid
    # skips trailing query CTAs. Prepared kernels must retain that stride.
    var pack_q_max_seq_len = (
        prepared_q_max_seq_len if prepared_q_max_seq_len > 0 else q_max_seq_len
    )
    var mla_decode_pack = MLA_Decode_Pack[
        ValidLengthType=ValidLengthType,
        MaskType=MaskType,
        SplitAccumType=SplitAccumType,
        prepared_bounds=prepared_bounds,
    ](
        mask,
        valid_len,
        lse_accum_split_ptr,
        num_partitions,
        batch_size,
        pack_q_max_seq_len,
    )
    var block_x = ceildiv(config_g.num_q_heads, config_g.BM)
    var grid_y = 1 if fold_shared_index else q_max_seq_len
    var grid_dim = (block_x * grid_y, 1, block_z)
    var block_dim = (config_g.num_threads, 1, 1)

    comptime if config_g.decode_layout_g:
        logger.info(
            "------ Dispatching to SM100 Sparse MLA-DECODE (unified,"
            " Layout G) ------"
        )
    else:
        logger.info(
            "------ Dispatching to SM100 Sparse MLA-DECODE (unified) ------"
        )

    comptime KernelType = MLA_SM100_Decode_Sparse_QKV_FP8[
        q_type=q_type,
        KVLUTType=KVLUTType,
        output_type=output_type,
        SplitAccumType=SplitAccumType,
        MaskType=MaskType,
        config=config_g,
        ValidLengthType=ValidLengthType,
        _is_cache_length_accurate=_is_cache_length_accurate,
        ragged=ragged,
        AttnSinkPtrType=AttnSinkPtrType,
        has_extra_kv=has_extra_kv,
        TopkLengthsPtrType=TopkLengthsPtrType,
        fold_shared_index=fold_shared_index,
        q_len_fold=q_len_fold,
        physical_indices=physical_indices,
        sparse_row=sparse_row,
        skip_masked_tiles=skip_masked_tiles,
        Engine=scalar_args_buf.Engine,
        prepared_bounds=prepared_bounds,
    ]
    # The pair's entry declares its (2, 1, 1) cluster and takes the rope
    # gather's descriptor; `kernel` serves every unpaired form, including
    # the fused combine's launch-time cluster. The two signatures differ
    # under the pair, so each enqueue site names its entry directly (a
    # comptime ternary over the two function types stays symbolic).
    comptime assert (
        config_g.cta_group != 2 or config_g.direct_gather
    ), "the pair takes the direct gather"
    comptime pdl_level = PDLLevel.OVERLAP_AT_END if (
        config_g.decoding_warp_split_k or _MLA_PDL_ALWAYS
    ) else PDLLevel.OFF
    # Extra SMEM beyond config.smem_used (unified-gather native FP8, no
    # cvt_blk_bars -- the second KV buffer + its pipeline are already
    # folded into config.smem_used via native_fp8_unified_gather):
    #   - idx_bars: 2*N barriers (depth matches config.num_kv_stages, N)
    #   - ptr_tmem_addr (4 bytes, UInt32)
    #   - idx_smem, N-deep (N * BN_QK * sizeof(Int32)), twice that with an
    #     NVFP4 cache, whose stages also hold the tile's scale rows
    #   - up to 12 bytes to 16-byte align idx_smem when the gather4 indices
    #     are loaded in ld.shared.v4 batches
    comptime sparse_extra_smem = (
        # The pair's config accounting already carries the index ring, TMEM
        # pointer, alignment pad and idx barriers (its `smem_used` is the
        # kernel's exact carve), at the K/V ring's depth; a deeper index
        # ring (`_SPARSE_PAIR_IDX_STAGES`) adds its extra stages.
        (KernelType.idx_stages - config_g.num_kv_stages)
        * (
            2 * config_g.mbar_size + KernelType.idx_stage_slots * 4
        ) if config_g.cta_group
        == 2 else (
            2 * config_g.num_kv_stages * config_g.mbar_size
            + 4
            + KernelType.idx_smem_align_pad
            + config_g.num_kv_stages * KernelType.idx_stage_slots * 4
        )
    )
    comptime corr_extra_smem = 512 if KernelType.corr_smem else 0
    comptime sparse_smem_used = (
        config_g.smem_used
        + sparse_extra_smem
        + corr_extra_smem
        + KernelType.role_metadata_bytes
        + STAMP_SMEM_BYTES
    )
    comptime if config_g.BN_QK == 128:
        comptime assert config_g.supported(), "unsupported 128-key config"
    comptime if config_g.decode_layout_g:
        comptime assert config_g.supported(), "unsupported Layout G config"
        comptime assert (
            sparse_smem_used <= config_g.sm100_smem_carveout
        ), "Layout G sparse SMEM exceeds the carveout"
    comptime if KernelType.is_nvfp4_kv:
        # The config sizes the rings for one-part index rows; the scale rows
        # take the slack it leaves.
        comptime assert (
            sparse_smem_used <= config_g.sm100_smem_carveout
        ), "NVFP4 sparse SMEM exceeds the carveout"
    comptime if config_g.cta_group == 2:
        comptime assert config_g.supported(), "unsupported paired config"
        comptime assert (
            sparse_smem_used <= config_g.sm100_smem_carveout
        ), "paired sparse SMEM exceeds the carveout"
    # Compile-time only. A carve that overran its allocation would read and
    # write past it, or into the `MLA_STAMP` reserve at its end.
    comptime assert KernelType.carve_bytes == sparse_qkv_fp8_carve_bytes(
        config_g,
        KernelType.idx_smem_align_pad,
        KernelType.sparse_row_parts,
        KernelType.corr_smem,
        KernelType.role_metadata_bytes,
    ), "sparse_qkv_fp8_carve_bytes disagrees with the kernel's carve"
    comptime assert (
        KernelType.carve_bytes
        <= config_g.smem_used
        + sparse_extra_smem
        + corr_extra_smem
        + KernelType.role_metadata_bytes
    ), "the sparse kernel's SMEM carve overruns its allocation"
    comptime if get_defined_int["MLA_SPARSE_TRACE_LAUNCH", 0]() != 0:
        var effective_keys = _sparse_pair_keys(
            indices_stride // (2 if KernelType.is_nvfp4_kv else 1),
            extra_indices_stride,
            dispatch_cache_bound,
        )
        print(
            t"MLA_SPARSE_LAUNCH"
            t" tile={config_g.BN_QK} group={config_g.cta_group} np={num_partitions} split_page={config_g.split_page_size} ctas_per_partition={block_x * grid_y * batch_size} stages={config_g.num_kv_stages} mma_stages={config_g.num_kv_mma_stages} smem={sparse_smem_used} tmem={config_g.sm100_tmem_cols} effective_keys={effective_keys} effective_cache_bound={dispatch_cache_bound}"
        )
    var stamp_addr: UInt64 = 0
    comptime if MLA_STAMP:
        var env = getenv("MLA_STAMP_ADDR")
        if env:
            stamp_addr = UInt64(atol(env))

    var kernel_scalar_args: KernelType.ScalarArgsType
    comptime if prepared_bounds:
        kernel_scalar_args = rebind[KernelType.ScalarArgsType](
            StaticTuple[Int64, 0]()
        )
    else:
        kernel_scalar_args = rebind[KernelType.ScalarArgsType](
            scalar_args_buf.value()
        )
    comptime if config_g.direct_gather:
        # FP8 SWIZZLE_64B (or 128B) gather4: one 64- (or 128-) column box per
        # column group, at the k-major offset the MMA descriptors read (as in
        # the cg1 sparse prefill).
        var k_sw64_tma = kv_lut.create_gather4_tma_tile[
            tile_width=KernelType.kv_gather4_tile_width,
            tile_stride=KernelType.kv_gather4_tile_width,
            swizzle_mode=KernelType.kv_gather4_swizzle,
            tile_height=KernelType.GatherTileRows,
            tma_dtype=KernelType.k_tma_dtype,
            l2_promotion=TensorMapL2Promotion.L2_128B,
        ](ctx)
        var extra_k_sw64_tma = extra_kv_lut.create_gather4_tma_tile[
            tile_width=KernelType.kv_gather4_tile_width,
            tile_stride=KernelType.kv_gather4_tile_width,
            swizzle_mode=KernelType.kv_gather4_swizzle,
            tile_height=KernelType.GatherTileRows,
            tma_dtype=KernelType.k_tma_dtype,
            l2_promotion=TensorMapL2Promotion.L2_128B,
        ](ctx)
        comptime if config_g.cta_group == 2:
            var packed_v_descriptor = KernelType.PackedVTMA(
                k_sw64_tma.descriptor
            )
            comptime if KernelType.is_nvfp4_kv:
                var packed_v_tma = kv_lut.create_gather4_tma_tile[
                    tile_width=8,
                    tile_stride=KernelType.kv_row_words,
                    swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
                    tile_height=KernelType.GatherTileRows,
                    tma_dtype=DType.int64,
                    l2_promotion=TensorMapL2Promotion.L2_128B,
                ](ctx)
                packed_v_descriptor = KernelType.PackedVTMA(
                    packed_v_tma.descriptor
                )
            ctx.enqueue_function[KernelType.kernel_pair](
                ClusterAttributeLaunchConfig(
                    grid_dim=grid_dim,
                    block_dim=block_dim,
                    # The two head tiles of one token are adjacent CTAs, so a
                    # (2, 1, 1) cluster pairs exactly the CTAs that share an
                    # index list.
                    cluster_dim=Dim(2, 1, 1),
                    shared_mem_bytes=sparse_smem_used,
                    attributes=pdl_launch_attributes(pdl_level),
                ),
                q_tma,
                KernelType.KTMA(k_sw64_tma.descriptor),
                packed_v_descriptor,
                o_tma,
                kv_lut,
                scale,
                mla_decode_pack,
                d_indices,
                Int32(indices_stride),
                logical_indices,
                topk_lengths,
                scales_ptr,
                attn_sink_ptr,
                KernelType.KTMA(extra_k_sw64_tma.descriptor),
                extra_kv_lut,
                extra_d_indices,
                extra_topk_lengths,
                Int32(extra_indices_stride),
                extra_scales_ptr,
                kernel_scalar_args,
                Int32(selection_lag),
                stamp_addr,
            )
        else:
            ctx.enqueue_function[KernelType.kernel](
                GridBlockAttributeLaunchConfig(
                    grid_dim=grid_dim,
                    block_dim=block_dim,
                    shared_mem_bytes=sparse_smem_used,
                    attributes=pdl_launch_attributes(pdl_level),
                ),
                q_tma,
                KernelType.KTMA(k_sw64_tma.descriptor),
                o_tma,
                kv_lut,
                scale,
                mla_decode_pack,
                d_indices,
                Int32(indices_stride),
                logical_indices,
                topk_lengths,
                scales_ptr,
                attn_sink_ptr,
                KernelType.KTMA(extra_k_sw64_tma.descriptor),
                extra_kv_lut,
                extra_d_indices,
                extra_topk_lengths,
                Int32(extra_indices_stride),
                extra_scales_ptr,
                kernel_scalar_args,
                Int32(selection_lag),
                stamp_addr,
            )
    else:
        # INT64, SWIZZLE_NONE, one box per 576-byte row for the linear front
        # end (16 gather4 instructions per 64 keys instead of the
        # SW64-fragmented 144); the re-swizzle warpgroup reproduces the SW64
        # layout in SMEM.
        comptime kv_tile_width = KernelType.kv_row_words
        var k_tma = kv_lut.create_gather4_tma_tile[
            tile_width=kv_tile_width,
            tile_stride=kv_tile_width,
            swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
            tile_height=KernelType.GatherTileRows,
            tma_dtype=DType.int64,
            l2_promotion=TensorMapL2Promotion.L2_128B,
        ](ctx)
        var extra_k_tma = extra_kv_lut.create_gather4_tma_tile[
            tile_width=kv_tile_width,
            tile_stride=kv_tile_width,
            swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
            tile_height=KernelType.GatherTileRows,
            tma_dtype=DType.int64,
            l2_promotion=TensorMapL2Promotion.L2_128B,
        ](ctx)
        comptime assert (
            type_of(k_tma).dtype == KernelType.k_tma_dtype
            and type_of(k_tma).tile_shape.element_types[1].static_value
            == KernelType.kv_gather4_box_w
        ), "k_tma does not match the kernel's gather descriptor"
        ctx.enqueue_function[KernelType.kernel](
            GridBlockAttributeLaunchConfig(
                grid_dim=grid_dim,
                block_dim=block_dim,
                shared_mem_bytes=sparse_smem_used,
                attributes=pdl_launch_attributes(pdl_level),
            ),
            q_tma,
            KernelType.KTMA(k_tma.descriptor),
            o_tma,
            kv_lut,
            scale,
            mla_decode_pack,
            d_indices,
            Int32(indices_stride),
            logical_indices,
            topk_lengths,
            scales_ptr,
            attn_sink_ptr,
            KernelType.KTMA(extra_k_tma.descriptor),
            extra_kv_lut,
            extra_d_indices,
            extra_topk_lengths,
            Int32(extra_indices_stride),
            extra_scales_ptr,
            kernel_scalar_args,
            Int32(selection_lag),
            stamp_addr,
        )
