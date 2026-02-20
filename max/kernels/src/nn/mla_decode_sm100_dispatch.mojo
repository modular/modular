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

from math import ceildiv, clamp, gcd
from sys import size_of
from gpu.host import DeviceContext, FuncAttribute
from gpu.memory import AddressSpace
from gpu.primitives.grid_controls import pdl_launch_attributes, PDLLevel
from layout.layout import (
    Layout,
)
from logger import Logger

from layout.layout_tensor import (
    LayoutTensor,
)
from nn.mha_fa3_utils import (
    NonNullPointer,
    NullPointer,
    OptionalPointer,
)
from nn.mha_mask import MHAMask
from nn.mha_operand import MHAOperand
from nn.mha_score_mod import ScoreModTrait
from nn.mha_utils import (
    MHAConfig,
)
from nn.mha_fa3_utils import KVTMATile
from layout.runtime_layout import RuntimeLayout
from utils.numerics import get_accum_type
from utils.index import Index

comptime logger = Logger()
from nn.mla_decode_sm100_utils import (
    MLA_SM100_Decode_Config,
    QOTMATile,
    tma_tile_qo,
    MLA_Decode_Pack,
    num_matrix_view_rows_decode,
)
from nn.mla_decode_sm100_kv_bf16 import MLA_SM100_Decode_KV_BF16
from nn.mla_decode_sm100_kv_fp8 import MLA_SM100_Decode_KV_FP8
from nn.mla_decode_sm100_combine import mla_decode_combine_partial_outputs


# ------------------------------------------------------------------------------
# MLA decoding implementation for SM100
# ------------------------------------------------------------------------------
fn mla_decode_sm100_dispatch[
    q_type: DType,
    q_layout: Layout,
    k_t: MHAOperand,
    output_type: DType,
    output_layout: Layout,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    valid_layout: Layout,
    config: MHAConfig,
    depth: Int,
    num_heads: Int,
    group: Int = 1,
    *,
    use_score_mod: Bool = False,
    ragged: Bool = False,
    _is_cache_length_accurate: Bool = False,
    decoding_warp_split_k: Bool = False,
](
    q: LayoutTensor[
        q_type, q_layout, address_space = AddressSpace.GENERIC, ...
    ],
    k: k_t,
    output: LayoutTensor[
        output_type, output_layout, address_space = AddressSpace.GENERIC, ...
    ],
    scale: Float32,
    batch_size: Int,
    max_cache_valid_length: Int,  # longest KV cache entry
    q_max_seq_len: Int,
    valid_length: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    mask: mask_t,
    score_mod: score_mod_t,
    ctx: DeviceContext,
) raises:
    # Get the base pointer to the scales tensor from the operand.
    var scales_ptr = k.scales_raw_ptr()

    # CRITICAL: The kernel's OffsetPosition adds q_max_seq_len to num_keys when
    # _is_cache_length_accurate=False. We must use the same effective num_keys
    # here to compute num_partitions correctly, otherwise there's a mismatch
    # between how the dispatcher divides work and how the kernel sees it.
    var effective_max_cache_len = max_cache_valid_length

    comptime if not _is_cache_length_accurate:
        effective_max_cache_len += q_max_seq_len

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
    if effective_max_cache_len <= 512 and batch_size >= 32:
        _mla_decode_sm100_dispatch_impl[
            q_type=q_type,
            q_layout=q_layout,
            k_t=k_t,
            output_type=output_type,
            output_layout=output_layout,
            mask_t=mask_t,
            score_mod_t=score_mod_t,
            valid_layout=valid_layout,
            config=config,
            depth=depth,
            num_heads=num_heads,
            group=group,
            use_score_mod=use_score_mod,
            ragged=ragged,
            _is_cache_length_accurate=_is_cache_length_accurate,
            decoding_warp_split_k=decoding_warp_split_k,
            split_page_size=64,
        ](
            q,
            k,
            output,
            scale,
            batch_size,
            effective_max_cache_len,
            max_cache_valid_length,
            q_max_seq_len,
            valid_length,
            mask,
            score_mod,
            scales_ptr,
            ctx,
        )
    else:
        _mla_decode_sm100_dispatch_impl[
            q_type=q_type,
            q_layout=q_layout,
            k_t=k_t,
            output_type=output_type,
            output_layout=output_layout,
            mask_t=mask_t,
            score_mod_t=score_mod_t,
            valid_layout=valid_layout,
            config=config,
            depth=depth,
            num_heads=num_heads,
            group=group,
            use_score_mod=use_score_mod,
            ragged=ragged,
            _is_cache_length_accurate=_is_cache_length_accurate,
            decoding_warp_split_k=decoding_warp_split_k,
            split_page_size=128,
        ](
            q,
            k,
            output,
            scale,
            batch_size,
            effective_max_cache_len,
            max_cache_valid_length,
            q_max_seq_len,
            valid_length,
            mask,
            score_mod,
            scales_ptr,
            ctx,
        )


# ------------------------------------------------------------------------------
# Inner dispatch implementation parameterized on split_page_size
# ------------------------------------------------------------------------------
fn _mla_decode_sm100_dispatch_impl[
    q_type: DType,
    q_layout: Layout,
    k_t: MHAOperand,
    output_type: DType,
    output_layout: Layout,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    valid_layout: Layout,
    config: MHAConfig,
    depth: Int,
    num_heads: Int,
    group: Int = 1,
    *,
    use_score_mod: Bool = False,
    ragged: Bool = False,
    _is_cache_length_accurate: Bool = False,
    decoding_warp_split_k: Bool = False,
    split_page_size: Int = 128,
](
    q: LayoutTensor[
        q_type, q_layout, address_space = AddressSpace.GENERIC, ...
    ],
    k: k_t,
    output: LayoutTensor[
        output_type, output_layout, address_space = AddressSpace.GENERIC, ...
    ],
    scale: Float32,
    batch_size: Int,
    effective_max_cache_len: Int,
    max_cache_valid_length: Int,
    q_max_seq_len: Int,
    valid_length: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    mask: mask_t,
    score_mod: score_mod_t,
    scales_ptr: UnsafePointer[Scalar[DType.float32], origin=MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    comptime hw_info = ctx.default_device_info
    comptime sm_count = hw_info.sm_count
    # Maximum number of splits the combine kernel is instantiated for.
    # Optimized for the DeepSeek V3/R1 production config (num_heads=128).
    #
    # wave_quantum = sm_count / gcd(ctas_per_partition, sm_count).
    # For DeepSeek (num_heads=128, BM=64): ctas_per_partition >= 2,
    # so wave_quantum = 148 / gcd(2, 148) = 148 / 2 = 74.
    #
    # Models with num_heads <= 64 have ctas_per_partition=1, giving
    # wave_quantum = 148, which exceeds this limit. Since this kernel
    # targets DeepSeek, 74 is sufficient and avoids 2x compile-time
    # cost of the combine kernel's @parameter for loops.
    comptime max_num_splits = 74

    comptime AccumType = get_accum_type[output.dtype]()
    comptime v_depth = depth - 64

    # Ensure KV cache page_size is evenly divisible by split_page_size.
    # If the KV cache page_size shrinks in the future, splits must not
    # straddle physical page boundaries.
    comptime assert (
        k_t.page_size % split_page_size == 0
    ), "KV cache page_size must be divisible by split_page_size"

    # Compute num_kv_cache_pages using the parametric split_page_size.
    # This determines how finely KV work is divided across split-K partitions.
    var num_kv_cache_pages = ceildiv(effective_max_cache_len, split_page_size)

    # Wave-aligned split count: num_partitions is chosen to make total_CTAs as
    # close as possible to multiple of sm_count, eliminating GPU wave quantization waste.
    #
    # Total CTAs launched = ctas_per_partition * num_partitions, where
    # ctas_per_partition = ceildiv(num_heads, BM) * q_max_seq_len * batch_size.
    # For perfect wave alignment: total_CTAs % sm_count == 0.
    # This requires num_partitions to be a multiple of:
    #   wave_quantum = sm_count / gcd(ctas_per_partition, sm_count)
    #
    # On B200 (sm_count=148, BM=64, num_heads=128 => ctas_per_partition=2*bs):
    #   bs=1: wave_quantum=74, bs>=2: wave_quantum=37
    #
    # We start with 1 wave quantum and add more wave quanta only if pages_per_split
    # exceeds the max threshold, keeping combine overhead low while ensuring enough parallelism.
    var ctas_per_partition = ceildiv(num_heads, 64) * q_max_seq_len * batch_size
    var wave_quantum = sm_count // gcd(ctas_per_partition, sm_count)

    # Minimum partitions to keep pages_per_split <= max threshold.
    # 18 pages * 128 tokens/page = 2304 tokens per split.
    # This threshold is chosen to balance decode work per split against wave
    # quantization. At 18 pages, the largest context (cl=163840, 1281 pages)
    # needs ceil(1281/18)=72 splits, which fits in 1 decode wave (72*2=144
    # CTAs on 148 SMs, 97.3% efficiency). For all other configs
    # (cl<=131072), max_pages_per_split doesn't affect the final np since
    # those are bounded by wave_quantum or max_num_splits instead.
    comptime max_pages_per_split = 18
    var min_partitions_for_work = ceildiv(
        num_kv_cache_pages, max_pages_per_split
    )

    # The key is to have enough splits so total CTAs fill the GPU,
    # but not so many that each split has trivial work (< min_pages_per_split pages).
    # This prevents the wave_quantum from creating 32 splits with only
    # 1-2 pages each when batch_size is moderate (16-32), while still
    # giving 2-4 splits for large batch (64-128) to improve SM utilization.
    #
    # min_pages_per_split is batch-size-aware to jointly optimize (np, wph):
    #  - Small batch (bs<=8): min_pages_per_split=4, allows many splits (37-74)
    #    with high wph (8) for parallelism since combine CTAs are few.
    #  - Medium batch (bs=16-32): min_pages_per_split=4, moderate splits (7-13)
    #    with moderate wph (4-8).
    #  - Large batch (bs>=64): min_pages_per_split=8, caps splits at 2-4 to
    #    keep combine CTA count low. E.g., bs=64/2K gets np=2 with wph=4
    #    (few-split regime prefers wph=4 over wph=2 for lower per-CTA latency).
    #
    # Note: When split_page_size=64 (short cache path), pages are half the
    # size of 128. The same min_pages_per_split thresholds still work correctly
    # because the resulting num_kv_cache_pages // min_pages_per_split naturally
    # produces low split counts (often 0), letting target_partitions dominate,
    # which gives the right np for these configs.
    var min_pages_per_split: Int = 8 if batch_size >= 64 else 4

    var target_partitions = ceildiv(sm_count, ctas_per_partition)
    # Use wave_quantum for alignment when it gives reasonable split sizes,
    # otherwise use the SM-fill target directly.
    var num_waves = max(1, ceildiv(min_partitions_for_work, wave_quantum))
    var wave_aligned = num_waves * wave_quantum
    # Pick the smaller of wave-aligned and page-constrained to avoid
    # over-splitting. Ensure at least target_partitions for SM fill.
    var num_partitions = max(
        target_partitions,
        min(wave_aligned, num_kv_cache_pages // min_pages_per_split),
    )

    # Clamp num_partitions to:
    # 1. max_num_splits (74) - combine kernel supports up to 74 splits
    # 2. num_kv_cache_pages - at least 1 page per split to avoid empty splits
    #    (empty splits cause hangs due to barrier deadlocks or infinite loops)
    # 3. min_partitions floor:
    #    - Allow np=1 when cache is very short and batch is large enough
    #      (combine overhead dominates, np=1 eliminates it entirely).
    #      Tested extensively for bs>=64 with cache_len<=256 (<=2 pages @128).
    #    - Otherwise require np>=2 when we have enough pages.
    #    - Fall back to np=1 for very short cache (<=1 page) as safety net.
    var min_partitions: Int
    if effective_max_cache_len <= 256 and batch_size >= 64:
        min_partitions = 1
    elif num_kv_cache_pages >= 2:
        min_partitions = 2
    else:
        min_partitions = 1
    num_partitions = clamp(
        num_partitions, min_partitions, min(max_num_splits, num_kv_cache_pages)
    )

    # Eliminate empty splits caused by ceil division mismatch.
    # The main kernel uses pages_per_split = ceildiv(total_pages, num_partitions),
    # which means only ceildiv(total_pages, pages_per_split) splits actually have
    # work. Splits beyond that have start_page >= total_pages and return early
    # with uninitialized LSE, causing combine kernel corruption.
    # Recompute to ensure every split has at least 1 page of work.
    if num_partitions > 1 and num_kv_cache_pages > 0:
        var pages_per_split = ceildiv(num_kv_cache_pages, num_partitions)
        num_partitions = ceildiv(num_kv_cache_pages, pages_per_split)
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
        var o_accum_split = LayoutTensor[output_type, Layout.row_major[5]()](
            o_accum_split_data.unsafe_ptr(),
            RuntimeLayout[Layout.row_major[5]()].row_major(
                Index(
                    num_partitions,
                    batch_size,
                    q_max_seq_len,
                    Int(num_heads),
                    Int(v_depth),
                )
            ),
        )
        # Create LSE accumulator buffer (AccumType = float32 for numerical stability)
        var lse_accum_data = ctx.enqueue_create_buffer[AccumType](
            Int(num_partitions * batch_size * q_max_seq_len * num_heads)
        )
        var lse_accum_split = LayoutTensor[AccumType, Layout.row_major[4]()](
            lse_accum_data.unsafe_ptr(),
            RuntimeLayout[Layout.row_major[4]()].row_major(
                Index(
                    num_partitions,
                    batch_size,
                    q_max_seq_len,
                    Int(num_heads),
                )
            ),
        )
        var lse_accum_split_ptr: SplitAccumType = {
            lse_accum_split.to_device_buffer(ctx).unsafe_ptr()
        }

        # Launch main MLA decode kernel (writes partial results to accumulators)
        mla_decode_sm100_sink_split_k[
            q_type=q_type,
            q_layout=q_layout,
            k_t=k_t,
            output_type=output_type,
            mask_t=mask_t,
            score_mod_t=score_mod_t,
            valid_layout=valid_layout,
            config=config,
            depth=depth,
            num_heads=num_heads,
            SplitAccumType=SplitAccumType,
            group=group,
            use_score_mod=use_score_mod,
            ragged=ragged,
            _is_cache_length_accurate=_is_cache_length_accurate,
            decoding_warp_split_k=True,
            split_page_size=split_page_size,
        ](
            q,
            k,
            o_accum_split,
            lse_accum_split_ptr,
            scale,
            batch_size,
            block_z,
            num_partitions,
            max_cache_valid_length,
            q_max_seq_len,
            valid_length,
            mask,
            score_mod,
            scales_ptr,
            ctx,
        )

        # Get input_row_offsets pointer for combine kernel's ragged output writes.
        var input_row_offsets_ptr = rebind[
            UnsafePointer[Scalar[DType.uint32], origin=MutAnyOrigin]
        ](valid_length.to_device_buffer(ctx).unsafe_ptr())

        # Dispatch to specialized kernel based on num_partitions for compile-time unrolling.
        # Supports up to max_num_splits splits to allow higher SM utilization on B200.
        @parameter
        fn launch_combine[n_splits: Int, wph: Int]() raises:
            mla_decode_combine_partial_outputs[
                output_type=output_type,
                accum_type=AccumType,
                head_dim=v_depth,
                num_splits=n_splits,
                ragged=ragged,
                warps_per_head=wph,
            ](
                o_accum_split,
                lse_accum_split,
                output,
                input_row_offsets_ptr,
                batch_size,
                q_max_seq_len,
                Int(num_heads),
                ctx,
            )

        # Choose warps_per_head (wph) for the combine kernel.
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
        #   ctas >= 4096 AND np <= 4 AND cache_len <= 1280:  wph=1
        #   ctas >= 2048 AND np > 4:                   wph=2
        #   ctas >= 512:                               wph=4
        #   ctas < 512 (small grid):                   wph=8
        var combine_ctas_base = (
            batch_size * q_max_seq_len * ceildiv(num_heads, 4)
        )
        if (
            combine_ctas_base >= 4096
            and num_partitions <= 4
            and effective_max_cache_len <= 1280
        ):
            # Very large combine grid with small split count AND short KV
            # cache: use wph=1 to aggressively minimize CTA count. With
            # only 1-4 partials to reduce, per-CTA work is negligible. The
            # bottleneck is purely wave count (CTAs / sm_count).
            #
            # The cache length guard (effective_max_cache_len <= 1280, i.e.
            # <=10 pages at page_size=128) ensures we only use wph=1 when the
            # decode kernel finishes quickly (short cache per split), making
            # combine the dominant cost. We compare against
            # effective_max_cache_len directly for clarity, independent of
            # the split_page_size used.
            # For larger caches (e.g., bs=256/cl=2048 with effective ~2049),
            # the decode takes longer and combine is partially hidden
            # behind PDL overlap, so wph=4 with more intra-CTA
            # parallelism is preferred.
            #
            # Threshold 4096 corresponds to bs >= 128 for DeepSeek V3/R1
            # (num_heads=128). Example configs:
            #   bs=128, cl=1024: np=2, wph=1, 2048 CTAs (14 waves)
            #     vs wph=4: 8192 CTAs (56 waves). 4x fewer waves.
            #   bs=128, cl=128:  np=2, wph=1, 2048 CTAs (14 waves)
            #   bs=256, cl=2048: effective ~2049 > 1280, wph=4 (excluded by guard)

            comptime for i in range(2, max_num_splits + 1):
                if num_partitions == i:
                    launch_combine[i, 1]()
        elif combine_ctas_base >= 2048 and num_partitions > 4:
            # Large combine grid with many splits: use wph=2 to minimize the
            # number of combine CTAs. Each CTA has enough reduction work
            # (>4 splits) to amortize launch overhead.

            comptime for i in range(5, max_num_splits + 1):
                if num_partitions == i:
                    launch_combine[i, 2]()
        elif combine_ctas_base >= 512:
            # Medium combine grid, OR large grid with few splits (np <= 4):
            # use wph=4. For few-split large-grid configs (e.g., bs=128/1K
            # with np=2), the combine work per CTA is tiny and wph=4 hides
            # per-CTA latency better than wph=2.

            comptime for i in range(2, max_num_splits + 1):
                if num_partitions == i:
                    launch_combine[i, 4]()
        else:
            # Small combine grid (< 512 CTAs at wph=2): maximize intra-head
            # parallelism with wph=8. The extra CTAs from higher wph are not
            # a concern since the grid is small.

            comptime for i in range(2, max_num_splits + 1):
                if num_partitions == i:
                    launch_combine[i, 8]()
    else:
        comptime SplitAccumType = NullPointer[AccumType]
        var lse_accum_split_ptr: SplitAccumType = {}

        mla_decode_sm100_sink_split_k[
            q_type=q_type,
            q_layout=q_layout,
            k_t=k_t,
            output_type=output_type,
            mask_t=mask_t,
            score_mod_t=score_mod_t,
            valid_layout=valid_layout,
            config=config,
            depth=depth,
            num_heads=num_heads,
            SplitAccumType=SplitAccumType,
            group=group,
            use_score_mod=use_score_mod,
            ragged=ragged,
            _is_cache_length_accurate=_is_cache_length_accurate,
            decoding_warp_split_k=False,
            split_page_size=split_page_size,
        ](
            q,
            k,
            output,
            lse_accum_split_ptr,
            scale,
            batch_size,
            block_z,
            num_partitions,
            max_cache_valid_length,
            q_max_seq_len,
            valid_length,
            mask,
            score_mod,
            scales_ptr,
            ctx,
        )


fn mla_decode_sm100_sink_split_k[
    q_type: DType,
    q_layout: Layout,
    k_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    *,
    score_mod_t: ScoreModTrait,
    valid_layout: Layout,
    config: MHAConfig,
    depth: Int,
    num_heads: Int,
    SplitAccumType: OptionalPointer,
    group: Int,
    use_score_mod: Bool,
    ragged: Bool,
    _is_cache_length_accurate: Bool,
    decoding_warp_split_k: Bool,
    split_page_size: Int = 128,
](
    q: LayoutTensor[
        q_type, q_layout, address_space = AddressSpace.GENERIC, ...
    ],
    k: k_t,
    output: LayoutTensor[address_space = AddressSpace.GENERIC, ...],
    lse_accum_split_ptr: SplitAccumType,
    scale: Float32,
    batch_size: Int,
    block_z: Int,
    num_partitions: Int,
    max_cache_valid_length: Int,  # longest KV cache entry
    q_max_seq_len: Int,
    valid_length: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    mask: mask_t,
    score_mod: score_mod_t,
    scales_ptr: UnsafePointer[Scalar[DType.float32], origin=MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    comptime _scale_block_size = k_t.quantization_granularity if k_t.quantization_enabled else 0
    comptime mla_config = MLA_SM100_Decode_Config(
        num_q_heads=num_heads,
        group=group,  # num_q_heads/h_k(1)
        depth=(depth - 64),  # 512
        q_depth=depth,  # 576
        dtype_size=size_of[q_type](),
        kv_type_size=size_of[k_t.dtype](),
        swizzle_mode=config.swizzle_mode,
        kv_mma_swizzle_mode=config.swizzle_mode,
        page_size=k_t.page_size,
        decoding_warp_split_k=decoding_warp_split_k,
        split_page_size=split_page_size,
        scale_block_size=_scale_block_size,
    )
    var num_rows_q = num_matrix_view_rows_decode(q)
    q_ptr = rebind[UnsafePointer[Scalar[q_type], origin=MutAnyOrigin]](
        q.to_device_buffer(ctx).unsafe_ptr()
    )
    q_tma_op = tma_tile_qo[
        swizzle_mode = mla_config.swizzle_mode,
        BM = mla_config.BM,  # tile_m =64
        BK = mla_config.BK0,  # tile_n =576
        depth = mla_config.q_depth,
    ](ctx, q_ptr, num_rows_q)

    k_tma_op = k.create_tma_tile[
        BN = mla_config.BK1,  # tile_m =64
        depth = mla_config.q_depth,
        BK = mla_config.BK0,  # tile_n =576
        swizzle_mode = mla_config.kv_tma_swizzle_mode,
    ](ctx)
    o_ptr = rebind[UnsafePointer[Scalar[output_type], origin=MutAnyOrigin]](
        output.to_device_buffer(ctx).unsafe_ptr()
    )
    var num_rows_o = num_matrix_view_rows_decode(output)
    o_tma_op = tma_tile_qo[
        swizzle_mode = mla_config.swizzle_mode,
        BM = mla_config.out_rows,
        BK = mla_config.BN,
        depth = mla_config.depth,
    ](ctx, o_ptr, num_rows_o)

    if ragged:
        comptime ValidLengthType = NonNullPointer[DType.uint32]
        var valid_len: ValidLengthType = {
            valid_length.to_device_buffer(ctx).unsafe_ptr()
        }
        launch_mla_sm100_decode_enqueue_kernel[
            q_type=q_type,
            KVLUTType=k_t,
            output_type=output_type,
            SplitAccumType=SplitAccumType,
            MaskType=mask_t,
            ScoreModType=score_mod_t,
            config=mla_config,
            use_score_mod=use_score_mod,
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
            max_cache_valid_length,
            q_max_seq_len,
            valid_len,
            mask,
            score_mod,
            scales_ptr,
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
            ScoreModType=score_mod_t,
            config=mla_config,
            use_score_mod=use_score_mod,
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
            max_cache_valid_length,
            q_max_seq_len,
            valid_len,
            mask,
            score_mod,
            scales_ptr,
            ctx,
        )


@always_inline
fn launch_mla_sm100_decode_enqueue_kernel[
    q_type: DType,
    KVLUTType: MHAOperand,
    output_type: DType,
    SplitAccumType: OptionalPointer,
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait,
    config: MLA_SM100_Decode_Config,
    use_score_mod: Bool,
    ValidLengthType: OptionalPointer,
    _is_cache_length_accurate: Bool = False,
    ragged: Bool = False,
](
    q_tma: QOTMATile[
        dtype=q_type,
        BM = config.BM,  # tile_m =64
        BK = config.BK0,  # tile_n =576
        swizzle_mode = config.swizzle_mode,
    ],
    k_tma: KVTMATile[
        dtype = KVLUTType.dtype,
        swizzle_mode = config.kv_tma_swizzle_mode,
        BN = config.BK1,  # tile_m =64
        BK = config.BK0,  # tile_n =576
    ],
    o_tma: QOTMATile[
        dtype=output_type,
        BM = config.out_rows,
        BK = config.BN,
        swizzle_mode = config.swizzle_mode,
    ],
    kv_lut: KVLUTType,
    lse_accum_split_ptr: SplitAccumType,
    scale: Float32,
    batch_size: Int,
    block_z: Int,
    num_partitions: Int,
    max_cache_valid_length: Int,  # longest KV cache entry,
    q_max_seq_len: Int,
    valid_len: ValidLengthType,
    mask: MaskType,
    score_mod: ScoreModType,
    scales_ptr: UnsafePointer[Scalar[DType.float32], origin=MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    var mla_decode_pack = MLA_Decode_Pack[
        ValidLengthType=ValidLengthType,
        MaskType=MaskType,
        ScoreModType=ScoreModType,
        SplitAccumType=SplitAccumType,
    ](mask, score_mod, valid_len, lse_accum_split_ptr)
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
        "config.BN:",
        config.BN,
        "config.BK0:",
        config.BK0,
        "config.BK1:",
        config.BK1,
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
        "max_cache_valid_length:",
        max_cache_valid_length,
        "output_tile_width:",
        (config.BN // 2) * (4 // size_of[output_type]()),
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
        "Max Cache Valid Length:",
        max_cache_valid_length,
    )

    # Dispatch to the appropriate kernel:
    # - FP8: float8_e4m3fn KV cache (handles both tensorwise and blockwise scaling
    #   via @parameter if Self.config.scale_block_size > 0 conditionals)
    # - BF16: bfloat16 KV cache (no conversion needed)
    comptime kernel = MLA_SM100_Decode_KV_FP8[
        q_type=q_type,
        KVLUTType=KVLUTType,
        output_type=output_type,
        SplitAccumType=SplitAccumType,
        MaskType=MaskType,
        ScoreModType=ScoreModType,
        config=config,
        use_score_mod=use_score_mod,
        ValidLengthType=ValidLengthType,
        _is_cache_length_accurate=_is_cache_length_accurate,
        ragged=ragged,
    ].kernel if KVLUTType.dtype == DType.float8_e4m3fn else MLA_SM100_Decode_KV_BF16[
        q_type=q_type,
        KVLUTType=KVLUTType,
        output_type=output_type,
        SplitAccumType=SplitAccumType,
        MaskType=MaskType,
        ScoreModType=ScoreModType,
        config=config,
        use_score_mod=use_score_mod,
        ValidLengthType=ValidLengthType,
        _is_cache_length_accurate=_is_cache_length_accurate,
        ragged=ragged,
    ].kernel
    # Enable PDL (Programmatic Dependent Launch) for split-K mode to chain
    # the MLA decode kernel with the combine kernel, reducing host synchronization.
    comptime pdl_level = PDLLevel.OVERLAP_AT_END if config.decoding_warp_split_k else PDLLevel.OFF
    ctx.enqueue_function[kernel, kernel](
        q_tma,
        k_tma,
        o_tma,
        kv_lut,
        scale,
        batch_size,
        q_max_seq_len,
        num_partitions,
        max_cache_valid_length,
        mla_decode_pack,
        scales_ptr,
        grid_dim=grid_dim,
        block_dim=block_dim,
        shared_mem_bytes=config.smem_used,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(config.smem_used)
        ),
        attributes=pdl_launch_attributes(pdl_level),
    )
