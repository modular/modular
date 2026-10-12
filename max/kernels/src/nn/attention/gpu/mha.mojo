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
"""GPU flash-attention kernels and dispatch logic for prefill and decode.

Implements FA2 and FA3 flash-attention for NVIDIA and AMD GPUs, a naive
two-BMM reference path, split-K decode partitioning, and the host-side
dispatch layer (`flash_attention_dispatch`) that selects among them based
on dtype, head depth, and target architecture. Dispatch uses `TileTensor`
operands; the legacy FA2 kernels still use `LayoutTensor` iterators.
"""

from std.math import ceildiv, recip
from std.math.uutils import umod, ufloordiv, udivmod
from std.os.env import getenv
from std.math.constants import log2e
from std.collections import OptionalReg
from std.sys import (
    CompilationTarget,
    align_of,
    get_defined_bool,
    get_defined_int,
    has_amd_rdna_gpu_accelerator,
    default_accelerator,
    is_amd_gpu,
    is_nvidia_gpu,
    simd_width_of,
    size_of,
)
from std.sys.info import _is_amd_rdna
import max.gpu.primitives.warp as warp
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from max.algorithm import elementwise
from std.algorithm.functional import tile_and_unswitch, unswitch, vectorize
from std.bit import next_power_of_two
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_dim,
    block_idx,
    global_idx,
    lane_id,
    thread_idx,
    warp_id,
)
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.host import Dim as LaunchDim
from max.gpu.host import FuncAttribute
from max.gpu.host.info import A100, H100, GPUInfo, _is_sm10x_gpu
from max.gpu.memory import (
    async_copy_commit_group,
    async_copy_wait_all,
    external_memory,
)
from kv_cache.types import KVCacheT
from layout import (
    lt_to_tt_idx,
    Coord,
    Idx,
    IntTuple,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
    TensorLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
    coord_to_index_list,
)
from layout.layout import *
from layout.layout_tensor import (
    LayoutTensorIter,
    copy_dram_to_sram_async,
    copy_local_to_dram,
    copy_local_to_shared,
    copy_sram_to_dram,
)
from layout.swizzle import make_swizzle
from layout.tile_tensor import stack_allocation
from layout.tensor_core import get_fragment_size, get_mma_shape
from linalg.matmul.gpu._multistage_gemm_gpu import multistage_mma
from linalg.utils_gpu import _apple_m5_allow_lossy_f32_attention
from std.memory import unsafe_stack_allocation

from .amd_rdna.attention import AttentionRDNA
from .amd_rdna.mha_decode import AttentionRDNA
from .amd_rdna.mha_prefill import AttentionRDNA
from .apple.naive_fa_decode import (
    naive_fa_decode_apple,
    naive_fa_decode_apple_supports_depth,
)
from .apple.fa_prefill import FA_PREFILL_APPLE_MAX_HEAD_DIM, fa_prefill_apple
from .amd_structured.attention import Attention
from .amd_structured.config import (
    _MHA_DECODE_FOLD_MAX_ROWS,
    _MHA_DECODE_FOLD_WM,
    _mha_decode_fold_warp_m,
    decode_mma_shape,
    mha_decode_fold_tile_q_seq_len,
    mha_decode_fold_wide_mma,
)
from .amd_structured.mha_prefill_v2 import (
    MhaConfigV2,
    MhaPrefillV2,
    mha_prefill_v2,
    mha_prefill_v2_ragged,
)
from .amd_structured.mha_decode import Attention
from .amd_structured.mha_decode_streaming import Attention
from .amd_structured.mha_prefill import Attention
from nn.attention.mha_mask import (
    CausalMask,
    MaterializedMask,
    MHAMask,
    NullMask,
    TileMaskStatus,
)
from nn.attention.mha_operand import (
    KVCacheMHAOperand,
    MHAOperand,
    TileTensorMHAOperand,
    RaggedMHAOperand,
)
from nn.attention.gpu.mha_decode_partition_heuristic import (
    mha_decoding_max_num_partitions,
    mha_decoding_num_partitions,
)
from nn.attention.gpu.nvidia.sm90.mha import mha_sm90_dispatch
from nn.attention.gpu.nvidia.common import (
    ImmutTileTensor1D,
    immut_tile_tensor_1d,
)
from nn.attention.gpu.nvidia.sm100.dispatch import (
    mha_sm100_dispatch as mha_sm100_2q_dispatch,
)
from nn.attention.gpu.nvidia.sm100.mha_depth512 import (
    mha_sm100_depth512_dispatch,
)
from nn.attention.gpu.nvidia.sm100.fa4_splitk_combine import P_MAX
from nn.attention.mha_utils import (
    DynamicInt,
    FlashAttentionAlgorithm,
    MHA_PDL_LEVEL,
    MHAConfig,
    NoPartition,
    SplitKPartition,
    StaticInt,
    _copy_frag_to_smem,
    _kernel_mask,
    get_start_and_end_for_partitions,
)
from max.runtime.tracing import Trace, TraceLevel, trace_arg

from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type, min_or_neg_inf
from std.utils.static_tuple import StaticTuple

from nn.softmax import (
    _exp2_concrete,
    _exp_concrete,
    _online_softmax_iter_for_mma_output,
    _online_softmax_iter_for_mma_output_split_warp_reduce,
    _softmax_gpu,
    softmax_inline,
)

# ===-----------------------------------------------------------------------===#
# Flash attention
# ===-----------------------------------------------------------------------===#


def flash_attention[
    dtype: DType,
    q_layout: TensorLayout,
    //,
    config: MHAConfig[dtype] = {
        q_layout.static_shape[2],
        q_layout.static_shape[3],
    },
    decoding_warp_split_k: Bool = False,
    naive_kernel: Bool = False,
    sink: Bool = False,
](
    output: TileTensor[mut=True, address_space=.GENERIC, ...],
    q: TileTensor[mut=False, dtype, q_layout, address_space=.GENERIC, ...],
    k: TileTensor[mut=False, address_space=.GENERIC, ...],
    v: TileTensor[mut=False, address_space=.GENERIC, ...],
    mask: TileTensor[mut=False, address_space=.GENERIC, ...],
    scale: Float32,
    context: DeviceContext,
    num_partitions: Optional[Int] = None,
    sink_weights: OptionalReg[ImmutTileTensor1D[dtype]] = None,
) raises:
    """Run flash attention with a dense mask tensor on the current device.

    Wraps the mask tensor in a `MaterializedMask` and delegates to the
    shared dense implementation. Selects the flash-attention algorithm variant
    (FA2 / FA3 / naive) based on `config.algorithm` and the detected GPU.

    Parameters:
        dtype: Element type of the query tensor.
        q_layout: Compile-time layout of the query tensor.
        config: Tile/pipeline configuration; defaults are derived from the
            query layout's last two dimensions.
        decoding_warp_split_k: Enable warp-level split-K for decode.
        naive_kernel: Force the fallback naive attention kernel.
        sink: Enable attention-sink mode (first tokens always attend).

    Args:
        output: Destination tensor for attention output.
        q: Query tensor.
        k: Key tensor.
        v: Value tensor.
        mask: Dense attention mask tensor.
        scale: Softmax temperature scale applied to Q·Kᵀ.
        context: GPU device context for kernel dispatch.
        num_partitions: Override the number of split-K partitions; `None`
            selects automatically.
        sink_weights: Optional sink-token weight tensor for attention sinks.
    """

    @inline(.always)
    def description_fn() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("q", coord_to_index_list(q.layout.shape_coord())),
                    trace_arg("k", coord_to_index_list(k.layout.shape_coord())),
                    trace_arg("v", coord_to_index_list(v.layout.shape_coord())),
                    trace_arg(
                        "output",
                        coord_to_index_list(output.layout.shape_coord()),
                    ),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=context.default_device_info.api](
        "flash_attention",
        Trace[
            TraceLevel.OP, target=context.default_device_info.api
        ]._get_detail_str(description_fn),
        task_id=Int(context.id()),
    ):
        return _flash_attention_dense[
            config=config,
            decoding_warp_split_k=decoding_warp_split_k,
            naive_kernel=naive_kernel,
            sink=sink,
        ](
            output,
            q,
            k,
            v,
            MaterializedMask(
                TileTensor(
                    mask.unsafe_ptr(),
                    row_major(mask.layout.shape_coord()),
                )
            ),
            scale,
            context,
            num_partitions,
            sink_weights=sink_weights,
        )


# Speculative-decode token fold (AMD). A verify step has query length S > 1, so
# decode's `seq_len == 1` gate would send it to prefill. The fold instead packs
# the S tokens into the MMA M dimension as heads-inner rows
# (row = token*group + head), so one CTA covers all S tokens of its KV head.
# At `group > 1` that spares prefill's `group` re-reads of each KV head; at
# `group == 1` it buys the key-dim split-K prefill lacks, which otherwise pins
# parallelism at `num_heads*batch` CTAs (half-idle on 256 CUs below batch 16).
#
# Each eligible S is its own instantiation, so the cap is a compile-time budget,
# not a hardware limit. It must clear the largest query length a caller
# DECLARES: for a 1+K speculative cycle that is K+2, not K+1 — the target
# verifies K+1 tokens and an EAGLE3 draft's step-0 catch-up declares that width
# plus one, so K <= 7 needs 9.
comptime _MHA_DECODE_FOLD_MAX_S = 9


@inline(.always)
def _mha_decode_fold_ok[
    dtype: DType,
    depth: Int,
    num_heads: Int,
    group: Int,
    S: Int,
    sink: Bool = False,
    use_valid_length: Bool = False,
    ragged: Bool = False,
    check_mask_during_decoding: Bool = False,
]() -> Bool:
    """Return whether AMD decode can fold `S` query tokens into the M dimension.

    Both the `is_token_generation` gate and the dispatch ladder call this, so the
    shapes routed to decode and the kernels instantiated cannot drift apart. An
    ineligible shape keeps taking the prefill path: slower, but correct.

    Parameters:
        dtype: Element type shared by Q, K, and V.
        depth: Attention head depth.
        num_heads: Number of query heads.
        group: GQA group size (query heads per KV head).
        S: Number of query tokens per sequence.
        sink: Whether attention-sink mode is active.
        use_valid_length: Whether per-sequence valid lengths are read, which
            allows a sequence shorter than `S`.
        ragged: Whether query rows are packed by `input_row_offsets`.
        check_mask_during_decoding: Whether the mask needs a per-tile status
            check during decode.

    Returns:
        `True` when the token fold applies to this shape.
    """

    return (
        default_accelerator().is_amd_gpu()
        and not has_amd_rdna_gpu_accelerator()
        # fp16 shares bf16's decode MMA shape and fp32 has its own, but neither
        # is tested through the fold; widen once a test covers them.
        and (dtype == .bfloat16 or dtype.is_float8())
        # The row bounds below count `_MHA_DECODE_FOLD_WM`-row M-tiles, so this
        # arm's starting shape must be that tall. Asks for the UNFOLDED shape on
        # purpose: a width that clears `mha_decode_fold_wide_mma` overrides it
        # to 32 rows, which tile 16 evenly either way.
        and decode_mma_shape[dtype, depth, num_heads]()[0]
        == _MHA_DECODE_FOLD_WM
        # At WN == BN each warp's K and V register tiles span the whole depth,
        # so anything wider spills.
        and depth <= 128
        # The sink lookup `q_head_idx()` (= lane % MMA_M) is the folded head only
        # while num_heads == MMA_M.
        # TODO: derive the sink index from the absolute fold row instead.
        and not sink
        # A PADDED batch can hold a sequence shorter than S, whose pad rows'
        # split-K stats are never written and which `mha_splitk_reduce` does not
        # skip — its `scale > 0` blend guard accepts uninitialized floats. Ragged
        # is exempt: pad rows are the next sequence's, written by its own CTA.
        # TODO: skip pad rows in `mha_splitk_reduce` to lift this.
        and not (use_valid_length and not ragged)
        # `Attention.mask_status` bases its S-row span at `num_keys - S`, which
        # underflows for a sequence holding fewer than S keys (a mixed ragged
        # batch can). Only `check_mask_during_decoding` masks consult it.
        # TODO: clamp that span to the runtime seq_len and widen this.
        and not check_mask_during_decoding
        # `mha_decode_streaming` carries no fold geometry, and dispatch routes the
        # same `Attention` to it under this flag; instantiating a fold arm would
        # turn an opt-in perf flag into a build break on its `comptime assert`.
        and not get_defined_bool["MHA_STREAMING_DECODE", False]()
        and S > 1
        and S <= _MHA_DECODE_FOLD_MAX_S
        # A CTA owns `group*S` rows and addresses them with ONE Q/O row stride
        # (`Attention._fold_token_strided` picks it), which admits only two
        # groups. Each brings its own block geometry:
        and (
            (
                # Single KV head: `num_heads*S` rows stack over 16-row M-tiles,
                # so they must tile evenly and need two — at num_warps_m == 1 P
                # stops being warp-local and falls back to the register-resident
                # chain never exercised there. `_mha_decode_fold_warp_m` picks
                # how many tiles a warp takes, within both bounds.
                group == num_heads
                and (num_heads * S) % _MHA_DECODE_FOLD_WM == 0
                and num_heads * S >= 2 * _MHA_DECODE_FOLD_WM
                and num_heads * S <= _MHA_DECODE_FOLD_MAX_ROWS
            )
            or (
                # One query head per KV head: a CTA owns just its S token rows,
                # which fit one 16-row M-tile, so it keeps the single-token
                # geometry (`_fold_narrow` below) instead of stacking tiles. No
                # 2-warp floor: that geometry splits N across 4 warps, so P
                # stays SMEM-backed via `BN != WN`. The row bound is redundant
                # while the cap is <= WM; keep it so raising the cap past WM
                # cannot silently overflow the tile. WM stands in for that
                # geometry's BM here — the two coincide at 16, and a wider BM
                # would only make the bound conservative.
                group == 1
                and S <= _MHA_DECODE_FOLD_WM
            )
        )
    )


def get_mha_decoding_num_partitions[
    num_heads: Int, group: Int
](batch_size: Int, num_keys: Int, ctx: DeviceContext) raises -> Int:
    """Return the recommended number of split-K partitions for MHA decoding.

    Computes the number of CTAs (partitions) that maximally utilise the GPU
    for decoding, given the batch size and key-sequence length. The result
    feeds the split-K launcher and is also stored in
    `MHADecodeDispatchMetadata`.

    Parameters:
        num_heads: Total number of query heads.
        group: GQA group size (query heads per key/value head).

    Args:
        batch_size: Number of sequences in the batch.
        num_keys: Maximum key-sequence length (cache length).
        ctx: GPU device context used to query SM count and other properties.

    Returns:
        The number of split-K partitions to launch.
    """

    return mha_decoding_num_partitions(
        batch_size,
        num_keys,
        num_heads // group,
        ctx,
    )


def get_mha_decoding_max_num_partitions[
    num_heads: Int, group: Int
](batch_size: Int, num_keys: Int, ctx: DeviceContext) raises -> Int:
    """Return the maximum number of split-K partitions for CUDA-graph-stable launches.

    Returns an upper bound on the partition count that remains constant for
    a given batch size and key length, allowing the kernel grid to be
    captured in a CUDA graph. CTAs whose partition index exceeds the
    runtime `num_partitions` early-exit without doing work.

    Parameters:
        num_heads: Total number of query heads.
        group: GQA group size (query heads per key/value head).

    Args:
        batch_size: Number of sequences in the batch.
        num_keys: Maximum key-sequence length (cache length).
        ctx: GPU device context used to query SM count and other properties.

    Returns:
        The stable upper bound on the number of split-K partitions.
    """

    return mha_decoding_max_num_partitions(
        batch_size,
        num_keys,
        num_heads // group,
        ctx,
    )


@fieldwise_init
struct MHADecodeDispatchMetadata(TrivialRegisterPassable):
    """Runtime metadata required to dispatch an MHA decode kernel launch.

    Bundles the batch size, maximum query sequence length, split-K partition
    count, and maximum cache length so that callers can construct the correct
    grid shape for the decode kernel without recomputing partition counts.
    Use `from_runtime_values()` to construct this from raw scalars.
    """

    var batch_size: Int
    var q_max_seq_len: Int
    var num_partitions: Int
    var max_cache_valid_length: Int

    @staticmethod
    @inline(.always)
    def from_runtime_values[
        num_heads: Int,
        group: Int,
    ](
        batch_size: Int,
        q_max_seq_len: Int,
        max_cache_valid_length: Int,
        ctx: DeviceContext,
    ) raises -> Self:
        return Self(
            batch_size,
            q_max_seq_len,
            get_mha_decoding_num_partitions[num_heads, group](
                batch_size,
                max_cache_valid_length,
                ctx,
            ),
            max_cache_valid_length,
        )


def flash_attention_hw_supported[qkv_type: DType]() -> Bool:
    """Return `True` if the current GPU supports flash attention for `qkv_type`.

    NVIDIA GPUs support all dtypes. AMD GPUs require `bfloat16` or a
    float8 type. Returns `False` on CPUs and unsupported GPU types so
    callers can gracefully fall back to a reference implementation.

    Parameters:
        qkv_type: The element data type of the Q/K/V tensors.

    Returns:
        `True` when flash attention is available for `qkv_type` on the
        detected accelerator.
    """

    return default_accelerator().is_nvidia_gpu() or (
        (qkv_type == .bfloat16 or qkv_type.is_float8())
        and default_accelerator().is_amd_gpu()
    )


def depth_supported_by_gpu[
    depth: Int,
    mask_t: MHAMask,
    config: MHAConfig,
    info: GPUInfo,
]() -> Bool:
    """Return `True` if the given head depth is supported for flash attention on this GPU.

    Checks the combination of `depth`, GPU architecture (`info`), and
    algorithm variant to decide whether the optimised kernel path is
    available. For example, depth 128 is universally supported, depth 64
    requires SM80+, depth 512 requires SM100 or AMD.

    Parameters:
        depth: Attention head depth (key/value dimension per head).
        mask_t: Mask type; some depths require `mask_safe_out_of_bounds`.
        config: MHA tile configuration, used to check the algorithm variant.
        info: GPU architecture descriptor.

    Returns:
        `True` when the optimised flash-attention kernel supports `depth`
        on the given GPU.
    """
    comptime target = CompilationTarget.from[info]()

    comptime is_sm100 = _is_sm10x_gpu(info)
    comptime is_sm90or100 = is_sm100 or (info == H100)

    comptime head_depth_supported = depth == 128 or (
        depth == 64 and (is_sm90or100 or info == A100 or target.is_amd_gpu())
    ) or (
        depth == 80 and config.dtype == .bfloat16 and target.is_amd_gpu()
    ) or (
        depth == 256
        and (
            target.is_amd_gpu()
            or (is_sm90or100 and mask_t.mask_safe_out_of_bounds)
        )
    ) or (
        depth in (72, 80, 96)
        and is_sm90or100
        and config.algorithm == FlashAttentionAlgorithm(3)
    ) or (
        (is_sm100 or target.is_amd_gpu()) and depth == 512
    )
    return head_depth_supported


# Entry point for flash_attention with batch_size > 1.
@inline(.always)
def _flash_attention_kv_cache[
    cache_t: KVCacheT,
    mask_t: MHAMask,
    dtype: DType,
    q_layout: TensorLayout,
    //,
    config: MHAConfig[dtype] = {
        q_layout.static_shape[q_layout.rank - 2],
        q_layout.static_shape[q_layout.rank - 1],
    },
    ragged: Bool = False,
    sink: Bool = False,
    decoding_warp_split_k: Bool = False,
    naive_kernel: Bool = False,
](
    output: TileTensor[mut=True, address_space=.GENERIC, ...],
    q: TileTensor[mut=False, dtype, q_layout, address_space=.GENERIC, ...],
    k: cache_t,
    v: cache_t,
    mask_functor: mask_t,
    valid_length: ImmutTileTensor1D[.uint32],
    scale: Float32,
    ctx: DeviceContext,
    q_max_seq_len: Optional[Int] = None,
    kv_input_row_offsets: OptionalReg[ImmutTileTensor1D[.uint32]] = None,
    num_partitions: Optional[Int] = None,
    sink_weights: OptionalReg[ImmutTileTensor1D[dtype]] = None,
    decode_dispatch_metadata: OptionalReg[MHADecodeDispatchMetadata] = None,
) raises:
    """Shared implementation behind the KV-cache `flash_attention` overloads.

    Picks the flash-attention variant for the detected GPU and launches it
    through `flash_attention_dispatch`. All tensor operands are
    `TileTensor`s.
    """
    comptime assert (
        ragged or q.rank == 4
    ), "only support rank 4 inputs for non-ragged inputs."
    comptime assert (
        not ragged or q.rank == 3
    ), "only support rank 3 inputs for ragged inputs."

    # Native FP8 path (SM100 tcgen05, or AMD MFMA on gfx950). K/V are FP8 in
    # the paged cache.  Q@K^T and P@V run as FP8 MMAs at tensorwise scale=1.
    # Output is BF16.
    comptime is_native_fp8_bf16_out = (
        q.dtype.is_float8()
        and cache_t.dtype.is_float8()
        and output.dtype == .bfloat16
        and (_is_sm10x_gpu(ctx.default_device_info) or ctx.target.is_amd_gpu())
    )

    comptime assert (
        q.dtype == cache_t.dtype == output.dtype or is_native_fp8_bf16_out
    ), (
        "Q, K, V, output should have same dtype, or Q=K=V=float8,"
        " output=bfloat16 for the native FP8 path."
    )
    comptime assert (
        q.dtype == .float32
        or q.dtype.is_half_float()
        or (q.dtype.is_float8() and ctx.target.is_amd_gpu())
        or is_native_fp8_bf16_out
    ), (
        "Only support single, half, float8 (AMD), and float8->bfloat16"
        " (SM100 or AMD) precision."
    )

    # TODO docstring
    @inline(.always)
    def description_fn() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("q", coord_to_index_list(q.layout.shape_coord())),
                    trace_arg(
                        "output",
                        coord_to_index_list(output.layout.shape_coord()),
                    ),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=ctx.default_device_info.api](
        "flash_attention",
        Trace[
            TraceLevel.OP, target=ctx.default_device_info.api
        ]._get_detail_str(description_fn),
        task_id=Int(ctx.id()),
    ):
        var max_prompt_len: Int
        var num_keys = Int(k.max_context_length())

        if q_max_seq_len:
            max_prompt_len = q_max_seq_len.value()
        elif decode_dispatch_metadata:
            max_prompt_len = decode_dispatch_metadata.value().q_max_seq_len
        else:
            max_prompt_len = Int(k.max_prompt_length())

        # Whether head and depth are static. With BSHD, B and S are dynamic.
        # H and D are always known for opaque KVCache types, we only check Q.
        # fmt: off
        comptime head_depth_known = q.static_shape[q.rank - 2] != -1 and q.static_shape[q.rank - 1] != -1
        comptime depth = q.static_shape[q.rank-1]
        comptime gpu_info = ctx.default_device_info
        comptime head_depth_supported = depth_supported_by_gpu[depth, mask_t, config, gpu_info]()
        comptime flash_attention_applicable = flash_attention_hw_supported[dtype]() and head_depth_known and head_depth_supported and not naive_kernel
        # fmt: on
        comptime kv_num_heads = cache_t.kv_params.num_heads

        # TODO: This helps differentiate between CE/TG. Not batch-specific.
        #       We'll just implement a flag on the cache object which is true
        #       when the batch contains all cache_lens == 0. Remove this when
        #       such flag (part of ContiguousKVCache) is implemented.
        # AMD decode also handles a speculative-verify query length S > 1 by
        # folding the S tokens into the MMA M dimension. `max_prompt_len` is the
        # batch maximum, so a mixed batch dispatches at S = max and each
        # sequence's own CTA clamps to its runtime length; a real prefill batch is
        # unaffected because its max exceeds _MHA_DECODE_FOLD_MAX_S. This
        # `comptime for` mirrors the dispatch ladder's instantiation set, and both
        # read the shape from `config` (which `flash_attention_dispatch`
        # comptime-asserts against Q's layout), so a routed shape always has a
        # kernel. `use_valid_length=True` because this overload exposes no
        # `_use_valid_length` parameter and dispatch takes that default.
        var fold_seq_len_ok = False
        comptime for S in range(2, _MHA_DECODE_FOLD_MAX_S + 1):
            comptime if _mha_decode_fold_ok[
                dtype,
                config.depth,
                config.num_heads,
                config.num_heads // kv_num_heads,
                S,
                sink=sink,
                use_valid_length=True,
                ragged=ragged,
                check_mask_during_decoding=mask_t.check_mask_during_decoding,
            ]():
                if max_prompt_len == S:
                    fold_seq_len_ok = True

        var is_token_generation = (
            max_prompt_len == 1 or fold_seq_len_ok
        ) and not k.empty_cache()

        var k_operand = KVCacheMHAOperand(k)
        var v_operand = KVCacheMHAOperand(v)

        flash_attention_dispatch[
            kv_num_heads=kv_num_heads,
            config=config,
            ragged=ragged,
            sink=sink,
            _is_flash_attention_applicable=flash_attention_applicable,
            decoding_warp_split_k=decoding_warp_split_k,
        ](
            output,
            q,
            k_operand,
            v_operand,
            mask_functor,
            max_prompt_len,
            num_keys,
            scale,
            is_token_generation,
            ctx,
            valid_length,
            kv_input_row_offsets,
            num_partitions,
            sink_weights,
            decode_dispatch_metadata=decode_dispatch_metadata,
        )


def flash_attention[
    cache_t: KVCacheT,
    mask_t: MHAMask,
    dtype: DType,
    output_type: DType,
    q_tt_layout: TensorLayout,
    output_tt_layout: TensorLayout,
    valid_length_tt_layout: TensorLayout,
    //,
    config: MHAConfig[dtype] = {
        q_tt_layout.static_shape[q_tt_layout.rank - 2],
        q_tt_layout.static_shape[q_tt_layout.rank - 1],
    },
    ragged: Bool = False,
    sink: Bool = False,
    decoding_warp_split_k: Bool = False,
    naive_kernel: Bool = False,
](
    output: TileTensor[
        mut=True, output_type, output_tt_layout, address_space=.GENERIC, ...
    ],
    q: TileTensor[mut=False, dtype, q_tt_layout, address_space=.GENERIC, ...],
    k: cache_t,
    v: cache_t,
    mask_functor: mask_t,
    valid_length: TileTensor[
        mut=False,
        .uint32,
        valid_length_tt_layout,
        address_space=.GENERIC,
        ...,
    ],
    scale: Float32,
    ctx: DeviceContext,
    q_max_seq_len: Optional[Int] = None,
    kv_input_row_offsets: OptionalReg[ImmutTileTensor1D[.uint32]] = None,
    num_partitions: Optional[Int] = None,
    sink_weights: OptionalReg[ImmutTileTensor1D[dtype]] = None,
    decode_dispatch_metadata: OptionalReg[MHADecodeDispatchMetadata] = None,
) raises:
    """Runs flash attention with KV-cache operands.

    Delegates native tensor operands to the shared KV-cache implementation.

    Parameters:
        cache_t: KV-cache type backing the key and value tensors (inferred).
        mask_t: Attention mask type implementing `MHAMask` (inferred).
        dtype: Element type of the query tensor (inferred).
        output_type: Element type of the output tensor, which may differ
            from `dtype` on the native FP8 path (inferred).
        q_tt_layout: Compile-time `TensorLayout` of the query tensor
            (inferred).
        output_tt_layout: Compile-time `TensorLayout` of the output tensor
            (inferred).
        valid_length_tt_layout: Compile-time `TensorLayout` of the
            valid-length tensor (inferred).
        config: Tile/pipeline configuration; defaults are derived from the
            query layout's last two dimensions.
        ragged: `True` for ragged-batch (variable-length) inputs (defaults
            to `False`).
        sink: `True` to enable attention-sink mode where the first tokens
            always attend (defaults to `False`).
        decoding_warp_split_k: `True` to enable warp-level split-K for
            decode (defaults to `False`).
        naive_kernel: `True` to force the fallback naive attention kernel
            (defaults to `False`).

    Args:
        output: Mutable destination `TileTensor` for the attention output.
        q: Query `TileTensor` with BSHD layout.
        k: Key operand backed by a KV cache.
        v: Value operand backed by a KV cache.
        mask_functor: Mask instance used to apply the attention mask.
        valid_length: Per-sequence valid lengths for masking padded batches.
        scale: Softmax temperature scale applied to Q·Kᵀ.
        ctx: GPU device context for kernel dispatch.
        q_max_seq_len: Maximum query sequence length in the batch; `None`
            infers it from the KV cache.
        kv_input_row_offsets: Row offsets for ragged KV inputs; `None` for
            self-attention.
        num_partitions: Override the number of split-K partitions; `None`
            selects automatically.
        sink_weights: Optional sink-token weight tensor for attention sinks.
        decode_dispatch_metadata: Pre-computed decode dispatch metadata;
            `None` recomputes it.
    """
    _flash_attention_kv_cache[
        config=config,
        ragged=ragged,
        sink=sink,
        decoding_warp_split_k=decoding_warp_split_k,
        naive_kernel=naive_kernel,
    ](
        output,
        q,
        k,
        v,
        mask_functor,
        immut_tile_tensor_1d(valid_length.ptr, valid_length.num_elements()),
        scale,
        ctx,
        q_max_seq_len,
        kv_input_row_offsets,
        num_partitions,
        sink_weights,
        decode_dispatch_metadata,
    )


@inline(.always)
def q_num_matrix_view_rows[
    dtype: DType, //
](q: TileTensor[mut=False, dtype, ...]) -> Int:
    """Return the number of matrix rows when viewing Q as a 2-D tensor for TMA.

    For decoding, Q is viewed as `rows x depth`; for prefill it is viewed as
    `rows x (depth * num_heads)`. The row count is the product of all
    leading dimensions except the last two (head and depth).

    Parameters:
        dtype: Element type of the query tensor.

    Args:
        q: The query `TileTensor`.

    Returns:
        The number of logical rows in the 2-D TMA view of Q.
    """
    var num_rows: Int = Int(q.dim[0]())

    comptime for i in range(1, q.rank - 2):
        num_rows *= Int(q.dim[i]())
    return num_rows


def _apple_naive_fa_decode_enabled() -> Bool:
    return getenv("MODULAR_ENABLE_APPLE_NAIVE_FA_DECODE", "1") != "0"


def _apple_fa_prefill_enabled() -> Bool:
    return getenv("MODULAR_ENABLE_APPLE_FA_PREFILL", "1") != "0"


@inline(.always)
def flash_attention_dispatch[
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    dtype: DType,
    q_layout: TensorLayout,
    //,
    kv_num_heads: Int,
    config: MHAConfig[dtype] = {
        q_layout.static_shape[q_layout.rank - 2],
        q_layout.static_shape[q_layout.rank - 1],
    },
    ragged: Bool = False,
    sink: Bool = False,
    _is_flash_attention_applicable: Bool = True,
    # Work arounds to unify KVCache and dense tensor inputs:
    # Differentiate two cases, KV cache's length is before adding the latest
    # tokens e.g. zero for CE, and KV NBuffer's length is the latest length
    # e.g. prompt length for CE.
    _is_cache_length_accurate: Bool = False,
    # valid_length is needed for KV cache inputs and is empty for homogeneous
    # dense tensor inputs to avoid overhead in benchmark.
    _use_valid_length: Bool = True,
    # we might also want to use valid length for padded dense inputs
    _padded_ndbuffer: Bool = False,
    decoding_warp_split_k: Bool = False,
](
    output: TileTensor[mut=True, address_space=.GENERIC, ...],
    q: TileTensor[mut=False, dtype, q_layout, address_space=.GENERIC, ...],
    k: k_t,
    v: v_t,
    mask_functor: mask_t,
    max_prompt_len: Int,
    max_cache_valid_length: Int,
    scale: Float32,
    is_token_generation: Bool,
    ctx: DeviceContext,
    valid_length: OptionalReg[ImmutTileTensor1D[.uint32]] = None,
    kv_input_row_offsets: OptionalReg[ImmutTileTensor1D[.uint32]] = None,
    num_partitions: Optional[Int] = None,
    sink_weights: OptionalReg[ImmutTileTensor1D[dtype]] = None,
    decode_dispatch_metadata: OptionalReg[MHADecodeDispatchMetadata] = None,
) raises:
    """Dispatch a flash-attention kernel for prefill or decode over dense or KV-cache operands.

    Central dispatch point that inspects `is_token_generation`, `dtype`,
    `depth`, and the target GPU to select among FA2, FA3, and naive
    implementations. Handles both prefill (context encoding) and
    incremental decode, routing ragged-batch and paged-KV-cache inputs
    through appropriate kernel paths.

    Parameters:
        k_t: KV-cache or dense operand type for the key tensor.
        v_t: KV-cache or dense operand type for the value tensor.
        mask_t: Attention mask type implementing `MHAMask`.
        dtype: Element type of Q (K/V type is inferred from `k_t`).
        q_layout: Compile-time layout of the query tensor.
        kv_num_heads: Number of key/value heads (for GQA).
        config: Tile/pipeline configuration; defaults from query shape.
        ragged: `True` for ragged-batch (variable-length) inputs.
        sink: `True` to enable attention-sink mode.
        _is_flash_attention_applicable: Internal flag to suppress FA path.
        _is_cache_length_accurate: `True` when KV cache length already
            includes the newest tokens.
        _use_valid_length: `True` to mask output with per-sequence lengths.
        _padded_ndbuffer: `True` when the NBuffer has padded dense inputs.
        decoding_warp_split_k: Enable warp-level split-K for decode.

    Args:
        output: Mutable output tensor.
        q: Query tensor.
        k: Key operand (KV cache or dense tensor).
        v: Value operand (KV cache or dense tensor).
        mask_functor: Mask instance.
        max_prompt_len: Maximum query sequence length in the batch.
        max_cache_valid_length: Maximum key/value sequence length.
        scale: Softmax temperature scale.
        is_token_generation: `True` for decode mode; `False` for prefill.
        ctx: GPU device context.
        valid_length: Per-sequence valid lengths for masked output.
        kv_input_row_offsets: Row offsets for ragged KV inputs.
        num_partitions: Override split-K partition count; `None` for auto. On
            the SM100 FA4 route (`depth <= 128` half-float/fp8) the count is
            honored exactly, and supplying one additionally forces a BM < 256
            tile because the 2Q config has no split-K mechanism. It is
            unsatisfiable, and asserts, for a mask that reports `UNKNOWN_MASK`
            (`MaterializedMask`, `AndMask`), which has split-K comptime-pruned.
        sink_weights: Optional sink-token weight tensor.
        decode_dispatch_metadata: Pre-computed decode dispatch metadata. Only
            `max_cache_valid_length` is consumed on the SM100 FA4 route; its
            `num_partitions` field is unused there.
    """

    comptime num_heads = config.num_heads
    comptime depth = config.depth
    comptime group = config.num_heads // kv_num_heads

    # K V smem is only separate for GPUs with shared memory greater or equal to A100's.
    comptime is_shared_kv = ctx.default_device_info.shared_memory_per_multiprocessor < A100.shared_memory_per_multiprocessor

    comptime assert depth == q.static_shape[q.rank - 1]
    comptime assert num_heads == q.static_shape[q.rank - 2]
    var batch_size: Int

    comptime if ragged:
        batch_size = Int(valid_length.value().dim[0]()) - 1
    # This branch holds for both KVCache and dense tensor inputs.
    # Q is BSHD, S is either homogeneous or padded to same length.
    else:
        batch_size = Int(q.dim[0]())

    comptime q_half_float = dtype in (DType.float16, DType.bfloat16)
    comptime q_half_float_or_fp32 = dtype == DType.float32 or q_half_float
    comptime q_fp8_depth512 = dtype.is_float8() and (
        depth == 256 or depth == 512
    )
    comptime q_fp8_2q = dtype.is_float8() and (depth == 64 or depth == 128)

    var q_device = q.to_device_buffer(ctx)
    var output_device = output.to_device_buffer(ctx)

    comptime if _is_flash_attention_applicable:
        comptime is_sm90 = ctx.default_device_info == H100
        comptime is_sm100 = _is_sm10x_gpu(ctx.default_device_info)
        # The half-float arm needs `(ragged or not _use_valid_length)` because
        # FA4 DROPS `valid_length` when `ragged` is False, so a padded caller
        # would attend over its pad rows and read uninitialized KV. fp8 is not
        # gated on it: the prefill fallthrough below re-admits fp8 with no such
        # conjunct onto the same dispatch, so the gate would only move fp8
        # non-ragged decode, and production fp8 callers are ragged.
        comptime fa4_route = is_sm100 and mask_t.mask_safe_out_of_bounds and (
            dtype.is_float8()
            or (q_half_float and (ragged or not _use_valid_length))
        )
        comptime if fa4_route:
            # Decode callers supply a graph-computed cache length; prefill does
            # not, and `kv_cache_ragged.mojo` passes `decode_dispatch_metadata`
            # unconditionally -- so the `is_token_generation` guard is what
            # keeps prefill on the raw argument.
            var fa4_max_cache_len = max_cache_valid_length
            if is_token_generation:
                if decode_dispatch_metadata:
                    fa4_max_cache_len = (
                        decode_dispatch_metadata.value().max_cache_valid_length
                    )
            # An explicit `num_partitions` is forwarded verbatim as FA4's exact
            # partition count (`0` == auto), not snapped to the `_bucket_ws`
            # ladder; honoring it forces a BM < 256 route, since the 2Q config
            # has no split-K mechanism. `dispatch_metadata.num_partitions` is
            # unused here -- FA4 sizes its own auto P.
            var fa4_num_partitions: Int = 0
            if num_partitions:
                fa4_num_partitions = num_partitions.value()
            # d256/d512 large-prefill carveout: a long prompt goes to the
            # pair-CTA `mha_sm100_depth512_dispatch`; short prefill and all
            # decode fall through to FA4 below. The `comptime if` is
            # load-bearing, not stylistic -- it stops the depth512 call from
            # ELABORATING at depth {64,72,80,96,128}, where its config is
            # degenerate (`mha_operand.mojo`'s `nonfull_sets` fails at d64),
            # even though the runtime branch never executes there.
            comptime if depth == 256 or depth == 512:
                if not is_token_generation and max_prompt_len * group > 32:
                    mha_sm100_depth512_dispatch[
                        config=config,
                        group=group,
                        ragged=ragged,
                        _is_cache_length_accurate=_is_cache_length_accurate,
                        sink=sink,
                    ](
                        output.to_device_buffer(ctx),
                        q.to_device_buffer(ctx).unsafe_ptr(),
                        k,
                        rebind[k_t](v),
                        q_num_matrix_view_rows(q),
                        mask_functor,
                        valid_length.value().to_device_buffer(ctx).unsafe_ptr(),
                        DynamicInt(max_prompt_len),
                        max_cache_valid_length,
                        scale,
                        kv_input_row_offsets,
                        batch_size,
                        NoPartition[get_accum_type[q.dtype]()](),
                        ctx,
                        sink_weights,
                    )
                    return
                # Only the deep arm runs the workspace split-K combine, so its
                # `P_MAX` ceiling is checked here. DIAGNOSTIC, not the
                # correctness bound -- the combine's own `debug_assert` protects
                # the reduction but compiles out with assertions off, and this
                # raise keeps a failure a negative control can see.
                if fa4_num_partitions > P_MAX:
                    raise Error(
                        "ws_shared_key deep route cannot serve num_partitions"
                        " above fa4_splitk_combine's P_MAX; raise it with"
                        " -D FA4_COMBINE_P_MAX=<N>"
                    )
            mha_sm100_2q_dispatch[
                config=config,
                group=group,
                ragged=ragged,
                sink=sink,
                _is_cache_length_accurate=_is_cache_length_accurate,
            ](
                output.to_device_buffer(ctx),
                q.to_device_buffer(ctx).unsafe_ptr(),
                k,
                rebind[k_t](v),
                q_num_matrix_view_rows(q),
                mask_functor,
                valid_length.value().to_device_buffer(ctx).unsafe_ptr(),
                DynamicInt(max_prompt_len),
                fa4_max_cache_len,
                scale,
                kv_input_row_offsets,
                batch_size,
                ctx,
                sink_weights,
                num_partitions_override=fa4_num_partitions,
            )
            # Early return rather than wrapping everything below in an `else:`.
            # Semantically identical -- this `comptime if` is the last statement
            # in the enclosing block -- but it keeps the legacy tree at its
            # original indentation instead of re-indenting ~815 lines.
            return
        if not is_token_generation:
            # TODO note that we have to handle mask tensor alignment here.
            # Choose matmul parameters based on dtype.
            comptime if (
                (is_sm90 or is_sm100)
                and (
                    (
                        q_half_float
                        and (ragged or not _use_valid_length)
                        and config.algorithm == FlashAttentionAlgorithm(3)
                    )
                    or (is_sm100 and (q_fp8_depth512 or q_fp8_2q))
                )
            ):
                var num_rows_q = q_num_matrix_view_rows(q)

                comptime if is_sm90:
                    mha_sm90_dispatch[
                        config=config,
                        group=group,
                        ragged=ragged,
                        sink=sink,
                        _is_cache_length_accurate=_is_cache_length_accurate,
                    ](
                        output.to_device_buffer(ctx),
                        q.to_device_buffer(ctx),
                        k,
                        rebind[k_t](v),
                        num_rows_q,
                        mask_functor,
                        valid_length.value().to_device_buffer(ctx),
                        DynamicInt(max_prompt_len),
                        max_cache_valid_length,
                        scale,
                        kv_input_row_offsets,
                        batch_size,
                        NoPartition[get_accum_type[q.dtype]()](),
                        ctx,
                        sink_weights,
                    )
                else:
                    comptime assert is_sm100

                    # Both arms below are the mask-not-safe residual:
                    # `fa4_route` excluded `MaterializedMask` by mask and
                    # returned every geometric-mask shape above. Padded
                    # non-ragged half-float reaches neither (both gates carry
                    # `(ragged or not _use_valid_length)`) and falls to the FA2
                    # `else:` branch.
                    comptime if depth == 512 or depth == 256:
                        mha_sm100_depth512_dispatch[
                            config=config,
                            group=group,
                            ragged=ragged,
                            _is_cache_length_accurate=_is_cache_length_accurate,
                            sink=sink,
                        ](
                            output.to_device_buffer(ctx),
                            q.to_device_buffer(ctx).unsafe_ptr(),
                            k,
                            rebind[k_t](v),
                            num_rows_q,
                            mask_functor,
                            valid_length.value()
                            .to_device_buffer(ctx)
                            .unsafe_ptr(),
                            DynamicInt(max_prompt_len),
                            max_cache_valid_length,
                            scale,
                            kv_input_row_offsets,
                            batch_size,
                            NoPartition[get_accum_type[q.dtype]()](),
                            ctx,
                            sink_weights,
                        )
                    else:
                        mha_sm100_2q_dispatch[
                            config=config,
                            group=group,
                            ragged=ragged,
                            sink=sink,
                            _is_cache_length_accurate=_is_cache_length_accurate,
                        ](
                            output.to_device_buffer(ctx),
                            q.to_device_buffer(ctx).unsafe_ptr(),
                            k,
                            rebind[k_t](v),
                            num_rows_q,
                            mask_functor,
                            valid_length.value()
                            .to_device_buffer(ctx)
                            .unsafe_ptr(),
                            DynamicInt(max_prompt_len),
                            max_cache_valid_length,
                            scale,
                            kv_input_row_offsets,
                            batch_size,
                            ctx,
                            sink_weights,
                        )

            else:
                # Long-context AMD CDNA prefill gate. Routes BF16
                # prefill to the 8-warp structured kernel in
                # `amd_structured/mha_prefill_v2.mojo`; otherwise falls
                # through to the FA2 launch below. Gate (all comptime
                # except the seq-length / page-size run-time checks):
                # - BF16 throughout;
                # - depth in (64, 128) (MFMA shape `32x32x16_bf16`);
                # - any `MHAMask` (`MhaPrefillV2` handles Causal natively + the
                #   generic `_maybe_apply_mask` path covers
                #   SlidingWindow / Chunked / Null / etc.);
                # - no attention sink;
                # - AMD CDNA (not RDNA, not Nvidia);
                # - K/V operand is either contiguous (`page_size == 0`)
                #   or paged with `page_size >= KV_BLOCK = 64`;
                # - max_prompt_len >= 4096 (perf gate).
                # The `scale_vec=1` reset in
                # `_pv_strip_with_partial_softmax`'s else-branch ensures
                # non-causal masks don't blow up `norm_vec` in the
                # epilogue (see comment there).
                comptime _v2_eligible = (
                    # TODO(KERN-3053): Disable this kernel to debug race
                    # conditions that lead to E2E model failures.
                    False
                    # config.dtype == DType.bfloat16
                    # and output.dtype == DType.bfloat16
                    # and (config.depth == 64 or config.depth == 128)
                    # and ctx.target.is_amd_gpu()
                    # and not has_amd_rdna_gpu_accelerator()
                    # and (k_t.page_size == 0 or k_t.page_size >= 64)
                )

                comptime if _v2_eligible:
                    # Long-context perf threshold. Below this the kernel
                    # doesn't fill the GPU at BM=256 and FA2 (BM=128) wins.
                    # Partial-Q-tile masking inside the kernel now handles
                    # `seq_len % 256 != 0` correctly for the Q-side
                    # writeback skip, so the alignment guard is gone;
                    # per-block early-return + writeback skip together
                    # handle mixed-length multi-sequence ragged.
                    #
                    # NullMask + partial-K (`num_keys % KV_BLOCK != 0`,
                    # e.g. FLUX.2-dev i2i at seq_len=8623 → 135 K tiles,
                    # last tile 47/64 valid) is now handled in-kernel: the
                    # SRD clamp hardware-zeros the partial tile's OOB
                    # columns, `_apply_kbound_mask_fast` excludes them from
                    # softmax, and the even-tile-count round-up fixes the
                    # odd-`N` main-loop/epilogue double-count that was the
                    # real i2i corruption. FLUX i2i is SSIM 0.994 through
                    # the kernel (was 0.50), so the prior carve-out to FA2
                    # is gone.
                    if max_prompt_len >= 4096:
                        comptime v2_config = MhaConfigV2(
                            q_block_size=32,
                            kv_block=64,
                            depth=config.depth,
                            num_heads=config.num_heads,
                            num_kv_heads=kv_num_heads,
                            num_warps=8,
                            output_dtype=DType.bfloat16,
                        )
                        comptime if ragged:
                            # Ragged batch: per-sequence setup happens
                            # inside the dedicated ragged `MhaPrefillV2`
                            # kernel so it keeps its tuned single-kernel
                            # register-allocation context. Avoids the
                            # ~14% perf hit observed when the kernel is
                            # inlined into the FA2 host `def mha[]`.
                            #
                            # Handles any `batch_size`: the `ragged:
                            # Bool` flag inside `MhaPrefillV2.run` forces
                            # the Q/O batch coord to 0 so each block
                            # reads from the per-sequence pre-offset
                            # pointer regardless of `block_idx.z` (the
                            # singleton batch_dim in the ragged BSHD
                            # view would OOB-read otherwise). The
                            # partial-Q-tile writeback skip in
                            # `_store_o_to_gmem` covers non-BM-aligned
                            # sequence lengths.
                            #
                            # Cross-attention path (`cross_attention`
                            # comptime flag on the launcher) fires when
                            # the caller passed `kv_input_row_offsets`
                            # — encoder/decoder workloads with K/V
                            # length independent of Q. Self-attention
                            # path is bit-identical to the pre-Phase-10
                            # codegen at the comptime monomorphization
                            # level.
                            var q_off_ptr = (
                                valid_length.value().as_unsafe_any_origin().ptr
                            )
                            # Sink-weights pointer: when `sink=True` the
                            # caller MUST pass non-None `sink_weights`
                            # (mirrors the existing FA2 contract). The
                            # gate-time `comptime if sink:` selects the
                            # launcher instantiation; the runtime
                            # `if kv_input_row_offsets:` selects the
                            # cross-attention vs self-attention variant.
                            comptime if sink:
                                var sw_ptr = (
                                    sink_weights.value()
                                    .as_unsafe_any_origin()
                                    .ptr
                                )
                                if kv_input_row_offsets:
                                    mha_prefill_v2_ragged[
                                        config=v2_config,
                                        cross_attention=True,
                                        sink=True,
                                    ](
                                        q.as_unsafe_any_origin().unsafe_ptr(),
                                        k,
                                        v,
                                        output.as_unsafe_any_origin().unsafe_ptr(),
                                        mask_functor,
                                        scale,
                                        q_off_ptr,
                                        kv_input_row_offsets.value()
                                        .as_unsafe_any_origin()
                                        .ptr,
                                        max_prompt_len,
                                        batch_size,
                                        ctx,
                                        sw_ptr,
                                    )
                                else:
                                    mha_prefill_v2_ragged[
                                        config=v2_config, sink=True
                                    ](
                                        q.as_unsafe_any_origin().unsafe_ptr(),
                                        k,
                                        v,
                                        output.as_unsafe_any_origin().unsafe_ptr(),
                                        mask_functor,
                                        scale,
                                        q_off_ptr,
                                        q_off_ptr,
                                        max_prompt_len,
                                        batch_size,
                                        ctx,
                                        sw_ptr,
                                    )
                            else:
                                if kv_input_row_offsets:
                                    mha_prefill_v2_ragged[
                                        config=v2_config, cross_attention=True
                                    ](
                                        q.as_unsafe_any_origin().unsafe_ptr(),
                                        k,
                                        v,
                                        output.as_unsafe_any_origin().unsafe_ptr(),
                                        mask_functor,
                                        scale,
                                        q_off_ptr,
                                        kv_input_row_offsets.value()
                                        .as_unsafe_any_origin()
                                        .ptr,
                                        max_prompt_len,
                                        batch_size,
                                        ctx,
                                    )
                                else:
                                    mha_prefill_v2_ragged[config=v2_config](
                                        q.as_unsafe_any_origin().unsafe_ptr(),
                                        k,
                                        v,
                                        output.as_unsafe_any_origin().unsafe_ptr(),
                                        mask_functor,
                                        scale,
                                        q_off_ptr,
                                        q_off_ptr,
                                        max_prompt_len,
                                        batch_size,
                                        ctx,
                                    )
                            return
                        else:
                            # `mask_functor` is the dispatcher's
                            # `mask_t` instance (the gate filtered to
                            # `CausalMask` for now; future phases will
                            # widen to sliding/chunked causal).
                            # `start_pos = max_cache_valid_length -
                            # max_prompt_len` is the number of pre-existing
                            # KV entries before this prefill batch's tokens
                            # — zero for fresh prefill, positive for cache
                            # reuse.
                            comptime if sink:
                                mha_prefill_v2[v2_config, sink=True](
                                    q,
                                    k,
                                    v,
                                    output,
                                    mask_functor,
                                    scale,
                                    max_cache_valid_length,
                                    max_cache_valid_length - max_prompt_len,
                                    ctx,
                                    sink_weights.value()
                                    .as_unsafe_any_origin()
                                    .ptr,
                                )
                            else:
                                mha_prefill_v2[v2_config](
                                    q,
                                    k,
                                    v,
                                    output,
                                    mask_functor,
                                    scale,
                                    max_cache_valid_length,
                                    max_cache_valid_length - max_prompt_len,
                                    ctx,
                                )
                            return

                comptime BM = config.block_m()
                comptime smem_use = config.shared_mem_bytes[is_shared_kv]()
                comptime kernel = mha[
                    config.dtype,
                    k_t,
                    v_t,
                    output.dtype,
                    mask_t,
                    config,
                    group=group,
                    ragged=ragged,
                    is_shared_kv=is_shared_kv,
                    sink=sink,
                    _use_valid_length=_use_valid_length,
                    _is_cache_length_accurate=_is_cache_length_accurate,
                    _padded_ndbuffer=_padded_ndbuffer,
                ]

                var grid_dim = LaunchDim(
                    ceildiv(max_prompt_len, BM),
                    config.num_heads,
                    batch_size,
                ) if ctx.target.is_nvidia_gpu() else LaunchDim(
                    config.num_heads,
                    ceildiv(max_prompt_len, BM),
                    batch_size,
                )

                ctx.enqueue_function[kernel](
                    q_device,
                    k,
                    v,
                    output_device,
                    scale,
                    Int32(batch_size),
                    Int32(max_prompt_len),
                    Int32(max_cache_valid_length),
                    valid_length.value(),
                    kv_input_row_offsets,
                    sink_weights,
                    mask_functor,
                    grid_dim=grid_dim,
                    block_dim=(config.num_threads(), 1, 1),
                    shared_mem_bytes=smem_use,
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        UInt32(smem_use)
                    ),
                )
        # FA3 decoding impl only support half precision, while fp32 is supported
        # for fp32 as well.
        elif (
            q_half_float_or_fp32
            or (dtype.is_float8() and ctx.target.is_amd_gpu())
            or (dtype.is_float8() and is_sm100)
        ) and is_token_generation:
            comptime if depth <= 576 and (
                not dtype.is_float8()
                or ctx.target.is_amd_gpu()
                or (dtype.is_float8() and is_sm100 and depth <= 512)
            ):
                # AMD bf16: 4 warps (256 threads) with 16x16 MMA.
                # BN=128 WN=32: each warp owns a full [16,32] P block.
                # AMD fp8: 16x16x128 MMA when depth%128==0, else 32x32x64.
                comptime _fp8_small_mma = (
                    dtype.is_float8() and depth % 128 == 0
                )
                comptime BM = (
                    (16 if _fp8_small_mma else 32) if (
                        dtype.is_float8() and ctx.target.is_amd_gpu()
                    ) else 16
                )
                comptime BN = 128 if ctx.target.is_amd_gpu() else (
                    depth if ctx.target.is_nvidia_gpu() else 128
                )
                comptime BK = (
                    (128 if _fp8_small_mma else 64) if dtype.is_float8() else 32
                ) if ctx.target.is_amd_gpu() else (
                    16 if q.dtype == .float32 else 32
                )
                comptime WM = BM
                comptime WN = 32
                # num warps in M and N, multiplied by warp size.
                comptime num_threads = (BM // WM) * (BN // WN) * WARP_SIZE

                # Key tile for the single-KV-head fold arm below, the only one
                # that sets WN == BN (one warp owns the whole key tile, so
                # softmax needs no cross-warp reduction); the `group == 1` arm
                # keeps the single-token BN/WN instead. So this knob sets that
                # arm's per-warp K/V register pressure and LDS footprint. 64
                # measured 0.9-8.2% slower: the kernel is HBM-bound, so a
                # narrower tile only buys more, shorter bursts.
                comptime fold_BN = 128

                comptime accum_type = get_accum_type[q.dtype]()
                comptime num_pipeline_stages = 4
                # smem for q
                var shared_mem_bytes = BM * depth * size_of[q.dtype]()

                # separate KV smem if we have enough smem
                comptime if not is_shared_kv:
                    shared_mem_bytes += 2 * BN * depth * size_of[k_t.dtype]()
                else:
                    shared_mem_bytes += (
                        num_pipeline_stages * BN * BK * size_of[k_t.dtype]()
                    )

                comptime num_warps = ceildiv(num_threads, WARP_SIZE)

                # smem for p and warp_scratch
                shared_mem_bytes += (
                    BM * BN * size_of[k_t.dtype]()
                    + 2 * num_warps * BM * size_of[accum_type]()
                )
                comptime num_blocks_y = num_heads // group

                # Total query rows in the batch. The S>1 fold packs several tokens
                # per sequence into one CTA, so the split-K buffers and the reduce
                # grid are keyed on rows, not sequences. Read from Q's own shape
                # rather than `batch_size * max_prompt_len` so it is right for
                # both layouts: ragged rank-3 gives total_q, BSHD rank-4 gives
                # B*S, and either way == batch_size for single-token decode.
                var num_q_rows = q_num_matrix_view_rows(q)

                var dispatch_metadata: MHADecodeDispatchMetadata
                if decode_dispatch_metadata:
                    dispatch_metadata = decode_dispatch_metadata.value()
                else:
                    dispatch_metadata = (
                        MHADecodeDispatchMetadata.from_runtime_values[
                            num_heads,
                            group,
                        ](
                            batch_size,
                            max_prompt_len,
                            max_cache_valid_length,
                            ctx,
                        )
                    )
                var max_cache_valid_length_value = (
                    dispatch_metadata.max_cache_valid_length
                )
                var partition_num_keys = max_cache_valid_length_value
                # `BM` here is the single-token geometry even when the S>1 fold
                # launches with BM = num_heads*S. That only shifts the partition
                # COUNT below, not any address or bound: the kernel derives its
                # tiling from the `BM_S`/`BN_S` it was instantiated with, and
                # `get_start_and_end_for_partitions` divides `num_keys` by
                # whatever count it is handed. (`check_mask_during_decoding`
                # masks are excluded from the fold anyway.)
                comptime if mask_t.check_mask_during_decoding:
                    if partition_num_keys > 0:
                        partition_num_keys -= Int(
                            mask_functor.start_column[BM, BN, k_t.page_size](
                                # Pre-launch dispatch is batch-aggregate; no
                                # per-sequence id is available. Masks whose
                                # start_column depends on seq_id should not
                                # be used through this decode path.
                                UInt32(0),
                                UInt32(partition_num_keys - 1),
                            )
                        )
                        if partition_num_keys <= 0:
                            partition_num_keys = 1

                var num_partitions_value: Int
                # Upper bound on num_partitions_value, independent of num_keys.
                # The SM100 1Q decode grid launches this many partition CTAs so
                # the grid shape is stable across num_keys (one CUDA graph per
                # batch size); CTAs beyond num_partitions_value early-return.
                # For explicit/override partition counts we do not over-launch,
                # so max == actual.
                var max_num_partitions_value: Int
                if num_partitions:
                    num_partitions_value = num_partitions.value()
                    max_num_partitions_value = num_partitions_value
                elif (
                    dispatch_metadata.num_partitions > 0
                    and partition_num_keys == max_cache_valid_length_value
                ):
                    num_partitions_value = dispatch_metadata.num_partitions
                    max_num_partitions_value = num_partitions_value
                else:
                    num_partitions_value = get_mha_decoding_num_partitions[
                        num_heads, group
                    ](batch_size, partition_num_keys, ctx)
                    max_num_partitions_value = (
                        get_mha_decoding_max_num_partitions[num_heads, group](
                            batch_size, partition_num_keys, ctx
                        )
                    )

                # The launched (max) count must bound the actual count, else the
                # over-launched SM100 1Q grid would under-launch and silently
                # drop partitions. (Also keeps max_num_partitions_value used on
                # targets where the SM100 1Q construction is comptime-elided.)
                debug_assert(
                    max_num_partitions_value >= num_partitions_value,
                    "max_num_partitions must be >= num_partitions",
                )

                # `fa4_route` above claimed every FA4-eligible SM100 shape and
                # returned, so this gates only the residual it excluded
                # (`MaterializedMask`, Float32, padded non-ragged half-float)
                # plus the sm90 half-float decode. `MaterializedMask` keeps the
                # FA2 `mha_decoding` split-K path below.
                comptime use_fa3_kernel = (
                    (is_sm90 or is_sm100)
                    and mask_t.mask_safe_out_of_bounds
                    and (
                        (
                            q_half_float
                            and (ragged or not _use_valid_length)
                            and config.algorithm == FlashAttentionAlgorithm(3)
                        )
                        or (is_sm100 and dtype.is_float8() and depth <= 512)
                    )
                )

                comptime if (not use_fa3_kernel) and (depth % 64) != 0:
                    # FA2 kernel only supports depth % 64 == 0
                    # Assumes BSHD.
                    mha_gpu_naive[
                        ragged=ragged,
                        sink=sink,
                        _use_valid_length=_use_valid_length,
                        _is_cache_length_accurate=_is_cache_length_accurate,
                    ](
                        q,
                        k,
                        v,
                        mask_functor,
                        output,
                        valid_length.value(),
                        scale,
                        batch_size,
                        max_prompt_len,
                        max_cache_valid_length_value,
                        num_heads,
                        depth,
                        group,
                        ctx,
                        sink_weights,
                    )
                else:
                    # One `mha_decoding` per folded query length S; S == 1 is the
                    # legacy single-token geometry, verbatim. Both the direct and
                    # the split-K launch go through this one ladder, so their
                    # instantiation sets cannot drift. Returns whether a launch
                    # happened.
                    def launch_mha_decoding[
                        out_dtype: DType
                    ](
                        out_buf: DeviceBuffer[out_dtype],
                        exp_sum_buf: DeviceBuffer[accum_type],
                        qk_max_buf: DeviceBuffer[accum_type],
                        grid_x: Int,
                    ) raises {imm} -> Bool:
                        comptime for S in range(1, _MHA_DECODE_FOLD_MAX_S + 1):
                            comptime if S == 1 or _mha_decode_fold_ok[
                                dtype,
                                depth,
                                num_heads,
                                group,
                                S,
                                sink=sink,
                                use_valid_length=_use_valid_length,
                                ragged=ragged,
                                check_mask_during_decoding=mask_t.check_mask_during_decoding,
                            ]():
                                if max_prompt_len == S:
                                    # `group == 1` folds only the S token rows,
                                    # which fit one M-tile, so it keeps the
                                    # S == 1 geometry; a single KV head stacks
                                    # `num_heads*S` rows over warp M-tiles.
                                    comptime _fold_narrow = S == 1 or group == 1
                                    # Token SLOTS the tile is built from: only
                                    # the geometry below follows this, never a
                                    # stride. The kernel is handed the REAL
                                    # width and re-derives this from the SAME
                                    # expression, so keep the two textually
                                    # identical rather than guarding either
                                    # side, or the agreement is a coincidence.
                                    comptime TILE_S = mha_decode_fold_tile_q_seq_len[
                                        dtype, num_heads, group, S
                                    ]()
                                    # Rows the stacked arm's tile holds, dead
                                    # ones included.
                                    comptime fold_rows = num_heads * TILE_S
                                    comptime _fold_wide = (
                                        not _fold_narrow
                                        and mha_decode_fold_wide_mma[
                                            dtype, num_heads, group, TILE_S
                                        ]()
                                    )
                                    comptime BM_S = (
                                        BM if _fold_narrow else fold_rows
                                    )
                                    comptime BN_S = (
                                        BN if _fold_narrow else fold_BN
                                    )
                                    comptime WM_S = (
                                        WM if _fold_narrow else _mha_decode_fold_warp_m[
                                            dtype, num_heads, group, TILE_S
                                        ]()
                                    )
                                    comptime WN_S = WN if _fold_narrow else BN_S
                                    # One MMA strip per SMEM block, so BK is READ
                                    # OFF the shape the kernel will pick rather
                                    # than restated beside it — a restated
                                    # literal mis-counts the QK/PV strip loops.
                                    comptime BK_S = decode_mma_shape[
                                        dtype,
                                        depth,
                                        num_heads,
                                        fold_wide_mma=True,
                                    ]()[2] if _fold_wide else BK
                                    comptime num_threads_S = (
                                        (BM_S // WM_S)
                                        * (BN_S // WN_S)
                                        * WARP_SIZE
                                    )
                                    comptime kernel = mha_decoding[
                                        q.dtype,
                                        k_t,
                                        v_t,
                                        out_dtype,
                                        mask_t,
                                        BM=BM_S,
                                        BN=BN_S,
                                        BK=BK_S,
                                        WM=WM_S,
                                        WN=WN_S,
                                        depth=depth,
                                        num_heads=num_heads,
                                        num_threads=num_threads_S,
                                        num_pipeline_stages=num_pipeline_stages,
                                        group=group,
                                        ragged=ragged,
                                        is_shared_kv=is_shared_kv,
                                        sink=sink,
                                        _use_valid_length=_use_valid_length,
                                        _is_cache_length_accurate=_is_cache_length_accurate,
                                        decoding_warp_split_k=decoding_warp_split_k,
                                        q_seq_len=S,
                                    ]
                                    ctx.enqueue_function[kernel](
                                        q_device,
                                        k,
                                        v,
                                        out_buf,
                                        exp_sum_buf,
                                        qk_max_buf,
                                        scale,
                                        Int32(batch_size),
                                        Int32(num_partitions_value),
                                        valid_length.value(),
                                        sink_weights,
                                        mask_functor,
                                        grid_dim=(
                                            grid_x,
                                            num_blocks_y,
                                            batch_size,
                                        ),
                                        block_dim=(num_threads_S, 1, 1),
                                        shared_mem_bytes=shared_mem_bytes if ctx.target.is_nvidia_gpu() else 0,
                                        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                                            UInt32(
                                                (
                                                    ctx.default_device_info.shared_memory_per_multiprocessor
                                                    - 4096
                                                ) if ctx.target.is_nvidia_gpu() else 0
                                            )
                                        ),
                                    )
                                    return True
                        return False

                    if num_partitions_value == 1:
                        comptime if use_fa3_kernel:
                            var num_rows_q = q_num_matrix_view_rows(q)

                            comptime if is_sm90:
                                mha_sm90_dispatch[
                                    config=config,
                                    group=group,
                                    ragged=ragged,
                                    sink=sink,
                                    _is_cache_length_accurate=_is_cache_length_accurate,
                                ](
                                    output.to_device_buffer(ctx),
                                    q.to_device_buffer(ctx),
                                    k,
                                    rebind[k_t](v),
                                    num_rows_q,
                                    mask_functor,
                                    valid_length.value().to_device_buffer(ctx),
                                    StaticInt[1](),
                                    max_cache_valid_length_value,
                                    scale,
                                    kv_input_row_offsets,
                                    batch_size,
                                    NoPartition[accum_type](),
                                    ctx,
                                    sink_weights,
                                )
                        else:
                            var nullptr_device = DeviceBuffer[accum_type].empty(
                                ctx
                            )
                            # Unreachable (the gate and the ladder share one
                            # predicate), but falling through would launch nothing
                            # and leave the output buffer untouched.
                            if not launch_mha_decoding[output.dtype](
                                output_device,
                                nullptr_device,
                                nullptr_device,
                                1,
                            ):
                                raise Error(
                                    t"no mha_decoding instantiation for query"
                                    t" length {max_prompt_len}"
                                )
                        return

                    else:
                        # This `num_partitions_value > 1` split-K branch still
                        # elaborates, but `fa4_route` now claims decode at
                        # d256/d512 far above (the `ws_shared_key` deep route),
                        # so for the residual shapes that reach here the runtime
                        # `FA4_COMBINE_P_MAX` ceiling lives in the `fa4_route`
                        # block, scoped to that deep arm. This split launches the
                        # 1Q / `mha_decoding` fall-back and reduces.
                        # We split partitions and then reduce
                        # allocate memory for intermediate results
                        # q # [B, S, H, D]

                        # Determine intermediate buffer type based on platform
                        # AMD uses float32 for higher precision with aggressive split-k
                        comptime intermediate_dtype = output.dtype

                        var output_intermediate_data = (
                            ctx.enqueue_create_buffer[intermediate_dtype](
                                num_heads
                                * depth
                                * num_q_rows
                                * num_partitions_value
                            )
                        )

                        var output_intermediate = TileTensor(
                            output_intermediate_data,
                            row_major(
                                num_partitions_value,
                                num_q_rows,
                                Idx[num_heads],
                                Idx[depth],
                            ),
                        )

                        var data_len = (
                            num_heads * num_q_rows * num_partitions_value
                        )
                        var stats_layout = row_major(
                            num_partitions_value, num_q_rows, Idx[num_heads]
                        )
                        var exp_sum_qk_max_data = ctx.enqueue_create_buffer[
                            accum_type
                        ](2 * data_len)

                        var exp_sum = TileTensor(
                            exp_sum_qk_max_data.unsafe_ptr(), stats_layout
                        )
                        var qk_max = TileTensor(
                            exp_sum_qk_max_data.unsafe_ptr() + data_len,
                            stats_layout,
                        )

                        var exp_sum_device = exp_sum.to_device_buffer(ctx)
                        var qk_max_device = qk_max.to_device_buffer(ctx)

                        comptime if use_fa3_kernel:
                            var num_rows_q = q_num_matrix_view_rows(q)

                            comptime if is_sm90:
                                mha_sm90_dispatch[
                                    config=config,
                                    group=group,
                                    ragged=ragged,
                                    sink=sink,
                                    _is_cache_length_accurate=_is_cache_length_accurate,
                                ](
                                    output_intermediate.to_device_buffer(ctx),
                                    q.to_device_buffer(ctx),
                                    k,
                                    rebind[k_t](v),
                                    num_rows_q,
                                    mask_functor,
                                    valid_length.value().to_device_buffer(ctx),
                                    StaticInt[1](),
                                    max_cache_valid_length_value,
                                    scale,
                                    kv_input_row_offsets,
                                    batch_size,
                                    SplitKPartition(
                                        exp_sum_qk_max_data.unsafe_ptr().as_unsafe_any_origin(),
                                        UInt32(num_partitions_value),
                                        # sm90 does not over-launch: max == actual.
                                        UInt32(num_partitions_value),
                                    ),
                                    ctx,
                                    sink_weights,
                                )
                        else:
                            # Same ladder, on the intermediate dtype and writing
                            # per-partition planes. Without a launch the reduce
                            # below would read an uninitialized workspace.
                            if not launch_mha_decoding[intermediate_dtype](
                                output_intermediate_data,
                                exp_sum_device,
                                qk_max_device,
                                num_partitions_value,
                            ):
                                raise Error(
                                    t"no mha_decoding split-k instantiation for"
                                    t" query length {max_prompt_len}"
                                )

                        # AMD decoding kernels always use exp2 for softmax,
                        # while NVIDIA uses exp2 only with FA3 kernels.
                        comptime reduce_use_exp2 = use_fa3_kernel or ctx.target.is_amd_gpu()
                        comptime kernel_reduce = mha_splitk_reduce[
                            intermediate_dtype,
                            output.dtype,
                            depth=depth,
                            num_heads=num_heads,
                            num_threads=WARP_SIZE,
                            use_exp2=reduce_use_exp2,
                        ]

                        # One warp per (query row, head). The reducer is keyed on
                        # rows throughout — its `batch_size` argument strides the
                        # per-partition planes — so BSHD `[B, S, H, D]` reduces as
                        # `[B*S, H, D]` with no reducer change.
                        ctx.enqueue_function[kernel_reduce](
                            output_intermediate_data,
                            output_device,
                            exp_sum_device,
                            qk_max_device,
                            Int32(num_q_rows),
                            Int32(num_partitions_value),
                            grid_dim=(
                                1,
                                num_heads,
                                num_q_rows,
                            ),
                            block_dim=(WARP_SIZE, 1, 1),
                            attributes=pdl_launch_attributes(MHA_PDL_LEVEL),
                        )
                        _ = exp_sum_qk_max_data^
                        _ = output_intermediate_data^
            else:
                # Not supported by contexting and decoding, e.g cross-attention or depth != 128
                # Assumes BSHD.
                mha_gpu_naive[
                    ragged=ragged,
                    sink=sink,
                    _use_valid_length=_use_valid_length,
                    _is_cache_length_accurate=_is_cache_length_accurate,
                ](
                    q,
                    k,
                    v,
                    mask_functor,
                    output,
                    valid_length,
                    scale,
                    batch_size,
                    max_prompt_len,
                    max_cache_valid_length,
                    num_heads,
                    depth,
                    group,
                    ctx,
                    sink_weights,
                )
        else:
            # Assumes BSHD.
            mha_gpu_naive[
                ragged=ragged,
                sink=sink,
                _use_valid_length=_use_valid_length,
                _is_cache_length_accurate=_is_cache_length_accurate,
            ](
                q,
                k,
                v,
                mask_functor,
                output,
                valid_length,
                scale,
                batch_size,
                max_prompt_len,
                max_cache_valid_length,
                num_heads,
                depth,
                group,
                ctx,
                sink_weights,
            )

    # Not supported by fast flash attention kernel.
    else:
        # Assumes BSHD.
        comptime if ctx.target.is_apple_gpu():
            # Apple attention. Decode (1 query row) -> `naive_fa_decode_apple`
            # (head dim split across lanes; that kernel owns which head dims it
            # can specialize). Prefill ->
            # MMA-based `fa_prefill_apple` when depth % 16 == 0 and KV is
            # contiguous or 16-aligned-paged; otherwise `mha_gpu_naive`. The KV
            # gate is COMPTIME because the prefill resolves a page per 16-row
            # sub-tile and comptime-asserts page_size % 16 == 0 (an odd page
            # could bisect a sub-tile) -- KB apple-paged-kv-prefill-per-sub-tile.
            if (
                is_token_generation
                and _apple_naive_fa_decode_enabled()
                and naive_fa_decode_apple_supports_depth(depth)
            ):
                naive_fa_decode_apple[
                    ragged=ragged,
                    sink=sink,
                    _use_valid_length=_use_valid_length,
                    _is_cache_length_accurate=_is_cache_length_accurate,
                ](
                    q,
                    k,
                    v,
                    mask_functor,
                    output,
                    valid_length.value(),
                    scale,
                    batch_size,
                    max_prompt_len,
                    max_cache_valid_length,
                    num_heads,
                    depth,
                    group,
                    ctx,
                    sink_weights,
                )
            else:
                comptime apple_prefill_kv_ok = (
                    k_t.page_size == 0 or k_t.page_size % 16 == 0
                )
                comptime apple_prefill_depth_ok = (
                    depth <= FA_PREFILL_APPLE_MAX_HEAD_DIM and depth % 16 == 0
                )
                comptime if apple_prefill_kv_ok and apple_prefill_depth_ok:
                    # Wide-threadgroup no-SMEM prefill (num_simdgroups=16): 16
                    # simdgroups / 256 query rows share a threadgroup and read
                    # K/V from DRAM (no staging, no barriers). It beat both the
                    # block_dim=32 base and the SMEM-staged variant at every
                    # shape measured (KB kernels/apple-m5-fa-prefill).
                    # The 16x16 simdgroup MMA needs M5+ and truncates fp32
                    # operands to fp19.
                    comptime f32_in = dtype == DType.float32
                    if (
                        not is_token_generation
                        and _apple_fa_prefill_enabled()
                        and ctx.compute_capability() >= 5
                        and (
                            not f32_in or _apple_m5_allow_lossy_f32_attention()
                        )
                    ):
                        fa_prefill_apple[
                            ragged=ragged,
                            sink=sink,
                            _use_valid_length=_use_valid_length,
                            _is_cache_length_accurate=_is_cache_length_accurate,
                        ](
                            q,
                            k,
                            v,
                            mask_functor,
                            output,
                            valid_length.value(),
                            scale,
                            batch_size,
                            max_prompt_len,
                            max_cache_valid_length,
                            num_heads,
                            depth,
                            group,
                            ctx,
                            sink_weights,
                        )
                    else:
                        mha_gpu_naive[
                            ragged=ragged,
                            _use_valid_length=_use_valid_length,
                            _is_cache_length_accurate=_is_cache_length_accurate,
                            sink=sink,
                        ](
                            q,
                            k,
                            v,
                            mask_functor,
                            output,
                            valid_length,
                            scale,
                            batch_size,
                            max_prompt_len,
                            max_cache_valid_length,
                            num_heads,
                            depth,
                            group,
                            ctx,
                            sink_weights,
                        )
                else:
                    mha_gpu_naive[
                        ragged=ragged,
                        _use_valid_length=_use_valid_length,
                        _is_cache_length_accurate=_is_cache_length_accurate,
                        sink=sink,
                    ](
                        q,
                        k,
                        v,
                        mask_functor,
                        output,
                        valid_length,
                        scale,
                        batch_size,
                        max_prompt_len,
                        max_cache_valid_length,
                        num_heads,
                        depth,
                        group,
                        ctx,
                        sink_weights,
                    )
        else:
            mha_gpu_naive[
                ragged=ragged,
                _use_valid_length=_use_valid_length,
                _is_cache_length_accurate=_is_cache_length_accurate,
                sink=sink,
            ](
                q,
                k,
                v,
                mask_functor,
                output,
                valid_length,
                scale,
                batch_size,
                max_prompt_len,
                max_cache_valid_length,
                num_heads,
                depth,
                group,
                ctx,
                sink_weights,
            )


def _flash_attention_dense[
    mask_t: MHAMask,
    dtype: DType,
    q_layout: TensorLayout,
    //,
    config: MHAConfig[dtype] = {
        q_layout.static_shape[2],
        q_layout.static_shape[3],
    },
    decoding_warp_split_k: Bool = False,
    _use_valid_length: Bool = False,
    _padded_ndbuffer: Bool = False,
    naive_kernel: Bool = False,
    sink: Bool = False,
](
    output: TileTensor[mut=True, address_space=.GENERIC, ...],
    q: TileTensor[mut=False, dtype, q_layout, address_space=.GENERIC, ...],
    k: TileTensor[mut=False, address_space=.GENERIC, ...],
    v: TileTensor[mut=False, address_space=.GENERIC, ...],
    mask_functor: mask_t,
    scale: Float32,
    ctx: DeviceContext,
    # if not set, we select num_partitions based on heuristics
    num_partitions: Optional[Int] = None,
    valid_length: OptionalReg[ImmutTileTensor1D[.uint32]] = None,
    sink_weights: OptionalReg[ImmutTileTensor1D[dtype]] = None,
) raises:
    """Shared implementation behind the dense `flash_attention` overloads.

    Wraps K and V in `TileTensorMHAOperand` adapters and delegates to
    `flash_attention_dispatch`. All tensor operands are `TileTensor`s.
    """
    # See the kV cache overloads for comments.

    comptime assert q.rank == 4, "only support rank 4 inputs."

    # Runtime dimensions.
    var batch_size = Int(q.dim[0]())
    var seq_len = Int(q.dim[1]())
    var num_keys = Int(k.dim[1]())

    # Zero-sized attention (e.g. VAE mid-block attention on a
    # ``(B, C, 0, 0)`` placeholder image flattens to ``seq_len=0``):
    # nothing to compute. The output buffer is pre-allocated zero
    # element by the caller; softmax over an empty sequence has no
    # defined value and the downstream readers also have zero seq.
    # Skipping the dispatch avoids zero-grid kernel launches and
    # undefined behavior in TMA descriptors with empty extents.
    if batch_size == 0 or seq_len == 0 or num_keys == 0:
        return

    # Whether head and depth are static. With BSHD, B and S are dynamic.
    # H and D are always known.
    # fmt: off
    comptime head_depth_known = q.static_shape[2] != -1 and q.static_shape[3] != -1 and k.static_shape[2] != -1
    comptime depth = q.static_shape[q.rank-1]
    comptime gpu_info = ctx.default_device_info
    comptime head_depth_supported = depth_supported_by_gpu[depth, mask_t, config, gpu_info]()
    comptime flash_attention_applicable = flash_attention_hw_supported[dtype]() and head_depth_known and head_depth_supported and not naive_kernel

    comptime q_half_float = q.dtype in (DType.float16, DType.bfloat16)
    comptime kv_num_heads = k.static_shape[2]
    # fmt: on

    # AMD decode also handles S > 1 (speculative verify) by folding the S query
    # tokens into the MMA M dimension; every other shape still needs seq_len == 1.
    # The `comptime for` mirrors the dispatch ladder's instantiation set, and both
    # read the shape from `config` (which `flash_attention_dispatch`
    # comptime-asserts against Q's layout), so a routed shape always has a kernel.
    var fold_seq_len_ok = False
    comptime for S in range(2, _MHA_DECODE_FOLD_MAX_S + 1):
        comptime if _mha_decode_fold_ok[
            dtype,
            config.depth,
            config.num_heads,
            config.num_heads // kv_num_heads,
            S,
            sink=sink,
            use_valid_length=_use_valid_length,
            ragged=False,
            check_mask_during_decoding=mask_t.check_mask_during_decoding,
        ]():
            if seq_len == S:
                fold_seq_len_ok = True

    var is_token_generation = (
        seq_len == 1 or fold_seq_len_ok
    ) and num_keys > seq_len

    # Build the row-major K/V TileTensors directly (no throwaway LayoutTensor
    # round-trip). BSHD layout: batch/seq are runtime, head/depth static, so
    # mirror `k`'s static pattern with `Idx` for the known dims. The operand
    # infers `buffer_layout` from the passed TileTensor.
    var k_operand = TileTensorMHAOperand(
        TileTensor(
            k.unsafe_ptr(),
            row_major(
                Int(k.dim[0]()),
                Int(k.dim[1]()),
                Idx[kv_num_heads],
                Idx[depth],
            ),
        )
    )
    var v_operand = TileTensorMHAOperand(
        TileTensor(
            v.unsafe_ptr(),
            row_major(
                Int(v.dim[0]()),
                Int(v.dim[1]()),
                Idx[kv_num_heads],
                Idx[depth],
            ),
        )
    )

    flash_attention_dispatch[
        kv_num_heads=kv_num_heads,
        config=config,
        ragged=False,
        sink=sink,
        _is_flash_attention_applicable=flash_attention_applicable,
        _is_cache_length_accurate=True,
        _use_valid_length=_use_valid_length,
        _padded_ndbuffer=_padded_ndbuffer,
        decoding_warp_split_k=decoding_warp_split_k,
    ](
        output,
        q,
        k_operand,
        v_operand,
        mask_functor,
        seq_len,
        num_keys,
        scale,
        is_token_generation,
        ctx,
        valid_length,
        None,
        num_partitions,
        sink_weights,
    )


def flash_attention[
    mask_t: MHAMask,
    dtype: DType,
    k_type: DType,
    v_type: DType,
    output_type: DType,
    q_tt_layout: TensorLayout,
    k_tt_layout: TensorLayout,
    v_tt_layout: TensorLayout,
    output_tt_layout: TensorLayout,
    //,
    config: MHAConfig[dtype] = {
        q_tt_layout.static_shape[2],
        q_tt_layout.static_shape[3],
    },
    decoding_warp_split_k: Bool = False,
    _use_valid_length: Bool = False,
    _padded_ndbuffer: Bool = False,
    naive_kernel: Bool = False,
    sink: Bool = False,
](
    output: TileTensor[
        mut=True, output_type, output_tt_layout, address_space=.GENERIC, ...
    ],
    q: TileTensor[mut=False, dtype, q_tt_layout, address_space=.GENERIC, ...],
    k: TileTensor[mut=False, k_type, k_tt_layout, address_space=.GENERIC, ...],
    v: TileTensor[mut=False, v_type, v_tt_layout, address_space=.GENERIC, ...],
    mask_functor: mask_t,
    scale: Float32,
    ctx: DeviceContext,
    # if not set, we select num_partitions based on heuristics
    num_partitions: Optional[Int] = None,
    valid_length: OptionalReg[ImmutTileTensor1D[.uint32]] = None,
    sink_weights: OptionalReg[ImmutTileTensor1D[dtype]] = None,
) raises:
    """Runs flash attention with dense tensor operands.

    Delegates native tensor operands to the shared dense implementation.

    Parameters:
        mask_t: Attention mask type implementing `MHAMask` (inferred).
        dtype: Element type of the query tensor (inferred).
        k_type: Element type of the key tensor (inferred).
        v_type: Element type of the value tensor (inferred).
        output_type: Element type of the output tensor, which may differ
            from `dtype` (inferred).
        q_tt_layout: Compile-time `TensorLayout` of the query tensor
            (inferred).
        k_tt_layout: Compile-time `TensorLayout` of the key tensor
            (inferred).
        v_tt_layout: Compile-time `TensorLayout` of the value tensor
            (inferred).
        output_tt_layout: Compile-time `TensorLayout` of the output tensor
            (inferred).
        config: Tile/pipeline configuration; defaults are derived from
            the query layout's last two dimensions.
        decoding_warp_split_k: `True` to enable warp-level split-K for
            decode (defaults to `False`).
        _use_valid_length: `True` to mask output with per-sequence lengths
            (defaults to `False`).
        _padded_ndbuffer: `True` when the NBuffer holds padded dense inputs
            (defaults to `False`).
        naive_kernel: `True` to force the fallback naive attention kernel
            (defaults to `False`).
        sink: `True` to enable attention-sink mode where the first tokens
            always attend (defaults to `False`).

    Args:
        output: Mutable destination `TileTensor` for the attention output.
        q: Query `TileTensor`.
        k: Key `TileTensor`.
        v: Value `TileTensor`.
        mask_functor: Mask instance used to apply the attention mask.
        scale: Softmax temperature scale applied to Q·Kᵀ.
        ctx: GPU device context for kernel dispatch.
        num_partitions: Override the number of split-K partitions; `None`
            selects automatically.
        valid_length: Per-sequence valid lengths for masking padded batches.
        sink_weights: Optional sink-token weight tensor for attention sinks.
    """
    _flash_attention_dense[
        config=config,
        decoding_warp_split_k=decoding_warp_split_k,
        _use_valid_length=_use_valid_length,
        _padded_ndbuffer=_padded_ndbuffer,
        naive_kernel=naive_kernel,
        sink=sink,
    ](
        output,
        q,
        k,
        v,
        mask_functor,
        scale,
        ctx,
        num_partitions,
        valid_length,
        sink_weights,
    )


def flash_attention_ragged[
    mask_t: MHAMask,
    type: DType,
    q_layout: TensorLayout,
    //,
    config: MHAConfig[type] = MHAConfig[type](
        q_layout.static_shape[q_layout.rank - 2],
        q_layout.static_shape[q_layout.rank - 1],
    ),
    decoding_warp_split_k: Bool = False,
    naive_kernel: Bool = False,
](
    output: TileTensor[mut=True, address_space=.GENERIC, ...],
    q: TileTensor[mut=False, type, q_layout, address_space=.GENERIC, ...],
    k: TileTensor[mut=False, address_space=.GENERIC, ...],
    v: TileTensor[mut=False, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    max_prompt_len: TileTensor[mut=False, .uint32, address_space=.GENERIC, ...],
    mask_functor: mask_t,
    scale: Float32,
    ctx: DeviceContext,
    # if not set, we select num_partitions based on heuristics
    num_partitions: Optional[Int] = None,
) raises:
    """Run flash attention on ragged (variable-length) batch inputs.

    Accepts Q/K/V as flat rank-3 tensors with shape
    `[total_seq_len, num_heads, head_dim]` and a CSR-style
    `input_row_offsets` tensor of length `batch + 1` that encodes per-sequence
    boundaries. Dispatches the same kernel paths as the dense overload but
    wraps K/V in `RaggedMHAOperand` adapters.

    Parameters:
        mask_t: Attention mask type implementing `MHAMask`.
        type: Element data type for Q/K/V and the output.
        q_layout: Compile-time layout of the query tensor.
        config: Tile/pipeline configuration; defaults from query shape.
        decoding_warp_split_k: Enable warp-level split-K for decode.
        naive_kernel: Force the fallback naive attention kernel.

    Args:
        output: Mutable output tensor, same shape as Q.
        q: Query tensor `[total_seq_len, num_heads, head_dim]`.
        k: Key tensor `[total_seq_len, kv_heads, head_dim]`.
        v: Value tensor `[total_seq_len, kv_heads, head_dim]`.
        input_row_offsets: Contiguous scalar CSR row offsets `[batch + 1]`.
        max_prompt_len: Scalar tensor holding the maximum sequence length.
        mask_functor: Mask instance.
        scale: Softmax temperature scale.
        ctx: GPU device context.
        num_partitions: Override split-K partition count; `None` for auto.

    Raises:
        If `input_row_offsets` does not have unit stride.
    """

    # See the kV cache overloads for comments.

    comptime assert q.rank == 3, "only support rank 3 inputs for ragged inputs."
    comptime assert input_row_offsets.rank == input_row_offsets.flat_rank == 1
    comptime assert input_row_offsets.element_size == 1
    if Int(input_row_offsets.layout.stride[0]().value()) != 1:
        raise Error("input_row_offsets must have unit stride")
    comptime assert (
        q.dtype == k.dtype == v.dtype == output.dtype
    ), "Q, K, V, output should have same type."

    comptime assert (
        q.dtype == .float32
        or q.dtype.is_half_float()
        or (q.dtype.is_float8() and ctx.target.is_amd_gpu())
    ), "Only support single, half, and float8 (AMD only) precision."

    # Runtime dimensions.
    # For ragged inputs: [total_seq_len, num_heads, head_dim]
    # fmt: off
    comptime head_depth_known = q.static_shape[1] != UNKNOWN_VALUE and q.static_shape[2] != UNKNOWN_VALUE and k.static_shape[1] != UNKNOWN_VALUE
    comptime depth = q.static_shape[q.rank - 1]
    comptime gpu_info = ctx.default_device_info
    comptime head_depth_supported = depth_supported_by_gpu[depth, mask_t, config, gpu_info]()
    comptime flash_attention_applicable = flash_attention_hw_supported[type]() and head_depth_known and head_depth_supported and not naive_kernel
    comptime kv_num_heads = k.static_shape[1]
    # fmt: on

    var is_token_generation = False

    var cache_row_offsets = immut_tile_tensor_1d(
        input_row_offsets.unsafe_ptr(), input_row_offsets.num_elements()
    )
    var k_operand = RaggedMHAOperand(
        TileTensor(
            k.unsafe_ptr(),
            row_major(Int(k.dim[0]()), Int(k.dim[1]()), Int(k.dim[2]())),
        ),
        cache_row_offsets,
    )
    var v_operand = RaggedMHAOperand(
        TileTensor(
            v.unsafe_ptr(),
            row_major(Int(v.dim[0]()), Int(v.dim[1]()), Int(v.dim[2]())),
        ),
        cache_row_offsets,
    )
    flash_attention_dispatch[
        kv_num_heads=kv_num_heads,
        config=config,
        ragged=True,
        _is_flash_attention_applicable=flash_attention_applicable,
        _is_cache_length_accurate=True,
        decoding_warp_split_k=decoding_warp_split_k,
    ](
        output,
        q,
        k_operand,
        v_operand,
        mask_functor,
        Int(max_prompt_len[0]),
        Int(max_prompt_len[0]),
        scale,
        is_token_generation,
        ctx,
        OptionalReg[ImmutTileTensor1D[.uint32]](cache_row_offsets),
        None,
        num_partitions,
    )


def get_waves_per_eu(depth: Int) -> Int:
    """Return the recommended `rocdl.waves_per_eu` hint for an AMD MHA kernel.

    AMD GCN/CDNA schedulers use this hint to decide how many wavefronts to
    co-issue per execution unit. Shallow heads (depth 64 or 128) benefit
    from two waves to hide memory latency, while deeper heads use one wave
    to conserve register file capacity.

    Args:
        depth: Attention head depth (key/value dimension per head).

    Returns:
        `2` for depth 64 or 128, `1` otherwise.
    """

    if depth in [64, 128]:
        return 2
    else:
        return 1


# ===-----------------------------------------------------------------------===#
# Flash attention for context encoding
# ===-----------------------------------------------------------------------===#


@__llvm_metadata(
    `rocdl.waves_per_eu`=SIMDLength(get_waves_per_eu(config.depth))
)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(config.num_threads())
    )
)
# Force the AMDGPU register allocator off AGPRs for this kernel. With AGPRs
# unavailable the `AMDGPURewriteAGPRCopyMFMA` pass has no copies to rewrite,
# sidestepping the SSA-verifier failures it otherwise produces at depth=512
# under heavy register pressure (e.g. gemma4 24Q/3KV BF16 prefill, where it
# breaks "virtual register defs don't dominate all uses"; see also the
# `IntervalMap.h "Overlapping insert"` variant covered by
# `test_mha_gemma4_sink_repro.mojo`). Harmless on NVIDIA — the attribute is
# AMDGPU-specific and ignored elsewhere.
@__llvm_metadata(`rocdl.no_agpr`=SIMDLength(1))
@__name(
    t"mha_depth{config.depth}_{q_type}_{output_type}_{ragged}_{is_shared_kv}_nqh{config.num_heads}_nkvh{config.num_heads // group}",
)
def mha[
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    config: MHAConfig,
    group: Int = 1,
    ragged: Bool = False,
    is_shared_kv: Bool = False,
    sink: Bool = False,
    _use_valid_length: Bool = False,
    _is_cache_length_accurate: Bool = False,
    _padded_ndbuffer: Bool = False,
](
    q_ptr: UnsafePointer[Scalar[q_type], ImmutAnyOrigin],
    k: k_t,
    v: v_t,
    output_ptr: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    scale: Float32,
    batch_size: Int32,
    seq_len_arg: Int32,
    num_keys_arg: Int32,
    valid_length: ImmutTileTensor1D[.uint32],
    kv_input_row_offsets: OptionalReg[ImmutTileTensor1D[.uint32]],
    sink_weights: OptionalReg[ImmutTileTensor1D[q_type]],
    mask: mask_t,
):
    var _batch_size = Int(batch_size)
    var _seq_len_arg = Int(seq_len_arg)
    var _num_keys_arg = Int(num_keys_arg)
    var batch_idx = block_idx.z

    # mha inputs
    var seq_len: Int
    var max_seq_len = _seq_len_arg
    var num_keys: Int
    var mask_tensor_col = _num_keys_arg
    var start_pos: UInt32 = 0

    @inline(.always)
    def q_block_idx() -> Int:
        return block_idx.x if is_nvidia_gpu() else block_idx.y

    var q_batch_offset: Int
    comptime if ragged:
        # treat valid_lengths as a input_row_offsets
        var start_of_seq = Int(valid_length[batch_idx])
        var end_of_seq = Int(valid_length[batch_idx + 1])
        seq_len = end_of_seq - start_of_seq

        if seq_len < q_block_idx() * config.block_m():
            return

        comptime if not _is_cache_length_accurate:
            start_pos = UInt32(k.cache_length(batch_idx))

        # this is used for cross attention where we get the num_keys
        # from kv_input_row_offsets. This is when num_keys != seq_len
        if kv_input_row_offsets:
            var kv_row_offsets = kv_input_row_offsets.value()
            var kv_seq_start = Int(kv_row_offsets[batch_idx])
            var kv_seq_end = Int(kv_row_offsets[batch_idx + 1])
            var cur_kv_len = kv_seq_end - kv_seq_start
            num_keys = cur_kv_len + Int(start_pos)
        else:
            num_keys = seq_len + Int(start_pos)

        q_batch_offset = start_of_seq * config.depth * config.num_heads

    # KVCache inputs, prompt lengths are all padded to the max in batch.
    elif _use_valid_length and not _padded_ndbuffer:
        # treat valid_lengths as valid lengths
        seq_len = Int(valid_length[batch_idx])

        if seq_len < q_block_idx() * config.block_m():
            return

        comptime if not _is_cache_length_accurate:
            var cache_length = k.cache_length(batch_idx)
            start_pos = UInt32(cache_length)

        num_keys = seq_len + k.cache_length(batch_idx)
        q_batch_offset = (
            config.depth * config.num_heads * max_seq_len * batch_idx
        )
    # Dense tensor inputs, homogeneous and padded batching.
    else:
        comptime if _padded_ndbuffer:
            seq_len = Int(valid_length[batch_idx])
            num_keys = seq_len
        else:
            seq_len = _seq_len_arg
            num_keys = _num_keys_arg

        if seq_len < q_block_idx() * config.block_m():
            return
        q_batch_offset = (
            config.depth * config.num_heads * max_seq_len * batch_idx
        )

        # When cache length (num_keys) is greater, we assume it has
        # prefix preceding the input seq_len.
        start_pos = UInt32(num_keys - seq_len)

    comptime if is_nvidia_gpu():
        comptime if is_shared_kv:
            mha_single_batch_pipelined[
                config=config,
                group=group,
                sink=sink,
            ](
                q_ptr + q_batch_offset,
                k,
                v,
                output_ptr + q_batch_offset,
                scale,
                seq_len,
                max_seq_len,
                start_pos,
                num_keys,
                mask_tensor_col,
                mask,
                batch_idx,
                sink_weights,
            )
        else:
            mha_single_batch[
                config=config,
                group=group,
                sink=sink,
            ](
                q_ptr + q_batch_offset,
                k,
                v,
                output_ptr + q_batch_offset,
                scale,
                seq_len,
                max_seq_len,
                start_pos,
                num_keys,
                mask_tensor_col,
                mask,
                batch_idx,
                sink_weights,
            )
    elif is_amd_gpu():
        # Single unified prefill kernel — handles BF16+FP8, any mask,
        # depth∈{64,128,256,512}, with/without sink. Depth-supported asserts
        # live in the kernel itself. Branches on `_is_amd_rdna()` because
        # gfx950 (CDNA) and gfx11/12 (RDNA) need different fragment
        # geometry / wave size / WMMA intrinsics.
        var sink_weights_ptr = OptionalReg[
            UnsafePointer[Scalar[q_type], ImmutAnyOrigin]
        ]()
        comptime if sink:
            sink_weights_ptr = sink_weights.value().ptr

        comptime if _is_amd_rdna():
            var attention = AttentionRDNA[config, group, sink](
                output_ptr + q_batch_offset,
                q_ptr + q_batch_offset,
                k,
                v,
                mask,
                sink_weights_ptr,
                batch_idx,
                scale,
                seq_len,
                num_keys,
                Int(start_pos),
            )
            attention.mha_prefill()
        else:
            # AMD CDNA prefill via FA2. The long-context `MhaPrefillV2` path
            # is dispatched host-side from `flash_attention_dispatch` so the
            # kernel keeps its tuned single-kernel register-allocation context
            # (`def mha[]`'s body holding the FA2 fallback inflates spills when
            # the kernel is inlined here — measured ~14% loss vs `MhaPrefillV2`
            # as a top-level kernel).
            var attention = Attention[config, group, sink](
                output_ptr + q_batch_offset,
                q_ptr + q_batch_offset,
                k,
                v,
                mask,
                sink_weights_ptr,
                batch_idx,
                scale,
                seq_len,
                num_keys,
                Int(start_pos),
            )
            attention.mha_prefill()
    else:
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name()
        ]()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(config.num_threads())
    )
)
@inline(.always)
@__name(
    t"mha_single_batch_depth{config.depth}_{q_type}_{output_type}_nqh{config.num_heads}_nkvh{config.num_heads // group}",
)
def mha_single_batch[
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    *,
    config: MHAConfig,
    group: Int = 1,
    sink: Bool = False,
](
    q_ptr: UnsafePointer[Scalar[q_type], ImmutAnyOrigin],
    k: k_t,
    v: v_t,
    output_ptr: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    scale: Float32,
    seq_len: Int,  # valid sequence length i.e. w/o padding.
    max_seq_len: Int,  # sequence length after padding.
    start_pos: UInt32,
    num_keys: Int,
    mask_tensor_col: Int,  # second dimension of mask tensor
    mask: mask_t,
    batch_idx: Int,
    sink_weights: OptionalReg[ImmutTileTensor1D[q_type]],
):
    """MHA for token gen where seqlen = 1 and num_keys >= 1.

    The general data layout and steps conform to flash attention. Two exceptions:

    1 Partition across B, H, and num_keys (TODO). The last one is split-K and
      will need a separate reduction kernel at the end.

    2 First bmm becomes gemv and second bmm becomes gevm.
      TODO: use more optimized kernels for them

    Parameters:
        q_type: Element type of the query tensor.
        k_t: Key operand type implementing `MHAOperand`.
        v_t: Value operand type implementing `MHAOperand`.
        output_type: Element type of the output tensor.
        mask_t: Attention mask type implementing `MHAMask`.
        config: Tile and pipeline configuration for the kernel.
        group: GQA group size, query heads per key/value head (defaults to 1).
        sink: `True` to enable attention-sink mode where the first tokens
            always attend (defaults to `False`).

    Args:
        q_ptr: Pointer to the query tensor data in global memory.
        k: Key operand backed by a KV cache or dense tensor.
        v: Value operand backed by a KV cache or dense tensor.
        output_ptr: Pointer to the output buffer in global memory.
        scale: Softmax temperature scale applied to Q·Kᵀ.
        seq_len: Valid query sequence length excluding padding.
        max_seq_len: Padded query sequence length used for batch offsets.
        start_pos: Starting position of the current tokens in the KV cache.
        num_keys: Number of key/value entries to attend over.
        mask_tensor_col: Second dimension of the mask tensor, equal to the
            key sequence length.
        mask: Mask instance used to apply the attention mask.
        batch_idx: Index of the current sequence within the batch.
        sink_weights: Optional sink-token weight tensor; required when `sink`
            is `True`.
    """
    comptime accum_type = get_accum_type[q_type]()
    comptime k_type = k_t.dtype
    comptime v_type = v_t.dtype
    comptime assert q_type == k_type and k_type == v_type

    comptime simd_size = simd_width_of[q_type]()

    comptime num_warps_m = config.num_warps_m()
    comptime num_warps_n = config.num_warps_n()
    comptime num_threads = config.num_threads()
    comptime BM = config.block_m()
    comptime BN = config.block_n()
    comptime BK = config.block_k()
    comptime num_heads = config.num_heads
    comptime depth = config.depth

    comptime assert num_warps_m * num_warps_n == (
        num_threads // WARP_SIZE
    ), "Number of warps doesn't match warp tile sizes."

    var tid = UInt32(thread_idx.x)
    var warp_id: UInt32 = warp.broadcast(tid // UInt32(WARP_SIZE))
    var lane = UInt32(lane_id())

    # Coordinates of the current warp.
    var warp_y, warp_x = divmod(warp_id, UInt32(num_warps_n))

    # The entire query block (BM x depth) is tiled in shared memory.
    comptime alignment = align_of[SIMD[q_type, simd_size]]()
    comptime q_smem_size = config.q_smem_size()
    var q_smem = external_memory[
        Scalar[q_type],
        address_space=.SHARED,
        alignment=alignment,
    ]()
    comptime IteratorTypeQ = LayoutTensorIter[
        q_type,
        Layout.row_major(BM, BK),
        _,
        address_space=.SHARED,
        alignment=alignment,
    ]
    var q_smem_iter = IteratorTypeQ(
        rebind[
            type_of(
                LayoutTensorIter[
                    q_type,
                    Layout.row_major(BM, BK),
                    q_smem.origin,
                    address_space=.SHARED,
                    alignment=alignment,
                ]().ptr
            )
        ](q_smem),
        IteratorTypeQ.layout_uint_type(q_smem_size),
    )
    # There is one pre-allocated dynamic shared buffer.
    # Need to explicitly offset key after at query's end.
    comptime k_smem_size = config.k_smem_size()
    var k_smem = (q_smem + q_smem_size).bitcast[Scalar[k_type]]()
    comptime IteratorTypeK = LayoutTensorIter[
        k_type,
        Layout.row_major(BN, BK),
        _,
        address_space=.SHARED,
        circular=True,
    ]
    var k_smem_iter = IteratorTypeK(
        k_smem, IteratorTypeK.layout_uint_type(k_smem_size)
    )

    comptime v_smem_size = config.v_smem_size()
    var v_smem = (k_smem + k_smem_size).bitcast[Scalar[v_type]]()
    comptime IteratorTypeV = LayoutTensorIter[
        v_type,
        Layout.row_major(BK, BN),
        _,
        address_space=.SHARED,
        circular=True,
    ]
    var v_smem_iter = IteratorTypeV(
        v_smem, IteratorTypeV.layout_uint_type(v_smem_size)
    )

    var head_idx = UInt32(block_idx.y)
    var q_tile_idx = UInt32(block_idx.x)

    # Query global memory iterator
    comptime q_gmem_layout = Layout(
        IntTuple(BM, depth), IntTuple(num_heads * depth, 1)
    )
    var q_tile_num_rows = min(
        UInt32(BM), UInt32(seq_len) - q_tile_idx * UInt32(BM)
    )
    var q_offset = UInt32(depth) * (
        head_idx + UInt32(num_heads) * q_tile_idx * UInt32(BM)
    )
    var q_gmem_block = LayoutTensor[
        q_type,
        q_gmem_layout,
        layout_int_type=.int32,
        linear_idx_type=.int32,
        masked=True,
    ](
        q_ptr + Int(q_offset),
        RuntimeLayout[element_type=.int32, linear_idx_type=.int32](
            RuntimeTuple[q_gmem_layout.shape, element_type=.int32](
                Int(q_tile_num_rows), depth
            ),
            RuntimeTuple[q_gmem_layout.stride, element_type=.int32](
                num_heads * depth, 1
            ),
        ),
    )
    var q_gmem_iter = q_gmem_block.tiled_iterator[BM, BK, axis=1](0, 0)
    # q tile has valid shape q_tile_num_rows x depth
    # q_tile_num_rows could be less than BM when seqlen % BM != 0

    comptime mma_shape = get_mma_shape[q_type, accum_type]()
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]
    comptime WM = config.WM
    comptime WN = config.WN
    comptime num_m_mmas = WM // MMA_M
    comptime num_n_mmas = WN // MMA_N

    comptime frag_size = get_fragment_size[mma_shape]()
    comptime p_frag_size = frag_size[2]
    comptime p_frag_simdwidth = p_frag_size // 2
    comptime p_frag_align = align_of[SIMD[accum_type, p_frag_size]]()

    var p_reg_tile_native = stack_allocation[
        accum_type, address_space=.LOCAL, alignment=p_frag_align
    ](row_major[num_m_mmas * num_n_mmas, p_frag_size]())
    var output_reg_tile_native = stack_allocation[
        accum_type, address_space=.LOCAL, alignment=p_frag_align
    ](row_major[num_m_mmas * num_n_mmas, p_frag_size]()).fill(0)
    # Copy helpers still require these legacy scalar views.
    var p_reg_tile = p_reg_tile_native.to_layout_tensor().as_unsafe_any_origin()
    var output_reg_tile = (
        output_reg_tile_native.to_layout_tensor().as_unsafe_any_origin()
    )

    # Rowwise max and sum for online softmax
    comptime row_alignment = align_of[
        SIMD[accum_type, simd_width_of[accum_type]()]
    ]()
    var rowmax = unsafe_stack_allocation[
        WM, accum_type, alignment=row_alignment
    ]()
    var rowsum = unsafe_stack_allocation[
        WM, accum_type, alignment=row_alignment
    ]()

    comptime for i in range(0, WM, 2):
        comptime if sink:
            assert Bool(
                sink_weights
            ), "expect sink_weights to be non-null when sink=true"
            var sink_logit_log2 = (
                sink_weights.value()[Int(head_idx)].cast[accum_type]() * log2e
            )
            rowmax.store(
                i,
                SIMD[accum_type, 2](sink_logit_log2),
            )
            # exp(sink_val-sink_val) = exp(0) = 1
            rowsum.store(i, SIMD[accum_type, 2](1))
        else:
            rowmax.store(i, SIMD[accum_type, 2](min_or_neg_inf[accum_type]()))
            rowsum.store(i, SIMD[accum_type, 2](0))

    # Shared memory for P = Q * K^t
    # This overlaps key tile but are used at the same time i.e. no race condition.
    var p_smem = (v_smem + v_smem_size).bitcast[Scalar[v_type]]()
    comptime IteratorTypeP = LayoutTensorIter[
        v_type,
        Layout.row_major(BM, BK),
        _,
        address_space=.SHARED,
        circular=True,
    ]
    var p_smem_iter = IteratorTypeP(
        p_smem, IteratorTypeP.layout_uint_type(BM * BN)
    )

    # Scratch shared memory for reduction across warps.
    var warp_scratch = TileTensor(
        (p_smem + (BM * BN if num_warps_n > 1 else 0)).bitcast[
            Scalar[accum_type]
        ](),
        row_major[2 * num_warps_n, BM](),
    )

    # Mask global memory iterator.
    var mask_block_row = q_tile_idx * UInt32(BM)
    var mask_warp_row = warp_y * UInt32(WM)
    var mask_warp_col = warp_x * UInt32(WN)

    # Account for group query.
    comptime kv_num_heads = num_heads // group

    comptime num_pipeline_stages = config.num_pipeline_stages

    comptime q_num_vecs = BM * BK // simd_size

    comptime async_copy_q_layout = Layout.row_major(
        min(num_threads, q_num_vecs) * simd_size // BK,
        BK // simd_size,
    )

    comptime for q_id in range(depth // BK):
        var q_smem_tile = q_smem_iter.next_unsafe(
            q_smem_iter.layout_uint_type(q_id)
        )[]

        copy_dram_to_sram_async[
            thread_layout=async_copy_q_layout,
            swizzle=True,
            num_threads=num_threads,
        ](
            q_smem_tile.vectorize[1, simd_size](),
            q_gmem_iter[].vectorize[1, simd_size](),
        )

        # we `async_copy_commit_group()` and after we finish copying `k`.

        q_gmem_iter._incr()

    # Iterate over KV, equivalent to the following with if hoisted out.
    #   ```
    #   for i in range(kv_tile_start_row, seq_len, tile_size):
    #     if i + tile_size >= seq_len:
    #       loop_over_kvcache[tile_size, False]
    #     else:
    #       loop_over_kvcache[tile_size, True]
    #   ```
    # Only the last iteration is doing boundary check.
    @inline(.always)
    def loop_over_kvcache[
        tile_size: Int, not_last_iter: Bool
    ](kv_tile_start_row: Int, end: Int) {
        var seq_len,
        var num_keys,
        var start_pos,
        mut mask_warp_col,
        mut k_smem_iter,
        mut v_smem_iter,
        imm,
    }:
        if (
            mask.status(
                UInt32(batch_idx),
                Index[dtype=DType.uint32](
                    Int(q_tile_idx * UInt32(BM) + start_pos),
                    kv_tile_start_row,
                ),
                Index[dtype=DType.uint32](BM, BN),
            )
            == TileMaskStatus.FULL_MASK
        ):
            mask_warp_col += UInt32(BN)
            return

        comptime kv_gmem_layout = Layout(
            IntTuple(BN, depth),
            IntTuple(kv_num_heads * depth, 1),
        )
        var kv_tile_num_rows = min(tile_size, end - kv_tile_start_row)

        # kv cache gmem has to clip num rows as runtime layout
        var kv_runtime_layout = RuntimeLayout[kv_gmem_layout](
            {kv_tile_num_rows, depth},
            {kv_num_heads * depth, 1},
        )

        var k_gmem_block = LayoutTensor[
            k_type,
            kv_gmem_layout,
            layout_int_type=.int32,
            linear_idx_type=.int32,
            masked=not not_last_iter,
        ](
            k.block_paged_ptr[BN](
                UInt32(batch_idx),
                UInt32(kv_tile_start_row),
                UInt32(Int(head_idx // UInt32(group))),
                0,
            ),
            kv_runtime_layout,
        )
        var k_gmem_iter = k_gmem_block.tiled_iterator[BN, BK, axis=1](0, 0)

        var v_gmem_block = LayoutTensor[
            v_type,
            kv_gmem_layout,
            layout_int_type=.int32,
            linear_idx_type=.int32,
            masked=not not_last_iter,
        ](
            v.block_paged_ptr[BN](
                UInt32(batch_idx),
                UInt32(kv_tile_start_row),
                UInt32(Int(head_idx // UInt32(group))),
                0,
            ),
            kv_runtime_layout,
        )
        var v_gmem_iter = v_gmem_block.tiled_iterator[BK, BN, axis=0](0, 0)

        # P = Q @ K, register tile holding mma result.
        _ = p_reg_tile.fill(0)

        # Retained legacy: both the input (a legacy iterator deref) and the
        # consumer (`copy_dram_to_sram_async`) are legacy-only, so a native
        # slice view would just round-trip.
        @inline(.always)
        def _mask_tensor_row(
            tensor: LayoutTensor, num_rows: Int, out result: type_of(tensor)
        ) {imm}:
            return {
                tensor.ptr,
                type_of(tensor.runtime_layout)(
                    type_of(tensor.runtime_layout.shape)(
                        num_rows, tensor.dim[1]()
                    ),
                    tensor.runtime_layout.stride,
                ),
            }

        comptime kv_num_vecs = BN * BK // simd_size
        comptime async_copy_k_layout = Layout.row_major(
            min(num_threads, kv_num_vecs)
            * simd_size
            // k_smem_iter.layout.stride[0].value(),
            k_smem_iter.layout.stride[0].value() // simd_size,
        )

        # load K tile into smem
        comptime for k_id in range(depth // BK):
            var k_smem_tile = k_smem_iter.next_unsafe(
                k_smem_iter.layout_uint_type(k_id)
            )[]

            copy_dram_to_sram_async[
                thread_layout=async_copy_k_layout,
                swizzle=True,
                num_threads=num_threads,
            ](
                k_smem_tile.vectorize[1, simd_size](),
                k_gmem_iter[].vectorize[1, simd_size](),
            )

            k_gmem_iter._incr()

        async_copy_commit_group()
        # synchronize here since we can overlap q tile and first k tile copy
        async_copy_wait_all()
        barrier()

        multistage_mma[
            BM,
            BN,
            BK,
            WM,
            WN,
            num_threads,
            num_pipeline_stages,
            True,  # transpose_b
            swizzle_a=True,
            prefetch_init=False,
            static_num_iters=ufloordiv(depth, BK),
            k_group_size=config.k_group_size,
        ](
            lt_to_tt_idx[linear_idx_type=p_reg_tile.linear_idx_type](
                p_reg_tile
            ),
            q_smem_iter,
            k_smem_iter,
            q_smem_iter,
            k_smem_iter,
            ufloordiv(depth, BK),
        )

        # Vectorize by 2.
        var p_reg_vec2 = p_reg_tile.vectorize[1, p_frag_simdwidth]()

        def _apply_mask[masked: Bool]() {imm}:
            var scale_log2e: Scalar[accum_type] = (
                scale.cast[
                    accum_type
                ]() if mask_t.apply_log2e_after_mask else scale.cast[
                    accum_type
                ]()
                * log2e
            )

            comptime for m_mma in range(num_m_mmas):
                comptime for n_mma in range(num_n_mmas):
                    comptime mma_id = n_mma * num_m_mmas + m_mma

                    # Coordinates in mask for current mma tile.
                    var mask_frag_row = mask_warp_row + UInt32(m_mma * MMA_M)
                    var mask_frag_col = mask_warp_col + UInt32(n_mma * MMA_N)

                    # Offset to current thread's fragment
                    mask_frag_row += lane // UInt32(MMA_N // p_frag_simdwidth)
                    mask_frag_col += (
                        lane * UInt32(p_frag_simdwidth) % UInt32(MMA_N)
                    )

                    comptime for i in range(2):
                        # The row in score matrix of shape seq_len x num_keys.
                        # Mask col is score col since we don't partition in col.
                        var score_row = (
                            mask_block_row
                            + mask_frag_row
                            + UInt32(i * MMA_M // 2)
                        )
                        var score_col = mask_frag_col

                        var score_row_with_start_pos = score_row + start_pos

                        comptime if masked:
                            p_reg_vec2[mma_id, i] = mask.mask(
                                IndexList[4, element_type=.uint32](
                                    block_idx.z,
                                    block_idx.y,
                                    Int(score_row_with_start_pos),
                                    Int(score_col),
                                ),
                                p_reg_vec2[mma_id, i] * scale_log2e,
                            )
                        else:
                            p_reg_vec2[mma_id, i] = (
                                p_reg_vec2[mma_id, i] * scale_log2e
                            )

                        comptime if mask_t.apply_log2e_after_mask:
                            p_reg_vec2[mma_id, i] = (
                                p_reg_vec2[mma_id, i] * log2e
                            )

                        comptime if not not_last_iter:
                            p_reg_vec2[mma_id, i] = _kernel_mask(
                                IndexList[2, element_type=.uint32](
                                    Int(score_row), Int(score_col)
                                ),
                                IndexList[2, element_type=.uint32](
                                    seq_len,
                                    num_keys,
                                ),
                                p_reg_vec2[mma_id, i],
                            )

        unswitch(
            mask.status(
                UInt32(batch_idx),
                Index[dtype=DType.uint32](
                    Int(q_tile_idx * UInt32(BM) + start_pos),
                    kv_tile_start_row,
                ),
                Index[dtype=DType.uint32](BM, BN),
            )
            == TileMaskStatus.PARTIAL_MASK,
            _apply_mask,
        )

        # Increment mask to next BM x BN block.
        mask_warp_col += UInt32(BN)

        comptime reg_layout_by_mma_unit = Layout.row_major(
            2 * num_m_mmas * num_n_mmas, 2
        )
        _online_softmax_iter_for_mma_output[
            accum_type,
            # score layout by mma unit
            # TODO: generalize beyond 16x8 layout
            Layout.row_major(2 * num_m_mmas, num_n_mmas),
            # threads layout by warp
            Layout.row_major(num_warps_m, num_warps_n),
            Layout.row_major(8, 4),
            use_exp2=True,
        ](
            lt_to_tt_idx[linear_idx_type=output_reg_tile.linear_idx_type](
                output_reg_tile.reshape[reg_layout_by_mma_unit]()
            ).vectorize[1, 2](),
            lt_to_tt_idx[linear_idx_type=p_reg_tile.linear_idx_type](
                p_reg_tile.reshape[reg_layout_by_mma_unit]()
            ).vectorize[1, 2](),
            warp_scratch.tile[2 * num_warps_n, WM](Coord(0, Int(warp_y))),
            rowmax,
            rowsum,
        )

        comptime async_copy_v_layout = Layout.row_major(
            min(num_threads, kv_num_vecs)
            * simd_size
            // v_smem_iter.layout.stride[0].value(),
            v_smem_iter.layout.stride[0].value() // simd_size,
        )

        var v_tensor: type_of(v_gmem_iter[])
        # load V tile into smem
        comptime for v_id in range(BN // BK):
            var v_smem_tile = v_smem_iter.next_unsafe(
                v_smem_iter.layout_uint_type(v_id)
            )[]

            comptime if not not_last_iter:
                var num_rows_bound = min(
                    BK, end - (kv_tile_start_row + v_id * BK)
                )
                v_tensor = _mask_tensor_row(v_gmem_iter[], num_rows_bound)
            else:
                v_tensor = v_gmem_iter[]

            copy_dram_to_sram_async[
                thread_layout=async_copy_v_layout,
                swizzle=v_smem_tile.dtype.is_half_float(),
                num_threads=num_threads,
            ](
                v_smem_tile.vectorize[1, simd_size](),
                v_tensor.vectorize[1, simd_size](),
            )

            v_gmem_iter._incr()

        async_copy_commit_group()

        comptime if num_warps_n > 1:
            # Pack the per-thread fragments in shared memory for 2nd mma.
            _copy_frag_to_smem[
                BM,
                BN,
                BK,
                WM,
                WN,
                MMA_M,
                MMA_N,
                p_frag_simdwidth,
            ](p_smem_iter, p_reg_tile, warp_x, warp_y)

            async_copy_wait_all()
            barrier()

            multistage_mma[
                BM,
                BN,
                BK,
                WM,
                WN,
                num_threads,
                num_pipeline_stages,
                False,  # transpose_b
                swizzle_a=True,
                prefetch_init=False,
                static_num_iters=ufloordiv(BN, BK),
                k_group_size=config.k_group_size,
            ](
                lt_to_tt_idx[linear_idx_type=output_reg_tile.linear_idx_type](
                    output_reg_tile
                ),
                p_smem_iter,
                v_smem_iter,
                p_smem_iter,
                v_smem_iter,
                ufloordiv(BN, BK),
            )

        else:
            # Reuse 1st mma output (MMA_M, MMA_N) as 2nd mma's input (MMA_M, MMA_K).
            # The num_n_mmas dim becomes "num_k_mmas" for 2nd mma.
            var p_reg_iter = p_reg_tile.tiled_iterator[
                MMA_K // MMA_N * num_m_mmas, p_frag_size
            ](0, 0)

            async_copy_wait_all()
            barrier()

            multistage_mma[
                BM,
                BN,
                BK,
                WM,
                WN,
                num_threads,
                num_pipeline_stages,
                False,  # transpose_b
                swizzle_a=False,
                prefetch_init=False,
                static_num_iters=ufloordiv(BN, BK),
                k_group_size=config.k_group_size,
            ](
                lt_to_tt_idx[linear_idx_type=output_reg_tile.linear_idx_type](
                    output_reg_tile
                ),
                p_reg_iter,
                v_smem_iter,
                p_smem_iter,
                v_smem_iter,
                ufloordiv(BN, BK),
            )

    tile_and_unswitch[[BN]](0, num_keys, loop_over_kvcache)

    # Apply softmax denumerator.
    comptime for m_mma in range(num_m_mmas):
        var rowsum_inv0 = recip(rowsum[2 * m_mma])
        var rowsum_inv1 = recip(rowsum[2 * m_mma + 1])

        comptime for n_mma in range(num_n_mmas):
            comptime for i in range(p_frag_size // 2):
                output_reg_tile[n_mma * num_m_mmas + m_mma, i] *= rowsum_inv0
                output_reg_tile[
                    n_mma * num_m_mmas + m_mma, i + p_frag_size // 2
                ] *= rowsum_inv1

    comptime output_gmem_layout = Layout(
        IntTuple(BM, depth), IntTuple(num_heads * depth, 1)
    )
    var output_gmem_tile = LayoutTensor[
        output_type,
        output_gmem_layout,
        layout_int_type=.int32,
        linear_idx_type=.int32,
        masked=True,
    ](
        output_ptr + Int(q_offset),
        RuntimeLayout[element_type=.int32, linear_idx_type=.int32](
            RuntimeTuple[output_gmem_layout.shape, element_type=.int32](
                Int(q_tile_num_rows), depth
            ),
            RuntimeTuple[output_gmem_layout.stride, element_type=.int32](
                num_heads * depth, 1
            ),
        ),
    )
    var output_gmem_warp_tile = output_gmem_tile.tile[WM, WN](
        Int(warp_y), Int(warp_x)
    )

    # Write to global memory.
    comptime if output_type.is_half_float():
        comptime swizzle = make_swizzle[
            num_rows=MMA_M // 2, row_size=WN, access_size=MMA_N
        ]()
        # Reuse a_smem for c tile in smem
        var accum_smem_tile = LayoutTensor[
            output_type,
            Layout.row_major(BM, depth),
            address_space=.SHARED,
        ](q_smem.bitcast[Scalar[output_type]]())

        var accum_smem_warp_tile = accum_smem_tile.tile[WM, WN](
            Int(warp_y), Int(warp_x)
        )
        copy_local_to_shared[
            thread_layout=Layout.row_major(8, 4), swizzle=swizzle
        ](
            accum_smem_warp_tile.vectorize[1, 2](),
            output_reg_tile.vectorize[1, 2]().transpose(),
        )

        # Guard writing to shared memory.
        barrier()

        # Vectorized copy from shared to global memory, during which every 2 FP32
        # are cast to 2 BF16 so that 2 4xFP32 vectors are merged into 1 8xBF16
        # vector and stored using 16B store instruction.
        copy_sram_to_dram[
            thread_layout=Layout.row_major(
                num_threads * simd_size // depth,
                depth // simd_size,
            ),
            swizzle=swizzle,
        ](
            output_gmem_tile.vectorize[1, simd_size](),
            accum_smem_tile.vectorize[1, simd_size](),
        )
    else:
        copy_local_to_dram[dst_thread_layout=Layout.row_major(8, 4)](
            output_gmem_warp_tile.vectorize[1, 2](),
            output_reg_tile.vectorize[1, 2]().transpose(),
        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(config.num_threads())
    )
)
@inline(.always)
@__name(
    t"mha_single_batch_pipelined_depth{config.depth}_{q_type}_{output_type}_nqh{config.num_heads}_nkvh{config.num_heads // group}",
)
def mha_single_batch_pipelined[
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    *,
    config: MHAConfig,
    group: Int = 1,
    sink: Bool = False,
](
    q_ptr: UnsafePointer[Scalar[q_type], ImmutAnyOrigin],
    k: k_t,
    v: v_t,
    output_ptr: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    scale: Float32,
    seq_len: Int,  # valid sequence length i.e. w/o padding.
    max_seq_len: Int,  # sequence length after padding.
    start_pos: UInt32,
    num_keys: Int,
    mask_tensor_col: Int,  # second dimension of mask tensor
    mask: mask_t,
    batch_idx: Int,
    sink_weights: OptionalReg[ImmutTileTensor1D[q_type]],
):
    """MHA for token gen where seqlen = 1 and num_keys >= 1.

    The general data layout and steps conform to flash attention. Two exceptions:

    1 Partition across B, H, and num_keys (TODO). The last one is split-K and
      will need a separate reduction kernel at the end.

    2 First bmm becomes gemv and second bmm becomes gevm.
      TODO: use more optimized kernels for them

    Parameters:
        q_type: Element type of the query tensor.
        k_t: Key operand type backing the key tensor (KV cache or dense).
        v_t: Value operand type backing the value tensor (KV cache or dense).
        output_type: Element type of the output tensor.
        mask_t: Attention mask type implementing `MHAMask`.
        config: Tile and pipeline configuration for the kernel.
        group: GQA group size, the ratio of query heads to key/value heads
            (defaults to 1).
        sink: `True` to enable attention-sink mode where the first tokens
            always attend (defaults to `False`).

    Args:
        q_ptr: Pointer to the query tensor in global memory.
        k: Key operand backed by a KV cache or dense tensor.
        v: Value operand backed by a KV cache or dense tensor.
        output_ptr: Pointer to the output tensor in global memory.
        scale: Softmax temperature scale applied to Q·Kᵀ.
        seq_len: Valid query sequence length excluding padding.
        max_seq_len: Padded query sequence length.
        start_pos: Starting position of the current query in the KV cache,
            used for mask row indexing.
        num_keys: Number of key entries in the KV cache.
        mask_tensor_col: Second dimension of the mask tensor.
        mask: Mask instance used to apply the attention mask.
        batch_idx: Index of the sequence within the batch.
        sink_weights: Optional sink-token weight tensor for attention sinks;
            required when `sink` is `True`.
    """
    comptime accum_type = get_accum_type[q_type]()
    comptime k_type = k_t.dtype
    comptime v_type = v_t.dtype
    comptime assert q_type == k_type and k_type == v_type

    comptime simd_size = simd_width_of[q_type]()

    comptime num_warps_m = config.num_warps_m()
    comptime num_warps_n = config.num_warps_n()
    comptime num_threads = config.num_threads()
    comptime BM = config.block_m()
    comptime BN = config.block_n()
    comptime BK = config.block_k()
    comptime num_heads = config.num_heads
    comptime depth = config.depth

    comptime assert num_warps_m * num_warps_n == (
        num_threads // WARP_SIZE
    ), "Number of warps doesn't match warp tile sizes."

    var tid = UInt32(thread_idx.x)
    var warp_id: UInt32 = warp.broadcast(tid // UInt32(WARP_SIZE))
    var lane = UInt32(lane_id())

    # Coordinates of the current warp.
    var warp_y, warp_x = divmod(warp_id, UInt32(num_warps_n))

    # The entire query block (BM x depth) is tiled in shared memory.
    comptime alignment = align_of[SIMD[q_type, simd_size]]()
    comptime q_smem_size = config.q_smem_size()
    var q_smem = external_memory[
        Scalar[q_type],
        address_space=.SHARED,
        alignment=alignment,
    ]()
    comptime IteratorTypeQ = LayoutTensorIter[
        q_type,
        Layout.row_major(BM, BK),
        _,
        address_space=.SHARED,
        alignment=alignment,
    ]
    var q_smem_iter = IteratorTypeQ(
        rebind[
            type_of(
                LayoutTensorIter[
                    q_type,
                    Layout.row_major(BM, BK),
                    q_smem.origin,
                    address_space=.SHARED,
                    alignment=alignment,
                ]().ptr
            )
        ](q_smem),
        IteratorTypeQ.layout_uint_type(q_smem_size),
    )
    # There is one pre-allocated dynamic shared buffer.
    # Need to explicitly offset key after at query's end.
    comptime k_smem_size = config.kv_smem_size()
    var k_smem = (q_smem + q_smem_size).bitcast[Scalar[k_type]]()
    comptime IteratorTypeK = LayoutTensorIter[
        k_type,
        Layout.row_major(BN, BK),
        _,
        address_space=.SHARED,
        circular=True,
    ]
    var k_smem_iter = IteratorTypeK(
        k_smem, IteratorTypeK.layout_uint_type(k_smem_size)
    )

    var head_idx = UInt32(block_idx.y)
    var q_tile_idx = UInt32(block_idx.x)

    # Query global memory iterator
    comptime q_gmem_layout = Layout(
        IntTuple(BM, depth), IntTuple(num_heads * depth, 1)
    )
    var q_tile_num_rows = min(
        UInt32(BM), UInt32(seq_len) - q_tile_idx * UInt32(BM)
    )
    var q_offset = UInt32(depth) * (
        head_idx + UInt32(num_heads) * q_tile_idx * UInt32(BM)
    )
    var q_gmem_block = LayoutTensor[
        q_type,
        q_gmem_layout,
        layout_int_type=.int32,
        linear_idx_type=.int32,
        masked=True,
    ](
        q_ptr + Int(q_offset),
        RuntimeLayout[element_type=.int32, linear_idx_type=.int32](
            RuntimeTuple[q_gmem_layout.shape, element_type=.int32](
                Int(q_tile_num_rows), depth
            ),
            RuntimeTuple[q_gmem_layout.stride, element_type=.int32](
                num_heads * depth, 1
            ),
        ),
    )
    var q_gmem_iter = q_gmem_block.tiled_iterator[BM, BK, axis=1](0, 0)
    # q tile has valid shape q_tile_num_rows x depth
    # q_tile_num_rows could be less than BM when seqlen % BM != 0

    comptime mma_shape = get_mma_shape[q_type, accum_type]()
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]
    comptime WM = config.WM
    comptime WN = config.WN
    comptime num_m_mmas = WM // MMA_M
    comptime num_n_mmas = WN // MMA_N

    comptime frag_size = get_fragment_size[mma_shape]()
    comptime p_frag_size = frag_size[2]
    comptime p_frag_simdwidth = p_frag_size // 2
    comptime p_frag_align = align_of[SIMD[accum_type, p_frag_size]]()

    var p_reg_tile_native = stack_allocation[
        accum_type, address_space=.LOCAL, alignment=p_frag_align
    ](row_major[num_m_mmas * num_n_mmas, p_frag_size]())
    var output_reg_tile_native = stack_allocation[
        accum_type, address_space=.LOCAL, alignment=p_frag_align
    ](row_major[num_m_mmas * num_n_mmas, p_frag_size]()).fill(0)
    # MMA and softmax helpers retain their legacy scalar views.
    var p_reg_tile = p_reg_tile_native.to_layout_tensor().as_unsafe_any_origin()
    var output_reg_tile = (
        output_reg_tile_native.to_layout_tensor().as_unsafe_any_origin()
    )

    # Rowwise max and sum for online softmax
    comptime row_alignment = align_of[
        SIMD[accum_type, simd_width_of[accum_type]()]
    ]()
    var rowmax = unsafe_stack_allocation[
        WM, accum_type, alignment=row_alignment
    ]()
    var rowsum = unsafe_stack_allocation[
        WM, accum_type, alignment=row_alignment
    ]()

    comptime for i in range(0, WM, p_frag_simdwidth):
        comptime if sink:
            assert Bool(
                sink_weights
            ), "expect sink_weights to be non-null when sink=true"
            var sink_logit_log2 = (
                sink_weights.value()[Int(head_idx)].cast[accum_type]() * log2e
            )
            rowmax.store(
                i,
                SIMD[accum_type, p_frag_simdwidth](sink_logit_log2),
            )
            # exp(sink_val-sink_val) = exp(0) = 1
            rowsum.store(i, SIMD[accum_type, p_frag_simdwidth](1))
        else:
            rowmax.store(
                i,
                SIMD[accum_type, p_frag_simdwidth](
                    min_or_neg_inf[accum_type]()
                ),
            )
            rowsum.store(i, SIMD[accum_type, p_frag_simdwidth](0))

    # Shared memory for P = Q * K^t
    # Only use BN/BK tiles. Setting circular so that the prefetch in matmul
    # doesn't go OOB at the last tile.
    var p_smem = (k_smem + k_smem_size).bitcast[Scalar[v_type]]()
    comptime IteratorTypeP = LayoutTensorIter[
        v_type,
        Layout.row_major(BM, BK),
        _,
        address_space=.SHARED,
        circular=True,
    ]
    var p_smem_iter = IteratorTypeP(
        p_smem, IteratorTypeP.layout_uint_type(BM * BN)
    )

    # Scratch shared memory for reduction across warps.
    var warp_scratch = TileTensor(
        (p_smem + (BM * BN if num_warps_n > 1 else 0)).bitcast[
            Scalar[accum_type]
        ](),
        row_major[p_frag_simdwidth * num_warps_n, BM](),
    )

    # Mask global memory iterator.
    var mask_block_row = q_tile_idx * UInt32(BM)
    var mask_warp_row = warp_y * UInt32(WM)
    var mask_warp_col = warp_x * UInt32(WN)

    # Account for group query.
    comptime kv_num_heads = num_heads // group

    comptime num_pipeline_stages = config.num_pipeline_stages
    var is_first_iter = True

    # Iterate over KV, equivalent to the following with if hoisted out.
    #   ```
    #   for i in range(start, end, tile_size):
    #     if i + tile_size >= end:
    #       loop_over_kvcache[tile_size, False]
    #     else:
    #       loop_over_kvcache[tile_size, True]
    #   ```
    # Only the last iteration is doing boundary check.
    @inline(.always)
    def loop_over_kvcache[
        tile_size: Int, not_last_iter: Bool
    ](kv_tile_start_row: Int, end: Int) {
        mut mask_warp_col,
        mut is_first_iter,
        mut k_smem_iter,
        imm,
    }:
        if (
            mask.status(
                UInt32(batch_idx),
                Index[dtype=DType.uint32](
                    Int(q_tile_idx * UInt32(BM) + start_pos),
                    kv_tile_start_row,
                ),
                Index[dtype=DType.uint32](BM, BN),
            )
            == TileMaskStatus.FULL_MASK
        ):
            mask_warp_col += UInt32(BN)
            return

        comptime kv_gmem_layout = Layout(
            IntTuple(BN, depth),
            IntTuple(kv_num_heads * depth, 1),
        )
        var kv_tile_num_rows = min(tile_size, end - kv_tile_start_row)

        # kv cache gmem has to clip num rows as runtime layout
        var kv_runtime_layout = RuntimeLayout[
            element_type=.int32, linear_idx_type=.int32
        ](
            RuntimeTuple[kv_gmem_layout.shape, element_type=.int32](
                kv_tile_num_rows, depth
            ),
            RuntimeTuple[kv_gmem_layout.stride, element_type=.int32](
                kv_num_heads * depth, 1
            ),
        )

        var k_gmem_block = LayoutTensor[
            k_type,
            kv_gmem_layout,
            layout_int_type=.int32,
            linear_idx_type=.int32,
            masked=not not_last_iter,
        ](
            k.block_paged_ptr[BN](
                UInt32(batch_idx),
                UInt32(kv_tile_start_row),
                UInt32(Int(head_idx // UInt32(group))),
                0,
            ),
            kv_runtime_layout,
        )
        var k_gmem_iter = k_gmem_block.tiled_iterator[BN, BK, axis=1](0, 0)

        var v_gmem_block = LayoutTensor[
            v_type,
            kv_gmem_layout,
            layout_int_type=.int32,
            linear_idx_type=.int32,
            masked=not not_last_iter,
        ](
            v.block_paged_ptr[BN](
                UInt32(batch_idx),
                UInt32(kv_tile_start_row),
                UInt32(Int(head_idx // UInt32(group))),
                0,
            ),
            kv_runtime_layout,
        )
        var v_gmem_iter = v_gmem_block.tiled_iterator[BK, BN, axis=0](0, 0)

        # P = Q @ K, register tile holding mma result.
        _ = p_reg_tile.fill(0)

        var num_b_rows = Optional[Int]() if not_last_iter else Optional[Int](
            kv_tile_num_rows
        )

        # First iteration load q from global memory to shared memory.
        if is_first_iter:
            multistage_mma[
                BM,
                BN,
                BK,
                WM,
                WN,
                num_threads,
                num_pipeline_stages,
                True,  # transpose_b
                swizzle_a=True,
                continue_prefetch_b=True,
                b_next_smem_layout=Layout.row_major(BK, BN),
                next_op_b_iter_masked=type_of(v_gmem_iter).masked,
                next_op_b_layout_int_type=type_of(v_gmem_iter).layout_int_type,
                next_op_b_linear_idx_type=type_of(v_gmem_iter).linear_idx_type,
                k_group_size=config.k_group_size,
            ](
                lt_to_tt_idx[linear_idx_type=p_reg_tile.linear_idx_type](
                    p_reg_tile
                ),
                q_gmem_iter,
                k_gmem_iter,
                q_smem_iter,
                k_smem_iter,
                ufloordiv(depth, BK),
                next_op_b_iter=v_gmem_iter.bitcast[k_type](),
                num_b_rows=num_b_rows,
            )
            is_first_iter = False
        # Subsequent iterations just use q in share memory.
        # TODO: Figure out a better function interface instead of passing in
        # shared memory iterator twice.
        else:
            multistage_mma[
                BM,
                BN,
                BK,
                WM,
                WN,
                num_threads,
                num_pipeline_stages,
                True,  # transpose_b
                swizzle_a=True,
                continue_prefetch_b=True,
                b_next_smem_layout=Layout.row_major(BK, BN),
                next_op_b_iter_masked=type_of(v_gmem_iter).masked,
                next_op_b_layout_int_type=type_of(v_gmem_iter).layout_int_type,
                next_op_b_linear_idx_type=type_of(v_gmem_iter).linear_idx_type,
                k_group_size=config.k_group_size,
            ](
                lt_to_tt_idx[linear_idx_type=p_reg_tile.linear_idx_type](
                    p_reg_tile
                ),
                # Pass shared memory iterator to hint not loading from global memory.
                q_smem_iter,
                k_gmem_iter,
                q_smem_iter,
                k_smem_iter,
                ufloordiv(depth, BK),
                next_op_b_iter=v_gmem_iter.bitcast[k_type](),
                num_b_rows=num_b_rows,
            )

        # Increment V iterator since it's prefetched inside 1st matmul.
        v_gmem_iter += num_pipeline_stages - 1

        # Vectorize by 2.
        var p_reg_vec2 = p_reg_tile.vectorize[1, p_frag_simdwidth]()

        def _apply_mask[masked: Bool]() {imm}:
            var scale_log2e: Scalar[accum_type] = (
                scale.cast[
                    accum_type
                ]() if mask_t.apply_log2e_after_mask else scale.cast[
                    accum_type
                ]()
                * log2e
            )

            comptime for m_mma in range(num_m_mmas):
                comptime for n_mma in range(num_n_mmas):
                    comptime mma_id = n_mma * num_m_mmas + m_mma

                    # Coordinates in mask for current mma tile.
                    var mask_frag_row = mask_warp_row + UInt32(m_mma) * UInt32(
                        MMA_M
                    )

                    var mask_frag_col = mask_warp_col + UInt32(n_mma) * UInt32(
                        MMA_N
                    )

                    mask_frag_row += lane // UInt32(MMA_N // p_frag_simdwidth)
                    mask_frag_col += (
                        lane * UInt32(p_frag_simdwidth) % UInt32(MMA_N)
                    )

                    comptime for i in range(2):
                        # The row in score matrix of shape seq_len x num_keys.
                        # Mask col is score col since we don't partition in col.
                        var score_row = (
                            mask_block_row
                            + mask_frag_row
                            + UInt32(i * MMA_M // 2)
                        )
                        var score_col = mask_frag_col

                        var score_row_with_start_pos = score_row + start_pos

                        comptime if masked:
                            p_reg_vec2[mma_id, i] = mask.mask(
                                IndexList[4, element_type=.uint32](
                                    block_idx.z,
                                    block_idx.y,
                                    Int(score_row_with_start_pos),
                                    Int(score_col),
                                ),
                                p_reg_vec2[mma_id, i] * scale_log2e,
                            )

                        else:
                            p_reg_vec2[mma_id, i] = (
                                p_reg_vec2[mma_id, i] * scale_log2e
                            )

                        comptime if mask_t.apply_log2e_after_mask:
                            p_reg_vec2[mma_id, i] = (
                                p_reg_vec2[mma_id, i] * log2e
                            )

                        comptime if not not_last_iter:
                            p_reg_vec2[mma_id, i] = _kernel_mask(
                                IndexList[
                                    2,
                                    element_type=.uint32,
                                ](Int(score_row), Int(score_col)),
                                IndexList[
                                    2,
                                    element_type=.uint32,
                                ](seq_len, num_keys),
                                p_reg_vec2[mma_id, i],
                            )

        unswitch(
            mask.status(
                UInt32(batch_idx),
                Index[dtype=DType.uint32](
                    Int(q_tile_idx * UInt32(BM) + start_pos),
                    kv_tile_start_row,
                ),
                Index[dtype=DType.uint32](BM, BN),
            )
            == TileMaskStatus.PARTIAL_MASK,
            _apply_mask,
        )

        # Increment mask to next BM x BN block.
        mask_warp_col += UInt32(BN)

        comptime reg_layout_by_mma_unit = Layout.row_major(
            2 * num_m_mmas * num_n_mmas, 2
        )

        _online_softmax_iter_for_mma_output[
            accum_type,
            # score layout by mma unit
            # TODO: generalize beyond 16x8 layout
            Layout.row_major(2 * num_m_mmas, num_n_mmas),
            # threads layout by warp
            Layout.row_major(num_warps_m, num_warps_n),
            Layout.row_major(8, 4),
            use_exp2=True,
        ](
            lt_to_tt_idx[linear_idx_type=output_reg_tile.linear_idx_type](
                output_reg_tile.reshape[reg_layout_by_mma_unit]()
            ).vectorize[1, p_frag_simdwidth](),
            lt_to_tt_idx[linear_idx_type=p_reg_tile.linear_idx_type](
                p_reg_tile.reshape[reg_layout_by_mma_unit]()
            ).vectorize[1, p_frag_simdwidth](),
            warp_scratch.tile[2 * num_warps_n, WM](Coord(0, Int(warp_y))),
            rowmax,
            rowsum,
        )

        # V reuse K's smem iterator. They has same smem footage expect for different layouts.
        var v_smem_iter = k_smem_iter.reshape[
            Layout.row_major(BK, BN)
        ]().bitcast[v_type]()

        comptime if num_warps_n > 1:
            # Pack the per-thread fragments in shared memory for 2nd mma.
            _copy_frag_to_smem[
                BM,
                BN,
                BK,
                WM,
                WN,
                MMA_M,
                MMA_N,
                p_frag_simdwidth,
            ](p_smem_iter, p_reg_tile, warp_x, warp_y)
            barrier()

            multistage_mma[
                BM,
                BN,
                BK,
                WM,
                WN,
                num_threads,
                num_pipeline_stages,
                False,  # transpose_b
                swizzle_a=True,
                prefetch_init=False,
                k_group_size=config.k_group_size,
            ](
                lt_to_tt_idx[linear_idx_type=output_reg_tile.linear_idx_type](
                    output_reg_tile
                ),
                p_smem_iter,
                v_gmem_iter,
                p_smem_iter,
                v_smem_iter,
                ufloordiv(BN, BK),
                num_b_rows=num_b_rows,
            )
        else:
            # Reuse 1st mma output (MMA_M, MMA_N) as 2nd mma's input (MMA_M, MMA_K).
            # The num_n_mmas dim becomes "num_k_mmas" for 2nd mma.
            var p_reg_iter = p_reg_tile.tiled_iterator[
                MMA_K // MMA_N * num_m_mmas, p_frag_size
            ](0, 0)

            multistage_mma[
                BM,
                BN,
                BK,
                WM,
                WN,
                num_threads,
                num_pipeline_stages,
                False,  # transpose_b
                swizzle_a=True,
                static_num_iters=ufloordiv(BN, BK),
                prefetch_init=False,
                k_group_size=config.k_group_size,
            ](
                lt_to_tt_idx[linear_idx_type=output_reg_tile.linear_idx_type](
                    output_reg_tile
                ),
                p_reg_iter,
                v_gmem_iter,
                p_smem_iter,
                v_smem_iter,
                ufloordiv(BN, BK),
                num_b_rows=num_b_rows,
            )

    tile_and_unswitch[[BN]](0, num_keys, loop_over_kvcache)

    comptime for m_mma in range(num_m_mmas):
        var rowsum_inv0 = recip(rowsum[2 * m_mma])
        var rowsum_inv1 = recip(rowsum[2 * m_mma + 1])

        comptime for n_mma in range(num_n_mmas):
            comptime for i in range(p_frag_size // 2):
                output_reg_tile[n_mma * num_m_mmas + m_mma, i] *= rowsum_inv0
                output_reg_tile[
                    n_mma * num_m_mmas + m_mma, i + p_frag_size // 2
                ] *= rowsum_inv1

    comptime output_gmem_layout = Layout(
        IntTuple(BM, depth), IntTuple(num_heads * depth, 1)
    )
    var output_gmem_tile = LayoutTensor[
        output_type,
        output_gmem_layout,
        layout_int_type=.int32,
        linear_idx_type=.int32,
        masked=True,
    ](
        output_ptr + Int(q_offset),
        RuntimeLayout[element_type=.int32, linear_idx_type=.int32](
            RuntimeTuple[output_gmem_layout.shape, element_type=.int32](
                Int(q_tile_num_rows), depth
            ),
            RuntimeTuple[output_gmem_layout.stride, element_type=.int32](
                num_heads * depth, 1
            ),
        ),
    )
    var output_gmem_warp_tile = output_gmem_tile.tile[WM, WN](
        Int(warp_y), Int(warp_x)
    )

    # Write to global memory.
    comptime if output_type.is_half_float():
        # Reuse a_smem for c tile in smem
        var accum_smem_tile = LayoutTensor[
            output_type,
            Layout.row_major(BM, depth),
            address_space=.SHARED,
        ](q_smem.bitcast[Scalar[output_type]]())

        var accum_smem_warp_tile = accum_smem_tile.tile[WM, WN](
            Int(warp_y), Int(warp_x)
        )

        comptime swizzle = make_swizzle[
            num_rows=MMA_M // 2, row_size=WN, access_size=MMA_N
        ]()
        copy_local_to_shared[
            thread_layout=Layout.row_major(8, 4), swizzle=swizzle
        ](
            accum_smem_warp_tile.vectorize[1, 2](),
            output_reg_tile.vectorize[1, 2]().transpose(),
        )
        barrier()
        copy_sram_to_dram[
            thread_layout=Layout.row_major(
                num_threads * simd_size // depth,
                depth // simd_size,
            ),
            swizzle=swizzle,
        ](
            output_gmem_tile.vectorize[1, simd_size](),
            accum_smem_tile.vectorize[1, simd_size](),
        )

        # Guard writing to shared memory.

        barrier()

        # Vectorized copy from shared to global memory, during which every 2 FP32
        # are cast to 2 BF16 so that 2 4xFP32 vectors are merged into 1 8xBF16
        # vector and stored using 16B store instruction.

    else:
        copy_local_to_dram[dst_thread_layout=Layout.row_major(8, 4)](
            output_gmem_warp_tile.vectorize[1, 2](),
            output_reg_tile.vectorize[1, 2]().transpose(),
        )


# ===-----------------------------------------------------------------------===#
# Flash decoding for token generation
# ===-----------------------------------------------------------------------===#


# Entry point for mha_decoding with batch_size > 1.
# A CTA's warps are dealt round-robin over the CU's 4 SIMDs, and a depth-128
# decode CTA's ~80 KB of LDS holds the CU to 2 CTAs -- too few to even out a
# short SIMD -- so 2 waves/EU needs a warp count that is a multiple of 4.
# Hinting it on a 2- or 3-warp token fold makes the register allocator target an
# occupancy it cannot reach: 7-27% slower. AMD single-token decode is 4 warps, so
# its emitted amdgcn is unchanged; depth 64 fits 4 CTAs, where a 2-warp CTA could
# reach 2 waves/EU and is merely hinted conservatively. NVIDIA sets BN == depth,
# so its depth-64 decode is 2 warps and lands here too -- inert, since
# `rocdl.waves_per_eu` is AMDGPU-only metadata.
@__llvm_metadata(
    `rocdl.waves_per_eu`=SIMDLength(
        get_waves_per_eu(depth) if (num_threads // WARP_SIZE) % 4 == 0 else 1
    )
)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__name(
    t"mha_decoding_depth{depth}_{q_type}_{output_type}_{BM}x{BN}x{BK}_{ragged}_nqh{num_heads}_nkvh{num_heads // group}",
)
def mha_decoding[
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    BM: Int,  # number of queries per block
    BN: Int,  # number of keys per block
    BK: Int,  # tile size in depth dimension
    WM: Int,
    WN: Int,
    depth: Int,
    num_heads: Int,
    num_threads: Int,
    num_pipeline_stages: Int,
    group: Int = 1,
    ragged: Bool = False,
    is_shared_kv: Bool = False,
    sink: Bool = False,
    _use_valid_length: Bool = False,
    _is_cache_length_accurate: Bool = False,
    decoding_warp_split_k: Bool = False,
    q_seq_len: Int = 1,
](
    q_ptr: UnsafePointer[Scalar[q_type], ImmutAnyOrigin],
    k: k_t,
    v: v_t,
    output_ptr: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    exp_sum_ptr: UnsafePointer[Scalar[get_accum_type[q_type]()], MutAnyOrigin],
    qk_max_ptr: UnsafePointer[Scalar[get_accum_type[q_type]()], MutAnyOrigin],
    scale: Float32,
    batch_size: Int32,
    num_partitions: Int32,
    valid_length: ImmutTileTensor1D[.uint32],  # valid length per batch
    sink_weights: OptionalReg[ImmutTileTensor1D[q_type]],
    mask: mask_t,
):
    """Flash-attention decode GPU kernel with optional split-K partitioning.

    Each CTA processes one split-K partition for one `(batch, head)` pair.
    Computes online softmax over its key slice and writes partial
    `exp_sum` and `qk_max` statistics alongside the partial output so the
    `mha_splitk_reduce` kernel can merge them. When `num_partitions == 1`
    the output is final and no reduction is needed.

    Parameters:
        q_type: Element type of the query tensor.
        k_t: Key operand type (dense or KV-cache).
        v_t: Value operand type (dense or KV-cache).
        output_type: Element type of the output and partial output buffer.
        mask_t: Attention mask type.
        BM: Query tile height (rows per CTA).
        BN: Key tile width (columns per CTA).
        BK: Tile size along the head-depth dimension.
        WM: Warp tile height.
        WN: Warp tile width.
        depth: Attention head depth.
        num_heads: Number of query heads.
        num_threads: Total threads per CTA.
        num_pipeline_stages: Number of software pipeline stages for KV loads.
        group: GQA group size (query heads per KV head).
        ragged: `True` for ragged-batch inputs.
        is_shared_kv: `True` when K and V share an SMEM buffer.
        sink: `True` to enable attention-sink mode.
        _use_valid_length: `True` to read per-sequence valid lengths.
        _is_cache_length_accurate: `True` when cache length is exact.
        decoding_warp_split_k: Enable warp-level split-K within a CTA.
        q_seq_len: Query tokens per sequence folded into the MMA M dimension;
            1 is plain decode. A property of the TENSOR, not the geometry —
            the fold may build a taller tile (`mha_decode_fold_tile_q_seq_len`).
            The non-ragged arms take every Q/output/split-K stride from it;
            ragged recovers the true length from `input_row_offsets`.

    Args:
        q_ptr: Pointer to query data.
        k: Key operand.
        v: Value operand.
        output_ptr: Pointer to the partial/final output buffer.
        exp_sum_ptr: Pointer to the partial exponential-sum buffer.
        qk_max_ptr: Pointer to the partial softmax-maximum buffer.
        scale: Softmax temperature scale.
        batch_size: Number of sequences in the batch.
        num_partitions: Number of split-K partitions.
        valid_length: Per-sequence valid lengths (or row offsets for ragged).
        sink_weights: Sink-token weights for attention-sink mode.
        mask: Mask instance.
    """

    var _batch_size = Int(batch_size)
    var _num_partitions = Int(num_partitions)
    comptime accum_type = get_accum_type[q_type]()
    var batch_idx = block_idx.z

    comptime assert not (q_seq_len > 1 and decoding_warp_split_k), (
        "the q_seq_len > 1 token fold leaves no cross-warp N split for"
        " decoding_warp_split_k to divide"
    )
    # `mha_decode_streaming` carries no fold geometry (no per-warp V depth split,
    # no fold-row handling), so it would silently compute only token 0.
    comptime assert not (
        q_seq_len > 1 and get_defined_bool["MHA_STREAMING_DECODE", False]()
    ), "MHA_STREAMING_DECODE does not implement the q_seq_len > 1 token fold"
    comptime assert not (
        q_seq_len > 1 and is_nvidia_gpu()
    ), "the q_seq_len > 1 decode token fold is AMD-only"
    # Token SLOTS the tile is built from. DERIVED rather than passed: the host
    # sizes `BM`/`WM`/`BK` from this same function, so launch and kernel cannot
    # disagree. Sizes geometry ONLY — every stride below uses `q_seq_len`.
    comptime tile_q_seq_len = mha_decode_fold_tile_q_seq_len[
        q_type, num_heads, group, q_seq_len
    ]()
    # `Attention.mask_status` re-derives the fold width from the TILE and uses
    # it as the decode span base `num_keys - fold_seq_len`, which a padded tile
    # overstates. `_mha_decode_fold_ok` excludes such masks today; this keeps
    # lifting that exclusion a build break rather than a wrong span.
    comptime assert not (
        tile_q_seq_len != q_seq_len and mask_t.check_mask_during_decoding
    ), (
        "a padded fold tile and check_mask_during_decoding are mutually"
        " exclusive: mask_status would span the tile's slots, not the tokens"
    )
    # `_mha_decode_fold_ok` checks that ceiling against the width a sequence
    # CARRIES, and the pad raises the tile above it — so the height that really
    # launches gets its own check, read off `BM` rather than re-derived.
    comptime assert (
        not (q_seq_len > 1 and group == num_heads)
        or BM <= _MHA_DECODE_FOLD_MAX_ROWS
    ), "the padded fold tile exceeds the fold's query-row ceiling"

    var seq_len: Int
    var q_batch_offset: Int
    # A sequence contributes `num_heads * seq_len` query rows, so the split-K
    # workspaces are keyed on query ROWS: `row_base` is this sequence's first row,
    # `row_stride` the batch's total row count (the stride between per-partition
    # planes). Both collapse to `batch_idx` / `batch_size` at q_seq_len == 1.
    var row_base: Int
    var row_stride: Int
    var start_pos: UInt32 = 0

    comptime if ragged:
        # treat valid_lengths as a input_row_offsets
        var start_of_seq = Int(valid_length[batch_idx])
        var end_of_seq = Int(valid_length[batch_idx + 1])
        seq_len = end_of_seq - start_of_seq
        q_batch_offset = start_of_seq * depth * num_heads
        # Rows are already packed by `input_row_offsets`, so this sequence's row
        # base IS its row offset and total_q is the last entry of the same array —
        # no extra kernel argument needed. This is what makes a non-uniform batch
        # correct: each sequence writes exactly the rows it owns.
        row_base = start_of_seq
        row_stride = Int(valid_length[_batch_size])
    elif _use_valid_length:
        # treat valid_lengths as valid lengths
        q_batch_offset = depth * num_heads * q_seq_len * batch_idx
        seq_len = Int(valid_length[batch_idx])
        row_base = q_seq_len * batch_idx
        row_stride = q_seq_len * _batch_size
    else:
        seq_len = q_seq_len
        q_batch_offset = depth * num_heads * q_seq_len * batch_idx
        row_base = q_seq_len * batch_idx
        row_stride = q_seq_len * _batch_size

    # split-k offsets
    var partition_idx = block_idx.x
    var output_batch_offset = (
        depth * num_heads * row_base
        + depth * num_heads * row_stride * partition_idx
    )
    var qk_max_offset = (
        num_heads * row_base + num_heads * row_stride * partition_idx
    )
    var exp_sum_offset = qk_max_offset

    # split-k intermediate buffers — only used when _num_partitions > 1
    var qk_max_batch_ptr = qk_max_ptr
    if _num_partitions > 1:
        qk_max_batch_ptr = qk_max_ptr + qk_max_offset

    var exp_sum_batch_ptr = exp_sum_ptr
    if _num_partitions > 1:
        exp_sum_batch_ptr = exp_sum_ptr + exp_sum_offset

    var num_keys = k.cache_length(batch_idx)

    comptime if not _is_cache_length_accurate:
        num_keys += seq_len

    comptime if is_nvidia_gpu():
        comptime if is_shared_kv:
            mha_decoding_single_batch_pipelined[
                BM=BM,
                BN=BN,
                BK=BK,
                WM=WM,
                WN=WN,
                depth=depth,
                num_heads=num_heads,
                num_threads=num_threads,
                num_pipeline_stages=num_pipeline_stages,
                group=group,
                decoding_warp_split_k=decoding_warp_split_k,
                sink=sink,
            ](
                q_ptr + q_batch_offset,
                k,
                v,
                output_ptr + output_batch_offset,
                exp_sum_batch_ptr,
                qk_max_batch_ptr,
                scale,
                num_keys,
                _num_partitions,
                sink_weights,
                mask,
                batch_idx,
            )
        else:
            mha_decoding_single_batch[
                BM=BM,
                BN=BN,
                BK=BK,
                WM=WM,
                WN=WN,
                depth=depth,
                num_heads=num_heads,
                num_threads=num_threads,
                num_pipeline_stages=num_pipeline_stages,
                group=group,
                decoding_warp_split_k=decoding_warp_split_k,
                sink=sink,
            ](
                q_ptr + q_batch_offset,
                k,
                v,
                output_ptr + output_batch_offset,
                exp_sum_batch_ptr,
                qk_max_batch_ptr,
                scale,
                num_keys,
                _num_partitions,
                mask,
                batch_idx,
                sink_weights,
            )
    elif is_amd_gpu():
        comptime config = MHAConfig[q_type](
            num_heads,
            depth,
            num_queries_per_block=BM,
            num_keys_per_block=BN,
            BK=BK,
            WM=WM,
            WN=WN,
            num_pipeline_stages=num_pipeline_stages,
            k_group_size=group,
        )

        comptime use_streaming_decode = get_defined_bool[
            "MHA_STREAMING_DECODE", False
        ]()

        var sink_weights_ptr = OptionalReg[
            UnsafePointer[Scalar[q_type], ImmutAnyOrigin]
        ]()
        comptime if sink:
            sink_weights_ptr = sink_weights.value().ptr

        comptime if _is_amd_rdna():
            # MHA_STREAMING_DECODE is CDNA-only and is intentionally
            # ignored on RDNA, which has only `mha_decode`.
            # `AttentionRDNA` has no `q_seq_len`, so it would compute token 0 and
            # leave the rest of the output stale.
            comptime assert (
                q_seq_len == 1
            ), "RDNA MHA decode does not implement the query-token fold"
            var attention = AttentionRDNA[config, group, sink, token_gen=True](
                output_ptr + output_batch_offset,
                q_ptr + q_batch_offset,
                k,
                v,
                mask,
                sink_weights_ptr,
                batch_idx,
                scale,
                1,
                num_keys,
                0,
            )
            attention.mha_decode(
                exp_sum_batch_ptr,
                qk_max_batch_ptr,
                _num_partitions,
            )
        else:
            # The fold needs the runtime query length (it drives the per-token
            # causal position and the live-row clamp). Plain decode keeps passing
            # the literal 1: on the `_use_valid_length` path `seq_len` can differ
            # from 1, and there it feeds only `num_keys`.
            var attn_seq_len = 1
            comptime if q_seq_len > 1:
                attn_seq_len = seq_len

            var attention = Attention[
                config, group, sink, token_gen=True, q_seq_len=tile_q_seq_len
            ](
                output_ptr + output_batch_offset,
                q_ptr + q_batch_offset,
                k,
                v,
                mask,
                sink_weights_ptr,
                batch_idx,
                scale,
                attn_seq_len,
                num_keys,
                0,
            )
            comptime if use_streaming_decode:
                attention.mha_decode_streaming(
                    exp_sum_batch_ptr,
                    qk_max_batch_ptr,
                    _num_partitions,
                )
            else:
                attention.mha_decode(
                    exp_sum_batch_ptr,
                    qk_max_batch_ptr,
                    _num_partitions,
                )
    else:
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name()
        ]()


@inline(.always)
def scale_and_mask_helper[
    p_type: DType,
    p_layout: TensorLayout,
    mask_t: MHAMask,
    group: Int,
    num_n_mmas: Int,
    WN: Int,
    MMA_N: Int,
    simd_width: Int,
](
    p_reg_tile: TileTensor[
        mut=True, p_type, p_layout, address_space=.LOCAL, ...
    ],
    scale_log2e: Float32,
    num_keys: Int,
    bound: Int,
    lane: Int,
    warp: Int,
    mask: mask_t,
    kv_tile_start_row: Int,
):
    """Apply softmax scaling and masking to one P = Q·Kᵀ MMA result tile in registers.

    Multiplies each element by `scale_log2e` and then applies `mask` to
    out-of-bounds and masked positions. Only threads with `lane < 4 * group`
    carry meaningful data; other threads return immediately. Designed for the
    decode inner loop where P is a 1-D column-vector across the key dimension.
    """
    # The masking body still indexes through the legacy scalar view.
    var p_reg_tile_legacy = p_reg_tile.to_layout_tensor()

    # Apply mask and scale to mma result. Only the first row (lane 0-3) has
    # meaningful data, other fragments are zero. The mask is an 1D vector.
    # The dimension of mask are assumed dynamic here so still using index calculation.
    # TODO: check if the explicit index calculation can be avoided.

    # For mma output, thread 0-3 are on the first row, 4-7 second row, etc.
    if lane >= 4 * group:
        return
    var batch_cache_valid_length = num_keys - 1
    var warp_offset = warp * WN

    # Number of groups updated by each thread. E.g. for group=16 and 16x8x16 mma,
    # Each thread updates 2 rows in mma output, mapped to 2 groups.
    # When group % 8 != 0, some work are OOB, e.g. updating 15-th row when there are
    # only 12 groups. Such results are ignored when output to global memory.
    comptime num_groups_per_thread = ceildiv(group, 8)

    comptime for n_mma in range(num_n_mmas):
        # offset in fragment
        var frag_offset = n_mma * MMA_N
        # Current thread's offset mapped in num_keys dim
        var key_offset = warp_offset + frag_offset
        # Current thread's index in current mma tile, e.g. T1 and T5 are 1 in 16x8 mma output.
        var frag_lane_col = umod(lane, 4) * simd_width

        comptime for i_group in range(num_groups_per_thread):
            var group_idx = i_group * 8 + ufloordiv(lane, 4)
            var q_head_idx = block_idx.y * group + group_idx

            comptime for i in range(simd_width):
                var score_row = batch_cache_valid_length
                var score_col = (
                    kv_tile_start_row + key_offset + frag_lane_col + i
                )

                p_reg_tile_legacy[n_mma, i + i_group * simd_width] = mask.mask(
                    Index(
                        block_idx.z,
                        q_head_idx,
                        score_row,
                        score_col,
                    ),
                    p_reg_tile_legacy[n_mma, i + i_group * simd_width]
                    * scale_log2e.cast[p_type](),
                )

                comptime if mask_t.apply_log2e_after_mask:
                    p_reg_tile_legacy[n_mma, i + i_group * simd_width] = (
                        p_reg_tile_legacy[n_mma, i + i_group * simd_width]
                        * log2e
                    )

                p_reg_tile_legacy[
                    n_mma, i + i_group * simd_width
                ] = _kernel_mask(
                    Index(score_row, score_col),
                    Index(
                        batch_cache_valid_length + 1,
                        # The following setting ensures that out of bound check happens at
                        # every function call, also it corrects the bounds to be exact.
                        # Previous version was using batch_cache_valid_length + 1 which was fine
                        # with the non-split-k based mha as the ooo would have been triggered only
                        # for the last iteration of the outer loop. So while the bound was not exact, it
                        # led to correct output.
                        kv_tile_start_row + bound,
                    ),
                    p_reg_tile_legacy[n_mma, i + i_group * simd_width],
                )


def mha_decoding_single_batch[
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    *,
    BM: Int,  # number of queries per block
    BN: Int,  # number of keys per block
    BK: Int,  # tile size in depth dimension
    WM: Int,
    WN: Int,
    depth: Int,
    num_heads: Int,
    num_threads: Int,
    num_pipeline_stages: Int,
    group: Int = 1,
    decoding_warp_split_k: Bool = False,
    sink: Bool = False,
](
    q_ptr: UnsafePointer[Scalar[q_type], ImmutAnyOrigin],
    k: k_t,
    v: v_t,
    output_ptr: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    exp_sum_ptr: UnsafePointer[Scalar[get_accum_type[q_type]()], MutAnyOrigin],
    qk_max_ptr: UnsafePointer[Scalar[get_accum_type[q_type]()], MutAnyOrigin],
    scale: Float32,
    num_keys: Int,
    num_partitions: Int,
    mask: mask_t,
    batch_idx: Int,
    sink_weights: OptionalReg[ImmutTileTensor1D[q_type]],
):
    """Flash attention v2 algorithm.

    Parameters:
        q_type: Element type of the query tensor.
        k_t: Key operand type (KV cache or dense tensor).
        v_t: Value operand type (KV cache or dense tensor).
        output_type: Element type of the output tensor.
        mask_t: Attention mask type implementing `MHAMask`.
        BM: Number of query rows per thread block.
        BN: Number of key columns per thread block.
        BK: Tile size in the depth dimension for shared-memory tiles.
        WM: Warp tile height in the query (M) dimension.
        WN: Warp tile width in the key (N) dimension.
        depth: Attention head depth (key/value dimension per head).
        num_heads: Total number of query heads.
        num_threads: Number of threads per thread block.
        num_pipeline_stages: Number of software-pipeline stages for async
            copies.
        group: GQA group size, query heads per key/value head (defaults
            to 1).
        decoding_warp_split_k: Enable warp-level split-K reduction
            (defaults to `False`).
        sink: Enable attention-sink mode where the first tokens always
            attend (defaults to `False`).

    Args:
        q_ptr: Pointer to the query tensor in global memory.
        k: Key operand backed by a KV cache or dense tensor.
        v: Value operand backed by a KV cache or dense tensor.
        output_ptr: Pointer to the output tensor in global memory.
        exp_sum_ptr: Pointer to the per-head online-softmax denominator
            (sum of exponentials) for cross-partition reduction.
        qk_max_ptr: Pointer to the per-head online-softmax running
            maximum for cross-partition reduction.
        scale: Softmax temperature scale applied to Q·Kᵀ.
        num_keys: Number of valid key/value entries (cache length).
        num_partitions: Number of split-K partitions along the key
            dimension.
        mask: Mask instance used to apply the attention mask.
        batch_idx: Index of the sequence within the batch.
        sink_weights: Optional sink-token weight tensor for attention
            sinks.
    """
    comptime accum_type = get_accum_type[q_type]()
    comptime k_type = k_t.dtype
    comptime v_type = v_t.dtype
    comptime assert q_type == k_type and k_type == v_type

    comptime simd_size = simd_width_of[q_type]()

    comptime num_warps_m = BM // WM
    comptime num_warps_n = BN // WN

    comptime assert num_warps_m * num_warps_n == (
        num_threads // WARP_SIZE
    ), "Number of warps doesn't match warp tile sizes."

    # It's because in online-softmax we only use the top 8x4 sub-matrix
    # in the 16x8 mma output for Nvidia GPU. It shouldn't matter for AMD
    comptime assert group <= 16, String(
        "Only support GQA with group <= 16 for Nvidia, but got a group = '",
        group,
        "'.",
    )

    var warp_id = warp_id[broadcast=True]()
    var lane = lane_id()

    # Coordinates of the current warp.
    var warp_y, warp_x = udivmod(warp_id, num_warps_n)

    # The entire query block (BM x depth) is tiled in shared memory.
    comptime alignment = align_of[SIMD[q_type, simd_size]]()
    comptime q_smem_size = BM * depth
    var q_smem = external_memory[
        Scalar[q_type],
        address_space=.SHARED,
        alignment=alignment,
    ]()
    comptime IteratorTypeQ = LayoutTensorIter[
        q_type,
        Layout.row_major(BM, BK),
        _,
        address_space=.SHARED,
        alignment=alignment,
    ]
    var q_smem_iter = IteratorTypeQ(
        rebind[
            type_of(
                LayoutTensorIter[
                    q_type,
                    Layout.row_major(BM, BK),
                    q_smem.origin,
                    address_space=.SHARED,
                    alignment=alignment,
                ]().ptr
            )
        ](q_smem),
        IteratorTypeQ.layout_uint_type(q_smem_size),
    )

    comptime k_smem_size = BN * depth
    var k_smem = (q_smem + q_smem_size).bitcast[Scalar[k_type]]()
    comptime IteratorTypeK = LayoutTensorIter[
        k_type,
        Layout.row_major(BN, BK),
        _,
        address_space=.SHARED,
        circular=True,
    ]
    var k_smem_iter = IteratorTypeK(
        k_smem, IteratorTypeK.layout_uint_type(k_smem_size)
    )

    comptime v_smem_size = BN * BN
    var v_smem = (k_smem + k_smem_size).bitcast[Scalar[v_type]]()
    comptime IteratorTypeV = LayoutTensorIter[
        v_type,
        Layout.row_major(BK, BN),
        _,
        address_space=.SHARED,
        circular=True,
    ]
    var v_smem_iter = IteratorTypeV(
        v_smem, IteratorTypeV.layout_uint_type(v_smem_size)
    )

    var kv_head_idx = block_idx.y
    var q_head_idx = kv_head_idx * group + ufloordiv(thread_idx.x, 4)
    var partition_idx = block_idx.x

    comptime mma_shape = get_mma_shape[q_type, accum_type]()
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]
    comptime num_m_mmas = WM // MMA_M
    comptime num_n_mmas = WN // MMA_N

    comptime frag_size = get_fragment_size[mma_shape]()
    comptime p_frag_size = frag_size[2]
    comptime p_frag_simdwidth = p_frag_size // 2
    comptime p_frag_align = align_of[SIMD[accum_type, p_frag_size]]()

    var p_reg_tile = stack_allocation[
        accum_type, address_space=.LOCAL, alignment=p_frag_align
    ](row_major[num_m_mmas * num_n_mmas, p_frag_size]())

    # Note that
    # num_warps_n * num_n_mmas == BN // WN * num_n_mmas
    # so we can use multistage_mma
    comptime num_output_rows = num_m_mmas * num_n_mmas
    comptime num_output_rows_full = num_warps_n * num_output_rows if decoding_warp_split_k else num_output_rows
    var output_reg_tile = stack_allocation[
        accum_type, address_space=.LOCAL, alignment=p_frag_align
    ](row_major[num_output_rows_full, p_frag_size]()).fill(0.0)

    # Rowwise max and sum for online softmax
    comptime row_align = align_of[
        SIMD[accum_type, simd_width_of[accum_type]()]
    ]()
    var rowmax = unsafe_stack_allocation[WM, accum_type, alignment=row_align]()
    var rowsum = unsafe_stack_allocation[WM, accum_type, alignment=row_align]()

    comptime for i in range(WM):
        comptime if sink:
            assert Bool(
                sink_weights
            ), "expect sink_weights to be non-null when sink=true"
            if thread_idx.x < 4 * group:
                var sink_logit_log2 = (
                    sink_weights.value()[q_head_idx].cast[accum_type]() * log2e
                )
                rowmax[i] = sink_logit_log2
                if partition_idx == 0 and umod(thread_idx.x, 4) == 0:
                    rowsum[i] = 1.0
                else:
                    rowsum[i] = 0.0
            else:
                rowmax[i] = min_or_neg_inf[accum_type]()
                rowsum[i] = 0.0
        else:
            rowmax[i] = min_or_neg_inf[accum_type]()
            rowsum[i] = 0.0

    # Shared memory for P = Q * K^t
    # This overlaps key tile but are used at the same time i.e. no race condition.
    var p_smem = (v_smem + v_smem_size).bitcast[Scalar[v_type]]()
    comptime IteratorTypeP = LayoutTensorIter[
        v_type,
        Layout.row_major(BM, BK),
        _,
        address_space=.SHARED,
    ]
    var p_smem_iter = IteratorTypeP(
        p_smem, IteratorTypeP.layout_uint_type(BM * BN)
    )

    # Scratch shared memory for reduction across warps.
    var warp_scratch = TileTensor(
        (p_smem + BM * BN).bitcast[Scalar[accum_type]](),
        row_major[2 * num_warps_n, BM](),
    )

    # Account for group query.
    comptime kv_num_heads = num_heads // group

    var q_offset = depth * kv_head_idx * group

    comptime q_gmem_layout = Layout.row_major(BM, depth)
    var q_gmem_block = LayoutTensor[
        q_type,
        q_gmem_layout,
        layout_int_type=.int32,
        linear_idx_type=.int32,
        masked=True,
    ](
        q_ptr + q_offset,
        RuntimeLayout[element_type=.int32, linear_idx_type=.int32](
            RuntimeTuple[q_gmem_layout.shape, element_type=.int32](
                group, depth
            ),
            RuntimeTuple[q_gmem_layout.stride, element_type=.int32](depth, 1),
        ),
    )
    var q_gmem_iter = q_gmem_block.tiled_iterator[BM, BK, axis=1](0, 0)

    var start, end = get_start_and_end_for_partitions[BN](
        num_keys, num_partitions, block_idx.x
    )

    comptime q_num_vecs = BM * BK // simd_size

    comptime async_copy_q_layout = Layout.row_major(
        min(num_threads, q_num_vecs) * simd_size // BK,
        BK // simd_size,
    )

    # Retained legacy: both the input (a legacy iterator deref) and the
    # consumer (`copy_dram_to_sram_async`) are legacy-only.
    @inline(.always)
    def _mask_tensor_row(
        tensor: LayoutTensor, num_rows: Int
    ) {imm} -> type_of(tensor):
        return {
            tensor.ptr,
            {{num_rows, tensor.dim[1]()}, tensor.runtime_layout.stride},
        }

    comptime for q_id in range(depth // BK):
        var q_smem_tile = q_smem_iter.next_unsafe(
            q_smem_iter.layout_uint_type(q_id)
        )[]

        copy_dram_to_sram_async[
            thread_layout=async_copy_q_layout,
            swizzle=True,
            num_threads=num_threads,
        ](
            q_smem_tile.vectorize[1, simd_size](),
            q_gmem_iter[].vectorize[1, simd_size](),
        )

        # we `async_copy_commit_group()` and after we finish copying `k`.

        q_gmem_iter._incr()

    var scale_log2e: Float32 = (
        scale.cast[
            DType.float32
        ]() if mask_t.apply_log2e_after_mask else scale.cast[.float32]()
        * log2e
    )

    @inline(.always)
    def loop_over_kvcache[
        tile_size: Int, not_last_iter: Bool
    ](kv_tile_start_row: Int, end: Int) {mut k_smem_iter, mut v_smem_iter, imm}:
        var k_ptr = k.block_paged_ptr[BN](
            UInt32(batch_idx), UInt32(kv_tile_start_row), UInt32(kv_head_idx), 0
        )
        var k_gmem_block = LayoutTensor[
            k_type,
            Layout(
                IntTuple(BN, depth),
                IntTuple(kv_num_heads * depth, 1),
            ),
            masked=not not_last_iter,
        ](k_ptr)
        var k_gmem_iter = k_gmem_block.tiled_iterator[BN, BK, axis=1](0, 0)

        var kv_tile_num_rows = min(BN, end - kv_tile_start_row)

        _ = p_reg_tile.fill(0)

        comptime kv_num_vecs = BN * BK // simd_size
        comptime async_copy_k_layout = Layout.row_major(
            min(num_threads, kv_num_vecs)
            * simd_size
            // k_smem_iter.layout.stride[0].value(),
            k_smem_iter.layout.stride[0].value() // simd_size,
        )

        var k_tensor: type_of(k_gmem_iter[])
        # load K tile into smem
        comptime for k_id in range(depth // BK):
            var k_smem_tile = k_smem_iter.next_unsafe(
                k_smem_iter.layout_uint_type(k_id)
            )[]

            comptime if not not_last_iter:
                k_tensor = _mask_tensor_row(k_gmem_iter[], kv_tile_num_rows)
            else:
                k_tensor = k_gmem_iter[]

            copy_dram_to_sram_async[
                thread_layout=async_copy_k_layout,
                swizzle=True,
                num_threads=num_threads,
            ](
                k_smem_tile.vectorize[1, simd_size](),
                k_tensor.vectorize[1, simd_size](),
            )

            k_gmem_iter._incr()

        async_copy_commit_group()

        async_copy_wait_all()
        barrier()

        multistage_mma[
            BM,
            BN,
            BK,
            WM,
            WN,
            num_threads,
            num_pipeline_stages,
            True,  # transpose_b
            swizzle_a=True,
            prefetch_init=False,
            static_num_iters=ufloordiv(depth, BK),
        ](
            p_reg_tile,
            q_smem_iter,
            k_smem_iter,
            q_smem_iter,
            k_smem_iter,
            ufloordiv(depth, BK),
        )

        scale_and_mask_helper[
            num_n_mmas=num_n_mmas,
            WN=WN,
            MMA_N=MMA_N,
            simd_width=p_frag_simdwidth,
            group=group,
        ](
            p_reg_tile,
            scale_log2e,
            num_keys,
            kv_tile_num_rows,
            lane,
            warp_id,
            mask,
            kv_tile_start_row,
        )

        # For 16x8 mma output, group <= 8 only uses the first 8x8 matrix
        # each thread only has one fragment vector of size 2.
        comptime if group <= 8:
            _online_softmax_iter_for_mma_output[
                accum_type,
                Layout.row_major(num_m_mmas, num_n_mmas),
                Layout.row_major(num_warps_m, num_warps_n),
                Layout.row_major(8, 4),
                warp_split_k=decoding_warp_split_k,
                use_exp2=True,
            ](
                output_reg_tile.tile[num_output_rows_full, p_frag_size // 2](
                    0, 0
                ).vectorize[1, p_frag_simdwidth](),
                p_reg_tile.tile[num_m_mmas * num_n_mmas, p_frag_size // 2](
                    0, 0
                ).vectorize[1, p_frag_simdwidth](),
                warp_scratch.tile[2 * num_warps_n, WM](0, warp_y),
                rowmax,
                rowsum,
            )
        else:
            comptime out_view_layout = row_major[
                2 * num_output_rows_full, p_frag_simdwidth
            ]()
            comptime p_view_layout = row_major[
                2 * num_m_mmas * num_n_mmas, p_frag_simdwidth
            ]()
            _online_softmax_iter_for_mma_output[
                accum_type,
                Layout.row_major(2 * num_m_mmas, num_n_mmas),
                Layout.row_major(num_warps_m, num_warps_n),
                Layout.row_major(8, 4),
                warp_split_k=decoding_warp_split_k,
                use_exp2=True,
            ](
                output_reg_tile.reshape[type_of(out_view_layout)](
                    out_view_layout
                ).vectorize[1, p_frag_simdwidth](),
                p_reg_tile.reshape[type_of(p_view_layout)](
                    p_view_layout
                ).vectorize[1, p_frag_simdwidth](),
                warp_scratch.tile[2 * num_warps_n, WM](0, warp_y),
                rowmax,
                rowsum,
            )

        var v_ptr = v.block_paged_ptr[BN](
            UInt32(batch_idx), UInt32(kv_tile_start_row), UInt32(kv_head_idx), 0
        )
        var v_gmem_block = LayoutTensor[
            v_type,
            Layout(
                IntTuple(BN, depth),
                IntTuple(kv_num_heads * depth, 1),
            ),
            masked=not not_last_iter,
        ](v_ptr)
        var v_gmem_iter = v_gmem_block.tiled_iterator[BK, BN, axis=0](0, 0)

        comptime async_copy_v_layout = Layout.row_major(
            min(num_threads, kv_num_vecs) * simd_size // BN,
            BN // simd_size,
        )

        var v_tensor: type_of(v_gmem_iter[])
        # load V tile into smem
        comptime for v_id in range(BN // BK):
            var v_smem_tile = v_smem_iter.next_unsafe(
                v_smem_iter.layout_uint_type(v_id)
            )[]

            comptime if not not_last_iter:
                var num_rows_bound = max(
                    0, end - (kv_tile_start_row + v_id * BK)
                )
                v_tensor = _mask_tensor_row(v_gmem_iter[], num_rows_bound)
            else:
                v_tensor = v_gmem_iter[]

            copy_dram_to_sram_async[
                thread_layout=async_copy_v_layout,
                swizzle=v_smem_tile.dtype.is_half_float(),
                num_threads=num_threads,
            ](
                v_smem_tile.vectorize[1, simd_size](),
                v_tensor.vectorize[1, simd_size](),
            )

            v_gmem_iter._incr()

        async_copy_commit_group()

        comptime if not decoding_warp_split_k:
            # Copy score fragments to shared memory with swizzling to resolve bank
            # conflicts for ldmatrix in the 2nd matmul.
            # warp_split_k does not need the copy as warps don't perform reduction
            # iterating across tiles, but use extra registers to perform MMAs
            # with warp-local data.
            _copy_frag_to_smem[
                BM,
                BN,
                BK,
                WM,
                WN,
                MMA_M,
                MMA_N,
                p_frag_simdwidth,
            ](p_smem_iter, p_reg_tile, UInt32(warp_x), UInt32(warp_y))

        async_copy_wait_all()
        barrier()

        # if decoding_warp_split_k:
        #   S[m, (0:WN) + n*WN] @ V[(0:WN) + n*WN, :]
        # else:
        #   S[m, :] @ V[:, (0:WN) + n*WN]
        comptime if decoding_warp_split_k:
            # Warp split-k is dead in every current instantiation; keep the
            # legacy iterator view local to this branch until multistage_mma
            # grows a native register iterator.
            var p_reg_iter = p_reg_tile.to_layout_tensor().tiled_iterator[
                MMA_K // MMA_N * num_m_mmas, p_frag_size
            ](0, 0)
            comptime IteratorTypeVSub = LayoutTensorIter[
                v_type,
                Layout.row_major(WN, BN),
                _,
                address_space=.SHARED,
                circular=True,
            ]
            var v_smem_sub = IteratorTypeVSub(
                v_smem + BN * WN * warp_x,
                IteratorTypeVSub.layout_uint_type(v_smem_size),
            )
            multistage_mma[
                BM,
                BN,
                WN,  # BK
                WM,
                BN,  # WN
                num_threads,
                num_pipeline_stages,
                False,  # transpose_b
                swizzle_a=True,
                prefetch_init=False,
                static_num_iters=1,
            ](
                output_reg_tile,
                p_reg_iter,
                v_smem_sub,
                p_smem_iter,
                v_smem_sub,
                1,
            )
        else:
            multistage_mma[
                BM,
                BN,
                BK,
                WM,
                WN,
                num_threads,
                num_pipeline_stages,
                False,  # transpose_b
                swizzle_a=True,
                prefetch_init=False,
                static_num_iters=ufloordiv(BN, BK),
            ](
                output_reg_tile,
                p_smem_iter,
                v_smem_iter,
                p_smem_iter,
                v_smem_iter,
                ufloordiv(BN, BK),
            )

    tile_and_unswitch[[BN]](start, end, loop_over_kvcache)

    comptime if decoding_warp_split_k:
        var output_reg_vecs = output_reg_tile.tile[
            num_warps_n * num_m_mmas * num_n_mmas, p_frag_size // 2
        ](0, 0).vectorize[1, p_frag_size // 2]()
        # offset on the pointer is to avoid possible races
        # with `accum_smem_warp_tile`.
        var o_smem_ptr = q_smem.bitcast[Scalar[accum_type]]()
        var scratch = TileTensor[
            accum_type,
            type_of(row_major[2 * num_warps_n, BM]()),
            MutAnyOrigin,
            address_space=.SHARED,
        ](
            ptr=(
                o_smem_ptr + num_warps_n * (num_warps_n - 1) * WM * WN
            ).unsafe_origin_cast[MutAnyOrigin](),
            layout=row_major[2 * num_warps_n, BM](),
        )

        # Note: Sink handling is done after warp reduction in partition-specific logic below.
        # The warp reduction just combines warps; sink contribution is added to rowsum later.
        _online_softmax_iter_for_mma_output_split_warp_reduce[
            accum_type,
            Layout.row_major(num_m_mmas, num_n_mmas),
            Layout.row_major(num_warps_m, num_warps_n),
            Layout.row_major(8, 4),
            WM,
            WN,
            use_exp2=True,
        ](
            output_reg_vecs,
            scratch.tile[2 * num_warps_n, WM](0, warp_y),
            o_smem_ptr,
        )

    # Apply softmax denumerator.
    comptime for m_mma in range(num_m_mmas):
        comptime if m_mma * MMA_M < group:
            var rowsum_inv = recip(rowsum[2 * m_mma])

            comptime for n_mma in range(num_n_mmas):
                output_reg_tile[n_mma * num_m_mmas + m_mma, 0] *= rowsum_inv
                output_reg_tile[n_mma * num_m_mmas + m_mma, 1] *= rowsum_inv

        comptime if m_mma * MMA_M + (MMA_M // 2) < group:
            var rowsum_inv = recip(rowsum[2 * m_mma + 1])

            comptime for n_mma in range(num_n_mmas):
                output_reg_tile[n_mma * num_m_mmas + m_mma, 2] *= rowsum_inv
                output_reg_tile[n_mma * num_m_mmas + m_mma, 3] *= rowsum_inv

    if num_partitions > 1:
        if umod(thread_idx.x, 4) == 0 and thread_idx.x < 4 * group:
            var row_sum = rowsum[0]
            var row_max = rowmax[0]
            exp_sum_ptr[q_head_idx] = row_sum
            qk_max_ptr[q_head_idx] = row_max

    # Pack results in shared memory for wider simd width.
    var accum_smem_warp_ptr = (
        q_smem.bitcast[Scalar[output_type]]() + warp_id * WM * WN
    )

    comptime if decoding_warp_split_k:
        accum_smem_warp_ptr += ufloordiv(
            (
                (num_warps_n * (num_warps_n - 1))
                * WM
                * WN
                * size_of[accum_type]()
            ),
            size_of[output_type](),
        )
    var accum_smem_warp_tile = LayoutTensor[
        output_type,
        Layout.row_major(WM, WN),
        address_space=.SHARED,
    ](accum_smem_warp_ptr)

    comptime swizzle = make_swizzle[
        num_rows=MMA_M // 2, row_size=WN, access_size=MMA_N
    ]()

    # The FP32 -> half downcast copy still lives on the legacy path.
    var output_reg_tile_legacy = output_reg_tile.to_layout_tensor()

    comptime if decoding_warp_split_k:
        copy_local_to_shared[
            thread_layout=Layout.row_major(8, 4), swizzle=swizzle
        ](
            accum_smem_warp_tile.vectorize[1, 2](),
            output_reg_tile_legacy.tile[num_output_rows, p_frag_size](0, 0)
            .vectorize[1, 2]()
            .transpose(),
        )
    else:
        copy_local_to_shared[
            thread_layout=Layout.row_major(8, 4), swizzle=swizzle
        ](
            accum_smem_warp_tile.vectorize[1, 2](),
            output_reg_tile_legacy.vectorize[1, 2]().transpose(),
        )

    # Guard writing to shared memory.
    barrier()

    # FIXME: Using RuntimeLayout to override the layout of the output tensor.
    comptime output_gmem_layout = Layout.row_major(BM, depth)
    var output_gmem_runtime_layout = RuntimeLayout[
        output_gmem_layout
    ].row_major(Index(group, depth))
    var output_gmem_tile = LayoutTensor[
        output_type,
        output_gmem_layout,
        layout_int_type=.int32,
        linear_idx_type=.int32,
        masked=True,
    ](output_ptr + q_offset, output_gmem_runtime_layout)
    var output_gmem_warp_tile = output_gmem_tile.tile[WM, WN](warp_y, warp_x)

    copy_sram_to_dram[
        thread_layout=Layout.row_major(
            WARP_SIZE * simd_size // WN, WN // simd_size
        ),
        swizzle=swizzle,
    ](
        output_gmem_warp_tile.vectorize[1, simd_size](),
        accum_smem_warp_tile.vectorize[1, simd_size](),
    )


def mha_decoding_single_batch_pipelined[
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    *,
    BM: Int,  # number of queries per block
    BN: Int,  # number of keys per block
    BK: Int,  # tile size in depth dimension
    WM: Int,
    WN: Int,
    depth: Int,
    num_heads: Int,
    num_threads: Int,
    num_pipeline_stages: Int,
    group: Int = 1,
    decoding_warp_split_k: Bool = False,
    sink: Bool = False,
](
    q_ptr: UnsafePointer[Scalar[q_type], ImmutAnyOrigin],
    k: k_t,
    v: v_t,
    output_ptr: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    exp_sum_ptr: UnsafePointer[Scalar[get_accum_type[q_type]()], MutAnyOrigin],
    qk_max_ptr: UnsafePointer[Scalar[get_accum_type[q_type]()], MutAnyOrigin],
    scale: Float32,
    num_keys: Int,
    num_partitions: Int,
    sink_weights: OptionalReg[ImmutTileTensor1D[q_type]],
    mask: mask_t,
    batch_idx: Int,
):
    """Flash attention v2 decode kernel for a single batch element with pipelined multistage MMA.

    Computes attention for the decoding (single-query) case using the FA2
    online-softmax algorithm with multistage pipelining of K/V loads. When
    `num_partitions` exceeds 1, each block processes a contiguous slice of
    the key dimension and writes partial `exp_sum` and `qk_max` statistics
    for a subsequent `mha_splitk_reduce` pass.

    Parameters:
        q_type: Element type of the query tensor (inferred).
        k_t: Key operand type backing the key tensor (inferred).
        v_t: Value operand type backing the value tensor (inferred).
        output_type: Element type of the output tensor (inferred).
        mask_t: Attention mask type implementing `MHAMask` (inferred).
        BM: Number of query rows processed per thread block.
        BN: Number of key columns per thread block tile.
        BK: Tile size in the head-depth dimension.
        WM: Warp tile height in the query (M) dimension.
        WN: Warp tile width in the key (N) dimension.
        depth: Attention head depth (key/value dimension per head).
        num_heads: Total number of query heads.
        num_threads: Number of threads per thread block.
        num_pipeline_stages: Number of pipeline stages for the multistage
            MMA loads.
        group: GQA group size, query heads per key/value head (defaults
            to 1).
        decoding_warp_split_k: Enable warp-level split-K for decode
            (defaults to `False`).
        sink: Enable attention-sink mode where the first tokens always
            attend (defaults to `False`).

    Args:
        q_ptr: Pointer to the query tensor for this batch element.
        k: Key operand backed by a KV cache.
        v: Value operand backed by a KV cache.
        output_ptr: Pointer to the output tensor for this batch element.
        exp_sum_ptr: Pointer to the online-softmax exponent sum buffer
            for this batch.
        qk_max_ptr: Pointer to the online-softmax running maximum buffer
            for this batch.
        scale: Softmax temperature scale applied to Q·Kᵀ.
        num_keys: Total number of key/value entries (cache length) for
            this batch.
        num_partitions: Number of split-K partitions dividing the key
            dimension.
        sink_weights: Optional sink-token weight tensor for attention
            sinks.
        mask: Mask instance used to apply the attention mask.
        batch_idx: Index of the batch element this block processes.
    """
    comptime accum_type = get_accum_type[q_type]()
    comptime k_type = k_t.dtype
    comptime v_type = v_t.dtype
    comptime assert q_type == k_type and k_type == v_type

    comptime simd_size = simd_width_of[q_type]()

    comptime num_warps_m = BM // WM
    comptime num_warps_n = BN // WN

    comptime assert (
        num_warps_m * num_warps_n == num_threads // WARP_SIZE
    ), "Number of warps doesn't match warp tile sizes."

    comptime assert group <= 8, String(
        "Only support GQA with group <= 8 for Nvidia, but got a group = '",
        group,
        "'.",
    )

    var warp_id = warp_id[broadcast=True]()
    var lane = lane_id()

    # Coordinates of the current warp.
    var warp_y, warp_x = udivmod(warp_id, num_warps_n)

    # The entire query block (BM x depth) is tiled in shared memory.
    comptime alignment = align_of[SIMD[q_type, simd_size]]()
    comptime q_smem_size = BM * depth
    var q_smem = external_memory[
        Scalar[q_type],
        address_space=.SHARED,
        alignment=alignment,
    ]()
    comptime IteratorTypeQ = LayoutTensorIter[
        q_type,
        Layout.row_major(BM, BK),
        _,
        address_space=.SHARED,
        alignment=alignment,
    ]
    var q_smem_iter = IteratorTypeQ(
        rebind[
            type_of(
                LayoutTensorIter[
                    q_type,
                    Layout.row_major(BM, BK),
                    q_smem.origin,
                    address_space=.SHARED,
                    alignment=alignment,
                ]().ptr
            )
        ](q_smem),
        IteratorTypeQ.layout_uint_type(q_smem_size),
    )

    # There is one pre-allocated dynamic shared buffer.
    # Need to explicitly offset key after at query's end.
    comptime k_smem_size = num_pipeline_stages * BN * BK
    var k_smem = (q_smem + q_smem_size).bitcast[Scalar[k_type]]()
    comptime IteratorTypeK = LayoutTensorIter[
        k_type,
        Layout.row_major(BN, BK),
        MutAnyOrigin,
        address_space=.SHARED,
        circular=True,
    ]
    var k_smem_iter = IteratorTypeK(
        k_smem.as_unsafe_any_origin(),
        IteratorTypeK.layout_uint_type(k_smem_size),
    )

    var kv_head_idx = block_idx.y

    comptime mma_shape = get_mma_shape[q_type, accum_type]()
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]
    comptime num_m_mmas = WM // MMA_M
    comptime num_n_mmas = WN // MMA_N

    comptime frag_size = get_fragment_size[mma_shape]()
    comptime p_frag_size = frag_size[2]
    comptime p_frag_simdwidth = p_frag_size // 2
    comptime p_frag_align = align_of[SIMD[accum_type, p_frag_size]]()

    var p_reg_tile_native = stack_allocation[
        accum_type, address_space=.LOCAL, alignment=p_frag_align
    ](row_major[num_m_mmas * num_n_mmas, p_frag_size]())
    var output_reg_tile_native = stack_allocation[
        accum_type, address_space=.LOCAL, alignment=p_frag_align
    ](row_major[num_m_mmas * num_n_mmas, p_frag_size]()).fill(0.0)
    # MMA and softmax helpers retain their legacy scalar views.
    var p_reg_tile = p_reg_tile_native.to_layout_tensor().as_unsafe_any_origin()
    var output_reg_tile = (
        output_reg_tile_native.to_layout_tensor().as_unsafe_any_origin()
    )

    # Account for group query.
    comptime kv_num_heads = num_heads // group
    var q_head_idx = kv_head_idx * group + ufloordiv(thread_idx.x, 4)

    # Rowwise max and sum for online softmax
    comptime row_align = align_of[
        SIMD[accum_type, simd_width_of[accum_type]()]
    ]()
    var rowmax = unsafe_stack_allocation[WM, accum_type, alignment=row_align]()
    var rowsum = unsafe_stack_allocation[WM, accum_type, alignment=row_align]()

    var partition_idx = block_idx.x

    comptime for i in range(WM):
        comptime if sink:
            assert Bool(
                sink_weights
            ), "expect sink_weights to be non-null when sink=true"
            if thread_idx.x < 4 * group:
                var sink_logit_log2 = (
                    sink_weights.value()[q_head_idx].cast[accum_type]() * log2e
                )
                rowmax[i] = sink_logit_log2
                if partition_idx == 0 and umod(thread_idx.x, 4) == 0:
                    rowsum[i] = 1.0
                else:
                    rowsum[i] = 0.0
            else:
                rowmax[i] = min_or_neg_inf[accum_type]()
                rowsum[i] = 0.0
        else:
            rowmax[i] = min_or_neg_inf[accum_type]()
            rowsum[i] = 0.0

    # Share memory tile for Value, reuse K's shared memory tile.
    comptime v_smem_size = num_pipeline_stages * BN * BK
    var v_smem = k_smem.bitcast[Scalar[v_type]]()
    comptime IteratorTypeV = LayoutTensorIter[
        v_type,
        Layout.row_major(BK, BN),
        MutAnyOrigin,
        address_space=.SHARED,
        circular=True,
    ]
    var v_smem_iter = IteratorTypeV(
        v_smem.as_unsafe_any_origin(),
        IteratorTypeV.layout_uint_type(v_smem_size),
    )

    # Shared memory for P = Q * K^t
    # This overlaps key tile but are used at the same time i.e. no race condition.
    var p_smem = (v_smem + v_smem_size).bitcast[Scalar[v_type]]()
    comptime p_smem_size = BM * BN
    comptime IteratorTypeP = LayoutTensorIter[
        v_type,
        Layout.row_major(BM, BK),
        _,
        address_space=.SHARED,
        circular=True,
    ]
    var p_smem_iter = IteratorTypeP(
        p_smem, IteratorTypeP.layout_uint_type(p_smem_size)
    )

    # Scratch shared memory for reduction across warps.
    var warp_scratch = TileTensor(
        (p_smem + BM * BN).bitcast[Scalar[accum_type]]().as_unsafe_any_origin(),
        row_major[p_frag_simdwidth * num_warps_n, BM](),
    )

    var q_offset = depth * kv_head_idx * group

    comptime q_gmem_layout = Layout.row_major(BM, depth)
    var q_gmem_block = LayoutTensor[
        q_type,
        q_gmem_layout,
        layout_int_type=.int32,
        linear_idx_type=.int32,
        masked=True,
    ](
        q_ptr + q_offset,
        RuntimeLayout[element_type=.int32, linear_idx_type=.int32](
            RuntimeTuple[q_gmem_layout.shape, element_type=.int32](
                group, depth
            ),
            RuntimeTuple[q_gmem_layout.stride, element_type=.int32](depth, 1),
        ),
    )
    var q_gmem_iter = q_gmem_block.tiled_iterator[BM, BK, axis=1](0, 0)

    # Loop over Key and Value tiles
    var start, end = get_start_and_end_for_partitions[BN](
        num_keys, num_partitions, block_idx.x
    )

    var scale_log2e: Float32 = (
        scale.cast[
            DType.float32
        ]() if mask_t.apply_log2e_after_mask else scale.cast[.float32]()
        * log2e
    )

    @inline(.always)
    def loop_over_kvcache[
        tile_size: Int, not_last_iter: Bool
    ](kv_tile_start_row: Int, seq_len: Int) {
        mut k_smem_iter, mut v_smem_iter, imm
    }:
        var k_ptr = k.block_paged_ptr[BN](
            UInt32(batch_idx), UInt32(kv_tile_start_row), UInt32(kv_head_idx), 0
        )
        var k_gmem_block = LayoutTensor[
            k_type,
            Layout(
                IntTuple(BN, depth),
                IntTuple(kv_num_heads * depth, 1),
            ),
            masked=not not_last_iter,
        ](k_ptr)
        var k_gmem_iter = k_gmem_block.tiled_iterator[BN, BK, axis=1](0, 0)

        var kv_tile_num_rows = min(BN, end - kv_tile_start_row)

        _ = p_reg_tile.fill(0)

        if kv_tile_start_row == start:
            multistage_mma[
                BM,
                BN,
                BK,
                WM,
                WN,
                num_threads,
                num_pipeline_stages,
                True,  # transpose_b
                swizzle_a=True,
            ](
                lt_to_tt_idx[linear_idx_type=p_reg_tile.linear_idx_type](
                    p_reg_tile
                ),
                q_gmem_iter,
                k_gmem_iter,
                q_smem_iter,
                k_smem_iter,
                ufloordiv(depth, BK),
                num_b_rows=kv_tile_num_rows,
            )
        else:
            multistage_mma[
                BM,
                BN,
                BK,
                WM,
                WN,
                num_threads,
                num_pipeline_stages,
                True,  # transpose_b
                swizzle_a=True,
            ](
                lt_to_tt_idx[linear_idx_type=p_reg_tile.linear_idx_type](
                    p_reg_tile
                ),
                q_smem_iter,
                k_gmem_iter,
                q_smem_iter,
                k_smem_iter,
                ufloordiv(depth, BK),
                num_b_rows=kv_tile_num_rows,
            )

        scale_and_mask_helper[
            num_n_mmas=num_n_mmas,
            WN=WN,
            MMA_N=MMA_N,
            simd_width=p_frag_simdwidth,
            group=group,
        ](
            lt_to_tt_idx[linear_idx_type=p_reg_tile.linear_idx_type](
                p_reg_tile
            ),
            scale_log2e,
            num_keys,
            kv_tile_num_rows,
            lane,
            warp_id,
            mask,
            kv_tile_start_row,
        )

        # For 16x8 mma output, only the top 8x4 matrix matters for GQA since
        # G <= 8 typically holds

        _online_softmax_iter_for_mma_output[
            accum_type,
            Layout.row_major(num_m_mmas, num_n_mmas),
            Layout.row_major(num_warps_m, num_warps_n),
            Layout.row_major(8, 4),
            use_exp2=True,
        ](
            lt_to_tt_idx[linear_idx_type=output_reg_tile.linear_idx_type](
                output_reg_tile.tile[num_m_mmas * num_n_mmas, p_frag_size // 2](
                    0, 0
                )
            ).vectorize[1, p_frag_size // 2](),
            lt_to_tt_idx[linear_idx_type=p_reg_tile.linear_idx_type](
                p_reg_tile.tile[num_m_mmas * num_n_mmas, p_frag_size // 2](0, 0)
            ).vectorize[1, p_frag_size // 2](),
            warp_scratch.tile[2 * num_warps_n, WM](0, warp_y),
            rowmax,
            rowsum,
        )

        var v_ptr = v.block_paged_ptr[BN](
            UInt32(batch_idx), UInt32(kv_tile_start_row), UInt32(kv_head_idx), 0
        )
        var v_gmem_block = LayoutTensor[
            v_type,
            Layout(
                IntTuple(BN, depth),
                IntTuple(kv_num_heads * depth, 1),
            ),
            masked=not not_last_iter,
        ](v_ptr)
        var v_gmem_iter = v_gmem_block.tiled_iterator[BK, BN, axis=0](0, 0)

        # Copy score fragments to shared memory with swizzling to resolve bank
        # conflicts for ldmatrix in the 2nd matmul.
        _copy_frag_to_smem[
            BM,
            BN,
            BK,
            WM,
            WN,
            MMA_M,
            MMA_N,
            p_frag_simdwidth,
        ](p_smem_iter, p_reg_tile, UInt32(warp_x), UInt32(warp_y))
        barrier()

        multistage_mma[
            BM,
            BN,
            BK,
            WM,
            WN,
            num_threads,
            num_pipeline_stages,
            False,  # transpose_b
            swizzle_a=True,
        ](
            lt_to_tt_idx[linear_idx_type=output_reg_tile.linear_idx_type](
                output_reg_tile
            ),
            p_smem_iter,
            v_gmem_iter,
            p_smem_iter,
            v_smem_iter,
            ufloordiv(BN, BK),
            num_b_rows=kv_tile_num_rows,
        )

    tile_and_unswitch[[BN]](start, end, loop_over_kvcache)

    # Apply softmax denumerator.

    comptime for m_mma in range(num_m_mmas):
        var rowsum_inv0 = 1.0 / rowsum[2 * m_mma]

        comptime for n_mma in range(num_n_mmas):
            output_reg_tile[n_mma, 0] *= rowsum_inv0
            output_reg_tile[n_mma, 1] *= rowsum_inv0

    if num_partitions > 1:
        if umod(thread_idx.x, 4) == 0 and thread_idx.x < 4 * group:
            var row_sum = rowsum[0]
            var row_max = rowmax[0]
            var q_head_idx = kv_head_idx * group + ufloordiv(thread_idx.x, 4)
            exp_sum_ptr[q_head_idx] = row_sum
            qk_max_ptr[q_head_idx] = row_max

    # Pack results in shared memory for wider simd width.
    var accum_smem_warp_tile = LayoutTensor[
        output_type,
        Layout.row_major(WM, WN),
        address_space=.SHARED,
    ](q_smem.bitcast[Scalar[output_type]]() + warp_id * WM * WN)

    comptime swizzle = make_swizzle[
        num_rows=MMA_M // 2, row_size=WN, access_size=MMA_N
    ]()
    copy_local_to_shared[thread_layout=Layout.row_major(8, 4), swizzle=swizzle](
        accum_smem_warp_tile.vectorize[1, 2](),
        output_reg_tile.vectorize[1, 2]().transpose(),
    )
    # Guard writing to shared memory.
    barrier()
    comptime output_gmem_layout = Layout.row_major(BM, depth)
    var output_gmem_runtime_layout = RuntimeLayout[
        element_type=.int32, linear_idx_type=.int32
    ](
        RuntimeTuple[output_gmem_layout.shape, element_type=.int32](
            group, depth
        ),
        RuntimeTuple[output_gmem_layout.stride, element_type=.int32](depth, 1),
    )
    var output_gmem_tile = LayoutTensor[
        output_type,
        Layout.row_major(BM, depth),
        layout_int_type=.int32,
        linear_idx_type=.int32,
        masked=True,
    ](output_ptr + q_offset, output_gmem_runtime_layout)
    var output_gmem_warp_tile = output_gmem_tile.tile[WM, WN](warp_y, warp_x)
    copy_sram_to_dram[
        thread_layout=Layout.row_major(
            WARP_SIZE * simd_size // WN, WN // simd_size
        ),
        swizzle=swizzle,
    ](
        output_gmem_warp_tile.vectorize[1, simd_size](),
        accum_smem_warp_tile.vectorize[1, simd_size](),
    )


@__name(t"mha_splitk_reduce_{intermediate_type}_{output_type}")
def mha_splitk_reduce[
    intermediate_type: DType,
    output_type: DType,
    depth: Int,
    num_heads: Int,
    num_threads: Int,
    use_exp2: Bool = False,
](
    intermediate_ptr: UnsafePointer[Scalar[intermediate_type], ImmutAnyOrigin],
    output_ptr: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    exp_sum_ptr: UnsafePointer[
        Scalar[get_accum_type[output_type]()], ImmutAnyOrigin
    ],
    qk_max_ptr: UnsafePointer[
        Scalar[get_accum_type[output_type]()], ImmutAnyOrigin
    ],
    batch_size: Int32,
    num_partitions: Int32,
):
    """Single-warp reduction kernel that merges split-K partial attention outputs.

    Reads `num_partitions` partial attention outputs together with their
    running `exp_sum` (denominator) and `qk_max` statistics, re-weights
    each partial output by its softmax scale relative to the global maximum,
    sums them, and writes the normalised result to `output_ptr`. Must be
    launched with exactly one warp (`WARP_SIZE` threads) per output row.

    Parameters:
        intermediate_type: Element type of the partial output buffer written
            by the split-K decode kernel.
        output_type: Element type of the final attention output.
        depth: Attention head depth.
        num_heads: Number of query heads.
        num_threads: Must equal `WARP_SIZE`.
        use_exp2: `True` to use base-2 exponentiation for numerics matching
            FA3 kernels that fuse scale * log2e.

    Args:
        intermediate_ptr: Pointer to partial output buffer
            `[num_partitions, batch, heads, depth]`.
        output_ptr: Pointer to final output buffer `[batch, heads, depth]`.
        exp_sum_ptr: Pointer to partial exp-sum buffer
            `[num_partitions, batch, heads]`.
        qk_max_ptr: Pointer to partial softmax-max buffer
            `[num_partitions, batch, heads]`.
        batch_size: Number of sequences in the batch.
        num_partitions: Number of split-K partitions to reduce.
    """

    # we only reduce over a warp so limit number of warps to 1
    var _batch_size = Int(batch_size)
    var _num_partitions = Int(num_partitions)
    comptime assert num_threads == WARP_SIZE, (
        "num_threads: "
        + String(num_threads)
        + " should be equal to the warp_size:"
        + String(WARP_SIZE)
    )
    assert (
        block_dim.x == WARP_SIZE
    ), "block_dim.x should be equal to the warp_size"

    # Programmatic Dependent Launch. Single-warp kernel with no early returns,
    # so the function entry is a divergence-free point all threads reach before
    # the first read of the producer's partial outputs / exp_sum / qk_max
    # below. `wait` fences here so those reads only happen after the split-K
    # producer grid has flushed them; `launch` lets the successor grid's
    # prologue overlap this reduction. No-op on non-SM90+ and when MHA_PDL=off.
    comptime if MHA_PDL_LEVEL > PDLLevel.OFF:
        wait_on_dependent_grids()
        launch_dependent_grids()

    comptime accum_type = get_accum_type[output_type]()
    var batch_idx = block_idx.z
    var q_head_idx = block_idx.y

    assert (
        _num_partitions <= WARP_SIZE
    ), "number of partitions should be less than or equal to the warp_size"
    var partition_idx = thread_idx.x

    var qk_max_offset = (
        num_heads * batch_idx
        + num_heads * _batch_size * partition_idx
        + q_head_idx
    )
    var l = min_or_neg_inf[accum_type]()
    if partition_idx < _num_partitions:
        l = qk_max_ptr[qk_max_offset]

    var qk_max = warp.lane_group_max[WARP_SIZE](l)

    # since _num_partitions <= WARP_SIZE, allocate buffer using WARP_SIZE
    var exp_sums = TileTensor(
        unsafe_stack_allocation[
            WARP_SIZE,
            Scalar[accum_type],
            address_space=.SHARED,
        ](),
        row_major[WARP_SIZE](),
    )

    var intermediate_output = TileTensor(
        intermediate_ptr,
        row_major((_num_partitions, _batch_size, Idx[num_heads], Idx[depth])),
    )
    var output = TileTensor(
        output_ptr, row_major((_batch_size, Idx[num_heads], Idx[depth]))
    )

    var rescaled_exp_sum: Scalar[accum_type] = 0
    comptime exp_fn = _exp2_concrete if use_exp2 else _exp_concrete
    if partition_idx < _num_partitions:
        rescaled_exp_sum = exp_sum_ptr[qk_max_offset] * exp_fn(l - qk_max)
        exp_sums[partition_idx] = rescaled_exp_sum

    # ensure exp_sums is written to before reading
    barrier()

    var exp_sum = warp.sum(rescaled_exp_sum)

    var inv_global_exp_sum = 1.0 / exp_sum

    comptime width = next_power_of_two(ceildiv(depth, num_threads))
    comptime assert depth % width == 0, "depth must be divisible by width"
    comptime assert (
        width * num_threads >= depth
    ), "width * num_threads must be greater than or equal to depth"

    var acc = SIMD[accum_type, width](0)
    # Kahan summation compensation for improved precision with many partitions
    var compensation = SIMD[accum_type, width](0)
    var depth_idx = thread_idx.x * width

    # Keep captured Kahan state inlined across scalar-tail calls on Metal.
    @inline(.always)
    def accum_fn[
        simd_width: Int
    ](partition_idx: Int) {
        exp_sums,
        intermediate_output,
        imm batch_idx,
        imm q_head_idx,
        imm depth_idx,
        mut,
    }:
        var partition_exp_sum = exp_sums.vectorize[simd_width]()[
            partition_idx // simd_width
        ]

        comptime for i in range(simd_width):
            var x_load = intermediate_output.load[
                width=width,
                alignment=width * size_of[intermediate_type](),
            ]((partition_idx + i, batch_idx, q_head_idx, depth_idx)).cast[
                accum_type
            ]()
            var scale = partition_exp_sum[i]
            var mask = SIMD[.bool, width](fill=scale > 0)
            var safe_load = mask.select(x_load, type_of(x_load)(0))
            var term = safe_load * type_of(safe_load)(scale)

            # Kahan summation: compensate for lost low-order bits
            var y = term - compensation
            var t = acc + y
            compensation = (t - acc) - y
            acc = t

    if depth_idx < depth:
        # simd_width=8 is based on experimentation
        # we may want to use a lower value if number of partitions are lower
        vectorize[8](_num_partitions, accum_fn)

        acc *= inv_global_exp_sum

        output.store[alignment=width * size_of[output_type]()](
            (batch_idx, q_head_idx, depth_idx), acc.cast[output_type]()
        )


# ===-----------------------------------------------------------------------===#
# Naive GPU multihead attention supporting flexible dimensions and
# _batch_size > 1.
# ===-----------------------------------------------------------------------===#

comptime _NAIVE_BMM_BLOCK_DIM = LaunchDim(32, 16, 1)
comptime _NAIVE_BMM_BLOCK_TUPLE = StaticTuple[Int32, 1](
    Int32(
        _NAIVE_BMM_BLOCK_DIM.x()
        * _NAIVE_BMM_BLOCK_DIM.y()
        * _NAIVE_BMM_BLOCK_DIM.z()
    )
)


def mha_gpu_naive[
    output_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    //,
    ragged: Bool = False,
    sink: Bool = False,
    _use_valid_length: Bool = False,
    _is_cache_length_accurate: Bool = False,
](
    q: TileTensor[mut=False, address_space=.GENERIC, ...],
    k: k_t,
    v: v_t,
    mask_functor: mask_t,
    output: TileTensor[mut=True, output_type, address_space=.GENERIC, ...],
    valid_length: OptionalReg[ImmutTileTensor1D[.uint32]],
    scale: Float32,
    batch_size: Int,
    max_prompt_len: Int,
    max_cache_size: Int,
    num_heads: Int,
    depth: Int,
    group: Int,
    ctx: DeviceContext,
    sink_weights: OptionalReg[ImmutTileTensor1D[q.dtype]] = None,
) raises:
    """Launch the naive (two-pass BMM) GPU attention implementation.

    Computes attention as two separate batched matrix multiplications using
    temporary GMEM storage for the P = softmax(Q·Kᵀ / scale) intermediate.
    This is slower than flash attention but supports any head depth and serves
    as a correctness reference. Dispatches three sequential kernels:
    `_bmm0_bs` (Q·Kᵀ + mask), softmax normalisation, and `_bmm1_bs` (P·V).

    Parameters:
        output_type: Element type of the attention output.
        k_t: Key operand type (dense or KV-cache).
        v_t: Value operand type (dense or KV-cache).
        mask_t: Attention mask type.
        ragged: `True` for ragged-batch inputs.
        sink: `True` to enable attention-sink mode.
        _use_valid_length: `True` to read per-sequence valid lengths.
        _is_cache_length_accurate: `True` when cache length is exact.

    Args:
        q: Query tensor.
        k: Key operand.
        v: Value operand.
        mask_functor: Mask instance.
        output: Mutable output tensor.
        valid_length: Per-sequence valid lengths; `None` for dense inputs.
        scale: Softmax temperature scale.
        batch_size: Number of sequences in the batch.
        max_prompt_len: Maximum query sequence length.
        max_cache_size: Maximum key/value sequence length.
        num_heads: Number of query heads.
        depth: Attention head depth.
        group: GQA group size.
        ctx: GPU device context.
        sink_weights: Sink-token weights for attention-sink mode.
    """

    comptime q_type = q.dtype
    comptime k_type = k_t.dtype
    comptime v_type = k_type

    var num_keys = max_cache_size

    if batch_size == 0 or num_keys == 0 or max_prompt_len == 0:
        return

    comptime p_type = get_accum_type[q_type]()
    var p_device = ctx.enqueue_create_buffer[p_type](
        batch_size * num_heads * max_prompt_len * num_keys
    )
    # FIXME: RUNP-356 Direct access to CUDA within DeviceContext
    var p_buffer = TileTensor(
        # FIXME: GEX-4123 Force use of DefaultEngine until the
        # `input_fn_device` legacy closure is replaced.
        p_device.unsafe_ptr(),
        row_major(
            (
                batch_size * num_heads,
                max_prompt_len,
                num_keys,
            )
        ),
    )
    var q_device = q.to_device_buffer(ctx)
    var output_device = output.to_device_buffer(ctx)
    comptime kernel = _bmm0_bs[
        q_type,
        k_t,
        mask_t,
        p_type,
        ragged=ragged,
        _use_valid_length=_use_valid_length,
        _is_cache_length_accurate=_is_cache_length_accurate,
    ]

    ctx.enqueue_function[kernel](
        p_device,
        q_device,
        k,
        valid_length,
        scale,
        Int32(batch_size),
        Int32(max_prompt_len),
        Int32(max_cache_size),
        Int32(num_heads),
        Int32(depth),
        Int32(group),
        mask_functor,
        grid_dim=(
            ceildiv(num_keys, 32),
            ceildiv(max_prompt_len, 16),
            num_heads * batch_size,
        ),
        block_dim=_NAIVE_BMM_BLOCK_DIM,
    )

    @__parameter
    @__copy_capture(p_buffer)
    def input_fn_device[
        _simd_width: Int
    ](coords: Coord) -> SIMD[p_type, _simd_width]:
        comptime assert p_buffer.flat_rank >= coords.flat_rank
        return p_buffer.load[width=_simd_width](coords)

    _softmax_gpu[p_type, 1, 3, input_fn_device, sink=sink](
        Coord(batch_size * num_heads, max_prompt_len, num_keys),
        p_buffer,
        2,
        ctx,
        sink_weights=sink_weights,
    )
    comptime kernel_1 = _bmm1_bs[
        output_type,
        p_type,
        v_t,
        ragged=ragged,
        _use_valid_length=_use_valid_length,
        _is_cache_length_accurate=_is_cache_length_accurate,
    ]
    ctx.enqueue_function[kernel_1](
        output_device,
        p_device,
        v,
        valid_length,
        Int32(max_prompt_len),
        Int32(max_cache_size),
        Int32(num_heads),
        Int32(depth),
        Int32(group),
        grid_dim=(
            ceildiv(depth, 32),
            ceildiv(max_prompt_len, 16),
            num_heads * batch_size,
        ),
        block_dim=_NAIVE_BMM_BLOCK_DIM,
    )

    _ = p_device^


@inline(.always)
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=_NAIVE_BMM_BLOCK_TUPLE)
@__name(t"mha_bmm0_{q_type}_{p_type}_{ragged}")
def _bmm0_bs[
    q_type: DType,
    k_t: MHAOperand,
    mask_t: MHAMask,
    p_type: DType,
    ragged: Bool = False,
    _use_valid_length: Bool = False,
    _is_cache_length_accurate: Bool = False,
](
    p_ptr: UnsafePointer[Scalar[p_type], MutAnyOrigin],
    q_ptr: UnsafePointer[Scalar[q_type], ImmutAnyOrigin],
    k: k_t,
    valid_length: OptionalReg[ImmutTileTensor1D[.uint32]],
    scale: Float32,
    batch_size: Int32,
    max_prompt_len: Int32,
    max_cache_size: Int32,
    num_heads: Int32,
    depth: Int32,
    group: Int32,
    mask_functor: mask_t,
):
    # In the num_keys dim.
    var _batch_size = Int(batch_size)
    var _max_prompt_len = Int(max_prompt_len)
    var _max_cache_size = Int(max_cache_size)
    var _num_heads = Int(num_heads)
    var _depth = Int(depth)
    var _group = Int(group)
    var x = global_idx.x
    # In the prompt length dim.
    var y = global_idx.y

    comptime k_type = k_t.dtype

    var batch_head = block_idx.z
    var batch, head = udivmod(batch_head, _num_heads)

    var cur_query_len: Int
    var q_offset: Int
    var cur_cache_len: Int
    var padded_num_keys = _max_cache_size
    var p_offset = batch_head * _max_prompt_len * padded_num_keys
    var start_pos: UInt32 = 0

    comptime if ragged:
        comptime if not _is_cache_length_accurate:
            start_pos = UInt32(k.cache_length(batch))

        var seq_start = Int(valid_length.value()[batch])
        var seq_end = Int(valid_length.value()[batch + 1])
        cur_query_len = seq_end - seq_start
        q_offset = _depth * (seq_start * _num_heads + head)
        cur_cache_len = Int(start_pos) + cur_query_len
    elif _use_valid_length:
        cur_query_len = Int(valid_length.value()[batch])
        q_offset = _depth * (head + _num_heads * _max_prompt_len * batch)
        comptime if _is_cache_length_accurate:
            cur_cache_len = cur_query_len
        else:
            cur_cache_len = k.cache_length(batch) + cur_query_len
    # When inputs are all dense tensors i.e. all sequences in batch have the same
    # length and same cache length
    else:
        cur_query_len = _max_prompt_len
        q_offset = _depth * (head + _num_heads * _max_prompt_len * batch)
        cur_cache_len = _max_cache_size
        p_offset = batch_head * _max_prompt_len * _max_cache_size

    assert cur_query_len <= _max_prompt_len, "Invalid cur_query_len"
    assert cur_cache_len <= padded_num_keys, "Invalid cur_cache_len"

    if x >= padded_num_keys or y >= _max_prompt_len:
        return

    var q = q_ptr + q_offset

    var kv_head = ufloordiv(head, _group)

    var p = p_ptr + p_offset

    var accum = Scalar[p_type](0.0)

    if x < cur_cache_len and y < cur_query_len:
        var k_ptr = k.block_paged_ptr[1](
            UInt32(batch), UInt32(x), UInt32(kv_head), 0
        )

        comptime accum_width = min(
            simd_width_of[q_type](), simd_width_of[k_type]()
        )
        var accum_vec = SIMD[p_type, accum_width](0)

        def accum_fn[
            width: Int
        ](offset: Int) {q, y, _num_heads, _depth, k_ptr, mut}:
            var q_val = q.load[
                width=width, alignment=align_of[SIMD[q_type, width]]()
            ](y * _num_heads * _depth + offset)
            var k_val = k_ptr.load[
                width=width, alignment=align_of[SIMD[k_type, width]]()
            ](offset)
            var qk_val = q_val.cast[p_type]() * k_val.cast[p_type]()

            comptime if width == 1:
                accum += rebind[type_of(accum)](qk_val)
            else:
                accum_vec += rebind[type_of(accum_vec)](qk_val)

        if _depth % accum_width == 0:
            vectorize[accum_width](_depth, accum_fn)
            accum += accum_vec.reduce_add()
        else:
            vectorize[1](_depth, accum_fn)

    var score_row = y + cur_cache_len - cur_query_len
    var score_col = x
    p[y * padded_num_keys + x] = mask_functor.mask(
        Index(
            batch,
            head,
            score_row,
            score_col,
        ),
        accum * scale.cast[p_type](),
    )

    if x >= cur_cache_len or y >= cur_query_len:
        p[y * padded_num_keys + x] = min_or_neg_inf[p_type]()


@inline(.always)
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=_NAIVE_BMM_BLOCK_TUPLE)
@__name(t"mha_bmm1_{output_type}_{p_type}_{ragged}")
def _bmm1_bs[
    output_type: DType,
    p_type: DType,
    v_t: MHAOperand,
    ragged: Bool = False,
    _use_valid_length: Bool = False,
    _is_cache_length_accurate: Bool = False,
](
    output_ptr: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    p_ptr: UnsafePointer[Scalar[p_type], ImmutAnyOrigin],
    v: v_t,
    valid_length: OptionalReg[ImmutTileTensor1D[.uint32]],
    max_prompt_len: Int32,
    max_cache_size: Int32,
    num_heads: Int32,
    depth: Int32,
    group: Int32,
):
    var _max_prompt_len = Int(max_prompt_len)
    var _max_cache_size = Int(max_cache_size)
    var _num_heads = Int(num_heads)
    var _depth = Int(depth)
    var _group = Int(group)
    comptime v_type = v_t.dtype

    # In the _depth dim.
    var x = global_idx.x
    # IN the sequence length dim.
    var y = global_idx.y

    var batch_head = block_idx.z
    var batch, head = udivmod(batch_head, _num_heads)

    var cur_query_len: Int
    var output_offset: Int
    var cur_cache_len: Int
    var padded_num_keys = _max_cache_size
    var p_offset = batch_head * _max_prompt_len * padded_num_keys
    var start_pos: UInt32 = 0

    comptime if ragged:
        comptime if not _is_cache_length_accurate:
            start_pos = UInt32(v.cache_length(batch))

        var seq_start = Int(valid_length.value()[batch])
        var seq_end = Int(valid_length.value()[batch + 1])
        cur_query_len = seq_end - seq_start
        output_offset = (seq_start * _num_heads + head) * _depth
        cur_cache_len = cur_query_len + Int(start_pos)
    elif _use_valid_length:
        cur_query_len = Int(valid_length.value()[batch])
        output_offset = _depth * (head + _num_heads * _max_prompt_len * batch)
        comptime if _is_cache_length_accurate:
            cur_cache_len = cur_query_len
        else:
            cur_cache_len = cur_query_len + v.cache_length(batch)
    # When inputs are all dense tensors i.e. all sequences in batch have the same
    # length and same cache length
    else:
        cur_query_len = _max_prompt_len
        output_offset = _depth * (head + _num_heads * _max_prompt_len * batch)
        cur_cache_len = _max_cache_size
        p_offset = batch_head * _max_prompt_len * _max_cache_size

    assert cur_query_len <= _max_prompt_len, "Invalid cur_query_len"

    if x >= _depth or y >= cur_query_len:
        return

    var p = p_ptr + p_offset

    var kv_head = ufloordiv(head, _group)
    var output = output_ptr + output_offset

    var accum = Float32(0.0)

    for i in range(cur_cache_len):
        var v_ptr = v.block_paged_ptr[1](
            UInt32(batch), UInt32(i), UInt32(kv_head), UInt32(x)
        )
        accum += (
            p[y * padded_num_keys + i].cast[.float32]()
            * v_ptr[0].cast[.float32]()
        )

    output[y * _num_heads * _depth + x] = accum.cast[output_type]()


# ===-----------------------------------------------------------------------===#
# Naive GPU multihead attention supporting flexible dimensions.
# ===-----------------------------------------------------------------------===#


def mha_gpu_naive[
    q_type: DType,
    k_type: DType,
    v_type: DType,
    output_type: DType,
    mask_type: DType,
    q_tt_layout: TensorLayout,
    k_tt_layout: TensorLayout,
    v_tt_layout: TensorLayout,
    mask_tt_layout: TensorLayout,
    output_tt_layout: TensorLayout,
    //,
    sink: Bool = False,
](
    q: TileTensor[mut=False, q_type, q_tt_layout, address_space=.GENERIC, ...],
    k: TileTensor[mut=False, k_type, k_tt_layout, address_space=.GENERIC, ...],
    v: TileTensor[mut=False, v_type, v_tt_layout, address_space=.GENERIC, ...],
    mask: TileTensor[
        mut=False, mask_type, mask_tt_layout, address_space=.GENERIC, ...
    ],
    output: TileTensor[
        mut=True, output_type, output_tt_layout, address_space=.GENERIC, ...
    ],
    scale: Float32,
    batch_size: Int,
    seq_len: Int,
    num_keys: Int,
    num_heads: Int,
    depth: Int,
    group: Int,
    ctx: DeviceContext,
    sink_weights: OptionalReg[ImmutTileTensor1D[q_type]] = None,
) raises:
    """Computes reference GPU attention with a materialized mask.

    Parameters:
        q_type: Element type of the query tensor.
        k_type: Element type of the key tensor.
        v_type: Element type of the value tensor.
        output_type: Element type of the output tensor.
        mask_type: Element type of the dense attention mask tensor.
        q_tt_layout: Compile-time `TensorLayout` of the query tensor.
        k_tt_layout: Compile-time `TensorLayout` of the key tensor.
        v_tt_layout: Compile-time `TensorLayout` of the value tensor.
        mask_tt_layout: Compile-time `TensorLayout` of the mask tensor.
        output_tt_layout: Compile-time `TensorLayout` of the output tensor.
        sink: `True` to enable attention-sink mode (defaults to `False`).

    Args:
        q: Query `TileTensor` with BSHD layout.
        k: Key `TileTensor`.
        v: Value `TileTensor`.
        mask: Dense attention mask `TileTensor`.
        output: Mutable output `TileTensor`.
        scale: Softmax temperature scale applied to Q·Kᵀ.
        batch_size: Number of sequences in the batch.
        seq_len: Maximum query sequence length in the batch.
        num_keys: Maximum key/value sequence length.
        num_heads: Number of query heads.
        depth: Attention head depth (key/value dimension per head).
        group: GQA group size (query heads per key/value head).
        ctx: GPU device context for kernel dispatch.
        sink_weights: Optional sink-token weight tensor for attention sinks.
    """
    mha_gpu_naive[sink=sink](
        q,
        k,
        v,
        MaterializedMask(
            TileTensor(mask.unsafe_ptr(), row_major(mask.layout.shape_coord()))
        ),
        output,
        scale,
        batch_size,
        seq_len,
        num_keys,
        num_heads,
        depth,
        group,
        ctx,
        sink_weights,
    )


def mha_gpu_naive[
    q_type: DType,
    k_type: DType,
    v_type: DType,
    output_type: DType,
    MaskType: MHAMask,
    q_tt_layout: TensorLayout,
    k_tt_layout: TensorLayout,
    v_tt_layout: TensorLayout,
    output_tt_layout: TensorLayout,
    //,
    sink: Bool = False,
](
    q: TileTensor[mut=False, q_type, q_tt_layout, address_space=.GENERIC, ...],
    k: TileTensor[mut=False, k_type, k_tt_layout, address_space=.GENERIC, ...],
    v: TileTensor[mut=False, v_type, v_tt_layout, address_space=.GENERIC, ...],
    mask: MaskType,
    output: TileTensor[
        mut=True, output_type, output_tt_layout, address_space=.GENERIC, ...
    ],
    scale: Float32,
    batch_size: Int,
    seq_len: Int,
    num_keys: Int,
    num_heads: Int,
    depth: Int,
    group: Int,
    ctx: DeviceContext,
    sink_weights: OptionalReg[ImmutTileTensor1D[q_type]] = None,
) raises:
    """Computes reference GPU attention with an `MHAMask` functor.

    Parameters:
        q_type: Element type of the query tensor.
        k_type: Element type of the key tensor.
        v_type: Element type of the value tensor.
        output_type: Element type of the output tensor.
        MaskType: Attention mask type implementing `MHAMask`.
        q_tt_layout: Compile-time `TensorLayout` of the query tensor.
        k_tt_layout: Compile-time `TensorLayout` of the key tensor.
        v_tt_layout: Compile-time `TensorLayout` of the value tensor.
        output_tt_layout: Compile-time `TensorLayout` of the output tensor.
        sink: `True` to enable attention-sink mode (defaults to `False`).

    Args:
        q: Query `TileTensor` with BSHD layout.
        k: Key `TileTensor`.
        v: Value `TileTensor`.
        mask: Mask instance used to apply the attention mask.
        output: Mutable output `TileTensor`.
        scale: Softmax temperature scale applied to Q·Kᵀ.
        batch_size: Number of sequences in the batch.
        seq_len: Maximum query sequence length in the batch.
        num_keys: Maximum key/value sequence length.
        num_heads: Number of query heads.
        depth: Attention head depth (key/value dimension per head).
        group: GQA group size (query heads per key/value head).
        ctx: GPU device context for kernel dispatch.
        sink_weights: Optional sink-token weight tensor for attention sinks.
    """
    var k_operand = TileTensorMHAOperand(
        TileTensor(k.unsafe_ptr(), row_major(k.layout.shape_coord()))
    )
    var v_operand = TileTensorMHAOperand(
        TileTensor(v.unsafe_ptr(), row_major(v.layout.shape_coord()))
    )
    mha_gpu_naive[_is_cache_length_accurate=True, sink=sink](
        q,
        k_operand,
        v_operand,
        mask,
        output,
        None,
        scale,
        batch_size,
        seq_len,
        num_keys,
        num_heads,
        depth,
        group,
        ctx,
        sink_weights,
    )


def mha_gpu_naive[
    q_type: DType,
    output_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    q_tt_layout: TensorLayout,
    output_tt_layout: TensorLayout,
    valid_length_tt_layout: TensorLayout,
    //,
    ragged: Bool = False,
    sink: Bool = False,
    _use_valid_length: Bool = False,
    _is_cache_length_accurate: Bool = False,
](
    q: TileTensor[mut=False, q_type, q_tt_layout, address_space=.GENERIC, ...],
    k: k_t,
    v: v_t,
    mask_functor: mask_t,
    output: TileTensor[
        mut=True, output_type, output_tt_layout, address_space=.GENERIC, ...
    ],
    valid_length: TileTensor[
        mut=False,
        .uint32,
        valid_length_tt_layout,
        address_space=.GENERIC,
        ...,
    ],
    scale: Float32,
    batch_size: Int,
    max_prompt_len: Int,
    max_cache_size: Int,
    num_heads: Int,
    depth: Int,
    group: Int,
    ctx: DeviceContext,
    sink_weights: OptionalReg[ImmutTileTensor1D[q_type]] = None,
) raises:
    """Computes reference GPU attention with `MHAOperand` key/value.

    Parameters:
        q_type: Element type of the query tensor.
        output_type: Element type of the attention output.
        k_t: Key operand type (dense or KV-cache).
        v_t: Value operand type (dense or KV-cache).
        mask_t: Attention mask type.
        q_tt_layout: Compile-time `TensorLayout` of the query tensor.
        output_tt_layout: Compile-time `TensorLayout` of the output tensor.
        valid_length_tt_layout: Compile-time `TensorLayout` of the valid-length
            tensor.
        ragged: `True` for ragged-batch inputs.
        sink: `True` to enable attention-sink mode.
        _use_valid_length: `True` to honour per-sequence valid lengths.
        _is_cache_length_accurate: `True` when the cache length already
            excludes the current prompt.

    Args:
        q: Query `TileTensor` with BSHD layout.
        k: Key operand.
        v: Value operand.
        mask_functor: Mask instance used to apply the attention mask.
        output: Mutable output `TileTensor`.
        valid_length: Per-sequence valid lengths as a `TileTensor`.
        scale: Softmax temperature scale applied to Q·Kᵀ.
        batch_size: Number of sequences in the batch.
        max_prompt_len: Maximum query sequence length in the batch.
        max_cache_size: Maximum key/value sequence length.
        num_heads: Number of query heads.
        depth: Attention head depth (key/value dimension per head).
        group: GQA group size (query heads per key/value head).
        ctx: GPU device context for kernel dispatch.
        sink_weights: Optional sink-token weight tensor for attention sinks.
    """
    mha_gpu_naive[
        ragged=ragged,
        sink=sink,
        _use_valid_length=_use_valid_length,
        _is_cache_length_accurate=_is_cache_length_accurate,
    ](
        q,
        k,
        v,
        mask_functor,
        output,
        OptionalReg[ImmutTileTensor1D[.uint32]](
            immut_tile_tensor_1d(valid_length.ptr, valid_length.num_elements())
        ),
        scale,
        batch_size,
        max_prompt_len,
        max_cache_size,
        num_heads,
        depth,
        group,
        ctx,
        sink_weights,
    )


def mha_gpu_naive[
    q_type: DType,
    output_type: DType,
    cache_t: KVCacheT,
    mask_t: MHAMask,
    q_tt_layout: TensorLayout,
    output_tt_layout: TensorLayout,
    valid_length_tt_layout: TensorLayout,
    //,
    ragged: Bool = False,
    sink: Bool = False,
](
    q: TileTensor[mut=False, q_type, q_tt_layout, address_space=.GENERIC, ...],
    k: cache_t,
    v: cache_t,
    mask_functor: mask_t,
    output: TileTensor[
        mut=True, output_type, output_tt_layout, address_space=.GENERIC, ...
    ],
    valid_length: TileTensor[
        mut=False,
        .uint32,
        valid_length_tt_layout,
        address_space=.GENERIC,
        ...,
    ],
    scale: Float32,
    batch_size: Int,
    max_prompt_len: Int,
    max_cache_size: Int,
    num_heads: Int,
    depth: Int,
    group: Int,
    ctx: DeviceContext,
    sink_weights: OptionalReg[ImmutTileTensor1D[q_type]] = None,
) raises:
    """Computes reference GPU attention with KV-cache key/value.

    Wraps the cache operands and delegates to the native `MHAOperand` overload.

    Parameters:
        q_type: Element type of the query tensor.
        output_type: Element type of the attention output.
        cache_t: KV-cache type backing the key and value tensors.
        mask_t: Attention mask type.
        q_tt_layout: Compile-time `TensorLayout` of the query tensor.
        output_tt_layout: Compile-time `TensorLayout` of the output tensor.
        valid_length_tt_layout: Compile-time `TensorLayout` of the valid-length
            tensor.
        ragged: `True` for ragged-batch inputs.
        sink: `True` to enable attention-sink mode.

    Args:
        q: Query `TileTensor` with BSHD layout.
        k: Key operand backed by a KV cache.
        v: Value operand backed by a KV cache.
        mask_functor: Mask instance used to apply the attention mask.
        output: Mutable output `TileTensor`.
        valid_length: Per-sequence valid lengths as a `TileTensor`.
        scale: Softmax temperature scale applied to Q·Kᵀ.
        batch_size: Number of sequences in the batch.
        max_prompt_len: Maximum query sequence length in the batch.
        max_cache_size: Maximum key/value sequence length.
        num_heads: Number of query heads.
        depth: Attention head depth (key/value dimension per head).
        group: GQA group size (query heads per key/value head).
        ctx: GPU device context for kernel dispatch.
        sink_weights: Optional sink-token weight tensor for attention sinks.
    """
    var k_operand = KVCacheMHAOperand(k)
    var v_operand = KVCacheMHAOperand(v)

    mha_gpu_naive[
        ragged=ragged,
        _use_valid_length=True,
        _is_cache_length_accurate=False,
        sink=sink,
    ](
        q,
        k_operand,
        v_operand,
        mask_functor,
        output,
        OptionalReg[ImmutTileTensor1D[.uint32]](
            immut_tile_tensor_1d(valid_length.ptr, valid_length.num_elements())
        ),
        scale,
        batch_size,
        max_prompt_len,
        max_cache_size,
        num_heads,
        depth,
        group,
        ctx,
        sink_weights,
    )
