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
"""A/B of the SM100 sparse MLA prefill and decode kernels, as one sweep.

Every arm reads the same Q, paged latent cache and causal top-k index rows,
built the way the indexer packs them (position `p` draws `min(top_k, p + 1)`
distinct keys and pads with -1), at the configurations `mla_graph.mojo`
launches each kernel with, so any kernel can be timed on another's shape.
Before an arm is timed its output is checked against the cell's reference,
and a mismatch fails the run.

Comptime arms, run innermost:

- `-D sweep_cache=1|2|3` selects the FP8, BF16 or packed NVFP4 latent
  cache. FP8 pairs the native-FP8 prefill kernel with FP8 Q, BF16 pairs
  `mla_sm100_prefill_sparse` with BF16 Q, and NVFP4 runs FP8 Q over the
  packed cache. Unset, all three run.
- `-D sweep_h=<heads>` pins the head count; unset runs 64 and 128.

Runtime axes and flags:

- `--cells=b:s:c,...`: the cells, by batch, query tokens and cached tokens.
  The default is prompts and decode-shaped batches, each on an empty and a
  long context.
- `--kernels=`: the arms to time, of `prefill`, `decode` (plain sparse
  decode), `skip` (decode with `skip_masked_tiles`, the prefill-on-decode
  route), `route` (whichever of the two the graph runs: `skip` past 8 query
  tokens) and `convert` (NVFP4's convert-once route). Default
  `prefill,decode,skip`.
- `--np=0,...`: split-K counts the decode arms take; 0 keeps the
  dispatch's own. A forced count reaches the grid and the kernel alike.
- `--preset=np_sweep`: the forced split-K grid -- `route` over every count
  on the cells a split policy is graded on.
- `--graph_repeats=64`: launches per captured graph; 0 times eager launches.
  `--verify_capture=1` also checks that replays reproduce an eager launch.
- `--check=1|0|report`: gate on agreement (default), skip it, or report
  without failing.
- `--launch_only=1`: launch each arm once, untimed and unchecked, for
  `MLA_SPARSE_TRACE_LAUNCH` builds.

The reference is the prefill kernel, or for NVFP4 the FP8 prefill kernel
over the FP8 mirror of the packed cache, which is its exact dequantization.
Kernel A/B knobs (`MLA_SPARSE_CG2`, `MLA_SMX_*`, ...) are `-D` defines, one
arm target each in `BUILD.bazel`. With `-D MLA_STAMP=1` every decode arm
dumps its clock stamps before it is timed. The measurement flags
(`--target_time`, `--json`, ...) are `SweepHarness`'s.
"""

from std.math import ceildiv
from std.sys import get_defined_int

from bench_sweep import (
    Config,
    Fields,
    Footprint,
    PagedBlocks,
    PageTable,
    SweepHarness,
    arm_enabled,
    device_budget,
    fill_affine,
    fill_sparse_indices,
    fill_uniform,
    plan_groups,
)
from internal_utils import arg_parse
from kv_cache.types import (
    KVCacheStaticParams,
    KVCacheT,
    PagedKVCacheCollection,
)
from layout import Coord, Idx, TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext
from mla_sweep import (
    PACKED_ROW,
    QUANT_GRANULARITY,
    SF_PER_ROW,
    PackedLatent,
    cells_from_arg,
    check_agreement,
    dump_stamps,
    fill_topk_lengths,
    force_split,
    interleave_index_rows,
    parse_names,
    verify_capture,
)
from nn.attention.gpu.mla import flare_mla_decoding
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
)
from nn.attention.gpu.nvidia.sm100.mla_nvfp4_kv_to_fp8 import (
    NVFP4PrefillWorkspace,
    NVFP4PrefillWorkspaceView,
    mla_prefill_sparse_nvfp4_via_fp8,
    nvfp4_prefill_workspace,
)
from nn.attention.gpu.nvidia.sm100.mla_prefill import (
    mla_sm100_prefill_sparse,
)
from nn.attention.gpu.nvidia.sm100.mla_prefill_sparse_qkv_fp8 import (
    mla_prefill_sparse_qkv_fp8,
)
from nn.attention.gpu.nvidia.sm100.mla_prefill_sparse_utils import (
    MLASparseConfig,
)
from nn.attention.mha_mask import CausalMask
from nn.attention.mha_utils import MHAConfig, NullPointer
from std.benchmark import BenchId
from nn.attention.gpu.mla import SM100MLA

comptime fp8 = DType.float8_e4m3fn
comptime bf16 = DType.bfloat16
comptime KV_LATENT_DIM = 512
comptime ROPE_DEPTH = 64
comptime Q_DEPTH = KV_LATENT_DIM + ROPE_DEPTH
comptime V_DEPTH = KV_LATENT_DIM
comptime PAGE_SIZE = get_defined_int["page_size", 128]()
comptime TOP_K = get_defined_int["top_k", 2048]()
comptime SCALE = Float32(0.125)
# The most query tokens the graph still sends to plain decode.
comptime MAX_FOLD_Q = 8

# Cache formats, the values of `-D sweep_cache`.
comptime CACHE_FP8 = 1
comptime CACHE_BF16 = 2
comptime CACHE_NVFP4 = 3

comptime PINNED_HEADS = get_defined_int["sweep_h", 0]()
comptime MAX_HEADS = PINNED_HEADS if PINNED_HEADS > 0 else 128

comptime DEFAULT_CELLS = (
    "1:8192:0,1:8192:32768,4:2048:0,4:2048:32768,4:6:0,4:6:32768,"
    "8:6:0,8:6:32768,32:6:0,32:6:32768"
)
comptime NP_SWEEP_CELLS = (
    "1:1:32768,4:1:32768,8:1:32768,16:1:32768,32:1:32768,64:1:32768,"
    "1:2:32768,4:2:32768,8:2:32768,16:2:32768,32:2:32768,"
    "1:6:32768,4:6:32768,8:6:32768,16:6:32768,32:6:32768,"
    "1:16:0,1:32:0,4:16:0,1:16:32768,1:32:32768,4:16:32768"
)


def heads_enabled[heads: Int]() -> Bool:
    return PINNED_HEADS == heads or (PINNED_HEADS == 0 and heads >= 64)


def cache_name[cache: Int]() -> String:
    comptime if cache == CACHE_FP8:
        return "fp8"
    elif cache == CACHE_BF16:
        return "bf16"
    return "nvfp4"


@fieldwise_init
struct Options(Copyable, Movable):
    """The runtime flags every cell reads."""

    var kernels: List[String]
    var nps: List[Int]
    var graph_repeats: Int
    var verify_capture: Bool
    var check: Bool
    var raise_on_mismatch: Bool
    var launch_only: Bool

    def wants(self, kernel: String) -> Bool:
        for ref k in self.kernels:
            if k == kernel:
                return True
        return False


struct Workspace(Movable):
    """Per-token buffers, sized for the largest cell of a sweep group."""

    var q_fp8: DeviceBuffer[fp8]
    var q_bf16: DeviceBuffer[bf16]
    var out_ref: DeviceBuffer[bf16]
    var out: DeviceBuffer[bf16]
    var indices: DeviceBuffer[DType.int32]
    var scale_indices: DeviceBuffer[DType.int32]
    var nvfp4_indices: DeviceBuffer[DType.int32]
    var topk_lengths: DeviceBuffer[DType.uint32]
    var row_offsets: DeviceBuffer[DType.uint32]
    var scale_lut: DeviceBuffer[DType.uint32]

    def __init__(out self, ctx: DeviceContext[], footprint: Footprint) raises:
        var tokens = footprint["tokens"]
        var q_elems = tokens * MAX_HEADS * Q_DEPTH
        var slots = tokens * TOP_K
        comptime nvfp4 = arm_enabled["sweep_cache", CACHE_NVFP4]()
        self.q_fp8 = ctx.enqueue_create_buffer[fp8](q_elems)
        self.q_bf16 = ctx.enqueue_create_buffer[bf16](
            q_elems if arm_enabled["sweep_cache", CACHE_BF16]() else 1
        )
        self.out_ref = ctx.enqueue_create_buffer[bf16](
            tokens * MAX_HEADS * V_DEPTH
        )
        self.out = ctx.enqueue_create_buffer[bf16](tokens * MAX_HEADS * V_DEPTH)
        self.indices = ctx.enqueue_create_buffer[DType.int32](slots)
        self.scale_indices = ctx.enqueue_create_buffer[DType.int32](
            slots if nvfp4 else 1
        )
        self.nvfp4_indices = ctx.enqueue_create_buffer[DType.int32](
            2 * slots if nvfp4 else 1
        )
        self.topk_lengths = ctx.enqueue_create_buffer[DType.uint32](tokens)
        self.row_offsets = ctx.enqueue_create_buffer[DType.uint32](
            footprint["batch"] + 1
        )
        self.scale_lut = ctx.enqueue_create_buffer[DType.uint32](
            footprint["lut_entries"] if nvfp4 else 1
        )
        fill_uniform(ctx, self.q_fp8, 10)
        comptime if arm_enabled["sweep_cache", CACHE_BF16]():
            fill_uniform(ctx, self.q_bf16, 10)

    @staticmethod
    def bytes_for(footprint: Footprint) -> Int:
        var tokens = footprint["tokens"]
        var q_bytes = tokens * MAX_HEADS * Q_DEPTH
        comptime if arm_enabled["sweep_cache", CACHE_BF16]():
            q_bytes *= 3
        var index_words = tokens * TOP_K
        comptime if arm_enabled["sweep_cache", CACHE_NVFP4]():
            index_words *= 4
        return (
            q_bytes
            + 4 * tokens * MAX_HEADS * V_DEPTH
            + 4 * (index_words + tokens + footprint["batch"] + 1)
            + 4 * footprint["lut_entries"]
        )


struct Cell(ImplicitlyCopyable):
    """One cell's shape and the values derived from it."""

    var batch_size: Int
    var seq_len: Int
    var cache_len: Int
    var tokens: Int
    var pages: Int

    def __init__(out self, cfg: Config) raises:
        self.batch_size = cfg["batch_size"]
        self.seq_len = cfg["seq_len"]
        self.cache_len = cfg["cache_len"]
        self.tokens = self.batch_size * self.seq_len
        self.pages = ceildiv(self.cache_len + self.seq_len, PAGE_SIZE)


def _view[
    dtype: DType
](
    ctx: DeviceContext, buf: DeviceBuffer[dtype, ...], n: Int
) raises -> DeviceBuffer[dtype]:
    return DeviceBuffer[dtype](ctx, buf.unsafe_ptr(), n, owning=False)


@inline(.always)
def _dev[
    dtype: DType
](buf: DeviceBuffer[dtype, ...]) -> Pointer[Scalar[dtype], MutAnyOrigin]:
    """Returns `buf`'s device pointer, untied from the workspace that owns
    it, so a launch closure and the workspace can be passed side by side."""
    return (
        buf.unsafe_ptr()
        .unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutAnyOrigin]()
    )


def _bench_name[
    cache: Int, heads: Int
](kernel: String, cell: Cell, num_partitions: Int) -> String:
    # fmt: off
    return String(
        "mla_sparse_", kernel, " : ",
        "cache=", cache_name[cache](), ", ",
        "num_heads=", heads, ", ",
        "top_k=", TOP_K, " : ",
        "batch_size=", cell.batch_size, ", ",
        "seq_len=", cell.seq_len, ", ",
        "cache_len=", cell.cache_len, ", ",
        "np=", num_partitions,
    )
    # fmt: on


def run_arm[
    LaunchFn: def(DeviceContext[]) raises -> None,
    //,
    cache: Int,
    heads: Int,
](
    ctx: DeviceContext[],
    mut h: SweepHarness,
    launch: LaunchFn,
    kernel: String,
    cell: Cell,
    num_partitions: Int,
    ws: Workspace,
    opts: Options,
    is_reference: Bool,
) raises:
    """Checks one arm against the cell's reference, then times it.

    The reference arm writes `ws.out_ref`, every other arm `ws.out`.
    """
    var name = _bench_name[cache, heads](kernel, cell, num_partitions)
    if opts.launch_only:
        print("BENCH_LAUNCH", name)
        launch(ctx)
        ctx.synchronize()
        return
    var count = cell.tokens * heads * V_DEPTH
    if not is_reference:
        if kernel == "decode" or kernel == "skip" or kernel == "route":
            dump_stamps(
                ctx,
                launch,
                String(
                    kernel,
                    "_",
                    cache_name[cache](),
                    "_h",
                    heads,
                    "_b",
                    cell.batch_size,
                    "_s",
                    cell.seq_len,
                    "_c",
                    cell.cache_len,
                    "_np",
                    num_partitions,
                ),
            )
        if opts.check:
            launch(ctx)
            ctx.synchronize()
            _ = check_agreement(
                ctx,
                ws.out_ref,
                ws.out,
                count,
                name,
                raise_on_mismatch=opts.raise_on_mismatch,
            )
    if opts.verify_capture:
        verify_capture(
            ctx,
            launch,
            ws.out_ref if is_reference else ws.out,
            count,
            opts.graph_repeats,
        )
    h.timer.reset()
    var fields = Fields()
    fields.add("kernel", kernel)
    fields.add("cache", cache_name[cache]())
    fields.add("num_heads", heads)
    fields.add("page_size", PAGE_SIZE)
    fields.add("top_k", TOP_K)
    fields.add("batch_size", cell.batch_size)
    fields.add("seq_len", cell.seq_len)
    fields.add("cache_len", cell.cache_len)
    fields.add("np", num_partitions)
    h.bench_graph(ctx, launch, BenchId(name), fields^, opts.graph_repeats)


@inline(.always)
def _decode[
    cache_t: KVCacheT,
    q_dtype: DType,
    //,
    heads: Int,
    skip: Bool,
](
    output: TileTensor[mut=True, address_space=.GENERIC, ...],
    q: TileTensor[q_dtype, address_space=.GENERIC, ...],
    cache: cache_t,
    row_offsets: TileTensor[.uint32, address_space=.GENERIC, ...],
    scalars: TileTensor[.int64, address_space=.GENERIC, ...],
    indices: MutPointer[Int32, MutAnyOrigin],
    indices_stride: Int,
    num_partitions: Int,
    ctx: DeviceContext,
) raises:
    flare_mla_decoding[
        rank=3,
        config=MHAConfig[q_dtype](heads, Q_DEPTH),
        ragged=True,
        sparse=True,
        skip_masked_tiles=skip,
        kernels=SM100MLA,
    ](
        output,
        q,
        cache,
        CausalMask(),
        row_offsets,
        SCALE,
        ctx,
        scalars,
        d_indices=indices,
        indices_stride=indices_stride,
        num_partitions_in=Optional[Int](num_partitions) if num_partitions
        > 0 else None,
    )


def run_decode_arms[
    cache_t: KVCacheT,
    q_dtype: DType,
    //,
    cache: Int,
    heads: Int,
](
    ctx: DeviceContext[],
    mut h: SweepHarness,
    kv_cache: cache_t,
    q_tt: TileTensor[q_dtype, address_space=.GENERIC, ...],
    indices: MutPointer[Int32, MutAnyOrigin],
    indices_stride: Int,
    cell: Cell,
    ws: Workspace,
    opts: Options,
) raises:
    """Runs the decode-kernel arms the options ask for, at every split."""
    var out_tt = TileTensor(
        _dev(ws.out),
        row_major((cell.tokens, Idx[heads], Idx[V_DEPTH])),
    )
    var row_offsets_tt = TileTensor(
        _dev(ws.row_offsets), row_major(cell.batch_size + 1)
    )
    for np in opts.nps:
        var args = MLADispatchScalarArgs[
            num_heads=heads, is_fp8_kv=cache != CACHE_BF16
        ](cell.batch_size, cell.cache_len, cell.seq_len, ctx)
        if np > 0:
            force_split(ctx, args.gpu_buf, cell.batch_size, cell.seq_len, np)
        var scalars_tt = args.gpu_tile_tensor()

        @inline(.always)
        def decode_launch(launch_ctx: DeviceContext[]) raises {mut out_tt, imm}:
            _decode[heads, False](
                out_tt,
                q_tt,
                kv_cache,
                row_offsets_tt,
                scalars_tt,
                indices,
                indices_stride,
                np,
                launch_ctx,
            )

        @inline(.always)
        def skip_launch(launch_ctx: DeviceContext[]) raises {mut out_tt, imm}:
            _decode[heads, True](
                out_tt,
                q_tt,
                kv_cache,
                row_offsets_tt,
                scalars_tt,
                indices,
                indices_stride,
                np,
                launch_ctx,
            )

        if opts.wants("decode"):
            run_arm[cache, heads](
                ctx, h, decode_launch, "decode", cell, np, ws, opts, False
            )
        if opts.wants("skip"):
            run_arm[cache, heads](
                ctx, h, skip_launch, "skip", cell, np, ws, opts, False
            )
        if opts.wants("route"):
            if cell.seq_len > MAX_FOLD_Q:
                run_arm[cache, heads](
                    ctx, h, skip_launch, "route", cell, np, ws, opts, False
                )
            else:
                run_arm[cache, heads](
                    ctx, h, decode_launch, "route", cell, np, ws, opts, False
                )
        _ = args^


# The native-FP8 prefill tile the graph's sparse prefill branches select.
comptime PrefillConfig[heads: Int] = MLASparseConfig[
    DType.bfloat16,
    b_topk_=128 if heads == 128 else 64,
    num_mbars_=2 if heads == 128 else 4,
    cta_group_=2 if heads == 128 else 1,
]


def _prefill_config[heads: Int, indices_stride: Int]() -> PrefillConfig[heads]:
    return PrefillConfig[heads](
        num_q_heads=heads,
        num_kv_heads=1,
        qk_depth=Q_DEPTH,
        input_qk_depth=Q_DEPTH,
        v_depth=V_DEPTH,
        indices_stride=indices_stride,
        group=heads,
    )


def run_dense[
    latent_dtype: DType, //, cache: Int, heads: Int
](
    ctx: DeviceContext[],
    mut h: SweepHarness,
    table: PageTable[PAGE_SIZE],
    mut latent: PagedBlocks[latent_dtype],
    cell: Cell,
    mut ws: Workspace,
    opts: Options,
) raises:
    """Runs a cell over an FP8 or BF16 cache, with Q in the cache's dtype."""
    comptime kv_dtype = latent_dtype
    comptime assert kv_dtype == (fp8 if cache == CACHE_FP8 else bf16)
    comptime kv_params = KVCacheStaticParams(
        num_heads=1, head_size=Q_DEPTH, is_mla=True
    )
    comptime Collection = PagedKVCacheCollection[
        kv_dtype,
        kv_params,
        PAGE_SIZE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    comptime blocks_layout = Collection.blocks_tt_layout
    var blocks_shape = Coord[*blocks_layout.shape_types]()
    blocks_shape[0] = Int64(table.num_blocks)
    blocks_shape[2] = Int64(1)
    var blocks_strides = Coord[*blocks_layout.stride_types]()
    blocks_strides[1] = blocks_shape[2] * Int64(blocks_strides[2].value())
    blocks_strides[0] = Int64(blocks_shape[1].value()) * blocks_strides[1]
    var lut = table.copy_lut(ctx, 0, cell.batch_size, cell.pages)
    var kv_collection = Collection(
        TileTensor(
            latent.buf, blocks_layout(blocks_shape, blocks_strides)
        ).as_unsafe_any_origin(),
        TileTensor(table.cache_lengths, row_major(Int64(cell.batch_size)))
        .as_imm()
        .as_unsafe_any_origin(),
        TileTensor(lut, row_major(Int64(cell.batch_size), Int64(cell.pages)))
        .as_imm()
        .as_unsafe_any_origin(),
        UInt32(cell.seq_len),
        UInt32(cell.cache_len),
    )
    var kv_cache = kv_collection.get_key_cache(0)

    var q_ptr: Pointer[Scalar[kv_dtype], MutAnyOrigin]
    comptime if cache == CACHE_FP8:
        q_ptr = rebind[Pointer[Scalar[kv_dtype], MutAnyOrigin]](_dev(ws.q_fp8))
    else:
        q_ptr = rebind[Pointer[Scalar[kv_dtype], MutAnyOrigin]](_dev(ws.q_bf16))
    var q_tt = TileTensor(
        q_ptr, row_major((cell.tokens, Idx[heads], Idx[Q_DEPTH]))
    )
    var ref_tt = TileTensor(
        _dev(ws.out_ref),
        row_major((cell.tokens, Idx[heads], Idx[V_DEPTH])),
    )
    # int32 slots reinterpreted as uint32, as the graph does: -1 becomes
    # 0xFFFFFFFF, which the gather rejects.
    var indices_tt = TileTensor(
        _dev(ws.indices).unsafe_bitcast[UInt32](),
        row_major(cell.tokens * TOP_K),
    )
    var topk_lengths_tt = TileTensor(
        _dev(ws.topk_lengths), row_major(cell.tokens)
    )
    comptime config = _prefill_config[heads, TOP_K]()

    @inline(.always)
    def reference_launch(launch_ctx: DeviceContext[]) raises {mut ref_tt, imm}:
        comptime if cache == CACHE_FP8:
            mla_prefill_sparse_qkv_fp8[
                config=config,
                group=heads,
                q_depth=Q_DEPTH,
                scale_block_size=0,
            ](
                ref_tt,
                q_tt,
                kv_cache,
                indices_tt,
                topk_lengths_tt,
                None,
                SCALE,
                Int32(TOP_K),
                launch_ctx,
            )
        else:
            mla_sm100_prefill_sparse[
                num_q_heads=heads,
                qk_depth=Q_DEPTH,
                v_depth=V_DEPTH,
                indices_stride=TOP_K,
            ](
                ref_tt,
                q_tt,
                kv_cache,
                indices_tt,
                topk_lengths_tt,
                None,
                SCALE,
                launch_ctx,
            )

    if opts.check:
        reference_launch(ctx)
        ctx.synchronize()
    h.timer.lap("config_setup")
    if opts.wants("prefill"):
        run_arm[cache, heads](
            ctx, h, reference_launch, "prefill", cell, 0, ws, opts, True
        )
    run_decode_arms[cache, heads](
        ctx,
        h,
        kv_cache,
        q_tt,
        _dev(ws.indices),
        TOP_K,
        cell,
        ws,
        opts,
    )
    _ = lut^


def run_nvfp4[
    heads: Int
](
    ctx: DeviceContext[],
    mut h: SweepHarness,
    table: PageTable[PAGE_SIZE],
    mut packed: PackedLatent,
    cell: Cell,
    mut ws: Workspace,
    opts: Options,
) raises:
    """Runs a cell over the packed NVFP4 cache, with FP8 Q.

    The reference is the FP8 prefill kernel over the FP8 mirror; `prefill`
    is the same kernel's NVFP4 gather over the packed cache.
    """
    comptime packed_params = KVCacheStaticParams(
        num_heads=1,
        head_size=PACKED_ROW,
        is_mla=True,
        fp8_tail_size=ROPE_DEPTH,
    )
    comptime PackedCollection = PagedKVCacheCollection[
        DType.uint8,
        packed_params,
        PAGE_SIZE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
        scale_dtype_=fp8,
        quantization_granularity_=QUANT_GRANULARITY,
    ]
    comptime mirror_params = KVCacheStaticParams(
        num_heads=1, head_size=Q_DEPTH, is_mla=True
    )
    comptime MirrorCollection = PagedKVCacheCollection[
        fp8,
        mirror_params,
        PAGE_SIZE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var lut = table.copy_lut(ctx, 0, cell.batch_size, cell.pages)
    var cache_lengths_tt = (
        TileTensor(table.cache_lengths, row_major(Int64(cell.batch_size)))
        .as_imm()
        .as_unsafe_any_origin()
    )
    var lut_tt = (
        TileTensor(lut, row_major(Int64(cell.batch_size), Int64(cell.pages)))
        .as_imm()
        .as_unsafe_any_origin()
    )
    var scale_lut_tt = (
        TileTensor(
            ws.scale_lut,
            row_major(Int64(cell.batch_size), Int64(cell.pages)),
        )
        .as_imm()
        .as_unsafe_any_origin()
    )
    comptime packed_layout = PackedCollection.blocks_tt_layout
    var packed_shape = Coord[*packed_layout.shape_types]()
    packed_shape[0] = Int64(table.num_blocks)
    packed_shape[2] = Int64(1)
    var packed_strides = Coord[*packed_layout.stride_types]()
    packed_strides[1] = packed_shape[2] * Int64(packed_strides[2].value())
    packed_strides[0] = Int64(packed_shape[1].value()) * packed_strides[1]
    var packed_blocks = TileTensor(
        packed.packed, packed_layout(packed_shape, packed_strides)
    ).as_unsafe_any_origin()
    var scales = TileTensor(
        packed.scales,
        row_major(
            Int64(table.num_blocks),
            Idx[1],
            Int64(1),
            Idx[PAGE_SIZE],
            Idx[1],
            Idx[SF_PER_ROW],
        ),
    ).as_unsafe_any_origin()

    def packed_cache(
        max_context: Int,
    ) raises {imm} -> PackedCollection.CacheType:
        return PackedCollection(
            packed_blocks,
            cache_lengths_tt,
            lut_tt,
            UInt32(cell.seq_len),
            UInt32(max_context),
            rebind[PackedCollection.scales_tt_type](scales),
            scale_lut_tt,
        ).get_key_cache(0)

    var kv_cache = packed_cache(cell.cache_len)
    # The convert-once route sizes its scratch from the context after this
    # step's tokens, as the graph passes it.
    var conversion_cache = packed_cache(cell.cache_len + cell.seq_len)
    var mirror_blocks = TileTensor(
        packed.mirror,
        row_major(
            Int64(table.num_blocks),
            Idx[1],
            Int64(1),
            Idx[PAGE_SIZE],
            Idx[1],
            Idx[Q_DEPTH],
        ),
    ).as_unsafe_any_origin()
    var mirror_cache = MirrorCollection(
        rebind[MirrorCollection.blocks_tt_type](mirror_blocks),
        cache_lengths_tt,
        lut_tt,
        UInt32(cell.seq_len),
        UInt32(cell.cache_len),
    ).get_key_cache(0)

    var q_tt = TileTensor(
        _dev(ws.q_fp8),
        row_major((cell.tokens, Idx[heads], Idx[Q_DEPTH])),
    )
    var ref_tt = TileTensor(
        _dev(ws.out_ref),
        row_major((cell.tokens, Idx[heads], Idx[V_DEPTH])),
    )
    var out_tt = TileTensor(
        _dev(ws.out),
        row_major((cell.tokens, Idx[heads], Idx[V_DEPTH])),
    )
    var row_offsets_tt = TileTensor(
        _dev(ws.row_offsets), row_major(cell.batch_size + 1)
    )
    var mirror_indices_tt = TileTensor(
        _dev(ws.indices).unsafe_bitcast[UInt32](),
        row_major(cell.tokens * TOP_K),
    )
    var indices_tt = TileTensor(
        _dev(ws.nvfp4_indices).unsafe_bitcast[UInt32](),
        row_major(cell.tokens * 2 * TOP_K),
    )
    var indices = _dev(ws.nvfp4_indices)
    var topk_lengths_tt = TileTensor(
        _dev(ws.topk_lengths), row_major(cell.tokens)
    )
    # Both gathers take their row width at run time, which the NVFP4 one
    # needs and which keeps the reference valid at any `top_k`.
    comptime gather_config = _prefill_config[heads, 0]()
    comptime convert_config = _prefill_config[heads, 2 * TOP_K]()

    @inline(.always)
    def reference_launch(launch_ctx: DeviceContext[]) raises {mut ref_tt, imm}:
        mla_prefill_sparse_qkv_fp8[
            config=gather_config,
            group=heads,
            q_depth=Q_DEPTH,
            scale_block_size=0,
        ](
            ref_tt,
            q_tt,
            mirror_cache,
            mirror_indices_tt,
            topk_lengths_tt,
            None,
            SCALE,
            Int32(TOP_K),
            launch_ctx,
        )

    @inline(.always)
    def prefill_launch(launch_ctx: DeviceContext[]) raises {mut out_tt, imm}:
        mla_prefill_sparse_qkv_fp8[
            config=gather_config,
            group=heads,
            q_depth=Q_DEPTH,
            scale_block_size=0,
        ](
            out_tt,
            q_tt,
            kv_cache,
            indices_tt,
            topk_lengths_tt,
            None,
            SCALE,
            Int32(2 * TOP_K),
            launch_ctx,
        )

    if opts.check:
        reference_launch(ctx)
        ctx.synchronize()
    h.timer.lap("config_setup")
    if opts.wants("prefill"):
        run_arm[CACHE_NVFP4, heads](
            ctx, h, prefill_launch, "prefill", cell, 0, ws, opts, False
        )
    run_decode_arms[CACHE_NVFP4, heads](
        ctx, h, kv_cache, q_tt, indices, 2 * TOP_K, cell, ws, opts
    )
    if opts.wants("convert"):
        var workspace = nvfp4_prefill_workspace[config=convert_config](
            q_tt, conversion_cache, row_offsets_tt, ctx
        )
        var view = Optional[NVFP4PrefillWorkspaceView](workspace.view())
        for np in opts.nps:

            @inline(.always)
            def convert_launch(
                launch_ctx: DeviceContext[],
            ) raises {mut out_tt, imm}:
                mla_prefill_sparse_nvfp4_via_fp8[
                    config=convert_config,
                    group=heads,
                    q_depth=Q_DEPTH,
                    causal=True,
                ](
                    out_tt,
                    q_tt,
                    conversion_cache,
                    indices,
                    2 * TOP_K,
                    topk_lengths_tt,
                    row_offsets_tt,
                    NullPointer[DType.float32](),
                    SCALE,
                    launch_ctx,
                    num_partitions_in=Optional[Int](np) if np > 0 else None,
                    workspace=view,
                )

            run_arm[CACHE_NVFP4, heads](
                ctx, h, convert_launch, "convert", cell, np, ws, opts, False
            )
        ctx.synchronize()
        _ = workspace^
    _ = lut^


def setup_cell[
    cache: Int
](
    ctx: DeviceContext[],
    mut table: PageTable[PAGE_SIZE],
    packed: PackedLatent,
    cell: Cell,
    mut ws: Workspace,
) raises:
    """Writes the cell's page table, index rows, lengths and row offsets."""
    var slots = cell.tokens * TOP_K
    table.set_cache_len(ctx, cell.cache_len)
    table.write_copies(ctx, 1, cell.batch_size, cell.pages)
    var lut = table.copy_lut(ctx, 0, cell.batch_size, cell.pages)
    fill_sparse_indices(
        ctx,
        _view(ctx, ws.indices, slots),
        lut,
        cell.pages,
        PAGE_SIZE,
        True,
        cell.batch_size,
        cell.seq_len,
        cell.cache_len,
        TOP_K,
    )
    fill_topk_lengths(
        ctx,
        ws.topk_lengths,
        cell.batch_size,
        cell.seq_len,
        cell.cache_len,
        TOP_K,
    )
    fill_affine(ctx, ws.row_offsets, cell.seq_len, 0, cell.batch_size + 1)
    comptime if cache == CACHE_NVFP4:
        var entries = cell.batch_size * cell.pages
        packed.scale_lut(ctx, ws.scale_lut, lut, entries)
        fill_sparse_indices(
            ctx,
            _view(ctx, ws.scale_indices, slots),
            _view(ctx, ws.scale_lut, entries),
            cell.pages,
            PAGE_SIZE,
            True,
            cell.batch_size,
            cell.seq_len,
            cell.cache_len,
            TOP_K,
        )
        interleave_index_rows(
            ctx,
            ws.nvfp4_indices,
            ws.indices,
            ws.scale_indices,
            cell.tokens,
            TOP_K,
        )
    ctx.synchronize()


def main() raises:
    var h = SweepHarness("prefill_vs_decode", default_l2="warm")

    var preset = String(arg_parse("preset", ""))
    var np_sweep = preset == "np_sweep"
    if preset.byte_length() > 0 and not np_sweep:
        raise Error("--preset must be np_sweep, not ", preset)
    var configs = cells_from_arg(
        "cells", NP_SWEEP_CELLS if np_sweep else DEFAULT_CELLS
    )
    var nps = List[Int]()
    for n in parse_names("np", "0,1,2,3,4,8,16" if np_sweep else "0"):
        nps.append(Int(n))
    var check = String(arg_parse("check", "1"))
    if check != "1" and check != "0" and check != "report":
        raise Error("--check must be 1, 0 or report, not ", check)
    var opts = Options(
        parse_names("kernels", "route" if np_sweep else "prefill,decode,skip"),
        nps^,
        arg_parse("graph_repeats", 64),
        arg_parse("verify_capture", 0) != 0,
        check != "0",
        check == "1",
        arg_parse("launch_only", 0) != 0,
    )
    for ref k in opts.kernels:
        if not (
            k == "prefill"
            or k == "decode"
            or k == "skip"
            or k == "route"
            or k == "convert"
        ):
            raise Error("unknown kernel in --kernels: ", k)

    h.timer.lap("main_prologue")
    with DeviceContext() as ctx:
        h.timer.lap("device_init")
        h.start(ctx)

        def require(cfg: Config, mut fp: Footprint) raises:
            var cell = Cell(cfg)
            PageTable[PAGE_SIZE].require(
                fp, cell.batch_size, cell.cache_len + cell.seq_len
            )
            fp.need("tokens", cell.tokens)
            fp.need("lut_entries", cell.batch_size * cell.pages)

        def bytes(fp: Footprint) raises -> Int:
            var rows = PageTable[PAGE_SIZE].rows_for(fp)
            var latent = 0
            comptime if arm_enabled["sweep_cache", CACHE_FP8]():
                latent += rows * Q_DEPTH
            comptime if arm_enabled["sweep_cache", CACHE_BF16]():
                latent += 2 * rows * Q_DEPTH
            comptime if arm_enabled["sweep_cache", CACHE_NVFP4]():
                latent += PackedLatent.bytes_for(rows)
            return (
                PageTable[PAGE_SIZE].bytes_for(fp)
                + latent
                + Workspace.bytes_for(fp)
            )

        var groups = plan_groups(configs, require, bytes, device_budget(ctx))
        for ref group in groups:
            ref fp = group.footprint
            var table = PageTable[PAGE_SIZE].from_footprint(ctx, fp)
            var fp8_latent = PagedBlocks[fp8](
                ctx,
                table,
                Q_DEPTH if arm_enabled["sweep_cache", CACHE_FP8]() else 1,
                seed=0,
            )
            var bf16_latent = PagedBlocks[bf16](
                ctx,
                table,
                Q_DEPTH if arm_enabled["sweep_cache", CACHE_BF16]() else 1,
                seed=0,
            )
            var packed = PackedLatent(
                ctx,
                table.rows() if arm_enabled[
                    "sweep_cache", CACHE_NVFP4
                ]() else PAGE_SIZE,
                PAGE_SIZE,
                seed=0,
            )
            ctx.synchronize()
            h.timer.lap("fixture_fill")
            var ws = Workspace(ctx, fp)
            ctx.synchronize()
            h.timer.lap("workspace_fill")
            var info = Fields()
            info.add(
                "fixture_bytes",
                table.bytes()
                + fp8_latent.bytes()
                + bf16_latent.bytes()
                + packed.bytes(),
            )
            info.add("planned_bytes", group.bytes)
            info.extend(fp)
            info.add("configs", len(group.configs))
            h.note(info^)
            for ref cfg in group.configs:
                var cell = Cell(cfg)
                comptime for cache in [CACHE_FP8, CACHE_BF16, CACHE_NVFP4]:
                    comptime if arm_enabled["sweep_cache", cache]():
                        h.timer.reset()
                        setup_cell[cache](ctx, table, packed, cell, ws)
                        comptime for heads in [8, 16, 32, 64, 128]:
                            comptime if heads_enabled[heads]():
                                comptime if cache == CACHE_FP8:
                                    run_dense[cache, heads](
                                        ctx,
                                        h,
                                        table,
                                        fp8_latent,
                                        cell,
                                        ws,
                                        opts,
                                    )
                                elif cache == CACHE_BF16:
                                    run_dense[cache, heads](
                                        ctx,
                                        h,
                                        table,
                                        bf16_latent,
                                        cell,
                                        ws,
                                        opts,
                                    )
                                else:
                                    run_nvfp4[heads](
                                        ctx, h, table, packed, cell, ws, opts
                                    )
            h.timer.reset()
        h.stop(ctx)
    h.timer.lap("device_teardown")
    h.finish()
