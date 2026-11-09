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
"""Sparse MLA prefill through the graph entry points, as one sweep.

Times what a model's prefill step runs, preamble included: RoPE, RMSNorm and
the cache write, the attention, and the W_UV projection. Every cell checks
its arms against a reference before any is timed, and a mismatch fails the
run.

`-D sweep_cache` selects the latent cache, each with its own pair of arms:

- 1, FP8, and 2, BF16: `prefill`, the retained sparse prefill kernel through
  `mla_prefill_branch_sparse_fp8`, is the reference for `graph`, the
  combined op `mla_prefill_decode_graph_fp8`, which runs the sparse decode
  kernel with tile skipping.
- 3, packed NVFP4, on the BF16-weights entry production reaches it through:
  `packed` runs `mla_prefill_decode_graph_bf16` over the packed cache
  (convert once and run the prefill-on-decode kernel, or gather the packed
  rows directly when the context is too large to convert), and `mirror` the
  retained FP8 prefill kernel over the FP8 mirror of the same bytes, which is
  the round's baseline and the reference. The mirror through the graph op
  and the packed cache through the NVFP4 prefill kernel are checked too, so a
  mismatch says whether the route or the cache format differs.

Every packed row is written by the graph's own NVFP4 row writer
(`mla_fused_rope_rmsnorm_quantization`), as in production, before the
mirror is built from it. Each arm then writes the cell's new tokens with
its own writer, so the latent input and RMSNorm weights sit on a grid that
FP8 and NVFP4 both store exactly; otherwise a prompt cell would compare two
different caches. `-D sweep_h` pins the head count; unset runs 64
and 128, the production head counts.

Runtime flags:

- `--cells=b:s:c,...`: batch, query tokens and cached tokens. The default
  is the prompt cells; `--preset=fallback` runs the NVFP4 prefills too large
  to convert once, and `--preset=diag` the one-token shapes that separate
  the decode branch and the selection width.
- `--np=<n>`: force the NVFP4 route's split-K count.
- `--split_break=1`: launch the NVFP4 route with the plain decode split
  instead of the dispatch key's, the wiring before the split was forwarded.
  It must fail the checks.
- `--report=1`: print the checks without failing (the `diag` preset's
  default); `--diag=1` also prints per-row cosines.
"""

from std.math import ceildiv, sqrt
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
    fill_sparse_indices,
    plan_groups,
    uniform_bits,
)
from internal_utils import arg_parse
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Coord, Idx, TileTensor, row_major
from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from mla_sweep import (
    PACKED_ROW,
    QUANT_GRANULARITY,
    SF_PER_ROW,
    PackedLatent,
    cells_from_arg,
    check_agreement,
    fill_topk_lengths,
    interleave_index_rows,
)
from nn.attention.gpu.mla_graph import (
    mla_fused_rope_rmsnorm_quantization,
    mla_prefill_branch_sparse_bf16,
    mla_prefill_branch_sparse_fp8,
    mla_prefill_decode_graph_bf16,
    mla_prefill_decode_graph_fp8,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLA_SPARSE_KV_FORM_NVFP4,
    MLADispatchScalarArgs,
    compute_mla_dispatch_key_runtime,
    compute_mla_dispatch_scalars,
)
from nn.attention.mha_utils import NonNullPointer
from std.benchmark import BenchId
from std.utils import IndexList
from nn.attention.gpu.mla_graph import SM100MLAGraph

comptime fp8 = DType.float8_e4m3fn
comptime bf16 = DType.bfloat16
comptime KV_LATENT_DIM = 512
comptime ROPE_DEPTH = 64
comptime K_CACHE_DIM = KV_LATENT_DIM + ROPE_DEPTH
comptime QK_NOPE_HEAD_DIM = 128
comptime Q_HEAD_DIM = QK_NOPE_HEAD_DIM + ROPE_DEPTH
comptime V_HEAD_DIM = KV_LATENT_DIM
comptime PAGE_SIZE = 128
comptime TOP_K = 2048
comptime SCALE = Float32(0.125)
comptime EPSILON = Float32(1e-6)
comptime M_SCALE_GRAN = 1
comptime N_SCALE_GRAN = 128
comptime K_SCALE_GRAN = 128
comptime W_UK_SCALE_N = KV_LATENT_DIM // N_SCALE_GRAN
comptime W_UK_SCALE_K = ceildiv(QK_NOPE_HEAD_DIM, K_SCALE_GRAN)
comptime W_UV_SCALE_N = V_HEAD_DIM // N_SCALE_GRAN
comptime W_UV_SCALE_K = KV_LATENT_DIM // K_SCALE_GRAN
# Production's convert-once thresholds (`nvfp4_mla_prefill_convert_ratio`):
# convert while the context stays under 150 rows per query token and the
# scratch fits 524288 rows.
comptime NVFP4_CONVERT_RATIO = 150
comptime NVFP4_CONVERT_MAX_ROWS = 524288
# Rows the NVFP4 writer primes per pass; its buffers scale with the count.
comptime PRIME_CHUNK = 4096

# Cache formats, the values of `-D sweep_cache`.
comptime CACHE_FP8 = 1
comptime CACHE_BF16 = 2
comptime CACHE_NVFP4 = 3

comptime PINNED_HEADS = get_defined_int["sweep_h", 0]()
comptime MAX_HEADS = PINNED_HEADS if PINNED_HEADS > 0 else 128

comptime PROMPT_CELLS = "1:8192:0,1:8192:32768,4:2048:0,4:2048:32768"
# NVFP4 prefills whose context is too large to convert once (more than 150
# context rows per query token, or more than 524288 scratch rows).
comptime FALLBACK_CELLS = (
    "1:100:32768,1:218:32768,4:64:32768,64:16:8192,128:16:8192"
)
# One query token over a growing context: the plain decode branch at every
# selection width up to the full window and past it.
comptime DIAG_CELLS = (
    "1:1:512,1:1:1024,1:1:1400,1:1:2048,1:1:4096,1:1:8192,1:1:16384,1:1:32768"
)


def heads_enabled[heads: Int]() -> Bool:
    return PINNED_HEADS == heads or (PINNED_HEADS == 0 and heads >= 64)


def cache_name[cache: Int]() -> String:
    comptime if cache == CACHE_FP8:
        return "fp8"
    elif cache == CACHE_BF16:
        return "bf16"
    return "nvfp4"


def _fill_periodic[
    dtype: DType, period: Int
](
    output: Pointer[Scalar[dtype], MutAnyOrigin],
    count: Int64,
    amplitude: Float32,
):
    """Writes `(i % period - period // 2) * amplitude`: deterministic, signed
    and nonzero, so no output is identically zero."""
    var i = Int(global_idx.x)
    if i < Int(count):
        output[unsafe_offset=i] = (
            Float32(i % period - period // 2) * amplitude
        ).cast[dtype]()


def _fill_kv_in(output: Pointer[BFloat16, MutAnyOrigin], count: Int64):
    """Writes latent entries of `±1` with hashed signs, so each row's RMS is
    1 and RMSNorm leaves it on the grid `_fill_gamma` scales; the rope dims
    stay a small sawtooth."""
    var i = Int(global_idx.x)
    if i < Int(count):
        var col = i % K_CACHE_DIM
        var value: Float32
        if col < KV_LATENT_DIM:
            value = 1.0 if uniform_bits(7, i) & 1 == 0 else -1.0
        else:
            value = Float32(i % 61 - 30) * 0.03
        output[unsafe_offset=i] = value.cast[DType.bfloat16]()


def _fill_gamma(output: Pointer[BFloat16, MutAnyOrigin]):
    """Writes per-channel weights from {1, 1.5, 2, 3}, with a 3 opening each
    16-channel NVFP4 group. A normalized `±1` row times these is exact in
    E4M3 and, at group scale 0.5, in E2M1, so the FP8 and NVFP4 writers
    store the same keys and the arms can be compared exactly."""
    var c = Int(global_idx.x)
    if c < KV_LATENT_DIM:
        var pick = 3 if c % 16 == 0 else Int(uniform_bits(11, c) & 3)
        var value = SIMD[DType.float32, 4](1.0, 1.5, 2.0, 3.0)[pick]
        output[unsafe_offset=c] = value.cast[DType.bfloat16]()


def fill_kv_in(
    ctx: DeviceContext, buf: DeviceBuffer[bf16, ...], count: Int
) raises:
    ctx.enqueue_function[_fill_kv_in](
        _dev(buf),
        Int64(count),
        grid_dim=ceildiv(count, 256),
        block_dim=256,
    )


def _fill_freqs(output: Pointer[Float32, MutAnyOrigin], count: Int64):
    var i = Int(global_idx.x)
    if i < Int(count):
        output[unsafe_offset=i] = Float32(0.5) if i % 2 == 0 else Float32(0.25)


def fill_periodic[
    dtype: DType, period: Int
](
    ctx: DeviceContext,
    buf: DeviceBuffer[dtype, ...],
    count: Int,
    amplitude: Float32,
) raises:
    ctx.enqueue_function[_fill_periodic[dtype, period]](
        buf.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin(),
        Int64(count),
        amplitude,
        grid_dim=ceildiv(count, 256),
        block_dim=256,
    )


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


@fieldwise_init
struct Options(Copyable, Movable):
    var raise_on_mismatch: Bool
    var diag: Bool
    var forced_np: Int
    var split_break: Bool


struct Workspace(Movable):
    """Inputs, weights and outputs, sized for the largest cell and the
    largest head count of a sweep group."""

    var q: DeviceBuffer[bf16]
    var kv_in: DeviceBuffer[bf16]
    var freqs: DeviceBuffer[DType.float32]
    var gamma: DeviceBuffer[bf16]
    var w_uk_fp8: DeviceBuffer[fp8]
    var w_uk_scale: DeviceBuffer[DType.float32]
    var w_uv_fp8: DeviceBuffer[fp8]
    var w_uv_scale: DeviceBuffer[DType.float32]
    var w_uk_bf16: DeviceBuffer[bf16]
    var w_uv_bf16: DeviceBuffer[bf16]
    var outs: List[DeviceBuffer[bf16]]
    var indices: DeviceBuffer[DType.int32]
    var nvfp4_indices: DeviceBuffer[DType.int32]
    var lengths: DeviceBuffer[DType.int32]
    var row_offsets: DeviceBuffer[DType.uint32]
    var cache_offsets: DeviceBuffer[DType.uint32]
    var dispatch_args: DeviceBuffer[DType.int64]

    def __init__(out self, ctx: DeviceContext[], footprint: Footprint) raises:
        var tokens = footprint["tokens"]
        comptime nvfp4 = arm_enabled["sweep_cache", CACHE_NVFP4]()
        comptime fp8_weights = arm_enabled[
            "sweep_cache", CACHE_FP8
        ]() or arm_enabled["sweep_cache", CACHE_BF16]()
        var q_elems = tokens * MAX_HEADS * Q_HEAD_DIM
        var uk = MAX_HEADS * KV_LATENT_DIM * QK_NOPE_HEAD_DIM
        var uv = MAX_HEADS * V_HEAD_DIM * KV_LATENT_DIM
        var uk_scales = MAX_HEADS * W_UK_SCALE_N * W_UK_SCALE_K
        var uv_scales = MAX_HEADS * W_UV_SCALE_N * W_UV_SCALE_K
        self.q = ctx.enqueue_create_buffer[bf16](q_elems)
        self.kv_in = ctx.enqueue_create_buffer[bf16](tokens * K_CACHE_DIM)
        self.freqs = ctx.enqueue_create_buffer[DType.float32](
            footprint["freq_rows"] * ROPE_DEPTH
        )
        self.gamma = ctx.enqueue_create_buffer[bf16](KV_LATENT_DIM)
        self.w_uk_fp8 = ctx.enqueue_create_buffer[fp8](uk if fp8_weights else 1)
        self.w_uk_scale = ctx.enqueue_create_buffer[DType.float32](uk_scales)
        self.w_uv_fp8 = ctx.enqueue_create_buffer[fp8](uv if fp8_weights else 1)
        self.w_uv_scale = ctx.enqueue_create_buffer[DType.float32](uv_scales)
        self.w_uk_bf16 = ctx.enqueue_create_buffer[bf16](uk if nvfp4 else 1)
        self.w_uv_bf16 = ctx.enqueue_create_buffer[bf16](uv if nvfp4 else 1)
        self.outs = []
        for _ in range(4 if nvfp4 else 2):
            self.outs.append(
                ctx.enqueue_create_buffer[bf16](tokens * MAX_HEADS * V_HEAD_DIM)
            )
        self.indices = ctx.enqueue_create_buffer[DType.int32](tokens * TOP_K)
        self.nvfp4_indices = ctx.enqueue_create_buffer[DType.int32](
            2 * tokens * TOP_K if nvfp4 else 1
        )
        self.lengths = ctx.enqueue_create_buffer[DType.int32](tokens)
        self.row_offsets = ctx.enqueue_create_buffer[DType.uint32](
            footprint["batch"] + 1
        )
        self.cache_offsets = ctx.enqueue_create_buffer[DType.uint32](
            footprint["batch"]
        )
        self.dispatch_args = ctx.enqueue_create_buffer[DType.int64](3)

        fill_periodic[bf16, 97](ctx, self.q, q_elems, 0.05)
        fill_kv_in(ctx, self.kv_in, tokens * K_CACHE_DIM)
        var freq_elems = footprint["freq_rows"] * ROPE_DEPTH
        ctx.enqueue_function[_fill_freqs](
            _dev(self.freqs),
            Int64(freq_elems),
            grid_dim=ceildiv(freq_elems, 256),
            block_dim=256,
        )
        ctx.enqueue_function[_fill_gamma](
            _dev(self.gamma),
            grid_dim=ceildiv(KV_LATENT_DIM, 256),
            block_dim=256,
        )
        self.w_uk_scale.enqueue_fill(0.0125)
        self.w_uv_scale.enqueue_fill(0.0125)
        comptime if fp8_weights:
            fill_periodic[fp8, 29](ctx, self.w_uk_fp8, uk, 0.25)
            fill_periodic[fp8, 23](ctx, self.w_uv_fp8, uv, 0.25)
        comptime if nvfp4:
            fill_periodic[bf16, 29](ctx, self.w_uk_bf16, uk, 0.25)
            fill_periodic[bf16, 23](ctx, self.w_uv_bf16, uv, 0.25)

    @staticmethod
    def bytes_for(footprint: Footprint) -> Int:
        var tokens = footprint["tokens"]
        comptime nvfp4 = arm_enabled["sweep_cache", CACHE_NVFP4]()
        var weights = (
            MAX_HEADS * KV_LATENT_DIM * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)
        )
        return (
            2 * tokens * (MAX_HEADS * Q_HEAD_DIM + K_CACHE_DIM)
            + 4 * footprint["freq_rows"] * ROPE_DEPTH
            + weights * (3 if nvfp4 else 1)
            + (4 if nvfp4 else 2) * 2 * tokens * MAX_HEADS * V_HEAD_DIM
            + 4 * tokens * TOP_K * (3 if nvfp4 else 1)
            + 4 * tokens
            + 8 * (footprint["batch"] + 1)
        )


struct Cell(ImplicitlyCopyable):
    var batch_size: Int
    var seq_len: Int
    var cache_len: Int
    var tokens: Int
    var keys: Int
    var pages: Int

    def __init__(out self, cfg: Config) raises:
        self.batch_size = cfg["batch_size"]
        self.seq_len = cfg["seq_len"]
        self.cache_len = cfg["cache_len"]
        self.tokens = self.batch_size * self.seq_len
        self.keys = self.cache_len + self.seq_len
        self.pages = ceildiv(self.keys, PAGE_SIZE)


def _row_diagonals(
    ctx: DeviceContext,
    reference: DeviceBuffer[bf16, ...],
    candidate: DeviceBuffer[bf16, ...],
    rows: Int,
    row_elems: Int,
) raises:
    """Prints, for each of the first candidate rows, its cosine with its own
    reference row and the reference row it matches best, so a row offset
    or a row-local fault shows as a diagonal other than the identity."""
    var n = min(rows, 8) * row_elems
    var ref_view = DeviceBuffer[bf16](
        ctx, reference.unsafe_ptr(), n, owning=False
    )
    var got_view = DeviceBuffer[bf16](
        ctx, candidate.unsafe_ptr(), n, owning=False
    )
    with ref_view.map_to_host() as expected:
        with got_view.map_to_host() as got:
            for t in range(min(rows, 4)):
                var best = -2.0
                var best_u = -1
                var own = 0.0
                for u in range(min(rows, 8)):
                    var d = Float64(0)
                    var na = Float64(0)
                    var nb = Float64(0)
                    for j in range(row_elems):
                        var r = expected[u * row_elems + j].cast[
                            DType.float64
                        ]()
                        var g = got[t * row_elems + j].cast[DType.float64]()
                        d += r * g
                        na += r * r
                        nb += g * g
                    var c = 0.0 if na == 0 or nb == 0 else d / sqrt(na * nb)
                    if u == t:
                        own = c
                    if c > best:
                        best = c
                        best_u = u
                print(t"  row {t}: cosine={own} best=[{best_u}] {best}")


def _check(
    ctx: DeviceContext,
    reference: DeviceBuffer[bf16, ...],
    candidate: DeviceBuffer[bf16, ...],
    count: Int,
    rows: Int,
    name: String,
    opts: Options,
) raises -> Bool:
    var ok = check_agreement(
        ctx, reference, candidate, count, name, raise_on_mismatch=False
    )
    if opts.diag:
        _row_diagonals(ctx, reference, candidate, rows, count // max(rows, 1))
    return ok


def _fields[cache: Int, heads: Int](arm: String, cell: Cell) -> Fields:
    var fields = Fields()
    fields.add("arm", arm)
    fields.add("cache", cache_name[cache]())
    fields.add("num_heads", heads)
    fields.add("batch_size", cell.batch_size)
    fields.add("seq_len", cell.seq_len)
    fields.add("cache_len", cell.cache_len)
    return fields^


def _bench_name[cache: Int, heads: Int](arm: String, cell: Cell) -> String:
    return String(
        "mla_graph_",
        arm,
        " : cache=",
        cache_name[cache](),
        ", heads=",
        heads,
        " : batch=",
        cell.batch_size,
        ", seq=",
        cell.seq_len,
        ", cache=",
        cell.cache_len,
    )


def run_dense[
    latent_dtype: DType, //, cache: Int, heads: Int
](
    ctx: DeviceContext[],
    mut h: SweepHarness,
    table: PageTable[PAGE_SIZE],
    mut latent: PagedBlocks[latent_dtype],
    cell: Cell,
    ws: Workspace,
    opts: Options,
) raises:
    """Runs the prefill branch against the combined op over an FP8 or BF16
    cache, with FP8 weights."""
    comptime kv_params = KVCacheStaticParams(
        num_heads=1, head_size=K_CACHE_DIM, is_mla=True
    )
    comptime Collection = PagedKVCacheCollection[
        latent_dtype,
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

    var q_tt = TileTensor(
        _dev(ws.q), row_major((cell.tokens, Idx[heads], Idx[Q_HEAD_DIM]))
    )
    var ref_tt = TileTensor(
        _dev(ws.outs[0]), row_major((cell.tokens, Idx[heads], Idx[V_HEAD_DIM]))
    )
    var out_tt = TileTensor(
        _dev(ws.outs[1]), row_major((cell.tokens, Idx[heads], Idx[V_HEAD_DIM]))
    )
    var kv_in_tt = TileTensor(
        _dev(ws.kv_in), row_major((cell.tokens, Idx[K_CACHE_DIM]))
    )
    var freqs_tt = TileTensor(
        _dev(ws.freqs), row_major((cell.keys, Idx[ROPE_DEPTH]))
    )
    var gamma_tt = TileTensor(_dev(ws.gamma), row_major(Idx[KV_LATENT_DIM]))
    var w_uk_tt = TileTensor(
        _dev(ws.w_uk_fp8),
        row_major((Idx[heads], Idx[KV_LATENT_DIM], Idx[QK_NOPE_HEAD_DIM])),
    )
    var w_uk_scale_tt = TileTensor(
        _dev(ws.w_uk_scale),
        row_major((Idx[heads], Idx[W_UK_SCALE_N], Idx[W_UK_SCALE_K])),
    )
    var w_uv_tt = TileTensor(
        _dev(ws.w_uv_fp8),
        row_major((Idx[heads], Idx[V_HEAD_DIM], Idx[KV_LATENT_DIM])),
    )
    var w_uv_scale_tt = TileTensor(
        _dev(ws.w_uv_scale),
        row_major((Idx[heads], Idx[W_UV_SCALE_N], Idx[W_UV_SCALE_K])),
    )
    var w_k_tt = TileTensor(
        _dev(ws.w_uk_fp8),
        row_major(Idx[heads * QK_NOPE_HEAD_DIM], Idx[KV_LATENT_DIM]),
    )
    var w_k_scale_tt = TileTensor(
        _dev(ws.w_uk_scale), row_major(Idx[heads], Idx[W_UK_SCALE_N])
    )
    var row_offsets_tt = TileTensor(
        _dev(ws.row_offsets), row_major(cell.batch_size + 1)
    ).as_imm()
    var cache_offsets_tt = TileTensor(
        _dev(ws.cache_offsets), row_major(cell.batch_size)
    )
    var indices = _dev(ws.indices)
    var lengths = _dev(ws.lengths)
    var mla_args = MLADispatchScalarArgs[num_heads=heads, is_fp8_kv=True](
        cell.batch_size, cell.cache_len, cell.seq_len, ctx
    )
    var scalar_args_tt = mla_args.gpu_tile_tensor()

    @inline(.always)
    def kv_input_fn[
        width: Int
    ](coords: IndexList[2]) {var kv_in_tt} -> SIMD[bf16, width]:
        comptime assert type_of(kv_in_tt).flat_rank == 2
        return kv_in_tt.load[width=width](Coord(coords))

    @inline(.always)
    def prefill_launch(launch_ctx: DeviceContext[]) raises {mut ref_tt, imm}:
        mla_prefill_branch_sparse_fp8[
            m_scale_granularity=M_SCALE_GRAN,
            n_scale_granularity=N_SCALE_GRAN,
            k_scale_granularity=K_SCALE_GRAN,
            indices_stride=TOP_K,
            kernels=SM100MLAGraph,
        ](
            ref_tt,
            q_tt,
            row_offsets_tt,
            freqs_tt,
            gamma_tt,
            kv_collection,
            UInt32(0),
            SCALE,
            EPSILON,
            w_uk_tt.as_imm(),
            w_uk_scale_tt.as_imm(),
            w_uv_tt,
            w_uv_scale_tt,
            scalar_args_tt,
            kv_input_fn,
            launch_ctx,
            indices,
            lengths,
            None,
        )

    @inline(.always)
    def graph_launch(
        launch_ctx: DeviceContext[],
    ) raises {mut out_tt, mut cache_offsets_tt, imm}:
        mla_prefill_decode_graph_fp8[
            m_scale_granularity=M_SCALE_GRAN,
            n_scale_granularity=N_SCALE_GRAN,
            k_scale_granularity=K_SCALE_GRAN,
            mask_str="causal",
            sparse_mla=True,
            sparse_indices_stride=TOP_K,
            kernels=SM100MLAGraph,
        ](
            out_tt,
            q_tt,
            row_offsets_tt,
            freqs_tt,
            gamma_tt,
            kv_collection,
            UInt32(0),
            SCALE,
            EPSILON,
            row_offsets_tt,
            cache_offsets_tt,
            cell.batch_size * cell.keys,
            cell.seq_len,
            w_k_tt.as_imm(),
            w_k_scale_tt.as_imm(),
            w_uk_tt.as_imm(),
            w_uk_scale_tt.as_imm(),
            w_uv_tt,
            w_uv_scale_tt,
            scalar_args_tt,
            kv_input_fn,
            launch_ctx,
            d_indices=indices,
            indices_stride=TOP_K,
            topk_lengths=NonNullPointer[DType.int32](lengths),
        )

    prefill_launch(ctx)
    graph_launch(ctx)
    ctx.synchronize()
    var ok = _check(
        ctx,
        ws.outs[0],
        ws.outs[1],
        cell.tokens * heads * V_HEAD_DIM,
        cell.tokens,
        _bench_name[cache, heads]("graph", cell),
        opts,
    )
    if not ok and opts.raise_on_mismatch:
        raise Error("the combined graph op differs from the prefill branch")
    h.timer.lap("config_setup")
    h.bench(
        ctx,
        prefill_launch,
        BenchId(_bench_name[cache, heads]("prefill", cell)),
        _fields[cache, heads]("prefill", cell),
    )
    h.bench(
        ctx,
        graph_launch,
        BenchId(_bench_name[cache, heads]("graph", cell)),
        _fields[cache, heads]("graph", cell),
    )
    _ = mla_args^
    _ = lut^


def prime_packed(
    ctx: DeviceContext[],
    table: PageTable[PAGE_SIZE],
    mut packed: PackedLatent,
    ws: Workspace,
) raises:
    """Writes every row of the packed pool with the graph's NVFP4 row writer
    and rebuilds the FP8 mirror from those bytes.

    Primes the pool as one sequence over the pool's page table, chunk by
    chunk, so a cell's page table, whichever blocks it picks, reads rows
    the writer produced. Its scales share the values' page table, as the
    graph's do.
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
    var pool_pages = len(table.lut)
    var rows = pool_pages * PAGE_SIZE
    var chunk = min(PRIME_CHUNK, rows)
    comptime packed_layout = PackedCollection.blocks_tt_layout
    var packed_shape = Coord[*packed_layout.shape_types]()
    packed_shape[0] = Int64(table.num_blocks)
    packed_shape[2] = Int64(1)
    var packed_strides = Coord[*packed_layout.stride_types]()
    packed_strides[1] = packed_shape[2] * Int64(packed_strides[2].value())
    packed_strides[0] = Int64(packed_shape[1].value()) * packed_strides[1]
    var lut_tt = (
        TileTensor(table.lut, row_major(Int64(1), Int64(pool_pages)))
        .as_imm()
        .as_unsafe_any_origin()
    )
    var collection = PackedCollection(
        TileTensor(
            packed.packed, packed_layout(packed_shape, packed_strides)
        ).as_unsafe_any_origin(),
        TileTensor(table.cache_lengths, row_major(Int64(1)))
        .as_imm()
        .as_unsafe_any_origin(),
        lut_tt,
        UInt32(chunk),
        UInt32(rows),
        rebind[PackedCollection.scales_tt_type](
            TileTensor(
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
        ),
        lut_tt,
    )
    var kv_in = ctx.enqueue_create_buffer[bf16](chunk * K_CACHE_DIM)
    fill_kv_in(ctx, kv_in, chunk * K_CACHE_DIM)
    var q_rope = ctx.enqueue_create_buffer[fp8](chunk * ROPE_DEPTH)
    var q_rope_out = ctx.enqueue_create_buffer[fp8](chunk * ROPE_DEPTH)
    var row_offsets = ctx.enqueue_create_buffer[DType.uint32](2)
    var host = ctx.enqueue_create_host_buffer[DType.uint32](2)
    ctx.synchronize()
    host[0] = 0
    host[1] = UInt32(chunk)
    ctx.enqueue_copy(row_offsets, host)
    var kv_in_tt = TileTensor(_dev(kv_in), row_major((chunk, Idx[K_CACHE_DIM])))
    var freqs_tt = TileTensor(
        _dev(ws.freqs), row_major((rows, Idx[ROPE_DEPTH]))
    ).as_imm()

    @inline(.always)
    def kv_fn[
        width: Int
    ](coords: IndexList[2]) {var kv_in_tt} -> SIMD[bf16, width]:
        return kv_in_tt.load[width=width](Coord(coords))

    var done = 0
    while done < rows:
        # The writer writes at `cache_length + token`, so each pass starts a
        # cache at its first row; the last pass overlaps the one before it
        # rather than leaving a short tail.
        var start = min(done, rows - chunk)
        table.cache_lengths.enqueue_fill(UInt32(start))
        mla_fused_rope_rmsnorm_quantization(
            TileTensor(
                _dev(q_rope_out), row_major((chunk, Idx[1], Idx[ROPE_DEPTH]))
            ),
            TileTensor(
                _dev(q_rope), row_major((chunk, Idx[1], Idx[ROPE_DEPTH]))
            ).as_imm(),
            TileTensor(_dev(row_offsets), row_major(2)).as_imm(),
            freqs_tt,
            TileTensor(_dev(ws.gamma), row_major(Idx[KV_LATENT_DIM])).as_imm(),
            collection,
            UInt32(0),
            EPSILON,
            kv_fn,
            ctx,
        )
        done += chunk
    packed.rebuild_mirror(ctx)
    ctx.synchronize()
    print("primed", rows, "packed rows with the graph's writer")


def run_nvfp4[
    heads: Int
](
    ctx: DeviceContext[],
    mut h: SweepHarness,
    table: PageTable[PAGE_SIZE],
    mut packed: PackedLatent,
    cell: Cell,
    ws: Workspace,
    opts: Options,
) raises:
    """Runs the packed-cache graph route against the FP8 prefill kernel over
    the mirror, with BF16 weights."""
    comptime indices_stride = 2 * TOP_K
    comptime packed_params = KVCacheStaticParams(
        num_heads=1,
        head_size=PACKED_ROW,
        is_mla=True,
        fp8_tail_size=ROPE_DEPTH,
    )
    comptime mirror_params = KVCacheStaticParams(
        num_heads=1, head_size=K_CACHE_DIM, is_mla=True
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
    comptime packed_layout = PackedCollection.blocks_tt_layout
    var packed_shape = Coord[*packed_layout.shape_types]()
    packed_shape[0] = Int64(table.num_blocks)
    packed_shape[2] = Int64(1)
    var packed_strides = Coord[*packed_layout.stride_types]()
    packed_strides[1] = packed_shape[2] * Int64(packed_strides[2].value())
    packed_strides[0] = Int64(packed_shape[1].value()) * packed_strides[1]
    # Production passes the longest prompt and the context after this step's
    # tokens; the converted scratch forwards both to the decode dispatch.
    var packed_collection = PackedCollection(
        TileTensor(
            packed.packed, packed_layout(packed_shape, packed_strides)
        ).as_unsafe_any_origin(),
        cache_lengths_tt,
        lut_tt,
        UInt32(cell.seq_len),
        UInt32(cell.keys),
        rebind[PackedCollection.scales_tt_type](
            TileTensor(
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
        ),
        lut_tt,
    )
    var mirror_collection = MirrorCollection(
        rebind[MirrorCollection.blocks_tt_type](
            TileTensor(
                packed.mirror,
                row_major(
                    Int64(table.num_blocks),
                    Idx[1],
                    Int64(1),
                    Idx[PAGE_SIZE],
                    Idx[1],
                    Idx[K_CACHE_DIM],
                ),
            ).as_unsafe_any_origin()
        ),
        cache_lengths_tt,
        lut_tt,
        UInt32(cell.seq_len),
        UInt32(cell.keys),
    )

    var out_shape = row_major((cell.tokens, Idx[heads], Idx[V_HEAD_DIM]))
    var mirror_tt = TileTensor(_dev(ws.outs[0]), out_shape)
    var packed_tt = TileTensor(_dev(ws.outs[1]), out_shape)
    var mirror_decode_tt = TileTensor(_dev(ws.outs[2]), out_shape)
    var packed_prefill_tt = TileTensor(_dev(ws.outs[3]), out_shape)
    var q_tt = TileTensor(
        _dev(ws.q), row_major((cell.tokens, Idx[heads], Idx[Q_HEAD_DIM]))
    )
    var kv_in_tt = TileTensor(
        _dev(ws.kv_in), row_major((cell.tokens, Idx[K_CACHE_DIM]))
    )
    var freqs_tt = TileTensor(
        _dev(ws.freqs), row_major((cell.keys, Idx[ROPE_DEPTH]))
    )
    var gamma_tt = TileTensor(_dev(ws.gamma), row_major(Idx[KV_LATENT_DIM]))
    var w_uk_tt = TileTensor(
        _dev(ws.w_uk_bf16),
        row_major((Idx[heads], Idx[KV_LATENT_DIM], Idx[QK_NOPE_HEAD_DIM])),
    )
    var w_uv_tt = TileTensor(
        _dev(ws.w_uv_bf16),
        row_major((Idx[heads], Idx[V_HEAD_DIM], Idx[KV_LATENT_DIM])),
    )
    var w_k_tt = TileTensor(
        _dev(ws.w_uk_bf16),
        row_major(Idx[heads * QK_NOPE_HEAD_DIM], Idx[KV_LATENT_DIM]),
    )
    var row_offsets_tt = TileTensor(
        _dev(ws.row_offsets), row_major(cell.batch_size + 1)
    ).as_imm()
    var cache_offsets_tt = TileTensor(
        _dev(ws.cache_offsets), row_major(cell.batch_size)
    )
    var indices = _dev(ws.nvfp4_indices)
    var indices_fp8 = _dev(ws.indices)
    var lengths = _dev(ws.lengths)

    # Production stages the dispatch key's split count into the op's
    # `num_partitions` tensor, and the op forwards it so the grid and the
    # kernel's split agree. `--split_break` keeps the plain decode split,
    # the wiring before the forwarding, which the checks must catch.
    comptime sm_count = ctx.default_device_info.sm_count
    var launch_key = compute_mla_dispatch_key_runtime(
        cell.batch_size,
        cell.keys,
        cell.seq_len,
        heads,
        True,
        indices_stride,
        sm_count,
        kv_form=MLA_SPARSE_KV_FORM_NVFP4,
        page_size=PAGE_SIZE,
        # The prefill passes per-token top-k lengths.
        fixed_topk=False,
        physical_indices=True,
        depth=K_CACHE_DIM,
    )
    var plain_scalars = compute_mla_dispatch_scalars[heads, is_fp8_kv=True](
        cell.batch_size, cell.keys, cell.seq_len, sm_count
    )
    var num_partitions = plain_scalars[2] if opts.split_break else launch_key[2]
    if opts.forced_np > 0:
        num_partitions = opts.forced_np
    print(
        t"dispatch split: key={launch_key[2]} plain={plain_scalars[2]}",
        t" launched={num_partitions}",
    )
    var args_host = ctx.enqueue_create_host_buffer[DType.int64](3)
    ctx.synchronize()
    args_host[0] = Int64(cell.batch_size)
    args_host[1] = Int64(cell.seq_len)
    args_host[2] = Int64(num_partitions)
    ctx.enqueue_copy(ws.dispatch_args, args_host)
    var scalar_args_tt = TileTensor(
        _dev(ws.dispatch_args), row_major((Idx[3],))
    )

    @inline(.always)
    def kv_input_fn[
        width: Int
    ](coords: IndexList[2]) {var kv_in_tt} -> SIMD[bf16, width]:
        comptime assert type_of(kv_in_tt).flat_rank == 2
        return kv_in_tt.load[width=width](Coord(coords))

    @inline(.always)
    def mirror_launch(launch_ctx: DeviceContext[]) raises {mut mirror_tt, imm}:
        mla_prefill_branch_sparse_bf16[
            indices_stride=TOP_K, kernels=SM100MLAGraph
        ](
            mirror_tt,
            q_tt,
            row_offsets_tt,
            freqs_tt,
            gamma_tt,
            mirror_collection,
            UInt32(0),
            SCALE,
            EPSILON,
            w_uk_tt.as_imm(),
            w_uv_tt.as_imm(),
            kv_input_fn,
            launch_ctx,
            indices_fp8,
            lengths,
        )

    @inline(.always)
    def packed_launch(
        launch_ctx: DeviceContext[],
    ) raises {mut packed_tt, mut cache_offsets_tt, imm}:
        mla_prefill_decode_graph_bf16[
            mask_str="causal",
            sparse_mla=True,
            sparse_indices_stride=indices_stride,
            nvfp4_convert_ratio=NVFP4_CONVERT_RATIO,
            nvfp4_convert_max_rows=NVFP4_CONVERT_MAX_ROWS,
            kernels=SM100MLAGraph,
        ](
            packed_tt,
            q_tt,
            row_offsets_tt,
            freqs_tt,
            gamma_tt,
            packed_collection,
            UInt32(0),
            SCALE,
            EPSILON,
            row_offsets_tt,
            cache_offsets_tt,
            cell.batch_size * cell.keys,
            cell.seq_len,
            w_k_tt.as_imm(),
            w_uk_tt.as_imm(),
            w_uv_tt.as_imm(),
            scalar_args_tt,
            kv_input_fn,
            launch_ctx,
            d_indices=indices,
            indices_stride=indices_stride,
            topk_lengths=NonNullPointer[DType.int32](lengths),
            num_partitions_in=Optional[Int](num_partitions),
        )

    @inline(.always)
    def mirror_decode_launch(
        launch_ctx: DeviceContext[],
    ) raises {mut mirror_decode_tt, mut cache_offsets_tt, imm}:
        mla_prefill_decode_graph_bf16[
            mask_str="causal",
            sparse_mla=True,
            sparse_indices_stride=TOP_K,
            kernels=SM100MLAGraph,
        ](
            mirror_decode_tt,
            q_tt,
            row_offsets_tt,
            freqs_tt,
            gamma_tt,
            mirror_collection,
            UInt32(0),
            SCALE,
            EPSILON,
            row_offsets_tt,
            cache_offsets_tt,
            cell.batch_size * cell.keys,
            cell.seq_len,
            w_k_tt.as_imm(),
            w_uk_tt.as_imm(),
            w_uv_tt.as_imm(),
            scalar_args_tt,
            kv_input_fn,
            launch_ctx,
            d_indices=indices_fp8,
            indices_stride=TOP_K,
            topk_lengths=NonNullPointer[DType.int32](lengths),
            num_partitions_in=Optional[Int](num_partitions),
        )

    @inline(.always)
    def packed_prefill_launch(
        launch_ctx: DeviceContext[],
    ) raises {mut packed_prefill_tt, imm}:
        mla_prefill_branch_sparse_bf16[
            indices_stride=indices_stride,
            kernels=SM100MLAGraph,
        ](
            packed_prefill_tt,
            q_tt,
            row_offsets_tt,
            freqs_tt,
            gamma_tt,
            packed_collection,
            UInt32(0),
            SCALE,
            EPSILON,
            w_uk_tt.as_imm(),
            w_uv_tt.as_imm(),
            kv_input_fn,
            launch_ctx,
            indices,
            lengths,
        )

    mirror_launch(ctx)
    packed_launch(ctx)
    mirror_decode_launch(ctx)
    packed_prefill_launch(ctx)
    ctx.synchronize()
    var count = cell.tokens * heads * V_HEAD_DIM
    var name = _bench_name[CACHE_NVFP4, heads]("packed", cell)
    var ok = _check(
        ctx,
        ws.outs[0],
        ws.outs[1],
        count,
        cell.tokens,
        String(name, ": packed vs FP8 mirror"),
        opts,
    )
    ok = (
        _check(
            ctx,
            ws.outs[2],
            ws.outs[1],
            count,
            cell.tokens,
            String(name, ": FP8 mirror on the decode route vs packed"),
            opts,
        )
        and ok
    )
    ok = (
        _check(
            ctx,
            ws.outs[2],
            ws.outs[0],
            count,
            cell.tokens,
            String(name, ": FP8 mirror on the decode route vs prefill kernel"),
            opts,
        )
        and ok
    )
    ok = (
        _check(
            ctx,
            ws.outs[3],
            ws.outs[0],
            count,
            cell.tokens,
            String(name, ": packed NVFP4 prefill kernel vs FP8 mirror"),
            opts,
        )
        and ok
    )
    ok = (
        _check(
            ctx,
            ws.outs[3],
            ws.outs[1],
            count,
            cell.tokens,
            String(name, ": packed NVFP4 prefill kernel vs packed decode"),
            opts,
        )
        and ok
    )
    if not ok and opts.raise_on_mismatch:
        raise Error("a packed NVFP4 arm differs from its reference")
    h.timer.lap("config_setup")
    h.bench(
        ctx,
        mirror_launch,
        BenchId(_bench_name[CACHE_NVFP4, heads]("mirror", cell)),
        _fields[CACHE_NVFP4, heads]("mirror", cell),
    )
    var fields = _fields[CACHE_NVFP4, heads]("packed", cell)
    fields.add("num_partitions", num_partitions)
    h.bench(ctx, packed_launch, BenchId(name), fields^)
    _ = lut^


def setup_cell[
    cache: Int
](
    ctx: DeviceContext[],
    mut table: PageTable[PAGE_SIZE],
    cell: Cell,
    ws: Workspace,
) raises:
    """Writes the cell's page table, index rows, lengths and offsets."""
    var slots = cell.tokens * TOP_K
    table.set_cache_len(ctx, cell.cache_len)
    table.write_copies(ctx, 1, cell.batch_size, cell.pages)
    var lut = table.copy_lut(ctx, 0, cell.batch_size, cell.pages)
    var indices = DeviceBuffer[DType.int32](
        ctx, ws.indices.unsafe_ptr(), slots, owning=False
    )
    fill_sparse_indices(
        ctx,
        indices,
        lut,
        cell.pages,
        PAGE_SIZE,
        True,
        cell.batch_size,
        cell.seq_len,
        cell.cache_len,
        TOP_K,
    )
    comptime if cache == CACHE_NVFP4:
        # The scales share the values' page table, so a pick's scale row is
        # its value row.
        interleave_index_rows(
            ctx, ws.nvfp4_indices, indices, indices, cell.tokens, TOP_K
        )
    fill_topk_lengths(
        ctx, ws.lengths, cell.batch_size, cell.seq_len, cell.cache_len, TOP_K
    )
    var host = ctx.enqueue_create_host_buffer[DType.uint32](cell.batch_size + 1)
    ctx.synchronize()
    for i in range(cell.batch_size + 1):
        host[i] = UInt32(i * cell.seq_len)
    ctx.enqueue_copy(
        DeviceBuffer[DType.uint32](
            ctx, ws.row_offsets.unsafe_ptr(), cell.batch_size + 1, owning=False
        ),
        host,
    )
    ws.cache_offsets.enqueue_fill(UInt32(cell.cache_len))
    ctx.synchronize()


def main() raises:
    var h = SweepHarness("graph_prefill", default_l2="warm")

    var preset = String(arg_parse("preset", ""))
    var cells = PROMPT_CELLS
    if preset == "fallback":
        cells = FALLBACK_CELLS
    elif preset == "diag":
        cells = DIAG_CELLS
    elif preset.byte_length() > 0:
        raise Error("--preset must be fallback or diag, not ", preset)
    var configs = cells_from_arg("cells", cells)
    var opts = Options(
        arg_parse("report", 1 if preset == "diag" else 0) == 0,
        arg_parse("diag", 0) != 0,
        arg_parse("np", 0),
        arg_parse("split_break", 0) != 0,
    )

    h.timer.lap("main_prologue")
    with DeviceContext() as ctx:
        h.timer.lap("device_init")
        h.start(ctx)

        def require(cfg: Config, mut fp: Footprint) raises:
            var cell = Cell(cfg)
            PageTable[PAGE_SIZE].require(fp, cell.batch_size, cell.keys)
            fp.need("tokens", cell.tokens)
            fp.need("freq_rows", cell.keys)
            # The NVFP4 writer primes the whole pool at absolute positions.
            comptime if arm_enabled["sweep_cache", CACHE_NVFP4]():
                fp.need(
                    "freq_rows",
                    PageTable[PAGE_SIZE].rows_for(fp) - PAGE_SIZE,
                )

        def bytes(fp: Footprint) raises -> Int:
            var rows = PageTable[PAGE_SIZE].rows_for(fp)
            var latent = 0
            comptime if arm_enabled["sweep_cache", CACHE_FP8]():
                latent += rows * K_CACHE_DIM
            comptime if arm_enabled["sweep_cache", CACHE_BF16]():
                latent += 2 * rows * K_CACHE_DIM
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
                K_CACHE_DIM if arm_enabled["sweep_cache", CACHE_FP8]() else 1,
                seed=0,
                lo=-2.0,
                hi=2.0,
            )
            var bf16_latent = PagedBlocks[bf16](
                ctx,
                table,
                K_CACHE_DIM if arm_enabled["sweep_cache", CACHE_BF16]() else 1,
                seed=0,
                lo=-2.0,
                hi=2.0,
            )
            var packed = PackedLatent(
                ctx,
                table.rows() if arm_enabled[
                    "sweep_cache", CACHE_NVFP4
                ]() else PAGE_SIZE,
                PAGE_SIZE,
                seed=0,
                permute_scales=False,
            )
            var ws = Workspace(ctx, fp)
            comptime if arm_enabled["sweep_cache", CACHE_NVFP4]():
                prime_packed(ctx, table, packed, ws)
            ctx.synchronize()
            h.timer.lap("fixture_fill")
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
                        setup_cell[cache](ctx, table, cell, ws)
                        comptime for heads in [64, 128]:
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
