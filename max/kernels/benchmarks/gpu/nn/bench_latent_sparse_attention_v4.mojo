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
"""DeepSeek-V4 latent sparse attention decode: portable kernel vs SM100 route.

Per call, at V4 decode shapes (bf16 512-wide latent, one query row per
sequence):

* `slow`: the portable `_latent_sparse_attention_gpu_kernel`, launched
  directly exactly as the op launched it before the SM100 route.
* `fast`: `latent_sparse_attention_ragged_paged`, the op's entry, which on
  SM100 runs the plan kernel, the sparse MLA decode and its split-K combine.
* `fast_nosplit`: the same entry with `split_k=False`, one CTA per row and no
  combine, so no split partial is rounded to bf16.
* `slow_pbf16`: `slow` with each attention weight rounded to bf16 before its
  value product (diagnostic: the SM100 decode's P rounding, alone).

Each shape first prints a `CHECK` line per fast arm with its max |slow - arm|.
`--case=N`
runs one shape only (for per-kernel nsys); -1 runs all.
"""

from std.math import ceildiv
from std.random import randn_float64, seed
from std.utils.index import IndexList

from max.benchmark import bencher_iter_custom
from std.benchmark import Bench, Bencher, BenchId
from max.gpu import WARP_SIZE
from max.gpu.host import DeviceBuffer, DeviceContext
from internal_utils import arg_parse
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from nn.attention.latent_sparse_attention import (
    _latent_sparse_attention_gpu_kernel,
    latent_sparse_attention_ragged_paged,
)

comptime HEAD_DIM = 512
comptime WINDOW = 128
comptime WIN_PAGE = 128
# A ratio-4 zone leaf holds page_size // 4 entries per page.
comptime COMP_PAGE = 32

comptime bf16 = DType.bfloat16
comptime f32 = DType.float32


def _collection[
    page: Int
](
    blocks: DeviceBuffer[bf16],
    num_pages: Int,
    cache_lengths: DeviceBuffer[DType.uint32],
    lut: DeviceBuffer[DType.uint32],
    batch: Int,
    pages_per_seq: Int,
    max_cache_len: Int,
) -> PagedKVCacheCollection[
    bf16,
    KVCacheStaticParams(num_heads=1, head_size=HEAD_DIM, is_mla=True),
    page,
    MutAnyOrigin,
    ImmutAnyOrigin,
    ImmutAnyOrigin,
    MutUntrackedOrigin,
]:
    comptime blocks_layout = Layout.row_major[6]()
    comptime cl_layout = Layout(UNKNOWN_VALUE)
    comptime lut_layout = Layout.row_major[2]()
    return PagedKVCacheCollection[
        bf16,
        KVCacheStaticParams(num_heads=1, head_size=HEAD_DIM, is_mla=True),
        page,
    ](
        LayoutTensor[bf16, blocks_layout, MutAnyOrigin](
            rebind[Pointer[Scalar[bf16], MutAnyOrigin]](blocks.unsafe_ptr()),
            RuntimeLayout[blocks_layout].row_major(
                IndexList[6](num_pages, 1, 1, page, 1, HEAD_DIM)
            ),
        ),
        LayoutTensor[DType.uint32, cl_layout, ImmutAnyOrigin](
            cache_lengths.unsafe_ptr().as_unsafe_any_origin(),
            RuntimeLayout[cl_layout].row_major(IndexList[1](batch)),
        ),
        LayoutTensor[DType.uint32, lut_layout, ImmutAnyOrigin](
            lut.unsafe_ptr().as_unsafe_any_origin(),
            RuntimeLayout[lut_layout].row_major(
                IndexList[2](batch, pages_per_seq)
            ),
        ),
        UInt32(1),
        UInt32(max_cache_len + 1),
    )


def _fill_randn(ctx: DeviceContext, buf: DeviceBuffer[bf16], n: Int) raises:
    with buf.map_to_host() as h:
        for i in range(n):
            h[i] = Float32(randn_float64()).cast[bf16]()


def _paged_lut(
    ctx: DeviceContext, batch: Int, pages_per_seq: Int
) raises -> DeviceBuffer[DType.uint32]:
    # Pages mapped in reverse so consecutive keys are not physically adjacent.
    var total = batch * pages_per_seq
    var lut = ctx.enqueue_create_buffer[DType.uint32](total)
    with lut.map_to_host() as h:
        for i in range(total):
            h[i] = UInt32(total - 1 - i)
    return lut


def run_shape[
    num_heads: Int
](
    ctx: DeviceContext,
    mut m: Bench,
    batch: Int,
    pos: Int,
    num_comp: Int,
    comp_valid: Int,
    comp_total: Int,
) raises:
    """Benchmarks `slow` and `fast` at one decode shape and checks they agree.

    Every sequence has one query row at position `pos`. `num_comp` is the
    compressed-index row width, of which the first `comp_valid` entries are
    real (the rest `-1`), drawn from `comp_total` stored entries.
    """
    seed(0x5EED)
    var tag = String(
        "H=",
        num_heads,
        " B=",
        batch,
        " pos=",
        pos,
        " ncomp=",
        num_comp,
        " valid=",
        comp_valid,
    )

    var win_pps = ceildiv(pos + 1, WIN_PAGE)
    var win_pages = batch * win_pps
    var cmp_pps = ceildiv(max(comp_total, 1), COMP_PAGE)
    var cmp_pages = batch * cmp_pps

    var win_blocks = ctx.enqueue_create_buffer[bf16](
        win_pages * WIN_PAGE * HEAD_DIM
    )
    var cmp_blocks = ctx.enqueue_create_buffer[bf16](
        cmp_pages * COMP_PAGE * HEAD_DIM
    )
    _fill_randn(ctx, win_blocks, win_pages * WIN_PAGE * HEAD_DIM)
    _fill_randn(ctx, cmp_blocks, cmp_pages * COMP_PAGE * HEAD_DIM)
    var win_lut = _paged_lut(ctx, batch, win_pps)
    var cmp_lut = _paged_lut(ctx, batch, cmp_pps)
    var win_len = ctx.enqueue_create_buffer[DType.uint32](batch)
    var cmp_len = ctx.enqueue_create_buffer[DType.uint32](batch)
    with win_len.map_to_host() as w, cmp_len.map_to_host() as c:
        for b in range(batch):
            w[b] = UInt32(pos)
            c[b] = UInt32(comp_total)

    var idx_cols = max(num_comp, 1)
    var idx = ctx.enqueue_create_buffer[DType.int32](batch * idx_cols)
    with idx.map_to_host() as h:
        for b in range(batch):
            for j in range(idx_cols):
                var e = Int32(-1)
                if j < comp_valid and j < num_comp and comp_total > 0:
                    e = Int32((j * 7 + 3 + b) % comp_total)
                h[b * idx_cols + j] = e

    var q = ctx.enqueue_create_buffer[bf16](batch * num_heads * HEAD_DIM)
    _fill_randn(ctx, q, batch * num_heads * HEAD_DIM)
    var sink = ctx.enqueue_create_buffer[f32](num_heads)
    with sink.map_to_host() as h:
        for i in range(num_heads):
            h[i] = Float32(randn_float64())
    var offs = ctx.enqueue_create_buffer[DType.uint32](batch + 1)
    with offs.map_to_host() as h:
        for b in range(batch + 1):
            h[b] = UInt32(b)
    var out_slow = ctx.enqueue_create_buffer[bf16](batch * num_heads * HEAD_DIM)
    var out_fast = ctx.enqueue_create_buffer[bf16](batch * num_heads * HEAD_DIM)
    var out_nosplit = ctx.enqueue_create_buffer[bf16](
        batch * num_heads * HEAD_DIM
    )
    var out_pbf16 = ctx.enqueue_create_buffer[bf16](
        batch * num_heads * HEAD_DIM
    )
    ctx.synchronize()

    var win = _collection[WIN_PAGE](
        win_blocks, win_pages, win_len, win_lut, batch, win_pps, pos
    ).get_key_cache(0)
    var cmp = _collection[COMP_PAGE](
        cmp_blocks, cmp_pages, cmp_len, cmp_lut, batch, cmp_pps, comp_total
    ).get_key_cache(0)

    var q_tt = TileTensor(
        q.unsafe_ptr().as_imm().as_unsafe_any_origin(),
        row_major(batch, Idx[num_heads], Idx[HEAD_DIM]),
    )
    var out_tt = TileTensor(
        out_fast.unsafe_ptr().as_unsafe_any_origin(),
        row_major(batch, Idx[num_heads], Idx[HEAD_DIM]),
    )
    var out_nosplit_tt = TileTensor(
        out_nosplit.unsafe_ptr().as_unsafe_any_origin(),
        row_major(batch, Idx[num_heads], Idx[HEAD_DIM]),
    )
    var offs_tt = TileTensor(
        offs.unsafe_ptr().as_imm().as_unsafe_any_origin(),
        row_major(batch + 1),
    )
    var sink_tt = TileTensor(
        sink.unsafe_ptr().as_imm().as_unsafe_any_origin(),
        row_major(num_heads),
    )
    var idx_tt = TileTensor(
        idx.unsafe_ptr().as_imm().as_unsafe_any_origin(),
        row_major(batch, idx_cols),
    )
    var scale = Float32(0.044194173824159216)

    var op_slow = rebind[Pointer[BFloat16, MutAnyOrigin]](out_slow.unsafe_ptr())
    var op_pbf16 = rebind[Pointer[BFloat16, MutAnyOrigin]](
        out_pbf16.unsafe_ptr()
    )
    var qp = q.unsafe_ptr().as_imm().as_unsafe_any_origin()
    var rp = offs.unsafe_ptr().as_imm().as_unsafe_any_origin()
    var ip = idx.unsafe_ptr().as_imm().as_unsafe_any_origin()
    var sp = sink.unsafe_ptr().as_imm().as_unsafe_any_origin()
    # The launch the op made before the SM100 route, kept verbatim.
    comptime heads_per_warp = 4
    comptime warps_per_block = 8
    comptime slow_kernel = _latent_sparse_attention_gpu_kernel[
        swa_t=type_of(win),
        comp_t=type_of(cmp),
        q_type=bf16,
        out_type=bf16,
        head_dim=HEAD_DIM,
        heads_per_warp=heads_per_warp,
        warps_per_block=warps_per_block,
        window=WINDOW,
    ]
    comptime pbf16_kernel = _latent_sparse_attention_gpu_kernel[
        swa_t=type_of(win),
        comp_t=type_of(cmp),
        q_type=bf16,
        out_type=bf16,
        head_dim=HEAD_DIM,
        heads_per_warp=heads_per_warp,
        warps_per_block=warps_per_block,
        window=WINDOW,
        round_p=True,
    ]

    @inline(.always)
    def launch_slow(
        launch_ctx: DeviceContext,
    ) raises {
        imm op_slow,
        imm qp,
        imm rp,
        imm ip,
        imm sp,
        imm win,
        imm cmp,
        imm batch,
        imm idx_cols,
        imm scale,
    }:
        launch_ctx.enqueue_function[slow_kernel](
            op_slow,
            qp,
            rp,
            ip,
            sp,
            win,
            cmp,
            Int32(num_heads),
            Int32(batch),
            Int32(idx_cols),
            Int32(num_heads * HEAD_DIM),
            Int32(HEAD_DIM),
            Int32(num_heads * HEAD_DIM),
            Int32(HEAD_DIM),
            Int32(idx_cols),
            Int32(1),
            scale,
            grid_dim=(
                batch,
                ceildiv(num_heads, heads_per_warp * warps_per_block),
            ),
            block_dim=WARP_SIZE * warps_per_block,
        )

    @inline(.always)
    def launch_pbf16(
        launch_ctx: DeviceContext,
    ) raises {
        imm op_pbf16,
        imm qp,
        imm rp,
        imm ip,
        imm sp,
        imm win,
        imm cmp,
        imm batch,
        imm idx_cols,
        imm scale,
    }:
        launch_ctx.enqueue_function[pbf16_kernel](
            op_pbf16,
            qp,
            rp,
            ip,
            sp,
            win,
            cmp,
            Int32(num_heads),
            Int32(batch),
            Int32(idx_cols),
            Int32(num_heads * HEAD_DIM),
            Int32(HEAD_DIM),
            Int32(num_heads * HEAD_DIM),
            Int32(HEAD_DIM),
            Int32(idx_cols),
            Int32(1),
            scale,
            grid_dim=(
                batch,
                ceildiv(num_heads, heads_per_warp * warps_per_block),
            ),
            block_dim=WARP_SIZE * warps_per_block,
        )

    @inline(.always)
    def launch_fast(
        launch_ctx: DeviceContext,
    ) raises {
        imm out_tt,
        imm q_tt,
        imm offs_tt,
        imm idx_tt,
        imm sink_tt,
        imm win,
        imm cmp,
        imm scale,
    }:
        latent_sparse_attention_ragged_paged[target="gpu", window=WINDOW](
            out_tt, q_tt, offs_tt, idx_tt, sink_tt, win, cmp, scale, launch_ctx
        )

    @inline(.always)
    def launch_nosplit(
        launch_ctx: DeviceContext,
    ) raises {
        imm out_nosplit_tt,
        imm q_tt,
        imm offs_tt,
        imm idx_tt,
        imm sink_tt,
        imm win,
        imm cmp,
        imm scale,
    }:
        latent_sparse_attention_ragged_paged[
            target="gpu", window=WINDOW, split_k=False
        ](
            out_nosplit_tt,
            q_tt,
            offs_tt,
            idx_tt,
            sink_tt,
            win,
            cmp,
            scale,
            launch_ctx,
        )

    launch_slow(ctx)
    launch_fast(ctx)
    launch_nosplit(ctx)
    launch_pbf16(ctx)
    ctx.synchronize()
    var n = batch * num_heads * HEAD_DIM
    with out_slow.map_to_host() as sh, out_fast.map_to_host() as fh, out_nosplit.map_to_host() as nh, out_pbf16.map_to_host() as ph:
        var max_ref = Float64(0)
        var max_fast = Float64(0)
        var max_nosplit = Float64(0)
        var max_pbf16 = Float64(0)
        for i in range(n):
            var a = Float64(sh[i].cast[DType.float64]())
            max_ref = max(max_ref, abs(a))
            max_fast = max(
                max_fast, abs(a - Float64(fh[i].cast[DType.float64]()))
            )
            max_nosplit = max(
                max_nosplit, abs(a - Float64(nh[i].cast[DType.float64]()))
            )
            max_pbf16 = max(
                max_pbf16, abs(a - Float64(ph[i].cast[DType.float64]()))
            )
        print("CHECK", tag, "max|slow-fast|=", max_fast, "max|slow|=", max_ref)
        print(
            "CHECK",
            tag,
            "max|slow-fast_nosplit|=",
            max_nosplit,
            "max|slow|=",
            max_ref,
        )
        print(
            "CHECK",
            tag,
            "max|slow-slow_pbf16|=",
            max_pbf16,
            "max|slow|=",
            max_ref,
        )

    @inline(.always)
    def bench_slow(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, launch_slow, ctx)

    @inline(.always)
    def bench_fast(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, launch_fast, ctx)

    @inline(.always)
    def bench_nosplit(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, launch_nosplit, ctx)

    @inline(.always)
    def bench_pbf16(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, launch_pbf16, ctx)

    m.bench_function(bench_slow, BenchId(String("slow ", tag)))
    m.bench_function(bench_fast, BenchId(String("fast ", tag)))
    m.bench_function(bench_nosplit, BenchId(String("fast_nosplit ", tag)))
    m.bench_function(bench_pbf16, BenchId(String("slow_pbf16 ", tag)))

    _ = win_blocks^
    _ = cmp_blocks^
    _ = win_lut^
    _ = cmp_lut^
    _ = win_len^
    _ = cmp_len^
    _ = idx^
    _ = q^
    _ = sink^
    _ = offs^
    _ = out_slow^
    _ = out_fast^
    _ = out_nosplit^
    _ = out_pbf16^


def run_heads[
    num_heads: Int
](ctx: DeviceContext, mut m: Bench, only: Int, base: Int) raises:
    # The four census shapes of the feasibility bench, at c1, plus the two
    # compressed shapes at c8.
    # Window-only layer.
    if only < 0 or only == base + 0:
        run_shape[num_heads](ctx, m, 1, 400, 0, 0, 0)
    # Ratio-128 layer: a handful of closed windows.
    if only < 0 or only == base + 1:
        run_shape[num_heads](ctx, m, 1, 400, 4, 3, 3)
    # Ratio-4 layer at census context: 512 slots, ~100 real.
    if only < 0 or only == base + 2:
        run_shape[num_heads](ctx, m, 1, 400, 512, 100, 100)
    # Ratio-4 layer, saturated top-k (long context): 640 keys.
    if only < 0 or only == base + 3:
        run_shape[num_heads](ctx, m, 1, 8192, 512, 512, 2048)
    if only < 0 or only == base + 4:
        run_shape[num_heads](ctx, m, 8, 400, 512, 100, 100)
    if only < 0 or only == base + 5:
        run_shape[num_heads](ctx, m, 8, 8192, 512, 512, 2048)


def main() raises:
    var only = arg_parse("case", -1)
    var m = Bench()
    with DeviceContext() as ctx:
        # Cases 0-5: 32 heads (TP2 per-GPU); 6-11: 64 heads.
        run_heads[32](ctx, m, only, 0)
        run_heads[64](ctx, m, only, 6)
    m.dump_report()
