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
"""AMD sparse MLA decode read in place must match gather-then-decode.

The in-place arm (`SparseRowsMHAOperand[predicated=True]` with
`SparseRowsDecodeMask`) derives each token's valid length in closed form
from the cache length, splits the top-k selection into partitions smaller
than a tile, and skips the copies of keys outside a partition or past the
valid length. Its output is compared with the production fallback, the
gather kernel that writes the same closed-form lengths followed by
`flare_mla_decoding`, at the GLM-5.3 per-rank shape (8 heads, depth 576,
top-k 2048).

Every cache row no batch entry selects holds a saturated poison pattern,
and the slots past each entry's top-k length name poison rows, so reading
any row the selection excludes moves the output well past tolerance.
Partition counts cover one multi-tile partition (1), tile-sized and
sub-tile partitions (8 to 128), and more partitions than strips (256),
which leaves the tail partitions empty.
"""

from std.math import ceildiv, exp
from std.collections import List

from max.gpu.host import DeviceContext, default_accelerator
from internal_utils.fp8_utils import cast_saturating
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Idx, TileTensor, row_major
from layout.tile_layout import Layout as TileLayout
from nn.attention.gpu.mla import flare_mla_decoding, flare_mla_decoding_rows
from nn.attention.gpu.mla_graph import (
    _AMD_SPARSE_MLA_GATHER_VEC,
    _amd_sparse_mla_gather_kv_rows_and_lengths,
)
from nn.attention.mha_mask import PaddingMask, SparseRowsDecodeMask
from nn.attention.mha_operand import KVCacheMHAOperand, SparseRowsMHAOperand
from nn.attention.mha_utils import MHAConfig
from std.testing import assert_true

comptime NUM_HEADS = 8
comptime K_CACHE_DIM = 576
comptime KV_LATENT_DIM = 512
comptime PAGE_SIZE = 128
comptime TOPK = 2048
comptime NUM_PAGES = 256
comptime NUM_ROWS = NUM_PAGES * PAGE_SIZE
comptime GATHER_ROWS_PER_BLOCK = 7

# The two paths differ only in split-K order and in rounding each
# partition's output to bf16 before the reduce.
comptime ATOL = Float64(0.02)
comptime RTOL = Float64(0.02)


def _row_value(row: Int, d: Int) -> Float32:
    return Float32(((row * 131 + d * 37) % 61) - 30) * 0.03125


def _oracle_max_err[
    dtype: DType
](
    got: TileTensor[DType.bfloat16, ...],
    q_host: TileTensor[dtype, ...],
    blocks_host: TileTensor[dtype, ...],
    idx_host: TileTensor[DType.int32, ...],
    valid: List[Int],
    scale: Float64,
) -> Float64:
    """Returns the largest |got - softmax(scale * q.k) v| over all outputs.

    The fp32 reference attends each entry's first `valid[b]` slots, from the
    same fp8 values the kernels read.
    """
    var max_err = Float64(0)
    var scores = List[Float64](length=TOPK, fill=0)
    var acc = List[Float64](length=KV_LATENT_DIM, fill=0)
    for b in range(len(valid)):
        var n = valid[b]
        for h in range(NUM_HEADS):
            var q_base = (b * NUM_HEADS + h) * K_CACHE_DIM
            var m = Float64(-1e300)
            for j in range(n):
                var r = Int(idx_host.ptr[b * TOPK + j])
                var dot = Float64(0)
                for d in range(K_CACHE_DIM):
                    dot += Float64(
                        q_host.ptr[q_base + d].cast[DType.float64]()
                    ) * Float64(
                        blocks_host.ptr[r * K_CACHE_DIM + d].cast[
                            DType.float64
                        ]()
                    )
                scores[j] = dot * scale
                m = max(m, scores[j])
            var l = Float64(0)
            for d in range(KV_LATENT_DIM):
                acc[d] = 0
            for j in range(n):
                var p = exp(scores[j] - m)
                l += p
                var r = Int(idx_host.ptr[b * TOPK + j])
                for d in range(KV_LATENT_DIM):
                    acc[d] += p * Float64(
                        blocks_host.ptr[r * K_CACHE_DIM + d].cast[
                            DType.float64
                        ]()
                    )
            var o_base = (b * NUM_HEADS + h) * KV_LATENT_DIM
            for d in range(KV_LATENT_DIM):
                # An entry with no visible key attends nothing: zeros.
                var expected = acc[d] / l if n > 0 else Float64(0)
                var g = Float64(got.ptr[o_base + d].cast[DType.float64]())
                var err = abs(g - expected)
                # NaN must fail, and NaN compares false.
                max_err = max(max_err, err) if err == err else Float64(1e300)
    return max_err


def _run_case(
    name: String,
    cache_lengths: List[Int],
    topk_lengths: List[Int],
    has_topk_lengths: Bool,
    selection_lag: Int,
    ctx: DeviceContext,
) raises:
    comptime dtype = DType.float8_e4m3fn
    comptime out_dtype = DType.bfloat16
    comptime scale = Float32(0.0625)
    comptime kv_params = KVCacheStaticParams(
        num_heads=1, head_size=K_CACHE_DIM, is_mla=True
    )
    var batch_size = len(cache_lengths)
    var valid = List[Int]()

    # Each entry's selection: its valid slots name distinct real rows, the
    # slots past its top-k length (while still inside the visible prefix)
    # name poison rows, and the rest are `-1`. Rows are drawn from a
    # stride-coprime walk so neighbouring slots land on different pages.
    var idx_host = ctx.enqueue_create_host_buffer[.int32](batch_size * TOPK)
    var selected = List[Bool](length=NUM_ROWS, fill=False)
    var poison_rows = List[Int]()
    for r in range(NUM_ROWS // 2, NUM_ROWS):
        poison_rows.append(r)
    for b in range(batch_size):
        var visible = cache_lengths[b] + 1 - selection_lag
        var topk_len = topk_lengths[b] if has_topk_lengths else TOPK
        var n = min(topk_len, max(visible, 0))
        valid.append(n)
        for i in range(TOPK):
            var v: Int32 = -1
            if i < n:
                var row = ((b * TOPK + i) * 7919 + 17) % (NUM_ROWS // 2)
                selected[row] = True
                v = Int32(row)
            elif i < min(TOPK, max(visible, 0)):
                v = Int32(poison_rows[(b * 31 + i) % len(poison_rows)])
            idx_host[b * TOPK + i] = v

    var block_elems = NUM_ROWS * K_CACHE_DIM
    var blocks_host = ctx.enqueue_create_host_buffer[dtype](block_elems)
    for r in range(NUM_ROWS):
        for d in range(K_CACHE_DIM):
            var v = _row_value(r, d)
            if not selected[r]:
                v = v * 1000.0
            blocks_host[r * K_CACHE_DIM + d] = cast_saturating[dtype](v)
    var blocks_device = ctx.enqueue_create_buffer[dtype](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)

    var lut_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size * NUM_PAGES
    )
    for b in range(batch_size):
        for p in range(NUM_PAGES):
            lut_host[b * NUM_PAGES + p] = UInt32(p)
    var lut_device = ctx.enqueue_create_buffer[.uint32](batch_size * NUM_PAGES)
    ctx.enqueue_copy(lut_device, lut_host)

    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    var max_cache_len = 0
    for b in range(batch_size):
        cache_lengths_host[b] = UInt32(cache_lengths[b])
        max_cache_len = max(max_cache_len, cache_lengths[b])
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

    var idx_device = ctx.enqueue_create_buffer[.int32](batch_size * TOPK)
    ctx.enqueue_copy(idx_device, idx_host)
    var topk_host = ctx.enqueue_create_host_buffer[.int32](batch_size)
    for b in range(batch_size):
        topk_host[b] = Int32(topk_lengths[b])
    var topk_device = ctx.enqueue_create_buffer[.int32](batch_size)
    ctx.enqueue_copy(topk_device, topk_host)

    var q_size = batch_size * NUM_HEADS * K_CACHE_DIM
    var q_host = ctx.enqueue_create_host_buffer[dtype](q_size)
    for i in range(q_size):
        q_host[i] = Scalar[dtype](Float32((i % 97) - 48) * 0.05)
    var q_device = ctx.enqueue_create_buffer[dtype](q_size)
    ctx.enqueue_copy(q_device, q_host)

    var out_size = batch_size * NUM_HEADS * KV_LATENT_DIM
    var out_gather = ctx.enqueue_create_buffer[out_dtype](out_size)
    var out_rows = ctx.enqueue_create_buffer[out_dtype](out_size)
    var scratch = ctx.enqueue_create_buffer[dtype](
        batch_size * TOPK * K_CACHE_DIM
    )
    var gather_lengths = ctx.enqueue_create_buffer[.uint32](batch_size)
    var scalar_args = ctx.enqueue_create_buffer[.int64](3)
    ctx.enqueue_memset(scalar_args, 0)

    var blocks_tt = TileTensor(
        blocks_device,
        row_major(
            Int64(NUM_PAGES),
            Idx[1],
            Int64(1),
            Idx[PAGE_SIZE],
            Idx[1],
            Idx[K_CACHE_DIM],
        ),
    )
    var cache_lengths_tt = TileTensor(
        cache_lengths_device, row_major(Int64(batch_size))
    )
    var lut_tt = TileTensor(
        lut_device, row_major(Int64(batch_size), Int64(NUM_PAGES))
    )
    comptime Collection = PagedKVCacheCollection[
        dtype,
        kv_params,
        PAGE_SIZE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var kv_collection = Collection(
        rebind[Collection.blocks_tt_type](blocks_tt.as_unsafe_any_origin()),
        cache_lengths_tt.as_imm().as_unsafe_any_origin(),
        lut_tt.as_imm().as_unsafe_any_origin(),
        UInt32(1),
        UInt32(max_cache_len),
    )
    var k_cache = kv_collection.get_key_cache(0)
    var blocks_base = k_cache.blocks_raw_ptr()

    var q_tt = TileTensor(
        q_device.unsafe_ptr(),
        TileLayout(
            (batch_size, Idx[1], Idx[NUM_HEADS], Idx[K_CACHE_DIM]),
            (
                Idx[NUM_HEADS * K_CACHE_DIM],
                Idx[NUM_HEADS * K_CACHE_DIM],
                Idx[K_CACHE_DIM],
                Idx[1],
            ),
        ),
    )
    var out_layout = TileLayout(
        (batch_size, Idx[1], Idx[NUM_HEADS], Idx[KV_LATENT_DIM]),
        (
            Idx[NUM_HEADS * KV_LATENT_DIM],
            Idx[NUM_HEADS * KV_LATENT_DIM],
            Idx[KV_LATENT_DIM],
            Idx[1],
        ),
    )
    var scalar_args_tt = TileTensor(scalar_args.unsafe_ptr(), row_major[3]())

    # Reference: the production fallback.
    comptime gather_kernel = _amd_sparse_mla_gather_kv_rows_and_lengths[
        type_of(k_cache), K_CACHE_DIM, GATHER_ROWS_PER_BLOCK
    ]
    comptime gather_kernel_no_topk = _amd_sparse_mla_gather_kv_rows_and_lengths[
        type_of(k_cache),
        K_CACHE_DIM,
        GATHER_ROWS_PER_BLOCK,
        has_topk_lengths=False,
    ]
    var gather_grid = (ceildiv(TOPK, GATHER_ROWS_PER_BLOCK), batch_size, 1)
    var gather_block = (
        K_CACHE_DIM // _AMD_SPARSE_MLA_GATHER_VEC,
        GATHER_ROWS_PER_BLOCK,
        1,
    )
    if has_topk_lengths:
        ctx.enqueue_function[gather_kernel](
            k_cache,
            blocks_base,
            idx_device.unsafe_ptr(),
            topk_device.unsafe_ptr(),
            scratch.unsafe_ptr(),
            gather_lengths.unsafe_ptr(),
            Int32(TOPK),
            Int32(selection_lag),
            grid_dim=gather_grid,
            block_dim=gather_block,
        )
    else:
        ctx.enqueue_function[gather_kernel_no_topk](
            k_cache,
            blocks_base,
            idx_device.unsafe_ptr(),
            topk_device.unsafe_ptr(),
            scratch.unsafe_ptr(),
            gather_lengths.unsafe_ptr(),
            Int32(TOPK),
            Int32(selection_lag),
            grid_dim=gather_grid,
            block_dim=gather_block,
        )
    flare_mla_decoding[
        config=MHAConfig[dtype](NUM_HEADS, K_CACHE_DIM),
        decoding_warp_split_k=False,
    ](
        TileTensor(out_gather.unsafe_ptr(), out_layout),
        q_tt,
        TileTensor(
            scratch, row_major(batch_size, TOPK, Idx[1], Idx[K_CACHE_DIM])
        ),
        PaddingMask(
            TileTensor(gather_lengths.unsafe_ptr(), row_major(batch_size))
        ),
        scale,
        ctx,
        scalar_args_tt,
    )
    var gather_host = ctx.enqueue_create_host_buffer[out_dtype](out_size)
    ctx.enqueue_copy(gather_host, out_gather)
    ctx.synchronize()
    var q_view = TileTensor(q_host.unsafe_ptr(), row_major(q_size))
    var blocks_view = TileTensor(
        blocks_host.unsafe_ptr(), row_major(block_elems)
    )
    var idx_view = TileTensor(
        idx_host.unsafe_ptr(), row_major(batch_size * TOPK)
    )
    var gather_err = _oracle_max_err(
        TileTensor(gather_host.unsafe_ptr(), row_major(out_size)),
        q_view,
        blocks_view,
        idx_view,
        valid,
        Float64(scale),
    )

    var k_rows = SparseRowsMHAOperand[row_width=K_CACHE_DIM, predicated=True](
        KVCacheMHAOperand(k_cache),
        blocks_base,
        idx_device.unsafe_ptr(),
        TOPK,
        topk_device.unsafe_ptr(),
        has_topk_lengths,
        selection_lag,
    )
    var rows_host = ctx.enqueue_create_host_buffer[out_dtype](out_size)
    for num_partitions in [1, 8, 32, 64, 128, 256]:
        # A sentinel, so an output row the kernel never writes fails.
        ctx.enqueue_memset(out_rows, -12345.0)
        flare_mla_decoding_rows[
            config=MHAConfig[dtype](NUM_HEADS, K_CACHE_DIM)
        ](
            TileTensor(out_rows.unsafe_ptr(), out_layout),
            q_tt,
            k_rows,
            TOPK,
            SparseRowsDecodeMask(k_rows),
            scale,
            ctx,
            scalar_args_tt,
            num_partitions=num_partitions,
        )
        ctx.enqueue_copy(rows_host, out_rows)
        ctx.synchronize()

        var max_abs = Float64(0)
        var max_ref = Float64(0)
        var bad = 0
        for i in range(out_size):
            var expected = Float64(gather_host[i].cast[DType.float64]())
            var got = Float64(rows_host[i].cast[DType.float64]())
            var err = abs(got - expected)
            # NaN fails every comparison, so test the passing condition.
            if not (err <= ATOL + RTOL * abs(expected)):
                bad += 1
            if err == err:
                max_abs = max(max_abs, err)
            max_ref = max(max_ref, abs(expected))
        var rows_err = _oracle_max_err(
            TileTensor(rows_host.unsafe_ptr(), row_major(out_size)),
            q_view,
            blocks_view,
            idx_view,
            valid,
            Float64(scale),
        )
        print(
            name,
            " num_partitions=",
            num_partitions,
            " fp32_err rows=",
            rows_err,
            " gather=",
            gather_err,
            " max_abs_diff=",
            max_abs,
            " max_abs_ref=",
            max_ref,
            " mismatches=",
            bad,
            "/",
            out_size,
            sep="",
        )
        assert_true(bad == 0, "in-place sparse decode differs from gather")
        # Split-K changes rounding (fp8 P per partition, bf16 partials), not
        # the keys attended: the in-place error against the fp32 reference
        # must stay on the gather path's scale.
        assert_true(
            rows_err <= 2.0 * gather_err + 0.002,
            "in-place sparse decode is further from fp32 than the gather",
        )
        # A zero output would match a broken reference; the selections are
        # chosen so every entry attends at least one real row.
        assert_true(max_ref > 0.01, "reference output is degenerate")

    _ = blocks_device
    _ = lut_device
    _ = cache_lengths_device
    _ = idx_device
    _ = topk_device
    _ = q_device
    _ = out_gather
    _ = out_rows
    _ = scratch
    _ = gather_lengths
    _ = scalar_args


def main() raises:
    comptime if default_accelerator().is_amd_gpu():
        with DeviceContext() as ctx:
            # Long context: every slot valid.
            _run_case("batch1_long", [84000], [TOPK], False, 0, ctx)
            # Mixed lengths, including valid counts off the 16-row strip and
            # off the partition boundaries, a lone key, and an exact fit.
            var mixed: List[Int] = [84000, 3000, 2047, 2046, 1000, 100, 0, 37]
            var full_topk = List[Int](length=8, fill=TOPK)
            _run_case("batch8_mixed", mixed, full_topk, False, 0, ctx)
            # Per-entry top-k shorter than the visible prefix: the slots past
            # it name poison rows and must stay masked.
            var short_topk: List[Int] = [1500, 2048, 777, 2048, 64, 2048, 1, 9]
            _run_case("batch8_short_topk", mixed, short_topk, True, 0, ctx)
            # A row selected one token earlier holds one fewer visible key,
            # so at cache length 0 the entry has none and its output must
            # still be written.
            var lagged: List[Int] = [84000, 1000, 101, 2048, 0]
            var lag_topk = List[Int](length=5, fill=TOPK)
            _run_case("batch4_lag1", lagged, lag_topk, False, 1, ctx)
            print("test_mla_decode_sparse_rows_inplace_amd: ALL PASSED")
