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
"""AMD sparse MLA decode: gather-to-scratch vs reading the rows in place.

Times, at the GLM-5.3 per-rank shape (8 heads, 576-wide FP8 latent rows,
top-k 2048, page 128), one decode step's attention per path:

  gather   : `_amd_sparse_mla_gather_kv_rows_and_lengths` copies each
             token's selected rows into a dense scratch and writes the valid
             lengths, then `flare_mla_decoding` (split-K decode and reduce)
             runs over the scratch. This is the production decode arm.
  in place : `flare_mla_decoding_rows` over a `SparseRowsMHAOperand`, which
             reads the same rows straight from the paged cache, at each
             split-K partition count.

Each timed call walks enough distinct top-k selections to stream
`STREAM_BYTES` of selected rows, past the MI355X's 256 MB last-level cache,
because a decode step's layers each read different rows and a bench that
re-reads one selection measures cache hits. Every batch entry runs with all
top-k slots valid and with half of them `-1`, as when decoding at a 1k
context. Prints microseconds per decode step (median and range of `REPEATS`
timed calls) and each in-place output's distance from the gather output.

Optional arguments narrow a run, for example under a profiler:
`--batch N`, `--np N` (predicated in-place arm only, at N partitions),
`--gather` (gather arm only), and `--warm` (re-read one selection).
"""

from std.math import ceildiv
from std.collections import List
from std.sys import argv

from max.gpu.host import DeviceContext, default_accelerator
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

comptime NUM_HEADS = 8
comptime K_CACHE_DIM = 576
comptime KV_LATENT_DIM = 512
comptime PAGE_SIZE = 128
comptime TOPK = 2048
comptime CACHE_LEN = 2097152
comptime GATHER_ROWS_PER_BLOCK = 7
comptime STREAM_BYTES = 512 * 1024 * 1024
comptime REPEATS = 5


def _median(var xs: List[Float64]) -> Float64:
    for i in range(1, len(xs)):
        var j = i
        while j > 0 and xs[j - 1] > xs[j]:
            xs.swap_elements(j - 1, j)
            j -= 1
    return xs[len(xs) // 2]


def _report(
    name: String, batch_size: Int, valid_len: Int, var us: List[Float64]
):
    var lo = us[0]
    var hi = us[0]
    for u in us:
        lo = min(lo, u)
        hi = max(hi, u)
    print(
        "arm=",
        name,
        " tokens=",
        batch_size,
        " valid=",
        valid_len,
        " median_us=",
        _median(us^),
        " min_us=",
        lo,
        " max_us=",
        hi,
        sep="",
    )


@fieldwise_init
struct _Filter(Copyable, Movable):
    var batch: Int
    var num_partitions: Int
    var gather_only: Bool
    var warm: Bool


def _bench(
    batch_size: Int, valid_len: Int, filter: _Filter, ctx: DeviceContext
) raises:
    comptime dtype = DType.float8_e4m3fn
    comptime out_dtype = DType.bfloat16
    comptime scale = Float32(0.0625)
    comptime kv_params = KVCacheStaticParams(
        num_heads=1, head_size=K_CACHE_DIM, is_mla=True
    )
    comptime num_pages = CACHE_LEN // PAGE_SIZE
    var num_sets = 1 if filter.warm else max(
        4, ceildiv(STREAM_BYTES, batch_size * TOPK * K_CACHE_DIM)
    )

    # One shared cache: every batch entry maps page p to block p, like the
    # query tokens of one long prompt.
    var block_elems = num_pages * PAGE_SIZE * K_CACHE_DIM
    var blocks_host = ctx.enqueue_create_host_buffer[dtype](block_elems)
    for i in range(block_elems):
        blocks_host[i] = Scalar[dtype](Float32((i * 37 % 29) - 14) * 0.125)
    var blocks_device = ctx.enqueue_create_buffer[dtype](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)

    var lut_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size * num_pages
    )
    for b in range(batch_size):
        for p in range(num_pages):
            lut_host[b * num_pages + p] = UInt32(p)
    var lut_device = ctx.enqueue_create_buffer[.uint32](batch_size * num_pages)
    ctx.enqueue_copy(lut_device, lut_host)

    # The gather writes each row's valid length in closed form from the cache
    # length (`min(top_k, cache_length + 1)` at selection lag 0), so a row
    # with `valid_len` leading valid slots decodes at cache length
    # `valid_len - 1`. The selected rows themselves may lie anywhere.
    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for b in range(batch_size):
        cache_lengths_host[b] = UInt32(valid_len - 1)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

    # `num_sets` independent scattered selections; with the identity LUT the
    # encoded physical index (block * page_size + offset) is the token
    # position itself.
    var set_indices = batch_size * TOPK
    var n_indices = num_sets * set_indices
    var idx_host = ctx.enqueue_create_host_buffer[.int32](n_indices)
    for s in range(num_sets):
        for b in range(batch_size):
            for i in range(TOPK):
                var key = (s * batch_size + b) * TOPK + i
                var row = (key * 2654435761 + 12345) % CACHE_LEN
                idx_host[s * set_indices + b * TOPK + i] = (
                    Int32(row) if i < valid_len else -1
                )
    var idx_device = ctx.enqueue_create_buffer[.int32](n_indices)
    ctx.enqueue_copy(idx_device, idx_host)

    var lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for b in range(batch_size):
        lengths_host[b] = UInt32(valid_len)
    var lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(lengths_device, lengths_host)

    var q_size = batch_size * NUM_HEADS * K_CACHE_DIM
    var q_host = ctx.enqueue_create_host_buffer[dtype](q_size)
    for i in range(q_size):
        q_host[i] = Scalar[dtype](Float32((i % 97) - 48) * 0.05)
    var q_device = ctx.enqueue_create_buffer[dtype](q_size)
    ctx.enqueue_copy(q_device, q_host)

    var out_size = batch_size * NUM_HEADS * KV_LATENT_DIM
    var out_gather = ctx.enqueue_create_buffer[out_dtype](out_size)
    var out_inplace = ctx.enqueue_create_buffer[out_dtype](out_size)
    var scratch = ctx.enqueue_create_buffer[dtype](
        batch_size * TOPK * K_CACHE_DIM
    )
    var gather_lengths = ctx.enqueue_create_buffer[.uint32](batch_size)
    var zero_row = ctx.enqueue_create_buffer[dtype](K_CACHE_DIM)
    ctx.enqueue_memset(zero_row, 0)
    var scalar_args = ctx.enqueue_create_buffer[.int64](3)
    ctx.enqueue_memset(scalar_args, 0)
    ctx.synchronize()

    var blocks_tt = TileTensor(
        blocks_device,
        row_major(
            Int64(num_pages),
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
        lut_device, row_major(Int64(batch_size), Int64(num_pages))
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
        UInt32(0),
        UInt32(CACHE_LEN),
    )
    var k_cache = kv_collection.get_key_cache(0)

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
    var out_gather_tt = TileTensor(out_gather.unsafe_ptr(), out_layout)
    var out_inplace_tt = TileTensor(out_inplace.unsafe_ptr(), out_layout)

    var lengths_tt = TileTensor(
        lengths_device.unsafe_ptr(), row_major(batch_size)
    )
    var gather_lengths_tt = TileTensor(
        gather_lengths.unsafe_ptr(), row_major(batch_size)
    )
    var scalar_args_tt = TileTensor(scalar_args.unsafe_ptr(), row_major[3]())
    var k_scratch = TileTensor(
        scratch, row_major(batch_size, TOPK, Idx[1], Idx[K_CACHE_DIM])
    )
    var blocks_base = k_cache.blocks_raw_ptr()

    @inline(.always)
    def gather_step(c: DeviceContext, s: Int) raises {imm}:
        c.enqueue_function[
            _amd_sparse_mla_gather_kv_rows_and_lengths[
                type_of(k_cache),
                K_CACHE_DIM,
                GATHER_ROWS_PER_BLOCK,
                has_topk_lengths=False,
            ]
        ](
            k_cache,
            blocks_base,
            idx_device.unsafe_ptr().unsafe_offset(s * set_indices),
            MutPointer[Int32, MutAnyOrigin].unsafe_dangling(),
            scratch.unsafe_ptr(),
            gather_lengths.unsafe_ptr(),
            Int32(TOPK),
            Int32(0),
            grid_dim=(ceildiv(TOPK, GATHER_ROWS_PER_BLOCK), batch_size, 1),
            block_dim=(
                K_CACHE_DIM // _AMD_SPARSE_MLA_GATHER_VEC,
                GATHER_ROWS_PER_BLOCK,
                1,
            ),
        )
        flare_mla_decoding[
            config=MHAConfig[dtype](NUM_HEADS, K_CACHE_DIM),
            decoding_warp_split_k=False,
        ](
            out_gather_tt,
            q_tt,
            k_scratch,
            PaddingMask(gather_lengths_tt),
            scale,
            c,
            scalar_args_tt,
        )

    @inline(.always)
    def rows_step(c: DeviceContext, s: Int, num_partitions: Int) raises {imm}:
        var k_rows = SparseRowsMHAOperand[
            row_width=K_CACHE_DIM, predicated=True
        ](
            KVCacheMHAOperand(k_cache),
            blocks_base,
            idx_device.unsafe_ptr().unsafe_offset(s * set_indices),
            TOPK,
            MutPointer[Int32, MutAnyOrigin].unsafe_dangling(),
            False,
            0,
        )
        flare_mla_decoding_rows[
            config=MHAConfig[dtype](NUM_HEADS, K_CACHE_DIM)
        ](
            out_inplace_tt,
            q_tt,
            k_rows,
            TOPK,
            SparseRowsDecodeMask(k_rows),
            scale,
            c,
            scalar_args_tt,
            num_partitions=num_partitions,
        )

    @inline(.always)
    def inplace_step(
        c: DeviceContext, s: Int, num_partitions: Int
    ) raises {imm}:
        var k_rows = SparseRowsMHAOperand[row_width=K_CACHE_DIM](
            KVCacheMHAOperand(k_cache),
            blocks_base,
            zero_row.unsafe_ptr(),
            idx_device.unsafe_ptr().unsafe_offset(s * set_indices),
            lengths_device.unsafe_ptr(),
            TOPK,
        )
        flare_mla_decoding_rows[
            config=MHAConfig[dtype](NUM_HEADS, K_CACHE_DIM)
        ](
            out_inplace_tt,
            q_tt,
            k_rows,
            TOPK,
            PaddingMask(lengths_tt),
            scale,
            c,
            scalar_args_tt,
            num_partitions=num_partitions,
        )

    @inline(.always)
    def gather_all(c: DeviceContext) raises {imm}:
        for s in range(num_sets):
            gather_step(c, s)

    var gather_host = ctx.enqueue_create_host_buffer[out_dtype](out_size)
    var inplace_host = ctx.enqueue_create_host_buffer[out_dtype](out_size)
    gather_step(ctx, 0)
    ctx.enqueue_copy(gather_host, out_gather)
    ctx.synchronize()

    var gather_arm = filter.num_partitions == 0
    var us = List[Float64]()
    for _ in range(REPEATS if gather_arm else 0):
        us.append(
            Float64(ctx.execution_time(gather_all, 1))
            / Float64(num_sets)
            / 1000
        )
    if gather_arm:
        _report("gather", batch_size, valid_len, us^)

    var arms: List[Int] = [8, -16, -32, -64]
    if filter.gather_only:
        arms = []
    elif filter.num_partitions != 0:
        arms = [-filter.num_partitions]
    for arm in arms:
        # Positive: the zero-row in-place path; negative: predicated rows.
        var num_partitions = abs(arm)
        var predicated = arm < 0
        if predicated:
            rows_step(ctx, 0, num_partitions)
        else:
            inplace_step(ctx, 0, num_partitions)
        ctx.enqueue_copy(inplace_host, out_inplace)
        ctx.synchronize()
        var max_abs = Float64(0)
        var max_ref = Float64(0)
        for i in range(out_size):
            var ref_v = Float64(gather_host[i].cast[DType.float64]())
            var got = Float64(inplace_host[i].cast[DType.float64]())
            max_abs = max(max_abs, abs(got - ref_v))
            max_ref = max(max_ref, abs(ref_v))

        @inline(.always)
        def inplace_all(c: DeviceContext) raises {imm}:
            for s in range(num_sets):
                if predicated:
                    rows_step(c, s, num_partitions)
                else:
                    inplace_step(c, s, num_partitions)

        var us_in = List[Float64]()
        for _ in range(REPEATS):
            us_in.append(
                Float64(ctx.execution_time(inplace_all, 1))
                / Float64(num_sets)
                / 1000
            )
        _report(
            String(t"rows_np{num_partitions}") if predicated else String(
                t"zero_row_np{num_partitions}"
            ),
            batch_size,
            valid_len,
            us_in^,
        )
        print(
            "  vs gather: max_abs_diff=",
            max_abs,
            " max_abs_ref=",
            max_ref,
            sep="",
        )

    _ = blocks_device
    _ = lut_device
    _ = cache_lengths_device
    _ = idx_device
    _ = lengths_device
    _ = q_device
    _ = out_gather
    _ = out_inplace
    _ = scratch
    _ = gather_lengths
    _ = zero_row
    _ = scalar_args


def main() raises:
    var filter = _Filter(0, 0, False, False)
    var args = argv()
    var i = 1
    while i < len(args):
        if (args[i] == "--batch" or args[i] == "--np") and i + 1 >= len(args):
            raise Error(String(t"usage: {args[i]} N"))
        if args[i] == "--batch":
            filter.batch = Int(args[i + 1])
            i += 1
        elif args[i] == "--np":
            filter.num_partitions = Int(args[i + 1])
            i += 1
        elif args[i] == "--gather":
            filter.gather_only = True
        elif args[i] == "--warm":
            filter.warm = True
        i += 1
    comptime if default_accelerator().is_amd_gpu():
        with DeviceContext() as ctx:
            for batch_size in [1, 8, 16]:
                if filter.batch != 0 and batch_size != filter.batch:
                    continue
                for valid_len in [TOPK, TOPK // 2]:
                    if filter.batch != 0 and valid_len != TOPK:
                        continue
                    _bench(batch_size, valid_len, filter, ctx)
