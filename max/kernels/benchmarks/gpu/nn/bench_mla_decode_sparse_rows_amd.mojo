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
top-k 2048, page 128), the three launches of the two paths:

  gather : `_amd_sparse_mla_gather_kv_rows` copies each token's selected rows
           into a dense scratch.
  decode : `flare_mla_decoding` over that scratch with a `PaddingMask`.
  fused  : `flare_mla_decoding_rows` over a `SparseRowsMHAOperand`, which
           reads the same rows straight from the paged cache.

Every token selects 2048 scattered rows from a 1.2 GB cache, larger than
the MI355X's 256 MB last-level cache. Each token count runs with all slots
valid and with half of them `-1`, as when decoding at a 1k context. Prints
microseconds per launch (best of `REPEATS` timed batches) and checks the
fused output matches the two-step output bit for bit.
"""

from std.math import ceildiv
from std.memory import bitcast
from std.sys import default_accelerator

from max.gpu.host import DeviceContext
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Idx, TileTensor, row_major
from layout.tile_layout import Layout as TileLayout
from nn.attention.gpu.mla import flare_mla_decoding, flare_mla_decoding_rows
from nn.attention.gpu.mla_graph import (
    _AMD_SPARSE_MLA_GATHER_VEC,
    _amd_sparse_mla_gather_kv_rows,
)
from nn.attention.mha_mask import PaddingMask
from nn.attention.mha_operand import KVCacheMHAOperand, SparseRowsMHAOperand
from nn.attention.mha_utils import MHAConfig
from std.utils import IndexList

comptime NUM_HEADS = 8
comptime K_CACHE_DIM = 576
comptime KV_LATENT_DIM = 512
comptime PAGE_SIZE = 128
comptime TOPK = 2048
comptime CACHE_LEN = 2097152
comptime GATHER_ROWS_PER_BLOCK = 7
comptime ITERS = 20
# Remote executors are shared; the best of several timed batches screens out
# a neighbor's burst better than one long average.
comptime REPEATS = 5


def _bench(batch_size: Int, valid_len: Int, ctx: DeviceContext) raises:
    comptime dtype = DType.float8_e4m3fn
    comptime out_dtype = DType.bfloat16
    comptime scale = Float32(0.0625)
    comptime kv_params = KVCacheStaticParams(
        num_heads=1, head_size=K_CACHE_DIM, is_mla=True
    )
    comptime num_pages = CACHE_LEN // PAGE_SIZE

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

    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for b in range(batch_size):
        cache_lengths_host[b] = UInt32(CACHE_LEN)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

    # Scattered selections; with the identity LUT the encoded physical index
    # (block * page_size + offset) is the token position itself.
    var n_indices = batch_size * TOPK
    var idx_host = ctx.enqueue_create_host_buffer[.int32](n_indices)
    for b in range(batch_size):
        for i in range(TOPK):
            var row = (i * 40503 + b * 7919 + 1) % CACHE_LEN
            idx_host[b * TOPK + i] = Int32(row) if i < valid_len else -1
    var idx_device = ctx.enqueue_create_buffer[.int32](n_indices)
    ctx.enqueue_copy(idx_device, idx_host)

    var lengths_host = ctx.enqueue_create_host_buffer[.int32](batch_size)
    for b in range(batch_size):
        lengths_host[b] = Int32(valid_len)
    var lengths_device = ctx.enqueue_create_buffer[.int32](batch_size)
    ctx.enqueue_copy(lengths_device, lengths_host)

    var q_size = batch_size * NUM_HEADS * K_CACHE_DIM
    var q_host = ctx.enqueue_create_host_buffer[dtype](q_size)
    for i in range(q_size):
        q_host[i] = Scalar[dtype](Float32((i % 97) - 48) * 0.05)
    var q_device = ctx.enqueue_create_buffer[dtype](q_size)
    ctx.enqueue_copy(q_device, q_host)

    var out_size = batch_size * NUM_HEADS * KV_LATENT_DIM
    var out_two_step = ctx.enqueue_create_buffer[out_dtype](out_size)
    var out_fused = ctx.enqueue_create_buffer[out_dtype](out_size)
    var scratch = ctx.enqueue_create_buffer[dtype](
        batch_size * TOPK * K_CACHE_DIM
    )
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
    var out_two_step_tt = TileTensor(out_two_step.unsafe_ptr(), out_layout)
    var out_fused_tt = TileTensor(out_fused.unsafe_ptr(), out_layout)

    var lengths_tt = TileTensor(
        lengths_device.unsafe_ptr().bitcast[UInt32](), row_major(batch_size)
    )
    var scalar_args_tt = TileTensor(scalar_args.unsafe_ptr(), row_major[3]())
    var k_scratch = TileTensor(
        scratch, row_major(batch_size, TOPK, Idx[1], Idx[K_CACHE_DIM])
    )
    var k_rows = SparseRowsMHAOperand[row_width=K_CACHE_DIM](
        KVCacheMHAOperand(k_cache),
        k_cache.blocks_raw_ptr(),
        zero_row.unsafe_ptr(),
        idx_device.unsafe_ptr(),
        lengths_device.unsafe_ptr().bitcast[UInt32](),
        TOPK,
    )
    var blocks_base = k_cache.blocks_raw_ptr()

    @inline(.always)
    def gather(c: DeviceContext) raises {imm}:
        c.enqueue_function[
            _amd_sparse_mla_gather_kv_rows[
                type_of(k_cache), K_CACHE_DIM, GATHER_ROWS_PER_BLOCK
            ]
        ](
            k_cache,
            blocks_base,
            idx_device.unsafe_ptr(),
            lengths_device.unsafe_ptr(),
            scratch.unsafe_ptr(),
            Int32(TOPK),
            grid_dim=(ceildiv(TOPK, GATHER_ROWS_PER_BLOCK), batch_size, 1),
            block_dim=(
                K_CACHE_DIM // _AMD_SPARSE_MLA_GATHER_VEC,
                GATHER_ROWS_PER_BLOCK,
                1,
            ),
        )

    @inline(.always)
    def decode(c: DeviceContext) raises {imm}:
        flare_mla_decoding[
            config=MHAConfig[dtype](NUM_HEADS, K_CACHE_DIM),
            decoding_warp_split_k=False,
        ](
            out_two_step_tt,
            q_tt,
            k_scratch,
            PaddingMask(lengths_tt),
            scale,
            c,
            scalar_args_tt,
        )

    @inline(.always)
    def fused(c: DeviceContext) raises {imm}:
        flare_mla_decoding_rows[
            config=MHAConfig[dtype](NUM_HEADS, K_CACHE_DIM)
        ](
            out_fused_tt,
            q_tt,
            k_rows,
            TOPK,
            PaddingMask(lengths_tt),
            scale,
            c,
            scalar_args_tt,
        )

    gather(ctx)
    decode(ctx)
    fused(ctx)
    ctx.synchronize()

    var two_step_host = ctx.enqueue_create_host_buffer[out_dtype](out_size)
    var fused_host = ctx.enqueue_create_host_buffer[out_dtype](out_size)
    ctx.enqueue_copy(two_step_host, out_two_step)
    ctx.enqueue_copy(fused_host, out_fused)
    ctx.synchronize()
    var mismatches = 0
    for i in range(out_size):
        if bitcast[DType.uint16](two_step_host[i]) != bitcast[DType.uint16](
            fused_host[i]
        ):
            mismatches += 1

    var gather_ns = Int.MAX
    var decode_ns = Int.MAX
    var fused_ns = Int.MAX
    for _ in range(REPEATS):
        gather_ns = min(gather_ns, ctx.execution_time(gather, ITERS))
        decode_ns = min(decode_ns, ctx.execution_time(decode, ITERS))
        fused_ns = min(fused_ns, ctx.execution_time(fused, ITERS))
    var gather_us = Float64(gather_ns) / ITERS / 1000
    var decode_us = Float64(decode_ns) / ITERS / 1000
    var fused_us = Float64(fused_ns) / ITERS / 1000
    var two_step_us = gather_us + decode_us
    print(
        "tokens=",
        batch_size,
        " valid=",
        valid_len,
        " gather_us=",
        gather_us,
        " decode_us=",
        decode_us,
        " two_step_us=",
        two_step_us,
        " fused_us=",
        fused_us,
        " speedup=",
        two_step_us / fused_us,
        " mismatches=",
        mismatches,
        sep="",
    )

    _ = blocks_device
    _ = lut_device
    _ = cache_lengths_device
    _ = idx_device
    _ = lengths_device
    _ = q_device
    _ = out_two_step
    _ = out_fused
    _ = scratch
    _ = zero_row
    _ = scalar_args


def main() raises:
    comptime if default_accelerator().is_amd_gpu():
        with DeviceContext() as ctx:
            for batch_size in [1, 8, 64, 256, 1024]:
                for valid_len in [TOPK, TOPK // 2]:
                    _bench(batch_size, valid_len, ctx)
