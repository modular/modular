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
# The SM100 BF16 grouped matmul widens its token tile when experts hold many
# rows. `grouped_matmul` must then match the 32-token tile bit for bit (same K
# order) and the naive kernel within bf16 rounding, including experts that span
# several tiles, where a store past an expert's last row clobbers the next one.

from std.math import ceildiv

from max.gpu import block_dim, global_idx, grid_dim
from max.gpu.host import DeviceBuffer, DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from linalg.grouped_matmul import grouped_matmul, naive_grouped_matmul
from linalg.grouped_matmul_sm100 import grouped_matmul_sm100_persistent
from linalg.utils_gpu import MatmulConfig
from std.testing import assert_equal, assert_true
from std.utils.numerics import nan

comptime dtype = DType.bfloat16

# How the output is written: the kernel's store, a compute epilogue ahead of
# it, or a store epilogue into another buffer.
comptime PLAIN = 0
comptime COMPUTE = 1
comptime STORE = 2


def fill_kernel(
    ptr: MutPointer[Scalar[dtype], MutAnyOrigin], n: Int64, stream: UInt32
):
    var i = Int(global_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    while i < Int(n):
        var x = UInt32(i & 0xFFFFFFFF) * 0x9E3779B9 + stream * 0x85EBCA6B
        x ^= UInt32(i >> 32) * 0xC2B2AE35
        x ^= x >> 16
        x *= 0x7FEB352D
        x ^= x >> 15
        ptr[i] = Scalar[dtype](Float32(Int32(x >> 24) - 128) / 128.0)
        i += stride


@inline(.always)
def shift[
    dt: DType, width: SIMDLength
](row: Int, col: Int, val: SIMD[dt, width]) -> SIMD[dt, width]:
    var out = val
    comptime for i in range(width):
        out[i] += Scalar[dt]((row + col + i) % 21 - 10)
    return out


def fill(ctx: DeviceContext, buf: DeviceBuffer[dtype], stream: UInt32) raises:
    ctx.enqueue_function[fill_kernel](
        buf.unsafe_ptr(), Int64(len(buf)), stream, grid_dim=4096, block_dim=256
    )


def run[
    N: Int, K: Int, E: Int, mode: Int, tile_32: Bool
](
    ctx: DeviceContext,
    c: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    out_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
    a: TileTensor[dtype, address_space=.GENERIC, ...],
    b: TileTensor[dtype, address_space=.GENERIC, ...],
    offsets: TileTensor[DType.uint32, address_space=.GENERIC, ...],
    ids: TileTensor[DType.int32, address_space=.GENERIC, ...],
    stats: TileTensor[DType.uint32, address_space=.GENERIC, ...],
) raises:
    """Runs `grouped_matmul`, or the 32-token tile it used before."""

    @inline(.always)
    def compute_fn[
        dt: DType, width: SIMDLength, *, alignment: Int
    ](idx: Coord, val: SIMD[dt, width]) -> SIMD[dt, width]:
        return shift(Int(idx[0]), Int(idx[1]), val)

    @inline(.always)
    def store_fn[
        dt: DType, width: SIMDLength, *, alignment: Int
    ](idx: Coord, val: SIMD[dt, width]) {var out_ptr}:
        out_ptr.unsafe_offset(Int(idx[0]) * N + Int(idx[1])).unsafe_store[
            width=width, alignment=alignment
        ](shift(Int(idx[0]), Int(idx[1]), val).cast[dtype]())

    comptime if tile_32:
        comptime config = MatmulConfig[dtype, dtype, dtype, True](
            block_tile_shape=(128, 16, 64),
            mma_shape=(256, 32, 16),
            cluster_shape=(2, 1, 1),
        )
        grouped_matmul_sm100_persistent[
            c_type=dtype,
            a_type=dtype,
            b_type=dtype,
            transpose_b=True,
            config=config,
            cta_group=2,
            has_epilogue_fn=mode == STORE,
            has_compute_fn=mode == COMPUTE,
        ](
            c,
            a.as_imm(),
            offsets.as_imm(),
            b.as_imm(),
            ids.as_imm(),
            stats.as_imm(),
            ctx,
            store_fn,
            compute_fn,
        )
    else:
        grouped_matmul[
            has_epilogue_fn=mode == STORE, has_compute_fn=mode == COMPUTE
        ](
            c,
            a.as_imm(),
            b.as_imm(),
            offsets.as_imm(),
            ids.as_imm(),
            stats.as_imm(),
            ctx,
            epilogue_fn=store_fn,
            compute_fn=compute_fn,
        )


def check[
    N: Int, K: Int, E: Int, mode: Int = PLAIN
](ctx: DeviceContext, counts: List[Int], expert_ids: List[Int]) raises:
    comptime assert N % 256 == 0, "the token tile varies only at two CTAs"
    var num_active = len(counts)
    var M = 0
    var max_count = 0
    for n in counts:
        M += n
        max_count = max(max_count, n)
    print("N", N, "K", K, "E", E, "mode", mode, "M", M, "active", num_active)

    var a = ctx.enqueue_create_buffer[dtype](max(M, 1) * K)
    var b = ctx.enqueue_create_buffer[dtype](E * N * K)
    # One output per arm: the kernel's own store, or the store epilogue's.
    var c = ctx.enqueue_create_buffer[dtype](max(M, 1) * N)
    var c_32 = ctx.enqueue_create_buffer[dtype](max(M, 1) * N)
    var c_ref = ctx.enqueue_create_buffer[dtype](max(M, 1) * N)
    var offsets = ctx.enqueue_create_buffer[DType.uint32](E + 1)
    var ids = ctx.enqueue_create_buffer[DType.int32](E)
    var stats = ctx.enqueue_create_buffer[DType.uint32](2)
    fill(ctx, a, 1)
    fill(ctx, b, 2)
    # A row nobody writes stays NaN and fails the naive comparison.
    ctx.enqueue_memset(c, nan[dtype]())
    ctx.enqueue_memset(c_32, nan[dtype]())
    with offsets.map_to_host() as h:
        h[0] = 0
        for i in range(E):
            h[i + 1] = h[i] + UInt32(counts[i] if i < num_active else 0)
    with ids.map_to_host() as h:
        for i in range(E):
            h[i] = Int32(expert_ids[i] if i < num_active else 0)
    with stats.map_to_host() as h:
        h[0] = UInt32(max_count)
        h[1] = UInt32(num_active)

    var a_t = TileTensor(a, row_major(M, Idx[K]))
    var b_t = TileTensor(b, row_major[E, N, K]())
    var offsets_t = TileTensor(offsets, row_major(E + 1))
    var ids_t = TileTensor(ids, row_major(E))
    var stats_t = TileTensor(stats, row_major(2))
    var c_t = TileTensor(c, row_major(M, Idx[N]))
    var c_32_t = TileTensor(c_32, row_major(M, Idx[N]))
    # A store epilogue writes through the pointer; `c` only gives the shape.
    var scratch = ctx.enqueue_create_buffer[dtype](max(M, 1) * N)
    var scratch_t = TileTensor(scratch, row_major(M, Idx[N]))
    if mode == STORE:
        run[N, K, E, mode, False](
            ctx,
            scratch_t,
            c.unsafe_ptr().as_unsafe_any_origin(),
            a_t,
            b_t,
            offsets_t,
            ids_t,
            stats_t,
        )
        run[N, K, E, mode, True](
            ctx,
            scratch_t,
            c_32.unsafe_ptr().as_unsafe_any_origin(),
            a_t,
            b_t,
            offsets_t,
            ids_t,
            stats_t,
        )
    else:
        run[N, K, E, mode, False](
            ctx,
            c_t,
            c.unsafe_ptr().as_unsafe_any_origin(),
            a_t,
            b_t,
            offsets_t,
            ids_t,
            stats_t,
        )
        run[N, K, E, mode, True](
            ctx,
            c_32_t,
            c_32.unsafe_ptr().as_unsafe_any_origin(),
            a_t,
            b_t,
            offsets_t,
            ids_t,
            stats_t,
        )
    if M == 0:
        ctx.synchronize()
        return

    naive_grouped_matmul(
        TileTensor(c_ref, row_major(M, Idx[N])),
        a_t.as_imm(),
        b_t.as_imm(),
        offsets_t.as_imm(),
        ids_t.as_imm(),
        max_count,
        num_active,
        ctx,
    )
    var got = ctx.enqueue_create_host_buffer[dtype](M * N)
    var got_32 = ctx.enqueue_create_host_buffer[dtype](M * N)
    var want = ctx.enqueue_create_host_buffer[dtype](M * N)
    ctx.enqueue_copy(got, c)
    ctx.enqueue_copy(got_32, c_32)
    ctx.enqueue_copy(want, c_ref)
    ctx.synchronize()

    var differ = 0
    var bad = 0
    var abs_sum: Float64 = 0
    for i in range(M * N):
        var row, col = divmod(i, N)
        var expect = want[i]
        comptime if mode != PLAIN:
            expect = shift(row, col, expect)
        var x = got[i].cast[DType.float32]()
        var y = expect.cast[DType.float32]()
        abs_sum += Float64(abs(y))
        if got[i].to_bits() != got_32[i].to_bits():
            differ += 1
        if not (abs(x - y) <= 0.02 * abs(y) + 0.5):
            bad += 1
            if bad <= 8:
                print(
                    "  row",
                    row,
                    "col",
                    col,
                    "got",
                    got[i],
                    "tile32",
                    got_32[i],
                    "want",
                    expect,
                )
    print("  differ_from_32", differ, "bad", bad)
    assert_true(abs_sum > 0, "reference is all zeros")
    assert_equal(differ, 0)
    assert_equal(bad, 0)
    _ = scratch^


# Rows per expert of a 16K-token prefill chunk on one GPU of eight.
comptime PREFILL_COUNTS = """
67,74,68,66,64,55,77,63,61,63,68,73,64,68,61,57,
50,57,60,64,54,79,60,76,72,58,85,53,67,48,79,57,
65,55,63,66,54,75,65,70,82,67,69,64,54,64,70,51,
69,86,61,65,46,71,56,61,60,69,51,70,72,76,45,55,
63,62,71,58,77,66,62,62,69,75,68,72,69,64,50,60,
66,60,62,52,61,57,57,66,60,52,66,63,67,80,89,69,
65,62,56,69,55,50,63,71,66,53,82,70,76,66,62,63,
72,50,65,68,80,68,59,70,86,73,76,72,63,65,65,48,
72,56,75,68,49,53,66,52,57,68,60,73,69,61,57,76,
64,60,56,79,52,69,74,68,49,71,67,58,69,66,60,62,
78,71,50,63,70,58,47,76,64,62,65,61,59,60,57,81,
75,48,57,54,61,71,64,68,67,57,51,55,60,72,68,66,
67,73,64,59,75,74,71,42,73,68,65,67,55,59,71,70,
82,69,59,60,66,63,60,65,87,62,64,62,74,60,67,63,
60,60,73,68,79,61,66,65,68,58,65,72,64,56,60,67,
72,67,51,77,62,50,73,65,69,69,59,64,39,63,50,63
"""


def parse_counts(s: StringSpan) raises -> List[Int]:
    var out = List[Int]()
    for part in s.split(","):
        out.append(Int(part.strip()))
    return out^


def iota(n: Int) -> List[Int]:
    var out = List[Int]()
    for i in range(n):
        out.append(i)
    return out^


def main() raises:
    with DeviceContext() as ctx:
        # Ragged and prime counts over 7 of 11 experts; the mean picks
        # 32, 64 and 128 in turn, with partial last tiles.
        comptime E = 11
        var ids: List[Int] = [9, 0, 4, 10, 2, 7, 5]
        check[512, 1024, E](ctx, [1, 3, 2, 7, 1, 5, 2], ids)
        check[512, 1024, E](ctx, [37, 53, 41, 29, 61, 47, 43], ids)
        check[512, 1024, E](ctx, [97, 131, 89, 113, 101, 127, 109], ids)
        check[512, 1024, E](ctx, [211, 3, 1, 197, 89, 2, 151], ids)
        # One row, and an empty batch.
        check[512, 1024, E](ctx, [1], [3])
        check[512, 1024, E](ctx, List[Int](), List[Int]())
        # Inactive groups (expert -1) write zeros, at every width.
        check[512, 1024, E](ctx, [5, 7], [-1, 6])
        check[512, 1024, E](ctx, [173, 139, 157], [-1, 4, -1])
        check[512, 1024, E](ctx, [161, 183, 147], [2, -1, 8])
        # Both epilogues, at 64- and 128-token tiles.
        check[512, 1024, E, COMPUTE](ctx, [37, 53, 41, 29, 61, 47, 43], ids)
        check[512, 1024, E, COMPUTE](ctx, [211, 3, 1, 197, 89, 2, 151], ids)
        check[512, 1024, E, STORE](ctx, [37, 53, 41, 29, 61, 47, 43], ids)
        check[512, 1024, E, STORE](ctx, [211, 3, 1, 197, 89, 2, 151], ids)
        # The largest a Muse Spark server sends: its 16K-token prefill chunk
        # over 256 BF16 experts (gate_up, then down).
        var counts = parse_counts(PREFILL_COUNTS)
        check[8192, 4096, 256](ctx, counts, iota(256))
        check[4096, 4096, 256](ctx, counts, iota(256))
