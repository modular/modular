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
"""Checks `multistage_mma` over `TileTensorIter` operands against a host
reference.

Each block runs one BMxBN output tile and dumps the per-thread accumulator
registers. The host maps every register back to its `(m, n)` through the
m16n8 accumulator fragment layout and compares it with a float64 matmul of
the same inputs. Covered: global-memory pipelines (both transpose_b modes), a
masked A operand with a partial last row tile (whose padded rows must come
out zero), a static-iteration shared memory pipeline, and an fp32 register A
operand. The shared-memory operands are staged with the pipeline's own
swizzled async copy, as the attention kernels stage Q.
"""

from std.sys import size_of
from std.math import ceildiv
from std.testing import assert_almost_equal, assert_equal, assert_true
from std.utils.numerics import get_accum_type

from layout import Coord, TileTensor, row_major
from layout.coord import ComptimeInt
from layout.tensor_core import get_fragment_size, get_mma_shape
from layout.tile_async_copy import copy_dram_to_sram_async_ldmatrix
from layout.tile_iterator import TileTensorIter
from layout.tile_layout import Layout as TileLayout
from layout.tile_tensor import stack_allocation

from linalg.matmul.gpu._multistage_gemm_gpu import multistage_mma
from max.gpu import WARP_SIZE, block_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, HostBuffer
from max.gpu.memory import (
    AddressSpace,
    async_copy_commit_group,
    async_copy_wait_all,
    external_memory,
)

comptime dtype = DType.bfloat16
comptime accum_type = get_accum_type[dtype]()
comptime BM = 64
comptime BN = 64
comptime BK = 32
comptime WM = 32
comptime WN = 32
comptime num_threads = (BM // WM) * (BN // WN) * WARP_SIZE
comptime num_stages = 3
comptime K = 256
comptime N = 128

comptime mma_shape = get_mma_shape[dtype, accum_type]()
comptime frag = get_fragment_size[mma_shape]()
comptime num_m_mmas = WM // mma_shape[0]
comptime num_n_mmas = WN // mma_shape[1]
comptime c_regs = num_m_mmas * num_n_mmas * frag[2]

comptime atol = 1e-2
comptime rtol = 1e-3


def _a_value(i: Int) -> Scalar[dtype]:
    return Scalar[dtype](Float32((i * 7 + 3) % 29) / 8 - 1.5)


def _b_value(i: Int) -> Scalar[dtype]:
    return Scalar[dtype](Float32((i * 5 + 1) % 31) / 8 - 1.75)


def _gmem_layout[
    rows: Int, cols: Int, row_stride: Int
]() -> TileLayout[
    shape_types=Coord[ComptimeInt[rows], ComptimeInt[cols]].element_types,
    stride_types=Coord[ComptimeInt[row_stride], ComptimeInt[1]].element_types,
]:
    return {
        Coord[ComptimeInt[rows], ComptimeInt[cols]](),
        Coord[ComptimeInt[row_stride], ComptimeInt[1]](),
    }


def _frag_coords[warp_tile_n: Int](thread: Int, reg: Int) -> Tuple[Int, Int]:
    """Maps a thread's accumulator register to its `(row, col)` in the block
    tile. `reg` indexes the dumped `(num_m_mmas * num_n_mmas, 4)` register
    tile row-major; mma `n_mma * num_m_mmas + m_mma` holds one m16n8 C
    fragment.
    """
    comptime warps_n = BN // warp_tile_n
    comptime m_mmas = WM // mma_shape[0]
    var warp, lane = divmod(thread, WARP_SIZE)
    var warp_y, warp_x = divmod(warp, warps_n)
    var mma, j = divmod(reg, frag[2])
    var n_mma, m_mma = divmod(mma, m_mmas)
    var row = warp_y * WM + m_mma * mma_shape[0] + lane // 4 + 8 * (j // 2)
    var col = (
        warp_x * warp_tile_n + n_mma * mma_shape[1] + (lane % 4) * 2 + (j % 2)
    )
    return (row, col)


def _dump(
    c: TileTensor[mut=True, accum_type, ...],
    out_ptr: MutPointer[Scalar[accum_type], MutAnyOrigin],
):
    var base = (
        (block_idx.y * (N // BN) + block_idx.x) * num_threads + thread_idx.x
    ) * c_regs
    comptime for i in range(num_m_mmas * num_n_mmas):
        comptime for j in range(frag[2]):
            out_ptr[base + i * frag[2] + j] = c[i, j][0]


def global_pipeline_kernel[
    transpose_b: Bool,
    masked_a: Bool,
](
    a_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
    b_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
    out_ptr: MutPointer[Scalar[accum_type], MutAnyOrigin],
    M_arg: Int32,
):
    var M = Int(M_arg)
    comptime a_smem_size = num_stages * BM * BK
    comptime b_smem_size = num_stages * BK * BN
    comptime BD_0 = BN if transpose_b else BK
    comptime BD_1 = BK if transpose_b else BN

    var a_smem = external_memory[
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
        alignment=16,
    ]()
    var b_smem = a_smem.unsafe_offset(a_smem_size).unsafe_bitcast[
        Scalar[dtype]
    ]()

    var a_rows = min(BM, M - block_idx.y * BM)
    var a_block = a_ptr.unsafe_offset(block_idx.y * BM * K)
    var b_block = b_ptr.unsafe_offset(
        block_idx.x * BN * K if transpose_b else block_idx.x * BN
    )

    var c = stack_allocation[
        dtype=accum_type, address_space=AddressSpace.LOCAL
    ](row_major[num_m_mmas * num_n_mmas, frag[2]]()).fill(0)

    comptime a_smem_layout = row_major[BM, BK]()
    comptime b_smem_layout = row_major[BD_0, BD_1]()
    var a_smem_iter = TileTensorIter[
        dtype,
        type_of(a_smem_layout),
        _,
        address_space=AddressSpace.SHARED,
        circular=True,
    ](a_smem, a_smem_layout, bound=a_smem_size)
    var b_smem_iter = TileTensorIter[
        dtype,
        type_of(b_smem_layout),
        _,
        address_space=AddressSpace.SHARED,
        circular=True,
    ](b_smem, b_smem_layout, bound=b_smem_size)

    comptime a_gmem = _gmem_layout[BM, BK, K]()
    var a_iter = TileTensorIter[dtype, type_of(a_gmem), _, masked=masked_a](
        a_block,
        a_gmem,
        bound=BM * K,
        stride=BK,
        num_valid_rows=a_rows,
    )
    comptime b_gmem = _gmem_layout[BD_0, BD_1, K if transpose_b else N]()
    var b_iter = TileTensorIter[dtype, type_of(b_gmem), _](
        b_block,
        b_gmem,
        bound=K * N,
        stride=BK if transpose_b else BK * N,
    )
    multistage_mma[
        BM,
        BN,
        BK,
        WM,
        WN,
        num_threads,
        num_stages,
        transpose_b,
    ](c, a_iter, b_iter, a_smem_iter, b_smem_iter, K // BK)

    _dump(c, out_ptr)


comptime smem_depth = 128


def _stage_tiles[
    rows: Int, cols: Int, row_stride: Int, threads: Int
](
    smem: MutPointer[Scalar[dtype], _, address_space=AddressSpace.SHARED],
    gmem: MutPointer[Scalar[dtype], MutAnyOrigin],
    num_tiles: Int,
    tile_step: Int,
):
    """Stages `num_tiles` row-major `(rows, cols)` tiles of a `row_stride`
    matrix back to back in shared memory, swizzled for `ldmatrix`.
    """
    comptime simd = 8
    comptime thread_layout = row_major[
        min(threads, rows * cols // simd) * simd // cols, cols // simd
    ]()
    comptime smem_layout = row_major[rows, cols]()
    comptime gmem_layout = _gmem_layout[rows, cols, row_stride]()
    for t in range(num_tiles):
        var dst = TileTensor(smem.unsafe_offset(t * rows * cols), smem_layout)
        var src = TileTensor(gmem.unsafe_offset(t * tile_step), gmem_layout)
        copy_dram_to_sram_async_ldmatrix[
            thread_layout, simd_size=simd, swizzle=True, num_threads=threads
        ](dst, src)
    async_copy_commit_group()
    async_copy_wait_all()
    barrier()


def smem_pipeline_kernel(
    a_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
    b_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
    out_ptr: MutPointer[Scalar[accum_type], MutAnyOrigin],
):
    """QK-style: both operands already in shared memory, static trip count.
    A is `(BM, depth)` and B is `(BN, depth)`, both row-major in global
    memory.
    """
    comptime num_tiles = smem_depth // BK
    comptime a_smem_size = num_tiles * BM * BK
    comptime b_smem_size = num_tiles * BN * BK

    var a_smem = external_memory[
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
        alignment=16,
    ]()
    var b_smem = a_smem.unsafe_offset(a_smem_size).unsafe_bitcast[
        Scalar[dtype]
    ]()
    _stage_tiles[BM, BK, smem_depth, num_threads](a_smem, a_ptr, num_tiles, BK)
    _stage_tiles[BN, BK, smem_depth, num_threads](b_smem, b_ptr, num_tiles, BK)

    var c = stack_allocation[
        dtype=accum_type, address_space=AddressSpace.LOCAL
    ](row_major[num_m_mmas * num_n_mmas, frag[2]]()).fill(0)

    comptime a_layout = row_major[BM, BK]()
    comptime b_layout = row_major[BN, BK]()
    var a_iter = TileTensorIter[
        dtype, type_of(a_layout), _, address_space=AddressSpace.SHARED
    ](a_smem, a_layout, bound=a_smem_size)
    var b_iter = TileTensorIter[
        dtype,
        type_of(b_layout),
        _,
        address_space=AddressSpace.SHARED,
        circular=True,
    ](b_smem, b_layout, bound=b_smem_size)
    multistage_mma[
        BM,
        BN,
        BK,
        WM,
        WN,
        num_threads,
        num_stages,
        True,
        prefetch_init=False,
        static_num_iters=num_tiles,
    ](c, a_iter, b_iter, a_iter, b_iter, num_tiles)

    _dump(c, out_ptr)


comptime LWN = 64
comptime local_threads = (BM // WM) * (BN // LWN) * WARP_SIZE
comptime local_num_n_mmas = LWN // mma_shape[1]
comptime local_c_regs = num_m_mmas * local_num_n_mmas * frag[2]
comptime p_rows = num_m_mmas * local_num_n_mmas


def _p_value(thread: Int, i: Int, j: Int) -> Scalar[accum_type]:
    return Scalar[accum_type]((thread * 13 + i * 5 + j * 3) % 37) / 7 - 2.3


def local_a_kernel(
    b_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
    out_ptr: MutPointer[Scalar[accum_type], MutAnyOrigin],
):
    """PV-style: A operand is an fp32 register tile, B comes from shared.
    B is `(BN, BN)` row-major in global memory.
    """
    comptime a_smem_size = num_stages * BM * BK
    comptime b_smem_size = (BN // BK) * BK * BN
    comptime p_tile_rows = mma_shape[2] // mma_shape[1] * num_m_mmas

    var a_smem = external_memory[
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
        alignment=16,
    ]()
    var b_smem = a_smem.unsafe_offset(a_smem_size).unsafe_bitcast[
        Scalar[dtype]
    ]()
    _stage_tiles[BK, BN, BN, local_threads](b_smem, b_ptr, BN // BK, BK * BN)

    var p = stack_allocation[
        dtype=accum_type, address_space=AddressSpace.LOCAL
    ](row_major[p_rows, frag[2]]())
    comptime for i in range(p_rows):
        comptime for j in range(frag[2]):
            p[i, j] = _p_value(Int(thread_idx.x), i, j)

    var c = stack_allocation[
        dtype=accum_type, address_space=AddressSpace.LOCAL
    ](row_major[p_rows, frag[2]]()).fill(0)

    comptime p_layout = row_major[p_tile_rows, frag[2]]()
    comptime b_layout = row_major[BK, BN]()
    comptime a_smem_layout = row_major[BM, BK]()
    var p_iter = TileTensorIter[
        accum_type,
        type_of(p_layout),
        _,
        address_space=AddressSpace.LOCAL,
    ](p, p_layout, bound=p_rows * frag[2])
    var b_iter = TileTensorIter[
        dtype,
        type_of(b_layout),
        _,
        address_space=AddressSpace.SHARED,
        circular=True,
    ](b_smem, b_layout, bound=b_smem_size)
    var a_smem_iter = TileTensorIter[
        dtype,
        type_of(a_smem_layout),
        _,
        address_space=AddressSpace.SHARED,
        circular=True,
    ](a_smem, a_smem_layout, bound=a_smem_size)
    multistage_mma[
        BM,
        BN,
        BK,
        WM,
        LWN,
        local_threads,
        num_stages,
        False,
        swizzle_a=False,
        prefetch_init=False,
        static_num_iters=BN // BK,
    ](c, p_iter, b_iter, a_smem_iter, b_iter, BN // BK)

    var base = thread_idx.x * local_c_regs
    comptime for i in range(p_rows):
        comptime for j in range(frag[2]):
            out_ptr[base + i * frag[2] + j] = c[i, j][0]


def _check_block[
    warp_tile_n: Int, block_threads: Int
](
    regs_host: HostBuffer[accum_type],
    out_base: Int,
    expected: List[Float64],
    expected_cols: Int,
    row0: Int,
    col0: Int,
    valid_rows: Int,
) raises:
    """Compares one block's dumped registers with `expected[row, col]`; rows
    at or past `valid_rows` must be exactly zero.
    """
    comptime regs = WM // mma_shape[0] * (warp_tile_n // mma_shape[1]) * frag[2]
    var nonzero = 0
    for t in range(block_threads):
        for r in range(regs):
            var row, col = _frag_coords[warp_tile_n](t, r)
            var got = Float64(regs_host[out_base + t * regs + r])
            if row >= valid_rows:
                assert_equal(got, 0.0)
                continue
            var want = expected[(row0 + row) * expected_cols + col0 + col]
            assert_almost_equal(got, want, atol=atol, rtol=rtol)
            if got != 0:
                nonzero += 1
    # An all-zero result would match a zero reference trivially.
    assert_true(nonzero > valid_rows * BN // 2)


def run_smem(ctx: DeviceContext) raises:
    comptime out_len = num_threads * c_regs
    comptime num_tiles = smem_depth // BK
    comptime a_len = BM * smem_depth
    comptime b_len = BN * smem_depth
    var a_host = ctx.enqueue_create_host_buffer[dtype](a_len)
    var b_host = ctx.enqueue_create_host_buffer[dtype](b_len)
    ctx.synchronize()
    for i in range(a_len):
        a_host[i] = _a_value(i)
    for i in range(b_len):
        b_host[i] = _b_value(i)
    var a_dev = ctx.enqueue_create_buffer[dtype](a_len)
    var b_dev = ctx.enqueue_create_buffer[dtype](b_len)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    var out = ctx.enqueue_create_buffer[accum_type](out_len)
    comptime smem_bytes = num_tiles * (BM + BN) * BK * size_of[dtype]()
    ctx.enqueue_function[smem_pipeline_kernel](
        a_dev,
        b_dev,
        out,
        grid_dim=1,
        block_dim=num_threads,
        shared_mem_bytes=smem_bytes,
    )

    var expected = List[Float64](length=BM * BN, fill=0)
    for m in range(BM):
        for n in range(BN):
            var acc = Float64(0)
            for k in range(smem_depth):
                acc += Float64(a_host[m * smem_depth + k]) * Float64(
                    b_host[n * smem_depth + k]
                )
            expected[m * BN + n] = acc

    with out.map_to_host() as host:
        _check_block[WN, num_threads](host, 0, expected, BN, 0, 0, BM)
    print("shared-memory static pipeline: matches reference")


def run_local_a(ctx: DeviceContext) raises:
    comptime out_len = local_threads * local_c_regs
    comptime b_len = BN * BN
    var b_host = ctx.enqueue_create_host_buffer[dtype](b_len)
    ctx.synchronize()
    for i in range(b_len):
        b_host[i] = _b_value(i)
    var b_dev = ctx.enqueue_create_buffer[dtype](b_len)
    ctx.enqueue_copy(b_dev, b_host)
    var out = ctx.enqueue_create_buffer[accum_type](out_len)
    comptime smem_bytes = (
        num_stages * BM * BK + (BN // BK) * BK * BN
    ) * size_of[dtype]()
    ctx.enqueue_function[local_a_kernel](
        b_dev,
        out,
        grid_dim=1,
        block_dim=local_threads,
        shared_mem_bytes=smem_bytes,
    )

    # P is a (BM, BN) fp32 accumulator tile spread over the threads in m16n8
    # fragments; the pipeline casts it to bf16 before the mma.
    var p = List[Float64](length=BM * BN, fill=0)
    for t in range(local_threads):
        for r in range(local_c_regs):
            var row, col = _frag_coords[LWN](t, r)
            var i, j = divmod(r, frag[2])
            p[row * BN + col] = Float64(_p_value(t, i, j).cast[dtype]())
    var expected = List[Float64](length=BM * BN, fill=0)
    for m in range(BM):
        for n in range(BN):
            var acc = Float64(0)
            for k in range(BN):
                acc += p[m * BN + k] * Float64(b_host[k * BN + n])
            expected[m * BN + n] = acc

    with out.map_to_host() as host:
        _check_block[LWN, local_threads](host, 0, expected, BN, 0, 0, BM)
    print("register A operand pipeline: matches reference")


def run_global[
    transpose_b: Bool, masked_a: Bool
](ctx: DeviceContext, M: Int) raises:
    var a_len = ceildiv(M, BM) * BM * K
    var b_len = K * N
    var a_host = ctx.enqueue_create_host_buffer[dtype](a_len)
    var b_host = ctx.enqueue_create_host_buffer[dtype](b_len)
    ctx.synchronize()
    for i in range(a_len):
        a_host[i] = _a_value(i)
    for i in range(b_len):
        b_host[i] = _b_value(i)
    var a_dev = ctx.enqueue_create_buffer[dtype](a_len)
    var b_dev = ctx.enqueue_create_buffer[dtype](b_len)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    var grid_y = ceildiv(M, BM)
    var grid_x = N // BN
    var out_len = grid_y * grid_x * num_threads * c_regs
    var out = ctx.enqueue_create_buffer[accum_type](out_len)

    comptime smem_bytes = num_stages * (BM * BK + BK * BN) * size_of[dtype]()
    ctx.enqueue_function[global_pipeline_kernel[transpose_b, masked_a]](
        a_dev,
        b_dev,
        out,
        Int32(M),
        grid_dim=(grid_x, grid_y),
        block_dim=num_threads,
        shared_mem_bytes=smem_bytes,
    )

    # A is (M, K) row-major; B is (N, K) when transposed, else (K, N).
    var expected = List[Float64](length=M * N, fill=0)
    for m in range(M):
        for n in range(N):
            var acc = Float64(0)
            for k in range(K):
                var b = b_host[n * K + k] if transpose_b else b_host[k * N + n]
                acc += Float64(a_host[m * K + k]) * Float64(b)
            expected[m * N + n] = acc

    with out.map_to_host() as host:
        for by in range(grid_y):
            for bx in range(grid_x):
                var block = by * grid_x + bx
                _check_block[WN, num_threads](
                    host,
                    block * num_threads * c_regs,
                    expected,
                    N,
                    by * BM,
                    bx * BN,
                    min(BM, M - by * BM),
                )
    print(
        "global pipeline transpose_b=",
        transpose_b,
        " masked_a=",
        masked_a,
        " M=",
        M,
        ": matches reference",
    )


def main() raises:
    # The host-side fragment mapping is the m16n8 accumulator layout.
    comptime assert mma_shape[0] == 16 and mma_shape[1] == 8 and frag[2] == 4
    with DeviceContext() as ctx:
        run_global[transpose_b=True, masked_a=False](ctx, 128)
        run_global[transpose_b=False, masked_a=False](ctx, 128)
        run_global[transpose_b=True, masked_a=True](ctx, 96)
        run_global[transpose_b=False, masked_a=True](ctx, 96)
        run_smem(ctx)
        run_local_a(ctx)
