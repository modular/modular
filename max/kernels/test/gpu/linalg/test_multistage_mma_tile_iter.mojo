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
"""Checks that the `TileTensorIter` overload of `multistage_mma` reproduces the
`LayoutTensorIter` overload bit for bit.

Each block runs one BMxBN output tile through both overloads on identical
inputs and dumps the per-thread accumulator registers, which are compared
exactly on the host. Covered: global-memory pipelines (both transpose_b modes),
a masked A operand with a partial last row tile, a static-iteration shared
memory pipeline, and an fp32 register A operand.
"""

from std.sys import size_of
from std.math import ceildiv
from std.testing import assert_equal, assert_true
from std.utils.numerics import get_accum_type

from layout import Coord, TileTensor, row_major
from layout.coord import ComptimeInt
from layout.int_tuple import IntTuple
from layout.layout import Layout
from layout.layout_tensor import LayoutTensor, LayoutTensorIter
from layout.runtime_layout import RuntimeLayout
from layout.runtime_tuple import RuntimeTuple
from layout.tensor_core import get_fragment_size, get_mma_shape
from layout.tile_iterator import TileTensorIter
from layout.tile_layout import Layout as TileLayout
from layout.tile_tensor import stack_allocation

from linalg.matmul.gpu._multistage_gemm_gpu import multistage_mma
from max.gpu import WARP_SIZE, block_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, HostBuffer
from max.gpu.memory import AddressSpace, external_memory

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


def _dump(
    c: TileTensor[mut=True, accum_type, ...],
    out_ptr: UnsafePointer[Scalar[accum_type], MutAnyOrigin],
):
    var base = (
        (Int(block_idx.y) * (N // BN) + Int(block_idx.x)) * num_threads
        + Int(thread_idx.x)
    ) * c_regs
    comptime for i in range(num_m_mmas * num_n_mmas):
        comptime for j in range(frag[2]):
            out_ptr[base + i * frag[2] + j] = c[i, j][0]


def global_pipeline_kernel[
    use_tile_iter: Bool,
    transpose_b: Bool,
    masked_a: Bool,
](
    a_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    out_ptr: UnsafePointer[Scalar[accum_type], MutAnyOrigin],
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
    var b_smem = (a_smem + a_smem_size).bitcast[Scalar[dtype]]()

    var a_rows = min(BM, M - Int(block_idx.y) * BM)
    var a_block = a_ptr + Int(block_idx.y) * BM * K
    var b_block = b_ptr + (
        Int(block_idx.x) * BN * K if transpose_b else Int(block_idx.x) * BN
    )

    var c = stack_allocation[
        dtype=accum_type, address_space=AddressSpace.LOCAL
    ](row_major[num_m_mmas * num_n_mmas, frag[2]]()).fill(0)

    comptime if use_tile_iter:
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
    else:
        comptime a_smem_layout = Layout.row_major(BM, BK)
        comptime b_smem_layout = Layout.row_major(BD_0, BD_1)
        comptime ASmemIter = LayoutTensorIter[
            dtype,
            a_smem_layout,
            _,
            address_space=AddressSpace.SHARED,
            circular=True,
        ]
        var a_smem_iter = ASmemIter(
            a_smem, ASmemIter.linear_uint_type(a_smem_size)
        )
        comptime BSmemIter = LayoutTensorIter[
            dtype,
            b_smem_layout,
            _,
            address_space=AddressSpace.SHARED,
            circular=True,
        ]
        var b_smem_iter = BSmemIter(
            b_smem, BSmemIter.linear_uint_type(b_smem_size)
        )

        comptime a_layout = Layout(IntTuple(BM, K), IntTuple(K, 1))
        var a_tensor = LayoutTensor[
            dtype,
            a_layout,
            layout_int_type=DType.int32,
            linear_idx_type=DType.int32,
            masked=masked_a,
        ](
            a_block,
            RuntimeLayout[
                element_type=DType.int32, linear_idx_type=DType.int32
            ](
                RuntimeTuple[a_layout.shape, element_type=DType.int32](
                    a_rows, K
                ),
                RuntimeTuple[a_layout.stride, element_type=DType.int32](K, 1),
            ),
        )
        var a_iter = a_tensor.tiled_iterator[BM, BK, axis=1](0, 0)
        comptime b_layout = Layout(
            IntTuple(K, N) if not transpose_b else IntTuple(N, K),
            IntTuple(N, 1) if not transpose_b else IntTuple(K, 1),
        )
        var b_tensor = LayoutTensor[dtype, b_layout](b_block)
        var b_iter = b_tensor.tiled_iterator[
            BD_0, BD_1, axis=1 if transpose_b else 0
        ](0, 0)
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


def smem_pipeline_kernel[
    use_tile_iter: Bool
](out_ptr: UnsafePointer[Scalar[accum_type], MutAnyOrigin],):
    """QK-style: both operands already in shared memory, static trip count."""
    comptime depth = 128
    comptime num_tiles = depth // BK
    comptime a_smem_size = num_tiles * BM * BK
    comptime b_smem_size = num_tiles * BN * BK

    var a_smem = external_memory[
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
        alignment=16,
    ]()
    var b_smem = (a_smem + a_smem_size).bitcast[Scalar[dtype]]()
    for i in range(Int(thread_idx.x), a_smem_size, num_threads):
        a_smem[i] = Scalar[dtype](Float32((i * 7 + 3) % 29) / 8 - 1.5)
    for i in range(Int(thread_idx.x), b_smem_size, num_threads):
        b_smem[i] = Scalar[dtype](Float32((i * 5 + 1) % 31) / 8 - 1.75)
    barrier()

    var c = stack_allocation[
        dtype=accum_type, address_space=AddressSpace.LOCAL
    ](row_major[num_m_mmas * num_n_mmas, frag[2]]()).fill(0)

    comptime if use_tile_iter:
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
    else:
        comptime AIter = LayoutTensorIter[
            dtype,
            Layout.row_major(BM, BK),
            _,
            address_space=AddressSpace.SHARED,
        ]
        comptime BIter = LayoutTensorIter[
            dtype,
            Layout.row_major(BN, BK),
            _,
            address_space=AddressSpace.SHARED,
            circular=True,
        ]
        var a_iter = AIter(a_smem, AIter.linear_uint_type(a_smem_size))
        var b_iter = BIter(b_smem, BIter.linear_uint_type(b_smem_size))
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


def local_a_kernel[
    use_tile_iter: Bool
](out_ptr: UnsafePointer[Scalar[accum_type], MutAnyOrigin],):
    """PV-style: A operand is an fp32 register tile, B comes from shared."""
    comptime a_smem_size = num_stages * BM * BK
    comptime b_smem_size = (BN // BK) * BK * BN
    comptime p_rows = num_m_mmas * local_num_n_mmas
    comptime p_tile_rows = mma_shape[2] // mma_shape[1] * num_m_mmas

    var a_smem = external_memory[
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
        alignment=16,
    ]()
    var b_smem = (a_smem + a_smem_size).bitcast[Scalar[dtype]]()
    for i in range(Int(thread_idx.x), b_smem_size, local_threads):
        b_smem[i] = Scalar[dtype](Float32((i * 5 + 1) % 31) / 8 - 1.75)
    barrier()

    var p = stack_allocation[
        dtype=accum_type, address_space=AddressSpace.LOCAL
    ](row_major[p_rows, frag[2]]())
    comptime for i in range(p_rows):
        comptime for j in range(frag[2]):
            p[i, j] = (
                Scalar[accum_type](
                    (Int(thread_idx.x) * 13 + i * 5 + j * 3) % 37
                )
                / 7
                - 2.3
            )

    var c = stack_allocation[
        dtype=accum_type, address_space=AddressSpace.LOCAL
    ](row_major[p_rows, frag[2]]()).fill(0)

    comptime if use_tile_iter:
        comptime p_layout = row_major[p_tile_rows, frag[2]]()
        comptime b_layout = row_major[BK, BN]()
        comptime a_smem_layout = row_major[BM, BK]()
        var p_iter = TileTensorIter[
            accum_type,
            type_of(p_layout),
            _,
            address_space=AddressSpace.LOCAL,
        ](p.ptr, p_layout, bound=p_rows * frag[2])
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
    else:
        var p_iter = p.to_layout_tensor().tiled_iterator[p_tile_rows, frag[2]](
            0, 0
        )
        comptime BIter = LayoutTensorIter[
            dtype,
            Layout.row_major(BK, BN),
            _,
            address_space=AddressSpace.SHARED,
            circular=True,
        ]
        comptime AIter = LayoutTensorIter[
            dtype,
            Layout.row_major(BM, BK),
            _,
            address_space=AddressSpace.SHARED,
            circular=True,
        ]
        var b_iter = BIter(b_smem, BIter.linear_uint_type(b_smem_size))
        var a_smem_iter = AIter(a_smem, AIter.linear_uint_type(a_smem_size))
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

    var base = Int(thread_idx.x) * local_c_regs
    comptime for i in range(p_rows):
        comptime for j in range(frag[2]):
            out_ptr[base + i * frag[2] + j] = c[i, j][0]


def run_smem(ctx: DeviceContext) raises:
    comptime out_len = num_threads * c_regs
    var legacy = ctx.enqueue_create_buffer[accum_type](out_len)
    var native = ctx.enqueue_create_buffer[accum_type](out_len)
    comptime smem_bytes = (128 // BK) * (BM + BN) * BK * size_of[dtype]()
    ctx.enqueue_function[smem_pipeline_kernel[False]](
        legacy, grid_dim=1, block_dim=num_threads, shared_mem_bytes=smem_bytes
    )
    ctx.enqueue_function[smem_pipeline_kernel[True]](
        native, grid_dim=1, block_dim=num_threads, shared_mem_bytes=smem_bytes
    )
    with legacy.map_to_host() as lg, native.map_to_host() as nt:
        _assert_bitwise_equal(lg, nt, out_len)
    print("shared-memory static pipeline: bitwise match")


def run_local_a(ctx: DeviceContext) raises:
    comptime out_len = local_threads * local_c_regs
    var legacy = ctx.enqueue_create_buffer[accum_type](out_len)
    var native = ctx.enqueue_create_buffer[accum_type](out_len)
    comptime smem_bytes = (
        num_stages * BM * BK + (BN // BK) * BK * BN
    ) * size_of[dtype]()
    ctx.enqueue_function[local_a_kernel[False]](
        legacy,
        grid_dim=1,
        block_dim=local_threads,
        shared_mem_bytes=smem_bytes,
    )
    ctx.enqueue_function[local_a_kernel[True]](
        native,
        grid_dim=1,
        block_dim=local_threads,
        shared_mem_bytes=smem_bytes,
    )
    with legacy.map_to_host() as lg, native.map_to_host() as nt:
        _assert_bitwise_equal(lg, nt, out_len)
    print("register A operand pipeline: bitwise match")


def _assert_bitwise_equal(
    lg: HostBuffer[accum_type], nt: HostBuffer[accum_type], length: Int
) raises:
    var mismatches = 0
    var nonzero = 0
    for i in range(length):
        if lg[i] != nt[i]:
            mismatches += 1
        if lg[i] != 0:
            nonzero += 1
    assert_equal(mismatches, 0)
    # An all-zero result would match trivially.
    assert_true(nonzero > length // 2)


def run_global[
    transpose_b: Bool, masked_a: Bool
](ctx: DeviceContext, M: Int) raises:
    var a_len = ceildiv(M, BM) * BM * K
    var b_len = K * N
    var a_host = ctx.enqueue_create_host_buffer[dtype](a_len)
    var b_host = ctx.enqueue_create_host_buffer[dtype](b_len)
    ctx.synchronize()
    for i in range(a_len):
        a_host[i] = Scalar[dtype](Float32((i * 7 + 3) % 29) / 8 - 1.5)
    for i in range(b_len):
        b_host[i] = Scalar[dtype](Float32((i * 5 + 1) % 31) / 8 - 1.75)
    var a_dev = ctx.enqueue_create_buffer[dtype](a_len)
    var b_dev = ctx.enqueue_create_buffer[dtype](b_len)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    var grid_y = ceildiv(M, BM)
    var grid_x = N // BN
    var out_len = grid_y * grid_x * num_threads * c_regs
    var legacy = ctx.enqueue_create_buffer[accum_type](out_len)
    var native = ctx.enqueue_create_buffer[accum_type](out_len)

    comptime smem_bytes = num_stages * (BM * BK + BK * BN) * size_of[dtype]()
    comptime legacy_kernel = global_pipeline_kernel[
        False, transpose_b, masked_a
    ]
    comptime native_kernel = global_pipeline_kernel[True, transpose_b, masked_a]
    ctx.enqueue_function[legacy_kernel](
        a_dev,
        b_dev,
        legacy,
        Int32(M),
        grid_dim=(grid_x, grid_y),
        block_dim=num_threads,
        shared_mem_bytes=smem_bytes,
    )
    ctx.enqueue_function[native_kernel](
        a_dev,
        b_dev,
        native,
        Int32(M),
        grid_dim=(grid_x, grid_y),
        block_dim=num_threads,
        shared_mem_bytes=smem_bytes,
    )

    with legacy.map_to_host() as lg, native.map_to_host() as nt:
        _assert_bitwise_equal(lg, nt, out_len)
    print(
        "global pipeline transpose_b=",
        transpose_b,
        " masked_a=",
        masked_a,
        " M=",
        M,
        ": bitwise match",
    )


def main() raises:
    with DeviceContext() as ctx:
        run_global[transpose_b=True, masked_a=False](ctx, 128)
        run_global[transpose_b=False, masked_a=False](ctx, 128)
        run_global[transpose_b=True, masked_a=True](ctx, 96)
        run_global[transpose_b=False, masked_a=True](ctx, 96)
        run_smem(ctx)
        run_local_a(ctx)
