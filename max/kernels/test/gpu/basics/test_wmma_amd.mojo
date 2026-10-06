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

from std._gpu import lane_id
from std.math import align_up, ceildiv, isclose
from std.math.uutils import umod, udivmod
from std.random import rand
from std.utils import IndexList

from max.gpu import WARP_SIZE, block_idx
from max.gpu.host import DeviceContext
from max.gpu.host.info import MI455X
from max.gpu.compute.mma import mma
from std.testing import assert_equal


def matmul_naive[
    a_dtype: DType,
    b_dtype: DType,
    out_dtype: DType,
    //,
    accum_dtype: DType,
    mma_k: Int,
](
    a: Pointer[Scalar[a_dtype], _],
    b: Pointer[Scalar[b_dtype], _],
    c: MutPointer[Scalar[out_dtype], _],
    m: Int,
    n: Int,
    k: Int,
):
    for i in range(m):
        for j in range(n):
            var outer = Scalar[accum_dtype](0)
            for lo in range(0, k, mma_k):
                var inner = outer.cast[.float32]()
                for li in range(lo, lo + mma_k):
                    var av = a[unsafe_offset=k * i + li].cast[.float32]()
                    var bv = b[unsafe_offset=n * li + j].cast[.float32]()
                    inner += av * bv
                outer = inner.cast[accum_dtype]()
            c[unsafe_offset=n * i + j] = outer.cast[out_dtype]()


@inline(.always)
def load_matrix_a[
    dtype: DType, //, mma_m: Int, mma_k: Int
](
    a_ptr: Pointer[mut=False, Scalar[dtype], _],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
    out fragment: SIMD[dtype, (mma_m * mma_k) // WARP_SIZE],
):
    var thread_y, thread_x = udivmod(lane_id(), 16)
    fragment = SIMD[dtype, fragment.length]()

    comptime for i in range(fragment.length):
        var a_idx = (
            ldm * (tile_row + thread_x)
            + tile_col
            + i
            + fragment.length * thread_y
        )
        fragment[i] = a_ptr[unsafe_offset=a_idx]


@inline(.always)
def load_matrix_b[
    dtype: DType, //, mma_n: Int, mma_k: Int
](
    b_ptr: Pointer[mut=False, Scalar[dtype], _],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
    out fragment: SIMD[dtype, (mma_n * mma_k) // WARP_SIZE],
):
    var thread_y, thread_x = udivmod(lane_id(), 16)
    fragment = SIMD[dtype, fragment.length]()

    comptime for i in range(fragment.length):
        var b_idx = (
            ldm * (tile_row + fragment.length * thread_y + i)
            + tile_col
            + thread_x
        )
        fragment[i] = b_ptr[unsafe_offset=b_idx]


@inline(.always)
def store_matrix_d[
    dtype: DType
](
    d_ptr: Pointer[mut=True, Scalar[dtype], _],
    d: SIMD[dtype, _],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
):
    var thread_y, thread_x = udivmod(lane_id(), 16)

    comptime for i in range(d.length):
        var d_idx = (
            ldm * (tile_row + d.length * thread_y + i) + tile_col + thread_x
        )
        d_ptr[unsafe_offset=d_idx] = d[i]


def mma_kernel[
    in_dtype: DType,
    accum_dtype: DType,
    out_dtype: DType,
    mma_m: Int,
    mma_n: Int,
    mma_k: Int,
](
    a_ptr: ImmPointer[Scalar[in_dtype], ImmutAnyOrigin],
    b_ptr: ImmPointer[Scalar[in_dtype], ImmutAnyOrigin],
    c_ptr: MutPointer[Scalar[out_dtype], MutAnyOrigin],
    m_dev: Int32,
    n_dev: Int32,
    k_dev: Int32,
):
    var m = Int(m_dev)
    var n = Int(n_dev)
    var k = Int(k_dev)

    var c_reg = SIMD[accum_dtype, (mma_m * mma_n) // WARP_SIZE](0)
    var d_reg = SIMD[out_dtype, (mma_m * mma_n) // WARP_SIZE](0)
    var tile_loops = k // mma_k

    for l in range(tile_loops):
        var a_tile_row = block_idx.x * mma_m
        var a_tile_col = l * mma_k
        var b_tile_row = l * mma_k
        var b_tile_col = block_idx.y * mma_n

        var a_reg = load_matrix_a[mma_m, mma_k](
            a_ptr, a_tile_row, a_tile_col, k
        )
        var b_reg = load_matrix_b[mma_n, mma_k](
            b_ptr, b_tile_row, b_tile_col, n
        )

        if l == tile_loops - 1:
            mma(d_reg, a_reg, b_reg, c_reg)
        else:
            mma(c_reg, a_reg, b_reg, c_reg)

    var c_tile_row = block_idx.x * mma_m
    var c_tile_col = block_idx.y * mma_n
    store_matrix_d(c_ptr, d_reg, c_tile_row, c_tile_col, n)


def run_mma[
    in_dtype: DType,
    accum_dtype: DType,
    out_dtype: DType,
    mma_m: Int,
    mma_n: Int,
    mma_k: Int,
](M: Int, N: Int, K: Int, ctx: DeviceContext) raises:
    print(
        t"== run_matmul {in_dtype}.{accum_dtype}.{out_dtype} matrix core kernel"
        t" shape={M},{N},{K}"
    )

    var a_host = ctx.enqueue_create_host_buffer[in_dtype](M * K)
    var b_host = ctx.enqueue_create_host_buffer[in_dtype](K * N)
    var c_host = ctx.enqueue_create_host_buffer[out_dtype](M * N)
    var c_host_ref = ctx.enqueue_create_host_buffer[out_dtype](M * N)

    rand(a_host.unsafe_ptr(), M * K)
    rand(b_host.unsafe_ptr(), K * N)

    for i in range(M * N):
        c_host[i] = 0
        c_host_ref[i] = 0

    var a_device = ctx.enqueue_create_buffer[in_dtype](M * K)
    var b_device = ctx.enqueue_create_buffer[in_dtype](K * N)
    var c_device = ctx.enqueue_create_buffer[out_dtype](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)
    ctx.enqueue_copy(c_device, c_host)

    comptime kernel = mma_kernel[
        in_dtype, accum_dtype, out_dtype, mma_m, mma_n, mma_k
    ]

    ctx.enqueue_function[kernel](
        a_device,
        b_device,
        c_device,
        Int32(M),
        Int32(N),
        Int32(K),
        grid_dim=(ceildiv(M, mma_m), ceildiv(N, mma_n)),
        block_dim=WARP_SIZE,
    )

    ctx.enqueue_copy(c_host, c_device)
    ctx.synchronize()

    _ = a_device
    _ = b_device
    _ = c_device

    matmul_naive[accum_dtype, mma_k](
        a_host.unsafe_ptr(),
        b_host.unsafe_ptr(),
        c_host_ref.unsafe_ptr(),
        M,
        N,
        K,
    )

    var errors = 0
    for i in range(M * N):
        if not isclose(c_host[i], c_host_ref[i], atol=1e-2, rtol=1e-2):
            errors += 1

    if errors == 0:
        print("Success 🎉: Results match.")
    else:
        print("Failed ❌: results mismatch.")

    assert_equal(errors, 0)


def run_mma[
    in_dtype: DType,
    mma_m: Int,
    mma_n: Int,
    mma_k: Int,
    *,
    out_dtype: DType = .float32,
    accum_dtype: DType = out_dtype,
](ctx: DeviceContext) raises:
    comptime shape_list: List[IndexList[3]] = [
        (16, 16, 16),
        (384, 512, 768),
    ]

    comptime for shape in shape_list:
        run_mma[in_dtype, accum_dtype, out_dtype, mma_m, mma_n, mma_k](
            shape[0], shape[1], align_up(shape[2], mma_k), ctx
        )


def main() raises:
    with DeviceContext() as ctx:
        run_mma[.float32, 16, 16, 4](ctx)

        comptime if ctx.default_device_info == MI455X:
            run_mma[.float16, 16, 16, 32](ctx)
            run_mma[.float16, 16, 16, 32, out_dtype=.float16](ctx)

            run_mma[.bfloat16, 16, 16, 32](ctx)
            run_mma[.bfloat16, 16, 16, 32, out_dtype=.bfloat16](ctx)
            run_mma[
                .bfloat16, 16, 16, 32, accum_dtype=.float32, out_dtype=.bfloat16
            ](ctx)

            comptime for mma_k in [64, 128]:
                comptime for out_dtype in [DType.float16, DType.float32]:
                    run_mma[.float8_e4m3fn, 16, 16, mma_k, out_dtype=out_dtype](
                        ctx
                    )
                    run_mma[.float8_e5m2, 16, 16, mma_k, out_dtype=out_dtype](
                        ctx
                    )

        else:
            run_mma[.float16, 16, 16, 16](ctx)
            run_mma[.bfloat16, 16, 16, 16](ctx)
