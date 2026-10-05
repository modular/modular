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
from std.math import ceildiv
from std.math.uutils import umod, udivmod
from std.random import random_si64
from std.utils import IndexList

from max.gpu import WARP_SIZE, block_idx
from max.gpu.host import DeviceContext
from max.gpu.compute.mma import mma
from std.testing import assert_equal


def matmul_naive[
    a_dtype: DType, b_dtype: DType, c_dtype: DType
](
    a: Pointer[Scalar[a_dtype], _],
    b: Pointer[Scalar[b_dtype], _],
    c: MutPointer[Scalar[c_dtype], _],
    m: Int,
    n: Int,
    k: Int,
):
    for i in range(m):
        for l in range(k):
            for j in range(n):
                var av = a[unsafe_offset=k * i + l].cast[c_dtype]()
                var bv = b[unsafe_offset=n * l + j].cast[c_dtype]()
                c[unsafe_offset=n * i + j] += av * bv


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
    d: SIMD[dtype, 4],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
):
    var thread_y, thread_x = udivmod(lane_id(), 16)

    comptime for i in range(4):
        var d_idx = ldm * (tile_row + 4 * thread_y + i) + tile_col + thread_x
        d_ptr[unsafe_offset=d_idx] = d[i]


def mma_kernel[
    in_dtype: DType, mma_m: Int, mma_n: Int, mma_k: Int
](
    a_ptr: ImmPointer[Scalar[in_dtype], ImmutAnyOrigin],
    b_ptr: ImmPointer[Scalar[in_dtype], ImmutAnyOrigin],
    c_ptr: MutPointer[Float32, MutAnyOrigin],
    m_dev: Int32,
    n_dev: Int32,
    k_dev: Int32,
):
    var m = Int(m_dev)
    var n = Int(n_dev)
    var k = Int(k_dev)

    var d_reg: SIMD[.float32, 4] = 0
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
        mma(d_reg, a_reg, b_reg, d_reg)

    var c_tile_row = block_idx.x * mma_m
    var c_tile_col = block_idx.y * mma_n
    store_matrix_d(c_ptr, d_reg, c_tile_row, c_tile_col, n)


def run_mma[
    in_dtype: DType, mma_m: Int, mma_n: Int, mma_k: Int
](
    M: Int,
    N: Int,
    K: Int,
    rand_min: Int64,
    rand_max: Int64,
    ctx: DeviceContext,
) raises:
    print(
        t"== run_matmul {in_dtype}.float32 matrix core kernel shape={M},{N},{K}"
    )

    var a_host = ctx.enqueue_create_host_buffer[in_dtype](M * K)
    var b_host = ctx.enqueue_create_host_buffer[in_dtype](K * N)
    var c_host = ctx.enqueue_create_host_buffer[.float32](M * N)
    var c_host_ref = ctx.enqueue_create_host_buffer[.float32](M * N)

    for i in range(M * K):
        a_host[i] = random_si64(rand_min, rand_max).cast[in_dtype]()

    for i in range(K * N):
        b_host[i] = random_si64(rand_min, rand_max).cast[in_dtype]()

    for i in range(M * N):
        c_host[i] = 0
        c_host_ref[i] = 0

    var a_device = ctx.enqueue_create_buffer[in_dtype](M * K)
    var b_device = ctx.enqueue_create_buffer[in_dtype](K * N)
    var c_device = ctx.enqueue_create_buffer[.float32](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)
    ctx.enqueue_copy(c_device, c_host)

    comptime kernel = mma_kernel[in_dtype, mma_m, mma_n, mma_k]

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

    matmul_naive(
        a_host.unsafe_ptr(),
        b_host.unsafe_ptr(),
        c_host_ref.unsafe_ptr(),
        M,
        N,
        K,
    )

    var errors = 0
    for i in range(M * N):
        if c_host[i] != c_host_ref[i]:
            errors += 1

    _ = a_device
    _ = b_device
    _ = c_device

    if errors == 0:
        print("Success 🎉: Results match.")
    else:
        print("Failed ❌: results mismatch.")

    assert_equal(errors, 0)


def run_mma[
    in_dtype: DType, mma_m: Int, mma_n: Int, mma_k: Int
](ctx: DeviceContext) raises:
    comptime shape_list: List[IndexList[3]] = [
        (16, 16, 16),
        (384, 512, 768),
        (1280, 768, 2048),
    ]

    comptime for shape in shape_list:
        run_mma[in_dtype, mma_m, mma_n, mma_k](
            shape[0], shape[1], shape[2], -100, 100, ctx
        )


def main() raises:
    with DeviceContext() as ctx:
        run_mma[.float32, 16, 16, 4](ctx)
        run_mma[.float16, 16, 16, 16](ctx)
        run_mma[.bfloat16, 16, 16, 16](ctx)
