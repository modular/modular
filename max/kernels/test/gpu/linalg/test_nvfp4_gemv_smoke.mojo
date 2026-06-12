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
"""Correctness + quick throughput check for the fused NVFP4 dequant-GEMV."""

from std.math import ceildiv
from std.random import random_ui64, seed
from std.testing import assert_almost_equal
from std.time import perf_counter_ns

from std.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from linalg.nvfp4_gemv import nvfp4_gemv
from linalg.fp4_utils import E2M1_TO_FLOAT32


def _decode_nibble(nib: UInt8) -> Float32:
    return E2M1_TO_FLOAT32[Int(nib)]


def _check[M: Int, N: Int, K: Int](ctx: DeviceContext) raises:
    comptime packed_cols = K // 2
    comptime scale_cols = K // 16

    var a_host = ctx.enqueue_create_host_buffer[DType.bfloat16](M * K)
    var w_host = ctx.enqueue_create_host_buffer[DType.uint8](N * packed_cols)
    var s_host = ctx.enqueue_create_host_buffer[DType.float32](N * scale_cols)
    var c_host = ctx.enqueue_create_host_buffer[DType.bfloat16](M * N)
    ctx.synchronize()

    seed(42)
    for i in range(M * K):
        a_host[i] = (Float32(Int(random_ui64(0, 200)) - 100) / 50.0).cast[
            DType.bfloat16
        ]()
    for i in range(N * packed_cols):
        w_host[i] = UInt8(random_ui64(0, 255))
    for i in range(N * scale_cols):
        s_host[i] = Float32(Int(random_ui64(1, 100))) / 1000.0

    var a_dev = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var w_dev = ctx.enqueue_create_buffer[DType.uint8](N * packed_cols)
    var s_dev = ctx.enqueue_create_buffer[DType.float32](N * scale_cols)
    var c_dev = ctx.enqueue_create_buffer[DType.bfloat16](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(w_dev, w_host)
    ctx.enqueue_copy(s_dev, s_host)
    ctx.synchronize()

    var c_tt = TileTensor(c_dev, row_major[M, N]())
    var a_tt = TileTensor(a_dev, row_major[M, K]())
    var w_tt = TileTensor(w_dev, row_major[N, packed_cols]())
    var s_tt = TileTensor(s_dev, row_major[N, scale_cols]())

    nvfp4_gemv(ctx, c_tt, a_tt, w_tt, s_tt, M, N, K)
    ctx.enqueue_copy(c_host, c_dev)
    ctx.synchronize()

    # CPU reference
    var max_rel_err: Float32 = 0.0
    for mm in range(M):
        for nn in range(N):
            var expected: Float32 = 0.0
            for kk in range(K):
                var byte = w_host[nn * packed_cols + kk // 2]
                var nib = (byte >> UInt8((kk % 2) * 4)) & 0x0F
                var wval = _decode_nibble(nib)
                var scale = s_host[nn * scale_cols + kk // 16]
                expected += (
                    wval * scale * a_host[mm * K + kk].cast[DType.float32]()
                )
            var got = c_host[mm * N + nn].cast[DType.float32]()
            # bf16 output rounding: tolerate ~1% relative.
            assert_almost_equal(got, expected, atol=0.05, rtol=0.02)
    print("correctness OK for", M, "x", N, "x", K)


def _bench[M: Int, N: Int, K: Int](ctx: DeviceContext) raises:
    comptime packed_cols = K // 2
    comptime scale_cols = K // 16
    comptime ITERS = 50

    var a_dev = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var w_dev = ctx.enqueue_create_buffer[DType.uint8](N * packed_cols)
    var s_dev = ctx.enqueue_create_buffer[DType.float32](N * scale_cols)
    var c_dev = ctx.enqueue_create_buffer[DType.bfloat16](M * N)
    ctx.synchronize()

    var c_tt = TileTensor(c_dev, row_major[M, N]())
    var a_tt = TileTensor(a_dev, row_major[M, K]())
    var w_tt = TileTensor(w_dev, row_major[N, packed_cols]())
    var s_tt = TileTensor(s_dev, row_major[N, scale_cols]())

    nvfp4_gemv(ctx, c_tt, a_tt, w_tt, s_tt, M, N, K)
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        nvfp4_gemv(ctx, c_tt, a_tt, w_tt, s_tt, M, N, K)
    ctx.synchronize()
    var t1 = perf_counter_ns()

    var ms = Float64(t1 - t0) / 1e6 / ITERS
    var bytes_read = Float64(N * packed_cols + N * scale_cols * 4 + M * K * 2)
    var gbps = bytes_read / (ms * 1e6)
    print(
        "gemv M=",
        M,
        " [",
        N,
        "x",
        K,
        "]: ",
        ms,
        " ms -> ",
        gbps,
        " GB/s (packed)",
    )


def main() raises:
    var ctx = DeviceContext()
    _check[1, 64, 128](ctx)
    _check[3, 96, 256](ctx)
    _check[5, 33, 1024](ctx)
    # >1 K-chunk per lane (K/32 > WARP_SIZE) and the production K.
    _check[2, 64, 2048](ctx)
    _check[1, 48, 3840](ctx)
    _bench[1, 15360, 3840](ctx)
    _bench[4, 15360, 3840](ctx)
    _bench[16, 15360, 3840](ctx)
