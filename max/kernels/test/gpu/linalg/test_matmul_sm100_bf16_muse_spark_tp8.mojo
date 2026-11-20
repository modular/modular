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
# Muse Spark's per-GPU bf16 matmuls at TP8: the attention projection (Q, K, V
# and the output gate as one matmul, N=2304) and the LM head (vocab / 8,
# N=25256), both [M, 8192] x [N, 8192]^T. Every M goes through the production
# dispatch (tuning rows, heuristic, GEMV at M=1), so this checks each against
# cuBLAS at prime and ragged M, the edges of the decode tuning bands, one row,
# an empty batch and the 16384-token prefill chunk. The output starts as a
# sentinel, so a tile the kernel skips fails too.

import linalg.matmul.vendor.blas as vendor_blas
from layout import Idx, TileTensor, row_major
from linalg.matmul.gpu import _matmul_gpu
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.random import random_float64, seed
from std.testing import assert_false, assert_true

comptime dt = DType.bfloat16
comptime K = 8192
comptime SENTINEL = -7.0
comptime QKV_GATE_N = 2304
comptime QKV_GATE_MS = [
    0,
    1,
    2,
    3,
    7,
    8,
    9,
    13,
    16,
    17,
    31,
    33,
    62,
    64,
    65,
    97,
    127,
    131,
    257,
    1021,
    4099,
    16384,
]
comptime LM_HEAD_N = 25256
comptime LM_HEAD_MS = [1, 2, 7, 8, 9, 16, 17, 31, 33, 47, 48, 49, 62, 64, 65]


def _close(
    got: HostBuffer[dt], expected: HostBuffer[dt], n: Int, max_ref: Float64
) -> Bool:
    # bf16 output of an fp32-accumulated K=8192 dot: a few output ulps.
    for i in range(n):
        var g = Float64(got[i].cast[.float32]())
        var e = Float64(expected[i].cast[.float32]())
        if abs(g - e) > 1e-2 * max_ref:
            return False
    return True


def weights[N: Int](ctx: DeviceContext) raises -> DeviceBuffer[dt]:
    var b_host = ctx.enqueue_create_host_buffer[dt](N * K)
    ctx.synchronize()
    seed(N)
    for i in range(N * K):
        b_host[i] = Scalar[dt](random_float64(-1, 1) * 0.02)
    var b = ctx.enqueue_create_buffer[dt](N * K)
    ctx.enqueue_copy(b, b_host)
    ctx.synchronize()
    return b


def check[N: Int](ctx: DeviceContext, b: DeviceBuffer[dt], m: Int) raises:
    var rows = max(m, 1)
    var a_host = ctx.enqueue_create_host_buffer[dt](rows * K)
    ctx.synchronize()
    seed(m)
    for i in range(rows * K):
        a_host[i] = Scalar[dt](random_float64(-1, 1))

    var a = ctx.enqueue_create_buffer[dt](rows * K)
    var c = ctx.enqueue_create_buffer[dt](rows * N)
    var c_ref = ctx.enqueue_create_buffer[dt](rows * N)
    ctx.enqueue_copy(a, a_host)
    c.enqueue_fill(Scalar[dt](SENTINEL))
    c_ref.enqueue_fill(Scalar[dt](SENTINEL))

    var a_t = TileTensor(a, row_major(m, Idx[K]))
    var b_t = TileTensor(b, row_major(Idx[N], Idx[K]))
    var c_t = TileTensor(c, row_major(m, Idx[N]))

    if m == 0:
        # An empty batch is refused (the launch fails on an empty grid) or
        # does nothing; either way the output must stay untouched.
        try:
            _matmul_gpu[use_tensor_core=True, transpose_b=True](
                c_t, a_t.as_imm(), b_t.as_imm(), ctx
            )
            ctx.synchronize()
        except e:
            print("M=0 refused:", e)
        var c_host = ctx.enqueue_create_host_buffer[dt](N)
        ctx.enqueue_copy(c_host, c)
        ctx.synchronize()
        for i in range(N):
            assert_true(
                c_host[i] == Scalar[dt](SENTINEL), "M=0 wrote the output"
            )
        print("PASS N=", N, " M=0")
        return

    _matmul_gpu[use_tensor_core=True, transpose_b=True](
        c_t, a_t.as_imm(), b_t.as_imm(), ctx
    )
    vendor_blas.matmul(
        ctx,
        TileTensor(c_ref, row_major(m, Idx[N])),
        a_t,
        b_t,
        c_row_major=True,
        transpose_b=True,
    )
    var c_host = ctx.enqueue_create_host_buffer[dt](m * N)
    var ref_host = ctx.enqueue_create_host_buffer[dt](m * N)
    ctx.enqueue_copy(c_host, c)
    ctx.enqueue_copy(ref_host, c_ref)
    ctx.synchronize()

    var max_ref = 0.0
    for i in range(m * N):
        max_ref = max(max_ref, abs(Float64(ref_host[i].cast[.float32]())))
    assert_true(max_ref > 0, "the reference is all zero")
    assert_true(
        _close(c_host, ref_host, m * N, max_ref),
        String("N=", N, " M=", m, ": MAX differs from cuBLAS"),
    )

    # Planted faults: the comparison must reject a skipped corner of the last
    # tile and one element off by 5% of the output range.
    var last = m * N - 1
    var saved = c_host[last]
    c_host[last] = Scalar[dt](SENTINEL)
    assert_false(_close(c_host, ref_host, m * N, max_ref), "missed a skip")
    c_host[last] = saved + Scalar[dt](0.05 * max_ref)
    assert_false(_close(c_host, ref_host, m * N, max_ref), "missed an error")
    print("PASS N=", N, " M=", m, " max_ref=", max_ref)


def main() raises:
    with DeviceContext() as ctx:
        var qkv_gate = weights[QKV_GATE_N](ctx)
        comptime for m in QKV_GATE_MS:
            check[QKV_GATE_N](ctx, qkv_gate, m)
        var lm_head = weights[LM_HEAD_N](ctx)
        comptime for m in LM_HEAD_MS:
            check[LM_HEAD_N](ctx, lm_head, m)
