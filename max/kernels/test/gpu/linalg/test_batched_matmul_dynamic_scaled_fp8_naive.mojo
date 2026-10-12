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

"""Kernel-level accuracy test for the batched blockwise FP8 naive fallback.

GLM-5.3 triggers `batched_matmul_dynamic_scaled_fp8` with scale
granularity (1, 64, 64) when the absorbed latent dimension is not a
multiple of 128. The dispatcher routes this to
`batched_matmul_dynamic_scaled_fp8_naive` on non-SM10x GPUs. This test
checks the dispatcher and its naive fallback against an independent
host-side reference at two GLM-like shapes.
"""

from std.math import isnan
from std.random import rand

from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from linalg.bmm import (
    batched_matmul_dynamic_scaled_fp8,
    batched_matmul_dynamic_scaled_fp8_naive,
)
from std.utils.index import Index, IndexList


# fp8 values accumulate in f32 then cast to bf16, so tolerate a few
# percent round-off (rtol/atol 1e-2) while catching real divergence.
def test_batched_matmul_dynamic_scaled_fp8_naive_granularity_64[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    batch_size: Int,
    M: Int,
    N: Int,
    K: Int,
](ctx: DeviceContext) raises:
    comptime m_scale = 1
    comptime n_scale = 64
    comptime k_scale = 64
    comptime BLOCK_SCALE_K = k_scale

    var a_size = batch_size * M * K
    var b_size = batch_size * N * K
    var c_size = batch_size * M * N
    var a_scales_size = batch_size * (K // BLOCK_SCALE_K) * M
    var b_scales_size = batch_size * (N // n_scale) * (K // BLOCK_SCALE_K)

    print(
        "== test_batched_matmul_dynamic_scaled_fp8_naive_granularity_64",
        "B=",
        batch_size,
        "M=",
        M,
        "N=",
        N,
        "K=",
        K,
        "scale=(",
        m_scale,
        ",",
        n_scale,
        ",",
        k_scale,
        ")",
    )

    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](a_size)
    var a_host = TileTensor(a_host_ptr, row_major[batch_size, M, K]())
    var b_host_ptr = ctx.enqueue_create_host_buffer[b_type](b_size)
    var b_host = TileTensor(b_host_ptr, row_major[batch_size, N, K]())
    var c_host_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_host = TileTensor(c_host_ptr, row_major[batch_size, M, N]())
    var c_ref_host_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)

    var a_scales_host_ptr = ctx.enqueue_create_host_buffer[.float32](
        a_scales_size
    )
    var a_scales_host = TileTensor(
        a_scales_host_ptr, row_major[batch_size, K // BLOCK_SCALE_K, M]()
    )
    var b_scales_host_ptr = ctx.enqueue_create_host_buffer[.float32](
        b_scales_size
    )
    var b_scales_host = TileTensor(
        b_scales_host_ptr,
        row_major[batch_size, N // n_scale, K // BLOCK_SCALE_K](),
    )

    rand(a_host._storage, a_host.num_elements())
    rand(b_host._storage, b_host.num_elements())
    _ = c_host.fill(0)
    rand(a_scales_host._storage, a_scales_host.num_elements())
    rand(b_scales_host._storage, b_scales_host.num_elements())

    var a_dev = ctx.enqueue_create_buffer[a_type](a_size)
    var b_dev = ctx.enqueue_create_buffer[b_type](b_size)
    var c_dev = ctx.enqueue_create_buffer[c_type](c_size)
    var a_scales_dev = ctx.enqueue_create_buffer[.float32](a_scales_size)
    var b_scales_dev = ctx.enqueue_create_buffer[.float32](b_scales_size)

    var a_tt = TileTensor(a_dev, row_major[batch_size, M, K]())
    var b_tt = TileTensor(b_dev, row_major[batch_size, N, K]())
    var c_tt = TileTensor(c_dev, row_major[batch_size, M, N]())
    var a_scales_tt = TileTensor(
        a_scales_dev, row_major[batch_size, K // BLOCK_SCALE_K, M]()
    )
    var b_scales_tt = TileTensor(
        b_scales_dev,
        row_major[batch_size, N // n_scale, K // BLOCK_SCALE_K](),
    )

    ctx.enqueue_copy(a_dev, a_host_ptr)
    ctx.enqueue_copy(b_dev, b_host_ptr)
    ctx.enqueue_copy(c_dev, c_host_ptr)
    ctx.enqueue_copy(a_scales_dev, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_dev, b_scales_host_ptr)

    # Call the naive per-batch fallback directly: the public AMD wrapper now
    # routes these shapes to the tiled kernel, which has its own test.
    _ = batched_matmul_dynamic_scaled_fp8_naive[
        scales_granularity_mnk=Index(1, n_scale, k_scale),
        transpose_b=True,
    ](c_tt, a_tt, b_tt, a_scales_tt, b_scales_tt, ctx)

    ctx.synchronize()

    # Independent reference: host-side blockwise scaled matmul in f32.
    for batch in range(batch_size):
        for m in range(M):
            for n in range(N):
                var accum = Scalar[DType.float32](0)
                for k in range(K):
                    var a_val = a_host_ptr[batch * M * K + m * K + k].cast[
                        DType.float32
                    ]()
                    var b_val = b_host_ptr[batch * N * K + n * K + k].cast[
                        DType.float32
                    ]()
                    var a_scale = a_scales_host_ptr[
                        batch * (K // k_scale) * M + (k // k_scale) * M + m
                    ]
                    var b_scale = b_scales_host_ptr[
                        batch * (N // n_scale) * (K // k_scale)
                        + (n // n_scale) * (K // k_scale)
                        + (k // k_scale)
                    ]
                    accum += (
                        a_val
                        * b_val
                        * a_scale.cast[DType.float32]()
                        * b_scale.cast[DType.float32]()
                    )
                c_ref_host_ptr[batch * M * N + m * N + n] = accum.cast[c_type]()

    ctx.enqueue_copy(c_host_ptr, c_dev)
    ctx.synchronize()

    comptime rtol = 1e-2
    comptime atol = 1e-2
    var ndiff = 0
    for i in range(c_size):
        var got = c_host_ptr[i]
        var ref_val = c_ref_host_ptr[i]
        var got_f32 = got.cast[DType.float32]()
        var ref_f32 = ref_val.cast[DType.float32]()
        var diff = got_f32 - ref_f32
        var abs_diff = diff if diff >= 0 else -diff
        var abs_ref = ref_f32 if ref_f32 >= 0 else -ref_f32
        if isnan(got_f32) or abs_diff > atol + rtol * abs_ref:
            ndiff += 1
            if ndiff <= 5:
                print("  diff @", i, "got=", got, "ref=", ref_val)
    if ndiff > 0:
        raise Error(
            "granularity-64 batched matmul diverged from references at "
            + String(ndiff)
            + " of "
            + String(c_size)
            + " positions"
        )
    print("  PASSED")


def main() raises:
    with DeviceContext() as ctx:
        test_batched_matmul_dynamic_scaled_fp8_naive_granularity_64[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            batch_size=8,
            M=528,
            N=192,
            K=512,
        ](ctx)
        test_batched_matmul_dynamic_scaled_fp8_naive_granularity_64[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            batch_size=8,
            M=1024,
            N=256,
            K=448,
        ](ctx)
