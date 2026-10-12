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
"""Correctness test for the tiled blockwise-scaled FP8 dense matmul (gfx950).

Compares `blockwise_scaled_fp8_matmul_amd` against the trusted
`naive_blockwise_scaled_fp8_matmul` reference (and, on the small shape,
an independent host fp32 loop) at the GLM dense shapes.

The scales are drawn LOG-UNIFORM in `[2^-4, 2^4]` per block. A constant
scale hides a scale-indexing bug (the single most likely failure mode),
so non-constant scales are load-bearing here, not cosmetic.
"""

from std.math import ceildiv, exp2

from max.gpu.host import DeviceContext
from internal_utils import assert_almost_equal
from std.random import rand
from std.testing import assert_true
from layout import CoordLike, Coord, Idx, TileTensor, row_major
from linalg.fp8_quantization import (
    naive_blockwise_scaled_fp8_matmul,
    matmul_dynamic_scaled_fp8,
    _amd_blockwise_tiled_ok,
)
from linalg.matmul.gpu.amd.blockwise_scaled_fp8_matmul_amd import (
    blockwise_scaled_fp8_matmul_amd,
)

from std.utils.index import Index, IndexList


def test_blockwise_fp8_matmul_amd[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    c_type: DType,
    check_cpu: Bool = False,
    via_dispatch: Bool = False,
](ctx: DeviceContext, m: MType, n: NType, k: KType) raises:
    comptime input_type = DType.float8_e4m3fn
    comptime transpose_b = True
    comptime BLOCK_SCALE_M = 1
    comptime BLOCK_SCALE_N = 128
    comptime BLOCK_SCALE_K = 128

    var M = Int(m.value())
    var N = Int(n.value())
    var K = Int(k.value())

    print(
        "== test_blockwise_fp8_matmul_amd c_type=",
        c_type,
        " MxNxK=",
        M,
        "x",
        N,
        "x",
        K,
        " via_dispatch=",
        via_dispatch,
        sep="",
    )

    var a_shape = Coord(m, k)
    var b_shape = Coord(n, k)  # transpose_b: B is [N, K]
    var c_shape = Coord(m, n)
    var a_scale_shape = Coord(
        ceildiv(K, BLOCK_SCALE_K), ceildiv(M, BLOCK_SCALE_M)
    )
    var b_scale_shape = Coord(
        ceildiv(N, BLOCK_SCALE_N), ceildiv(K, BLOCK_SCALE_K)
    )

    var a_size = M * K
    var b_size = N * K
    var c_size = M * N
    var a_scale_size = ceildiv(K, BLOCK_SCALE_K) * ceildiv(M, BLOCK_SCALE_M)
    var b_scale_size = ceildiv(N, BLOCK_SCALE_N) * ceildiv(K, BLOCK_SCALE_K)

    var a_host_ptr = ctx.enqueue_create_host_buffer[input_type](a_size)
    var b_host_ptr = ctx.enqueue_create_host_buffer[input_type](b_size)
    var a_scale_host_ptr = ctx.enqueue_create_host_buffer[.float32](
        a_scale_size
    )
    var b_scale_host_ptr = ctx.enqueue_create_host_buffer[.float32](
        b_scale_size
    )

    rand(a_host_ptr.unsafe_ptr(), a_size)
    rand(b_host_ptr.unsafe_ptr(), b_size)
    rand(a_scale_host_ptr.unsafe_ptr(), a_scale_size)
    rand(b_scale_host_ptr.unsafe_ptr(), b_scale_size)
    # Rewrite the [0, 1) samples as LOG-UNIFORM scales in [2^-4, 2^4).
    for i in range(a_scale_size):
        a_scale_host_ptr[i] = exp2(Float32(8.0) * a_scale_host_ptr[i] - 4.0)
    for i in range(b_scale_size):
        b_scale_host_ptr[i] = exp2(Float32(8.0) * b_scale_host_ptr[i] - 4.0)

    # Outputs: my tiled kernel (c_type) + naive reference (float32).
    var c_my_host = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_my_f32 = ctx.enqueue_create_host_buffer[.float32](c_size)
    var c_ref_host = ctx.enqueue_create_host_buffer[.float32](c_size)

    var a_device = ctx.enqueue_create_buffer[input_type](a_size)
    var b_device = ctx.enqueue_create_buffer[input_type](b_size)
    var c_my_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_ref_device = ctx.enqueue_create_buffer[.float32](c_size)
    var a_scale_device = ctx.enqueue_create_buffer[.float32](a_scale_size)
    var b_scale_device = ctx.enqueue_create_buffer[.float32](b_scale_size)

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(a_scale_device, a_scale_host_ptr)
    ctx.enqueue_copy(b_scale_device, b_scale_host_ptr)

    var a_dev = TileTensor(a_device, row_major(a_shape))
    var b_dev = TileTensor(b_device, row_major(b_shape))
    var c_my_dev = TileTensor(c_my_device, row_major(c_shape))
    var c_ref_dev = TileTensor(c_ref_device, row_major(c_shape))
    var a_scale_dev = TileTensor(a_scale_device, row_major(a_scale_shape))
    var b_scale_dev = TileTensor(b_scale_device, row_major(b_scale_shape))

    comptime if via_dispatch:
        # Exercises the wired dispatch (block/block granularity on AMD
        # routes to the tiled kernel). `c_my_dev == c_ref_dev` below cannot
        # by itself prove the tiled route was taken -- a gate regressed to
        # False falls back to the (also-correct) naive kernel and this test
        # would still pass. Assert the gate directly (PR02 review N3).
        assert_true(
            _amd_blockwise_tiled_ok[
                c_type=c_type,
                a_type=input_type,
                b_type=input_type,
                a_scales_type=.float32,
                b_scales_type=.float32,
                transpose_b=transpose_b,
                scales_granularity_mnk=Index(
                    BLOCK_SCALE_M, BLOCK_SCALE_N, BLOCK_SCALE_K
                ),
                elementwise_lambda_fn=None,
                device_info=ctx.default_device_info,
                a_static_k=type_of(a_dev).static_shape[1],
                c_static_n=type_of(c_my_dev).static_shape[1],
            ](),
            (
                "AMD tiled blockwise-FP8 gate (_amd_blockwise_tiled_ok) is"
                " False for a production dtype/shape -- dispatch would"
                " silently fall back to the naive kernel"
            ),
        )
        matmul_dynamic_scaled_fp8[
            input_scale_granularity="block",
            weight_scale_granularity="block",
            m_scale_granularity=BLOCK_SCALE_M,
            n_scale_granularity=BLOCK_SCALE_N,
            k_scale_granularity=BLOCK_SCALE_K,
            transpose_b=transpose_b,
            target="gpu",
        ](c_my_dev, a_dev, b_dev, a_scale_dev, b_scale_dev, ctx)
    else:
        blockwise_scaled_fp8_matmul_amd[
            transpose_b=transpose_b,
            N_SCALE=BLOCK_SCALE_N,
            K_SCALE=BLOCK_SCALE_K,
        ](c_my_dev, a_dev, b_dev, a_scale_dev, b_scale_dev, ctx)

    naive_blockwise_scaled_fp8_matmul[
        BLOCK_DIM=16,
        transpose_b=transpose_b,
        scales_granularity_mnk=Index(
            BLOCK_SCALE_M, BLOCK_SCALE_N, BLOCK_SCALE_K
        ),
    ](
        c_ref_dev,
        a_dev,
        b_dev,
        a_scale_dev,
        b_scale_dev,
        ctx,
    )

    ctx.enqueue_copy(c_my_host, c_my_device)
    ctx.enqueue_copy(c_ref_host, c_ref_device)
    ctx.synchronize()

    for i in range(c_size):
        c_my_f32[i] = c_my_host[i].cast[.float32]()

    # Max abs / rel error vs the naive reference (reported for the record).
    var max_abs = Float32(0.0)
    var max_rel = Float32(0.0)
    for i in range(c_size):
        var got = c_my_f32[i]
        var expected = c_ref_host[i]
        var abs_err = abs(got - expected)
        if abs_err > max_abs:
            max_abs = abs_err
        var denom = abs(expected)
        if denom > 1e-6:
            var rel = abs_err / denom
            if rel > max_rel:
                max_rel = rel
    print("   max_abs_err=", max_abs, " max_rel_err=", max_rel, sep="")

    # float32 output isolates scale-indexing correctness (only MFMA vs
    # sequential accumulation order differs); bfloat16 adds a looser rounding band.
    comptime rtol = 1e-2 if c_type == .float32 else 4e-2
    comptime atol = 1e-2 if c_type == .float32 else 2e-1
    assert_almost_equal(
        c_my_f32.unsafe_ptr(),
        c_ref_host.unsafe_ptr(),
        c_size,
        "tiled vs naive",
        atol=atol,
        rtol=rtol,
    )

    comptime if check_cpu:
        # Independent host fp32 oracle (small shapes only — O(M*N*K)).
        var a_host = TileTensor(a_host_ptr, row_major(a_shape))
        var b_host = TileTensor(b_host_ptr, row_major(b_shape))
        var a_scale_host = TileTensor(
            a_scale_host_ptr, row_major(a_scale_shape)
        )
        var b_scale_host = TileTensor(
            b_scale_host_ptr, row_major(b_scale_shape)
        )
        var c_cpu_ptr = ctx.enqueue_create_host_buffer[.float32](c_size)
        for _m in range(M):
            for _n in range(N):
                var res: Float32 = 0.0
                for _k in range(K):
                    var a_scale = a_scale_host[
                        Coord(_k // BLOCK_SCALE_K, _m // BLOCK_SCALE_M)
                    ]
                    var b_scale = b_scale_host[
                        Coord(_n // BLOCK_SCALE_N, _k // BLOCK_SCALE_K)
                    ]
                    res += (
                        a_host[_m, _k].cast[.float32]()
                        * b_host[_n, _k].cast[.float32]()
                        * a_scale
                        * b_scale
                    )
                c_cpu_ptr[_m * N + _n] = res
        assert_almost_equal(
            c_my_f32.unsafe_ptr(),
            c_cpu_ptr.unsafe_ptr(),
            c_size,
            "tiled vs cpu fp32",
            atol=atol,
            rtol=rtol,
        )

    print("   PASSED")


def main() raises:
    with DeviceContext() as ctx:
        # Small shape: independent CPU fp32 oracle + tight fp32 band.
        test_blockwise_fp8_matmul_amd[.float32, check_cpu=True](
            ctx, Idx[256], Idx[512], Idx[256]
        )
        test_blockwise_fp8_matmul_amd[.bfloat16, check_cpu=True](
            ctx, Idx[256], Idx[512], Idx[256]
        )

        # GLM q_b: [2048, 2048] x [2048, 2048].
        test_blockwise_fp8_matmul_amd[.float32](
            ctx, Idx[2048], Idx[2048], Idx[2048]
        )
        test_blockwise_fp8_matmul_amd[.bfloat16](
            ctx, Idx[2048], Idx[2048], Idx[2048]
        )

        # GLM o_proj: [2048, 2048] x [2048, 6144].
        test_blockwise_fp8_matmul_amd[.float32](
            ctx, Idx[2048], Idx[6144], Idx[2048]
        )
        test_blockwise_fp8_matmul_amd[.bfloat16](
            ctx, Idx[2048], Idx[6144], Idx[2048]
        )

        # Dispatch wiring: block/block granularity routes to the tiled
        # kernel on AMD.
        test_blockwise_fp8_matmul_amd[.bfloat16, via_dispatch=True](
            ctx, Idx[2048], Idx[2048], Idx[2048]
        )

        # Partial M / N tiles (OOB masking + clamped scale reads): M=200 and
        # N=380 (not a multiple of 128).
        test_blockwise_fp8_matmul_amd[.float32, check_cpu=True](
            ctx, Idx[200], Idx[380], Idx[256]
        )

        # Decode split-K (PR02 review B2): M in {1, 8, 64} at K=2048/N=6144
        # takes `_launch_blockwise_split_k` (num_splits=4 at this shape) --
        # every shape above has M >= 200, so BM=64's `M <= 64` split branch,
        # its grid.z band filter, and the stacked-workspace reduce were
        # previously reachable only from the `manual` bench, which never
        # compares against a reference. M=64 is the split band's edge
        # (`M <= min(BM, _ws_max_m)` with `BM == SK_MAX_M == 64`).
        test_blockwise_fp8_matmul_amd[.bfloat16, check_cpu=True](
            ctx, Idx[1], Idx[6144], Idx[2048]
        )
        test_blockwise_fp8_matmul_amd[.bfloat16, check_cpu=True](
            ctx, Idx[8], Idx[6144], Idx[2048]
        )
        test_blockwise_fp8_matmul_amd[.bfloat16](
            ctx, Idx[64], Idx[6144], Idx[2048]
        )

        # K % BK tail (PR02 review B3): every case above has K in {256,
        # 2048}, both multiples of BK=128, so the K-tail zero-masking
        # (`comptime if K % BK != 0`) and the `swizzle=None` `MmaOp` it
        # forces were comptime-dead everywhere. K=192 leaves a round
        # `c_valid == 64`; K=300 leaves `c_valid == 44` (not a multiple of
        # 64) and also lands on a partial N tile (300 is not a multiple of
        # BN=64), so together they cover both an aligned and a ragged
        # partial K-slab width.
        test_blockwise_fp8_matmul_amd[.float32, check_cpu=True](
            ctx, Idx[64], Idx[256], Idx[192]
        )
        test_blockwise_fp8_matmul_amd[.float32, check_cpu=True](
            ctx, Idx[64], Idx[300], Idx[300]
        )
