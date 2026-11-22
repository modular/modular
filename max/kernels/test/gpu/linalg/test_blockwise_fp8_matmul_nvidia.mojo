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
"""Correctness and bandwidth for the tiled blockwise-scaled FP8 matmul on
NVIDIA parts without tcgen05 (GB10 / sm_121a).

Numerics: every case compares `blockwise_scaled_fp8_matmul_nvidia` against the
trusted `naive_blockwise_scaled_fp8_matmul` on identical operands, and the
small shapes against an independent host fp32 loop. Operands are signed,
uniform in `[-1, 1)`, so block sums cancel the way real activations do and a
dropped sign or a lost term does not hide in a sum of positives. Scales are drawn
LOG-UNIFORM in `[2^-4, 2^4)` per block because a constant scale hides a
scale-indexing bug, which is this kernel's most plausible failure mode
(`a_scale` is indexed `[kb, m]`, `b_scale` `[n // 128, kb]`).

Wiring: `via_dispatch` runs the production entry point
(`matmul_dynamic_scaled_fp8` with block/block granularity) and asserts
`_nvidia_blockwise_tiled_ok` is True -- a numeric match alone cannot prove the
tiled route was taken, since a gate that regressed to False falls back to the
(also correct) naive kernel.

Routes: the launcher runs three tiles -- the GEMV for decode (`M <= 4`), the
8-row SIMT tile for small batches and any forced tile width or split, and the
128-row tensor-core tile for prefill. `expect_gemv` and `expect_tc` assert a
case lands where it should through the launcher's own
`_nvidia_blockwise_gemv_ok` / `_nvidia_blockwise_tiles` /
`_nvidia_blockwise_tc_ok`, for the same reason: every tile is numerically
correct, so only the route tells them apart.

"""

from std.math import ceildiv, exp2

from max.gpu.host import DeviceContext
from max.gpu.host.info import (
    DGXSpark,
    GPUInfo,
    MI455X,
    RTX5090,
    Radeon9060,
    Radeon9070,
)
from internal_utils import assert_almost_equal
from std.random import rand
from std.testing import assert_false, assert_true
from layout import CoordLike, Coord, Idx, TileTensor, row_major
from linalg.fp8_quantization import (
    naive_blockwise_scaled_fp8_matmul,
    matmul_dynamic_scaled_fp8,
    blockwise_scaled_fp8_with_epilogue,
)
from linalg.matmul.gpu.blockwise_scaled_fp8_matmul_nvidia import (
    _nvidia_blockwise_gemv_ok,
    _nvidia_blockwise_tc_ok,
    _nvidia_blockwise_tiles,
    _nvidia_blockwise_tiled_ok,
    blockwise_scaled_fp8_matmul_nvidia,
)

comptime _BLOCK_SCALE_M = 1
comptime _BLOCK_SCALE_N = 128
comptime _BLOCK_SCALE_K = 128


def test_blockwise_fp8_matmul_nvidia[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    c_type: DType,
    check_cpu: Bool = False,
    via_dispatch: Bool = False,
    via_epilogue: Bool = False,
    bn: Int = 0,
    split: Int = 0,
    expect_tc: Bool = False,
    expect_gemv: Bool = False,
    scale_ramp: Bool = False,
    zero_scales: Bool = False,
](ctx: DeviceContext, m: MType, n: NType, k: KType) raises:
    """Compares one shape against the naive reference, and optionally the host.

    Parameters:
        MType: Type of the M argument (static `Idx` or dynamic `Int`).
        NType: Type of the N argument.
        KType: Type of the K argument.
        c_type: Output dtype under test.
        check_cpu: Also compare against an independent host fp32 loop.
        via_dispatch: Launch through `matmul_dynamic_scaled_fp8` instead of the
            kernel entry point, and assert the dispatch gate is open.
        via_epilogue: Launch through the `has_epilogue_fn` path, whose epilogue
            stores the output element itself (the route the serving graph
            takes). Which tile then runs depends on the shape.
        bn: Force the n tile width (0 leaves it to the launcher).
        split: Force the split-K factor (0 leaves it to the launcher).
        expect_tc: Assert the shape routes to the 128-row tensor-core tile.
        expect_gemv: Assert the shape routes to the decode GEMV.
        scale_ramp: Replace the random scales with one that grows by 4x per
            128-wide K block, so a bug in the running-ratio rescale cannot
            hide behind scales that happen to be similar.
        zero_scales: Zero whole scale blocks in the middle of K (an `a_scale`
            row for every token, a `b_scale` column for every n block), one
            token's `a_scale` and the first n block's leading `b_scale`, so a
            slab whose scale is zero must contribute nothing without erasing
            the slabs around it.

    Args:
        ctx: Device context.
        m: Number of rows.
        n: Number of columns of the output.
        k: Reduction depth.
    """
    comptime input_type = DType.float8_e4m3fn
    comptime transpose_b = True

    var M = Int(m)
    var N = Int(n)
    var K = Int(k)

    print(
        (
            t"== tiled blockwise fp8 nvidia c_type={c_type}"
            t" MxNxK={M}x{N}x{K} dispatch={via_dispatch}"
            t" epilogue={via_epilogue}"
        ),
        sep="",
    )

    # The route the launcher takes for this shape: the 8-row SIMT tile, or
    # prefill's 128-row tensor-core tile. A numeric match cannot tell them
    # apart -- both are correct -- so the route is asserted where it matters.
    var route_bn, route_split = _nvidia_blockwise_tiles(
        M,
        N,
        ceildiv(K, _BLOCK_SCALE_K),
        ctx.device().multiprocessor_count(),
        bn,
        split,
    )
    var routes_to_gemv = _nvidia_blockwise_gemv_ok(M, K, bn, split)
    var routes_to_tc = (not routes_to_gemv) and _nvidia_blockwise_tc_ok(
        M, route_bn, route_split
    )
    print(
        "   route: gemv=",
        routes_to_gemv,
        " bn=",
        route_bn,
        " split=",
        route_split,
        " tensor_core=",
        routes_to_tc,
        sep="",
    )
    comptime if expect_tc:
        assert_true(
            routes_to_tc,
            "this shape must take the 128-row tensor-core prefill path",
        )
    comptime if expect_gemv:
        assert_true(routes_to_gemv, "this shape must take the decode GEMV")

    var a_shape = Coord(m, k)
    var b_shape = Coord(n, k)
    var c_shape = Coord(m, n)
    var a_scale_shape = Coord(
        ceildiv(K, _BLOCK_SCALE_K), ceildiv(M, _BLOCK_SCALE_M)
    )
    var b_scale_shape = Coord(
        ceildiv(N, _BLOCK_SCALE_N), ceildiv(K, _BLOCK_SCALE_K)
    )

    var a_size = M * K
    var b_size = N * K
    var c_size = M * N
    var a_scale_size = ceildiv(K, _BLOCK_SCALE_K) * ceildiv(M, _BLOCK_SCALE_M)
    var b_scale_size = ceildiv(N, _BLOCK_SCALE_N) * ceildiv(K, _BLOCK_SCALE_K)

    var a_host = ctx.enqueue_create_host_buffer[input_type](a_size)
    var b_host = ctx.enqueue_create_host_buffer[input_type](b_size)
    var a_scale_host = ctx.enqueue_create_host_buffer[.float32](a_scale_size)
    var b_scale_host = ctx.enqueue_create_host_buffer[.float32](b_scale_size)

    rand(a_host.unsafe_ptr(), a_size)
    rand(b_host.unsafe_ptr(), b_size)
    # Rewrite the [0, 1) samples as signed values in [-1, 1).
    for i in range(a_size):
        a_host[i] = (a_host[i].cast[.float32]() * 2.0 - 1.0).cast[input_type]()
    for i in range(b_size):
        b_host[i] = (b_host[i].cast[.float32]() * 2.0 - 1.0).cast[input_type]()
    rand(a_scale_host.unsafe_ptr(), a_scale_size)
    rand(b_scale_host.unsafe_ptr(), b_scale_size)
    # Rewrite the [0, 1) samples as LOG-UNIFORM scales in [2^-4, 2^4).
    for i in range(a_scale_size):
        a_scale_host[i] = exp2(Float32(8.0) * a_scale_host[i] - 4.0)
    for i in range(b_scale_size):
        b_scale_host[i] = exp2(Float32(8.0) * b_scale_host[i] - 4.0)

    comptime if scale_ramp:
        # `a_scale` and `b_scale` each double per 128-wide K block, so the
        # product that a slab is accumulated in grows by 4x per slab. A ratio
        # left at 1, or applied to the wrong slab, then cannot cancel out the
        # way it can when neighbouring slabs carry similar scales.
        for i in range(a_scale_size):
            a_scale_host[i] = exp2(Float32(i // M))
        for i in range(b_scale_size):
            b_scale_host[i] = exp2(Float32(i % ceildiv(K, _BLOCK_SCALE_K)))

    comptime if zero_scales:
        # The operands stay random, so a zero-scale slab whose raw products
        # leak into the accumulator, or that rescales it to zero, shows up.
        var k_blocks = ceildiv(K, _BLOCK_SCALE_K)
        var n_blocks = ceildiv(N, _BLOCK_SCALE_N)
        var a_zero_kb = k_blocks // 3
        var b_zero_kb = 2 * k_blocks // 3
        for i in range(M):
            a_scale_host[a_zero_kb * M + i] = 0.0
        for i in range(n_blocks):
            b_scale_host[i * k_blocks + b_zero_kb] = 0.0
        # One token's scale alone, and a zero first slab for one n block, so
        # the scale that precedes every nonzero one is zero.
        a_scale_host[(a_zero_kb + 1) * M + min(5, M - 1)] = 0.0
        b_scale_host[0] = 0.0

    var c_my_host = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_my_f32 = ctx.enqueue_create_host_buffer[.float32](c_size)
    var c_ref_host = ctx.enqueue_create_host_buffer[.float32](c_size)

    var a_device = ctx.enqueue_create_buffer[input_type](a_size)
    var b_device = ctx.enqueue_create_buffer[input_type](b_size)
    var c_my_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_scratch_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_ref_device = ctx.enqueue_create_buffer[.float32](c_size)
    var c_plain_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_plain_host = ctx.enqueue_create_host_buffer[c_type](c_size)
    var a_scale_device = ctx.enqueue_create_buffer[.float32](a_scale_size)
    var b_scale_device = ctx.enqueue_create_buffer[.float32](b_scale_size)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)
    ctx.enqueue_copy(a_scale_device, a_scale_host)
    ctx.enqueue_copy(b_scale_device, b_scale_host)

    var a_dev = TileTensor(a_device, row_major(a_shape))
    var b_dev = TileTensor(b_device, row_major(b_shape))
    var c_my_dev = TileTensor(c_my_device, row_major(c_shape))
    var c_ref_dev = TileTensor(c_ref_device, row_major(c_shape))
    var a_scale_dev = TileTensor(a_scale_device, row_major(a_scale_shape))
    var b_scale_dev = TileTensor(b_scale_device, row_major(b_scale_shape))

    comptime if via_epilogue:
        # The epilogue stores the output itself, so the kernel is handed a
        # scratch output: this is prefill's route (`has_epilogue_fn=True`).
        var c_scratch_dev = TileTensor(c_scratch_device, row_major(c_shape))
        var c_other_dev = TileTensor(c_my_device, row_major(c_shape))
        var c_plain_dev = TileTensor(c_plain_device, row_major(c_shape))
        ctx.enqueue_memset[c_type](c_scratch_device, 0)
        ctx.enqueue_memset[c_type](c_my_device, 0)

        # The same operands through the plain entry point: the epilogue path
        # must reproduce this bit for bit.
        blockwise_scaled_fp8_matmul_nvidia[
            transpose_b=transpose_b, bn_override=bn, split_override=split
        ](c_plain_dev, a_dev, b_dev, a_scale_dev, b_scale_dev, ctx)

        @always_inline
        def store_epilogue[
            dtype: DType, width: SIMDLength, *, alignment: Int
        ](coords: Coord, val: SIMD[dtype, width]) {var c_other_dev}:
            comptime assert coords.rank == 2
            c_other_dev.store[alignment=alignment](
                coords, rebind[SIMD[c_type, width]](val)
            )

        blockwise_scaled_fp8_with_epilogue[
            transpose_b=transpose_b,
            scales_granularity_mnk=Coord[Int, Int, Int](
                _BLOCK_SCALE_M, _BLOCK_SCALE_N, _BLOCK_SCALE_K
            ),
            has_epilogue_fn=True,
        ](
            c_scratch_dev,
            a_dev,
            b_dev,
            a_scale_dev,
            b_scale_dev,
            ctx,
            store_epilogue,
        )
    elif via_dispatch:
        assert_true(
            _nvidia_blockwise_tiled_ok[
                c_type=c_type,
                a_type=input_type,
                b_type=input_type,
                a_scales_type=.float32,
                b_scales_type=.float32,
                transpose_b=transpose_b,
                scales_granularity_mnk=(
                    _BLOCK_SCALE_M,
                    _BLOCK_SCALE_N,
                    _BLOCK_SCALE_K,
                ),
                device_info=ctx.default_device_info,
            ](),
            (
                "tiled blockwise-FP8 NVIDIA gate (_nvidia_blockwise_tiled_ok)"
                " is False for a production dtype/shape -- dispatch would"
                " silently fall back to the naive kernel"
            ),
        )
        matmul_dynamic_scaled_fp8[
            input_scale_granularity="block",
            weight_scale_granularity="block",
            m_scale_granularity=_BLOCK_SCALE_M,
            n_scale_granularity=_BLOCK_SCALE_N,
            k_scale_granularity=_BLOCK_SCALE_K,
            transpose_b=transpose_b,
        ](c_my_dev, a_dev, b_dev, a_scale_dev, b_scale_dev, ctx)
    else:
        blockwise_scaled_fp8_matmul_nvidia[
            transpose_b=transpose_b, bn_override=bn, split_override=split
        ](c_my_dev, a_dev, b_dev, a_scale_dev, b_scale_dev, ctx)

    naive_blockwise_scaled_fp8_matmul[
        BLOCK_DIM=16,
        transpose_b=transpose_b,
        scales_granularity_mnk=Coord[Int, Int, Int](
            _BLOCK_SCALE_M, _BLOCK_SCALE_N, _BLOCK_SCALE_K
        ),
    ](c_ref_dev, a_dev, b_dev, a_scale_dev, b_scale_dev, ctx)

    ctx.enqueue_copy(c_my_host, c_my_device)
    ctx.enqueue_copy(c_ref_host, c_ref_device)
    ctx.synchronize()

    for i in range(c_size):
        c_my_f32[i] = c_my_host[i].cast[.float32]()

    var max_abs = Float32(0.0)
    var max_rel = Float32(0.0)
    var max_ref = Float32(0.0)
    for i in range(c_size):
        var got = c_my_f32[i]
        var expected = c_ref_host[i]
        var abs_err = abs(got - expected)
        if abs_err > max_abs:
            max_abs = abs_err
        var mag = abs(expected)
        if mag > max_ref:
            max_ref = mag
        if mag > 1e-6:
            var rel = abs_err / mag
            if rel > max_rel:
                max_rel = rel
    comptime if via_epilogue:
        ctx.enqueue_copy(c_plain_host, c_plain_device)
        ctx.synchronize()
        var mismatches = 0
        for _i in range(c_size):
            if c_plain_host[_i] != c_my_host[_i]:
                mismatches += 1
        print(
            "   epilogue-vs-plain mismatches=", mismatches, "/", c_size, sep=""
        )
        assert_true(
            mismatches == 0,
            "the has_epilogue_fn path must reproduce the plain path exactly",
        )

    print(
        "   max_abs_err=",
        max_abs,
        " max_rel_err=",
        max_rel,
        " max|ref|=",
        max_ref,
        sep="",
    )

    # float32 output isolates scale-indexing correctness: only the accumulation
    # order differs, so the error is fp32 summation noise, a few ulps of the
    # largest partial sum. Signed operands cancel, so an element can sit far
    # below the sums that formed it, and the absolute floor follows the
    # output's magnitude. bfloat16 adds the reference's own rounding band.
    comptime rtol = 1e-4 if c_type == .float32 else 4e-2
    var atol = Float64(max_ref) * 1e-5 if c_type == .float32 else 2e-1
    assert_almost_equal(
        c_my_f32.unsafe_ptr(),
        c_ref_host.unsafe_ptr(),
        c_size,
        "tiled vs naive",
        atol=atol,
        rtol=rtol,
    )

    comptime if check_cpu:
        # Independent host fp32 oracle (small shapes only -- O(M*N*K)).
        var a_host_t = TileTensor(a_host, row_major(a_shape))
        var b_host_t = TileTensor(b_host, row_major(b_shape))
        var a_scale_host_t = TileTensor(a_scale_host, row_major(a_scale_shape))
        var b_scale_host_t = TileTensor(b_scale_host, row_major(b_scale_shape))
        var c_cpu = ctx.enqueue_create_host_buffer[.float32](c_size)
        for _m in range(M):
            for _n in range(N):
                var res: Float32 = 0.0
                for _k in range(K):
                    var a_scale = a_scale_host_t[
                        Coord(_k // _BLOCK_SCALE_K, _m // _BLOCK_SCALE_M)
                    ]
                    var b_scale = b_scale_host_t[
                        Coord(_n // _BLOCK_SCALE_N, _k // _BLOCK_SCALE_K)
                    ]
                    res += (
                        a_host_t[_m, _k].cast[.float32]()
                        * b_host_t[_n, _k].cast[.float32]()
                        * a_scale
                        * b_scale
                    )
                c_cpu[_m * N + _n] = res
        assert_almost_equal(
            c_my_f32.unsafe_ptr(),
            c_cpu.unsafe_ptr(),
            c_size,
            "tiled vs cpu fp32",
            atol=atol,
            rtol=rtol,
        )

    print("   PASSED")


def test_gate_excludes_amd() raises:
    """AMD parts that report compute 12.x must not take the NVIDIA kernel.

    `_is_sm12x_gpu` compares only the compute capability, which MI455X
    (gfx1250, 12.5), Radeon 9070 (gfx1201, 12.0) and Radeon 9060 (gfx1200,
    12.0) also satisfy. The gate is compile-time, so this runs on any device.
    """

    def gate[info: GPUInfo]() -> Bool:
        return _nvidia_blockwise_tiled_ok[
            c_type=.bfloat16,
            a_type=.float8_e4m3fn,
            b_type=.float8_e4m3fn,
            a_scales_type=.float32,
            b_scales_type=.float32,
            transpose_b=True,
            scales_granularity_mnk=(
                _BLOCK_SCALE_M,
                _BLOCK_SCALE_N,
                _BLOCK_SCALE_K,
            ),
            device_info=info,
        ]()

    assert_false(gate[MI455X.gpu_info](), "MI455X must not take this kernel")
    assert_false(
        gate[Radeon9070.gpu_info](), "Radeon 9070 must not take this kernel"
    )
    assert_false(
        gate[Radeon9060.gpu_info](), "Radeon 9060 must not take this kernel"
    )
    assert_true(gate[DGXSpark.gpu_info](), "GB10 must take this kernel")
    assert_true(gate[RTX5090.gpu_info](), "RTX 5090 must take this kernel")


def main() raises:
    test_gate_excludes_amd()
    with DeviceContext() as ctx:
        # Small shape with an independent host fp32 oracle, both output dtypes.
        test_blockwise_fp8_matmul_nvidia[.float32, check_cpu=True](
            ctx, Idx[16], Idx[256], Idx[256]
        )
        test_blockwise_fp8_matmul_nvidia[.bfloat16, check_cpu=True](
            ctx, Idx[16], Idx[256], Idx[256]
        )

        # Decode and prefill-ish tiles around the 8-row block edge.
        test_blockwise_fp8_matmul_nvidia[.bfloat16, expect_gemv=True](
            ctx, Idx[1], Idx[1024], Idx[1024]
        )
        test_blockwise_fp8_matmul_nvidia[.bfloat16](
            ctx, Idx[8], Idx[1024], Idx[1024]
        )
        test_blockwise_fp8_matmul_nvidia[.bfloat16](
            ctx, Idx[9], Idx[1024], Idx[1024]
        )
        test_blockwise_fp8_matmul_nvidia[.bfloat16](
            ctx, Idx[64], Idx[1024], Idx[1024]
        )

        # Dynamic M (the serving graph's token count) and dynamic N/K.
        test_blockwise_fp8_matmul_nvidia[.bfloat16](
            ctx, 1, Idx[1024], Idx[1024]
        )
        test_blockwise_fp8_matmul_nvidia[.bfloat16](ctx, 5, 1024, 640)

        # Partial N tile (380 is not a multiple of 128) and partial M tile.
        test_blockwise_fp8_matmul_nvidia[.float32, check_cpu=True](
            ctx, Idx[5], Idx[380], Idx[256]
        )
        test_blockwise_fp8_matmul_nvidia[.bfloat16](
            ctx, Idx[200], Idx[380], Idx[256]
        )

        # K tail: K=192 leaves a 64-wide partial slab, K=300 a ragged 44-wide
        # one that also lands on a partial N tile.
        test_blockwise_fp8_matmul_nvidia[.float32, check_cpu=True](
            ctx, Idx[8], Idx[256], Idx[192]
        )
        test_blockwise_fp8_matmul_nvidia[.float32](
            ctx, Idx[8], Idx[300], Idx[300]
        )

        # Production wiring: the dynamic-scaled entry point with block/block
        # granularity must route here (the gate assertion is in the test).
        # M=1 runs the decode GEMV; M=16 has too few CTAs to fill the SMs, so
        # it splits K and runs the 8-row SIMT tile; M=256 has enough and runs
        # the 128-row tensor-core tile, which `expect_tc` asserts.
        test_blockwise_fp8_matmul_nvidia[
            .bfloat16, via_dispatch=True, expect_gemv=True
        ](ctx, Idx[1], Idx[1024], Idx[1024])
        test_blockwise_fp8_matmul_nvidia[.bfloat16, via_dispatch=True](
            ctx, Idx[16], Idx[2048], Idx[1024]
        )
        test_blockwise_fp8_matmul_nvidia[
            .bfloat16, via_dispatch=True, expect_tc=True
        ](ctx, Idx[256], Idx[1024], Idx[1024])

        # The serving graph's route: `has_epilogue_fn=True` with the epilogue
        # storing. M=16 splits K and runs the SIMT tile; M=256 runs the
        # tensor-core tile.
        test_blockwise_fp8_matmul_nvidia[.bfloat16, via_epilogue=True](
            ctx, Idx[16], Idx[1024], Idx[1024]
        )
        test_blockwise_fp8_matmul_nvidia[
            .bfloat16, via_epilogue=True, expect_tc=True
        ](ctx, Idx[256], Idx[1024], Idx[1024])

        # The decode GEMV: every row count it takes, against the host where
        # the shape is small; an N tail (380 is not a multiple of its 8
        # columns or of 128); a K that ends mid-slab (192) and one that
        # crosses a 512-wide warp step unevenly (1040 = 2 * 512 + 16); the
        # serving graph's epilogue route; and the real projection shape.
        test_blockwise_fp8_matmul_nvidia[
            .float32, check_cpu=True, expect_gemv=True
        ](ctx, Idx[1], Idx[256], Idx[256])
        test_blockwise_fp8_matmul_nvidia[
            .float32, check_cpu=True, expect_gemv=True
        ](ctx, Idx[2], Idx[384], Idx[512])
        test_blockwise_fp8_matmul_nvidia[.float32, expect_gemv=True](
            ctx, Idx[3], Idx[512], Idx[1024]
        )
        test_blockwise_fp8_matmul_nvidia[
            .bfloat16, check_cpu=True, expect_gemv=True
        ](ctx, Idx[4], Idx[256], Idx[256])
        test_blockwise_fp8_matmul_nvidia[
            .float32, check_cpu=True, expect_gemv=True
        ](ctx, Idx[1], Idx[380], Idx[256])
        test_blockwise_fp8_matmul_nvidia[
            .float32, check_cpu=True, expect_gemv=True
        ](ctx, Idx[3], Idx[256], Idx[192])
        test_blockwise_fp8_matmul_nvidia[.float32, expect_gemv=True](
            ctx, Idx[1], Idx[640], Idx[1040]
        )
        test_blockwise_fp8_matmul_nvidia[
            .bfloat16, via_epilogue=True, expect_gemv=True
        ](ctx, Idx[1], Idx[1024], Idx[1024])
        test_blockwise_fp8_matmul_nvidia[.bfloat16, expect_gemv=True](
            ctx, 1, Idx[5120], Idx[17408]
        )
        # K not a multiple of 16 leaves the GEMV for the 8-row tile.
        test_blockwise_fp8_matmul_nvidia[.float32, check_cpu=True](
            ctx, Idx[1], Idx[256], Idx[200]
        )
        assert_false(
            _nvidia_blockwise_gemv_ok(1, 200, 0, 0),
            "K % 16 != 0 must not take the decode GEMV",
        )
        assert_false(
            _nvidia_blockwise_gemv_ok(5, 1024, 0, 0),
            "M > 4 must not take the decode GEMV",
        )

        # Split-K correctness: forced factors, at the decode K.
        test_blockwise_fp8_matmul_nvidia[.bfloat16, bn=128, split=4](
            ctx, Idx[1], Idx[512], Idx[5120]
        )
        test_blockwise_fp8_matmul_nvidia[.bfloat16, bn=64, split=2](
            ctx, Idx[3], Idx[300], Idx[1024]
        )
        test_blockwise_fp8_matmul_nvidia[.float32, bn=128, split=4](
            ctx, Idx[9], Idx[256], Idx[1024]
        )

        # The 128-row tensor-core tile. Shapes here keep at least 170 CTAs
        # (or fewer than 8 K blocks) so they do not split K on a 48-SM GB10
        # or a 170-SM RTX 5090. One m tile, then two,
        # then one that is cut to a single row (M=129) so the m-boundary path
        # -- zero-filled rows and the clamped `a_scale` row -- is measured.
        test_blockwise_fp8_matmul_nvidia[.float32, expect_tc=True](
            ctx, Idx[128], Idx[768], Idx[256]
        )
        test_blockwise_fp8_matmul_nvidia[
            .bfloat16, check_cpu=True, expect_tc=True
        ](ctx, Idx[129], Idx[768], Idx[256])
        test_blockwise_fp8_matmul_nvidia[.bfloat16, expect_tc=True](
            ctx, Idx[256], Idx[512], Idx[512]
        )

        # N tail inside the tensor-core tile: 900 = 7 * 128 + 4.
        test_blockwise_fp8_matmul_nvidia[.float32, expect_tc=True](
            ctx, Idx[256], Idx[900], Idx[512]
        )

        # K tail inside the last 128-wide slab: 1088 = 8 * 128 + 64.
        test_blockwise_fp8_matmul_nvidia[.bfloat16, expect_tc=True](
            ctx, Idx[256], Idx[768], Idx[1088]
        )

        # K not a multiple of 16: rows of `a` and `b` are no longer 16-byte
        # aligned, so the tensor-core tile stages every slab element by element
        # instead of with 16-byte loads. 1000 = 7 * 128 + 104 = 62 * 16 + 8;
        # 200 also ends on a ragged slab, checked against the host.
        test_blockwise_fp8_matmul_nvidia[.float32, expect_tc=True](
            ctx, Idx[256], Idx[768], Idx[1000]
        )
        test_blockwise_fp8_matmul_nvidia[
            .bfloat16, check_cpu=True, expect_tc=True
        ](ctx, Idx[129], Idx[768], Idx[200])

        # Scales that grow 4x per K slab, so the running rescale cannot hide.
        test_blockwise_fp8_matmul_nvidia[
            .float32, expect_tc=True, scale_ramp=True
        ](ctx, Idx[128], Idx[768], Idx[640])

        # Zero scale blocks in the middle of K on the tensor-core tile: the
        # slab must add nothing and leave the slabs before and after intact.
        # The decode GEMV covers the same scales at M=1, and the SIMT tile
        # at M=5.
        test_blockwise_fp8_matmul_nvidia[
            .float32, expect_tc=True, zero_scales=True
        ](ctx, Idx[256], Idx[768], Idx[1024])
        test_blockwise_fp8_matmul_nvidia[
            .float32, zero_scales=True, expect_gemv=True
        ](ctx, Idx[1], Idx[768], Idx[1024])
        test_blockwise_fp8_matmul_nvidia[.float32, zero_scales=True](
            ctx, Idx[5], Idx[768], Idx[1024]
        )
