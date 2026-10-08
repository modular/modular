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
"""Production-layout numerics test for the AMD tiled batched blockwise-FP8
dispatch (gfx950).

`test_batched_matmul_dynamic_scaled_fp8_amd_tiled.mojo` drives the batched
dispatch with contiguous rank-3 operands at M <= 8. The GLM-5.3 MLA absorb
GEMMs (`quantize_and_bmm_fp8_helper`, `nn/attention/gpu/mla_graph.mojo`)
never look like that:

  * `c` is a per-head slab of a `[seq_len, num_heads, N]` buffer viewed as
    `[num_heads, seq_len, N]`: batch stride `N`, row stride `num_heads * N`.
  * `a_scales` is `[B, ceildiv(K, g), align_up(M, 4)]`: the row stride is
    the padded M, and the quantizer never writes the padded columns.
  * M is the decode batch (one to hundreds of tokens), not 1..8.

This test builds exactly those layouts around the two absorb shapes
(`[M, 512, 192]` = Q @ w_uk, `[M, 256, 512]` = O @ w_uv) at GLM-5.3's
`(1, 64, 64)` granularity and sweeps M across one and several M-tiles, with
`(1, 128, 128)` as a control. Inputs are signed e4m3 with per-block
log-uniform scales, so every `(m, k, n)` term and every scale block is
load-bearing; the padded `a_scales` columns are NaN-poisoned so any read of
them surfaces. The reference is an fp32 host dequant-matmul over the same
strided buffers. The naive per-batch kernel runs on the same fixture as a
second oracle so a fixture bug and a tiled-kernel bug are told apart.
"""

from std.math import align_up, ceildiv, exp2, isnan
from std.random import rand
from std.sys import size_of
from std.utils.index import Index
from std.utils.numerics import nan

from max.gpu.host import DeviceContext
from layout import Idx, TileTensor, row_major
from layout.tile_layout import Layout as TileLayout
from linalg.bmm import (
    _batched_matmul_dynamic_scaled_fp8_impl,
    batched_matmul_dynamic_scaled_fp8_naive,
)


def run_case[
    c_type: DType,
    B: Int,
    N: Int,
    K: Int,
    gran: Int,
](ctx: DeviceContext, M: Int, mut failures: List[String]) raises:
    comptime a_type = DType.float8_e4m3fn
    comptime K_BLOCKS = ceildiv(K, gran)
    comptime N_BLOCKS = ceildiv(N, gran)
    # Mirrors `quantize_and_bmm_fp8_helper`: scales are `[B, K_BLOCKS,
    # align_up(M, 16 bytes)]`.
    comptime SCALES_M_PAD = 16 // size_of[DType.float32]()
    var padded_m = align_up(M, SCALES_M_PAD)

    var a_size = B * M * K
    var b_size = B * N * K
    var c_size = B * M * N
    var a_scales_size = B * K_BLOCKS * padded_m
    var b_scales_size = B * N_BLOCKS * K_BLOCKS

    var case_name = String(
        "c=",
        c_type,
        " B=",
        B,
        " M=",
        M,
        " N=",
        N,
        " K=",
        K,
        " gran=",
        gran,
    )
    print("== ", case_name, sep="")

    var a_host = ctx.enqueue_create_host_buffer[a_type](a_size)
    var b_host = ctx.enqueue_create_host_buffer[a_type](b_size)
    var c_disp_host = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_naive_host = ctx.enqueue_create_host_buffer[c_type](c_size)
    var a_scales_host = ctx.enqueue_create_host_buffer[DType.float32](
        a_scales_size
    )
    var b_scales_host = ctx.enqueue_create_host_buffer[DType.float32](
        b_scales_size
    )
    var u_host = ctx.enqueue_create_host_buffer[DType.float32](
        max(a_size, b_size)
    )
    ctx.synchronize()

    # Signed e4m3 inputs with magnitude in [0.25, 2): every term contributes,
    # and cancellation makes a wrongly-scaled block visible.
    rand(u_host.unsafe_ptr(), a_size, min=-1.0, max=1.0)
    for i in range(a_size):
        var u = u_host[i]
        var mag = Float32(0.25) + Float32(1.75) * abs(u)
        a_host[i] = (mag if u >= 0 else -mag).cast[a_type]()
    rand(u_host.unsafe_ptr(), b_size, min=-1.0, max=1.0)
    for i in range(b_size):
        var u = u_host[i]
        var mag = Float32(0.25) + Float32(1.75) * abs(u)
        b_host[i] = (mag if u >= 0 else -mag).cast[a_type]()

    # Log-uniform per-block scales. An e4m3 output narrows the range so no
    # K=192 sum approaches the 448 max.
    comptime scale_lo = Float32(-3.0) if c_type.is_float8() else Float32(-2.0)
    comptime scale_span = Float32(3.0) if c_type.is_float8() else Float32(4.0)
    rand(a_scales_host.unsafe_ptr(), a_scales_size)
    for b in range(B):
        for kb in range(K_BLOCKS):
            for m in range(padded_m):
                var idx = (b * K_BLOCKS + kb) * padded_m + m
                if m < M:
                    a_scales_host[idx] = exp2(
                        scale_span * a_scales_host[idx] + scale_lo
                    )
                else:
                    # The quantizer never writes these; the kernel must
                    # never read them.
                    a_scales_host[idx] = nan[DType.float32]()
    rand(b_scales_host.unsafe_ptr(), b_scales_size)
    for i in range(b_scales_size):
        b_scales_host[i] = exp2(scale_span * b_scales_host[i] + scale_lo)

    # Poison the outputs so an unwritten element cannot pass by luck.
    for i in range(c_size):
        c_disp_host[i] = nan[c_type]()
        c_naive_host[i] = nan[c_type]()

    var a_dev = ctx.enqueue_create_buffer[a_type](a_size)
    var b_dev = ctx.enqueue_create_buffer[a_type](b_size)
    var c_disp_dev = ctx.enqueue_create_buffer[c_type](c_size)
    var c_naive_dev = ctx.enqueue_create_buffer[c_type](c_size)
    var a_scales_dev = ctx.enqueue_create_buffer[DType.float32](a_scales_size)
    var b_scales_dev = ctx.enqueue_create_buffer[DType.float32](b_scales_size)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(c_disp_dev, c_disp_host)
    ctx.enqueue_copy(c_naive_dev, c_naive_host)
    ctx.enqueue_copy(a_scales_dev, a_scales_host)
    ctx.enqueue_copy(b_scales_dev, b_scales_host)

    # Production layouts (see `mla_decode_branch_fp8`): `c` is a per-head slab of
    # `[M, B, N]`, `a` contiguous `[B, M, K]`, `a_scales` carries padded-M stride.
    var c_disp_tt = TileTensor(
        c_disp_dev,
        TileLayout((Idx[B], M, Idx[N]), (Idx[N], Idx[B * N], Idx[1])),
    )
    var c_naive_tt = TileTensor(
        c_naive_dev,
        TileLayout((Idx[B], M, Idx[N]), (Idx[N], Idx[B * N], Idx[1])),
    )
    var a_tt = TileTensor(a_dev, row_major(Idx[B], M, Idx[K]))
    var b_tt = TileTensor(b_dev, row_major[B, N, K]())
    var a_scales_tt = TileTensor(
        a_scales_dev, row_major((Idx[B], Idx[K_BLOCKS], padded_m))
    )
    var b_scales_tt = TileTensor(
        b_scales_dev, row_major[B, N_BLOCKS, K_BLOCKS]()
    )

    # The production entry: on gfx950 this is one batched tiled launch.
    var dispatched = _batched_matmul_dynamic_scaled_fp8_impl[
        input_scale_granularity="block",
        weight_scale_granularity="block",
        m_scale_granularity=1,
        n_scale_granularity=gran,
        k_scale_granularity=gran,
        transpose_b=True,
        target="gpu",
    ](c_disp_tt, a_tt, b_tt, a_scales_tt, b_scales_tt, ctx)
    # Second oracle on the identical fixture.
    _ = batched_matmul_dynamic_scaled_fp8_naive[
        scales_granularity_mnk=Index(1, gran, gran), transpose_b=True
    ](c_naive_tt, a_tt, b_tt, a_scales_tt, b_scales_tt, ctx)

    ctx.enqueue_copy(c_disp_host, c_disp_dev)
    ctx.enqueue_copy(c_naive_host, c_naive_dev)
    ctx.synchronize()

    if dispatched != 1:
        failures.append(
            case_name
            + ": expected 1 batched dispatch, got "
            + String(dispatched)
        )

    # e4m3 (3 mantissa bits) compares against the un-rounded fp32 sum within
    # half an ULP; bf16/f32 compare against the sum rounded through c_type.
    comptime rtol = Float32(
        1.0 / 16.0 + 1e-3
    ) if c_type.is_float8() else Float32(1e-2)
    comptime atol = Float32(1e-2)

    var ndiff_disp = 0
    var ndiff_naive = 0
    var nnan_disp = 0
    for b in range(B):
        for m in range(M):
            for n in range(N):
                var accum = Float32(0)
                for k in range(K):
                    var a_val = a_host[(b * M + m) * K + k].cast[
                        DType.float32
                    ]()
                    var b_val = b_host[(b * N + n) * K + k].cast[
                        DType.float32
                    ]()
                    var a_s = a_scales_host[
                        (b * K_BLOCKS + k // gran) * padded_m + m
                    ]
                    var b_s = b_scales_host[
                        (b * N_BLOCKS + n // gran) * K_BLOCKS + k // gran
                    ]
                    accum += a_val * b_val * a_s * b_s
                var ref_val = accum
                comptime if not c_type.is_float8():
                    ref_val = accum.cast[c_type]().cast[DType.float32]()
                var c_idx = (m * B + b) * N + n
                var got = c_disp_host[c_idx].cast[DType.float32]()
                var got_naive = c_naive_host[c_idx].cast[DType.float32]()
                var tol = atol + rtol * abs(ref_val)
                if isnan(got):
                    nnan_disp += 1
                if not (abs(got - ref_val) <= tol):
                    ndiff_disp += 1
                    if ndiff_disp <= 4:
                        print(
                            "  tiled diff @ (b,m,n)=(",
                            b,
                            ",",
                            m,
                            ",",
                            n,
                            ") got=",
                            got,
                            " ref=",
                            ref_val,
                            " naive=",
                            got_naive,
                            sep="",
                        )
                if not (abs(got_naive - ref_val) <= tol):
                    ndiff_naive += 1

    if ndiff_naive > 0:
        failures.append(
            case_name
            + ": NAIVE oracle diverged from host ref at "
            + String(ndiff_naive)
            + " of "
            + String(c_size)
            + " (fixture problem)"
        )
    if ndiff_disp > 0:
        failures.append(
            case_name
            + ": tiled batched diverged from host ref at "
            + String(ndiff_disp)
            + " of "
            + String(c_size)
            + " positions ("
            + String(nnan_disp)
            + " NaN)"
        )
        print(
            "  FAILED:",
            ndiff_disp,
            "of",
            c_size,
            "positions differ (",
            nnan_disp,
            "NaN )",
        )
    else:
        print("  ok (", c_size, "positions )")


def sweep_m[
    c_type: DType,
    B: Int,
    N: Int,
    K: Int,
    gran: Int,
](ctx: DeviceContext, ms: List[Int], mut failures: List[String]) raises:
    for m in ms:
        run_case[c_type, B, N, K, gran](ctx, m, failures)


def main() raises:
    var failures = List[String]()
    # One M-tile (BM=64) at several fill levels, then two and three M-tiles,
    # including M-tiles that MMA rows and the a_scales pad both straddle.
    var ms: List[Int] = [1, 2, 3, 5, 8, 9, 16, 17, 33, 48, 64, 65, 100, 129]
    var ms_short: List[Int] = [1, 8, 33, 65]

    with DeviceContext() as ctx:
        # GLM-5.3 absorb shapes, 8 heads per GPU (TP8), production dtypes.
        # O @ w_uv: bf16 output, N=256, K=512 (four full slabs).
        sweep_m[.bfloat16, 8, 256, 512, 64](ctx, ms, failures)
        # Q @ w_uk over an FP8 latent cache: e4m3 output, N=512, K=192.
        sweep_m[.float8_e4m3fn, 8, 512, 192, 64](ctx, ms, failures)
        # Same two shapes in f32, where the tiled kernel is fully visible.
        sweep_m[.float32, 8, 256, 512, 64](ctx, ms_short, failures)
        sweep_m[.float32, 8, 512, 192, 64](ctx, ms_short, failures)
        # Q @ w_uk with a bf16 output (a bf16 latent cache).
        sweep_m[.bfloat16, 8, 512, 192, 64](ctx, ms_short, failures)
        # All 64 heads on one GPU (attention data-parallel).
        sweep_m[.bfloat16, 64, 256, 512, 64](ctx, ms_short, failures)
        # Control: the on-disk (1,128,128) granularity on the same layouts.
        sweep_m[.bfloat16, 8, 256, 512, 128](ctx, ms_short, failures)

    if len(failures) > 0:
        print("\n", len(failures), "FAILING CASES:")
        for f in failures:
            print("  ", f)
        raise Error(
            String(len(failures))
            + " production-layout case(s) failed; see list above"
        )
    print("ALL PASSED")
