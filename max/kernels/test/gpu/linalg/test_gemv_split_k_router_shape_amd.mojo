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
"""Checks cached GEMV and the small-M row-sharing route.

The cached-load, all-fp32 multirow launch is a combination no other test runs:
`test_gemv.mojo` covers all-fp32 multirow launches only with the default
`weight_non_temporal=True`,
and `weight_non_temporal=False` only with bf16 activations.

The N=896, K=7168 cells exercise the direct kernel, outside the dispatch
rule. A host fp64 reference catches wrong math; bit-exact agreement with M
separate `grid.x=1` launches catches indexing errors in multirow launches.
"""

from std.math import ceildiv
from std.random import randn, random_float64, seed
from std.sys import get_defined_bool, get_defined_int, simd_width_of
from std.sys.info import _accelerator_arch
from std.testing import assert_equal, assert_true

from internal_utils import assert_almost_equal

from layout import Idx, TileTensor, row_major
from linalg.gemv import (
    _amd_small_m_gemv_route,
    _amd_small_m_static_k_threads,
    gemv_split_k,
)
from linalg.matmul.gpu import _matmul_gpu
from max.gpu.host import DeviceContext, default_accelerator


def _run_case[
    M: Int,
    N: Int,
    K: Int,
    use_b: Bool,
    a_type: DType = .float32,
    c_type: DType = .float32,
](ctx: DeviceContext) raises:
    """Checks direct GEMV and selected matmul dispatches against references."""
    comptime gfx950 = _accelerator_arch() == "amdgpu:gfx950"
    # The gfx950 bf16 128-thread static-K instantiation: K % 1024 == 0 but
    # K % 2048 != 0, inside the config-B route, above the N >= 64 CTA floor,
    # with the M bands the dispatch stages (2..8, plus the N=576 column at
    # 9..16 mirroring the tuned 576x4096 cell).
    comptime static_k_128 = (
        use_b
        and a_type == .bfloat16
        and c_type in (DType.bfloat16, DType.float32)
        and gfx950
        and K > 0
        and K % 1024 == 0
        and K % 2048 != 0
        and N >= 64
        and (M >= 4 or N >= 576)
        and (2 <= M <= 8 or (9 <= M <= 16 and N == 576))
    )
    # The gfx950 static-K tiers: rows per block the dispatch
    # stages for the M bands past 16 and every N band past 576, mirroring
    # `_amd_small_m_gemv_route`'s tiers and the arm cascade in
    # `_gemv_gpu_dispatch_impl`; 0 keeps an arm below. N in (128, 576]
    # has no tier: the static-K extension to M=24 there measured slower
    # than the tiled GEMM warm and was rejected.
    comptime sk_threads = _amd_small_m_static_k_threads[a_type, a_type, K]()
    comptime sk_small_n_tier = (
        gfx950 and use_b and a_type == .bfloat16 and N == 128 and sk_threads > 0
    )
    comptime sk_wide_n_tier = (
        gfx950
        and use_b
        and a_type == .bfloat16
        and N > 576
        and (N <= 2048 or K >= 7168)
        and sk_threads > 0
    )
    comptime sk_beta_tier = (
        gfx950 and use_b and a_type == .float32 and N < 16 and sk_threads > 0
    )
    comptime tier_rpb = (
        4 if (sk_small_n_tier and 17 <= M <= 64) else 2 if (
            sk_wide_n_tier and N <= 2048 and M == 2
        ) else 4 if (sk_wide_n_tier and N <= 2048 and 3 <= M <= 4) else 8 if (
            sk_wide_n_tier and N <= 2048 and 5 <= M <= 8
        ) else 2 if (
            sk_wide_n_tier and N > 2048 and M == 2
        ) else 2 if (
            sk_beta_tier and 2 <= M <= 32
        ) else 4 if (
            sk_beta_tier and 33 <= M <= 64
        ) else 0
    )
    comptime tier_static_k = tier_rpb > 0
    comptime row_sharing = static_k_128 or tier_static_k or (
        use_b
        and (
            (
                M == 1
                and a_type == .float32
                and (
                    (N == 32 and K in (4096, 6144)) or (N == 256 and K == 6144)
                )
            )
            or (2 <= M <= 8 and a_type == .float32 and N == 32 and K == 4096)
            or (2 <= M <= 16 and N in (128, 256, 288) and K in (4096, 6144))
        )
    )
    comptime one_column_bf16 = (
        use_b
        and a_type == .bfloat16
        and c_type == .bfloat16
        and M == 8
        and N == 32
        and K == 4096
    )
    comptime tuned_m2m3 = a_type == .float32 and M in (2, 3) and (
        (N == 288 and K == 4096) or (N in (128, 256) and K == 6144)
    )
    comptime num_threads = (
        128 if row_sharing
        and a_type == .float32
        and N == 288
        and K == 4096
        and M
        in (
            3,
            4,
        ) else 128 if static_k_128 else sk_threads if tier_static_k else 256 if row_sharing else 128 if use_b else 256
    )
    comptime tile_n = 1 if row_sharing or one_column_bf16 else 2
    comptime unroll_factor = 2
    comptime tile_m = (
        tier_rpb if tier_static_k else (
            1 if (N == 32 and (row_sharing or a_type == .bfloat16))
            or M == 1
            or (M == 2 and not (tuned_m2m3 and N != 128)) else 2 if M <= 4
            or (a_type == .float32 and N == 128 and M <= 8) else 4 if M
            <= 8 else 8
        ) if row_sharing else 1
    )
    comptime k_chunk = (
        1 if row_sharing
        and (
            (a_type == .float32 and M == 8 and N == 256 and K == 6144)
            or (tuned_m2m3 and ((M == 2 and N == 128) or (M == 3 and N == 288)))
        ) else 0
    )
    print(t"== gemv_split_k router-gate shape M= {M} N= {N} K= {K}")

    comptime b_type = a_type
    comptime simd_width = simd_width_of[a_type, target=ctx.target]()
    comptime check_bounds_n = N % tile_n != 0

    var a_host = ctx.enqueue_create_host_buffer[a_type](M * K)
    var b_host = ctx.enqueue_create_host_buffer[b_type](N * K)
    var c_host = ctx.enqueue_create_host_buffer[c_type](M * N)
    var c_rowwise_host = ctx.enqueue_create_host_buffer[c_type](M * N)
    var c_expected = ctx.enqueue_create_host_buffer[c_type](M * N)

    for i in range(M * K):
        a_host[i] = random_float64(min=-1.0, max=1.0).cast[a_type]()
    randn(b_host.unsafe_ptr(), N * K)

    var a_dev = ctx.enqueue_create_buffer[a_type](M * K)
    var b_dev = ctx.enqueue_create_buffer[b_type](N * K)
    var c_dev = ctx.enqueue_create_buffer[c_type](M * N)
    var c_rowwise_dev = ctx.enqueue_create_buffer[c_type](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    var a_tt = TileTensor(a_dev, row_major(M, Idx[K])).as_imm()
    var b_tt = TileTensor(b_dev, row_major(Idx[N], Idx[K])).as_imm()
    var c_tt = TileTensor(c_dev, row_major(M, Idx[N]))

    # Check 1: the full M-row launch.
    comptime kernel_full = gemv_split_k[
        c_type,
        a_type,
        b_type,
        type_of(c_tt).LayoutType,
        type_of(a_tt).LayoutType,
        type_of(b_tt).LayoutType,
        type_of(c_tt).Engine,
        type_of(a_tt).Engine,
        type_of(b_tt).Engine,
        simd_width=simd_width,
        tile_m=tile_m,
        tile_n=tile_n,
        num_threads=num_threads,
        unroll_factor=unroll_factor,
        weight_non_temporal=not use_b,
        check_bounds_m=tile_m > 1,
        check_bounds_n=check_bounds_n,
        small_m_static_k=row_sharing,
        small_m_k_chunk=k_chunk,
    ]
    ctx.enqueue_function[kernel_full](
        c_tt,
        a_tt,
        b_tt,
        Int32(M),
        Int32(N),
        Int32(K),
        grid_dim=(ceildiv(M, tile_m), ceildiv(N, tile_n)),
        block_dim=num_threads,
    )

    comptime if (
        (row_sharing and (M == 1 or N == 32))
        or one_column_bf16
        or (K == 4096 and N in (128, 288) and M in (4, 5, 8, 16) and use_b)
        or (
            K == 4096
            and N in (128, 288)
            and M == 2
            and not use_b
            and not tuned_m2m3
        )
        or (tuned_m2m3 and use_b)
        or (N == 256 and K == 6144 and M == 8 and use_b)
        or static_k_128
        or tier_static_k
    ):
        var dispatch_dev = ctx.enqueue_create_buffer[c_type](M * N)
        var dispatch_host = ctx.enqueue_create_host_buffer[c_type](M * N)
        comptime assert c_tt.static_shape[1] == N
        comptime assert a_tt.static_shape[1] == K
        comptime assert b_tt.static_shape[0] == N
        comptime assert b_tt.static_shape[1] == K
        var dispatch_tt = TileTensor(dispatch_dev, row_major(M, Idx[N]))
        _matmul_gpu[use_tensor_core=True, transpose_b=True](
            dispatch_tt, a_tt, b_tt, ctx
        )
        ctx.enqueue_copy(dispatch_host, dispatch_dev)
        ctx.enqueue_copy(c_host, c_dev)
        ctx.synchronize()
        for i in range(M * N):
            assert_equal(dispatch_host[i], c_host[i])
        _ = dispatch_dev^

    # Check 2: M independent grid.x=1 dispatches of the same kernel. The row
    # views' layout is static, so one instance binds the kernel type for every row.
    var a_rows = TileTensor(a_dev, row_major(M, K)).as_imm()
    var c_rows = TileTensor(c_rowwise_dev, row_major(M, N))
    var a_row0_tt = a_rows.tile[1, K](0, 0)
    var c_row0_tt = c_rows.tile[1, N](0, 0)
    comptime kernel_row = gemv_split_k[
        c_type,
        a_type,
        b_type,
        type_of(c_row0_tt).LayoutType,
        type_of(a_row0_tt).LayoutType,
        type_of(b_tt).LayoutType,
        type_of(c_row0_tt).Engine,
        type_of(a_row0_tt).Engine,
        type_of(b_tt).Engine,
        simd_width=simd_width,
        tile_m=1,
        tile_n=tile_n,
        num_threads=num_threads,
        unroll_factor=unroll_factor,
        weight_non_temporal=not use_b,
        check_bounds_m=False,
        check_bounds_n=check_bounds_n,
        small_m_static_k=row_sharing,
    ]
    # Only their types were needed, to bind `kernel_row` above.
    _ = a_row0_tt
    _ = c_row0_tt
    for row in range(M):
        var a_row_tt = a_rows.tile[1, K](row, 0)
        var c_row_tt = c_rows.tile[1, N](row, 0)
        ctx.enqueue_function[kernel_row](
            c_row_tt,
            a_row_tt,
            b_tt,
            Int32(1),
            Int32(N),
            Int32(K),
            grid_dim=(1, ceildiv(N, tile_n)),
            block_dim=num_threads,
        )

    ctx.enqueue_copy(c_host, c_dev)
    ctx.enqueue_copy(c_rowwise_host, c_rowwise_dev)
    ctx.synchronize()

    # DRIV-199: keep device buffers alive past synchronize.
    _ = a_dev^
    _ = b_dev^
    _ = c_dev^
    _ = c_rowwise_dev^

    comptime f32_roundoff = 1.1920928955078125e-7
    comptime reference_error_factor = 4.0
    # The bound catches magnitude-scaled accumulation error; the dense check catches near-zero absolute error.
    for m in range(M):
        for n in range(N):
            var acc = Float64(0)
            var sum_abs = Float64(0)
            for kk in range(K):
                var product = (
                    a_host[m * K + kk].cast[.float64]()
                    * b_host[n * K + kk].cast[.float64]()
                )
                acc += product
                sum_abs += abs(product)
            c_expected[m * N + n] = acc.cast[c_type]()
            var error = abs(c_host[m * N + n].cast[.float64]() - acc)
            assert_true(
                error
                <= reference_error_factor * Float64(K) * f32_roundoff * sum_abs,
                msg=String("reference error at ", m, ",", n),
            )

    assert_almost_equal(
        c_host.unsafe_ptr(),
        c_expected.unsafe_ptr(),
        num_elements=M * N,
        atol=1e-3,
        rtol=1e-2,
    )

    # Check 2: bit-exact, since the K partition and reduction order match.
    for i in range(M * N):
        assert_equal(
            c_host[i],
            c_rowwise_host[i],
            "multirow output disagrees with the M=1 reference at element "
            + String(i),
        )

    print("PASS")


comptime WK_SENTINEL = Float32(-12345.0)
comptime WK_GUARD_ELEMS = 8


def _run_wide_k_case[M: Int, tile_m: Int](ctx: DeviceContext) raises:
    """Checks the `act_wide_k` mixed bf16-activation x fp32-weight GEMV at
    the wide router-gate shape (each lane owns 8 consecutive K, so a bf16
    row is one 16 B load per K-block and the row folds a SIMD accumulator
    once after the K loop).

    The M one-row launches use the same wide K map, so the per-row summation
    order matches and the full launch must agree bit-exactly with them; the
    fp64 oracle stays independent of the kernel's fp32 arithmetic. The signed
    activation and weight data exercise cancellation, the sentinel catches
    OOB stores, and the partial last M block exercises the clamped row loads
    and the dead-row accumulator fold.
    """
    comptime N = 896
    comptime K = 7168
    comptime num_threads = 128
    comptime tile_n = 1
    print(
        "== gemv_split_k act_wide_k M=",
        M,
        "tm=",
        tile_m,
        "N=",
        N,
        "K=",
        K,
    )

    comptime c_elems = M * N + WK_GUARD_ELEMS
    var a_host = ctx.enqueue_create_host_buffer[.bfloat16](M * K)
    var b_host = ctx.enqueue_create_host_buffer[.float32](N * K)
    var c_host = ctx.enqueue_create_host_buffer[.float32](c_elems)
    var c_rowwise_host = ctx.enqueue_create_host_buffer[.float32](c_elems)
    var c_expected = ctx.enqueue_create_host_buffer[.float32](c_elems)

    for i in range(M * K):
        a_host[i] = random_float64(min=-1.0, max=1.0).cast[.bfloat16]()
    randn(b_host.unsafe_ptr(), N * K)
    _ = TileTensor(c_host, row_major(c_elems)).fill(WK_SENTINEL)
    _ = TileTensor(c_rowwise_host, row_major(c_elems)).fill(WK_SENTINEL)
    _ = TileTensor(c_expected, row_major(c_elems)).fill(WK_SENTINEL)

    var a_dev = ctx.enqueue_create_buffer[.bfloat16](M * K)
    var b_dev = ctx.enqueue_create_buffer[.float32](N * K)
    var c_dev = ctx.enqueue_create_buffer[.float32](c_elems)
    var c_rowwise_dev = ctx.enqueue_create_buffer[.float32](c_elems)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(c_dev, c_host)
    ctx.enqueue_copy(c_rowwise_dev, c_rowwise_host)

    var a_tt = TileTensor(a_dev, row_major(M, Idx[K])).as_imm()
    var b_tt = TileTensor(b_dev, row_major(Idx[N], Idx[K])).as_imm()
    var c_tt = TileTensor(c_dev, row_major(M, Idx[N]))

    comptime kernel_full = gemv_split_k[
        .float32,
        .bfloat16,
        .float32,
        type_of(c_tt).LayoutType,
        type_of(a_tt).LayoutType,
        type_of(b_tt).LayoutType,
        type_of(c_tt).Engine,
        type_of(a_tt).Engine,
        type_of(b_tt).Engine,
        simd_width=4,
        tile_m=tile_m,
        tile_n=tile_n,
        num_threads=num_threads,
        unroll_factor=2,
        weight_non_temporal=False,
        check_bounds_m=tile_m > 1,
        check_bounds_n=False,
        small_m_static_k=True,
        small_m_k_chunk=1,
        expert_major_grid=True,
        act_wide_k=True,
    ]
    ctx.enqueue_function[kernel_full](
        c_tt,
        a_tt,
        b_tt,
        Int32(M),
        Int32(N),
        Int32(K),
        grid_dim=(ceildiv(N, tile_n), ceildiv(M, tile_m)),
        block_dim=num_threads,
    )

    # M independent one-row launches of the same wide kernel: the per-row K
    # map and summation order are tile_m-independent, so the full launch
    # must agree bit-exactly.
    var a_rows = TileTensor(a_dev, row_major(M, K)).as_imm()
    var c_rows = TileTensor(c_rowwise_dev, row_major(M, N))
    var a_row0_tt = a_rows.tile[1, K](0, 0)
    var c_row0_tt = c_rows.tile[1, N](0, 0)
    comptime kernel_row = gemv_split_k[
        .float32,
        .bfloat16,
        .float32,
        type_of(c_row0_tt).LayoutType,
        type_of(a_row0_tt).LayoutType,
        type_of(b_tt).LayoutType,
        type_of(c_row0_tt).Engine,
        type_of(a_row0_tt).Engine,
        type_of(b_tt).Engine,
        simd_width=4,
        tile_m=1,
        tile_n=tile_n,
        num_threads=num_threads,
        unroll_factor=2,
        weight_non_temporal=False,
        check_bounds_m=False,
        check_bounds_n=False,
        small_m_static_k=True,
        small_m_k_chunk=1,
        expert_major_grid=True,
        act_wide_k=True,
    ]
    # Only their types were needed, to bind `kernel_row` above.
    _ = a_row0_tt
    _ = c_row0_tt
    for row in range(M):
        var a_row_tt = a_rows.tile[1, K](row, 0)
        var c_row_tt = c_rows.tile[1, N](row, 0)
        ctx.enqueue_function[kernel_row](
            c_row_tt,
            a_row_tt,
            b_tt,
            Int32(1),
            Int32(N),
            Int32(K),
            grid_dim=(ceildiv(N, tile_n), 1),
            block_dim=num_threads,
        )

    ctx.enqueue_copy(c_host, c_dev)
    ctx.enqueue_copy(c_rowwise_host, c_rowwise_dev)
    ctx.synchronize()

    # DRIV-199: keep device buffers alive past synchronize.
    _ = a_dev^
    _ = b_dev^
    _ = c_dev^
    _ = c_rowwise_dev^

    comptime f32_roundoff = 1.1920928955078125e-7
    comptime reference_error_factor = 4.0
    var a_host_tt = TileTensor(a_host, row_major(M, Idx[K]))
    var b_host_tt = TileTensor(b_host, row_major(Idx[N], Idx[K]))
    var c_host_tt = TileTensor(c_host, row_major(M, Idx[N]))
    var c_expected_tt = TileTensor(c_expected, row_major(M, Idx[N]))
    for m in range(M):
        for n in range(N):
            var acc = Float64(0)
            var sum_abs = Float64(0)
            for kk in range(K):
                var product = (
                    a_host_tt[m, kk].cast[.float64]()
                    * b_host_tt[n, kk].cast[.float64]()
                )
                acc += product
                sum_abs += abs(product)
            c_expected_tt[m, n] = acc.cast[.float32]()
            var error = abs(c_host_tt[m, n].cast[.float64]() - acc)
            assert_true(
                error
                <= reference_error_factor * Float64(K) * f32_roundoff * sum_abs,
                msg=String("wide-k reference error at ", m, ",", n),
            )

    assert_almost_equal(
        c_host.unsafe_ptr(),
        c_expected.unsafe_ptr(),
        num_elements=c_elems,
        atol=1e-3,
        rtol=1e-2,
    )

    # Bit-exact against the one-row launches, sentinel-exact on the guard.
    for i in range(c_elems):
        assert_equal(
            c_host[i],
            c_rowwise_host[i],
            "wide-k output disagrees with the one-row reference at "
            + String(i),
        )
        if i >= M * N:
            assert_equal(
                c_host[i],
                WK_SENTINEL,
                "wide-k launch wrote a guarded output element " + String(i),
            )

    print("PASS")


def _check_bf16_route[N: Int, K: Int, c_type: DType](expected: Bool) raises:
    var route = _amd_small_m_gemv_route[c_type, .bfloat16, .bfloat16, N, K](8)
    assert_equal(route[0], expected)


# Pre-written into the padded output buffer; elements the kernel must not
# touch still hold it afterwards.
comptime EM_SENTINEL = Float32(-12345.0)
comptime EM_GUARD_ELEMS = 8


def _run_expert_major_case[
    M: Int, tile_m: Int, chunk: Int
](ctx: DeviceContext,) raises:
    """Checks the expert-major (`expert_major_grid=True`) `gemv_split_k`
    decode at the wide router-gate shape.

    The full-M launch reads block x as the weight-row tile and block y as
    the M tile (grid `(N, ceildiv(M, tile_m))`); the per-row reference
    launches run the same expert-major kernel on a single M block (grid
    `(N, 1)`), where the decode degenerates to the plain one-row walk, so
    bit-exact agreement between them proves the full launch's swapped
    block decode addressed the same (row, weight-row) pairs. The fp64
    oracle catches wrong math, and the sentinel catches OOB stores.
    """
    comptime N = 896
    comptime K = 7168
    comptime num_threads = 128
    comptime tile_n = 1
    print(
        "== gemv_split_k expert-major M=",
        M,
        "tm=",
        tile_m,
        "N=",
        N,
        "K=",
        K,
    )

    comptime c_elems = M * N + EM_GUARD_ELEMS
    var a_host = ctx.enqueue_create_host_buffer[.float32](M * K)
    var b_host = ctx.enqueue_create_host_buffer[.float32](N * K)
    var c_host = ctx.enqueue_create_host_buffer[.float32](c_elems)
    var c_rowwise_host = ctx.enqueue_create_host_buffer[.float32](c_elems)
    var c_expected = ctx.enqueue_create_host_buffer[.float32](c_elems)

    for i in range(M * K):
        a_host[i] = random_float64(min=-1.0, max=1.0).cast[.float32]()
    randn(b_host.unsafe_ptr(), N * K)
    _ = TileTensor(c_host, row_major(c_elems)).fill(EM_SENTINEL)
    _ = TileTensor(c_rowwise_host, row_major(c_elems)).fill(EM_SENTINEL)
    _ = TileTensor(c_expected, row_major(c_elems)).fill(EM_SENTINEL)

    var a_dev = ctx.enqueue_create_buffer[.float32](M * K)
    var b_dev = ctx.enqueue_create_buffer[.float32](N * K)
    var c_dev = ctx.enqueue_create_buffer[.float32](c_elems)
    var c_rowwise_dev = ctx.enqueue_create_buffer[.float32](c_elems)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(c_dev, c_host)
    ctx.enqueue_copy(c_rowwise_dev, c_rowwise_host)

    var a_tt = TileTensor(a_dev, row_major(M, Idx[K])).as_imm()
    var b_tt = TileTensor(b_dev, row_major(Idx[N], Idx[K])).as_imm()
    var c_tt = TileTensor(c_dev, row_major(M, Idx[N]))

    comptime kernel_full = gemv_split_k[
        .float32,
        .float32,
        .float32,
        type_of(c_tt).LayoutType,
        type_of(a_tt).LayoutType,
        type_of(b_tt).LayoutType,
        type_of(c_tt).Engine,
        type_of(a_tt).Engine,
        type_of(b_tt).Engine,
        simd_width=4,
        tile_m=tile_m,
        tile_n=tile_n,
        num_threads=num_threads,
        unroll_factor=2,
        weight_non_temporal=False,
        check_bounds_m=tile_m > 1,
        check_bounds_n=False,
        small_m_static_k=True,
        small_m_k_chunk=chunk,
        expert_major_grid=True,
    ]
    ctx.enqueue_function[kernel_full](
        c_tt,
        a_tt,
        b_tt,
        Int32(M),
        Int32(N),
        Int32(K),
        grid_dim=(ceildiv(N, tile_n), ceildiv(M, tile_m)),
        block_dim=num_threads,
    )

    # M independent one-row launches of the same kernel with the one-M-tile
    # grid; the per-row reduction order is identical, so the full launch
    # must agree bit-exactly. The kernel type binds from the tile views'
    # layouts (the full tensors' index types differ).
    var a_rows = TileTensor(a_dev, row_major(M, K)).as_imm()
    var c_rows = TileTensor(c_rowwise_dev, row_major(M, N))
    var a_row0_tt = a_rows.tile[1, K](0, 0)
    var c_row0_tt = c_rows.tile[1, N](0, 0)
    comptime kernel_row = gemv_split_k[
        .float32,
        .float32,
        .float32,
        type_of(c_row0_tt).LayoutType,
        type_of(a_row0_tt).LayoutType,
        type_of(b_tt).LayoutType,
        type_of(c_row0_tt).Engine,
        type_of(a_row0_tt).Engine,
        type_of(b_tt).Engine,
        simd_width=4,
        tile_m=1,
        tile_n=tile_n,
        num_threads=num_threads,
        unroll_factor=2,
        weight_non_temporal=False,
        check_bounds_m=False,
        check_bounds_n=False,
        small_m_static_k=True,
        expert_major_grid=True,
    ]
    # Only their types were needed, to bind `kernel_row` above.
    _ = a_row0_tt
    _ = c_row0_tt
    for row in range(M):
        var a_row_tt = a_rows.tile[1, K](row, 0)
        var c_row_tt = c_rows.tile[1, N](row, 0)
        ctx.enqueue_function[kernel_row](
            c_row_tt,
            a_row_tt,
            b_tt,
            Int32(1),
            Int32(N),
            Int32(K),
            grid_dim=(ceildiv(N, tile_n), 1),
            block_dim=num_threads,
        )

    ctx.enqueue_copy(c_host, c_dev)
    ctx.enqueue_copy(c_rowwise_host, c_rowwise_dev)
    ctx.synchronize()

    # DRIV-199: keep device buffers alive past synchronize.
    _ = a_dev^
    _ = b_dev^
    _ = c_dev^
    _ = c_rowwise_dev^

    comptime f32_roundoff = 1.1920928955078125e-7
    comptime reference_error_factor = 4.0
    var a_host_tt = TileTensor(a_host, row_major(M, Idx[K]))
    var b_host_tt = TileTensor(b_host, row_major(Idx[N], Idx[K]))
    var c_host_tt = TileTensor(c_host, row_major(M, Idx[N]))
    var c_expected_tt = TileTensor(c_expected, row_major(M, Idx[N]))
    for m in range(M):
        for n in range(N):
            var acc = Float64(0)
            var sum_abs = Float64(0)
            for kk in range(K):
                var product = (
                    a_host_tt[m, kk].cast[.float64]()
                    * b_host_tt[n, kk].cast[.float64]()
                )
                acc += product
                sum_abs += abs(product)
            c_expected_tt[m, n] = acc.cast[.float32]()
            var error = abs(c_host_tt[m, n].cast[.float64]() - acc)
            assert_true(
                error
                <= reference_error_factor * Float64(K) * f32_roundoff * sum_abs,
                msg=String("reference error at ", m, ",", n),
            )

    assert_almost_equal(
        c_host.unsafe_ptr(),
        c_expected.unsafe_ptr(),
        num_elements=c_elems,
        atol=1e-3,
        rtol=1e-2,
    )

    # Bit-exact: the K partition and reduction order match, so only the
    # block decode differs between the two launch orders.
    for i in range(c_elems):
        assert_equal(
            c_host[i],
            c_rowwise_host[i],
            "expert-major output disagrees with the one-row reference at "
            + String(i),
        )
        if i >= M * N:
            assert_equal(
                c_host[i],
                EM_SENTINEL,
                "expert-major launch wrote a guarded output element "
                + String(i),
            )

    print("PASS")


def main() raises:
    comptime if not default_accelerator().is_amd_gpu():
        print("SKIP: AMD GPU not available")
        return

    seed(0)

    comptime tier_admitted = _accelerator_arch() == "amdgpu:gfx950"

    comptime if (
        get_defined_bool["AUTOTUNING_MODE", False]()
        or get_defined_int["TUNE_4WAVE_KERNEL", 0]() != 0
    ):
        _check_bf16_route[576, 7168, .bfloat16](True)
        _check_bf16_route[580, 7168, .bfloat16](False)
        _check_bf16_route[2048, 7168, .bfloat16](False)
        var wide = _amd_small_m_gemv_route[
            .bfloat16, .bfloat16, .bfloat16, 4608, 7168
        ](2)
        assert_equal(wide[0], False)
        var r = _amd_small_m_gemv_route[.float32, .float32, .float32, 12, 2560](
            8
        )
        assert_equal(r[0], False)
        return

    _check_bf16_route[288, 4096, .bfloat16](True)
    _check_bf16_route[288, 4096, .float32](True)
    _check_bf16_route[384, 7168, .bfloat16](True)
    _check_bf16_route[576, 2048, .bfloat16](True)
    _check_bf16_route[128, 4096, .bfloat16](True)
    # gfx950 admits bf16 N past 576 into the static-K tiers; 2048x2048
    # reaches M=8 there, every other AMD target keeps N <= 576.
    _check_bf16_route[2048, 2048, .bfloat16](tier_admitted)
    _check_bf16_route[4096, 4096, .bfloat16](False)
    _check_bf16_route[6144, 4096, .bfloat16](False)
    _check_bf16_route[128256, 4096, .bfloat16](False)
    _check_bf16_route[576, 8192, .bfloat16](False)

    var r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 384, 7168](
        2
    )
    assert_equal(r[0], True)
    assert_equal(r[1], False)
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 576, 7168](2)
    assert_equal(r[0], True)
    assert_equal(r[1], True)
    r = _amd_small_m_gemv_route[.float32, .float32, .float32, 384, 7168](2)
    assert_equal(r[0], True)
    assert_equal(r[1], True)

    r = _amd_small_m_gemv_route[.float32, .float32, .float32, 288, 4096](8)
    assert_equal(r[0], True)
    assert_equal(r[1], True)
    r = _amd_small_m_gemv_route[.float32, .float32, .float32, 288, 4096](3)
    assert_equal(r[0], True)
    assert_equal(r[1], False)
    r = _amd_small_m_gemv_route[.float32, .float32, .float32, 288, 4096](17)
    assert_equal(r[0], False)
    r = _amd_small_m_gemv_route[.float32, .float32, .float32, 288, 4096](1)
    assert_equal(r[0], False)
    r = _amd_small_m_gemv_route[.float32, .float32, .float32, 24, 16384](8)
    assert_equal(r[0], True)
    assert_equal(r[1], False)
    r = _amd_small_m_gemv_route[.float32, .float32, .float32, 576, 2048](2)
    assert_equal(r[0], True)
    assert_equal(r[1], True)
    r = _amd_small_m_gemv_route[.float32, .float32, .float32, 580, 2048](2)
    assert_equal(r[0], False)
    r = _amd_small_m_gemv_route[.float32, .float32, .float32, 896, 7168](2)
    assert_equal(r[0], False)
    r = _amd_small_m_gemv_route[.float32, .float32, .float32, 1024, 4096](8)
    assert_equal(r[0], False)

    # The narrow fp32 band (N in [8, 16); production shape: the KDA beta
    # 12x7168 projection) is admitted on gfx950 only. Other AMD targets keep
    # the N >= 16 band and the tiled split-K + reduce launch.
    r = _amd_small_m_gemv_route[.float32, .float32, .float32, 8, 7168](8)
    assert_equal(r[0], tier_admitted)
    r = _amd_small_m_gemv_route[.float32, .float32, .float32, 12, 7168](2)
    assert_equal(r[0], tier_admitted)
    # Narrow fp32 keeps config A: the config-B floor is N >= 64 for fp32.
    assert_equal(r[1], False)
    r = _amd_small_m_gemv_route[.float32, .float32, .float32, 12, 7168](16)
    assert_equal(r[0], tier_admitted)
    r = _amd_small_m_gemv_route[.float32, .float32, .float32, 12, 7168](1)
    assert_equal(r[0], False)  # the m == 1 GEMV arm owns M=1
    r = _amd_small_m_gemv_route[.float32, .float32, .float32, 12, 7168](17)
    # The beta tier reaches M=64 through the static-K stage on gfx950;
    # other AMD targets keep the tiled split-K arm above M=16.
    assert_equal(r[0], tier_admitted)
    r = _amd_small_m_gemv_route[.float32, .float32, .float32, 7, 7168](8)
    assert_equal(r[0], False)  # below the N floor
    r = _amd_small_m_gemv_route[.float32, .float32, .float32, 12, 1024](8)
    assert_equal(r[0], False)  # below the K floor
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 12, 7168](8)
    assert_equal(r[0], False)  # bf16 keeps N >= 16

    # The static-K stage needs K % (simd_width * num_threads) == 0. bf16
    # K % 2048 == 0 keeps the 256-thread instantiation (the 576x2048 and
    # 128x4096 cells); on gfx950, bf16 K = 7 x 1024 (the KDA f_a 128x7168
    # and MLA kv_a 576x7168 projections) gets the 128-thread one instead.
    # Other targets and dtypes keep the 256-thread tiling, and K % 1024 != 0
    # takes no static-K stage at all.
    assert_equal(
        _amd_small_m_static_k_threads[.bfloat16, .bfloat16, 7168](),
        128 if tier_admitted else 0,
    )
    assert_equal(
        _amd_small_m_static_k_threads[.bfloat16, .bfloat16, 5120](),
        128 if tier_admitted else 0,
    )
    assert_equal(
        _amd_small_m_static_k_threads[.bfloat16, .bfloat16, 2048](), 256
    )
    assert_equal(
        _amd_small_m_static_k_threads[.bfloat16, .bfloat16, 4096](), 256
    )
    assert_equal(_amd_small_m_static_k_threads[.bfloat16, .bfloat16, 2560](), 0)
    assert_equal(_amd_small_m_static_k_threads[.float32, .float32, 7168](), 256)
    # 128x7168 and 576x7168 at M=4 sit in the config-B route with a legal
    # 128-thread static-K tiling, so the M-band arms stage them through the
    # static-K instantiation instead of the cached-load config.
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 128, 7168](4)
    assert_equal(r[0], True)
    assert_equal(r[1], True)
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 576, 7168](4)
    assert_equal(r[0], True)
    assert_equal(r[1], True)

    # gfx950 static-K tier band edges: first and last M of
    # every tier plus one M outside, N-tier boundaries, and the K-legality
    # gate. Other AMD targets keep every one of these points on the tiled
    # GEMM (max_m=16 at N <= 576, nothing admitted above).
    # bf16 N = 128 reaches M=64 (the 128x7168 KDA f_a projection); it is
    # the only measured N in the tier, so every other N keeps M <= 16.
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 128, 7168](17)
    assert_equal(r[0], tier_admitted)
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 128, 7168](24)
    assert_equal(r[0], tier_admitted)
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 128, 7168](64)
    assert_equal(r[0], tier_admitted)
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 128, 7168](65)
    assert_equal(r[0], False)
    # N = 64 was never measured: it keeps the M <= 16 band every N below
    # 128 keeps, at every M past 16.
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 64, 7168](17)
    assert_equal(r[0], False)
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 64, 7168](24)
    assert_equal(r[0], False)
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 64, 7168](64)
    assert_equal(r[0], False)
    # N below the static-K CTA floor keeps M <= 16.
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 32, 7168](17)
    assert_equal(r[0], False)
    # bf16 N in (128, 576] keeps M=16: the static-K extension past 16 there
    # measured slower than the tiled GEMM warm, so every M past the
    # existing M=16 band keeps the tiled GEMM on every AMD target.
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 576, 7168](17)
    assert_equal(r[0], False)
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 576, 7168](24)
    assert_equal(r[0], False)
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 576, 7168](25)
    assert_equal(r[0], False)
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 132, 7168](24)
    assert_equal(r[0], False)
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 132, 7168](25)
    assert_equal(r[0], False)
    # bf16 N in (576, 2048] reaches M=8 (the 1536x7168 KDA out-gate and the
    # 2048x4096 GLM KDA projection).
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 1536, 7168](2)
    assert_equal(r[0], tier_admitted)
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 1536, 7168](8)
    assert_equal(r[0], tier_admitted)
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 1536, 7168](9)
    assert_equal(r[0], False)
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 2048, 4096](8)
    assert_equal(r[0], tier_admitted)
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 2048, 4096](9)
    assert_equal(r[0], False)
    # bf16 N past 2048 admits M=2 only, and only at the measured K
    # (K >= 7168: the 3584x7168 shared-expert gate/up and the 4608x7168
    # KDA merged qkv). GLM's 4096x2048 KDA o_proj and every other in-band
    # K below that keeps main's tiled GEMM arm.
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 3584, 7168](2)
    assert_equal(r[0], tier_admitted)
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 3584, 7168](3)
    assert_equal(r[0], False)
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 4608, 7168](2)
    assert_equal(r[0], tier_admitted)
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 4608, 7168](3)
    assert_equal(r[0], False)
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 4096, 2048](2)
    assert_equal(r[0], False)
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 4608, 5120](2)
    assert_equal(r[0], False)
    # N past the widened band keeps the tiled GEMM.
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 4612, 7168](2)
    assert_equal(r[0], False)
    # The N-tier boundaries: the small-N tier is N=128 only, the wide
    # tier takes over above N=576 (same M band as N=1536), and N past 2048
    # admits M=2 only at K >= 7168.
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 64, 7168](2)
    assert_equal(r[0], True)
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 64, 7168](16)
    assert_equal(r[0], True)
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 580, 7168](8)
    assert_equal(r[0], tier_admitted)
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 580, 7168](9)
    assert_equal(r[0], False)
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 2049, 7168](2)
    assert_equal(r[0], tier_admitted)
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 2049, 4096](2)
    assert_equal(r[0], False)
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 2049, 4096](3)
    assert_equal(r[0], False)
    # A K the static-K tiling cannot cover keeps the tiled GEMM at every
    # tier point (N > 576 and M > 16 have static-K-only arms); the M <= 16
    # band at N <= 576 keeps its cached-load arms.
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 1536, 2560](2)
    assert_equal(r[0], False)
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 128, 2816](17)
    assert_equal(r[0], False)
    r = _amd_small_m_gemv_route[.bfloat16, .bfloat16, .bfloat16, 128, 2816](8)
    assert_equal(r[0], True)
    # The fp32 beta tier: M 2..32 at rpb2, 33..64 at rpb4, one past the
    # band and the K-legality gate.
    r = _amd_small_m_gemv_route[.float32, .float32, .float32, 12, 7168](32)
    assert_equal(r[0], tier_admitted)
    r = _amd_small_m_gemv_route[.float32, .float32, .float32, 12, 7168](33)
    assert_equal(r[0], tier_admitted)
    r = _amd_small_m_gemv_route[.float32, .float32, .float32, 12, 7168](64)
    assert_equal(r[0], tier_admitted)
    r = _amd_small_m_gemv_route[.float32, .float32, .float32, 12, 7168](65)
    assert_equal(r[0], False)
    r = _amd_small_m_gemv_route[.float32, .float32, .float32, 12, 2560](17)
    assert_equal(r[0], False)
    r = _amd_small_m_gemv_route[.float32, .float32, .float32, 12, 2560](8)
    assert_equal(r[0], tier_admitted)

    with DeviceContext() as ctx:
        # The N=896 cells check direct GEMV even though dispatch uses tiled GEMM.
        _run_case[1, 896, 7168, True](ctx)
        # Expert-major grid: block x walks the weight rows. One full-block
        # and one partial-last-block M per band tile_m (2, 4, 8); each is
        # checked against the fp64 oracle, bit-exact one-row launches, and
        # the output sentinel.
        _run_expert_major_case[3, 2, 2](ctx)
        _run_expert_major_case[6, 2, 2](ctx)
        _run_expert_major_case[7, 4, 2](ctx)
        _run_expert_major_case[12, 4, 2](ctx)
        _run_expert_major_case[13, 8, 1](ctx)
        _run_expert_major_case[16, 8, 1](ctx)
        # act_wide_k: 8 consecutive K per lane, SIMD accumulator, partial
        # last M block at M=3 (tm2) and M=5 (tm4).
        _run_wide_k_case[2, 2](ctx)
        _run_wide_k_case[3, 2](ctx)
        _run_wide_k_case[5, 4](ctx)
        _run_wide_k_case[16, 8](ctx)
        _run_case[2, 896, 7168, True](ctx)
        _run_case[3, 896, 7168, True](ctx)
        _run_case[5, 896, 7168, True](ctx)
        _run_case[8, 896, 7168, True](ctx)
        _run_case[9, 896, 7168, True](ctx)
        _run_case[13, 896, 7168, True](ctx)
        _run_case[16, 896, 7168, True](ctx)
        _run_case[2, 288, 4096, False](ctx)
        _run_case[8, 288, 4096, False](ctx)
        _run_case[2, 288, 4096, True](ctx)
        _run_case[3, 288, 4096, True](ctx)
        _run_case[8, 288, 4096, True](ctx)
        _run_case[4, 288, 4096, True](ctx)
        _run_case[5, 288, 4096, True](ctx)
        _run_case[16, 288, 4096, True](ctx)
        _run_case[2, 256, 6144, False](ctx)
        _run_case[1, 256, 6144, True](ctx)
        _run_case[8, 256, 6144, False](ctx)
        _run_case[2, 256, 6144, True](ctx)
        _run_case[3, 256, 6144, True](ctx)
        _run_case[8, 256, 6144, True](ctx)
        _run_case[2, 128, 6144, True](ctx)
        _run_case[3, 128, 6144, True](ctx)
        _run_case[8, 128, 6144, True](ctx)
        _run_case[2, 24, 16384, False](ctx)
        _run_case[8, 24, 16384, False](ctx)
        _run_case[2, 24, 16384, True](ctx)
        _run_case[8, 24, 16384, True](ctx)
        _run_case[2, 32, 4096, False](ctx)
        _run_case[1, 32, 4096, True](ctx)
        _run_case[1, 32, 6144, True](ctx)
        _run_case[8, 32, 4096, False](ctx)
        _run_case[2, 32, 4096, True](ctx)
        _run_case[8, 32, 4096, True](ctx)
        _run_case[8, 32, 4096, True, .bfloat16, .bfloat16](ctx)
        _run_case[2, 288, 4096, False, .bfloat16, .bfloat16](ctx)
        _run_case[8, 288, 4096, True, .bfloat16, .bfloat16](ctx)
        _run_case[2, 128, 4096, False, .bfloat16, .bfloat16](ctx)
        _run_case[8, 128, 4096, True, .bfloat16, .bfloat16](ctx)
        _run_case[2, 288, 4096, False, .bfloat16, .float32](ctx)
        _run_case[8, 288, 4096, True, .bfloat16, .float32](ctx)
        # The gfx950 bf16 128-thread static-K cells (K = 7 x 1024): the KDA
        # f_a 128x7168 and MLA kv_a 576x7168 projections, at the M-band
        # boundaries -- rpb 1 at M=2, 2 at M=4, 4 at M=8, 8 at M in 13..16
        # (the N=576 column), with a bf16-to-fp32 output cell at M=4.
        _run_case[2, 576, 7168, True, .bfloat16, .bfloat16](ctx)
        _run_case[4, 128, 7168, True, .bfloat16, .bfloat16](ctx)
        _run_case[4, 576, 7168, True, .bfloat16, .bfloat16](ctx)
        _run_case[4, 128, 7168, True, .bfloat16, .float32](ctx)
        _run_case[8, 128, 7168, True, .bfloat16, .bfloat16](ctx)
        _run_case[8, 576, 7168, True, .bfloat16, .bfloat16](ctx)
        _run_case[13, 576, 7168, True, .bfloat16, .bfloat16](ctx)
        _run_case[16, 576, 7168, True, .bfloat16, .bfloat16](ctx)
        # The gfx950 static-K tier cells (band widening): the M
        # bands past 16 and the N bands past 576, including the partial
        # last row blocks, both thread instantiations, and the fp32 beta
        # tier at its rpb switch. Each is checked against the fp64 oracle,
        # bit-exact against per-row launches, and against the production
        # dispatch output.
        _run_case[2, 1536, 7168, True, .bfloat16, .bfloat16](ctx)
        _run_case[3, 1536, 7168, True, .bfloat16, .bfloat16](ctx)
        _run_case[4, 1536, 7168, True, .bfloat16, .bfloat16](ctx)
        _run_case[5, 1536, 7168, True, .bfloat16, .bfloat16](ctx)
        _run_case[7, 1536, 7168, True, .bfloat16, .bfloat16](ctx)
        _run_case[8, 1536, 7168, True, .bfloat16, .bfloat16](ctx)
        _run_case[4, 1536, 7168, True, .bfloat16, .float32](ctx)
        _run_case[2, 2048, 4096, True, .bfloat16, .bfloat16](ctx)
        _run_case[4, 2048, 4096, True, .bfloat16, .bfloat16](ctx)
        _run_case[8, 2048, 4096, True, .bfloat16, .bfloat16](ctx)
        _run_case[2, 3584, 7168, True, .bfloat16, .bfloat16](ctx)
        _run_case[2, 4608, 7168, True, .bfloat16, .bfloat16](ctx)
        _run_case[17, 128, 7168, True, .bfloat16, .bfloat16](ctx)
        _run_case[23, 128, 7168, True, .bfloat16, .bfloat16](ctx)
        _run_case[32, 128, 7168, True, .bfloat16, .bfloat16](ctx)
        _run_case[33, 128, 7168, True, .bfloat16, .bfloat16](ctx)
        _run_case[63, 128, 7168, True, .bfloat16, .bfloat16](ctx)
        _run_case[64, 128, 7168, True, .bfloat16, .bfloat16](ctx)
        _run_case[32, 128, 4096, True, .bfloat16, .bfloat16](ctx)
        _run_case[2, 12, 7168, True, .float32, .float32](ctx)
        _run_case[17, 12, 7168, True, .float32, .float32](ctx)
        _run_case[32, 12, 7168, True, .float32, .float32](ctx)
        _run_case[33, 12, 7168, True, .float32, .float32](ctx)
        _run_case[64, 12, 7168, True, .float32, .float32](ctx)

    print("ALL TESTS PASSED")
