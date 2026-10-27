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
from std.sys import simd_width_of
from std.testing import assert_equal, assert_true

from internal_utils import assert_almost_equal

from layout import Idx, TileTensor, row_major
from linalg.gemv import _amd_small_m_gemv_route, gemv_split_k
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
    comptime row_sharing = use_b and (
        (
            M == 1
            and a_type == .float32
            and ((N == 32 and K in (4096, 6144)) or (N == 256 and K == 6144))
        )
        or (2 <= M <= 8 and a_type == .float32 and N == 32 and K == 4096)
        or (2 <= M <= 16 and N in (128, 256, 288) and K in (4096, 6144))
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
        and M in (3, 4) else 256 if row_sharing else 128 if use_b else 256
    )
    comptime tile_n = 1 if row_sharing or one_column_bf16 else 2
    comptime unroll_factor = 2
    comptime tile_m = (
        1 if (N == 32 and (row_sharing or a_type == .bfloat16))
        or M == 1
        or (M == 2 and not (tuned_m2m3 and N != 128)) else 2 if M <= 4
        or (a_type == .float32 and N == 128 and M <= 8) else 4 if M <= 8 else 8
    ) if row_sharing else 1
    comptime k_chunk = (
        1 if row_sharing
        and (
            (a_type == .float32 and M == 8 and N == 256 and K == 6144)
            or (tuned_m2m3 and ((M == 2 and N == 128) or (M == 3 and N == 288)))
        ) else 0
    )
    print(
        "== gemv_split_k router-gate shape M=",
        M,
        "N=",
        N,
        "K=",
        K,
    )

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


def _check_bf16_route[N: Int, K: Int, c_type: DType](expected: Bool) raises:
    var route = _amd_small_m_gemv_route[c_type, .bfloat16, .bfloat16, N, K](8)
    assert_equal(route[0], expected)


def main() raises:
    comptime if not default_accelerator().is_amd_gpu():
        print("SKIP: AMD GPU not available")
        return

    seed(0)

    _check_bf16_route[288, 4096, .bfloat16](True)
    _check_bf16_route[288, 4096, .float32](True)
    _check_bf16_route[384, 7168, .bfloat16](True)
    _check_bf16_route[576, 2048, .bfloat16](True)
    _check_bf16_route[128, 4096, .bfloat16](True)
    _check_bf16_route[2048, 2048, .bfloat16](False)
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

    with DeviceContext() as ctx:
        # The N=896 cells check direct GEMV even though dispatch uses tiled GEMM.
        _run_case[1, 896, 7168, True](ctx)
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

    print("ALL TESTS PASSED")
