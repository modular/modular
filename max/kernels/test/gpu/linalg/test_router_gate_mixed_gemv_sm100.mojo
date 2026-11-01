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
"""SM100 coverage for the `router_gate_mixed_gemv` launcher.

The launcher takes the bf16 router activation directly and runs `gemv_split_k`
with a per-`M` output tiling. Every SM100 tiling keeps the 256-thread,
`simd_width=4` K split of the standard fp32 router dispatch, so for `M <= 64`,
where that dispatch also runs `gemv_split_k`, the fused result must equal
"cast to fp32, then `matmul`" bit for bit. Above 64 the standard dispatch
switches to TF32 tensor cores, so those rows are checked against an fp64 host
reference instead.

Shapes mirror MiniMax-M3's router gate (`N = 128`, `K = 6144`) with runtime
`M`, as the `mo.router.gate.mixed.gemv` custom op calls it. An odd `N`
exercises the `check_bounds_n` guard of the `tile_n = 2` buckets.
"""

from std.random import randn, random_float64, seed
from std.testing import assert_equal, assert_false, assert_true

from internal_utils import assert_almost_equal
from layout import Idx, TileTensor, row_major
from linalg.gemv import router_gate_mixed_gemv, router_gate_use_mixed_gemv
from linalg.matmul import matmul
from max.gpu.host import DeviceContext, default_accelerator

comptime ROUTER_K = 6144

# Pre-written into the output. Elements the launch must not touch (the
# trailing guard, and every element when `m == 0`) still hold it afterwards.
comptime SENTINEL = Float32(-12345.0)
comptime GUARD_ELEMS = 8


def _run_case[static_N: Int](ctx: DeviceContext, m: Int) raises:
    """Runs the launcher at `[m, K] x [static_N, K]^T` and checks the output.

    Parameters:
        static_N: Output width.

    Args:
        ctx: The device context.
        m: Runtime row count. `m == 0` exercises the empty-launch guard.
    """
    comptime K = ROUTER_K
    print("== router_gate_mixed_gemv sm100 m=", m, "N=", static_N)

    var rows = m if m > 0 else 1
    var c_elems = rows * static_N + GUARD_ELEMS

    var a_host = ctx.enqueue_create_host_buffer[.bfloat16](rows * K)
    var a_f32_host = ctx.enqueue_create_host_buffer[.float32](rows * K)
    var b_host = ctx.enqueue_create_host_buffer[.float32](static_N * K)
    var c_host = ctx.enqueue_create_host_buffer[.float32](c_elems)
    var c_ref_host = ctx.enqueue_create_host_buffer[.float32](c_elems)
    var c_expected = ctx.enqueue_create_host_buffer[.float32](c_elems)

    for i in range(rows * K):
        a_host[i] = random_float64(min=-1.0, max=1.0).cast[.bfloat16]()
        a_f32_host[i] = a_host[i].cast[.float32]()
    randn(b_host.unsafe_ptr(), static_N * K)
    for i in range(c_elems):
        c_host[i] = SENTINEL
        c_ref_host[i] = SENTINEL

    var a_dev = ctx.enqueue_create_buffer[.bfloat16](rows * K)
    var a_f32_dev = ctx.enqueue_create_buffer[.float32](rows * K)
    var b_dev = ctx.enqueue_create_buffer[.float32](static_N * K)
    var c_dev = ctx.enqueue_create_buffer[.float32](c_elems)
    var c_ref_dev = ctx.enqueue_create_buffer[.float32](c_elems)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(a_f32_dev, a_f32_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(c_dev, c_host)
    ctx.enqueue_copy(c_ref_dev, c_ref_host)

    var c_tt = TileTensor(c_dev.unsafe_ptr(), row_major(m, Idx[static_N]))
    var a_tt = TileTensor(a_dev.unsafe_ptr(), row_major(m, Idx[K]))
    var b_tt = TileTensor(b_dev.unsafe_ptr(), row_major[static_N, K]())

    router_gate_mixed_gemv[static_N](
        c_tt, a_tt.as_imm(), b_tt.as_imm(), m, static_N, K, ctx
    )

    # The standard path the fused launch replaces: fp32 activation, ordinary
    # matmul dispatch. Skipped at m == 0, which the op never reaches it with.
    if m > 0:
        var c_ref_tt = TileTensor(
            c_ref_dev.unsafe_ptr(), row_major(m, Idx[static_N])
        )
        var a_f32_tt = TileTensor(a_f32_dev.unsafe_ptr(), row_major(m, Idx[K]))
        matmul[transpose_b=True, target="gpu"](
            c_ref_tt, a_f32_tt.as_imm(), b_tt.as_imm(), ctx
        )

    ctx.enqueue_copy(c_host, c_dev)
    ctx.enqueue_copy(c_ref_host, c_ref_dev)
    ctx.synchronize()

    # DRIV-199: keep device buffers alive past synchronize.
    _ = a_dev^
    _ = a_f32_dev^
    _ = b_dev^
    _ = c_dev^
    _ = c_ref_dev^

    for i in range(c_elems):
        c_expected[i] = SENTINEL
    for row in range(m):
        for col in range(static_N):
            var acc = Float64(0)
            for kk in range(K):
                acc += (
                    a_host[row * K + kk].cast[.float64]()
                    * b_host[col * K + kk].cast[.float64]()
                )
            c_expected[row * static_N + col] = acc.cast[.float32]()

    # fp32 accumulation over K = 6144 terms of magnitude ~1 against fp64.
    assert_almost_equal(
        c_host.unsafe_ptr(),
        c_expected.unsafe_ptr(),
        num_elements=c_elems,
        atol=1e-2,
        rtol=1e-4,
    )

    for row in range(m):
        var got_arg = 0
        var expected_arg = 0
        for col in range(1, static_N):
            if c_host[row * static_N + col] > c_host[row * static_N + got_arg]:
                got_arg = col
            if (
                c_expected[row * static_N + col]
                > c_expected[row * static_N + expected_arg]
            ):
                expected_arg = col
        assert_equal(
            got_arg,
            expected_arg,
            "top-1 routing mismatch on row " + String(row),
        )

    # Up to M = 64 the standard dispatch is the same gemv_split_k reduction.
    if m > 0 and m <= 64:
        for i in range(c_elems):
            assert_equal(
                c_host[i],
                c_ref_host[i],
                "not bit-identical to cast + matmul at element " + String(i),
            )

    print("PASS")


def test_router_gate_use_mixed_gemv() raises:
    """Pins the M windows: 64 by default (gfx950) and 128 on SM100."""
    assert_false(router_gate_use_mixed_gemv(0, 128))
    assert_true(router_gate_use_mixed_gemv(1, 128))
    assert_true(router_gate_use_mixed_gemv(128, 128))
    assert_false(router_gate_use_mixed_gemv(129, 128))
    assert_true(router_gate_use_mixed_gemv(64))
    assert_false(router_gate_use_mixed_gemv(65))


def main() raises:
    comptime if not default_accelerator().is_nvidia_gpu():
        print("SKIP: NVIDIA GPU not available")
        return

    seed(0)
    test_router_gate_use_mixed_gemv()

    with DeviceContext() as ctx:
        # One case per tiling bucket edge (m <= 8, 24, 32, else), the
        # non-multiples of tile_m that exercise `check_bounds_m`, the decode
        # widths (8 positions per row), and the window edge at 128.
        _run_case[128](ctx, 1)
        _run_case[128](ctx, 8)
        _run_case[128](ctx, 9)
        _run_case[128](ctx, 17)
        _run_case[128](ctx, 24)
        _run_case[128](ctx, 25)
        _run_case[128](ctx, 32)
        _run_case[128](ctx, 33)
        _run_case[128](ctx, 36)
        _run_case[128](ctx, 64)
        _run_case[128](ctx, 96)
        _run_case[128](ctx, 127)
        _run_case[128](ctx, 128)
        # Odd N exercises check_bounds_n in the tile_n = 2 buckets.
        _run_case[127](ctx, 2)
        _run_case[127](ctx, 36)
        # Empty-launch guard: no launch, no fault, output untouched.
        _run_case[128](ctx, 0)

    print("ALL TESTS PASSED")
