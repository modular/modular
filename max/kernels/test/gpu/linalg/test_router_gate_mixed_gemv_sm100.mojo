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

Shapes mirror both gates the `moe.router.gate.mixed.gemv` custom op serves:
MiniMax-M3's (`N = 128`, `K = 6144`) and MiMo-V2.6-Flash's under four-way
tensor parallelism (`N = 256`, `K = 4096`), each with runtime `M`. An odd
`N` exercises the `check_bounds_n` guard.

Every case also pins the launcher's own band table: the same call is
replayed through two bands the launcher takes at few row counts -- the
narrowest `(1, 1)` one, and the `(8, 2)` one it picked above 32 rows before the
per-shape rows existed -- and every output must match bit for bit. Bands
differ only in the output tile a block covers, so a shape that starts taking a
different band must produce the same logits -- which is what keeps the
router's expert selection, and the scores its combine weights come from,
unchanged.
"""

from std.math import ceildiv
from std.random import randn, random_float64, seed
from std.testing import assert_equal, assert_false, assert_true

from internal_utils import assert_almost_equal
from layout import Idx, TileTensor, row_major
from linalg.gemv import (
    _sm100_router_gate_band,
    gemv_split_k,
    router_gate_mixed_gemv,
    router_gate_use_mixed_gemv,
)
from linalg.matmul import matmul
from max.gpu.host import (
    DeviceBuffer,
    DeviceContext,
    HostBuffer,
    default_accelerator,
)
from max.gpu.primitives.grid_controls import PDLLevel

comptime ROUTER_K = 6144
comptime MIMO_K = 4096

# Pre-written into the output. Elements the launch must not touch (the
# trailing guard, and every element when `m == 0`) still hold it afterwards.
comptime SENTINEL = Float32(-12345.0)
comptime GUARD_ELEMS = 8


def _assert_band_identity[
    static_N: Int, static_K: Int, ref_tile_m: Int, ref_tile_n: Int
](
    ctx: DeviceContext,
    m: Int,
    got: HostBuffer[DType.float32],
    a_dev: DeviceBuffer[DType.bfloat16],
    b_dev: DeviceBuffer[DType.float32],
) raises:
    """Runs one `gemv_split_k` band and asserts the launcher matches it.

    The band keeps the launcher's 256-thread, `simd_width=4` K split and
    changes only the output tile a block covers, so its logits must be
    bit-identical to the launcher's. It runs with the row guard on, which is
    valid at every row count. `unroll_factor` is held at 2 for both references,
    the K unroll the launcher's wide bands use, so the `(1, 1)` reference pins
    the split's reduction order rather than the launcher's own `(1, 1)` band,
    which unrolls once.

    Parameters:
        static_N: Output width.
        static_K: Contraction dim.
        ref_tile_m: Rows per block of the reference band.
        ref_tile_n: Columns per block of the reference band.

    Args:
        ctx: The device context.
        m: Runtime row count.
        got: The launcher's logits, `[m, static_N]`.
        a_dev: The bf16 activation, `[m, static_K]`.
        b_dev: The fp32 weight, `[static_N, static_K]`.
    """
    var c_dev = ctx.enqueue_create_buffer[.float32](m * static_N)
    var c_tt = TileTensor(c_dev.unsafe_ptr(), row_major(m, Idx[static_N]))
    var a_tt = TileTensor(a_dev.unsafe_ptr(), row_major(m, Idx[static_K]))
    var b_tt = TileTensor(b_dev.unsafe_ptr(), row_major[static_N, static_K]())
    comptime kernel = gemv_split_k[
        DType.float32,
        DType.bfloat16,
        DType.float32,
        type_of(c_tt).LayoutType,
        type_of(a_tt).LayoutType,
        type_of(b_tt).LayoutType,
        type_of(c_tt).Engine,
        type_of(a_tt).Engine,
        type_of(b_tt).Engine,
        simd_width=4,
        tile_m=ref_tile_m,
        tile_n=ref_tile_n,
        num_threads=256,
        unroll_factor=2,
        weight_non_temporal=False,
        check_bounds_m=True,
        check_bounds_n=static_N % ref_tile_n != 0,
        pdl_level=PDLLevel.OFF,
    ]
    ctx.enqueue_function[kernel](
        c_tt,
        a_tt.as_imm(),
        b_tt.as_imm(),
        Int32(m),
        Int32(static_N),
        Int32(static_K),
        grid_dim=(ceildiv(m, ref_tile_m), ceildiv(static_N, ref_tile_n)),
        block_dim=256,
    )
    var c_host = ctx.enqueue_create_host_buffer[.float32](m * static_N)
    ctx.enqueue_copy(c_host, c_dev)
    ctx.synchronize()

    # DRIV-199: keep device buffers alive past synchronize.
    _ = c_dev^

    for i in range(m * static_N):
        assert_equal(
            got[i],
            c_host[i],
            "not bit-identical to the ("
            + String(ref_tile_m)
            + ", "
            + String(ref_tile_n)
            + ") band at element "
            + String(i),
        )


def _run_case[
    static_N: Int, static_K: Int = ROUTER_K
](ctx: DeviceContext, m: Int) raises:
    """Runs the launcher at `[m, static_K] x [static_N, static_K]^T`.

    Parameters:
        static_N: Output width.
        static_K: Contraction dim.

    Args:
        ctx: The device context.
        m: Runtime row count. `m == 0` exercises the empty-launch guard.
    """
    comptime K = static_K
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

    router_gate_mixed_gemv[.float32, static_N](
        c_tt, a_tt.as_imm(), b_tt.as_imm(), m, static_N, K, ctx
    )

    # The standard path the fused launch replaces: fp32 activation, ordinary
    # matmul dispatch. Skipped at m == 0, which the op never reaches it with.
    if m > 0:
        var c_ref_tt = TileTensor(
            c_ref_dev.unsafe_ptr(), row_major(m, Idx[static_N])
        )
        var a_f32_tt = TileTensor(a_f32_dev.unsafe_ptr(), row_major(m, Idx[K]))
        matmul[transpose_b=True](
            c_ref_tt, a_f32_tt.as_imm(), b_tt.as_imm(), ctx
        )

    ctx.enqueue_copy(c_host, c_dev)
    ctx.enqueue_copy(c_ref_host, c_ref_dev)
    ctx.synchronize()

    if m > 0:
        # Bands the launcher takes at few row counts: the narrowest, and the
        # one it picked above 32 rows before the per-shape rows existed.
        _assert_band_identity[static_N, static_K, 1, 1](
            ctx, m, c_host, a_dev, b_dev
        )
        _assert_band_identity[static_N, static_K, 8, 2](
            ctx, m, c_host, a_dev, b_dev
        )

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


def test_sm100_router_gate_band() raises:
    """Pins the band every row count takes, and each shape's fallback."""
    # MiniMax-M3's `[128, 6144]` gate, and every other shape: the bands this
    # launcher shipped with, at every row count.
    assert_equal(_sm100_router_gate_band[False](1), 0)  # (1, 1)
    assert_equal(_sm100_router_gate_band[False](8), 0)
    assert_equal(_sm100_router_gate_band[False](12), 1)  # (4, 1)
    assert_equal(_sm100_router_gate_band[False](24), 1)
    assert_equal(_sm100_router_gate_band[False](28), 2)  # (4, 2)
    assert_equal(_sm100_router_gate_band[False](32), 2)
    assert_equal(_sm100_router_gate_band[False](64), 3)  # (8, 2)
    assert_equal(_sm100_router_gate_band[False](128), 3)
    # A count no row's `tile_m` divides keeps the shipped band and takes the
    # row guard: `(4, 1)` at 9-23, `(4, 2)` at 25-31, `(8, 2)` above.
    assert_equal(_sm100_router_gate_band[False](9), 1)
    assert_equal(_sm100_router_gate_band[False](17), 1)
    assert_equal(_sm100_router_gate_band[False](25), 2)
    assert_equal(_sm100_router_gate_band[False](33), 3)
    assert_equal(_sm100_router_gate_band[False](127), 3)
    # MiMo-V2.6-Flash's `[256, 4096]` gate.
    assert_equal(_sm100_router_gate_band[True](1), 0)  # (1, 1)
    assert_equal(_sm100_router_gate_band[True](4), 1)  # (4, 1)
    assert_equal(_sm100_router_gate_band[True](16), 1)
    assert_equal(_sm100_router_gate_band[True](32), 3)  # (8, 2)
    assert_equal(_sm100_router_gate_band[True](64), 5)  # (16, 4)
    assert_equal(_sm100_router_gate_band[True](72), 4)  # (8, 4)
    assert_equal(_sm100_router_gate_band[True](128), 4)
    assert_equal(_sm100_router_gate_band[True](12), 1)
    # (1, 1) divides every count it covers.
    assert_equal(_sm100_router_gate_band[True](2), 0)
    assert_equal(_sm100_router_gate_band[True](3), 0)
    # A count no row divides takes the first guarded row that covers it:
    # (4, 1) to 16 rows, the (8, 4) fallback above.
    assert_equal(_sm100_router_gate_band[True](6), 1)
    assert_equal(_sm100_router_gate_band[True](9), 1)
    assert_equal(_sm100_router_gate_band[True](20), 4)
    assert_equal(_sm100_router_gate_band[True](65), 4)


def main() raises:
    comptime if not default_accelerator().is_nvidia_gpu():
        print("SKIP: NVIDIA GPU not available")
        return

    seed(0)
    test_router_gate_use_mixed_gemv()
    test_sm100_router_gate_band()

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
        # MiMo-V2.6-Flash's gate shape, at its own band edges: the c1 decode
        # step's 8 rows, c2's 16, c4's 32, c8's 64 and the window edge 128.
        _run_case[256, MIMO_K](ctx, 1)
        _run_case[256, MIMO_K](ctx, 2)
        _run_case[256, MIMO_K](ctx, 3)
        _run_case[256, MIMO_K](ctx, 8)
        _run_case[256, MIMO_K](ctx, 9)
        _run_case[256, MIMO_K](ctx, 16)
        _run_case[256, MIMO_K](ctx, 32)
        _run_case[256, MIMO_K](ctx, 33)
        _run_case[256, MIMO_K](ctx, 64)
        _run_case[256, MIMO_K](ctx, 65)
        _run_case[256, MIMO_K](ctx, 128)
        # Empty-launch guard: no launch, no fault, output untouched.
        _run_case[128](ctx, 0)

    print("ALL TESTS PASSED")
