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
"""KERN-3352: CI coverage for the `router_gate_mixed_gemv` launcher.

`router_gate_mixed_gemv` is a host-side launcher generic over `static_N` plus
six layout/storage parameters, so its body is only type-checked once something
specializes it. Its sole production caller is the `mo.router.gate.mixed.gemv`
custom op, reachable only from a `manual`-tagged target -- which is how #91194
("Treat Int and UInt as non-device passable") swept the `Int32(...)` casts
through `gemv.mojo` while silently skipping this launch site and breaking
MiniMax-M3 on main. These tests exist to instantiate the launcher body itself
from a cheap, always-run target; calling the inner `gemv_split_k` instead would
not have caught that break (`test_gemv.mojo` already does exactly that).

The instantiation mirrors the production call site
(`graph_compiler/builtin_kernels/linalg.mojo`): runtime `M`, compile-time
`N`/`K`. A static `M` would specialize a shape production never compiles.

The `[896, 7168]` (Kimi K3) cells cover every cold-weight band the AMD
dispatch keys on `static_N == 896 and static_K == 7168`, with an fp64 host
oracle (independent of the kernel's fp32 arithmetic), the top-1 routing
check, and the untouched-sentinel check on the padded output buffer. The
m == 2..32 bands run the `act_wide_k` K map (8 consecutive K per lane, one
SIMD-accumulator fold per row), whose fp32 summation order differs from the
cast-then-GEMV chain; the fp64 oracle covers that by design. The
`[128, 6144]` cells pin the older router shape's config and the
`static_K == 7168` key's other side (N=896 with K=6144 must take the N=128
sweep config, not the 896 table).
"""

from std.random import randn, random_float64, seed
from std.testing import assert_equal, assert_false, assert_true

from internal_utils import assert_almost_equal
from layout import Idx, TileTensor, row_major
from linalg.gemv import (
    _amd_router_gate_896x7168_config,
    router_gate_mixed_gemv,
    router_gate_use_mixed_gemv,
)
from max.gpu.host import DeviceContext, default_accelerator

# MiniMax-M3's router-gate contraction depth. `_k_iter_body` has no K-tail
# guard and the launcher tiles K by `simd_width * num_threads == 4 * 128`, so
# any K here must stay a multiple of 512 (6144 == 12 * 512).
comptime ROUTER_K = 6144

# Kimi K3's router-gate shape: the wide-N cold-weight table keys on this
# static `[N, K]` pair. A multiple of 1024 (the widest tile_k any band uses:
# simd_width=4 x 256 threads).
comptime K3_ROUTER_N = 896
comptime K3_ROUTER_K = 7168

# Pre-written into the output. Elements the kernel must not touch -- the
# trailing guard, and every element when `m == 0` -- still hold it afterwards.
comptime SENTINEL = Float32(-12345.0)
comptime GUARD_ELEMS = 8


def _run_case[static_N: Int, K: Int](ctx: DeviceContext, m: Int) raises:
    """Runs the launcher at `[m, K] x [static_N, K]^T` and checks the output.

    Parameters:
        static_N: Output width. Drives the launcher's `check_bounds_n` guard
            (`static_N % 2 != 0`), so an even and an odd value compile the two
            halves of its compile-time branching.
        K: Contraction depth, a multiple of 512.

    Args:
        ctx: The device context.
        m: Runtime row count. `m == 0` exercises the empty-launch guard.
    """
    print(t"== router_gate_mixed_gemv m= {m} N= {static_N} K= {K}")

    # Buffers stay non-empty at m == 0 so the launcher gets valid pointers,
    # which is exactly what the graph-capture warmup hands it.
    var rows = m if m > 0 else 1
    var c_elems = rows * static_N + GUARD_ELEMS

    var a_host = ctx.enqueue_create_host_buffer[.bfloat16](rows * K)
    var b_host = ctx.enqueue_create_host_buffer[.float32](static_N * K)
    var c_host = ctx.enqueue_create_host_buffer[.float32](c_elems)
    var c_expected = ctx.enqueue_create_host_buffer[.float32](c_elems)

    for i in range(rows * K):
        a_host[i] = random_float64(min=-1.0, max=1.0).cast[.bfloat16]()
    randn(b_host.unsafe_ptr(), static_N * K)
    for i in range(c_elems):
        c_host[i] = SENTINEL

    var a_dev = ctx.enqueue_create_buffer[.bfloat16](rows * K)
    var b_dev = ctx.enqueue_create_buffer[.float32](static_N * K)
    var c_dev = ctx.enqueue_create_buffer[.float32](c_elems)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(c_dev, c_host)

    var c_tt = TileTensor(c_dev.unsafe_ptr(), row_major(m, Idx[static_N]))
    var a_tt = TileTensor(a_dev.unsafe_ptr(), row_major(m, Idx[K]))
    var b_tt = TileTensor(b_dev.unsafe_ptr(), row_major[static_N, K]())

    router_gate_mixed_gemv[static_N](
        c_tt, a_tt.as_imm(), b_tt.as_imm(), m, static_N, K, ctx
    )

    ctx.enqueue_copy(c_host, c_dev)
    ctx.synchronize()

    # DRIV-199: keep device buffers alive past synchronize.
    _ = a_dev^
    _ = b_dev^
    _ = c_dev^

    # bf16 -> fp32 widening is lossless and the reduction structure matches
    # `gemv_split_k`, so an fp32 host reference is legitimate at a tight
    # tolerance. Untouched elements must still read back as the sentinel.
    for i in range(c_elems):
        c_expected[i] = SENTINEL
    for row in range(m):
        for col in range(static_N):
            var acc = Float32(0)
            for kk in range(K):
                acc += (
                    a_host[row * K + kk].cast[.float32]() * b_host[col * K + kk]
                )
            c_expected[row * static_N + col] = acc

    assert_almost_equal(
        c_host.unsafe_ptr(),
        c_expected.unsafe_ptr(),
        num_elements=c_elems,
        atol=1e-4,
        rtol=1e-2,
    )

    # Top-1 is what the MoE gate actually consumes: a tolerance check alone
    # would pass through a routing flip.
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

    print("PASS")


def _run_k3_case[static_N: Int, K: Int](ctx: DeviceContext, m: Int) raises:
    """Runs the launcher at `[m, K] x [static_N, K]^T` against an fp64 oracle.

    Unlike `_run_case`, the reference accumulates in fp64, so it is fully
    independent of the kernel's fp32 arithmetic. The data is signed and
    mixed-magnitude (per-row and per-column power-of-two scales on top of
    uniform and normal draws), so no band can pass on positive-only data and
    cancellation is exercised. Untouched output elements must still read back
    as the sentinel, exactly.

    Parameters:
        static_N: Output width (896 for the cold-weight table's key).
        K: Contraction depth, a multiple of 1024.

    Args:
        ctx: The device context.
        m: Runtime row count. `m == 0` exercises the empty-launch guard.
    """
    print("== router_gate_mixed_gemv k3 m=", m, "N=", static_N, "K=", K)

    var rows = m if m > 0 else 1
    var c_elems = rows * static_N + GUARD_ELEMS

    var a_host = ctx.enqueue_create_host_buffer[.bfloat16](rows * K)
    var b_host = ctx.enqueue_create_host_buffer[.float32](static_N * K)
    var c_host = ctx.enqueue_create_host_buffer[.float32](c_elems)
    var c_expected = ctx.enqueue_create_host_buffer[.float64](c_elems)

    var a_host_tt = TileTensor(a_host, row_major(rows, Idx[K]))
    var b_host_tt = TileTensor(b_host, row_major[static_N, K]())

    # Power-of-two scales spread magnitudes over several decades while
    # staying exactly representable.
    for row in range(rows):
        var row_scale = Float64(2) ** Float64((row % 8) - 4)
        for kk in range(K):
            a_host_tt[row, kk] = (
                random_float64(min=-1.0, max=1.0) * row_scale
            ).cast[.bfloat16]()
    randn(b_host.unsafe_ptr(), static_N * K)
    for col in range(static_N):
        var col_scale = Float32(2) ** Float32((col % 4) - 1)
        for kk in range(K):
            b_host_tt[col, kk] = b_host_tt[col, kk] * col_scale

    _ = TileTensor(c_host, row_major(c_elems)).fill(SENTINEL)
    _ = TileTensor(c_expected, row_major(c_elems)).fill(Float64(SENTINEL))

    var a_dev = ctx.enqueue_create_buffer[.bfloat16](rows * K)
    var b_dev = ctx.enqueue_create_buffer[.float32](static_N * K)
    var c_dev = ctx.enqueue_create_buffer[.float32](c_elems)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(c_dev, c_host)

    var c_tt = TileTensor(c_dev.unsafe_ptr(), row_major(m, Idx[static_N]))
    var a_tt = TileTensor(a_dev.unsafe_ptr(), row_major(m, Idx[K]))
    var b_tt = TileTensor(b_dev.unsafe_ptr(), row_major[static_N, K]())

    router_gate_mixed_gemv[static_N](
        c_tt, a_tt.as_imm(), b_tt.as_imm(), m, static_N, K, ctx
    )

    ctx.enqueue_copy(c_host, c_dev)
    ctx.synchronize()

    # DRIV-199: keep device buffers alive past synchronize.
    _ = a_dev^
    _ = b_dev^
    _ = c_dev^

    # fp64 oracle: independent of the kernel's fp32 accumulation order.
    var c_host_tt = TileTensor(c_host, row_major(m, Idx[static_N]))
    var c_expected_tt = TileTensor(c_expected, row_major(m, Idx[static_N]))
    for row in range(m):
        for col in range(static_N):
            var acc = Float64(0)
            for kk in range(K):
                acc += (
                    a_host_tt[row, kk].cast[.float64]()
                    * b_host_tt[col, kk].cast[.float64]()
                )
            c_expected_tt[row, col] = acc

    comptime atol = 1e-4
    comptime rtol = 1e-2
    var max_abs_err = Float64(0)
    var max_ref = Float64(0)
    for i in range(c_elems):
        var ref_val = c_expected[i]
        var got = Float64(c_host[i])
        if i >= m * static_N:
            # Untouched sentinel: exact, not tolerance-compared.
            assert_equal(
                c_host[i],
                SENTINEL,
                "kernel wrote a guarded output element " + String(i),
            )
            continue
        var err = abs(got - ref_val)
        max_abs_err = max(max_abs_err, err)
        max_ref = max(max_ref, abs(ref_val))
        assert_true(
            err <= atol + rtol * abs(ref_val),
            "fp64 oracle mismatch at "
            + String(i)
            + ": got "
            + String(got)
            + " ref "
            + String(ref_val),
        )
    print(
        "oracle m=",
        m,
        " max_abs_err=",
        max_abs_err,
        " max_ref=",
        max_ref,
    )

    # Top-1 is what the MoE gate actually consumes: a tolerance check alone
    # would pass through a routing flip.
    for row in range(m):
        var got_arg = 0
        var expected_arg = 0
        for col in range(1, static_N):
            if c_host_tt[row, col] > c_host_tt[row, got_arg]:
                got_arg = col
            if c_expected_tt[row, col] > c_expected_tt[row, expected_arg]:
                expected_arg = col
        assert_equal(
            got_arg,
            expected_arg,
            "top-1 routing mismatch on row " + String(row),
        )

    print("PASS")


def _assert_k3_config[
    tile_m: Int,
    tile_n: Int,
    num_threads: Int,
    unroll_factor: Int,
    weight_non_temporal: Bool,
    small_m_static_k: Bool,
    small_m_k_chunk: Int,
    expert_major_grid: Bool,
    act_wide_k: Bool,
](m: Int) raises:
    """Asserts the config the cold-weight table returns at this `m`."""
    var cfg = _amd_router_gate_896x7168_config(m)
    assert_equal(cfg.tile_m, tile_m, "tile_m at m=" + String(m))
    assert_equal(cfg.tile_n, tile_n, "tile_n at m=" + String(m))
    assert_equal(cfg.num_threads, num_threads, "threads at m=" + String(m))
    assert_equal(cfg.unroll_factor, unroll_factor, "unroll at m=" + String(m))
    assert_equal(
        cfg.weight_non_temporal,
        weight_non_temporal,
        "weight_non_temporal at m=" + String(m),
    )
    assert_equal(
        cfg.small_m_static_k, small_m_static_k, "static_k at m=" + String(m)
    )
    assert_equal(
        cfg.small_m_k_chunk, small_m_k_chunk, "k_chunk at m=" + String(m)
    )
    assert_equal(
        cfg.expert_major_grid,
        expert_major_grid,
        "expert_major_grid at m=" + String(m),
    )
    assert_equal(
        cfg.act_wide_k,
        act_wide_k,
        "act_wide_k at m=" + String(m),
    )


def test_amd_router_gate_896x7168_config() raises:
    """Pins every band edge of the cold-weight `[896, 7168]` config table."""
    # m == 1 streams the weight once (no row reuse to exploit).
    _assert_k3_config[1, 1, 256, 2, True, False, 0, False, False](1)
    # m == 2..6: 2-row blocks; the expert-major grid keeps an expert's
    # M-blocks on one XCD, so the shared weight row is an L2 hit, and the
    # wide-K map loads each activation row as one 16 B vector per K-block.
    for m in range(2, 7):
        _assert_k3_config[2, 1, 128, 2, False, True, 1, True, True](m)
    # m == 7..12: 4-row blocks (a tm8 2nd block would waste up to 7 rows).
    _assert_k3_config[4, 1, 128, 2, False, True, 1, True, True](7)
    _assert_k3_config[4, 1, 128, 2, False, True, 1, True, True](8)
    _assert_k3_config[4, 1, 128, 2, False, True, 1, True, True](12)
    # m == 13..32: 8-row blocks again (the 4th 4-row block would be mostly
    # dead from m == 13).
    _assert_k3_config[8, 1, 128, 2, False, True, 1, True, True](13)
    _assert_k3_config[8, 1, 128, 2, False, True, 1, True, True](16)
    _assert_k3_config[8, 1, 128, 2, False, True, 1, True, True](17)
    _assert_k3_config[8, 1, 128, 2, False, True, 1, True, True](24)
    _assert_k3_config[8, 1, 128, 2, False, True, 1, True, True](31)
    _assert_k3_config[8, 1, 128, 2, False, True, 1, True, True](32)
    # m == 33..64: the tile_n=4 band.
    _assert_k3_config[4, 4, 128, 2, False, False, 0, False, False](33)
    _assert_k3_config[4, 4, 128, 2, False, False, 0, False, False](64)
    # Above the fused window: the wide-N fallback config.
    _assert_k3_config[1, 4, 128, 2, False, False, 0, False, False](65)


def test_router_gate_use_mixed_gemv() raises:
    """Pins the M window the fused path claims (`0 < m <= 64`)."""
    assert_false(router_gate_use_mixed_gemv(0))
    assert_true(router_gate_use_mixed_gemv(1))
    assert_true(router_gate_use_mixed_gemv(16))
    assert_true(router_gate_use_mixed_gemv(17))
    assert_true(router_gate_use_mixed_gemv(64))
    assert_false(router_gate_use_mixed_gemv(65))


def main() raises:
    comptime if not default_accelerator().is_amd_gpu():
        print("SKIP: AMD GPU not available")
        return

    seed(0)
    test_router_gate_use_mixed_gemv()
    test_amd_router_gate_896x7168_config()

    with DeviceContext() as ctx:
        _run_case[128, ROUTER_K](ctx, 1)
        _run_case[128, ROUTER_K](ctx, 2)
        _run_case[128, ROUTER_K](ctx, 16)
        _run_case[128, ROUTER_K](ctx, 17)
        _run_case[128, ROUTER_K](ctx, 24)
        _run_case[128, ROUTER_K](ctx, 32)
        _run_case[128, ROUTER_K](ctx, 64)
        # Odd N exercises check_bounds_n in both tile_n bands.
        _run_case[127, ROUTER_K](ctx, 2)
        _run_case[127, ROUTER_K](ctx, 24)
        # Empty-launch guard: no launch, no fault, output untouched.
        _run_case[128, ROUTER_K](ctx, 0)
        # Kimi K3's [896, 7168] router gate: m values across every
        # cold-weight band, against the fp64 oracle.
        comptime k3_case_ms = (
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            12,
            16,
            17,
            24,
            25,
            32,
            33,
            64,
        )
        comptime for i in range(len(k3_case_ms)):
            _run_k3_case[K3_ROUTER_N, K3_ROUTER_K](ctx, k3_case_ms[i])
        # The table keys on the static [N, K] pair: N=896 with K=6144 must
        # take the N=128 sweep config, not the 896 table.
        _run_case[896, ROUTER_K](ctx, 2)

    print("ALL TESTS PASSED")
