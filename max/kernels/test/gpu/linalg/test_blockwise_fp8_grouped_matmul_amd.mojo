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
"""Correctness test for the tiled blockwise-scaled FP8 grouped matmul (gfx950).

Compares `blockwise_scaled_fp8_grouped_matmul_amd` (directly and via the
`grouped_matmul_dynamic_scaled_fp8` dispatch) against the trusted
`naive_blockwise_scaled_fp8_grouped_matmul` reference at GLM MoE shapes.

The hard part of a grouped matmul is scale indexing under a ragged,
expert-dispatched layout, so this test deliberately stresses it:

  - scales are LOG-UNIFORM in `[2^-4, 2^4]` per block (a constant scale
    hides an index bug),
  - rows per expert are UNEVEN, with at least one ZERO-row expert and at
    least one expert whose rows are not a multiple of the tile,
  - at least one active slot has `expert_id == -1` (skipped; output 0),
  - A carries NaN poison rows OUTSIDE every expert's range -- if the
    kernel reads past an expert's rows the output goes NaN, which an
    explicit scan catches (an `almost_equal` band silently passes NaNs).
"""

from std.math import ceildiv, exp2, isnan

from max.gpu.host import DeviceContext
from internal_utils import assert_almost_equal
from std.testing import assert_equal
from std.random import rand
from layout import Coord, Idx, TileTensor, row_major
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_grouped_matmul
from linalg.matmul.gpu.amd.blockwise_scaled_fp8_grouped_matmul_amd import (
    blockwise_scaled_fp8_grouped_matmul_amd,
)
from linalg.grouped_matmul_sm100_blockwise_fp8 import (
    grouped_matmul_dynamic_scaled_fp8,
)

from std.utils.index import Index


def test_grouped_blockwise_fp8_matmul_amd[
    c_type: DType,
    num_experts: Int,
    N: Int,
    K: Int,
    via_dispatch: Bool = False,
    static_grid_z: Bool = False,
    check_cpu: Bool = False,
](
    ctx: DeviceContext,
    var row_counts: List[Int],
    var expert_id_list: List[Int],
) raises:
    comptime input_type = DType.float8_e4m3fn
    comptime transpose_b = True
    comptime BLOCK_SCALE_M = 1
    comptime BLOCK_SCALE_N = 128
    comptime BLOCK_SCALE_K = 128
    comptime N_BLOCKS = N // BLOCK_SCALE_N
    comptime K_BLOCKS = K // BLOCK_SCALE_K

    var num_active = len(row_counts)

    comptime if static_grid_z:
        if num_active != num_experts:
            raise Error(
                "test setup: static_grid_z requires row_counts to cover"
                " every expert slot"
            )

    # Total active rows and the largest expert. Poison rows past the last
    # expert's range live at [sum_rows, total_tokens).
    var sum_rows = 0
    var max_tokens = 0
    for i in range(num_active):
        sum_rows += row_counts[i]
        if row_counts[i] > max_tokens:
            max_tokens = row_counts[i]

    comptime POISON = 64
    var total_tokens = sum_rows + POISON

    print(
        "== test_grouped_blockwise_fp8_matmul_amd c_type=",
        c_type,
        " num_experts=",
        num_experts,
        " num_active=",
        num_active,
        " sum_rows=",
        sum_rows,
        " NxK=",
        N,
        "x",
        K,
        " via_dispatch=",
        via_dispatch,
        " static_grid_z=",
        static_grid_z,
        sep="",
    )

    var a_size = total_tokens * K
    var b_size = num_experts * N * K
    var c_size = total_tokens * N
    var a_scale_size = K_BLOCKS * total_tokens
    var b_scale_size = num_experts * N_BLOCKS * K_BLOCKS

    var a_host = ctx.enqueue_create_host_buffer[input_type](a_size)
    var b_host = ctx.enqueue_create_host_buffer[input_type](b_size)
    var a_scale_host = ctx.enqueue_create_host_buffer[.float32](a_scale_size)
    var b_scale_host = ctx.enqueue_create_host_buffer[.float32](b_scale_size)

    rand(a_host.unsafe_ptr(), a_size)
    rand(b_host.unsafe_ptr(), b_size)
    rand(a_scale_host.unsafe_ptr(), a_scale_size)
    rand(b_scale_host.unsafe_ptr(), b_scale_size)
    for i in range(a_scale_size):
        a_scale_host[i] = exp2(Float32(8.0) * a_scale_host[i] - 4.0)
    for i in range(b_scale_size):
        b_scale_host[i] = exp2(Float32(8.0) * b_scale_host[i] - 4.0)

    # NaN-poison every A row outside the active range. 0x7F is a NaN in
    # E4M3 (OCP). A correct kernel never touches these rows.
    var a_bytes = a_host.unsafe_ptr().bitcast[UInt8]()
    for r in range(sum_rows, total_tokens):
        for kk in range(K):
            a_bytes[r * K + kk] = UInt8(0x7F)

    var c_my_host = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_my_f32 = ctx.enqueue_create_host_buffer[.float32](c_size)
    var c_ref_host = ctx.enqueue_create_host_buffer[.float32](c_size)

    var a_dev_buf = ctx.enqueue_create_buffer[input_type](a_size)
    var b_dev_buf = ctx.enqueue_create_buffer[input_type](b_size)
    var c_my_buf = ctx.enqueue_create_buffer[c_type](c_size)
    var c_ref_buf = ctx.enqueue_create_buffer[.float32](c_size)
    var a_scale_buf = ctx.enqueue_create_buffer[.float32](a_scale_size)
    var b_scale_buf = ctx.enqueue_create_buffer[.float32](b_scale_size)
    var a_offsets_buf = ctx.enqueue_create_buffer[.uint32](num_active + 1)
    var expert_ids_buf = ctx.enqueue_create_buffer[.int32](num_active)

    var a_offsets_h_buf = ctx.enqueue_create_host_buffer[.uint32](
        num_active + 1
    )
    var expert_ids_h_buf = ctx.enqueue_create_host_buffer[.int32](num_active)
    var running = 0
    for i in range(num_active):
        a_offsets_h_buf[i] = UInt32(running)
        running += row_counts[i]
        expert_ids_h_buf[i] = Int32(expert_id_list[i])
    a_offsets_h_buf[num_active] = UInt32(running)

    ctx.enqueue_copy(a_dev_buf, a_host)
    ctx.enqueue_copy(b_dev_buf, b_host)
    ctx.enqueue_copy(a_scale_buf, a_scale_host)
    ctx.enqueue_copy(b_scale_buf, b_scale_host)
    ctx.enqueue_copy(a_offsets_buf, a_offsets_h_buf)
    ctx.enqueue_copy(expert_ids_buf, expert_ids_h_buf)
    # Skipped (-1) slots and untouched rows must read 0, so pre-zero C.
    c_my_buf.enqueue_fill(0)
    c_ref_buf.enqueue_fill(0)

    var a_dev = TileTensor(a_dev_buf, row_major(Coord(total_tokens, Idx[K])))
    var b_dev = TileTensor(b_dev_buf, row_major[num_experts, N, K]())
    var c_my_dev = TileTensor(c_my_buf, row_major(Coord(total_tokens, Idx[N])))
    var c_ref_dev = TileTensor(
        c_ref_buf, row_major(Coord(total_tokens, Idx[N]))
    )
    var a_scale_dev = TileTensor(
        a_scale_buf, row_major(Coord(Idx[K_BLOCKS], total_tokens))
    )
    var b_scale_dev = TileTensor(
        b_scale_buf, row_major[num_experts, N_BLOCKS, K_BLOCKS]()
    )
    var a_offsets_dev = TileTensor(
        a_offsets_buf, row_major(Coord(num_active + 1))
    )
    var expert_ids_dev = TileTensor(
        expert_ids_buf, row_major(Coord(num_active))
    )

    comptime if via_dispatch:
        grouped_matmul_dynamic_scaled_fp8[
            input_scale_granularity="block",
            weight_scale_granularity="block",
            m_scale_granularity=BLOCK_SCALE_M,
            n_scale_granularity=BLOCK_SCALE_N,
            k_scale_granularity=BLOCK_SCALE_K,
            transpose_b=transpose_b,
            static_grid_z=static_grid_z,
            target="gpu",
        ](
            c_my_dev,
            a_dev,
            b_dev,
            a_scale_dev,
            b_scale_dev,
            a_offsets_dev,
            expert_ids_dev,
            max_num_tokens_per_expert=max_tokens,
            num_active_experts=num_active,
            ctx=ctx,
        )
    else:
        blockwise_scaled_fp8_grouped_matmul_amd[
            transpose_b=transpose_b,
            N_SCALE=BLOCK_SCALE_N,
            K_SCALE=BLOCK_SCALE_K,
            static_grid_z=static_grid_z,
        ](
            c_my_dev,
            a_dev,
            b_dev,
            a_scale_dev,
            b_scale_dev,
            a_offsets_dev,
            expert_ids_dev,
            max_num_tokens_per_expert=max_tokens,
            num_active_experts=num_active,
            ctx=ctx,
        )

    naive_blockwise_scaled_fp8_grouped_matmul[
        BLOCK_DIM_M=16,
        BLOCK_DIM_N=16,
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
        a_offsets_dev,
        expert_ids_dev,
        max_tokens,
        num_active,
        ctx,
    )

    ctx.enqueue_copy(c_my_host, c_my_buf)
    ctx.enqueue_copy(c_ref_host, c_ref_buf)
    ctx.synchronize()

    var nan_count = 0
    for i in range(c_size):
        c_my_f32[i] = c_my_host[i].cast[.float32]()
        if isnan(c_my_f32[i]):
            nan_count += 1
    if nan_count != 0:
        print("   FAILED: ", nan_count, " NaN outputs (poison row read?)")
    assert_equal(
        nan_count, 0, msg="kernel produced NaN outputs -- poison A rows read"
    )

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

    comptime rtol = 1e-2 if c_type == .float32 else 4e-2
    comptime atol = 1e-2 if c_type == .float32 else 2e-1
    assert_almost_equal(
        c_my_f32.unsafe_ptr(),
        c_ref_host.unsafe_ptr(),
        c_size,
        "grouped tiled vs naive",
        atol=atol,
        rtol=rtol,
    )

    comptime if check_cpu:
        # Independent host fp32 oracle mirroring the naive scale indexing.
        var c_cpu = ctx.enqueue_create_host_buffer[.float32](c_size)
        for i in range(c_size):
            c_cpu[i] = 0.0
        var a_start = 0
        for slot in range(num_active):
            var eid = expert_id_list[slot]
            var m_count = row_counts[slot]
            if eid == -1:
                a_start += m_count
                continue
            for ml in range(m_count):
                var mg = a_start + ml
                for nn in range(N):
                    var acc = Float32(0.0)
                    for kk in range(K):
                        var av = a_host[mg * K + kk].cast[.float32]()
                        var bv = b_host[eid * N * K + nn * K + kk].cast[
                            .float32
                        ]()
                        var asc = a_scale_host[
                            (kk // BLOCK_SCALE_K) * total_tokens + mg
                        ]
                        var bsc = b_scale_host[
                            eid * N_BLOCKS * K_BLOCKS
                            + (nn // BLOCK_SCALE_N) * K_BLOCKS
                            + (kk // BLOCK_SCALE_K)
                        ]
                        acc += av * bv * asc * bsc
                    c_cpu[mg * N + nn] = acc
            a_start += m_count
        assert_almost_equal(
            c_my_f32.unsafe_ptr(),
            c_cpu.unsafe_ptr(),
            c_size,
            "grouped tiled vs cpu fp32",
            atol=atol,
            rtol=rtol,
        )

    print("   PASSED")


def test_grouped_static_grid_z[
    c_type: DType,
    num_experts: Int,
    N: Int,
    K: Int,
](
    ctx: DeviceContext,
    var row_counts: List[Int],
    var expert_id_list: List[Int],
) raises:
    """`static_grid_z=True` must produce identical output to the default.

    Sizes grid.z from `b`'s comptime expert count instead of the runtime
    `num_active_experts` tensor read -- safe here because `num_active_experts
    == num_experts` on every call this test builds (mirrors the
    `moe_create_indices` routing contract documented on the parameter).
    """
    comptime input_type = DType.float8_e4m3fn
    comptime transpose_b = True
    comptime BLOCK_SCALE_N = 128
    comptime BLOCK_SCALE_K = 128
    comptime N_BLOCKS = N // BLOCK_SCALE_N
    comptime K_BLOCKS = K // BLOCK_SCALE_K

    var num_active = len(row_counts)
    if num_active != num_experts:
        raise Error("test setup: row_counts must cover every expert slot")

    var sum_rows = 0
    var max_tokens = 0
    for i in range(num_active):
        sum_rows += row_counts[i]
        if row_counts[i] > max_tokens:
            max_tokens = row_counts[i]
    var total_tokens = sum_rows

    print(
        "== test_grouped_static_grid_z c_type=",
        c_type,
        " num_experts=",
        num_experts,
        " sum_rows=",
        sum_rows,
        sep="",
    )

    var a_size = total_tokens * K
    var b_size = num_experts * N * K
    var c_size = total_tokens * N
    var a_scale_size = K_BLOCKS * total_tokens
    var b_scale_size = num_experts * N_BLOCKS * K_BLOCKS

    var a_host = ctx.enqueue_create_host_buffer[input_type](a_size)
    var b_host = ctx.enqueue_create_host_buffer[input_type](b_size)
    var a_scale_host = ctx.enqueue_create_host_buffer[.float32](a_scale_size)
    var b_scale_host = ctx.enqueue_create_host_buffer[.float32](b_scale_size)
    rand(a_host.unsafe_ptr(), a_size)
    rand(b_host.unsafe_ptr(), b_size)
    rand(a_scale_host.unsafe_ptr(), a_scale_size)
    rand(b_scale_host.unsafe_ptr(), b_scale_size)
    for i in range(a_scale_size):
        a_scale_host[i] = exp2(Float32(8.0) * a_scale_host[i] - 4.0)
    for i in range(b_scale_size):
        b_scale_host[i] = exp2(Float32(8.0) * b_scale_host[i] - 4.0)

    var a_dev_buf = ctx.enqueue_create_buffer[input_type](a_size)
    var b_dev_buf = ctx.enqueue_create_buffer[input_type](b_size)
    var c_default_buf = ctx.enqueue_create_buffer[c_type](c_size)
    var c_static_buf = ctx.enqueue_create_buffer[c_type](c_size)
    var a_scale_buf = ctx.enqueue_create_buffer[.float32](a_scale_size)
    var b_scale_buf = ctx.enqueue_create_buffer[.float32](b_scale_size)
    var a_offsets_buf = ctx.enqueue_create_buffer[.uint32](num_active + 1)
    var expert_ids_buf = ctx.enqueue_create_buffer[.int32](num_active)

    var a_offsets_h_buf = ctx.enqueue_create_host_buffer[.uint32](
        num_active + 1
    )
    var expert_ids_h_buf = ctx.enqueue_create_host_buffer[.int32](num_active)
    var running = 0
    for i in range(num_active):
        a_offsets_h_buf[i] = UInt32(running)
        running += row_counts[i]
        expert_ids_h_buf[i] = Int32(expert_id_list[i])
    a_offsets_h_buf[num_active] = UInt32(running)

    ctx.enqueue_copy(a_dev_buf, a_host)
    ctx.enqueue_copy(b_dev_buf, b_host)
    ctx.enqueue_copy(a_scale_buf, a_scale_host)
    ctx.enqueue_copy(b_scale_buf, b_scale_host)
    ctx.enqueue_copy(a_offsets_buf, a_offsets_h_buf)
    ctx.enqueue_copy(expert_ids_buf, expert_ids_h_buf)
    c_default_buf.enqueue_fill(0)
    c_static_buf.enqueue_fill(0)

    var a_dev = TileTensor(a_dev_buf, row_major(Coord(total_tokens, Idx[K])))
    var b_dev = TileTensor(b_dev_buf, row_major[num_experts, N, K]())
    var c_default_dev = TileTensor(
        c_default_buf, row_major(Coord(total_tokens, Idx[N]))
    )
    var c_static_dev = TileTensor(
        c_static_buf, row_major(Coord(total_tokens, Idx[N]))
    )
    var a_scale_dev = TileTensor(
        a_scale_buf, row_major(Coord(Idx[K_BLOCKS], total_tokens))
    )
    var b_scale_dev = TileTensor(
        b_scale_buf, row_major[num_experts, N_BLOCKS, K_BLOCKS]()
    )
    var a_offsets_dev = TileTensor(
        a_offsets_buf, row_major(Coord(num_active + 1))
    )
    var expert_ids_dev = TileTensor(
        expert_ids_buf, row_major(Coord(num_active))
    )

    blockwise_scaled_fp8_grouped_matmul_amd[
        transpose_b=transpose_b,
        N_SCALE=BLOCK_SCALE_N,
        K_SCALE=BLOCK_SCALE_K,
    ](
        c_default_dev,
        a_dev,
        b_dev,
        a_scale_dev,
        b_scale_dev,
        a_offsets_dev,
        expert_ids_dev,
        max_num_tokens_per_expert=max_tokens,
        num_active_experts=num_active,
        ctx=ctx,
    )
    blockwise_scaled_fp8_grouped_matmul_amd[
        transpose_b=transpose_b,
        N_SCALE=BLOCK_SCALE_N,
        K_SCALE=BLOCK_SCALE_K,
        static_grid_z=True,
    ](
        c_static_dev,
        a_dev,
        b_dev,
        a_scale_dev,
        b_scale_dev,
        a_offsets_dev,
        expert_ids_dev,
        max_num_tokens_per_expert=max_tokens,
        num_active_experts=num_active,
        ctx=ctx,
    )

    var c_default_host = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_static_host = ctx.enqueue_create_host_buffer[c_type](c_size)
    ctx.enqueue_copy(c_default_host, c_default_buf)
    ctx.enqueue_copy(c_static_host, c_static_buf)
    ctx.synchronize()

    for i in range(c_size):
        assert_equal(
            c_default_host[i],
            c_static_host[i],
            msg=(
                "static_grid_z changed the result -- grid.z sourced from"
                " b's comptime expert count must match the runtime"
                " num_active_experts read bit-for-bit"
            ),
        )
    print("   PASSED")


def test_grouped_decode_grid_m_cap_capture_safety[
    c_type: DType,
    num_experts: Int,
    N: Int,
    K: Int,
](ctx: DeviceContext, var row_counts: List[Int],) raises:
    """Proves a stale (capture-frozen) `max_num_tokens_per_expert` is safe.

    Simulates a `max_num_tokens_per_expert` frozen at an earlier, smaller
    warmup step (a stale value well below this call's true per-expert max)
    -- exactly what a captured-and-replayed graph would hand the kernel if
    a later step routes more rows to some expert than warmup saw. The
    kernel's M-tile stride loop makes grid.y a packing hint rather than a
    coverage bound, and the launcher clamps grid.y to this call's own row
    count, so BOTH the ungated launch and the `decode_grid_m_cap` launch
    (retained for source compatibility) must cover every row: the silent
    row-drop hazard this test used to demonstrate as its negative control
    is structurally gone, with no gate opt-in required.
    """
    comptime input_type = DType.float8_e4m3fn
    comptime transpose_b = True
    comptime BLOCK_SCALE_N = 128
    comptime BLOCK_SCALE_K = 128
    comptime N_BLOCKS = N // BLOCK_SCALE_N
    comptime K_BLOCKS = K // BLOCK_SCALE_K

    var num_active = len(row_counts)
    var sum_rows = 0
    var true_max_tokens = 0
    for i in range(num_active):
        sum_rows += row_counts[i]
        if row_counts[i] > true_max_tokens:
            true_max_tokens = row_counts[i]
    var total_tokens = sum_rows

    # A stale value under the true per-expert max: what a graph captured at a
    # lighter-routing warmup step would freeze into grid.y on every replay.
    var stale_max_tokens = 8

    print(
        "== test_grouped_decode_grid_m_cap_capture_safety c_type=",
        c_type,
        " num_experts=",
        num_experts,
        " sum_rows=",
        sum_rows,
        " true_max=",
        true_max_tokens,
        " stale_max=",
        stale_max_tokens,
        sep="",
    )

    var a_size = total_tokens * K
    var b_size = num_experts * N * K
    var c_size = total_tokens * N
    var a_scale_size = K_BLOCKS * total_tokens
    var b_scale_size = num_experts * N_BLOCKS * K_BLOCKS

    var a_host = ctx.enqueue_create_host_buffer[input_type](a_size)
    var b_host = ctx.enqueue_create_host_buffer[input_type](b_size)
    var a_scale_host = ctx.enqueue_create_host_buffer[.float32](a_scale_size)
    var b_scale_host = ctx.enqueue_create_host_buffer[.float32](b_scale_size)
    rand(a_host.unsafe_ptr(), a_size)
    rand(b_host.unsafe_ptr(), b_size)
    rand(a_scale_host.unsafe_ptr(), a_scale_size)
    rand(b_scale_host.unsafe_ptr(), b_scale_size)
    for i in range(a_scale_size):
        a_scale_host[i] = exp2(Float32(8.0) * a_scale_host[i] - 4.0)
    for i in range(b_scale_size):
        b_scale_host[i] = exp2(Float32(8.0) * b_scale_host[i] - 4.0)

    var c_broken_host = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_fixed_host = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_ref_host = ctx.enqueue_create_host_buffer[.float32](c_size)

    var a_dev_buf = ctx.enqueue_create_buffer[input_type](a_size)
    var b_dev_buf = ctx.enqueue_create_buffer[input_type](b_size)
    var c_broken_buf = ctx.enqueue_create_buffer[c_type](c_size)
    var c_fixed_buf = ctx.enqueue_create_buffer[c_type](c_size)
    var c_ref_buf = ctx.enqueue_create_buffer[.float32](c_size)
    var a_scale_buf = ctx.enqueue_create_buffer[.float32](a_scale_size)
    var b_scale_buf = ctx.enqueue_create_buffer[.float32](b_scale_size)
    var a_offsets_buf = ctx.enqueue_create_buffer[.uint32](num_active + 1)
    var expert_ids_buf = ctx.enqueue_create_buffer[.int32](num_active)

    var a_offsets_h_buf = ctx.enqueue_create_host_buffer[.uint32](
        num_active + 1
    )
    var expert_ids_h_buf = ctx.enqueue_create_host_buffer[.int32](num_active)
    var running = 0
    for i in range(num_active):
        a_offsets_h_buf[i] = UInt32(running)
        running += row_counts[i]
        expert_ids_h_buf[i] = Int32(i)
    a_offsets_h_buf[num_active] = UInt32(running)

    ctx.enqueue_copy(a_dev_buf, a_host)
    ctx.enqueue_copy(b_dev_buf, b_host)
    ctx.enqueue_copy(a_scale_buf, a_scale_host)
    ctx.enqueue_copy(b_scale_buf, b_scale_host)
    ctx.enqueue_copy(a_offsets_buf, a_offsets_h_buf)
    ctx.enqueue_copy(expert_ids_buf, expert_ids_h_buf)
    c_broken_buf.enqueue_fill(0)
    c_fixed_buf.enqueue_fill(0)
    c_ref_buf.enqueue_fill(0)

    var a_dev = TileTensor(a_dev_buf, row_major(Coord(total_tokens, Idx[K])))
    var b_dev = TileTensor(b_dev_buf, row_major[num_experts, N, K]())
    var c_broken_dev = TileTensor(
        c_broken_buf, row_major(Coord(total_tokens, Idx[N]))
    )
    var c_fixed_dev = TileTensor(
        c_fixed_buf, row_major(Coord(total_tokens, Idx[N]))
    )
    var c_ref_dev = TileTensor(
        c_ref_buf, row_major(Coord(total_tokens, Idx[N]))
    )
    var a_scale_dev = TileTensor(
        a_scale_buf, row_major(Coord(Idx[K_BLOCKS], total_tokens))
    )
    var b_scale_dev = TileTensor(
        b_scale_buf, row_major[num_experts, N_BLOCKS, K_BLOCKS]()
    )
    var a_offsets_dev = TileTensor(
        a_offsets_buf, row_major(Coord(num_active + 1))
    )
    var expert_ids_dev = TileTensor(
        expert_ids_buf, row_major(Coord(num_active))
    )

    # "Broken": no cap, stale (too-small) max_num_tokens_per_expert -- the
    # exact shape of the hazard under device graph capture.
    blockwise_scaled_fp8_grouped_matmul_amd[
        transpose_b=transpose_b,
        N_SCALE=BLOCK_SCALE_N,
        K_SCALE=BLOCK_SCALE_K,
    ](
        c_broken_dev,
        a_dev,
        b_dev,
        a_scale_dev,
        b_scale_dev,
        a_offsets_dev,
        expert_ids_dev,
        max_num_tokens_per_expert=stale_max_tokens,
        num_active_experts=num_active,
        ctx=ctx,
    )

    # "Fixed": same stale value, but the decode-band gate is active and this
    # call's shape qualifies, so grid.y comes from `a`'s own row count.
    blockwise_scaled_fp8_grouped_matmul_amd[
        transpose_b=transpose_b,
        N_SCALE=BLOCK_SCALE_N,
        K_SCALE=BLOCK_SCALE_K,
    ](
        c_fixed_dev,
        a_dev,
        b_dev,
        a_scale_dev,
        b_scale_dev,
        a_offsets_dev,
        expert_ids_dev,
        max_num_tokens_per_expert=stale_max_tokens,
        num_active_experts=num_active,
        ctx=ctx,
        decode_grid_m_cap=total_tokens,
    )

    naive_blockwise_scaled_fp8_grouped_matmul[
        BLOCK_DIM_M=16,
        BLOCK_DIM_N=16,
        transpose_b=transpose_b,
        scales_granularity_mnk=Index(1, BLOCK_SCALE_N, BLOCK_SCALE_K),
    ](
        c_ref_dev,
        a_dev,
        b_dev,
        a_scale_dev,
        b_scale_dev,
        a_offsets_dev,
        expert_ids_dev,
        true_max_tokens,
        num_active,
        ctx,
    )

    ctx.enqueue_copy(c_broken_host, c_broken_buf)
    ctx.enqueue_copy(c_fixed_host, c_fixed_buf)
    ctx.enqueue_copy(c_ref_host, c_ref_buf)
    ctx.synchronize()

    comptime rtol = 1e-2 if c_type == .float32 else 4e-2
    comptime atol = 1e-2 if c_type == .float32 else 2e-1

    # Ungated launch: the M-tile stride loop covers the stale, undersized
    # grid.y bound anyway -- no silent row drops.
    var broken_c = ctx.enqueue_create_host_buffer[.float32](c_size)
    for i in range(c_size):
        broken_c[i] = c_broken_host[i].cast[.float32]()
    assert_almost_equal(
        broken_c.unsafe_ptr(),
        c_ref_host.unsafe_ptr(),
        c_size,
        "grouped stale-max (no cap) vs naive",
        atol=atol,
        rtol=rtol,
    )

    # The gated launch: decode_grid_m_cap active (retained for source
    # compatibility) with the same stale runtime value is also correct.
    var fixed_c = ctx.enqueue_create_host_buffer[.float32](c_size)
    for i in range(c_size):
        fixed_c[i] = c_fixed_host[i].cast[.float32]()
    assert_almost_equal(
        fixed_c.unsafe_ptr(),
        c_ref_host.unsafe_ptr(),
        c_size,
        "grouped decode_grid_m_cap-fixed vs naive",
        atol=atol,
        rtol=rtol,
    )
    print("   PASSED")


def test_grouped_grid_y_decode_bounded_bitidentity[
    c_type: DType,
    num_experts: Int,
    N: Int,
    K: Int,
](
    ctx: DeviceContext,
    var row_counts: List[Int],
    prod_max_tokens: Int,
    decode_max_tokens: Int,
) raises:
    """Proves the decode-bounded grid.y packing hint is bit-identical.

    The GLM-5.3 EP decode call pads `a` to the pre-allocated recv buffer's
    row count, so the launcher's `min(max_num_tokens_per_expert, a.rows)`
    clamp cannot mask the metadata: the prefill-sized metadata
    (`max_tokens_per_rank * n_ranks`, 2048 in production) launches grid.y=32
    whose trailing CTAs exit at the bounds check, while the decode-bounded
    metadata (this step's estimated received rows) launches grid.y=1 and
    each CTA strides its expert's M-tiles. Both launches cover the same
    M-tiles with the same one-tile body, so the outputs must match
    bit-for-bit. `a` carries NaN poison rows past the routed rows to prove
    neither launch reads them.
    """
    comptime input_type = DType.float8_e4m3fn
    comptime transpose_b = True
    comptime BLOCK_SCALE_N = 128
    comptime BLOCK_SCALE_K = 128
    comptime N_BLOCKS = N // BLOCK_SCALE_N
    comptime K_BLOCKS = K // BLOCK_SCALE_K
    comptime BM = 64  # launcher default; grid.y = ceildiv(m_cap, BM)

    var num_active = len(row_counts)
    var sum_rows = 0
    for i in range(num_active):
        sum_rows += row_counts[i]

    # Pad `a` past the routed rows to the production recv-buffer count (>=
    # prod_max_tokens so the clamp keeps the prefill bound); pad rows are NaN.
    var pad_rows = prod_max_tokens + 64

    print(
        "== test_grouped_grid_y_decode_bounded_bitidentity c_type=",
        c_type,
        " num_experts=",
        num_experts,
        " num_active=",
        num_active,
        " sum_rows=",
        sum_rows,
        " pad_rows=",
        pad_rows,
        " NxK=",
        N,
        "x",
        K,
        sep="",
    )

    var a_size = pad_rows * K
    var b_size = num_experts * N * K
    var c_size = sum_rows * N
    var a_scale_size = K_BLOCKS * sum_rows
    var b_scale_size = num_experts * N_BLOCKS * K_BLOCKS

    var a_host = ctx.enqueue_create_host_buffer[input_type](a_size)
    var b_host = ctx.enqueue_create_host_buffer[input_type](b_size)
    var a_scale_host = ctx.enqueue_create_host_buffer[.float32](a_scale_size)
    var b_scale_host = ctx.enqueue_create_host_buffer[.float32](b_scale_size)

    rand(a_host.unsafe_ptr(), a_size)
    rand(b_host.unsafe_ptr(), b_size)
    rand(a_scale_host.unsafe_ptr(), a_scale_size)
    rand(b_scale_host.unsafe_ptr(), b_scale_size)
    for i in range(a_scale_size):
        a_scale_host[i] = exp2(Float32(8.0) * a_scale_host[i] - 4.0)
    for i in range(b_scale_size):
        b_scale_host[i] = exp2(Float32(8.0) * b_scale_host[i] - 4.0)

    # NaN-poison every A row outside the routed range (0x7F is a NaN in
    # E4M3); a correct launch never reads them.
    var a_bytes = a_host.unsafe_ptr().bitcast[UInt8]()
    for r in range(sum_rows, pad_rows):
        for kk in range(K):
            a_bytes[r * K + kk] = UInt8(0x7F)

    var a_dev_buf = ctx.enqueue_create_buffer[input_type](a_size)
    var b_dev_buf = ctx.enqueue_create_buffer[input_type](b_size)
    var a_scale_buf = ctx.enqueue_create_buffer[.float32](a_scale_size)
    var b_scale_buf = ctx.enqueue_create_buffer[.float32](b_scale_size)
    var c_a_buf = ctx.enqueue_create_buffer[c_type](c_size)
    var c_b_buf = ctx.enqueue_create_buffer[c_type](c_size)
    var a_offsets_buf = ctx.enqueue_create_buffer[.uint32](num_active + 1)
    var expert_ids_buf = ctx.enqueue_create_buffer[.int32](num_active)

    var a_offsets_h_buf = ctx.enqueue_create_host_buffer[.uint32](
        num_active + 1
    )
    var expert_ids_h_buf = ctx.enqueue_create_host_buffer[.int32](num_active)
    var running = 0
    for i in range(num_active):
        a_offsets_h_buf[i] = UInt32(running)
        running += row_counts[i]
        expert_ids_h_buf[i] = Int32(i)
    a_offsets_h_buf[num_active] = UInt32(running)

    ctx.enqueue_copy(a_dev_buf, a_host)
    ctx.enqueue_copy(b_dev_buf, b_host)
    ctx.enqueue_copy(a_scale_buf, a_scale_host)
    ctx.enqueue_copy(b_scale_buf, b_scale_host)
    ctx.enqueue_copy(a_offsets_buf, a_offsets_h_buf)
    ctx.enqueue_copy(expert_ids_buf, expert_ids_h_buf)
    c_a_buf.enqueue_fill(0)
    c_b_buf.enqueue_fill(0)

    var a_dev = TileTensor(a_dev_buf, row_major(Coord(pad_rows, Idx[K])))
    var b_dev = TileTensor(b_dev_buf, row_major[num_experts, N, K]())
    var c_a_dev = TileTensor(c_a_buf, row_major(Coord(sum_rows, Idx[N])))
    var c_b_dev = TileTensor(c_b_buf, row_major(Coord(sum_rows, Idx[N])))
    var a_scale_dev = TileTensor(
        a_scale_buf, row_major(Coord(Idx[K_BLOCKS], sum_rows))
    )
    var b_scale_dev = TileTensor(
        b_scale_buf, row_major[num_experts, N_BLOCKS, K_BLOCKS]()
    )
    var a_offsets_dev = TileTensor(
        a_offsets_buf, row_major(Coord(num_active + 1))
    )
    var expert_ids_dev = TileTensor(
        expert_ids_buf, row_major(Coord(num_active))
    )

    # Prefill-sized metadata on the padded recv buffer -> grid.y = 32.
    blockwise_scaled_fp8_grouped_matmul_amd[
        transpose_b=transpose_b,
        N_SCALE=BLOCK_SCALE_N,
        K_SCALE=BLOCK_SCALE_K,
    ](
        c_a_dev,
        a_dev,
        b_dev,
        a_scale_dev,
        b_scale_dev,
        a_offsets_dev,
        expert_ids_dev,
        max_num_tokens_per_expert=prod_max_tokens,
        num_active_experts=num_active,
        ctx=ctx,
    )

    # Decode-bounded metadata on the same padded buffer -> grid.y = 1, each
    # CTA striding its expert's M-tiles.
    blockwise_scaled_fp8_grouped_matmul_amd[
        transpose_b=transpose_b,
        N_SCALE=BLOCK_SCALE_N,
        K_SCALE=BLOCK_SCALE_K,
    ](
        c_b_dev,
        a_dev,
        b_dev,
        a_scale_dev,
        b_scale_dev,
        a_offsets_dev,
        expert_ids_dev,
        max_num_tokens_per_expert=decode_max_tokens,
        num_active_experts=num_active,
        ctx=ctx,
    )

    # Effective grid.y mirrors the launcher (decode_grid_m_cap unset): the
    # metadata clamped to `a`'s own row count.
    var grid_y_prod = ceildiv(min(prod_max_tokens, pad_rows), BM)
    var grid_y_decode = ceildiv(min(decode_max_tokens, pad_rows), BM)
    print(
        "   grid=(x=",
        ceildiv(N, 64),
        ", y=",
        grid_y_prod,
        "/",
        grid_y_decode,
        ", z=",
        num_active,
        ")",
        sep="",
    )
    assert_equal(
        grid_y_prod,
        32,
        msg="prod_max_tokens=2048 on a padded buffer must launch grid.y=32",
    )
    assert_equal(
        grid_y_decode,
        1,
        msg="decode_max_tokens <= 64 must launch grid.y=1",
    )

    var c_a_host = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_b_host = ctx.enqueue_create_host_buffer[c_type](c_size)
    ctx.enqueue_copy(c_a_host, c_a_buf)
    ctx.enqueue_copy(c_b_host, c_b_buf)
    ctx.synchronize()

    for i in range(c_size):
        assert_equal(
            c_a_host[i],
            c_b_host[i],
            msg=(
                "grid.y packing changed the result -- the decode-bounded"
                " grid.y=1 launch must match the grid.y=32 launch"
                " bit-for-bit"
            ),
        )
        if isnan(c_a_host[i].cast[.float32]()):
            raise Error(
                "NaN output -- a poison pad row was read (grid.y=32 arm)"
            )
        if isnan(c_b_host[i].cast[.float32]()):
            raise Error(
                "NaN output -- a poison pad row was read (grid.y=1 arm)"
            )
    print("   PASSED")


def main() raises:
    with DeviceContext() as ctx:
        # Small shape vs CPU fp32 oracle: uneven rows, a zero-row expert, a
        # skipped -1 slot, and partial-tile counts (37, 63).
        test_grouped_blockwise_fp8_matmul_amd[
            .float32, num_experts=4, N=256, K=256, check_cpu=True
        ](ctx, [100, 0, 37, 63], [0, 1, 2, -1])
        test_grouped_blockwise_fp8_matmul_amd[
            .bfloat16, num_experts=4, N=256, K=256, check_cpu=True
        ](ctx, [100, 0, 37, 63], [0, 1, 2, -1])

        # GLM down proj decode band: N=6144, K=2048, ~64 total rows.
        test_grouped_blockwise_fp8_matmul_amd[
            .bfloat16, num_experts=8, N=6144, K=2048
        ](ctx, [10, 0, 16, 7, 15, 16], [0, 1, 2, 3, -1, 5])

        # GLM gate_up prefill band: N=4096, K=6144, 16384 total rows,
        # unevenly split with a zero-row expert and a skipped slot.
        test_grouped_blockwise_fp8_matmul_amd[
            .bfloat16, num_experts=8, N=4096, K=6144
        ](
            ctx,
            [3000, 0, 4096, 2011, 128, 5000, 1149, 1000],
            [0, 1, 2, 3, 4, 5, 6, -1],
        )

        # Dispatch wiring: fp32 scales route to the tiled kernel on AMD.
        test_grouped_blockwise_fp8_matmul_amd[
            .bfloat16, num_experts=8, N=6144, K=2048, via_dispatch=True
        ](ctx, [10, 0, 16, 7, 15, 16], [0, 1, 2, 3, 4, 5])

        # Compile-time routing proof: the SM100 and naive-AMD arms both
        # `comptime assert not static_grid_z`, so if the dispatch gate ever
        # stops routing AMD to `blockwise_scaled_fp8_grouped_matmul_amd` (the
        # only arm that honors `static_grid_z`), this fails to BUILD instead
        # of silently passing with naive's identical numbers.
        test_grouped_blockwise_fp8_matmul_amd[
            .bfloat16,
            num_experts=8,
            N=6144,
            K=2048,
            via_dispatch=True,
            static_grid_z=True,
        ](ctx, [10, 3, 16, 7, 15, 16, 0, 5], [0, 1, 2, 3, 4, 5, 6, 7])

        # static_grid_z must match the default bit-for-bit.
        test_grouped_static_grid_z[.bfloat16, num_experts=8, N=6144, K=2048](
            ctx, [10, 3, 16, 7, 15, 16, 0, 5], [0, 1, 2, 3, 4, 5, 6, 7]
        )

        # decode_grid_m_cap: prove it recovers correctness from a stale
        # (capture-frozen) max_num_tokens_per_expert that is wrong without it.
        test_grouped_decode_grid_m_cap_capture_safety[
            .bfloat16, num_experts=4, N=256, K=256
        ](ctx, [50, 0, 150, 30])

        # grid.y packing: decode-bounded (grid.y=1) must match prefill-sized
        # (grid.y=32) bit-for-bit; poison rows prove neither reads pads.
        test_grouped_grid_y_decode_bounded_bitidentity[
            .bfloat16, num_experts=4, N=256, K=256
        ](ctx, [10, 0, 7, 3], 2048, 16)
        test_grouped_grid_y_decode_bounded_bitidentity[
            .bfloat16, num_experts=8, N=4096, K=6144
        ](ctx, [3, 0, 5, 2, 1, 7, 4, 2], 2048, 16)

    print("\nAll grouped blockwise FP8 matmul tests passed!")
