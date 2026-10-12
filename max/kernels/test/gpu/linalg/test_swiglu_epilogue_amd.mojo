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
"""Correctness tests for the fused SwiGLU + MXFP8 epilogue in the preb kernel.

Runs the same grouped MXFP8 problem twice through
`PreShuffledBGroupedGEMM.launch`: once unfused (bf16 C, full N) and once with
`fuse_swiglu_mxfp8=True` (E4M3 C at N/2 plus E8M0 block scales). The unfused
result is the reference -- taking it from the kernel itself keeps the
preshuffled-B layout out of the comparison.

Each WN is covered because each takes a different path through the block-max
reduction. The dispatch cases then run M3's gate+up shape through
`block_scaled_grouped_matmul_amd_preb`, once per MXFP8 band of its tactic table,
with the scales written in the per-expert slot layout production uses.
Currently MI355X-only.
"""

from max.gpu.host import DeviceContext, HostBuffer
from max.gpu.host.info import MI355X
from std.math import align_up, ceildiv, exp, exp2
from std.random import random_ui64, seed

from layout import Coord, Idx, TileTensor, row_major
from linalg.arch.amd.block_scaled_mma import CDNA4F8F6F4MatrixFormat
from linalg.matmul.gpu.amd import (
    PreShuffledBGroupedGEMM,
    Shuffler,
    block_scaled_grouped_matmul_amd_preb,
)

comptime SWIGLU_ALPHA = Float32(1.702)
comptime SWIGLU_LIMIT = Float32(7.0)
comptime MX_BLOCK = 32
# E8M0 bias: the stored byte is the exponent plus this.
comptime E8M0_BIAS = 127


def _fill_random_fp8(buf: HostBuffer[.uint8], n: Int):
    """Fills with E4M3 bytes in a modest magnitude band.

    Avoids 0x7F / 0xFF (E4M3 NaN) and keeps the exponent small enough that the
    dot products do not all saturate the SwiGLU clamp, which would make the
    comparison vacuous.
    """
    for i in range(n):
        var mag = UInt8(0x30 + Int(random_ui64(0, 7)))  # ~0.5 .. 1.9
        var sign = UInt8(Int(random_ui64(0, 1)) << 7)
        buf[i] = mag | sign


def _swiglu(gate_f32: Float32, up_f32: Float32) -> Float32:
    """The OAI clamped SwiGLU the epilogue implements, bf16 round-trip and
    all -- see the epilogue's note on why that round-trip is load-bearing.
    """
    var gate = gate_f32.cast[DType.bfloat16]().cast[DType.float32]()
    var up = up_f32.cast[DType.bfloat16]().cast[DType.float32]()
    var g_c = min(gate, SWIGLU_LIMIT)
    var u_c = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    return g_c / (1.0 + exp(-(g_c * SWIGLU_ALPHA))) * (u_c + 1.0)


def _check_fused[
    N: Int
](
    c_ref_h: HostBuffer[.float32],
    c_q_h: HostBuffer[.float8_e4m3fn],
    c_sc_hh: HostBuffer[.uint8],
    total_tokens: Int,
) raises:
    """Holds the fused output to the SwiGLU of the unfused reference.

    `c_sc_hh` holds the E8M0 scales row-major, one byte per 32 outputs.
    """
    comptime OUT_N = N // 2
    comptime SCALES_PER_ROW = OUT_N // MX_BLOCK
    var worst = Float32(0)
    var shown = 0
    var nonzero = 0
    var distinct_hi = 0
    for r in range(total_tokens):
        # All 32 outputs in an E8M0 block share one scale, so error is only
        # meaningful relative to the block's own dynamic range.
        for blk in range(SCALES_PER_ROW):
            var block_max = Float32(0)
            for t in range(MX_BLOCK):
                var j = blk * MX_BLOCK + t
                var w = _swiglu(
                    c_ref_h[r * N + 2 * j], c_ref_h[r * N + 2 * j + 1]
                )
                block_max = max(block_max, abs(w))
            var e8m0 = Int(c_sc_hh[r * SCALES_PER_ROW + blk])
            var unit = exp2(Float32(e8m0 - E8M0_BIAS))
            for t in range(MX_BLOCK):
                var j = blk * MX_BLOCK + t
                var want = _swiglu(
                    c_ref_h[r * N + 2 * j], c_ref_h[r * N + 2 * j + 1]
                )
                var dq = Float32(c_q_h[r * OUT_N + j]) * unit
                var err = abs(dq - want) / max(block_max, Float32(1e-6))
                if err > 0.09 and shown < 8:
                    shown += 1
                    print(
                        "      MISMATCH r=",
                        r,
                        " j=",
                        j,
                        " gate=",
                        c_ref_h[r * N + 2 * j],
                        " up=",
                        c_ref_h[r * N + 2 * j + 1],
                        " want=",
                        want,
                        " dq=",
                        dq,
                        " blockmax=",
                        block_max,
                        " e8m0=",
                        e8m0,
                    )
                worst = max(worst, err)
                if abs(want) > 1e-3:
                    nonzero += 1
                if abs(want) > 0.5:
                    distinct_hi += 1

    # An all-zero or fully-clamped reference would pass any tolerance.
    if nonzero * 4 < total_tokens * OUT_N:
        raise Error(
            "degenerate reference: only "
            + String(nonzero)
            + " of "
            + String(total_tokens * OUT_N)
            + " outputs are nonzero -- the comparison would be vacuous"
        )
    if distinct_hi == 0:
        raise Error(
            "degenerate reference: no output exceeds 0.5, so the SwiGLU is"
            " operating only near zero"
        )

    print("     worst error (vs block max):", worst, " nonzero:", nonzero)
    # E4M3 carries 3 mantissa bits, so an element quantizes to within ~2^-4
    # of the block max, plus headroom for the max landing between powers of 2.
    if worst > 0.09:
        raise Error("fused SwiGLU+MXFP8 epilogue mismatch: " + String(worst))
    print("     PASS")


def _run_case[
    num_experts: Int,
    N: Int,
    K: Int,
    BM: Int,
    BN: Int,
    WN: Int,
    persistent: Bool,
](name: String, num_tokens_by_expert: List[Int], ctx: DeviceContext) raises:
    comptime packed_K = K  # MXFP8: one byte per element
    comptime scale_K = K // MX_BLOCK
    comptime OUT_N = N // 2
    comptime SCALES_PER_ROW = OUT_N // MX_BLOCK
    comptime WARPS_PER_BLOCK = 64 // WN

    var total_tokens = 0
    var max_tokens = 0
    for ne in num_tokens_by_expert:
        total_tokens += ne
        max_tokens = max(max_tokens, ne)
    var num_active = len(num_tokens_by_expert)
    print(
        "  ",
        "[persistent]" if persistent else "[direct]    ",
        name,
        " N=",
        N,
        " K=",
        K,
        " BM=",
        BM,
        " BN=",
        BN,
        " WN=",
        WN,
        " warps/E8M0-block=",
        WARPS_PER_BLOCK,
    )

    var a_h = ctx.enqueue_create_host_buffer[.uint8](total_tokens * packed_K)
    var b_h = ctx.enqueue_create_host_buffer[.uint8](num_experts * N * packed_K)
    var a_sc_h = ctx.enqueue_create_host_buffer[.uint8](total_tokens * scale_K)
    var b_sc_h = ctx.enqueue_create_host_buffer[.uint8](
        num_experts * N * scale_K
    )
    var a_off_h = ctx.enqueue_create_host_buffer[.uint32](num_active + 1)
    var eid_h = ctx.enqueue_create_host_buffer[.int32](num_active)
    ctx.synchronize()

    _fill_random_fp8(a_h, total_tokens * packed_K)
    _fill_random_fp8(b_h, num_experts * N * packed_K)
    # Constant 2^-1 on both operands puts the dot product's standard
    # deviation near 3, spanning the SwiGLU's linear region, its knee, and
    # enough tail to exercise the clamp at 7.
    for i in range(total_tokens * scale_K):
        a_sc_h[i] = UInt8(E8M0_BIAS - 1)
    for i in range(num_experts * N * scale_K):
        b_sc_h[i] = UInt8(E8M0_BIAS - 1)

    var acc = 0
    for e in range(num_active):
        a_off_h[e] = UInt32(acc)
        acc += num_tokens_by_expert[e]
        eid_h[e] = Int32(e)
    a_off_h[num_active] = UInt32(acc)
    var max_padded_M = align_up(max_tokens, 32)

    var a_d = ctx.enqueue_create_buffer[.uint8](total_tokens * packed_K)
    var b_d = ctx.enqueue_create_buffer[.uint8](num_experts * N * packed_K)
    var b_pre_d = ctx.enqueue_create_buffer[.uint8](num_experts * N * packed_K)
    var a_sc_d = ctx.enqueue_create_buffer[.uint8](total_tokens * scale_K)
    var a_sc_pre_d = ctx.enqueue_create_buffer[.uint8](
        num_experts * max_padded_M * scale_K
    )
    var b_sc_pre_d = ctx.enqueue_create_buffer[.uint8](
        num_experts * N * scale_K
    )
    var a_off_d = ctx.enqueue_create_buffer[.uint32](num_active + 1)
    var eid_d = ctx.enqueue_create_buffer[.int32](num_active)
    var c_ref_d = ctx.enqueue_create_buffer[.float32](total_tokens * N)
    var c_q_d = ctx.enqueue_create_buffer[.float8_e4m3fn](total_tokens * OUT_N)
    var c_sc_d = ctx.enqueue_create_buffer[.uint8](
        total_tokens * SCALES_PER_ROW
    )
    c_ref_d.enqueue_fill(Float32(0.0))
    c_q_d.enqueue_fill(Float8_e4m3fn(0.0))
    c_sc_d.enqueue_fill(UInt8(0))

    ctx.enqueue_copy(a_d, a_h)
    ctx.enqueue_copy(b_d, b_h)
    ctx.enqueue_copy(a_sc_d, a_sc_h)
    ctx.enqueue_copy(a_off_d, a_off_h)
    ctx.enqueue_copy(eid_d, eid_h)

    Shuffler[num_experts].preshuffle_b_5d[N=N, K_BYTES=packed_K](
        TileTensor(b_d, row_major[num_experts, N, packed_K]()).as_imm(),
        TileTensor(
            b_pre_d,
            Shuffler[num_experts].b_5d_grouped_layout[N=N, K_BYTES=packed_K],
        ),
        ctx,
    )
    Shuffler[1].preshuffle_grouped_scale_4d_gpu[K_SCALES=scale_K](
        TileTensor(
            a_sc_d, row_major(Coord(total_tokens, Idx[scale_K]))
        ).as_imm(),
        TileTensor(
            a_sc_pre_d,
            row_major(Coord(num_experts * max_padded_M, Idx[scale_K])),
        ),
        TileTensor(a_off_d, row_major(Coord(num_active + 1))).as_imm(),
        num_active,
        max_tokens,
        ctx.default_device_info.sm_count * 2,
        ctx,
    )
    var b_sc_pre_h = ctx.enqueue_create_host_buffer[.uint8](
        num_experts * N * scale_K
    )
    ctx.synchronize()
    Shuffler[num_experts].preshuffle_scale_4d[MN=N, K_SCALES=scale_K](
        TileTensor(
            b_sc_h.unsafe_ptr(),
            row_major(Coord(Idx[num_experts], Idx[N], Idx[scale_K])),
        ),
        b_sc_pre_h,
    )
    ctx.enqueue_copy(b_sc_pre_d, b_sc_pre_h)

    var a_tt = TileTensor(
        a_d, row_major(Coord(total_tokens, Idx[packed_K]))
    ).as_imm()
    var b_pre_tt = TileTensor(
        b_pre_d, row_major[num_experts, N * packed_K]()
    ).as_imm()
    var a_sc_tt = TileTensor(
        a_sc_pre_d.unsafe_ptr().bitcast[Float8_e8m0fnu](),
        row_major(Coord(num_experts * max_padded_M, Idx[scale_K])),
    ).as_imm()
    var b_sc_tt = TileTensor(
        b_sc_pre_d.unsafe_ptr().bitcast[Float8_e8m0fnu](),
        row_major[num_experts, N, scale_K](),
    ).as_imm()
    var a_off_tt = TileTensor(a_off_d, row_major(Coord(num_active + 1)))
    var eid_tt = TileTensor(eid_d, row_major(Coord(num_active)))

    # `matrix_format` defaults to FP4, so calling `launch` directly has to
    # name MXFP8 or the kernel reads these bytes as packed FP4 pairs.
    comptime GEMM = PreShuffledBGroupedGEMM[
        cu_count=ctx.default_device_info.sm_count,
        matrix_format=CDNA4F8F6F4MatrixFormat.FLOAT8_E4M3,
    ]

    # Reference arm: unfused, bf16 C at the full matmul N.
    GEMM.launch[BM=BM, BN=BN, BK_ELEMS=256, WN=WN, persistent=persistent](
        TileTensor(c_ref_d, row_major(Coord(total_tokens, Idx[N]))),
        a_tt,
        b_pre_tt,
        a_sc_tt,
        b_sc_tt,
        a_off_tt,
        eid_tt,
        max_tokens,
        num_active,
        ctx,
    )

    # Fused arm: E4M3 C at N/2 plus row-major E8M0 block scales.
    GEMM.launch[
        BM=BM,
        BN=BN,
        BK_ELEMS=256,
        WN=WN,
        persistent=persistent,
        fuse_swiglu_mxfp8=True,
    ](
        TileTensor(c_q_d, row_major(Coord(total_tokens, Idx[OUT_N]))),
        a_tt,
        b_pre_tt,
        a_sc_tt,
        b_sc_tt,
        a_off_tt,
        eid_tt,
        max_tokens,
        num_active,
        ctx,
        -1,
        c_sc_d.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        UInt32(0),
    )

    var c_ref_h = ctx.enqueue_create_host_buffer[.float32](total_tokens * N)
    var c_q_h = ctx.enqueue_create_host_buffer[.float8_e4m3fn](
        total_tokens * OUT_N
    )
    var c_sc_hh = ctx.enqueue_create_host_buffer[.uint8](
        total_tokens * SCALES_PER_ROW
    )
    ctx.enqueue_copy(c_ref_h, c_ref_d)
    ctx.enqueue_copy(c_q_h, c_q_d)
    ctx.enqueue_copy(c_sc_hh, c_sc_d)
    ctx.synchronize()

    _check_fused[N](c_ref_h, c_q_h, c_sc_hh, total_tokens)

    _ = a_d^
    _ = b_d^
    _ = b_pre_d^
    _ = a_sc_d^
    _ = a_sc_pre_d^
    _ = b_sc_pre_d^
    _ = a_off_d^
    _ = eid_d^
    _ = c_ref_d^
    _ = c_q_d^
    _ = c_sc_d^


def _run_dispatch_case[
    num_experts: Int, N: Int, K: Int
](
    name: String,
    num_tokens_by_expert: List[Int],
    decode_grid_m_cap: Int,
    ctx: DeviceContext,
) raises:
    """Runs the fused epilogue through the production dispatcher.

    The token count picks the tactic, and the scales land in the per-expert
    slot layout the down projection reads, as M3 always writes them. The slot
    stride is wider than the tokens need, so a writer that strides by the
    runtime max instead of the stride it is given puts every expert after the
    first in the wrong slot. Expert ids are reversed so a slot and its expert
    id differ.
    """
    comptime packed_K = K
    comptime scale_K = K // MX_BLOCK
    comptime OUT_N = N // 2
    comptime SCALES_PER_ROW = OUT_N // MX_BLOCK

    var total_tokens = 0
    var max_tokens = 0
    for ne in num_tokens_by_expert:
        total_tokens += ne
        max_tokens = max(max_tokens, ne)
    var num_active = len(num_tokens_by_expert)
    var max_padded_M = align_up(max_tokens, 32)
    var slot_stride = max_padded_M + 32
    print(
        "  ",
        name,
        " N=",
        N,
        " K=",
        K,
        " tokens=",
        total_tokens,
        " decode_cap=",
        decode_grid_m_cap,
        " slot_stride=",
        slot_stride,
    )

    var a_h = ctx.enqueue_create_host_buffer[.uint8](total_tokens * packed_K)
    var b_h = ctx.enqueue_create_host_buffer[.uint8](num_experts * N * packed_K)
    var a_sc_h = ctx.enqueue_create_host_buffer[.uint8](total_tokens * scale_K)
    var b_sc_h = ctx.enqueue_create_host_buffer[.uint8](
        num_experts * N * scale_K
    )
    var a_off_h = ctx.enqueue_create_host_buffer[.uint32](num_active + 1)
    var eid_h = ctx.enqueue_create_host_buffer[.int32](num_active)
    ctx.synchronize()

    _fill_random_fp8(a_h, total_tokens * packed_K)
    _fill_random_fp8(b_h, num_experts * N * packed_K)
    # A 2^-4 scale product keeps the K=6144 dot product's standard deviation
    # near 2.5, the same regime the K=256 cases reach with 2^-2.
    for i in range(total_tokens * scale_K):
        a_sc_h[i] = UInt8(E8M0_BIAS - 2)
    for i in range(num_experts * N * scale_K):
        b_sc_h[i] = UInt8(E8M0_BIAS - 2)

    var acc = 0
    for e in range(num_active):
        a_off_h[e] = UInt32(acc)
        acc += num_tokens_by_expert[e]
        eid_h[e] = Int32(num_active - 1 - e)
    a_off_h[num_active] = UInt32(acc)

    var a_d = ctx.enqueue_create_buffer[.uint8](total_tokens * packed_K)
    var b_d = ctx.enqueue_create_buffer[.uint8](num_experts * N * packed_K)
    var b_pre_d = ctx.enqueue_create_buffer[.uint8](num_experts * N * packed_K)
    var a_sc_d = ctx.enqueue_create_buffer[.uint8](total_tokens * scale_K)
    var a_sc_pre_d = ctx.enqueue_create_buffer[.uint8](
        num_experts * max_padded_M * scale_K
    )
    var b_sc_pre_d = ctx.enqueue_create_buffer[.uint8](
        num_experts * N * scale_K
    )
    var a_off_d = ctx.enqueue_create_buffer[.uint32](num_active + 1)
    var eid_d = ctx.enqueue_create_buffer[.int32](num_active)
    var c_ref_d = ctx.enqueue_create_buffer[.float32](total_tokens * N)
    var c_q_d = ctx.enqueue_create_buffer[.float8_e4m3fn](total_tokens * OUT_N)
    var slot_bytes = num_active * slot_stride * SCALES_PER_ROW
    var c_sc_slot_d = ctx.enqueue_create_buffer[.uint8](slot_bytes)
    c_ref_d.enqueue_fill(Float32(0.0))
    c_q_d.enqueue_fill(Float8_e4m3fn(0.0))
    # 0xFF is E8M0 NaN; any byte still holding it after the launch was never
    # written, which the gather below rejects.
    c_sc_slot_d.enqueue_fill(UInt8(0xFF))

    ctx.enqueue_copy(a_d, a_h)
    ctx.enqueue_copy(b_d, b_h)
    ctx.enqueue_copy(a_sc_d, a_sc_h)
    ctx.enqueue_copy(a_off_d, a_off_h)
    ctx.enqueue_copy(eid_d, eid_h)

    Shuffler[num_experts].preshuffle_b_5d[N=N, K_BYTES=packed_K](
        TileTensor(b_d, row_major[num_experts, N, packed_K]()).as_imm(),
        TileTensor(
            b_pre_d,
            Shuffler[num_experts].b_5d_grouped_layout[N=N, K_BYTES=packed_K],
        ),
        ctx,
    )
    Shuffler[1].preshuffle_grouped_scale_4d_gpu[K_SCALES=scale_K](
        TileTensor(
            a_sc_d, row_major(Coord(total_tokens, Idx[scale_K]))
        ).as_imm(),
        TileTensor(
            a_sc_pre_d,
            row_major(Coord(num_experts * max_padded_M, Idx[scale_K])),
        ),
        TileTensor(a_off_d, row_major(Coord(num_active + 1))).as_imm(),
        num_active,
        max_tokens,
        ctx.default_device_info.sm_count * 2,
        ctx,
    )
    var b_sc_pre_h = ctx.enqueue_create_host_buffer[.uint8](
        num_experts * N * scale_K
    )
    ctx.synchronize()
    Shuffler[num_experts].preshuffle_scale_4d[MN=N, K_SCALES=scale_K](
        TileTensor(
            b_sc_h.unsafe_ptr(),
            row_major(Coord(Idx[num_experts], Idx[N], Idx[scale_K])),
        ),
        b_sc_pre_h,
    )
    ctx.enqueue_copy(b_sc_pre_d, b_sc_pre_h)
    # Only the preshuffled copy is read from here on; freeing the raw B keeps
    # the prefill case inside the test's device-memory budget.
    _ = b_d^

    var a_tt = TileTensor(
        a_d, row_major(Coord(total_tokens, Idx[packed_K]))
    ).as_imm()
    var b_pre_tt = TileTensor(
        b_pre_d, row_major[num_experts, N * packed_K]()
    ).as_imm()
    var a_sc_tt = TileTensor(
        a_sc_pre_d.unsafe_ptr().bitcast[Float8_e8m0fnu](),
        row_major(Coord(num_experts * max_padded_M, Idx[scale_K])),
    ).as_imm()
    var b_sc_tt = TileTensor(
        b_sc_pre_d.unsafe_ptr().bitcast[Float8_e8m0fnu](),
        row_major[num_experts, N, scale_K](),
    ).as_imm()
    var a_off_tt = TileTensor(
        a_off_d, row_major(Coord(num_active + 1))
    ).as_imm()
    var eid_tt = TileTensor(eid_d, row_major(Coord(num_active))).as_imm()

    # Reference arm: the unfused kernel at a fixed, separately tested tactic,
    # so the reference does not move with the tactic under test.
    comptime GEMM = PreShuffledBGroupedGEMM[
        cu_count=ctx.default_device_info.sm_count,
        matrix_format=CDNA4F8F6F4MatrixFormat.FLOAT8_E4M3,
    ]
    GEMM.launch[BM=64, BN=128, BK_ELEMS=256, WN=64, persistent=True](
        TileTensor(c_ref_d, row_major(Coord(total_tokens, Idx[N]))),
        a_tt,
        b_pre_tt,
        a_sc_tt,
        b_sc_tt,
        a_off_tt,
        eid_tt,
        max_tokens,
        num_active,
        ctx,
    )

    block_scaled_grouped_matmul_amd_preb[fuse_swiglu_mxfp8=True](
        TileTensor(c_q_d, row_major(Coord(total_tokens, Idx[OUT_N]))),
        a_tt,
        b_pre_tt,
        a_sc_tt,
        b_sc_tt,
        a_off_tt,
        eid_tt,
        max_tokens,
        num_active,
        ctx,
        estimated_total_m=total_tokens,
        decode_grid_m_cap=decode_grid_m_cap,
        decode_grid_m_rows=max_tokens,
        c_scales_ptr=c_sc_slot_d.unsafe_ptr().unsafe_origin_cast[
            MutAnyOrigin
        ](),
        c_scales_max_padded_m=slot_stride,
    )

    var c_ref_h = ctx.enqueue_create_host_buffer[.float32](total_tokens * N)
    var c_q_h = ctx.enqueue_create_host_buffer[.float8_e4m3fn](
        total_tokens * OUT_N
    )
    var c_sc_slot_h = ctx.enqueue_create_host_buffer[.uint8](slot_bytes)
    var c_sc_hh = ctx.enqueue_create_host_buffer[.uint8](
        total_tokens * SCALES_PER_ROW
    )
    ctx.enqueue_copy(c_ref_h, c_ref_d)
    ctx.enqueue_copy(c_q_h, c_q_d)
    ctx.enqueue_copy(c_sc_slot_h, c_sc_slot_d)
    ctx.synchronize()

    # Read each scale back where the down projection will look for it.
    for e in range(num_active):
        for row in range(num_tokens_by_expert[e]):
            for blk in range(SCALES_PER_ROW):
                var off = Shuffler[1].scale_4d_slot_byte_off[SCALES_PER_ROW](
                    e, row, blk, slot_stride
                )
                var scale_byte = c_sc_slot_h[off]
                if scale_byte == 0xFF:
                    raise Error(
                        "scale never written: slot="
                        + String(e)
                        + " row="
                        + String(row)
                        + " block="
                        + String(blk)
                    )
                c_sc_hh[
                    (Int(a_off_h[e]) + row) * SCALES_PER_ROW + blk
                ] = scale_byte

    _check_fused[N](c_ref_h, c_q_h, c_sc_hh, total_tokens)

    _ = a_d^
    _ = b_pre_d^
    _ = a_sc_d^
    _ = a_sc_pre_d^
    _ = b_sc_pre_d^
    _ = a_off_d^
    _ = eid_d^
    _ = c_ref_d^
    _ = c_q_d^
    _ = c_sc_slot_d^


def main() raises:
    seed(0)
    with DeviceContext() as ctx:
        comptime assert (
            ctx.default_device_info == MI355X
        ), "the preb fused-SwiGLU epilogue is MI355X-only"
        print("fused SwiGLU + MXFP8 epilogue")
        # WN=64: one warp owns a 32-column E8M0 block (butterfly only).
        _run_case[2, 256, 256, 64, 128, 64, True](
            "WN64 persistent", [64, 96], ctx
        )
        _run_case[2, 256, 256, 64, 128, 64, False](
            "WN64 direct    ", [64, 96], ctx
        )
        # WN=32: two warps share a block, adding the LDS cross-warp reduction.
        _run_case[2, 256, 256, 64, 128, 32, True](
            "WN32 persistent", [64, 96], ctx
        )
        _run_case[2, 256, 256, 64, 128, 32, False](
            "WN32 direct    ", [64, 96], ctx
        )
        # WN=16: four warps per block -- widest cross-warp fold, and the band
        # production decode dispatches. BN must supply at least
        # WARPS_PER_BLOCK N-warps; BN=128 gives eight.
        _run_case[2, 256, 256, 64, 128, 16, True](
            "WN16 persistent", [64, 96], ctx
        )
        _run_case[2, 256, 256, 64, 128, 16, False](
            "WN16 direct    ", [64, 96], ctx
        )

        # M3 gate+up (N = K = 6144) through the dispatcher, one case per
        # MXFP8 band of its tactic table. Decode: BM=16, BN=64, WN=16, so BN/WN
        # gives exactly WARPS_PER_BLOCK N-warps; odd counts exercise the BM=16
        # cell straddle.
        print("M3 gate+up via block_scaled_grouped_matmul_amd_preb")
        _run_dispatch_case[2, 6144, 6144](
            "decode (capped direct)", [5, 11], 64, ctx
        )
        _run_dispatch_case[2, 6144, 6144]("etm <= 256", [40, 72], -1, ctx)
        _run_dispatch_case[2, 6144, 6144]("etm <= 512", [150, 250], -1, ctx)
        _run_dispatch_case[2, 6144, 6144]("etm <= 2048", [300, 500], -1, ctx)
        _run_dispatch_case[2, 6144, 6144](
            "prefill (BM=128, BN=256)", [1000, 1100], -1, ctx
        )
