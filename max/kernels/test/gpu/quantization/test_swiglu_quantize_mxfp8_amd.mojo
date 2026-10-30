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
"""Tests the AMD CDNA4 (MI355X) branch of `swiglu_quantize_dynamic_block_scaled`.

The fused SiTU + MXFP8 kernel must produce the exact bytes of the chain it
replaces: a `situ` elementwise kernel writing the bfloat16 activation, then
`quantize_mx_amd`. That gate catches a missing bfloat16 rounding, a swapped
gate/up, a launch geometry computed on 2N, and a skipped scale column; the
sentinel-filled buffers and the guard tails after them catch unwritten or
out-of-bounds stores.

A second, independent host fp64 oracle (SiTU in fp64 -> bfloat16 round ->
a block quant written here, mirroring the kernel's fp32 scale arithmetic)
guards the fused kernel's math itself: the exact-byte gate above would pass
if the shared `situ` thin function were wrong in both arms. Scales must
match exactly except at most 0.1% of blocks, each off by one E8M0 step
(fp64 tanh vs the fp32 polynomial can flip a bfloat16 rounding at a
block's max element), and data bytes within one E4M3 code step, nine
under a one-step scale shift.

Two mutants must FAIL, proving those gates have teeth: the same kernel run
with `limit = 0` (linear_beta None) against the linear_beta = 25 reference,
and a host reference built without the bfloat16 rounding.

The public launcher has its own negative case: `swiglu_quantize_mx_amd`
called directly with a wrongly shaped scales tile raises before the kernel
is enqueued, because the kernel stores into the scales tile unchecked.
"""

from std.math import ceildiv, divmod, exp, isfinite, tanh
from std.memory import bitcast
from std.random import random_float64, seed
from std.sys import align_of
from std.testing import assert_raises, assert_true
from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from layout import Idx, TileTensor, row_major
from linalg.block_scaled_quantization import (
    quantize_mx_amd,
    swiglu_quantize_dynamic_block_scaled,
    swiglu_quantize_mx_amd,
)
from linalg.fp4_utils import MXFP8_SF_VECTOR_SIZE
from linalg.swiglu_activation import (
    apply_swiglu_activation,
    get_swiglu_activation,
    swiglu_activation_type,
)

comptime SEED = 0x5EED
comptime BETA = Float32(4.0)
comptime LINEAR_BETA = Float32(25.0)
# E4M3's largest finite magnitude. The kernel multiplies each block's max
# by fp32(1 / 448); this oracle mirrors that multiply, not a divide by 448.
comptime E4M3_MAXABS_RECIP = Float32(1.0) / Float32(448.0)
# Guard tails after each output buffer: a geometry computed on 2N or a
# doubled scale column writes past its tile, into the tail.
comptime OUT_GUARD_BYTES = 32
comptime SF_GUARD_BYTES = 4


def _block_magnitude(block_idx: Int) -> Float64:
    """Returns a power of two in [2^-10, 2^10], cycling over the block index.

    Every test needs the ladder: at a flat magnitude a straddling lane group
    picks a scale that hides a wrong-block bug by accident.
    """
    return Float64(1 << (block_idx % 21)) / 1024.0


def _fill_gate_up(mut values: List[Float32], m: Int, gate_N: Int):
    """Builds the deterministic BF16 `[gate | up]` input, widened to f32.

    Signed values on the per-block magnitude ladder; every 13th column's
    gate saturates the tanh (|gate / beta| > 9) and drives sigmoid toward
    0/1; every 17th column's up passes the soft saturation (|up| > 25);
    every 5th row's first block is all zero (a zero-max block); row 7's
    second block holds one outlier against 31 tiny values.
    """
    seed(SEED)
    values.clear()
    for row in range(m):
        for col in range(2 * gate_N):
            var half, in_col = divmod(col, gate_N)
            var mag = _block_magnitude(in_col // MXFP8_SF_VECTOR_SIZE)
            var v = random_float64(-4.0, 4.0) * mag
            if in_col % 13 == 0:
                v = 40.0 if in_col % 26 == 0 else -40.0
            if half == 1 and in_col % 17 == 0:
                v = 100.0 if in_col % 34 == 0 else -100.0
            if row % 5 == 0 and in_col < MXFP8_SF_VECTOR_SIZE:
                v = 0.0
            if row == 7 and in_col // MXFP8_SF_VECTOR_SIZE == 1:
                v = (
                    2.0**10 if in_col % MXFP8_SF_VECTOR_SIZE
                    == 31 else 2.0**-20
                )
            values.append(v.cast[.bfloat16]().cast[.float32]())


def _situ_fp64(
    gate: Float64, up: Float64, beta: Float64, linear_beta: Float64
) -> Float64:
    """SiTU in fp64, in the graph op order the kernel mirrors in fp32."""
    var s = beta * tanh(gate / beta)
    s = s * (Float64(1.0) / (Float64(1.0) + exp(-gate)))
    var u = up
    if linear_beta != 0.0:
        u = linear_beta * tanh(up / linear_beta)
    return s * u


def _host_situ_fp64[
    N: Int, M: Int
](values: List[Float32], beta: Float64, linear_beta: Float64) raises -> List[
    Float64
]:
    """Returns the fp64 SiTU over the whole `[gate | up]` input, unrounded."""
    var h = List[Float64]()
    for row in range(M):
        for col in range(N):
            var gate = Float64(values[row * 2 * N + col])
            var up = Float64(values[row * 2 * N + N + col])
            h.append(_situ_fp64(gate, up, beta, linear_beta))
    return h^


def _situ_activation[
    activation: swiglu_activation_type, M: Int, N: Int
](
    h: TileTensor[.bfloat16, type_of(row_major[M, N]()), MutUntrackedOrigin],
    gate_up: TileTensor[
        .bfloat16, type_of(row_major[M, 2 * N]()), ImmUntrackedOrigin
    ],
):
    """Elementwise `h = activation(gate_up)`, bfloat16 rounded: the arm the
    fused kernel replaces."""
    # The 8-wide accesses claim only element alignment, not the 16-byte
    # vector width that GPU loads and stores default to.
    comptime alignment = align_of[BFloat16]()
    var i = global_idx.x
    if i >= M * (N // 8):
        return
    var row, c = divmod(i, N // 8)
    var h_vec = h.vectorize[1, 8]()
    var gate_up_vec = gate_up.vectorize[1, 8]()
    h_vec.store[alignment=alignment](
        (row, c),
        apply_swiglu_activation[activation](
            gate_up_vec.load[alignment=alignment]((row, c)),
            gate_up_vec.load[alignment=alignment]((row, c + N // 8)),
        ),
    )


def _upload_gate_up[
    M: Int, GATE_N: Int
](ctx: DeviceContext, values: List[Float32]) raises -> DeviceBuffer[.bfloat16]:
    """Uploads the host `values` as a BF16 `[M, 2 * GATE_N]` device buffer."""
    var gate_up_dev = ctx.enqueue_create_buffer[.bfloat16](M * 2 * GATE_N)
    with gate_up_dev.map_to_host() as host:
        for i in range(M * 2 * GATE_N):
            host[i] = values[i].cast[.bfloat16]()
    return gate_up_dev


def _run_unfused[
    N: Int, M: Int, activation: swiglu_activation_type
](
    ctx: DeviceContext,
    values: List[Float32],
    mut data: List[UInt8],
    mut scales: List[UInt8],
) raises:
    """The chain the fused kernel replaces: the `situ` elementwise kernel
    writing bfloat16, then `quantize_mx_amd`."""
    comptime scale_N = N // MXFP8_SF_VECTOR_SIZE
    var gate_up_dev = _upload_gate_up[M, N](ctx, values)
    var h_dev = ctx.enqueue_create_buffer[.bfloat16](M * N)
    var gate_up_tt = TileTensor(gate_up_dev, row_major((Idx[M], Idx[2 * N])))
    var h_tt = TileTensor(h_dev, row_major((Idx[M], Idx[N])))
    ctx.enqueue_function[_situ_activation[activation, M, N]](
        h_tt,
        gate_up_tt.as_imm(),
        grid_dim=ceildiv(M * (N // 8), 256),
        block_dim=256,
    )

    var data_dev = ctx.enqueue_create_buffer[.float8_e4m3fn](M * N)
    var scales_dev = ctx.enqueue_create_buffer[.float8_e8m0fnu](M * scale_N)
    var data_tt = TileTensor(data_dev, row_major((Idx[M], Idx[N])))
    var scales_tt = TileTensor(scales_dev, row_major((Idx[M], Idx[scale_N])))
    quantize_mx_amd(ctx, data_tt, scales_tt, h_tt)

    var data_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](M * N)
    var scales_host = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        M * scale_N
    )
    ctx.enqueue_copy(data_host, data_dev)
    ctx.enqueue_copy(scales_host, scales_dev)
    ctx.synchronize()

    data.clear()
    for i in range(M * N):
        data.append(bitcast[.uint8](data_host[i]))
    scales.clear()
    for i in range(M * scale_N):
        scales.append(bitcast[.uint8](scales_host[i]))


def _run_fused[
    N: Int, M: Int, activation: swiglu_activation_type
](
    ctx: DeviceContext,
    values: List[Float32],
    mut data: List[UInt8],
    mut scales: List[UInt8],
) raises:
    """The fused kernel through the same dispatcher the graph op calls, into
    sentinel-filled buffers with guard tails."""
    comptime scale_N = N // MXFP8_SF_VECTOR_SIZE
    var gate_up_dev = _upload_gate_up[M, N](ctx, values)

    var out_dev = ctx.enqueue_create_buffer[.uint8](M * N + OUT_GUARD_BYTES)
    out_dev.enqueue_fill(0xAB)
    var sf_dev = ctx.enqueue_create_buffer[.uint8](M * scale_N + SF_GUARD_BYTES)
    sf_dev.enqueue_fill(0xCD)

    swiglu_quantize_dynamic_block_scaled[
        SF_VECTOR_SIZE=MXFP8_SF_VECTOR_SIZE, activation=activation
    ](
        TileTensor(
            out_dev.unsafe_ptr().unsafe_bitcast[Scalar[.float8_e4m3fn]](),
            row_major((Idx[M], Idx[N])),
        ),
        TileTensor(
            sf_dev.unsafe_ptr().unsafe_bitcast[Scalar[.float8_e8m0fnu]](),
            row_major((Idx[M], Idx[scale_N])),
        ),
        TileTensor(gate_up_dev, row_major((Idx[M], Idx[2 * N]))),
        Float32(1.0),
        ctx,
    )

    var raw_host = ctx.enqueue_create_host_buffer[.uint8](
        M * N + OUT_GUARD_BYTES
    )
    ctx.enqueue_copy(raw_host, out_dev)
    ctx.synchronize()
    for i in range(OUT_GUARD_BYTES):
        assert_true(
            raw_host[M * N + i] == 0xAB,
            (
                "an output guard tail byte changed: the kernel writes past"
                " [M, N], a launch geometry computed on 2N"
            ),
        )
    data.clear()
    for i in range(M * N):
        data.append(raw_host[i])

    var raw_sf_host = ctx.enqueue_create_host_buffer[.uint8](
        M * scale_N + SF_GUARD_BYTES
    )
    ctx.enqueue_copy(raw_sf_host, sf_dev)
    ctx.synchronize()
    for i in range(SF_GUARD_BYTES):
        assert_true(
            raw_sf_host[M * scale_N + i] == 0xCD,
            (
                "a scales guard tail byte changed: the kernel writes past"
                " [M, N // 32], a doubled scale column"
            ),
        )
    scales.clear()
    for i in range(M * scale_N):
        scales.append(raw_sf_host[i])


@inline(.always)
def _signed_e4m3_code(byte: UInt8) -> Int:
    """The E4M3 byte as a signed code: sign-magnitude, so adjacent codes of
    the same sign differ by 1 and +0/-0 (0x00/0x80) compare equal."""
    var magnitude = Int(byte & 0x7F)
    if byte & 0x80:
        magnitude = -magnitude
    return magnitude


def _host_quantize(
    h: List[Float32],
    m: Int,
    N: Int,
    mut data: List[UInt8],
    mut scales: List[UInt8],
) raises:
    """The host MXFP8 block quant, mirroring the kernel's fp32 arithmetic:
    `group_max` in fp32, the E8M0 scale from `group_max * fp32(1/448)`, the
    data scaled by the reciprocal. Written here rather than shared with the
    kernel, so it stays an independent oracle.

    Dead blocks (where the device's approximate reciprocal of the E8M0
    scale is non-finite and the kernel zeroes the data) are not modeled;
    the test input must avoid them.
    """
    var scale_N = N // MXFP8_SF_VECTOR_SIZE
    for row in range(m):
        for block in range(scale_N):
            var base = row * N + block * MXFP8_SF_VECTOR_SIZE
            var group_max = Float32(0)
            for e in range(MXFP8_SF_VECTOR_SIZE):
                group_max = max(group_max, abs(h[base + e]))

            var e8m0 = (group_max * E4M3_MAXABS_RECIP).cast[.float8_e8m0fnu]()
            var multiplier = Float32(0.0)
            if group_max != 0.0:
                multiplier = Float32(1.0) / e8m0.cast[.float32]()
                assert_true(
                    isfinite(multiplier),
                    (
                        "test input produced a dead MX block; the host oracle"
                        " does not model that path"
                    ),
                )
            scales.append(bitcast[.uint8](e8m0))
            for e in range(MXFP8_SF_VECTOR_SIZE):
                data.append(
                    bitcast[.uint8](
                        (h[base + e] * multiplier).cast[.float8_e4m3fn]()
                    )
                )


def _count_byte_diffs[
    N: Int, M: Int
](
    label: StaticString,
    kind: StaticString,
    want: List[UInt8],
    got: List[UInt8],
    mut diffs: Int,
) raises:
    """Counts differing bytes into `diffs`, printing the first 8 overall."""
    for i in range(len(want)):
        if want[i] != got[i]:
            diffs += 1
            if diffs <= 8:
                print(
                    label,
                    "M =",
                    M,
                    "N =",
                    N,
                    kind,
                    i,
                    "fused",
                    got[i],
                    "vs unfused",
                    want[i],
                )


def test_fused_matches_unfused[
    N: Int, M: Int, activation: swiglu_activation_type
](ctx: DeviceContext, label: StaticString) raises:
    """Fused bytes == the unfused chain's bytes, data and scales both."""
    var values = List[Float32]()
    _fill_gate_up(values, M, N)

    var ref_data = List[UInt8]()
    var ref_scales = List[UInt8]()
    _run_unfused[N, M, activation](ctx, values, ref_data, ref_scales)
    var got_data = List[UInt8]()
    var got_scales = List[UInt8]()
    _run_fused[N, M, activation](ctx, values, got_data, got_scales)

    var diffs = 0
    _count_byte_diffs[N, M](label, "output byte", ref_data, got_data, diffs)
    _count_byte_diffs[N, M](label, "scale byte", ref_scales, got_scales, diffs)
    assert_true(
        diffs == 0,
        String(label)
        + ": fused differs from the unfused chain in "
        + String(diffs)
        + " bytes",
    )
    print(label, ": fused matches unfused, M =", M, "N =", N)


def test_fused_matches_host_fp64[
    N: Int,
    M: Int,
    activation: swiglu_activation_type,
    beta: Float32,
    linear_beta: Float32,
](ctx: DeviceContext) raises:
    """Fused bytes vs the independent host fp64 oracle, with the bounded
    tolerances a fp64-vs-fp32 transcendental gap needs."""
    var values = List[Float32]()
    _fill_gate_up(values, M, N)

    var got_data = List[UInt8]()
    var got_scales = List[UInt8]()
    _run_fused[N, M, activation](ctx, values, got_data, got_scales)

    # The fp64 activation, rounded to bf16 as the fused kernel does.
    var h = _host_situ_fp64[N, M](values, Float64(beta), Float64(linear_beta))
    var h_bf16 = List[Float32]()
    for v in h:
        h_bf16.append(v.cast[.bfloat16]().cast[.float32]())

    var want_data = List[UInt8]()
    var want_scales = List[UInt8]()
    _host_quantize(h_bf16, M, N, want_data, want_scales)

    comptime scale_N = N // MXFP8_SF_VECTOR_SIZE
    var num_blocks = M * scale_N
    var scale_mismatches = 0
    var byte_mismatches = 0
    var dequant_worst = Float32(0)
    for row in range(M):
        for block in range(scale_N):
            var sf_idx = row * scale_N + block
            var got_scale = got_scales[sf_idx]
            var want_scale = want_scales[sf_idx]
            if got_scale != want_scale:
                scale_mismatches += 1
                assert_true(
                    abs(Int(got_scale) - Int(want_scale)) == 1,
                    String("scale at row ")
                    + String(row)
                    + " block "
                    + String(block)
                    + " is more than one E8M0 step from the fp64 oracle: "
                    + String(got_scale)
                    + " vs "
                    + String(want_scale),
                )
            for e in range(MXFP8_SF_VECTOR_SIZE):
                var idx = row * N + block * MXFP8_SF_VECTOR_SIZE + e
                var got_code = _signed_e4m3_code(got_data[idx])
                var want_code = _signed_e4m3_code(want_data[idx])
                # A one-step scale shift moves the data a whole exponent
                # (8 codes), plus one code step of rounding slack.
                var bound = 1 if got_scale == want_scale else 9
                if abs(got_code - want_code) > bound:
                    byte_mismatches += 1
                    if byte_mismatches <= 8:
                        print(
                            "fp64 oracle: M =",
                            M,
                            "N =",
                            N,
                            "byte",
                            idx,
                            "fused",
                            got_data[idx],
                            "vs",
                            want_data[idx],
                        )
                if got_scale == want_scale:
                    var block_scale = bitcast[.float8_e8m0fnu](got_scale).cast[
                        .float32
                    ]()
                    var dequant_got = (
                        bitcast[.float8_e4m3fn](got_data[idx]).cast[.float32]()
                        * block_scale
                    )
                    var ref_val = h_bf16[idx]
                    # The byte gate above allows one E4M3 code step from
                    # the oracle's rounding, so the dequant error is
                    # bounded by 1.5 steps of the value's binade: two
                    # normal steps are `|ref| * 2^-2`, two subnormal
                    # steps are `block_scale * 2^-8`.
                    var allowed = max(
                        block_scale * Float32(2.0) ** Float32(-8),
                        abs(ref_val) * Float32(0.25),
                    )
                    dequant_worst = max(
                        dequant_worst, abs(dequant_got - ref_val) / allowed
                    )

    assert_true(
        scale_mismatches * 1000 <= num_blocks,
        String("fp64 oracle: ")
        + String(scale_mismatches)
        + " of "
        + String(num_blocks)
        + " scales mismatch (over 0.1%)",
    )
    assert_true(
        byte_mismatches == 0,
        String("fp64 oracle: ")
        + String(byte_mismatches)
        + " data bytes are more than the allowed code steps from the fp64"
        + " oracle",
    )
    assert_true(
        dequant_worst <= Float32(1.0),
        (
            "fp64 oracle: a dequantized value is off by more than two E4M3"
            " code steps"
        ),
    )
    print(
        "fp64 oracle: M =",
        M,
        "N =",
        N,
        "scale mismatches",
        scale_mismatches,
        "of",
        num_blocks,
        "blocks, dequant worst rel",
        dequant_worst,
    )


def test_limit_zero_mutant_fails[N: Int, M: Int](ctx: DeviceContext) raises:
    """Mutant: the same kernel with `limit = 0` (linear_beta None) must
    differ from the linear_beta = 25 reference. `25 * tanh(up / 25)` is a
    soft saturation, not a clamp, so the data needs |up| large enough that
    it differs from `up` outright."""
    comptime reference = get_swiglu_activation["situ", BETA, LINEAR_BETA]
    comptime mutant = get_swiglu_activation["situ", BETA, Float32(0.0)]

    var values = List[Float32]()
    _fill_gate_up(values, M, N)
    var ref_data = List[UInt8]()
    var ref_scales = List[UInt8]()
    _run_fused[N, M, reference](ctx, values, ref_data, ref_scales)
    var mutant_data = List[UInt8]()
    var mutant_scales = List[UInt8]()
    _run_fused[N, M, mutant](ctx, values, mutant_data, mutant_scales)

    var diffs = 0
    for i in range(len(ref_data)):
        if ref_data[i] != mutant_data[i]:
            diffs += 1
    for i in range(len(ref_scales)):
        if ref_scales[i] != mutant_scales[i]:
            diffs += 1
    assert_true(
        diffs > 0,
        (
            "limit = 0 mutant produced the linear_beta = 25 bytes: the"
            " up saturation has no effect on this data"
        ),
    )
    print(
        "limit=0 mutant: differs from the linear_beta=25 reference in",
        diffs,
        "bytes (must be > 0)",
    )


def test_no_bf16_round_reference_differs[
    N: Int, M: Int
](ctx: DeviceContext) raises:
    """Mutant: a host reference built WITHOUT the bfloat16 rounding must
    mismatch the kernel's bytes. The bf16 round shifts ~2^-9 relative,
    which moves ~1-2% of E4M3 bytes by one code step; a gate that cannot
    see it would pass a kernel that skips the rounding."""
    comptime activation = get_swiglu_activation["situ", BETA, LINEAR_BETA]
    var values = List[Float32]()
    _fill_gate_up(values, M, N)

    var got_data = List[UInt8]()
    var got_scales = List[UInt8]()
    _run_fused[N, M, activation](ctx, values, got_data, got_scales)

    # The fp64 activation, deliberately NOT rounded to bfloat16.
    var h = _host_situ_fp64[N, M](values, Float64(BETA), Float64(LINEAR_BETA))
    var h_noround = List[Float32]()
    for v in h:
        h_noround.append(v.cast[.float32]())

    var want_data = List[UInt8]()
    var want_scales = List[UInt8]()
    _host_quantize(h_noround, M, N, want_data, want_scales)

    comptime scale_N = N // MXFP8_SF_VECTOR_SIZE
    var diffs = 0
    for row in range(M):
        for block in range(scale_N):
            var sf_idx = row * scale_N + block
            if got_scales[sf_idx] != want_scales[sf_idx]:
                continue
            for e in range(MXFP8_SF_VECTOR_SIZE):
                var idx = row * N + block * MXFP8_SF_VECTOR_SIZE + e
                if _signed_e4m3_code(got_data[idx]) != _signed_e4m3_code(
                    want_data[idx]
                ):
                    diffs += 1
    print(
        "no-bf16-round reference: differs from the fused kernel in",
        diffs,
        "of",
        len(got_data),
        "bytes (must be > 0)",
    )
    assert_true(
        diffs > 0,
        (
            "the no-bf16-round reference matches the kernel: the exact-byte"
            " gate cannot catch a missing bfloat16 rounding"
        ),
    )
    assert_true(
        diffs * 20 < len(got_data),
        (
            "the no-bf16-round reference differs in more than 5% of bytes;"
            " something other than the rounding is wrong"
        ),
    )


def test_rejects_wrong_scales_shape[
    N: Int, M: Int, activation: swiglu_activation_type
](ctx: DeviceContext) raises:
    """A direct call with a wrongly shaped scales tile must raise before
    the kernel is enqueued: the launcher is public and its kernel stores
    each row's scales unchecked, so a short scales tile would write out
    of bounds."""
    comptime scale_N = N // MXFP8_SF_VECTOR_SIZE
    var values = List[Float32]()
    _fill_gate_up(values, M, N)
    var gate_up_dev = _upload_gate_up[M, N](ctx, values)

    var out_dev = ctx.enqueue_create_buffer[.float8_e4m3fn](M * N)
    # One scale column short of the [M, N // 32] a [M, 2 * N] gate_up
    # needs.
    var sf_dev = ctx.enqueue_create_buffer[.float8_e8m0fnu](M * (scale_N - 1))

    with assert_raises(contains="scales [M, N // 32]"):
        swiglu_quantize_mx_amd[activation=activation](
            ctx,
            TileTensor(out_dev, row_major((Idx[M], Idx[N]))),
            TileTensor(sf_dev, row_major((Idx[M], Idx[scale_N - 1]))),
            TileTensor(gate_up_dev, row_major((Idx[M], Idx[2 * N]))),
        )
    print("wrong scales shape: raised before enqueue, M =", M, "N =", N)


def main() raises:
    comptime act = get_swiglu_activation["situ", BETA, LINEAR_BETA]
    comptime act_passthrough = get_swiglu_activation["situ", BETA]
    with DeviceContext() as ctx:
        print("fused vs the on-GPU unfused chain:")
        test_fused_matches_unfused[384, 16, act](ctx, "situ")
        test_fused_matches_unfused[384, 64, act](ctx, "situ")
        test_fused_matches_unfused[384, 128, act](ctx, "situ")
        test_fused_matches_unfused[384, 300, act](ctx, "situ")
        test_fused_matches_unfused[3072, 8, act](ctx, "situ")
        # More rows than the grid the launcher derives at this width, so
        # the kernel's row loop strides.
        test_fused_matches_unfused[3072, 1300, act](ctx, "situ")
        # Twice the columns one 512-thread block pass covers, so the
        # kernel's column loop strides within a row.
        test_fused_matches_unfused[8192, 8, act](ctx, "situ")
        test_fused_matches_unfused[384, 16, act_passthrough](
            ctx, "linear_beta=None"
        )
        test_fused_matches_unfused[384, 128, act_passthrough](
            ctx, "linear_beta=None"
        )

        print("fused vs the host fp64 oracle:")
        test_fused_matches_host_fp64[384, 16, act, BETA, LINEAR_BETA](ctx)
        test_fused_matches_host_fp64[384, 64, act, BETA, LINEAR_BETA](ctx)
        test_fused_matches_host_fp64[384, 128, act, BETA, LINEAR_BETA](ctx)
        test_fused_matches_host_fp64[3072, 8, act, BETA, LINEAR_BETA](ctx)
        test_fused_matches_host_fp64[8192, 8, act, BETA, LINEAR_BETA](ctx)

        print("mutants:")
        test_limit_zero_mutant_fails[384, 128](ctx)
        test_no_bf16_round_reference_differs[384, 128](ctx)

        print("launcher validation:")
        test_rejects_wrong_scales_shape[64, 3, act](ctx)

        print("PASS")
