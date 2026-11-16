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
# The iHC projection and the collapse + RMSNorm against a float64 model, at no
# rows, one, primes, the 64-row block edges and a prefill chunk, with both
# gate bands (an iHC site) and `pre` alone (the iHC head); then each planted
# fault, which must fail.
#
# The gates are held to 1e-5: the kernel keeps about 16 bits of the weight
# and sums in float32. The output is modeled from the kernel's own `pre`, with
# the kernel's roundings (the collapse and the normalized value rounded to
# bfloat16 before gamma). The bar is 2^-6 of the element. Where the collapse
# sits on a bfloat16 tie (within 1e-5), the kernel's float32 sum may round to
# either neighbour and the two later roundings can carry that one step to
# three, so the output from both neighbours is accepted.

from std.math import exp, sqrt
from std.testing import assert_false, assert_true
from std.utils.numerics import isfinite, nan

from layout import Idx, TileTensor, row_major
from max.gpu.host import DeviceContext, HostBuffer
from nn.ihc import (
    IHC_POISONS,
    ihc_collapse_rms_norm,
    ihc_mix_gemv,
    ihc_mix_gemv_splits,
)

comptime HC = 4
comptime MAGNITUDE = Float32(2.0)
comptime HC_EPS = Float32(1e-6)
comptime STREAMS_EPS = Float32(1e-5)
comptime NORM_EPS = Float32(1e-5)
comptime GATE_TOL = 1e-5
comptime SENTINEL = nan[DType.float32]()
"""Written to the outputs first, so a row the kernel skips shows."""


def _hashed(seed: Int, i: Int) -> Float64:
    """A hashed value in `[-0.5, 0.5)`."""
    var h = UInt64(seed) * 0x9E3779B97F4A7C15 + UInt64(i) * 0xBF58476D1CE4E5B9
    h ^= h >> 33
    h *= 0xFF51AFD7ED558CCD
    h ^= h >> 33
    return Float64(UInt32(h >> 40)) / Float64(1 << 24) - 0.5


def _sigmoid(x: Float64) -> Float64:
    return 1 / (1 + exp(-x))


def _bf16(x: Float64) -> Float64:
    return Float64(Scalar[DType.bfloat16](Float32(x)).cast[DType.float32]())


def run_case[
    hidden: Int, num_mix: Int, poison: Int = 0
](ctx: DeviceContext, rows: Int) raises -> Bool:
    """Runs one case; returns whether every element is within its bar."""
    comptime k = HC * hidden
    comptime splits = ihc_mix_gemv_splits[k]()
    comptime bands = num_mix // HC
    var n = max(rows, 1)

    var streams_h = ctx.enqueue_create_host_buffer[DType.bfloat16](n * k)
    var fn_h = ctx.enqueue_create_host_buffer[DType.float32](num_mix * k)
    var base_h = ctx.enqueue_create_host_buffer[DType.float32](num_mix)
    var scale_h = ctx.enqueue_create_host_buffer[DType.float32](bands)
    var gamma_h = ctx.enqueue_create_host_buffer[DType.bfloat16](hidden)
    # Uniform with unit variance; `fn` keeps the logits near one.
    for i in range(n * k):
        streams_h[i] = Scalar[DType.bfloat16](
            Float32(_hashed(rows + 1, i) * 3.4641)
        )
    for i in range(num_mix * k):
        fn_h[i] = Float32(_hashed(7, i) * 3.4641 / sqrt(Float64(k)))
    for i in range(num_mix):
        base_h[i] = Float32(_hashed(8, i))
    for i in range(bands):
        scale_h[i] = Float32(0.6 + 0.3 * Float64(i))
    for i in range(hidden):
        gamma_h[i] = Scalar[DType.bfloat16](Float32(1 + 0.2 * _hashed(9, i)))

    var streams_d = ctx.enqueue_create_buffer[DType.bfloat16](n * k)
    var fn_d = ctx.enqueue_create_buffer[DType.float32](num_mix * k)
    var base_d = ctx.enqueue_create_buffer[DType.float32](num_mix)
    var scale_d = ctx.enqueue_create_buffer[DType.float32](bands)
    var gamma_d = ctx.enqueue_create_buffer[DType.bfloat16](hidden)
    var partials_d = ctx.enqueue_create_buffer[DType.float32](
        splits * n * (num_mix + 1)
    )
    var gates_d = ctx.enqueue_create_buffer[DType.float32](n * num_mix)
    var out_d = ctx.enqueue_create_buffer[DType.bfloat16](n * hidden)
    ctx.enqueue_copy(streams_d, streams_h)
    ctx.enqueue_copy(fn_d, fn_h)
    ctx.enqueue_copy(base_d, base_h)
    ctx.enqueue_copy(scale_d, scale_h)
    ctx.enqueue_copy(gamma_d, gamma_h)
    gates_d.enqueue_fill(SENTINEL)
    out_d.enqueue_fill(SENTINEL.cast[DType.bfloat16]())

    ihc_mix_gemv[k=k, num_mix=num_mix, poison=poison](
        TileTensor(
            partials_d, row_major((Idx[splits], rows, Idx[num_mix + 1]))
        ),
        TileTensor(streams_d, row_major((rows, Idx[k]))).as_imm(),
        TileTensor(fn_d, row_major((Idx[num_mix], Idx[k]))).as_imm(),
        ctx,
    )
    ihc_collapse_rms_norm[
        hc_mult=HC,
        num_mix=num_mix,
        num_splits=splits,
        hidden=hidden,
        poison=poison,
    ](
        TileTensor(out_d, row_major((rows, Idx[hidden]))),
        TileTensor(gates_d, row_major((rows, Idx[num_mix]))),
        TileTensor(streams_d, row_major((rows, Idx[HC], Idx[hidden]))).as_imm(),
        TileTensor(
            partials_d, row_major((Idx[splits], rows, Idx[num_mix + 1]))
        ).as_imm(),
        TileTensor(base_d, row_major((Idx[num_mix],))).as_imm(),
        TileTensor(scale_d, row_major((Idx[bands],))).as_imm(),
        TileTensor(gamma_d, row_major((Idx[hidden],))).as_imm(),
        MAGNITUDE,
        HC_EPS,
        STREAMS_EPS,
        NORM_EPS,
        ctx,
    )
    var gates_h = ctx.enqueue_create_host_buffer[DType.float32](n * num_mix)
    var out_h = ctx.enqueue_create_host_buffer[DType.bfloat16](n * hidden)
    ctx.enqueue_copy(gates_h, gates_d)
    ctx.enqueue_copy(out_h, out_d)
    ctx.synchronize()

    var bad_gates = 0
    var bad_out = 0
    var worst_gate = 0.0
    for r in range(rows):
        var dots = List[Float64](length=num_mix, fill=0.0)
        var squares = 0.0
        for c in range(k):
            var x = Float64(streams_h[r * k + c].cast[DType.float32]())
            squares += x * x
            for m in range(num_mix):
                dots[m] += x * Float64(fn_h[m * k + c])
        var inv_rms = 1 / sqrt(squares / Float64(k) + Float64(STREAMS_EPS))
        for m in range(num_mix):
            var band = m // HC
            var g = _sigmoid(
                dots[m] * inv_rms * Float64(scale_h[band]) + Float64(base_h[m])
            )
            var want = (Float64(MAGNITUDE) * g if band else g) + Float64(HC_EPS)
            var err = abs(Float64(gates_h[r * num_mix + m]) - want)
            worst_gate = max(worst_gate, err)
            if not (err <= GATE_TOL * max(1.0, abs(want))):
                bad_gates += 1

        # The output from the kernel's own `pre`, with its roundings.
        var exact = List[Float64](length=hidden, fill=0.0)
        var sum_sq = 0.0
        for d in range(hidden):
            for j in range(HC):
                exact[d] += Float64(gates_h[r * num_mix + j]) * Float64(
                    streams_h[r * k + j * hidden + d].cast[DType.float32]()
                )
            var c = _bf16(exact[d])
            sum_sq += c * c
        var factor = 1 / sqrt(sum_sq / Float64(hidden) + Float64(NORM_EPS))
        for d in range(hidden):
            var g = Float64(gamma_h[d].cast[DType.float32]())
            var got = Float64(out_h[r * hidden + d].cast[DType.float32]())
            var ok = False
            for nudge in [0.0, 1e-5, -1e-5]:
                var want = _bf16(
                    _bf16(_bf16(exact[d] * (1 + nudge)) * factor) * g
                )
                ok = (
                    ok
                    or abs(got - want) <= max(abs(got), abs(want)) / 64 + 1e-6
                )
            # An infinite `got` would pass the relative bound above.
            if not (ok and isfinite(got)):
                bad_out += 1
    print(
        "  hidden ",
        hidden,
        " num_mix ",
        num_mix,
        " poison ",
        poison,
        " rows ",
        rows,
        ": gates off ",
        bad_gates,
        " (worst ",
        worst_gate,
        "), output off ",
        bad_out,
        sep="",
    )
    return bad_gates == 0 and bad_out == 0


def main() raises:
    with DeviceContext() as ctx:
        var ok = True
        for rows in [0, 1, 2, 3, 7, 16, 17, 24, 61, 64, 65, 127, 300]:
            ok = run_case[6144, 8](ctx, rows) and ok
            ok = run_case[6144, 4](ctx, rows) and ok
            ok = run_case[512, 8](ctx, rows) and ok
        # Prefill: a 4096-token chunk (full row blocks), a prime, the chunk
        # with six decode requests of 4 verify rows each (4096 + 6 * 4 = 4120),
        # and the chunk with six decode tokens and 3 draft rows for each of the
        # seven requests (4096 + 6 + 7 * 3 = 4123).
        for rows in [4093, 4096, 4120, 4123]:
            ok = run_case[6144, 8](ctx, rows) and ok
            ok = run_case[6144, 4](ctx, rows) and ok
        assert_true(ok, "a case is off")
        comptime for p in range(1, IHC_POISONS + 1):
            assert_false(run_case[6144, 8, p](ctx, 61), String("poison ", p))
            assert_false(
                run_case[6144, 8, p](ctx, 4123),
                String("poison ", p, " prefill"),
            )
        print("All tests passed!")
