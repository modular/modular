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
# The gated residual boundary against a float64 model with the kernel's
# roundings, at prime and ragged row counts on both sides of the SM count,
# scaled and unscaled, persistent and cluster-split; then each planted
# fault, which must fail.
#
# The model sums rows in float64, so the branch's norm `n` can round one ulp
# apart, which moves a stream channel by `beta * ulp(n)`: many of its own ulps
# where the mix cancels. The bar: at most 1% of channels differ, each by no
# more than its own rounding plus the error it inherits (the flip for the
# stream, the stream's error through the next norm for its output), or 2^-6
# of the row's RMS if larger.

from std.math import exp, sqrt
from std.random import randn, seed
from std.testing import assert_false, assert_true

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from nn.gated_residual_rms_norm import (
    HIDDEN,
    POISONS,
    GatedResidualArgs,
    In,
    Out,
    gated_residual_rms_norm,
    gated_residual_rms_norm_split,
)

comptime POST_EPS = Float32(1e-8)
comptime NORM_EPS = Float32(1e-5)
comptime TEMPERATURE = Float32(2.0)
comptime GATE_EPS = Float32(1e-3)
comptime MISMATCH_BAR = 1e-2
comptime ROW_RMS_BAR = 1.0 / 64
comptime SENTINEL = Float32(-12288.0)
"""Written to the outputs first, so a row the kernel skips shows."""


def _in[T: DType](b: DeviceBuffer[T]) -> In[T]:
    return b.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin]()


def _out[T: DType](mut b: DeviceBuffer[T]) -> Out[T]:
    return b.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()


def _bf16(x: Float64) -> Float64:
    return Float64(Scalar[DType.bfloat16](Float32(x)).cast[DType.float32]())


def _excess(
    got: Float64, want: Float64, flip: Float64, rms: Float64
) -> Float64:
    """The difference over its allowance: 2^-7 of the larger value (one
    bf16 ulp or more) plus the inherited error `flip`, or 2^-6 of the row's
    RMS."""
    var ulp = max(abs(got), abs(want)) / 128
    return abs(got - want) / max(ulp + flip, rms * ROW_RMS_BAR)


struct Data:
    var rows: Int
    var branch: HostBuffer[DType.bfloat16]
    var residual: HostBuffer[DType.bfloat16]
    var post_w: HostBuffer[DType.bfloat16]
    var norm_w: HostBuffer[DType.bfloat16]
    var gate: HostBuffer[DType.bfloat16]
    var alpha: HostBuffer[DType.float32]
    var beta: HostBuffer[DType.float32]
    var d_branch: DeviceBuffer[DType.bfloat16]
    var d_residual: DeviceBuffer[DType.bfloat16]
    var d_post_w: DeviceBuffer[DType.bfloat16]
    var d_norm_w: DeviceBuffer[DType.bfloat16]
    var d_gate: DeviceBuffer[DType.bfloat16]

    def __init__(out self, ctx: DeviceContext, rows: Int) raises:
        var n = max(rows, 1) * HIDDEN
        self.rows = rows
        self.branch = ctx.enqueue_create_host_buffer[DType.bfloat16](n)
        self.residual = ctx.enqueue_create_host_buffer[DType.bfloat16](n)
        self.post_w = ctx.enqueue_create_host_buffer[DType.bfloat16](HIDDEN)
        self.norm_w = ctx.enqueue_create_host_buffer[DType.bfloat16](HIDDEN)
        self.gate = ctx.enqueue_create_host_buffer[DType.bfloat16](HIDDEN)
        self.alpha = ctx.enqueue_create_host_buffer[DType.float32](HIDDEN)
        self.beta = ctx.enqueue_create_host_buffer[DType.float32](HIDDEN)
        ctx.synchronize()
        seed(rows)
        var tmp = List[Float32](length=max(n, 3 * HIDDEN), fill=0)
        randn(tmp.unsafe_ptr(), n)
        for i in range(n):
            self.branch[i] = Scalar[DType.bfloat16](tmp[i] * 3)
        randn(tmp.unsafe_ptr(), n)
        for i in range(n):
            self.residual[i] = Scalar[DType.bfloat16](tmp[i])
        randn(tmp.unsafe_ptr(), 3 * HIDDEN)
        for c in range(HIDDEN):
            self.post_w[c] = Scalar[DType.bfloat16](tmp[c] * 0.2)
            self.norm_w[c] = Scalar[DType.bfloat16](tmp[HIDDEN + c] * 0.2)
            # the model's coefficients of the gate logit, in fp32 on the host
            self.gate[c] = Scalar[DType.bfloat16](tmp[2 * HIDDEN + c] * 0.5)
            var w = Float32(self.gate[c]) * (1 / TEMPERATURE)
            var b = 1 / (1 + exp(-w))
            self.beta[c] = b
            self.alpha[c] = sqrt(max((1 / (1 + exp(w))) * (1 + b), GATE_EPS))
        self.d_branch = ctx.enqueue_create_buffer[DType.bfloat16](n)
        self.d_residual = ctx.enqueue_create_buffer[DType.bfloat16](n)
        self.d_post_w = ctx.enqueue_create_buffer[DType.bfloat16](HIDDEN)
        self.d_norm_w = ctx.enqueue_create_buffer[DType.bfloat16](HIDDEN)
        self.d_gate = ctx.enqueue_create_buffer[DType.bfloat16](HIDDEN)
        ctx.enqueue_copy(self.d_branch, self.branch)
        ctx.enqueue_copy(self.d_residual, self.residual)
        ctx.enqueue_copy(self.d_post_w, self.post_w)
        ctx.enqueue_copy(self.d_norm_w, self.norm_w)
        ctx.enqueue_copy(self.d_gate, self.gate)
        ctx.synchronize()

    def model(
        self,
        row: Int,
        scale: Bool,
        mut stream: List[Float64],
        mut normed: List[Float64],
        mut flip_s: List[Float64],
        mut flip_n: List[Float64],
    ):
        """One row through the kernel's roundings, sums in float64, and the
        error each output inherits from a one-ulp flip of the branch's norm
        (and, for the next norm, the stream's own rounding)."""
        var x = List[Float64](length=HIDDEN, fill=0)
        var s1 = Float64(0)
        for c in range(HIDDEN):
            var v = Float64(self.branch[row * HIDDEN + c].cast[DType.float32]())
            if scale:
                v = _bf16(
                    v * _bf16(1 + Float64(self.post_w[c].cast[DType.float32]()))
                )
            x[c] = v
            s1 += v * v
        var inv1 = 1 / sqrt(s1 / HIDDEN + Float64(POST_EPS))
        var s2 = Float64(0)
        for c in range(HIDDEN):
            var n = _bf16(x[c] * inv1)
            var r = Float64(
                self.residual[row * HIDDEN + c].cast[DType.float32]()
            )
            stream[c] = _bf16(
                Float64(self.alpha[c]) * r + Float64(self.beta[c]) * n
            )
            flip_s[c] = Float64(self.beta[c]) * abs(n) / 128
            s2 += stream[c] * stream[c]
        var inv2 = 1 / sqrt(s2 / HIDDEN + Float64(NORM_EPS))
        for c in range(HIDDEN):
            var g = _bf16(1 + Float64(self.norm_w[c].cast[DType.float32]()))
            normed[c] = _bf16(stream[c] * inv2 * g)
            var stream_err = abs(stream[c]) / 128 + flip_s[c]
            flip_n[c] = stream_err * inv2 * abs(g)


def run_case[
    poison: Int = 0, split: Bool = False
](ctx: DeviceContext, rows: Int, scale: Bool) raises -> Bool:
    """Whether the kernel's outputs pass the bar on every row; `split` runs
    the cluster-split launcher instead of the persistent one."""
    var d = Data(ctx, rows)
    var n = max(rows, 1) * HIDDEN
    var d_stream = ctx.enqueue_create_buffer[DType.bfloat16](n)
    var d_normed = ctx.enqueue_create_buffer[DType.bfloat16](n)
    d_stream.enqueue_fill(Scalar[DType.bfloat16](SENTINEL))
    d_normed.enqueue_fill(Scalar[DType.bfloat16](SENTINEL))
    var args = GatedResidualArgs(
        _in(d.d_branch),
        _in(d.d_residual),
        _in(d.d_post_w),
        _in(d.d_gate),
        _in(d.d_norm_w),
        _out(d_stream),
        _out(d_normed),
        Int64(rows),
        POST_EPS,
        NORM_EPS,
        TEMPERATURE,
        GATE_EPS,
    )
    comptime if split:
        if scale:
            gated_residual_rms_norm_split[True](args, ctx)
        else:
            gated_residual_rms_norm_split[False](args, ctx)
    else:
        if scale:
            gated_residual_rms_norm[True, poison](args, ctx)
        else:
            gated_residual_rms_norm[False, poison](args, ctx)
    var h_stream = ctx.enqueue_create_host_buffer[DType.bfloat16](n)
    var h_normed = ctx.enqueue_create_host_buffer[DType.bfloat16](n)
    ctx.enqueue_copy(h_stream, d_stream)
    ctx.enqueue_copy(h_normed, d_normed)
    ctx.synchronize()
    if rows == 0:
        # nothing launched, nothing written
        return Float32(h_stream[0].cast[DType.float32]()) == SENTINEL

    var want_s = List[Float64](length=HIDDEN, fill=0)
    var want_n = List[Float64](length=HIDDEN, fill=0)
    var flip_s = List[Float64](length=HIDDEN, fill=0)
    var flip_n = List[Float64](length=HIDDEN, fill=0)
    var differ = 0
    var worst = Float64(0)
    var worst_at = String("")
    for row in range(rows):
        d.model(row, scale, want_s, want_n, flip_s, flip_n)
        var ms = Float64(0)
        var mn = Float64(0)
        for c in range(HIDDEN):
            ms += want_s[c] * want_s[c]
            mn += want_n[c] * want_n[c]
        var rms_s = sqrt(ms / HIDDEN)
        var rms_n = sqrt(mn / HIDDEN)
        for c in range(HIDDEN):
            var gs = Float64(h_stream[row * HIDDEN + c].cast[DType.float32]())
            var gn = Float64(h_normed[row * HIDDEN + c].cast[DType.float32]())
            if gs != want_s[c]:
                differ += 1
            if gn != want_n[c]:
                differ += 1
            var es = _excess(gs, want_s[c], flip_s[c], rms_s)
            var en = _excess(gn, want_n[c], flip_n[c], rms_n)
            if max(es, en) > worst:
                worst = max(es, en)
                worst_at = String(
                    "row ",
                    row,
                    " col ",
                    c,
                    " stream ",
                    gs,
                    "/",
                    want_s[c],
                    " normed ",
                    gn,
                    "/",
                    want_n[c],
                    " flip ",
                    flip_s[c],
                    " rms ",
                    rms_s,
                )
    var frac = Float64(differ) / Float64(2 * rows * HIDDEN)
    var ok = frac <= MISMATCH_BAR and worst <= 1
    print(
        "  rows ",
        rows,
        " scale ",
        scale,
        " poison ",
        poison,
        ": ",
        frac,
        " differ, worst ",
        worst,
        " of the bar -> ",
        "pass" if ok else String("FAIL at ", worst_at),
        sep="",
    )
    return ok


def main() raises:
    with DeviceContext() as ctx:
        # one row, primes, the SM count and past it (the persistent walk)
        for rows in [0, 1, 2, 3, 7, 61, 148, 149, 301]:
            for scale in [True, False]:
                assert_true(run_case(ctx, rows, scale))
        # the cluster split: one row, primes, the last row count under SMs
        var sms = ctx.default_device_info.sm_count
        for rows in [0, 1, 2, 3, 7, 61, sms - 1]:
            for scale in [True, False]:
                assert_true(run_case[split=True](ctx, rows, scale))
        comptime for p in range(1, POISONS + 1):
            # each planted fault fails at a multi-trip row count
            assert_false(run_case[p](ctx, 301, True), String("poison ", p))
        print("All tests passed!")
