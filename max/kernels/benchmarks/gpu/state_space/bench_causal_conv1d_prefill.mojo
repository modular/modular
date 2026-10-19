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
"""Benchmarks the registered dense causal-conv prefill operation.

The adjacent YAML manifest owns the shape grids. Named CLI arguments select
one shape and input view. Token-major storage represents the first-channel
half of Mamba's projected `[B, L, 2*C]` tensor, viewed logically as `[B, C, L]`.
The output is contiguous in both cases. Timings include the live registered
host launch path; these are operation-event probes, not pure GPU durations.
"""

from std.benchmark import Batch, Bench, BenchmarkInfo, Report
from std.builtin.sort import sort
from std.math import abs, exp, isinf, isnan, nan
from std.memory import bitcast
from std.random import rand, seed
from std.testing import assert_true
from std.utils.index import Index

from extensibility import IOSpec, ManagedTensorSlice, StaticTensorSpec
from internal_utils import arg_parse
from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from state_space.causal_conv1d import causal_conv1d_channel_first_fwd_cpu
from state_space.causal_conv1d_ops import CausalConv1D

comptime WARMUP = 5
comptime SAMPLES = 30
comptime SEED = 1234
comptime GUARD = 32


def _median(times: List[Float64]) -> Float64:
    var ordered = times.copy()
    sort(Span(ordered))
    var middle = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return (ordered[middle - 1] + ordered[middle]) / 2
    return ordered[middle]


@inline(.always)
def _rounded[dtype: DType](value: Float64) -> Float64:
    return Float64(Scalar[dtype](value))


def _gpu_order_reference[
    dtype: DType
](
    input_row: ImmPointer[Scalar[dtype], _],
    weights: ImmPointer[Scalar[dtype], _],
    bias: Scalar[dtype],
    position: Int,
    width: Int,
    silu_activation: Bool,
) raises -> Scalar[dtype]:
    var products = SIMD[DType.float64, 4](0)
    var exact_pre = Float64(bias)
    var magnitude = abs(Float64(bias))
    for tap in range(width):
        var source = position - (width - 1 - tap)
        var product = Float64(0)
        if source >= 0:
            product = Float64(input_row[source]) * Float64(weights[tap])
        products[tap] = _rounded[dtype](product)
        exact_pre += product
        magnitude += abs(product)
    var convolution: Float64
    match width:
        case 1:
            convolution = products[0]
        case 2:
            convolution = _rounded[dtype](products[0] + products[1])
        case 3:
            convolution = _rounded[dtype](
                _rounded[dtype](products[0] + products[1]) + products[2]
            )
        case _:
            # SIMD.reduce splits halves, so four lanes combine as (0+2)+(1+3).
            convolution = _rounded[dtype](
                _rounded[dtype](products[0] + products[2])
                + _rounded[dtype](products[1] + products[3])
            )
    var rounded_pre = _rounded[dtype](Float64(bias) + convolution)
    comptime unit_roundoff = 1.0 / 256 if dtype == DType.bfloat16 else 1.0 / 16777216
    comptime double_roundoff = 1.0 / 9007199254740992
    # Maximum product-to-bias paths have 2/3/4/4 roundings for widths 1/2/3/4.
    # The standard gamma_n bound is absolute, scaled by |bias| + sum(|x*w|),
    # so cancellation cannot make this check spuriously strict.
    var operations = min(width + 1, 4)
    var gamma = (
        Float64(operations)
        * unit_roundoff
        / (1 - Float64(operations) * unit_roundoff)
    )
    var convolution_bound = gamma * magnitude + 8 * double_roundoff * max(
        1.0, magnitude
    )
    assert_true(
        abs(rounded_pre - exact_pre) <= convolution_bound,
        "Scalar rounding oracle exceeds its gamma_n convolution bound",
    )
    if not silu_activation:
        return Scalar[dtype](rounded_pre)

    var ideal_activation = exact_pre / (1 + exp(-exact_pre))
    var rounded_activation = rounded_pre / (1 + exp(-rounded_pre))
    # |SiLU'| <= 1 + 1/e < 2; this bound covers convolution error propagation,
    # independently of the CPU/GPU Float32 exponential implementations.
    assert_true(
        abs(rounded_activation - ideal_activation)
        <= 2 * convolution_bound + 16 * double_roundoff,
        "Scalar oracle exceeds its Float64 SiLU propagation bound",
    )
    var value = Float32(rounded_pre)
    return Scalar[dtype](value * (Float32(1) / (Float32(1) + exp(-value))))


def _bench[
    dtype: DType
](
    ctx: DeviceContext,
    batch: Int,
    channels: Int,
    seqlen: Int,
    width: Int,
    silu_activation: Bool,
    token_major: Bool,
) raises:
    assert_true(batch > 0 and channels > 0 and seqlen > 0)
    assert_true(1 <= width <= 4)
    var elements = batch * channels * seqlen
    var input_size = 2 * elements if token_major else elements
    var input_batch_stride = (
        2 * channels * seqlen if token_major else channels * seqlen
    )
    var input_channel_stride = 1 if token_major else seqlen
    var input_seq_stride = 2 * channels if token_major else 1
    var layout_name = String("token_major") if token_major else String(
        "contiguous"
    )
    var activation = String("silu") if silu_activation else String("none")
    var description = String(
        t"kernel=causal_conv1d_prefill"
        t" dtype={dtype} batch={batch} channels={channels} seqlen={seqlen} width={width} activation={activation} layout={layout_name} input_strides={input_batch_stride}:{input_channel_stride}:{input_seq_stride} output_strides={channels * seqlen}:{seqlen}:1"
        t" threads=128 elements_per_thread=4"
        t" seed={SEED} correctness=scalar_gpu_order cpu_helper=diagnostic"
        t" registered_execute=true"
    )
    print(String("case ") + description)
    var compact_h = ctx.enqueue_create_host_buffer[dtype](elements)
    var input_h = ctx.enqueue_create_host_buffer[dtype](input_size + GUARD)
    var weights_h = ctx.enqueue_create_host_buffer[dtype](channels * width)
    var bias_h = ctx.enqueue_create_host_buffer[dtype](channels)
    var reference_h = ctx.enqueue_create_host_buffer[dtype](elements)
    var output_h = ctx.enqueue_create_host_buffer[dtype](elements + GUARD)
    seed(SEED)
    rand(compact_h.unsafe_ptr(), elements)
    rand(weights_h.unsafe_ptr(), channels * width)
    rand(bias_h.unsafe_ptr(), channels)
    for i in range(input_size + GUARD):
        input_h[i] = nan[dtype]()
    for i in range(elements + GUARD):
        output_h[i] = nan[dtype]()
    # Dyadic products permit exact Float64 mathematical checks; bias keeps outputs positive.
    for b in range(batch):
        for c in range(channels):
            for l in range(seqlen):
                var index = (b * channels + c) * seqlen + l
                compact_h[index] = Scalar[dtype](
                    Float32(Int(Float32(compact_h[index]) * 32) - 16) * 0.015625
                )
                var input_index = (
                    b * input_batch_stride
                    + c * input_channel_stride
                    + l * input_seq_stride
                )
                input_h[input_index] = compact_h[index]
                reference_h[index] = nan[dtype]()
    for i in range(channels * width):
        weights_h[i] = Scalar[dtype](
            Float32(Int(Float32(weights_h[i]) * 8) + 1) * 0.03125
        )
    for i in range(channels):
        bias_h[i] = Scalar[dtype](
            Float32(Int(Float32(bias_h[i]) * 8) + 8) * 0.0625
        )
    causal_conv1d_channel_first_fwd_cpu[dtype, dtype, dtype, dtype](
        batch,
        channels,
        seqlen,
        width,
        TileTensor(compact_h, row_major(batch, channels, seqlen)),
        TileTensor(weights_h, row_major(channels, width)),
        TileTensor(reference_h, row_major(batch, channels, seqlen)),
        TileTensor(bias_h, row_major(channels)),
        silu_activation,
    )

    var input_d = ctx.enqueue_create_buffer[dtype](input_size + GUARD)
    var weights_d = ctx.enqueue_create_buffer[dtype](channels * width)
    var bias_d = ctx.enqueue_create_buffer[dtype](channels)
    var output_d = ctx.enqueue_create_buffer[dtype](elements + GUARD)
    ctx.enqueue_copy(input_d, input_h)
    ctx.enqueue_copy(weights_d, weights_h)
    ctx.enqueue_copy(bias_d, bias_h)
    ctx.enqueue_copy(output_d, output_h)
    comptime spec3 = StaticTensorSpec[dtype, 3].get_unknown()
    comptime spec2 = StaticTensorSpec[dtype, 2].get_unknown()
    comptime spec1 = StaticTensorSpec[dtype, 1].get_unknown()
    var input_arg = ManagedTensorSlice[io_spec=IOSpec.Input, static_spec=spec3](
        input_d.unsafe_ptr(),
        Index(batch, channels, seqlen),
        Index(input_batch_stride, input_channel_stride, input_seq_stride),
    )
    var weights_arg = ManagedTensorSlice[
        io_spec=IOSpec.Input, static_spec=spec2
    ](weights_d.unsafe_ptr(), (channels, width))
    var bias_arg = ManagedTensorSlice[io_spec=IOSpec.Input, static_spec=spec1](
        bias_d.unsafe_ptr(), (channels,)
    )
    var output_arg = ManagedTensorSlice[
        io_spec=IOSpec.Output, static_spec=spec3
    ](output_d.unsafe_ptr(), (batch, channels, seqlen))

    def launch(
        lctx: DeviceContext,
    ) raises {
        imm input_arg,
        imm weights_arg,
        imm bias_arg,
        imm output_arg,
        imm silu_activation,
    }:
        if silu_activation:
            CausalConv1D["silu"].execute[target="gpu"](
                output_arg, input_arg, weights_arg, bias_arg, lctx
            )
        else:
            CausalConv1D["none"].execute[target="gpu"](
                output_arg, input_arg, weights_arg, bias_arg, lctx
            )

    ctx.synchronize()
    launch(ctx)
    ctx.enqueue_copy(output_h, output_d)
    ctx.synchronize()
    comptime bits_dtype = DType.uint16 if dtype == DType.bfloat16 else DType.uint32
    # GPU exp uses an approximate exp2 path, while host exp uses a polynomial.
    # Allow at most one BF16 output ULP (or eight FP32 ULPs) for activation,
    # not a relative tolerance on the differently ordered CPU convolution.
    var activation_ulps = 1 if dtype == DType.bfloat16 else 8
    var allowed_ulps = activation_ulps if silu_activation else 0
    var cpu_differences = 0
    var cpu_max_error = Float64(0)
    for i in range(elements):
        assert_true(not isnan(output_h[i]) and not isinf(output_h[i]))
        assert_true(not isnan(reference_h[i]) and not isinf(reference_h[i]))
        var row, position = divmod(i, seqlen)
        var channel = row % channels
        var expected = _gpu_order_reference[dtype](
            compact_h.unsafe_ptr().unsafe_offset(row * seqlen),
            weights_h.unsafe_ptr().unsafe_offset(channel * width),
            bias_h[channel],
            position,
            width,
            silu_activation,
        )
        var output_bits = Int(bitcast[bits_dtype](output_h[i]))
        var expected_bits = Int(bitcast[bits_dtype](expected))
        assert_true(output_h[i] > 0 and expected > 0)
        var distance = abs(output_bits - expected_bits)
        if distance > allowed_ulps:
            print(
                String(t"mismatch {description} index={i} ")
                + String(t"actual={output_h[i]} rounded_oracle={expected} ")
                + String(
                    t"cpu_bias_first={reference_h[i]} ulps={distance} allowed_ulps={allowed_ulps}"
                )
            )
        assert_true(
            distance <= allowed_ulps,
            "Output differs from scalar GPU-order oracle",
        )
        if bitcast[bits_dtype](output_h[i]) != bitcast[bits_dtype](
            reference_h[i]
        ):
            cpu_differences += 1
        cpu_max_error = max(
            cpu_max_error, abs(Float64(output_h[i]) - Float64(reference_h[i]))
        )
    print(
        String(
            t"validated {description} allowed_activation_ulps={allowed_ulps} "
        )
        + String(
            t"cpu_bias_first_differences={cpu_differences} cpu_bias_first_max_abs={cpu_max_error}"
        )
    )
    for i in range(elements, elements + GUARD):
        assert_true(isnan(output_h[i]), "Output tail guard was overwritten")
    for _ in range(WARMUP):
        launch(ctx)
    ctx.synchronize()
    var times = List[Float64]()
    for _ in range(SAMPLES):
        times.append(Float64(ctx.execution_time(launch, 1)))
    ctx.enqueue_copy(output_h, output_d)
    ctx.synchronize()
    for i in range(elements, elements + GUARD):
        assert_true(
            isnan(output_h[i]), "Timed output tail guard was overwritten"
        )

    for sample in range(SAMPLES):
        print(t"sample {description} index={sample} ns={times[sample]}")
    print(t"summary {description} samples={SAMPLES} median_ns={_median(times)}")
    var benchmark = Bench()
    var report = Report()
    for elapsed in times:
        report.runs.append(Batch(Int(elapsed), 1, True))
    benchmark.info_vec.append(BenchmarkInfo(description, report^))
    benchmark.dump_report()
    _ = input_d^
    _ = weights_d^
    _ = bias_d^
    _ = output_d^


def main() raises:
    var dtype = arg_parse("dtype", String(""))
    if dtype != "bfloat16" and dtype != "float32":
        raise Error("dtype must be bfloat16 or float32")
    var batch = arg_parse("batch", 0)
    var channels = arg_parse("channels", 0)
    var seqlen = arg_parse("seqlen", 0)
    var width = arg_parse("width", 0)
    var silu_activation = arg_parse("silu", True)
    var token_major = arg_parse("token_major", False)
    with DeviceContext() as ctx:
        assert_true(ctx.is_compatible(), "A compatible GPU is required")
        var api = ctx.api()
        assert_true(
            api == "cuda" or api == "hip", "CUDA or HIP GPU is required"
        )
        print(
            t"config api={api} warmup={WARMUP} samples={SAMPLES} "
            t"launches_per_sample=1 l2_flush=false timing=operation_event"
            t" csv_statistic=mean raw_statistic=median"
        )
        if dtype == "bfloat16":
            _bench[.bfloat16](
                ctx,
                batch,
                channels,
                seqlen,
                width,
                silu_activation,
                token_major,
            )
        else:
            _bench[.float32](
                ctx,
                batch,
                channels,
                seqlen,
                width,
                silu_activation,
                token_major,
            )
