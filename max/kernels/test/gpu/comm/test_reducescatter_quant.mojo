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
"""Correctness tests for `comm.reducescatter_quant`.

Each peer's contribution is quantized in groups of 32, so an element of the
sum may sit off the exact sum by at most each peer's per-group error bound:
half an INT4 step (`group_max / 14`, plus the bf16 scale's rounding) or one E4M3 rounding (`group_max / 16`),
plus the bf16 rounding of the result. The test checks every element against
that bound and that the mean error stays well inside it, which a scale read
from the wrong group or a misplaced row would break.
"""

from std.math import ceildiv
from std.sys import size_of

from comm import MAX_GPUS, Signal
from comm.reducescatter import ReduceScatterConfig
from comm.reducescatter_quant import (
    RS_QUANT_GROUP,
    reducescatter_quant,
    rs_quant_scratch_bytes,
)
from comm.sync import enable_p2p, init_signal_buffer
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.testing import assert_true


def input_value(gpu: Int, row: Int, col: Int) -> Float32:
    """Signed values whose magnitude varies by group, row and GPU."""
    var h = (gpu * 1_000_003 + row * 7919 + col * 104729) % 65521
    var unit = Float32(h) / 65521.0 * 2.0 - 1.0
    var group_scale = Float32(1 + (col // 32 + row + gpu) % 7)
    return unit * group_scale


def residual_value(row: Int, col: Int) -> Float32:
    return Float32((row * 31 + col * 17) % 101) / 50.0 - 1.0


def run_case[
    ngpus: Int, quant_bits: Int, has_residual: Bool
](list_of_ctx: List[DeviceContext[]], rows: Int, cols: Int) raises:
    comptime dtype = DType.bfloat16
    var n = rows * cols
    var config = ReduceScatterConfig[dtype, ngpus](
        axis_size=rows, unit_numel=cols, threads_per_gpu=0
    )
    var scratch = rs_quant_scratch_bytes[quant_bits, ngpus](rows, cols)

    var in_bufs = List[DeviceBuffer[dtype]](capacity=ngpus)
    var out_bufs = List[DeviceBuffer[dtype]](capacity=ngpus)
    var res_bufs = List[DeviceBuffer[dtype]](capacity=ngpus)
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var host_in = List[HostBuffer[dtype]](capacity=ngpus)
    var host_res = list_of_ctx[0].enqueue_create_host_buffer[dtype](n)
    for r in range(rows):
        for c in range(cols):
            host_res[r * cols + c] = residual_value(r, c).cast[dtype]()

    for g in range(ngpus):
        var ctx = list_of_ctx[g]
        in_bufs.append(ctx.enqueue_create_buffer[dtype](n))
        out_bufs.append(
            ctx.enqueue_create_buffer[dtype](
                max(config.rank_units(g), 1) * cols
            )
        )
        res_bufs.append(ctx.enqueue_create_buffer[dtype](n))
        var h = ctx.enqueue_create_host_buffer[dtype](n)
        for r in range(rows):
            for c in range(cols):
                h[r * cols + c] = input_value(g, r, c).cast[dtype]()
        ctx.enqueue_copy(in_bufs[g], h)
        ctx.enqueue_copy(res_bufs[g], host_res)
        host_in.append(h^)
        signal_buffers.append(
            ctx.create_buffer_sync[.uint8](size_of[Signal]() + scratch)
        )
        init_signal_buffer(signal_buffers[g], ctx)

    var rank_sigs = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> MutPointer[
            Signal, MutAnyOrigin
        ]: Signal.unsafe_ptr_from(signal_buffers[i])
    )
    for g in range(ngpus):
        list_of_ctx[g].synchronize()

    for g in range(ngpus):
        reducescatter_quant[dtype, ngpus, quant_bits, has_residual](
            in_bufs[g].unsafe_ptr().as_unsafe_any_origin().as_imm(),
            out_bufs[g].unsafe_ptr().as_unsafe_any_origin(),
            res_bufs[g].unsafe_ptr().as_unsafe_any_origin().as_imm(),
            rows,
            cols,
            rank_sigs,
            g,
            list_of_ctx[g],
        )

    var host_out = List[HostBuffer[dtype]](capacity=ngpus)
    for g in range(ngpus):
        var h = list_of_ctx[g].enqueue_create_host_buffer[dtype](
            max(config.rank_units(g), 1) * cols
        )
        list_of_ctx[g].enqueue_copy(h, out_bufs[g])
        host_out.append(h^)
    for g in range(ngpus):
        list_of_ctx[g].synchronize()

    comptime step = Float64(1.0 / 14.0 + 1.0 / 128.0) if quant_bits == 4 else (
        Float64(1.0 / 16.0)
    )
    var err_sum = Float64(0)
    var bound_sum = Float64(0)
    for g in range(ngpus):
        var start = config.rank_unit_start(g)
        for s in range(config.rank_units(g)):
            var row = start + s
            for c in range(cols):
                var exact = Float64(0)
                var bound = Float64(0)
                for p in range(ngpus):
                    exact += Float64(host_in[p][row * cols + c])
                    if p == g:
                        continue
                    var g0 = c // RS_QUANT_GROUP * RS_QUANT_GROUP
                    var gmax = Float64(0)
                    for k in range(RS_QUANT_GROUP):
                        gmax = max(
                            gmax, abs(Float64(host_in[p][row * cols + g0 + k]))
                        )
                    bound += step * gmax
                var res = Float64(0)
                comptime if has_residual:
                    res = Float64(host_res[row * cols + c])
                var expected = exact + res
                bound += (abs(exact) + abs(expected)) / 128.0 + 1e-6
                var got = Float64(host_out[g][s * cols + c])
                var err = abs(got - expected)
                err_sum += err
                bound_sum += bound
                assert_true(
                    err <= bound,
                    String(
                        t"ngpus={ngpus} bits={quant_bits} rows={rows}"
                        t" cols={cols} gpu={g} row={row} col={c}: |{got} -"
                        t" {expected}| > {bound}"
                    ),
                )
    assert_true(
        err_sum < 0.5 * bound_sum,
        String(
            t"ngpus={ngpus} bits={quant_bits} rows={rows} cols={cols}: mean"
            t" error {err_sum} is not well inside the bound {bound_sum}"
        ),
    )
    _ = in_bufs^
    _ = out_bufs^
    _ = res_bufs^
    _ = signal_buffers^


def run_shapes[
    ngpus: Int, quant_bits: Int
](list_of_ctx: List[DeviceContext[]]) raises:
    run_case[ngpus, quant_bits, False](list_of_ctx, 4 * ngpus * 16, 6144)
    run_case[ngpus, quant_bits, True](list_of_ctx, 4 * ngpus * 16 + 3, 6144)
    run_case[ngpus, quant_bits, True](list_of_ctx, 2 * ngpus * 600 + 1, 256)
    run_case[ngpus, quant_bits, False](list_of_ctx, ngpus - 1, 64)


def main() raises:
    assert_true(
        DeviceContext.number_of_devices() > 1, "must have multiple GPUs"
    )
    assert_true(enable_p2p(), "failed to enable P2P access between GPUs")
    var list_of_ctx = List[DeviceContext[]](capacity=MAX_GPUS)
    for i in range(DeviceContext.number_of_devices()):
        list_of_ctx.append(DeviceContext(i))

    run_shapes[2, 4](list_of_ctx)
    run_shapes[2, 8](list_of_ctx)
    if len(list_of_ctx) >= 4:
        run_shapes[4, 4](list_of_ctx)
        run_shapes[4, 8](list_of_ctx)
