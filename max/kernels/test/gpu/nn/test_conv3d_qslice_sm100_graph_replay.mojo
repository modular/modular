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
"""`dispatch_qslice_conv3d_sm100` is replay-safe in a device graph.

The dispatcher accumulates Q per-slice 2-D convs into an fp32 scratch
buffer. Its result must not depend on what that scratch buffer held before
the call: a device graph replays the recorded work N times over the same
scratch memory, so any initialization that ran eagerly at record time (rather
than as a recorded node) leaves replay k holding k times the answer.

This test records the dispatcher, replays it several times without touching
the output in between, and requires every replay to match the eager call bit
for bit and the host reference within tolerance.

NVIDIA SM100 only: the dispatcher instantiates tcgen05 kernels.
"""

from std.math import isclose
from std.random import rand
from layout import Coord, TileTensor, row_major
from max.gpu.host import DeviceContext, DeviceGraph, DeviceGraphBuilder
from nn.conv.conv import Naive2dConvolution
from nn.conv.gpu.nvidia.sm100.qslice_conv3d import (
    dispatch_qslice_conv3d_sm100,
)
from std.testing import assert_almost_equal, assert_equal, assert_true

from std.utils.index import Index

comptime NUM_REPLAYS = 4


def test_qslice_graph_replay[
    N: Int,
    D: Int,
    H: Int,
    W: Int,
    C: Int,
    F: Int,
    with_compute_fn: Bool,
](ctx: DeviceContext) raises:
    comptime dtype = DType.bfloat16
    comptime Q = 3
    comptime R = 3
    comptime S = 3

    comptime stride = Coord(1, 1, 1)
    comptime dilation = Coord(1, 1, 1)
    comptime pad = Coord(0, 1, 1)
    comptime D_out = D - Q + 1
    comptime H_out = H + 2 - R + 1
    comptime W_out = W + 2 - S + 1

    comptime input_size = N * D * H * W * C
    comptime filter_size = Q * R * S * C * F
    comptime output_size = N * D_out * H_out * W_out * F

    print(
        t"test_qslice_graph_replay: N={N} D={D} HxW={H}x{W} C={C} F={F}"
        t" compute_fn={with_compute_fn}"
    )

    var input_host = ctx.enqueue_create_host_buffer[dtype](input_size)
    var filter_host = ctx.enqueue_create_host_buffer[dtype](filter_size)
    rand(input_host.as_span())
    rand(filter_host.as_span())

    var ref_accum = ctx.enqueue_create_host_buffer[.float32](output_size)
    Naive2dConvolution[.float32, dtype, dtype].run(
        ref_accum.unsafe_ptr(),
        input_host.unsafe_ptr(),
        filter_host.unsafe_ptr(),
        Coord(N, D_out, H_out, W_out, F),
        Coord(N, D, H, W, C),
        Coord(Q, R, S, C, F),
        Coord(0, 0),
        Coord(1, 1),
        Coord(1, 1),
        stride,
        dilation,
        1,
    )
    comptime scale = Float32(2.0) if with_compute_fn else Float32(1.0)

    var input_dev = ctx.enqueue_create_buffer[dtype](input_size)
    var filter_dev = ctx.enqueue_create_buffer[dtype](filter_size)
    var output_dev = ctx.enqueue_create_buffer[dtype](output_size)
    ctx.enqueue_copy(input_dev, input_host)
    ctx.enqueue_copy(filter_dev, filter_host)
    ctx.enqueue_memset(output_dev, Scalar[dtype](0))

    var input_ptr = input_dev.unsafe_ptr()
    var filter_ptr = filter_dev.unsafe_ptr()
    var output_ptr = output_dev.unsafe_ptr()

    @inline(.always)
    def scale_compute[
        _dtype: DType, _width: SIMDLength, _alignment: Int = 1
    ](coords: Coord, val: SIMD[_dtype, _width]) -> SIMD[_dtype, _width]:
        comptime assert coords.rank == 5
        return (val.cast[.float32]() * 2.0).cast[_dtype]()

    # The same call serves the eager launch and the recording context.
    def run(c: DeviceContext) raises {imm}:
        var input_tt = TileTensor(input_ptr, row_major[N, D, H, W, C]())
        var filter_tt = TileTensor(filter_ptr, row_major[Q, R, S, C, F]())
        var output_tt = TileTensor(
            output_ptr, row_major[N, D_out, H_out, W_out, F]()
        )
        var handled: Bool
        comptime if with_compute_fn:
            handled = dispatch_qslice_conv3d_sm100[has_compute_fn=True](
                input_tt,
                filter_tt,
                output_tt,
                stride,
                dilation,
                pad,
                1,
                c,
                compute_fn=scale_compute,
            )
        else:
            handled = dispatch_qslice_conv3d_sm100(
                input_tt, filter_tt, output_tt, stride, dilation, pad, 1, c
            )
        assert_true(handled, "the dispatcher must take this shape")

    def read_output() raises {imm} -> List[Float32]:
        var host = ctx.enqueue_create_host_buffer[dtype](output_size)
        ctx.enqueue_copy(host, output_dev)
        ctx.synchronize()
        var out = List[Float32](capacity=output_size)
        for i in range(output_size):
            out.append(host[i].cast[.float32]())
        return out^

    def check_vs_reference(got: List[Float32], what: String) raises {imm}:
        for i in range(output_size):
            var want = ref_accum[i] * scale
            if not isclose(got[i], want, atol=2e-2, rtol=2e-2):
                print(t"{what} mismatch at {i} want {want} got {got[i]}")
            assert_almost_equal(got[i], want, rtol=2e-2, atol=2e-2)

    # Eager call: also warms the kernel compiles before recording.
    run(ctx)
    var expected = read_output()
    check_vs_reference(expected, "eager")

    def record(mut gb: DeviceGraphBuilder) raises {imm}:
        run(gb.recording_context())

    ctx.synchronize()
    var graph = DeviceGraph.create(ctx, record)

    # Do not clear the output between replays: the harness that exposed the
    # bug replays back to back, and a stale scratch buffer is the thing under
    # test.
    for k in range(NUM_REPLAYS):
        graph.replay()
        var got = read_output()
        for i in range(output_size):
            if got[i] != expected[i]:
                print(
                    t"replay {k} mismatch at {i} eager {expected[i]}"
                    t" replay {got[i]}"
                )
            assert_equal(got[i], expected[i])
        check_vs_reference(got, "replay")

    _ = graph^
    _ = input_dev^
    _ = filter_dev^
    _ = output_dev^


def main() raises:
    with DeviceContext() as ctx:
        # Q=3, D_out=1, the WAN mid-block shape class, plain and fused.
        test_qslice_graph_replay[1, 3, 16, 16, 128, 128, False](ctx)
        test_qslice_graph_replay[1, 3, 16, 16, 128, 128, True](ctx)
        # D_out=2 and batch=2: the per-batch accumulator offsets.
        test_qslice_graph_replay[2, 4, 16, 16, 128, 192, True](ctx)
