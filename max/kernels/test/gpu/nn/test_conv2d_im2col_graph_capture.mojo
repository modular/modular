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

"""`dispatch_im2col_matmul_conv2d` records into a device graph.

The dispatcher used to end with `ctx.synchronize()`, which a recording context
rejects, so the bf16 patch-embed and strided Conv1D shapes could not be graph
captured. This test records the dispatcher, replays the graph, and requires
the replayed output to match the eager call bit for bit, including after the
allocator has been churned between replays (a scratch buffer that was returned
to the pool at record time would be overwritten by the churn).

NVIDIA only: the recording context does not exist on other backends.
"""

from layout import Coord, TileTensor, row_major
from max.gpu.host import (
    DeviceBuffer,
    DeviceContext,
    DeviceGraph,
    DeviceGraphBuilder,
)
from nn.conv.gpu.im2col_matmul_2d import (
    _DEFAULT_M_TILE_BYTE_BUDGET,
    _fp16_im2col_wins,
    dispatch_im2col_matmul_conv2d,
)
from std.random import rand
from std.testing import assert_almost_equal, assert_equal, assert_true


def _to_host_f32[
    dtype: DType
](ctx: DeviceContext, buf: DeviceBuffer[dtype], size: Int) raises -> List[
    Float32
]:
    var host = ctx.enqueue_create_host_buffer[dtype](size)
    ctx.enqueue_copy(host, buf)
    ctx.synchronize()
    var out = List[Float32](capacity=size)
    for i in range(size):
        out.append(host[i].cast[.float32]())
    return out^


def _expect_equal(
    expected: List[Float32], actual: List[Float32], what: String
) raises:
    for i in range(len(expected)):
        if expected[i] != actual[i]:
            print(t"{what} mismatch at {i} {expected[i]} {actual[i]}")
        assert_equal(expected[i], actual[i])


def test_graph_capture[
    H: Int,
    W: Int,
    C: Int,
    F: Int,
    R: Int,
    S: Int,
    stride_h: Int,
    stride_w: Int,
    filter_is_fcrs: Bool,
    dtype: DType = .bfloat16,
    m_tile_byte_budget: Int = _DEFAULT_M_TILE_BYTE_BUDGET,
](ctx: DeviceContext) raises:
    comptime H_out = (H - R) // stride_h + 1
    comptime W_out = (W - S) // stride_w + 1
    comptime input_size = H * W * C
    comptime filter_size = R * S * C * F
    comptime output_size = H_out * W_out * F
    comptime stride = Coord(stride_h, stride_w)
    comptime dilation = Coord(1, 1)
    comptime pad = Coord(0, 0)

    print(
        t"test_graph_capture: {H}x{W}x{C} -> {F} filter {R}x{S} stride"
        t" {stride_h}x{stride_w} fcrs={filter_is_fcrs} {dtype}"
    )

    var input_host = ctx.enqueue_create_host_buffer[dtype](input_size)
    var filter_host = ctx.enqueue_create_host_buffer[dtype](filter_size)
    rand(input_host.as_span())
    rand(filter_host.as_span())

    var input_dev = ctx.enqueue_create_buffer[dtype](input_size)
    var filter_dev = ctx.enqueue_create_buffer[dtype](filter_size)
    var output_dev = ctx.enqueue_create_buffer[dtype](output_size)
    ctx.enqueue_copy(input_dev, input_host)
    ctx.enqueue_copy(filter_dev, filter_host)
    ctx.enqueue_memset(output_dev, Scalar[dtype](0))

    # `run` captures pointers rather than buffers: the replays below fill
    # `output_dev` while the closure is still alive.
    var input_ptr = input_dev.unsafe_ptr()
    var filter_ptr = filter_dev.unsafe_ptr()
    var output_ptr = output_dev.unsafe_ptr()

    # The same call serves the eager launch and the recording context.
    def run(c: DeviceContext) raises {imm}:
        var input_tt = TileTensor(input_ptr, row_major[1, H, W, C]())
        var output_tt = TileTensor(output_ptr, row_major[1, H_out, W_out, F]())
        var handled: Bool
        comptime if filter_is_fcrs:
            var filter_tt = TileTensor(filter_ptr, row_major[F, C, R, S]())
            handled = dispatch_im2col_matmul_conv2d[
                filter_is_fcrs=True, m_tile_byte_budget=m_tile_byte_budget
            ](input_tt, filter_tt, output_tt, stride, dilation, pad, 1, c)
        else:
            var filter_tt = TileTensor(filter_ptr, row_major[R, S, C, F]())
            handled = dispatch_im2col_matmul_conv2d[
                m_tile_byte_budget=m_tile_byte_budget
            ](input_tt, filter_tt, output_tt, stride, dilation, pad, 1, c)
        assert_true(handled, "the dispatcher must take this shape")

    # Eager reference; it also warms the kernel compiles before recording.
    run(ctx)
    var expected = _to_host_f32(ctx, output_dev, output_size)

    # The eager result must also be the convolution, in fp32 on the host, at
    # the first, middle and last output pixels (all channels).
    for pix in range(3):
        var ho = 0 if pix == 0 else (H_out // 2 if pix == 1 else H_out - 1)
        var wo = 0 if pix == 0 else (W_out // 2 if pix == 1 else W_out - 1)
        for f in range(F):
            var acc = Float32(0)
            for r in range(R):
                for sx in range(S):
                    for c in range(C):
                        var x = input_host[
                            ((ho * stride_h + r) * W + wo * stride_w + sx) * C
                            + c
                        ].cast[.float32]()
                        var widx: Int
                        comptime if filter_is_fcrs:
                            widx = ((f * C + c) * R + r) * S + sx
                        else:
                            widx = ((r * S + sx) * C + c) * F + f
                        acc += x * filter_host[widx].cast[.float32]()
            assert_almost_equal(
                acc,
                expected[(ho * W_out + wo) * F + f],
                rtol=2e-2,
                atol=2e-2,
            )

    def record(mut gb: DeviceGraphBuilder) raises {imm}:
        run(gb.recording_context())

    ctx.synchronize()
    var graph = DeviceGraph.create(ctx, record)

    ctx.enqueue_memset(output_dev, Scalar[dtype](0))
    graph.replay()
    _expect_equal(
        expected, _to_host_f32(ctx, output_dev, output_size), "replay"
    )

    # Churn the allocator, then replay again: a scratch buffer that went back
    # to the pool at record time would now be overwritten.
    var churn = List[DeviceBuffer[dtype]]()
    for _ in range(8):
        var b = ctx.enqueue_create_buffer[dtype](input_size * 4 + 4096)
        ctx.enqueue_memset(b, Scalar[dtype](7))
        churn.append(b^)
    ctx.enqueue_memset(output_dev, Scalar[dtype](0))
    graph.replay()
    _expect_equal(
        expected, _to_host_f32(ctx, output_dev, output_size), "churned replay"
    )

    _ = churn^
    _ = graph^
    _ = input_dev^
    _ = filter_dev^
    _ = output_dev^


def main() raises:
    with DeviceContext() as ctx:
        comptime if not ctx.target.is_nvidia_gpu():
            print("SKIP: device graphs are recorded on NVIDIA GPUs only")
            return

        # Patch embed: 14x14 stride-14 over C=3, FCRS filter.
        test_graph_capture[112, 112, 3, 128, 14, 14, 14, 14, True](ctx)
        # Unaligned K (588) with an RSCF filter pads K like the FCRS one.
        test_graph_capture[112, 112, 3, 128, 14, 14, 14, 14, False](ctx)
        # Strided Conv1D as a rank-4 conv: 1x5 stride 2, FCRS and RSCF filters.
        test_graph_capture[1, 59, 128, 256, 1, 5, 1, 2, True](ctx)
        test_graph_capture[1, 59, 128, 256, 1, 5, 1, 2, False](ctx)
        # fp16 takes the same path for the big patchify and strided Conv1D
        # classes; the Conv1D widths vary M, as a vendor GEMM under graph
        # capture is shape dependent.
        test_graph_capture[672, 672, 3, 1152, 14, 14, 14, 14, True, .float16](
            ctx
        )
        test_graph_capture[1, 37, 512, 1024, 1, 5, 1, 2, True, .float16](ctx)
        test_graph_capture[1, 133, 512, 1024, 1, 5, 1, 2, True, .float16](ctx)
        test_graph_capture[1, 601, 512, 1024, 1, 5, 1, 2, True, .float16](ctx)

        # fp16 stays on cuDNN except for big patchify and strided Conv1D: the
        # WAN VAE shapes (cases 0, 3, 6), the stride-1 Conv1D (case 15) and
        # the strided 3x3 / 7x7 / small-patch convs measured faster on cuDNN.
        assert_true(not _fp16_im2col_wins(Coord(1, 1), 3, 3, 99840, 96, 864))
        assert_true(not _fp16_im2col_wins(Coord(1, 1), 3, 3, 99840, 96, 1728))
        assert_true(not _fp16_im2col_wins(Coord(1, 1), 3, 3, 24960, 192, 1728))
        assert_true(not _fp16_im2col_wins(Coord(1, 1), 1, 5, 546, 1024, 2560))
        assert_true(not _fp16_im2col_wins(Coord(2, 2), 3, 3, 1024, 256, 1152))
        assert_true(not _fp16_im2col_wins(Coord(2, 2), 7, 7, 12544, 64, 147))
        assert_true(not _fp16_im2col_wins(Coord(4, 4), 4, 4, 3136, 96, 48))
        assert_true(not _fp16_im2col_wins(Coord(2, 2), 3, 3, 64, 512, 4608))
        # The patch embed and the strided Conv1D (cases 12, 18) take it.
        assert_true(_fp16_im2col_wins(Coord(14, 14), 14, 14, 4096, 1152, 588))
        assert_true(_fp16_im2col_wins(Coord(1, 2), 1, 5, 136, 1024, 2560))

        # Several M tiles: the loop reuses one scratch buffer across tiles.
        test_graph_capture[
            1, 4096, 64, 64, 1, 5, 1, 2, False, m_tile_byte_budget=64 * 1024
        ](ctx)
