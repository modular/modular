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
"""GPU benchmark for the channel-first causal conv1d forward kernel.

Default workload mirrors Mamba(1)-130m prefill: B=1, dim=1536, L=256, width=4,
FP32, channel-first (B, C, L) — the layout Dao-AILab/causal-conv1d uses. Sweeps
several (kNThreads, kNElts) block configs in a single run (one GPU init) to find
the best occupancy/tiling for the workload. Benchmarks the kernel directly via
`compile_function` + `enqueue_function` (no graph-op registration needed).
"""

from std.math import ceildiv
from std.sys import get_defined_int, get_defined_string

from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from std.gpu.host import DeviceContext
from std.random import rand
from layout import TileTensor, row_major
from state_space.causal_conv1d import (
    causal_conv1d_channel_first_fwd_gpu,
    causal_conv1d_channel_last_fwd_gpu,
)


def main() raises:
    comptime dtype = DType._from_str(
        get_defined_string["dtype", "DType.float32"]()
    )
    comptime batch = get_defined_int["batch", 1]()
    comptime dim = get_defined_int["dim", 1536]()
    comptime seqlen = get_defined_int["seqlen", 256]()
    comptime kWidth = get_defined_int["width", 4]()

    var n = batch * dim * seqlen

    var m = Bench()
    with DeviceContext() as ctx:
        var x_host = ctx.enqueue_create_host_buffer[dtype](n)
        var w_host = ctx.enqueue_create_host_buffer[dtype](dim * kWidth)
        var b_host = ctx.enqueue_create_host_buffer[dtype](dim)
        ctx.synchronize()
        rand[dtype](x_host.unsafe_ptr(), n)
        rand[dtype](w_host.unsafe_ptr(), dim * kWidth)
        rand[dtype](b_host.unsafe_ptr(), dim)

        var x_dev = ctx.enqueue_create_buffer[dtype](n)
        var w_dev = ctx.enqueue_create_buffer[dtype](dim * kWidth)
        var b_dev = ctx.enqueue_create_buffer[dtype](dim)
        var o_dev = ctx.enqueue_create_buffer[dtype](n)
        with ctx.push_context():
            ctx.enqueue_copy(x_dev, x_host.unsafe_ptr())
            ctx.enqueue_copy(w_dev, w_host.unsafe_ptr())
            ctx.enqueue_copy(b_dev, b_host.unsafe_ptr())

        var x = TileTensor(x_dev, row_major(batch, dim, seqlen))
        var o = TileTensor(o_dev, row_major(batch, dim, seqlen))
        var w = TileTensor(w_dev, row_major(dim, kWidth))
        var b = TileTensor(b_dev, row_major(dim))
        var x_batch_stride: UInt32 = UInt32(dim * seqlen)
        var x_c_stride: UInt32 = UInt32(seqlen)
        var x_l_stride: UInt32 = 1

        @parameter
        @always_inline
        def bench_cfg[kNThreads: Int, kNElts: Int, kCPB: Int = 1]() raises:
            # kCPB = channels folded per block (block_dim.y). >1 raises warps/
            # block to improve occupancy on the latency-bound launch.
            var compiled = ctx.compile_function[
                causal_conv1d_channel_first_fwd_gpu[
                    dtype,
                    dtype,
                    dtype,
                    kNThreads,
                    kWidth,
                    kNElts,
                    dtype,
                    dtype,
                    x.LayoutType,
                    w.LayoutType,
                    o.LayoutType,
                    b.LayoutType,
                    b.LayoutType,
                ]
            ]()
            var grid = (
                ceildiv(seqlen, kNThreads * kNElts),
                ceildiv(dim, kCPB),
                batch,
            )

            @parameter
            @always_inline
            def run(mut bn: Bencher):
                @parameter
                @always_inline
                def launch(c: DeviceContext, i: Int) raises:
                    c.enqueue_function(
                        compiled,
                        batch,
                        dim,
                        seqlen,
                        kWidth,
                        x.as_immut(),
                        w.as_immut(),
                        o,
                        b.as_immut(),
                        b.as_immut(),
                        x_batch_stride,
                        x_c_stride,
                        x_l_stride,
                        UInt32(kWidth),
                        UInt32(1),
                        x_batch_stride,
                        x_c_stride,
                        x_l_stride,
                        UInt32(1),
                        UInt32(0),
                        UInt32(0),
                        Int8(1),
                        Int8(0),
                        Int8(1),
                        grid_dim=grid,
                        block_dim=(kNThreads, kCPB),
                    )

                bn.iter_custom[launch](ctx)

            m.bench_function[run](
                BenchId(
                    "cf",
                    input_id=String(
                        "nthreads=", kNThreads, "/nelts=", kNElts, "/cpb=", kCPB
                    ),
                ),
                [ThroughputMeasure(BenchMetric.elements, n)],
            )

        # Winning config: kNThreads*kNElts == seqlen tile, full L utilization,
        # one channel/block. kNElts is the 128-bit vector width: 4 for fp32,
        # 8 for 16-bit dtypes. Channel folding (cpb>1) did not help.
        comptime if dtype == DType.float32:
            bench_cfg[64, 4, 1]()
        else:
            # 16-bit at L=256: full utilization (tile==L) beats a wider but
            # half-idle load. 64x4 (tile 256, 64-bit load) measured fastest.
            bench_cfg[64, 4, 1]()

        # ---- Channel-last (B, L, C). Reuse the same device data, reinterpreted
        # with channel-last strides (c_stride=1, l_stride=dim). Vectorization is
        # across the kNElts contiguous channels.
        var xcl = TileTensor(x_dev, row_major(batch, seqlen, dim))
        var ocl = TileTensor(o_dev, row_major(batch, seqlen, dim))
        var cl_batch_stride: UInt32 = UInt32(seqlen * dim)
        var cl_l_stride: UInt32 = UInt32(dim)
        var cl_c_stride: UInt32 = 1

        @parameter
        @always_inline
        def bench_cl[kNThreads: Int, kNElts: Int]() raises:
            var compiled = ctx.compile_function[
                causal_conv1d_channel_last_fwd_gpu[
                    dtype,
                    dtype,
                    dtype,
                    kNThreads,
                    kWidth,
                    kNElts,
                    dtype,
                    dtype,
                    xcl.LayoutType,
                    w.LayoutType,
                    ocl.LayoutType,
                    b.LayoutType,
                    b.LayoutType,
                ]
            ]()
            var grid = (
                ceildiv(seqlen, kNElts),
                ceildiv(dim, kNThreads),
                batch,
            )

            @parameter
            @always_inline
            def run(mut bn: Bencher):
                @parameter
                @always_inline
                def launch(c: DeviceContext, i: Int) raises:
                    c.enqueue_function(
                        compiled,
                        batch,
                        dim,
                        seqlen,
                        kWidth,
                        xcl.as_immut(),
                        w.as_immut(),
                        ocl,
                        b.as_immut(),
                        b.as_immut(),
                        cl_batch_stride,
                        cl_c_stride,
                        cl_l_stride,
                        UInt32(kWidth),
                        UInt32(1),
                        cl_batch_stride,
                        cl_c_stride,
                        cl_l_stride,
                        UInt32(1),
                        UInt32(0),
                        UInt32(0),
                        Int8(1),
                        Int8(0),
                        Int8(1),
                        grid_dim=grid,
                        block_dim=(kNThreads),
                    )

                bn.iter_custom[launch](ctx)

            m.bench_function[run](
                BenchId(
                    "cl",
                    input_id=String("nthreads=", kNThreads, "/nelts=", kNElts),
                ),
                [ThroughputMeasure(BenchMetric.elements, n)],
            )

        # 64 channels/block (coalesced), 8 positions/thread: kNElts=8 amortizes
        # the sliding window best (11 loads / 8 outputs). Measured fastest.
        bench_cl[64, 8]()

        ctx.synchronize()

    m.dump_report()
