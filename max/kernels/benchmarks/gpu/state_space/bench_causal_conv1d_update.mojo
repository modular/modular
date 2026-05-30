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
"""GPU benchmark for the causal conv1d update (autoregressive decode) kernel.

Decode step: seqlen=1, conv_state holds the prior width-1 inputs. Mirrors the
Dao-AILab causal_conv1d_update workload (dim=1536, width=4). Sweeps batch to
cover single-stream decode (B=1) and datacenter batched decode (B=128).
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
from state_space.causal_conv1d import causal_conv1d_update_gpu


def main() raises:
    comptime dtype = DType._from_str(
        get_defined_string["dtype", "DType.float32"]()
    )
    comptime batch = get_defined_int["batch", 1]()
    comptime dim = get_defined_int["dim", 1536]()
    comptime kWidth = get_defined_int["width", 4]()
    comptime seqlen = 1
    comptime state_len = kWidth  # conv_state holds kWidth taps

    var n = batch * dim * seqlen
    var ns = batch * dim * state_len

    var m = Bench()
    with DeviceContext() as ctx:
        var x_host = ctx.enqueue_create_host_buffer[dtype](n)
        var cs_host = ctx.enqueue_create_host_buffer[dtype](ns)
        var w_host = ctx.enqueue_create_host_buffer[dtype](dim * kWidth)
        var b_host = ctx.enqueue_create_host_buffer[dtype](dim)
        ctx.synchronize()
        rand[dtype](x_host.unsafe_ptr(), n)
        rand[dtype](cs_host.unsafe_ptr(), ns)
        rand[dtype](w_host.unsafe_ptr(), dim * kWidth)
        rand[dtype](b_host.unsafe_ptr(), dim)

        var x_dev = ctx.enqueue_create_buffer[dtype](n)
        var cs_dev = ctx.enqueue_create_buffer[dtype](ns)
        var w_dev = ctx.enqueue_create_buffer[dtype](dim * kWidth)
        var b_dev = ctx.enqueue_create_buffer[dtype](dim)
        var o_dev = ctx.enqueue_create_buffer[dtype](n)
        with ctx.push_context():
            ctx.enqueue_copy(x_dev, x_host.unsafe_ptr())
            ctx.enqueue_copy(cs_dev, cs_host.unsafe_ptr())
            ctx.enqueue_copy(w_dev, w_host.unsafe_ptr())
            ctx.enqueue_copy(b_dev, b_host.unsafe_ptr())

        var x = TileTensor(x_dev, row_major(batch, dim, seqlen))
        var cs = TileTensor(cs_dev, row_major(batch, dim, state_len))
        var w = TileTensor(w_dev, row_major(dim, kWidth))
        var o = TileTensor(o_dev, row_major(batch, dim, seqlen))
        var b = TileTensor(b_dev, row_major(dim))

        var x_batch_stride = UInt32(dim * seqlen)
        var x_c_stride = UInt32(seqlen)
        var cs_batch_stride = UInt32(dim * state_len)
        var cs_c_stride = UInt32(state_len)
        var o_batch_stride = UInt32(dim * seqlen)
        var o_c_stride = UInt32(seqlen)

        @parameter
        @always_inline
        def bench_cfg[kNThreads: Int]() raises:
            var compiled = ctx.compile_function[
                causal_conv1d_update_gpu[
                    dtype,
                    dtype,
                    dtype,
                    dtype,
                    dtype,
                    kNThreads,
                    x.LayoutType,
                    cs.LayoutType,
                    w.LayoutType,
                    o.LayoutType,
                    b.LayoutType,
                ]
            ]()
            var grid = (batch, ceildiv(dim, kNThreads))

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
                        state_len,
                        x,
                        cs,
                        w,
                        o,
                        b,
                        x_batch_stride,
                        x_c_stride,
                        UInt32(1),
                        cs_batch_stride,
                        cs_c_stride,
                        UInt32(1),
                        UInt32(kWidth),
                        UInt32(1),
                        o_batch_stride,
                        o_c_stride,
                        UInt32(1),
                        Int8(1),
                        Int8(1),
                        grid_dim=grid,
                        block_dim=(kNThreads),
                    )

                bn.iter_custom[launch](ctx)

            m.bench_function[run](
                BenchId(
                    "update",
                    input_id=String("nthreads=", kNThreads, "/batch=", batch),
                ),
                [ThroughputMeasure(BenchMetric.elements, n)],
            )

        bench_cfg[128]()

        ctx.synchronize()

    m.dump_report()
