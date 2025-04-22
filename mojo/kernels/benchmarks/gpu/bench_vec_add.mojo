# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
# RUN: %mojo-build-no-debug-no-assert %s

from pathlib import Path
from sys import env_get_int

from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from builtin._closure import __ownership_keepalive
from gpu import *
from gpu.host import DeviceContext
from testing import assert_equal


fn vec_func(
    in0: UnsafePointer[Float32],
    in1: UnsafePointer[Float32],
    out: UnsafePointer[Float32],
    len: Int,
):
    var tid = global_idx.x
    if tid >= len:
        return
    out[tid] = in0[tid] + in1[tid]


@no_inline
fn bench_vec_add(
    mut b: Bench, *, block_dim: Int, length: Int, context: DeviceContext
) raises:
    alias dtype = DType.float32
    var in0_host = UnsafePointer[Scalar[dtype]].alloc(length)
    var in1_host = UnsafePointer[Scalar[dtype]].alloc(length)
    var out_host = UnsafePointer[Scalar[dtype]].alloc(length)

    for i in range(length):
        in0_host[i] = i
        in1_host[i] = 2

    var in0_device = context.enqueue_create_buffer[dtype](length)
    var in1_device = context.enqueue_create_buffer[dtype](length)
    var out_device = context.enqueue_create_buffer[dtype](length)

    context.enqueue_copy(in0_device, in0_host)
    context.enqueue_copy(in1_device, in1_host)

    @always_inline
    @parameter
    fn run_func() raises:
        context.enqueue_function[vec_func](
            in0_device,
            in1_device,
            out_device,
            length,
            grid_dim=(length // block_dim),
            block_dim=(block_dim),
        )

    @parameter
    @always_inline
    fn bench_func(mut b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            run_func()

        b.iter_custom[kernel_launch](context)

    b.bench_function[bench_func](
        BenchId("vec_add", input_id=String("block_dim=", block_dim)),
        ThroughputMeasure(BenchMetric.flops, length),
    )
    context.synchronize()
    context.enqueue_copy(out_host, out_device)

    for i in range(length):
        assert_equal(i + 2, out_host[i])

    __ownership_keepalive(in0_device, in1_device, out_device)

    in0_host.free()
    in1_host.free()
    out_host.free()


# CHECK-NOT: CUDA_ERROR
def main():
    alias block_dim = env_get_int["block_dim", 32]()
    var m = Bench()

    with DeviceContext() as ctx:
        bench_vec_add(m, block_dim=block_dim, length=32 * 1024, context=ctx)

    m.dump_report()
