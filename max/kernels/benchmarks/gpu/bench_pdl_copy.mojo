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

from pathlib import Path
from sys import env_get_int

from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from builtin._closure import __ownership_keepalive
from gpu import *
from gpu.grid_controls import PDL, pdl_launch_attributes
from gpu.host import DeviceContext
from testing import assert_equal
from memory import UnsafePointer


fn copy1(
    a: UnsafePointer[Float32],
    b: UnsafePointer[Float32],
    n: Int,
):
    var tmp = Scalar[DType.float32]()
    for i in range(
        block_idx.x * block_dim.x + thread_idx.x, n, block_dim.x * grid_dim.x
    ):
        tmp += b[i]

    launch_dependent_grids()

    for i in range(
        block_idx.x * block_dim.x + thread_idx.x, n, block_dim.x * grid_dim.x
    ):
        b[i] = a[i] + tmp


fn copy2(
    b: UnsafePointer[Float32],
    c: UnsafePointer[Float32],
    d: UnsafePointer[Float32],
    n: Int,
):
    var result = Scalar[DType.float32]()
    for i in range(
        block_idx.x * block_dim.x + thread_idx.x, n, block_dim.x * grid_dim.x
    ):
        result += d[i]

    wait_on_dependent_grids()

    for i in range(
        block_idx.x * block_dim.x + thread_idx.x, n, block_dim.x * grid_dim.x
    ):
        c[i] = b[i] + result + 2.0


fn copy1_n(
    a: UnsafePointer[Float32],
    b: UnsafePointer[Float32],
    n: Int,
):
    var tmp = Scalar[DType.float32]()
    for i in range(
        block_idx.x * block_dim.x + thread_idx.x, n, block_dim.x * grid_dim.x
    ):
        tmp += b[i]

    for i in range(
        block_idx.x * block_dim.x + thread_idx.x, n, block_dim.x * grid_dim.x
    ):
        b[i] = a[i] + tmp


fn copy2_n(
    b: UnsafePointer[Float32],
    c: UnsafePointer[Float32],
    d: UnsafePointer[Float32],
    n: Int,
):
    var result = Scalar[DType.float32]()
    for i in range(
        block_idx.x * block_dim.x + thread_idx.x, n, block_dim.x * grid_dim.x
    ):
        result += d[i]

    for i in range(
        block_idx.x * block_dim.x + thread_idx.x, n, block_dim.x * grid_dim.x
    ):
        c[i] = b[i] + result + 2.0


@no_inline
fn bench_pdl_copy(mut b: Bench, *, length: Int, context: DeviceContext) raises:
    alias dtype = DType.float32
    var a_host = UnsafePointer[Scalar[dtype]].alloc(length)
    var b_host = UnsafePointer[Scalar[dtype]].alloc(length)
    var c_host = UnsafePointer[Scalar[dtype]].alloc(length)
    var d_host = UnsafePointer[Scalar[dtype]].alloc(length)

    alias grid_dim = 16
    alias block_dim = 256

    for i in range(length):
        a_host[i] = i
        b_host[i] = i
        d_host[i] = i

    for i in range(length):
        c_host[i] = 0

    var a_device = context.enqueue_create_buffer[dtype](length)
    var b_device = context.enqueue_create_buffer[dtype](length)
    var c_device = context.enqueue_create_buffer[dtype](length)
    var d_device = context.enqueue_create_buffer[dtype](length)

    context.enqueue_copy(a_device, a_host)
    context.enqueue_copy(b_device, b_host)
    context.enqueue_copy(c_device, c_host)
    context.enqueue_copy(d_device, d_host)

    @always_inline
    @parameter
    fn run_func() raises:
        for _ in range(10):
            context.enqueue_function[copy1](
                a_device,
                b_device,
                length,
                grid_dim=(grid_dim),
                block_dim=(block_dim),
                attributes=pdl_launch_attributes(),
            )
            context.enqueue_function[copy2](
                b_device,
                c_device,
                d_device,
                length,
                grid_dim=(grid_dim),
                block_dim=(block_dim),
                attributes=pdl_launch_attributes(),
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
        BenchId("copy_pdl", input_id=String("length=", length)),
    )
    context.synchronize()
    context.enqueue_copy(c_host, c_device)

    __ownership_keepalive(a_device, b_device, c_device, d_device)

    a_host.free()
    b_host.free()
    c_host.free()
    d_host.free()


@no_inline
fn bench_copy(mut b: Bench, *, length: Int, context: DeviceContext) raises:
    alias dtype = DType.float32
    var a_host = UnsafePointer[Scalar[dtype]].alloc(length)
    var b_host = UnsafePointer[Scalar[dtype]].alloc(length)
    var c_host = UnsafePointer[Scalar[dtype]].alloc(length)
    var d_host = UnsafePointer[Scalar[dtype]].alloc(length)

    alias grid_dim = 16
    alias block_dim = 256

    for i in range(length):
        a_host[i] = i
        b_host[i] = i
        d_host[i] = i

    for i in range(length):
        c_host[i] = 0

    var a_device = context.enqueue_create_buffer[dtype](length)
    var b_device = context.enqueue_create_buffer[dtype](length)
    var c_device = context.enqueue_create_buffer[dtype](length)
    var d_device = context.enqueue_create_buffer[dtype](length)

    context.enqueue_copy(a_device, a_host)
    context.enqueue_copy(b_device, b_host)
    context.enqueue_copy(c_device, c_host)
    context.enqueue_copy(d_device, d_host)

    @always_inline
    @parameter
    fn run_func() raises:
        for _ in range(10):
            context.enqueue_function[copy1_n](
                a_device,
                b_device,
                length,
                grid_dim=(grid_dim),
                block_dim=(block_dim),
            )
            context.enqueue_function[copy2_n](
                b_device,
                c_device,
                d_device,
                length,
                grid_dim=(grid_dim),
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
        BenchId("copy_n", input_id=String("length=", length)),
    )
    context.synchronize()
    context.enqueue_copy(c_host, c_device)

    __ownership_keepalive(a_device, b_device, c_device, d_device)

    a_host.free()
    b_host.free()
    c_host.free()
    d_host.free()


def main():
    alias length = env_get_int["length", 4096]()
    var m = Bench()

    with DeviceContext() as ctx:
        bench_pdl_copy(m, length=length, context=ctx)
        bench_copy(m, length=length, context=ctx)

    m.dump_report()
