# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s -t

from gpu.host.device_context import DeviceContext
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from internal_utils import DeviceNDBuffer
from gpu.host._utils import _human_memory


@value
struct Direction:
    var value: Int
    alias DEVICE_TO_HOST = Self(0)
    alias HOST_TO_DEVICE = Self(1)

    @no_inline
    fn __str__(self) -> String:
        if self is Self.DEVICE_TO_HOST:
            return "device_to_host"
        else:
            return "host_to_device"

    fn __is__(self, other: Self) -> Bool:
        return self.value == other.value


@no_inline
fn bench_memcpy[
    direction: Direction
](inout b: Bench, *, length: Int, context: DeviceContext) raises:
    alias dtype = DType.float32
    var mem_host = DTypePointer[dtype].alloc(length)

    var mem_device = context.create_buffer[dtype](length)

    @parameter
    @always_inline
    fn bench_func(inout b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            @parameter
            if direction is Direction.DEVICE_TO_HOST:
                context.enqueue_copy_from_device(mem_host, mem_device)
            else:
                context.enqueue_copy_to_device(mem_device, mem_host)

        b.iter_custom[kernel_launch](context)

    b.bench_function[bench_func](
        BenchId(
            "memcpy_" + str(direction),
            input_id="length=" + _human_memory(length),
        ),
        ThroughputMeasure(BenchMetric.bytes, length * sizeof[dtype]()),
    )
    context.synchronize()

    mem_host.free()
    _ = mem_device


def main():
    with DeviceContext() as ctx:
        var m = Bench()
        for log2_length in range(28, 32):
            var length = 1 << log2_length
            bench_memcpy[Direction.DEVICE_TO_HOST](
                m, length=length, context=ctx
            )
            bench_memcpy[Direction.HOST_TO_DEVICE](
                m, length=length, context=ctx
            )
        m.dump_report()
