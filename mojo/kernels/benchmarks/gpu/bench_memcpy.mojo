# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-build-no-debug %s

from sys import sizeof, env_get_int
from math import floor
from memory import UnsafePointer
from gpu.host import DeviceContext
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure


fn _pretty_print_float(val: Float64) -> String:
    """This converts the float value to a string, but omits the fractional part
    if not needed (e.g. prints 2 instead of 2.0).
    """
    if Float64(floor(val)) == val:
        return str(int(val))
    return str(val)


fn _human_memory(size: Int) -> String:
    alias KB = 1024
    alias MB = KB * KB
    alias GB = MB * KB

    if size >= GB:
        return _pretty_print_float(Float64(size) / GB) + "GB"

    if size >= MB:
        return _pretty_print_float(Float64(size) / MB) + "MB"

    if size >= KB:
        return _pretty_print_float(Float64(size) / KB) + "KB"

    return str(size) + "B"


@value
struct Config:
    var direction: Int
    var pinned_memory: Bool
    # Definitions for direction field.
    alias DToH = 0
    alias HToD = 1
    alias DToD = 2
    # Different possible configurations.
    alias DEVICE_TO_HOST = Self(Self.DToH, False)
    alias DEVICE_TO_HOST_PINNED = Self(Self.DToH, True)
    alias HOST_TO_DEVICE = Self(Self.HToD, False)
    alias HOST_PINNED_TO_DEVICE = Self(Self.HToD, True)
    alias DEVICE_TO_DEVICE = Self(Self.DToD, False)

    @no_inline
    fn __str__(self) -> String:
        var host_str: String
        if self.pinned_memory:
            host_str = "host_pinned"
        else:
            host_str = "host"
        if self.direction == Self.DToH:
            return "device_to_" + host_str
        elif self.direction == Self.HToD:
            return host_str + "_to_device"
        else:
            return "device_to_device"


@no_inline
fn bench_memcpy[
    config: Config
](inout b: Bench, *, length: Int, context: DeviceContext) raises:
    alias dtype = DType.float32
    var mem_host = context.malloc_host[Scalar[dtype]](
        length
    ) if config.pinned_memory else UnsafePointer[Scalar[dtype]].alloc(length)

    var mem_device = context.create_buffer[dtype](length)
    var mem2_device = context.create_buffer[dtype](length)

    @parameter
    @always_inline
    fn bench_func(inout b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            @parameter
            if config.direction == Config.DToH:
                context.enqueue_copy_from_device(mem_host, mem_device)
            elif config.direction == Config.HToD:
                context.enqueue_copy_to_device(mem_device, mem_host)
            else:
                context.enqueue_copy_device_to_device(mem_device, mem2_device)

        b.iter_custom[kernel_launch](context)

    b.bench_function[bench_func](
        BenchId(
            "memcpy_" + str(config),
            input_id="length=" + _human_memory(length),
        ),
        ThroughputMeasure(BenchMetric.bytes, length * sizeof[dtype]()),
    )
    context.synchronize()

    if config.pinned_memory:
        context.free_host(mem_host)
    else:
        mem_host.free()
    _ = mem_device
    _ = mem2_device


def main():
    alias log2_length = env_get_int["log2_length", 20]()
    constrained[log2_length > 0]()
    var m = Bench()
    with DeviceContext() as ctx:
        var length = 1 << log2_length
        bench_memcpy[Config.DEVICE_TO_HOST](m, length=length, context=ctx)
        bench_memcpy[Config.DEVICE_TO_HOST_PINNED](
            m, length=length, context=ctx
        )
        bench_memcpy[Config.HOST_TO_DEVICE](m, length=length, context=ctx)
        bench_memcpy[Config.HOST_PINNED_TO_DEVICE](
            m, length=length, context=ctx
        )
        bench_memcpy[Config.DEVICE_TO_DEVICE](m, length=length, context=ctx)
    m.dump_report()
