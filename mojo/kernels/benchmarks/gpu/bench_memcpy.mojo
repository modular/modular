# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-build-no-debug-no-assert %s

from sys import sizeof, env_get_int
from os import abort
from math import iota, floor
from algorithm.functional import parallelize_over_rows
from memory import UnsafePointer
from gpu.host import DeviceContext
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from testing import assert_almost_equal
from utils import IndexList


fn _pretty_print_float(val: Float64) -> String:
    """This converts the float value to a string, but omits the fractional part
    if not needed (e.g. prints 2 instead of 2.0).
    """
    if Float64(floor(val)) == val:
        return String(Int(val))
    return String(val)


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

    return String(size) + "B"


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
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        if self.direction == Self.DToD:
            writer.write("device_to_device")
            return

        if self.direction == Self.DToH:
            writer.write("device_to_")

        if self.pinned_memory:
            writer.write("host_pinned")
        else:
            writer.write("host")

        if self.direction == Self.HToD:
            writer.write("_to_device")


@no_inline
fn bench_memcpy[
    config: Config
](mut b: Bench, *, length: Int, context: DeviceContext) raises:
    alias dtype = DType.float32
    var mem_host = context.malloc_host[Scalar[dtype]](
        length
    ) if config.pinned_memory else UnsafePointer[Scalar[dtype]].alloc(length)

    var mem_device = context.enqueue_create_buffer[dtype](length)
    var mem2_device = context.enqueue_create_buffer[dtype](length)

    @parameter
    @always_inline
    fn bench_func(mut b: Bencher):
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
            String("memcpy_", config),
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


@no_inline
fn bench_p2p(
    mut b: Bench, *, length: Int, ctx1: DeviceContext, ctx2: DeviceContext
) raises:
    alias dtype = DType.float32

    # Create host buffers for verification
    var host_ptr = UnsafePointer[Scalar[dtype]].alloc(length)

    # Initialize source data with known pattern
    iota(host_ptr, length)

    # Create and initialize device buffers
    var src_buf = ctx1.enqueue_create_buffer[dtype](length)
    var dst_buf = ctx2.enqueue_create_buffer[dtype](length)

    # Copy initial data to source buffer
    ctx1.enqueue_copy_to_device(src_buf, host_ptr)
    ctx1.synchronize()

    @parameter
    @always_inline
    fn bench_func(mut b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            ctx2.enqueue_copy_device_to_device(dst_buf, src_buf)

        b.iter_custom[kernel_launch](ctx1)

    # Calculate bytes transferred
    var bytes_transferred = length * sizeof[dtype]()

    # Create list of throughput measures
    var measures = List[ThroughputMeasure](
        # Raw bandwidth (considering only one transfer)
        ThroughputMeasure(BenchMetric.bytes, bytes_transferred),
    )

    b.bench_function[bench_func](
        BenchId(
            "memcpy_p2p",
            input_id="length=" + _human_memory(bytes_transferred),
        ),
        measures=measures,
    )

    # Copy back for verification
    ctx2.enqueue_copy_from_device(host_ptr, dst_buf)
    ctx2.synchronize()

    # Parallel verification
    @parameter
    fn verify_chunk(start: Int, end: Int):
        for i in range(start, end):
            try:
                assert_almost_equal(host_ptr[i], i)
            except e:
                print("Verification failed at index", i)
                print("Expected:", i, "Got:", host_ptr[i])
                abort(e)

    # Parallelize verification using sync_parallelize
    var shape = IndexList[1](
        length,
    )
    parallelize_over_rows[verify_chunk](shape, 0, 256)

    # Cleanup
    host_ptr.free()
    _ = src_buf
    _ = dst_buf


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

    var num_devices = DeviceContext.number_of_devices()
    if num_devices > 1:
        # Create contexts for both same-device and peer device transfers
        var ctx1 = DeviceContext(device_id=0)
        var ctx2 = DeviceContext(device_id=1)

        var length = 1 << log2_length
        # Benchmark peer context D2D
        bench_p2p(m, length=length, ctx1=ctx1, ctx2=ctx2)
    else:
        print("Only one device found, skipping peer-to-peer benchmarks")

    m.dump_report()
