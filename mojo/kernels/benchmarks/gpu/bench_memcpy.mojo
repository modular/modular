# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-build-no-debug-no-assert %s

from math import floor, iota
from os import abort
from sys import env_get_int, sizeof

from algorithm.functional import parallelize_over_rows
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from gpu.host import DeviceContext
from memory import UnsafePointer
from testing import assert_almost_equal, assert_true

from utils import IndexList
from internal_utils import arg_parse


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
    alias P2P = 3
    # Different possible configurations.
    alias DEVICE_TO_HOST = Self(Self.DToH, False)
    alias DEVICE_TO_HOST_PINNED = Self(Self.DToH, True)
    alias HOST_TO_DEVICE = Self(Self.HToD, False)
    alias HOST_PINNED_TO_DEVICE = Self(Self.HToD, True)
    alias DEVICE_TO_DEVICE = Self(Self.DToD, False)
    alias PEER_TO_PEER = Self(Self.P2P, False)
    alias UNDEFINED = Self(-1, False)

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    fn __eq__(self, other: Self) -> Bool:
        return (
            self.direction == other.direction
            and self.pinned_memory == other.pinned_memory
        )

    @staticmethod
    fn get(handle: String) -> Self:
        if handle == "host_to_device":
            return Self.HOST_TO_DEVICE
        elif handle == "host_pinned_to_device":
            return Self.HOST_PINNED_TO_DEVICE
        elif handle == "device_to_host":
            return Self.DEVICE_TO_HOST
        elif handle == "device_to_host_pinned":
            return Self.DEVICE_TO_HOST_PINNED
        elif handle == "device_to_device":
            return Self.DEVICE_TO_DEVICE
        elif handle == "peer_to_peer":
            return Self.PEER_TO_PEER
        else:
            print("UNDEFINED")
            print(
                "options: host_to_device, host_pinned_to_device,"
                " device_to_host, device_to_host_pinned, device_to_device,"
                " peer_to_peer"
            )
            return Self.UNDEFINED

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
fn bench_memcpy(
    mut b: Bench, *, length: Int, config: Config, context: DeviceContext
) raises:
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
    var m = Bench()

    var log2_length = arg_parse("log2_length", 20)
    var mode = arg_parse("mode", "host_to_device")
    assert_true(log2_length > 0)
    var length = 1 << log2_length
    var config = Config.get(mode)

    if not (config == Config.UNDEFINED) and not (config == Config.PEER_TO_PEER):
        with DeviceContext() as ctx:
            bench_memcpy(m, length=length, config=config, context=ctx)

    elif config.direction == Config.P2P:
        var num_devices = DeviceContext.number_of_devices()
        if num_devices > 1:
            # Create contexts for both same-device and peer device transfers
            var ctx1 = DeviceContext(device_id=0)
            var ctx2 = DeviceContext(device_id=1)

            # Benchmark peer context D2D
            bench_p2p(m, length=length, ctx1=ctx1, ctx2=ctx2)
        else:
            print("Only one device found, skipping peer-to-peer benchmarks")

    m.dump_report()
