# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Launches `vec_add`, `axpy`, `add` and `matmul` over every dtype through the
typed `Queue.execute` and `Stream.execute`, which pack HAL buffers, device
pointers, scalars and `TileTensor`s as each kernel declares them.

The same source runs on every HAL device: each launch compiles its kernel
behind the plugin's function wrapper and launches it with whichever execution
config the plugin takes.
"""

from std.sys import size_of
from std.testing import assert_equal

from layout import TensorLayout, TileTensor, row_major
from max._driver import Driver, Context
from max._driver.max_license.hal.buffer import Buffer
from max._driver.max_license.hal._machine import DeviceSpec
from max._driver.max_license.hal.queue import Queue

from ukl_test_support import (
    alloc_zeroed,
    device_ptr,
    download,
    driver_plugins,
    first_item,
    item_stride,
    launch,
    launch_config,
    upload,
)

comptime LENGTH = 1000
comptime M = 6
comptime K = 5
comptime N = 7


# ===-----------------------------------------------------------------------===#
# Kernels
# ===-----------------------------------------------------------------------===#


def vec_add[
    dtype: DType
](
    a: ImmPointer[Scalar[dtype], ImmUntrackedOrigin],
    b: ImmPointer[Scalar[dtype], ImmUntrackedOrigin],
    output: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    n: Int32,
):
    for i in range(first_item(), Int(n), item_stride()):
        output[unsafe_offset=i] = a[unsafe_offset=i] + b[unsafe_offset=i]


def axpy[
    dtype: DType, n: Int
](
    a: Scalar[dtype],
    x: ImmPointer[Scalar[dtype], ImmUntrackedOrigin],
    y: ImmPointer[Scalar[dtype], ImmUntrackedOrigin],
    output: MutPointer[Scalar[dtype], MutUntrackedOrigin],
):
    for i in range(first_item(), n, item_stride()):
        output[unsafe_offset=i] = a * x[unsafe_offset=i] + y[unsafe_offset=i]


def add[
    dtype: DType
](
    a: Scalar[dtype],
    b: Scalar[dtype],
    output: MutPointer[Scalar[dtype], MutUntrackedOrigin],
):
    if first_item() == 0:
        output[] = a + b


def no_args():
    pass


def matmul[
    dtype: DType, LayoutA: TensorLayout, LayoutB: TensorLayout
](
    a: TileTensor[dtype, LayoutA, ImmUntrackedOrigin],
    b: TileTensor[dtype, LayoutB, ImmUntrackedOrigin],
    output: MutPointer[Scalar[dtype], MutUntrackedOrigin],
):
    """Writes the row-major product of `a` (`m x k`) and `b` (`k x n`)."""
    var k = Int(a.dim[1]())
    var n = Int(b.dim[1]())
    for i in range(first_item(), Int(a.dim[0]()) * n, item_stride()):
        var row = i // n
        var col = i % n
        var acc = Scalar[dtype](0)
        for step in range(k):
            acc += rebind[Scalar[dtype]](a[row, step]) * rebind[Scalar[dtype]](
                b[step, col]
            )
        output[unsafe_offset=i] = acc


# ===-----------------------------------------------------------------------===#
# Helpers
# ===-----------------------------------------------------------------------===#


def _expect[
    dtype: DType, spec: DeviceSpec
](
    queue: Queue[spec],
    buf: Buffer[spec],
    expected: List[Scalar[dtype]],
    what: String,
) raises:
    """Copies `buf` back once the queue's work is done and checks it holds
    `expected`."""
    var result = download[dtype](queue, buf, len(expected))
    for i in range(len(expected)):
        assert_equal(
            result[i], expected[i], String(what, " ", dtype, "[", i, "]")
        )


def _ramp[dtype: DType](length: Int, period: Int) -> List[Scalar[dtype]]:
    """Returns `i % period` for each `i`, small enough that no kernel below
    overflows any dtype."""
    var values = List[Scalar[dtype]](capacity=length)
    for i in range(length):
        values.append(Scalar[dtype](i % period))
    return values^


# ===-----------------------------------------------------------------------===#
# Tests
# ===-----------------------------------------------------------------------===#


def test_vec_add[
    dtype: DType, spec: DeviceSpec
](ctx: Context[spec], queue: Queue[spec]) raises:
    """Fills the pointers with `Buffer`s and a `BufferView`, and passes the
    length at run time as an `Int32`."""
    print("vec_add", dtype)
    var a_host = _ramp[dtype](LENGTH, 97)
    var b_host = _ramp[dtype](LENGTH, 89)
    var a = upload(ctx, queue, a_host)
    var b = upload(ctx, queue, b_host)
    var out = alloc_zeroed[dtype](ctx, queue, LENGTH)

    launch[vec_add[dtype]](queue, LENGTH, a, b.view(), out, Int32(LENGTH))

    var expected = List[Scalar[dtype]](capacity=LENGTH)
    for i in range(LENGTH):
        expected.append(a_host[i] + b_host[i])
    _expect(queue, out, expected, "vec_add")
    ctx.free_sync(out^)
    ctx.free_sync(b^)
    ctx.free_sync(a^)


def test_axpy[
    dtype: DType, spec: DeviceSpec
](ctx: Context[spec], queue: Queue[spec]) raises:
    """Passes a scalar inline and a raw device pointer."""
    print("axpy", dtype)
    comptime alpha = Scalar[dtype](3)
    var x_host = _ramp[dtype](LENGTH, 31)
    var y_host = _ramp[dtype](LENGTH, 17)
    var x = upload(ctx, queue, x_host)
    var y = upload(ctx, queue, y_host)
    var out = alloc_zeroed[dtype](ctx, queue, LENGTH)

    launch[axpy[dtype, LENGTH]](
        queue, LENGTH, alpha, device_ptr[dtype](ctx, x), y, out.view()
    )

    var expected = List[Scalar[dtype]](capacity=LENGTH)
    for i in range(LENGTH):
        expected.append(alpha * x_host[i] + y_host[i])
    _expect(queue, out, expected, "axpy")
    ctx.free_sync(out^)
    ctx.free_sync(y^)
    ctx.free_sync(x^)


def test_add[
    dtype: DType, spec: DeviceSpec
](ctx: Context[spec], queue: Queue[spec]) raises:
    """Passes only scalars and the output."""
    print("add", dtype)
    var out = alloc_zeroed[dtype](ctx, queue, 1)
    launch[add[dtype]](queue, 1, Scalar[dtype](7), Scalar[dtype](5), out)
    _expect(queue, out, [Scalar[dtype](12)], "add")
    ctx.free_sync(out^)


def _matmul_expected[
    dtype: DType
](a: List[Scalar[dtype]], b: List[Scalar[dtype]]) -> List[Scalar[dtype]]:
    var c = List[Scalar[dtype]](capacity=M * N)
    for row in range(M):
        for col in range(N):
            var acc = Scalar[dtype](0)
            for step in range(K):
                acc += a[row * K + step] * b[step * N + col]
            c.append(acc)
    return c^


def test_matmul[
    dtype: DType, spec: DeviceSpec
](ctx: Context[spec], queue: Queue[spec]) raises:
    """Passes two static-shape `TileTensor`s over device pointers."""
    print("matmul", dtype)
    var a_host = _ramp[dtype](M * K, 7)
    var b_host = _ramp[dtype](K * N, 5)
    var a = upload(ctx, queue, a_host)
    var b = upload(ctx, queue, b_host)
    var out = alloc_zeroed[dtype](ctx, queue, M * N)

    var a_tile = TileTensor(
        ptr=device_ptr[dtype](ctx, a), layout=row_major[M, K]()
    )
    var b_tile = TileTensor(
        ptr=device_ptr[dtype](ctx, b), layout=row_major[K, N]()
    )
    launch[matmul[dtype, a_tile.LayoutType, b_tile.LayoutType]](
        queue, M * N, a_tile, b_tile, out
    )

    _expect(queue, out, _matmul_expected(a_host, b_host), "matmul")
    ctx.free_sync(out^)
    ctx.free_sync(b^)
    ctx.free_sync(a^)


def test_matmul_dynamic_shape[
    spec: DeviceSpec
](ctx: Context[spec], queue: Queue[spec]) raises:
    """Passes a `TileTensor` whose shape is only known at run time, so its
    dims travel with it."""
    print("matmul, dynamic shape")
    comptime dtype = DType.float32
    var a_host = _ramp[dtype](M * K, 7)
    var b_host = _ramp[dtype](K * N, 5)
    var a = upload(ctx, queue, a_host)
    var b = upload(ctx, queue, b_host)
    var out = alloc_zeroed[dtype](ctx, queue, M * N)

    var m = M
    var k = K
    var a_tile = TileTensor(
        ptr=device_ptr[dtype](ctx, a), layout=row_major(m, k)
    )
    var b_tile = TileTensor(
        ptr=device_ptr[dtype](ctx, b), layout=row_major[K, N]()
    )
    launch[matmul[dtype, a_tile.LayoutType, b_tile.LayoutType]](
        queue, M * N, a_tile, b_tile, out
    )

    _expect(queue, out, _matmul_expected(a_host, b_host), "dynamic matmul")
    ctx.free_sync(out^)
    ctx.free_sync(b^)
    ctx.free_sync(a^)


def test_no_args[spec: DeviceSpec](queue: Queue[spec]) raises:
    """Launches a kernel that takes no arguments, so packs no slots."""
    print("no arguments")
    launch[no_args](queue, 1)
    queue.synchronize()


def test_axpy_on_stream[spec: DeviceSpec](ctx: Context[spec]) raises:
    """Orders the launch after the stream's earlier copies, and fills the
    pointers with offset `BufferView`s."""
    print("axpy on a stream")
    comptime dtype = DType.float32
    comptime half = LENGTH // 2
    comptime half_bytes = UInt64(half * size_of[Scalar[dtype]]())
    var stream = ctx.create_stream()
    var x = ctx.alloc_sync(2 * half_bytes)
    var out = ctx.alloc_sync(2 * half_bytes)
    var x_host = _ramp[dtype](LENGTH, 31)
    stream[].copy_to_device(
        x.view(), x_host.unsafe_ptr().unsafe_bitcast[UInt8]()
    )
    stream[].set_memory(out.view(), 0)

    # The upper half of `out` from the upper half of `x`.
    var x_upper = x.view(byte_offset=half_bytes, byte_size=half_bytes)
    var out_upper = out.view(byte_offset=half_bytes, byte_size=half_bytes)
    stream[].execute[axpy[dtype, half]](
        Float32(2),
        x_upper,
        x_upper,
        out_upper,
        execution_config=launch_config[spec](half),
    )

    var result = List[Scalar[dtype]](length=LENGTH, fill=0)
    stream[].copy_from_device(
        result.unsafe_ptr().unsafe_bitcast[UInt8](), out.view()
    )
    stream[].synchronize()
    for i in range(LENGTH):
        var expected = Float32(0) if i < half else 3 * x_host[i]
        assert_equal(result[i], expected, String("axpy on stream [", i, "]"))
    ctx.free_sync(out^)
    ctx.free_sync(x^)


def test_every_kernel[
    dtype: DType, spec: DeviceSpec
](ctx: Context[spec], queue: Queue[spec]) raises:
    test_vec_add[dtype](ctx, queue)
    test_axpy[dtype](ctx, queue)
    test_add[dtype](ctx, queue)
    test_matmul[dtype](ctx, queue)


def main() raises:
    var driver = Driver.create(driver_plugins())
    var device = driver[].get_device[0]()
    var ctx = device[].get_context()

    var queue = ctx[].create_queue()
    test_every_kernel[DType.float32](ctx[], queue[])
    test_every_kernel[DType.int32](ctx[], queue[])
    test_every_kernel[DType.int64](ctx[], queue[])
    test_every_kernel[DType.uint32](ctx[], queue[])
    test_every_kernel[DType.uint64](ctx[], queue[])
    test_matmul_dynamic_shape(ctx[], queue[])
    test_no_args(queue[])
    # A context may allow only one live queue, so this one goes before the
    # stream makes its own.
    _ = queue^
    test_axpy_on_stream(ctx[])
