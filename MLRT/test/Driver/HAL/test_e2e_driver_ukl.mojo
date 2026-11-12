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
"""Launches kernels through the typed `Queue.execute` and `Stream.execute`,
which pack HAL buffers, device pointers and scalars as the kernel declares
them."""

from std.os import getenv
from std.memory import ArcPointer
from std.sys import size_of
from std.testing import assert_equal

from max._driver import Driver, Context
from max._driver.max_license.hal.device import (
    Device,
    DeviceSpec,
    get_device_spec,
)
from max.gpu import global_idx
from max.gpu.host import GridBlockLaunchConfig

comptime LENGTH = 1024
comptime BLOCK = 256
comptime BYTES = UInt64(LENGTH * size_of[Float32]())


def vec_add(
    a: ImmPointer[Float32, ImmUntrackedOrigin],
    b: ImmPointer[Float32, ImmUntrackedOrigin],
    result: MutPointer[Float32, MutUntrackedOrigin],
    n: Int32,
):
    var tid = global_idx.x
    if tid >= Int(n):
        return
    result[unsafe_offset=tid] = a[unsafe_offset=tid] + b[unsafe_offset=tid]


def axpy(
    alpha: Float32,
    x: ImmPointer[Float32, ImmUntrackedOrigin],
    y: MutPointer[Float32, MutUntrackedOrigin],
    n: UInt32,
):
    var tid = global_idx.x
    if tid >= Int(n):
        return
    y[unsafe_offset=tid] = alpha * x[unsafe_offset=tid] + y[unsafe_offset=tid]


def _config(n: Int) -> GridBlockLaunchConfig:
    return GridBlockLaunchConfig(
        grid_dim=(n + BLOCK - 1) // BLOCK, block_dim=BLOCK
    )


def test_vec_add_on_queue[spec: DeviceSpec](mut ctx: Context[spec]) raises:
    """Passes a `Buffer`, a `BufferView` and an `Int32`."""
    # Unloading the bundle unloads `func`, so it stays alive to the end.
    var compiled = ctx.compile[type_of(vec_add), vec_add]()
    var func = ctx.load_function(compiled[0], compiled[1].function_name)
    var queue = ctx.create_queue()

    var a = ctx.alloc_sync(BYTES)
    var b = ctx.alloc_sync(BYTES)
    var out = ctx.alloc_sync(BYTES)
    var host_a = List[Float32](capacity=LENGTH)
    var host_b = List[Float32](capacity=LENGTH)
    for i in range(LENGTH):
        host_a.append(Float32(i))
        host_b.append(Float32(2 * i))
    queue[].copy_to_device(
        a.view(), host_a.unsafe_ptr().unsafe_bitcast[UInt8]()
    )
    queue[].copy_to_device(
        b.view(), host_b.unsafe_ptr().unsafe_bitcast[UInt8]()
    )

    queue[].execute[vec_add](
        a,
        b.view(),
        out,
        Int32(LENGTH),
        function=func,
        execution_config=_config(LENGTH),
    )

    var result = List[Float32](length=LENGTH, fill=0)
    queue[].copy_from_device(
        result.unsafe_ptr().unsafe_bitcast[UInt8](), out.view()
    )
    queue[].synchronize()
    for i in range(LENGTH):
        assert_equal(result[i], Float32(3 * i))

    ctx.unload_function(func)
    _ = compiled^
    ctx.free_sync(out^)
    ctx.free_sync(b^)
    ctx.free_sync(a^)


def test_axpy_on_stream[spec: DeviceSpec](mut ctx: Context[spec]) raises:
    """Passes a scalar inline, a raw device pointer and an offset
    `BufferView`, and orders the launch after the stream's earlier copies."""
    # Unloading the bundle unloads `func`, so it stays alive to the end.
    var compiled = ctx.compile[type_of(axpy), axpy]()
    var func = ctx.load_function(compiled[0], compiled[1].function_name)
    var stream = ctx.create_stream()

    comptime half = LENGTH // 2
    var x = ctx.alloc_sync(BYTES)
    var y = ctx.alloc_sync(BYTES)
    var host_x = List[Float32](capacity=LENGTH)
    var host_y = List[Float32](capacity=LENGTH)
    for i in range(LENGTH):
        host_x.append(Float32(i))
        host_y.append(Float32(1))
    stream[].copy_to_device(
        x.view(), host_x.unsafe_ptr().unsafe_bitcast[UInt8]()
    )
    stream[].copy_to_device(
        y.view(), host_y.unsafe_ptr().unsafe_bitcast[UInt8]()
    )

    # The upper half of `y` from the lower half of `x`.
    var x_ptr = ImmPointer[Float32, ImmUntrackedOrigin](
        unsafe_from_address=Int(ctx.memory_get_address(x))
    )
    var y_upper = y.view(
        byte_offset=UInt64(half * size_of[Float32]()),
        byte_size=UInt64(half * size_of[Float32]()),
    )
    stream[].execute[axpy](
        Float32(2),
        x_ptr,
        y_upper,
        UInt32(half),
        function=func,
        execution_config=_config(half),
    )

    var result = List[Float32](length=LENGTH, fill=0)
    stream[].copy_from_device(
        result.unsafe_ptr().unsafe_bitcast[UInt8](), y.view()
    )
    stream[].synchronize()
    for i in range(LENGTH):
        var expected = Float32(1) if i < half else Float32(2 * (i - half) + 1)
        assert_equal(result[i], expected)

    ctx.unload_function(func)
    _ = compiled^
    ctx.free_sync(y^)
    ctx.free_sync(x^)


def main() raises:
    var plugin_spec = getenv("MODULAR_DRIVER_PLUGINS")
    if not plugin_spec:
        raise Error("MODULAR_DRIVER_PLUGINS not set")

    var driver = Driver.create(plugin_spec)
    var device: ArcPointer[Device[get_device_spec[0]()]] = driver[].get_device[
        0
    ]()
    var ctx = device[].get_context()

    test_vec_add_on_queue(ctx[])
    test_axpy_on_stream(ctx[])
