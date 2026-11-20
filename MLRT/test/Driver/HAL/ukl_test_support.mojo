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
"""Shared pieces of the typed `Queue.execute` kernel suites: the portable
work split kernels use, the launch config each device takes, and buffer
upload, download and checks."""

from std.builtin.rebind import downcast, rebind_var
from std.math import ceildiv
from std.os import getenv
from std.reflection import call_location
from std.sys import size_of
from std.sys.info import is_gpu
from std.testing import assert_almost_equal

from max._driver import Context
from max._driver.max_license.hal.buffer import Buffer
from max._driver.max_license.hal._machine import DeviceRef, DeviceSpec
from max._driver.max_license.hal.execution_config import CoresExecutionConfig
from max._driver.max_license.hal.queue import Queue
from max.gpu import block_dim, global_idx, grid_dim
from max.gpu.host import (
    ExecutionConfig,
    GridBlockExecutionConfig,
    GridBlockLaunchConfig,
)

comptime BLOCK = 256
# A cores config enables the first `CORES` cores, so the set has no gaps.
comptime CORES = 2


# ===-----------------------------------------------------------------------===#
# Device side
# ===-----------------------------------------------------------------------===#


@inline(.always)
def first_item() -> Int:
    """Returns the first work item this thread owns."""
    comptime if is_gpu():
        return global_idx.x
    else:
        return 0


# A device that isn't SIMT runs the kernel once on every enabled thread with no
# portable index to split the work by, so each thread covers every item. A
# kernel must then never read back what it stores: two threads running the
# same item would race on the partial value.
@inline(.always)
def item_stride() -> Int:
    """Returns the distance between this thread's work items."""
    comptime if is_gpu():
        return Int(grid_dim.x) * Int(block_dim.x)
    else:
        return 1


# ===-----------------------------------------------------------------------===#
# Host side
# ===-----------------------------------------------------------------------===#


comptime _Hooks[spec: DeviceSpec] = Queue[spec]._HALHooks
comptime _PluginConfig[spec: DeviceSpec]: ExecutionConfig = _Hooks[
    spec
].ExecutionConfigType[DeviceRef(spec, 0, private=())]

comptime LaunchConfig[spec: DeviceSpec]: ExecutionConfig = (
    _PluginConfig[spec] if (
        _Hooks[spec].Launcher.launches_config
        and not conforms_to(_PluginConfig[spec], GridBlockExecutionConfig)
    ) else GridBlockLaunchConfig
)
"""The plugin's own config when its launcher takes one that isn't grid/block
shaped, and a portable grid/block config otherwise."""


def launch_config[
    spec: DeviceSpec
](items: Int) -> downcast[
    LaunchConfig[spec], ExecutionConfig & Deinitable & Movable
]:
    """Returns the config that runs `items` work items on the device."""
    comptime Config = LaunchConfig[spec]
    comptime Result = downcast[Config, ExecutionConfig & Deinitable & Movable]
    comptime if conforms_to(Config, GridBlockExecutionConfig):
        return rebind_var[Result](
            GridBlockLaunchConfig(
                grid_dim=ceildiv(items, BLOCK), block_dim=BLOCK
            )
        )
    else:
        comptime assert conforms_to(
            Config, CoresExecutionConfig
        ), "the device takes neither grid/block nor cores"
        return rebind_var[Result](
            downcast[Config, CoresExecutionConfig & Deinitable](num_cores=CORES)
        )


@inline(.always)
def launch[
    spec: DeviceSpec,
    declared: TypeList[Trait=AnyType, ...],
    //,
    kernel: def(*args: *declared) thin -> None,
    *Ts: AnyType,
](queue: Queue[spec], items: Int, *args: *Ts) raises:
    """Launches `kernel` over `items` work items with the config the device
    takes. The queue's context compiles and loads `kernel` on first use and
    keeps it loaded; a failure names the caller's line."""
    queue.execute[kernel](
        *args,
        execution_config=launch_config[spec](items),
        location=call_location(),
    )


def driver_plugins() raises -> String:
    """Returns the plugin spec to create the test's `Driver` from.

    A toolchain that sets `MODULAR_DRIVER_PLUGINS` to a plugin of its own
    leaves the test's plugin in `MODULAR_HAL_TEST_DRIVER_PLUGINS`.
    """
    var plugins = getenv("MODULAR_HAL_TEST_DRIVER_PLUGINS")
    if not plugins:
        plugins = getenv("MODULAR_DRIVER_PLUGINS")
    if not plugins:
        raise Error("MODULAR_DRIVER_PLUGINS not set")
    return plugins^


def device_ptr[
    dtype: DType, spec: DeviceSpec, *, mut: Bool = False
](ctx: Context[spec], buf: Buffer[spec]) raises -> Pointer[
    Scalar[dtype], UntrackedOrigin[mut=mut]
]:
    """Returns `buf`'s device address as a kernel's pointer argument, an
    output when `mut` is true."""
    return Pointer[Scalar[dtype], UntrackedOrigin[mut=mut]](
        unsafe_from_address=Int(ctx.memory_get_address(buf))
    )


def upload[
    dtype: DType, spec: DeviceSpec
](
    ctx: Context[spec], queue: Queue[spec], values: List[Scalar[dtype]]
) raises -> Buffer[spec]:
    """Returns a new device buffer holding `values`."""
    var buf = ctx.alloc_sync(UInt64(len(values) * size_of[Scalar[dtype]]()))
    queue.copy_to_device(
        buf.view(), values.unsafe_ptr().unsafe_bitcast[UInt8]()
    )
    # The copy reads `values`, which the caller may drop once this returns.
    queue.synchronize()
    return buf^


def alloc_zeroed[
    dtype: DType, spec: DeviceSpec
](ctx: Context[spec], queue: Queue[spec], length: Int) raises -> Buffer[spec]:
    """Returns a new device buffer of `length` zeros."""
    var buf = ctx.alloc_sync(UInt64(length * size_of[Scalar[dtype]]()))
    queue.set_memory(buf.view(), 0)
    return buf^


def download[
    dtype: DType, spec: DeviceSpec
](queue: Queue[spec], buf: Buffer[spec], length: Int) raises -> List[
    Scalar[dtype]
]:
    """Copies the first `length` values of `buf` back once the queue's work is
    done."""
    var result = List[Scalar[dtype]](length=length, fill=0)
    queue.copy_from_device(
        result.unsafe_ptr().unsafe_bitcast[UInt8](),
        buf.view(
            byte_offset=0,
            byte_size=UInt64(length * size_of[Scalar[dtype]]()),
        ),
    )
    queue.synchronize()
    return result^


def expect_close[
    dtype: DType, spec: DeviceSpec
](
    queue: Queue[spec],
    buf: Buffer[spec],
    expected: List[Scalar[dtype]],
    what: String,
    *,
    atol: Float64,
    rtol: Float64,
) raises:
    """Checks `buf` holds `expected` to within `atol + rtol * |expected|`."""
    var result = download[dtype](queue, buf, len(expected))
    for i in range(len(expected)):
        assert_almost_equal(
            result[i],
            expected[i],
            String(what, "[", i, "]"),
            atol=atol,
            rtol=rtol,
        )
