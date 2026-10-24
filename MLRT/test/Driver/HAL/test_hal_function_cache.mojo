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
"""Tests the HAL `Context` loaded-function cache."""

from std.os import getenv
from max._driver import Context, Driver
from max._driver.max_license.hal.device import (
    Device,
    DeviceSpec,
    get_device_spec,
)
from max.gpu import global_idx
from std.memory import (
    ArcPointer,
    OpaquePointer,
    Pointer,
    unsafe_stack_allocation,
)
from std.sys import size_of
from std.testing import assert_equal, assert_not_equal


def scale_kernel(data: Pointer[Float32, MutAnyOrigin], n: Int):
    var tid = global_idx.x
    if tid < n:
        data[unsafe_offset=tid] = data[unsafe_offset=tid] * 2.0


def test_content_dedupes_across_buffers[
    device_spec: DeviceSpec,
](mut ctx: Context[device_spec]) raises:
    """Identical code in two distinct buffers loads once."""
    var info = ctx._compile_inner[type_of(scale_kernel), scale_kernel]()
    var name = String(info.function_name)

    # Append rather than `String(info.asm)`, which only views the static buffer;
    # two views would share an address and an address key would "dedupe" them,
    # so the assert below guards against a vacuous pass.
    var copy_a = String()
    copy_a += info.asm
    var copy_b = String()
    copy_b += info.asm
    assert_not_equal(Int(copy_a.unsafe_ptr()), Int(copy_b.unsafe_ptr()))

    var func_a = ctx.get_or_load_function_dynamic(
        StringSlice(copy_a), name.copy()
    )
    var func_b = ctx.get_or_load_function_dynamic(
        StringSlice(copy_b), name.copy()
    )
    assert_equal(Int(func_a), Int(func_b))


def test_static_and_dynamic_agree[
    device_spec: DeviceSpec,
](mut ctx: Context[device_spec]) raises:
    """The address fast path resolves to the same entry as a content lookup."""
    var info = ctx._compile_inner[type_of(scale_kernel), scale_kernel]()
    var name = String(info.function_name)

    var first = ctx.get_or_load_function(info.asm, name.copy())
    var second = ctx.get_or_load_function(info.asm, name.copy())
    assert_equal(Int(first), Int(second))

    var copied = String()
    copied += info.asm
    var dynamic = ctx.get_or_load_function_dynamic(
        StringSlice(copied), name.copy()
    )
    assert_equal(Int(first), Int(dynamic))


def test_compile_function_inline_uses_cache[
    device_spec: DeviceSpec,
](mut ctx: Context[device_spec]) raises:
    """`compile_function_inline` resolves to its function's cached entry."""
    var first = ctx.compile_function_inline[scale_kernel]()
    var second = ctx.compile_function_inline[scale_kernel]()
    assert_equal(Int(first), Int(second))

    var info = ctx._compile_inner[type_of(scale_kernel), scale_kernel]()
    var direct = ctx.get_or_load_function(info.asm, String(info.function_name))
    assert_equal(Int(first), Int(direct))


def test_compile_function_inline_launches[
    device_spec: DeviceSpec,
](mut ctx: Context[device_spec]) raises:
    """The returned handle launches the kernel it was compiled from."""
    var func = ctx.compile_function_inline[scale_kernel]()

    comptime n = 4
    var host = SIMD[DType.float32, n](1.0, 2.0, 3.0, 4.0)
    var buf = ctx.alloc_sync(UInt64(size_of[type_of(host)]()))
    ctx.copy_to_device(buf.view(), Pointer(to=host).unsafe_bitcast[UInt8]())

    var data_addr = ctx.memory_get_address(buf)
    var n_val = Int(n)
    var args = unsafe_stack_allocation[2, OpaquePointer[MutUntrackedOrigin]]()
    var arg_sizes = unsafe_stack_allocation[2, UInt64]()
    args[unsafe_offset=0] = rebind[OpaquePointer[MutUntrackedOrigin]](
        Pointer(to=data_addr)
    )
    args[unsafe_offset=1] = rebind[OpaquePointer[MutUntrackedOrigin]](
        Pointer(to=n_val)
    )
    arg_sizes[unsafe_offset=0] = UInt64(size_of[UInt64]())
    arg_sizes[unsafe_offset=1] = UInt64(size_of[Int]())

    var queue = ctx.create_queue()
    queue[].execute(
        func,
        grid=(UInt32(1), UInt32(1), UInt32(1)),
        block=(UInt32(n), UInt32(1), UInt32(1)),
        args=rebind[Pointer[OpaquePointer[MutUntrackedOrigin], MutAnyOrigin]](
            args
        ),
        arg_sizes=rebind[Pointer[UInt64, MutAnyOrigin]](arg_sizes),
        num_args=UInt32(2),
    )
    queue[].synchronize()

    var result = SIMD[DType.float32, n](0.0)
    ctx.copy_from_device(Pointer(to=result).unsafe_bitcast[UInt8](), buf.view())
    assert_equal(result, SIMD[DType.float32, n](2.0, 4.0, 6.0, 8.0))
    ctx.free_sync(buf^)


def test_cache_key_distinguishes[device_spec: DeviceSpec]() raises:
    """Every component of the key participates, so nothing aliases on a partial
    match."""
    comptime Ctx = Context[device_spec]
    var base = Ctx._function_cache_key(1, 10, "kernel", None, 0)

    assert_not_equal(base, Ctx._function_cache_key(2, 10, "kernel", None, 0))
    assert_not_equal(base, Ctx._function_cache_key(1, 11, "kernel", None, 0))
    assert_not_equal(base, Ctx._function_cache_key(1, 10, "other", None, 0))
    assert_not_equal(
        base, Ctx._function_cache_key(1, 10, "kernel", Int32(8), 0)
    )
    assert_not_equal(
        Ctx._function_cache_key(1, 10, "kernel", Int32(8), 0),
        Ctx._function_cache_key(1, 10, "kernel", Int32(8), 1),
    )
    assert_equal(base, Ctx._function_cache_key(1, 10, "kernel", None, 0))


def main() raises:
    var plugin_spec = getenv("MODULAR_DRIVER_PLUGINS")
    if not plugin_spec:
        raise Error("MODULAR_DRIVER_PLUGINS not set")

    comptime device_spec = get_device_spec[0]()
    test_cache_key_distinguishes[device_spec]()

    var driver = Driver.create(plugin_spec)
    var device: ArcPointer[Device[device_spec]] = driver[].get_device[0]()
    var ctx: ArcPointer[Context[device_spec]] = device[].get_context()

    test_content_dedupes_across_buffers(ctx[])
    test_static_and_dynamic_agree(ctx[])
    test_compile_function_inline_uses_cache(ctx[])
    test_compile_function_inline_launches(ctx[])
    print("Function cache tests OK")
