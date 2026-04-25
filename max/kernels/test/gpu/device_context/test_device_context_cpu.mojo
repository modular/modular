# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

from std.gpu.host import DeviceContext
from std.memory.unsafe_pointer import alloc
from std.testing import assert_equal, assert_true


def test_empty_func(ctx: DeviceContext) raises:
    @parameter
    def empty() -> None:
        pass

    ctx.enqueue_cpu_function[empty]()
    ctx.synchronize()


def test_func_writes_to_memory(ctx: DeviceContext) raises:
    var ptr = alloc[Int](1)
    ptr[] = 0

    @parameter
    def write_42() -> None:
        ptr[] = 42

    ctx.enqueue_cpu_function[write_42]()
    ctx.synchronize()
    assert_equal(ptr[], 42)
    ptr.free()


def test_func_closure_writes_to_memory(ctx: DeviceContext) raises:
    var ptr = alloc[Int](1)
    ptr[] = 0
    var expected = 42

    def write_val() {read}:
        ptr[] = expected

    ctx.enqueue_cpu_function(write_val)
    ctx.synchronize()
    assert_equal(ptr[], 42)
    ptr.free()


def test_multiple_funcs_execute_in_order(ctx: DeviceContext) raises:
    var ptr = alloc[Int](1)
    ptr[] = 0

    @parameter
    def write_1() -> None:
        ptr[] = 1

    @parameter
    def write_2() -> None:
        ptr[] = 2

    @parameter
    def write_3() -> None:
        ptr[] = 3

    ctx.enqueue_cpu_function[write_1]()
    ctx.enqueue_cpu_function[write_2]()
    ctx.enqueue_cpu_function[write_3]()
    ctx.synchronize()
    # Stream semantics: functions execute in order, last write wins.
    assert_equal(ptr[], 3)
    ptr.free()


def test_func_accumulates(ctx: DeviceContext) raises:
    var ptr = alloc[Int](1)
    ptr[] = 0

    @parameter
    def increment() -> None:
        ptr[] += 1

    ctx.enqueue_cpu_function[increment]()
    ctx.enqueue_cpu_function[increment]()
    ctx.enqueue_cpu_function[increment]()
    ctx.synchronize()
    assert_equal(ptr[], 3)
    ptr.free()


def test_range_writes_indices(ctx: DeviceContext) raises:
    comptime count = 16
    var ptr = alloc[Int](count)
    for i in range(count):
        ptr[i] = -1

    @parameter
    def write_index(i: Int) -> None:
        ptr[i] = i

    ctx.enqueue_cpu_range[write_index](count=count)
    ctx.synchronize()
    for i in range(count):
        assert_equal(ptr[i], i)
    ptr.free()


def test_range_large(ctx: DeviceContext) raises:
    comptime count = 1024
    var ptr = alloc[Int](count)
    for i in range(count):
        ptr[i] = 0

    @parameter
    def write_squared(i: Int) -> None:
        ptr[i] = i * i

    ctx.enqueue_cpu_range[write_squared](count=count)
    ctx.synchronize()
    for i in range(count):
        assert_equal(ptr[i], i * i)
    ptr.free()


def test_func_then_range(ctx: DeviceContext) raises:
    comptime count = 8
    var ptr = alloc[Int](count)
    for i in range(count):
        ptr[i] = 0

    @parameter
    def set_all_to_one() -> None:
        for j in range(count):
            ptr[j] = 1

    @parameter
    def add_index(i: Int) -> None:
        ptr[i] += i

    # First fill with 1s, then add the index.
    ctx.enqueue_cpu_function[set_all_to_one]()
    ctx.enqueue_cpu_range[add_index](count=count)
    ctx.synchronize()
    for i in range(count):
        assert_equal(ptr[i], 1 + i)
    ptr.free()


def test_two_ranges_sequential(ctx: DeviceContext) raises:
    comptime count = 8
    var ptr = alloc[Int](count)
    for i in range(count):
        ptr[i] = 0

    @parameter
    def write_index(i: Int) -> None:
        ptr[i] = i

    @parameter
    def double_value(i: Int) -> None:
        ptr[i] *= 2

    ctx.enqueue_cpu_range[write_index](count=count)
    ctx.enqueue_cpu_range[double_value](count=count)
    ctx.synchronize()
    for i in range(count):
        assert_equal(ptr[i], i * 2)
    ptr.free()


def main() raises:
    with DeviceContext(api="cpu") as ctx:
        test_empty_func(ctx)
        test_func_writes_to_memory(ctx)
        test_func_closure_writes_to_memory(ctx)
        test_multiple_funcs_execute_in_order(ctx)
        test_func_accumulates(ctx)
        test_range_writes_indices(ctx)
        test_range_large(ctx)
        test_func_then_range(ctx)
        test_two_ranges_sequential(ctx)
