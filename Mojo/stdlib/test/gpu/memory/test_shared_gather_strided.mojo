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
"""Regression tests for gather and strided accesses through shared pointers."""

from std.memory import unsafe_stack_allocation
from std.sys.intrinsics import strided_load, strided_store
from std.testing import TestSuite, assert_equal

from max.gpu.host import DeviceBuffer, DeviceContext

comptime _ALL = SIMD[.bool, 4](fill=True)
comptime _EVEN = SIMD[.bool, 4](True, False, True, False)


def _read_shared[
    gather: Bool, masked: Bool
](output: MutPointer[Float32, MutAnyOrigin],):
    var shared = unsafe_stack_allocation[
        8, DType.float32, address_space=.SHARED
    ]()
    for i in range(8):
        shared.unsafe_offset(i).unsafe_store(Float32(i))

    comptime mask = _EVEN if masked else _ALL
    var values: SIMD[.float32, 4]
    comptime if gather:
        values = shared.unsafe_gather(
            SIMD[.int32, 4](6, 4, 2, 0), mask, SIMD[.float32, 4](-1)
        )
    else:
        values = strided_load[4](shared.as_imm(), 2, mask)
    output.unsafe_store[width=4](0, values)


def _store_shared[masked: Bool](output: MutPointer[Float32, MutAnyOrigin]):
    var shared = unsafe_stack_allocation[
        8, DType.float32, address_space=.SHARED
    ]()
    for i in range(8):
        shared.unsafe_offset(i).unsafe_store(Float32(-1))

    comptime mask = _EVEN if masked else _ALL
    strided_store(SIMD[.float32, 4](10, 20, 30, 40), shared, 2, mask)
    for i in range(8):
        output.unsafe_offset(i).unsafe_store(
            shared.unsafe_offset(i).unsafe_load()
        )


def _check[
    width: SIMDLength
](
    ctx: DeviceContext,
    output: DeviceBuffer[DType.float32],
    expected: SIMD[.float32, width],
) raises:
    ctx.synchronize()
    with output.map_to_host() as host:
        for i in range(width):
            assert_equal(host[i], expected[i], msg=String("element ", i))


def test_shared_gather() raises:
    with DeviceContext() as ctx:
        var buffer = ctx.enqueue_create_buffer[.float32](4)
        ctx.enqueue_function[_read_shared[True, False]](
            buffer, grid_dim=1, block_dim=1
        )
        _check(ctx, buffer, SIMD[.float32, 4](6, 4, 2, 0))
        ctx.enqueue_function[_read_shared[True, True]](
            buffer, grid_dim=1, block_dim=1
        )
        _check(ctx, buffer, SIMD[.float32, 4](6, -1, 2, -1))


def test_shared_strided_load() raises:
    with DeviceContext() as ctx:
        var buffer = ctx.enqueue_create_buffer[.float32](4)
        ctx.enqueue_function[_read_shared[False, False]](
            buffer, grid_dim=1, block_dim=1
        )
        _check(ctx, buffer, SIMD[.float32, 4](0, 2, 4, 6))
        ctx.enqueue_function[_read_shared[False, True]](
            buffer, grid_dim=1, block_dim=1
        )
        _check(ctx, buffer, SIMD[.float32, 4](0, 0, 4, 0))


def test_shared_strided_store() raises:
    with DeviceContext() as ctx:
        var buffer = ctx.enqueue_create_buffer[.float32](8)
        ctx.enqueue_function[_store_shared[False]](
            buffer, grid_dim=1, block_dim=1
        )
        _check(ctx, buffer, SIMD[.float32, 8](10, -1, 20, -1, 30, -1, 40, -1))
        ctx.enqueue_function[_store_shared[True]](
            buffer, grid_dim=1, block_dim=1
        )
        _check(ctx, buffer, SIMD[.float32, 8](10, -1, -1, -1, 30, -1, -1, -1))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
