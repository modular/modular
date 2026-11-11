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
"""Tests strided intrinsics on GPU in shared memory."""

from std.memory import unsafe_stack_allocation
from std.sys.intrinsics import strided_load, strided_store
from std.testing import TestSuite, assert_equal

from max.gpu.host import DeviceBuffer, DeviceContext

comptime _EVEN_LANES = SIMD[.bool, 4](True, False, True, False)


def _strided_load_shared_kernel[
    masked: Bool
](output: MutPointer[Float32, MutAnyOrigin]):
    var shared = unsafe_stack_allocation[
        16, DType.float32, address_space=.SHARED
    ]()
    for i in range(16):
        shared.unsafe_offset(i).unsafe_store(Float32(i))

    var values: SIMD[.float32, 4]
    comptime if masked:
        values = strided_load[4](shared.as_imm(), 4, _EVEN_LANES)
    else:
        values = strided_load[4](shared.as_imm(), 4)
    output.unsafe_store[width=4](0, values)


def _strided_store_shared_kernel[
    masked: Bool
](output: MutPointer[Float32, MutAnyOrigin]):
    var shared = unsafe_stack_allocation[
        8, DType.float32, address_space=.SHARED
    ]()
    for i in range(8):
        shared.unsafe_offset(i).unsafe_store(Float32(0))

    var values = SIMD[.float32, 4](99, 12, 23, 56)
    comptime if masked:
        strided_store(values, shared, 2, _EVEN_LANES)
    else:
        strided_store(values, shared, 2)
    for i in range(8):
        output.unsafe_offset(i).unsafe_store(
            shared.unsafe_offset(i).unsafe_load()
        )


def _check[
    width: SIMDLength
](
    ctx: DeviceContext,
    out_device: DeviceBuffer[DType.float32],
    expected: SIMD[DType.float32, width],
) raises:
    ctx.synchronize()
    with out_device.map_to_host() as out_host:
        for i in range(width):
            assert_equal(out_host[i], expected[i], msg=String("element ", i))


def test_strided_load_shared() raises:
    with DeviceContext() as ctx:
        var out_device = ctx.enqueue_create_buffer[.float32](4)
        ctx.enqueue_function[_strided_load_shared_kernel[masked=False]](
            out_device, grid_dim=1, block_dim=1
        )
        _check(ctx, out_device, SIMD[DType.float32, 4](0, 4, 8, 12))


def test_strided_load_shared_masked() raises:
    with DeviceContext() as ctx:
        var out_device = ctx.enqueue_create_buffer[.float32](4)
        ctx.enqueue_function[_strided_load_shared_kernel[masked=True]](
            out_device, grid_dim=1, block_dim=1
        )
        # Strided loads return zero in masked-off lanes.
        _check(ctx, out_device, SIMD[DType.float32, 4](0, 0, 8, 0))


def test_strided_store_shared() raises:
    with DeviceContext() as ctx:
        var out_device = ctx.enqueue_create_buffer[.float32](8)
        ctx.enqueue_function[_strided_store_shared_kernel[masked=False]](
            out_device, grid_dim=1, block_dim=1
        )
        _check(
            ctx, out_device, SIMD[DType.float32, 8](99, 0, 12, 0, 23, 0, 56, 0)
        )


def test_strided_store_shared_masked() raises:
    with DeviceContext() as ctx:
        var out_device = ctx.enqueue_create_buffer[.float32](8)
        ctx.enqueue_function[_strided_store_shared_kernel[masked=True]](
            out_device, grid_dim=1, block_dim=1
        )
        # Only lanes 0 and 2 write; all other positions stay zero.
        _check(
            ctx, out_device, SIMD[DType.float32, 8](99, 0, 0, 0, 23, 0, 0, 0)
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
