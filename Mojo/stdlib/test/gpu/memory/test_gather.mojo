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
"""Tests `Pointer.unsafe_gather` on GPU in shared memory."""

from std.memory import unsafe_stack_allocation
from std.testing import TestSuite, assert_equal

from max.gpu.host import DeviceBuffer, DeviceContext

comptime _OFFSETS = SIMD[.int32, 4](3, 0, 2, 1)
comptime _EVEN_LANES = SIMD[.bool, 4](True, False, True, False)


def _gather_shared_kernel[
    masked: Bool
](output: MutPointer[Float32, MutAnyOrigin]):
    var shared = unsafe_stack_allocation[
        4, DType.float32, address_space=.SHARED
    ]()
    for i in range(4):
        shared.unsafe_offset(i).unsafe_store(Float32(i))

    var values: SIMD[.float32, 4]
    comptime if masked:
        values = shared.unsafe_gather(
            _OFFSETS, _EVEN_LANES, SIMD[.float32, 4](-1)
        )
    else:
        values = shared.unsafe_gather(_OFFSETS)
    output.unsafe_store[width=4](0, values)


def _check(
    ctx: DeviceContext,
    out_device: DeviceBuffer[DType.float32],
    expected: SIMD[DType.float32, 4],
) raises:
    ctx.synchronize()
    with out_device.map_to_host() as out_host:
        for i in range(4):
            assert_equal(out_host[i], expected[i], msg=String("lane ", i))


def test_gather_shared() raises:
    with DeviceContext() as ctx:
        var out_device = ctx.enqueue_create_buffer[.float32](4)
        ctx.enqueue_function[_gather_shared_kernel[masked=False]](
            out_device, grid_dim=1, block_dim=1
        )
        _check(ctx, out_device, SIMD[DType.float32, 4](3, 0, 2, 1))


def test_gather_shared_masked() raises:
    with DeviceContext() as ctx:
        var out_device = ctx.enqueue_create_buffer[.float32](4)
        ctx.enqueue_function[_gather_shared_kernel[masked=True]](
            out_device, grid_dim=1, block_dim=1
        )
        # Masked-off lanes retain the supplied default, -1.
        _check(ctx, out_device, SIMD[DType.float32, 4](3, -1, 2, -1))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
