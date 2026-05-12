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

"""Smoke tests for kernel launches that bind pointer arguments via
`DevicePointer`.

Two cases are exercised:

1. Every pointer arg bound from a `DevicePointer` (the new path under
   development).
2. A mix of `DeviceBuffer` and `DevicePointer` args in a single launch.

Both must produce identical results, which guards against regressions in the
existing `DeviceBuffer` arg-translation path while `DevicePointer` is being
developed.
"""

from asyncrt_test_utils import create_test_device_context
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.testing import TestSuite, assert_equal


def vec_add(
    in0: UnsafePointer[Float32, MutAnyOrigin],
    in1: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    len: Int,
):
    var tid = global_idx.x
    if tid >= len:
        return
    output[tid] = in0[tid] + in1[tid]


comptime _LENGTH = 1024
comptime _BLOCK_DIM = 32


def test_kernel_with_device_pointers() raises:
    """Bind every pointer arg from a `DevicePointer`."""
    var ctx = create_test_device_context()

    var in0 = ctx.enqueue_create_buffer[DType.float32](_LENGTH)
    var in1 = ctx.enqueue_create_buffer[DType.float32](_LENGTH)
    var out = ctx.enqueue_create_buffer[DType.float32](_LENGTH)

    with in0.map_to_host() as in0_host, in1.map_to_host() as in1_host:
        for i in range(_LENGTH):
            in0_host[i] = Float32(i)
            in1_host[i] = Float32(2 * i + 1)

    var kernel = ctx.compile_function[vec_add]()
    ctx.enqueue_function(
        kernel,
        in0.device_ptr(),
        in1.device_ptr(),
        out.device_ptr(),
        _LENGTH,
        grid_dim=(_LENGTH // _BLOCK_DIM),
        block_dim=_BLOCK_DIM,
    )

    with out.map_to_host() as out_host:
        for i in range(_LENGTH):
            assert_equal(
                out_host[i],
                Float32(i) + Float32(2 * i + 1),
                String("at index ", i, " the value is ", out_host[i]),
            )


def test_kernel_mixed_buffer_and_device_pointer() raises:
    """Mix `DeviceBuffer` and `DevicePointer` args in a single launch."""
    var ctx = create_test_device_context()

    var in0 = ctx.enqueue_create_buffer[DType.float32](_LENGTH)
    var in1 = ctx.enqueue_create_buffer[DType.float32](_LENGTH)
    var out = ctx.enqueue_create_buffer[DType.float32](_LENGTH)

    with in0.map_to_host() as in0_host, in1.map_to_host() as in1_host:
        for i in range(_LENGTH):
            in0_host[i] = Float32(i)
            in1_host[i] = Float32(2 * i + 1)

    var kernel = ctx.compile_function[vec_add]()
    ctx.enqueue_function(
        kernel,
        in0,  # DeviceBuffer
        in1.device_ptr(),  # DevicePointer
        out,  # DeviceBuffer
        _LENGTH,
        grid_dim=(_LENGTH // _BLOCK_DIM),
        block_dim=_BLOCK_DIM,
    )

    with out.map_to_host() as out_host:
        for i in range(_LENGTH):
            assert_equal(
                out_host[i],
                Float32(i) + Float32(2 * i + 1),
                String("at index ", i, " the value is ", out_host[i]),
            )


def main() raises:
    # TODO(MOCO-2556): Use automatic discovery when it can handle global_idx.
    # TestSuite.discover_tests[__functions_in_module()]().run()
    var suite = TestSuite()

    suite.test[test_kernel_with_device_pointers]()
    suite.test[test_kernel_mixed_buffer_and_device_pointer]()

    suite^.run()
