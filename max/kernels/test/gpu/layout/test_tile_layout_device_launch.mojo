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
"""Launches kernels that take a bare `Layout` as a positional argument.

Each kernel writes the decoded shape, the decoded strides, and a trailing
argument back to device memory. The trailing argument catches a launch whose
argument count or order is wrong, which a multi-dynamic-leaf `Coord` passed
bare causes (MOCO-4307).
"""

from layout import Idx
from layout.tile_layout import Layout, TensorLayout, row_major
from max.gpu.host import DeviceContext
from std.testing import assert_equal, TestSuite

comptime _TAG = 0x5A5A_1234


def _read_back_kernel[
    L: TensorLayout
](layout: L, tag: Int64, dst: Pointer[Int64, MutAnyOrigin]):
    comptime for i in range(L.rank):
        dst[unsafe_offset=i] = Int64(layout.shape[i]().value())
        dst[unsafe_offset=L.rank + i] = Int64(layout.stride[i]().value())
    dst[unsafe_offset=2 * L.rank] = tag


def _check_launch(layout: Some[TensorLayout], expected: List[Int]) raises:
    comptime L = type_of(layout)
    comptime count = 2 * L.rank + 1
    assert_equal(len(expected), count - 1)
    with DeviceContext() as ctx:
        var out_dev = ctx.enqueue_create_buffer[.int64](count)
        out_dev.enqueue_fill(-1)
        ctx.enqueue_function[_read_back_kernel[L]](
            layout,
            Int64(_TAG),
            out_dev.unsafe_ptr(),
            grid_dim=1,
            block_dim=1,
        )
        var out_host = ctx.enqueue_create_host_buffer[.int64](count)
        ctx.enqueue_copy(out_host, out_dev)
        ctx.synchronize()
        for i in range(count - 1):
            assert_equal(Int(out_host[i]), expected[i])
        assert_equal(Int(out_host[count - 1]), _TAG)
        _ = out_dev^


def test_static_layout() raises:
    _check_launch(row_major[3, 5](), [3, 5, 5, 1])


def test_dynamic_shape_and_stride() raises:
    var rows = 3
    var cols = 5
    _check_launch(row_major(rows, cols), [3, 5, 5, 1])


def test_dynamic_shape_static_stride() raises:
    # Only the shape occupies bytes, so the layout's one non-empty field holds
    # two dynamic leaves.
    var rows = 3
    var cols = 5
    _check_launch(
        Layout(shape=(rows, cols), stride=(Idx[7], Idx[1])), [3, 5, 7, 1]
    )


def test_three_dynamic_dims() raises:
    var d0 = 2
    var d1 = 3
    var d2 = 5
    _check_launch(row_major(d0, d1, d2), [2, 3, 5, 15, 5, 1])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
