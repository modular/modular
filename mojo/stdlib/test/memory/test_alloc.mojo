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

from std.memory.alloc import alloc, free, Layout
from std.sys import align_of, size_of

from test_utils import ObservableDel, check_write_to
from std.testing import (
    assert_equal,
    assert_false,
    assert_true,
    TestSuite,
)


def test_layout_with_default_alignment_uses_type_alignment() raises:
    var layout = Layout[Int32](count=4)
    assert_equal(layout.count(), 4)
    assert_equal(layout.alignment(), align_of[Int32]())


def test_layout_with_runtime_alignment() raises:
    var layout = Layout[Int32](count=4, alignment=64)
    assert_equal(layout.count(), 4)
    assert_equal(layout.alignment(), 64)


def test_layout_aligned_uses_comptime_alignment() raises:
    var layout = Layout[Int32].aligned[128](count=4)
    assert_equal(layout.count(), 4)
    assert_equal(layout.alignment(), 128)


def test_layout_single_has_count_one_and_default_alignment() raises:
    var layout = Layout[Int64].single()
    assert_equal(layout.count(), 1)
    assert_equal(layout.alignment(), align_of[Int64]())


def test_layout_as_byte_layout_scales_count_and_preserves_alignment() raises:
    var layout = Layout[Int32](count=4, alignment=64)
    var byte_layout = layout.as_byte_layout()
    assert_equal(byte_layout.count(), 4 * size_of[Int32]())
    assert_equal(byte_layout.alignment(), 64)


def test_layout_write_to_and_repr() raises:
    var layout = Layout[Int](count=8, alignment=64)
    check_write_to(
        layout, expected="Layout[Int](count=8, alignment=64)", is_repr=False
    )
    check_write_to(
        layout, expected="Layout[Int](count=8, alignment=64)", is_repr=True
    )


def test_alloc_and_free_round_trip_reads_and_writes_values() raises:
    var layout = Layout[Int](count=5)
    var ptr = alloc(layout)
    for i in range(5):
        (ptr + i).init_pointee_move(i)
    for i in range(5):
        assert_equal(ptr[i], i)
    free(ptr, layout)


def test_alloc_returns_pointer_meeting_layout_alignment() raises:
    var layout = Layout[UInt8](count=1, alignment=128)
    var ptr = alloc(layout)
    assert_equal(Int(ptr) % 128, 0)
    free(ptr, layout)


def test_alloc_with_layout_single_supports_one_element() raises:
    var layout = Layout[Int64].single()
    var ptr = alloc(layout)
    ptr.init_pointee_move(42)
    assert_equal(ptr[], 42)
    free(ptr, layout)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
