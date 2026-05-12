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

"""Tests for `DevicePointer` offset arithmetic and bounds checking.

These tests exercise the host-side offset math on `DevicePointer` without
touching the device. They run against any `DeviceContext` backend, including
`api="cpu"`, so no GPU is required.
"""

from asyncrt_test_utils import create_test_device_context
from std.gpu.host import DeviceContext, DevicePointer
from std.testing import (
    TestSuite,
    assert_equal,
    assert_raises,
    assert_true,
    assert_false,
)


comptime _LENGTH = 64


def _make_pointer(ctx: DeviceContext) raises -> DevicePointer[DType.float32]:
    var buf = ctx.enqueue_create_buffer[DType.float32](_LENGTH)
    return buf.device_ptr()


# ===-----------------------------------------------------------------------===#
# Acquisition
# ===-----------------------------------------------------------------------===#


def test_device_ptr_starts_at_offset_zero() raises:
    var ctx = create_test_device_context()
    var p = _make_pointer(ctx)
    assert_equal(p.offset(), 0)


def test_device_ptr_accepts_length_minus_1() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[DType.float32](_LENGTH)
    var p = DevicePointer(buf, _LENGTH - 1)
    assert_equal(p.offset(), _LENGTH - 1)


def test_device_ptr_preserves_buffer_size() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[DType.float32](_LENGTH)
    var p = buf.device_ptr()
    assert_equal(len(p.buffer()), _LENGTH)


def test_init_zero_size_buffer_raises() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[DType.float32](0)
    with assert_raises(contains="size of DeviceBuffer must not be 0"):
        _ = DevicePointer(buf)


def test_init_zero_size_buffer_with_offset_raises() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[DType.float32](0)
    with assert_raises(contains="invalid DeviceBuffer of size '0'"):
        _ = DevicePointer(buf, 0)


def test_init_negative_offset_raises() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[DType.float32](_LENGTH)
    with assert_raises(contains="invalid offset '-1'"):
        _ = DevicePointer(buf, -1)


def test_init_offset_equal_to_size_raises() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[DType.float32](_LENGTH)
    with assert_raises(contains="invalid offset"):
        _ = DevicePointer(buf, _LENGTH)


def test_init_offset_past_end_raises() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[DType.float32](_LENGTH)
    with assert_raises(contains="invalid offset"):
        _ = DevicePointer(buf, _LENGTH + 1)


# ===-----------------------------------------------------------------------===#
# Forward arithmetic: __add__
# ===-----------------------------------------------------------------------===#


def test_add_advances_offset() raises:
    var ctx = create_test_device_context()
    var p = _make_pointer(ctx)
    var q = p + 8
    assert_equal(q.offset(), 8)
    # Original is unchanged.
    assert_equal(p.offset(), 0)


def test_add_zero_is_noop() raises:
    var ctx = create_test_device_context()
    var p = _make_pointer(ctx)
    var q = p + 0
    assert_equal(q.offset(), 0)


def test_add_chained() raises:
    var ctx = create_test_device_context()
    var p = _make_pointer(ctx)
    var q = (p + 4) + 4
    assert_equal(q.offset(), 8)


def test_add_negative_offsets_backward() raises:
    var ctx = create_test_device_context()
    var p = _make_pointer(ctx)
    # Move forward, then add a negative.
    var q = p + 10
    var r = q + -3
    assert_equal(r.offset(), 7)


def test_add_below_zero_raises() raises:
    var ctx = create_test_device_context()
    var p = _make_pointer(ctx)
    with assert_raises(contains="invalid offset"):
        _ = p + -1


def test_add_past_end_raises() raises:
    var ctx = create_test_device_context()
    var p = _make_pointer(ctx)
    with assert_raises(contains="invalid offset"):
        _ = p + _LENGTH


# ===-----------------------------------------------------------------------===#
# Backward arithmetic: __sub__
# ===-----------------------------------------------------------------------===#


def test_sub_decreases_offset() raises:
    var ctx = create_test_device_context()
    var p = _make_pointer(ctx)
    var q = p + 16
    var r = q - 4
    assert_equal(r.offset(), 12)


def test_sub_zero_is_noop() raises:
    var ctx = create_test_device_context()
    var p = _make_pointer(ctx)
    var q = (p + 8) - 0
    assert_equal(q.offset(), 8)


def test_sub_round_trip() raises:
    """`(p + n) - n` returns to the original offset."""
    var ctx = create_test_device_context()
    var p = _make_pointer(ctx)
    var q = (p + 16) - 16
    assert_equal(q.offset(), p.offset())


def test_sub_negative_advances_forward() raises:
    """`p - (-n)` is equivalent to `p + n`."""
    var ctx = create_test_device_context()
    var p = _make_pointer(ctx)
    var q = p - -8
    assert_equal(q.offset(), 8)


def test_sub_below_zero_raises() raises:
    var ctx = create_test_device_context()
    var p = _make_pointer(ctx)
    with assert_raises(contains="invalid offset"):
        _ = p - 1


def test_sub_negative_past_end_raises() raises:
    """`p - (-n)` is equivalent to `p + n`."""
    var ctx = create_test_device_context()
    var p = _make_pointer(ctx)
    with assert_raises(contains="invalid offset"):
        var q = p - -_LENGTH


# ===-----------------------------------------------------------------------===#
# In-place arithmetic: __iadd__ / __isub__
# ===-----------------------------------------------------------------------===#


def test_iadd_mutates_offset() raises:
    var ctx = create_test_device_context()
    var p = _make_pointer(ctx)
    p += 12
    assert_equal(p.offset(), 12)


def test_isub_mutates_offset() raises:
    var ctx = create_test_device_context()
    var p = _make_pointer(ctx)
    p += 20
    p -= 5
    assert_equal(p.offset(), 15)


def test_iadd_past_end_raises() raises:
    var ctx = create_test_device_context()
    var p = _make_pointer(ctx)
    with assert_raises(contains="invalid offset"):
        p += _LENGTH + 1
    # After a failed in-place op, offset should be unchanged.
    assert_equal(p.offset(), 0)


def test_isub_below_zero_raises() raises:
    var ctx = create_test_device_context()
    var p = _make_pointer(ctx)
    p += 4
    with assert_raises(contains="invalid offset"):
        p -= 5
    # After a failed in-place op, offset should be unchanged.
    assert_equal(p.offset(), 4)


# ===-----------------------------------------------------------------------===#
# Comparison ordering
# ===-----------------------------------------------------------------------===#


def test_equality_same_buffer_same_offset() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[DType.float32](_LENGTH)
    var p = buf.device_ptr()
    var q = buf.device_ptr()
    assert_true(p == q)
    assert_false(p != q)


def test_equality_same_buffer_different_offset() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[DType.float32](_LENGTH)
    var p = buf.device_ptr()
    var q = buf.device_ptr() + 4
    assert_false(p == q)
    assert_true(p != q)


def test_equality_different_buffer() raises:
    var ctx = create_test_device_context()
    var abuf = ctx.enqueue_create_buffer[DType.float32](_LENGTH)
    var a = abuf.device_ptr()
    var bbuf = ctx.enqueue_create_buffer[DType.float32](_LENGTH)
    var b = bbuf.device_ptr()
    assert_false(a == b)
    assert_true(a != b)


def test_ordering_same_buffer() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[DType.float32](_LENGTH)
    var p = buf.device_ptr()
    var q = buf.device_ptr() + 4
    assert_true(p < q)
    assert_true(p <= q)
    assert_true(q > p)
    assert_true(q >= p)


def test_lt_cross_buffer_raises() raises:
    var ctx = create_test_device_context()
    var buf_a = ctx.enqueue_create_buffer[DType.float32](_LENGTH)
    var buf_b = ctx.enqueue_create_buffer[DType.float32](_LENGTH)
    var p = buf_a.device_ptr()
    var q = buf_b.device_ptr()
    with assert_raises(contains="DeviceBuffer does not match"):
        _ = p < q
    with assert_raises(contains="DeviceBuffer does not match"):
        _ = p <= q
    with assert_raises(contains="DeviceBuffer does not match"):
        _ = q > p
    with assert_raises(contains="DeviceBuffer does not match"):
        _ = q >= p


# ===-----------------------------------------------------------------------===#
# Raw pointer access (on backends that expose it)
# ===-----------------------------------------------------------------------===#


def test_unsafe_ptr_matches_buffer_at_offset_zero() raises:
    """At offset 0, `unsafe_ptr()` should resolve to the same address as
    `DeviceBuffer.unsafe_ptr()`. The CPU backend exposes raw pointers."""
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[DType.float32](_LENGTH)
    var p = buf.device_ptr()
    assert_true(p.unsafe_ptr() == buf.unsafe_ptr())


def test_unsafe_ptr_advances_by_offset() raises:
    """`(p + n).unsafe_ptr()` must point `n` elements past `p.unsafe_ptr()`."""
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[DType.float32](_LENGTH)
    var p = buf.device_ptr()
    var q = p + 8
    assert_true(q.unsafe_ptr() == p.unsafe_ptr() + 8)


# ===-----------------------------------------------------------------------===#
# Writable
# ===-----------------------------------------------------------------------===#


def test_write_to() raises:
    var ctx = create_test_device_context()
    var buf = ctx.enqueue_create_buffer[DType.float32](_LENGTH)
    var p = DevicePointer(buf, 5)
    var expected = String(
        t"DevicePointer[{DType.float32}](buffer=DeviceBuffer(size={len(buf)}),"
        t" offset=5)"
    )
    assert_equal(String(p), expected)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
