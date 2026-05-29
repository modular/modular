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
# Tests that the poison check does NOT produce false positives for legitimate
# values, including NaN/Inf bit patterns, near-poison integers, and masked-off
# lanes.

from std.memory import UnsafePointer, alloc
from std.sys.intrinsics import masked_load
from std.testing import assert_true


def test_normal_float32() raises:
    """Loading a properly initialized Float32 should not trigger abort."""
    var value = Float32(42.0)
    var val = UnsafePointer(to=value).load()
    assert_true(val == 42.0)


def test_qnan_not_flagged():
    """Canonical qNaN (0x7FC00000) is not the poison pattern; loading must
    not trigger abort. The poison pattern is intentionally non-NaN so the
    uninit-read check coexists with the nan-check pass."""
    var value = UInt32(0x7FC00000)
    var ptr = UnsafePointer(to=value).bitcast[Float32]()
    _ = ptr.load()


def test_snan_not_flagged():
    """A signaling NaN (0x7F800001) is not the poison pattern."""
    var value = UInt32(0x7F800001)
    var ptr = UnsafePointer(to=value).bitcast[Float32]()
    _ = ptr.load()


def test_inf_not_flagged():
    """Positive infinity (0x7F800000) is not the poison pattern."""
    var value = UInt32(0x7F800000)
    var ptr = UnsafePointer(to=value).bitcast[Float32]()
    _ = ptr.load()


def test_integer_not_checked() raises:
    """Integer types should not be checked for poison patterns."""
    var value = UInt32(0xFFFFFFFF)
    var val = UnsafePointer(to=value).load()
    assert_true(val == 0xFFFFFFFF)


def test_near_poison_float():
    """A float value close to but not equal to poison (FLT_MAX - 1 ulp =
    0x7F7FFFFE) should not trigger abort."""
    var value = UInt32(0x7F7FFFFE)
    var ptr = UnsafePointer(to=value).bitcast[Float32]()
    _ = ptr.load()


def test_masked_load_poison_in_masked_off_lane() raises:
    """Poison in a masked-off lane (passthrough) should not trigger abort."""
    # masked_load requires a contiguous buffer, so alloc is needed here.
    var ptr = alloc[Float32](4)

    ptr.store(0, Float32(1.0))
    ptr.store(1, Float32(2.0))
    ptr.store(2, Float32(3.0))
    ptr.store(3, Float32(4.0))

    # Poison lanes 2 and 3 with the debug allocator poison pattern
    # (FLT_MAX bits = 0x7F7FFFFF). Masked-off lanes must not trigger.
    (ptr + 2).bitcast[UInt32]().store(UInt32(0x7F7FFFFF))
    (ptr + 3).bitcast[UInt32]().store(UInt32(0x7F7FFFFF))

    # mask=False for lanes 2,3 means those lanes use passthrough, not memory.
    var mask = SIMD[DType.bool, 4](True, True, False, False)
    var passthrough = SIMD[DType.float32, 4](0)
    var val = masked_load(ptr, mask, passthrough)

    assert_true(val[0] == 1.0)
    assert_true(val[1] == 2.0)

    ptr.free()


def test_normal_float64() raises:
    """Loading a properly initialized Float64 should not trigger abort."""
    var value = Float64(3.14159)
    var val = UnsafePointer(to=value).load()
    assert_true(val == 3.14159)


def test_normal_float16():
    """Loading a properly initialized Float16 should not trigger abort."""
    var value = Float16(1.5)
    _ = UnsafePointer(to=value).load()


def test_normal_bfloat16():
    """Loading a properly initialized BFloat16 should not trigger abort."""
    var value = BFloat16(1.5)
    _ = UnsafePointer(to=value).load()


def main() raises:
    test_normal_float32()
    test_qnan_not_flagged()
    test_snan_not_flagged()
    test_inf_not_flagged()
    test_integer_not_checked()
    test_near_poison_float()
    test_masked_load_poison_in_masked_off_lane()
    test_normal_float64()
    test_normal_float16()
    test_normal_bfloat16()
