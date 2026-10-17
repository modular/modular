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
#
# Verifies that loading the debug allocator poison pattern aborts when built
# with -D MOJO_STDLIB_SIMD_UNINIT_CHECK=true.
#
# ===----------------------------------------------------------------------=== #

from std.memory import Pointer, alloc
from std.sys.intrinsics import masked_load
from std.testing import _assert_aborts, TestSuite

comptime POISON_MESSAGE = "load matched debug allocator poison sentinel"


def test_float32_poison() raises:
    # FLT_MAX = 0x7F7FFFFF
    def trigger() raises {} -> None:
        var value = UInt32(0x7F7FFFFF)
        var ptr = Pointer(to=value).unsafe_bitcast[Float32]()
        _ = ptr.unsafe_load()

    _assert_aborts(trigger, contains=POISON_MESSAGE)


def test_float16_poison() raises:
    # 65504 = 0x7BFF
    def trigger() raises {} -> None:
        var value = UInt16(0x7BFF)
        var ptr = Pointer(to=value).unsafe_bitcast[Float16]()
        _ = ptr.unsafe_load()

    _assert_aborts(trigger, contains=POISON_MESSAGE)


def test_bfloat16_poison() raises:
    # Largest finite BFloat16 = 0x7F7F
    def trigger() raises {} -> None:
        var value = UInt16(0x7F7F)
        var ptr = Pointer(to=value).unsafe_bitcast[BFloat16]()
        _ = ptr.unsafe_load()

    _assert_aborts(trigger, contains=POISON_MESSAGE)


def test_float64_poison() raises:
    # DBL_MAX = 0x7FEFFFFFFFFFFFFF
    def trigger() raises {} -> None:
        var value = UInt64(0x7FEFFFFFFFFFFFFF)
        var ptr = Pointer(to=value).unsafe_bitcast[Float64]()
        _ = ptr.unsafe_load()

    _assert_aborts(trigger, contains=POISON_MESSAGE)


def test_simd_vector_poison() raises:
    # A 4-wide load where only element 2 is poisoned.
    def trigger() raises {} -> None:
        var allocation = alloc[Float32]({count = 4}).into_managed()
        var ptr = allocation.unsafe_ptr()
        ptr.unsafe_store(0, Float32(1.0))
        ptr.unsafe_store(1, Float32(2.0))
        ptr.unsafe_store(2, Float32(3.0))
        ptr.unsafe_store(3, Float32(4.0))
        ptr.unsafe_offset(2).unsafe_bitcast[UInt32]().unsafe_store(
            UInt32(0x7F7FFFFF)
        )
        _ = ptr.unsafe_load[width=4]()

    _assert_aborts(trigger, contains=POISON_MESSAGE)


def test_masked_load_poison() raises:
    # Lane 1 is unmasked (loaded from memory) and poisoned.
    def trigger() raises {} -> None:
        var allocation = alloc[Float32]({count = 4}).into_managed()
        var ptr = allocation.unsafe_ptr()
        ptr.unsafe_store(0, Float32(1.0))
        ptr.unsafe_store(1, Float32(2.0))
        ptr.unsafe_store(2, Float32(3.0))
        ptr.unsafe_store(3, Float32(4.0))
        ptr.unsafe_offset(1).unsafe_bitcast[UInt32]().unsafe_store(
            UInt32(0x7F7FFFFF)
        )
        var mask = SIMD[.bool, 4](True, True, False, False)
        var passthrough = SIMD[.float32, 4](0)
        _ = masked_load(ptr, mask, passthrough)

    _assert_aborts(trigger, contains=POISON_MESSAGE)


def test_gather_poison() raises:
    # Lane 2 is unmasked and poisoned.
    def trigger() raises {} -> None:
        var allocation = alloc[Float32]({count = 4}).into_managed()
        var ptr = allocation.unsafe_ptr()
        ptr.unsafe_store(0, Float32(1.0))
        ptr.unsafe_store(1, Float32(2.0))
        ptr.unsafe_store(2, Float32(3.0))
        ptr.unsafe_store(3, Float32(4.0))
        ptr.unsafe_offset(2).unsafe_bitcast[UInt32]().unsafe_store(
            UInt32(0x7F7FFFFF)
        )
        var offset = SIMD[.int64, 4](0, 1, 2, 3)
        var mask = SIMD[.bool, 4](True, True, True, False)
        _ = ptr.unsafe_gather(offset=offset, mask=mask)

    _assert_aborts(trigger, contains=POISON_MESSAGE)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
