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
from std.testing import (
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_true,
    TestSuite,
)

from std.ffi import external_call
from std.memory import UnsafeMaybeUninit
from std.memory._nonnull import NonNullUnsafePointer
from std.sys import align_of, size_of


def test_non_null_niche() raises:
    var x = 42
    comptime NonNull = NonNullUnsafePointer[Int, ImmutOrigin(origin_of(x))]
    assert_equal(size_of[NonNull](), size_of[Optional[NonNull]]())

    var storage = UnsafeMaybeUninit[NonNull]()
    NonNull.write_niche(UnsafePointer(to=storage))
    assert_true(NonNull.isa_niche(UnsafePointer(to=storage)))

    storage.init_from(NonNull(to=x))
    assert_false(NonNull.isa_niche(UnsafePointer(to=storage)))


def test_non_null_dangling() raises:
    var int_ptr = NonNullUnsafePointer[Int, MutExternalOrigin].dangling()
    assert_equal(Int(int_ptr) % align_of[Int](), 0)

    var str_ptr = NonNullUnsafePointer[String, MutExternalOrigin].dangling()
    assert_equal(Int(str_ptr) % align_of[String](), 0)


def test_optional_non_null_across_c_ffi() raises:
    var string = "abc"
    comptime Result = Optional[NonNullUnsafePointer[Int8, origin_of(string)]]

    var not_found = external_call[
        "strchr",
        Result,
    ](string.as_c_string_slice(), Int8(ord("z")))
    assert_false(not_found)

    var found = external_call[
        "strchr",
        Result,
    ](string.as_c_string_slice(), Int8(ord("a")))
    assert_true(found)
    assert_equal(Int(found[]), Int(string.unsafe_ptr()))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
