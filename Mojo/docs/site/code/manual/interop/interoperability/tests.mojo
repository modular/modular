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
# tests.mojo
# Tests for interop/interoperability.mdx claims and code examples. Banners
# follow the page's headings.
#
# A C program calling the exported `add()` is tested in mylib_c_host.mojo.
#
# Not tested (needs a host or toolchain this file can't provide):
#   - The page's build commands as written (`cc ... -lmylib` with rpath, and
#     `install_name_tool` on macOS). mylib_c_host.mojo loads the library with
#     `dlopen()` instead, because Bazel can't link against a library built in
#     a lit `RUN:` line.
#   - Python loading the library through `ctypes`
#   - Calling the library from C++ or Rust
from std.ffi import c_int, c_long, c_size_t, external_call
from std.sys import size_of
from std.testing import assert_equal, assert_true


# --- What is the C ABI? ---


def test_c_long_is_64_bit() raises:
    # The size of C's `long` depends on the platform. On every platform Mojo
    # supports today (64-bit macOS and Linux), `c_long` is 64 bits.
    assert_equal(size_of[c_long](), 8)
    assert_equal(size_of[c_int](), 4)


def test_call_c_function() raises:
    # long labs(long n); Mojo calls C through the platform's C ABI.
    assert_equal(external_call["labs", c_long](c_long(-42)), c_long(42))


# Data representations: numbers, pointers, function pointers, and structs
# with C-compatible layouts.


@fieldwise_init
struct DivT(RegisterPassable):  # Matches C's div_t.
    var quot: c_int
    var rem: c_int


def test_struct_by_value() raises:
    var result = external_call["div", DivT](c_int(17), c_int(5))
    assert_equal(result.quot, c_int(3))
    assert_equal(result.rem, c_int(2))


@fieldwise_init
struct TimeSpec:  # Matches C's struct timespec.
    var seconds: c_long
    var nanoseconds: c_long


def test_struct_by_pointer() raises:
    var now = TimeSpec(0, 0)
    var status = external_call["clock_gettime", c_int](
        c_int(0), Pointer(to=now)  # CLOCK_REALTIME
    )
    assert_equal(status, c_int(0))
    assert_true(now.seconds > 1_700_000_000)
    assert_true(now.nanoseconds >= 0 and now.nanoseconds < 1_000_000_000)


def ascending(
    a: ImmOpaquePointer[ImmUntrackedOrigin],
    b: ImmOpaquePointer[ImmUntrackedOrigin],
) abi("C") -> c_int:
    var a_value = a.unsafe_bitcast[c_int]()[]
    var b_value = b.unsafe_bitcast[c_int]()[]
    if a_value < b_value:
        return c_int(-1)
    return c_int(a_value > b_value)


def test_function_pointer() raises:
    var values: List[c_int] = [42, 4, 23, 8, 16, 15]
    external_call["qsort", NoneType](
        values.unsafe_ptr(),
        c_size_t(len(values)),
        c_size_t(size_of[c_int]()),
        ascending,
    )
    assert_equal(values, [c_int(4), 8, 15, 16, 23, 42])


# --- Example: Call Mojo from C ---


# The page's mylib.mojo, called from Mojo here and from C in
# mylib_c_host.mojo.
@export("add")
def add(x: c_long, y: c_long) abi("C") -> c_long:
    return x + y


def test_exported_add() raises:
    assert_equal(add(5, 103), c_long(108))


def main() raises:
    test_c_long_is_64_bit()
    test_call_c_function()
    test_struct_by_value()
    test_struct_by_pointer()
    test_function_pointer()
    test_exported_add()
