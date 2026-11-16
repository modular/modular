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
# Tests for interop/libraries.mdx claims and code examples that don't need
# the custom C library. Banners follow the page's headings. The *_lit.mojo
# files test the examples that build or load libcustom.
#
# Not tested (needs a host or toolchain this file can't provide):
#   - The page's `cc` commands for building libcustom. Bazel builds the same
#     custom.c instead, because remote test machines may not have `cc`.
#   - The unversioned `libm.so` failing to load on Linux: where the C
#     development package installs it, it's a linker script, and not every
#     test machine has the package.
#   - `mojo build --emit shared-lib` on Linux building a library that uses a
#     missing symbol, and loading that library failing: it needs a separate
#     host process to load the library.
#   - Wrong argument types and counts returning nonsense: that's undefined
#     behavior, so no result is stable enough to assert.
from std.ffi import (
    OwnedDLHandle,
    c_char,
    c_double,
    c_int,
    c_long,
    c_size_t,
    external_call,
)
from std.sys.info import CompilationTarget, platform_map
from std.testing import assert_equal, assert_false, assert_raises, assert_true

comptime LIBM = platform_map["libm", linux="libm.so.6", macos="libm.dylib"]()


# --- Processes ---


def test_libc_and_libm_in_process() raises:
    # libc and libm are already in the process, with no setup.
    assert_equal(external_call["abs", c_int](c_int(-65)), c_int(65))
    assert_equal(
        external_call["hypot", c_double](c_double(3.0), c_double(4.0)), 5.0
    )


# --- Incorrect function names ---


def test_typo_naming_another_function_isnt_caught() raises:
    # A typo that spells another real function links and runs: `labs`
    # instead of `abs`.
    assert_equal(external_call["labs", c_long](c_long(-65)), c_long(65))


def test_get_function_missing_name_raises() raises:
    var process = OwnedDLHandle()
    with assert_raises(contains="symbol not found"):
        _ = process.get_function[c_int]("mojo_interop_test_not_a_symbol")


def test_check_symbol() raises:
    var process = OwnedDLHandle()
    assert_true(process.check_symbol("abs"))
    assert_false(process.check_symbol("mojo_interop_test_not_a_symbol"))


# --- Library handles and file paths ---


def test_platform_map_selects_library_name() raises:
    comptime LIBCUSTOM = platform_map[
        "libcustom", linux="./libcustom.so", macos="./libcustom.dylib"
    ]()
    comptime if CompilationTarget.is_linux():
        assert_equal(LIBCUSTOM, "./libcustom.so")
    else:
        assert_equal(LIBCUSTOM, "./libcustom.dylib")


def test_bare_versioned_name_loads() raises:
    # A bare name searches the system library directories.
    var libm = OwnedDLHandle(LIBM)
    var hypot = libm.get_function[c_double]("hypot")
    assert_equal(hypot(c_double(3.0), c_double(4.0)), 5.0)


def test_missing_path_raises() raises:
    with assert_raises(contains="dlopen failed"):
        _ = OwnedDLHandle("./libmojo-interop-test-not-installed.so")


# --- Call global libraries ---


def test_global_abs() raises:
    # [name, return type](arguments)
    var n = external_call["abs", c_int](c_int(-65))
    assert_equal(n, c_int(65))


# --- Call runtime lookups ---


def test_function_name_chosen_at_run_time() raises:
    # get_function() and check_symbol() take a run-time String.
    var name = String("ab") + "s"
    var process = OwnedDLHandle()
    assert_true(process.check_symbol(name))
    var abs_fn = process.get_function[c_int](name)
    assert_equal(abs_fn(c_int(-65)), c_int(65))


# --- Call variadic functions with external_call() ---


def test_variadic_snprintf() raises:
    # int snprintf(char *buf, size_t size, const char *fmt, ...);
    var buf = Array[c_char, 64](uninitialized=True)
    var written = external_call["snprintf", c_int, num_fixed_args=3](
        buf.unsafe_ptr(),
        c_size_t(64),
        "score: %d/%d".as_c_string_span(),
        c_int(7),
        c_int(10),
    )
    assert_equal(written, c_int(11))
    assert_equal(String(unsafe_from_utf8_ptr=buf.unsafe_ptr()), "score: 7/10")


def main() raises:
    test_libc_and_libm_in_process()
    test_typo_naming_another_function_isnt_caught()
    test_get_function_missing_name_raises()
    test_check_symbol()
    test_platform_map_selects_library_name()
    test_bare_versioned_name_loads()
    test_missing_path_raises()
    test_global_abs()
    test_function_name_chosen_at_run_time()
    test_variadic_snprintf()
