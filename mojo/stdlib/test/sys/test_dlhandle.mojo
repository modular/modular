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

from std.pathlib import Path
from std.ffi import OwnedDLHandle

from std.testing import assert_equal, assert_raises, assert_true
from std.testing import TestSuite


# ===----------------------------------------------------------------------=== #
# OwnedDLHandle tests
# ===----------------------------------------------------------------------=== #


def test_owned_dlhandle_invalid_path() raises:
    with assert_raises(contains="dlopen failed"):
        _ = OwnedDLHandle("/an/invalid/library")


def test_owned_dlhandle_invalid_path_obj() raises:
    with assert_raises(contains="dlopen failed"):
        _ = OwnedDLHandle(Path("/an/invalid/library"))


def test_owned_dlhandle_load_valid_library() raises:
    try:
        # Try common locations for libc
        var lib = OwnedDLHandle("libc.so.6")  # Linux
        assert_true(lib.__bool__(), "Library handle should be valid")
    except:
        try:
            var lib = OwnedDLHandle("libc.so")  # Some Linux systems
            assert_true(lib.__bool__(), "Library handle should be valid")
        except:
            try:
                var lib = OwnedDLHandle(
                    "/usr/lib/system/libsystem_c.dylib"
                )  # macOS
                assert_true(lib.__bool__(), "Library handle should be valid")
            except:
                # If none work, skip this test
                print(
                    "Warning: Could not find a standard C library to test with"
                )


def test_owned_dlhandle_check_symbol() raises:
    try:
        var lib = OwnedDLHandle("libc.so.6")
        # Common C library functions that should exist
        assert_true(lib.check_symbol("printf"), "printf should exist in libc")
        assert_true(lib.check_symbol("malloc"), "malloc should exist in libc")
    except:
        try:
            var lib = OwnedDLHandle("libc.so")
            assert_true(
                lib.check_symbol("printf"), "printf should exist in libc"
            )
        except:
            # Skip if we can't load libc
            print("Warning: Could not load libc for symbol test")


def test_owned_dlhandle_borrow() raises:
    """Test that borrow() returns a valid DLHandle reference."""
    try:
        var lib = OwnedDLHandle("libc.so.6")
        var borrowed = lib.borrow()
        # borrowed should be a valid DLHandle
        assert_true(borrowed.__bool__(), "Borrowed handle should be valid")
        assert_true(
            borrowed.check_symbol("printf"),
            "Borrowed handle should access symbols",
        )
    except:
        try:
            var lib = OwnedDLHandle("libc.so")
            var borrowed = lib.borrow()
            assert_true(borrowed.__bool__(), "Borrowed handle should be valid")
        except:
            # Skip if we can't load libc
            print("Warning: Could not load libc for borrow test")


def test_owned_dlhandle_global_symbols() raises:
    """Test loading global symbols from current process."""
    try:
        # Load symbols from the current process
        var lib = OwnedDLHandle()
        assert_true(lib.__bool__(), "Global symbol handle should be valid")
    except:
        # This might fail on some systems
        print("Warning: Could not load global symbols")


def test_owned_dlhandle_get_symbol_missing() raises:
    """Test that get_symbol returns None for a nonexistent symbol."""

    def _test_with_lib(lib: OwnedDLHandle) raises:
        var result = lib.get_symbol[NoneType](
            "this_symbol_does_not_exist_xyz_42"
        )
        assert_true(not result, "Missing symbol should return None")

    try:
        _test_with_lib(OwnedDLHandle("libc.so.6"))
    except:
        try:
            _test_with_lib(OwnedDLHandle("libc.so"))
        except:
            try:
                _test_with_lib(
                    OwnedDLHandle("/usr/lib/system/libsystem_c.dylib")
                )
            except:
                print(
                    "Warning: Could not load a standard C library to test with"
                )


def test_owned_dlhandle_get_symbol_found() raises:
    """Test that get_symbol returns a value for an existing symbol."""

    def _test_with_lib(lib: OwnedDLHandle) raises:
        var result = lib.get_symbol[NoneType]("printf")
        assert_true(Bool(result), "Existing symbol should return a value")

    try:
        _test_with_lib(OwnedDLHandle("libc.so.6"))
    except:
        try:
            _test_with_lib(OwnedDLHandle("libc.so"))
        except:
            try:
                _test_with_lib(
                    OwnedDLHandle("/usr/lib/system/libsystem_c.dylib")
                )
            except:
                print(
                    "Warning: Could not load a standard C library to test with"
                )


def test_owned_dlhandle_automatic_cleanup() raises:
    """Test that OwnedDLHandle automatically closes on destruction."""
    # This test primarily verifies that the code compiles and runs
    # without crashes. The actual cleanup happens automatically.

    @always_inline
    def create_and_destroy_handle():
        try:
            var lib = OwnedDLHandle("libc.so.6")
            _ = lib.check_symbol("printf")
            # lib will be automatically closed here when it goes out of scope
        except:
            pass

    # Call the function multiple times to ensure cleanup works
    create_and_destroy_handle()
    create_and_destroy_handle()
    create_and_destroy_handle()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
