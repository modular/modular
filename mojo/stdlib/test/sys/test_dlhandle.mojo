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

from std.sys.info import CompilationTarget
from std.testing import assert_equal, assert_raises, assert_true
from std.testing import TestSuite


def _load_libc() raises -> OwnedDLHandle:
    """Loads libc from the standard location for this platform.

    Selects platform-appropriate paths up front so that a failure on
    (say) Linux doesn't propagate a confusing macOS-path error message.
    """
    comptime if CompilationTarget.is_linux():
        try:
            return OwnedDLHandle("libc.so.6")  # glibc
        except:
            pass
        return OwnedDLHandle("libc.so")  # musl / BSD
    elif CompilationTarget.is_macos():
        return OwnedDLHandle("/usr/lib/system/libsystem_c.dylib")
    else:
        comptime assert False, "libc discovery not implemented for platform"


def _load_libm() raises -> OwnedDLHandle:
    """Loads libm (math functions) from the standard location for this
    platform.

    On glibc, math lives in libm.so.6. On macOS and musl, math symbols
    are folded into libc, so we fall back to `_load_libc` there.
    """
    comptime if CompilationTarget.is_linux():
        try:
            return OwnedDLHandle("libm.so.6")  # glibc
        except:
            pass
        # musl folds math into libc.
        return _load_libc()
    else:
        # On macOS, math symbols are available through libSystem / libc.
        return _load_libc()


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


def test_owned_dlhandle_get_function_keepalive() raises:
    """Inline resolve and call with no later use of the handle."""
    var lib = _load_libc()
    # Inline resolve + call, no subsequent use of `lib`.
    var pid = lib.get_function[Int32]("getpid")()
    assert_true(pid > 0, "getpid should return a positive pid")


def test_owned_dlhandle_get_function_stored_callable() raises:
    var lib = _load_libc()
    var getpid_fn = lib.get_function[Int32]("getpid")
    assert_true(getpid_fn() > 0, "call 1")
    assert_true(getpid_fn() > 0, "call 2")
    assert_true(getpid_fn() > 0, "call 3")


def test_owned_dlhandle_get_function_multiple_inline_calls() raises:
    """Repeated inline resolve and call; the last call previously crashed."""
    var lib = _load_libc()
    _ = lib.get_function[Int32]("getpid")()
    _ = lib.get_function[Int32]("getpid")()
    _ = lib.get_function[Int32]("getpid")()
    # The final call is the one that used to crash.
    _ = lib.get_function[Int32]("getpid")()


def test_owned_dlhandle_get_function_with_args() raises:
    """Exercise the variadic argument-forwarding path with a scalar-in,
    scalar-out function, where the Mojo calling convention is safe."""
    var lib = _load_libc()
    var abs_fn = lib.get_function[Int32]("abs")
    assert_equal(abs_fn(Int32(-5)), Int32(5), "abs(-5) should return 5")
    assert_equal(abs_fn(Int32(42)), Int32(42), "abs(42) should return 42")
    assert_equal(abs_fn(Int32(0)), Int32(0), "abs(0) should return 0")


def test_owned_dlhandle_get_function_missing_symbol_raises() raises:
    """A missing symbol raises `Error` (previously aborted the process).
    Ensures callers that probe for optional symbols can recover."""
    var lib = _load_libc()
    with assert_raises(contains="symbol not found"):
        _ = lib.get_function[Int32]("this_symbol_does_not_exist_xyz_42")


def test_owned_dlhandle_get_function_float64_return() raises:
    """Exercises the `Float64` return-type path to match the docstring
    example and ensure non-`Int32` scalars round-trip through the Mojo
    ABI forwarding correctly."""
    var lib = _load_libm()
    var sqrt_fn = lib.get_function[Float64]("sqrt")
    assert_equal(sqrt_fn(Float64(4.0)), Float64(2.0), "sqrt(4.0)")
    assert_equal(sqrt_fn(Float64(0.0)), Float64(0.0), "sqrt(0.0)")
    assert_equal(sqrt_fn(Float64(1.0)), Float64(1.0), "sqrt(1.0)")


def test_owned_dlhandle_get_function_default_return_type() raises:
    """Exercises the default `NoneType` return type (omitted type param)
    against a void-returning libc function. `srand(unsigned)` takes a
    scalar and returns void."""
    var lib = _load_libc()
    var srand_fn = lib.get_function("srand")
    srand_fn(UInt32(42))
    srand_fn(UInt32(0))


def test_owned_dlhandle_get_function_explicit_nonetype_return() raises:
    """Same as the default-return-type test, but with `NoneType` stated
    explicitly — covers the shape used by `TVMFFIErrorMoveFromRaised`
    call sites."""
    var lib = _load_libc()
    var srand_fn = lib.get_function[NoneType]("srand")
    srand_fn(UInt32(7))


def test_owned_dlhandle_automatic_cleanup() raises:
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
