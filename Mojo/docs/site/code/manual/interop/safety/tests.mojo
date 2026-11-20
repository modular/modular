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
# Tests for interop/safety.mdx claims whose results are stable on every
# platform. Banners follow the page's headings.
#
# Not tested (no stable result to assert from a single process):
#   - "Caught at build time": every entry is a compile error, so a program
#     that contains one never runs
#   - Undefined behavior that varies by platform or run: a narrower return
#     type, `get_function()` arguments of the wrong type, a raw `String`
#     passed through `get_function()`, a variadic call without
#     `num_fixed_args`, callbacks without `abi("C")`, capturing lambdas,
#     mixed allocators, `ImmStaticOrigin` on library-owned data, pointers
#     native code keeps, and non-nul-terminated buffers
#   - Export entries (`@export` signatures, `initialize_runtime()`, shipping
#     the runtime, stack traces, `sys.argv()`), which need a native host
#   - `--as-needed`, `mojo run` versus `mojo build`, and symbol-name rules
#     for C++, Rust, and Swift, which need separate builds
from std.ffi import OwnedDLHandle, c_char, c_double
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.python.numpy import from_numpy_array
from std.sys.info import CompilationTarget, platform_map
from std.testing import assert_equal, assert_false, assert_raises, assert_true

comptime LIBM = platform_map["libm", linux="libm.so.6", macos="libm.dylib"]()


@fieldwise_init
struct Sensor(Movable, Writable):
    var id: Int


@fieldwise_init
struct Unregistered(Movable, Writable):
    var id: Int


def register_sensor() raises:
    var builder = PythonModuleBuilder("sensors")
    _ = builder.add_type[Sensor]("Sensor")
    _ = builder.finalize()


# --- What Mojo checks for you ---


def test_checked_downcast_raises() raises:
    with assert_raises(contains="expected Mojo"):
        _ = PythonObject(7).downcast_value_ptr[Sensor]()


def test_python_exception_becomes_error() raises:
    with assert_raises(contains="division by zero"):
        _ = Python.evaluate("1 / 0")


# --- Availability checks ---


def test_missing_library_raises() raises:
    with assert_raises(contains="dlopen failed"):
        _ = OwnedDLHandle("libmojo-interop-test-not-installed.so.1")


def test_missing_symbol_raises() raises:
    var libm = OwnedDLHandle(LIBM)
    assert_false(libm.check_symbol("mojo_interop_test_not_a_symbol"))
    with assert_raises(contains="symbol not found"):
        _ = libm.get_function[c_double]("mojo_interop_test_not_a_symbol")


def test_exp10_check_symbol() raises:
    var libm = OwnedDLHandle(LIBM)
    comptime if CompilationTarget.is_linux():
        assert_true(libm.check_symbol("exp10"))
    else:
        assert_false(libm.check_symbol("exp10"))


def test_missing_python_module_raises() raises:
    with assert_raises(contains="No module named"):
        _ = Python.import_module("mojo_interop_test_not_a_module")


def test_unregistered_alloc_raises() raises:
    with assert_raises(contains="No Python type object registered"):
        _ = PythonObject(alloc=Unregistered(9))


# --- Silently wrong at runtime ---


def test_char_signedness_varies() raises:
    comptime if CompilationTarget.is_linux() and CompilationTarget.is_arm():
        assert_false(c_char.dtype.is_signed())
    else:
        assert_true(c_char.dtype.is_signed())


def test_narrow_integer_conversions_wrap() raises:
    var builtins = Python.import_module("builtins")
    assert_equal(UInt8(py=PythonObject(300)), 44)
    assert_equal(Int8(py=PythonObject(200)), -56)
    assert_equal(Int32(py=builtins.pow(2, 40)), 0)
    with assert_raises():
        _ = UInt8(py=PythonObject(-1))
    with assert_raises():
        _ = Int64(py=builtins.pow(2, 63))
    with assert_raises():
        _ = Int64(py=builtins.pow(2, 70))


def test_permissive_conversions() raises:
    assert_equal(String(py=PythonObject(None)), "None")
    assert_equal(Int(py=PythonObject(3.9)), 3)
    assert_equal(Int(py=PythonObject("12")), 12)
    assert_false(Bool(py=Python.list()))
    assert_true(Bool(py=PythonObject("x")))


def test_numpy_view_shares_buffer() raises:
    # A view, not a copy: it points at the array's own buffer. Whether a
    # later resize moves that buffer is up to the allocator, so the test
    # doesn't assert it.
    var np = Python.import_module("numpy")
    var arr = np.zeros(4)
    var view = from_numpy_array[.float64](arr)
    assert_equal(Int(view.unsafe_ptr()), Int(py=arr.ctypes.data))


# --- Limitations ---


def test_supported_python() raises:
    var sys = Python.import_module("sys")
    var sysconfig = Python.import_module("sysconfig")
    assert_equal(Int(py=sys.version_info.major), 3)
    assert_true(Int(py=sys.version_info.minor) >= 10)
    assert_false(Bool(py=sysconfig.get_config_var("Py_GIL_DISABLED")))


def main() raises:
    register_sensor()
    test_checked_downcast_raises()
    test_python_exception_becomes_error()
    test_missing_library_raises()
    test_missing_symbol_raises()
    test_exp10_check_symbol()
    test_missing_python_module_raises()
    test_unregistered_alloc_raises()
    test_char_signedness_varies()
    test_narrow_integer_conversions_wrap()
    test_permissive_conversions()
    test_numpy_view_shares_buffer()
    test_supported_python()
