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
"""Typed-argument fast-path wrappers for `PythonModuleBuilder`.

For a Mojo function whose Python-facing signature is *concrete* (e.g.
`def f(a: Int, b: Int) -> Int`), we can build a CPython wrapper that
unwraps positional arguments with the type-specific CPython API
directly (`PyLong_AsSsize_t`, `PyFloat_AsDouble`, ...) instead of
routing through `PythonObject` and `Int(py=...)`. The latter goes
through `__int__()` -> `PyNumber_Long`, which allocates a fresh `int`
object per call.
"""

# TODO: These wrappers use `METH_VARARGS` for now. Once the `METH_FASTCALL`
# plumbing lands (see issue #6521 / Joe's stack), switching them over is a
# single-line change of the underlying `PyMethodDef` flags.
from std.python import Python
from std.python._cpython import (
    CPython,
    PyObjectPtr,
    Py_ssize_t,
)


# Compile-time aliases for the supported user-function shapes.
# A single raising signature per arity; non-raising user functions are
# implicitly compatible with the raising form (they simply never raise),
# which avoids overload ambiguity between raises and non-raises shapes.
comptime _FnIntToInt = def(Int) thin raises -> Int
comptime _FnIntIntToInt = def(Int, Int) thin raises -> Int


@always_inline
def _set_exception(cpy: CPython, var msg: String):
    var exc_type = cpy.get_error_global("PyExc_Exception")
    cpy.PyErr_SetString(exc_type, msg.as_c_string_slice().unsafe_ptr())


@always_inline
def _check_arity(
    cpy: CPython, args: PyObjectPtr, expected: Py_ssize_t
) -> Bool:
    """Return True if `args` (a tuple) has exactly `expected` items.

    On mismatch, sets `PyExc_TypeError` and returns False so the caller
    can return NULL to surface the error to Python (matching the
    `TypeError: f() takes N positional arguments but M were given`
    convention).
    """
    var got = cpy.PyObject_Length(args)
    if got == expected:
        return True
    var exc_type = cpy.get_error_global("PyExc_TypeError")
    cpy.PyErr_SetString(
        exc_type,
        StaticString("wrong number of positional arguments")
        .unsafe_ptr()
        .bitcast[Int8](),
    )
    return False


@always_inline
def _get_int_arg(
    cpy: CPython, args: PyObjectPtr, i: Py_ssize_t
) -> Tuple[Int, Bool]:
    """Fetch the i-th positional arg and unwrap it as an Int.

    Returns `(value, ok)`. On error (out-of-range index, non-int arg,
    overflow), `ok` is False and a Python exception is already set.
    """
    var ptr = cpy.PyTuple_GetItem(args, i)  # borrowed; NULL + IndexError if OOR
    if not ptr:
        return (0, False)
    var raw = cpy.PyLong_AsSsize_t(ptr)
    if raw == -1 and cpy.PyErr_Occurred():
        return (0, False)
    return (raw, True)


# ===-----------------------------------------------------------------------===#
# (Int) -> Int  /  (Int) raises -> Int
# ===-----------------------------------------------------------------------===#


def _typed_int_to_int_wrapper[
    user_func: _FnIntToInt
](_self: PyObjectPtr, args: PyObjectPtr) -> PyObjectPtr:
    ref cpy = Python().cpython()
    try:
        if not _check_arity(cpy, args, 1):
            return PyObjectPtr()
        var a = _get_int_arg(cpy, args, 0)
        if not a[1]:
            return PyObjectPtr()
        return cpy.PyLong_FromSsize_t(user_func(a[0]))
    except e:
        _set_exception(cpy, String(e))
        return PyObjectPtr()


# ===-----------------------------------------------------------------------===#
# (Int, Int) -> Int  /  (Int, Int) raises -> Int
# ===-----------------------------------------------------------------------===#


def _typed_int_int_to_int_wrapper[
    user_func: _FnIntIntToInt
](_self: PyObjectPtr, args: PyObjectPtr) -> PyObjectPtr:
    ref cpy = Python().cpython()
    try:
        if not _check_arity(cpy, args, 2):
            return PyObjectPtr()
        var a = _get_int_arg(cpy, args, 0)
        if not a[1]:
            return PyObjectPtr()
        var b = _get_int_arg(cpy, args, 1)
        if not b[1]:
            return PyObjectPtr()
        return cpy.PyLong_FromSsize_t(user_func(a[0], b[0]))
    except e:
        _set_exception(cpy, String(e))
        return PyObjectPtr()
