# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
# XFAIL: asan && !system-darwin
# RUN: %mojo %s

from python import Python, PythonObject
from python._cpython import Py_ssize_t, PyObjectPtr
from testing import (
    assert_false,
    assert_equal,
    assert_equal_pyobj,
    assert_raises,
    assert_true,
)


def test_PyObject_HasAttrString(mut python: Python):
    var cpython_env = python.cpython()

    var the_object = PythonObject(0)
    var result = cpython_env.PyObject_HasAttrString(
        the_object.py_object, "__contains__"
    )
    assert_equal(0, result)

    the_object = Python.list(1, 2, 3)
    result = cpython_env.PyObject_HasAttrString(
        the_object.py_object, "__contains__"
    )
    assert_equal(1, result)
    _ = the_object


def test_PyDict(mut python: Python):
    var cpy = python.cpython()

    var d = cpy.PyDict_New()
    var b = cpy.PyBool_FromLong(0)

    assert_true(cpy.PyDict_CheckExact(d))
    assert_false(cpy.PyDict_CheckExact(b))

    assert_equal(cpy.PyDict_SetItem(d, b, b), 0)
    assert_equal(cpy.PyDict_GetItemWithError(d, b), b)

    var key = PyObjectPtr()
    var value = PyObjectPtr()
    var pos: Py_ssize_t = 0

    var succ = cpy.PyDict_Next(
        d,
        UnsafePointer(to=pos),
        UnsafePointer(to=key),
        UnsafePointer(to=value),
    )
    assert_equal(pos, 1)
    assert_equal(key, b)
    assert_equal(value, b)
    assert_true(succ)

    succ = cpy.PyDict_Next(
        d,
        UnsafePointer(to=pos),
        UnsafePointer(to=key),
        UnsafePointer(to=value),
    )
    assert_false(succ)


fn destructor(capsule: PyObjectPtr) -> None:
    pass


def test_PyCapsule(mut python: Python):
    var cpython_env = python.cpython()

    # Passing an invalid PyCapsule so it should raise an error.
    var the_object = PythonObject(0)
    with assert_raises(
        contains="PyCapsule_GetPointer called with invalid PyCapsule object"
    ):
        _ = cpython_env.PyCapsule_GetPointer(the_object.py_object, "some_name")

    # Build a capsule and retrieve a pointer to it.
    var capsule_impl = UnsafePointer[UInt64].alloc(1)
    var capsule = cpython_env.PyCapsule_New(
        capsule_impl.bitcast[NoneType](), "some_name", destructor
    )
    var capsule_pointer = cpython_env.PyCapsule_GetPointer(capsule, "some_name")
    assert_equal(capsule_impl.bitcast[NoneType](), capsule_pointer)

    # PyCapsule for this name hasn't been created, so it should raise an error.
    with assert_raises(
        contains="PyCapsule_GetPointer called with incorrect name"
    ):
        _ = cpython_env.PyCapsule_GetPointer(
            capsule, "this name does not exist in the capsule"
        )


def main():
    # initializing Python instance calls init_python
    var python = Python()
    test_PyObject_HasAttrString(python)
    test_PyDict(python)
    test_PyCapsule(python)
