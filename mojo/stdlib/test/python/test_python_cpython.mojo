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

from memory import UnsafePointer
from python import Python, PythonObject
from python._cpython import PyObjectPtr
from testing import assert_equal, assert_false, assert_raises, assert_true


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
        _ = cpython_env.PyCapsule_GetPointer(capsule, "some_other_name")


def main():
    # initializing Python instance calls init_python
    var python = Python()
    test_PyObject_HasAttrString(python)
    test_PyCapsule(python)
