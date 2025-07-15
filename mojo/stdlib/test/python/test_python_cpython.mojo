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

from python import Python, PythonObject
from python._cpython import Py_eval_input, Py_ssize_t, PyMethodDef, PyObjectPtr
from testing import (
    assert_false,
    assert_equal,
    assert_equal_pyobj,
    assert_raises,
    assert_true,
)


def test_very_high_level_api(python: Python):
    var cpy = python.cpython()

    assert_equal(cpy.PyRun_SimpleString("None"), 0)

    var d = cpy.PyDict_New()
    assert_true(cpy.PyRun_String("42", Py_eval_input, d, d))

    var co = cpy.Py_CompileString("5", "test", Py_eval_input)
    assert_true(co)

    assert_true(cpy.PyEval_EvalCode(co, d, d))


def test_Py_IncRef_DecRef(mut python: Python):
    var cpy = python.cpython()

    # this is the smallest integer that's GC'd by the Python interpreter
    var n = cpy.PyLong_FromSsize_t(257)
    assert_equal(cpy._Py_REFCNT(n), 1)

    cpy.Py_IncRef(n)
    assert_equal(cpy._Py_REFCNT(n), 2)

    cpy.Py_DecRef(n)
    assert_equal(cpy._Py_REFCNT(n), 1)


def test_PyErr(python: Python):
    var cpy = python.cpython()

    var ValueError = cpy.get_error_global("PyExc_ValueError")
    var msg = "some error message"

    assert_false(cpy.PyErr_Occurred())

    cpy.PyErr_SetNone(ValueError)
    assert_true(cpy.PyErr_Occurred())
    cpy.PyErr_Clear()

    cpy.PyErr_SetString(ValueError, msg.unsafe_cstr_ptr())
    assert_true(cpy.PyErr_Occurred())

    if cpy.version.minor < 12:
        # PyErr_Fetch is deprecated since Python 3.12.
        assert_true(cpy.PyErr_Fetch())
        # Manually clear the error indicator.
        cpy.PyErr_Clear()
    else:
        # PyErr_GetRaisedException is new in Python 3.12.
        # PyErr_GetRaisedException clears the error indicator.
        assert_true(cpy.PyErr_GetRaisedException())

    _ = msg


def test_PyThread(python: Python):
    var cpy = python.cpython()

    var gstate = cpy.PyGILState_Ensure()
    var save = cpy.PyEval_SaveThread()
    cpy.PyEval_RestoreThread(save)
    cpy.PyGILState_Release(gstate)


def test_PyImport(python: Python):
    var cpy = python.cpython()

    assert_true(cpy.PyImport_ImportModule("builtins"))
    assert_true(cpy.PyImport_AddModule("test"))


def test_object_protocol_api(python: Python):
    var cpy = python.cpython()

    var n = cpy.PyLong_FromSsize_t(42)
    var z = cpy.PyLong_FromSsize_t(0)
    var l = cpy.PyList_New(1)
    _ = cpy.PyList_SetItem(l, 0, z)

    assert_equal(cpy.PyObject_HasAttrString(n, "__hash__"), 1)
    assert_true(cpy.PyObject_GetAttrString(n, "__hash__"))
    assert_equal(cpy.PyObject_SetAttrString(n, "attr", cpy.Py_None()), -1)
    cpy.PyErr_Clear()

    assert_true(cpy.PyObject_Str(n))
    assert_equal(cpy.PyObject_Hash(n), 42)
    assert_equal(cpy.PyObject_IsTrue(n), 1)
    assert_true(cpy.PyObject_Type(n))
    assert_equal(cpy.PyObject_Length(l), 1)

    assert_equal(cpy.PyObject_GetItem(l, z), z)
    assert_equal(cpy.PyObject_SetItem(l, z, n), 0)
    assert_equal(cpy.PyObject_GetItem(l, z), n)

    var it = cpy.PyObject_GetIter(l)
    assert_true(it)
    assert_equal(cpy.PyObject_GetIter(it), it)


def test_call_protocol_api(python: Python):
    var cpy = python.cpython()

    var dict_func = PyObjectPtr(upcast_from=cpy.PyDict_Type())
    var t = cpy.PyTuple_New(0)
    var d = cpy.PyDict_New()

    assert_true(cpy.PyObject_CallObject(dict_func, t))
    assert_true(cpy.PyObject_Call(dict_func, t, d))


def test_type_object_api(python: Python):
    var cpy = python.cpython()

    var dict_type = cpy.PyDict_Type()
    assert_true(cpy.PyType_GetName(dict_type))


def test_module_object_api(python: Python):
    var cpy = python.cpython()

    var mod = cpy.PyModule_Create("module")

    assert_true(mod)
    assert_true(cpy.PyModule_GetDict(mod))

    var funcs = InlineArray[PyMethodDef, 1](fill={})
    # returns 0 on success, -1 on failure
    assert_equal(cpy.PyModule_AddFunctions(mod, funcs.unsafe_ptr()), 0)
    _ = funcs

    if cpy.version.minor >= 10:
        var n = cpy.PyLong_FromSsize_t(0)
        var name = "n"
        # returns 0 on success, -1 on failure
        assert_equal(
            cpy.PyModule_AddObjectRef(mod, name.unsafe_cstr_ptr(), n), 0
        )
        _ = name


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
        _ = cpython_env.PyCapsule_GetPointer(the_object._obj_ptr, "some_name")

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


def test_common_object_structure_api(python: Python):
    var cpy = python.cpython()

    var n = cpy.PyLong_FromSsize_t(42)
    assert_true(cpy.Py_Is(n, n))

    var dict_type = cpy.PyDict_Type()
    var d = cpy.PyDict_New()

    var d_type = cpy.Py_TYPE(d)
    assert_equal(
        PyObjectPtr(upcast_from=d_type),
        PyObjectPtr(upcast_from=dict_type),
    )


def main():
    # initializing Python instance calls init_python
    var python = Python()

    # The Very High Level Layer
    test_very_high_level_api(python)

    # Reference Counting
    test_Py_IncRef_DecRef(python)

    # Exception Handling
    test_PyErr(python)

    # Initialization, Finalization, and Threads
    test_PyThread(python)

    # Importing Modules
    test_PyImport(python)

    # Abstract Objects Layer

    # Object Protocol
    test_object_protocol_api(python)

    # Call Protocol
    test_call_protocol_api(python)

    # Concrete Objects Layer

    # Type Objects
    test_type_object_api(python)

    # Module Objects
    test_module_object_api(python)

    test_PyDict(python)
    test_PyCapsule(python)

    # Object Implementation Support

    # Common Object Structures
    test_common_object_structure_api(python)
