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

from python import Python
from python._cpython import (
    CPython,
    Py_eval_input,
    Py_ssize_t,
    PyMethodDef,
    PyObjectPtr,
)
from testing import (
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)


def test_very_high_level_api(cpy: CPython):
    assert_equal(cpy.PyRun_SimpleString("None"), 0)

    var d = cpy.PyDict_New()
    assert_true(cpy.PyRun_String("42", Py_eval_input, d, d))

    var co = cpy.Py_CompileString("5", "test", Py_eval_input)
    assert_true(co)

    assert_true(cpy.PyEval_EvalCode(co, d, d))


def test_reference_counting_api(cpy: CPython):
    # this is the smallest integer that's GC'd by the Python interpreter
    var n = cpy.PyLong_FromSsize_t(257)
    assert_equal(cpy._Py_REFCNT(n), 1)

    cpy.Py_IncRef(n)
    assert_equal(cpy._Py_REFCNT(n), 2)

    cpy.Py_DecRef(n)
    assert_equal(cpy._Py_REFCNT(n), 1)


def test_exception_handling_api(cpy: CPython):
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


def test_threading_api(cpy: CPython):
    var gstate = cpy.PyGILState_Ensure()
    var save = cpy.PyEval_SaveThread()
    cpy.PyEval_RestoreThread(save)
    cpy.PyGILState_Release(gstate)


def test_importing_module_api(cpy: CPython):
    assert_true(cpy.PyImport_ImportModule("builtins"))
    assert_true(cpy.PyImport_AddModule("test"))


def test_object_protocol_api(cpy: CPython):
    var n = cpy.PyLong_FromSsize_t(42)
    var z = cpy.PyLong_FromSsize_t(0)
    var l = cpy.PyList_New(1)
    cpy.Py_IncRef(z)
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


def test_call_protocol_api(cpy: CPython):
    var dict_func = PyObjectPtr(upcast_from=cpy.PyDict_Type())
    var t = cpy.PyTuple_New(0)
    var d = cpy.PyDict_New()

    assert_true(cpy.PyObject_CallObject(dict_func, t))
    assert_true(cpy.PyObject_Call(dict_func, t, d))


def test_number_protocol_api(cpy: CPython):
    var n = cpy.PyLong_FromSsize_t(42)

    var long_value = cpy.PyNumber_Long(n)
    assert_true(long_value)
    assert_equal(cpy.PyLong_AsSsize_t(long_value), 42)

    var float_value = cpy.PyNumber_Float(n)
    assert_true(float_value)
    assert_equal(cpy.PyFloat_AsDouble(float_value), 42.0)


def test_iterator_protocol_api(cpy: CPython):
    var n = cpy.PyLong_FromSsize_t(42)
    var l = cpy.PyList_New(1)
    cpy.Py_IncRef(n)
    _ = cpy.PyList_SetItem(l, 0, n)

    var it = cpy.PyObject_GetIter(l)

    assert_false(cpy.PyIter_Check(n))
    assert_true(it)
    assert_true(cpy.PyIter_Next(it))


def test_type_object_api(cpy: CPython):
    var dict_type = cpy.PyDict_Type()
    assert_true(cpy.PyType_GetName(dict_type))


def test_integer_object_api(cpy: CPython):
    var n = cpy.PyLong_FromSsize_t(-42)
    assert_true(n)
    assert_equal(cpy.PyLong_AsSsize_t(n), -42)

    var z = cpy.PyLong_FromSize_t(57)
    assert_true(z)
    assert_equal(cpy.PyLong_AsSsize_t(z), 57)


def test_boolean_object_api(cpy: CPython):
    var t = cpy.PyBool_FromLong(1)
    assert_true(t)
    assert_equal(cpy.PyObject_IsTrue(t), 1)

    var f = cpy.PyBool_FromLong(0)
    assert_true(f)
    assert_equal(cpy.PyObject_IsTrue(f), 0)


def test_floating_point_object_api(cpy: CPython):
    var f = cpy.PyFloat_FromDouble(3.14)
    assert_true(f)
    assert_equal(cpy.PyFloat_AsDouble(f), 3.14)


def test_unicode_object_api(cpy: CPython):
    var str = "Hello, World!"

    var py_str = cpy.PyUnicode_DecodeUTF8(str)
    assert_true(py_str)

    var res = cpy.PyUnicode_AsUTF8AndSize(py_str)
    assert_equal(res, str)


def test_tuple_object_api(cpy: CPython):
    var n = cpy.PyLong_FromSsize_t(42)
    var t = cpy.PyTuple_New(1)
    assert_true(t)

    # PyTuple_SetItem steals a reference to the object
    cpy.Py_IncRef(n)
    assert_equal(cpy.PyTuple_SetItem(t, 0, n), 0)
    assert_equal(cpy.PyTuple_GetItem(t, 0), n)


def test_list_object_api(cpy: CPython):
    var n = cpy.PyLong_FromSsize_t(42)
    var l = cpy.PyList_New(1)
    assert_true(l)

    # PyList_SetItem steals a reference to the object
    cpy.Py_IncRef(n)
    assert_equal(cpy.PyList_SetItem(l, 0, n), 0)
    assert_equal(cpy.PyList_GetItem(l, 0), n)


def test_dictionary_object_api(cpy: CPython):
    var d = cpy.PyDict_New()
    var b = cpy.PyBool_FromLong(0)

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


def test_set_object_api(cpy: CPython):
    var s = cpy.PySet_New({})
    assert_true(s)

    var n = cpy.PyLong_FromSsize_t(42)
    assert_equal(cpy.PySet_Add(s, n), 0)


def test_module_object_api(cpy: CPython):
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


def test_slice_object_api(cpy: CPython):
    var n = cpy.PyLong_FromSsize_t(42)
    assert_true(cpy.PySlice_New(n, n, n))


def test_capsule_api(cpy: CPython):
    var o = PyObjectPtr()
    with assert_raises(contains="called with invalid PyCapsule object"):
        _ = cpy.PyCapsule_GetPointer(o, "some_name")

    var capsule_impl = UnsafePointer[UInt64].alloc(1)

    fn empty_dtor(capsule: PyObjectPtr):
        pass

    var capsule = cpy.PyCapsule_New(
        capsule_impl.bitcast[NoneType](), "some_name", empty_dtor
    )
    var capsule_pointer = cpy.PyCapsule_GetPointer(capsule, "some_name")
    assert_equal(capsule_impl.bitcast[NoneType](), capsule_pointer)

    with assert_raises(contains="called with incorrect name"):
        _ = cpy.PyCapsule_GetPointer(capsule, "some_other_name")

    capsule_impl.free()


def test_memory_management_api(cpy: CPython):
    var ptr = cpy.lib.call["PyObject_Malloc", UnsafePointer[NoneType]](64)
    assert_true(ptr)

    cpy.PyObject_Free(ptr)


def test_common_object_structure_api(cpy: CPython):
    var n = cpy.PyLong_FromSsize_t(42)
    assert_true(cpy.Py_Is(n, n))

    var dict_type = cpy.PyDict_Type()
    var d = cpy.PyDict_New()

    var d_type = cpy.Py_TYPE(d)
    assert_equal(
        PyObjectPtr(upcast_from=d_type),
        PyObjectPtr(upcast_from=dict_type),
    )


def test_with_cpython_very_high_level_api():
    var python = Python()
    ref cpython = python.cpython()
    test_very_high_level_api(cpython)


def test_with_cpython_reference_counting_api():
    var python = Python()
    ref cpython = python.cpython()
    test_reference_counting_api(cpython)


def test_with_cpython_exception_handling_api():
    var python = Python()
    ref cpython = python.cpython()
    test_exception_handling_api(cpython)


def test_with_cpython_threading_api():
    var python = Python()
    ref cpython = python.cpython()
    test_threading_api(cpython)


def test_with_cpython_importing_module_api():
    var python = Python()
    ref cpython = python.cpython()
    test_importing_module_api(cpython)


def test_with_cpython_object_protocol_api():
    var python = Python()
    ref cpython = python.cpython()
    test_object_protocol_api(cpython)


def test_with_cpython_call_protocol_api():
    var python = Python()
    ref cpython = python.cpython()
    test_call_protocol_api(cpython)


def test_with_cpython_number_protocol_api():
    var python = Python()
    ref cpython = python.cpython()
    test_number_protocol_api(cpython)


def test_with_cpython_iterator_protocol_api():
    var python = Python()
    ref cpython = python.cpython()
    test_iterator_protocol_api(cpython)


def test_with_cpython_type_object_api():
    var python = Python()
    ref cpython = python.cpython()
    test_type_object_api(cpython)


def test_with_cpython_integer_object_api():
    var python = Python()
    ref cpython = python.cpython()
    test_integer_object_api(cpython)


def test_with_cpython_boolean_object_api():
    var python = Python()
    ref cpython = python.cpython()
    test_boolean_object_api(cpython)


def test_with_cpython_floating_point_object_api():
    var python = Python()
    ref cpython = python.cpython()
    test_floating_point_object_api(cpython)


def test_with_cpython_unicode_object_api():
    var python = Python()
    ref cpython = python.cpython()
    test_unicode_object_api(cpython)


def test_with_cpython_tuple_object_api():
    var python = Python()
    ref cpython = python.cpython()
    test_tuple_object_api(cpython)


def test_with_cpython_list_object_api():
    var python = Python()
    ref cpython = python.cpython()
    test_list_object_api(cpython)


def test_with_cpython_dictionary_object_api():
    var python = Python()
    ref cpython = python.cpython()
    test_dictionary_object_api(cpython)


def test_with_cpython_set_object_api():
    var python = Python()
    ref cpython = python.cpython()
    test_set_object_api(cpython)


def test_with_cpython_module_object_api():
    var python = Python()
    ref cpython = python.cpython()
    test_module_object_api(cpython)


def test_with_cpython_slice_object_api():
    var python = Python()
    ref cpython = python.cpython()
    test_slice_object_api(cpython)


def test_with_cpython_capsule_api():
    var python = Python()
    ref cpython = python.cpython()
    test_capsule_api(cpython)


def test_with_cpython_memory_management_api():
    var python = Python()
    ref cpython = python.cpython()
    test_memory_management_api(cpython)


def test_with_cpython_common_object_structure_api():
    var python = Python()
    ref cpython = python.cpython()
    test_common_object_structure_api(cpython)


def main():
    var suite = TestSuite()

    suite.test[test_with_cpython_very_high_level_api]()
    suite.test[test_with_cpython_reference_counting_api]()
    suite.test[test_with_cpython_exception_handling_api]()
    suite.test[test_with_cpython_threading_api]()
    suite.test[test_with_cpython_importing_module_api]()
    suite.test[test_with_cpython_object_protocol_api]()
    suite.test[test_with_cpython_call_protocol_api]()
    suite.test[test_with_cpython_number_protocol_api]()
    suite.test[test_with_cpython_iterator_protocol_api]()
    suite.test[test_with_cpython_type_object_api]()
    suite.test[test_with_cpython_integer_object_api]()
    suite.test[test_with_cpython_boolean_object_api]()
    suite.test[test_with_cpython_floating_point_object_api]()
    suite.test[test_with_cpython_unicode_object_api]()
    suite.test[test_with_cpython_tuple_object_api]()
    suite.test[test_with_cpython_list_object_api]()
    suite.test[test_with_cpython_dictionary_object_api]()
    suite.test[test_with_cpython_set_object_api]()
    suite.test[test_with_cpython_module_object_api]()
    suite.test[test_with_cpython_slice_object_api]()
    suite.test[test_with_cpython_capsule_api]()
    suite.test[test_with_cpython_memory_management_api]()
    suite.test[test_with_cpython_common_object_structure_api]()

    suite^.run()
