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
from testing import assert_equal, assert_true, TestSuite


fn test_import(mut python: Python) raises:
    var my_module: PythonObject = Python.import_module("my_module")
    var py_string = my_module.my_function("Hello")
    var str = String(python.as_string_slice(py_string))
    assert_equal(str, "Formatting the string from Lit with Python: Hello")


fn test_list(mut python: Python) raises:
    var b: PythonObject = Python.import_module("builtins")
    var my_list: PythonObject = [1, 2.34, "False"]
    var py_string = String(my_list)
    assert_equal(py_string, "[1, 2.34, 'False']")


fn test_tuple(mut python: Python) raises:
    var b: PythonObject = Python.import_module("builtins")
    var my_tuple = Python.tuple(1, 2.34, "False")
    var py_string = String(my_tuple)
    assert_equal(py_string, "(1, 2.34, 'False')")


fn test_call_ownership(mut python: Python) raises:
    var obj: PythonObject = [1, "5"]
    var py_string = String(obj)
    var string = python.as_string_slice(py_string)
    assert_true(string == "[1, '5']")


fn test_getitem_ownership(mut python: Python) raises:
    var obj: PythonObject = [1, "5"]
    var py_string = String(obj[1])
    var string = python.as_string_slice(py_string)
    assert_true(string == "5")


fn test_getattr_ownership(mut python: Python) raises:
    var my_module: PythonObject = Python.import_module("my_module")
    var obj = my_module.Foo(4)
    var py_string = String(obj.bar)
    var string = python.as_string_slice(py_string)
    assert_true(string == "4")


def test_with_python_list():
    var python = Python()
    test_list(python)


def test_with_python_tuple():
    var python = Python()
    test_tuple(python)


def test_with_python_call_ownership():
    var python = Python()
    test_call_ownership(python)


def test_with_python_getitem_ownership():
    var python = Python()
    test_getitem_ownership(python)


def test_with_python_getattr_ownership():
    var python = Python()
    test_getattr_ownership(python)


def test_with_python_import():
    var python = Python()
    test_import(python)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
