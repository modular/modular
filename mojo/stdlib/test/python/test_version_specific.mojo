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
# Tests functionality that depends on the python version
# RUN: %mojo %s

from python.bindings import _get_type_name
from python import PythonObject, Python
from python._cpython import CPython
from testing import assert_equal


def _type_name_helper(obj: PythonObject, cpython: CPython) -> String:
    """Retrieve the name of the type of the object using the PyType_GetName method.
    """

    var actual_type = cpython.Py_TYPE(obj.unsafe_as_py_object_ptr())
    var actual_type_name = PythonObject(
        from_owned_ptr=cpython.PyType_GetName(actual_type)
    )
    return String(actual_type_name)


def test_get_type_name() -> None:
    """Test the PyType_GetName function, it has different implementations for Python pre and post 3.11.
    """
    var cpython = Python().cpython()

    var str_obj = PythonObject("hello")
    assert_equal(_type_name_helper(str_obj, cpython).__str__(), "str")
    var int_obj = PythonObject(32)
    assert_equal(_type_name_helper(int_obj, cpython).__str__(), "int")


def main():
    test_get_type_name()
