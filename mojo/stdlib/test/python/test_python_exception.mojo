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
"""Tests for PythonException and non-NULL PythonObject enforcement.

These tests verify that PythonObject handles NULL pointers safely and that
PythonException can be used to return errors from Python bindings without
causing segfaults.
"""

from std.python import Python, PythonException, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import PyFunctionResult
from std.testing import assert_equal, assert_true, assert_false, TestSuite
from std.utils import Variant


def test_python_exception_creation() raises:
    """Test that PythonException can be created from an error message."""
    var exc = PythonException("Test error message")
    var exc_obj = exc.get_exception_object()
    
    # Verify the exception object is valid
    assert_true(Bool(exc_obj))


def test_python_function_result_variant() raises:
    """Test that PyFunctionResult can hold either exception or object."""
    # Test with PythonObject
    var obj_result: PyFunctionResult = PythonObject(42)
    assert_true(obj_result.isa[PythonObject]())
    assert_false(obj_result.isa[PythonException]())
    
    # Test with PythonException
    var exc_result: PyFunctionResult = PythonException("Error")
    assert_true(exc_result.isa[PythonException]())
    assert_false(exc_result.isa[PythonObject]())


def test_python_object_null_safety() raises:
    """Test that PythonObject handles NULL pointers safely."""
    # Create a PythonObject with a NULL pointer
    var null_ptr = PyObjectPtr()
    assert_false(Bool(null_ptr))  # Verify it's NULL
    
    # Creating from NULL should be handled gracefully
    # The object should be safely destructible even with NULL pointer
    var obj = PythonObject(from_owned=null_ptr)
    
    # Destructor should not crash
    _ = obj^  # Destroys the object


def test_python_object_from_borrowed_null() raises:
    """Test that from_borrowed handles NULL gracefully."""
    var null_ptr = PyObjectPtr()
    
    # This should not crash - the NULL check should prevent issues
    var obj = PythonObject(from_borrowed=null_ptr)
    
    # Destructor should handle NULL safely
    _ = obj^


# ===-------------------------------------------------------------------===#
# main
# ===-------------------------------------------------------------------===#
def main() raises:
    TestSuite.discover_tests[__functions_in_module__()]().run()
