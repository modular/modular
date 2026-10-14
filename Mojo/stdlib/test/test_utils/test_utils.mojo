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
"""Provides general test utility functions for the Mojo standard library tests."""

from std.python import ConvertibleToPython, PythonObject
from std.testing import assert_equal, assert_true


def check_write_to(
    value: Some[Writable], *, expected: String, is_repr: Bool
) raises:
    """Check that the write_to or write_repr_to of the value is equal to the expected string.

    Args:
        value: The Writable value to check.
        expected: The expected string.
        is_repr: Whether to check the repr version of the value.

    Raises:
        Error: if the written string does not equal `expected`.
    """

    var string = String()
    if is_repr:
        value.write_repr_to(string)
    else:
        value.write_to(string)
    assert_equal(string, expected)


def check_write_to(
    value: Some[Writable], *, contains: String, is_repr: Bool
) raises:
    """Check that the write_to or write_repr_to of the value contains the expected string.

    Args:
        value: The Writable value to check.
        contains: The string to check for in the output.
        is_repr: Whether to check the repr version of the value.

    Raises:
        Error: if the written string does not contain `contains`.
    """
    var string = String()
    if is_repr:
        value.write_repr_to(string)
    else:
        value.write_to(string)
    assert_true(contains in string)


def check_convertible_to_python(
    var value: Some[ConvertibleToPython], expected: String
) raises:
    """Check that the value can be converted to a Python object.
    Tests for equality with the expected string.

    Args:
        value: The value to check.
        expected: The expected string representation of the Python object.

    Raises:
        Error: if the converted Python object does not equal `expected`.
    """
    var py_obj: PythonObject
    try:
        py_obj = value^.to_python_object()
    except:
        raise Error("Failed to convert value to PythonObject")

    assert_equal(String(py_obj), expected)


def check_python_object(var value: PythonObject, expected: String) raises:
    """Check that the value can be implicitly converted to a Python object.
    Tests for equality with the expected string.

    Args:
        value: The value to check.
        expected: The expected string representation of the Python object.

    Raises:
        Error: if the converted Python object does not equal `expected`.
    """
    assert_equal(String(value), expected)
