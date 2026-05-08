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
"""Implements PythonException for representing Python exceptions.

This module provides a type for representing Python exceptions that can be
returned from Mojo functions exposed to Python, enabling safer error handling
without relying on NULL pointer checks.

You can import these APIs from the `python` package. For example:

```mojo
from std.python import PythonException
```
"""

from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr


@fieldwise_init
struct PythonException(ImplicitlyCopyable, Movable):
    """Represents a Python exception raised during execution.

    This type is used to safely represent Python exceptions that occur when
    calling Mojo functions from Python. Instead of returning NULL (which can
    cause segfaults), functions can return a `Variant[PythonException, PythonObject]`
    that explicitly indicates whether an exception occurred.

    The exception object wraps the Python exception type and value, allowing
    proper error propagation across the Mojo/Python boundary.

    Example:
        ```mojo
        from std.python import PythonException, PythonObject
        from std.utils import Variant

        def my_function() raises -> Variant[PythonException, PythonObject]:
            try:
                # Some operation that might fail
                return PythonObject(42)
            except e:
                # Return exception instead of NULL
                return PythonException(String(e))
        ```
    """

    var _exception_obj: PythonObject
    """The underlying Python exception object."""

    def __init__(out self, error_message: String) raises:
        """Initialize a PythonException from an error message.

        This creates a standard Python Exception with the given message.

        Args:
            error_message: The error message for the exception.

        Raises:
            If the Python builtins module cannot be imported.
        """
        # Create a standard Exception using Python's builtins
        var builtins = Python.import_module("builtins")
        var exc_type = builtins["Exception"]
        self._exception_obj = exc_type(error_message)

    def __init__(out self, error_type: PythonObject, error_value: PythonObject):
        """Initialize a PythonException from an error type and value.

        Args:
            error_type: The Python exception type (e.g., ValueError, TypeError).
            error_value: The Python exception value/instance.
        """
        self._exception_obj = error_value

    def __init__(out self, *, from_cpython_error: PyObjectPtr):
        """Initialize a PythonException from a CPython error.

        This constructor takes ownership of the Python exception object
        retrieved via PyErr_GetRaisedException().

        Args:
            from_cpython_error: The exception object from CPython (owns reference).
        """
        self._exception_obj = PythonObject(from_owned=from_cpython_error)

    def get_exception_object(self) -> PythonObject:
        """Get the underlying Python exception object.

        Returns:
            The Python exception object wrapped by this exception.
        """
        return self._exception_obj

    def __del__(deinit self):
        """Destroy the exception.

        The underlying Python exception object will have its refcount decremented.
        """
        # PythonObject's __del__ handles the refcount
        pass
