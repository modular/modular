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
"""Provides functions for working with environment variables.

You can import these APIs from the `os` package. For example:

```mojo
from os import setenv
```
"""


from sys import CompilationTarget, external_call
from sys.ffi import c_int


fn setenv(var name: String, var value: String, overwrite: Bool = True) -> Bool:
    """Changes or adds an environment variable.

    Constraints:
      The function only works on macOS or Linux and returns False otherwise.

    Args:
      name: The name of the environment variable.
      value: The value of the environment variable.
      overwrite: If an environment variable with the given name already exists,
        its value is not changed unless `overwrite` is True.

    Returns:
      False if the name is empty or contains an `=` character. In any other
      case, True is returned.
    """

    @parameter
    if CompilationTarget.is_windows():
        return False

    var status = external_call["setenv", Int32](
        name.unsafe_cstr_ptr(),
        value.unsafe_cstr_ptr(),
        Int32(1 if overwrite else 0),
    )
    return status == 0


fn unsetenv(var name: String) -> Bool:
    """Unsets an environment variable.

    Args:
        name: The name of the environment variable.

    Returns:
        True if unsetting the variable succeeded. Otherwise, False is returned.
    """
    constrained[
        not CompilationTarget.is_windows(),
        "operating system must be Linux or macOS",
    ]()

    return external_call["unsetenv", c_int](name.unsafe_cstr_ptr()) == 0


fn getenv(var name: String, default: String = "") -> String:
    """Returns the value of the given environment variable.

    Constraints:
      The function only works on macOS or Linux and returns an empty string
      otherwise.

    Args:
      name: The name of the environment variable.
      default: The default value to return if the environment variable
        doesn't exist.

    Returns:
      The value of the environment variable.
    """

    @parameter
    if CompilationTarget.is_windows():
        return default

    var ptr = external_call["getenv", UnsafePointer[UInt8]](
        name.unsafe_cstr_ptr()
    )
    if not ptr:
        return default
    return String(unsafe_from_utf8_ptr=ptr)
