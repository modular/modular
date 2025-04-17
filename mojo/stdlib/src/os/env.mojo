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


from sys import external_call, os_is_linux, os_is_macos, os_is_windows
from sys.ffi import c_int, c_char

from memory import UnsafePointer


fn setenv(
    name: StringSlice, value: StringSlice, overwrite: Bool = True
) -> Bool:
    """Changes or adds an environment variable.

    Args:
      name: The name of the environment variable.
      value: The value of the environment variable.
      overwrite: If an environment variable with the given name already exists,
          its value is not changed unless `overwrite` is True.

    Returns:
        False if the name is empty or contains an `=` character. In any other
        case, True is returned.

    Constraints:
        The function only works on macOS or Linux and returns False otherwise.
    """

    @parameter
    if not (os_is_linux() or os_is_macos()):
        return False

    var status = external_call["setenv", Int32](
        name.unsafe_ptr().bitcast[c_char](),
        value.unsafe_ptr().bitcast[c_char](),
        Int32(overwrite),
    )
    return status == 0


fn unsetenv(name: StringSlice) -> Bool:
    """Unsets an environment variable.

    Args:
        name: The name of the environment variable.

    Returns:
        True if unsetting the variable succeeded. Otherwise, False is returned.
    """
    constrained[
        not os_is_windows(), "operating system must be Linux or macOS"
    ]()

    return (
        external_call["unsetenv", c_int](name.unsafe_ptr().bitcast[c_char]())
        == 0
    )


fn getenv[
    O: ImmutableOrigin = StaticConstantOrigin
](
    name: StringSlice,
    default: StringSlice[O] = rebind[StringSlice[O]](StringSlice("")),
) -> String:
    """Returns the value of the given environment variable.

    Parameters:
        O: The origin of the default `StringSlice`.

    Args:
        name: The name of the environment variable.
        default: The default value to return if the environment variable
            doesn't exist.

    Returns:
        The value of the environment variable.

    Constraints:
        The function only works on macOS or Linux and returns an empty string
        otherwise.
    """

    @parameter
    if not (os_is_linux() or os_is_macos()):
        return String(default)

    var ptr = external_call["getenv", UnsafePointer[UInt8]](
        name.unsafe_ptr().bitcast[c_char]()
    )
    if not ptr:
        return String(default)
    return String(StringSlice[ptr.origin](unsafe_from_utf8_ptr=ptr))
