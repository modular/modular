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
"""Implements functions that return compile-time information.
"""
from .param_env import env_get_int, env_get_string

# ===----------------------------------------------------------------------=== #
# OptimizationLevel
# ===----------------------------------------------------------------------=== #


@value
struct _OptimizationLevel(Intable):
    """Represents the optimization level used during compilation.

    The optimization level is determined by the __OPTIMIZATION_LEVEL environment
    variable, with a default value of 4 if not specified.

    Attributes:
        level: The integer value of the optimization level.
    """

    alias level = env_get_int["__OPTIMIZATION_LEVEL", 4]()

    fn __copyinit__(out self, existing: Self):
        """Copy initialization.

        Args:
            existing: The existing optimization level.
        """
        return

    fn __moveinit__(out self, owned existing: Self):
        """Move initialization.

        Args:
            existing: The existing optimization level.
        """
        return

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        """Writes the optimization level to a writer."""
        writer.write(Self.level)

    @no_inline
    fn __str__(self) -> String:
        """Returns the string representation of the optimization level.

        Returns:
            A string containing the optimization level value.
        """
        return String.write(self)

    fn __int__(self) -> Int:
        """Returns the integer value of the optimization level.

        Returns:
            The optimization level as an integer.
        """
        return Self.level


alias OptimizationLevel = _OptimizationLevel()
"""Represents the optimization level used during compilation."""

# ===----------------------------------------------------------------------=== #
# DebugLevel
# ===----------------------------------------------------------------------=== #


@value
struct _DebugLevel:
    """Represents the debug level used during compilation.

    The debug level is determined by the __DEBUG_LEVEL environment variable,
    with a default value of "none" if not specified.

    Attributes:
        level: The string value of the debug level.
    """

    alias level = env_get_string["__DEBUG_LEVEL", "none"]()

    fn __copyinit__(out self, existing: Self):
        """Copy initialization.

        Args:
            existing: The existing optimization level.
        """
        return

    fn __moveinit__(out self, owned existing: Self):
        """Move initialization.

        Args:
            existing: The existing optimization level.
        """
        return

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        """Writes the optimization level to a writer."""
        writer.write(Self.level)

    @no_inline
    fn __str__(self) -> String:
        """Returns the string representation of the debug level.

        Returns:
            The debug level as a string.
        """
        return Self.level


alias DebugLevel = _DebugLevel()
"""Represents the debug level used during compilation."""
