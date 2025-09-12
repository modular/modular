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

"""Provides logging functionality with different severity levels.

This module implements a simple logging system with configurable severity
levels: `NOTSET`, `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`. The
logging level can be set via the LOGGING_LEVEL environment variable.

The main components are:

- `Level`: An enum-like struct defining the available logging levels
- `Logger`: A struct that handles logging messages with different severity levels

Example:

```mojo
from logger import Logger

var logger = Logger()  # Uses default level from LOGGING_LEVEL env var
logger.info("Starting process")
logger.debug("Debug information")
logger.error("An error occurred")
```

The logger can be configured to write to different file descriptors (default
stdout). Messages below the configured level will be silently ignored.
"""

import sys
from os import abort
from sys.param_env import env_get_string
from io.write import _WriteBufferStack
from builtin._location import __call_location, _SourceLocation

# ===-----------------------------------------------------------------------===#
# DEFAULT_LEVEL
# ===-----------------------------------------------------------------------===#

alias DEFAULT_LEVEL = Level._from_str(
    env_get_string["LOGGING_LEVEL", "NOTSET"]()
)

# ===-----------------------------------------------------------------------===#
# Level
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct Level(
    EqualityComparable,
    Identifiable,
    ImplicitlyCopyable,
    Movable,
    Stringable,
    Writable,
):
    """Represents logging severity levels.

    Defines the available logging levels in ascending order of severity.
    """

    var _value: Int

    alias NOTSET = Self(0)
    """Lowest level, used when no level is set."""

    alias TRACE = Self(10)
    """Repetitive trace information, Indicates repeated execution or IO-coupled
    activity, typically only of interest when diagnosing hangs or ensuring a
    section of code is executing."""

    alias DEBUG = Self(20)
    """Detailed information, typically of interest only when diagnosing problems."""

    alias INFO = Self(30)
    """Confirmation that things are working as expected."""

    alias WARNING = Self(40)
    """Indication that something unexpected happened, or may happen in the near future."""

    alias ERROR = Self(50)
    """Due to a more serious problem, the software has not been able to perform some function."""

    alias CRITICAL = Self(60)
    """A serious error indicating that the program itself may be unable to continue running."""

    fn __eq__(self, other: Self) -> Bool:
        """Returns True if this level equals the other level.

        Args:
            other: The level to compare with.

        Returns:
            Bool: True if the levels are equal, False otherwise.
        """
        return self._value == other._value

    fn __gt__(self, other: Self) -> Bool:
        """Returns True if this level is greater than the other level.

        Args:
            other: The level to compare with.

        Returns:
            Bool: True if this level is greater than the other level, False otherwise.
        """
        return self._value > other._value

    fn __ge__(self, other: Self) -> Bool:
        """Returns True if this level is greater than or equal to the other level.

        Args:
            other: The level to compare with.

        Returns:
            Bool: True if this level is greater than or equal to the other level, False otherwise.
        """
        return self._value >= other._value

    fn __lt__(self, other: Self) -> Bool:
        """Returns True if this level is less than the other level.

        Args:
            other: The level to compare with.

        Returns:
            Bool: True if this level is less than the other level, False otherwise.
        """
        return self._value < other._value

    fn __le__(self, other: Self) -> Bool:
        """Returns True if this level is less than or equal to the other level.

        Args:
            other: The level to compare with.

        Returns:
            Bool: True if this level is less than or equal to the other level, False otherwise.
        """
        return self._value <= other._value

    fn __is__(self, other: Self) -> Bool:
        """Returns True if this level is identical to the other level.

        Args:
            other: The level to compare with.

        Returns:
            Bool: True if this level is identical to the other level, False otherwise.
        """
        return self == other

    @staticmethod
    fn _from_str(name: StringSlice) -> Self:
        """Converts a string level name to a Level value.

        Args:
            name: The level name as a string (case insensitive).

        Returns:
            The corresponding Level value, or NOTSET if not recognized.
        """
        var lname = name.lower()
        if lname == "notset":
            return Self.NOTSET
        if lname == "trace":
            return Self.TRACE
        if lname == "debug":
            return Self.DEBUG
        if lname == "info":
            return Self.INFO
        if lname == "warning":
            return Self.WARNING
        if lname == "error":
            return Self.ERROR
        if lname == "critical":
            return Self.CRITICAL
        return Self.NOTSET

    fn write_to(self, mut writer: Some[Writer]):
        """Writes the string representation of this level to a writer.

        Args:
            writer: The writer to write to.
        """
        if self is Self.NOTSET:
            writer.write("NOTSET")
        elif self is Self.TRACE:
            writer.write("TRACE")
        elif self is Self.DEBUG:
            writer.write("DEBUG")
        elif self is Self.INFO:
            writer.write("INFO")
        elif self is Self.WARNING:
            writer.write("WARNING")
        elif self is Self.ERROR:
            writer.write("ERROR")
        elif self is Self.CRITICAL:
            writer.write("CRITICAL")

    @no_inline
    fn __str__(self) -> String:
        """Returns the string representation of this level.

        Returns:
            String: A human-readable string representation of the level
                (e.g., "DEBUG", "INFO").
        """
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        """Returns the detailed string representation of this level.

        Returns:
            String: A string representation including the type name and level
                value (e.g., "Level.DEBUG").
        """
        return String("Level.", self)


# ===-----------------------------------------------------------------------===#
# Logger
# ===-----------------------------------------------------------------------===#


struct Logger[level: Level = DEFAULT_LEVEL](ImplicitlyCopyable):
    """A logger that outputs messages at or above a specified severity level.

    Parameters:
        level: The minimum severity level for messages to be logged.
    """

    var _fd: FileDescriptor
    var _prefix: String
    var _source_location: Bool

    fn __init__(
        out self,
        fd: FileDescriptor = sys.stdout,
        *,
        prefix: String = "",
        source_location: Bool = False,
    ):
        """Initializes a new Logger.

        Args:
            fd: The file descriptor to write log messages to (defaults to stdout).
            prefix: The prefix to prepend to each log message (defaults to an
                empty string).
            source_location: Whether to include the source location in the log
                message (defaults to False).
        """
        self._fd = fd
        self._prefix = prefix
        self._source_location = source_location

    @always_inline
    @staticmethod
    fn _is_disabled[target_level: Level]() -> Bool:
        """Returns True if logging at the target level is disabled.

        Parameters:
            target_level: The level to check if disabled.

        Returns:
            True if logging at the target level is disabled, False otherwise.
        """

        @parameter
        if level == Level.NOTSET:
            return True
        return level > target_level

    @always_inline
    fn trace[
        *Ts: Writable
    ](self, *values: *Ts, sep: StaticString = " ", end: StaticString = "\n"):
        """Logs a trace message.

        Parameters:
            Ts: The types of values to log.

        Args:
            values: The values to log.
            sep: The separator to use between values (defaults to a space).
            end: The string to append to the end of the message
                (defaults to a newline).
        """
        alias target_level = Level.TRACE

        @parameter
        if not Self._is_disabled[target_level]():
            self._write_out[target_level](
                values, sep=sep, end=end, location=__call_location()
            )

    @always_inline
    fn debug[
        *Ts: Writable
    ](self, *values: *Ts, sep: StaticString = " ", end: StaticString = "\n"):
        """Logs a debug message.

        Parameters:
            Ts: The types of values to log.

        Args:
            values: The values to log.
            sep: The separator to use between values (defaults to a space).
            end: The string to append to the end of the message
                (defaults to a newline).
        """
        alias target_level = Level.DEBUG

        @parameter
        if not Self._is_disabled[target_level]():
            self._write_out[target_level](
                values, sep=sep, end=end, location=__call_location()
            )

    @always_inline
    fn info[
        *Ts: Writable
    ](self, *values: *Ts, sep: StaticString = " ", end: StaticString = "\n"):
        """Logs an info message.

        Parameters:
            Ts: The types of values to log.

        Args:
            values: The values to log.
            sep: The separator to use between values (defaults to a space).
            end: The string to append to the end of the message
                (defaults to a newline).
        """
        alias target_level = Level.INFO

        @parameter
        if not Self._is_disabled[target_level]():
            self._write_out[target_level](
                values, sep=sep, end=end, location=__call_location()
            )

    @always_inline
    fn warning[
        *Ts: Writable
    ](self, *values: *Ts, sep: StaticString = " ", end: StaticString = "\n"):
        """Logs a warning message.

        Parameters:
            Ts: The types of values to log.

        Args:
            values: The values to log.
            sep: The separator to use between values (defaults to a space).
            end: The string to append to the end of the message
                (defaults to a newline).
        """
        alias target_level = Level.WARNING

        @parameter
        if not Self._is_disabled[target_level]():
            self._write_out[target_level](
                values, sep=sep, end=end, location=__call_location()
            )

    @always_inline
    fn error[
        *Ts: Writable
    ](self, *values: *Ts, sep: StaticString = " ", end: StaticString = "\n"):
        """Logs an error message.

        Parameters:
            Ts: The types of values to log.

        Args:
            values: The values to log.
            sep: The separator to use between values (defaults to a space).
            end: The string to append to the end of the message
                (defaults to a newline).
        """
        alias target_level = Level.ERROR

        @parameter
        if not Self._is_disabled[target_level]():
            self._write_out[target_level](
                values, sep=sep, end=end, location=__call_location()
            )

    @always_inline
    fn critical[
        *Ts: Writable
    ](self, *values: *Ts, sep: StaticString = " ", end: StaticString = "\n"):
        """Logs a critical message and aborts execution.

        Parameters:
            Ts: The types of values to log.

        Args:
            values: The values to log.
            sep: The separator to use between values (defaults to a space).
            end: The string to append to the end of the message
                (defaults to a newline).

        """
        alias target_level = Level.CRITICAL

        @parameter
        if not Self._is_disabled[target_level]():
            self._write_out[target_level](
                values, sep=sep, end=end, location=__call_location()
            )

        abort()

    fn _write_out[
        level: Level
    ](
        self,
        values: VariadicPack[element_trait=Writable],
        *,
        location: _SourceLocation,
        sep: StaticString = " ",
        end: StaticString = "\n",
    ):
        var file = self._fd
        var buffer = _WriteBufferStack(file)

        if self._prefix:
            buffer.write(self._prefix)
        else:
            buffer.write(level, "::: ")

        if self._source_location:
            buffer.write("[", location, "] ")

        alias length = values.__len__()

        @parameter
        for i in range(length):
            values[i].write_to(buffer)

            @parameter
            if i < length - 1:
                buffer.write(sep)

        buffer.write(end)
        buffer.flush()
