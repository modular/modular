# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Provides logging functionality with different severity levels.

This module implements a simple logging system with configurable severity levels:
NOTSET, DEBUG, INFO, WARNING, ERROR, and CRITICAL. The logging level can be set
via the LOGGING_LEVEL environment variable.

The main components are:

- Level: An enum-like struct defining the available logging levels
- Logger: A struct that handles logging messages with different severity levels

Example:
    ```mojo
    from logger import Logger

    var logger = Logger()  # Uses default level from LOGGING_LEVEL env var
    logger.info("Starting process")
    logger.debug("Debug information")
    logger.error("An error occurred")
    ```

The logger can be configured to write to different file descriptors (default stdout).
Messages below the configured level will be silently ignored.
"""

import sys
from os import abort
from sys.param_env import env_get_string

from utils import write_args

# ===-----------------------------------------------------------------------===#
# DEFAULT_LEVEL
# ===-----------------------------------------------------------------------===#

alias DEFAULT_LEVEL = Level._from_str(
    env_get_string["LOGGING_LEVEL", "NOTSET"]()
)

# ===-----------------------------------------------------------------------===#
# Level
# ===-----------------------------------------------------------------------===#


@value
struct Level:
    """Represents logging severity levels.

    Defines the available logging levels in ascending order of severity.
    """

    var _value: Int

    alias NOTSET = Self(0)  # Lowest level, used when no level is set
    alias DEBUG = Self(10)  # Detailed information for debugging
    alias INFO = Self(20)  # General information about program execution
    alias WARNING = Self(30)  # Indicates a potential problem
    alias ERROR = Self(40)  # Error that prevents normal program execution
    alias CRITICAL = Self(
        50
    )  # Critical error that may lead to program termination

    fn __eq__(self, other: Self) -> Bool:
        """Returns True if this level equals the other level.

        Args:
            other: The level to compare with.
        """
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        """Returns True if this level does not equal the other level.

        Args:
            other: The level to compare with.
        """
        return not (self == other)

    fn __gt__(self, other: Self) -> Bool:
        """Returns True if this level is greater than the other level.

        Args:
            other: The level to compare with.
        """
        return self._value > other._value

    fn __ge__(self, other: Self) -> Bool:
        """Returns True if this level is greater than or equal to the other level.

        Args:
            other: The level to compare with.
        """
        return self._value >= other._value

    fn __lt__(self, other: Self) -> Bool:
        """Returns True if this level is less than the other level.

        Args:
            other: The level to compare with.
        """
        return self._value < other._value

    fn __le__(self, other: Self) -> Bool:
        """Returns True if this level is less than or equal to the other level.

        Args:
            other: The level to compare with.
        """
        return self._value <= other._value

    fn __is__(self, other: Self) -> Bool:
        """Returns True if this level is identical to the other level.

        Args:
            other: The level to compare with.
        """
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        """Returns True if this level is not identical to the other level.

        Args:
            other: The level to compare with.
        """
        return self != other

    @staticmethod
    fn _from_str(name: String) -> Self:
        """Converts a string level name to a Level value.

        Args:
            name: The level name as a string (case insensitive).

        Returns:
            The corresponding Level value, or NOTSET if not recognized.
        """
        var lname = name.lower()
        if lname == "notset":
            return Self.NOTSET
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

    fn write_to[W: Writer](self, mut writer: W):
        """Writes the string representation of this level to a writer.

        Args:
            writer: The writer to write to.
        """
        if self is Self.NOTSET:
            writer.write("NOTSET")
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
        """Returns the string representation of this level."""
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        """Returns the detailed string representation of this level."""
        return String("Level.", self)


# ===-----------------------------------------------------------------------===#
# Logger
# ===-----------------------------------------------------------------------===#


struct Logger[level: Level = DEFAULT_LEVEL]:
    """A logger that outputs messages at or above a specified severity level.

    Parameters:
        level: The minimum severity level for messages to be logged.
    """

    var _fd: FileDescriptor

    @implicit
    fn __init__(out self, fd: FileDescriptor = sys.stdout):
        """Initializes a new Logger.

        Args:
            fd: The file descriptor to write log messages to (defaults to stdout).
        """
        self._fd = fd

    @always_inline
    @staticmethod
    fn _is_disabled[target_level: Level]() -> Bool:
        """Returns True if logging at the target level is disabled.

        Parameters:
            target_level: The level to check if disabled.
        """
        if level == Level.NOTSET:
            return False
        return level > target_level

    fn debug[*Ts: Writable](self, *values: *Ts):
        """Logs a debug message.

        Parameters:
            Ts: The types of values to log.
        Args:
            *values: The values to log.
        """
        alias target_level = Level.DEBUG

        @parameter
        if Self._is_disabled[target_level]():
            return

        var writer = self._fd

        writer.write(String(target_level), "::: ")
        write_args(writer, values, sep=" ", end="\n")

    fn info[*Ts: Writable](self, *values: *Ts):
        """Logs an info message.

        Parameters:
            Ts: The types of values to log.
        Args:
            *values: The values to log.
        """
        alias target_level = Level.INFO

        @parameter
        if Self._is_disabled[target_level]():
            return

        var writer = self._fd

        writer.write(String(target_level), "::: ")
        write_args(writer, values, sep=" ", end="\n")

    fn warning[*Ts: Writable](self, *values: *Ts):
        """Logs a warning message.

        Parameters:
            Ts: The types of values to log.
        Args:
            *values: The values to log.
        """
        alias target_level = Level.WARNING

        @parameter
        if Self._is_disabled[target_level]():
            return

        var writer = self._fd

        writer.write(String(target_level), "::: ")
        write_args(writer, values, sep=" ", end="\n")

    fn error[*Ts: Writable](self, *values: *Ts):
        """Logs an error message.

        Parameters:
            Ts: The types of values to log.
        Args:
            *values: The values to log.
        """
        alias target_level = Level.ERROR

        @parameter
        if Self._is_disabled[target_level]():
            return

        var writer = self._fd

        writer.write(String(target_level), "::: ")
        write_args(writer, values, sep=" ", end="\n")

    fn critical[*Ts: Writable](self, *values: *Ts):
        """Logs a critical message and aborts execution.

        Parameters:
            Ts: The types of values to log.
        Args:
            *values: The values to log.
        """
        alias target_level = Level.CRITICAL

        @parameter
        if Self._is_disabled[target_level]():
            return

        var writer = self._fd

        writer.write(String(target_level), "::: ")
        write_args(writer, values, sep=" ", end="\n")

        abort()

    # Ideally we can remove the duplication and have something like, but
    # forwarding variadics does not work with mojo atm so we are forced to
    # copy/paste code :(
    # fn _log[
    #     target_level: Level, *Ts: Writable
    # ](self, *values: VariadicPack[_, Writable, Ts]):
    #     @parameter
    #     fn get_length() -> Int:
    #         return len(values)

    #     alias variadic_length = get_length()

    #     var writer = sys.stdout

    #     writer.write(target_level, "::: ")

    #     @parameter
    #     for i in range(variadic_length):
    #         var elem = values[i]
    #         elem.write(writer)

    #         @parameter
    #         if i < variadic_length - 1:
    #             writer.write(" ")
