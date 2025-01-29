# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

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
    var _value: Int

    alias NOTSET = Self(0)
    alias DEBUG = Self(10)
    alias INFO = Self(20)
    alias WARNING = Self(30)
    alias ERROR = Self(40)
    alias CRITICAL = Self(50)

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __gt__(self, other: Self) -> Bool:
        return self._value > other._value

    fn __ge__(self, other: Self) -> Bool:
        return self._value >= other._value

    fn __lt__(self, other: Self) -> Bool:
        return self._value < other._value

    fn __le__(self, other: Self) -> Bool:
        return self._value <= other._value

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @staticmethod
    fn _from_str(name: String) -> Self:
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
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return String("Level.", self)


# ===-----------------------------------------------------------------------===#
# Logger
# ===-----------------------------------------------------------------------===#


struct Logger[level: Level = DEFAULT_LEVEL]:
    var _fd: FileDescriptor

    @implicit
    fn __init__(out self, fd: FileDescriptor = sys.stdout):
        self._fd = fd

    @always_inline
    @staticmethod
    fn _is_disabled[target_level: Level]() -> Bool:
        if level == Level.NOTSET:
            return False
        return level > target_level

    fn debug[*Ts: Writable](self, *values: *Ts):
        alias target_level = Level.DEBUG

        @parameter
        if Self._is_disabled[target_level]():
            return

        var writer = self._fd

        writer.write(String(target_level), "::: ")
        write_args(writer, values, sep=" ", end="\n")

    fn info[*Ts: Writable](self, *values: *Ts):
        alias target_level = Level.INFO

        @parameter
        if Self._is_disabled[target_level]():
            return

        var writer = self._fd

        writer.write(String(target_level), "::: ")
        write_args(writer, values, sep=" ", end="\n")

    fn warning[*Ts: Writable](self, *values: *Ts):
        alias target_level = Level.WARNING

        @parameter
        if Self._is_disabled[target_level]():
            return

        var writer = self._fd

        writer.write(String(target_level), "::: ")
        write_args(writer, values, sep=" ", end="\n")

    fn error[*Ts: Writable](self, *values: *Ts):
        alias target_level = Level.CRITICAL

        @parameter
        if Self._is_disabled[target_level]():
            return

        var writer = self._fd

        writer.write(String(target_level), "::: ")
        write_args(writer, values, sep=" ", end="\n")

    fn critical[*Ts: Writable](self, *values: *Ts):
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
