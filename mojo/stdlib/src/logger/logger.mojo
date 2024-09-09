# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import sys
from os import abort
from sys.param_env import env_get_string

from utils import Formatter

# ===----------------------------------------------------------------------===#
# DEFAULT_LEVEL
# ===----------------------------------------------------------------------===#

alias DEFAULT_LEVEL = Level._from_str(
    env_get_string["LOGGING_LEVEL", "NOTSET"]()
)

# ===----------------------------------------------------------------------===#
# Level
# ===----------------------------------------------------------------------===#


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

    fn format_to(self, inout writer: Formatter):
        if self is Self.NOTSET:
            writer.write_str("NOTSET")
        elif self is Self.DEBUG:
            writer.write_str("DEBUG")
        elif self is Self.INFO:
            writer.write_str("INFO")
        elif self is Self.WARNING:
            writer.write_str("WARNING")
        elif self is Self.ERROR:
            writer.write_str("ERROR")
        elif self is Self.CRITICAL:
            writer.write_str("CRITICAL")

    @no_inline
    fn __str__(self) -> String:
        return String.format_sequence(self)

    @no_inline
    fn __repr__(self) -> String:
        return "Level." + str(self)


# ===----------------------------------------------------------------------===#
# Logger
# ===----------------------------------------------------------------------===#


struct Logger[level: Level = DEFAULT_LEVEL]:
    var _fd: FileDescriptor

    fn __init__(inout self, fd: FileDescriptor = sys.stdout):
        self._fd = fd

    @always_inline
    @staticmethod
    fn _is_active[target_level: Level]() -> Bool:
        if level == Level.NOTSET:
            return False
        return level > target_level

    fn debug[*Ts: Formattable](self, *values: *Ts):
        alias target_level = Level.DEBUG

        @parameter
        if Self._is_active[target_level]():
            return

        var writer = Formatter(fd=self._fd)

        writer.write(str(target_level), "::: ")

        @parameter
        fn print_with_separator[i: Int, T: Formattable](value: T):
            writer.write(value)

            @parameter
            if i < len(VariadicList(Ts)) - 1:
                writer.write(" ")

        values.each_idx[print_with_separator]()

        writer.write("\n")

    fn info[*Ts: Formattable](self, *values: *Ts):
        alias target_level = Level.INFO

        @parameter
        if Self._is_active[target_level]():
            return

        var writer = Formatter(fd=self._fd)

        writer.write(str(target_level), "::: ")

        @parameter
        fn print_with_separator[i: Int, T: Formattable](value: T):
            writer.write(value)

            @parameter
            if i < len(VariadicList(Ts)) - 1:
                writer.write(" ")

        values.each_idx[print_with_separator]()

        writer.write("\n")

    fn warning[*Ts: Formattable](self, *values: *Ts):
        alias target_level = Level.WARNING

        @parameter
        if Self._is_active[target_level]():
            return

        var writer = Formatter(fd=self._fd)

        writer.write(str(target_level), "::: ")

        @parameter
        fn print_with_separator[i: Int, T: Formattable](value: T):
            writer.write(value)

            @parameter
            if i < len(VariadicList(Ts)) - 1:
                writer.write(" ")

        values.each_idx[print_with_separator]()

        writer.write("\n")

    fn error[*Ts: Formattable](self, *values: *Ts):
        alias target_level = Level.CRITICAL

        @parameter
        if Self._is_active[target_level]():
            return

        var writer = Formatter(fd=self._fd)

        writer.write(str(target_level), "::: ")

        @parameter
        fn print_with_separator[i: Int, T: Formattable](value: T):
            writer.write(value)

            @parameter
            if i < len(VariadicList(Ts)) - 1:
                writer.write(" ")

        values.each_idx[print_with_separator]()

        writer.write("\n")

    fn critical[*Ts: Formattable](self, *values: *Ts):
        alias target_level = Level.CRITICAL

        @parameter
        if Self._is_active[target_level]():
            return

        var writer = Formatter(fd=self._fd)

        writer.write(str(target_level), "::: ")

        @parameter
        fn print_with_separator[i: Int, T: Formattable](value: T):
            writer.write(value)

            @parameter
            if i < len(VariadicList(Ts)) - 1:
                writer.write(" ")

        values.each_idx[print_with_separator]()

        writer.write("\n")

        abort()

    # Ideally we can remove the duplication and have something like, but
    # forwarding variadics does not work with mojo atm so we are forced to
    # copy/paste code :(
    # fn _log[
    #     target_level: Level, *Ts: Formattable
    # ](self, *values: VariadicPack[_, Formattable, Ts]):
    #     @parameter
    #     fn get_length() -> Int:
    #         return len(values)

    #     alias variadic_length = get_length()

    #     var writer = Formatter(fd=sys.stdout)

    #     writer.write(target_level, "::: ")

    #     @parameter
    #     for i in range(variadic_length):
    #         var elem = values[i]
    #         elem.write(writer)

    #         @parameter
    #         if i < variadic_length - 1:
    #             writer.write(" ")
