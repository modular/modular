# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen -elaborate %s -S -o - | FileCheck %s --check-prefix=ELABORATE
# RUN: mojo %s | FileCheck %s

from std.collections.string.string_slice import StaticString, get_static_string


@fieldwise_init
struct StringParam[value: String]:
    def print_it(self):
        print(Self.value)


# ELABORATE: kgen.func @{{.*}}stringInputParam[[FUNC:.*]]()
# ELABORATE-NOT: kgen.func {{.*}}stringInputParam
@no_inline
def stringInputParam[value: String]():
    print(value)


@always_inline
def stringInputParamInline[value: String]():
    print(value)


def instantiateElsewhere():
    # ELABORATE: kgen.call {{.*}}stringInputParam[[FUNC]]()
    stringInputParam["thrice"]()


def test_literal_from_comptime_string[s: String]() -> StaticString:
    return get_static_string[s, "-", s]()


def test_parameter_string():
    # CHECK: hello world
    StringParam["hello" + " " + "world"]().print_it()

    comptime strValue: String = "thrice"
    # CHECK-COUNT-4: thrice
    # ELABORATE: kgen.call {{.*}}stringInputParam[[FUNC]]()
    stringInputParam[strValue]()
    instantiateElsewhere()

    stringInputParamInline[strValue]()
    print(strValue)

    # CHECK: hihi-hihi
    comptime hi: String = "hi"
    var str = test_literal_from_comptime_string[hi * 2]()
    print(str)

    # CHECK: 33
    print(get_static_string[String(33)]())

    # CHECK: 42
    print(get_static_string[String(Int64(42))]())


def test_string_global_constant():
    comptime A = String(11.1)

    # CHECK: alias is 11.1
    print("alias is ", A)


def main():
    test_parameter_string()
    test_string_global_constant()
