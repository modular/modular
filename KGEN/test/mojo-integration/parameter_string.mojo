# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen -elaborate %s -S -o - | FileCheck %s --check-prefix=ELABORATE
# RUN: mojo %s | FileCheck %s


@value
struct StringParam[value: String]:
    fn print_it(self):
        print(value)


# ELABORATE: kgen.func @{{.*}}stringInputParam[[FUNC:.*]]()
# ELABORATE-NOT: kgen.func {{.*}}stringInputParam
@no_inline
fn stringInputParam[value: String]():
    print(value)


@always_inline
fn stringInputParamInline[value: String]():
    print(value)


fn instantiateElsewhere():
    # ELABORATE: kgen.call @{{.*}}stringInputParam[[FUNC]]()
    stringInputParam["thrice"]()


fn test_literal_from_comptime_string[s: String]() -> StringLiteral:
    return get_string_literal[s + "-" + s]()


fn main():
    # CHECK: hello world
    StringParam[String("hello") + " " + "world"]().print_it()

    alias strValue: String = "thrice"
    # CHECK-COUNT-4: thrice
    # ELABORATE: kgen.call @{{.*}}stringInputParam[[FUNC]]()
    stringInputParam[strValue]()
    instantiateElsewhere()

    # ELABORATE-COUNT-2: kgen.param.materialize: struct<(struct<(pointer<
    stringInputParamInline[strValue]()
    print(strValue)

    # CHECK: hihi-hihi
    alias hi: String = "hi"
    var strlit: StringLiteral = test_literal_from_comptime_string[hi * 2]()
    print(strlit)

    # CHECK: 33
    print(get_string_literal[33]())

    # CHECK: 42
    print(get_string_literal[Int64(42)]())

    # CHECK: [1, 2, 3, 4]
    print(get_string_literal[SIMD[DType.int64, 4](1, 2, 3, 4)]())
