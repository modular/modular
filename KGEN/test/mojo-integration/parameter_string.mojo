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
    return StringLiteral.from_string[s + "-" + s]()


fn to_string_literal(i: Int) -> StringLiteral:
    return __mlir_op.`pop.string.create`(i)


fn to_string_literal(i: SIMD) -> StringLiteral:
    return __mlir_op.`pop.string.create`(i)


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

    alias s = to_string_literal(33)

    # CHECK: 33
    print(s)

    alias t = to_string_literal(Int64(42))

    # CHECK: #pop<simd 42> : !pop.scalar<si64>
    print(t)
