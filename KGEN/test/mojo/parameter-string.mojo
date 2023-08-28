# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s | FileCheck %s
# RUN: kgen -elaborate %s -S -o - | FileCheck %s --check-prefix=ELABORATE


@value
struct StringParam[value: String]:
    fn print_it(self):
        print(value)


# ELABORATE: kgen.func @"$parameter-string::stringInputParam[[FUNC:.*]]"() -> !pop.array<0, i1>
# ELABORATE-NOT: kgen.func {{.*}}stringInputParam
@no_inline
fn stringInputParam[value: String]():
    print(value)


@always_inline
fn stringInputParamInline[value: String]():
    print(value)


fn instantiateElsewhere():
    # ELABORATE: kgen.call @"$parameter-string::stringInputParam[[FUNC]]"() : () -> !pop.array<0, i1>
    stringInputParam["thrice"]()


fn main():
    # CHECK: hello world
    StringParam[String("hello") + " " + "world"]().print_it()

    alias strValue: String = "thrice"
    # CHECK-COUNT-4: thrice
    # ELABORATE: kgen.call @"$parameter-string::stringInputParam[[FUNC]]"() : () -> !pop.array<0, i1>
    stringInputParam[strValue]()
    instantiateElsewhere()

    # ELABORATE-COUNT-2: kgen.param.materialize: struct<pointer<
    stringInputParamInline[strValue]()
    print(strValue)
