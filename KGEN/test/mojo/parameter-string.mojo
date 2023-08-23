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


# ELABORATE: kgen.func {{.*}}stringInputParam
# ELABORATE-NOT: kgen.func {{.*}}stringInputParam
fn stringInputParam[value: String]():
    print(value)


fn main():
    # CHECK: hello world
    # ELABORATE: kgen.call
    StringParam[String("hello") + " " + "world"]().print_it()

    alias strValue: String = "thrice"
    # CHECK-COUNT-3: thrice
    print(strValue)
    # ELABORATE: kgen.call @[[INSTANTIATION:.*]]() : () -> !pop.array<0, i1>
    stringInputParam[strValue]()
    # ELABORATE: kgen.call
    instantiateElsewhere()


fn instantiateElsewhere():
    # ELABORATE: kgen.call @[[INSTANTIATION]]() : () -> !pop.array<0, i1>
    stringInputParam["thrice"]()
