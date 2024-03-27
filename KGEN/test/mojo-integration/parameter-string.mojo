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
