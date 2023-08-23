# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo %s | FileCheck %s


@value
struct StringParam[value: String]:
    fn print_it(self):
        print(value)


fn main():
    # CHECK: hello world
    StringParam[String("hello") + " " + "world"]().print_it()
