# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s

##===----------------------------------------------------------------------===##
# late binding
##===----------------------------------------------------------------------===##

comptime myIntAdd[x: Int, y: Int] = x + y
comptime myIntMul[x: Int, y: Int] = x * y
comptime myIntFMA[x: Int, y: Int, z: Int] = x * y + z


@no_inline
fn just_print[n: Int]():
    print(n)


@no_inline
fn bind_unop_and_print[unop: type_of(myIntAdd[2])]():
    just_print[unop[7]]()


@no_inline
fn bind_binop_and_print[binop: type_of(myIntAdd)]():
    bind_unop_and_print[binop[5]]()


fn main():
    # CHECK: 12
    bind_binop_and_print[myIntAdd]()
    # CHECK: 35
    bind_binop_and_print[myIntMul]()
    # CHECK: 37
    bind_binop_and_print[myIntFMA[z=2]]()
    # CHECK: 17
    bind_binop_and_print[myIntFMA[y=2]]()
