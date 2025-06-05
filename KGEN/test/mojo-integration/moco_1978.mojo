# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s


fn f0(not_used: String, values: List[List[String]], i: Int) -> List[String]:
    return values[i]


fn main():
    # CHECK: ['hello']
    # CHECK-NEXT: ['world']
    alias not_used = String("not_used")
    alias res0 = f0(
        not_used, List(List[String]("hello"), List[String]("world")), 0
    )
    print(res0.__str__())
    alias res1 = f0(
        not_used, List(List[String]("hello"), List[String]("world")), 1
    )
    print(res1.__str__())
