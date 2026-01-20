# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s


fn f0(not_used: String, values: List[List[String]], i: Int) -> List[String]:
    return values[i].copy()


fn main():
    # CHECK: ['hello']
    # CHECK-NEXT: ['world']
    comptime not_used = String("not_used")
    comptime res0 = f0(not_used, [["hello"], ["world"]], 0)
    print(materialize[res0.__str__()]())
    comptime res1 = f0(not_used, [["hello"], ["world"]], 1)
    print(materialize[res1.__str__()]())
