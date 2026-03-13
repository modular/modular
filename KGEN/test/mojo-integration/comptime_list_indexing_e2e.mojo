# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s


def f0(not_used: String, values: List[List[String]], i: Int) -> List[String]:
    return values[i].copy()


def main():
    # CHECK: [hello]
    # CHECK-NEXT: [world]
    comptime not_used = String("not_used")
    comptime res0 = f0(not_used, [["hello"], ["world"]], 0)
    print(materialize[String(res0)]())
    comptime res1 = f0(not_used, [["hello"], ["world"]], 1)
    print(materialize[String(res1)]())
