# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: asan
# RUN: %mojo %s | FileCheck %s

from runtime.llcl import Runtime


@value
struct MemType:
    var value: Int


async fn might_raise(value: MemType, c: Bool) raises -> MemType:
    if c:
        raise "whoops"
    return MemType(value.value)


async fn call(value: MemType, c: Bool) raises -> MemType:
    return await might_raise(value, c)


fn main():
    # CHECK: async
    print("async")
    with Runtime() as rt:
        try:
            # CHECK-NEXT: 42
            print(rt.run(call(42, False)).value)
        except e:
            print(e)
        try:
            print(rt.run(call(24, True)).value)
        except e:
            # CHECK-NEXT: whoops
            print(e)
