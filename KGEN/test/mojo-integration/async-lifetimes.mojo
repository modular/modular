# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# UNSUPPORTED: asan
# RUN: %mojo %s | FileCheck %s

from runtime.llcl import Runtime

# CHECK: foo
# CHECK-NEXT: hello async


async fn byref_capture(inout value: String):
    value += " async"
    print(value)


fn main():
    var x: String = "hello"

    var coro = byref_capture(x)
    print("foo")
    with Runtime() as rt:
        rt.run(coro^)
