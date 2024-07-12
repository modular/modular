# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo build %s -o %t
# RUN: %t | FileCheck %s

# Test that we can build executables that rely on LLCL and the runtime
# libraries.

from runtime.llcl import Runtime


# CHECK-LABEL: test_runtime_task
fn main():
    print("== test_runtime_task")

    @parameter
    async fn test_llcl_add[lhs: Int](rhs: Int) -> Int:
        return lhs + rhs

    @parameter
    async fn test_llcl_add_two_of_them(rt: Runtime, a: Int, b: Int) -> Int:
        return await rt.create_task(test_llcl_add[1](a)) + await rt.create_task(
            test_llcl_add[2](b)
        )

    with Runtime() as rt:
        var task = rt.create_task(test_llcl_add_two_of_them(rt, 10, 20))
        # CHECK: 33
        print(task.wait())
