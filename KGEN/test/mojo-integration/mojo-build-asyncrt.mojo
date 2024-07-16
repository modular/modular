# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo build %s -o %t
# RUN: %t | FileCheck %s

# Test that we can build executables that rely on AsyncRT and the runtime
# libraries.

from runtime.asyncrt import create_task


# CHECK-LABEL: test_runtime_task
fn main():
    print("== test_runtime_task")

    @parameter
    async fn test_asyncrt_add[lhs: Int](rhs: Int) -> Int:
        return lhs + rhs

    @parameter
    async fn test_asyncrt_add_two_of_them(a: Int, b: Int) -> Int:
        return await create_task(test_asyncrt_add[1](a)) + await create_task(
            test_asyncrt_add[2](b)
        )

    var task = create_task(test_asyncrt_add_two_of_them(10, 20))
    # CHECK: 33
    print(task.wait())
