# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: not --crash %bare-mojo -D ASSERT=all %s 2>&1 | FileCheck %s -check-prefix=CHECK-FAIL

from gpu.host.device_context import DeviceContext


# CHECK-FAIL-LABEL: test_fail
def main():
    print("== test_fail")
    # CHECK-FAIL: block: [0,0,0], thread: [0,0,0] Assertion `forcing failure` failed
    with DeviceContext() as ctx:

        fn fail_assert():
            debug_assert(False, "forcing failure")
            # CHECK-NOT: won't print this due to assert failure
            print("won't print this due to assert failure")

        var fail_assert_launch = ctx.compile_function[fail_assert]()

        ctx.enqueue_function(
            fail_assert_launch,
            grid_dim=(2, 1, 1),
            block_dim=(2, 1, 1),
        )

        ctx.synchronize()
