# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# FIXME: KERN-1738
# UNSUPPORTED: AMD-GPU
# RUN: not --crash %bare-mojo -D ASSERT=all %s 2>&1 | FileCheck %s -check-prefix=CHECK-FAIL

from gpu.host import DeviceContext


# CHECK-FAIL-LABEL: test_fail
def main():
    print("== test_fail")
    # CHECK-FAIL: block: [0,0,0] thread: [0,0,0] Assert Error: forcing failure
    with DeviceContext() as ctx:

        fn fail_assert():
            debug_assert(False, "forcing failure")
            # CHECK-NOT: won't print this due to assert failure
            print("won't print this due to assert failure")

        ctx.enqueue_function[fail_assert](
            grid_dim=(2, 1, 1),
            block_dim=(2, 1, 1),
        )

        ctx.synchronize()
