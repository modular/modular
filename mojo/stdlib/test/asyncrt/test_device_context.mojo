# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from gpu.host import DeviceContext
from common_test_smoke import test_smoke
from common_test_copies import test_copies
from common_test_function import test_function
from common_test_timing import test_timing
from common_test_elementwise import test_elementwise


def main():
    # Create an instance of DeviceContext.
    var ctx = DeviceContext()

    # Execute CUDA tests with the original DeviceContext.
    # test_smoke[DeviceContext](ctx)
    # test_copies(ctx)
    # test_function(ctx)
    # test_timing(ctx)
    test_elementwise(ctx)
