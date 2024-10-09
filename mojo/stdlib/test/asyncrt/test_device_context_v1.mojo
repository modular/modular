# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V1=cuda %s

from smoke_test_utils import create_test_device_context

from smoke_test_utils import create_test_device_context

from common_test_capture import test_capture
from common_test_copies import test_copies
from common_test_function import test_function
from common_test_memset import test_memset
from common_test_smoke import test_smoke
from common_test_timing import test_timing


def main():
    # Create an instance of the DeviceContextVariant based on the value of
    # the `MODULAR_ASYNCRT_DEVICE_CONTEXT_V2` define.
    var ctx = create_test_device_context()

    # Execute CUDA tests with the original DeviceContext.
    test_smoke(ctx)
    test_copies(ctx)
    test_memset(ctx)
    test_timing(ctx)
    test_function(ctx)
    test_capture(ctx)
