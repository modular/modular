# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V2=cpu %s

from smoke_test_utils import create_test_device_context
from common_test_smoke import test_smoke
from common_test_timing import test_timing


def main():
    # Create an instance of the DeviceContextVariant based on the value of
    # the `MODULAR_ASYNCRT_DEVICE_CONTEXT_V2` define.
    var ctx = create_test_device_context()

    # Execute cpu tests with the context
    test_smoke(ctx)
    test_timing(ctx)
