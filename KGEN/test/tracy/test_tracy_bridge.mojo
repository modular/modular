# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from std.testing import assert_equal, assert_true
from std.ffi import external_call


def test_tracy_bridge_symbols_exist_and_disabled_by_default() raises:
    # Query whether CompilerRT was built with TRACY_ENABLE.
    enabled = external_call["KGEN_CompilerRT_TracyIsEnabled", Int]()

    # Begin/End should behave based on `enabled`.
    name = "tracy_bridge_test"
    ctx = external_call["KGEN_CompilerRT_TracyZoneBegin", UInt64](
        name.unsafe_ptr(), name.byte_length(), 0
    )
    if enabled != 0:
        assert_true(ctx != UInt64(0))
    else:
        assert_equal(UInt64(0), ctx)
    external_call["KGEN_CompilerRT_TracyZoneEnd", NoneType](ctx)


def main() raises:
    test_tracy_bridge_symbols_exist_and_disabled_by_default()
