# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: system-windows
# RUN: %mojo -debug-level full %s

from max.engine import (
    get_version,
    InferenceSession,
)
from testing import assert_true


fn test_engine_version() raises:
    var version_str = get_version()
    assert_true(version_str)


fn test_session() raises:
    var session = InferenceSession()


fn main() raises:
    test_engine_version()
    test_session()
