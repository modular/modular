# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: system-windows
# RUN: %mojo -debug-level full %s

from max.engine import InferenceSession
from testing import assert_false, assert_true


fn test_bool_value() raises:
    var session = InferenceSession()

    var false_value = session.new_bool_value(False)
    assert_false(false_value.as_bool())

    var true_value = session.new_bool_value(True)
    assert_true(true_value.as_bool())


fn main() raises:
    test_bool_value()
