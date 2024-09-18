# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: system-windows
# RUN: %mojo -debug-level full %s

from testing import assert_equal, assert_false, assert_true

from max.engine import InferenceSession


fn test_list_value() raises:
    var session = InferenceSession()
    var list_value = session.new_list_value()
    var list = list_value.as_list()

    assert_equal(len(list), 0)

    var false_value = session.new_bool_value(False)
    var true_value = session.new_bool_value(True)

    list.append(false_value)
    assert_equal(len(list), 1)

    list.append(true_value)
    assert_equal(len(list), 2)

    assert_false(list[0].as_bool())
    assert_true(list[1].as_bool())

    _ = list_value^


fn main() raises:
    test_list_value()
