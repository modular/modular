# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: %mojo -I %engine_pkg_dir %s | FileCheck %s

from max.engine import InferenceSession


fn test_bool_value() raises:
    # CHECK-LABEL: ====test_bool_value
    print("====test_bool_value")

    var session = InferenceSession()

    var false_value = session.new_bool_value(False)
    # CHECK: False
    print(false_value.as_bool())

    var true_value = session.new_bool_value(True)
    # CHECK: True
    print(true_value.as_bool())


fn main() raises:
    test_bool_value()
