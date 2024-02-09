# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: %mojo -I %engine_pkg_dir -I %test_utils_pkg_dir %s | FileCheck %s

from max.engine import EngineTensorView, InferenceSession
from tensor import Tensor, TensorShape
from test_utils import linear_fill
from utils.index import Index
from python import Python


fn test_list_value() raises:
    # CHECK-LABEL: ====test_list_value
    print("====test_list_value")

    let session = InferenceSession()
    let list_value = session.new_list_value()
    let list = list_value.as_list()

    # CHECK: 0
    print(len(list))

    let false_value = session.new_bool_value(False)
    let true_value = session.new_bool_value(True)

    list.append(false_value)
    # CHECK: 1
    print(len(list))

    list.append(true_value)
    # CHECK: 2
    print(len(list))

    # CHECK: False
    # CHECK: True
    print(list[0].as_bool())
    print(list[1].as_bool())

    _ = list_value ^


fn main() raises:
    test_list_value()
