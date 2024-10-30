# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from register import *


@mogg_register("test_override_dummy_op")
fn my_func_one():
    return


@mogg_register("test_override_dummy_op")
fn my_func_two():
    return
