# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from register import *


@register_internal("test_override_dummy_op")
fn my_func_one():
    return


@register_internal("test_override_dummy_op")
fn my_func_two():
    return
