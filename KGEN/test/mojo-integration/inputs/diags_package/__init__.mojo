# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


def fn_missing_constraint[n: Int]() where n > 0:
    pass


def overloaded_function(n: Int):
    pass


def overloaded_function(n: Float64):
    pass


def overloaded_function(n: Int, m: Float64):
    pass


@fieldwise_init
struct PosOnlyStruct[a: Int, b: Int, /, c: Int = 9]:
    pass
