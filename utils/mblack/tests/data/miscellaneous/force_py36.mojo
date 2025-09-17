# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# ===----------------------------------------------------------------------=== #
#
# File originates from:
#   Repo:   git@github.com:psf/black.git
#   Commit: d4a85643a465f5fae2113d07d22d021d4af4795a
#   Path:   tests/data/miscellaneous/force_py36.py
#
# ===----------------------------------------------------------------------=== #

# The input source must not contain any Py36-specific syntax (e.g. argument type
# annotations, trailing comma after *rest) or this test becomes invalid.
def long_function_name(
    argument_one,
    argument_two,
    argument_three,
    argument_four,
    argument_five,
    argument_six,
    *rest,
):
    ...


# output
# The input source must not contain any Py36-specific syntax (e.g. argument type
# annotations, trailing comma after *rest) or this test becomes invalid.
def long_function_name(
    argument_one,
    argument_two,
    argument_three,
    argument_four,
    argument_five,
    argument_six,
    *rest,
):
    ...
