# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# This file is a test input that defines a module within a nested package.


def nested_function():
    return


# A function with a non-inferrable parameter, used to check that an "Included
# from" stack flows through a re-exporting module down to the definition.
def parametric_fn[n: Int]() -> Int:
    return n
