# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@always_inline
def nested_callee(a: Int):
    var nested_var = a
    print(nested_var)  # breakpoint


@always_inline("nodebug")
def nodebug_wrapper(b: Int):
    nested_callee(b)


def main():
    nodebug_wrapper(2)
