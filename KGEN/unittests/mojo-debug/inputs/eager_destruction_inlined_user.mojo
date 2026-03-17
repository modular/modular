# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@always_inline
def use_string(s: String) -> Int:
    if len(s):
        return 2
    return 8


def main():
    var foo = "4" + "2"
    print(use_string(foo))  # breakpoint
