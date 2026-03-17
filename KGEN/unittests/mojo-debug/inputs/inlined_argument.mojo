# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@always_inline
def modify(mut x: Int):
    x = 42


def use_int(x: Int):
    pass


def main():
    var m: Int = 5
    modify(m)
    use_int(m)  # breakpoint
