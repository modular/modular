# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


def use_int(x: Int):
    pass


def use_string(y: String):
    pass


def main():
    var x = 123
    x = x + 345
    use_int(x)  # breakpoint

    var y: String
    y = "world"
    use_string(y)  # breakpoint
