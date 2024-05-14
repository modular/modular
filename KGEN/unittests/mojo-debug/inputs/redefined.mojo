# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn use_int(x: Int):
    pass


fn use_string(y: String):
    pass


fn main():
    var x = 123
    x = x + 345
    use_int(x)  # breakpoint

    var y: String = "hello"
    y = "world"
    use_string(y)  # breakpoint
