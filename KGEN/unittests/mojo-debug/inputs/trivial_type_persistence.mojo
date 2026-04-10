# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test whether trivial types (Int, Float, Bool) persist in the debugger
# past their last use point through the end of the scope.


def use_int(x: Int):
    pass


def use_float(x: Float64):
    pass


def main():
    # Define trivial-type variables and use them early.
    var my_int = 42
    var my_float: Float64 = 3.14
    var my_bool = True
    use_int(my_int)  # last use of my_int
    use_float(my_float)  # last use of my_float

    # This breakpoint is PAST the last use of all three trivial variables.
    # If trivial types persist through the scope, they should still be visible.
    print("after last use")  # breakpoint

    # Use a non-trivial type to contrast behavior.
    var my_string = String("hello")
    print(my_string)  # breakpoint

    # Final breakpoint: my_string's last use was above, and all trivial
    # variables are still in scope but long past their last use.
    print("end")  # breakpoint
