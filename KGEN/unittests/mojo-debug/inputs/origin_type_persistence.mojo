# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that extending debug lifetimes of trivial types does not
# transitively keep non-trivial values alive.  A trivial Int derived
# from a non-trivial String (via len()) must not prevent the String
# from being ASAP-destroyed.
#
# Note: origin-carrying types like UnsafePointer and Span are
# TrivialRegisterPassable and not tracked by CheckLifetimes at all,
# so the debug lifetime extension cannot affect them.


def use_int(x: Int):
    pass


def get_string() -> String:
    var s = "hel"
    s += "lo"
    return s


def main():
    # Create a non-trivial value and a trivial value derived from it.
    var my_string = get_string()
    var length = len(my_string)
    use_int(length)  # last use of length

    # my_string should still be alive because print uses it.
    print(my_string)  # breakpoint

    # After this point, my_string should be ASAP-destroyed (non-trivial).
    # The trivial `length` should NOT keep my_string alive.
    print("after string use")  # breakpoint
