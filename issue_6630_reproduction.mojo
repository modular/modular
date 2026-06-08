# ===----------------------------------------------------------------------=== #
# Reproduction of Issue #6630
# ===----------------------------------------------------------------------=== #
# This file demonstrates the bug where hard keywords can be used as function
# names but cannot be called.

# BUG: This should fail at declaration time, not at call time
def match():
    """This function uses a hard keyword as its name without escaping."""
    print("In match function")


def class():
    """Another hard keyword used as function name."""
    print("In class function")


def yield():
    """Yet another hard keyword as function name."""
    print("In yield function")


def del():
    """Hard keyword 'del' as function name."""
    print("In del function")


# Attempting to call these functions will fail:
def main():
    # These calls will all fail with "unexpected token in expression"
    # because the parser sees the keyword, not the function name

    # match()     # Error: unexpected token in expression
    # class()     # Error: unexpected token in expression
    # yield()     # Error: unexpected token in expression
    # del()       # Error: unexpected token in expression

    print("Cannot call any of the functions defined above!")
    print("The bug is that they shouldn't be allowed to be declared in the first place.")


# WORKAROUND: Use backtick escaping
def `match_escaped`():
    """This works because the keyword is escaped."""
    print("In escaped match function")


def main_workaround():
    `match_escaped`()  # This works!
    print("Escaped version works correctly.")
