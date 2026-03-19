# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test input for verifying that ASAP-destroyed non-trivial variables (String)
# show as "not available" in the debugger after their last use, rather than
# showing wrong/garbage values (MOTO-1424).


fn get_string() -> String:
    var s = "hel"
    s += "lo"  # defeat the string literal optimization.
    return s


fn main():
    var text = get_string()
    print(text)  # breakpoint
    print("done")  # breakpoint
