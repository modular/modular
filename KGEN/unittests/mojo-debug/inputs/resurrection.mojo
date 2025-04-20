# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn get_string() -> String:
    var s = String("hel")
    s += "lo"  # defeat the string literal optimization.
    return s


fn main():
    var text1 = get_string()
    var text2 = text1^
    print(text2)  # breakpoint
    text1 = get_string()
    print(text1)  # breakpoint
