# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn get_string() -> String:
    return "hello"


fn main():
    var text1 = get_string()
    var text2 = text1^
    print(text2)  # breakpoint
    text1 = get_string()
    print(text1)  # breakpoint
