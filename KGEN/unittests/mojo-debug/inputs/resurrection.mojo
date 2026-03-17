# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


def get_string() -> String:
    var s = "hel"
    s += "lo"  # defeat the string literal optimization.
    return s


def main():
    var text1 = get_string()
    var text2 = text1^
    print(text2)  # breakpoint
    text1 = get_string()
    print(text1)  # breakpoint
