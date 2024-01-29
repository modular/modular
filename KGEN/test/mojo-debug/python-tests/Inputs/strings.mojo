# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn main():
    let literal = "string_literal"
    let s1 = String("let_string")
    var s2 = String("")
    for i in range(0, 100):
        s2 += str(i)
    var s3 = String()
    print(literal, s1, s2, s3)  # breakpoint
