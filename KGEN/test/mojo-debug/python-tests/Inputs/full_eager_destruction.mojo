# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn get_string() -> String:
    return "hello"


fn get_number() -> Int:
    return 8


fn main():
    var text = get_string()
    print(text)  # breakpoint

    var number = get_number()
    for i in range(2):
        print(number)  # breakpoint
    print(0)  # breakpoint

    var simd = SIMD[DType.int16, 4](1, 2, 3, 4)
    if simd[0] < 0:
        print(simd)
    else:
        print(0)  # breakpoint
