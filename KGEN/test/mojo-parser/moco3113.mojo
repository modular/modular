# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


# RUN: %parse-mojo-isolated %s | FileCheck %s


struct SomeStruct[x: Int, y: Int]:
    @implicit
    fn __init__(out self: SomeStruct[3, 4], other: SomeStruct[1, 1]):
        pass


fn take_x_and_plus_one[x: Int](s: SomeStruct[x, x + 1]):
    pass


fn test_some_struct(s: SomeStruct[1, 1]):
    # This should compile to a call
    # CHECK: lit.call @moco3113::@"take_x_and_plus_one{{.*}}#SomeStruct <:!Int {3}, :!Int {4}>>
    take_x_and_plus_one(s)
