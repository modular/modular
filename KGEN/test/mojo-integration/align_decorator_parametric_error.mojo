# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not kgen -elaborate %s -o /dev/null 2>&1 | FileCheck %s


# Error tests for parametric @align decorator.
@no_inline
def calculate_illegal_alignment(x: Int) -> Int:
    return x * 3


@align(calculate_illegal_alignment(alignment))
@fieldwise_init
struct AlignedTrivialParam[alignment: Int](TrivialRegisterPassable):
    pass


@no_inline
def return_illegally_aligned_struct() -> AlignedTrivialParam[23]:
    return {}


# CHECK: function instantiation failed
def main():
    # CHECK: struct alignment must be a positive power of 2, got 69
    print(return_illegally_aligned_struct().alignment)
