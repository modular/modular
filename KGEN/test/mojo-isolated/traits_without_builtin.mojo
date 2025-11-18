# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s --mojo-disable-builtins | FileCheck %s


struct Int:
    pass


trait A:
    pass


trait E(A):
    pass


struct List[T: A]():
    pass

    fn __init__(out self: List[Self.T]):
        pass


# CHECK-LABEL: lit.fn @"test_upcast_trait
fn test_upcast_trait[T: E](tuples: List[T]):
    pass
