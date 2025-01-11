# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo --mojo-disable-builtins %s | FileCheck %s


struct Int:
    pass


trait A:
    pass


trait E(A):
    pass


struct List[T: A]():
    pass

    fn __init__(out self: List[T]):
        pass


# CHECK-LABEL: lit.fn @"test_upcast_trait
fn test_upcast_trait[T: E](tuples: List[T]):
    pass
