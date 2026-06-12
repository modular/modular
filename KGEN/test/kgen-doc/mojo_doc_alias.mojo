# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-doc %s | FileCheck %s

from std.utils import Index

# CHECK-LABEL: "name": "x1"
# CHECK: "value": "Index[Int, Int, Int](Int(16), Int(16), Int(16))"
comptime x1 = Index(16, 16, 16)

# CHECK-LABEL: "name": "x2"
# CHECK: "value": "Tuple(Index[Int, Int, Int](Int(64), Int(8), Int(8)))"
comptime x2 = (Index(64, 8, 8),)

# CHECK-LABEL: "name": "x3"
# CHECK: "value": "Tuple(Int(1), Int(1))"
comptime x3: Tuple[Int, Int] = (1, 1)

# Do not truncate non-functions.
# CHECK-LABEL: "name": "x4"
# CHECK: "value": "Indexer"
comptime x4 = Indexer


def Indexing[T: Indexer](x: T):
    pass


# Do not truncate functions not literally "Index".
# CHECK-LABEL: "name": "x5"
# CHECK: "value": "Indexing[Int](Int(8))"
comptime x5 = Indexing[Int](8)


struct S[a: Int, b: Int]:
    pass


# CHECK-LABEL: "name": "S1"
# CHECK: "parameters": [
# CHECK:   "description": "An integer, naturally.",
# CHECK:   "name": "z",
# CHECK:   "type": "Int"
# CHECK: "signature": "comptime S1[z: Int]"
# CHECK: "value": "S[Int(1), z]"
comptime S1[z: Int] = S[1, z]
"""Returns an S with two Zs.

Parameters:
    z: An integer, naturally.
"""
