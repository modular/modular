# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo doc %s | FileCheck %s

from utils import Index

# CHECK-LABEL: "name": "x1"
# CHECK: "value": "Index(16, 16, 16)"
comptime x1 = Index(16, 16, 16)

# CHECK-LABEL: "name": "x2"
# CHECK: "value": "Tuple[IndexList[3]](VariadicPack[True, True, origin_of(), Copyable & Movable, IndexList[3]](Index(64, 8, 8)))"
comptime x2 = (Index(64, 8, 8),)

# CHECK-LABEL: "name": "x3"
# CHECK: "value": "Tuple[Int, Int](VariadicPack[True, True, origin_of(), Copyable & Movable, Int, Int](1, 1))"
comptime x3: Tuple[Int, Int] = (1, 1)

# Do not truncate non-functions.
# CHECK-LABEL: "name": "x4"
# CHECK: "value": "Indexer"
comptime x4 = Indexer


fn Indexing[T: Indexer](x: T):
    pass


# Do not truncate functions not literally "Index".
# CHECK-LABEL: "name": "x5"
# CHECK: "value": "Indexing[Int](8)"
comptime x5 = Indexing[Int](8)


struct S[a: Int, b: Int]:
    pass


# CHECK-LABEL: "name": "S1"
# CHECK: "parameters": [
# CHECK:   "description": "An integer, naturally.",
# CHECK:   "name": "z",
# CHECK:   "type": "Int"
# CHECK: "signature": "comptime S1[z: Int]"
# CHECK: "value": "S[1, z]"
comptime S1[z: Int] = S[1, z]
"""Returns an S with two Zs.

Parameters:
    z: An integer, naturally.
"""
