# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo doc %s | FileCheck %s

from utils import Index

# CHECK: "value": "Index(16, 16, 16)"
alias x1 = Index(16, 16, 16)

# CHECK: "value": "Tuple(VariadicPack(Index(64, 8, 8)))"
alias x2 = (Index(64, 8, 8),)

# CHECK: "value": "Tuple(VariadicPack(1, 1))"
alias x3: Tuple[Int, Int] = (1, 1)

# Do not truncate non-functions.
# CHECK: "value": "Indexer"
alias x4 = Indexer


fn Indexing[T: Indexer](x: T):
    pass


# Do not truncate functions not literally "Index".
# CHECK: "value": "Indexing[
alias x5 = Indexing[Int](8)


struct S[a: Int, b: Int]:
    pass


# CHECK: "name": "S1"
# CHECK: "parameters": [
# CHECK:   "name": "z",
# CHECK:   "type": "Int"
# CHECK: "value": "S[1, z]"
alias S1[z: Int] = S[1, z]
