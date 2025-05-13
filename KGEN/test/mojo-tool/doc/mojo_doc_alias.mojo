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
