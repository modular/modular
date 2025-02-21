# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo doc %s | FileCheck %s

from utils import Index

# CHECK: "value": "Index[::Intable,::Intable,::Intable,::Int,::Bool](16, 16, 16)"
alias x1 = Index(16, 16, 16)

# CHECK: "value": "Tuple(VariadicPack(RefPack(Index[::Intable,::Intable,::Intable,::Int,::Bool](64, 8, 8)), True))"
alias x2 = (Index(64, 8, 8),)
