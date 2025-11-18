# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -split-input-file | FileCheck %s

from stdlib.builtin.variadics import *

# CHECK-LABEL: lit.alias.decl *"T`0x": meta<!lit.struct<#Tuple <:variadic<!AnyType>
# CHECH-SAME: [!Int, !Int, !Int, !Int, !Int, !Int, !Int, !Int, !Int, !Int]
alias T = Tuple[*VariadicSplat[Int, 10]]
