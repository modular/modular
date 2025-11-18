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


alias VA_SIZE[*Ts: AnyType] = variadic_size(Ts)
# CHECK: lit.alias.decl *"Folded`{{.*}}": !Int = <{3}>
alias Folded = VA_SIZE[Int, Int, Int]
