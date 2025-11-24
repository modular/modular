# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -split-input-file | FileCheck %s


from stdlib.builtin.variadics import *


fn unfoldable[
    elt0: VariadicOf[AnyType], elt1: VariadicOf[AnyType]
](t0: Tuple[*elt0], t1: Tuple[*elt1]) -> Tuple[*ZipToTuple[elt0, elt1]]:
    pass


# CHECK-LABEL:  lit.fn @"foldable
fn foldable(
    t0: Tuple[Int, Int, Int],
    t1: Tuple[FloatDyn, FloatDyn, FloatDyn]
    # CHECK-SAME:  %__result__: !lit.ref<!lit.struct<#Tuple <:variadic<!AnyType>
    # CHECK-SAME: [
    # CHECK-SAME:  @Tuple<:variadic<!AnyType> [!Int, !FloatDyn]>,
    # CHECK-SAME:  @Tuple<:variadic<!AnyType> [!Int, !FloatDyn]>,
    # CHECK-SAME:  @Tuple<:variadic<!AnyType> [!Int, !FloatDyn]>
    # CHECK-SAME: ]>
) -> type_of(unfoldable(t0, t1)):
    pass
