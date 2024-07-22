# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


@value
struct VarParamStruct[*args: int]:
    pass

struct StructWithDefault[a: int, b: int, c: int = `1`, d: int = `2`]:
    pass


struct StructWithDefaultKwOnly[a: int, b: int, c: int = `1`, *, d: int = `2`]:
    pass


# CHECK-LABEL: lit.func @"test_unbound_pack
fn test_unbound_pack():
    # CHECK: lit.alias.decl *"all_unbound`": anystruct<#StructWithDefault <?, ?, ?, ?>, <"a": index, "b": index, "c": index = 1, "d": index = 2>>
    alias all_unbound = StructWithDefault[*_]

    # CHECK: lit.alias.decl *"first_bound`1": anystruct<#StructWithDefault <5, ?, ?, ?>, <"b": index, "c": index = 1, "d": index = 2>>
    alias first_bound = StructWithDefault[`5`, *_]

    # CHECK: lit.alias.decl *"last_bound_with_kw`2": anystruct<#StructWithDefaultKwOnly <8, ?, 1, ?>, <"b": index, *, "d": index = 2>>
    alias last_bound_with_kw = StructWithDefaultKwOnly[`8`, d=_, *_]
