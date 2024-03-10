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
    # CHECK: lit.alias.decl *"all_unbound`": metatype<#StructWithDefault <?, ?, ?, ?>, <"a": index, "b": index, "c": index = 1, "d": index = 2>>
    alias all_unbound = StructWithDefault[*_]

    # CHECK: lit.alias.decl *"first_bound`1": metatype<#StructWithDefault <5, ?, ?, ?>, <"b": index, "c": index = 1, "d": index = 2>>
    alias first_bound = StructWithDefault[`5`, *_]

    # CHECK: lit.alias.decl *"last_bound`2": metatype<#StructWithDefault <?, ?, ?, 6>, <"a": index, "b": index, "c": index = 1>>
    alias last_bound = StructWithDefault[*_, `6`]

    # CHECK: lit.alias.decl *"mid_unbound`3": metatype<#StructWithDefault <3, ?, ?, 4>, <"b": index, "c": index = 1>>
    alias mid_unbound = StructWithDefault[`3`, *_, `4`]

    # CHECK: lit.alias.decl *"last_bound_without_kw`4": metatype<#StructWithDefaultKwOnly <?, ?, 7, 2>, <"a": index, "b": index>>
    alias last_bound_without_kw = StructWithDefaultKwOnly[*_, `7`]

    # CHECK: lit.alias.decl *"last_bound_with_kw`5": metatype<#StructWithDefaultKwOnly <?, ?, 8, ?>, <"a": index, "b": index, *, "d": index = 2>>
    alias last_bound_with_kw = StructWithDefaultKwOnly[*_, `8`, d=_]
