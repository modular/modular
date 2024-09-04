# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


struct Parametric[a: int]:
    pass


struct StructWithDefault[a: int, b: int, c: int = `1`, d: int = `2`]:
    pass


struct StructWithDefaultKwOnly[a: int, b: int, c: int = `1`, *, d: int = `2`]:
    pass


# CHECK-LABEL: lit.func @"test_unbound_pack
fn test_unbound_pack():
    # CHECK: lit.alias.decl *"all_unbound`": anystruct<#StructWithDefault <?, ?, ?, ?>, <"a": index, "b": index, "c": index = 1, "d": index = 2>>
    alias all_unbound = StructWithDefault[*_]

    # CHECK: lit.alias.decl *"first_bound`{{.*}}": anystruct<#StructWithDefault <5, ?, ?, ?>, <"b": index, "c": index = 1, "d": index = 2>>
    alias first_bound = StructWithDefault[`5`, *_]

    # CHECK: lit.alias.decl *"last_bound_with_kw`{{.*}}": anystruct<#StructWithDefaultKwOnly <8, ?, ?, ?>, <"b": index, "c": index = 1, *, "d": index = 2>>
    alias last_bound_with_kw = StructWithDefaultKwOnly[`8`, d=_, *_]

    # CHECK: lit.alias.decl *"prev_bound_with_kw`{{.*}}: anystruct<#StructWithDefaultKwOnly <8, ?, ?, ?>, <"b": index, "c": index = 1, *, "d": index = 2>>
    alias prev_bound_with_kw = StructWithDefaultKwOnly[`8`, *_, d=_]

    # CHECK: lit.alias.decl *"unbound_between`{{.*}}": anystruct<#StructWithDefault <1, ?, ?, 2>, <"b": index, "c": index = 1>>
    alias unbound_between = StructWithDefault[`1`, *_, `2`]


# CHECK-LABEL: test_multiple_unbound_pack
fn test_multiple_unbound_pack():
    # CHECK: Parametric <1>
    alias t = Parametric[*_, `1`, *_]
