# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


struct Parametric[a: Index]:
    pass


struct StructWithDefault[a: Index, b: Index, c: Index = `1`, d: Index = `2`]:
    pass


struct StructWithDefaultKwOnly[a: Index, b: Index, c: Index = `1`, *, d: Index = `2`]:
    pass


struct StructWithVariadic[a: Index = `1`, *b: Index]:
    pass


struct DefaultPosOnly[a: Index = `1`, /, b: Index = `2`, *, c: Index = `3`]:
    pass


fn variadic_params[*a: Index]():
    pass


# CHECK-LABEL: lit.fn @"test_unbound_pack
fn test_unbound_pack():
    # CHECK: lit.alias.decl *"all_unbound`": meta<!lit.struct<#StructWithDefault <?, ?, ?, ?>, <"a": index, "b": index, "c": index = 1, "d": index = 2>>>
    alias all_unbound = StructWithDefault[*_]

    # CHECK: lit.alias.decl *"first_bound`{{.*}}": meta<!lit.struct<#StructWithDefault <5, ?, ?, ?>, <"b": index, "c": index = 1, "d": index = 2>>>
    alias first_bound = StructWithDefault[`5`, *_]

    # CHECK: lit.alias.decl *"last_bound_with_kw`{{.*}}": meta<!lit.struct<#StructWithDefaultKwOnly <8, ?, ?, ?>, <"b": index, "c": index = 1, *, "d": index = 2>>>
    alias last_bound_with_kw = StructWithDefaultKwOnly[`8`, d=_, *_]

    # CHECK: lit.alias.decl *"prev_bound_with_kw`{{.*}}: meta<!lit.struct<#StructWithDefaultKwOnly <8, ?, ?, ?>, <"b": index, "c": index = 1, *, "d": index = 2>>>
    alias prev_bound_with_kw = StructWithDefaultKwOnly[`8`, *_, d=_]

    # CHECK: lit.alias.decl *"kw_unpacked`{{.*}}: meta<!lit.struct<#StructWithDefaultKwOnly <?, ?, ?, ?>, <"a": index, "b": index, "c": index = 1, *, "d": index = 2>>>
    alias kw_unpacked = StructWithDefaultKwOnly[**_]

    # CHECK: lit.alias.decl *"unpack_both{{.*}}: meta<!lit.struct<#DefaultPosOnly <?, ?, ?>, <"a": index = 1, |, "b": index = 2, *, "c": index = 3>>>
    alias unpack_both = DefaultPosOnly[*_, **_]

    # CHECK: lit.alias.decl *"pos_only_kw_unpacked`{{.*}}: meta<!lit.struct<#DefaultPosOnly <1, ?, ?>, <"b": index = 2, *, "c": index = 3>>>
    alias pos_only_kw_unpacked = DefaultPosOnly[**_]

    # CHECK: lit.alias.decl *"unbound_variadic`{{.*}}": meta<!lit.struct<#StructWithVariadic <?, :variadic<index> ?>
    alias unbound_variadic = StructWithVariadic[*_]

    # CHECK: lit.alias.decl *"unpack_variadic`{{.*}}": !lit.generator<<"a": variadic<index> pos_vararg>() -> !kgen.none> = <{{.*}}variadic_params{{.*}}<:variadic<index> ?>>
    alias unpack_variadic = variadic_params[*_]

    # CHECK: lit.call {{.*}}variadic_params{{.*}}<:variadic<index> []>()
    unpack_variadic()

    # CHECK: call {{.*}}variadic_params{{.*}}<:variadic<index> []>
    variadic_params[*_]()
