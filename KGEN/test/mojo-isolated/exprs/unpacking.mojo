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


struct StructWithVariadic[a: int = `1`, *b: int]:
    pass


struct DefaultPosOnly[a: int = `1`, /, b: int = `2`, *, c: int = `3`]:
    pass


fn variadic_params[*a: int]():
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

    # CHECK: lit.alias.decl *"kw_unpacked`{{.*}}: anystruct<#StructWithDefaultKwOnly <?, ?, ?, ?>, <"a": index, "b": index, "c": index = 1, *, "d": index = 2>>
    alias kw_unpacked = StructWithDefaultKwOnly[**_]

    # CHECK: lit.alias.decl *"unpack_both{{.*}}: anystruct<#DefaultPosOnly <?, ?, ?>, <"a": index = 1, |, "b": index = 2, *, "c": index = 3>>
    alias unpack_both = DefaultPosOnly[*_, **_]

    # CHECK: lit.alias.decl *"pos_only_kw_unpacked`{{.*}}: anystruct<#DefaultPosOnly <1, ?, ?>, <"b": index = 2, *, "c": index = 3>>
    alias pos_only_kw_unpacked = DefaultPosOnly[**_]

    # CHECK: lit.alias.decl *"unbound_variadic`{{.*}}": anystruct<#StructWithVariadic <?, :variadic<index> ?>
    alias unbound_variadic = StructWithVariadic[*_]

    # CHECK: lit.alias.decl *"unpack_variadic`{{.*}}": !lit.generator<<"a": variadic<index> var>() -> !kgen.none> = <{{.*}}variadic_params{{.*}}<:variadic<index> ?>>
    alias unpack_variadic = variadic_params[*_]

    # CHECK: lit.call {{.*}}variadic_params{{.*}}<:variadic<index> []>()
    unpack_variadic()

    # CHECK: call {{.*}}variadic_params{{.*}}<:variadic<index> []>
    variadic_params[*_]()
