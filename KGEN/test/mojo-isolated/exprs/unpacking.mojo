# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


struct Parametric[a: Int]:
    pass


struct StructWithDefault[a: Int, b: Int, c: Int = 1, d: Int = 2]:
    pass


struct StructWithDefaultKwOnly[a: Int, b: Int, c: Int = 1, *, d: Int = 2]:
    pass


struct StructWithVariadic[a: Int = 1, *b: Int]:
    pass


struct DefaultPosOnly[a: Int = 1, /, b: Int = 2, *, c: Int = 3]:
    pass


fn variadic_params[*a: Int]():
    pass


# CHECK-LABEL: lit.fn @"test_unbound_pack
fn test_unbound_pack():
    # CHECK: lit.alias.decl *"all_unbound`": meta<!lit.struct<#StructWithDefault <:!Int ?, :!Int ?, :!Int ?, :!Int ?>, <"a": !Int, "b": !Int, "c": !Int = {{.*}}1{{.*}}, "d": !Int = {{.*}}2{{.*}}>>>
    alias all_unbound = StructWithDefault[*_]

    # CHECK: lit.alias.decl *"first_bound`{{.*}}": meta<!lit.struct<#StructWithDefault <:!Int {5}, :!Int ?, :!Int ?, :!Int ?>, <"b": !Int, "c": !Int = {{.*}}1{{.*}}, "d": !Int = {{.*}}2{{.*}}>>>
    alias first_bound = StructWithDefault[5, *_]

    # CHECK: lit.alias.decl *"last_bound_with_kw`{{.*}}": meta<!lit.struct<#StructWithDefaultKwOnly <:!Int {8}, :!Int ?, :!Int ?, :!Int ?>, <"b": !Int, "c": !Int = {{.*}}1{{.*}}, *, "d": !Int = {{.*}}2{{.*}}>>>
    alias last_bound_with_kw = StructWithDefaultKwOnly[8, d=_, *_]

    # CHECK: lit.alias.decl *"prev_bound_with_kw`{{.*}}: meta<!lit.struct<#StructWithDefaultKwOnly <:!Int {8}, :!Int ?, :!Int ?, :!Int ?>, <"b": !Int, "c": !Int = {{.*}}1{{.*}}, *, "d": !Int = {{.*}}2{{.*}}>>>
    alias prev_bound_with_kw = StructWithDefaultKwOnly[8, *_, d=_]

    # CHECK: lit.alias.decl *"kw_unpacked`{{.*}}: meta<!lit.struct<#StructWithDefaultKwOnly <:!Int ?, :!Int ?, :!Int ?, :!Int ?>, <"a": !Int, "b": !Int, "c": !Int = {{.*}}1{{.*}}, *, "d": !Int = {{.*}}2{{.*}}>>>
    alias kw_unpacked = StructWithDefaultKwOnly[**_]

    # CHECK: lit.alias.decl *"unpack_both{{.*}}: meta<!lit.struct<#DefaultPosOnly <:!Int ?, :!Int ?, :!Int ?>, <"a": !Int = {{.*}}1{{.*}}, |, "b": !Int = {{.*}}2{{.*}}, *, "c": !Int = {{.*}}3{{.*}}>>>
    alias unpack_both = DefaultPosOnly[*_, **_]

    # CHECK: lit.alias.decl *"pos_only_kw_unpacked`{{.*}}: meta<!lit.struct<#DefaultPosOnly <:!Int {1}, :!Int ?, :!Int ?>, <"b": !Int = {{.*}}2{{.*}}, *, "c": !Int = {{.*}}3{{.*}}>>>
    alias pos_only_kw_unpacked = DefaultPosOnly[**_]

    # CHECK: lit.alias.decl *"unbound_variadic`{{.*}}": meta<!lit.struct<#StructWithVariadic <:!Int ?, :variadic<!Int> ?>
    alias unbound_variadic = StructWithVariadic[*_]

    # CHECK: lit.alias.decl *"unpack_variadic`{{.*}}": !lit.generator<<"a": variadic<!Int> pos_vararg>() -> !kgen.none> = <{{.*}}variadic_params{{.*}}<:variadic<!Int> ?>>
    alias unpack_variadic = variadic_params[*_]

    # CHECK: lit.call {{.*}}variadic_params{{.*}}<:variadic<!Int> []>()
    unpack_variadic()

    # CHECK: call {{.*}}variadic_params{{.*}}<:variadic<!Int> []>
    variadic_params[*_]()
