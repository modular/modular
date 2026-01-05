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
    comptime all_unbound = StructWithDefault[...]

    # CHECK: lit.alias.decl *"first_bound`{{.*}}": meta<!lit.struct<#StructWithDefault <:!Int {5}, :!Int ?, :!Int ?, :!Int ?>, <"b": !Int, "c": !Int = {{.*}}1{{.*}}, "d": !Int = {{.*}}2{{.*}}>>>
    comptime first_bound = StructWithDefault[5, ...]

    # CHECK: lit.alias.decl *"last_bound_with_kw`{{.*}}": meta<!lit.struct<#StructWithDefaultKwOnly <:!Int {8}, :!Int ?, :!Int ?, :!Int ?>, <"b": !Int, "c": !Int = {{.*}}1{{.*}}, *, "d": !Int = {{.*}}2{{.*}}>>>
    comptime last_bound_with_kw = StructWithDefaultKwOnly[8, d=_, ...]

    # CHECK: lit.alias.decl *"prev_bound_with_kw`{{.*}}: meta<!lit.struct<#StructWithDefaultKwOnly <:!Int {8}, :!Int ?, :!Int ?, :!Int ?>, <"b": !Int, "c": !Int = {{.*}}1{{.*}}, *, "d": !Int = {{.*}}2{{.*}}>>>
    comptime prev_bound_with_kw = StructWithDefaultKwOnly[8, ..., d=_]

    # CHECK: lit.alias.decl *"kw_unpacked`{{.*}}: meta<!lit.struct<#StructWithDefaultKwOnly <:!Int ?, :!Int ?, :!Int ?, :!Int ?>, <"a": !Int, "b": !Int, "c": !Int = {{.*}}1{{.*}}, *, "d": !Int = {{.*}}2{{.*}}>>>
    comptime kw_unpacked = StructWithDefaultKwOnly[...]

    # CHECK: lit.alias.decl *"unpack_both{{.*}}: meta<!lit.struct<#DefaultPosOnly <:!Int ?, :!Int ?, :!Int ?>, <"a": !Int = {{.*}}1{{.*}}, |, "b": !Int = {{.*}}2{{.*}}, *, "c": !Int = {{.*}}3{{.*}}>>>
    comptime unpack_both = DefaultPosOnly[...]

    # CHECK: lit.alias.decl *"unpack_ellipsis{{.*}}: meta<!lit.struct<#DefaultPosOnly <:!Int ?, :!Int ?, :!Int ?>, <"a": !Int = {{.*}}1{{.*}}, |, "b": !Int = {{.*}}2{{.*}}, *, "c": !Int = {{.*}}3{{.*}}>>>
    comptime unpack_ellipsis = DefaultPosOnly[...]

    # CHECK: lit.alias.decl *"unbound_variadic`{{.*}}": meta<!lit.struct<#StructWithVariadic <:!Int ?, :variadic<!Int> ?>
    comptime unbound_variadic = StructWithVariadic[...]

    # CHECK: lit.alias.decl *"unpack_variadic`{{.*}}": !lit.generator<<"a": variadic<!Int> pos_vararg>() -> !kgen.none> = <{{.*}}variadic_params{{.*}}<:variadic<!Int> ?>>
    comptime unpack_variadic = variadic_params[...]

    # CHECK: lit.call {{.*}}variadic_params{{.*}}<:variadic<!Int> []>()
    unpack_variadic()

    # CHECK: call {{.*}}variadic_params{{.*}}<:variadic<!Int> []>
    variadic_params[...]()
