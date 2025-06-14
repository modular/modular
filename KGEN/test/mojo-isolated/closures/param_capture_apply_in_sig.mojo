# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


@fieldwise_init
@register_passable
struct Foo[x: Index](Copyable):
    var b: Index

    fn get(self) -> Index:
        return self.b


# CHECK: lit.struct.decl @"`_CI_{{.*}}"<a, Y: {{.*}}Foo<a>
# CHECK: lit.fn @"__call__
# CHECK-SAME: @Foo<apply(:{{.*}}@Foo::@"get{{.*}}"<a>), store_to_mem(Y))>


fn alias_ref_apply_in_sig[a: Index, Y: Foo[a]]():
    #alias Y = Foo[a](__mlir_attr.`2 : index`)

    fn p_capture(x: Index, y: Foo[Y.get()]) escaping -> Index:
        return Foo[a](x).get()
