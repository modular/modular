# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


@value
@register_passable
struct Foo[x: int]:
    var b: int

    fn get(self) -> int:
        return self.b


# CHECK: lit.struct.decl @"`_CI_{{.*}}"<a, Y: {{.*}}Foo<a>
# CHECK: lit.fn @"__call__
# CHECK-SAME: @Foo<apply(:{{.*}}@Foo::@"get{{.*}}"<a>), store_to_mem(Y))>


fn alias_ref_apply_in_sig[a: int, Y: Foo[a]]():
    #alias Y = Foo[a](__mlir_attr.`2 : index`)

    fn p_capture(x: int, y: Foo[Y.get()]) escaping -> int:
        return Foo[a](x).get()
