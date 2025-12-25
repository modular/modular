# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


@fieldwise_init
@register_passable
struct Foo[x: Int](ImplicitlyCopyable):
    var b: Int

    fn get(self) -> Int:
        return self.b


# CHECK: lit.struct.decl @"`_CI_{{.*}}"<a: !Int, Y: !lit.struct<#Foo <:!Int a>>
# CHECK: lit.fn @"__call__
# CHECK-SAME: #Foo <:!Int apply(:{{.*}}@Foo::@"get{{.*}}"<:!Int a>), store_to_mem(Y))>


fn alias_ref_apply_in_sig[a: Int, Y: Foo[a]]():
    # alias Y = Foo[a](__mlir_attr.`2 : index`)

    fn p_capture(x: Int, y: Foo[Y.get()]) escaping -> Int:
        return Foo[a](x).get()
