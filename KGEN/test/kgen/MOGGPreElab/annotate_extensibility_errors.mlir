// Test errors that may be caught in mogg.annotate process
// RUN: kgen-opt %s --mogg-annotate-kernels --split-input-file --verify-diagnostics

// Hard coded registration function, has special `mogg.intrinsic_register`
lit.fn @"register(::StringLiteral)"(%name: !lit.struct<@stdlib::@builtin::@string_literal::@StringLiteral>, %num_dps_outputs: !lit.struct<@stdlib::@builtin::@int::@Int> = {1}) -> !kgen.none attributes {mogg.intrinsic_register, sourceName = "register", specialFnKind = 0 : i8} {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// expected-error @below {{The kernel must have an entry point named execute}}
lit.struct.decl @just_shape(trait<@stdlib::@builtin::@anytype::@AnyType>)
  decorators <:none apply(:!lit.generator<("name": !lit.struct<@stdlib::@builtin::@string_literal::@StringLiteral>, "num_dps_outputs": !lit.struct<@stdlib::@builtin::@int::@Int> = {1}) -> !kgen.none> @"register(::StringLiteral)", {:string "imposter_add"}, {1})> {
  lit.fn export @"shape"(%x: !lit.struct<@test2<:@DType *"dtype">>, %y: !lit.struct<@test3>) -> !kgen.none
    attributes {
        isStatic,
        sourceName = "shape", specialFnKind = 0 : i8} {
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
  }
}
