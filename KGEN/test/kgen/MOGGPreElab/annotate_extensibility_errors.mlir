// Test errors that may be caught in mogg.annotate process
// RUN: kgen-opt %s --mogg-annotate --split-input-file --verify-diagnostics

// Hard coded registration function, has special `mogg.intrinsic_register`
lit.func @"register(::StringLiteral)"(%name: !lit.struct<@stdlib::@builtin::@string_literal::@StringLiteral>, %num_dps_outputs: !lit.struct<@stdlib::@builtin::@int::@Int> = {1}) -> !kgen.none attributes {mogg.intrinsic_register, sourceName = "register", specialFnKind = 0 : i8} {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// expected-error @below {{Struct based extensibility cannot have execute and initialize_output op!}}
lit.struct.decl @execute_shape_and_init(trait<@stdlib::@builtin::@anytype::@AnyType>)
  decorators <:none apply(:!lit.signature<("name": !lit.struct<@stdlib::@builtin::@string_literal::@StringLiteral>, "num_dps_outputs": !lit.struct<@stdlib::@builtin::@int::@Int> = {1}) -> !kgen.none> @"register(::StringLiteral)", {:string "imposter_add"}, {1})> {
  lit.func export @"execute"(%z: !lit.struct<@test1>, %x: !lit.struct<@test2<:@DType *"dtype">>, %y: !lit.struct<@test3>) -> !kgen.none
    attributes {
        isStatic,
        sourceName = "execute",
        specialFnKind = 0 : i8} {
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
  }

  lit.func export @"shape"(%x: !lit.struct<@test2<:@DType *"dtype">>, %y: !lit.struct<@test3>) -> !kgen.none
    attributes {
        isStatic,
        sourceName = "shape", specialFnKind = 0 : i8} {
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
  }

  lit.func export @"initialize_output"(%z: !lit.struct<@test1>, %x: !lit.struct<@test2<:@DType *"dtype">>, %y: !lit.struct<@test3>) -> !lit.struct<@test1>
    attributes {
        isStatic,
        sourceName = "initialize_output",
        specialFnKind = 0 : i8} {
    %none = kgen.param.constant: none = <#kgen.none>
    %hack = kgen.rebind %none : !kgen.none to !lit.struct<@test1>
    kgen.return %hack : !lit.struct<@test1>
  }
}

// -----

lit.func @"register(::StringLiteral)"(%name: !lit.struct<@stdlib::@builtin::@string_literal::@StringLiteral>) -> !kgen.none attributes {mogg.intrinsic_register, sourceName = "register", specialFnKind = 0 : i8} {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// expected-error @below {{Struct based extensibility cannot have initialize_output and shape op!}}
lit.struct.decl @init_and_shape(trait<@stdlib::@builtin::@anytype::@AnyType>)
  decorators <:none apply(:!lit.signature<("name": !lit.struct<@stdlib::@builtin::@string_literal::@StringLiteral>, "num_dps_outputs": !lit.struct<@stdlib::@builtin::@int::@Int> = {1}) -> !kgen.none> @"register(::StringLiteral)", {:string "imposter_add"}, {1})> {
  lit.func export @"shape"(%x: !lit.struct<@test2<:@DType *"dtype">>, %y: !lit.struct<@test3>) -> !kgen.none
    attributes {
        isStatic,
        sourceName = "shape", specialFnKind = 0 : i8} {
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
  }

  lit.func export @"initialize_output"(%z: !lit.struct<@test1>, %x: !lit.struct<@test2<:@DType *"dtype">>, %y: !lit.struct<@test3>) -> !lit.struct<@test1>
    attributes {
        isStatic,
        sourceName = "initialize_output",
        specialFnKind = 0 : i8} {
    %none = kgen.param.constant: none = <#kgen.none>
    %hack = kgen.rebind %none : !kgen.none to !lit.struct<@test1>
    kgen.return %hack : !lit.struct<@test1>
  }
}

// -----

lit.func @"register(::StringLiteral)"(%name: !lit.struct<@stdlib::@builtin::@string_literal::@StringLiteral>) -> !kgen.none attributes {mogg.intrinsic_register, sourceName = "register", specialFnKind = 0 : i8} {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// expected-error @below {{Struct based extensibility needs execute or initialize_output!}}
lit.struct.decl @just_shape(trait<@stdlib::@builtin::@anytype::@AnyType>)
  decorators <:none apply(:!lit.signature<("name": !lit.struct<@stdlib::@builtin::@string_literal::@StringLiteral>, "num_dps_outputs": !lit.struct<@stdlib::@builtin::@int::@Int> = {1}) -> !kgen.none> @"register(::StringLiteral)", {:string "imposter_add"}, {1})> {
  lit.func export @"shape"(%x: !lit.struct<@test2<:@DType *"dtype">>, %y: !lit.struct<@test3>) -> !kgen.none
    attributes {
        isStatic,
        sourceName = "shape", specialFnKind = 0 : i8} {
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
  }
}
