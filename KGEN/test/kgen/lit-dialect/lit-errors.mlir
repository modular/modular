// RUN: kgen-opt %s -verify-parameters -verify-diagnostics -split-input-file -o /dev/null

lit.struct.decl @SomeStruct {
  // expected-error @+1 {{invalid use of parameter with no declaration "ty"}}
  %size = lit.var.decl "size" var : !lit.ref<simd<1, ty>, mut origin>
}

// -----

// expected-error @below {{expected declaration body to have no arguments}}
"lit.struct.decl"() ({
^bb0(%arg0: i32):
}) {sym_name = "StructArgs",
    decorators = #kgen<decorators[]>,
    signature = !lit.type_signature,
    canonicalTrait = !lit.trait<@Foo>,
    params = #kgen<param.decls[]>
    } : () -> ()

// -----

// expected-error @below {{custom op 'lit.struct.decl' expected no result parameters}}
lit.struct.decl @StructReturns<() -> dtype> {}

// -----

// expected-error @below {{custom op 'lit.fn' expected no result parameters}}
lit.fn @func_param_return<() -> dtype> {}

// -----

lit.struct.decl @StructDuplicate {
  // expected-note @below {{see previous declaration here}}
  lit.struct.field x : i32
  lit.struct.field y : i32
  // expected-error @below {{duplicate struct field "x"}}
  lit.struct.field x : i32
}

// -----

lit.struct.decl @SomeType<v, b> {}

// expected-error @below {{'kgen.generator' op invalid use of parameter with no declaration "c"}}
kgen.generator @InvalidTypeParamValue<a>(%arg0: !lit.struct<@SomeType<a, c>>) {
  kgen.return
}

// -----

lit.struct.decl @Bar {}

kgen.generator @invalid_field_name(%a: index, %container: !lit.struct<@Bar>) {
  // expected-error @below {{struct @Bar has no field named "a"}}
  %0 = lit.struct.insert %a, %container[a] : index into !lit.struct<@Bar>
  kgen.return
}

// -----

lit.struct.decl @Bar {
  lit.struct.field a : i32
}

kgen.generator @invalid_field_name(%a: index, %container: !lit.struct<@Bar>) {
  // expected-error @below {{cannot insert value of type 'index' into struct field "a" which expected 'i32'}}
  %0 = lit.struct.insert %a, %container[a] : index into !lit.struct<@Bar>
  kgen.return
}

// -----

lit.struct.decl @Bar {}

kgen.generator @invalid_field_name(%a: index, %container: !lit.struct<@Bar>) {
  // expected-error @below {{struct @Bar has no field named "a"}}
  %0 = lit.struct.extract %container[a] : index from !lit.struct<@Bar>
  kgen.return
}

// -----

// expected-error @below {{expected SSA operand}}
lit.fn @no_names(index)

// -----

// expected-error @below {{only one '|' allowed in signature}}
lit.fn @twoSlash(%a: index, |, |, %b: index) {
  kgen.return
}

// -----

// expected-error @below {{only one '*' allowed in signature}}
lit.fn @twoStar(%a: index, *, *, %b: index) {
  kgen.return
}

// -----

// expected-error @below {{'*' cannot precede '|' in signature}}
lit.fn @slashAfterStar(%a: index, *, |, %b: index) {
  kgen.return
}

// -----

// expected-error @+1 {{expected variadic kind, got: stuff}}
lit.fn @incorrect_arg_variadicness(%a: index read|stuff) {
  kgen.return
}

// -----

// expected-error @+1 {{expected convention|variadicness, got: stuff}}
lit.fn @incorrect_arg_conv_and_variadicness(%a: index stuff) {
  kgen.return
}

// -----

// expected-error @+1 {{expected variadic kind, got: stuff}}
lit.fn @incorrect_param_variadicness<a: dtype stuff>() {
  kgen.return
}

// -----

// expected-error @below {{'lit.fn' expected positional parameter with default value}}
lit.fn @default_pos_params<a: dtype, b: dtype = f32, w: scalar<si32>>() attributes {isParametric} {
  %0 = kgen.param.constant: none = <#kgen.none>
  lit.end_fn
}

// -----

// expected-error @below {{'lit.fn' expected keyword-only parameter with default value}}
lit.fn @default_kw_only_params<a: dtype, b: dtype = f32, *, b: dtype = f16, w: scalar<si32>>() attributes {isParametric} {
  %0 = kgen.param.constant: none = <#kgen.none>
  lit.end_fn
}

// -----

kgen.generator @not_lit_func() {
  // expected-error @below {{'lit.return' op expected to be nested inside a `lit.fn` operation}}
  lit.return
  kgen.return
}

// -----

lit.fn @mismatched_return_types(%arg0: i64) -> i32 {
  // expected-error @below {{'lit.return' op operand #0 has type 'i64' but expected 'i32'}}
  lit.return %arg0 : i64
  lit.end_fn
}

// -----

lit.fn @does_not_throw() {
  // expected-error @below {{'lit.raise' op must be nested inside the 'try' region of a `lit.try` operation or a throwing function}}
  lit.raise
  lit.end_fn
}

// -----

lit.fn @not_async() {
  // expected-error @below {{'lit.async.call' op callable must be 'async'}}
  %0 = lit.async.call[() -> (): @not_async]()
  lit.end_fn
}

// -----

lit.fn @im_a_func() {
  kgen.return
}

lit.fn @struct_attr() {
  // expected-error @below {{invalid symbol use within this operator}}
  // expected-error @below {{struct attribute type @im_a_func does not refer to a struct declaration}}
  kgen.param.constant: @im_a_func = <#lit.struct<{}>>
  kgen.return
}

// -----

// expected-note @below {{see struct declaration here}}
lit.struct.decl @TwoFields {
  lit.struct.field a : index
  lit.struct.field b : index
}

lit.fn @struct_attr() {
  // expected-error @below {{invalid symbol use within this operator}}
  // expected-error @below {{struct declaration expected 2 fields but struct attribute has 0}}
  kgen.param.constant: @TwoFields = <#lit.struct<{}>>
  kgen.return
}

// -----

// expected-note @below {{see struct declaration here}}
lit.struct.decl @TwoFields {
  lit.struct.field a : index
  lit.struct.field b : index
}

lit.fn @struct_attr() {
  // expected-error @below {{invalid symbol use within this operator}}
  // expected-error @below {{struct attribute field name "c" at position #1 does not match the name "b" in the struct declaration}}
  kgen.param.constant: @TwoFields = <#lit.struct<{a = 1, c = 2}>>
  kgen.return
}

// -----

// expected-note @below {{see struct declaration here}}
lit.struct.decl @ParamField<ty: type> {
  lit.struct.field a : !kgen.param<ty>
}

lit.fn @struct_attr() {
  // expected-error @below {{invalid symbol use within this operator}}
  // expected-error @below {{struct attribute field #0 has type 'index' but corresponding struct field "a" expected 'i1'}}
  kgen.param.constant: @ParamField<:type i1> = <#lit.struct<{a = 5}>>
  kgen.return
}

// -----

lit.struct.decl @ParamField<ty: type> {
  lit.struct.field a : !kgen.param<ty>
}

lit.fn @struct_attr() {
  // expected-error @below {{'kgen.param.constant' op invalid use of parameter with no declaration "A"}}
  kgen.param.constant: @ParamField<:type i1> = <#lit.struct<{a: i1 = A}>>
  kgen.return
}

// -----

lit.fn @unbound_region() {
  // expected-error @below {{'lit.unbound_region' op is never valid. Was it not erased by the parser?}}
  "lit.unbound_region"() ({
  ^bb0(%arg0: index):
    hlcf.yield %arg0 : index
  }) : () -> ()
  kgen.return
}

// -----

// expected-error@below {{expected only `lit.file_module`, `lit.package`, `lit.unresolved_import`, or `lit.unresolved_wildcard_import` in its body}}
lit.package @MyPackage {
  // expected-note @below {{see operation defined here}}
  kgen.unreachable
}

// -----

lit.fn @declareWrongType() {
  // expected-error @below {{op declares a parameter with type 'index' but parameter expression has type 'i32'}}
  "lit.alias.decl"() {paramDecl = #kgen<param.decl p1 : index>, value = 1 : i32} : () -> ()
  kgen.return
}

// -----

lit.fn @wrong_error_return1(%arg0: i32) -> i1 {
  %0 = kgen.param.constant = <0>
  // expected-error @below {{'lit.error_return' op operand #0 has type 'index' but expected 'i1'}}
  lit.error_return %0 : index
}

// -----

lit.fn @wrong_error_return2(%arg0: i32) -> !kgen.variant<index> {
  %var = kgen.variant.create %arg0, 0 : <i32, index>
  // expected-error @below {{'lit.error_return' op operand #0 has type '!kgen.variant<i32, index>' but expected '!kgen.variant<index>'}}
  lit.error_return %var : !kgen.variant<i32, index>
}

// -----

// expected-error @below {{specified `declNameLoc` without `declName`}}
lit.unresolved_import @module as @newModule declNameLoc(loc(unknown))

// -----

// expected-error @below {{argument #0 with convention 'read_mem' in signature type should be a `!lit.ref` but got: 'index'}}
!type = !lit.generator<(index read_mem) -> ()>

// -----

// expected-error @below {{'?' cannot precede '|' in signature}}
!sig = !lit.generator<<?, |>() -> ()>

// -----

// expected-error @below {{'?' cannot precede '*' in signature}}
!sig = !lit.generator<<?, *>() -> ()>

// -----

// expected-error @below {{only one '?' allowed in signature}}
!sig = !lit.generator<<?, ?> -> ()>

// -----

// expected-error @below {{2 origins specified, but signature expected 1}}
lit.call @calls[imm a, mut b]() : !lit.generator<[1]() -> ()>

// -----

// expected-error @+1 {{custom op 'lit.call' implicit origin reference at depth 0 has an out-of-range index: 1 >= 1}}
lit.call @calls[mut a]() : !lit.generator<[1](!lit.ref<index, mut *[0,1]>) -> ()>

// -----

lit.fn @ref_immut<life: origin<0>>(%ref1: !lit.ref<index, imm life>) ->  !lit.ref<index, imm life> {
  %ref2 = lit.ref.immut %ref1: !lit.ref<index, imm life>
  kgen.return %ref2: !lit.ref<index, imm life>
}
