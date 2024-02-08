// RUN: kgen-opt %s -verify-parameters -verify-diagnostics -split-input-file -o /dev/null

lit.struct.decl @SomeStruct {
  // expected-error @+1 {{invalid use of parameter with no declaration "ty"}}
  %size = lit.varlet.decl "size" var : !lit.ref<simd<1, ty>, mut lifetime>
}

// -----

// expected-error @below {{expected declaration body to have no arguments}}
"lit.struct.decl"() ({
^bb0(%arg0: i32):
}) {sym_name = "StructArgs",
    decorators = #kgen<decorators[]>,
    signature = !lit.type_signature,
    parentTypes = #lit<type_lineage.array[]>,
    params = #kgen<param.decls[]>
    } : () -> ()

// -----

// expected-error @below {{custom op 'lit.struct.decl' expected no result parameters}}
lit.struct.decl @StructReturns<() -> dtype> {}

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
kgen.generator @InvalidTypeParamValue<a>(%arg0: !kgen.declref<@SomeType<a, c>>) {
  kgen.return
}

// -----

lit.struct.decl @Bar<a: type> {
  lit.struct.field x : !pop.array<32, a>
}

kgen.generator @invalid_field_type<c: type>(%a: !kgen.paramref<c>) {
  // expected-error @below {{perand #0 has type '!kgen.paramref<c>' but corresponding struct field "x" expected '!pop.array<32, index>'}}
  %0 = lit.struct.create(x=%a) : (!kgen.paramref<c>) -> !kgen.declref<@Bar<:type index>>
  kgen.return
}

// -----

lit.struct.decl @Baz {
  lit.struct.field x : i32
}

kgen.generator @invalid_field_name(%a: i32) {
  // expected-error @below {{'lit.struct.create' op the field name "y" at the position #0 did not match the name "x" in the op declaration}}
  %0 = lit.struct.create(y=%a) : (i32) -> !kgen.declref<@Baz>
  kgen.return
}

// -----

lit.struct.decl @Bar {
}

kgen.generator @invalid_num_fields(%a: index) {
  // expected-error @below {{'lit.struct.create' op expected 0 operands but got 1}}
  %0 = lit.struct.create(a=%a) : (index) -> !kgen.declref<@Bar>
  kgen.return
}

// -----

lit.struct.decl @Bar {}

kgen.generator @invalid_field_name(%a: index, %container: !kgen.declref<@Bar>) {
  // expected-error @below {{struct @Bar has no field named "a"}}
  %0 = lit.struct.insert %a, %container[a] : index into !kgen.declref<@Bar>
  kgen.return
}

// -----

lit.struct.decl @Bar {
  lit.struct.field a : i32
}

kgen.generator @invalid_field_name(%a: index, %container: !kgen.declref<@Bar>) {
  // expected-error @below {{cannot insert value of type 'index' into struct field "a" which expected 'i32'}}
  %0 = lit.struct.insert %a, %container[a] : index into !kgen.declref<@Bar>
  kgen.return
}

// -----

lit.struct.decl @Bar {}

kgen.generator @invalid_field_name(%a: index, %container: !kgen.declref<@Bar>) {
  // expected-error @below {{struct @Bar has no field named "a"}}
  %0 = lit.struct.extract %container[a] : index from !kgen.declref<@Bar>
  kgen.return
}

// -----

// expected-error @below {{expected SSA operand}}
lit.func @no_names(index)

// -----

// expected-error @below {{only one '|' allowed in signature}}
lit.func @twoSlash(%a: index, |, |, %b: index) {
  kgen.return
}

// -----

// expected-error @below {{only one '*' allowed in signature}}
lit.func @twoStar(%a: index, *, *, %b: index) {
  kgen.return
}

// -----

// expected-error @below {{'*' cannot precede '|' in signature}}
lit.func @slashAfterStar(%a: index, *, |, %b: index) {
  kgen.return
}

// -----

// expected-error @below {{'lit.func' expected positional parameter with default value}}
lit.func @default_pos_params<a: dtype, b: dtype = f32, w: scalar<si32>>() attributes {isParametric} {
  %0 = kgen.param.constant: none = <#kgen.none>
  lit.end_func
}

// -----

// expected-error @below {{'lit.func' expected keyword-only parameter with default value}}
lit.func @default_kw_only_params<a: dtype, b: dtype = f32, *, b: dtype = f16, w: scalar<si32>>() attributes {isParametric} {
  %0 = kgen.param.constant: none = <#kgen.none>
  lit.end_func
}

// -----

kgen.generator @not_lit_func() {
  // expected-error @below {{'lit.return' op expected to be nested inside a `lit.func` operation}}
  lit.return
  kgen.return
}

// -----

lit.func @mismatched_return_types(%arg0: i64) -> i32 {
  // expected-error @below {{'lit.return' op operand #0 has type 'i64' but expected 'i32'}}
  lit.return %arg0 : i64
  lit.end_func
}

// -----

lit.func @does_not_throw(%err: !kgen.declref<@Error>) {
  // expected-error @below {{'lit.raise' op must be nested inside the 'try' region of a `lit.try` operation}}
  lit.raise %err : <@Error>
  lit.end_func
}

// -----

lit.func @not_async() {
  // expected-error @below {{'lit.async.call' op callable must be 'async'}}
  %0 = lit.async.call[() -> (): @not_async]()
  lit.end_func
}

// -----

lit.func @im_a_func() {
  kgen.return
}

lit.func @struct_attr() {
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

lit.func @struct_attr() {
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

lit.func @struct_attr() {
  // expected-error @below {{struct attribute field name "c" at position #1 does not match the name "b" in the struct declaration}}
  kgen.param.constant: @TwoFields = <#lit.struct<{a = 1, c = 2}>>
  kgen.return
}

// -----

// expected-note @below {{see struct declaration here}}
lit.struct.decl @ParamField<ty: type> {
  lit.struct.field a : !kgen.paramref<ty>
}

lit.func @struct_attr() {
  // expected-error @below {{struct attribute field #0 has type 'index' but corresponding struct field "a" expected 'i1'}}
  kgen.param.constant: @ParamField<:type i1> = <#lit.struct<{a = 5}>>
  kgen.return
}

// -----

lit.struct.decl @ParamField<ty: type> {
  lit.struct.field a : !kgen.paramref<ty>
}

lit.func @struct_attr() {
  // expected-error @below {{'kgen.param.constant' op invalid use of parameter with no declaration "A"}}
  kgen.param.constant: @ParamField<:type i1> = <#lit.struct<{a: i1 = A}>>
  kgen.return
}

// -----

lit.func @unbound_region() {
  // expected-error @below {{'lit.unbound_region' op is never valid. Was it not erased by the parser?}}
  "lit.unbound_region"() ({
  ^bb0(%arg0: index):
    hlcf.yield %arg0 : index
  }) : () -> ()
  kgen.return
}

// -----

lit.func @no_struct_decl(%a: index) {
  // expected-error @below {{expected to find a struct decl for '!kgen.declref<@Bar<:type index>>'}}
  %0 = lit.struct.create(x=%a) : (index) -> !kgen.declref<@Bar<:type index>>
  lit.end_func
}

// -----

lit.func @caller() -> !kgen.none attributes {isParametric} {
  lit.try {
    %i = index.constant 0
    // expected-error @below {{'lit.handle_variant' op operand #0 must be A parametric variant type., but got 'index'}}
    %4 = lit.handle_variant %i, %i : (index, index) -> !kgen.none {
      kgen.unreachable
    } else {
      kgen.unreachable
    }
    lit.try.yield
  } except (%arg0: !kgen.declref<@Error>) {
    lit.try.yield
  } else {
    lit.try.yield
  }
  %6 = kgen.param.constant: none = <#kgen.none>
  kgen.return %6 : !kgen.none
}

// -----

lit.func @throwing_caller(%0: !kgen.variant<@Error, index, !kgen.none>) throws -> !kgen.variant<@Error, none> attributes {isParametric} {
    %y = lit.varlet.decl "y" let : !lit.ref<@MyStruct, mut *"lifetime">
    %yp = lit.ref.to_pointer %y : !lit.ref<@MyStruct, mut *"lifetime">
    // expected-error @below {{'lit.handle_variant' op expected the variant to have two types: a success type and an error type}}
    %1 = lit.handle_variant %0, %yp : (!kgen.variant<@Error, index, !kgen.none>, !kgen.pointer<@MyStruct>) -> !kgen.none
    {
      kgen.unreachable
    } else {
      kgen.unreachable
    }
    %6 = kgen.param.constant: none = <#kgen.none>
    kgen.return %6 : !kgen.none
}

// -----

// expected-error@below {{expected external function body to contain a single `lit.extern_func`}}
lit.func @post_elaboration() attributes {preCompiledModuleRef = @package} {
  lit.end_func
}

// -----

lit.func @non_external() {
  // expected-error@below {{expected an external parent function}}
  lit.extern_func
}

// -----

// expected-error@below {{expected only `lit.file_module`, `lit.package`, `lit.unresolved_import`, or `lit.unresolved_wildcard_import` in its body}}
lit.package @MyPackage {
  // expected-note @below {{see operation defined here}}
  kgen.unreachable
}

// -----

lit.func @declareWrongType() {
  // expected-error @below {{op declares a parameter with type 'index' but parameter expression has type 'i32'}}
  "lit.alias.decl"() {paramDecl = #kgen<param.decl p1 : index>, value = 1 : i32} : () -> ()
  kgen.return
}

// -----

lit.func @wrong_error_return1(%arg0: i32) -> !kgen.variant<index> {
  %var = kgen.variant.create %arg0, 0 : <i32>
  // expected-error @below {{'lit.error_return' op expected two types in the variant: an error type and a success type}}
  lit.error_return %var : !kgen.variant<i32>
}

// -----

lit.func @wrong_error_return2(%arg0: i32) -> !kgen.variant<index> {
  %var = kgen.variant.create %arg0, 0 : <i32, index>
  // expected-error @below {{'lit.error_return' op operand #0 has type '!kgen.variant<i32, index>' but expected '!kgen.variant<index>'}}
  lit.error_return %var : !kgen.variant<i32, index>
}

// -----

// expected-error @below {{specified `declNameLoc` without `declName`}}
lit.unresolved_import @module as @newModule declNameLoc(loc(unknown))

// -----

lit.func @f() -> !kgen.none {
  // expected-error @below {{'lit.trait_func' op expected a parent function in a trait}}
  lit.trait_func
}

// -----

// expected-error @below {{argument #0 with convention 'borrow_in_mem' in signature type should be a `!kgen.pointer` or `!lit.ref` but got: 'index'}}
!type = !lit.signature<(index borrow_in_mem) -> ()>

// -----

// expected-error @below {{'?' cannot precede '|' in signature}}
!sig = !lit.signature<<?, |>() -> ()>

// -----

// expected-error @below {{'?' cannot precede '*' in signature}}
!sig = !lit.signature<<?, *>() -> ()>

// -----

// expected-error @below {{only one '?' allowed in signature}}
!sig = !lit.signature<<?, ?> -> ()>

// -----

// expected-error @below {{2 lifetimes specified, but signature expected 1}}
lit.call @calls[imm a, mut b]() : !lit.signature<[1]() -> ()>

// -----

// expected-error @+1 {{custom op 'lit.call' implicit lifetime reference at depth 0 has an out-of-range index: 1 >= 1}}
lit.call @calls[mut a]() : !lit.signature<[1](!lit.ref<index, mut *[0,1]>) -> ()>

// -----
lit.func @ref_immut<life: lifetime<0>>(%ref1: !lit.ref<index, imm life>) ->  !lit.ref<index, imm life> {
  %ref2 = lit.ref.immut %ref1: !lit.ref<index, imm life>
  kgen.return %ref2: !lit.ref<index, imm life>
}
