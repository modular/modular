// RUN: kgen-opt %s -verify-parameters -verify-diagnostics -split-input-file -o /dev/null

lit.struct.decl @SomeStruct {
  // expected-error @+1 {{invalid use of parameter with no declaration "ty"}}
  %size = lit.varlet.decl "size" var : !lit.ref<mut simd<1, ty>, *"lifetime">
}

// -----

// expected-error @below {{expected declaration body to have no arguments}}
"lit.struct.decl"() ({
^bb0(%arg0: i32):
}) {sym_name = "StructArgs",
    decorators = #kgen<decorators[]>,
    signature = !lit.type_signature,
    parentTypes = #lit<type_lineage.array[]>,
    inputParams = #kgen<param.decls[]>
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

lit.struct.decl @Bar<a: regtype> {
  lit.struct.field x : !pop.array<32, a>
}

kgen.generator @invalid_field_type<c: regtype>(%a: !kgen.paramref<c>) {
  // expected-error @below {{perand #0 has type '!kgen.paramref<c>' but corresponding struct field "x" expected '!pop.array<32, index>'}}
  %0 = lit.struct.create(x=%a) : (!kgen.paramref<c>) -> !kgen.declref<@Bar<:regtype index>>
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

// expected-error @below {{'lit.func' expected parameter with default value}}
lit.func @default_params<a: dtype, b: dtype = f32, w: scalar<si32>>() attributes {isParametric} {
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

lit.func @invalid_break() {
  // expected-error @below {{'lit.break' op must be nested within a `lit.loop` operation}}
  lit.break
  lit.end_func
}

// -----

lit.func @invalid_continue() {
  // expected-error @below {{'lit.continue' op must be nested within a `lit.loop` operation}}
  lit.continue
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
lit.struct.decl @ParamField<ty: regtype> {
  lit.struct.field a : !kgen.paramref<ty>
}

lit.func @struct_attr() {
  // expected-error @below {{struct attribute field #0 has type 'index' but corresponding struct field "a" expected 'i1'}}
  kgen.param.constant: @ParamField<:regtype i1> = <#lit.struct<{a = 5}>>
  kgen.return
}

// -----

lit.struct.decl @ParamField<ty: regtype> {
  lit.struct.field a : !kgen.paramref<ty>
}

lit.func @struct_attr() {
  // expected-error @below {{'kgen.param.constant' op invalid use of parameter with no declaration "A"}}
  kgen.param.constant: @ParamField<:regtype i1> = <#lit.struct<{a: i1 = A}>>
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

lit.func @bad_param_results<() -> r0: dtype>() {
  // expected-error @below {{'lit.param_return' op parameter #0 has type 'index' but should be '!kgen.dtype'}}
  lit.param_return<2>
  lit.end_func
}

// -----

lit.func @no_struct_decl(%a: index) {
  // expected-error @below {{expected to find a struct decl for '!kgen.declref<@Bar<:regtype index>>'}}
  %0 = lit.struct.create(x=%a) : (index) -> !kgen.declref<@Bar<:regtype index>>
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
    %y = lit.varlet.decl "y" let : !lit.ref<mut @MyStruct, *"lifetime">
    %yp = lit.ref.to_pointer %y : !lit.ref<mut @MyStruct, *"lifetime">
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

// expected-note @below {{see function here}}
lit.func @wrong_error_return(%arg0: i32) -> !kgen.variant<index> {
  %var = kgen.variant.create %arg0, 0 : <i32>
  // expected-error @below {{'lit.error_return' op operand #0 type '!kgen.variant<i32>' does not match expected result type '!kgen.variant<index>'}}
  lit.error_return %var : !kgen.variant<i32>
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

// expected-error @below {{'bind_type' expected a metatyped type value}}
#bind = #lit.bind_type<:regtype index, []> : !lit.metatype<@Foo>

// -----

// expected-error @below {{'bind_type' result metatype parameter #0 does not match corresponding input parameter}}
#bind = #lit.bind_type<:metatype<@Foo<?>, <index>> T, [?]> : !lit.metatype<@Foo<1>>

// -----

// expected-error @below {{'bind_type' result metatype should have 1 parameter values, but got 0}}
#bind = #lit.bind_type<:metatype<@Foo, <index>> T, [?]> : !lit.metatype<@Foo<1>>

// -----

// expected-error @below {{'bind_type' cannot change the value of parameter #0}}
#bind = #lit.bind_type<:metatype<@Foo<2>> T, []> : !lit.metatype<@Foo<1>>

// -----

// expected-error @below {{'bind_type' result metatype parameter #0 does not match corresponding input parameter}}
#bind = #lit.bind_type<:metatype<@Foo<?>, <index>> T, [2]> : !lit.metatype<@Foo<3>>

// -----

// expected-error @below {{'bind_type' result metatype signature should have 0 input parameters}}
#bind = #lit.bind_type<:metatype<@Foo<?>, <index>> T, [1]> : !lit.metatype<@Foo<1>, <index>>

// -----

// expected-error @below {{result signature parameter #0 expected to be 'index' but got '!kgen.dtype'}}
#bind = #lit.bind_type<:metatype<@Foo<?>, <index>> T, [?]> : !lit.metatype<@Foo<?>, <dtype>>

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
lit.call @calls[a, b]() : !lit.signature<[1]() -> ()>

// -----

// expected-error @+1 {{custom op 'lit.call' implicit lifetime reference at depth 0 has an out-of-range index: 1 >= 1}}
lit.call @calls[a]() : !lit.signature<[1](!lit.ref<mut index, *[0,1]>) -> ()>
