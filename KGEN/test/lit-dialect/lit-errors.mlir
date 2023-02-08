// RUN: kgen-opt %s -verify-diagnostics -split-input-file -o /dev/null

lit.struct.decl @SomeStruct {
  // expected-error @+1 {{invalid use of parameter with no declaration "ty"}}
  %size = lit.var.decl "size" : !pop.pointer<simd<1, ty>>
}

// -----

// expected-error @below {{expected declaration body to have no arguments}}
"lit.struct.decl"() ({
^bb0(%arg0: i32):
}) {sym_name = "StructArgs", constraints = #kgen<constraints[]>,
    inputParamDecls = #kgen<param.decls[]>
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

// expected-error @below {{'kgen.generator.interface' op invalid use of parameter with no declaration "c"}}
kgen.generator.interface @InvalidTypeParamValue<a>() -> !kgen.declref<@SomeType<v = a, b = c>>

// -----

// expected-note @below {{@SomeType declared here}}
lit.struct.decl @SomeType<v, d> {}

// expected-error @below {{!kgen.declref symbol use input parameter #1 has name "b" but @SomeType expected name "d"}}
kgen.generator.interface @InvalidTypeParamValue<a, c>() ->
    !kgen.declref<@SomeType<v = a, b = c>>

// -----

lit.struct.decl @Bar<a: type> {
  lit.struct.field x : !pop.array<32, a>
}

kgen.generator @invalid_field_type<c: type>(%a: !kgen.paramref<c>) {
  // expected-error @below {{perand #0 has type '!kgen.paramref<c>' but corresponding struct field "x" expected '!pop.array<32, index>'}}
  %0 = lit.struct.create(x=%a) : (!kgen.paramref<c>) -> !kgen.declref<@Bar<a: type = index>>
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

// expected-error @below {{custom op 'lit.func' arguments requires SSA names}}
lit.func @no_names(index) {isInterface}

// -----

kgen.generator @not_lit_func() {
  // expected-error @below {{'lit.return' op expected to be nested inside a `lit.func` operation}}
  lit.return
  kgen.return
}

// -----

lit.func @mismatched_return_types(%arg0: i64) -> i32 {
  // expected-error @below {{'lit.return' op operand #0 has type 'i64' but should be 'i32'}}
  lit.return %arg0 : i64
  lit.end_func
}

// -----

lit.func @mismatched_result_parameter<() -> r1: i32>() {
  // expected-error @below {{'lit.return' op parameter #0 has type 'i64' but should be 'i32'}}
  lit.return<:i64 0>
  lit.end_func
}

// -----

lit.func @does_not_throw(%err: !kgen.declref<@Error>) {
  // expected-error @below {{'lit.raise' op must be nested inside the 'try' region of a `lit.try` operation or within a `lit.func` that throws}}
  lit.raise %err : <@Error>
  lit.end_func
}

// -----

lit.func @invalid_break() {
  // expected-error @below {{'lit.break' op must be nested within an `hlcf.loop` operation}}
  lit.break
  lit.end_func
}

// -----

lit.func @invalid_continue() {
  // expected-error @below {{'lit.continue' op must be nested within an `hlcf.loop` operation}}
  lit.continue
  lit.end_func
}

// -----

// expected-error @+1 {{argument #0 has type 'f32' but default argument has type 'index'}}
lit.func @mismatched_default_argument_type(%a: f32 = 1) {
  lit.end_func
}

// -----

lit.func @not_async() {
  // expected-error @below {{'lit.async_call' op callable must be 'async'}}
  %0 = lit.async_call[() -> (): @not_async]()
  lit.end_func
}
