// RUN: kgen-opt %s -verify-diagnostics -split-input-file -o /dev/null

kgen.struct.decl @SomeStruct {
  // expected-error @+1 {{invalid use of parameter with no declaration "ty"}}
  %size = lit.var.decl "size" : !pop.pointer<simd<1, ty>>
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

lit.func @mismatched_result_parameter<() -> i32>() {
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
lit.func @mismatched_default_argument_type(%a: f32) attributes {defaults = #lit<default.arguments[<0 : index = 1 : index>]>} {
  lit.end_func
}

// -----

lit.func @not_async() {
  // expected-error @below {{'lit.async_call' op callable must be 'async'}}
  %0 = lit.async_call[() -> (): @not_async]()
  lit.end_func
}
