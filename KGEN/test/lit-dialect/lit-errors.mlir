// RUN: kgen-opt %s -verify-diagnostics -split-input-file -o /dev/null

kgen.struct.decl @SomeStruct {
  // expected-error @+1 {{invalid use of parameter with no declaration "ty"}}
  %size = lit.var.decl "size" : !pop.pointer<simd<1, ty>>
}

// -----

// expected-error @below {{custom op 'lit.func' arguments requires SSA names}}
lit.func @no_names(index) {isInterface}

// -----

kgen.generator @throws(%arg0: !pop.variant<i32, si32>) {
  // expected-error @below {{'lit.unwrap_or_propagate' op must be contained in a `lit.func`}}
  %0 = lit.unwrap_or_propagate %arg0 : <i32, si32>
  kgen.return
}

// -----

lit.func @doesntThrow(%arg0: !pop.variant<i32, si32>) {
  // expected-error @below {{'lit.unwrap_or_propagate' op cannot propagate error in a function that does not throw}}
  %0 = lit.unwrap_or_propagate %arg0 : <i32, si32>
  kgen.return
}
