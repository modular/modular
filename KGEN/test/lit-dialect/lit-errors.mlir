// RUN: kgen-opt %s -verify-diagnostics -split-input-file -o /dev/null

kgen.struct.decl @SomeStruct {
  // expected-error @+1 {{invalid use of parameter with no declaration "ty"}}
  %size = lit.var.decl "size" : !pop.pointer<simd<1, ty>>
}

// -----
