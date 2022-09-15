// RUN: kgen-opt %s -split-input-file -convert-kgen-to-llvm=break-up-structs="kernel" -verify-diagnostics

// expected-error @below {{cannot find func}}
module {}

// -----

// expected-error @below {{func is not top-level}}
kgen.func public @kernel() {
  kgen.return
}

kgen.func public @toplevel() {
  // expected-note @below {{callsite here}}
  kgen.call @kernel() : () -> ()
  kgen.return
}

// -----

// expected-error @below {{cannot break up structs of an external function}}
llvm.func @kernel() -> i32

// -----

// expected-warning@+1 {{will not emit wrappers for this function marked private}}
kgen.func @kernel(%arg0 : f32) -> f32 {
  kgen.return %arg0 : f32
}
