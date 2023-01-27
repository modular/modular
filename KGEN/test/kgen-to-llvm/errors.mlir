// RUN: kgen-opt -lower-kgen-to-llvm -verify-diagnostics -split-input-file %s

// expected-error@+2 {{failed to convert func signature}}
// expected-error@+1 {{failed to legalize operation 'kgen.func'}}
kgen.func @unsupported(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  kgen.return %arg0 : tensor<4xf32>
}

// -----

kgen.func @inline() always_inline {
  kgen.return
}

kgen.func @call() {
  // expected-error@+2 {{failed to legalize operation 'kgen.call' that was explicitly marked illegal}}
  // expected-error@+1 {{cannot lower call with always_inline, it should have been removed by now}}
  kgen.call @inline() : () always_inline -> ()
  kgen.return
}
