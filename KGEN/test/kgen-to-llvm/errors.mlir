// RUN: kgen-opt -convert-kgen-to-llvm -verify-diagnostics -split-input-file %s

// expected-error@+2 {{cannot lower a kernel that is not fully specified}}
// expected-error@+1 {{failed to legalize operation 'kgen.kernel'}}
kgen.kernel @trivial_kernel<() -> a>(%arg0: si32) -> si32 {
  kgen.return<a = 3> %arg0 : si32
}

// -----

// expected-error@-3 {{could not convert 'tensor<4xf32>' to be an llvm-compatible type}}
// expected-error@+2 {{could not convert region types to be LLVM-compatible}}
// expected-error@+1 {{failed to legalize operation 'kgen.kernel'}}
kgen.kernel @simd_unsupported(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  kgen.return %arg0 : tensor<4xf32>
}

// -----

kgen.kernel @unspecified_call() {
  // expected-error@+2 {{cannot lower a call op that is not fully specified}}
  // expected-error@+1 {{failed to legalize operation 'kgen.call'}}
  kgen.call @trivial_kernel<p1 = 4>() : () -> ()
  kgen.return
}

kgen.generator @trivial_kernel<p1>() {
  kgen.return
}
