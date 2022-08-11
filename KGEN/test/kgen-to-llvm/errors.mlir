// RUN: kgen-opt -convert-kgen-to-llvm -verify-diagnostics -split-input-file %s

// expected-error@+2 {{failed to convert kernel signature}}
// expected-error@+1 {{failed to legalize operation 'kgen.kernel'}}
kgen.kernel @simd_unsupported(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  kgen.return %arg0 : tensor<4xf32>
}
