// RUN: kgen-opt -convert-kgen-to-llvm -verify-diagnostics -split-input-file %s

// expected-error@+2 {{cannot lower a kernel that is not fully specified}}
// expected-error@+1 {{failed to legalize operation 'kgen.kernel'}}
kgen.kernel @trivial_kernel<() -> a>(%arg0: si32) -> si32 {
  kgen.return<a = 3> %arg0 : si32
}

// -----
// expected-error@-2 {{could not convert '!meta.simd<4, f32>' to be an llvm-compatible type}}
// expected-error@+2 {{could not convert region types to be LLVM-compatible}}
// expected-error@+1 {{failed to legalize operation 'kgen.kernel'}}
kgen.kernel @simd_unsupported(%arg0: !meta.simd<4, f32>) -> !meta.simd<4, f32> {
  kgen.return %arg0 : !meta.simd<4, f32>
}
