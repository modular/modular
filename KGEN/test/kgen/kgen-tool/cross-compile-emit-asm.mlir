// RUN: kgen -emit-asm -march skylake-avx512 %s | FileCheck %s
// RUN: kgen -emit-asm-verbose -march skylake-avx512 %s | FileCheck %s --check-prefix=CHECK-VERBOSE

kgen.func export @return_zero() -> index {
  // CHECK: %eax
  // CHECK-VERBOSE: {{.*}}KGEN_EE_JIT_GlobalConstructor  {{.*}} -- Begin function {{.*}}KGEN_EE_JIT_GlobalConstructor
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}
