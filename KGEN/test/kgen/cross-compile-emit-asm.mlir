// RUN: kgen -emit-asm -march skylake-avx512 %s | FileCheck %s
// REQUIRES: x86-registered-target

kgen.func export @return_zero() -> index {
  // CHECK: %eax
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}
