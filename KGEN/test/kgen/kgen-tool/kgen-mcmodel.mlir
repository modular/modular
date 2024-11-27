// REQUIRES: x86_64-linux
// RUN: kgen -emit --mcmodel=medium --large-data-threshold=2 %s -o %t
// RUN: llvm-objdump %t -t | FileCheck %s

// COM: check that string constant is in .lrodata section
// (for any data size that's larger than large-data-threshold)
// CHECK: .lrodata
kgen.generator export @main() -> !kgen.string {
  %0 = kgen.param.constant : !kgen.string = <"I am a string.">
  kgen.return %0 : !kgen.string
}
