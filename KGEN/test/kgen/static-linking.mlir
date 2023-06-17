// XFAIL: windows
// RUN: kgen --emit %s -o %t -L "$MODULAR_DERIVED_PATH/build/lib" && llvm-objdump -t %t | FileCheck %s

kgen.link "libKGENCompilerRT.a" as @CompilerRT

kgen.generator @main() -> i1 {
  %0 = pop.external_call @KGEN_CompilerRT_Initialize() from @CompilerRT : () -> i1
  kgen.return %0 : i1
}

kgen.export @main to C

// CHECK-LABEL: Initialize.cpp.o
// CHECK: KGEN_CompilerRT_Initialize

// CHECK-LABEL: 0.o
// CHECK: main
