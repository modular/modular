// UNSUPPORTED: windows
// RUN: kgen --emit %s -o %t -L "$MODULAR_DERIVED_PATH/build/lib" && llvm-objdump -t %t | FileCheck %s

kgen.link "libKGENCompilerRT.a" as @CompilerRT

// kgen.extern.func doesn't pass through the elaborator, so we have to provide a dummy implementation.
kgen.func @KGEN_CompilerRT_Initialize() -> i1 no_inline attributes {precompiledBodyRef = @CompilerRT} {
  %0 = kgen.param.constant : i1 = <0>
  kgen.return %0 : i1
}

kgen.generator export C @main() -> i1 {
  %0 = kgen.call @KGEN_CompilerRT_Initialize(): () -> i1
  kgen.return %0 : i1
}

// CHECK-LABEL: Initialize.cpp.o
// CHECK: KGEN_CompilerRT_Initialize

// CHECK-LABEL: 0.o
// CHECK: main
