// RUN: kgen-opt -test-generate-elaborated-body %s -o %t.0.mlir
// RUN: cat %t.0.mlir | FileCheck %s --check-prefix=ATTACH
// RUN: kgen-opt %t.0.mlir -lower-lit -o %t.1.mlir
// RUN: kgen-opt %t.1.mlir -elaborate-generators -o %t.2.mlir
// RUN: cat %t.2.mlir | FileCheck %s --check-prefix=ELAB
// COM: this test is probably not needed now that MaterializePackage pass is gone.

#module_target = #M.target<triple="", arch="", features="", data_layout="",
                        simd_bit_width=128>

module {
// The `doNotExtern` attribute is a marker for the test pass - it won't attempt to generate a kgen.func for this lit.func.
// ATTACH-LABEL: lit.func @caller
// ATTACH-NOT: preCompiledModuleRef
// CHECK-LABEL: kgen.generator @caller
// ELAB-LABEL: kgen.func @caller
lit.func @caller() -> index attributes {test.target.0 = #module_target, doNotExtern} {
  %0 = index.constant 32
  // CHECK: kgen.call @precompiled_func
  // ELAB: kgen.call @precompiled_func
  %1 = kgen.call @precompiled_func(%0) : (index) -> index
  kgen.return %1 : index
}

lit.func @precompiled_func(%arg0: index) -> index attributes {
  test.target.0 = #module_target, doNotExtern
} {
  kgen.return %arg0 : index
}
}
