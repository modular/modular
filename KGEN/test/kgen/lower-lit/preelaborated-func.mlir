// REQUIRES: DISABLED
// (FIXME) clean up this test as follow up to simplify lower-lit for removing extern.generator.
// RUN: kgen-opt -test-generate-elaborated-body %s -o %t.0.mlir
// RUN: cat %t.0.mlir | FileCheck %s --check-prefix=ATTACH
// RUN: kgen-opt %t.0.mlir -lower-lit -o %t.1.mlir
// RUN: cat %t.1.mlir | FileCheck %s --check-prefix=LOWER_LIT
// RUN: kgen-opt %t.1.mlir -materialize-packages -o %t.2.mlir
// RUN: cat %t.2.mlir | FileCheck %s
// RUN: kgen-opt %t.2.mlir -elaborate-generators -o %t.3.mlir
// RUN: cat %t.3.mlir | FileCheck %s --check-prefix=ELAB

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

// Functions linked from other packages are "inflated" into the module being
// built. When this occurs, their `export` attribute is removed by the
// `-materialize-packages` pass.
// ATTACH: kgen.package.link @link_exported_func
// ATTACH-LABEL: lit.func export @exported_func
// ATTACH-SAME: preCompiledModuleRef = @link_exported_func

// LOWER_LIT-LABEL: kgen.extern.generator export @exported_func
// LOWER_LIT-SAME: preCompiledModuleRef = @link_exported_func

// CHECK: kgen.package.link @link_exported_func
// CHECK-SAME: post_parse(dense_resource<exported_func_generated_post_parse_attr
// CHECK-LABEL: kgen.generator @exported_func
// ELAB-LABEL: kgen.func @exported_func
lit.func export @exported_func(%arg0: index) -> index attributes {
  test.target.0 = #module_target
} {
  kgen.return %arg0 : index
}

// ATTACH: kgen.package.link @link_precompiled_func
// ATTACH-LABEL: lit.func @precompiled_func
// ATTACH-SAME: preCompiledModuleRef = @link_precompiled_func

// LOWER_LIT-LABEL: kgen.extern.generator @precompiled_func
// LOWER_LIT-SAME: preCompiledModuleRef = @link_precompiled_func

// CHECK: kgen.package.link @link_precompiled_func
// CHECK-LABEL: kgen.generator @precompiled_func
lit.func @precompiled_func(%arg0: index) -> index attributes {
  test.target.0 = #module_target
} {
  kgen.return %arg0 : index
}

// ATTACH: kgen.package.link @link_different_precompiled_func
// ATTACH-LABEL: lit.func @different_precompiled_func
// ATTACH-SAME: preCompiledModuleRef = @link_different_precompiled_func

// LOWER_LIT-LABEL: kgen.extern.generator @different_precompiled_func
// LOWER_LIT-SAME: preCompiledModuleRef = @link_different_precompiled_func

// CHECK: kgen.package.link @link_different_precompiled_func
// CHECK-LABEL: kgen.generator @different_precompiled_func
lit.func @different_precompiled_func(%arg0: index) -> index attributes {
  test.target.0 = #M.target<triple="", arch="", features="", data_layout="",
                            simd_bit_width=64>,
  test.target.1 = #module_target,
  test.target.2 = #M.target<triple="", arch="", features="", data_layout="",
                            simd_bit_width=256>
} {
  kgen.return %arg0 : index
}

// Check that we can still compile the module if none of the targets are
// correct, by using the pre-elaborated module.

// CHECK-LABEL: kgen.generator @compile_invalid_target
lit.func @compile_invalid_target(%arg0: index) -> index attributes {
  test.target.0 = #M.target<triple="", arch="", features="", data_layout="",
                            simd_bit_width=64>,
  test.target.1 = #M.target<triple="", arch="", features="", data_layout="",
                            simd_bit_width=256>
} {
  kgen.return %arg0 : index
}
}
