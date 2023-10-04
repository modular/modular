// RUN: kgen-opt -test-generate-elaborated-body %s | FileCheck %s --check-prefix=ATTACH
// RUN: kgen-opt -test-generate-elaborated-body -lower-lit %s | FileCheck %s --check-prefix=LOWER_LIT
// RUN: kgen-opt -test-generate-elaborated-body -lower-lit -materialize-packages %s | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
// The `doNotExtern` attribute is a marker for the test pass - it won't attempt to generate a kgen.func for this lit.func.
// ATTACH-LABEL: lit.func @caller
// ATTACH-NOT: preCompiledModuleRef
// CHECK-LABEL: kgen.generator @caller
lit.func @caller() -> index attributes {doNotExtern} {
  %0 = index.constant 32
  // CHECK: kgen.call @precompiled_func_precompiled
  %1 = kgen.call @precompiled_func(%0) : (index) -> index
  kgen.return %1 : index
}

// When the build target doesn't match the module target ("" in this case), func
// ops are "inflated." Inflated ops have their `export` attribute removed during
// pre-elaborated LIT lowering.
// ATTACH: kgen.package.link @link_exported_func
// ATTACH-SAME: archive(<target = {{.*}}, elaboratedModule = dense_resource<exported_func_generated_body_attr>
// ATTACH-LABEL: lit.func export @exported_func
// ATTACH-SAME: preCompiledModuleRef = @link_exported_func

// LOWER_LIT-LABEL: kgen.extern.generator export @exported_func_precompiled
// LOWER_LIT-SAME: preCompiledModuleRef = @link_exported_func

// CHECK: kgen.link dense_resource<exported_func_generated_body_attr> {{.*}} as @link_exported_func
// CHECK-LABEL: kgen.func @exported_func_precompiled
lit.func export @exported_func(%arg0: index) -> index {
  kgen.return %arg0 : index
}

// ATTACH: kgen.package.link @link_precompiled_func
// ATTACH-SAME: archive(<target = {{.*}}, elaboratedModule = dense_resource<precompiled_func_generated_body_attr>
// ATTACH-LABEL: lit.func @precompiled_func
// ATTACH-SAME: preCompiledModuleRef = @link_precompiled_func

// LOWER_LIT-LABEL: kgen.extern.generator @precompiled_func_precompiled
// LOWER_LIT-SAME: preCompiledModuleRef = @link_precompiled_func

// CHECK: kgen.link dense_resource<precompiled_func_generated_body_attr> {{.*}} as @link_precompiled_func
// CHECK-LABEL: kgen.func @precompiled_func_precompiled
lit.func @precompiled_func(%arg0: index) -> index {
  kgen.return %arg0 : index
}

// ATTACH: kgen.package.link @link_different_precompiled_func
// ATTACH-SAME: archive(<target = {{.*}}, elaboratedModule = dense_resource<different_precompiled_func_generated_body_attr>
// ATTACH-LABEL: lit.func @different_precompiled_func
// ATTACH-SAME: preCompiledModuleRef = @link_different_precompiled_func

// LOWER_LIT-LABEL: kgen.extern.generator @different_precompiled_func_precompiled
// LOWER_LIT-SAME: preCompiledModuleRef = @link_different_precompiled_func

// CHECK: kgen.link dense_resource<different_precompiled_func_generated_body_attr> {{.*}} as @link_different_precompiled_func
// CHECK-LABEL: kgen.func @different_precompiled_func_precompiled
lit.func @different_precompiled_func(%arg0: index) -> index {
  kgen.return %arg0 : index
}

// Check that we can still compile the module if the target is incorrect, by
// using the pre-elaboration module. We don't generate a link in this case,
// given the module is going to be fully compiled.

// CHECK-NOT: kgen.link dense_resource<compile_invalid_target_generated_body_attr> {{.*}} as @compile_invalid_target
// CHECK-LABEL: kgen.func @compile_invalid_target_precompiled
lit.func @compile_invalid_target(%arg0: index) -> index attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=256>}  {
  kgen.return %arg0 : index
}
}
