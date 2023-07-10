// RUN: kgen-opt -test-generate-elaborated-body %s | FileCheck %s --check-prefix=ATTACH
// RUN: kgen-opt -test-generate-elaborated-body -lower-lit %s | FileCheck %s --check-prefix=LOWER_LIT
// RUN: kgen-opt -test-generate-elaborated-body -lower-lit -lower-preelaborated-lit %s | FileCheck %s

module attributes {M.target_info = #M.target<triple="", cpu="", features="", data_layout="", simd_bit_width=128>} {
// The `doNotExtern` attribute is a marker for the test pass - it won't attempt to generate a kgen.func for this lit.func.
// ATTACH-LABEL: lit.func @caller
// ATTACH-NOT: postElaborationModuleRef
// CHECK-LABEL: kgen.generator @caller
lit.func @caller() -> index attributes {doNotExtern} {
  %0 = index.constant 32
  // CHECK: kgen.call @precompiled_func_elaborated
  %1 = kgen.call @precompiled_func(%0) : (index) -> index
  kgen.return %1 : index
}

// ATTACH: lit.package_link @link_precompiled_func
// ATTACH-SAME: post_elaboration(dense_resource<test_generated_post_elaboration_body_attr_{{[0-9]}}>
// ATTACH-LABEL: lit.func @precompiled_func
// ATTACH-SAME: postElaborationModuleRef = @link_precompiled_func

// LOWER_LIT-LABEL: lit.func @precompiled_func_elaborated
// LOWER_LIT-SAME: postElaborationModuleRef = @link_precompiled_func

// CHECK: kgen.link dense_resource<test_generated_post_elaboration_body_attr_{{[0-9]}}> {{.*}} as @link_precompiled_func
// CHECK-LABEL: kgen.func @precompiled_func_elaborated
lit.func @precompiled_func(%arg0: index) -> index {
  kgen.return %arg0 : index
}

// ATTACH: lit.package_link @link_different_precompiled_func
// ATTACH-SAME: post_elaboration(dense_resource<test_generated_post_elaboration_body_attr_{{[0-9]}}>
// ATTACH-LABEL: lit.func @different_precompiled_func
// ATTACH-SAME: postElaborationModuleRef = @link_different_precompiled_func

// LOWER_LIT-LABEL: lit.func @different_precompiled_func_elaborated
// LOWER_LIT-SAME: postElaborationModuleRef = @link_different_precompiled_func

// CHECK: kgen.link dense_resource<test_generated_post_elaboration_body_attr_{{[0-9]}}> {{.*}} as @link_different_precompiled_func
// CHECK-LABEL: kgen.func @different_precompiled_func_elaborated
lit.func @different_precompiled_func(%arg0: index) -> index {
  kgen.return %arg0 : index
}
}
