// RUN: kgen-opt -test-generate-elaborated-body %s | FileCheck %s --check-prefix=ATTACH
// RUN: kgen-opt -test-generate-elaborated-body -lower-lit %s | FileCheck %s

// The `doNotExtern` attribute is a marker for the test pass - it won't attempt to generate a kgen.func for this lit.func.
// ATTACH-LABEL: lit.func @caller
// ATTACH-NOT: postElaborationBodyRef
// CHECK-LABEL: kgen.generator @caller
lit.func @caller() -> index attributes {doNotExtern} {
  %0 = index.constant 32
  // CHECK: kgen.call @precompiled_func_elaborated
  %1 = kgen.call @precompiled_func(%0) : (index) -> index
  kgen.return %1 : index
}

// ATTACH-LABEL: lit.func @precompiled_func
// ATTACH-SAME: postElaborationBodyRef = dense_resource<test_generated_post_elaboration_body_attr_{{[0-9]}}>
// CHECK-LABEL: kgen.func @precompiled_func_elaborated
lit.func @precompiled_func(%arg0: index) -> index {
  kgen.return %arg0 : index
}
