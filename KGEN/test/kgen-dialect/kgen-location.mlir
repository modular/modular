// RUN: kgen-opt %s -mlir-print-debuginfo | FileCheck %s
// RUN: kgen-opt %s | FileCheck %s --check-prefix=NDEBUG

// COM: Check that location info is correctly propagated for sugared ops.

// CHECK: #[[LOC1:.*]] = loc("im_a_region")

kgen.generator @signature_call<fn: () -> ()>() {
  kgen.return
}

// CHECK-LABEL: @call_with_region
kgen.generator @call_with_region() {
  // CHECK: kgen.call @signature_call
  kgen.call @signature_call<fn: () -> () = region>() : () -> ()
  // CHECK-NEXT: fn()
  // CHECK-NEXT: kgen.return
  // CHECK-NEXT: } loc(#[[LOC1]])
  // NDEBUG-NOT: loc(#{{.*}})
  fn() { // kgen.region.body
    kgen.return
  } loc("im_a_region")
  kgen.return
}

// CHECK-LABEL: kgen.struct.decl @Struct
kgen.struct.decl @Struct {
  // CHECK-NEXT: kgen.struct.field x : i32 loc(#[[LOC2:.*]])
  // NDEBUG-NOT: loc(#{{.*}})
  kgen.struct.field x : i32 loc("im_a_field")
}

// CHECK: #[[LOC2]] = loc("im_a_field")
