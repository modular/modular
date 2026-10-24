// RUN: mef-translate -export-mef -allow-unregistered-dialect %s -o %t
// RUN: mef-translate -import-mef -allow-unregistered-dialect %t | FileCheck %s

// CHECK: module {

// CHECK-LABEL: "mef.callable" ["trivial"]
// CHECK-NEXT:    %{{.*}}:2 = mef.opaque "simple.primitive" []() : () -> (!mef.unknown, !mef.unknown)
// CHECK-NEXT:    mef.opaque "grt.invoke" [@trivial]() : () -> ()
// CHECK-NEXT:    mef.output
// CHECK-NEXT:  }
mef.callable @trivial() {
  %a, %b = "simple.primitive"() : () -> (i8, i16)
  grt.invoke @trivial() : () -> ()
  mef.output
}

// CHECK-LABEL: "mef.callable" ["trivial2"]
// CHECK-NEXT:    %{{.*}}:2 = mef.opaque "simple.primitive2" []() : () -> (!mef.unknown, !mef.unknown)
// CHECK-NEXT:    mef.opaque "grt.invoke" [@trivial2]() : () -> ()
// CHECK-NEXT:    mef.output
// CHECK-NEXT:  }
mef.callable @trivial2() {
  %a, %b = "simple.primitive2"() : () -> (i32, i64)
  grt.invoke @trivial2() : () -> ()
  mef.output
}

// CHECK-NEXT: }
