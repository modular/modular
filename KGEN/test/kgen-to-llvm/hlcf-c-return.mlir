// RUN: kgen-opt %s -lower-kgen-to-llvm=c-call=c_calling | FileCheck %s

kgen.export [@c_calling]

// CHECK-LABEL: llvm.func @c_calling(%arg0: i64, %arg1: !llvm.ptr<i64>) {
kgen.func @c_calling(%a: !pop.struct<index>) -> !pop.struct<index> {
  // CHECK: hlcf.loop
  hlcf.loop {
    // CHECK-NEXT: %[[V:.*]] = llvm.extractvalue
    // CHECK-NEXT: llvm.store %[[V]], %arg1
    // CHECK-NEXT: llvm.return
    hlcf.return %a : !pop.struct<index>
  }
  // CHECK: %[[V:.*]] = llvm.extractvalue
  // CHECK-NEXT: llvm.store %[[V]], %arg1
  // CHECK-NEXT: llvm.return
  kgen.return %a : !pop.struct<index>
}
