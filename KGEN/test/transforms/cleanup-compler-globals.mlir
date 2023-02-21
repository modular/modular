// RUN: kgen-opt %s -cleanup-compiler-globals | FileCheck %s

// CHECK-LABEL: @useAGlobalForNoReason
kgen.func @useAGlobalForNoReason() -> index {
  // CHECK-NEXT: index.constant
  %idx0 = index.constant 0
  pop.compiler.global_store "aGlobal", %idx0 : index
  %0 = pop.compiler.global_load "aGlobal" : index
  // CHECK-NEXT: kgen.return
  kgen.return %0 : index
}

// COM: This should not remove a load that's not paired with a store.
// CHECK-LABEL: @loadOnly
kgen.func @loadOnly() -> index {
  // CHECK-NEXT: index.constant
  %idx0 = index.constant 0
  // CHECK-NEXT: pop.compiler.global_load
  %0 = pop.compiler.global_load "aGlobal" : index
  // CHECK-NEXT: kgen.return
  kgen.return %0 : index
}

// CHECK-LABEL: @storeOnly
kgen.func @storeOnly() {
  // CHECK-NEXT: index.constant
  %idx0 = index.constant 0
  // CHECK-NOT: pop.compiler.global_store
  pop.compiler.global_store "aGlobal", %idx0 : index
  // CHECK-NEXT: kgen.return
  kgen.return
}

// CHECK-LABEL: @multiLoad
kgen.func @multiLoad() -> (index, index) {
  // CHECK-NEXT: index.constant
  %idx0 = index.constant 0
  pop.compiler.global_store "aGlobal", %idx0 : index
  %0 = pop.compiler.global_load "aGlobal" : index
  %1 = pop.compiler.global_load "aGlobal" : index
  // CHECK-NEXT: kgen.return
  kgen.return %0, %1 : index, index
}

// CHECK-LABEL: @multiLoadNested
kgen.func @multiLoadNested(%pred : i1) -> index {
  // CHECK-NEXT: index.constant
  %idx0 = index.constant 0
  pop.compiler.global_store "aGlobal", %idx0 : index
  %0 = pop.compiler.global_load "aGlobal" : index
  // CHECK-NEXT: hlcf.if %arg0 {
  hlcf.if %pred {
    // CHECK-NOT: pop.compiler.global_load
    %1 = pop.compiler.global_load "aGlobal" : index
    // CHECK-NEXT: hlcf.yield
    hlcf.yield
  } else {
    hlcf.yield
  }
  // CHECK: kgen.return
  kgen.return %0: index
}
