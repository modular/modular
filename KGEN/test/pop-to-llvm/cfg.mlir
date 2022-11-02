// RUN: kgen-opt %s -split-input-file -pass-pipeline='lower-kgen-to-llvm,llvm.func(lower-pop-to-llvm,lower-scf-to-llvm),llvm.func(reconcile-unrealized-casts)' | FileCheck %s

// CHECK-LABEL: @variant_visit
// CHECK-SAME: %[[A:.*]]:
kgen.func @variant_visit(%a: !pop.variant<i32, f32>) -> !pop.simd<1, si32> {
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK-NEXT: %[[PTR:.*]] = llvm.alloca
  // CHECK-NEXT: %[[CONTENT:.*]] = llvm.extractvalue %[[A]][0]
  // CHECK-NEXT: llvm.intr.lifetime.start 8, %[[PTR]]
  // CHECK-NEXT: llvm.store %[[CONTENT]], %[[PTR]]
  // CHECK-NEXT: %[[DISCR:.*]] = llvm.extractvalue %[[A]][1]
  // CHECK-NEXT: llvm.switch %[[DISCR]] : i1, ^bb2
  // CHECK-NEXT:   0: ^bb1
  %0 = pop.variant.visit %a : !pop.variant<i32, f32> -> !pop.simd<1, si32>
  // CHECK: ^bb1:
  case (%v: i32) {
    // CHECK-NEXT: %[[VPTR:.*]] = llvm.bitcast %[[PTR]]
    // CHECK-NEXT: %[[V:.*]] = llvm.load %[[VPTR]]
    // CHECK-NEXT: llvm.intr.lifetime.end 8, %[[PTR]]
    // CHECK-NEXT: %[[R:.*]] = llvm.mlir.constant(0 :
    %1 = pop.constant(0 : si32) : !pop.simd<1, si32>
    // CHECK-NEXT: llvm.br ^bb3(%[[R]]
    pop.yield %1 : !pop.simd<1, si32>
  }
  // CHECK: ^bb2:
  case (%v: f32) {
    // CHECK-NEXT: %[[VPTR:.*]] = llvm.bitcast %[[PTR]]
    // CHECK-NEXT: %[[V:.*]] = llvm.load %[[VPTR]]
    // CHECK-NEXT: llvm.intr.lifetime.end 8, %[[PTR]]
    // CHECK-NEXT: %[[R:.*]] = llvm.mlir.constant(1 :
    %1 = pop.constant(1 : si32) : !pop.simd<1, si32>
    // CHECK-NEXT: llvm.br ^bb3(%[[R]]
    pop.yield %1 : !pop.simd<1, si32>
  }
  // CHECK: ^bb3(%[[ARG:.*]]: i32
  // CHECK-NEXT: return %[[ARG]]
  kgen.return %0 : !pop.simd<1, si32>
}

// -----

// CHECK-LABEL: @variant_visit
kgen.func @variant_visit(%a: !pop.variant<simd<1, si32>, f32>) -> !pop.simd<1, si32> {
  %0 = pop.constant(1 : si32) : !pop.simd<1, si32>
  // CHECK: %[[PTR:.*]] = llvm.alloca
  // CHECK: llvm.switch %{{.*}} : i1, ^bb2
  // CHECK-NEXT: 0: ^bb1
  %1 = pop.variant.visit %a : !pop.variant<simd<1, si32>, f32> -> !pop.simd<1, si32>
  // CHECK: ^bb1:
  case (%v: !pop.simd<1, si32>) {
    // CHECK-NEXT: llvm.bitcast
    %2 = pop.add %v, %0 : !pop.simd<1, si32>
    pop.yield %2 : !pop.simd<1, si32>
  }
  // CHECK: ^bb2:
  default {
    // CHECK-NEXT: llvm.intr.lifetime.end 8, %[[PTR]]
    %2 = pop.sub %0, %0 : !pop.simd<1, si32>
    pop.yield %2 : !pop.simd<1, si32>
  }
  kgen.return %1 : !pop.simd<1, si32>
}

// -----

// CHECK-LABEL: @variant_visit
kgen.func @variant_visit(%a: !pop.variant<i1, i2, i3, i4>) {
  // CHECK: llvm.switch %{{.*}} : i2, ^bb3
  // CHECK-NEXT: 1: ^bb1
  // CHECK-NEXT: 3: ^bb2
  pop.variant.visit %a : !pop.variant<i1, i2, i3, i4>
  case (%v: i2) {
    pop.yield
  }
  case (%v: i4) {
    pop.yield
  }
  default {
    pop.yield
  }
  kgen.return
}

// -----

// Ensure `pop.variant.visit` nested inside SCF ops can be lowered.

// CHECK-LABEL: @visit_in_if
kgen.func @visit_in_if(%cond: i1, %variant: !pop.variant<i32, i64>, %a: index, %b: index) -> index {
  %0 = scf.if %cond -> index {
    // CHECK-NOT: pop.variant.visit
    %1 = pop.variant.visit %variant : !pop.variant<i32, i64> -> index
    case (%v: i32) {
      pop.yield %a : index
    }
    case (%v: i64) {
      pop.yield %b : index
    }
    scf.yield %1 : index
  } else {
    scf.yield %a : index
  }
  kgen.return %0 : index
}
