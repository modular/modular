// RUN: kgen-opt %s -split-input-file -pass-pipeline='builtin.module(lower-kgen-to-llvm,llvm.func(lower-pop-to-llvm,lower-scf-to-llvm),llvm.func(reconcile-unrealized-casts))' | FileCheck %s

// CHECK-LABEL: @variant_visit
// CHECK-SAME: %[[A:.*]]:
kgen.func @variant_visit(%a: !pop.variant<i32, f32>) -> !pop.scalar<si32> {
  // CHECK-NEXT: %[[CONTENT:.*]] = llvm.extractvalue %[[A]][0]
  // CHECK-NEXT: %[[V0:.*]] = llvm.extractvalue %[[CONTENT]][0]
  // CHECK-NEXT: %[[DISCR:.*]] = llvm.extractvalue %[[A]][1]
  // CHECK-NEXT: llvm.switch %[[DISCR]] : i1, ^bb2
  // CHECK-NEXT:   0: ^bb1
  %0 = pop.variant.visit %a : !pop.variant<i32, f32> -> !pop.scalar<si32>
  // CHECK: ^bb1:
  case (%v: i32) {
    // CHECK: %[[C0_i32:.*]] = llvm.mlir.constant(0 : i32)
    // CHECK: %[[P0:.*]] = llvm.trunc %{{.*}} : i64 to i32
    // CHECK-NEXT: llvm.or %[[C0_i32]], %[[P0]]
    // CHECK-NEXT: %[[R:.*]] = llvm.mlir.constant(0 :
    %1 = kgen.param.constant: !pop.scalar<si32> = <#pop.simd<0>>
    // CHECK-NEXT: llvm.br ^bb3(%[[R]]
    pop.yield %1 : !pop.scalar<si32>
  }
  // CHECK: ^bb2:
  case (%v: f32) {
    // CHECK: %[[C0_i32:.*]] = llvm.mlir.constant(0 : i32)
    // CHECK: %[[P0:.*]] = llvm.trunc %{{.*}} : i64 to i32
    // CHECK-NEXT: %[[P1:.*]] = llvm.or %[[C0_i32]], %[[P0]]
    // CHECK-NEXT: llvm.bitcast %[[P1]] : i32 to f32
    // CHECK-NEXT: %[[R:.*]] = llvm.mlir.constant(1 :
    %1 = kgen.param.constant: !pop.scalar<si32> = <#pop.simd<1>>
    // CHECK-NEXT: llvm.br ^bb3(%[[R]]
    pop.yield %1 : !pop.scalar<si32>
  }
  // CHECK: ^bb3(%[[ARG:.*]]: i32
  // CHECK-NEXT: return %[[ARG]]
  kgen.return %0 : !pop.scalar<si32>
}

// -----

// CHECK-LABEL: @variant_visit
kgen.func @variant_visit(%a: !pop.variant<scalar<si32>, f32>) -> !pop.scalar<si32> {
  %0 = kgen.param.constant: !pop.scalar<si32> = <#pop.simd<1>>
  // CHECK: llvm.switch %{{.*}} : i1, ^bb2
  // CHECK-NEXT: 0: ^bb1
  %1 = pop.variant.visit %a : !pop.variant<scalar<si32>, f32> -> !pop.scalar<si32>
  // CHECK: ^bb1:
  case (%v: !pop.scalar<si32>) {
    // CHECK: %[[C0_i32:.*]] = llvm.mlir.constant(0 : i32)
    // CHECK: %[[V:.*]] = llvm.or %[[C0_i32]], %{{.*}}
    // CHECK-NEXT: llvm.add %[[V]], %{{.*}} : i32
    %2 = pop.add %v, %0 : !pop.scalar<si32>
    pop.yield %2 : !pop.scalar<si32>
  }
  // CHECK: ^bb2:
  default {
    %2 = pop.sub %0, %0 : !pop.scalar<si32>
    pop.yield %2 : !pop.scalar<si32>
  }
  kgen.return %1 : !pop.scalar<si32>
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
