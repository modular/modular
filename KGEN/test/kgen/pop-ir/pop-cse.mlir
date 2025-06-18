// RUN: kgen-opt %s -cse | FileCheck %s

// CHECK-LABEL: @cse_intrinsic
kgen.func @cse_intrinsic() -> !pop.scalar<f32> {
  // CHECK-COUNT-1: pop.call_llvm_intrinsic side_effecting<0> "no_side_effect_intrinsic", ()
  %0 = pop.call_llvm_intrinsic side_effecting<0> "no_side_effect_intrinsic", () : () -> !pop.scalar<f32>
  %1 = pop.call_llvm_intrinsic side_effecting<0> "no_side_effect_intrinsic", () : () -> !pop.scalar<f32>
  %2 = pop.add %0, %1 : !pop.scalar<f32>
  // CHECK-COUNT-2: pop.call_llvm_intrinsic "side_effect_intrinsic", ()
  %3 = pop.call_llvm_intrinsic "side_effect_intrinsic", (): () -> !pop.scalar<f32>
  %4 = pop.call_llvm_intrinsic "side_effect_intrinsic", (): () -> !pop.scalar<f32>
  %5 = pop.add %3, %4 : !pop.scalar<f32>
  %6 = pop.add %5, %2 : !pop.scalar<f32>
  kgen.return %6 : !pop.scalar<f32>
}
