// RUN: kgen-opt -lower-hlkgen %s | FileCheck %s

// CHECK-LABEL: kgen.generator @trivial_generator(%arg0: si32) -> si32 {
// CHECK-NEXT:    kgen.return %arg0 : si32
// CHECK-NEXT:  }
hlkgen.generator @trivial_generator(%arg0: si32) -> si32 {
  kgen.return %arg0 : si32
}


kgen.generator.interface @add<ty: dtype>(%arg0: !meta.scalar<ty>, %arg1: !meta.scalar<ty>)
-> !meta.scalar<ty>

// This implementation is fine.
// CHECK-LABEL: kgen.generator @add_f32<ty: dtype>(%arg0: !meta.scalar<ty>, %arg1: !meta.scalar<ty>) -> !meta.scalar<ty> 
// CHECK-NEXT: constraints <eq_dtype(ty, f32), "f32 feels great"> 
// CHECK-NEXT: implements @add { 
kgen.generator @add_f32<ty: dtype>(%arg0 : !meta.scalar<ty>, %arg1 : !meta.scalar<ty>) -> !meta.scalar<ty>
  constraints <eq_dtype(ty, f32), "f32 feels great"> implements @add {
  
  // CHECK: %0 = meta.cast_to_builtin %arg0 : !meta.scalar<ty> to f32 
  %0 = meta.cast_to_builtin %arg0 : !meta.scalar<ty> to f32
  // CHECK: %1 = meta.cast_to_builtin %arg1 : !meta.scalar<ty> to f32 
  %1 = meta.cast_to_builtin %arg1 : !meta.scalar<ty> to f32

  // CHECK: %2 = llvm.fadd %0, %1 : f32 
  %2 = llvm.fadd %0, %1 : f32

  // CHECK: %3 = meta.cast_from_builtin %2 : f32 to !meta.scalar<ty>
  %3 = meta.cast_from_builtin %2 : f32 to !meta.scalar<ty>
  // CHECK: kgen.return %3 
  kgen.return %3 : !meta.scalar<ty>
}