// RUN: kgen-opt -lower-lit %s | FileCheck %s

// CHECK-LABEL: kgen.generator @trivial_generator
// CHECK-SAME: (%[[ARG0:.*]]: si32) -> si32 {
// CHECK-NEXT:    kgen.return %[[ARG0]] : si32
// CHECK-NEXT:  }
lit.func @trivial_generator(%arg0: si32) -> si32 {
  kgen.return %arg0 : si32
}

// CHECK-LABEL: kgen.generator public @varDecl
// CHECK-SAME:  (%[[ARG0:.*]]: index) -> index {
// CHECK-NEXT:    %[[VAR_A:.*]] = pop.stack_allocation 1 x index
// CHECK-NEXT:    kgen.return %[[ARG0]] : index
// CHECK-NEXT:  }

lit.func public @varDecl(%arg0: index) -> index {
  %a = lit.var.decl "a" : <index>
  kgen.return %arg0 : index
}

kgen.generator.interface @add<ty: dtype>(%arg0: !pop.scalar<ty>, %arg1: !pop.scalar<ty>)
-> !pop.scalar<ty>

// This implementation is fine.
// CHECK-LABEL: kgen.generator @add_f32<ty: dtype>
// CHECK-SAME: (%[[ARG0:.*]]: !pop.scalar<ty>, %[[ARG1:.*]]: !pop.scalar<ty>) -> !pop.scalar<ty>
// CHECK-NEXT: constraints <[eq(:dtype ty, f32), "f32 feels great", #
// CHECK-NEXT: implements @add {
kgen.generator @add_f32<ty: dtype>(%arg0 : !pop.scalar<ty>, %arg1 : !pop.scalar<ty>) -> !pop.scalar<ty>
  constraints <[eq(:dtype ty, f32), "f32 feels great"]> implements @add {

  // CHECK: %[[V0:.*]] = pop.cast_to_builtin %[[ARG0]] : !pop.scalar<ty> to f32
  %0 = pop.cast_to_builtin %arg0 : !pop.scalar<ty> to f32
  // CHECK: %[[V1:.*]] = pop.cast_to_builtin %[[ARG1]] : !pop.scalar<ty> to f32
  %1 = pop.cast_to_builtin %arg1 : !pop.scalar<ty> to f32

  // CHECK: %[[V2:.*]] = llvm.fadd %[[V0]], %[[V1]] : f32
  %2 = llvm.fadd %0, %1 : f32

  // CHECK: %[[V3:.*]] = pop.cast_from_builtin %[[V2]] : f32 to !pop.scalar<ty>
  %3 = pop.cast_from_builtin %2 : f32 to !pop.scalar<ty>
  // CHECK: kgen.return %[[V3]]
  kgen.return %3 : !pop.scalar<ty>
}

lit.func @add_64<ty: dtype>(%arg0 : !pop.scalar<f64>, %arg1 : !pop.scalar<f64>)
    -> !pop.scalar<ty> implements @add {
  %0 = pop.cast_to_builtin %arg0 : !pop.scalar<f64> to f64
  %1 = pop.cast_to_builtin %arg1 : !pop.scalar<f64> to f64
  %2 = llvm.fadd %0, %1 : f64
  %3 = pop.cast_from_builtin %2 : f64 to !pop.scalar<ty>
  kgen.return %3 : !pop.scalar<ty>
}

// CHECK-LABEL: kgen.generator @add_64_thunk
// CHECK-SAME: %[[ARG0:.*]]: !pop.scalar<{{.*}}>, %[[ARG1:.*]]: !pop.scalar<{{.*}}>
// CHECK-NEXT:    constraints <[eq(:dtype ty, f64), "argument #0 specifies 'ty' = f64", #
// CHECK-NEXT:    implements @add {
// CHECK-NEXT:    %[[V0:.*]] = kgen.rebind %[[ARG0]]
// CHECK-NEXT:    %[[V1:.*]] = kgen.rebind %[[ARG1]]
// CHECK-NEXT:    %[[V2:.*]] = kgen.call @add_64<ty: dtype = ty>(%[[V0]], %[[V1]])

//===----------------------------------------------------------------------===//
// Infer argument constraints.
//===----------------------------------------------------------------------===//

kgen.generator.interface @bufitf<size, ty: dtype -> index>(!zap.buffer<size, ty>) -> index

// This implementation infers that the ty argument must be f32.
lit.func @arg_inf<size, ty: dtype -> index>(%arg0: !zap.buffer<size, f32>) -> index
  implements @bufitf {
  %0 = zap.buffer.size %arg0 : !zap.buffer<size, f32>
  kgen.return<add(size, 2)> %0 : index
}

// This causes synthesization of a thunk for impl1 that adapts the calling convention.

// CHECK-LABEL: kgen.generator @arg_inf_thunk<size, ty: dtype -> index>
// CHECK-SAME: (%[[ARG0:.*]]: !zap.buffer<size, ty>) -> index
// CHECK-NEXT:  constraints <[eq(:dtype ty, f32), "argument #0 specifies 'ty' = f32", #
// CHECK-NEXT:  implements @bufitf {
// CHECK-NEXT:    %[[V0:.*]] = kgen.rebind %[[ARG0]] : !zap.buffer<size, ty> to !zap.buffer<size, f32>
// CHECK-NEXT:    %[[V1:.*]] = kgen.call @arg_inf<size = size, ty: dtype = ty -> resultParam0>(%[[V0]]) : (!zap.buffer<size, f32>) -> index
// CHECK-NEXT:    kgen.return<resultParam0> %[[V1]] : index
// CHECK-NEXT:  }

// CHECK-LABEL: kgen.generator @arg_inf<size, ty: dtype -> index>
// CHECK-SAME: (%[[ARG0:.*]]: !zap.buffer<size, f32>) -> index
// CHECK-NEXT: constraints <[eq(:dtype ty, f32), "argument #0 specifies 'ty' = f32", #
// CHECK: %{{.*}} = zap.buffer.size %[[ARG0]] : !zap.buffer<size, f32>

//===----------------------------------------------------------------------===//
// Infer result constraints.
//===----------------------------------------------------------------------===//

kgen.generator.interface @returnbufItf<size, ty: dtype>(!zap.buffer<123, f32>) -> !zap.buffer<size, ty>

// This implementation infers that the ty argument must be f32 and size must be 123
lit.func @returnbufItf_impl(%a : !zap.buffer<123, f32>) -> !zap.buffer<123, f32>
  implements @returnbufItf {
  kgen.return %a : !zap.buffer<123, f32>
}

// This causes synthesization of a thunk for impl1 that adapts the calling convention.

// CHECK-LABEL: kgen.generator @returnbufItf_impl_thunk<size, ty: dtype>
// CHECK-SAME: (%[[ARG0:.*]]: !zap.buffer<123, f32>) -> !zap.buffer<size, ty>
// CHECK-NEXT:   constraints <
// CHECK-NEXT:     [eq(size, 123), "result #0 specifies 'size' = 123", #
// CHECK-NEXT:     [eq(:dtype ty, f32), "result #0 specifies 'ty' = f32", #
// CHECK-NEXT:   implements @returnbufItf {
// CHECK-NEXT:   %[[V0:.*]] = kgen.call @returnbufItf_impl<size = size, ty: dtype = ty>(%[[ARG0]]
// CHECK-NEXT:   %[[V1:.*]] = kgen.rebind %[[V0]] : !zap.buffer<123, f32> to !zap.buffer<size, ty>
// CHECK-NEXT:   kgen.return %[[V1]]
// CHECK-NEXT: }

// CHECK-LABEL: kgen.generator @returnbufItf_impl<size, ty: dtype>
// CHECK-SAME: (%[[ARG0:.*]]: !zap.buffer<123, f32>) -> !zap.buffer<123, f32>


//===----------------------------------------------------------------------===//
// Simplify constraints.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: kgen.generator @SetIntersect<
lit.func @SetIntersect<a, b>()
// CHECK-NEXT: constraints <
// CHECK-NEXT: [in(a, [8, 57]), "a is prime, and a is even", #
// CHECK-NEXT: [in(b, [7, 8]), "thing Y", #
  constraints <
    [in(a, [7, 8, 57]), "a is prime"],
    [in(a, [57, 8, 2]), "a is even"],

    [in(b, [7, 8, 57]), "thing X"],  // superset of B.
    [in(b, [7, 8]), "thing Y"]
   > {
  kgen.return
}
