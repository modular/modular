// RUN: kgen-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: @rebind_folds
kgen.generator @rebind_folds<dtype: dtype, type: type>(
  %a: i32, %b: !pop.scalar<f32>, %c: !pop.scalar<dtype>, %d: !kgen.paramref<type>
) -> (
  i32, !pop.scalar<f32>, !pop.scalar<dtype>, !kgen.paramref<type>
) {
  // CHECK-NOT: kgen.rebind
  %0 = kgen.rebind %a : i32 to i32
  %1 = kgen.rebind %b : !pop.scalar<f32> to !pop.scalar<f32>
  %2 = kgen.rebind %c : !pop.scalar<dtype> to !pop.scalar<dtype>
  %3 = kgen.rebind %d : !kgen.paramref<type> to !kgen.paramref<type>
  kgen.return %0, %1, %2, %3 : i32, !pop.scalar<f32>, !pop.scalar<dtype>, !kgen.paramref<type>
}

// CHECK-LABEL: kgen.func @cast_from_folds
// CHECK-SAME: (%[[ARG0:.*]]: !pop.scalar<f32>) -> !pop.scalar<f32> {
kgen.func @cast_from_folds(%arg0: !pop.scalar<f32>) -> !pop.scalar<f32> {

  // A-B-A cast.
  %1 = pop.cast_to_builtin %arg0 : !pop.scalar<f32> to f32
  %2 = pop.cast_from_builtin %1 : f32 to !pop.scalar<f32>

  // CHECK: kgen.return %[[ARG0]]
  kgen.return %2 : !pop.scalar<f32>
}

// CHECK-LABEL: kgen.func @cast_to_folds
// CHECK-SAME: (%[[ARG0:.*]]: f32) -> f32 {
kgen.func @cast_to_folds(%arg0: f32) -> f32 {

  // A-B-A cast.
  %1 = pop.cast_from_builtin %arg0 : f32 to !pop.scalar<f32>
  %2 = pop.cast_to_builtin %1 : !pop.scalar<f32> to f32

  // CHECK: kgen.return %[[ARG0]]
  kgen.return %2 : f32
}

kgen.func @producesResultParam<() -> index>() {
  kgen.return<42>
}


// CHECK-LABEL: kgen.generator @param_assert_simplify<p1: i1, p2>()
// CHECK-NEXT: constraints <
// CHECK-NEXT:   [p1, "this is a constraint!", #
// CHECK-NEXT:   [eq(add(p2, 4), 17), "also a constraint", #
kgen.generator @param_assert_simplify<p1 : i1, p2>() {

  kgen.param.assert <p1>, "this is a constraint!"
  kgen.param.assert <eq(add(p2, 4), 17)>, "also a constraint"

  kgen.param.assert <1>, "this is pointless"

  // CHECK-NEXT:   kgen.param.assert <0>, "failing asserts must be kept"
  kgen.param.assert <eq(42, 41)>, "failing asserts must be kept"

  // CHECK-NEXT: kgen.call @producesResultParam
  kgen.call @producesResultParam<() -> result>() : () -> ()

  // CHECK-NEXT: kgen.param.assert <eq(result, 12)>, "this stays"
  kgen.param.assert <eq(result, 12)>, "this stays"
  kgen.return
}

kgen.func @trivial(%arg0: si32) -> si32 {
  kgen.return %arg0 : si32
}

kgen.generator @trivial_param<A>(%arg0: si32) -> si32 {
  kgen.return %arg0 : si32
}

// CHECK-LABEL: kgen.generator @call_param_canonicalize
kgen.generator @call_param_canonicalize(%arg0: si32) -> si32 {
  // CHECK: %0 = kgen.call @trivial(%arg0) : (si32) -> si32
  %0 = kgen.call_param[(si32) -> si32: @trivial](%arg0)
  // CHECK: %1 = kgen.call @trivial_param<A = 1>(%arg0)
  %1 = kgen.call_param[<A>(si32) -> si32: @trivial_param]<A = 1>(%arg0)
  kgen.return %0: si32
}

// CHECK-LABEL: kgen.generator @param_declare
// https://github.com/modularml/modular/issues/3042
kgen.generator @param_declare<simd_width, unroll_factor>() -> index {
  // CHECK: kgen.param.declare unroll_simd_size
  kgen.param.declare unroll_simd_size = <mul(simd_width, unroll_factor)>
  %result = kgen.param.constant = <unroll_simd_size>
  kgen.return %result : index
}

// -----

// Hoisting constants that reference parameters.
// https://github.com/modularml/modular/issues/4518

kgen.generator @callee<fn: <N>()->index>() {
  kgen.return
}

// CHECK-LABEL: @hoist_constant
kgen.generator @hoist_constant() {
  // CHECK-NEXT: call
  kgen.call @callee<fn: <N>()->index = region>() : () -> ()
  // CHECK-NEXT: fn<N>
  fn<N>() -> index {
    // CHECK-NEXT: kgen.param.constant = <N>
    %0 = kgen.param.constant = <N>
    kgen.return %0 : index
  }
  kgen.return
}

kgen.generator @call_me() {
  kgen.return
}

// CHECK-LABEL: @call_indirect_constant
kgen.generator @call_indirect_constant() {
  // CHECK-NEXT: kgen.call @call_me() : () -> ()
  %0 = kgen.param.constant: () -> () = <@call_me>
  kgen.call_indirect %0() : () -> ()
  kgen.return
}

// CHECK-LABEL: @call_indirect_partial_apply
kgen.generator @call_indirect_partial_apply(%fn: !kgen.signature<[], [], (index, i32) -> index>, %arg0: index, %arg1: i32) -> index {
  // CHECK-NEXT: %0 = kgen.call_indirect %arg0(%arg1, %arg2) : (index, i32) -> index
  %0 = kgen.partial_apply %fn(?, %arg1) : (index, i32) -> index
  %1 = kgen.call_indirect %0(%arg0) : (index) -> index
  // CHECK-NEXT: return %0
  kgen.return %1 : index
}

// CHECK-LABEL: @partial_apply_of_partial_apply
kgen.generator @partial_apply_of_partial_apply(%fn: !kgen.signature<[], [], (index, i32) -> index>, %arg0: index, %arg1: i32) -> !kgen.signature<[], [], () -> index> {
  // CHECK-NEXT: %0 = kgen.partial_apply %arg0(%arg1, %arg2) : (index, i32) -> index
  %0 = kgen.partial_apply %fn(?, %arg1) : (index, i32) -> index
  %1 = kgen.partial_apply %0(%arg0) : (index) -> index
  // CHECK-NEXT: return %0
  kgen.return %1 : !kgen.signature<[], [], () -> index>
}

kgen.generator @call_with_bound<A>() {
  kgen.return
}

// CHECK-LABEL: @call_param_bound_symbol
kgen.generator @call_param_bound_symbol() {
  // CHECK-NEXT: kgen.call @call_with_bound<A = 1>() : () -> ()
  kgen.call_param[() -> (): @call_with_bound<A = 1>]()
  kgen.return
}

// -----

kgen.struct.decl @Struct {
  lit.func @Nested() {
    kgen.return
  }
}

// CHECK-LABEL: @callNested
kgen.generator @callNested() {
  // CHECK-NEXT: kgen.call @Struct::@Nested
  kgen.call_param[() -> (): @Struct::@Nested]()
  kgen.return
}

// -----

kgen.generator @takeBody<fn: () -> ()>() {
  kgen.return
}

// CHECK-LABEL: @callParamWithBody
kgen.generator @callParamWithBody() {
  // CHECK-NEXT: kgen.call @takeBody<fn: () -> () = region>
  // CHECK-NEXT: fn() {
  kgen.call_param[<fn: () -> ()>() -> (): @takeBody]<fn: () -> () = region>()
  fn() {
    kgen.return
  }
  kgen.return
}
