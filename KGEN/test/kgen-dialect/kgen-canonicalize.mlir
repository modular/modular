// RUN: kgen-opt -canonicalize -mlir-print-debuginfo -split-input-file %s | FileCheck %s

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

// CHECK-LABEL: @rebind_canonicalize
kgen.generator @rebind_canonicalize<dt1: dtype, dt2: dtype, dt3: dtype>(%arg0: !pop.scalar<dt1>) -> !pop.scalar<si32> {
  // CHECK-NEXT: %0 = kgen.rebind %arg0 : !pop.scalar<dt1> to !pop.scalar<si32>
  %0 = kgen.rebind %arg0 : !pop.scalar<dt1> to !pop.scalar<dt2>
  %1 = kgen.rebind %0 : !pop.scalar<dt2> to !pop.scalar<dt3>
  %2 = kgen.rebind %1 : !pop.scalar<dt3> to !pop.scalar<si32>
  // CHECK-NEXT: return %0
  kgen.return %2 : !pop.scalar<si32>
}

// CHECK-LABEL: @rebind_across_scopes
kgen.generator @rebind_across_scopes<dt: dtype>(%arg0: !pop.scalar<dt>) {
  kgen.param.declare dt1 = <dt>
  // CHECK: rebind %arg0 : !pop.scalar<dt> to !pop.scalar<dt1>
  %0 = kgen.rebind %arg0 : !pop.scalar<dt> to !pop.scalar<dt1>
  // CHECK: param.declare.region
  kgen.param.declare.region F = <dt: dtype>() -> !pop.scalar<dt> {
    // CHECK: rebind %0 : !pop.scalar<dt1> to !pop.scalar<dt>
    %1 = kgen.rebind %0 : !pop.scalar<dt1> to !pop.scalar<dt>
    kgen.return %1 : !pop.scalar<dt>
  }
  kgen.return
}

// CHECK-LABEL: @param_materialize
kgen.generator @param_materialize() -> (i32, !pop.pointer<i32>) {
  // CHECK-NEXT: kgen.param.constant: i32 = <2>
  %0 = kgen.param.materialize: i32 = <2>
  // CHECK-NEXT: kgen.param.materialize
  %1 = kgen.param.materialize: pointer<i32> = <#M.memref<[(undef, heap, [])], 0, 0>>
  // CHECK-NOT: kgen.param.materialize
  %2 = kgen.param.materialize: pointer<i32> = <#M.memref<[(undef, heap, [])], 0, 0>>
  kgen.return %0, %1 : i32, !pop.pointer<i32>
}

// CHECK-LABEL: kgen.func @cast_from_folds
// CHECK-SAME: (%[[ARG0:.*]]: !pop.scalar<f32> loc({{.*}})) -> !pop.scalar<f32> {
kgen.func @cast_from_folds(%arg0: !pop.scalar<f32>) -> !pop.scalar<f32> {

  // A-B-A cast.
  %1 = pop.cast_to_builtin %arg0 : !pop.scalar<f32> to f32
  %2 = pop.cast_from_builtin %1 : f32 to !pop.scalar<f32>

  // CHECK: kgen.return %[[ARG0]]
  kgen.return %2 : !pop.scalar<f32>
}

// CHECK-LABEL: kgen.func @cast_to_folds
// CHECK-SAME: (%[[ARG0:.*]]: f32 loc({{.*}})) -> f32 {
kgen.func @cast_to_folds(%arg0: f32) -> f32 {

  // A-B-A cast.
  %1 = pop.cast_from_builtin %arg0 : f32 to !pop.scalar<f32>
  %2 = pop.cast_to_builtin %1 : !pop.scalar<f32> to f32

  // CHECK: kgen.return %[[ARG0]]
  kgen.return %2 : f32
}

kgen.generator @producesResultParam<() -> r1>() {
  kgen.param.result_bind<42>
  kgen.return
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
  kgen.call @producesResultParam<[] -> result>() : () -> ()

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
  // CHECK: %1 = kgen.call @trivial_param<1>(%arg0)
  %1 = kgen.call_param[(si32) -> si32: @trivial_param<1>](%arg0)
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

kgen.generator @callee<fn: <index>()->index>() {
  kgen.return
}

// CHECK-LABEL: @hoist_constant
kgen.generator @hoist_constant() {
  // CHECK-NEXT: kgen.param.declare.region fn
  kgen.param.declare.region fn = <N>() -> index {
    // CHECK-NEXT: kgen.param.constant = <N>
    %0 = kgen.param.constant = <N>
    kgen.return %0 : index
  }
  kgen.return
}

kgen.generator @call_me() {
  kgen.return
}

kgen.generator @call_with_bound<A>() {
  kgen.return
}

// CHECK-LABEL: @call_param_bound_symbol
kgen.generator @call_param_bound_symbol() {
  // CHECK-NEXT: kgen.call @call_with_bound<1>() : () -> ()
  kgen.call_param[() -> (): @call_with_bound<1>]()
  kgen.return
}

// -----

lit.struct.decl @Struct {
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

// COM: Check that constant are only hoisted from subprogram regions if there is
// COM: no debuginfo scope given.

#file = #debuginfo.file<"foo.mlir" in "/">
#compile_unit = #debuginfo.compile_unit<sourceLanguage = DW_LANG_C, file = #file, producer = "Mojo", isOptimized = true, emissionKind = Full>
#subprogram = #debuginfo.subprogram<
  compileUnit = #compile_unit,
  scope = #file,
  name = "foo",
  linkageName = "foo",
  file = #file,
  line = 44,
  scopeLine = 44,
  subprogramFlags = "Definition|Optimized"
> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#subprogram1 = #debuginfo.subprogram<
  compileUnit = #compile_unit,
  scope = #file,
  name = "SomeClosure",
  linkageName = "SomeClosure",
  file = #file,
  line = 325,
  scopeLine = 325,
  subprogramFlags = "Definition|Optimized"
> : !debuginfo.subroutine<() -> (): DW_CC_normal>

#loc1 = loc("foo.mlir":44:1)
#loc2 = loc("foo.mlir":325:11)
#loc3 = loc("bar.mlir":327:17)
#loc4 = loc(fused<#subprogram>[#loc1])
#loc5 = loc(fused<#subprogram1>[#loc2])
#loc6 = loc(fused<#subprogram1>[#loc3])

// CHECK-LABEL: kgen.func @no_hoist
kgen.func @no_hoist() {
  // CHECK-NEXT: kgen.stage_closure = () {
  %0 = kgen.stage_closure = () {
    // CHECK-NEXT: kgen.param.constant: array<1, index> = <[0]>
    %array = kgen.param.constant: array<1, index> = <[0]> loc(#loc6)
    %1 = pop.stack_allocation 1 x !pop.array<1, index>  loc(#loc6)
    pop.store %array, %1 : !pop.pointer<array<1, index>> loc(#loc6)
    kgen.return loc(#loc5)
  } callLoc(#loc4) loc(#loc5)
  kgen.call_signature %0() : () -> () loc(#loc4)
  kgen.return loc(#loc4)
} loc(#loc4)

// COM: Callee does not have debug info, but the caller does.
// CHECK-LABEL: kgen.func @no_hoist_nodebug_callee
kgen.func @no_hoist_nodebug_callee() {
  // CHECK-NEXT: kgen.stage_closure = () {
  %0 = kgen.stage_closure = () {
    // CHECK-NEXT: kgen.param.constant: array<1, index> = <[0]>
    %array = kgen.param.constant: array<1, index> = <[0]> loc(#loc6)
    %1 = pop.stack_allocation 1 x !pop.array<1, index>  loc(#loc6)
    pop.store %array, %1 : !pop.pointer<array<1, index>> loc(#loc6)
    kgen.return loc(#loc5)
  } callLoc(#loc4) loc(#loc2)
  kgen.call_signature %0() : () -> () loc(#loc4)
  kgen.return loc(#loc4)
} loc(#loc4)

// CHECK-LABEL: kgen.func @hoist
kgen.func @hoist() {
  // CHECK-NEXT: kgen.param.constant: array<1, index> = <[0]>
  // CHECK-NEXT: kgen.stage_closure = () {
  %0 = kgen.stage_closure = () -> () {
    // CHECK-NOT: kgen.param.constant: array<1, index> = <[0]>
    %array = kgen.param.constant: array<1, index> = <[0]>
    %1 = pop.stack_allocation 1 x !pop.array<1, index>
    pop.store %array, %1 : !pop.pointer<array<1, index>>
    kgen.return
  }
  kgen.call_signature %0() : () -> ()
  kgen.return
}
