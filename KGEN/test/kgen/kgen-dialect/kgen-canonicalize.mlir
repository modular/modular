// RUN: kgen-opt -verify-parameters -canonicalize -mlir-print-debuginfo %s | FileCheck %s

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

// CHECK-LABEL: @rebind_symbolic_ptr
kgen.generator @rebind_symbolic_ptr<t: type>() -> !kgen.pointer<t> {
  // CHECK-NEXT: %pointer = kgen.param.constant: pointer<t> = <#interp.symbolic_pointer<0>>
  %0 = kgen.param.constant: pointer<none> = <#interp.symbolic_pointer<0>>
  %1 = kgen.rebind %0 : !kgen.pointer<none> to !kgen.pointer<t>
  // CHECK-NEXT: return %pointer
  kgen.return %1 : !kgen.pointer<t>
}

// CHECK-LABEL: @rebind_across_scopes
kgen.generator @rebind_across_scopes<dt: dtype>(%arg0: !pop.scalar<dt>) {
  kgen.param.declare dt1: dtype = <dt>
  %0 = kgen.rebind %arg0 : !pop.scalar<dt> to !pop.scalar<dt1>
  // CHECK: param.declare.region
  kgen.param.declare.region F = <dt2: dtype>() -> !pop.scalar<dt2> {
    // CHECK: rebind %arg0 : !pop.scalar<dt> to !pop.scalar<dt2>
    %1 = kgen.rebind %0 : !pop.scalar<dt1> to !pop.scalar<dt2>
    kgen.return %1 : !pop.scalar<dt2>
  }
  kgen.return
}

// CHECK-LABEL: @param_materialize
kgen.generator @param_materialize() -> (i32, !kgen.pointer<i32>) {
  // CHECK-NEXT: kgen.param.constant: i32 = <2>
  %0 = kgen.param.materialize: i32 = <2>
  // CHECK-NEXT: kgen.param.materialize
  %1 = kgen.param.materialize: pointer<i32> = <#interp.memref<[(undef, heap, [])], 0, 0>>
  // CHECK-NOT: kgen.param.materialize
  %2 = kgen.param.materialize: pointer<i32> = <#interp.memref<[(undef, heap, [])], 0, 0>>
  kgen.return %0, %1 : i32, !kgen.pointer<i32>
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

// CHECK-LABEL: kgen.generator @param_assert_simplify<p1: i1, p2>()
kgen.generator @param_assert_simplify<p1 : i1, p2>() {
  // CHECK-NOT: assert <1>
  kgen.param.assert <1>, "this is pointless"
  // CHECK-NEXT: kgen.param.assert <0>, "failing asserts must be kept"
  kgen.param.assert <eq(42, 41)>, "failing asserts must be kept"
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


lit.struct.decl @Struct {
  lit.func @Nested() {
    kgen.return
  }
}

// CHECK-LABEL: @callNested
kgen.generator @callNested() {
  // CHECK-NEXT: kgen.call @Struct::@Nested
  kgen.call_param[!lit.signature<() -> ()>: @Struct::@Nested]()
  kgen.return
}


// Check that constant are only hoisted from subprogram regions if there is no
// debuginfo scope given.
#subprogram = #debuginfo.subprogram<name = <"foo">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#subprogram1 = #debuginfo.subprogram<name = <"SomeClosure">> : !debuginfo.subroutine<() -> (): DW_CC_normal>

#loc1 = loc("foo.mlir":44:1)
#loc2 = loc("foo.mlir":325:11)
#loc3 = loc("bar.mlir":327:17)
#loc4 = loc(fused<#subprogram>[#loc1])
#loc5 = loc(fused<#subprogram1>[#loc2])
#loc6 = loc(fused<#subprogram1>[#loc3])
#call_loc = #debuginfo.call_loc<#loc4>
#loc7 = loc(fused<#call_loc>[#loc2])
#loc8 = loc(fused<#subprogram1>[#loc7])
#loc9 = loc(fused<#call_loc>[#loc2])

// CHECK-LABEL: kgen.func @no_hoist
kgen.func @no_hoist() {
  // CHECK-NEXT: kgen.stage_closure = () {
  %0 = kgen.stage_closure = () {
    // CHECK-NEXT: kgen.param.constant: array<1, index> = <[0]>
    %array = kgen.param.constant: array<1, index> = <[0]> loc(#loc6)
    %1 = pop.stack_allocation 1 x !pop.array<1, index>  loc(#loc6)
    pop.store %array, %1 : !kgen.pointer<array<1, index>> loc(#loc6)
    kgen.return loc(#loc5)
  } loc(#loc8)
  kgen.call_indirect %0() : () -> () loc(#loc4)
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
    pop.store %array, %1 : !kgen.pointer<array<1, index>> loc(#loc6)
    kgen.return loc(#loc5)
  } loc(#loc9)
  kgen.call_indirect %0() : () -> () loc(#loc4)
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
    pop.store %array, %1 : !kgen.pointer<array<1, index>>
    kgen.return
  }
  kgen.call_indirect %0() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @int_literal_convert
kgen.func @int_literal_convert() -> si64 {
  // CHECK: constant: si64 = <123>
  %0 = kgen.param.constant: !kgen.int_literal = <123>
  %1 = kgen.int_literal.convert %0 : to si64
  kgen.return %1 : si64
}

// CHECK-LABEL: @pack_create(
kgen.func @pack_create() -> !kgen.pack<[ui8, i32]> {
  // CHECK-NEXT: kgen.param.constant{{.*}} = <<5, -55>>
  %0 = kgen.param.constant: ui8 = <5>
  %1 = kgen.param.constant: i32 = <-55>
  %2 = kgen.pack.create(%0, %1) : !kgen.pack<[ui8, i32]>
  kgen.return %2 : !kgen.pack<[ui8, i32]>
}

// CHECK-LABEL: @pack_create_empty(
kgen.func @pack_create_empty() -> !kgen.pack<[]> {
  // CHECK-NEXT: kgen.param.constant{{.*}} = <<>>
  %1 = kgen.pack.create() : !kgen.pack<[]>
  kgen.return %1 : !kgen.pack<[]>
}

// CHECK-LABEL: @pack_get(
kgen.func @pack_get() -> i4 {
  // CHECK-NEXT: kgen.param.constant: i4 = <3>
  %0 = kgen.param.constant: !kgen.pack<[f32, i4]> = <<-1.2, 3>>
  %1 = kgen.pack.extract %0[1] : !kgen.pack<[f32, i4]>
  kgen.return %1 : i4
}

// CHECK-LABEL: @pack_create_get(
kgen.func @pack_create_get(%arg0: f32, %arg1: si8) -> f32 {
  // CHECK-NEXT: kgen.return %arg0
  %0 = kgen.pack.create(%arg0, %arg1) : !kgen.pack<[f32, si8]>
  %1 = kgen.pack.extract %0[0] : !kgen.pack<[f32, si8]>
  kgen.return %1 : f32
}

// CHECK-LABEL: @pack_size(
kgen.func @pack_size() -> index {
  // CHECK-NEXT: kgen.param.constant = <2>
  %0 = kgen.param.constant: !kgen.pack<[i1, i4]> = <<0, 1>>
  %1 = kgen.pack.size %0 : !kgen.pack<[i1, i4]>
  kgen.return %1 : index
}

// CHECK-LABEL: @struct_create
kgen.func @struct_create() -> !kgen.struct<(si4, ui4)> {
  // CHECK-NEXT: constant: struct<(si4, ui4)> = <{ -3, 7 }>
  %0 = kgen.param.constant: si4 = <-3>
  %1 = kgen.param.constant: ui4 = <7>
  %2 = kgen.struct.create(%0, %1) : !kgen.struct<(si4, ui4)>
  kgen.return %2 : !kgen.struct<(si4, ui4)>
}

// CHECK-LABEL: @struct_get
kgen.func @struct_get() -> si4 {
  // CHECK-NEXT: constant: si4 = <-3>
  %0 = kgen.param.constant: struct<(si4, ui4)> = <{ -3, 7 }>
  %1 = kgen.struct.extract %0[0] : !kgen.struct<(si4, ui4)>
  kgen.return %1 : si4
}

// CHECK-LABEL: @struct_replace
kgen.func @struct_replace() -> !kgen.struct<(si4, ui4)> {
  // CHECK-NEXT: constant: struct<(si4, ui4)> = <{ -5, 7 }>
  %0 = kgen.param.constant: si4 = <-5>
  %1 = kgen.param.constant: struct<(si4, ui4)> = <{ -3, 7 }>
  %2 = kgen.struct.replace %0, %1[0] : !kgen.struct<(si4, ui4)>
  kgen.return %2 : !kgen.struct<(si4, ui4)>
}

// CHECK-LABEL: @variant_create
kgen.func @variant_create() -> !kgen.variant<si4, ui4> {
  // CHECK-NEXT: constant: variant<si4, ui4> = <#kgen.variant<:ui4 7, 1>>
  %0 = kgen.param.constant: ui4 = <7>
  %1 = kgen.variant.create %0, 1 : <si4, ui4>
  kgen.return %1 : !kgen.variant<si4, ui4>
}

// CHECK-LABEL: @variant_is
kgen.func @variant_is() -> i1 {
  // CHECK-NEXT: constant: i1 = <1>
  %0 = kgen.param.constant: variant<si4, ui4> = <#kgen.variant<:ui4 7, 1>>
  %1 = kgen.variant.is %0, 1 : <si4, ui4>
  kgen.return %1 : i1
}

// CHECK-LABEL: @variant_create_is_true
kgen.func @variant_create_is_true(%a: i32) -> i1 {
  // CHECK-NEXT: constant: i1 = <1>
  %0 = kgen.variant.create %a, 0 : <i32, f32>
  %1 = kgen.variant.is %0, 0 : <i32, f32>
  kgen.return %1 : i1
}

// CHECK-LABEL: @variant_create_is_false
kgen.func @variant_create_is_false(%a: i32) -> i1 {
  // CHECK-NEXT: constant: i1 = <0>
  %0 = kgen.variant.create %a, 0 : <i32, f32>
  %1 = kgen.variant.is %0, 1 : <i32, f32>
  kgen.return %1 : i1
}

// CHECK-LABEL: @variant_get
kgen.func @variant_get() -> ui4 {
  // CHECK-NEXT: constant: ui4 = <7>
  %0 = kgen.param.constant: variant<si4, ui4> = <#kgen.variant<:ui4 7, 1>>
  %1 = kgen.variant.get %0, 1 : <si4, ui4>
  kgen.return %1 : ui4
}

// CHECK-LABEL: @variant_get_ub
kgen.func @variant_get_ub() -> si4 {
  // CHECK: kgen.variant.get
  %0 = kgen.param.constant: variant<si4, ui4> = <#kgen.variant<:ui4 7, 1>>
  %1 = kgen.variant.get %0, 0 : <si4, ui4>
  kgen.return %1 : si4
}

// CHECK-LABEL: @variant_create_get
kgen.func @variant_create_get(%a: i32) -> i32 {
  %0 = kgen.variant.create %a, 0 : <i32, f32>
  %1 = kgen.variant.get %0, 0 : <i32, f32>
  // CHECK: return %arg0
  kgen.return %1 : i32
}

kgen.func @closure_callee(%arg0: index, %arg1: i32) -> i64 {
  kgen.unreachable
}

// CHECK-LABEL: @call_create_closure
kgen.func @call_create_closure(%arg0: index, %arg1: i32) -> i64 {
  // CHECK-NEXT: %0 = kgen.call @closure_callee(%arg0, %arg1)
  %0 = kgen.create_closure[(index, i32) -> i64: @closure_callee](%arg0)
  %1 = kgen.call_indirect %0(%arg1) : (i32) capturing -> i64
  // CHECK-NEXT: return %0
  kgen.return %1 : i64
}

// CHECK-LABEL: kgen.func @variant_take_then_create
kgen.func @variant_take_then_create(
  %input: !kgen.variant<f32, i32>
) -> !kgen.variant<f32, i32> {

  // These will be folded away.
  %f32 = kgen.variant.get %input, 0 : !kgen.variant<f32, i32>
  %res = kgen.variant.create %f32, 0 : !kgen.variant<f32, i32>

  // CHECK-NEXT: return %arg0
  kgen.return %res : !kgen.variant<f32, i32>
}

// CHECK-LABEL: kgen.func @variant_create_then_take
kgen.func @variant_create_then_take(%f32: f32) -> f32 {

  // These will be folded away.
  %variant = kgen.variant.create %f32, 0 : !kgen.variant<f32, i32>
  %res = kgen.variant.get %variant, 0 : !kgen.variant<f32, i32>

  // CHECK-NEXT: return %arg0
  kgen.return %res : f32
}

// CHECK-LABEL: kgen.func @variant_take_then_create_mismatch_index
kgen.func @variant_take_then_create_mismatch_index(
  %input: !kgen.variant<i32, i32>
) -> !kgen.variant<i32, i32> {

  // CHECK-NEXT: %0 = kgen.variant.get %arg0, 0
  // CHECK-NEXT: %1 = kgen.variant.create %0, 1
  %i32 = kgen.variant.get %input, 0 : !kgen.variant<i32, i32>
  %res = kgen.variant.create %i32, 1 : !kgen.variant<i32, i32>

  // CHECK-NEXT: return %1
  kgen.return %res : !kgen.variant<i32, i32>
}

// CHECK-LABEL: kgen.func @variant_create_then_take_mismatch_index
kgen.func @variant_create_then_take_mismatch_index(%f32: f32) -> f32 {

  // CHECK-NEXT: %0 = kgen.variant.create %arg0, 1
  // CHECK-NEXT: %1 = kgen.variant.get %0, 0
  %variant = kgen.variant.create %f32, 1 : !kgen.variant<f32, f32>
  %res = kgen.variant.get %variant, 0 : !kgen.variant<f32, f32>

  // CHECK-NEXT: return %1
  kgen.return %res : f32
}

// CHECK-LABEL: kgen.func @variant_take_then_create_mismatch_types
kgen.func @variant_take_then_create_mismatch_types(
  %input: !kgen.variant<i32, i32>
) -> !kgen.variant<i32, f32> {

  // CHECK-NEXT: %0 = kgen.variant.get %arg0, 0
  // CHECK-NEXT: %1 = kgen.variant.create %0, 0
  %i32 = kgen.variant.get %input, 0 : !kgen.variant<i32, i32>
  %res = kgen.variant.create %i32, 0 : !kgen.variant<i32, f32>

  // CHECK-NEXT: return %1
  kgen.return %res : !kgen.variant<i32, f32>
}

// CHECK-LABEL: @source_loc_pure
kgen.func @source_loc_pure() {
  %line, %col, %fileName = kgen.source_loc[1]
  // CHECK-NEXT: return
  kgen.return
}

// CHECK-LABEL: @param_if_known_trivial
kgen.generator @param_if_known_trivial(%arg0: index) -> index {
  // CHECK-NEXT: return %arg0
  %0 = kgen.param.if <1> -> index {
    kgen.param.yield %arg0 : index
  } else {
    %1 = kgen.param.constant = <0>
    kgen.param.yield %1 : index
  }
  kgen.return %0 : index
}

// CHECK-LABEL: @param_if_known_dead
kgen.generator @param_if_known_dead(%arg0: !kgen.pointer<index>, %arg1: index) {
  // CHECK-NEXT: param.if <0>
  kgen.param.if <0> {
    // CHECK-NEXT: unreachable
    pop.store %arg1, %arg0 : !kgen.pointer<index>
    kgen.param.yield
  } else {
    pop.store %arg1, %arg0 : !kgen.pointer<index>
    kgen.param.yield
  }
  kgen.return
}

// CHECK-LABEL: @trivial_struct_copy
kgen.func @trivial_struct_copy(%arg0: !kgen.struct<(i1)>, %arg1: !kgen.struct<(i1, i1)>) -> (!kgen.struct<()>, !kgen.struct<(i1)>, !kgen.struct<(i1, i1)>, !kgen.struct<(i1, i1)>) {
  // CHECK: %struct = kgen.param.constant: struct<()>
  %0 = kgen.struct.create () : !kgen.struct<()>

  %1 = kgen.struct.extract %arg0[0] : !kgen.struct<(i1)>
  %2 = kgen.struct.create (%1) : !kgen.struct<(i1)>

  %3 = kgen.struct.extract %arg1[1] : !kgen.struct<(i1, i1)>
  %4 = kgen.struct.extract %arg1[0] : !kgen.struct<(i1, i1)>
  %5 = kgen.struct.create (%4, %3) : !kgen.struct<(i1, i1)>

  // CHECK: [[CREATE:%.*]] = kgen.struct.create
  %6 = kgen.struct.create (%3, %4) : !kgen.struct<(i1, i1)>

  // CHECK: return %struct, %arg0, %arg1, [[CREATE:%.*]]
  kgen.return %0, %2, %5, %6 : !kgen.struct<()>, !kgen.struct<(i1)>, !kgen.struct<(i1, i1)>, !kgen.struct<(i1, i1)>
}
