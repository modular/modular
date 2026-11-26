// RUN: kgen-opt -lower-kgen-to-arith -allow-unregistered-dialect %s | FileCheck %s

// `lower-kgen-to-arith` lowers scalar constants and arithmetic to `arith`,
// bridging the still-`!kgen.scalar` function boundary with
// `unrealized_conversion_cast`s, and leaves SIMD forms for the NISA lowering.

// CHECK-LABEL: kgen.func @scalar_add
// CHECK-DAG:     %[[A:.*]] = builtin.unrealized_conversion_cast %arg0 : !kgen.scalar<f32> to f32
// CHECK-DAG:     %[[B:.*]] = builtin.unrealized_conversion_cast %arg1 : !kgen.scalar<f32> to f32
// CHECK:         %[[R:.*]] = arith.addf %[[A]], %[[B]] : f32
// CHECK:         %[[RC:.*]] = builtin.unrealized_conversion_cast %[[R]] : f32 to !kgen.scalar<f32>
// CHECK:         hlcf.return %[[RC]]
kgen.func @scalar_add(%a: !kgen.scalar<f32>, %b: !kgen.scalar<f32>) -> !kgen.scalar<f32> {
  %r = pop.add %a, %b : !kgen.scalar<f32>
  hlcf.return %r : !kgen.scalar<f32>
}

// CHECK-LABEL: kgen.func @scalar_cmp_select
// CHECK:         %[[CMP:.*]] = arith.cmpi slt
// CHECK:         arith.select %[[CMP]]
kgen.func @scalar_cmp_select(%a: !kgen.scalar<si32>, %b: !kgen.scalar<si32>) -> !kgen.scalar<si32> {
  %c = pop.cmp lt(%a, %b) : !kgen.scalar<si32>
  %r = pop.simd.select %c, %a, %b : !kgen.scalar<si32>
  hlcf.return %r : !kgen.scalar<si32>
}

// CHECK-LABEL: kgen.func @scalar_constant
// CHECK:         arith.constant 7 : i32
kgen.func @scalar_constant() -> !kgen.scalar<si32> {
  %c = kgen.param.constant: scalar<si32> = <7>
  hlcf.return %c : !kgen.scalar<si32>
}

// Index min/max stay in the `index` dialect for the loop-bound matcher.
// CHECK-LABEL: kgen.func @index_max
// CHECK:         index.maxs
kgen.func @index_max(%a: !kgen.scalar<index>, %b: !kgen.scalar<index>) -> !kgen.scalar<index> {
  %r = pop.max %a, %b : !kgen.scalar<index>
  hlcf.return %r : !kgen.scalar<index>
}

// SIMD forms are not this pass's business.
// CHECK-LABEL: kgen.func @simd_add_untouched
// CHECK:         pop.add %{{.*}}, %{{.*}} : !kgen.simd<128, f32>
kgen.func @simd_add_untouched(%a: !kgen.simd<128, f32>, %b: !kgen.simd<128, f32>) -> !kgen.simd<128, f32> {
  %r = pop.add %a, %b : !kgen.simd<128, f32>
  hlcf.return %r : !kgen.simd<128, f32>
}

// `bf16` and `f16` are both 16 bits wide, so the dtype, not the width, picks
// the float type.
// CHECK-LABEL: kgen.func @bf16_add
// CHECK:         arith.addf %{{.*}}, %{{.*}} : bf16
kgen.func @bf16_add(%a: !kgen.scalar<bf16>, %b: !kgen.scalar<bf16>) -> !kgen.scalar<bf16> {
  %r = pop.add %a, %b : !kgen.scalar<bf16>
  hlcf.return %r : !kgen.scalar<bf16>
}

// The builtin integer types are signless, so the predicate carries the
// signedness the dtype holds.
// CHECK-LABEL: kgen.func @unsigned_cmp
// CHECK:         arith.cmpi ult
kgen.func @unsigned_cmp(%a: !kgen.scalar<ui32>, %b: !kgen.scalar<ui32>) -> !kgen.scalar<bool> {
  %r = pop.cmp lt(%a, %b) : !kgen.scalar<ui32>
  hlcf.return %r : !kgen.scalar<bool>
}

// CHECK-LABEL: kgen.func @signed_cmp
// CHECK:         arith.cmpi slt
kgen.func @signed_cmp(%a: !kgen.scalar<si32>, %b: !kgen.scalar<si32>) -> !kgen.scalar<bool> {
  %r = pop.cmp lt(%a, %b) : !kgen.scalar<si32>
  hlcf.return %r : !kgen.scalar<bool>
}

// Same for min/max, in both the `arith` and the `index` forms.
// CHECK-LABEL: kgen.func @unsigned_max
// CHECK:         arith.maxui
kgen.func @unsigned_max(%a: !kgen.scalar<ui32>, %b: !kgen.scalar<ui32>) -> !kgen.scalar<ui32> {
  %r = pop.max %a, %b : !kgen.scalar<ui32>
  hlcf.return %r : !kgen.scalar<ui32>
}

// CHECK-LABEL: kgen.func @uindex_max
// CHECK:         index.maxu
kgen.func @uindex_max(%a: !kgen.scalar<uindex>, %b: !kgen.scalar<uindex>) -> !kgen.scalar<uindex> {
  %r = pop.max %a, %b : !kgen.scalar<uindex>
  hlcf.return %r : !kgen.scalar<uindex>
}

// The dtype picks an arithmetic or a logical right shift.
// CHECK-LABEL: kgen.func @signed_shr
// CHECK:         arith.shrsi
kgen.func @signed_shr(%a: !kgen.scalar<si32>, %b: !kgen.scalar<si32>) -> !kgen.scalar<si32> {
  %r = pop.shr %a, %b : !kgen.scalar<si32>
  hlcf.return %r : !kgen.scalar<si32>
}

// CHECK-LABEL: kgen.func @unsigned_shr
// CHECK:         arith.shrui
kgen.func @unsigned_shr(%a: !kgen.scalar<ui32>, %b: !kgen.scalar<ui32>) -> !kgen.scalar<ui32> {
  %r = pop.shr %a, %b : !kgen.scalar<ui32>
  hlcf.return %r : !kgen.scalar<ui32>
}

// CHECK-LABEL: kgen.func @index_shr
// CHECK:         index.shrs
kgen.func @index_shr(%a: !kgen.scalar<index>, %b: !kgen.scalar<index>) -> !kgen.scalar<index> {
  %r = pop.shr %a, %b : !kgen.scalar<index>
  hlcf.return %r : !kgen.scalar<index>
}

// CHECK-LABEL: kgen.func @uindex_shr
// CHECK:         index.shru
kgen.func @uindex_shr(%a: !kgen.scalar<uindex>, %b: !kgen.scalar<uindex>) -> !kgen.scalar<uindex> {
  %r = pop.shr %a, %b : !kgen.scalar<uindex>
  hlcf.return %r : !kgen.scalar<uindex>
}

// CHECK-LABEL: kgen.func @int_shl
// CHECK:         arith.shli %{{.*}}, %{{.*}} : i32
kgen.func @int_shl(%a: !kgen.scalar<si32>, %b: !kgen.scalar<si32>) -> !kgen.scalar<si32> {
  %r = pop.shl %a, %b : !kgen.scalar<si32>
  hlcf.return %r : !kgen.scalar<si32>
}

// CHECK-LABEL: kgen.func @int_and
// CHECK:         arith.andi %{{.*}}, %{{.*}} : i32
kgen.func @int_and(%a: !kgen.scalar<si32>, %b: !kgen.scalar<si32>) -> !kgen.scalar<si32> {
  %r = pop.simd.and %a, %b : !kgen.scalar<si32>
  hlcf.return %r : !kgen.scalar<si32>
}

// CHECK-LABEL: kgen.func @int_or
// CHECK:         arith.ori %{{.*}}, %{{.*}} : i32
kgen.func @int_or(%a: !kgen.scalar<si32>, %b: !kgen.scalar<si32>) -> !kgen.scalar<si32> {
  %r = pop.simd.or %a, %b : !kgen.scalar<si32>
  hlcf.return %r : !kgen.scalar<si32>
}

// CHECK-LABEL: kgen.func @int_xor
// CHECK:         arith.xori %{{.*}}, %{{.*}} : i32
kgen.func @int_xor(%a: !kgen.scalar<si32>, %b: !kgen.scalar<si32>) -> !kgen.scalar<si32> {
  %r = pop.simd.xor %a, %b : !kgen.scalar<si32>
  hlcf.return %r : !kgen.scalar<si32>
}

// The `index` forms stay in the `index` dialect, as for min/max.
// CHECK-LABEL: kgen.func @index_shl
// CHECK:         index.shl
kgen.func @index_shl(%a: !kgen.scalar<index>, %b: !kgen.scalar<index>) -> !kgen.scalar<index> {
  %r = pop.shl %a, %b : !kgen.scalar<index>
  hlcf.return %r : !kgen.scalar<index>
}

// CHECK-LABEL: kgen.func @index_and
// CHECK:         index.and
kgen.func @index_and(%a: !kgen.scalar<index>, %b: !kgen.scalar<index>) -> !kgen.scalar<index> {
  %r = pop.simd.and %a, %b : !kgen.scalar<index>
  hlcf.return %r : !kgen.scalar<index>
}

// CHECK-LABEL: kgen.func @index_or
// CHECK:         index.or
kgen.func @index_or(%a: !kgen.scalar<index>, %b: !kgen.scalar<index>) -> !kgen.scalar<index> {
  %r = pop.simd.or %a, %b : !kgen.scalar<index>
  hlcf.return %r : !kgen.scalar<index>
}

// CHECK-LABEL: kgen.func @index_xor
// CHECK:         index.xor
kgen.func @index_xor(%a: !kgen.scalar<index>, %b: !kgen.scalar<index>) -> !kgen.scalar<index> {
  %r = pop.simd.xor %a, %b : !kgen.scalar<index>
  hlcf.return %r : !kgen.scalar<index>
}

// `bool` converts to `i1`, so the bitwise ops lower to logical ops.
// CHECK-LABEL: kgen.func @bool_and
// CHECK:         arith.andi %{{.*}}, %{{.*}} : i1
kgen.func @bool_and(%a: !kgen.scalar<bool>, %b: !kgen.scalar<bool>) -> !kgen.scalar<bool> {
  %r = pop.simd.and %a, %b : !kgen.scalar<bool>
  hlcf.return %r : !kgen.scalar<bool>
}

// CHECK-LABEL: kgen.func @bool_or
// CHECK:         arith.ori %{{.*}}, %{{.*}} : i1
kgen.func @bool_or(%a: !kgen.scalar<bool>, %b: !kgen.scalar<bool>) -> !kgen.scalar<bool> {
  %r = pop.simd.or %a, %b : !kgen.scalar<bool>
  hlcf.return %r : !kgen.scalar<bool>
}

// CHECK-LABEL: kgen.func @bool_xor
// CHECK:         arith.xori %{{.*}}, %{{.*}} : i1
kgen.func @bool_xor(%a: !kgen.scalar<bool>, %b: !kgen.scalar<bool>) -> !kgen.scalar<bool> {
  %r = pop.simd.xor %a, %b : !kgen.scalar<bool>
  hlcf.return %r : !kgen.scalar<bool>
}

// SIMD shift and bitwise forms are left for the target lowering.
// CHECK-LABEL: kgen.func @simd_shl_untouched
// CHECK:         pop.shl %{{.*}}, %{{.*}} : !kgen.simd<128, si32>
kgen.func @simd_shl_untouched(%a: !kgen.simd<128, si32>, %b: !kgen.simd<128, si32>) -> !kgen.simd<128, si32> {
  %r = pop.shl %a, %b : !kgen.simd<128, si32>
  hlcf.return %r : !kgen.simd<128, si32>
}

// CHECK-LABEL: kgen.func @simd_shr_untouched
// CHECK:         pop.shr %{{.*}}, %{{.*}} : !kgen.simd<128, si32>
kgen.func @simd_shr_untouched(%a: !kgen.simd<128, si32>, %b: !kgen.simd<128, si32>) -> !kgen.simd<128, si32> {
  %r = pop.shr %a, %b : !kgen.simd<128, si32>
  hlcf.return %r : !kgen.simd<128, si32>
}

// CHECK-LABEL: kgen.func @simd_and_untouched
// CHECK:         pop.simd.and %{{.*}}, %{{.*}} : <128, si32>
kgen.func @simd_and_untouched(%a: !kgen.simd<128, si32>, %b: !kgen.simd<128, si32>) -> !kgen.simd<128, si32> {
  %r = pop.simd.and %a, %b : !kgen.simd<128, si32>
  hlcf.return %r : !kgen.simd<128, si32>
}

// CHECK-LABEL: kgen.func @simd_or_untouched
// CHECK:         pop.simd.or %{{.*}}, %{{.*}} : <128, si32>
kgen.func @simd_or_untouched(%a: !kgen.simd<128, si32>, %b: !kgen.simd<128, si32>) -> !kgen.simd<128, si32> {
  %r = pop.simd.or %a, %b : !kgen.simd<128, si32>
  hlcf.return %r : !kgen.simd<128, si32>
}

// CHECK-LABEL: kgen.func @simd_xor_untouched
// CHECK:         pop.simd.xor %{{.*}}, %{{.*}} : <128, si32>
kgen.func @simd_xor_untouched(%a: !kgen.simd<128, si32>, %b: !kgen.simd<128, si32>) -> !kgen.simd<128, si32> {
  %r = pop.simd.xor %a, %b : !kgen.simd<128, si32>
  hlcf.return %r : !kgen.simd<128, si32>
}
