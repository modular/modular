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
