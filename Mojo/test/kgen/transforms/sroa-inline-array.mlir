// RUN: kgen-opt -canonicalize -sroa %s | FileCheck %s

// An array of SIMD-wrapping structs is reached via an element-0
// `pop.pointer.bitcast` and a `pop.offset` neighbor walk. Canonicalization
// turns those into constant-index `pop.array.gep`, which unblocks SROA to
// fully decompose the array into per-element scalar allocs.
// CHECK-LABEL: @inline_array_del_pattern
// CHECK:      pop.stack_allocation 1 x simd<4, f32>
// CHECK-NEXT: pop.stack_allocation 1 x simd<4, f32>
// CHECK-NOT:  pop.pointer.bitcast
// CHECK-NOT:  pop.offset
// CHECK-NOT:  kgen.struct.gep
kgen.func @inline_array_del_pattern(%v: !kgen.simd<4, f32>) {
  %arr = pop.stack_allocation 1 x !pop.array<2, struct<(simd<4, f32>)>>
  %c1 = kgen.param.constant: scalar<index> = <1>
  %e0 = pop.pointer.bitcast %arr : !kgen.pointer<array<2, struct<(simd<4, f32>)>>> to !kgen.pointer<struct<(simd<4, f32>)>>
  %e1 = pop.offset %e0[%c1] : !kgen.pointer<struct<(simd<4, f32>)>>
  %f0 = kgen.struct.gep %e0[0] : <struct<(simd<4, f32>)>>
  %f1 = kgen.struct.gep %e1[0] : <struct<(simd<4, f32>)>>
  pop.store %v, %f0 : !kgen.pointer<simd<4, f32>>
  pop.store %v, %f1 : !kgen.pointer<simd<4, f32>>
  hlcf.return
}
