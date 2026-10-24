// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg | FileCheck %s

// `mogg.routine` carries the memory effects the MAP op holds, which `MOToMAP`
// took from the op it lowered. A kernel whose only observable effect is a
// mutable store has no used result, so without them it reports no effects at
// all and is dead: the store is dropped and the buffer never written.

// CHECK-LABEL: mo.graph @mutable_store
// The add produces a value, so it carries no effect.
// CHECK: mogg.routine
// CHECK: attributes {{.*}}hasReadEffect = false, hasWriteEffect = false
// The store returns only a chain, and writes.
// CHECK: mogg.routine {{.*}}-> !mo.chain
// CHECK: attributes {{.*}}hasWriteEffect = true
mo.graph @mutable_store(%buffer: !mo.buffer<[10, 10], f32>, %rhs: !mo.tensor<[10, 10], f32>) {
  %0 = mo.chain.create(%buffer) : !mo.buffer<[10, 10], f32>
  %outTensor, %outChain = mo.mutable.load(%buffer, %0) : (!mo.buffer<[10, 10], f32>, !mo.chain) -> (!mo.tensor<[10, 10], f32>, !mo.chain)
  %1 = mo.add(%outTensor, %rhs) : !mo.tensor<[10, 10], f32>
  %2 = mo.mutable.store(%buffer, %1, %outChain) : (!mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>, !mo.chain) -> !mo.chain
  mo.output
}

// CHECK-LABEL: mo.graph @pure_add
// CHECK: attributes {{.*}}hasReadEffect = false, hasWriteEffect = false
mo.graph @pure_add(%a: !mo.tensor<[10, 10], f32>, %b: !mo.tensor<[10, 10], f32>) -> !mo.tensor<[10, 10], f32> {
  %0 = mo.add(%a, %b) : !mo.tensor<[10, 10], f32>
  mo.output %0 : !mo.tensor<[10, 10], f32>
}
