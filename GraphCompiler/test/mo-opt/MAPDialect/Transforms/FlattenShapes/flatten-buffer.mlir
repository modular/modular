// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-kernels \
// RUN:   --canonicalize --map-fuse-mutable-loads \
// RUN:   --map-outline-structured-kernel-bodies --map-livm \
// RUN:   --map-flatten-shapes --map-to-mogg | FileCheck %s

// An in-place loop reads and writes a buffer through its collapse view, which
// stays as mutable as the buffer.

// CHECK-LABEL: mo.graph @add_in_place
// CHECK:       mogg.tensor.create.collapse(%{{.*}}) <elementType = f32, mut = false, outShape = #mosh<ape[100]> : !mosh.ape, groups = [2]> : !mogg.tensor
// CHECK:       mogg.tensor.create.collapse(%{{.*}}) <elementType = f32, mut = true, outShape = #mosh<ape[100]> : !mosh.ape, groups = [2]> : !mogg.buffer
// CHECK:       mogg.call.foreach
mo.graph @add_in_place(%buffer: !mo.buffer<[10, 10], f32>, %rhs: !mo.tensor<[10, 10], f32>) {
  %0 = mo.chain.create(%buffer) : !mo.buffer<[10, 10], f32>
  %lhs, %1 = mo.mutable.load(%buffer, %0) : (!mo.buffer<[10, 10], f32>, !mo.chain) -> (!mo.tensor<[10, 10], f32>, !mo.chain)
  %sum = mo.add(%lhs, %rhs) : !mo.tensor<[10, 10], f32>
  %2 = mo.mutable.store(%buffer, %sum, %1) : (!mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>, !mo.chain) -> !mo.chain
  mo.output
}
