// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-kernels --map-fuse-mutable-loads --map-outline-structured-kernel-bodies --map-to-mogg | FileCheck %s

// `map-fuse-mutable-loads` folds the load's mutable buffer into the loop, so
// the loaded and stored captures reach `mogg.call.foreach` as `!mogg.buffer`
// rather than `!mogg.tensor`.

// CHECK-LABEL: mo.graph @in_place_add
// CHECK: [[IN:%[^ ]+]] = mogg.tensor.create({{.*}}) <{{.*}}mut = true{{.*}}> : !mogg.buffer
// CHECK: [[RHS:%[^ ]+]] = mogg.tensor.create({{.*}}) <{{.*}}mut = false{{.*}}> : !mogg.tensor
// CHECK: [[OUT:%[^ ]+]] = mogg.tensor.create({{.*}}) <{{.*}}mut = true{{.*}}> : !mogg.buffer
// CHECK: mogg.call.foreach {{.*}} ([[IDX:%[^ :,]+]] : !map.index<2>) capturing {[[BIN:%[^ :,]+]] = [[IN]] : !mogg.buffer, [[BRHS:%[^ :,]+]] = [[RHS]] : !mogg.tensor, [[BOUT:%[^ :,]+]] = [[OUT]] : !mogg.buffer}
// CHECK-NEXT:   mogg.tensor.load [[BIN]][[[IDX]]] : !mogg.buffer, !map.index<2> -> !mogg.simd<f32>
// CHECK: mogg.tensor.store [[BOUT]][[[IDX]]] = {{%[^ ]+}} : !mogg.buffer, !map.index<2>, !mogg.simd<f32>
mo.graph @in_place_add(%buf: !mo.buffer<[2, 2], f32>, %rhs: !mo.tensor<[2, 2], f32>) -> (!mo.buffer<[2, 2], f32>, !mo.chain) {
  %ch0 = mo.chain.create(%buf) : !mo.buffer<[2, 2], f32>
  %vbuf, %ch1 = mo.mutable.load(%buf, %ch0) : (!mo.buffer<[2, 2], f32>, !mo.chain) -> (!mo.tensor<[2, 2], f32>, !mo.chain)
  %res = mo.add(%vbuf, %rhs) : !mo.tensor<[2, 2], f32>
  %ch2 = mo.mutable.store(%buf, %res, %ch1) : (!mo.buffer<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.chain) -> (!mo.chain)
  mo.output %buf, %ch2 : !mo.buffer<[2, 2], f32>, !mo.chain
}
