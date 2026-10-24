// The environment keeps glibc from handing freed memory straight back and
// scribbles over it, so a read of an erased op crashes instead of finding a
// recycled one.
// RUN: env GLIBC_TUNABLES=glibc.malloc.tcache_count=0:glibc.malloc.mxfast=0 MALLOC_PERTURB_=165 \
// RUN:   mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-horizontal --map-fuse-epilogue | FileCheck %s

// Epilogue fusion erases a consumer, and can rebuild its producer, while an
// `mo.if` branch reads one of them. The topology analysis that orders each
// block records that read against the op, so every block is sorted before
// anything fuses.

// The relu fuses into the matmul and is erased; a branch reads its result.
// CHECK-LABEL: mo.graph @consumer_read_in_branch
// CHECK:       %[[MM:[a-z0-9]+]] = map.iter.opaque
// CHECK:       map.call.elementwise @{{.*}}.ReLU
// CHECK:       map.call.opaque @{{.*}}.Matmul
// CHECK:       mo.if
// CHECK:       map.iter.opaque
// CHECK-NEXT:  inputs (%[[MM]]: !mo.tensor<[8, 8], f32>
mo.graph @consumer_read_in_branch(
  %c: !mo.tensor<[], bool>,
  %x: !mo.tensor<[8, 8], f32>,
  %y: !mo.tensor<[8, 8], f32>,
  %z: !mo.tensor<[8, 8], f32>
) -> !mo.tensor<[8, 8], f32> {
  %mm = mo.matmul(%x, %y) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
  %r = mo.relu(%mm) : !mo.tensor<[8, 8], f32>
  %t = mo.if (%c : !mo.tensor<[], bool>) -> !mo.tensor<[8, 8], f32> {
    %m = mo.matmul(%r, %z) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
    mo.yield %m : !mo.tensor<[8, 8], f32>
  } else {
    mo.yield %z : !mo.tensor<[8, 8], f32>
  }
  mo.output %t : !mo.tensor<[8, 8], f32>
}

// The paired adds fuse into the matmul, which is rebuilt with a result for the
// second add, and the pair is erased; a branch reads the second add.
// CHECK-LABEL: mo.graph @multi_output_consumer_read_in_branch
// CHECK:       %[[MM:[a-z0-9]+]]:2 = map.iter.opaque
// CHECK:       map.call.opaque @{{.*}}.Matmul
// CHECK:       mo.if
// CHECK:       map.iter.opaque
// CHECK-NEXT:  inputs (%[[MM]]#1: !mo.tensor<[8, 8], f32>
mo.graph @multi_output_consumer_read_in_branch(
  %c: !mo.tensor<[], bool>,
  %x: !mo.tensor<[8, 8], f32>,
  %y: !mo.tensor<[8, 8], f32>,
  %z: !mo.tensor<[8, 8], f32>,
  %w: !mo.tensor<[8, 8], f32>
) -> (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) {
  %mm = mo.matmul(%x, %y) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
  %a = mo.add(%mm, %z) : !mo.tensor<[8, 8], f32>
  %b = mo.add(%mm, %w) : !mo.tensor<[8, 8], f32>
  %t = mo.if (%c : !mo.tensor<[], bool>) -> !mo.tensor<[8, 8], f32> {
    %m = mo.matmul(%b, %z) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
    mo.yield %m : !mo.tensor<[8, 8], f32>
  } else {
    mo.yield %z : !mo.tensor<[8, 8], f32>
  }
  mo.output %a, %t : !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>
}
