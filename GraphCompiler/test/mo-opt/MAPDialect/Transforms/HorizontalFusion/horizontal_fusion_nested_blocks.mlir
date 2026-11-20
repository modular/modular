// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-horizontal | FileCheck %s

// Kernels inside `mo.if` branches fuse as well, each block with its own kernels
// only. Every kernel here also shares an operand with kernels in the other
// blocks, and those pairs stay apart: fusion never crosses a block.

// CHECK-LABEL: mo.graph @pairs_in_every_block
// CHECK:       %[[IF:[a-z0-9]+]]:2 = mo.if
// CHECK-NEXT:  %[[THEN:[a-z0-9]+]]:2 = map.iter.elementwise<[4]>
// CHECK:       map.call.elementwise @{{.*}}.Mul
// CHECK:       map.call.elementwise @{{.*}}.Sub
// CHECK:       mo.yield %[[THEN]]#0, %[[THEN]]#1
// CHECK-NEXT:  } else {
// CHECK-NEXT:  %[[ELSE:[a-z0-9]+]]:2 = map.iter.elementwise<[4]>
// CHECK:       map.call.elementwise @{{.*}}.Add
// CHECK:       map.call.elementwise @{{.*}}.Sub
// CHECK:       mo.yield %[[ELSE]]#0, %[[ELSE]]#1
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[OUTER:[a-z0-9]+]]:2 = map.iter.elementwise<[4]>
// CHECK:       map.call.elementwise @{{.*}}.Add
// CHECK:       map.call.elementwise @{{.*}}.Mul
// CHECK:       mo.output %[[OUTER]]#0, %[[OUTER]]#1, %[[IF]]#0, %[[IF]]#1
mo.graph @pairs_in_every_block(
  %cond: !mo.tensor<[], bool>,
  %a: !mo.tensor<[4], f32>,
  %b: !mo.tensor<[4], f32>,
  %c: !mo.tensor<[4], f32>
) -> (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>, !mo.tensor<[4], f32>, !mo.tensor<[4], f32>) {
  %x = mo.add(%a, %b) : !mo.tensor<[4], f32>
  %t:2 = mo.if (%cond : !mo.tensor<[], bool>) -> !mo.tensor<[4], f32>, !mo.tensor<[4], f32> {
    %m = mo.mul(%b, %c) : !mo.tensor<[4], f32>
    %n = mo.sub(%b, %a) : !mo.tensor<[4], f32>
    mo.yield %m, %n : !mo.tensor<[4], f32>, !mo.tensor<[4], f32>
  } else {
    %p = mo.add(%c, %a) : !mo.tensor<[4], f32>
    %q = mo.sub(%c, %b) : !mo.tensor<[4], f32>
    mo.yield %p, %q : !mo.tensor<[4], f32>, !mo.tensor<[4], f32>
  }
  %y = mo.mul(%a, %c) : !mo.tensor<[4], f32>
  mo.output %x, %y, %t#0, %t#1 : !mo.tensor<[4], f32>, !mo.tensor<[4], f32>, !mo.tensor<[4], f32>, !mo.tensor<[4], f32>
}

// The escaping-reader merge inside a branch: the branch yields the matmul's
// result and an add that reads it, so the add moves into the matmul's epilogue
// and the matmul keeps its own store for the yield.

// CHECK-LABEL: mo.graph @escaping_reader_in_branch
// CHECK:       %[[IF:[a-z0-9]+]]:2 = mo.if
// CHECK-NEXT:  %[[MM:[a-z0-9]+]]:2 = map.iter.opaque
// CHECK:       map.iter.epilogue
// CHECK-NEXT:  map.store
// CHECK-NEXT:  map.load
// CHECK-NEXT:  map.call.elementwise @{{.*}}.Add
// CHECK-NEXT:  map.store
// CHECK:       map.call.opaque @{{.*}}.Matmul
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.yield %[[MM]]#0, %[[MM]]#1
// CHECK-NEXT:  } else {
// CHECK:       mo.output %[[IF]]#0, %[[IF]]#1
mo.graph @escaping_reader_in_branch(
  %cond: !mo.tensor<[], bool>,
  %x: !mo.tensor<[8, 8], f32>,
  %y: !mo.tensor<[8, 8], f32>,
  %z: !mo.tensor<[8, 8], f32>
) -> (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) {
  %r:2 = mo.if (%cond : !mo.tensor<[], bool>) -> !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32> {
    %mm = mo.matmul(%x, %y) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
    %s = mo.add(%mm, %z) : !mo.tensor<[8, 8], f32>
    mo.yield %mm, %s : !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>
  } else {
    mo.yield %x, %y : !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>
  }
  mo.output %r#0, %r#1 : !mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>
}
