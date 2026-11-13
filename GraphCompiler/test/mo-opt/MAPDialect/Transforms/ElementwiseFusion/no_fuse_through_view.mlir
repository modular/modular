// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-kernels | FileCheck %s

// A consumer that reads an elementwise producer through a broadcast reads each
// producer element once per broadcast row. Fusing would compute the exp that
// many times, so the exp stays unfused.

// CHECK-LABEL: mo.graph @noFuseThroughBroadcast
// CHECK:       map.iter.elementwise<[1, 64]>
// CHECK:       map.call.elementwise @{{.*}}.Exp
// CHECK:       map.iter.elementwise<[8, 64]>
// CHECK:       map.index.broadcast
// CHECK:       map.call.elementwise @{{.*}}.Add
mo.graph @noFuseThroughBroadcast(%x: !mo.tensor<[1, 64], f32>, %y: !mo.tensor<[8, 64], f32>) -> !mo.tensor<[8, 64], f32> {
  %e = mo.exp(%x) : !mo.tensor<[1, 64], f32>
  %b = mo.static.broadcast_to(%e) : !mo.tensor<[1, 64], f32> -> !mo.tensor<[8, 64], f32>
  %res = mo.add(%b, %y) : !mo.tensor<[8, 64], f32>
  mo.output %res : !mo.tensor<[8, 64], f32>
}

// The exp does not fuse into a slice that is a pure view, so
// `RaiseBackContiguousViews` can still make the slice zero-copy.

// CHECK-LABEL: mo.graph @noFuseIntoPureView
// CHECK:       map.iter.elementwise<[8, 64]>
// CHECK:       map.call.elementwise @{{.*}}.Exp
// CHECK:       map.iter.elementwise<[8, 32]>
// CHECK-NOT:   map.call.elementwise
// CHECK:       mo.output
mo.graph @noFuseIntoPureView(%x: !mo.tensor<[8, 64], f32>) -> !mo.tensor<[8, 32], f32> {
  %e = mo.exp(%x) : !mo.tensor<[8, 64], f32>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %start = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop = mo.constant <#M.dense_array<8, 32> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %s = mo.slice(%e, %start, %stop, %step) : (
    !mo.tensor<[8, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[8, 32], f32>
  mo.output %s : !mo.tensor<[8, 32], f32>
}
