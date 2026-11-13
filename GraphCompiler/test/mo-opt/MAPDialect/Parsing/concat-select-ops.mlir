// RUN: mo-opt %s | FileCheck %s
// RUN: mo-opt %s | mo-opt

// Round-trip map.index.concat.select in a foreach: two static inputs, and three
// symbolic ones on the outer axis. Each region loads its own input at its
// argument, the index rebased into that input.

// CHECK-LABEL: func.func @concat_select_static
// CHECK:       map.iter.elementwise<[4, 24]>
// CHECK:       index   (%[[IDX:.*]]: !map.index<2>)
// CHECK-NEXT:  %[[V:.*]] = map.index.concat.select<#map.index.concat.spec<[4, 8], [4, 16] -> [4, 24], axis = 1>> (%[[IDX]]) : !map.index<2> -> f32 (%[[I0:.*]]: !map.index<2>) {
// CHECK-NEXT:    %[[A:.*]] = map.load %{{.*}}[%[[I0]]] : !mo.tensor<[4, 8], f32>
// CHECK-NEXT:    map.yield %[[A]] : f32
// CHECK-NEXT:  }, (%[[I1:.*]]: !map.index<2>) {
// CHECK-NEXT:    %[[B:.*]] = map.load %{{.*}}[%[[I1]]] : !mo.tensor<[4, 16], f32>
// CHECK-NEXT:    map.yield %[[B]] : f32
// CHECK-NEXT:  }
// CHECK-NEXT:  map.store %{{.*}}[%[[IDX]]] = %[[V]] : !mo.tensor<[4, 24], f32>
func.func @concat_select_static(%a: !mo.tensor<[4, 8], f32>, %b: !mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 24], f32> {
  %r = map.iter.elementwise<[4, 24]>
  inputs  (%a: !mo.tensor<[4, 8], f32>, %b: !mo.tensor<[4, 16], f32>)
  outputs (%out: !mo.tensor<[4, 24], f32>)
  index   (%idx: !map.index<2>) -> !mo.tensor<[4, 24], f32> {
    %v = map.index.concat.select<#map.index.concat.spec<[4, 8], [4, 16] -> [4, 24], axis = 1>> (%idx) : !map.index<2> -> f32 (%i: !map.index<2>) {
      %x = map.load %a[%i] : !mo.tensor<[4, 8], f32>
      map.yield %x : f32
    }, (%i: !map.index<2>) {
      %x = map.load %b[%i] : !mo.tensor<[4, 16], f32>
      map.yield %x : f32
    }
    map.store %out[%idx] = %v : !mo.tensor<[4, 24], f32>
  }
  return %r : !mo.tensor<[4, 24], f32>
}

// CHECK-LABEL: func.func @concat_select_symbolic
// CHECK:       map.index.concat.select<#map.index.concat.spec<[M, 8], [N, 8], [2, 8] -> [P, 8], axis = 0>>
// CHECK-COUNT-3: map.yield
func.func @concat_select_symbolic(%a: !mo.tensor<[M, 8], f32>, %b: !mo.tensor<[N, 8], f32>, %c: !mo.tensor<[2, 8], f32>) -> !mo.tensor<[P, 8], f32> {
  %r = map.iter.elementwise<[P, 8]>
  inputs  (%a: !mo.tensor<[M, 8], f32>, %b: !mo.tensor<[N, 8], f32>, %c: !mo.tensor<[2, 8], f32>)
  outputs (%out: !mo.tensor<[P, 8], f32>)
  index   (%idx: !map.index<2>) -> !mo.tensor<[P, 8], f32> {
    %v = map.index.concat.select<#map.index.concat.spec<[M, 8], [N, 8], [2, 8] -> [P, 8], axis = 0>> (%idx) : !map.index<2> -> f32 (%i: !map.index<2>) {
      %x = map.load %a[%i] : !mo.tensor<[M, 8], f32>
      map.yield %x : f32
    }, (%i: !map.index<2>) {
      %x = map.load %b[%i] : !mo.tensor<[N, 8], f32>
      map.yield %x : f32
    }, (%i: !map.index<2>) {
      %x = map.load %c[%i] : !mo.tensor<[2, 8], f32>
      map.yield %x : f32
    }
    map.store %out[%idx] = %v : !mo.tensor<[P, 8], f32>
  }
  return %r : !mo.tensor<[P, 8], f32>
}
