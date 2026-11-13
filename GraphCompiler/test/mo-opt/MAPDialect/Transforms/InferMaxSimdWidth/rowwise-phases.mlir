// RUN: mo-opt %s --mlir-very-unsafe-disable-verifier-on-parsing \
// RUN:   --mogg-import-packages --map-pipeline | FileCheck %s

// A cache and a map phase, like a fold, take the rowwise kernel's own index,
// so they are not index computation; nothing in this normalization limits the
// width, so its body gets no `max_simd_width`.

// CHECK-LABEL: mo.graph @rmsNorm
// CHECK:       map.iter.rowwise.body<[2, 4], axis = 1>
// CHECK-NOT:   max_simd_width
// CHECK:       map.iter.cache
// CHECK:       map.iter.map
// CHECK:       mo.output
mo.graph @rmsNorm(%x: !mo.tensor<[2, 4], f32>, %g: !mo.tensor<[4], f32>, %n: !mo.tensor<[], f32>) -> !mo.tensor<[2, 4], f32> {
  %r = map.iter.rowwise<[2, 4], axis = 1>
      inputs  (%x: !mo.tensor<[2, 4], f32>, %g: !mo.tensor<[4], f32>, %n: !mo.tensor<[], f32>)
      outputs (%out: !mo.tensor<[2, 4], f32>)
      row     (%row: !map.index<2>) -> !mo.tensor<[2, 4], f32> {
    %c = map.iter.cache index (%k: !map.index<2>) -> f32 {
      %v = map.load %x[%k] : !mo.tensor<[2, 4], f32>
      map.yield %v : f32
    }
    %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum index (%i: !map.index<2>) -> f32 {
      %v = map.iter.cache.load %c[%i] : !map.index<2>, f32
      %sq = map.call.elementwise @builtin_kernels.elementwise.Mul(%i, %v, %v) : !map.index<2>, f32, f32 -> f32
      map.yield %sq : f32
    }
    %si = map.index.broadcast<#map.index.broadcast.spec<[] -> [2, 1]>> (%row) : !map.index<2> -> !map.index<0>
    %nv = map.load %n[%si] : !mo.tensor<[], f32>
    %ms = map.call.elementwise @builtin_kernels.elementwise.Div(%row, %acc, %nv) : !map.index<2>, f32, f32 -> f32
    %inv = map.call.elementwise @builtin_kernels.elementwise.Rsqrt(%row, %ms) : !map.index<2>, f32 -> f32
    map.iter.map index (%j: !map.index<2>) {
      %xv = map.iter.cache.load %c[%j] : !map.index<2>, f32
      %nx = map.call.elementwise @builtin_kernels.elementwise.Mul(%j, %xv, %inv) : !map.index<2>, f32, f32 -> f32
      %gi = map.index.broadcast<#map.index.broadcast.spec<[4] -> [2, 4]>> (%j) : !map.index<2> -> !map.index<1>
      %gv = map.load %g[%gi] : !mo.tensor<[4], f32>
      %y = map.call.elementwise @builtin_kernels.elementwise.Mul(%j, %nx, %gv) : !map.index<2>, f32, f32 -> f32
      map.store %out[%j] = %y : !mo.tensor<[2, 4], f32>
    }
  }
  mo.output %r : !mo.tensor<[2, 4], f32>
}
