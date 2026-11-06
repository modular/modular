// RUN: mo-opt %s --mlir-very-unsafe-disable-verifier-on-parsing --gc-pipeline-pre-jit="dump-mojo-text=true" -o /dev/null 2>&1 | FileCheck %s

// Rowwise kernels with a map phase, written as MAP IR: the pipeline imports
// the monoids and elementwise kernels they name, so the parser can't verify
// them first. Row-level ops run once per row at the fold's width, and a map
// phase prints as `row.elementwise`, reading row values as `.slice[w]()`.

// An rms_norm written as MAP IR: the row-level divide, add and rsqrt run once
// per row at the fold's width; the map phase reads the result as `.slice[w]()`
// and loads gamma at the context's element alignment, like the hand-written
// kernel.
// CHECK-LABEL: def __iter_opaque(
// CHECK:       var [[ROW4:.*]] = algorithm.rowwise.Row[params, DType.float32, DType.float32, 1, {{.*}}, is_cached = True](row_coords, layout.Idx[128], row_ctx, [[LOAD4:load_+[0-9]+]])
// CHECK:       var [[ACC4:.*]] = [[ROW4]].reduce[builtin_kernels.monoids.ReduceSum[dtype = DType.float32, W = params.simd_width]]
// CHECK-NEXT:  var [[MS:.*]] = builtin_kernels.elementwise.Div.elementwise[dtype = DType.float32]([[ACC4]].acc, SIMD[DType.float32, params.simd_width](128.0))
// CHECK-NEXT:  var [[V:.*]] = builtin_kernels.elementwise.Add.elementwise[dtype = DType.float32]([[MS]], SIMD[DType.float32, params.simd_width](9.99999997E-7))
// CHECK-NEXT:  var [[INV:.*]] = builtin_kernels.elementwise.Rsqrt.elementwise[dtype = DType.float32]([[V]])
// CHECK-NEXT:  def [[EW:elementwise_+[0-9]+]][w : Int](tile: SIMD[DType.float32, w], idx: {{.*}}) {var}:
// CHECK-NEXT:    comptime [[AL:.*]] = algorithm.rowwise_types.Context[params].element_alignment[DType.float32, w]()
// CHECK-NEXT:    var [[NX:.*]] = builtin_kernels.elementwise.Mul.elementwise[dtype = DType.float32](tile, [[INV]].slice[w]())
// CHECK-NEXT:    var [[G:.*]] = managed_tensor_slice.simd_load_from_tensor_pointer[{{.*}}simd_width = w, element_alignment = [[AL]]]({{.*}}idx
// CHECK-NEXT:    var [[Y:.*]] = builtin_kernels.elementwise.Mul.elementwise[dtype = DType.float32]([[NX]], [[G]])
// CHECK-NEXT:    managed_tensor_slice.simd_store_into_tensor_pointer[{{.*}}simd_width = w, element_alignment = [[AL]]]({{.*}}idx{{.*}}[[Y]]))
// CHECK:       [[ROW4]].elementwise([[EW]], [[LOAD4]])
// CHECK:       {{.*}}._rowwise[builtin_kernels.monoids.ReduceSum[dtype = DType.float32, W = 1], DType.float32, DType.float32, axis = 1, num_phases = 2, computationally_expensive = True, dtype_size = 4]
mo.graph @rmsNorm(%x: !mo.tensor<[8, 128], f32>, %g: !mo.tensor<[128], f32>) -> !mo.tensor<[8, 128], f32> {
  %n = mo.constant <#M.dense_array<1.280000e+02> : tensor<f32>> : !mo.tensor<[], f32>
  %e = mo.constant <#M.dense_array<9.99999997E-7> : tensor<f32>> : !mo.tensor<[], f32>
  %r = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%x: !mo.tensor<[8, 128], f32>, %g: !mo.tensor<[128], f32>, %n: !mo.tensor<[], f32>, %e: !mo.tensor<[], f32>)
      outputs (%out: !mo.tensor<[8, 128], f32>)
      row     (%row: !map.index<2>) -> !mo.tensor<[8, 128], f32> {
    %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum index (%i: !map.index<2>) -> f32 {
      %v = map.load %x[%i] : !mo.tensor<[8, 128], f32>
      %sq = map.call.elementwise @builtin_kernels.elementwise.Mul(%i, %v, %v) : !map.index<2>, f32, f32 -> f32
      map.yield %sq : f32
    }
    %si = map.index.broadcast<#map.index.broadcast.spec<[] -> [8, 1]>> (%row) : !map.index<2> -> !map.index<0>
    %nv = map.load %n[%si] : !mo.tensor<[], f32>
    %ms = map.call.elementwise @builtin_kernels.elementwise.Div(%row, %acc, %nv) : !map.index<2>, f32, f32 -> f32
    %ev = map.load %e[%si] : !mo.tensor<[], f32>
    %vv = map.call.elementwise @builtin_kernels.elementwise.Add(%row, %ms, %ev) : !map.index<2>, f32, f32 -> f32
    %inv = map.call.elementwise @builtin_kernels.elementwise.Rsqrt(%row, %vv) : !map.index<2>, f32 -> f32
    map.iter.map index (%j: !map.index<2>) {
      %xv = map.load %x[%j] : !mo.tensor<[8, 128], f32>
      %nx = map.call.elementwise @builtin_kernels.elementwise.Mul(%j, %xv, %inv) : !map.index<2>, f32, f32 -> f32
      %gi = map.index.broadcast<#map.index.broadcast.spec<[128] -> [8, 128]>> (%j) : !map.index<2> -> !map.index<1>
      %gv = map.load %g[%gi] : !mo.tensor<[128], f32>
      %y = map.call.elementwise @builtin_kernels.elementwise.Mul(%j, %nx, %gv) : !map.index<2>, f32, f32 -> f32
      map.store %out[%j] = %y : !mo.tensor<[8, 128], f32>
    }
  }
  mo.output %r : !mo.tensor<[8, 128], f32>
}

// A map phase over a dynamic axis: the row is cached, the row-level rsqrt runs
// once per row, and the phase reads it as `.slice[w]()`.
// CHECK-LABEL: def __iter_opaque_0(
// CHECK:       var [[ROW3:.*]] = algorithm.rowwise.Row[params, DType.float32, DType.float32, 1, {{.*}}, is_cached = True](row_coords, Int({{.*}}[1].value()), row_ctx, [[LOAD3:load_+[0-9]+]])
// CHECK:       var [[ACC3:.*]] = [[ROW3]].reduce[
// CHECK-NEXT:  var [[INV:.*]] = builtin_kernels.elementwise.Rsqrt.elementwise[{{.*}}]([[ACC3]].acc)
// CHECK-NEXT:  def [[EW:elementwise_+[0-9]+]][w : Int](tile: SIMD[DType.float32, w], {{.*}}) {var}:
// CHECK-NEXT:    comptime [[AL:.*]] = algorithm.rowwise_types.Context[params].element_alignment[DType.float32, w]()
// CHECK-NEXT:    var [[N:.*]] = builtin_kernels.elementwise.Mul.elementwise[{{.*}}](tile, [[INV]].slice[w]())
// CHECK-NEXT:    managed_tensor_slice.simd_store_into_tensor_pointer[{{.*}}simd_width = w, element_alignment = [[AL]]]({{.*}}idx{{.*}}[[N]]))
// CHECK:       [[ROW3]].elementwise([[EW]], [[LOAD3]])
// CHECK:       _rowwise[{{.*}}num_phases = 2, computationally_expensive = True, dtype_size = 4]
mo.graph @normalize<M, N>(%x: !mo.tensor<[M, N], f32>) -> !mo.tensor<[M, N], f32> {
  %r = map.iter.rowwise<[M, N], axis = 1>
      inputs  (%x: !mo.tensor<[M, N], f32>)
      outputs (%out: !mo.tensor<[M, N], f32>)
      row     (%row: !map.index<2>) -> !mo.tensor<[M, N], f32> {
    %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum index (%i: !map.index<2>) -> f32 {
      %v = map.load %x[%i] : !mo.tensor<[M, N], f32>
      %sq = map.call.elementwise @builtin_kernels.elementwise.Mul(%i, %v, %v) : !map.index<2>, f32, f32 -> f32
      map.yield %sq : f32
    }
    %inv = map.call.elementwise @builtin_kernels.elementwise.Rsqrt(%row, %acc) : !map.index<2>, f32 -> f32
    map.iter.map index (%j: !map.index<2>) {
      %v = map.load %x[%j] : !mo.tensor<[M, N], f32>
      %y = map.call.elementwise @builtin_kernels.elementwise.Mul(%j, %v, %inv) : !map.index<2>, f32, f32 -> f32
      map.store %out[%j] = %y : !mo.tensor<[M, N], f32>
    }
  }
  mo.output %r : !mo.tensor<[M, N], f32>
}

// A per-row input a map phase reads through a row value: it loads once per
// row, one lane per row, and spreads to the fold's width.
// CHECK-LABEL: def __iter_opaque_1(
// CHECK:       var [[ROW5:.*]] = algorithm.rowwise.Row[
// CHECK:       var [[ACC5:.*]] = [[ROW5]].reduce[
// CHECK-NEXT:  var [[S:.*]] = managed_tensor_slice.simd_load_from_tensor_pointer[{{.*}}simd_width = params.emit_tile_width{{.*}}algorithm.rowwise_types.at_axis[1]([[ROW5]].row_coord, 0)
// CHECK-NEXT:  var [[SW:.*]] = {{.*}}._rowwise_splat_lane([[ROW5]], [[S]])
// CHECK-NEXT:  var [[INV:.*]] = builtin_kernels.elementwise.Mul.elementwise[{{.*}}]([[ACC5]].acc, [[SW]])
// CHECK-NEXT:  def [[EW:elementwise_+[0-9]+]][w : Int](tile: SIMD[DType.float32, w], {{.*}}) {var}:
// CHECK:         builtin_kernels.elementwise.Mul.elementwise[{{.*}}](tile, [[INV]].slice[w]())
mo.graph @scaleByRow(%x: !mo.tensor<[8, 128], f32>, %s: !mo.tensor<[8, 1], f32>) -> !mo.tensor<[8, 128], f32> {
  %r = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%x: !mo.tensor<[8, 128], f32>, %s: !mo.tensor<[8, 1], f32>)
      outputs (%out: !mo.tensor<[8, 128], f32>)
      row     (%row: !map.index<2>) -> !mo.tensor<[8, 128], f32> {
    %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum index (%i: !map.index<2>) -> f32 {
      %v = map.load %x[%i] : !mo.tensor<[8, 128], f32>
      map.yield %v : f32
    }
    %sv = map.load %s[%row] : !mo.tensor<[8, 1], f32>
    %inv = map.call.elementwise @builtin_kernels.elementwise.Mul(%row, %acc, %sv) : !map.index<2>, f32, f32 -> f32
    map.iter.map index (%j: !map.index<2>) {
      %v = map.load %x[%j] : !mo.tensor<[8, 128], f32>
      %y = map.call.elementwise @builtin_kernels.elementwise.Mul(%j, %v, %inv) : !map.index<2>, f32, f32 -> f32
      map.store %out[%j] = %y : !mo.tensor<[8, 128], f32>
    }
  }
  mo.output %r : !mo.tensor<[8, 128], f32>
}
