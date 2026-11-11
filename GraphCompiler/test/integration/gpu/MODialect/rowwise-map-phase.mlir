// RUN: mo-opt %s --mlir-very-unsafe-disable-verifier-on-parsing --gc-pipeline="constant-fold=false default-device-label=gpu" -o %t.mlir
// RUN: mt --execute --default-device-label=gpu --result-output-style=full %t.mlir | FileCheck %s

// The MAP IR names kernels the pipeline imports, so the parser can't verify
// it first.

// The CPU test of the same name, on the GPU: a rowwise kernel with a map
// phase, an rms_norm whose results are exact.

// Rows [2, -2, 2, -2] and [4, 0, 0, 0], gamma [1, 0.5, 2, -1].
// CHECK{LITERAL}: 'rmsNormInner' returned tensor<2x4xf32> [1.000000e+00, -5.000000e-01, 2.000000e+00, 1.000000e+00, 2.000000e+00, 0.000000e+00, 0.000000e+00, -0.000000e+00]
mo.graph @rmsNormInner() -> !mo.tensor<[2, 4], f32> {
  %x = mo.constant <#M.dense_array<2.0, -2.0, 2.0, -2.0, 4.0, 0.0, 0.0, 0.0> : tensor<2x4xf32>> : !mo.tensor<[2, 4], f32>
  %g = mo.constant <#M.dense_array<1.0, 0.5, 2.0, -1.0> : tensor<4xf32>> : !mo.tensor<[4], f32>
  %n = mo.constant <#M.dense_array<4.0> : tensor<f32>> : !mo.tensor<[], f32>
  %e = mo.constant <#M.dense_array<0.0> : tensor<f32>> : !mo.tensor<[], f32>
  %c0 = mo.chain.create()
  %x_gpu, %c1 = mo.transfer[%c0] %x : !mo.tensor<[2, 4], f32> to <"gpu", 0>
  %g_gpu, %c2 = mo.transfer[%c1] %g : !mo.tensor<[4], f32> to <"gpu", 0>
  %n_gpu, %c3 = mo.transfer[%c2] %n : !mo.tensor<[], f32> to <"gpu", 0>
  %e_gpu, %c4 = mo.transfer[%c3] %e : !mo.tensor<[], f32> to <"gpu", 0>
  %r_gpu = map.iter.rowwise<[2, 4], axis = 1>
      inputs  (%x_gpu: !mo.tensor<[2, 4], f32, gpu:0>, %g_gpu: !mo.tensor<[4], f32, gpu:0>, %n_gpu: !mo.tensor<[], f32, gpu:0>, %e_gpu: !mo.tensor<[], f32, gpu:0>)
      outputs (%out: !mo.tensor<[2, 4], f32, gpu:0>)
      row     (%row: !map.index<2>) -> !mo.tensor<[2, 4], f32, gpu:0> {
    %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum index (%i: !map.index<2>) -> f32 {
      %v = map.load %x_gpu[%i] : !mo.tensor<[2, 4], f32, gpu:0>
      %sq = map.call.elementwise @builtin_kernels.elementwise.Mul(%i, %v, %v) : !map.index<2>, f32, f32 -> f32
      map.yield %sq : f32
    }
    %si = map.index.broadcast<#map.index.broadcast.spec<[] -> [2, 1]>> (%row) : !map.index<2> -> !map.index<0>
    %nv = map.load %n_gpu[%si] : !mo.tensor<[], f32, gpu:0>
    %ms = map.call.elementwise @builtin_kernels.elementwise.Div(%row, %acc, %nv) : !map.index<2>, f32, f32 -> f32
    %ev = map.load %e_gpu[%si] : !mo.tensor<[], f32, gpu:0>
    %vv = map.call.elementwise @builtin_kernels.elementwise.Add(%row, %ms, %ev) : !map.index<2>, f32, f32 -> f32
    %inv = map.call.elementwise @builtin_kernels.elementwise.Rsqrt(%row, %vv) : !map.index<2>, f32 -> f32
    map.iter.map index (%j: !map.index<2>) {
      %xv = map.load %x_gpu[%j] : !mo.tensor<[2, 4], f32, gpu:0>
      %nx = map.call.elementwise @builtin_kernels.elementwise.Mul(%j, %xv, %inv) : !map.index<2>, f32, f32 -> f32
      %gi = map.index.broadcast<#map.index.broadcast.spec<[4] -> [2, 4]>> (%j) : !map.index<2> -> !map.index<1>
      %gv = map.load %g_gpu[%gi] : !mo.tensor<[4], f32, gpu:0>
      %y = map.call.elementwise @builtin_kernels.elementwise.Mul(%j, %nx, %gv) : !map.index<2>, f32, f32 -> f32
      map.store %out[%j] = %y : !mo.tensor<[2, 4], f32, gpu:0>
    }
  }
  %r, %cend = mo.transfer[%c4] %r_gpu : !mo.tensor<[2, 4], f32, gpu:0> to <"cpu", 0>
  mo.output %r : !mo.tensor<[2, 4], f32>
}

// The same rows as columns, reduced over axis 0, gamma along that axis.
// CHECK{LITERAL}: 'rmsNormOuter' returned tensor<4x2xf32> [1.000000e+00, 2.000000e+00, -5.000000e-01, 0.000000e+00, 2.000000e+00, 0.000000e+00, 1.000000e+00, -0.000000e+00]
mo.graph @rmsNormOuter() -> !mo.tensor<[4, 2], f32> {
  %x = mo.constant <#M.dense_array<2.0, 4.0, -2.0, 0.0, 2.0, 0.0, -2.0, 0.0> : tensor<4x2xf32>> : !mo.tensor<[4, 2], f32>
  %g = mo.constant <#M.dense_array<1.0, 0.5, 2.0, -1.0> : tensor<4x1xf32>> : !mo.tensor<[4, 1], f32>
  %n = mo.constant <#M.dense_array<4.0> : tensor<f32>> : !mo.tensor<[], f32>
  %e = mo.constant <#M.dense_array<0.0> : tensor<f32>> : !mo.tensor<[], f32>
  %c0 = mo.chain.create()
  %x_gpu, %c1 = mo.transfer[%c0] %x : !mo.tensor<[4, 2], f32> to <"gpu", 0>
  %g_gpu, %c2 = mo.transfer[%c1] %g : !mo.tensor<[4, 1], f32> to <"gpu", 0>
  %n_gpu, %c3 = mo.transfer[%c2] %n : !mo.tensor<[], f32> to <"gpu", 0>
  %e_gpu, %c4 = mo.transfer[%c3] %e : !mo.tensor<[], f32> to <"gpu", 0>
  %r_gpu = map.iter.rowwise<[4, 2], axis = 0>
      inputs  (%x_gpu: !mo.tensor<[4, 2], f32, gpu:0>, %g_gpu: !mo.tensor<[4, 1], f32, gpu:0>, %n_gpu: !mo.tensor<[], f32, gpu:0>, %e_gpu: !mo.tensor<[], f32, gpu:0>)
      outputs (%out: !mo.tensor<[4, 2], f32, gpu:0>)
      row     (%row: !map.index<2>) -> !mo.tensor<[4, 2], f32, gpu:0> {
    %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum index (%i: !map.index<2>) -> f32 {
      %v = map.load %x_gpu[%i] : !mo.tensor<[4, 2], f32, gpu:0>
      %sq = map.call.elementwise @builtin_kernels.elementwise.Mul(%i, %v, %v) : !map.index<2>, f32, f32 -> f32
      map.yield %sq : f32
    }
    %si = map.index.broadcast<#map.index.broadcast.spec<[] -> [1, 2]>> (%row) : !map.index<2> -> !map.index<0>
    %nv = map.load %n_gpu[%si] : !mo.tensor<[], f32, gpu:0>
    %ms = map.call.elementwise @builtin_kernels.elementwise.Div(%row, %acc, %nv) : !map.index<2>, f32, f32 -> f32
    %ev = map.load %e_gpu[%si] : !mo.tensor<[], f32, gpu:0>
    %vv = map.call.elementwise @builtin_kernels.elementwise.Add(%row, %ms, %ev) : !map.index<2>, f32, f32 -> f32
    %inv = map.call.elementwise @builtin_kernels.elementwise.Rsqrt(%row, %vv) : !map.index<2>, f32 -> f32
    map.iter.map index (%j: !map.index<2>) {
      %xv = map.load %x_gpu[%j] : !mo.tensor<[4, 2], f32, gpu:0>
      %nx = map.call.elementwise @builtin_kernels.elementwise.Mul(%j, %xv, %inv) : !map.index<2>, f32, f32 -> f32
      %gi = map.index.broadcast<#map.index.broadcast.spec<[4, 1] -> [4, 2]>> (%j) : !map.index<2> -> !map.index<2>
      %gv = map.load %g_gpu[%gi] : !mo.tensor<[4, 1], f32, gpu:0>
      %y = map.call.elementwise @builtin_kernels.elementwise.Mul(%j, %nx, %gv) : !map.index<2>, f32, f32 -> f32
      map.store %out[%j] = %y : !mo.tensor<[4, 2], f32, gpu:0>
    }
  }
  %r, %cend = mo.transfer[%c4] %r_gpu : !mo.tensor<[4, 2], f32, gpu:0> to <"cpu", 0>
  mo.output %r : !mo.tensor<[4, 2], f32>
}

// A per-row input read at row level, then by the map phase: each element
// is scaled by the row's sum of squares (16) times b[row], b = [0.25, -0.5].
// CHECK{LITERAL}: 'rowScaleInner' returned tensor<2x4xf32> [8.000000e+00, -8.000000e+00, 8.000000e+00, -8.000000e+00, -3.200000e+01, -0.000000e+00, -0.000000e+00, -0.000000e+00]
mo.graph @rowScaleInner() -> !mo.tensor<[2, 4], f32> {
  %x = mo.constant <#M.dense_array<2.0, -2.0, 2.0, -2.0, 4.0, 0.0, 0.0, 0.0> : tensor<2x4xf32>> : !mo.tensor<[2, 4], f32>
  %b = mo.constant <#M.dense_array<0.25, -0.5> : tensor<2x1xf32>> : !mo.tensor<[2, 1], f32>
  %c0 = mo.chain.create()
  %x_gpu, %c1 = mo.transfer[%c0] %x : !mo.tensor<[2, 4], f32> to <"gpu", 0>
  %b_gpu, %c2 = mo.transfer[%c1] %b : !mo.tensor<[2, 1], f32> to <"gpu", 0>
  %r_gpu = map.iter.rowwise<[2, 4], axis = 1>
      inputs  (%x_gpu: !mo.tensor<[2, 4], f32, gpu:0>, %b_gpu: !mo.tensor<[2, 1], f32, gpu:0>)
      outputs (%out: !mo.tensor<[2, 4], f32, gpu:0>)
      row     (%row: !map.index<2>) -> !mo.tensor<[2, 4], f32, gpu:0> {
    %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum index (%i: !map.index<2>) -> f32 {
      %v = map.load %x_gpu[%i] : !mo.tensor<[2, 4], f32, gpu:0>
      %sq = map.call.elementwise @builtin_kernels.elementwise.Mul(%i, %v, %v) : !map.index<2>, f32, f32 -> f32
      map.yield %sq : f32
    }
    %bv = map.load %b_gpu[%row] : !mo.tensor<[2, 1], f32, gpu:0>
    %s = map.call.elementwise @builtin_kernels.elementwise.Mul(%row, %acc, %bv) : !map.index<2>, f32, f32 -> f32
    map.iter.map index (%j: !map.index<2>) {
      %xv = map.load %x_gpu[%j] : !mo.tensor<[2, 4], f32, gpu:0>
      %y = map.call.elementwise @builtin_kernels.elementwise.Mul(%j, %xv, %s) : !map.index<2>, f32, f32 -> f32
      map.store %out[%j] = %y : !mo.tensor<[2, 4], f32, gpu:0>
    }
  }
  %r, %cend = mo.transfer[%c2] %r_gpu : !mo.tensor<[2, 4], f32, gpu:0> to <"cpu", 0>
  mo.output %r : !mo.tensor<[2, 4], f32>
}

// The same rows as columns, reduced over axis 0, b along the other axis.
// CHECK{LITERAL}: 'rowScaleOuter' returned tensor<4x2xf32> [8.000000e+00, -3.200000e+01, -8.000000e+00, -0.000000e+00, 8.000000e+00, -0.000000e+00, -8.000000e+00, -0.000000e+00]
mo.graph @rowScaleOuter() -> !mo.tensor<[4, 2], f32> {
  %x = mo.constant <#M.dense_array<2.0, 4.0, -2.0, 0.0, 2.0, 0.0, -2.0, 0.0> : tensor<4x2xf32>> : !mo.tensor<[4, 2], f32>
  %b = mo.constant <#M.dense_array<0.25, -0.5> : tensor<1x2xf32>> : !mo.tensor<[1, 2], f32>
  %c0 = mo.chain.create()
  %x_gpu, %c1 = mo.transfer[%c0] %x : !mo.tensor<[4, 2], f32> to <"gpu", 0>
  %b_gpu, %c2 = mo.transfer[%c1] %b : !mo.tensor<[1, 2], f32> to <"gpu", 0>
  %r_gpu = map.iter.rowwise<[4, 2], axis = 0>
      inputs  (%x_gpu: !mo.tensor<[4, 2], f32, gpu:0>, %b_gpu: !mo.tensor<[1, 2], f32, gpu:0>)
      outputs (%out: !mo.tensor<[4, 2], f32, gpu:0>)
      row     (%row: !map.index<2>) -> !mo.tensor<[4, 2], f32, gpu:0> {
    %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum index (%i: !map.index<2>) -> f32 {
      %v = map.load %x_gpu[%i] : !mo.tensor<[4, 2], f32, gpu:0>
      %sq = map.call.elementwise @builtin_kernels.elementwise.Mul(%i, %v, %v) : !map.index<2>, f32, f32 -> f32
      map.yield %sq : f32
    }
    %bv = map.load %b_gpu[%row] : !mo.tensor<[1, 2], f32, gpu:0>
    %s = map.call.elementwise @builtin_kernels.elementwise.Mul(%row, %acc, %bv) : !map.index<2>, f32, f32 -> f32
    map.iter.map index (%j: !map.index<2>) {
      %xv = map.load %x_gpu[%j] : !mo.tensor<[4, 2], f32, gpu:0>
      %y = map.call.elementwise @builtin_kernels.elementwise.Mul(%j, %xv, %s) : !map.index<2>, f32, f32 -> f32
      map.store %out[%j] = %y : !mo.tensor<[4, 2], f32, gpu:0>
    }
  }
  %r, %cend = mo.transfer[%c2] %r_gpu : !mo.tensor<[4, 2], f32, gpu:0> to <"cpu", 0>
  mo.output %r : !mo.tensor<[4, 2], f32>
}
