// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute %t.mlir --result-output-style=full | FileCheck %s

// Equiv: print(torch.arange(0, 15).reshape(3, 5)[1:3:1, 0:5:2])
mo.graph @basic_slice<() -> D0, D1>() -> !mo.tensor<[D0, D1], f32> {
  %input = mo.constant <#M.dense_array<0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0> : tensor<15xf32>> : !mo.tensor<[3, 5], f32>

  %start = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop = mo.constant <#M.dense_array<3, 5> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>

  %res = mo.slice<() -> D2, D3>(%input, %start, %stop, %step) : (
    !mo.tensor<[3, 5], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[D2, D3], f32>
  mo.output<D2, D3> %res : !mo.tensor<[D2, D3], f32>
}

// CHECK{LITERAL}: 'basic_slice' returned tensor<2x3xf32> [5.000000e+00, 7.000000e+00, 9.000000e+00, 1.000000e+01, 1.200000e+01, 1.400000e+01]


// Equiv: print(torch.arange(0, 15)[0:15:5])
mo.graph @slice_1D<() -> D0>() -> !mo.tensor<[D0], f32> {
  %input = mo.constant <#M.dense_array<0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0> : tensor<15xf32>> : !mo.tensor<[15], f32>

  %start = mo.constant <#M.dense_array<0> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %stop = mo.constant <#M.dense_array<15> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %step = mo.constant <#M.dense_array<5> : tensor<1xsi64>> : !mo.tensor<[1], si64>

  %res = mo.slice<() -> D1>(%input, %start, %stop, %step) : (
    !mo.tensor<[15], f32>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>
  ) -> !mo.tensor<[D1], f32>
  mo.output<D1> %res : !mo.tensor<[D1], f32>
}

// CHECK{LITERAL}: 'slice_1D' returned tensor<3xf32> [0.000000e+00, 5.000000e+00, 1.000000e+01]


// Equiv: print(torch.arange(0, 15)[-5:-1:1])
mo.graph @slice_1D_neg<() -> D0>() -> !mo.tensor<[D0], f32> {
  %input = mo.constant <#M.dense_array<0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0> : tensor<15xf32>> : !mo.tensor<[15], f32>

  %start = mo.constant <#M.dense_array<-5> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %stop = mo.constant <#M.dense_array<-1> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %step = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[1], si64>

  %res = mo.slice<() -> D1>(%input, %start, %stop, %step) : (
    !mo.tensor<[15], f32>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>
  ) -> !mo.tensor<[D1], f32>
  mo.output<D1> %res : !mo.tensor<[D1], f32>
}

// CHECK{LITERAL}: 'slice_1D_neg' returned tensor<4xf32> [1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01]


// Equiv: print(torch.arange(0, 15).reshape(3, 5)[-2:-1:1, -4:-1:2])
mo.graph @slice_2D_neg<() -> D0, D1>() -> !mo.tensor<[D0, D1], f32> {
  %input = mo.constant <#M.dense_array<0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0> : tensor<15xf32>> : !mo.tensor<[3, 5], f32>

  %start = mo.constant <#M.dense_array<-2, -4> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop = mo.constant <#M.dense_array<-1, -1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>

  %res = mo.slice<() -> D2, D3>(%input, %start, %stop, %step) : (
    !mo.tensor<[3, 5], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[D2, D3], f32>
  mo.output<D2, D3> %res : !mo.tensor<[D2, D3], f32>
}

// CHECK{LITERAL}: 'slice_2D_neg' returned tensor<1x2xf32> [6.000000e+00, 8.000000e+00]


// Equiv: print(np.arange(0, 15).reshape(5, 3)[-1:0:-2, -1:0:-1])
mo.graph @slice_2D_neg_step<() -> D0, D1>() -> !mo.tensor<[D0, D1], f32> {
  %input = mo.constant <#M.dense_array<0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0> : tensor<15xf32>> : !mo.tensor<[5, 3], f32>

  %start = mo.constant <#M.dense_array<-1, -1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<-2, -1> : tensor<2xsi64>> : !mo.tensor<[2], si64>

  %res = mo.slice<() -> D2, D3>(%input, %start, %stop, %step) : (
    !mo.tensor<[5, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[D2, D3], f32>
  mo.output<D2, D3> %res : !mo.tensor<[D2, D3], f32>
}

// CHECK{LITERAL}: 'slice_2D_neg_step' returned tensor<2x2xf32> [1.400000e+01, 1.300000e+01, 8.000000e+00, 7.000000e+00]


// Equiv: print(np.arange(0, 15).reshape(3, 5)[3:-2:-1, 5:-3:-1])
mo.graph @slice_neg_edge<() -> D0, D1>() -> !mo.tensor<[D0, D1], f32> {
  %input = mo.constant <#M.dense_array<0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0> : tensor<15xf32>> : !mo.tensor<[3, 5], f32>

  %start = mo.constant <#M.dense_array<3, 5> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop = mo.constant <#M.dense_array<-2, -3> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<-1, -1> : tensor<2xsi64>> : !mo.tensor<[2], si64>

  %res = mo.slice<() -> D2, D3>(%input, %start, %stop, %step) : (
    !mo.tensor<[3, 5], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[D2, D3], f32>
  mo.output<D2, D3> %res : !mo.tensor<[D2, D3], f32>
}

// CHECK{LITERAL}: 'slice_neg_edge' returned tensor<1x2xf32> [1.400000e+01, 1.300000e+01]


mo.graph @strided_slice_out_of_bounds_indices<() -> D0, D1>() -> !mo.tensor<[D0, D1], f32> {
  %input = mo.constant <#M.dense_array<0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0> : tensor<9xf32>> : !mo.tensor<[3, 3], f32>
  %0 = mo.constant <#M.dense_array<1, -5> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %1 = mo.constant <#M.dense_array<5, -1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %2 = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %3 = mo.slice<() -> D2, D3>(%input, %0, %1, %2) : (!mo.tensor<[3, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[D2, D3], f32>
  mo.output<D2, D3> %3 : !mo.tensor<[D2, D3], f32>
}

// CHECK{LITERAL}: 'strided_slice_out_of_bounds_indices' returned tensor<2x2xf32> [3.000000e+00, 4.000000e+00, 6.000000e+00, 7.000000e+00]


mo.graph @slice_fallback_shape_func<() -> D0, D1, D2>(
  %input: !mo.tensor<[10, 20, 30], f32>
) -> !mo.tensor<[D0, D1, D2], si32> {

  %start = mo.constant <#M.dense_array<0, 10, 5> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %stop = mo.constant <#M.dense_array<100, -100, 15> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %step = mo.constant <#M.dense_array<2, -2, 3> : tensor<3xsi32>> : !mo.tensor<[3], si32>

  mo.invoke_shape_func["mo.slice"]<() -> D3, D4, D5>(%input, %start, %stop, %step)
    [dataDeptTensors = [1 : i32, 2 : i32, 3 : i32]] :
    (!mo.tensor<[10, 20, 30], f32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>)

  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  %val = mo.static.broadcast_to(%one): !mo.tensor<[], si32> -> !mo.tensor<[D3, D4, D5], si32>

  mo.output<D3, D4, D5> %val : !mo.tensor<[D3, D4, D5], si32>
}
// CHECK{LITERAL}: 'slice_fallback_shape_func' returned tensor<5x6x4xsi32>
