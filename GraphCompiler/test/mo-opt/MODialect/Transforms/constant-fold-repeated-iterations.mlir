// UNSUPPORTED: asan
// RUN: mo-opt --mo-constant-fold=iterations=2 %s | FileCheck %s

// CHECK-LABEL: mo.graph @repeated_application
mo.graph @repeated_application(%e0 : !mo.tensor<[], si32>) -> !mo.tensor<[], si32> {
  // CHECK-NEXT: %{{.*}} = mo.constant <#M.dense_array<4>
  // CHECK-NEXT: mo.output
  %e1 = mo.constant <#M.dense_array<3> : tensor<1xsi32>> :
				!mo.tensor<[], si32>
  %e2 = mo.constant <#M.dense_array<2> : tensor<1xsi32>> :
				!mo.tensor<[], si32>
  %e3 = mo.add(%e1, %e1) : !mo.tensor<[], si32>
  %e4 = mo.greater_equal(%e3, %e2) :(!mo.tensor<[], si32>, !mo.tensor<[], si32>)
    -> !mo.tensor<[], bool>
  %e5 = mo.select(%e4, %e3, %e0) :
    (!mo.tensor<[], bool>, !mo.tensor<[], si32>, !mo.tensor<[], si32>)
    -> !mo.tensor<[], si32>
  %e6 = mo.sub(%e5, %e2) : !mo.tensor<[], si32>
  mo.output %e6 : !mo.tensor<[], si32>
}

// CHECK-LABEL: mo.graph @large_op
// CHECK-DAG: %{{.*}} = mo.constant <#M.dense_array<2, 3> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-DAG: %{{.*}} = mo.constant <#M.dense_array<2> : !M.array<1xsi64>> : !mo.tensor<[1], si64>
// CHECK-DAG: %{{.*}} = mo.constant <#M.dense_array<3, 1> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-DAG: %{{.*}} = mo.constant <#M.dense_array<0> : !M.array<1xsi64>> : !mo.tensor<[1], si64>
// CHECK-DAG: %{{.*}} = mo.constant <#M.dense_array<1> : !M.array<1xsi64>> : !mo.tensor<[1], si64>
// CHECK-DAG: %{{.*}} = mo.constant <#M.dense_array<1.000000e+00, 1.000000e+00, 1.000000e+00> : tensor<3xf32>> : !mo.tensor<[3], f32>
mo.graph @large_op(%arg0: !mo.tensor<[2, 2, 3], f32>) -> !mo.tensor<[2, 2], f32> {
  %0 = mo.constant <#M.dense_array<3> : !M.array<1xsi64>> : !mo.tensor<[1], si64>
  %1 = mo.constant <#M.dense_array<0> : !M.array<1xsi64>> : !mo.tensor<[1], si64>
  %2 = mo.constant <#M.dense_array<1.000000e+00, 1.000000e+00, 1.000000e+00> : tensor<3xf32>> : !mo.tensor<[3], f32>
  %3 = mo.constant <#M.dense_array<0> : !M.array<1xsi32>> : !mo.tensor<[], si32>
  %4 = mo.constant <#M.dense_array<1> : !M.array<1xsi64>> : !mo.tensor<[1], si64>
  %5 = mo.shape_of(%2) : (!mo.tensor<[3], f32>) -> !mo.tensor<[1], si64>
  %6 = mo.concat[0] (%5, %4) : (!mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[2], si64>
  %7 = mo.reshape<() -> D15>(%2, %6) : (!mo.tensor<[3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[D15, 1], f32>
  %8 = mo.shape_of(%arg0) : (!mo.tensor<[2, 2, 3], f32>) -> !mo.tensor<[3], si64>
  %9 = mo.rebind<() -> D14>(%7) : !mo.tensor<[D15, 1], f32> -> !mo.tensor<[D14, 1], f32>
  %10 = mo.shape_of(%9) : (!mo.tensor<[D14, 1], f32>) -> !mo.tensor<[2], si64>
  %11 = mo.slice(%8, %1, %4, %4) : (!mo.tensor<[3], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[1], si64>
  %12 = mo.slice(%8, %4, %0, %4) : (!mo.tensor<[3], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[2], si64>
  %13 = mo.slice(%10, %1, %1, %4) : (!mo.tensor<[2], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[0], si64>
  %14 = mo.broadcast_shape(%11, %13) : (!mo.tensor<[1], si64>, !mo.tensor<[0], si64>) -> !mo.tensor<[1], si64>
  %15 = mo.concat[0] (%14, %12) : (!mo.tensor<[1], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[3], si64>
  %16 = mo.concat[0] (%14, %10) : (!mo.tensor<[1], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[3], si64>
  %17 = mo.broadcast_to<() -> D11, D12, D13>(%arg0, %15) : (!mo.tensor<[2, 2, 3], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[D11, D12, D13], f32>
  %18 = mo.rebind<() -> D7>(%7) : !mo.tensor<[D15, 1], f32> -> !mo.tensor<[D7, 1], f32>
  %19 = mo.broadcast_to<() -> D8, D9, D10>(%18, %16) : (!mo.tensor<[D7, 1], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[D8, D9, D10], f32>
  %20 = mo.rebind<() -> D1, D2, D3>(%17) : !mo.tensor<[D11, D12, D13], f32> -> !mo.tensor<[D1, D2, D3], f32>
  %21 = mo.rebind<() -> D4, D5, D6>(%19) : !mo.tensor<[D8, D9, D10], f32> -> !mo.tensor<[D4, D5, D6], f32>
  %22 = mo.batch_matmul(%20, %21) : (!mo.tensor<[D1, D2, D3], f32>, !mo.tensor<[D4, D5, D6], f32>) -> !mo.tensor<[2, 2, 1], f32>
  %23 = mo.static.reshape(%22) : !mo.tensor<[2, 2, 1], f32> -> !mo.tensor<[2, 2], f32>
  mo.output %23 : !mo.tensor<[2, 2], f32>
}
