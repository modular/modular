// RUN: mo-opt %s --mo-symbolic-optimize | FileCheck %s

// CHECK:         @parallel_matmuls_static(%[[ACT:.*]]:{{.*}})
// CHECK:         %[[CONCAT:.*]] = mo.concat{{.*}} : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 8], f32>
// CHECK-NEXT:    %[[MATMUL:.*]] = mo.matmul(%[[ACT]], %[[CONCAT]]) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 8], f32>) -> !mo.tensor<[2, 8], f32>
// CHECK-NEXT:    %[[SLICE0:.*]] = mo.slice(%[[MATMUL]], {{.*}}) : (!mo.tensor<[2, 8], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
// CHECK-NEXT:    %[[SLICE1:.*]] = mo.slice(%[[MATMUL]], {{.*}}) : (!mo.tensor<[2, 8], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
// CHECK-NEXT:    %[[SLICE2:.*]] = mo.slice(%[[MATMUL]], {{.*}}) : (!mo.tensor<[2, 8], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
// CHECK-NEXT:    %[[SLICE3:.*]] = mo.slice(%[[MATMUL]], {{.*}}) : (!mo.tensor<[2, 8], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
// CHECK-NEXT:    %[[ADD0:.*]] = mo.add({{.*}}) : !mo.tensor<[2, 2], f32>
// CHECK-NEXT:    %[[ADD1:.*]] = mo.add({{.*}}) : !mo.tensor<[2, 2], f32>
// CHECK-NEXT:    %[[ADD2:.*]] = mo.add({{.*}}) : !mo.tensor<[2, 2], f32>
// CHECK-NEXT:    mo.output %[[ADD2]]

mo.graph @parallel_matmuls_static(%A: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %W0 = mo.constant <#M.dense_array<0.0, 0.0, 0.0, 0.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32>
  %W1 = mo.constant <#M.dense_array<1.0, 1.0, 1.0, 1.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32>
  %W2 = mo.constant <#M.dense_array<2.0, 2.0, 2.0, 2.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32>
  %W3 = mo.constant <#M.dense_array<3.0, 3.0, 3.0, 3.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32>
  %1 = mo.matmul(%A, %W0) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %2 = mo.matmul(%A, %W1) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %3 = mo.matmul(%A, %W2) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %4 = mo.matmul(%A, %W3) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %5 = mo.add(%1, %2) : !mo.tensor<[2, 2], f32>
  %6 = mo.add(%3, %4) : !mo.tensor<[2, 2], f32>
  %7 = mo.add(%5, %6) : !mo.tensor<[2, 2], f32>
  mo.output %7 : !mo.tensor<[2, 2], f32>
}

// Should fuse the same as parallel_matmuls_static
// CHECK:         @parallel_matmuls_static_external_constants(%[[ACT:.*]]:{{.*}})
// CHECK:         %[[CONCAT:.*]] = mo.concat{{.*}} : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 8], f32>
// CHECK-NEXT:    %[[MATMUL:.*]] = mo.matmul(%[[ACT]], %[[CONCAT]]) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 8], f32>) -> !mo.tensor<[2, 8], f32>
// CHECK-NEXT:    %[[SLICE0:.*]] = mo.slice(%[[MATMUL]], {{.*}}) : (!mo.tensor<[2, 8], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
// CHECK-NEXT:    %[[SLICE1:.*]] = mo.slice(%[[MATMUL]], {{.*}}) : (!mo.tensor<[2, 8], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
// CHECK-NEXT:    %[[SLICE2:.*]] = mo.slice(%[[MATMUL]], {{.*}}) : (!mo.tensor<[2, 8], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
// CHECK-NEXT:    %[[SLICE3:.*]] = mo.slice(%[[MATMUL]], {{.*}}) : (!mo.tensor<[2, 8], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
// CHECK-NEXT:    %[[ADD0:.*]] = mo.add({{.*}}) : !mo.tensor<[2, 2], f32>
// CHECK-NEXT:    %[[ADD1:.*]] = mo.add({{.*}}) : !mo.tensor<[2, 2], f32>
// CHECK-NEXT:    %[[ADD2:.*]] = mo.add({{.*}}) : !mo.tensor<[2, 2], f32>
// CHECK-NEXT:    mo.output %[[ADD2]]
mo.graph @parallel_matmuls_static_external_constants(%A: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %W0 = mo.constant.external [name = "blob0", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[2, 2], f32>
  %W1 = mo.constant.external [name = "blob1", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[2, 2], f32>
  %W2 = mo.constant.external [name = "blob2", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[2, 2], f32>
  %W3 = mo.constant.external [name = "blob3", align = 1, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[2, 2], f32>
  %1 = mo.matmul(%A, %W0) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %2 = mo.matmul(%A, %W1) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %3 = mo.matmul(%A, %W2) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %4 = mo.matmul(%A, %W3) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %5 = mo.add(%1, %2) : !mo.tensor<[2, 2], f32>
  %6 = mo.add(%3, %4) : !mo.tensor<[2, 2], f32>
  %7 = mo.add(%5, %6) : !mo.tensor<[2, 2], f32>
  mo.output %7 : !mo.tensor<[2, 2], f32>
}

// Do NOT optimize aliased external constant.
// CHECK:         @parallel_matmuls_static_external_constants_aliased(%[[ACT:.*]]:{{.*}})
// CHECK-NOT:     mo.concat
mo.graph @parallel_matmuls_static_external_constants_aliased(%A: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %W0 = mo.constant.external [name = "blob0", align = 1, device = #M.device_ref<"cpu", 0>, hasAlias = true] : !mo.tensor<[2, 2], f32>
  %W1 = mo.constant.external [name = "blob1", align = 1, device = #M.device_ref<"cpu", 0>, hasAlias = true] : !mo.tensor<[2, 2], f32>
  %W2 = mo.constant.external [name = "blob2", align = 1, device = #M.device_ref<"cpu", 0>, hasAlias = true] : !mo.tensor<[2, 2], f32>
  %W3 = mo.constant.external [name = "blob3", align = 1, device = #M.device_ref<"cpu", 0>, hasAlias = true] : !mo.tensor<[2, 2], f32>
  %1 = mo.matmul(%A, %W0) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %2 = mo.matmul(%A, %W1) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %3 = mo.matmul(%A, %W2) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %4 = mo.matmul(%A, %W3) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %5 = mo.add(%1, %2) : !mo.tensor<[2, 2], f32>
  %6 = mo.add(%3, %4) : !mo.tensor<[2, 2], f32>
  %7 = mo.add(%5, %6) : !mo.tensor<[2, 2], f32>
  mo.output %7 : !mo.tensor<[2, 2], f32>
}
// CHECK:         @parallel_batch_matmuls_static(%[[ACT:.*]]: {{.*}})
// CHECK:    %[[CONCAT:.*]] = mo.concat[2]{{.*}} : (!mo.tensor<[2, 2, 2], f32>, !mo.tensor<[2, 2, 2], f32>, !mo.tensor<[2, 2, 2], f32>, !mo.tensor<[2, 2, 2], f32>) -> !mo.tensor<[2, 2, 8], f32>
// CHECK-NEXT:    %[[MATMUL:.*]] = mo.batch_matmul(%[[ACT]], %[[CONCAT]]) : (!mo.tensor<[2, 2, 2], f32>, !mo.tensor<[2, 2, 8], f32>) -> !mo.tensor<[2, 2, 8], f32>
// CHECK-NEXT:    %[[SLICE0:.*]] = mo.slice(%[[MATMUL]], {{.*}}) : (!mo.tensor<[2, 2, 8], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[2, 2, 2], f32>
// CHECK-NEXT:    %[[SLICE1:.*]] = mo.slice(%[[MATMUL]], {{.*}}) : (!mo.tensor<[2, 2, 8], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[2, 2, 2], f32>
// CHECK-NEXT:    %[[SLICE2:.*]] = mo.slice(%[[MATMUL]], {{.*}}) : (!mo.tensor<[2, 2, 8], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[2, 2, 2], f32>
// CHECK-NEXT:    %[[SLICE3:.*]] = mo.slice(%[[MATMUL]], {{.*}}) : (!mo.tensor<[2, 2, 8], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[2, 2, 2], f32>
// CHECK-NEXT:    %[[ADD0:.*]] = mo.add({{.*}}) : !mo.tensor<[2, 2, 2], f32>
// CHECK-NEXT:    %[[ADD1:.*]] = mo.add({{.*}}) : !mo.tensor<[2, 2, 2], f32>
// CHECK-NEXT:    %[[ADD2:.*]] = mo.add({{.*}}) : !mo.tensor<[2, 2, 2], f32>
// CHECK-NEXT:    mo.output %[[ADD2]]
mo.graph @parallel_batch_matmuls_static(%A: !mo.tensor<[2, 2, 2], f32>) -> !mo.tensor<[2, 2, 2], f32> {
  %W0 = mo.constant <#M.dense_array<0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0> : tensor<2x2x2xf32>> : !mo.tensor<[2, 2, 2], f32>
  %W1 = mo.constant <#M.dense_array<1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0> : tensor<2x2x2xf32>> : !mo.tensor<[2, 2, 2], f32>
  %W2 = mo.constant <#M.dense_array<2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0> : tensor<2x2x2xf32>> : !mo.tensor<[2, 2, 2], f32>
  %W3 = mo.constant <#M.dense_array<3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0> : tensor<2x2x2xf32>> : !mo.tensor<[2, 2, 2], f32>
  %1 = mo.batch_matmul(%A, %W0) : (!mo.tensor<[2, 2, 2], f32>, !mo.tensor<[2, 2, 2], f32>) -> !mo.tensor<[2, 2, 2], f32>
  %2 = mo.batch_matmul(%A, %W1) : (!mo.tensor<[2, 2, 2], f32>, !mo.tensor<[2, 2, 2], f32>) -> !mo.tensor<[2, 2, 2], f32>
  %3 = mo.batch_matmul(%A, %W2) : (!mo.tensor<[2, 2, 2], f32>, !mo.tensor<[2, 2, 2], f32>) -> !mo.tensor<[2, 2, 2], f32>
  %4 = mo.batch_matmul(%A, %W3) : (!mo.tensor<[2, 2, 2], f32>, !mo.tensor<[2, 2, 2], f32>) -> !mo.tensor<[2, 2, 2], f32>
  %5 = mo.add(%1, %2) : !mo.tensor<[2, 2, 2], f32>
  %6 = mo.add(%3, %4) : !mo.tensor<[2, 2, 2], f32>
  %7 = mo.add(%5, %6) : !mo.tensor<[2, 2, 2], f32>
  mo.output %7 : !mo.tensor<[2, 2, 2], f32>
}


// CHECK:         @parallel_matmuls_insertion_point(%[[ACT:.*]]:{{.*}})
// CHECK:         %[[CONCAT:.*]] = mo.concat{{.*}} : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 8], f32>
// CHECK-NEXT:    %[[MATMUL:.*]] = mo.matmul(%[[ACT]], %[[CONCAT]]) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 8], f32>) -> !mo.tensor<[2, 8], f32>
// CHECK-NEXT:    %[[SLICE0:.*]] = mo.slice(%[[MATMUL]], {{.*}}) : (!mo.tensor<[2, 8], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
// CHECK-NEXT:    %[[SLICE1:.*]] = mo.slice(%[[MATMUL]], {{.*}}) : (!mo.tensor<[2, 8], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
// CHECK-NEXT:    %[[SLICE2:.*]] = mo.slice(%[[MATMUL]], {{.*}}) : (!mo.tensor<[2, 8], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
// CHECK-NEXT:    %[[SLICE3:.*]] = mo.slice(%[[MATMUL]], {{.*}}) : (!mo.tensor<[2, 8], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
// CHECK-NEXT:    %[[ADD0:.*]] = mo.add({{.*}}) : !mo.tensor<[2, 2], f32>
// CHECK-NEXT:    %[[ADD1:.*]] = mo.add({{.*}}) : !mo.tensor<[2, 2], f32>
// CHECK-NEXT:    %[[ADD2:.*]] = mo.add({{.*}}) : !mo.tensor<[2, 2], f32>
// CHECK-NEXT:    mo.output %[[ADD2]]
mo.graph @parallel_matmuls_insertion_point(%A: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %W0 = mo.constant <#M.dense_array<0.0, 0.0, 0.0, 0.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32>
  %W1 = mo.constant <#M.dense_array<1.0, 1.0, 1.0, 1.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32>
  %W2 = mo.constant <#M.dense_array<2.0, 2.0, 2.0, 2.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32>
  %W3 = mo.constant <#M.dense_array<3.0, 3.0, 3.0, 3.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32>
  %1 = mo.matmul(%A, %W0) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %2 = mo.matmul(%A, %W1) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %3 = mo.add(%1, %2) : !mo.tensor<[2, 2], f32>
  %4 = mo.matmul(%A, %W2) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %5 = mo.matmul(%A, %W3) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %6 = mo.add(%4, %5) : !mo.tensor<[2, 2], f32>
  %7 = mo.add(%3, %6) : !mo.tensor<[2, 2], f32>
  mo.output %7 : !mo.tensor<[2, 2], f32>
}

// CHECK:         @parallel_matmuls_static_with_device(%[[ACT:[a-zA-Z0-9_]*]]:{{.*}})
// CHECK:         %[[CONCAT:.*]] = mo.concat{{.*}} : (!mo.tensor<[2, 2], f32, dev:0>, !mo.tensor<[2, 2], f32, dev:0>, !mo.tensor<[2, 2], f32, dev:0>, !mo.tensor<[2, 2], f32, dev:0>) -> !mo.tensor<[2, 8], f32, dev:0>
// CHECK-NEXT:    %[[MATMUL:.*]] = mo.matmul(%[[ACT]], %[[CONCAT]]) : (!mo.tensor<[2, 2], f32, dev:0>, !mo.tensor<[2, 8], f32, dev:0>) -> !mo.tensor<[2, 8], f32, dev:0>
// CHECK-NEXT:    %[[SLICE0:.*]] = mo.slice(%[[MATMUL]], {{.*}}) : (!mo.tensor<[2, 8], f32, dev:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32, dev:0>
// CHECK-NEXT:    %[[SLICE1:.*]] = mo.slice(%[[MATMUL]], {{.*}}) : (!mo.tensor<[2, 8], f32, dev:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32, dev:0>
// CHECK-NEXT:    %[[SLICE2:.*]] = mo.slice(%[[MATMUL]], {{.*}}) : (!mo.tensor<[2, 8], f32, dev:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32, dev:0>
// CHECK-NEXT:    %[[SLICE3:.*]] = mo.slice(%[[MATMUL]], {{.*}}) : (!mo.tensor<[2, 8], f32, dev:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32, dev:0>
// CHECK-NEXT:    %[[ADD0:.*]] = mo.add({{.*}}) : !mo.tensor<[2, 2], f32, dev:0>
// CHECK-NEXT:    %[[ADD1:.*]] = mo.add({{.*}}) : !mo.tensor<[2, 2], f32, dev:0>
// CHECK-NEXT:    %[[ADD2:.*]] = mo.add({{.*}}) : !mo.tensor<[2, 2], f32, dev:0>
// CHECK-NEXT:    mo.output %[[ADD2]]

mo.graph @parallel_matmuls_static_with_device(%A: !mo.tensor<[2, 2], f32, dev:0>) -> !mo.tensor<[2, 2], f32, dev:0> {
  %W0 = mo.constant <#M.dense_array<0.0, 0.0, 0.0, 0.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32, dev:0>
  %W1 = mo.constant <#M.dense_array<1.0, 1.0, 1.0, 1.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32, dev:0>
  %W2 = mo.constant <#M.dense_array<2.0, 2.0, 2.0, 2.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32, dev:0>
  %W3 = mo.constant <#M.dense_array<3.0, 3.0, 3.0, 3.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32, dev:0>
  %1 = mo.matmul(%A, %W0) : (!mo.tensor<[2, 2], f32, dev:0>, !mo.tensor<[2, 2], f32, dev:0>) -> !mo.tensor<[2, 2], f32, dev:0>
  %2 = mo.matmul(%A, %W1) : (!mo.tensor<[2, 2], f32, dev:0>, !mo.tensor<[2, 2], f32, dev:0>) -> !mo.tensor<[2, 2], f32, dev:0>
  %3 = mo.matmul(%A, %W2) : (!mo.tensor<[2, 2], f32, dev:0>, !mo.tensor<[2, 2], f32, dev:0>) -> !mo.tensor<[2, 2], f32, dev:0>
  %4 = mo.matmul(%A, %W3) : (!mo.tensor<[2, 2], f32, dev:0>, !mo.tensor<[2, 2], f32, dev:0>) -> !mo.tensor<[2, 2], f32, dev:0>
  %5 = mo.add(%1, %2) : !mo.tensor<[2, 2], f32, dev:0>
  %6 = mo.add(%3, %4) : !mo.tensor<[2, 2], f32, dev:0>
  %7 = mo.add(%5, %6) : !mo.tensor<[2, 2], f32, dev:0>
  mo.output %7 : !mo.tensor<[2, 2], f32, dev:0>
}


// CHECK:         @parallel_matmuls_dynamic<[[D0:.*]]>(%[[ACT:.*]]:{{.*}})
// CHECK:         %[[CONCAT:.*]] = mo.concat{{.*}} : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 8], f32>
// CHECK-NEXT:    %[[MATMUL:.*]] = mo.matmul(%[[ACT]], %[[CONCAT]]) : (!mo.tensor<[D0, 2], f32>, !mo.tensor<[2, 8], f32>) -> !mo.tensor<[D0, 8], f32>
// CHECK-NEXT:    %[[SLICE0:.*]] = mo.slice(%[[MATMUL]], {{.*}}) : (!mo.tensor<[D0, 8], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[D0, 2], f32>
// CHECK-NEXT:    %[[SLICE1:.*]] = mo.slice(%[[MATMUL]], {{.*}}) : (!mo.tensor<[D0, 8], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[D0, 2], f32>
// CHECK-NEXT:    %[[SLICE2:.*]] = mo.slice(%[[MATMUL]], {{.*}}) : (!mo.tensor<[D0, 8], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[D0, 2], f32>
// CHECK-NEXT:    %[[SLICE3:.*]] = mo.slice(%[[MATMUL]], {{.*}}) : (!mo.tensor<[D0, 8], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[D0, 2], f32>
// CHECK-NEXT:    %[[ADD0:.*]] = mo.add({{.*}}) : !mo.tensor<[D0, 2], f32>
// CHECK-NEXT:    %[[ADD1:.*]] = mo.add({{.*}}) : !mo.tensor<[D0, 2], f32>
// CHECK-NEXT:    %[[ADD2:.*]] = mo.add({{.*}}) : !mo.tensor<[D0, 2], f32>
// CHECK-NEXT:    mo.output %[[ADD2]]

mo.graph @parallel_matmuls_dynamic<D0>(%A: !mo.tensor<[D0, 2], f32>) -> !mo.tensor<[D0, 2], f32> {
  %W0 = mo.constant <#M.dense_array<0.0, 0.0, 0.0, 0.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32>
  %W1 = mo.constant <#M.dense_array<1.0, 1.0, 1.0, 1.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32>
  %W2 = mo.constant <#M.dense_array<2.0, 2.0, 2.0, 2.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32>
  %W3 = mo.constant <#M.dense_array<3.0, 3.0, 3.0, 3.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32>
  %1 = mo.matmul(%A, %W0) : (!mo.tensor<[D0, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[D0, 2], f32>
  %2 = mo.matmul(%A, %W1) : (!mo.tensor<[D0, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[D0, 2], f32>
  %3 = mo.matmul(%A, %W2) : (!mo.tensor<[D0, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[D0, 2], f32>
  %4 = mo.matmul(%A, %W3) : (!mo.tensor<[D0, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[D0, 2], f32>
  %5 = mo.add(%1, %2) : !mo.tensor<[D0, 2], f32>
  %6 = mo.add(%3, %4) : !mo.tensor<[D0, 2], f32>
  %7 = mo.add(%5, %6) : !mo.tensor<[D0, 2], f32>
  mo.output %7 : !mo.tensor<[D0, 2], f32>
}

// CHECK:     @parallel_matmuls_but_dynamic_weights
// CHECK-NOT: mo.concat
// CHECK-NOT: mo.slice
// CHECK:     mo.output
mo.graph @parallel_matmuls_but_dynamic_weights(%A: !mo.tensor<[2, 2], f32>, %W0 : !mo.tensor<[2, 2], f32>, %W1 : !mo.tensor<[2, 2], f32>, %W2 : !mo.tensor<[2, 2], f32>, %W3 : !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %1 = mo.matmul(%A, %W0) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %2 = mo.matmul(%A, %W1) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %3 = mo.matmul(%A, %W2) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %4 = mo.matmul(%A, %W3) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %5 = mo.add(%1, %2) : !mo.tensor<[2, 2], f32>
  %6 = mo.add(%3, %4) : !mo.tensor<[2, 2], f32>
  %7 = mo.add(%5, %6) : !mo.tensor<[2, 2], f32>
  mo.output %7 : !mo.tensor<[2, 2], f32>
}

// CHECK: @parallel_matmuls_constant_indirection
// CHECK:         %[[CONCAT:.*]] = mo.concat
// CHECK-NEXT:    %[[MATMUL:.*]] = mo.matmul(%[[ACT]], %[[CONCAT]]) : (!mo.tensor<[4, 768], f32>, !mo.tensor<[768, 3072], f32>) -> !mo.tensor<[4, 3072], f32>
// CHECK-NEXT:    %[[SLICE0:.*]] = mo.slice(%[[MATMUL]], {{.*}}) : (!mo.tensor<[4, 3072], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 768], f32>
// CHECK-NEXT:    %[[SLICE1:.*]] = mo.slice(%[[MATMUL]], {{.*}}) : (!mo.tensor<[4, 3072], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 768], f32>
// CHECK-NEXT:    %[[SLICE2:.*]] = mo.slice(%[[MATMUL]], {{.*}}) : (!mo.tensor<[4, 3072], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 768], f32>
// CHECK-NEXT:    %[[SLICE3:.*]] = mo.slice(%[[MATMUL]], {{.*}}) : (!mo.tensor<[4, 3072], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 768], f32>
// CHECK-NEXT:    %[[ADD0:.*]] = mo.add({{.*}}) : !mo.tensor<[4, 768], f32>
// CHECK-NEXT:    %[[ADD1:.*]] = mo.add({{.*}}) : !mo.tensor<[4, 768], f32>
// CHECK-NEXT:    %[[ADD2:.*]] = mo.add({{.*}}) : !mo.tensor<[4, 768], f32>
// CHECK-NEXT:    mo.output %[[ADD2]]
mo.graph @parallel_matmuls_constant_indirection(%A: !mo.tensor<[4, 768], f32>) -> !mo.tensor<[4, 768], f32> {
  %W0 = mo.constant <#M.dense_array<0.0> : tensor<f32>> : !mo.tensor<[], f32>
  %W1 = mo.constant <#M.dense_array<1.0> : tensor<f32>> : !mo.tensor<[], f32>
  %W2 = mo.constant <#M.dense_array<2.0> : tensor<f32>> : !mo.tensor<[], f32>
  %W3 = mo.constant <#M.dense_array<3.0> : tensor<f32>> : !mo.tensor<[], f32>
  %shape = mo.constant <#M.dense_array<768, 768> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %W0_bcast = mo.broadcast_to(%W0, %shape) : (!mo.tensor<[], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[768, 768], f32>
  %W1_bcast = mo.broadcast_to(%W1, %shape) : (!mo.tensor<[], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[768, 768], f32>
  %W2_bcast = mo.broadcast_to(%W2, %shape) : (!mo.tensor<[], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[768, 768], f32>
  %W3_bcast = mo.broadcast_to(%W3, %shape) : (!mo.tensor<[], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[768, 768], f32>
  %1 = mo.matmul(%A, %W0_bcast) : (!mo.tensor<[4, 768], f32>, !mo.tensor<[768, 768], f32>) -> !mo.tensor<[4, 768], f32>
  %2 = mo.matmul(%A, %W1_bcast) : (!mo.tensor<[4, 768], f32>, !mo.tensor<[768, 768], f32>) -> !mo.tensor<[4, 768], f32>
  %3 = mo.matmul(%A, %W2_bcast) : (!mo.tensor<[4, 768], f32>, !mo.tensor<[768, 768], f32>) -> !mo.tensor<[4, 768], f32>
  %4 = mo.matmul(%A, %W3_bcast) : (!mo.tensor<[4, 768], f32>, !mo.tensor<[768, 768], f32>) -> !mo.tensor<[4, 768], f32>
  %5 = mo.add(%1, %2) : !mo.tensor<[4, 768], f32>
  %6 = mo.add(%3, %4) : !mo.tensor<[4, 768], f32>
  %7 = mo.add(%5, %6) : !mo.tensor<[4, 768], f32>
  mo.output %7 : !mo.tensor<[4, 768], f32>
}
