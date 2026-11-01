// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t_new_api.mlir
// RUN: mt --execute --result-output-style=full %t_new_api.mlir | FileCheck %s
// RUN: cat %t_new_api.mlir | FileCheck %s --check-prefix=CHECK-IR

// CHECK-IR-LABEL: mef.callable @concat_fuse_ins_out$init

// CHECK: 'concat_fuse_ins_out' returned tensor<2x6xf32> [1.000000e+00, 4.000000e+00, 1.500000e+00, 1.500000e+00, -3.000000e+00, -6.000000e+00, 3.000000e+00, 8.000000e+00, 7.500000e+00, 3.500000e+00, -5.000000e+00, -9.000000e+00]

mo.graph @concat_fuse_ins_out() -> !mo.tensor<[2, 6], f32> {
    %axis = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[], si64>
    %x0 = mo.constant <#M.dense_array<1.5, 1.0, 2.5, 1.5> : tensor<4xf32>> : !mo.tensor<[2, 2], f32>
    %x1 = mo.constant <#M.dense_array<-0.5, 1.0, 0.5, 2.5> : tensor<4xf32>> : !mo.tensor<[2, 2], f32>
    %y = mo.constant <#M.dense_array<0.5, 1.5, 2.5, 3.5> : tensor<4xf32>> : !mo.tensor<[2, 2], f32>
    %z0 = mo.constant <#M.dense_array<-1.0, -3.0, 1.5, -0.5> : tensor<4xf32>> : !mo.tensor<[2, 2], f32>
    %z1 = mo.constant <#M.dense_array<-0.5, 1.0, -4.0, -2.5> : tensor<4xf32>> : !mo.tensor<[2, 2], f32>
    %s = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0> : tensor<12xf32>> : !mo.tensor<[2, 6], f32>

    %x = mo.add(%x0, %x1) : !mo.tensor<[2, 2], f32>
    %z = mo.add(%z0, %z1) : !mo.tensor<[2, 2], f32>
    %tmp = mo.concat[1](%x, %y, %z) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 6], f32>
    %out = mo.mul(%tmp, %s) : !mo.tensor<[2, 6], f32>
    // Expected: [1.000000e+00, 4.000000e+00, 1.500000e+00, 1.500000e+00, -3.000000e+00, -6.000000e+00, 3.000000e+00, 8.000000e+00, 7.500000e+00, 3.500000e+00, -5.000000e+00, -9.000000e+00]
    mo.output %out : !mo.tensor<[2, 6], f32>
}
