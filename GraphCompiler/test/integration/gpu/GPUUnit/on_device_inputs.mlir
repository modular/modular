// RUN: mo-opt --gc-pipeline="default-device-label=gpu" %s | FileCheck %s

mo.graph @matmul<D>(%arg1: !mo.tensor<[D, D], f32, gpu:0>, %arg2: !mo.tensor<[D, D], f32, gpu:0>) -> !mo.tensor<[D, D], f32, gpu:0> {
  %1 = mo.matmul(%arg2, %arg1) : (!mo.tensor<[D, D], f32, gpu:0>, !mo.tensor<[D, D], f32, gpu:0>) -> !mo.tensor<[D, D], f32, gpu:0>
  mo.output %1: !mo.tensor<[D,D], f32, gpu:0>
}
// CHECK: mef.callable @matmul$exec({{.*}}: !mgp.tensor<?x?xf32, <"gpu", 0>>, {{.*}}: !mgp.tensor<?x?xf32, <"gpu", 0>>) -> ({{.*}}!mgp.tensor<?x?xf32, <"gpu", 0>>)
