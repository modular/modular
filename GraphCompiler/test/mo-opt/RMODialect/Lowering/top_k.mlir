// RUN: mo-opt --mo-legalize-rmo-operators="skip-rmo-rebinds" --mo-shape-inference %s | FileCheck %s --enable-var-scope

// CHECK-LABEL: @top_k
mo.graph @top_k<dim0, dim1>(%arg0: !mo.tensor<[dim0, dim1], si64, gpu:0>) -> (!mo.tensor<[dim0, 3], si64, gpu:0>, !mo.tensor<[dim0, 3], si64, gpu:0>) {
  // CHECK-DAG: [[K:%.+]] = mo.constant <{{.*}}3{{.*}}si64{{.*}}> : !mo.tensor<[], si64>
  // CHECK-DAG: [[AXIS:%.+]] = mo.constant <{{.*}}1{{.*}}si64{{.*}}> : !mo.tensor<[], si64>
  // CHECK-DAG: [[SORTED:%.+]] = mo.constant <{{.*}}1{{.*}}i8{{.*}}> : !mo.tensor<[], bool>
  // CHECK: mo.top_k(%arg0, [[K]], [[AXIS]], [[SORTED]])
  %values, %indices = rmo.top_k(%arg0) [k = 3, axis = 1] : (!mo.tensor<[dim0, dim1], si64, gpu:0>) -> (!mo.tensor<[dim0, 3], si64, gpu:0>, !mo.tensor<[dim0, 3], si64, gpu:0>)
  mo.output %values, %indices : !mo.tensor<[dim0, 3], si64, gpu:0>, !mo.tensor<[dim0, 3], si64, gpu:0>
}
