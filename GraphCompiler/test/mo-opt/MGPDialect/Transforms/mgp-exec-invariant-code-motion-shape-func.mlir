// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir --mlir-print-ir-after=mgp-exec-invariant-code-motion 2>&1 | FileCheck %s

// CHECK-LABEL: mgp.model @shape_func_consts

// Cast and save the output dimension in init
// CHECK-DAG: %[[DIMS:.*]]:2 = mogg.routine.async
// CHECK-DAG: %[[DIM_S:.*]] = pop.cast_to_builtin %[[DIMS]]#1 : !kgen.scalar<si64> to si64
// CHECK-DAG: %[[DIM_I:.*]] = index.casts %[[DIM_S]]
// CHECK-DAG: mgp.int.cache {{.*}} %[[DIM_I]]

// Load and cast the dimension in exec.
// CHECK: } execute (
// CHECK: %[[DIM2_I:.*]] = mgp.int.get_cached
// CHECK: %[[DIM2:.*]] = index.casts %[[DIM2_I]] : index to si64
// CHECK: %[[DIM3:.*]] = pop.cast_from_builtin %[[DIM2]] : si64 to !kgen.scalar<si64>


mo.graph @shape_func_consts() -> !mo.tensor<[D0], f32> {
  %0 = mo.constant <#M.dense_array<1.000000e+00> : tensor<f32>> : !mo.tensor<[], f32>
  %1 = mo.constant <#M.dense_array<0.000000e+00> : tensor<f32>> : !mo.tensor<[], f32>
  %2 = mo.constant <#M.dense_array<8.000000e+00> : !M.array<1xf32>> : !mo.tensor<[], f32>
  mo.invoke_shape_func["mo.range"]<() -> D0>(%1, %2, %0) [dataDeptTensors = [0 : i32, 1 : i32, 2 : i32]] : (!mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>)
  %3 = mo.range(%1, %2, %0) : (!mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[D0], f32>
  mo.output %3 : !mo.tensor<[D0], f32>
}
