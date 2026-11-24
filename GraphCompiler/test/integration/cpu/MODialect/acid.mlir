// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

// This test is useful to tickle concurrency issues between the MEF runtime,
// AsyncRT and our kgenned kernels.
// Avoid using `-gc-pipeline` to avoid fusing the kernels.

// CHECK-LABEL: --- Running 'acid':
mo.graph @acid() -> (!mo.tensor<[5], f32>) {
  %0 = "mo.constant"() {value = #M.dense_array<-1.0, 2.0, -3.0, -4.0, 5.0> : tensor<5xf32>} : () -> !mo.tensor<[5], f32>
  %1 = "mo.constant"() {value = #M.dense_array<3.0, 1.0, -6.0, 2.0, 5.0> : tensor<5xf32>} : () -> !mo.tensor<[5], f32>
  %2 = mo.add(%0, %1) : !mo.tensor<[5], f32>
  %3 = mo.mul(%0, %2) : !mo.tensor<[5], f32>
  %4 = mo.sub(%0, %2) : !mo.tensor<[5], f32>
  %5 = mo.relu(%4) : !mo.tensor<[5], f32>
  %6 = mo.add(%0, %5) : !mo.tensor<[5], f32>
  %7 = mo.relu(%6) : !mo.tensor<[5], f32>
  %8 = mo.erf(%7) : !mo.tensor<[5], f32>
  %9 = mo.add(%3, %8) : !mo.tensor<[5], f32>
  %10 = mo.add(%0, %1) : !mo.tensor<[5], f32>
  %11 = mo.add(%0, %1) : !mo.tensor<[5], f32>
  %12 = mo.add(%0, %1) : !mo.tensor<[5], f32>
  %13 = mo.add(%0, %1) : !mo.tensor<[5], f32>
  %14 = mo.add(%0, %1) : !mo.tensor<[5], f32>
  %15 = mo.add(%0, %1) : !mo.tensor<[5], f32>
  %16 = mo.add(%0, %1) : !mo.tensor<[5], f32>
  %17 = mo.add(%0, %1) : !mo.tensor<[5], f32>
  %18 = mo.add(%0, %1) : !mo.tensor<[5], f32>
  %19 = mo.add(%0, %1) : !mo.tensor<[5], f32>
  %20 = mo.add(%9, %10) : !mo.tensor<[5], f32>
  %21 = mo.add(%20, %11) : !mo.tensor<[5], f32>
  %22 = mo.add(%21, %12) : !mo.tensor<[5], f32>
  %23 = mo.add(%22, %13) : !mo.tensor<[5], f32>
  %24 = mo.add(%23, %14) : !mo.tensor<[5], f32>
  %25 = mo.add(%24, %15) : !mo.tensor<[5], f32>
  %26 = mo.add(%25, %16) : !mo.tensor<[5], f32>
  %27 = mo.add(%26, %17) : !mo.tensor<[5], f32>
  %28 = mo.add(%27, %18) : !mo.tensor<[5], f32>
  %29 = mo.add(%28, %19) : !mo.tensor<[5], f32>
  mo.output %29 : !mo.tensor<[5], f32>
}
// CHECK: 'acid' returned tensor<5xf32> [1.800000e+01, 3.699532e+01, -6.200002e+01, -1.200000e+01, 1.510000e+02]
