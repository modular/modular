// RUN: mo-opt %s --gc-pipeline -o %t.mlir
// RUN: mt --execute --result-output-style=full --model-inputs=ones --input-shapes=1x2x3 --input-shapes=1x2x3 --input-shapes=1x2x3 %t.mlir | FileCheck %s

// CHECK: 'fuse_through_rebinds' returned tensor<1x2x3xf32> [3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00]

// Despite rebinds (some of which are not topo. order), still fuse the adds
// CHECK-IR-LABEL: @fuse_through_rebinds
// CHECK-IR-COUNT-1: mogg.routine
// CHECK-IR-COUNT-2: mogg.call "mo.add"
mo.graph @fuse_through_rebinds<A, B, C>(
    %arg0: !mo.tensor<[A, 2, 3], f32>,
    %arg1: !mo.tensor<[B, 2, 3], f32>,
    %arg2: !mo.tensor<[C, 2, 3], f32>
) -> !mo.tensor<[C, 2, 3], f32> {
    %arg0_recast1 = mo.rebind(%arg0_recast0): !mo.tensor<[B, 2, 3], f32> -> !mo.tensor<[1, 2, 3], f32>
    %arg0_recast2 = mo.rebind(%arg0_recast1): !mo.tensor<[1, 2, 3], f32> -> !mo.tensor<[B, 2, 3], f32>
    %arg0_recast0 = mo.rebind(%arg0): !mo.tensor<[A, 2, 3], f32> -> !mo.tensor<[B, 2, 3], f32>
    %res1 = mo.add(%arg0_recast2, %arg1) : !mo.tensor<[B, 2, 3], f32>
    %res1_recast = mo.rebind(%res1): !mo.tensor<[B, 2, 3], f32> -> !mo.tensor<[C, 2, 3], f32>
    %res2 = mo.add(%res1_recast, %arg2) : !mo.tensor<[C, 2, 3], f32>
    mo.output %res2 : !mo.tensor<[C, 2, 3], f32>
}
