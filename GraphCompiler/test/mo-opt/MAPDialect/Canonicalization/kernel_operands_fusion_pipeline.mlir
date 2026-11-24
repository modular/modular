// RUN: mo-opt --mlir-print-local-scope %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_pkg" --mo-to-map --map-fuse-kernels --map-outline-structured-kernel-bodies --canonicalize --cse | FileCheck %s

// Kernel fusion duplicates %x/%y once per fused producer;
// simplify-kernel-operands uniques them, which makes the redundant loads and
// adds identical so the trailing CSE collapses them.

// CHECK-LABEL: mo.graph @fusion_diamond
// CHECK-SAME:  (%[[X:.*]]: !mo.tensor<[8], f32>, %[[Y:.*]]: !mo.tensor<[8], f32>)
// CHECK-NEXT:  %[[OUT:.*]] = map.iter.opaque
// CHECK-NEXT:  inputs (%[[X]]: !mo.tensor<[8], f32>, %[[Y]]: !mo.tensor<[8], f32>)
// CHECK-NEXT:  outputs (%[[O:.*]] : !mo.tensor<[8], f32>) {
// CHECK-NEXT:    map.iter.elementwise.body<[8]>
// CHECK-NEXT:      index (%[[IDX:.*]]: !map.index<1>)
// CHECK-NEXT:      capturing {%[[X_IN:.*]] = %[[X]] : !mo.tensor<[8], f32>, %[[Y_IN:.*]] = %[[Y]] : !mo.tensor<[8], f32>, %[[O_IN:.*]] = %[[O]] : !mo.tensor<[8], f32>} {
// CHECK-NEXT:    %[[XV:.*]] = map.load %[[X_IN]][%[[IDX]]] : !mo.tensor<[8], f32>
// CHECK-NEXT:    %[[YV:.*]] = map.load %[[Y_IN]][%[[IDX]]] : !mo.tensor<[8], f32>
// CHECK-NEXT:    %[[ADD0:.*]] = map.call.elementwise @{{.*}}.Add(%[[IDX]], %[[XV]], %[[YV]])
// CHECK-NEXT:    %[[MUL:.*]] = map.call.elementwise @{{.*}}.Mul(%[[IDX]], %[[ADD0]], %[[ADD0]])
// CHECK-NEXT:    %[[RELU:.*]] = map.call.elementwise @{{.*}}.ReLU(%[[IDX]], %[[ADD0]])
// CHECK-NEXT:    %[[ADD1:.*]] = map.call.elementwise @{{.*}}.Add(%[[IDX]], %[[MUL]], %[[RELU]])
// CHECK-NEXT:    map.store %[[O_IN]][%[[IDX]]] = %[[ADD1]] : !mo.tensor<[8], f32>
// CHECK-NEXT:  }
// CHECK:       } {device = #M.device_ref<"cpu", 0>, fusionInfo = #mogg.fusion<#mogg.fusion_elementwise<[#mogg.fusion_leaf<"mo.add"{{.*}}, #mogg.fusion_leaf<"mo.relu"{{.*}}, #mogg.fusion_leaf<"mo.mul"{{.*}}, #mogg.fusion_leaf<"mo.add"{{.*}}]>>, hasReadEffect = false, hasWriteEffect = false}
// CHECK-NEXT:  mo.output %[[OUT]] : !mo.tensor<[8], f32>
mo.graph @fusion_diamond(
  %x: !mo.tensor<[8], f32>,
  %y: !mo.tensor<[8], f32>
) -> !mo.tensor<[8], f32> {
  %t0 = mo.add(%x, %y) : !mo.tensor<[8], f32>
  %t1 = mo.mul(%t0, %t0) : !mo.tensor<[8], f32>
  %t2 = mo.relu(%t0) : !mo.tensor<[8], f32>
  %out = mo.add(%t1, %t2) : !mo.tensor<[8], f32>
  mo.output %out : !mo.tensor<[8], f32>
}
