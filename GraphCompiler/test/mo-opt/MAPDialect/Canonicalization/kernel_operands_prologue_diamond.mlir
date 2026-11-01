// RUN: mo-opt --mlir-print-local-scope %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels --canonicalize --cse --map-outline-structured-kernel-bodies --canonicalize | FileCheck %s

// CHECK-LABEL: mo.graph @diamond_gather
// CHECK-SAME:  (%[[X:[^:]*]]: !mo.tensor<[3, 3], f32>, %[[Y:[^:]*]]: !mo.tensor<[3, 3], f32>, %[[IDXS:[^:]*]]: !mo.tensor<[2, 2], si32>)
// CHECK-NEXT:  %[[RES:.*]] = map.iter.opaque
// CHECK-NEXT:  inputs (%[[IDXS]]: !mo.tensor<[2, 2], si32>, %[[X]]: !mo.tensor<[3, 3], f32>, %[[Y]]: !mo.tensor<[3, 3], f32>)
// CHECK-NEXT:  outputs (%[[O:[^ ]*]] : !mo.tensor<[2, 2, 3], f32>) {

// The data prologue: 6 captures down to 2. Capture dedup fires at the
// mid-pipeline canonicalize, before fusion ever reaches outlining, so the
// following CSE already collapsed the redundant loads and adds into a single
// diamond over one add.
// CHECK-NEXT:    %[[DATA:.*]] = map.iter.prologue (%[[I:[^ ]*]] : !map.index<2>) capturing {%[[CX:[^ ]*]] = %[[X]] : !mo.tensor<[3, 3], f32>, %[[CY:[^ ]*]] = %[[Y]] : !mo.tensor<[3, 3], f32>} {
// CHECK-NEXT:      %[[XV:.*]] = map.load %[[CX]][%[[I]]]
// CHECK-NEXT:      %[[YV:.*]] = map.load %[[CY]][%[[I]]]
// CHECK-NEXT:      %[[ADD:.*]] = map.call.elementwise @{{.*}}.Add(%[[I]], %[[XV]], %[[YV]])
// CHECK-NEXT:      %[[MUL:.*]] = map.call.elementwise @{{.*}}.Mul(%[[I]], %[[ADD]], %[[ADD]])
// CHECK-NEXT:      %[[RELU:.*]] = map.call.elementwise @{{.*}}.ReLU(%[[I]], %[[ADD]])
// CHECK-NEXT:      %[[SUM:.*]] = map.call.elementwise @{{.*}}.Add(%[[I]], %[[MUL]], %[[RELU]])
// CHECK-NEXT:      map.yield %[[SUM]] : f32
// CHECK-NEXT:    } : !mo.tensor<[3, 3], f32>

// The single-capture indices prologue and epilogue are untouched.
// CHECK-NEXT:    %[[IPRO:.*]] = map.iter.prologue (%[[I2:[^ ]*]] : !map.index<2>) capturing {%[[CI:[^ ]*]] = %[[IDXS]] : !mo.tensor<[2, 2], si32>} {
// CHECK-NEXT:      %[[IV:.*]] = map.load %[[CI]][%[[I2]]] : !mo.tensor<[2, 2], si32>
// CHECK-NEXT:      map.yield %[[IV]] : si32
// CHECK-NEXT:    } : !mo.tensor<[2, 2], si32>
// CHECK-NEXT:    %[[EPI:.*]] = map.iter.epilogue (%[[V:[^ ]*]] : f32, %[[I3:[^ ]*]] : !map.index<3>) capturing {%[[CO:[^ ]*]] = %[[O]] : !mo.tensor<[2, 2, 3], f32>} {
// CHECK-NEXT:      map.store %[[CO]][%[[I3]]] = %[[V]] : !mo.tensor<[2, 2, 3], f32>
// CHECK-NEXT:    } : !mo.tensor<[2, 2, 3], f32>
// CHECK-NEXT:    map.call.opaque @{{.*}}.Gather[axis = 0 : index] inputs(%[[DATA]], %[[IPRO]]) outputs(%[[EPI]])
// CHECK-NEXT:  } {device = #M.device_ref<"cpu", 0>, fusionInfo = #mogg.fusion<#mogg.fusion_prologue<[#mogg.fusion_elementwise<[#mogg.fusion_leaf<"mo.add"{{.*}}, #mogg.fusion_leaf<"mo.relu"{{.*}}, #mogg.fusion_leaf<"mo.mul"{{.*}}, #mogg.fusion_leaf<"mo.add"{{.*}}]>, #mogg.fusion_leaf<"mo.gather"{{.*}}], operandIndices = [0]>>, hasReadEffect = false, hasWriteEffect = false}
// CHECK-NEXT:  mo.output %[[RES]] : !mo.tensor<[2, 2, 3], f32>
mo.graph @diamond_gather(
  %x: !mo.tensor<[3, 3], f32>,
  %y: !mo.tensor<[3, 3], f32>,
  %indices: !mo.tensor<[2, 2], si32>
) -> !mo.tensor<[2, 2, 3], f32> {
  %t0 = mo.add(%x, %y) : !mo.tensor<[3, 3], f32>
  %t1 = mo.mul(%t0, %t0) : !mo.tensor<[3, 3], f32>
  %t2 = mo.relu(%t0) : !mo.tensor<[3, 3], f32>
  %d = mo.add(%t1, %t2) : !mo.tensor<[3, 3], f32>
  %out = mo.gather(%d, %indices) [axis = 0] : (!mo.tensor<[3, 3], f32>, !mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 2, 3], f32>
  mo.output %out : !mo.tensor<[2, 2, 3], f32>
}
