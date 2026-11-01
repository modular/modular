// RUN: mo-opt --gc-pipeline %s | FileCheck %s

// CHECK-LABEL: mef.callable @all_cpu$exec
// CHECK: mo.matmul
// CHECK-LABEL: mef.model @all_cpu
// CHECK-SAME: inputDeviceIndices [0 : ui32, 0 : ui32] resultDeviceIndices [0 : ui32]
// CHECK: mef.output {{.*}} !mgp.tensor<2x2xf32>
mo.graph @all_cpu(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[3, 2], f32>) -> (!mo.tensor<[2, 2], f32>) attributes {
  device_specs = #M.device_spec_collection<host = <"cpu", 0>, devices = [
    <ref = <"cpu", 0>, target = <triple = "", arch = "", features = "">>,
    <ref = <"gpu", 0>, target = <triple = "", arch = "", features = "">>
  ]>} {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[3, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %0 : !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL: mef.callable @all_device$exec
// CHECK: mo.matmul
// CHECK-LABEL: mef.model @all_device
// CHECK-SAME: inputDeviceIndices [1 : ui32, 1 : ui32] resultDeviceIndices [1 : ui32]
// CHECK: mef.output {{.*}} !mgp.tensor<2x2xf32, <"gpu", 0>>
mo.graph @all_device(%arg0: !mo.tensor<[2, 3], f32, gpu:0>, %arg1: !mo.tensor<[3, 2], f32, gpu:0>) -> (!mo.tensor<[2, 2], f32, gpu:0>) attributes {
  device_specs = #M.device_spec_collection<host = <"cpu", 0>, devices = [
    <ref = <"cpu", 0>, target = <triple = "", arch = "", features = "">>,
    <ref = <"gpu", 0>, target = <triple = "", arch = "", features = "">>
  ]>} {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[3, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
  mo.output %0 : !mo.tensor<[2, 2], f32, gpu:0>
}

// CHECK-LABEL: mef.callable @output_device$exec
// CHECK: host_to_device
// CHECK: host_to_device
// CHECK: mo.matmul
// CHECK-LABEL: mef.model @output_device
// CHECK-SAME: inputDeviceIndices [0 : ui32, 0 : ui32] resultDeviceIndices [1 : ui32]
// CHECK: mef.output {{.*}} !mgp.tensor<2x2xf32, <"gpu", 0>>
mo.graph @output_device(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[3, 2], f32>) -> (!mo.tensor<[2, 2], f32, gpu:0>) attributes {
  device_specs = #M.device_spec_collection<host = <"cpu", 0>, devices = [
    <ref = <"cpu", 0>, target = <triple = "", arch = "", features = "">>,
    <ref = <"gpu", 0>, target = <triple = "", arch = "", features = "">>
  ]>} {
  %ch0 = mo.chain.create()
  %0, %ch1 = mo.transfer[%ch0] %arg0 : !mo.tensor<[2, 3], f32> to <"gpu", 0>
  %1, %ch2 = mo.transfer[%ch1] %arg1 : !mo.tensor<[3, 2], f32> to <"gpu", 0>
  %2 = mo.matmul(%0, %1) : (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[3, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
  mo.output %2 : !mo.tensor<[2, 2], f32, gpu:0>
}

// CHECK-LABEL: mef.callable @inputs_device$exec
// CHECK: mo.matmul
// CHECK: device_to_host
// CHECK-LABEL: mef.model @inputs_device
// CHECK-SAME: inputDeviceIndices [1 : ui32, 1 : ui32] resultDeviceIndices [0 : ui32]
// CHECK: mef.output {{.*}} !mgp.tensor<2x2xf32>
mo.graph @inputs_device(%arg0: !mo.tensor<[2, 3], f32, gpu:0>, %arg1: !mo.tensor<[3, 2], f32, gpu:0>) -> (!mo.tensor<[2, 2], f32>) attributes {
  device_specs = #M.device_spec_collection<host = <"cpu", 0>, devices = [
    <ref = <"cpu", 0>, target = <triple = "", arch = "", features = "">>,
    <ref = <"gpu", 0>, target = <triple = "", arch = "", features = "">>
  ]>} {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[3, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
  %ch0 = mo.chain.create()
  %1, %ch1 = mo.transfer[%ch0] %0 : !mo.tensor<[2, 2], f32, gpu:0> to <"cpu", 0>
  mo.output %1 : !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL: mef.callable @mixture_device$exec
// CHECK: host_to_device
// CHECK: mo.matmul
// CHECK-LABEL: mef.model @mixture_device
// CHECK-SAME: inputDeviceIndices [0 : ui32, 1 : ui32] resultDeviceIndices [1 : ui32]
// CHECK: mef.output {{.*}} !mgp.tensor<2x2xf32, <"gpu", 0>>
mo.graph @mixture_device(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[3, 2], f32, gpu:0>) -> (!mo.tensor<[2, 2], f32, gpu:0>) attributes {
  device_specs = #M.device_spec_collection<host = <"cpu", 0>, devices = [
    <ref = <"cpu", 0>, target = <triple = "", arch = "", features = "">>,
    <ref = <"gpu", 0>, target = <triple = "", arch = "", features = "">>
  ]>} {
  %ch0 = mo.chain.create()
  %0, %ch1 = mo.transfer[%ch0] %arg0 : !mo.tensor<[2, 3], f32> to <"gpu", 0>
  %1 = mo.matmul(%0, %arg1) : (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[3, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
  mo.output %1 : !mo.tensor<[2, 2], f32, gpu:0>
}
