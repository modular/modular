// RUN: mo-opt %s --mogg-to-mgp | FileCheck %s


// CHECK-LABEL: mgp.model @transfer_d2h : mof
//       CHECK: } execute ([[CH:%.*]]: !mef.chain, [[CTX:%.*]]: !mgp.context, {{.*}}
// CHECK: [[CPY_CH:%.*]] = mgp.buffer.device_to_host {{.*}}, [[DBUF:%.*]], [[HBUF:%.*]], {{.*}} : !mgp.buffer
// CHECK-NEXT: [[SYNC_CH:%.*]] = mgp.sync([[CPY_CH]]
// CHECK-NEXT: mgp.extend.lifetime [[SYNC_CH]], [[HBUF]]

// COM: Device to host needs to wait on the sync before using the host buffer.

// CHECK-NEXT: [[OUT_TENSOR:%.*]] = mgp.tensor.create [[SYNC_CH]]
// CHECK: mef.output [[CH]], [[OUT_TENSOR]]

mo.graph @transfer_d2h(
  %input: !mo.tensor<[5], si64, gpu:0>
) -> !mo.tensor<[5], si64> attributes {
  device_specs = #M.device_spec_collection<host = <"cpu", 0>, devices = [
    <ref = <"cpu", 0>, target = <triple = "", arch = "", features = "">>,
    <ref = <"gpu", 0>, target = <triple = "", arch = "", features = "">>
  ]>} {
  %ch0 = mo.chain.create()
  %output, %ch1 = mo.transfer[%ch0] %input : !mo.tensor<[5], si64, gpu:0> to <"cpu", 0>
  mo.output %output : !mo.tensor<[5], si64>
}
