// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt --mlir-print-local-scope %s --mogg-import-packages --mo-to-map --map-fuse-mutable-loads | FileCheck %s

// GEX-4230 / GEX-3962: a mutable load feeding a *chainless* opaque kernel -- a
// gather that indexes a whole buffer -- must fuse. The pass used to skip an
// opaque kernel that had no chain to sequence the load against (only a
// chainless map.iter.elementwise was handled), leaving the mo.mutable.load behind.
// Every read of a large buffer (e.g. a KV-cache leaf gathered by row) then
// materialized a full-buffer copy before the gather. Fusing gives the kernel a
// chain in/out pair and lets it read the buffer directly -- an indexed read,
// matching the legacy MOGG fusion path.

// The whole-buffer load is gone; the gather takes the buffer as an input,
// gains a chain in/out pair, and its map.call.opaque reads the buffer directly.
// CHECK-LABEL: mo.graph @gather_chainless(
// CHECK-SAME:  %[[BUF:.*]]: !mo.buffer<[8, 4], f32>, %[[IDX:.*]]: !mo.tensor<[2, 1], si32>
// CHECK-NOT:   mo.mutable.load
// CHECK:       %[[CH0:.*]] = mo.chain.create()
// CHECK:       %[[RES:.*]]:2 = map.iter.opaque
// CHECK-NEXT:  inputs (%[[BUF]]: !mo.buffer<[8, 4], f32>, %[[IDX]]: !mo.tensor<[2, 1], si32>, %[[CH0]]: !mo.chain)
// CHECK-NEXT:  outputs ({{.*}}: !mo.tensor<[2, 4], f32>, %[[RES]]#1: !mo.chain)
// CHECK:       map.call.opaque @{{.*}}GatherND[{{.*}}] inputs(%[[BUF]], %[[IDX]])
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_mutable_load<[#mogg.fusion_leaf<"mo.mutable.load"{{.*}}, #mogg.fusion_leaf<"mo.gather_nd"{{.*}}]>>
// CHECK:       mo.output %[[RES]]#0
mo.graph @gather_chainless(%buf: !mo.buffer<[8, 4], f32>, %idx: !mo.tensor<[2, 1], si32>) -> !mo.tensor<[2, 4], f32> {
  %0 = mo.chain.create()
  %t, %oc = mo.mutable.load(%buf, %0) : (!mo.buffer<[8, 4], f32>, !mo.chain) -> (!mo.tensor<[8, 4], f32>, !mo.chain)
  %g = mo.gather_nd(%t, %idx) [batchDims = 0] : (!mo.tensor<[8, 4], f32>, !mo.tensor<[2, 1], si32>) -> !mo.tensor<[2, 4], f32>
  mo.output %g : !mo.tensor<[2, 4], f32>
}
