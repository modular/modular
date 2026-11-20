// RUN: mo-opt --mlir-print-local-scope %s --mogg-import-packages --mo-to-map --map-fuse-mutable-loads | FileCheck %s

// The first load fuses into the opaque kernel reading it: its buffer takes the
// tensor's input slot and the kernel's chain sequences the load.
//
// The second load does not. The kernel reading it takes a chain that merges
// that load's own out-chain, so fusing would make the kernel depend on a chain
// it now produces. Detaching the merge instead is not an option either: the
// mo.chain.create carrying it has a second user (the third load), which would
// observe the rewiring.
//
// The last two loads feed no kernel at all, so they are never candidates.
// CHECK-LABEL: mo.graph @cycle(
// CHECK-SAME:  %[[BUFA:.*]]: !mo.buffer<[7, 5], f32>, %[[BUFB:.*]]: !mo.buffer<[3, 2], si32>
// CHECK:       %[[CH0:.*]] = mo.chain.create()
// CHECK-NOT:   mo.mutable.load
// CHECK:       %[[K1:.*]] = map.iter.opaque
// CHECK-NEXT:  inputs (%[[BUFA]] as {{.*}}: !mo.buffer<[7, 5], f32>, %[[BUFB]] as %[[BUFBARG:.*]]: !mo.buffer<[3, 2], si32>,
// CHECK-SAME:  %[[CH0]]: !mo.chain)
// CHECK:       map.call.opaque @{{.*}}sampler_apply_penalties[] inputs({{.*}}%[[BUFBARG]],
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_mutable_load<[#mogg.fusion_leaf<"mo.mutable.load"{{.*}}, #mogg.fusion_leaf<"custom__sampler.apply_penalties"{{.*}}]>>

// CHECK:       %[[T2:.*]], %[[OC2:.*]] = mo.mutable.load(%[[BUFB]], %[[CH0]])
// CHECK:       %[[MERGE:.*]] = mo.chain.create(%[[CH0]], %[[K1]], %[[K1]], %[[OC2]])
// CHECK:       %[[K2:.*]] = map.iter.opaque
// CHECK-NEXT:  inputs (%[[BUFA]] as {{.*}}: !mo.buffer<[7, 5], f32>, %[[T2]] as {{.*}}: !mo.tensor<[3, 2], si32>,
// CHECK-SAME:  %[[MERGE]]: !mo.chain)
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"custom__sampler.apply_penalties"{{.*}}>>

// CHECK:       mo.mutable.load(%[[BUFB]], %[[MERGE]])
// CHECK:       mo.mutable.load(%[[BUFA]], %[[K2]])
mo.graph @cycle(%arg0: !mo.buffer<[7, 5], f32>, %arg1: !mo.buffer<[3, 2], si32>, %arg2: !mo.tensor<[11], ui32>, %arg3: !mo.tensor<[7], f32>, %arg4: !mo.tensor<[7], f32>) -> (!mo.tensor<[7, 5], f32>, !mo.chain, !mo.chain) {
  %2 = mo.chain.create()
  %outTensor, %outChain = mo.mutable.load(%arg1, %2) : (!mo.buffer<[3, 2], si32>, !mo.chain) -> (!mo.tensor<[3, 2], si32>, !mo.chain)
  %4 = mo.custom [symbol = "sampler.apply_penalties", device = #M.device_ref<"gpu", 0>](%arg0, %outTensor, %arg2, %arg4, %arg4, %arg4, %2) : (!mo.buffer<[7, 5], f32>, !mo.tensor<[3, 2], si32>, !mo.tensor<[11], ui32>, !mo.tensor<[7], f32>, !mo.tensor<[7], f32>, !mo.tensor<[7], f32>, !mo.chain) -> !mo.chain
  %outTensor_0, %outChain_1 = mo.mutable.load(%arg1, %2) : (!mo.buffer<[3, 2], si32>, !mo.chain) -> (!mo.tensor<[3, 2], si32>, !mo.chain)
  %5 = mo.chain.create(%2, %4, %outChain, %outChain_1) : !mo.chain, !mo.chain, !mo.chain, !mo.chain
  %8 = mo.custom [symbol = "sampler.apply_penalties", device = #M.device_ref<"gpu", 0>](%arg0, %outTensor_0, %arg2, %arg3, %arg4, %arg4, %5) : (!mo.buffer<[7, 5], f32>, !mo.tensor<[3, 2], si32>, !mo.tensor<[11], ui32>, !mo.tensor<[7], f32>, !mo.tensor<[7], f32>, !mo.tensor<[7], f32>, !mo.chain) -> !mo.chain
  // One other user to break fusion.
  %outTensor_2, %outChain_3 = mo.mutable.load(%arg1, %5) : (!mo.buffer<[3, 2], si32>, !mo.chain) -> (!mo.tensor<[3, 2], si32>, !mo.chain)
  %outTensor_3, %outChain_4 = mo.mutable.load(%arg0, %8) : (!mo.buffer<[7, 5], f32>, !mo.chain) -> (!mo.tensor<[7, 5], f32>, !mo.chain)
  mo.output %outTensor_3, %outChain_3, %outChain_4 : !mo.tensor<[7, 5], f32>, !mo.chain, !mo.chain
}
