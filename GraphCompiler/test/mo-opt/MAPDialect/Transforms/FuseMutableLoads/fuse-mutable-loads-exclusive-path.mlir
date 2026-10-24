// RUN: mo-opt --mlir-print-local-scope %s --mogg-import-packages --mo-to-map --map-fuse-mutable-loads | FileCheck %s

// Both loads fuse into the opaque kernels reading them.
//
// The second kernel's chain merges the second load's out-chain, which on its own
// would make the kernel depend on a chain it produces once fused. Here the
// mo.chain.create carrying the merge has exactly one user, so it can be rewired
// onto the load's *in*-chain instead -- severing the dependency without changing
// what anything else observes. The merge ends up carrying %[[CH0]] twice: once
// for the detached second load and once for the first load's out-chain, which
// became the first kernel's out-chain when that load fused.
// CHECK-LABEL: mo.graph @exclusive_path(
// CHECK-SAME:  %[[BUFA:.*]]: !mo.buffer<[7, 5], f32>, %[[BUFB:.*]]: !mo.buffer<[3, 2], si32>
// CHECK:       %[[CH0:.*]] = mo.chain.create()
// CHECK-NOT:   mo.mutable.load(%[[BUFB]]
// CHECK:       %[[K1:.*]] = map.iter.opaque
// CHECK-NEXT:  inputs (%[[BUFA]] as {{.*}}: !mo.buffer<[7, 5], f32>, %[[BUFB]] as %[[BUFBARG1:.*]]: !mo.buffer<[3, 2], si32>,
// CHECK-SAME:  %[[CH0]]: !mo.chain)
// CHECK:       map.call.opaque @{{.*}}sampler_apply_penalties[] inputs({{.*}}%[[BUFBARG1]],
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_mutable_load<[#mogg.fusion_leaf<"mo.mutable.load"{{.*}}, #mogg.fusion_leaf<"custom__sampler.apply_penalties"{{.*}}]>>

// CHECK:       %[[MERGE:.*]] = mo.chain.create(%[[CH0]], %[[K1]], %[[K1]], %[[CH0]])
// CHECK:       %[[K2CHAIN:.*]] = mo.chain.create(%[[MERGE]], %[[CH0]])
// CHECK:       %[[K2:.*]] = map.iter.opaque
// CHECK-NEXT:  inputs (%[[BUFA]] as {{.*}}: !mo.buffer<[7, 5], f32>, %[[BUFB]] as %[[BUFBARG2:.*]]: !mo.buffer<[3, 2], si32>,
// CHECK-SAME:  %[[K2CHAIN]]: !mo.chain)
// CHECK:       map.call.opaque @{{.*}}sampler_apply_penalties[] inputs({{.*}}%[[BUFBARG2]],
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_mutable_load<[#mogg.fusion_leaf<"mo.mutable.load"{{.*}}, #mogg.fusion_leaf<"custom__sampler.apply_penalties"{{.*}}]>>

// CHECK:       mo.mutable.load(%[[BUFA]], %[[K2]])
mo.graph @exclusive_path(%arg0: !mo.buffer<[7, 5], f32>, %arg1: !mo.buffer<[3, 2], si32>, %arg2: !mo.tensor<[11], ui32>, %arg3: !mo.tensor<[7], f32>, %arg4: !mo.tensor<[7], f32>) -> (!mo.tensor<[7, 5], f32>, !mo.chain) {
  %2 = mo.chain.create()
  %outTensor, %outChain = mo.mutable.load(%arg1, %2) : (!mo.buffer<[3, 2], si32>, !mo.chain) -> (!mo.tensor<[3, 2], si32>, !mo.chain)
  %4 = mo.custom [symbol = "sampler.apply_penalties", device = #M.device_ref<"gpu", 0>](%arg0, %outTensor, %arg2, %arg4, %arg4, %arg4, %2) : (!mo.buffer<[7, 5], f32>, !mo.tensor<[3, 2], si32>, !mo.tensor<[11], ui32>, !mo.tensor<[7], f32>, !mo.tensor<[7], f32>, !mo.tensor<[7], f32>, !mo.chain) -> !mo.chain
  %outTensor_0, %outChain_1 = mo.mutable.load(%arg1, %2) : (!mo.buffer<[3, 2], si32>, !mo.chain) -> (!mo.tensor<[3, 2], si32>, !mo.chain)
  %5 = mo.chain.create(%2, %4, %outChain, %outChain_1) : !mo.chain, !mo.chain, !mo.chain, !mo.chain
  %8 = mo.custom [symbol = "sampler.apply_penalties", device = #M.device_ref<"gpu", 0>](%arg0, %outTensor_0, %arg2, %arg3, %arg4, %arg4, %5) : (!mo.buffer<[7, 5], f32>, !mo.tensor<[3, 2], si32>, !mo.tensor<[11], ui32>, !mo.tensor<[7], f32>, !mo.tensor<[7], f32>, !mo.tensor<[7], f32>, !mo.chain) -> !mo.chain
  %outTensor_3, %outChain_3 = mo.mutable.load(%arg0, %8) : (!mo.buffer<[7, 5], f32>, !mo.chain) -> (!mo.tensor<[7, 5], f32>, !mo.chain)
  mo.output %outTensor_3, %outChain_3 : !mo.tensor<[7, 5], f32>, !mo.chain
}
