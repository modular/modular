// RUN: mo-opt --mlir-print-local-scope %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels --canonicalize | FileCheck %s

// MO-level counterpart of view_fusion_into_epilogue.mlir's single-view case:
// a chain of TWO view ops (slice, then broadcast) fuses into an existing
// epilogue capture, one hop per round of `--map-fuse-kernels`'s fixed point
// -- the round that absorbs the broadcast exposes the slice feeding it as a
// new "producer feeding an existing capture" candidate, which the next round
// then also absorbs. Driven from plain `mo.slice`/`mo.static.broadcast_to`
// through the real MOToMAP + fusion pipeline rather than hand-authored MAP
// IR: composing two chained transforms by hand would be tedious and easy to
// get wrong, where the real pipeline just produces it.

mo.graph @broadcastOfSliceIntoExistingEpilogue(
    %main: !mo.tensor<[8, 8], f32>,
    %bias_wide: !mo.tensor<[16], f32>
) -> !mo.tensor<[8, 8], f32> {
  %start = mo.constant <#M.dense_array<4> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %stop = mo.constant <#M.dense_array<12> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %step = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%main) : (!mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
  %sliced = mo.slice(%bias_wide, %start, %stop, %step) : (
    !mo.tensor<[16], f32>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>
  ) -> !mo.tensor<[8], f32>
  %bcast = mo.static.broadcast_to(%sliced) : !mo.tensor<[8], f32> -> !mo.tensor<[8, 8], f32>
  %out = mo.add(%c, %bcast) : !mo.tensor<[8, 8], f32>
  mo.output %out : !mo.tensor<[8, 8], f32>
}

// CHECK-LABEL: mo.graph @broadcastOfSliceIntoExistingEpilogue
// CHECK-SAME:  (%[[MAIN:.*]]: !mo.tensor<[8, 8], f32>, %[[BIAS:.*]]: !mo.tensor<[16], f32>)
// CHECK-NOT:   map.iter.elementwise
// CHECK:       %[[RES:.*]] = map.iter.opaque
// The slice bounds are single-element host constants, so
// `FuseConstantIterOpaqueInputs` rematerializes them in the body rather than
// passing them in: the kernel keeps only its two real tensor operands.
// CHECK-NEXT:  inputs (%[[MAIN]]: !mo.tensor<[8, 8], f32>, %[[BIAS]]: !mo.tensor<[16], f32>)
// CHECK-NEXT:  outputs (%[[O:.*]] : !mo.tensor<[8, 8], f32>) {
// CHECK-NEXT:  %[[START:.*]] = map.constant #M.dense_array<4> : tensor<1xsi64>
// CHECK-NEXT:  %[[STOP:.*]] = map.constant #M.dense_array<12> : tensor<1xsi64>
// CHECK-NEXT:  %[[STEP:.*]] = map.constant #M.dense_array<1> : tensor<1xsi64>
// CHECK-NEXT:  %[[EPI:.*]] = map.iter.epilogue (%[[VAL:.*]] : f32, %[[IDX:.*]] : !map.index<2>) capturing
// CHECK-SAME:      {%[[OCAP:.*]] = %[[O]] : !mo.tensor<[8, 8], f32>,
// CHECK-SAME:       %[[BCAP:.*]] = %[[BIAS]] : !mo.tensor<[16], f32>,
// CHECK-SAME:       %[[STARTCAP:.*]] = %[[START]] : !mo.tensor<[1], si64>,
// CHECK-SAME:       %[[STOPCAP:.*]] = %[[STOP]] : !mo.tensor<[1], si64>,
// CHECK-SAME:       %[[STEPCAP:.*]] = %[[STEP]] : !mo.tensor<[1], si64>}
// CHECK-NEXT:  %[[BIDX:.*]] = map.index.broadcast<{{.*}}> (%[[IDX]])
// CHECK-NEXT:  %[[SIDX:.*]] = map.index.slice<{{.*}}> (%[[BIDX]], %[[STARTCAP]], %[[STOPCAP]], %[[STEPCAP]])
// CHECK-NEXT:  %[[BV:.*]] = map.load %[[BCAP]][%[[SIDX]]] : !mo.tensor<[16], f32>
// CHECK-NEXT:  %[[SUM:.*]] = map.call.elementwise @{{.*}}.Add(%[[IDX]], %[[VAL]], %[[BV]])
// CHECK-NEXT:  map.store %[[OCAP]][%[[IDX]]] = %[[SUM]]
// CHECK-NEXT:  } : !mo.tensor<[8, 8], f32>
// CHECK-NEXT:  map.call.opaque @{{.*}}MapLowerOneFusedOutput{{.*}} inputs(%[[MAIN]]) outputs(%[[EPI]])
// The view chain lands in the epilogue emergently: ViewFuser folds the slice
// and broadcast into the add foreach, then output fusion pulls that foreach into
// the custom op's epilogue -- so the tree nests one view (slice, broadcast, add)
// under the epilogue.
// CHECK:       #mogg.fusion_epilogue<[#mogg.fusion_leaf<"custom__map_lower_one_fused_output"
// CHECK-SAME:  #mogg.fusion_view<[#mogg.fusion_leaf<"mo.slice"
// CHECK-SAME:  #mogg.fusion_leaf<"mo.static.broadcast_to"
// CHECK-SAME:  #mogg.fusion_leaf<"mo.add"
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[RES]] : !mo.tensor<[8, 8], f32>
