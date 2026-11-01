// RUN: mo-opt --mlir-print-local-scope %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels | FileCheck %s

// CHECK-LABEL: mo.graph @add_gather
// CHECK-SAME:  (%[[A:.*]]: !mo.tensor<[3, 3], f32>, %[[B:.*]]: !mo.tensor<[3, 3], f32>, %[[INDICES:.*]]: !mo.tensor<[2, 2], si32>)
// CHECK-NEXT:  %[[OUT:.*]] = map.iter.opaque
// CHECK-NEXT:  inputs (%[[INDICES]]: !mo.tensor<[2, 2], si32>, %[[A]]: !mo.tensor<[3, 3], f32>, %[[B]]: !mo.tensor<[3, 3], f32>)
// CHECK-NEXT:  outputs (%{{.*}} : !mo.tensor<[2, 2, 3], f32>) {
// CHECK-NEXT:  %[[PD:.*]] = map.iter.prologue (%{{.*}} : !map.index<2>) capturing {%[[CA:.*]] = %[[A]] : !mo.tensor<[3, 3], f32>, %[[CB:.*]] = %[[B]] : !mo.tensor<[3, 3], f32>}
// CHECK-NEXT:  %[[AV:.*]] = map.load %[[CA]][%{{.*}}] : !mo.tensor<[3, 3], f32>
// CHECK-NEXT:  %[[BV:.*]] = map.load %[[CB]][%{{.*}}] : !mo.tensor<[3, 3], f32>
// CHECK-NEXT:  %[[ADD:.*]] = map.call.elementwise @{{.*}}.Add(%{{.*}}, %[[AV]], %[[BV]])
// CHECK-NEXT:  map.yield %[[ADD]] : f32
// CHECK:       %[[PI:.*]] = map.iter.prologue (%{{.*}} : !map.index<2>) capturing {%[[CI:.*]] = %[[INDICES]] : !mo.tensor<[2, 2], si32>}
// CHECK-NEXT:  %{{.*}} = map.load %[[CI]][%{{.*}}] : !mo.tensor<[2, 2], si32>
// CHECK-NEXT:  map.yield
// CHECK:       map.call.opaque @{{.*}}.Gather{{.*}} inputs(%[[PD]], %[[PI]])
// CHECK:       #mogg.fusion_prologue<[#mogg.fusion_leaf<"mo.add"
// CHECK-SAME:  #mogg.fusion_leaf<"mo.gather"
// CHECK-SAME:  operandIndices = [0]
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[OUT]] : !mo.tensor<[2, 2, 3], f32>
mo.graph @add_gather(
  %a: !mo.tensor<[3, 3], f32>,
  %b: !mo.tensor<[3, 3], f32>,
  %indices: !mo.tensor<[2, 2], si32>
) -> !mo.tensor<[2, 2, 3], f32> {
  %data = mo.add(%a, %b) : !mo.tensor<[3, 3], f32>
  %out = mo.gather(%data, %indices) [axis = 0] : (!mo.tensor<[3, 3], f32>, !mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 2, 3], f32>
  mo.output %out : !mo.tensor<[2, 2, 3], f32>
}

// CHECK-LABEL: mo.graph @add_gather_indices
// CHECK-SAME:  (%[[DATA:.*]]: !mo.tensor<[3, 3], f32>, %[[I0:.*]]: !mo.tensor<[2, 2], si32>, %[[I1:.*]]: !mo.tensor<[2, 2], si32>)
// CHECK:       %[[PD:.*]] = map.iter.prologue (%{{.*}} : !map.index<2>) capturing {%{{.*}} = %[[DATA]] : !mo.tensor<[3, 3], f32>}
// CHECK-NOT:   map.call.elementwise
// CHECK:       %[[PI:.*]] = map.iter.prologue (%{{.*}} : !map.index<2>) capturing {%[[C0:.*]] = %[[I0]] : !mo.tensor<[2, 2], si32>, %[[C1:.*]] = %[[I1]] : !mo.tensor<[2, 2], si32>}
// CHECK:       map.call.elementwise @{{.*}}.Add(%{{.*}}) : !map.index<2>, si32, si32 -> si32
// CHECK:       map.call.opaque @{{.*}}.Gather{{.*}} inputs(%[[PD]], %[[PI]])
// CHECK:       #mogg.fusion_prologue<[#mogg.fusion_leaf<"mo.add"
// CHECK-SAME:  operandIndices = [1]
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output
mo.graph @add_gather_indices(
  %data: !mo.tensor<[3, 3], f32>,
  %i0: !mo.tensor<[2, 2], si32>,
  %i1: !mo.tensor<[2, 2], si32>
) -> !mo.tensor<[2, 2, 3], f32> {
  %indices = mo.add(%i0, %i1) : !mo.tensor<[2, 2], si32>
  %out = mo.gather(%data, %indices) [axis = 0] : (!mo.tensor<[3, 3], f32>, !mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 2, 3], f32>
  mo.output %out : !mo.tensor<[2, 2, 3], f32>
}

// CHECK-LABEL: mo.graph @chain
// CHECK-SAME:  (%[[A:.*]]: !mo.tensor<[2, 3], f32>)
// CHECK-NEXT:  %[[OUT:.*]] = map.iter.opaque
// CHECK-NEXT:  inputs (%[[A]]: !mo.tensor<[2, 3], f32>)
// CHECK:       %[[P:.*]] = map.iter.prologue
// CHECK:       map.call.elementwise @{{.*}}.Add(%{{.*}}) : !map.index<2>, f32, f32 -> f32
// CHECK-NEXT:  %[[CAST:.*]] = map.call.elementwise @{{.*}}.Cast(%{{.*}}) : !map.index<2>, f32 -> f64
// CHECK-NEXT:  map.yield %[[CAST]] : f64
// CHECK-NEXT:  } : !mo.tensor<[2, 3], f64>
// CHECK-NEXT:  map.call.opaque @{{.*}}.Softmax{{.*}} inputs(%[[P]])
// CHECK:       #mogg.fusion_prologue<[#mogg.fusion_elementwise<[#mogg.fusion_leaf<"mo.add"
// CHECK-SAME:  #mogg.fusion_leaf<"mo.cast"
// CHECK-SAME:  #mogg.fusion_leaf<"mo.reduce.softmax"
// CHECK-SAME:  operandIndices = [0]
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[OUT]] : !mo.tensor<[2, 3], f64>
mo.graph @chain(%a: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f64> {
  %add = mo.add(%a, %a) : !mo.tensor<[2, 3], f32>
  %cast = mo.cast(%add) : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f64>
  %out = mo.reduce.softmax(%cast) [axis = 1] : (!mo.tensor<[2, 3], f64>) -> !mo.tensor<[2, 3], f64>
  mo.output %out : !mo.tensor<[2, 3], f64>
}

// CHECK-LABEL: mo.graph @two_fused_inputs
// CHECK:       %[[P0:.*]] = map.iter.prologue
// CHECK:       map.call.elementwise @{{.*}}.Add
// CHECK:       %[[P1:.*]] = map.iter.prologue
// CHECK:       map.call.elementwise @{{.*}}.Mul
// CHECK:       map.call.opaque @{{.*}}MapLowerTwoFusedInputs{{.*}} inputs(%[[P0]], %[[P1]])
// CHECK:       #mogg.fusion_prologue<[#mogg.fusion_leaf<"mo.add"
// CHECK-SAME:  #mogg.fusion_leaf<"mo.mul"
// CHECK-SAME:  #mogg.fusion_leaf<"custom__map_lower_two_fused_inputs"
// CHECK-SAME:  operandIndices = [0, 1]
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output
mo.graph @two_fused_inputs(
  %a: !mo.tensor<[2, 2], f32>,
  %b: !mo.tensor<[2, 2], f32>,
  %c: !mo.tensor<[2, 2], f32>,
  %d: !mo.tensor<[2, 2], f32>
) -> !mo.tensor<[2, 2], f32> {
  %p0 = mo.add(%a, %b) : !mo.tensor<[2, 2], f32>
  %p1 = mo.mul(%c, %d) : !mo.tensor<[2, 2], f32>
  %out = mo.custom [symbol = "map_lower_two_fused_inputs", device = #M.device_ref<"cpu", 0>](%p0, %p1) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %out : !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL: mo.graph @partial_fused_inputs
// CHECK:       %[[EW:.*]] = map.iter.elementwise
// CHECK:       map.call.elementwise @{{.*}}.Mul
// CHECK:       map.iter.opaque
// CHECK:       %[[P:.*]] = map.iter.prologue
// CHECK:       map.call.elementwise @{{.*}}.Add
// CHECK:       map.call.opaque @{{.*}}MapLowerPartialFusedInputs{{.*}} inputs(%[[P]], %[[EW]])
// CHECK:       #mogg.fusion_prologue<[#mogg.fusion_leaf<"mo.add"
// CHECK-SAME:  #mogg.fusion_leaf<"custom__map_lower_partial_fused_inputs"
// CHECK-SAME:  operandIndices = [0]
// CHECK:       mo.output
mo.graph @partial_fused_inputs(
  %a: !mo.tensor<[2, 2], f32>,
  %b: !mo.tensor<[2, 2], f32>,
  %c: !mo.tensor<[2, 2], f32>,
  %d: !mo.tensor<[2, 2], f32>
) -> !mo.tensor<[2, 2], f32> {
  %p0 = mo.add(%a, %b) : !mo.tensor<[2, 2], f32>
  %p1 = mo.mul(%c, %d) : !mo.tensor<[2, 2], f32>
  %out = mo.custom [symbol = "map_lower_partial_fused_inputs", device = #M.device_ref<"cpu", 0>](%p0, %p1) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %out : !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL: mo.graph @fuse_in_if
// CHECK:       mo.if
// CHECK:       map.iter.opaque
// CHECK:       map.call.elementwise @{{.*}}.Add
// CHECK:       #mogg.fusion_prologue<[#mogg.fusion_leaf<"mo.add"
// CHECK-NOT:   map.iter.elementwise
// CHECK:       } else {
// CHECK:       map.iter.opaque
// CHECK:       map.call.elementwise @{{.*}}.Mul
// CHECK:       #mogg.fusion_prologue<[#mogg.fusion_leaf<"mo.mul"
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output
mo.graph @fuse_in_if(
  %a: !mo.tensor<[3, 3], f32>,
  %b: !mo.tensor<[3, 3], f32>,
  %indices: !mo.tensor<[2, 2], si32>,
  %cond: !mo.tensor<[], bool>
) -> !mo.tensor<[2, 2, 3], f32> {
  %res = mo.if (%cond : !mo.tensor<[], bool>) -> !mo.tensor<[2, 2, 3], f32> {
    %data = mo.add(%a, %b) : !mo.tensor<[3, 3], f32>
    %out = mo.gather(%data, %indices) [axis = 0] : (!mo.tensor<[3, 3], f32>, !mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 2, 3], f32>
    mo.yield %out : !mo.tensor<[2, 2, 3], f32>
  } else {
    %data = mo.mul(%a, %b) : !mo.tensor<[3, 3], f32>
    %out = mo.gather(%data, %indices) [axis = 0] : (!mo.tensor<[3, 3], f32>, !mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 2, 3], f32>
    mo.yield %out : !mo.tensor<[2, 2, 3], f32>
  }
  mo.output %res : !mo.tensor<[2, 2, 3], f32>
}

// CHECK-LABEL: mo.graph @no_fuse_across_blocks
// CHECK:       %[[EW:.*]] = map.iter.elementwise
// CHECK:       map.call.elementwise @{{.*}}.Add
// CHECK:       mo.if
// CHECK:       map.iter.opaque
// CHECK-NEXT:  inputs (%[[EW]]: !mo.tensor<[3, 3], f32>, %{{.*}}: !mo.tensor<[2, 2], si32>)
// CHECK-NOT:   map.call.elementwise
// CHECK:       #mogg.fusion<#mogg.fusion_leaf<"mo.gather"
mo.graph @no_fuse_across_blocks(
  %a: !mo.tensor<[3, 3], f32>,
  %b: !mo.tensor<[3, 3], f32>,
  %indices: !mo.tensor<[2, 2], si32>,
  %cond: !mo.tensor<[], bool>
) -> !mo.tensor<[2, 2, 3], f32> {
  %data = mo.add(%a, %b) : !mo.tensor<[3, 3], f32>
  %res = mo.if (%cond : !mo.tensor<[], bool>) -> !mo.tensor<[2, 2, 3], f32> {
    %out = mo.gather(%data, %indices) [axis = 0] : (!mo.tensor<[3, 3], f32>, !mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 2, 3], f32>
    mo.yield %out : !mo.tensor<[2, 2, 3], f32>
  } else {
    %z = mo.gather(%a, %indices) [axis = 0] : (!mo.tensor<[3, 3], f32>, !mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 2, 3], f32>
    mo.yield %z : !mo.tensor<[2, 2, 3], f32>
  }
  mo.output %res : !mo.tensor<[2, 2, 3], f32>
}

// CHECK-LABEL: mo.graph @no_fuse_multi_use
// CHECK:       %[[EW:.*]] = map.iter.elementwise
// CHECK:       map.call.elementwise @{{.*}}.Add
// CHECK:       %[[G:.*]] = map.iter.opaque
// CHECK-NEXT:  inputs (%[[EW]]: !mo.tensor<[3, 3], f32>, %{{.*}}: !mo.tensor<[2, 2], si32>)
// CHECK-NOT:   map.call.elementwise
// CHECK:       #mogg.fusion<#mogg.fusion_leaf<"mo.gather"
// CHECK:       mo.output %[[G]], %[[EW]]
mo.graph @no_fuse_multi_use(
  %a: !mo.tensor<[3, 3], f32>,
  %b: !mo.tensor<[3, 3], f32>,
  %indices: !mo.tensor<[2, 2], si32>
) -> (!mo.tensor<[2, 2, 3], f32>, !mo.tensor<[3, 3], f32>) {
  %data = mo.add(%a, %b) : !mo.tensor<[3, 3], f32>
  %out = mo.gather(%data, %indices) [axis = 0] : (!mo.tensor<[3, 3], f32>, !mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 2, 3], f32>
  mo.output %out, %data : !mo.tensor<[2, 2, 3], f32>, !mo.tensor<[3, 3], f32>
}

// Feeds both operands, so not fused; the duplicate inputs are `--canonicalize`'s.
// CHECK-LABEL: mo.graph @no_fuse_same_producer_twice
// CHECK:       %[[EW:.*]] = map.iter.elementwise
// CHECK:       map.call.elementwise @{{.*}}.Add
// CHECK:       map.iter.opaque
// CHECK-NEXT:  inputs (%[[EW]] as %{{.*}}: !mo.tensor<[2, 2], f32>, %[[EW]] as %{{.*}}: !mo.tensor<[2, 2], f32>)
// CHECK-NOT:   map.call.elementwise
// CHECK:       #mogg.fusion<#mogg.fusion_leaf<"custom__map_lower_two_fused_inputs"
mo.graph @no_fuse_same_producer_twice(
  %a: !mo.tensor<[2, 2], f32>,
  %b: !mo.tensor<[2, 2], f32>
) -> !mo.tensor<[2, 2], f32> {
  %p = mo.add(%a, %b) : !mo.tensor<[2, 2], f32>
  %out = mo.custom [symbol = "map_lower_two_fused_inputs", device = #M.device_ref<"cpu", 0>](%p, %p) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %out : !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL: mo.graph @no_fuse_plain_input
// CHECK:       %[[EW:.*]] = map.iter.elementwise
// CHECK:       map.call.elementwise @{{.*}}.Add
// CHECK:       map.iter.opaque
// CHECK-NOT:   map.iter.prologue
// CHECK:       map.call.opaque @{{.*}}.Matmul{{.*}} inputs(%[[EW]], %{{.*}})
mo.graph @no_fuse_plain_input(
  %x: !mo.tensor<[8, 8], f32>,
  %y: !mo.tensor<[8, 8], f32>,
  %z: !mo.tensor<[8, 8], f32>
) -> !mo.tensor<[8, 8], f32> {
  %p = mo.add(%x, %y) : !mo.tensor<[8, 8], f32>
  %out = mo.matmul(%p, %z) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
  mo.output %out : !mo.tensor<[8, 8], f32>
}
