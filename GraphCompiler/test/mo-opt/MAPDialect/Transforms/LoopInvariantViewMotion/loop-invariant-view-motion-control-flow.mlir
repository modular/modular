// RUN: mo-opt %s --pass-pipeline='builtin.module(mogg-import-packages, mo-to-map, map-outline-structured-kernel-bodies, mo.graph(map-livm, map.iter.opaque(canonicalize)), map-to-mogg)' | FileCheck %s

// A kernel nested in a control-flow region must be hoisted too. The rewrite is
// mandatory: `map.index.broadcast` is index-space and declares no
// `LowerableToMOGGInterface`, so an unhoisted one strands `map-to-mogg` with
// "unsupported op". Anchoring this pass on `map.iter.opaque` used to skip these
// kernels entirely, because MLIR's pass adaptor only visits ops in immediately
// nested regions.

// CHECK-LABEL: mo.graph @broadcast_add_in_while
// CHECK:       mogg.tensor.create.broadcast
// CHECK-NOT:   map.index.broadcast<
mo.graph @broadcast_add_in_while(
    %arg0: !mo.tensor<[2], f32>,
    %scalar: !mo.tensor<[], f32>,
    %cond: !mo.tensor<[], bool>) -> !mo.tensor<[2], f32> {
  %0:2 = mo.while (%arg0 as %argi0: !mo.tensor<[2], f32>, %cond as %condi: !mo.tensor<[], bool>) {
    mo.while.condition(%condi) %argi0, %condi : !mo.tensor<[2], f32>, !mo.tensor<[], bool>
  } do {
    %bcast = mo.static.broadcast_to(%scalar) : !mo.tensor<[], f32> -> !mo.tensor<[2], f32>
    %sum = mo.add(%argi0, %bcast) : !mo.tensor<[2], f32>
    mo.yield %sum, %condi : !mo.tensor<[2], f32>, !mo.tensor<[], bool>
  }
  mo.output %0#0 : !mo.tensor<[2], f32>
}

// The loop is incidental -- any control-flow region strands the view the same
// way, so cover a branch too.

// CHECK-LABEL: mo.graph @broadcast_add_in_if
// CHECK:       mogg.tensor.create.broadcast
// CHECK-NOT:   map.index.broadcast<
mo.graph @broadcast_add_in_if(
    %arg0: !mo.tensor<[2], f32>,
    %scalar: !mo.tensor<[], f32>,
    %cond: !mo.tensor<[], bool>) -> !mo.tensor<[2], f32> {
  %0 = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[2], f32> {
    %bcast = mo.static.broadcast_to(%scalar) : !mo.tensor<[], f32> -> !mo.tensor<[2], f32>
    %sum = mo.add(%arg0, %bcast) : !mo.tensor<[2], f32>
    mo.yield %sum : !mo.tensor<[2], f32>
  } else {
    mo.yield %arg0 : !mo.tensor<[2], f32>
  }
  mo.output %0 : !mo.tensor<[2], f32>
}
