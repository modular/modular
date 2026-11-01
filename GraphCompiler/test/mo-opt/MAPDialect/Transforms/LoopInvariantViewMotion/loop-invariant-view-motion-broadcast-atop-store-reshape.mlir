// RUN: mo-opt %s --pass-pipeline='builtin.module(mo.graph(map-livm, map.iter.opaque(canonicalize,cse)))' | FileCheck %s

// A load-side view (broadcast) whose index is computed THROUGH a store-index
// reshape -- the shape a reshape-into-epilogue fuses (the bias load of
// `reshape(matmul) + broadcast(c)` reads at the reshaped store index). The
// broadcast feeds only its load, so it hoists into a map.index.broadcast.stride
// view read at the reshaped index; the reshape feeds the store and stays in the
// loop. Before the per-op "does it transitively feed a store" rule, LoopInvariant
// ViewMotion bailed on the whole chain and left the un-lowerable broadcast
// behind, erroring "an index-transform op survived hoisting".

module {

// CHECK-LABEL: mo.graph @broadcastAtopStoreReshape
// CHECK:         map.iter.opaque
// CHECK:         inputs ([[IN:%[^ :,]+]]: !mo.tensor<[1], f32>)
// CHECK:         outputs ([[OUT:%[^ :,]+]] : !mo.tensor<[8], f32>) {
// The broadcast hoists to a loop-invariant stride view of the input.
// CHECK-NEXT:      [[BS:%[^ ]+]] = map.index.broadcast.stride<#map.index.broadcast.spec<[1] -> [8]>> ([[IN]])
// CHECK-NEXT:      map.iter.elementwise.body<[2, 4]>
// CHECK-NEXT:        index   ([[IDX:%[^ :,]+]]: !map.index<2>)
// CHECK-NEXT:        capturing {{{.*}}[[BCAP:%[^ ]+]] = [[BS]] : !mo.tensor<[8], f32>} {
// The reshape stays in the loop (it feeds the store); the load reads the hoisted
// broadcast view at the reshaped index. No load-side broadcast survives.
// CHECK-NEXT:      [[R:%[^ ]+]] = map.index.reshape<#map.index.reshape.spec<[8] -> [2, 4]>> ([[IDX]])
// CHECK-NEXT:      [[VAL:%[^ ]+]] = map.load [[BCAP]]{{\[}}[[R]]] : !mo.tensor<[8], f32>
// CHECK-NEXT:      map.store {{%[^ ]+}}{{\[}}[[R]]] = [[VAL]] : !mo.tensor<[8], f32>
// CHECK-NOT:       map.index.broadcast<
mo.graph @broadcastAtopStoreReshape(%c: !mo.tensor<[1], f32>)
    -> !mo.tensor<[8], f32> {
  %output = map.iter.opaque
      inputs (%c : !mo.tensor<[1], f32>)
      outputs (%out : !mo.tensor<[8], f32>) {
    map.iter.elementwise.body<[2, 4]>
        index (%idx: !map.index<2>)
        capturing {%c0 = %c : !mo.tensor<[1], f32>, %out0 = %out : !mo.tensor<[8], f32>} {
      %r = map.index.reshape<#map.index.reshape.spec<[8] -> [2, 4]>> (%idx)
          : !map.index<2> -> !map.index<1>
      %b = map.index.broadcast<#map.index.broadcast.spec<[1] -> [8]>> (%r)
          : !map.index<1> -> !map.index<1>
      %v = map.load %c0[%b] : !mo.tensor<[1], f32>
      map.store %out0[%r] = %v : !mo.tensor<[8], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %output : !mo.tensor<[8], f32>
}

}
