// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg --mogg-to-mgp | FileCheck %s
// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg --mogg-to-mgp --mgp-exec-invariant-code-motion | FileCheck %s --check-prefix=ICM

// A pure op after an mo.if that reads none of its results keeps the chain
// from before the if, so it lifts to init. Only the output waits on the if.

// CHECK-LABEL: mgp.model @pure_op_after_if
// CHECK:       } execute (%[[CH:[^:]*]]: !mef.chain
// CHECK:       %[[IF:.*]]:2 = grt.if
// CHECK:       } else {
// CHECK:       %[[PRE:.*]] = grt.chain.create (%[[CH]], %[[CH]])
// CHECK-NOT:   %[[IF]]#
// CHECK:       mogg.routine.async {{(dispatch\([a-z]+\) )?}}(%[[PRE]],
// CHECK:       mef.output %[[IF]]#0,

// ICM-LABEL:   mgp.model @pure_op_after_if
// ICM:         mogg.routine.async
// ICM:         } execute (
// ICM:         grt.if
// ICM:         } else {
// ICM:         }
// ICM-NOT:     mogg.routine.async
// ICM:         mef.output
mo.graph @pure_op_after_if(%buf: !mo.buffer<[2], f32>, %cond: !mo.tensor<[], bool>) -> !mo.tensor<[2], f32> {
  %ch0 = mo.chain.create()
  %ch1 = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.chain {
    %v = mo.constant <#M.dense_array<5.0, 6.0> : tensor<2xf32>> : !mo.tensor<[2], f32>
    %s = mo.mutable.store(%buf, %v, %ch0) : (!mo.buffer<[2], f32>, !mo.tensor<[2], f32>, !mo.chain) -> !mo.chain
    mo.yield %s : !mo.chain
  } else {
    mo.yield %ch0 : !mo.chain
  }
  %w = mo.constant <#M.dense_array<1.0, 2.0> : tensor<2xf32>> : !mo.tensor<[2], f32>
  %r = mo.negative(%w) : !mo.tensor<[2], f32>
  mo.output %r : !mo.tensor<[2], f32>
}

// A load after an mo.if of the buffer the if writes keeps the if's chain.

// CHECK-LABEL: mgp.model @load_after_if
// CHECK:       %[[IF:.*]]:2 = grt.if
// CHECK:       } else {
// CHECK:       mgp.buffer.device_to_device %[[IF]]#1,
mo.graph @load_after_if(%buf: !mo.buffer<[2], f32>, %cond: !mo.tensor<[], bool>) -> !mo.tensor<[2], f32> {
  %ch0 = mo.chain.create()
  %ch1 = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.chain {
    %v = mo.constant <#M.dense_array<5.0, 6.0> : tensor<2xf32>> : !mo.tensor<[2], f32>
    %s = mo.mutable.store(%buf, %v, %ch0) : (!mo.buffer<[2], f32>, !mo.tensor<[2], f32>, !mo.chain) -> !mo.chain
    mo.yield %s : !mo.chain
  } else {
    mo.yield %ch0 : !mo.chain
  }
  %t, %ch2 = mo.mutable.load(%buf, %ch1) : (!mo.buffer<[2], f32>, !mo.chain) -> (!mo.tensor<[2], f32>, !mo.chain)
  mo.output %t : !mo.tensor<[2], f32>
}

// A write to a slice of the buffer the if writes keeps the if's chain.

// CHECK-LABEL: mgp.model @store_slice_after_if
// CHECK:       %[[IF:.*]]:2 = grt.if
// CHECK:       } else {
// CHECK:       %[[AFTER:.*]] = grt.chain.create ({{.*}}%[[IF]]#1)
// CHECK:       mogg.routine.async {{(dispatch\([a-z]+\) )?}}(%[[AFTER]],
mo.graph @store_slice_after_if(%buf: !mo.buffer<[4], f32>, %cond: !mo.tensor<[], bool>) -> !mo.buffer<[4], f32> {
  %ch0 = mo.chain.create()
  %ch1 = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.chain {
    %v = mo.constant <#M.dense_array<5.0, 6.0, 7.0, 8.0> : tensor<4xf32>> : !mo.tensor<[4], f32>
    %s = mo.mutable.store(%buf, %v, %ch0) : (!mo.buffer<[4], f32>, !mo.tensor<[4], f32>, !mo.chain) -> !mo.chain
    mo.yield %s : !mo.chain
  } else {
    mo.yield %ch0 : !mo.chain
  }
  %x = mo.constant <#M.dense_array<1.0, 2.0> : tensor<2xf32>> : !mo.tensor<[2], f32>
  %start = mo.constant <#M.dense_array<0> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %stop = mo.constant <#M.dense_array<2> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %step = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %ch2 = mo.mutable.store.slice(%buf, %x, %start, %stop, %step, %ch1) : (!mo.buffer<[4], f32>, !mo.tensor<[2], f32>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.chain) -> !mo.chain
  mo.output %buf : !mo.buffer<[4], f32>
}

// A store after an mo.if to the buffer the if writes keeps the if's chain.

// CHECK-LABEL: mgp.model @store_after_if
// CHECK:       %[[IF:.*]]:2 = grt.if
// CHECK:       } else {
// CHECK:       %[[AFTER:.*]] = grt.chain.create (%{{.*}}, %[[IF]]#1)
// CHECK:       mogg.routine.async {{(dispatch\([a-z]+\) )?}}(%[[AFTER]],
mo.graph @store_after_if(%buf: !mo.buffer<[2], f32>, %cond: !mo.tensor<[], bool>) -> !mo.buffer<[2], f32> {
  %ch0 = mo.chain.create()
  %ch1 = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.chain {
    %v = mo.constant <#M.dense_array<5.0, 6.0> : tensor<2xf32>> : !mo.tensor<[2], f32>
    %s = mo.mutable.store(%buf, %v, %ch0) : (!mo.buffer<[2], f32>, !mo.tensor<[2], f32>, !mo.chain) -> !mo.chain
    mo.yield %s : !mo.chain
  } else {
    mo.yield %ch0 : !mo.chain
  }
  %w = mo.constant <#M.dense_array<1.0, 2.0> : tensor<2xf32>> : !mo.tensor<[2], f32>
  %ch2 = mo.mutable.store(%buf, %w, %ch1) : (!mo.buffer<[2], f32>, !mo.tensor<[2], f32>, !mo.chain) -> !mo.chain
  mo.output %buf : !mo.buffer<[2], f32>
}
