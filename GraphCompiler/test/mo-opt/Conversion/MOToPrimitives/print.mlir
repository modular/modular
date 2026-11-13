// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg --mogg-to-mgp | FileCheck %s
// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg --mogg-to-mgp --mgp-exec-invariant-code-motion | FileCheck %s --check-prefix=ICM

// CHECK-LABEL: mgp.model @print_string
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,
// CHECK:  = mgp.debug.print %{{.*}}, "debug string", "label"

mo.graph @print_string() {
  %ch0 = mo.chain.create()
  mo.debug.print(%ch0) [value = "debug string", label = "label"]
  mo.output
}


// CHECK-LABEL: mgp.model @print
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,
// CHECK: %[[ARG:[^:]*]]: !mgp.tensor<4x5xf32>)
// CHECK: %[[BUF:.*]] = mgp.tensor.extract.buffer %[[ARG]]
// CHECK: %[[SPEC:.*]] = mgp.tensor.extract.tensor_spec %[[ARG]]
// CHECK: %[[CH1:.*]] = mgp.debug.tensor.print %{{.*}}, %[[BUF]], %[[SPEC]], "label"

mo.graph @print(%arg0: !mo.tensor<[4, 5], f32>) {
  %ch0 = mo.chain.create()
  mo.debug.tensor.print(%ch0, %arg0) [label = "label"] : !mo.tensor<[4, 5], f32>
  mo.output
}

// CHECK-LABEL: mgp.model @print_chained

// CHECK: %[[CHAIN:.*]] = mogg.routine.async
// CHECK: %[[JOINED_CH:.*]] = grt.chain.create (%[[CHAIN]]
// CHECK: gp.debug.tensor.print %[[JOINED_CH]]

mo.graph @print_chained(%arg0: !mo.tensor<[4, 5], f32>) {
  %ch0 = mo.chain.create()
  %0 = mo.relu(%arg0): !mo.tensor<[4, 5], f32>
  mo.debug.tensor.print(%ch0, %0) [label = "label"] : !mo.tensor<[4, 5], f32>
  mo.output
}

// CHECK-LABEL: mgp.model @sequenced_print_string

// CHECK: [[ch0:%.*]] = grt.chain.create ()
// CHECK: [[ch1:%.*]] = mgp.debug.print [[ch0]]
// CHECK: [[ch2:%.*]] = mgp.debug.print [[ch1]]
mo.graph @sequenced_print_string() {
  %ch0 = mo.chain.create()
  %ch1 = mo.debug.print(%ch0) [value = "first debug string", label = "label"]
  %ch2 = mo.debug.print(%ch1) [value = "second debug string", label = "label"]
  mo.output
}

// CHECK-LABEL: mgp.model @sequenced_print

// CHECK: [[ch0:%.*]] = grt.chain.create ()
// CHECK: [[ch1:%.*]] = mgp.debug.tensor.print [[ch0]]
// CHECK: [[ch2:%.*]] = mgp.debug.tensor.print [[ch1]]
mo.graph @sequenced_print(%arg0: !mo.tensor<[4, 5], f32>) {
  %ch0 = mo.chain.create()
  %ch1 = mo.debug.tensor.print(%ch0, %arg0) [label = "label1"] : !mo.tensor<[4, 5], f32>
  %ch2 = mo.debug.tensor.print(%ch1, %arg0) [label = "label2"] : !mo.tensor<[4, 5], f32>
  mo.output
}

// A pure op after a print keeps the chain from before the print, so it lifts
// to init. Only the output waits on the print.

// CHECK-LABEL: mgp.model @pure_op_after_print
// CHECK:       } execute (%[[CH:[^:]*]]: !mef.chain
// CHECK:       %[[PRINT:.*]] = mgp.debug.tensor.print
// CHECK-NOT:   %[[PRINT]]
// CHECK:       %[[PRE:.*]] = grt.chain.create (%[[CH]], %[[CH]])
// CHECK-NOT:   %[[PRINT]]
// CHECK:       mogg.routine.async {{(dispatch\([a-z]+\) )?}}(%[[PRE]],
// CHECK:       mef.output %[[PRINT]],

// ICM-LABEL:   mgp.model @pure_op_after_print
// ICM:         mogg.routine.async
// ICM:         } execute (
// ICM:         mgp.debug.tensor.print
// ICM-NOT:     mogg.routine.async
// ICM:         mef.output
mo.graph @pure_op_after_print(%arg0: !mo.tensor<[4, 5], f32>) -> !mo.tensor<[2], f32> {
  %ch0 = mo.chain.create()
  %ch1 = mo.debug.tensor.print(%ch0, %arg0) [label = "label"] : !mo.tensor<[4, 5], f32>
  %w = mo.constant <#M.dense_array<1.0, 2.0> : tensor<2xf32>> : !mo.tensor<[2], f32>
  %r = mo.negative(%w) : !mo.tensor<[2], f32>
  mo.output %r : !mo.tensor<[2], f32>
}
