// RUN: mo-opt --gc-pipeline="compiled-framework-label=mof constant-fold=false default-device-label=gpu kernel-library-paths=%kernel_api_test_lib" %s -o %t
// RUN: mt --execute --default-device-label=gpu --model-inputs=ones --result-output-style=full %t | FileCheck %s
// RUN: mt --execute --default-device-label=gpu --model-inputs=ones --num-runs=2 --result-output-style=full %t | FileCheck %s --check-prefix=REPLAY

// Executing twice exercises the graph cache: the state context outlives an
// execution, so the second one replays the graph the first one built instead of
// recording it again. Both must produce all 8s.
// REPLAY-COUNT-2: 'main' returned tensor<2x3xui8> [8, 8, 8, 8, 8, 8]

// End-to-end device-graph synthesis + replay.

// CHECK-LABEL: --- Running 'main':
// CHECK: 'main' returned tensor<2x3xui8> [8, 8, 8, 8, 8, 8]
mo.graph @main(%x: !mo.tensor<[2, 3], ui8, gpu:0>) -> !mo.tensor<[2, 3], ui8> {
  %ch0 = mo.chain.create()
  %g = mo.call @device_graph_fill(%x)
      : (!mo.tensor<[2, 3], ui8, gpu:0>) -> !mo.tensor<[2, 3], ui8, gpu:0>
  %h, %ch1 = mo.transfer[%ch0] %g
      : !mo.tensor<[2, 3], ui8, gpu:0> to <"cpu", 0>
  mo.output %h : !mo.tensor<[2, 3], ui8>
}

mo.graph @device_graph_fill(%x: !mo.tensor<[2, 3], ui8, gpu:0>) -> !mo.tensor<[2, 3], ui8, gpu:0> isSubgraph isDeviceGraph {
  %0 = mo.custom [symbol = "device_graph_fill", device = #M.device_ref<"gpu", 0>]
      (%x) : (!mo.tensor<[2, 3], ui8, gpu:0>) -> !mo.tensor<[2, 3], ui8, gpu:0>
  %1 = mo.custom [symbol = "device_graph_increment", device = #M.device_ref<"gpu", 0>]
      (%0) : (!mo.tensor<[2, 3], ui8, gpu:0>) -> !mo.tensor<[2, 3], ui8, gpu:0>
  mo.output %1 : !mo.tensor<[2, 3], ui8, gpu:0>
}
