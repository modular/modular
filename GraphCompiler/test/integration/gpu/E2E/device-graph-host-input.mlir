// RUN: mo-opt --gc-pipeline="compiled-framework-label=mof constant-fold=false default-device-label=gpu kernel-library-paths=%kernel_api_test_lib" %s -o %t
// RUN: mt --execute --default-device-label=gpu --model-inputs=ones --result-output-style=full %t | FileCheck %s
// RUN: mt --execute --default-device-label=gpu --model-inputs=ones --num-runs=2 --result-output-style=full %t | FileCheck %s --check-prefix=REPLAY

// A host-resident input feeds the `isDeviceGraph` subgraph directly -- no
// `mo.transfer` workaround. The input gets a graph-stable host twin populated
// before recording; `device_graph_fill_from_host` reads its value on the host
// at record time (the dispatch-metadata pattern), baking it into the recorded
// fill; the contents key the graph cache. With `--model-inputs=ones` the fill
// value is 1 and the chained increment yields all 2s.

// Executing twice exercises the cache: the host bytes are identical, so the
// second run replays the graph the first one recorded. Both must produce 2s.
// REPLAY-COUNT-2: 'main' returned tensor<2x3xui8> [2, 2, 2, 2, 2, 2]

// CHECK-LABEL: --- Running 'main':
// CHECK: 'main' returned tensor<2x3xui8> [2, 2, 2, 2, 2, 2]
mo.graph @main(%h: !mo.tensor<[], ui8>) -> !mo.tensor<[2, 3], ui8> {
  %ch0 = mo.chain.create()
  %g = mo.call @device_graph_host_fill(%h)
      : (!mo.tensor<[], ui8>) -> !mo.tensor<[2, 3], ui8, gpu:0>
  %out, %ch1 = mo.transfer[%ch0] %g
      : !mo.tensor<[2, 3], ui8, gpu:0> to <"cpu", 0>
  mo.output %out : !mo.tensor<[2, 3], ui8>
}

mo.graph @device_graph_host_fill(%h: !mo.tensor<[], ui8>) -> !mo.tensor<[2, 3], ui8, gpu:0> isSubgraph isDeviceGraph {
  %0 = mo.custom [symbol = "device_graph_fill_from_host", device = #M.device_ref<"gpu", 0>]
      (%h) : (!mo.tensor<[], ui8>) -> !mo.tensor<[2, 3], ui8, gpu:0>
  %1 = mo.custom [symbol = "device_graph_increment", device = #M.device_ref<"gpu", 0>]
      (%0) : (!mo.tensor<[2, 3], ui8, gpu:0>) -> !mo.tensor<[2, 3], ui8, gpu:0>
  mo.output %1 : !mo.tensor<[2, 3], ui8, gpu:0>
}
