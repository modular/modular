// RUN: mo-opt --gc-pipeline="compiled-framework-label=mof constant-fold=false default-device-label=gpu kernel-library-paths=%kernel_api_test_lib" %s -o %t
// RUN: mt --execute --default-device-label=gpu --model-inputs=ones --result-output-style=full %t | FileCheck %s

// End-to-end device-graph synthesis + replay for the execute-only recording
// fallback. `imposter_add` exposes no `build_graph` method, so its recording
// stub calls `execute` with the builder's recording context as the launch
// argument; the kernels `execute` enqueues are recorded as graph nodes instead
// of running eagerly. The subgraph chains `device_graph_fill` (records via its
// `build_graph` method, all 7s) into the execute-only `imposter_add` with the
// all-ones graph input, so replaying the recorded graph must produce all 8s --
// proving the fallback-recorded dispatch ran and was ordered after the fill
// whose result it consumes.

// CHECK-LABEL: --- Running 'main':
// CHECK: 'main' returned tensor<2x3xui8> [8, 8, 8, 8, 8, 8]
mo.graph @main(%x: !mo.tensor<[2, 3], ui8>) -> !mo.tensor<[2, 3], ui8> {
  %ch0 = mo.chain.create()
  %x_gpu, %ch1 = mo.transfer[%ch0] %x
      : !mo.tensor<[2, 3], ui8> to <"gpu", 0>
  %g = mo.call @device_graph_add(%x_gpu)
      : (!mo.tensor<[2, 3], ui8, gpu:0>) -> !mo.tensor<[2, 3], ui8, gpu:0>
  %h, %ch2 = mo.transfer[%ch1] %g
      : !mo.tensor<[2, 3], ui8, gpu:0> to <"cpu", 0>
  mo.output %h : !mo.tensor<[2, 3], ui8>
}

mo.graph @device_graph_add(%x: !mo.tensor<[2, 3], ui8, gpu:0>) -> !mo.tensor<[2, 3], ui8, gpu:0> isSubgraph isDeviceGraph {
  %0 = mo.custom [symbol = "device_graph_fill", device = #M.device_ref<"gpu", 0>]
      (%x) : (!mo.tensor<[2, 3], ui8, gpu:0>) -> !mo.tensor<[2, 3], ui8, gpu:0>
  %1 = mo.custom [symbol = "imposter_add", device = #M.device_ref<"gpu", 0>]
      (%0, %x) : (!mo.tensor<[2, 3], ui8, gpu:0>, !mo.tensor<[2, 3], ui8, gpu:0>) -> !mo.tensor<[2, 3], ui8, gpu:0>
  mo.output %1 : !mo.tensor<[2, 3], ui8, gpu:0>
}
