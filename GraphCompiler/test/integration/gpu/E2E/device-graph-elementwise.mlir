// RUN: mo-opt --gc-pipeline="compiled-framework-label=mof constant-fold=false default-device-label=gpu" %s -o %t
// RUN: mt --execute --default-device-label=gpu --model-inputs=ones --result-output-style=full %t | FileCheck %s

// CHECK-LABEL: --- Running 'main':
// CHECK: 'main' returned tensor<2x3xui8> [2, 2, 2, 2, 2, 2]
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
  %0 = mo.add(%x, %x) : !mo.tensor<[2, 3], ui8, gpu:0>
  mo.output %0 : !mo.tensor<[2, 3], ui8, gpu:0>
}
