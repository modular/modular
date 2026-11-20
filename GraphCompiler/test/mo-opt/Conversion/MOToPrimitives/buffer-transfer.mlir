// RUN: mo-opt --mogg-to-mgp %s | FileCheck --enable-var-scope %s

// CHECK-LABEL: mgp.model @buffer_transfer_h2d
// CHECK: execute
// CHECK-SAME: (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[WREG:.*]]: !mgp.opaque<"weights_registry">, %[[CPU_CTX:.*]]: !mgp.device_context<<"cpu", 0>>, %[[GPU_CTX:.*]]: !mgp.device_context<<"gpu", 0>>, %[[ARG:.*]]: !mgp.tensor<2x2xf32, mut>)
// CHECK: %[[CH0:.*]] = grt.chain.create ()
// CHECK: %[[SPEC:.*]] = mgp.tensor_spec.create
// CHECK: %[[IDX4:.*]] = index.constant 4
// CHECK: %[[IDX16:.*]] = index.constant 16
// CHECK: %[[BUF:.*]] = mgp.buffer.alloc %[[CHAIN]], %[[IDX16]], %[[GPU_CTX]], align 256
// CHECK: %[[TENSOR:.*]] = mgp.tensor.create %[[CHAIN]], %[[BUF]], %[[SPEC]] mut : !mgp.tensor<2x2xf32, <"gpu", 0>, mut>
// CHECK: %[[SRC_BUF:.*]] = mgp.tensor.extract.buffer %[[ARG]]
// CHECK: %[[SRC_SPEC:.*]] = mgp.tensor.extract.tensor_spec %[[ARG]]
// CHECK: %[[CH1:.*]] = grt.chain.create (%[[CHAIN]], %[[CH0]])
// CHECK: %[[COPY:.*]] = mgp.buffer.host_to_device %[[CH1]], %[[SRC_BUF]], %[[BUF]], %[[GPU_CTX]]
// CHECK: %[[SYNC:.*]] = mgp.sync(%[[COPY]], %[[CTX]], %[[GPU_CTX]]) device <"gpu", 0>
// CHECK: %[[EXTEND:.*]] = mgp.extend.lifetime %[[SYNC]], %[[SRC_BUF]]
// CHECK: mef.output %[[CHAIN]], %[[TENSOR]]
mo.graph @buffer_transfer_h2d(%arg0: !mo.buffer<[2, 2], f32>) -> !mo.buffer<[2, 2], f32, gpu:0>  {
  %ch = mo.chain.create()
  %darg0 = mo.buffer.create : !mo.buffer<[2, 2], f32, gpu:0>

  mo.buffer.transfer(%arg0, %darg0, %ch) : !mo.buffer<[2, 2], f32>, !mo.buffer<[2, 2], f32, gpu:0>

  mo.output %darg0: !mo.buffer<[2, 2], f32, gpu:0>
}

// CHECK-LABEL: mgp.model @buffer_transfer_d2d
// CHECK: execute
// CHECK-SAME: (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[WREG:.*]]: !mgp.opaque<"weights_registry">, %[[CPU_CTX:.*]]: !mgp.device_context<<"cpu", 0>>, %[[GPU0_CTX:.*]]: !mgp.device_context<<"gpu", 0>>, %[[GPU1_CTX:.*]]: !mgp.device_context<<"gpu", 1>>, %[[ARG:.*]]: !mgp.tensor<2x2xf32, <"gpu", 0>, mut>)
// CHECK: %[[CH0:.*]] = grt.chain.create ()
// CHECK: %[[SPEC:.*]] = mgp.tensor_spec.create
// CHECK: %[[IDX4:.*]] = index.constant 4
// CHECK: %[[IDX16:.*]] = index.constant 16
// CHECK: %[[BUF:.*]] = mgp.buffer.alloc %[[CHAIN]], %[[IDX16]], %[[GPU1_CTX]], align 256
// CHECK: %[[TENSOR:.*]] = mgp.tensor.create %[[CHAIN]], %[[BUF]], %[[SPEC]] mut : !mgp.tensor<2x2xf32, <"gpu", 1>, mut>
// CHECK: %[[SRC_BUF:.*]] = mgp.tensor.extract.buffer %[[ARG]]
// CHECK: %[[SRC_SPEC:.*]] = mgp.tensor.extract.tensor_spec %[[ARG]]
// CHECK: %[[CH1:.*]] = grt.chain.create (%[[CHAIN]], %[[CH0]])
// CHECK: %[[BEFORE:.*]] = mgp.device_wait(%[[CH1]], %[[CTX]], %[[GPU1_CTX]], %[[GPU0_CTX]]) device <"gpu", 1>
// CHECK: %[[COPY:.*]] = mgp.buffer.device_to_device %[[BEFORE]], %[[SRC_BUF]], %[[GPU0_CTX]], %[[BUF]], %[[GPU1_CTX]]
// CHECK: mgp.device_wait(%[[COPY]], %[[CTX]], %[[GPU0_CTX]], %[[GPU1_CTX]]) device <"gpu", 0>
// CHECK: mef.output %[[CHAIN]], %[[TENSOR]]
mo.graph @buffer_transfer_d2d(%arg0: !mo.buffer<[2, 2], f32, gpu:0>) -> !mo.buffer<[2, 2], f32, gpu:1>  {
  %ch = mo.chain.create()
  %darg0 = mo.buffer.create : !mo.buffer<[2, 2], f32, gpu:1>

  mo.buffer.transfer(%arg0, %darg0, %ch) : !mo.buffer<[2, 2], f32, gpu:0>, !mo.buffer<[2, 2], f32, gpu:1>

  mo.output %darg0: !mo.buffer<[2, 2], f32, gpu:1>
}
