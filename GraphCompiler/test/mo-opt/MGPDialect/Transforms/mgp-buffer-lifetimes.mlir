// RUN: mo-opt %s --allow-unregistered-dialect --mgp-buffer-lifetimes  | FileCheck %s

mgp.model @serial : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  mef.output %0, %1, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>

} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %inTensor: !mgp.tensor<?x?xf32>) -> (!mef.chain, !mgp.tensor<?x?xf32>) {
  // CHECK: %[[INBUFFER:.*]] = mgp.tensor.extract.buffer
  %inBuffer = mgp.tensor.extract.buffer %inTensor : <?x?xf32>
  %inTensorSpec = mgp.tensor.extract.tensor_spec %inTensor : <?x?xf32>
  %inDims0 = mgp.tensor_spec.get_dim[0] %inTensorSpec : !mgp.tensor_spec<?x?xf32>
  %inDims1 = mgp.tensor_spec.get_dim[1] %inTensorSpec : !mgp.tensor_spec<?x?xf32>

  %allocSize = index.mul %inDims0, %inDims1
  // CHECK: %[[ALLOC0:.*]] = mgp.buffer.alloc {{.*}}
  // CHECK-NEXT: %[[ALLOC1:.*]] = mgp.buffer.alloc {{.*}}
  // CHECK-NEXT: %[[ALLOC2:.*]] = mgp.buffer.alloc {{.*}}

  %alloc0 = mgp.buffer.alloc %arg0, %allocSize, %cpu, align 16 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %alloc1 = mgp.buffer.alloc %arg0, %allocSize, %cpu, align 16 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %alloc2 = mgp.buffer.alloc %arg0, %allocSize, %cpu, align 16 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>

  // CHECK-DAG: %[[CH0:.*]] = "use_buffers"({{.*}}, %[[INBUFFER]], %[[ALLOC0]])
  %outChain_0 = "use_buffers"(%arg0, %inBuffer, %alloc0) : (!mef.chain, !mgp.buffer, !mgp.buffer) -> !mef.chain
  // CHECK-DAG: %[[CH1:.*]] = "use_buffers"(%[[CH0]], %[[ALLOC0]], %[[ALLOC1]])
  %outChain_1 = "use_buffers"(%outChain_0, %alloc0, %alloc1) : (!mef.chain, !mgp.buffer, !mgp.buffer) -> !mef.chain
  // CHECK-DAG: %[[FREE0:.*]] = mgp.buffer.free %[[CH1]], %[[ALLOC0]] : !mgp.buffer
  // CHECK-DAG: %[[CH2:.*]] = "use_buffers"(%[[FREE0]], %[[ALLOC1]], %[[ALLOC2]])
  %outChain_2 = "use_buffers"(%outChain_1, %alloc1, %alloc2) : (!mef.chain, !mgp.buffer, !mgp.buffer) -> !mef.chain
  // CHECK-DAG: %[[FREE1:.*]] = mgp.buffer.free %[[CH2]], %[[ALLOC1]] : !mgp.buffer
  // CHECK-NEXT: %[[OUTTENSOR:.*]] = mgp.tensor.create %[[FREE1]], {{.*}}
  %outTensor = mgp.tensor.create %outChain_2, %alloc2, %inTensorSpec : !mgp.tensor<?x?xf32>
  mef.output %arg0, %outTensor : !mef.chain, !mgp.tensor<?x?xf32>
}

// -----

mgp.model @parallel : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  mef.output %0, %1, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>

} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %inTensor: !mgp.tensor<?x?xf32>) -> (!mef.chain, !mgp.tensor<?x?xf32>) {
  // CHECK: %[[INBUFFER:.*]] = mgp.tensor.extract.buffer
  %inBuffer = mgp.tensor.extract.buffer %inTensor : <?x?xf32>
  %inTensorSpec = mgp.tensor.extract.tensor_spec %inTensor : <?x?xf32>
  %inDims0 = mgp.tensor_spec.get_dim[0] %inTensorSpec : !mgp.tensor_spec<?x?xf32>
  %inDims1 = mgp.tensor_spec.get_dim[1] %inTensorSpec : !mgp.tensor_spec<?x?xf32>

  %allocSize = index.mul %inDims0, %inDims1
  // CHECK: %[[ALLOC0:.*]] = mgp.buffer.alloc {{.*}}
  // CHECK-NEXT: %[[ALLOC1:.*]] = mgp.buffer.alloc {{.*}}
  // CHECK-NEXT: %[[ALLOC2:.*]] = mgp.buffer.alloc {{.*}}

  %alloc0 = mgp.buffer.alloc %arg0, %allocSize, %cpu, align 16 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %alloc1 = mgp.buffer.alloc %arg0, %allocSize, %cpu, align 16 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %alloc2 = mgp.buffer.alloc %arg0, %allocSize, %cpu, align 16 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>

  // CHECK-DAG: %[[CH0:.*]] = "use_buffers"({{.*}}, %[[INBUFFER]], %[[ALLOC0]])
  %outChain_0 = "use_buffers"(%arg0, %inBuffer, %alloc0) : (!mef.chain, !mgp.buffer, !mgp.buffer) -> !mef.chain
  // CHECK-DAG: %[[CH1:.*]] = "use_buffers"({{.*}}, %[[INBUFFER]], %[[ALLOC1]])
  %outChain_1 = "use_buffers"(%outChain_0, %inBuffer, %alloc1) : (!mef.chain, !mgp.buffer, !mgp.buffer) -> !mef.chain
  %merged = grt.chain.create (%outChain_0, %outChain_1)
  // CHECK-DAG: %[[MERGED:.*]] = grt.chain.create (%[[CH0]], %[[CH1]])

  // CHECK-DAG: %[[CH2:.*]] = "use_buffers"({{.*}}, %[[ALLOC0]], %[[ALLOC1]], %[[ALLOC2]])
  %outChain_2 = "use_buffers"(%merged, %alloc0, %alloc1, %alloc2) : (!mef.chain, !mgp.buffer, !mgp.buffer, !mgp.buffer) -> !mef.chain
  // CHECK-DAG: %[[FREE1:.*]] = mgp.buffer.free %[[CH2]], %[[ALLOC1]] : !mgp.buffer
  // CHECK-DAG: %[[FREE0:.*]] = mgp.buffer.free %[[FREE1]], %[[ALLOC0]] : !mgp.buffer
  // CHECK-NEXT: %[[OUTTENSOR:.*]] = mgp.tensor.create %[[FREE0]], {{.*}}
  %outTensor = mgp.tensor.create %outChain_2, %alloc2, %inTensorSpec : !mgp.tensor<?x?xf32>
  mef.output %arg0, %outTensor : !mef.chain, !mgp.tensor<?x?xf32>
}

// -----

mgp.model @tensor_create_aliased : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  mef.output %0, %1, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>

} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %inTensor: !mgp.tensor<?x?xf32>) -> (!mef.chain, !mgp.tensor<?x?xf32>) {
  // CHECK: %[[INBUFFER:.*]] = mgp.tensor.extract.buffer
  %inBuffer = mgp.tensor.extract.buffer %inTensor : <?x?xf32>
  %inTensorSpec = mgp.tensor.extract.tensor_spec %inTensor : <?x?xf32>
  %inDims0 = mgp.tensor_spec.get_dim[0] %inTensorSpec : !mgp.tensor_spec<?x?xf32>
  %inDims1 = mgp.tensor_spec.get_dim[1] %inTensorSpec : !mgp.tensor_spec<?x?xf32>

  %allocSize = index.mul %inDims0, %inDims1
  // CHECK: %[[ALLOC0:.*]] = mgp.buffer.alloc {{.*}}
  // CHECK-NEXT: %[[ALLOC1:.*]] = mgp.buffer.alloc {{.*}}
  // CHECK-NEXT: %[[ALLOC2:.*]] = mgp.buffer.alloc {{.*}}

  %alloc0 = mgp.buffer.alloc %arg0, %allocSize, %cpu, align 16 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %alloc1 = mgp.buffer.alloc %arg0, %allocSize, %cpu, align 16 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %alloc2 = mgp.buffer.alloc %arg0, %allocSize, %cpu, align 16 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>

  // CHECK-DAG: %[[CH0:.*]] = "use_buffers"({{.*}}, %[[INBUFFER]], %[[ALLOC0]])
  %outChain_0 = "use_buffers"(%arg0, %inBuffer, %alloc0) : (!mef.chain, !mgp.buffer, !mgp.buffer) -> !mef.chain

  %alias0_tensor = mgp.tensor.create %outChain_0, %alloc0, %inTensorSpec : !mgp.tensor<?x?xf32>
  // CHECK-DAG: %[[ALIAS0:.*]] = mgp.tensor.extract.buffer {{.*}}
  %alias0 = mgp.tensor.extract.buffer %alias0_tensor : <?x?xf32>

  // CHECK-DAG: %[[CH1:.*]] = "use_buffers"({{.*}}, %[[ALLOC0]], %[[ALLOC1]])
  %outChain_1 = "use_buffers"(%outChain_0, %alloc0, %alloc1) : (!mef.chain, !mgp.buffer, !mgp.buffer) -> !mef.chain

  // CHECK-DAG: %[[CH2:.*]] = "use_buffers"({{.*}}, %[[ALLOC1]], %[[ALLOC2]])
  %outChain_2 = "use_buffers"(%outChain_1, %alloc1, %alloc2) : (!mef.chain, !mgp.buffer, !mgp.buffer) -> !mef.chain

  // CHECK-DAG: mgp.buffer.free %[[CH2]], %[[ALLOC1]] : !mgp.buffer
  // CHECK-DAG: %[[CH3:.*]] = "use_buffers"({{.*}}, %[[ALIAS0]], %[[ALLOC2]])
  %outChain_3 = "use_buffers"(%outChain_2, %alias0, %alloc2) : (!mef.chain, !mgp.buffer, !mgp.buffer) -> !mef.chain
  // CHECK-DAG: mgp.buffer.free %[[CH3]], %[[ALLOC0]] : !mgp.buffer

  %outTensor = mgp.tensor.create %outChain_3, %alloc2, %inTensorSpec : !mgp.tensor<?x?xf32>
  mef.output %arg0, %outTensor : !mef.chain, !mgp.tensor<?x?xf32>
}

// -----

mgp.model @nested_block : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  mef.output %0, %1, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>

} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %inTensor: !mgp.tensor<?x?xf32>) -> (!mef.chain, !mgp.tensor<?x?xf32>) {
  // CHECK: %[[INBUFFER:.*]] = mgp.tensor.extract.buffer
  %inBuffer = mgp.tensor.extract.buffer %inTensor : <?x?xf32>
  %inTensorSpec = mgp.tensor.extract.tensor_spec %inTensor : <?x?xf32>

  %buf1 = mgp.buffer.constant #M.dense_array<1> : !M.array<1xui8>
  %b1 = mgp.buffer.to_bool %arg0, %buf1 : !mgp.buffer

  %ifchain, %outbuf = grt.if %b1 (%arg0: !mef.chain, %arg1: !mgp.context, %inBuffer: !mgp.buffer, %inTensorSpec: !mgp.tensor_spec<?x?xf32>, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.buffer) {
    %inDims0 = mgp.tensor_spec.get_dim[0] %inTensorSpec : !mgp.tensor_spec<?x?xf32>
    %inDims1 = mgp.tensor_spec.get_dim[1] %inTensorSpec : !mgp.tensor_spec<?x?xf32>

    %allocSize = index.mul %inDims0, %inDims1

    // CHECK: %[[ALLOC0:.*]] = mgp.buffer.alloc {{.*}}
    // CHECK-NEXT: %[[ALLOC1:.*]] = mgp.buffer.alloc {{.*}}
    // CHECK-NEXT: %[[ALLOC2:.*]] = mgp.buffer.alloc {{.*}}
    %alloc0 = mgp.buffer.alloc %arg0, %allocSize, %cpu, align 16 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
    %alloc1 = mgp.buffer.alloc %arg0, %allocSize, %cpu, align 16 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
    %alloc2 = mgp.buffer.alloc %arg0, %allocSize, %cpu, align 16 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>

    // CHECK-DAG: %[[CH0:.*]] = "use_buffers"({{.*}}, %[[INBUFFER]], %[[ALLOC0]])
    %outChain_0 = "use_buffers"(%arg0, %inBuffer, %alloc0) : (!mef.chain, !mgp.buffer, !mgp.buffer) -> !mef.chain

    // CHECK-DAG: %[[CH1:.*]] = "use_buffers"({{.*}}, %[[ALLOC0]], %[[ALLOC1]])
    %outChain_1 = "use_buffers"(%outChain_0, %alloc0, %alloc1) : (!mef.chain, !mgp.buffer, !mgp.buffer) -> !mef.chain

    // CHECK-DAG: mgp.buffer.free %[[CH1]], %[[ALLOC0]] : !mgp.buffer
    // CHECK-DAG: %[[CH2:.*]] = "use_buffers"({{.*}}, %[[ALLOC1]], %[[ALLOC2]])
    %outChain_2 = "use_buffers"(%outChain_1, %alloc1, %alloc2) : (!mef.chain, !mgp.buffer, !mgp.buffer) -> !mef.chain

    // CHECK-DAG: mgp.buffer.free %[[CH2]], %[[ALLOC1]] : !mgp.buffer
    mef.output %outChain_2, %alloc2 : !mef.chain, !mgp.buffer
  } else {
    mef.output %arg0, %inBuffer : !mef.chain, !mgp.buffer
  }
  %outTensor = mgp.tensor.create %ifchain, %outbuf, %inTensorSpec : !mgp.tensor<?x?xf32>
  mef.output %arg0, %outTensor : !mef.chain, !mgp.tensor<?x?xf32>
}
