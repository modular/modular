// Just check we still parse it.
// RUN: mo-opt %s --allow-unregistered-dialect

mgp.model @upsideDown : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %2 = mgp.runtime.create %0, %1[1]
  %1 = mgp.context.create 1
  %0 = grt.chain.create ()
  mef.output %2, %1, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %arg2: !mgp.tensor<8xf32>) -> (!mef.chain, !mgp.tensor<8xf32>) {
  %spec = mgp.tensor_spec.create : !mgp.tensor_spec<8xf32>
  %size = index.constant 32
  %buf = mgp.buffer.alloc %ch0, %size, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %3 = mgp.tensor.create %ch0, %buf, %spec : !mgp.tensor<8xf32>
  %2 = "mop.kernels.add"(%ch0, %bufferResult, %tensorSpecResult, %bufferResult_0, %tensorSpecResult_1, %0, %1) : (!mef.chain, !mgp.buffer, !mgp.tensor_spec<8xf32>, !mgp.buffer, !mgp.tensor_spec<8xf32>, !mgp.buffer, !mgp.tensor_spec<8xf32>) -> !mef.chain
  %1 = mgp.tensor_spec.create  : !mgp.tensor_spec<8xf32>
  %0 = mgp.buffer.alloc %ch0, %size, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %bufferResult = mgp.tensor.extract.buffer %arg2 : <8xf32>
  %tensorSpecResult = mgp.tensor.extract.tensor_spec %arg2 : <8xf32>
  %bufferResult_0 = mgp.tensor.extract.buffer %arg2 : <8xf32>
  %tensorSpecResult_1 = mgp.tensor.extract.tensor_spec %arg2 : <8xf32>
  mef.output %ch0, %3 : !mef.chain, !mgp.tensor<8xf32>
}

// Also check building a model op with a weights registry.
mgp.model @external_weights : mof
  {argument_names = ["input0"], result_names = ["output0"]}
  devices [#M.device_ref<"cpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %cpu_ctx: !mgp.device_context<<"cpu", 0>>) {
    %chain = grt.chain.create ()
    %mgp_ctx = mgp.context.create 1
    mef.output %chain, %mgp_ctx, %weights , %cpu_ctx:
      !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%chain: !mef.chain, %mgp_ctx: !mgp.context,
           %weights: !mgp.opaque<"weights_registry">,
           %cpu_ctx: !mgp.device_context<<"cpu", 0>>,
           %input: !mgp.tensor<4xf32>) -> (!mef.chain, !mgp.tensor<4xf32>) {
  %spec = mgp.tensor.extract.tensor_spec %input : <4xf32>
  %input_buf = mgp.tensor.extract.buffer %input : <4xf32>
  %output = mgp.tensor.create %chain, %input_buf, %spec : !mgp.tensor<4xf32>
  mef.output %chain, %output : !mef.chain, !mgp.tensor<4xf32>
}
