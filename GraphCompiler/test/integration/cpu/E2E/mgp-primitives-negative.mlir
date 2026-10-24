// RUN: mo-opt -gc-pipeline %s -o %t.mlir
// RUN: not mt --execute --model-inputs=ones --on-failure=continue --result-output-style=full %t.mlir | FileCheck %s

// CHECK-LABEL: --- Running 'assert_false'
// CHECK: <error:
// CHECK: the buffer must not contain 0>
mgp.model @assert_false : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %buf = mgp.buffer.constant #M.dense_array<0> : !M.array<1xui8>
  %0 = mgp.buffer.to_bool %ch0, %buf : !mgp.buffer
  %ch2 = mgp.assert %0, "the buffer must not contain 0"
  mef.output %ch2 : !mef.chain
}

// CHECK-LABEL: --- Running 'buffer_slice_out_of_range'
// CHECK: <error:
// CHECK: offset, len and elem_size are out of range
mgp.model @buffer_slice_out_of_range : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %offset = index.constant 0
  // The 8-byte slice does not fit within the 4-byte parent buffer.
  %size = index.constant 8
  %buf = mgp.buffer.constant #M.dense_array<1, 2, 3, 4> : !M.array<4xui8>
  %sliced = mgp.buffer.slice %buf, %offset, %size, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %spec = mgp.tensor_spec.create : !mgp.tensor_spec<8xui8>
  %ch2 = mgp.debug.tensor.print %ch0, %sliced, %spec, "sliced" : (!mef.chain, !mgp.buffer, !mgp.tensor_spec<8xui8>) -> !mef.chain
  mef.output %ch2 : !mef.chain
}
