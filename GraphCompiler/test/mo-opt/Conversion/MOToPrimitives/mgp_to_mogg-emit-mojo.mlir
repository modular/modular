// RUN: mo-opt %s --allow-unregistered-dialect --mgp-lower-primitives --split-input-file | FileCheck %s

// Check we can lower a given MGP primitive to a mogg kernel with emit-mojo enabled.
// In emit-mojo mode, constants are lowered to buffer constants and ops like
// index.add and mgp.buffer.slice are kept inline (not converted to mogg.call).

mgp.model @basic : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %chain = grt.chain.create ()
  %context = mgp.context.create 2
  %out_chain = mgp.runtime.create %chain, %context[1]

  mef.output %out_chain, %context, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %ch0 = grt.chain.create ()

  %lhs = index.constant 42
  %rhs = index.constant -20
  %res1 = index.add %lhs, %rhs
  %ch1 = "test.sink"(%res1) : (index) -> !mef.chain
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @basic

// In emit-mojo mode, index.constant is lowered to a buffer constant loaded at runtime
// CHECK: mogg.routine.async () : () -> index {
// CHECK: mgp.buffer.constant
// CHECK: mgp.buffer.to_index
// CHECK: mogg.async.pack
// CHECK: mogg.output

// CHECK: mogg.routine.async () : () -> index {
// CHECK: mgp.buffer.constant
// CHECK: mgp.buffer.to_index
// CHECK: mogg.async.pack
// CHECK: mogg.output

// index.add stays inline with async unpack/pack wrapping
// CHECK: mogg.routine.async
// CHECK: mogg.async.unpack
// CHECK: mogg.async.unpack
// CHECK: index.add
// CHECK: mogg.async.pack
// CHECK: mogg.output

// -----

mgp.model @buffer_borrow : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %arg0: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %arg0 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<?xsi32>) {
  %idx0 = index.constant 12
  %idx1 = index.constant 12
  %idx2 = index.constant 3
  %buf = mgp.buffer.constant #M.dense_array<1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0> : !M.array<24xui8>
  %sliced = mgp.buffer.slice %buf, %idx0, %idx1, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %spec = mgp.tensor_spec.create %idx2 : !mgp.tensor_spec<?xsi32>
  %tensor = mgp.tensor.create %ch0, %sliced, %spec : !mgp.tensor<?xsi32>
  mef.output %ch0, %tensor : !mef.chain, !mgp.tensor<?xsi32>
}


// CHECK-LABEL: mgp.model @buffer_borrow

// In emit-mojo mode, mgp.buffer.slice stays as the op (not converted to mogg.call)
// and is wrapped with async unpack/pack operations. Every buffer/tensor value
// carries its own storage handle, so the output is packed with the uniform
// owned pack (no borrow/lifetime-extension plumbing).
// CHECK: mogg.routine.async
// CHECK: mogg.async.unpack
// CHECK: mogg.async.unpack
// CHECK: mogg.async.unpack
// CHECK: mgp.buffer.slice
// CHECK: mogg.async.pack
// CHECK: mogg.output
