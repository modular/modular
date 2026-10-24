// RUN: mo-opt %s --mogg-import-packages --mgp-lower-primitives --mogg-sequence-fusion --mogg-outline-async-routines --mogg-add-try-except-blocks --canonicalize --mogg-emit-mojo --mogg-strip-kernel-decls --jit --mogg-lower-compiled-routines --mgp-outline-models-to-callables -o %t.mlir
// RUN: mt --execute %t.mlir | FileCheck %s

// Test it works without fusion.
// RUN: mo-opt %s --mogg-import-packages --mgp-lower-primitives --mogg-outline-async-routines --mogg-add-try-except-blocks --canonicalize --mogg-emit-mojo --mogg-strip-kernel-decls --jit --mogg-lower-compiled-routines --mgp-outline-models-to-callables -o %t_nofused.mlir
// RUN: mt --execute %t_nofused.mlir | FileCheck %s

// CHECK-LABEL:--- Running 'mgp_assert_true':
// CHECK-NEXT: 'mgp_assert_true' returned 1
mgp.model @mgp_assert_true : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
    %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, i1) {
  %ch0 = grt.chain.create ()
  %cond = index.bool.constant true
  %ch = mgp.assert %cond, "ASSERT"
  mef.output %ch, %cond: !mef.chain, i1
}

// CHECK-LABEL:--- Running 'mgp_alloc_cpu':
// CHECK-NEXT: 'mgp_alloc_cpu' returned tensor{{.*}}42{{.*}}
mgp.model @mgp_alloc_cpu : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
    %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<1x?xui8>) {
  %size = index.constant 4
  %buf0 = mgp.buffer.alloc %arg0, %size, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %spec0 = mgp.tensor_spec.create %size : !mgp.tensor_spec<1x?xui8>
  %idx0 = index.constant 42
  %ch2 = mgp.buffer.set_with_index %buf0, %idx0 : !mgp.buffer
  %tensor0 = mgp.tensor.create %ch2, %buf0, %spec0 : !mgp.tensor<1x?xui8>
  mef.output %arg0, %tensor0: !mef.chain, !mgp.tensor<1x?xui8>
}

// CHECK-LABEL:--- Running 'mgp_tensor_spec_create_static':
// CHECK-NEXT: 'mgp_tensor_spec_create_static' returned tensor
// CHECK-NEXT: shape=[2,2]
mgp.model @mgp_tensor_spec_create_static : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>

} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<2x2xui8>) {
  %const = mgp.buffer.constant dense_resource<resource> : !M.aligned_bytes<4, align 128>
  %spec = mgp.tensor_spec.create : !mgp.tensor_spec<2x2xui8>
  %tensor = mgp.tensor.create %ch0, %const, %spec : !mgp.tensor<2x2xui8>
  mef.output %ch0, %tensor : !mef.chain, !mgp.tensor<2x2xui8>
}

// CHECK-LABEL:--- Running 'mgp_tensor_spec_create_dynamic':
// CHECK-NEXT: 'mgp_tensor_spec_create_dynamic' returned tensor
// CHECK-NEXT: shape=[2,2]
mgp.model @mgp_tensor_spec_create_dynamic : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>

} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<2x?xui8>) {
  %size = index.constant 2
  %const = mgp.buffer.constant dense_resource<resource> : !M.aligned_bytes<4, align 128>
  %spec = mgp.tensor_spec.create %size : !mgp.tensor_spec<2x?xui8>
  %tensor = mgp.tensor.create %ch0, %const, %spec : !mgp.tensor<2x?xui8>
  mef.output %ch0, %tensor : !mef.chain, !mgp.tensor<2x?xui8>
}

{-#
  dialect_resources: {
    builtin: {
      resource: "0x80000000fffefdfc"
    }
  }
#-}

// CHECK-LABEL:--- Running 'mgp_set_with_index':
// CHECK{LITERAL}-NEXT:'mgp_set_with_index' returned tensor([[42, 17, 20]], dtype=si32, shape=[3])
mgp.model @mgp_set_with_index : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>

} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<3xsi32>) {
  %idx0 = index.constant 42
  %idx1 = index.constant 17
  %idx2 = index.constant 20
  %size = index.constant 12
  %buf = mgp.buffer.alloc %ch0, %size, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %ch2 = mgp.buffer.set_with_index %buf, %idx0, %idx1, %idx2 : !mgp.buffer
  %spec = mgp.tensor_spec.create : !mgp.tensor_spec<3xsi32>
  %tensor = mgp.tensor.create %ch2, %buf, %spec : !mgp.tensor<3xsi32>
  mef.output %ch0, %tensor : !mef.chain, !mgp.tensor<3xsi32>
}

// CHECK-LABEL: --- Running 'buffer_to_bool':
// CHECK-NEXT: 'buffer_to_bool' returned 1, 0, 1
mgp.model @buffer_to_bool : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>

} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, i1, i1, i1) {
  %buf1 = mgp.buffer.constant #M.dense_array<1> : !M.array<1xui8>
  %buf2 = mgp.buffer.constant #M.dense_array<0> : !M.array<1xui8>
  %buf3 = mgp.buffer.constant #M.dense_array<2> : !M.array<1xui8>
  %b1 = mgp.buffer.to_bool %ch0, %buf1 : !mgp.buffer
  %b2 = mgp.buffer.to_bool %ch0, %buf2 : !mgp.buffer
  %b3 = mgp.buffer.to_bool %ch0, %buf3 : !mgp.buffer
  mef.output %ch0, %b1, %b2, %b3 : !mef.chain, i1, i1, i1
}
