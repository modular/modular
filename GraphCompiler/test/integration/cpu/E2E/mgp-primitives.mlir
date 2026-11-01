// RUN: mo-opt --mogg-import-packages --mgp-lower-primitives --mogg-sequence-fusion --mogg-outline-async-routines --mogg-add-try-except-blocks --canonicalize --mogg-emit-mojo --mogg-strip-kernel-decls --jit --jit-pipeline %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

// CHECK-LABEL: --- Running 'debug_print'
// CHECK-NEXT: [label] This is a debug string
mgp.model @debug_print : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  %ch2 = mgp.runtime.create %ch0, %ctx[0]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %ch1 = mgp.debug.print %ch0, %ctx, "This is a debug string", "label"
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: --- Running 'debug_tensor_print'
// CHECK-NEXT: test = tensor<2x2xui8> [255, 254, 253, 252]
// CHECK-NEXT: sliced = tensor<1xsi16> [-771]
mgp.model @debug_tensor_print : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  %ch2 = mgp.runtime.create %ch0, %ctx[0]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain) {
  %one = index.constant 1
  %two = index.constant 2
  %buffer = mgp.buffer.constant dense_resource<resource> : !M.aligned_bytes<4, align 128>
  %spec = mgp.tensor_spec.create : !mgp.tensor_spec<2x2xui8>
  %ch1 = mgp.debug.tensor.print %ch0, %buffer, %spec, "test" : (!mef.chain, !mgp.buffer, !mgp.tensor_spec<2x2xui8>) -> !mef.chain
  %sliced = mgp.buffer.slice %buffer, %two, %two, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %slicedSpec = mgp.tensor_spec.create %one : !mgp.tensor_spec<?xsi16>
  %ch2 = mgp.debug.tensor.print %ch1, %sliced, %slicedSpec, "sliced" : (!mef.chain, !mgp.buffer, !mgp.tensor_spec<?xsi16>) -> !mef.chain
  mef.output %ch2 : !mef.chain
}

// CHECK-LABEL: --- Running 'tensor_spec_get_dim'
// CHECK: 'tensor_spec_get_dim' returned tensor<1x8xui8> [8, 0, 0, 0, 0, 0, 0, 0]
mgp.model @tensor_spec_get_dim : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  %ch2 = mgp.runtime.create %ch0, %ctx[0]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<1x?xui8>) {
  %size = index.constant 8
  %buf = mgp.buffer.alloc %ch0, %size, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %tensor_spec = mgp.tensor_spec.create %size : !mgp.tensor_spec<1x?xui8>
  %dim1 = mgp.tensor_spec.get_dim[1] %tensor_spec : !mgp.tensor_spec<1x?xui8>
  %ch2 = mgp.buffer.set_with_index %buf, %dim1 : !mgp.buffer
  %tensor0 = mgp.tensor.create %ch2, %buf, %tensor_spec : !mgp.tensor<1x?xui8>
  mef.output %ch0, %tensor0: !mef.chain, !mgp.tensor<1x?xui8>
}

// CHECK-LABEL: --- Running 'cache'
// CHECK-NEXT: tensor<4xui8> [255, 254, 253, 252]
mgp.model @cache : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 3
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<4xui8>) {
  %buf0 = mgp.buffer.constant dense_resource<resource> : !M.aligned_bytes<4, align 128>
  %ch1 = mgp.buffer.cache %ch0, %ctx[2], %buf0 : (!mef.chain, !mgp.context, !mgp.buffer) -> !mef.chain
  %idx = index.constant 2
  %buf1 = mgp.buffer.get_cached %ch1, %ctx[%idx] : !mgp.buffer
  %spec = mgp.tensor_spec.create : !mgp.tensor_spec<4xui8>
  %tensor = mgp.tensor.create %ch1, %buf1, %spec : !mgp.tensor<4xui8>
  mef.output %ch1, %tensor : !mef.chain, !mgp.tensor<4xui8>
}

// CHECK-LABEL: --- Running 'remove_cache'
// CHECK-NEXT: tensor<6xui8> [1, 2, 3, 4, 5, 6]
mgp.model @remove_cache : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 3
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<6xui8>) {
  %buf0 = mgp.buffer.constant dense_resource<resource> : !M.aligned_bytes<4, align 128>
  %buf1 = mgp.buffer.constant #M.dense_array<1, 2, 3, 4, 5, 6> : !M.array<6xui8>
  %idx = index.constant 2
  %ch4 = mgp.buffer.cache %ch0, %ctx[2], %buf1 : (!mef.chain, !mgp.context, !mgp.buffer) -> !mef.chain
  %buf1_fromcache = mgp.buffer.get_cached %ch4, %ctx[%idx] : !mgp.buffer
  %spec = mgp.tensor_spec.create : !mgp.tensor_spec<6xui8>
  %tensor = mgp.tensor.create %ch0, %buf1_fromcache, %spec : !mgp.tensor<6xui8>
  mef.output %ch4, %tensor : !mef.chain, !mgp.tensor<6xui8>
}

// CHECK-LABEL: --- Running 'buffer_get_size'
// CHECK-NEXT: tensor<1xsi64> [4]
mgp.model @buffer_get_size : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<1xsi64>) {
  %size = index.constant 8
  %buf0 = mgp.buffer.constant dense_resource<resource> : !M.aligned_bytes<4, align 128>
  %buf_size = mgp.buffer.get_size %ch0, %buf0 : !mgp.buffer
  %buf = mgp.buffer.alloc %ch0, %size, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %ch2 = mgp.buffer.set_with_index %buf, %buf_size : !mgp.buffer
  %spec = mgp.tensor_spec.create : !mgp.tensor_spec<1xsi64>
  %ten1 = mgp.tensor.create %ch2, %buf, %spec : !mgp.tensor<1xsi64>
  mef.output %ch2, %ten1 : !mef.chain, !mgp.tensor<1xsi64>
}

// CHECK-LABEL: --- Running 'to_tensor_from_tensor':
// CHECK-NEXT: 'to_tensor_from_tensor' returned tensor<1xsi32> [-50462977]
mgp.model @to_tensor_from_tensor : mof {argument_device_indices = [0 : ui32], result_device_indices = [0 : ui32]} devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"cpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 1
  %2 = mgp.runtime.create %0, %1[0]
  mef.output %2, %1, %arg0, %arg1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<1xsi32>) {
  %buffer = mgp.buffer.constant dense_resource<resource> : !M.aligned_bytes<4, align 128>
  %sliceLen = index.constant 4
  %idx0 = index.constant 0
  %3 = mgp.buffer.slice %buffer, %idx0, %sliceLen, %arg3 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %4 = mgp.buffer.to_index %arg0, %3 : !mgp.buffer
  %5 = mgp.tensor_spec.create %4 : !mgp.tensor_spec<?xsi64>
  %6 = mgp.tensor_spec.create  : !mgp.tensor_spec<1xsi32>
  %idx4 = index.constant 4
  %7 = mgp.buffer.alloc %arg0, %idx4, %arg3, align 64 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %8 = mgp.tensor_spec.get_dim[0] %5 : <?xsi64>
  %9 = mgp.buffer.set_with_index %7, %8 : !mgp.buffer
  %10 = mgp.tensor.create %9, %7, %6 : !mgp.tensor<1xsi32>
  mef.output %arg0, %10 : !mef.chain, !mgp.tensor<1xsi32>
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

// CHECK-LABEL: --- Running 'buffer_concat'
// CHECK-NEXT: tensor<10xui8> [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
mgp.model @buffer_concat : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<10xui8>) {
  %buf0 = mgp.buffer.constant #M.dense_array<1, 2, 3, 4> : !M.array<4xui8>
  %buf1 = mgp.buffer.constant #M.dense_array<5, 6, 7, 8, 9, 10> : !M.array<6xui8>
  %size = index.constant 10
  %out_buf = mgp.buffer.alloc %ch0, %size, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %ch2 = mgp.buffer.concat(%ch0, %out_buf, %buf0, %buf1, %cpu) : (!mef.chain, !mgp.buffer, !mgp.buffer, !mgp.buffer, !mgp.device_context<<"cpu", 0>>) -> !mef.chain
  %spec = mgp.tensor_spec.create : !mgp.tensor_spec<10xui8>
  %tensor = mgp.tensor.create %ch2, %out_buf, %spec : !mgp.tensor<10xui8>
  mef.output %ch0, %tensor  : !mef.chain, !mgp.tensor<10xui8>
}

mgp.model @tensor_create_extract : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<1x?xui8>) {
  %size = index.constant 4
  %const0 = mgp.buffer.constant dense_resource<resource> : !M.aligned_bytes<4, align 128>
  %spec0 = mgp.tensor_spec.create %size : !mgp.tensor_spec<1x?xui8>
  %tensor0 = mgp.tensor.create %ch0, %const0, %spec0 : !mgp.tensor<1x?xui8>
  %buf1 = mgp.tensor.extract.buffer %tensor0 : <1x?xui8>
  %spec1 = mgp.tensor.extract.tensor_spec %tensor0 : <1x?xui8>
  %tensor1 = mgp.tensor.create %ch0, %buf1, %spec1 : !mgp.tensor<1x?xui8>
  mef.output %ch0, %tensor1 : !mef.chain, !mgp.tensor<1x?xui8>
}

{-#
  dialect_resources: {
    builtin: {
      resource: "0x80000000fffefdfc"
    }
  }
#-}
