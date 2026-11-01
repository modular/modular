// UNSUPPORTED: ubsan

// RUN: mo-opt --mogg-import-packages --mgp-lower-primitives --mogg-sequence-fusion --mogg-outline-async-routines --mogg-add-try-except-blocks --canonicalize --mogg-emit-mojo --mogg-strip-kernel-decls --jit --mogg-lower-compiled-routines --mgp-outline-models-to-callables %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

//
// mgp.runtime.create is implicitly tested in all the following.
//

//
// mgp.module.load cannot be tested without a dependency on GraphCompiler.
//

//
// mgp.buffer.constant
//

// CHECK-LABEL: --- Running 'constant'
// CHECK-NEXT: tensor<1x4xui8> [255, 254, 253, 252]
// CHECK-NEXT: , tensor<0xui8> []
mgp.model @constant : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<1x4xui8>, !mgp.tensor<0xui8>) {
  %buf0 = mgp.buffer.constant dense_resource<resource> : !M.aligned_bytes<4, align 128>
  %spec0 = mgp.tensor_spec.create : !mgp.tensor_spec<1x4xui8>
  %tensor0 = mgp.tensor.create %ch0, %buf0, %spec0 : !mgp.tensor<1x4xui8>
  %buf1 = mgp.buffer.constant #M.aligned_bytes<"", align 128>
  %spec1 = mgp.tensor_spec.create : !mgp.tensor_spec<0xui8>
  %tensor1 = mgp.tensor.create %ch0, %buf1, %spec1 : !mgp.tensor<0xui8>
  mef.output %ch0, %tensor0, %tensor1 : !mef.chain, !mgp.tensor<1x4xui8>, !mgp.tensor<0xui8>
}

//
// mgp.buffer.alloc
//

// CHECK-LABEL: --- Running 'alloc'
// CHECK-NEXT: tensor<1x4xui8> [{{.*}}]
// CHECK-NEXT: , tensor<1x4xui8> [{{.*}}]
mgp.model @alloc : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 3
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<1x4xui8>, !mgp.tensor<1x?xui8>) {
  %size = index.constant 4
  %buf0 = mgp.buffer.alloc %ch0, %size, %cpu: !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %spec0 = mgp.tensor_spec.create : !mgp.tensor_spec<1x4xui8>
  %tensor0 = mgp.tensor.create %ch0, %buf0, %spec0 : !mgp.tensor<1x4xui8>
  %buf1 = mgp.buffer.alloc %ch0, %size, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %spec1 = mgp.tensor_spec.create %size : !mgp.tensor_spec<1x?xui8>
  %tensor1 = mgp.tensor.create %ch0, %buf1, %spec1 : !mgp.tensor<1x?xui8>
  mef.output %ch0, %tensor0, %tensor1 : !mef.chain, !mgp.tensor<1x4xui8>, !mgp.tensor<1x?xui8>
}

//
// mgp.buffer.{cache, get_cached}
//

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

//
// mgp.buffer.{cache, get_cached} over a `dense_array` constant, rather than the
// `dense_resource` one above.
//

// CHECK-LABEL: --- Running 'cache_dense_array'
// CHECK-NEXT: tensor<6xui8> [1, 2, 3, 4, 5, 6]
mgp.model @cache_dense_array : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 3
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<6xui8>) {
  %buf1 = mgp.buffer.constant #M.dense_array<1, 2, 3, 4, 5, 6> : !M.array<6xui8>
  %idx = index.constant 2
  %ch4 = mgp.buffer.cache %ch0, %ctx[2], %buf1 : (!mef.chain, !mgp.context, !mgp.buffer) -> !mef.chain
  %buf1_fromcache = mgp.buffer.get_cached %ch4, %ctx[%idx] : !mgp.buffer
  %spec = mgp.tensor_spec.create : !mgp.tensor_spec<6xui8>
  %tensor = mgp.tensor.create %ch4, %buf1_fromcache, %spec : !mgp.tensor<6xui8>
  mef.output %ch4, %tensor : !mef.chain, !mgp.tensor<6xui8>
}

//
// mgp.buffer.{host_to_device, device_to_host} are not supported for CPU.
//

//
// mgp.buffer.device_to_device
//

// CHECK-LABEL: --- Running 'device_to_device'
// CHECK-NEXT: tensor<4xui8> [255, 254, 253, 252]
mgp.model @device_to_device : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<4xui8>) {
  %buf0 = mgp.buffer.constant dense_resource<resource> : !M.aligned_bytes<4, align 128>
  %size = index.constant 4
  %buf1 = mgp.buffer.alloc %ch0, %size, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %ch2 = mgp.buffer.device_to_device %ch0, %buf0, %cpu, %buf1, %cpu : !mgp.buffer -> !mgp.buffer, !mgp.device_context<<"cpu", 0>>, !mgp.device_context<<"cpu", 0>>
  %spec = mgp.tensor_spec.create : !mgp.tensor_spec<4xui8>
  %tensor = mgp.tensor.create %ch2, %buf1, %spec : !mgp.tensor<4xui8>
  mef.output %ch2, %tensor : !mef.chain, !mgp.tensor<4xui8>
}

//
// mgp.buffer.get_size
//

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

//
// mgp.buffer.concat
//

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

//
// mgp.buffer.to_bool
//

// CHECK-LABEL: --- Running 'buffer_to_bool'
// CHECK-NEXT: tensor<1xui32> [42]
// CHECK-NEXT: , tensor<1xui32> [42]
mgp.model @buffer_to_bool : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<1xui32>, !mgp.tensor<1xui32>) {
  %buf1 = mgp.buffer.constant #M.dense_array<1> : !M.array<1xui8>
  %b1 = mgp.buffer.to_bool %ch0, %buf1 : !mgp.buffer
  %out1 = grt.if %b1 (%ch0: !mef.chain, %ctx: !mgp.context) -> (!mgp.buffer) {
    %out1 = mgp.buffer.constant #M.dense_array<42, 0, 0, 0> : !M.array<4xui8>
    mef.output %out1 : !mgp.buffer
  } else {
    %out1 = mgp.buffer.constant #M.dense_array<0, 0, 0, 0> : !M.array<4xui8>
    mef.output %out1 : !mgp.buffer
  }
  %spec = mgp.tensor_spec.create : !mgp.tensor_spec<1xui32>
  %ten1 = mgp.tensor.create %ch0, %out1, %spec : !mgp.tensor<1xui32>
  %buf2 = mgp.buffer.constant #M.dense_array<0> : !M.array<1xui8>
  %b2 = mgp.buffer.to_bool %ch0, %buf2 : !mgp.buffer
  %out2 = grt.if %b2 (%ch0: !mef.chain, %ctx: !mgp.context) -> (!mgp.buffer) {
    %out2 = mgp.buffer.constant #M.dense_array<0, 0, 0, 0> : !M.array<4xui8>
    mef.output %out2 : !mgp.buffer
  } else {
    %out2 = mgp.buffer.constant #M.dense_array<42, 0, 0, 0> : !M.array<4xui8>
    mef.output %out2 : !mgp.buffer
  }
  %ten2 = mgp.tensor.create %ch0, %out2, %spec : !mgp.tensor<1xui32>
  mef.output %ch0, %ten1, %ten2 : !mef.chain, !mgp.tensor<1xui32>, !mgp.tensor<1xui32>
}

//
// mgp.buffer.set_with_index
//

// CHECK-LABEL: --- Running 'buffer_set_with_index_si32'
// CHECK-NEXT: tensor<3xsi32> [42, 17, 20]
mgp.model @buffer_set_with_index_si32 : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
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

// CHECK-LABEL: --- Running 'buffer_set_with_index_si64'
// CHECK-NEXT: tensor<2xsi64> [29, 13]
mgp.model @buffer_set_with_index_si64 : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<2xsi64>) {
  %idx0 = index.constant 29
  %idx1 = index.constant 13
  %size = index.constant 16
  %buf = mgp.buffer.alloc %ch0, %size, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %ch2 = mgp.buffer.set_with_index %buf, %idx0, %idx1 : !mgp.buffer
  %spec = mgp.tensor_spec.create : !mgp.tensor_spec<2xsi64>
  %tensor = mgp.tensor.create %ch2, %buf, %spec : !mgp.tensor<2xsi64>
  mef.output %ch0, %tensor : !mef.chain, !mgp.tensor<2xsi64>
}

// CHECK-LABEL: --- Running 'buffer_slice'
// CHECK-NEXT: tensor<3xsi32> [4, 5, 6]
mgp.model @buffer_slice : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %arg0: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %arg0 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<3xsi32>) {
  %idx0 = index.constant 12
  %idx1 = index.constant 12
  %buf = mgp.buffer.constant #M.dense_array<1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0> : !M.array<24xui8>
  %sliced = mgp.buffer.slice %buf, %idx0, %idx1, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %spec = mgp.tensor_spec.create : !mgp.tensor_spec<3xsi32>
  %tensor = mgp.tensor.create %ch0, %sliced, %spec : !mgp.tensor<3xsi32>
  mef.output %ch0, %tensor : !mef.chain, !mgp.tensor<3xsi32>
}


// CHECK-LABEL: --- Running 'buffer_cache_slice'
// CHECK-NEXT: tensor<3xsi32> [4, 5, 6]
mgp.model @buffer_cache_slice : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %arg0: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %arg0 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<3xsi32>) {
  %idx0 = index.constant 12
  %idx1 = index.constant 12
  %buf1 = mgp.buffer.constant #M.dense_array<1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0> : !M.array<24xui8>
  %ch1 = mgp.buffer.cache %ch0, %ctx[0], %buf1 : (!mef.chain, !mgp.context, !mgp.buffer) -> !mef.chain
  %idx = index.constant 0
  %buf = mgp.buffer.get_cached %ch1, %ctx[%idx] : !mgp.buffer
  %sliced = mgp.buffer.slice %buf, %idx0, %idx1, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %spec = mgp.tensor_spec.create : !mgp.tensor_spec<3xsi32>
  %tensor = mgp.tensor.create %ch0, %sliced, %spec : !mgp.tensor<3xsi32>
  mef.output %ch0, %tensor : !mef.chain, !mgp.tensor<3xsi32>
}


{-#
  dialect_resources: {
    builtin: {
      resource: "0x80000000fffefdfc"
    }
  }
#-}
