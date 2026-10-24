// RUN: mo-opt --mogg-import-packages --mgp-lower-primitives --mogg-sequence-fusion --mogg-outline-async-routines --mogg-add-try-except-blocks --canonicalize --mogg-emit-mojo --mogg-strip-kernel-decls --jit --mogg-lower-compiled-routines --mgp-outline-models-to-callables %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

//
// mgp.runtime.create is implicitly tested in all the following.
//

//
// mgp.module.load cannot be tested without a dependency on GraphCompiler.
//

//
// mgp.tensor.{create, extract.buffer, extract.tensor_spec}
//

// CHECK-LABEL: --- Running 'tensor_create_extract'
// CHECK-NEXT: tensor<1x4xui8> [255, 254, 253, 252]
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

//
// mgp.debug.print
//

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


//
// mgp.debug.tensor.print
//

// CHECK-LABEL: --- Running 'debug_tensor_print'
// CHECK-NEXT: test = tensor<2x2xui8> [255, 254, 253, 252]
mgp.model @debug_tensor_print : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain) {
  %const = mgp.buffer.constant dense_resource<resource> : !M.aligned_bytes<4, align 128>
  %spec = mgp.tensor_spec.create : !mgp.tensor_spec<2x2xui8>
  %ch1 = mgp.debug.tensor.print %ch0, %const, %spec, "test" : (!mef.chain, !mgp.buffer, !mgp.tensor_spec<2x2xui8>) -> !mef.chain
  mef.output %ch1 : !mef.chain
}


{-#
  dialect_resources: {
    builtin: {
      resource: "0x80000000fffefdfc"
    }
  }
#-}
