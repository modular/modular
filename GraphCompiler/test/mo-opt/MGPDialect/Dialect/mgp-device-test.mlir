// RUN: mo-opt --allow-unregistered-dialect --canonicalize --cse %s | mo-opt --allow-unregistered-dialect | FileCheck %s

//
// mgp.runtime.create
//

// CHECK-LABEL: @runtime_create_host
func.func @runtime_create_host(%ctx : !mgp.context, %ch0 : !mef.chain) -> !mef.chain {
  // CHECK: %{{.*}} = mgp.runtime.create %{{.*}}, %{{.*}}[1]
  %ch1 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch1 : !mef.chain
}

//
// mgp.module.load
//

// CHECK-LABEL: @module_load_dense_attribute
func.func @module_load_dense_attribute(%ctx : !mgp.context, %ch0 : !mef.chain) -> !mef.chain {
  // CHECK: %{{.*}} = mgp.module.load %{{.*}}, %{{.*}}[{{.*}}], dense_resource<resource> : !M.aligned_bytes<4, align 64>, ["f1", "f2"],  ["g1", "g2"] device <"gpu", 3>
  %ch1 = mgp.module.load %ch0, %ctx[1][1], dense_resource<resource> : !M.aligned_bytes<4, align 64>, ["f1", "f2"], ["g1", "g2"] device <"gpu", 3>
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: @module_load_multiline
func.func @module_load_multiline(%ctx : !mgp.context, %ch0 : !mef.chain) -> !mef.chain {
  // CHECK: %{{.*}} = mgp.module.load %{{.*}}, %{{.*}}[{{.*}}], #M<multiline["a", "b", "c"]>, ["f"], [] device <"gpu", 3>
  %ch1 = mgp.module.load %ch0, %ctx[1][1], #M<multiline["a", "b", "c"]>, ["f"], [] device <"gpu", 3>
  mef.output %ch1 : !mef.chain
}

//
// mgp.buffer.constant.external
//
// CHECK-LABEL: @buffer_constant_external
func.func @buffer_constant_external(%ch: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">) -> (!mgp.buffer, !mgp.buffer) {
  // CHECK: %{{.*}} = mgp.string.create "foo"
  // CHECK: %{{.*}} = mgp.buffer.constant.external %{{.*}},  %{{.*}}[%{{.*}}, 16384, align 16] : !mgp.opaque<"weights_registry">, !mgp.opaque<"mojo_string"> -> !mgp.buffer
  %name1 = mgp.string.create "foo" : !mgp.opaque<"mojo_string">
  %buf1 = mgp.buffer.constant.external %ch, %weights[%name1, 16384, align 16] : !mgp.opaque<"weights_registry">, !mgp.opaque<"mojo_string"> -> !mgp.buffer
  // CHECK: %{{.*}} = mgp.string.create "bar"
  // CHECK: %{{.*}} = mgp.buffer.constant.external %{{.*}},  %{{.*}}[%{{.*}}, 16384, align 16] : !mgp.opaque<"weights_registry">, !mgp.opaque<"mojo_string"> -> !mgp.buffer
  %name2 = mgp.string.create "bar" : !mgp.opaque<"mojo_string">
  %buf2 = mgp.buffer.constant.external %ch, %weights[%name2, 16384, align 16] : !mgp.opaque<"weights_registry">, !mgp.opaque<"mojo_string"> -> !mgp.buffer
  mef.output %buf1, %buf2 : !mgp.buffer, !mgp.buffer
}

// CHECK-LABEL: @buffer_constant_mutable
func.func @buffer_constant_mutable(%ch: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">) -> !mgp.buffer {
  // CHECK: %{{.*}} = mgp.string.create "foo"
  // CHECK: %{{.*}} = mgp.buffer.constant.external %{{.*}},  %{{.*}}[%{{.*}}, 16384, align 16 has_alias true] : !mgp.opaque<"weights_registry">, !mgp.opaque<"mojo_string"> -> !mgp.buffer
  %name = mgp.string.create "foo" : !mgp.opaque<"mojo_string">
  %buf = mgp.buffer.constant.external %ch, %weights[%name, 16384, align 16 has_alias true] : !mgp.opaque<"weights_registry">, !mgp.opaque<"mojo_string"> -> !mgp.buffer
  mef.output %buf : !mgp.buffer
}

//
// mgp.buffer.constant
//

// CHECK-LABEL: @buffer_constant
func.func @buffer_constant() -> !mgp.buffer {
  // CHECK: %{{.*}} = mgp.buffer.constant dense_resource<resource> : !M.aligned_bytes<4, align 64>
  %buf = mgp.buffer.constant dense_resource<resource> : !M.aligned_bytes<4, align 64>
  mef.output %buf : !mgp.buffer
}

//
// mgp.buffer.alloc
//

// CHECK-LABEL: @buffer_alloc
func.func @buffer_alloc(%ctx : !mgp.context, %ch : !mef.chain, %size : index, %gpu : !mgp.device_context<<"gpu", 0>>) -> !mgp.buffer<<"gpu", 3>> {
  // CHECK: %{{.*}} = mgp.buffer.alloc %{{.*}}, %{{.*}} : !mgp.buffer<<"gpu", 3>>, !mgp.device_context<<"gpu", 0>>
  %buf = mgp.buffer.alloc %ch, %size, %gpu : !mgp.buffer<<"gpu", 3>>, !mgp.device_context<<"gpu", 0>>
  mef.output %buf : !mgp.buffer<<"gpu", 3>>
}


//
// mgp.buffer.cache
//

// CHECK-LABEL: @buffer_cache
func.func @buffer_cache(%ctx : !mgp.context, %ch0 : !mef.chain, %buf : !mgp.buffer<<"gpu", 3>>) -> !mef.chain {
  // CHECK: %{{.*}} = mgp.buffer.cache %{{.*}}, %{{.*}}[5], %{{.*}} : (!mef.chain, !mgp.context, !mgp.buffer<<"gpu", 3>>) -> !mef.chain
  %ch1 = mgp.buffer.cache %ch0, %ctx[5], %buf : (!mef.chain, !mgp.context, !mgp.buffer<<"gpu", 3>>) -> !mef.chain
  mef.output %ch1 : !mef.chain
}

//
// mgp.buffer.get_cached
//

// CHECK-LABEL: @buffer_get_cached
func.func @buffer_get_cached(%ctx : !mgp.context, %ch0 : !mef.chain, %idx : index) -> !mgp.buffer<<"gpu", 3>> {
  // CHECK: %{{.*}} = mgp.buffer.get_cached %{{.*}}, %{{.*}}[{{.*}}] : !mgp.buffer<<"gpu", 3>>
  %buf = mgp.buffer.get_cached %ch0, %ctx[%idx] : !mgp.buffer<<"gpu", 3>>
  mef.output %buf : !mgp.buffer<<"gpu", 3>>
}

//
// mgp.buffer.host_to_device
//

// CHECK-LABEL: @buffer_host_to_device_static
func.func @buffer_host_to_device_static(%ctx : !mgp.context, %ch0 : !mef.chain, %cpu_buf : !mgp.buffer, %gpu_buf : !mgp.buffer<<"gpu", 3>>, %gpu : !mgp.device_context<<"gpu", 0>>) -> !mef.chain {
  // CHECK: %{{.*}} = mgp.buffer.host_to_device %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !mgp.buffer -> !mgp.buffer<<"gpu", 3>>, !mgp.device_context<<"gpu", 0>>
  %ch1 = mgp.buffer.host_to_device %ch0, %cpu_buf, %gpu_buf, %gpu : !mgp.buffer -> !mgp.buffer<<"gpu", 3>>, !mgp.device_context<<"gpu", 0>>
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: @buffer_host_to_device_dynamic
func.func @buffer_host_to_device_dynamic(%ctx : !mgp.context, %ch0 : !mef.chain, %cpu_buf : !mgp.buffer, %gpu_buf : !mgp.buffer<<"gpu", 3>>, %gpu : !mgp.device_context<<"gpu", 0>>) -> !mef.chain {
  // CHECK: %{{.*}} = mgp.buffer.host_to_device %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !mgp.buffer -> !mgp.buffer<<"gpu", 3>>, !mgp.device_context<<"gpu", 0>>
  %ch1 = mgp.buffer.host_to_device %ch0, %cpu_buf, %gpu_buf, %gpu : !mgp.buffer -> !mgp.buffer<<"gpu", 3>>, !mgp.device_context<<"gpu", 0>>
  mef.output %ch1 : !mef.chain
}

//
// mgp.buffer.device_to_host
//

// CHECK-LABEL: @buffer_device_to_host_static
func.func @buffer_device_to_host_static(%ctx : !mgp.context, %ch0 : !mef.chain, %gpu_buf : !mgp.buffer<<"gpu", 3>>, %cpu_buf : !mgp.buffer, %gpu : !mgp.device_context<<"gpu", 0>>) -> !mef.chain {
  // CHECK: %{{.*}} = mgp.buffer.device_to_host %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !mgp.buffer<<"gpu", 3>> -> !mgp.buffer, !mgp.device_context<<"gpu", 0>>
  %ch1 = mgp.buffer.device_to_host %ch0, %gpu_buf, %cpu_buf, %gpu : !mgp.buffer<<"gpu", 3>> -> !mgp.buffer, !mgp.device_context<<"gpu", 0>>
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: @buffer_device_to_host_dynamic
func.func @buffer_device_to_host_dynamic(%ctx : !mgp.context, %ch0 : !mef.chain, %gpu_buf : !mgp.buffer<<"gpu", 3>>, %cpu_buf : !mgp.buffer, %gpu : !mgp.device_context<<"gpu", 0>>) -> !mef.chain {
  // CHECK: %{{.*}} = mgp.buffer.device_to_host %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !mgp.buffer<<"gpu", 3>> -> !mgp.buffer, !mgp.device_context<<"gpu", 0>>
  %ch1 = mgp.buffer.device_to_host %ch0, %gpu_buf, %cpu_buf, %gpu : !mgp.buffer<<"gpu", 3>> -> !mgp.buffer, !mgp.device_context<<"gpu", 0>>
  mef.output %ch1 : !mef.chain
}

//
// mgp.buffer.device_to_device
//

// CHECK-LABEL: @buffer_device_to_device_static
func.func @buffer_device_to_device_static(%ctx : !mgp.context, %ch0 : !mef.chain, %buf0 : !mgp.buffer<<"gpu", 3>>, %buf1 : !mgp.buffer<<"gpu", 1>>, %gpu : !mgp.device_context<<"gpu", 0>>) -> !mef.chain {
  // CHECK: %{{.*}} = mgp.buffer.device_to_device %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !mgp.buffer<<"gpu", 3>> -> !mgp.buffer<<"gpu", 1>>, !mgp.device_context<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %ch1 = mgp.buffer.device_to_device %ch0, %buf0, %gpu, %buf1, %gpu : !mgp.buffer<<"gpu", 3>> -> !mgp.buffer<<"gpu", 1>>, !mgp.device_context<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: @buffer_device_to_device_dynamic
func.func @buffer_device_to_device_dynamic(%ctx : !mgp.context, %ch0 : !mef.chain, %buf0 : !mgp.buffer<<"gpu", 3>>, %buf1 : !mgp.buffer<<"gpu", 1>>, %gpu : !mgp.device_context<<"gpu", 0>>) -> !mef.chain {
  // CHECK: %{{.*}} = mgp.buffer.device_to_device %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !mgp.buffer<<"gpu", 3>> -> !mgp.buffer<<"gpu", 1>>, !mgp.device_context<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %ch1 = mgp.buffer.device_to_device %ch0, %buf0, %gpu, %buf1, %gpu : !mgp.buffer<<"gpu", 3>> -> !mgp.buffer<<"gpu", 1>>, !mgp.device_context<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  mef.output %ch1 : !mef.chain
}

//
// mgp.buffer.get_size
//

// CHECK-LABEL: @buffer_get_size
func.func @buffer_get_size(%ch : !mef.chain, %in_buf : !mgp.buffer<<"gpu", 3>>) -> index {
  // CHECK:  %{{.*}} = mgp.buffer.get_size %{{.*}}, %{{.*}} : !mgp.buffer<<"gpu", 3>>
  %size = mgp.buffer.get_size %ch, %in_buf : !mgp.buffer<<"gpu", 3>>
  mef.output %size : index
}

//
// mgp.buffer.concat
//

// CHECK-LABEL: @buffer_concat
func.func @buffer_concat(%ctx : !mgp.context, %ch0 : !mef.chain, %out_buf : !mgp.buffer<<"gpu", 3>>, %buf0 : !mgp.buffer<<"gpu", 3>>, %buf1 : !mgp.buffer<<"gpu", 3>>, %gpu: !mgp.device_context<<"gpu", 3>>) -> !mef.chain {
  // CHECK: %{{.*}} = mgp.buffer
  %ch1 = mgp.buffer.concat(%ch0, %out_buf, %buf0, %buf1, %gpu) device <"gpu", 3> : (!mef.chain, !mgp.buffer<<"gpu", 3>>, !mgp.buffer<<"gpu", 3>>, !mgp.buffer<<"gpu", 3>>, !mgp.device_context<<"gpu", 3>>) -> !mef.chain
  mef.output %ch1 : !mef.chain
}

//
// mgp.buffer.to_bool
//

// CHECK-LABEL: @buffer_to_bool
func.func @buffer_to_bool(%ctx : !mgp.context, %ch0 : !mef.chain, %buf : !mgp.buffer<<"gpu", 3>>) -> i1 {
  // CHECK: %{{.*}} = mgp.buffer.to_bool %{{.*}}, %{{.*}} : !mgp.buffer<<"gpu", 3>>
  %b = mgp.buffer.to_bool %ch0, %buf : !mgp.buffer<<"gpu", 3>>
  mef.output %b : i1
}

//
// mgp.buffer.set_with_index
//

// CHECK-LABEL: @buffer_set_with_index
func.func @buffer_set_with_index(%ctx : !mgp.context, %buf: !mgp.buffer<<"gpu", 3>>, %idx: index) -> !mef.chain {
  // CHECK: %{{.*}} = mgp.buffer.set_with_index %{{.*}}, %{{.*}} : !mgp.buffer<<"gpu", 3>>
  %ch = mgp.buffer.set_with_index %buf, %idx : !mgp.buffer<<"gpu", 3>>
  mef.output %ch : !mef.chain
}

// CHECK-LABEL: @buffer_set_with_index_variadic
func.func @buffer_set_with_index_variadic(%ctx :  !mgp.context, %buf: !mgp.buffer<<"gpu", 3>>, %idx0: index, %idx1: index) -> !mef.chain {
  // CHECK: %{{.*}} = mgp.buffer.set_with_index %{{.*}}, %{{.*}}, %{{.*}} : !mgp.buffer<<"gpu", 3>>
  %ch = mgp.buffer.set_with_index %buf, %idx0, %idx1 : !mgp.buffer<<"gpu", 3>>
  mef.output %ch : !mef.chain
}

//
// mgp.tensor.create
//

// CHECK-LABEL: @tensor_create
func.func @tensor_create(%ch : !mef.chain, %buf : !mgp.buffer<<"gpu", 3>>, %spec : !mgp.tensor_spec<3x5xf32>) -> !mgp.tensor<3x5xf32, <"gpu", 3>> {
  // CHECK: %{{.*}} = mgp.tensor.create %{{.*}}, %{{.*}}, %{{.*}} : !mgp.tensor<3x5xf32, <"gpu", 3>>
  %0 = mgp.tensor.create %ch, %buf, %spec : !mgp.tensor<3x5xf32, <"gpu", 3>>
  mef.output %0 : !mgp.tensor<3x5xf32, <"gpu", 3>>
}

//
// mgp.tensor.extract.buffer
//

// CHECK-LABEL: @tensor_extract_buffer
func.func @tensor_extract_buffer(%ctx : !mgp.context, %tensor : !mgp.tensor<3x5xf32, <"gpu", 3>>) -> !mgp.buffer<<"gpu", 3>> {
  // CHECK: %{{.*}} = mgp.tensor.extract.buffer %{{.*}} : <3x5xf32, <"gpu", 3>>
  %buf = mgp.tensor.extract.buffer %tensor : <3x5xf32, <"gpu", 3>>
  mef.output %buf : !mgp.buffer<<"gpu", 3>>
}

//
// gpu.tensor.extract.tensor_spec
//

// CHECK-LABEL: @tensor_extract_tensor_spec
func.func @tensor_extract_tensor_spec(%ctx : !mgp.context, %tensor : !mgp.tensor<3x?xf32, <"gpu", 3>>) -> !mgp.tensor_spec<3x?xf32> {
  // CHECK: %{{.*}} = mgp.tensor.extract.tensor_spec %{{.*}} : <3x?xf32, <"gpu", 3>>
  %shape = mgp.tensor.extract.tensor_spec %tensor : <3x?xf32, <"gpu", 3>>
  mef.output %shape : !mgp.tensor_spec<3x?xf32>
}

//
// mgp.debug.tensor.print
//

// CHECK-LABEL: @debug_tensor_print
func.func @debug_tensor_print(%ctx : !mgp.context, %ch0 : !mef.chain, %buf : !mgp.buffer<<"gpu", 3>>, %spec : !mgp.tensor_spec<3x5xf32>) -> !mef.chain {
  // CHECK: %{{.*}} = mgp.debug.tensor.print %{{.*}}, %{{.*}}, %{{.*}} : (!mef.chain, !mgp.buffer<<"gpu", 3>>, !mgp.tensor_spec<3x5xf32>) -> !mef.chain
  %ch1 = mgp.debug.tensor.print %ch0, %buf, %spec : (!mef.chain, !mgp.buffer<<"gpu", 3>>, !mgp.tensor_spec<3x5xf32>) -> !mef.chain
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: @debug_print
func.func @debug_print(%ctx : !mgp.context, %ch0 : !mef.chain) -> !mef.chain {
  // CHECK: %{{.*}} = mgp.debug.print %{{.*}}, %{{.*}}, "debug string", "label"
  %ch1 = mgp.debug.print %ch0, %ctx, "debug string", "label"
  mef.output %ch1 : !mef.chain
}

{-#
  dialect_resources: {
    builtin: {
      resource:
        // First 4 bytes are alignment of 64 in little-endian form, then 4 bytes.
        "0x4000000000010203"
    }
  }
#-}
