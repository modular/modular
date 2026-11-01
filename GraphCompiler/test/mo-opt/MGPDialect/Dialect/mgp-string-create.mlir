// RUN: mo-opt --allow-unregistered-dialect %s | FileCheck %s

//
// mgp.string.create
//

// CHECK-LABEL: @string_create_basic
func.func @string_create_basic() -> !mgp.opaque<"mojo_string"> {
  // CHECK: %{{.*}} = mgp.string.create "hello world" : !mgp.opaque<"mojo_string">
  %str = mgp.string.create "hello world" : !mgp.opaque<"mojo_string">
  mef.output %str : !mgp.opaque<"mojo_string">
}

// CHECK-LABEL: @string_create_empty
func.func @string_create_empty() -> !mgp.opaque<"mojo_string"> {
  // CHECK: %{{.*}} = mgp.string.create "" : !mgp.opaque<"mojo_string">
  %str = mgp.string.create "" : !mgp.opaque<"mojo_string">
  mef.output %str : !mgp.opaque<"mojo_string">
}

// CHECK-LABEL: @string_create_special_chars
func.func @string_create_special_chars() -> !mgp.opaque<"mojo_string"> {
  // CHECK: %{{.*}} = mgp.string.create "path/to/file.txt" : !mgp.opaque<"mojo_string">
  %str = mgp.string.create "path/to/file.txt" : !mgp.opaque<"mojo_string">
  mef.output %str : !mgp.opaque<"mojo_string">
}

// CHECK-LABEL: @string_create_in_model
mgp.model @string_create_in_model : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %chain = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %chain, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%chain: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.opaque<"mojo_string">) {
  // CHECK: %{{.*}} = mgp.string.create "test_string" : !mgp.opaque<"mojo_string">
  %str = mgp.string.create "test_string" : !mgp.opaque<"mojo_string">
  mef.output %chain, %str : !mef.chain, !mgp.opaque<"mojo_string">
}

//
// mgp.string.concat
//

// CHECK-LABEL: @string_concat_basic
func.func @string_concat_basic() -> !mgp.opaque<"mojo_string"> {
  // CHECK: %[[PREFIX:.*]] = mgp.string.create "hello " : !mgp.opaque<"mojo_string">
  %prefix = mgp.string.create "hello " : !mgp.opaque<"mojo_string">
  // CHECK: %[[END:.*]] = mgp.string.create "world" : !mgp.opaque<"mojo_string">
  %end = mgp.string.create "world" : !mgp.opaque<"mojo_string">
  // CHECK: %{{.*}} = mgp.string.concat %[[PREFIX]], %[[END]] : (!mgp.opaque<"mojo_string">, !mgp.opaque<"mojo_string">) -> !mgp.opaque<"mojo_string">
  %result = mgp.string.concat %prefix, %end : (!mgp.opaque<"mojo_string">, !mgp.opaque<"mojo_string">) -> !mgp.opaque<"mojo_string">
  mef.output %result : !mgp.opaque<"mojo_string">
}

// CHECK-LABEL: @string_concat_empty_prefix
func.func @string_concat_empty_prefix() -> !mgp.opaque<"mojo_string"> {
  // CHECK: %[[PREFIX:.*]] = mgp.string.create "" : !mgp.opaque<"mojo_string">
  %prefix = mgp.string.create "" : !mgp.opaque<"mojo_string">
  // CHECK: %[[END:.*]] = mgp.string.create "world" : !mgp.opaque<"mojo_string">
  %end = mgp.string.create "world" : !mgp.opaque<"mojo_string">
  // CHECK: %{{.*}} = mgp.string.concat %[[PREFIX]], %[[END]] : (!mgp.opaque<"mojo_string">, !mgp.opaque<"mojo_string">) -> !mgp.opaque<"mojo_string">
  %result = mgp.string.concat %prefix, %end : (!mgp.opaque<"mojo_string">, !mgp.opaque<"mojo_string">) -> !mgp.opaque<"mojo_string">
  mef.output %result : !mgp.opaque<"mojo_string">
}

// CHECK-LABEL: @string_concat_empty_end
func.func @string_concat_empty_end() -> !mgp.opaque<"mojo_string"> {
  // CHECK: %[[PREFIX:.*]] = mgp.string.create "hello" : !mgp.opaque<"mojo_string">
  %prefix = mgp.string.create "hello" : !mgp.opaque<"mojo_string">
  // CHECK: %[[END:.*]] = mgp.string.create "" : !mgp.opaque<"mojo_string">
  %end = mgp.string.create "" : !mgp.opaque<"mojo_string">
  // CHECK: %{{.*}} = mgp.string.concat %[[PREFIX]], %[[END]] : (!mgp.opaque<"mojo_string">, !mgp.opaque<"mojo_string">) -> !mgp.opaque<"mojo_string">
  %result = mgp.string.concat %prefix, %end : (!mgp.opaque<"mojo_string">, !mgp.opaque<"mojo_string">) -> !mgp.opaque<"mojo_string">
  mef.output %result : !mgp.opaque<"mojo_string">
}

// CHECK-LABEL: @string_concat_in_model
mgp.model @string_concat_in_model : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %chain = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %chain, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%chain: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.opaque<"mojo_string">) {
  // CHECK: %[[PREFIX:.*]] = mgp.string.create "prefix_" : !mgp.opaque<"mojo_string">
  %prefix = mgp.string.create "prefix_" : !mgp.opaque<"mojo_string">
  // CHECK: %[[END:.*]] = mgp.string.create "suffix" : !mgp.opaque<"mojo_string">
  %end = mgp.string.create "suffix" : !mgp.opaque<"mojo_string">
  // CHECK: %{{.*}} = mgp.string.concat %[[PREFIX]], %[[END]] : (!mgp.opaque<"mojo_string">, !mgp.opaque<"mojo_string">) -> !mgp.opaque<"mojo_string">
  %result = mgp.string.concat %prefix, %end : (!mgp.opaque<"mojo_string">, !mgp.opaque<"mojo_string">) -> !mgp.opaque<"mojo_string">
  mef.output %chain, %result : !mef.chain, !mgp.opaque<"mojo_string">
}
