// Tests for MOGG async dispatch operations:
//   - mogg.call.async: Calls a kernel stub asynchronously via create_task
//   - mogg.async.join: Waits for async task handles and checks for errors
//
// These operations form the foundation for parallel kernel execution.

// RUN: mo-opt %s --mogg-import-packages --mgp-lower-primitives --mogg-outline-async-routines --mogg-add-try-except-blocks --canonicalize --mogg-emit-mojo --mogg-strip-kernel-decls --jit --mogg-lower-compiled-routines --mgp-outline-models-to-callables -o %t.mlir
// RUN: mt --execute %t.mlir | FileCheck %s

//===----------------------------------------------------------------------===//
// Kernel stubs for async dispatch testing
//===----------------------------------------------------------------------===//

mogg.func @noop_stub private ()
  attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "cpu", summary = "noop_stub", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"noop_stub">>} {
  mogg.output
}

mogg.func @noop_stub_2 private ()
  attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "cpu", summary = "noop_stub_2", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"noop_stub_2">>} {
  mogg.output
}

//===----------------------------------------------------------------------===//
// Test: Single async stub call with join
//===----------------------------------------------------------------------===//

// CHECK-LABEL: --- Running 'single_async_call':
// CHECK-NOT: <error
mgp.model @single_async_call : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch1, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {

  %out_chain = mogg.routine.async (%arg0) :
      (%ch: !mef.chain) -> !mef.chain {

    %task = mogg.call.async @noop_stub
    mogg.async.join(%task)
    %ready = mogg.async.ready

    mogg.output %ready : !mef.chain
  } canBeFusedInto = false canFuseIntoOther = false inOutSignature = <"i"> {device = #M.device_ref<"cpu", 0>, summary = "single_async_call"}

  mef.output %out_chain : !mef.chain
}

//===----------------------------------------------------------------------===//
// Test: Two parallel async stub calls with join
//===----------------------------------------------------------------------===//

// CHECK-LABEL: --- Running 'two_parallel_async_calls':
// CHECK-NOT: <error
mgp.model @two_parallel_async_calls : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch1, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {

  %out_chain = mogg.routine.async (%arg0) :
      (%ch: !mef.chain) -> !mef.chain {

    %t0 = mogg.call.async @noop_stub
    %t1 = mogg.call.async @noop_stub_2
    mogg.async.join(%t0, %t1)
    %ready = mogg.async.ready

    mogg.output %ready : !mef.chain
  } canBeFusedInto = false canFuseIntoOther = false inOutSignature = <"i"> {device = #M.device_ref<"cpu", 0>, summary = "two_parallel_async_calls"}

  mef.output %out_chain : !mef.chain
}

//===----------------------------------------------------------------------===//
// Test: Four parallel async stub calls with join
//===----------------------------------------------------------------------===//

// CHECK-LABEL: --- Running 'four_parallel_async_calls':
// CHECK-NOT: <error
mgp.model @four_parallel_async_calls : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch1, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {

  %out_chain = mogg.routine.async (%arg0) :
      (%ch: !mef.chain) -> !mef.chain {

    %t0 = mogg.call.async @noop_stub
    %t1 = mogg.call.async @noop_stub
    %t2 = mogg.call.async @noop_stub_2
    %t3 = mogg.call.async @noop_stub_2
    mogg.async.join(%t0, %t1, %t2, %t3)
    %ready = mogg.async.ready

    mogg.output %ready : !mef.chain
  } canBeFusedInto = false canFuseIntoOther = false inOutSignature = <"i"> {device = #M.device_ref<"cpu", 0>, summary = "four_parallel_async_calls"}

  mef.output %out_chain : !mef.chain
}
