// RUN: mo-opt -mogg-merge-identical-funcs

mogg.func @func_1()
  attributes {abi = 2 : i32, constantResources = [], device = "cpu", isAsync = true, fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"func_1">>} {
  mogg.output
}

mogg.func @func_2()
  attributes {abi = 2 : i32, constantResources = [], device = "cpu", isAsync = true, fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"func_2">>} {
  mogg.output
}

// CHECK-LABEL: @func_3
// CHECK:      mogg.call @func_1
// CHECK-NEXT: mogg.call @func_1
// CHECK-NOT:  mogg.call @func
mogg.func @func_3()
  attributes {abi = 2 : i32, constantResources = [], device = "cpu", isAsync = true, fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"func_3">>} {
  mogg.call @func_1
  mogg.call @func_2
  mogg.output
}

// CHECK-LABEL: @func_4
// CHECK:      mogg.call @func_1
// CHECK:      mogg.call @func_1
// CHECK-NOT:  mogg.call @func
mogg.func @func_4()
  attributes {abi = 2 : i32, constantResources = [], device = "cpu", isAsync = true, fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"func_4">>} {
  mogg.call @func_1
  mogg.call @func_2
  mogg.output
}

// CHECK-LABEL: @func_5
// CHECK:      mogg.call @func_3
// CHECK:      mogg.call @func_3
// CHECK-NOT:  mogg.call @func
mogg.func @func_5()
  attributes {abi = 2 : i32, constantResources = [], device = "cpu", isAsync = true, fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"func_5">>} {
  mogg.call @func_3
  mogg.call @func_4
  mogg.output
}

mogg.func @func_6()
  attributes {abi = 2 : i32, constantResources = [], device = "cpu", foo = "foo", isAsync = true, fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"func_6">>} {
  mogg.output
}

mogg.func @func_7()
  attributes {abi = 2 : i32, constantResources = [], device = "cpu", foo = "foo", isAsync = true, fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"func_7">>} {
  mogg.output
}

// CHECK-LABEL: @top_level
// CHECK: mogg.compiled.call @func_6
// CHECK: mogg.compiled.call @func_6
mgp.model @top_level subgraph : mof devices [#M.device_ref<"cpu", 0>, #M.device_ref<"gpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"cpu", 0>>, %arg2: !mgp.device_context<<"gpu", 0>>) {
    %0 = grt.chain.create ()
    %1 = mgp.context.create 2
    %2 = mgp.runtime.create %0, %1[1]
    mef.output %0, %1, %arg0, %arg1, %arg2 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.device_context<<"gpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"cpu", 0>>, %arg4: !mgp.device_context<<"gpu", 0>>, %input : !mgp.tensor<4x16xf32>, %idx: index) -> (!mef.chain) {
    mogg.compiled.call @func_6() fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mogg.routine.async">> constantResources = [], usedParams = [] {device = #M.device_ref<"cpu", 0>} : () -> ()
    mogg.compiled.call @func_7() fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mogg.routine.async">> constantResources = [], usedParams = [] {device = #M.device_ref<"cpu", 0>} : () -> ()
    mef.output %arg0 : !mef.chain
}
