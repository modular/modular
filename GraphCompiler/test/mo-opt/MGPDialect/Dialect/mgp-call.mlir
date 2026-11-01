// RUN: mo-opt %s --verify-diagnostics

module {
    mgp.model @noop_subgraph : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"cpu", 0>>) {
        %0 = grt.chain.create ()
        %1 = mgp.context.create 2
        %2 = mgp.runtime.create %0, %1[1]
        mef.output %2, %1, %arg0, %arg1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
    } execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"cpu", 0>>, %input : !mgp.tensor<4x16xf32>, %idx: index) -> (!mef.chain, !mgp.tensor<4x16xf32>) {
        %out = "mgp.generic.execute"(%arg1, %arg0, %input) {aRuntimeSlot = 0 : ui64, bModuleIndex = 0 : ui64, cFunctionIndex = 2 : ui64, dSummary = "do_stuff", eIsAsync = false, fConstantResource = []} : (!mgp.context, !mef.chain, !mgp.tensor<4x16xf32>) -> !mgp.tensor<4x16xf32>
        mef.output %arg0, %out : !mef.chain, !mgp.tensor<4x16xf32>
    }

    mgp.model @basic_call : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"cpu", 0>>) {
        %0 = grt.chain.create ()
        %1 = mgp.context.create 2
        %2 = mgp.runtime.create %0, %1[1]
        mef.output %2, %1, %arg0, %arg1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
    } execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"cpu", 0>>, %input : !mgp.tensor<4x16xf32>) -> (!mef.chain, !mgp.tensor<4x16xf32>) {
        %ch0, %res0 = mgp.call @noop_subgraph(%arg0, %arg1, %arg2, %arg3, %input) : (!mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.tensor<4x16xf32>) -> (!mef.chain, !mgp.tensor<4x16xf32>)
        %ch1, %res1 = mgp.call @noop_subgraph(%ch0, %arg1, %arg2, %arg3, %res0) : (!mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.tensor<4x16xf32>) -> (!mef.chain, !mgp.tensor<4x16xf32>)
        %ch2, %res2 = mgp.call @noop_subgraph(%ch1, %arg1, %arg2, %arg3, %res1) : (!mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.tensor<4x16xf32>) -> (!mef.chain, !mgp.tensor<4x16xf32>)
        mef.output %ch2, %res2 : !mef.chain, !mgp.tensor<4x16xf32>
    }
}
