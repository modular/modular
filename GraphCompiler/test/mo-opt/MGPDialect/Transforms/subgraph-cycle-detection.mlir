// RUN: not mo-opt %s --mgp-outline-models-to-callables --allow-unregistered-dialect 2>&1 | FileCheck %s
// CHECK: cycle detected in subgraph call graph
// CHECK: subgraph in cycle

// Test case: A -> B -> A cycle in subgraph calls

module {
    // Entry point model that calls into a cyclic subgraph
    mgp.model @cycle_test : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"cpu", 0>>) {
        %0 = grt.chain.create ()
        %1 = mgp.context.create 1
        %2 = mgp.runtime.create %0, %1[0]
        mef.output %0, %1, %arg0, %arg1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
    } execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"cpu", 0>>, %input : !mgp.tensor<4x4xf32>) -> (!mef.chain, !mgp.tensor<4x4xf32>) {
        // Start the cycle by calling model A
        %ch1, %result = mgp.call @model_a(%arg0, %arg1, %arg2, %arg3, %input) : (!mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.tensor<4x4xf32>) -> (!mef.chain, !mgp.tensor<4x4xf32>)
        mef.output %ch1, %result : !mef.chain, !mgp.tensor<4x4xf32>
    }

    // Subgraph A calls subgraph B
    mgp.model @model_a subgraph : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"cpu", 0>>) {
        %0 = grt.chain.create ()
        %1 = mgp.context.create 1
        %2 = mgp.runtime.create %0, %1[0]
        mef.output %0, %1, %arg0, %arg1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
    } execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"cpu", 0>>, %input : !mgp.tensor<4x4xf32>, %idx: index) -> (!mef.chain, !mgp.tensor<4x4xf32>) {
        %pre = "a_preprocess"(%arg1, %arg0, %input) : (!mgp.context, !mef.chain, !mgp.tensor<4x4xf32>) -> !mgp.tensor<4x4xf32>
        // Cyclical call to model B
        %ch1, %processed = mgp.call @model_b(%arg0, %arg1, %arg2, %arg3, %pre) : (!mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.tensor<4x4xf32>) -> (!mef.chain, !mgp.tensor<4x4xf32>)
        mef.output %ch1, %processed : !mef.chain, !mgp.tensor<4x4xf32>
    }

    // Subgraph B calls subgraph A, creating the cycle
    mgp.model @model_b subgraph : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"cpu", 0>>) {
        %0 = grt.chain.create ()
        %1 = mgp.context.create 1
        %2 = mgp.runtime.create %0, %1[0]
        mef.output %0, %1, %arg0, %arg1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
    } execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"cpu", 0>>, %input : !mgp.tensor<4x4xf32>, %idx: index) -> (!mef.chain, !mgp.tensor<4x4xf32>) {
        %pre = "b_preprocess"(%arg1, %arg0, %input) : (!mgp.context, !mef.chain, !mgp.tensor<4x4xf32>) -> !mgp.tensor<4x4xf32>
        // Cyclical call back to model A
        %ch1, %processed = mgp.call @model_a(%arg0, %arg1, %arg2, %arg3, %pre) : (!mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.tensor<4x4xf32>) -> (!mef.chain, !mgp.tensor<4x4xf32>)
        mef.output %ch1, %processed : !mef.chain, !mgp.tensor<4x4xf32>
    }
}
