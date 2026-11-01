// RUN: mo-opt %s --mlir-disable-threading --allow-unregistered-dialect --mgp-async-dataflow-dependencies-test | FileCheck %s

//
// Simple graph with no control flow
//

// For reference here's the model with vars matching the checks:
// CHECK-LABEL: "mgp.model"() <{{.*}}> ({
// ### Block 0 ###
// CHECK:   %[[SSA48:.*]] = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %[[SSA49:.*]] = "mgp.context.create"() <{numSlots = 2 : ui64}> : () -> !mgp.context
// CHECK:   %[[SSA51:.*]] = "mgp.runtime.create"(%[[SSA48]], %[[SSA49]])
// CHECK:   "mef.output"(%[[SSA51]], %[[SSA49]]
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%[[arg4:[a-z0-9]*]]: !mef.chain, %[[arg5:[a-z0-9]*]]: !mgp.context
// CHECK:   %[[SSA42:.*]] = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %[[SSA43:.*]] = "some.op"(%[[arg4]]) : (!mef.chain) -> !mef.chain
// CHECK:   %[[SSA44:.*]] = "some.op"(%[[SSA43]], %[[SSA42]]) : (!mef.chain, !mef.chain) -> !mef.chain
// CHECK:   %[[SSA45:.*]] = "some.op"(%[[arg4]]) : (!mef.chain) -> !mef.chain
// CHECK:   %[[SSA46:.*]] = "some.op"(%[[SSA45]], %[[SSA42]]) : (!mef.chain, !mef.chain) -> !mef.chain
// CHECK:   %[[SSA47:.*]] = "some.op"(%[[SSA44]], %[[SSA46]]) : (!mef.chain, !mef.chain) -> !mef.chain
// CHECK:   "mef.output"(%[[SSA47]]) : (!mef.chain) -> ()
// CHECK: })

// sameAs only ties the 'init' results to the 'execute args
// CHECK: sameAs begin
// CHECK:   %[[arg4]] |-> {%[[SSA51]][0]}
// CHECK:   %[[arg5]] |-> {%[[SSA49]][0]}
// CHECK: sameAs end

// maybeSameAs == sameAs
// CHECK: maybeSameAs begin
// CHECK:   %[[arg4]] |-> {%[[SSA51]][0]}
// CHECK:   %[[arg5]] |-> {%[[SSA49]][0]}
// CHECK: maybeSameAs end

// The two always ready chains (this is up to sameAs equivalence)
// CHECK: isAlwaysReady({%[[SSA48]][0], %[[SSA42]]})

// impliesReady captures transitive operand dependencies (plus identity)
// CHECK: impliesReady begin
// CHECK:   %[[SSA51]][0] |-> {%[[SSA48]][0], %[[SSA49]][0]}
// CHECK:   %[[arg4]] |-> {%[[SSA48]][0], %[[SSA49]][0], %[[SSA51]][0]}
// CHECK:   %[[arg5]] |-> {%[[SSA49]][0]}
// CHECK:   %[[SSA43]] |-> {%[[SSA48]][0], %[[SSA49]][0], %[[SSA51]][0], %[[arg4]]}
// CHECK:   %[[SSA44]] |-> {%[[SSA48]][0], %[[SSA49]][0], %[[SSA51]][0], %[[arg4]], %[[SSA42]], %[[SSA43]]}
// CHECK:   %[[SSA45]] |-> {%[[SSA48]][0], %[[SSA49]][0], %[[SSA51]][0], %[[arg4]]}
// CHECK:   %[[SSA46]] |-> {%[[SSA48]][0], %[[SSA49]][0], %[[SSA51]][0], %[[arg4]], %[[SSA42]], %[[SSA45]]}
// CHECK:   %[[SSA47]] |-> {%[[SSA48]][0], %[[SSA49]][0], %[[SSA51]][0], %[[arg4]], %[[SSA42]], %[[SSA43]], %[[SSA44]], %[[SSA45]], %[[SSA46]]}
// CHECK: impliesReady end

// No control flow so nothing in these relations.
// CHECK: structurallyAtMostOne begin
// CHECK: structurallyAtMostOne end
// CHECK: structurallyCannotOverlapRelation begin
// CHECK: structurallyCannotOverlapRelation end

// runsLater for which ops (identified by their result values) run after which
// CHECK: runsLater begin
// CHECK:   %[[SSA48]][0] |-> {%[[SSA51]][0], %[[SSA43]], %[[SSA44]], %[[SSA45]], %[[SSA46]], %[[SSA47]]}
// CHECK:   %[[SSA49]][0] |-> {%[[SSA51]][0], %[[SSA43]], %[[SSA44]], %[[SSA45]], %[[SSA46]], %[[SSA47]]}
// CHECK:   %[[SSA51]][0] |-> {%[[SSA43]], %[[SSA44]], %[[SSA45]], %[[SSA46]], %[[SSA47]]}
// CHECK:   %[[SSA42]] |-> {%[[SSA44]], %[[SSA46]], %[[SSA47]]}
// CHECK:   %[[SSA43]] |-> {%[[SSA44]], %[[SSA47]]}
// CHECK:   %[[SSA44]] |-> {%[[SSA47]]}
// CHECK:   %[[SSA45]] |-> {%[[SSA46]], %[[SSA47]]}
// CHECK:   %[[SSA46]] |-> {%[[SSA47]]}
// CHECK: runsLater end

// exclusive tells us which values can't have overlapping lifetimes.
// CHECK: exclusive begin
// CHECK:   %[[SSA48]][0] |-> {%[[SSA43]], %[[SSA44]], %[[SSA45]], %[[SSA46]], %[[SSA47]]}
// CHECK:   %[[SSA44]] |-> {%[[SSA48]][0]}
// CHECK:   %[[SSA45]] |-> {%[[SSA48]][0], %[[SSA47]]}
// CHECK:   %[[SSA46]] |-> {%[[SSA48]][0]}
// CHECK:   %[[SSA47]] |-> {%[[SSA48]][0], %[[SSA42]], %[[SSA43]], %[[SSA45]]}
// CHECK: exclusive end

mgp.model @simple : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %cxt: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %ch1 = grt.chain.create ()
  %ch2 = "some.op"(%ch0) : (!mef.chain) -> !mef.chain
  %ch3 = "some.op"(%ch2, %ch1) : (!mef.chain, !mef.chain) -> !mef.chain
  %ch4 = "some.op"(%ch0) : (!mef.chain) -> !mef.chain
  %ch5 = "some.op"(%ch4, %ch1) : (!mef.chain, !mef.chain) -> !mef.chain
  %ch6 = "some.op"(%ch3, %ch5) : (!mef.chain, !mef.chain) -> !mef.chain
  mef.output %ch6 : !mef.chain
}
