// RUN: mo-opt %s --mlir-disable-threading --allow-unregistered-dialect --mgp-async-dataflow-dependencies-test | FileCheck %s

//
// Simple grt.if
//

// For reference here's the model with vars matching the checks (modulo the block ids).
// CHECK-LABEL: "mgp.model"() <{{.*}}> ({
// ### Block 0 ###
// CHECK:   %[[SSA76:.*]] = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %[[SSA77:.*]] = "mgp.context.create"() <{numSlots = 2 : ui64}> : () -> !mgp.context
// CHECK:   %[[SSA79:.*]] = "mgp.runtime.create"(%[[SSA76]], %[[SSA77]])
// CHECK:   "mef.output"(%[[SSA79]], %[[SSA77]]
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%[[arg26:[a-z0-9]*]]: !mef.chain, %[[arg27:[a-z0-9]*]]: !mgp.context
// CHECK:   %[[SSA68:.*]] = "some.op"(%[[arg26]]) : (!mef.chain) -> i1
// CHECK:   %[[SSA69:.*]] = "some.op"(%[[arg26]]) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:   %[[SSA70:.*]] = "some.op"(%[[arg26]]) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:   %[[SSA71:.*]]:3 = "grt.if"(%[[SSA68]], %[[arg26]], %[[SSA69]], %[[SSA70]], %[[SSA70]]) ({
// ### Block 2 ###
// CHECK:   ^bb0(%[[arg32:[a-z0-9]*]]: !mef.chain, %[[arg33:[a-z0-9]*]]: !mgp.tensor<f32>,
// CHECK: %[[arg34:[a-z0-9]*]]: !mgp.tensor<f32>, %[[arg35:[a-z0-9]*]]: !mgp.tensor<f32>):
// CHECK:     %[[SSA74:.*]] = "some.op"(%[[arg32]]) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:     %[[SSA75:.*]] = "some.op"(%[[arg34]]) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:     "mef.output"(%[[SSA74]], %[[SSA75]], %[[arg35]]) : (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> ()
// CHECK:   }, {
// ### Block 3 ###
// CHECK:   ^bb0(%[[arg28:[a-z0-9]*]]: !mef.chain, %[[arg29:[a-z0-9]*]]: !mgp.tensor<f32>, %[[arg30:[a-z0-9]*]]: !mgp.tensor<f32>, %[[arg31:[a-z0-9]*]]: !mgp.tensor<f32>):
// CHECK:     %[[SSA73:.*]] = "some.op"(%[[arg30]]) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:     "mef.output"(%[[arg29]], %[[SSA73]], %[[arg31]]) : (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> ()
// CHECK:   }) : (i1, !mef.chain, !mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>)
// CHECK:   %[[SSA72:.*]] = "some.op"(%[[SSA71]]#0, %[[SSA71]]#1, %[[SSA71]]#2) : (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:   "mef.output"(%[[arg26]], %[[SSA72]]) : (!mef.chain, !mgp.tensor<f32>) -> ()
// CHECK: })

// Notice if results now capture both arms
// CHECK: maybeSameAs begin
// CHECK:   %[[arg26]] |-> {%[[SSA79]][0]}
// CHECK:   %[[arg27]] |-> {%[[SSA77]][0]}
// CHECK:   %[[arg32]][2] |-> {%[[SSA79]][0], %[[arg26]]}
// CHECK:   %[[arg33]][2] |-> {%[[SSA69]]}
// CHECK:   %[[arg34]][2] |-> {%[[SSA70]]}
// CHECK:   %[[arg35]][2] |-> {%[[SSA70]]}
// CHECK:   %[[arg28]][3] |-> {%[[SSA79]][0], %[[arg26]]}
// CHECK:   %[[arg29]][3] |-> {%[[SSA69]]}
// CHECK:   %[[arg30]][3] |-> {%[[SSA70]]}
// CHECK:   %[[arg31]][3] |-> {%[[SSA70]]}
// CHECK:   %[[SSA71]]#0 |-> {%[[SSA69]], %[[SSA74]][2], %[[arg29]][3]}
// CHECK:   %[[SSA71]]#1 |-> {%[[SSA75]][2], %[[SSA73]][3]}
// CHECK:   %[[SSA71]]#2 |-> {%[[SSA70]], %[[arg35]][2], %[[arg31]][3]}
// CHECK: maybeSameAs end

// CHECK: isAlwaysReady({%[[SSA76]][0]})

// impliesReady is actually 'implies ready or never bound', so
// %[[SSA71]]#1 captures that either one or other of the if branches will
// be executed.
// We know the if condition is ready if any of its results are ready.
// CHECK: impliesReady begin
// CHECK:   %[[SSA71]]#0 |-> {%[[SSA76]][0], %[[SSA77]][0], %[[SSA79]][0], %[[arg26]], %[[SSA68]], %[[SSA74]][2]}
// CHECK:   %[[SSA71]]#1 |-> {%[[SSA76]][0], %[[SSA77]][0], %[[SSA79]][0], %[[arg26]], %[[SSA68]], %[[SSA70]], %[[SSA75]][2], %[[SSA73]][3]}
// CHECK:   %[[SSA71]]#2 |-> {%[[SSA76]][0], %[[SSA77]][0], %[[SSA79]][0], %[[arg26]], %[[SSA68]], %[[SSA70]]}
// CHECK: impliesReady end

// Ops in the if branches are structurally disjoint
// CHECK: structurallyAtMostOne begin
// CHECK:   %[[SSA74]][2] |-> {%[[SSA73]][3]}
// CHECK:   %[[SSA75]][2] |-> {%[[SSA73]][3]}
// CHECK:   %[[SSA73]][3] |-> {%[[SSA74]][2], %[[SSA75]][2]}
// CHECK: structurallyAtMostOne end

// CHECK: structurallyCannotOverlapRelation begin
// CHECK:   %[[SSA74]][2] |-> {%[[SSA73]][3]}
// CHECK:   %[[SSA75]][2] |-> {%[[SSA73]][3]}
// CHECK:   %[[SSA73]][3] |-> {%[[SSA74]][2], %[[SSA75]][2]}
// CHECK: structurallyCannotOverlapRelation end

// runsLater is actually 'runs later or never runs'.
// CHECK: runsLater begin
// CHECK:   %[[SSA76]][0] |-> {%[[SSA79]][0], %[[SSA68]], %[[SSA69]], %[[SSA70]], %[[SSA71]]#0, %[[SSA74]][2], %[[SSA75]][2], %[[SSA73]][3], %[[SSA72]]}
// CHECK:   %[[SSA77]][0] |-> {%[[SSA79]][0], %[[SSA68]], %[[SSA69]], %[[SSA70]], %[[SSA71]]#0, %[[SSA74]][2], %[[SSA75]][2], %[[SSA73]][3], %[[SSA72]]}
// CHECK:   %[[SSA79]][0] |-> {%[[SSA68]], %[[SSA69]], %[[SSA70]], %[[SSA71]]#0, %[[SSA74]][2], %[[SSA75]][2], %[[SSA73]][3], %[[SSA72]]}
// CHECK:   %[[SSA68]] |-> {%[[SSA71]]#0, %[[SSA72]]}
// CHECK:   %[[SSA70]] |-> {%[[SSA75]][2], %[[SSA73]][3], %[[SSA72]]}
// CHECK:   %[[SSA71]]#0 |-> {%[[SSA72]]}
// CHECK:   %[[SSA74]][2] |-> {%[[SSA72]]}
// CHECK:   %[[SSA75]][2] |-> {%[[SSA72]]}
// CHECK:   %[[SSA73]][3] |-> {%[[SSA72]]}
// CHECK: runsLater end

// exclusive now accounts for nesting and the disjointness of the if branches
// CHECK: exclusive begin
// CHECK:   %[[SSA76]][0] |-> {%[[SSA68]], %[[SSA69]], %[[SSA70]], %[[arg33]][2], %[[arg34]][2], %[[arg35]][2], %[[SSA74]][2], %[[SSA75]][2], %[[arg29]][3], %[[arg30]][3], %[[arg31]][3], %[[SSA73]][3], %[[SSA71]]#0, %[[SSA71]]#1, %[[SSA71]]#2, %[[SSA72]]}
// CHECK:   %[[SSA68]] |-> {%[[SSA76]][0], %[[SSA72]]}
// CHECK:   %[[SSA69]] |-> {%[[SSA76]][0]}
// CHECK:   %[[SSA70]] |-> {%[[SSA76]][0]}
// CHECK:   %[[arg33]][2] |-> {%[[SSA76]][0]}
// CHECK:   %[[arg34]][2] |-> {%[[SSA76]][0]}
// CHECK:   %[[arg35]][2] |-> {%[[SSA76]][0]}
// CHECK:   %[[SSA74]][2] |-> {%[[SSA76]][0], %[[SSA73]][3]}
// CHECK:   %[[SSA75]][2] |-> {%[[SSA76]][0], %[[SSA73]][3]}
// CHECK:   %[[arg29]][3] |-> {%[[SSA76]][0]}
// CHECK:   %[[arg30]][3] |-> {%[[SSA76]][0]}
// CHECK:   %[[arg31]][3] |-> {%[[SSA76]][0]}
// CHECK:   %[[SSA73]][3] |-> {%[[SSA76]][0], %[[SSA74]][2], %[[SSA75]][2]}
// CHECK:   %[[SSA71]]#0 |-> {%[[SSA76]][0]}
// CHECK:   %[[SSA71]]#1 |-> {%[[SSA76]][0]}
// CHECK:   %[[SSA71]]#2 |-> {%[[SSA76]][0]}
// CHECK:   %[[SSA72]] |-> {%[[SSA76]][0], %[[SSA68]]}
// CHECK: exclusive end

mgp.model @simple : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>

} execute (%ch0: !mef.chain, %cxt: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<f32>) {
  %cond = "some.op"(%ch0) : (!mef.chain) -> i1
  %x = "some.op"(%ch0) : (!mef.chain) -> !mgp.tensor<f32>
  %y = "some.op"(%ch0) : (!mef.chain) -> !mgp.tensor<f32>
  %i, %j, %k = grt.if %cond (%ch0: !mef.chain, %x: !mgp.tensor<f32>, %y: !mgp.tensor<f32>, %y as %z: !mgp.tensor<f32>)
    -> (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) {
    %c = "some.op"(%ch0) : (!mef.chain) -> !mgp.tensor<f32>
    %d = "some.op"(%y) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
    mef.output %c, %d, %z : !mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>
  } else {
    %e = "some.op"(%y) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
    mef.output %x, %e, %z : !mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>
  }
  %l = "some.op"(%i, %j, %k) : (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> !mgp.tensor<f32>
  mef.output %ch0, %l : !mef.chain, !mgp.tensor<f32>
}

//
// Lots of redundant grt.chain.create operands.
//

// For reference here's the model with vars matching the checks (modulo the block ids).
// ### Block 0 ###
// CHECK-LABEL: "mgp.model"() <{{.*}}> ({
// CHECK:   %[[SSA64:.*]] = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %[[SSA65:.*]] = "mgp.context.create"() <{numSlots = 2 : ui64}> : () -> !mgp.context
// CHECK:   %[[SSA67:.*]] = "mgp.runtime.create"(%[[SSA64]], %[[SSA65]])
// CHECK:   "mef.output"(%[[SSA67]], %[[SSA65]]
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%[[arg22:[a-z0-9]*]]: !mef.chain, %[[arg23:[a-z0-9]*]]: !mgp.context
// CHECK:   %[[SSA49:.*]] = "grt.chain.create"(%[[arg22]]) : (!mef.chain) -> !mef.chain
// CHECK:   %[[SSA50:.*]] = "grt.chain.create"(%[[SSA49]], %[[SSA49]]) : (!mef.chain, !mef.chain) -> !mef.chain
// CHECK:   %[[SSA51:.*]] = "some.op"(%[[SSA50]]) : (!mef.chain) -> !mef.chain
// CHECK:   %[[SSA52:.*]] = "grt.chain.create"(%[[arg22]], %[[SSA51]]) : (!mef.chain, !mef.chain) -> !mef.chain
// CHECK:   %[[SSA53:.*]] = "some.op"(%[[SSA52]]) : (!mef.chain) -> !mef.chain
// CHECK:   %[[SSA54:.*]] = "some.op"(%[[SSA52]]) : (!mef.chain) -> !mef.chain
// CHECK:   %[[SSA55:.*]] = "some.op"(%[[SSA52]]) : (!mef.chain) -> !mef.chain
// CHECK:   %[[SSA56:.*]] = "grt.chain.create"(%[[SSA53]], %[[SSA54]]) : (!mef.chain, !mef.chain) -> !mef.chain
// CHECK:   %[[SSA57:.*]] = "grt.chain.create"(%[[SSA52]], %[[SSA56]], %[[SSA55]]) : (!mef.chain, !mef.chain, !mef.chain) -> !mef.chain
// CHECK:   %[[SSA58:.*]] = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %[[SSA59:.*]] = "grt.chain.create"(%[[SSA58]], %[[SSA57]]) : (!mef.chain, !mef.chain) -> !mef.chain
// CHECK:   %[[SSA60:.*]] = "some.op"() : () -> i1
// CHECK:   %[[SSA61:.*]] = "grt.if"(%[[SSA60]], %[[SSA58]]) ({
// ### Block 2 ###
// CHECK:   ^bb0(%[[arg25:[a-z0-9]*]]: !mef.chain):
// CHECK:     "mef.output"(%[[arg25]]) : (!mef.chain) -> ()
// CHECK:   }, {
// ### Block 3 ###
// CHECK:   ^bb0(%[[arg24:[a-z0-9]*]]: !mef.chain):
// CHECK:     %[[SSA63:.*]] = "grt.chain.create"() : () -> !mef.chain
// CHECK:     "mef.output"(%[[SSA63]]) : (!mef.chain) -> ()
// CHECK:   }) : (i1, !mef.chain) -> !mef.chain
// CHECK:   %[[SSA62:.*]] = "grt.chain.create"(%[[SSA58]], %[[SSA59]], %[[SSA61]]) : (!mef.chain, !mef.chain, !mef.chain) -> !mef.chain
// CHECK:   "mef.output"(%[[SSA62]]) : (!mef.chain) -> ()
// CHECK: })

// CHECK: sameAs begin
// CHECK:   %[[arg25]][2] |-> {%[[SSA58]]}
// CHECK:   %[[arg24]][3] |-> {%[[SSA58]]}
// CHECK: sameAs end

// CHECK: maybeSameAs begin
// CHECK:   %[[SSA61]] |-> {%[[SSA58]], %[[arg25]][2], %[[SSA63]][3]}
// CHECK: maybeSameAs end

// CHECK: isAlwaysReady({%[[SSA64]][0], %[[SSA58]], %[[arg25]][2], %[[arg24]][3], %[[SSA63]][3], %[[SSA61]]})

// CHECK: impliesReady begin
// CHECK:   %[[SSA51]] |-> {%[[SSA64]][0], %[[SSA65]][0], %[[SSA67]][0], %[[arg22]], %[[SSA49]], %[[SSA50]]}
// CHECK: impliesReady end

// CHECK: runsLater begin
// CHECK:   %[[SSA56]] |-> {%[[SSA57]], %[[SSA59]], %[[SSA62]]}
// CHECK: runsLater end

// CHECK: exclusive begin
// CHECK:   %[[SSA56]] |-> {%[[SSA64]][0], %[[SSA49]], %[[SSA50]], %[[SSA51]], %[[SSA59]], %[[SSA62]]}
// CHECK: exclusive end

mgp.model @redundant : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>

} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %ch1 = grt.chain.create (%ch0)
  %ch2 = grt.chain.create (%ch1, %ch1)
  %ch3 = "some.op"(%ch2) : (!mef.chain) -> !mef.chain
  %ch4 = grt.chain.create (%ch0, %ch3)
  %ch5 = "some.op"(%ch4) : (!mef.chain) -> !mef.chain
  %ch6 = "some.op"(%ch4) : (!mef.chain) -> !mef.chain
  %ch7 = "some.op"(%ch4) : (!mef.chain) -> !mef.chain
  %ch8 = grt.chain.create (%ch5, %ch6)
  %ch9 = grt.chain.create (%ch4, %ch8, %ch7)
  %ch10 = grt.chain.create ()
  %ch11 = grt.chain.create (%ch10, %ch9)
  %c = "some.op"() : () -> i1
  %ch12 = grt.if %c (%ch10 : !mef.chain) -> !mef.chain {
    mef.output %ch10 : !mef.chain
  } else {
    %ch13 = grt.chain.create ()
    mef.output %ch13 : !mef.chain
  }
  %ch14 = grt.chain.create (%ch10, %ch11, %ch12)
  mef.output %ch14 : !mef.chain
}
