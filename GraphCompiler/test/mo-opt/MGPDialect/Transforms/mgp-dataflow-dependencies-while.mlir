// RUN: mo-opt %s --mlir-disable-threading --allow-unregistered-dialect --mgp-async-dataflow-dependencies-test | FileCheck %s

// For reference here's the model with vars matching the checks (modulo the block ids).
// CHECK: "mgp.model"() <{{.*}}> ({
// ### Block 0 ###
// CHECK:   %[[SSA94:.*]] = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %[[SSA95:.*]] = "mgp.context.create"() <{numSlots = 2 : ui64}> : () -> !mgp.context
// CHECK:   %[[SSA97:.*]] = "mgp.runtime.create"(%[[SSA94]], %[[SSA95]])
// CHECK:   "mef.output"(%[[SSA97]], %[[SSA95]]
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%[[arg43:[a-z0-9]*]]: !mef.chain, %[[arg44:[a-z0-9]*]]: !mgp.context
// CHECK:   %[[SSA89:.*]] = "some.op"(%[[arg43]]) : (!mef.chain) -> si32
// CHECK:   %[[SSA90:.*]] = "some.op"(%[[arg43]]) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:   %[[SSA91:.*]] = "grt.repeat.i32"(%[[SSA89]], %[[SSA90]]) ({
// ### Block 2 ###
// CHECK:   ^bb0(%[[arg45:[a-z0-9]*]]: !mgp.tensor<f32>):
// CHECK:     %[[SSA93:.*]] = "some.op"(%[[arg45]]) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:     "mef.output"(%[[SSA93]]) : (!mgp.tensor<f32>) -> ()
// CHECK:   }) : (si32, !mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:   %[[SSA92:.*]] = "some.op"(%[[SSA91]]) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:   "mef.output"(%[[arg43]], %[[SSA92]]) : (!mef.chain, !mgp.tensor<f32>) -> ()
// CHECK: })


// Note the repeat input, result, sub-block arg and sub-block output are all
// in the same equivalence class.
// CHECK: maybeSameAs begin
// CHECK:   %[[arg45]][2] |-> {%[[SSA90]], %[[SSA93]][2]}
// CHECK:   %[[SSA91]] |-> {%[[SSA90]], %[[arg45]][2], %[[SSA93]][2]}
// CHECK: maybeSameAs end

mgp.model @repeat1 : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %cxt: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<f32>) {
  %c = "some.op"(%ch0) : (!mef.chain) -> si32
  %x = "some.op"(%ch0) : (!mef.chain) -> !mgp.tensor<f32>
  %z = grt.repeat.i32 %c (%x : !mgp.tensor<f32>) {
    %y = "some.op"(%x) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
    mef.output %y : !mgp.tensor<f32>
  }
  %out = "some.op"(%z) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
  mef.output %ch0, %out : !mef.chain, !mgp.tensor<f32>
}

// For reference here's the model with vars matching the checks (modulo the block ids).
// CHECK: "mgp.model"() <{{.*}}> ({
// ### Block ###
// CHECK:   %[[SSA85:.*]] = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %[[SSA86:.*]] = "mgp.context.create"() <{numSlots = 2 : ui64}> : () -> !mgp.context
// CHECK:   %[[SSA88:.*]] = "mgp.runtime.create"(%[[SSA85]], %[[SSA86]])
// CHECK:   "mef.output"(%[[SSA88]], %[[SSA86]]
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%[[arg40:[a-z0-9]*]]: !mef.chain, %[[arg41:[a-z0-9]*]]: !mgp.context
// CHECK:   %[[SSA81:.*]] = "some.op"(%[[arg40]]) : (!mef.chain) -> si32
// CHECK:   %[[SSA82:.*]] = "some.op"(%[[arg40]]) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:   %[[SSA83:.*]] = "grt.repeat.i32"(%[[SSA81]], %[[SSA82]]) ({
// ### Block 2 ###
// CHECK:   ^bb0(%[[arg42:[a-z0-9]*]]: !mgp.tensor<f32>):
// CHECK:     "mef.output"(%[[arg42]]) : (!mgp.tensor<f32>) -> ()
// CHECK:   }) : (si32, !mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:   %[[SSA84:.*]] = "some.op"(%[[SSA83]]) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:   "mef.output"(%[[arg40]], %[[SSA84]]) : (!mef.chain, !mgp.tensor<f32>) -> ()
// CHECK: })

// CHECK: maybeSameAs begin
// CHECK:   %[[arg42]][2] |-> {%[[SSA82]], %[[arg42]][2]}
// CHECK:   %[[SSA83]] |-> {%[[SSA82]], %[[arg42]][2]}
// CHECK: maybeSameAs end

mgp.model @repeat2 : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %cxt: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<f32>) {
  %c = "some.op"(%ch0) : (!mef.chain) -> si32
  %x = "some.op"(%ch0) : (!mef.chain) -> !mgp.tensor<f32>
  %z = grt.repeat.i32 %c (%x : !mgp.tensor<f32>) {
    mef.output %x : !mgp.tensor<f32>
  }
  %out = "some.op"(%z) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
  mef.output %ch0, %out : !mef.chain, !mgp.tensor<f32>
}

// For reference here's the model with vars matching the checks (modulo the block ids).
// CHECK: "mgp.model"() <{{.*}}> ({
// ### Block 0 ###
// CHECK:   %[[SSA77:.*]] = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %[[SSA78:.*]] = "mgp.context.create"() <{numSlots = 2 : ui64}> : () -> !mgp.context
// CHECK:   %[[SSA80:.*]] = "mgp.runtime.create"(%[[SSA77]], %[[SSA78]])
// CHECK:   "mef.output"(%[[SSA80]], %[[SSA78]]
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%[[arg36:[a-z0-9]*]]: !mef.chain, %[[arg37:[a-z0-9]*]]: !mgp.context
// CHECK:   %[[SSA72:.*]] = "some.op"(%[[arg36]]) : (!mef.chain) -> si32
// CHECK:   %[[SSA73:.*]] = "some.op"(%[[arg36]]) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:   %[[SSA74:.*]]:2 = "grt.repeat.i32"(%[[SSA72]], %[[SSA73]], %[[SSA73]]) ({
// ### Block 2 ###
// CHECK:   ^bb0(%[[arg38:[a-z0-9]*]]: !mgp.tensor<f32>, %[[arg39:[a-z0-9]*]]: !mgp.tensor<f32>):
// CHECK:     %[[SSA76:.*]] = "some.op"(%[[arg38]]) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:     "mef.output"(%[[SSA76]], %[[arg39]]) : (!mgp.tensor<f32>, !mgp.tensor<f32>) -> ()
// CHECK:   }) : (si32, !mgp.tensor<f32>, !mgp.tensor<f32>) -> (!mgp.tensor<f32>, !mgp.tensor<f32>)
// CHECK:   %[[SSA75:.*]] = "some.op"(%[[SSA74]]#0) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:   "mef.output"(%[[arg36]], %[[SSA75]]) : (!mef.chain, !mgp.tensor<f32>) -> ()
// CHECK: })

// CHECK: maybeSameAs begin
// CHECK:   %[[arg38]][2] |-> {%[[SSA73]], %[[SSA76]][2]}
// CHECK:   %[[arg39]][2] |-> {%[[SSA73]], %[[arg39]][2]}
// CHECK:   %[[SSA74]]#0 |-> {%[[SSA73]], %[[arg38]][2], %[[SSA76]][2]}
// CHECK:   %[[SSA74]]#1 |-> {%[[SSA73]], %[[arg39]][2]}
// CHECK: maybeSameAs end

mgp.model @repeat3 : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %cxt: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<f32>) {
  %c = "some.op"(%ch0) : (!mef.chain) -> si32
  %x = "some.op"(%ch0) : (!mef.chain) -> !mgp.tensor<f32>
  %z, %dummy = grt.repeat.i32 %c (%x as %x0: !mgp.tensor<f32>, %x as %x1: !mgp.tensor<f32>)  {
    %y = "some.op"(%x0) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
    mef.output %y, %x1 : !mgp.tensor<f32>, !mgp.tensor<f32>
  }
  %out = "some.op"(%z) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
  mef.output %ch0, %out : !mef.chain, !mgp.tensor<f32>
}

//
// Simple grt.while
//

// For reference here's the model with vars matching the checks (modulo the block ids).
// ### Block 0 ###
// CHECK: "mgp.model"() <{{.*}}> ({
// CHECK:   %[[SSA68:.*]] = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %[[SSA69:.*]] = "mgp.context.create"() <{numSlots = 2 : ui64}> : () -> !mgp.context
// CHECK:   %[[SSA71:.*]] = "mgp.runtime.create"(%[[SSA68]], %[[SSA69]])
// CHECK:   "mef.output"(%[[SSA71]], %[[SSA69]]
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%[[arg26:[a-z0-9]*]]: !mef.chain, %[[arg27:[a-z0-9]*]]: !mgp.context
// CHECK:   %[[SSA60:.*]] = "some.op"(%[[arg26]]) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:   %[[SSA61:.*]] = "some.op"(%[[arg26]]) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:   %[[SSA62:.*]]:4 = "grt.while"(%[[arg26]], %[[SSA60]], %[[SSA61]], %[[SSA61]]) ({
// ### Block 2 ###
// CHECK:   ^bb0(%[[arg32:[a-z0-9]*]]: !mef.chain, %[[arg33:[a-z0-9]*]]: !mgp.tensor<f32>, %[[arg34:[a-z0-9]*]]: !mgp.tensor<f32>, %[[arg35:[a-z0-9]*]]: !mgp.tensor<f32>):
// CHECK:     %[[SSA65:.*]] = "some.op"(%[[arg32]]) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:     %[[SSA66:.*]] = "some.op"(%[[arg34]]) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:     %[[SSA67:.*]] = "some.op"(%[[arg32]]) : (!mef.chain) -> i1
// CHECK:     "mef.output"(%[[arg32]], %[[SSA65]], %[[SSA66]], %[[arg35]], %[[SSA67]]) : (!mef.chain, !mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>, i1) -> ()
// CHECK:   }, {
// ### Block 3 ###
// CHECK:   ^bb0(%[[arg28:[a-z0-9]*]]: !mef.chain, %[[arg29:[a-z0-9]*]]: !mgp.tensor<f32>, %[[arg30:[a-z0-9]*]]: !mgp.tensor<f32>, %[[arg31:[a-z0-9]*]]: !mgp.tensor<f32>):
// CHECK:     %[[SSA64:.*]] = "some.op"(%[[arg30]]) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:     "mef.output"(%[[arg28]], %[[arg29]], %[[SSA64]], %[[arg31]]) : (!mef.chain, !mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> ()
// CHECK:   }) : (!mef.chain, !mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> (!mef.chain, !mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>)
// CHECK:   %[[SSA63:.*]] = "some.op"(%[[SSA62]]#1, %[[SSA62]]#2, %[[SSA62]]#3) : (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:   "mef.output"(%[[SSA62]]#0, %[[SSA63]]) : (!mef.chain, !mgp.tensor<f32>) -> ()
// CHECK: })

// CHECK: sameAs begin
// CHECK:   %[[SSA62]]#0 |-> {%[[SSA71]][0], %[[arg26]], %[[arg32]][2], %[[arg28]][3]}
// CHECK:   %[[SSA62]]#1 |-> {%[[SSA65]][2], %[[arg29]][3]}
// CHECK:   %[[SSA62]]#2 |-> {%[[SSA66]][2], %[[arg30]][3]}
// CHECK:   %[[SSA62]]#3 |-> {%[[SSA61]], %[[arg35]][2], %[[arg31]][3]}
// CHECK: sameAs end

// CHECK: maybeSameAs begin
// CHECK:   %[[SSA62]]#0 |-> {%[[SSA71]][0], %[[arg26]], %[[arg32]][2], %[[arg28]][3]}
// CHECK:   %[[SSA62]]#1 |-> {%[[SSA65]][2], %[[arg29]][3]}
// CHECK:   %[[SSA62]]#2 |-> {%[[SSA66]][2], %[[arg30]][3]}
// CHECK:   %[[SSA62]]#3 |-> {%[[SSA61]], %[[arg35]][2], %[[arg31]][3]}
// CHECK: maybeSameAs end

// CHECK: impliesReady begin
// CHECK:   %[[SSA64]][3] |-> {%[[SSA68]][0], %[[SSA69]][0], %[[SSA71]][0], %[[arg26]], %[[SSA61]], %[[arg34]][2], %[[SSA66]][2], %[[arg30]][3]}
// CHECK:   %[[SSA62]]#0 |-> {%[[SSA68]][0], %[[SSA69]][0], %[[SSA71]][0], %[[arg26]], %[[SSA61]], %[[arg32]][2], %[[arg34]][2], %[[arg35]][2], %[[SSA65]][2], %[[SSA66]][2], %[[arg28]][3], %[[arg29]][3], %[[arg30]][3], %[[arg31]][3], %[[SSA64]][3]}
// CHECK: impliesReady end

// No branches, so this relation is empty.
// CHECK: structurallyAtMostOne begin
// CHECK-NEXT: structurallyAtMostOne end

// But the loop cond and body blocks are known not to overlap.
// CHECK: structurallyCannotOverlapRelation begin
// CHECK:   %[[SSA65]][2] |-> {%[[SSA64]][3]}
// CHECK:   %[[SSA66]][2] |-> {%[[SSA64]][3]}
// CHECK:   %[[SSA67]][2] |-> {%[[SSA64]][3]}
// CHECK:   %[[SSA64]][3] |-> {%[[SSA65]][2], %[[SSA66]][2], %[[SSA67]][2]}
// CHECK: structurallyCannotOverlapRelation end

// CHECK: runsLater begin
// CHECK:   %[[SSA65]][2] |-> {%[[SSA63]]}
// CHECK: runsLater end

mgp.model @simple : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %cxt: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<f32>) {
  %x = "some.op"(%ch0) : (!mef.chain) -> !mgp.tensor<f32>
  %y = "some.op"(%ch0) : (!mef.chain) -> !mgp.tensor<f32>
  %ch1, %i, %j, %k = grt.while (%ch0: !mef.chain, %x: !mgp.tensor<f32>, %y: !mgp.tensor<f32>, %y as %z: !mgp.tensor<f32>) {
    %c = "some.op"(%ch0) : (!mef.chain) -> !mgp.tensor<f32>
    %d = "some.op"(%y) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
    %cond = "some.op"(%ch0) : (!mef.chain) -> i1
    mef.output %ch0, %c, %d, %z, %cond : !mef.chain, !mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>, i1
  } do {
    %e = "some.op"(%y) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
    mef.output %ch0, %x, %e, %z : !mef.chain, !mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>
  }
  %l = "some.op"(%i, %j, %k) : (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> !mgp.tensor<f32>
  mef.output %ch1, %l : !mef.chain, !mgp.tensor<f32>
}

//
// Unstable loop
//

// For reference here's the model with vars matching the checks (modulo the block ids).
// ### Block 0 ###
// CHECK: "mgp.model"() <{{.*}}> ({
// CHECK:   %[[SSA56:.*]] = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %[[SSA57:.*]] = "mgp.context.create"() <{numSlots = 2 : ui64}> : () -> !mgp.context
// CHECK:   %[[SSA59:.*]] = "mgp.runtime.create"(%[[SSA56]], %[[SSA57]])
// CHECK:   "mef.output"(%[[SSA59]], %[[SSA57]]
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%[[arg20:[a-z0-9]*]]: !mef.chain, %[[arg21:[a-z0-9]*]]: !mgp.context
// CHECK:   %[[SSA51:.*]] = "some.op"(%[[arg20]]) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:   %[[SSA52:.*]] = "some.op"(%[[arg20]]) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:   %[[SSA53:.*]]:2 = "grt.while"(%[[SSA51]], %[[SSA52]]) ({
// ### Block 2 ###
// CHECK:   ^bb0(%[[arg24:[a-z0-9]*]]: !mgp.tensor<f32>, %[[arg25:[a-z0-9]*]]: !mgp.tensor<f32>):
// CHECK:     %[[SSA55:.*]] = "some.op"() : () -> i1
// CHECK:     "mef.output"(%[[arg24]], %[[arg25]], %[[SSA55]]) : (!mgp.tensor<f32>, !mgp.tensor<f32>, i1) -> ()
// CHECK:   }, {
// ### Block 3 ###
// CHECK:   ^bb0(%[[arg22:[a-z0-9]*]]: !mgp.tensor<f32>, %[[arg23:[a-z0-9]*]]: !mgp.tensor<f32>):
// CHECK:     "mef.output"(%[[arg23]], %[[arg22]]) : (!mgp.tensor<f32>, !mgp.tensor<f32>) -> ()
// CHECK:   }) : (!mgp.tensor<f32>, !mgp.tensor<f32>) -> (!mgp.tensor<f32>, !mgp.tensor<f32>)
// CHECK:   %[[SSA54:.*]] = "some.op"(%[[SSA53]]#0, %[[SSA53]]#1) : (!mgp.tensor<f32>, !mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:   "mef.output"(%[[arg20]], %[[SSA54]]) : (!mef.chain, !mgp.tensor<f32>) -> ()
// CHECK: })

// CHECK: sameAs begin
// CHECK:   %[[SSA53]]#0 |-> {%[[arg24]][2], %[[arg22]][3]}
// CHECK:   %[[SSA53]]#1 |-> {%[[arg25]][2], %[[arg23]][3]}
// CHECK: sameAs end

// CHECK: maybeSameAs begin
// CHECK:   %[[SSA53]]#0 |-> {%[[SSA51]], %[[SSA52]], %[[arg24]][2], %[[arg25]][2], %[[arg22]][3], %[[arg23]][3]}
// CHECK:   %[[SSA53]]#1 |-> {%[[SSA51]], %[[SSA52]], %[[arg24]][2], %[[arg25]][2], %[[arg22]][3], %[[arg23]][3]}
// CHECK: maybeSameAs end

mgp.model @crissCross : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %cxt: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<f32>) {
  %x = "some.op"(%ch0) : (!mef.chain) -> !mgp.tensor<f32>
  %y = "some.op"(%ch0) : (!mef.chain) -> !mgp.tensor<f32>
  %i, %j = grt.while (%x: !mgp.tensor<f32>, %y: !mgp.tensor<f32>) {
    %c = "some.op"() : () -> i1
    mef.output %x, %y, %c : !mgp.tensor<f32>, !mgp.tensor<f32>, i1
  } do {
    mef.output %y, %x: !mgp.tensor<f32>, !mgp.tensor<f32>
  }
  %l = "some.op"(%i, %j) : (!mgp.tensor<f32>, !mgp.tensor<f32>) -> !mgp.tensor<f32>
  mef.output %ch0, %l : !mef.chain, !mgp.tensor<f32>
}

//
// Loop requiring more than one iteration to reach fixed point.
//

// For reference here's the model with vars matching the checks (modulo the block ids).
// ### Block 0 ###
// CHECK: "mgp.model"() <{{.*}}> ({
// CHECK:   %[[SSA47:.*]] = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %[[SSA48:.*]] = "mgp.context.create"() <{numSlots = 2 : ui64}> : () -> !mgp.context
// CHECK:   %[[SSA50:.*]] = "mgp.runtime.create"(%[[SSA47]], %[[SSA48]])
// CHECK:   "mef.output"(%[[SSA50]], %[[SSA48]]
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%[[arg12:[a-z0-9]*]]: !mef.chain, %[[arg13:[a-z0-9]*]]: !mgp.context
// CHECK:   %[[SSA42:.*]] = "some.op"(%[[arg12]]) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:   %[[SSA43:.*]]:3 = "grt.while"(%[[SSA42]], %[[SSA42]], %[[SSA42]]) ({
// ### Block 2 ###
// CHECK:   ^bb0(%[[arg17:[a-z0-9]*]]: !mgp.tensor<f32>, %[[arg18:[a-z0-9]*]]: !mgp.tensor<f32>, %[[arg19:[a-z0-9]*]]: !mgp.tensor<f32>):
// CHECK:     %[[SSA46:.*]] = "some.op"() : () -> i1
// CHECK:     "mef.output"(%[[arg17]], %[[arg18]], %[[arg19]], %[[SSA46]]) : (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>, i1) -> ()
// CHECK:   }, {
// ### Block 3 ###
// CHECK:   ^bb0(%[[arg14:[a-z0-9]*]]: !mgp.tensor<f32>, %[[arg15:[a-z0-9]*]]: !mgp.tensor<f32>, %[[arg16:[a-z0-9]*]]: !mgp.tensor<f32>):
// CHECK:     %[[SSA45:.*]] = "some.op"() : () -> !mgp.tensor<f32>
// CHECK:     "mef.output"(%[[arg15]], %[[arg16]], %[[SSA45]]) : (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> ()
// CHECK:   }) : (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>)
// CHECK:   %[[SSA44:.*]] = "some.op"(%[[SSA43]]#0, %[[SSA43]]#1, %[[SSA43]]#2) : (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:   "mef.output"(%[[arg12]], %[[SSA44]]) : (!mef.chain, !mgp.tensor<f32>) -> ()
// CHECK: })

// CHECK: sameAs begin
// CHECK:   %[[SSA43]]#0 |-> {%[[arg17]][2], %[[arg14]][3]}
// CHECK:   %[[SSA43]]#1 |-> {%[[arg18]][2], %[[arg15]][3]}
// CHECK:   %[[SSA43]]#2 |-> {%[[arg19]][2], %[[arg16]][3]}
// CHECK: sameAs end

// CHECK: maybeSameAs begin
// CHECK:   %[[SSA43]]#0 |-> {%[[SSA42]], %[[arg17]][2], %[[arg18]][2], %[[arg19]][2], %[[arg14]][3], %[[arg15]][3], %[[arg16]][3], %[[SSA45]][3]}
// CHECK:   %[[SSA43]]#1 |-> {%[[SSA42]], %[[arg18]][2], %[[arg19]][2], %[[arg15]][3], %[[arg16]][3], %[[SSA45]][3]}
// CHECK:   %[[SSA43]]#2 |-> {%[[SSA42]], %[[arg19]][2], %[[arg16]][3], %[[SSA45]][3]}
// CHECK: maybeSameAs end

// CHECK: impliesReady begin
// CHECK:   %[[SSA44]] |-> {%[[arg17]][2], %[[arg18]][2], %[[arg19]][2], %[[arg14]][3], %[[arg15]][3], %[[arg16]][3], %[[SSA45]][3], %[[SSA43]]#0, %[[SSA43]]#1, %[[SSA43]]#2}
// CHECK: impliesReady end

mgp.model @shift : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %cxt: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<f32>) {
  %x = "some.op"(%ch0) : (!mef.chain) -> !mgp.tensor<f32>
  %i, %j, %k = grt.while (%x: !mgp.tensor<f32>, %x as %y: !mgp.tensor<f32>, %x as %z: !mgp.tensor<f32>) {
    %c = "some.op"() : () -> i1
    mef.output %x, %y, %z, %c : !mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>, i1
  } do {
    %n = "some.op"() : () -> !mgp.tensor<f32>
    mef.output %y, %z, %n: !mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>
  }
  %l = "some.op"(%i, %j, %k) : (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> !mgp.tensor<f32>
  mef.output %ch0, %l : !mef.chain, !mgp.tensor<f32>
}
