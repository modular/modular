// RUN: mo-opt %s -mogg-add-try-except-blocks 2>&1 | FileCheck %s --check-prefix=OFF
// RUN: mo-opt %s -mogg-add-profile-scope-tracing 2>&1 | FileCheck %s --check-prefix=ON
// RUN: mo-opt %s -mogg-add-profile-scope-tracing="profile-scope-tracing-trace-unlabelled=true" 2>&1 | FileCheck %s --check-prefix=TRACE

// OFF-LABEL: mogg.func @region_952 private
// OFF-NOT: mogg.profile_scope

// ON-LABEL: mogg.func @region_952 private
// ON: mogg.profile_scope.begin "[draft_forward] region_952"
// ON: mogg.profile_scope.end

// TRACE-LABEL: mogg.func @region_952 private
// TRACE: mogg.profile_scope.begin "[draft_forward] region_952"
// TRACE: mogg.profile_scope.end

mogg.func @region_952 private dispatch(deferred) (%arg0: !mogg.async<!mef.chain>, %arg1: !mogg.async<!mgp.buffer>, %arg2: !mogg.async<!mgp.buffer>, %arg3: !mogg.async<!mgp.device_context<<"cpu", 0>>>) -> !mef.chain attributes {abi = #mogg.abi<MGP>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.cast">>} {
  %0 = mogg.async.unpack(%arg1) : !mgp.buffer
  %1 = mogg.async.unpack(%arg2) : !mgp.buffer
  %2 = mogg.async.unpack(%arg3) : !mgp.device_context<<"cpu", 0>>
  %3 = mogg.tensor.__init__(%0) io = Input, mut = false : !mgp.buffer -> !mo.tensor<[10], f32, {io = #mo.io<Input>, mut = false}>
  %4 = mogg.tensor.__init__(%1) io = Output, mut = true : !mgp.buffer -> !mo.tensor<[10], f32, {io = #mo.io<Output>, mut = true}>
  mogg.call @cast_mogg %3, %4, %2 : !mo.tensor<[10], f32, {io = #mo.io<Input>, mut = false}>, !mo.tensor<[10], f32, {io = #mo.io<Output>, mut = true}>, !mgp.device_context<<"cpu", 0>> loc(#mogg.profile_scope<"draft_forward", loc(unknown)>)
  %5 = mogg.async.ready
  mogg.output %5 : !mef.chain
}

mogg.func @cast_mogg private dispatch(deferred) (%arg0: !mo.tensor<[10], f32, {io = #mo.io<Input>, mut = false}>, %arg1: !mo.tensor<[10], f32, {io = #mo.io<Output>, mut = true}>, %arg2: !mogg.context) attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.cast">>} {
  mogg.output
}

// ---

// ON-LABEL: mogg.func @unlabeled_region private
// ON-NOT: mogg.profile_scope.begin "unlabeled_region"

// TRACE-LABEL: mogg.func @unlabeled_region private
// TRACE: mogg.profile_scope.begin "unlabeled_region"
// TRACE: mogg.profile_scope.end

mogg.func @unlabeled_region private dispatch(deferred) (%arg0: !mogg.async<!mef.chain>, %arg1: !mogg.async<!mgp.buffer>, %arg2: !mogg.async<!mgp.buffer>, %arg3: !mogg.async<!mgp.device_context<<"cpu", 0>>>) -> !mef.chain attributes {abi = #mogg.abi<MGP>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.cast">>} {
  %0 = mogg.async.unpack(%arg1) : !mgp.buffer
  %1 = mogg.async.unpack(%arg2) : !mgp.buffer
  %2 = mogg.async.unpack(%arg3) : !mgp.device_context<<"cpu", 0>>
  %3 = mogg.tensor.__init__(%0) io = Input, mut = false : !mgp.buffer -> !mo.tensor<[10], f32, {io = #mo.io<Input>, mut = false}>
  %4 = mogg.tensor.__init__(%1) io = Output, mut = true : !mgp.buffer -> !mo.tensor<[10], f32, {io = #mo.io<Output>, mut = true}>
  mogg.call @cast_mogg_unlabeled %3, %4, %2 : !mo.tensor<[10], f32, {io = #mo.io<Input>, mut = false}>, !mo.tensor<[10], f32, {io = #mo.io<Output>, mut = true}>, !mgp.device_context<<"cpu", 0>>
  %5 = mogg.async.ready
  mogg.output %5 : !mef.chain
}

mogg.func @cast_mogg_unlabeled private dispatch(deferred) (%arg0: !mo.tensor<[10], f32, {io = #mo.io<Input>, mut = false}>, %arg1: !mo.tensor<[10], f32, {io = #mo.io<Output>, mut = true}>, %arg2: !mogg.context) attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.cast">>} {
  mogg.output
}

// ---

// A host-side bookkeeping op with no device dispatch of its own (here,
// mogg.tensor.__init__) must never split two device-dispatching ops that
// share the same profile_scope layer into separate runs, even if that
// bookkeeping op happens to carry a stale/foreign layer -- e.g. a CSE'd
// cached-buffer lookup shared across non-adjacent profile_scope() call
// sites, whose location was never re-attributed to the step it's actually
// read from here. One continuous range should cover both calls, not two.
#glueStepA = loc(#mogg.profile_scope<"draft_step_1", #mogg.profile_scope<"draft_forward", loc(unknown)>>)
#glueStepForeign = loc(#mogg.profile_scope<"draft_step_0", #mogg.profile_scope<"draft_forward", loc(unknown)>>)

// ON-LABEL: mogg.func @region_glue_mismatch private
// ON: mogg.profile_scope.begin "[draft_forward/draft_step_1] region_glue_mismatch"
// ON: mogg.call @cast_step
// ON-NOT: mogg.profile_scope.end
// ON: mogg.tensor.__init__
// ON-NOT: mogg.profile_scope.begin
// ON: mogg.call @cast_step
// ON: mogg.profile_scope.end

mogg.func @region_glue_mismatch private dispatch(deferred) (%arg0: !mogg.async<!mef.chain>, %arg1: !mogg.async<!mgp.buffer>, %arg2: !mogg.async<!mgp.buffer>, %arg3: !mogg.async<!mgp.device_context<<"cpu", 0>>>) -> !mef.chain attributes {abi = #mogg.abi<MGP>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.cast">>} {
  %g0 = mogg.async.unpack(%arg1) : !mgp.buffer
  %g1 = mogg.async.unpack(%arg2) : !mgp.buffer
  %g2 = mogg.async.unpack(%arg3) : !mgp.device_context<<"cpu", 0>>
  %g3 = mogg.tensor.__init__(%g0) io = Input, mut = false : !mgp.buffer -> !mo.tensor<[10], f32, {io = #mo.io<Input>, mut = false}>
  %g4 = mogg.tensor.__init__(%g1) io = Output, mut = true : !mgp.buffer -> !mo.tensor<[10], f32, {io = #mo.io<Output>, mut = true}>
  mogg.call @cast_step %g3, %g4, %g2 : !mo.tensor<[10], f32, {io = #mo.io<Input>, mut = false}>, !mo.tensor<[10], f32, {io = #mo.io<Output>, mut = true}>, !mgp.device_context<<"cpu", 0>> loc(#glueStepA)
  %g5 = mogg.tensor.__init__(%g0) io = Input, mut = false : !mgp.buffer -> !mo.tensor<[10], f32, {io = #mo.io<Input>, mut = false}> loc(#glueStepForeign)
  mogg.call @cast_step %g3, %g4, %g2 : !mo.tensor<[10], f32, {io = #mo.io<Input>, mut = false}>, !mo.tensor<[10], f32, {io = #mo.io<Output>, mut = true}>, !mgp.device_context<<"cpu", 0>> loc(#glueStepA)
  %g6 = mogg.async.ready
  mogg.output %g6 : !mef.chain
}

mogg.func @cast_step private dispatch(deferred) (%arg0: !mo.tensor<[10], f32, {io = #mo.io<Input>, mut = false}>, %arg1: !mo.tensor<[10], f32, {io = #mo.io<Output>, mut = true}>, %arg2: !mogg.context) attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.cast">>} {
  mogg.output
}

// ---

// MergeIdenticalFuncs can canonicalize structurally-identical, non-nested
// profile scope call sites (e.g. the same matmul kernel invoked from
// draft_forward and target_forward) onto one shared surviving FuncOp.
// Every merged duplicate's location is fused onto the stub's own location.
// Locations of the ops *inside* that body still carry an arbitrary
// original, single-scope location
// In-body bracketing must be suppressed in this case, otherwise
// the generated range would show one scope and the ops inside
// would be labeled with a different scope.
#shared_a = #mogg.profile_scope<"draft_forward", loc(unknown)>
#shared_b = #mogg.profile_scope<"target_forward", loc(unknown)>

// ON-LABEL: mogg.func @region_shared private
// ON-NOT: mogg.profile_scope

mogg.func @region_shared private dispatch(deferred) (%arg0: !mogg.async<!mef.chain>, %arg1: !mogg.async<!mgp.buffer>, %arg2: !mogg.async<!mgp.buffer>, %arg3: !mogg.async<!mgp.device_context<<"cpu", 0>>>) -> !mef.chain attributes {abi = #mogg.abi<MGP>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.cast">>} {
  %sh0 = mogg.async.unpack(%arg1) : !mgp.buffer
  %sh1 = mogg.async.unpack(%arg2) : !mgp.buffer
  %sh2 = mogg.async.unpack(%arg3) : !mgp.device_context<<"cpu", 0>>
  %sh3 = mogg.tensor.__init__(%sh0) io = Input, mut = false : !mgp.buffer -> !mo.tensor<[10], f32, {io = #mo.io<Input>, mut = false}>
  %sh4 = mogg.tensor.__init__(%sh1) io = Output, mut = true : !mgp.buffer -> !mo.tensor<[10], f32, {io = #mo.io<Output>, mut = true}>
  mogg.call @cast_shared %sh3, %sh4, %sh2 : !mo.tensor<[10], f32, {io = #mo.io<Input>, mut = false}>, !mo.tensor<[10], f32, {io = #mo.io<Output>, mut = true}>, !mgp.device_context<<"cpu", 0>> loc(#shared_a)
  %sh5 = mogg.async.ready
  mogg.output %sh5 : !mef.chain
} loc(fused[#shared_a, #shared_b])

mogg.func @cast_shared private dispatch(deferred) (%arg0: !mo.tensor<[10], f32, {io = #mo.io<Input>, mut = false}>, %arg1: !mo.tensor<[10], f32, {io = #mo.io<Output>, mut = true}>, %arg2: !mogg.context) attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.cast">>} {
  mogg.output
}

// Color is region-only: a single consistent color on every op in a run
// resolves to a `color` attribute on the begin op.
// ON-LABEL: mogg.func @region_color private
// ON: mogg.profile_scope.begin "[colored] region_color" color = "orange"

mogg.func @region_color private dispatch(deferred) (%arg0: !mogg.async<!mef.chain>, %arg1: !mogg.async<!mgp.buffer>, %arg2: !mogg.async<!mgp.buffer>, %arg3: !mogg.async<!mgp.device_context<<"cpu", 0>>>) -> !mef.chain attributes {abi = #mogg.abi<MGP>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.cast">>} {
  %c0 = mogg.async.unpack(%arg1) : !mgp.buffer
  %c1 = mogg.async.unpack(%arg2) : !mgp.buffer
  %c2 = mogg.async.unpack(%arg3) : !mgp.device_context<<"cpu", 0>>
  %c3 = mogg.tensor.__init__(%c0) io = Input, mut = false : !mgp.buffer -> !mo.tensor<[10], f32, {io = #mo.io<Input>, mut = false}>
  %c4 = mogg.tensor.__init__(%c1) io = Output, mut = true : !mgp.buffer -> !mo.tensor<[10], f32, {io = #mo.io<Output>, mut = true}>
  mogg.call @cast_mogg_color %c3, %c4, %c2 : !mo.tensor<[10], f32, {io = #mo.io<Input>, mut = false}>, !mo.tensor<[10], f32, {io = #mo.io<Output>, mut = true}>, !mgp.device_context<<"cpu", 0>> loc(#mogg.profile_scope<"colored", loc(unknown), color = "orange">)
  %c5 = mogg.async.ready
  mogg.output %c5 : !mef.chain
}

mogg.func @cast_mogg_color private dispatch(deferred) (%arg0: !mo.tensor<[10], f32, {io = #mo.io<Input>, mut = false}>, %arg1: !mo.tensor<[10], f32, {io = #mo.io<Output>, mut = true}>, %arg2: !mogg.context) attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.cast">>} {
  mogg.output
}

// ---

// Disagreeing colors on ops in the same run resolve to no color attribute,
// rather than guessing one.
// ON-LABEL: mogg.func @region_color_conflict private
// ON: mogg.profile_scope.begin "[conflict] region_color_conflict"

mogg.func @region_color_conflict private dispatch(deferred) (%arg0: !mogg.async<!mef.chain>, %arg1: !mogg.async<!mgp.buffer>, %arg2: !mogg.async<!mgp.buffer>, %arg3: !mogg.async<!mgp.device_context<<"cpu", 0>>>) -> !mef.chain attributes {abi = #mogg.abi<MGP>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.cast">>} {
  %x0 = mogg.async.unpack(%arg1) : !mgp.buffer
  %x1 = mogg.async.unpack(%arg2) : !mgp.buffer
  %x2 = mogg.async.unpack(%arg3) : !mgp.device_context<<"cpu", 0>>
  %x3 = mogg.tensor.__init__(%x0) io = Input, mut = false : !mgp.buffer -> !mo.tensor<[10], f32, {io = #mo.io<Input>, mut = false}>
  %x4 = mogg.tensor.__init__(%x1) io = Output, mut = true : !mgp.buffer -> !mo.tensor<[10], f32, {io = #mo.io<Output>, mut = true}>
  mogg.call @cast_mogg_conflict1 %x3, %x4, %x2 : !mo.tensor<[10], f32, {io = #mo.io<Input>, mut = false}>, !mo.tensor<[10], f32, {io = #mo.io<Output>, mut = true}>, !mgp.device_context<<"cpu", 0>> loc(#mogg.profile_scope<"conflict", loc(unknown), color = "orange">)
  mogg.call @cast_mogg_conflict2 %x3, %x4, %x2 : !mo.tensor<[10], f32, {io = #mo.io<Input>, mut = false}>, !mo.tensor<[10], f32, {io = #mo.io<Output>, mut = true}>, !mgp.device_context<<"cpu", 0>> loc(#mogg.profile_scope<"conflict", loc(unknown), color = "blue">)
  %x5 = mogg.async.ready
  mogg.output %x5 : !mef.chain
}

mogg.func @cast_mogg_conflict1 private dispatch(deferred) (%arg0: !mo.tensor<[10], f32, {io = #mo.io<Input>, mut = false}>, %arg1: !mo.tensor<[10], f32, {io = #mo.io<Output>, mut = true}>, %arg2: !mogg.context) attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.cast">>} {
  mogg.output
}

mogg.func @cast_mogg_conflict2 private dispatch(deferred) (%arg0: !mo.tensor<[10], f32, {io = #mo.io<Input>, mut = false}>, %arg1: !mo.tensor<[10], f32, {io = #mo.io<Output>, mut = true}>, %arg2: !mogg.context) attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.cast">>} {
  mogg.output
}

// ---

// Two adjacent leaf runs nested under the same outer scope get an ancestor
// range bracketing both, in addition to each step's own narrower range.
// The color on the outer scope alone still resolves on both the ancestor and
// the leaf ranges because color propagates through every active scope layer.
#step0 = loc(#mogg.profile_scope<"draft_step_0", #mogg.profile_scope<"draft_forward", loc(unknown), color = "orange">>)
#step1 = loc(#mogg.profile_scope<"draft_step_1", #mogg.profile_scope<"draft_forward", loc(unknown), color = "orange">>)

// ON-LABEL: mogg.func @region_ancestor private
// ON: mogg.profile_scope.begin "[draft_forward] region_ancestor" color = "orange"
// ON: mogg.profile_scope.begin "[draft_forward/draft_step_0] region_ancestor" color = "orange"
// ON: mogg.call @cast_step
// ON: mogg.profile_scope.end
// ON: mogg.profile_scope.begin "[draft_forward/draft_step_1] region_ancestor" color = "orange"
// ON: mogg.call @cast_step
// ON: mogg.profile_scope.end
// ON: mogg.profile_scope.end

mogg.func @region_ancestor private dispatch(deferred) (%arg0: !mogg.async<!mef.chain>, %arg1: !mogg.async<!mgp.buffer>, %arg2: !mogg.async<!mgp.buffer>, %arg3: !mogg.async<!mgp.device_context<<"cpu", 0>>>) -> !mef.chain attributes {abi = #mogg.abi<MGP>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.cast">>} {
  %a0 = mogg.async.unpack(%arg1) : !mgp.buffer
  %a1 = mogg.async.unpack(%arg2) : !mgp.buffer
  %a2 = mogg.async.unpack(%arg3) : !mgp.device_context<<"cpu", 0>>
  %a3 = mogg.tensor.__init__(%a0) io = Input, mut = false : !mgp.buffer -> !mo.tensor<[10], f32, {io = #mo.io<Input>, mut = false}>
  %a4 = mogg.tensor.__init__(%a1) io = Output, mut = true : !mgp.buffer -> !mo.tensor<[10], f32, {io = #mo.io<Output>, mut = true}>
  mogg.call @cast_step %a3, %a4, %a2 : !mo.tensor<[10], f32, {io = #mo.io<Input>, mut = false}>, !mo.tensor<[10], f32, {io = #mo.io<Output>, mut = true}>, !mgp.device_context<<"cpu", 0>> loc(#step0)
  mogg.call @cast_step %a3, %a4, %a2 : !mo.tensor<[10], f32, {io = #mo.io<Input>, mut = false}>, !mo.tensor<[10], f32, {io = #mo.io<Output>, mut = true}>, !mgp.device_context<<"cpu", 0>> loc(#step1)
  %a5 = mogg.async.ready
  mogg.output %a5 : !mef.chain
}

// ---

// A single nested run with no sibling under the same outer scope gets no
// ancestor range -- it would just duplicate the leaf range's exact span under
// a less specific name.
#step_only = loc(#mogg.profile_scope<"draft_step_0", #mogg.profile_scope<"draft_forward", loc(unknown)>>)

// ON-LABEL: mogg.func @region_ancestor_single private
// ON-NOT: mogg.profile_scope.begin "[draft_forward] region_ancestor_single"
// ON: mogg.profile_scope.begin "[draft_forward/draft_step_0] region_ancestor_single"

mogg.func @region_ancestor_single private dispatch(deferred) (%arg0: !mogg.async<!mef.chain>, %arg1: !mogg.async<!mgp.buffer>, %arg2: !mogg.async<!mgp.buffer>, %arg3: !mogg.async<!mgp.device_context<<"cpu", 0>>>) -> !mef.chain attributes {abi = #mogg.abi<MGP>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.cast">>} {
  %s0 = mogg.async.unpack(%arg1) : !mgp.buffer
  %s1 = mogg.async.unpack(%arg2) : !mgp.buffer
  %s2 = mogg.async.unpack(%arg3) : !mgp.device_context<<"cpu", 0>>
  %s3 = mogg.tensor.__init__(%s0) io = Input, mut = false : !mgp.buffer -> !mo.tensor<[10], f32, {io = #mo.io<Input>, mut = false}>
  %s4 = mogg.tensor.__init__(%s1) io = Output, mut = true : !mgp.buffer -> !mo.tensor<[10], f32, {io = #mo.io<Output>, mut = true}>
  mogg.call @cast_step_single %s3, %s4, %s2 : !mo.tensor<[10], f32, {io = #mo.io<Input>, mut = false}>, !mo.tensor<[10], f32, {io = #mo.io<Output>, mut = true}>, !mgp.device_context<<"cpu", 0>> loc(#step_only)
  %s5 = mogg.async.ready
  mogg.output %s5 : !mef.chain
}

mogg.func @cast_step_single private dispatch(deferred) (%arg0: !mo.tensor<[10], f32, {io = #mo.io<Input>, mut = false}>, %arg1: !mo.tensor<[10], f32, {io = #mo.io<Output>, mut = true}>, %arg2: !mogg.context) attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.cast">>} {
  mogg.output
}
