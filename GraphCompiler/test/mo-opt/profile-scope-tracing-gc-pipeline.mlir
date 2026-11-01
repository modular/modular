// RUN: mo-opt %s --gc-pipeline-pre-jit="enable-profile-scope-tracing=true dump-mojo-text=true" -o /dev/null 2>&1 | FileCheck %s

// Regression for wiring `enable-profile-scope-tracing` through the actual
// registered pipeline entry point (what InferenceSession.compile() and the
// MODULAR_MAX_DEBUG_PROFILE_SCOPE_TRACING env var drive), not just the
// standalone `-mogg-add-try-except-blocks` pass. GraphCompiler.cpp's
// buildMgpPipeline must copy the option from MgpPipelineOptions into the
// EmitMojoOptions it builds before constructing the EmitMojo pass, or the
// flag never reaches AddTryExceptBlocks's in-region range bracketing here
// even though a standalone single-pass invocation of that same pass works
// fine in isolation.

// CHECK: var [[RANGE:.*]] = max.runtime.tracing.Trace[max.runtime.tracing.TraceLevel.OP, target = StaticString("cpu")]("[draft_forward]
// CHECK-NEXT: [[RANGE]].start()
// CHECK: [[RANGE]].end()

mo.graph @simple_fusion(
    %x: !mo.tensor<[10], f32>,
    %y: !mo.tensor<[10], f32>
) -> !mo.tensor<[10], f32> {
    %relu = mo.relu(%x): !mo.tensor<[10], f32> loc(#mogg.profile_scope<"draft_forward", loc(unknown)>)
    %add = mo.add(%relu, %y): !mo.tensor<[10], f32> loc(#mogg.profile_scope<"draft_forward", loc(unknown)>)
    mo.output %add: !mo.tensor<[10], f32>
}
