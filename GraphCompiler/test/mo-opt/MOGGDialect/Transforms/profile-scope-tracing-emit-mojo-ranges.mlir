// RUN: mo-opt %s --mogg-import-packages -mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg --mogg-pipeline -mogg-to-mgp -mogg-outline-async-routines -mogg-add-profile-scope-tracing -mogg-add-try-except-blocks --mogg-emit-mojo -mogg-dump-mojo 2>&1 | FileCheck %s

// Full-pipeline regression for the in-region range bracketing mechanism.
// After outlining to FuncOp and enabling profile-scope-tracing, the
// emitted Mojo must contain a Trace call whose name contains the kernel stub
// symbol ("__..._async"), confirming in-region range bracketing is emitted,
// not just the per-kernel trace-name suffix from WithTraceOp.

// CHECK: var [[RANGE:.*]] = max.runtime.tracing.Trace[max.runtime.tracing.TraceLevel.OP, target = StaticString("cpu")]("[draft_forward] __add_async")
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
