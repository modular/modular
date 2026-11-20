// RUN: mo-opt %s --gc-pipeline-pre-jit="dump-mojo-text=true" -o /dev/null 2>&1 | FileCheck %s

// Full-pipeline regression: color is region-only, so even though the
// profile scope carries color="orange", the per-kernel trace name from
// WithTraceOp::emitMojo (the per-kernel trace-name suffix) must not include a color
// The suffix still carries the scope label so the kernel is still labeled.

// CHECK: with max.runtime.tracing.Trace[max.runtime.tracing.TraceLevel.OP, target = StaticString("cpu")]("Elementwise(relu, add) [colored_scope]"
// CHECK-NOT: color=
// CHECK-NOT: Color(

mo.graph @simple_fusion(
    %x: !mo.tensor<[10], f32>,
    %y: !mo.tensor<[10], f32>
) -> !mo.tensor<[10], f32> {
    %relu = mo.relu(%x): !mo.tensor<[10], f32> loc(#mogg.profile_scope<"colored_scope", loc(unknown), color = "orange">)
    %add = mo.add(%relu, %y): !mo.tensor<[10], f32> loc(#mogg.profile_scope<"colored_scope", loc(unknown), color = "orange">)
    mo.output %add: !mo.tensor<[10], f32>
}
