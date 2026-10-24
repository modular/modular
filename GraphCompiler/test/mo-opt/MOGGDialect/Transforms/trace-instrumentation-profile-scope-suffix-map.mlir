// RUN: mo-opt %s --gc-pipeline-pre-jit="dump-mojo-text=true" -o /dev/null 2>&1 | FileCheck %s

// The MAP-ABI counterpart of trace-instrumentation-profile-scope-suffix.mlir,
// which drives `--mo-to-mogg` directly and so only ever covers the legacy stub.
// CHECK: with max.runtime.tracing.Trace[max.runtime.tracing.TraceLevel.OP, target = StaticString("cpu")]("Elementwise(relu, add) [my_scope]"{{.*}}):

mo.graph @simple_fusion(
    %x: !mo.tensor<[10], f32>,
    %y: !mo.tensor<[10], f32>
) -> !mo.tensor<[10], f32> {
    %relu = mo.relu(%x): !mo.tensor<[10], f32> loc(#mogg.profile_scope<"my_scope", loc(unknown)>)
    %add = mo.add(%relu, %y): !mo.tensor<[10], f32> loc(#mogg.profile_scope<"my_scope", loc(unknown)>)
    mo.output %add: !mo.tensor<[10], f32>
}
