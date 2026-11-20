// RUN: mo-opt --gc-pipeline="fuse-jit-kernels=false" %s | FileCheck %s

mo.graph @updside_down(
    %arg0: !mo.tensor<[256], f32>
) -> (!mo.tensor<[256], f32>) {
    %relu = mo.relu(%add) : !mo.tensor<[256], f32>
    %add = mo.add(%mul, %arg0) : !mo.tensor<[256], f32>
    %mul = mo.mul(%arg0, %arg0) : !mo.tensor<[256], f32>
    mo.output %relu : !mo.tensor<[256], f32>
}

// CHECK-LABEL: mef.callable @updside_down$exec
// The three ops fuse into one kernel, so the dependency order this graph's
// reversed source order tests shows up inside a single summary.
// CHECK: "mgp.generic.execute"{{.*}}dSummary = "{{.*}}Elementwise(mo.mul, mo.add, mo.relu){{.*}}"
// CHECK: mef.model @updside_down
