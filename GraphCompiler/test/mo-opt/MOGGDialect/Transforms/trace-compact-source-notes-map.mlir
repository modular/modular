// RUN: mo-opt %s --gc-pipeline-pre-jit="dump-mojo-text=true" -o /dev/null 2>&1 | FileCheck %s

// The MAP-ABI counterpart of trace-compact-source-notes.mlir, which drives
// `--mo-to-mogg` directly and so only ever covers the legacy stub.
//
// Test that WithTraceOp emits compact Python source notes as the Trace detail
// argument. Long paths are trimmed to the last 3 components, and only the
// bottom 10 (controlled by envar MODULAR_TRACE_MAX_FRAMES=N) frames of each
// traceback are included.
//
// The CHECK lines only match the detail string content, not the surrounding
// generated code, to avoid coupling to unrelated pipeline changes.

// --- Single short-path source note: leading \n separates from kernel name ---
// CHECK: Trace[{{.*}}]("relu", "\nmodel.py:10:forward\n"

// --- No source notes: no detail string between name and task_id ---
// CHECK: Trace[{{.*}}]("relu", task_id

// --- Deep stack with long bazel paths: all frames shown (default 10),
// ---   paths trimmed to last 3 components ---
// CHECK: Trace[{{.*}}]("relu", "\ngraph/ops/custom.py:173:inplace_custom
// CHECK-SAME: nn/comm/allreduce.py:42:__call__
// CHECK-SAME: architectures/deepseekV3/model.py:187:build_graph
// CHECK-SAME: comm/ep/ep_manager.py:656:_build_ep_init_graph
// CHECK-SAME: "

// Each graph uses a different tensor shape to prevent kernel stub dedup.

mo.graph @single_source_note(
    %arg0: !mo.tensor<[4], f32>
) -> (!mo.tensor<[4], f32>) {
  %0 = mo.relu(%arg0) : !mo.tensor<[4], f32> loc(fused<"  File \22model.py\22, line 10, in forward\n    x = relu(x)\n">["model.py":10:0])
  mo.output %0 : !mo.tensor<[4], f32>
}

mo.graph @no_source_notes(
    %arg0: !mo.tensor<[8], f32>
) -> (!mo.tensor<[8], f32>) {
  %0 = mo.relu(%arg0) : !mo.tensor<[8], f32>
  mo.output %0 : !mo.tensor<[8], f32>
}

mo.graph @deep_stack_trimmed(
    %arg0: !mo.tensor<[16], f32>
) -> (!mo.tensor<[16], f32>) {
  %0 = mo.relu(%arg0) : !mo.tensor<[16], f32> loc(fused<"  File \22/home/ubuntu/.cache/bazel/_bazel_ubuntu/aa43b6dd/execroot/_main/bazel-out/k8-dbg/bin/smoke_test.runfiles/_main/max/python/max/graph/ops/custom.py\22, line 173, in inplace_custom\n    (*results, out_chain), custom_op = graph._add_op(\n  File \22/home/ubuntu/.cache/bazel/_bazel_ubuntu/aa43b6dd/execroot/_main/bazel-out/k8-dbg/bin/smoke_test.runfiles/_main/max/python/max/nn/comm/allreduce.py\22, line 42, in __call__\n    return ops.allreduce.sum(\n  File \22/home/ubuntu/.cache/bazel/_bazel_ubuntu/aa43b6dd/execroot/_main/bazel-out/k8-dbg/bin/smoke_test.runfiles/_main/max/python/max/pipelines/architectures/deepseekV3/model.py\22, line 187, in build_graph\n    out = self.allreduce(x)\n  File \22/home/ubuntu/.cache/bazel/_bazel_ubuntu/aa43b6dd/execroot/_main/bazel-out/k8-dbg/bin/smoke_test.runfiles/_main/max/python/max/nn/comm/ep/ep_manager.py\22, line 656, in _build_ep_init_graph\n    dev_ptrs, my_rank = call_ep_init(\n">["ep_manager.py":656:0])
  mo.output %0 : !mo.tensor<[16], f32>
}
