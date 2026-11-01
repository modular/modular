// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mogg-rewrite-opaque-types --mo-to-map --map-to-mogg | FileCheck %s

// An `!mo.opaque` crosses a `mogg.routine` boundary as itself, not as a
// `!mogg.pointer` it is rebuilt from: it carries no shape, so there is
// nothing to reconstruct. Which side it sits on decides whether it gets a body
// block argument at all.
//
// As an INPUT it is destination-passing like a tensor, so it gets a block
// argument -- typed with the opaque itself, and passed straight to the call.
// As a RESULT it is not: the kernel allocates it internally and returns it, so
// there is no caller-allocated destination to point at. It gets no block
// argument, `mogg.call.opaque` grows a real result for it, and the kernel
// hands it out with `mogg.output`.
//
// Tensors keep their existing treatment throughout, so the consumer below
// shows both conventions side by side in one kernel.

// An opaque RESULT: no `inputs`/`outputs` groups at all, just the context.
// CHECK-LABEL: mo.graph @opaque_result
// CHECK: mogg.routine () -> !mo.opaque<"{{.*}}SIMDPair", {S0 = 4 : i64, S1 = 8 : i64}>
// CHECK-NOT: outputs (
// CHECK: context (%[[CTX:[^ ,)]+]])
// CHECK: %[[CALL:.*]] = mogg.call.opaque @{{.*}}MakeSimdPair[P0 = 4 : i64, P1 = 8 : i64] -> !mo.opaque<"{{.*}}SIMDPair", {S0 = 4 : i64, S1 = 8 : i64}>
// CHECK: mogg.output %[[CALL]] : !mo.opaque<"{{.*}}SIMDPair", {S0 = 4 : i64, S1 = 8 : i64}>
mo.graph @opaque_result() -> !mo.opaque<"SIMDPair", {S0 = 4, S1 = 8}> {
  %pair = mo.custom [symbol = "make_simd_pair", device = #M.device_ref<"cpu", 0>, parameters = {}] () : () -> (!mo.opaque<"SIMDPair", {S0 = 4, S1 = 8}>)
  mo.output %pair : !mo.opaque<"SIMDPair", {S0 = 4, S1 = 8}>
}

// An opaque INPUT beside a tensor output: the opaque is its own block arg and
// reaches the call directly, while the tensor still goes through
// `mogg.shape.create` / `mogg.tensor.create` from a pointer.
// CHECK-LABEL: mo.graph @opaque_input
// CHECK: mogg.routine (%{{.*}} : !mo.opaque<"{{.*}}SIMDPair", {S0 = 4 : i64, S1 = 8 : i64}>) -> !mo.tensor<[12], si32>
// CHECK: inputs (%[[IN:[^ ,)]+]]: !mo.opaque<"{{.*}}SIMDPair", {S0 = 4 : i64, S1 = 8 : i64}>)
// CHECK: outputs (%[[OP:[^ ,)]+]])
// CHECK: context (%[[CTX:[^ ,)]+]])
// CHECK: %[[SH:.*]] = mogg.shape.create() <shape = #mosh<ape[12]> : !mosh.ape>
// CHECK: %[[OT:.*]] = mogg.tensor.create(%[[OP]], %[[SH]]) {{.*}} : !mogg.tensor
// CHECK: mogg.call.opaque @{{.*}}ParameterizedOpaqueType[P0 = 4 : i64] inputs(%[[IN]] : !mo.opaque<"{{.*}}SIMDPair", {S0 = 4 : i64, S1 = 8 : i64}>) outputs(%[[OT]] : !mogg.tensor)
mo.graph @opaque_input(
    %pair: !mo.opaque<"SIMDPair", {S0 = 4, S1 = 8}>
) -> !mo.tensor<[12], si32> {
  %out = mo.custom [symbol = "kernel_with_parameterized_opaque", device = #M.device_ref<"cpu", 0>] (%pair) : (!mo.opaque<"SIMDPair", {S0 = 4, S1 = 8}>) -> (!mo.tensor<[12], si32>)
  mo.output %out : !mo.tensor<[12], si32>
}
