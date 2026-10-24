// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-allocate-accumulating-buffers --map-to-mogg --mogg-outline-routines | FileCheck %s

// OutlineRoutines moves each `mogg.routine` body into a module-level
// `mogg.func` labeled with the MOGG ABI, and replaces the kernel body
// with a single `mogg.call` forwarding the pointer / dim / context block
// args. The `mogg.routine` container is kept for `MOGGToMGP`.
//
// The stub is named `__<ops it fuses>`; two kernels that fuse the same ops share
// a preferred name, so the symbol table appends a numeric suffix to the second
// (`..._0`), as `@dyn` and `@one_plain` show below.

// Plain output: the kernel body becomes one `mogg.call`; the shape/tensor
// reconstruction + `mogg.call.opaque` move into the MAP-ABI stub.
// CHECK-LABEL: mo.graph @one_plain
// CHECK: mogg.routine (%{{.*}} : !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
// CHECK:   inputs (%[[IP:[^ ,)]+]])
// CHECK:   outputs (%[[OP:[^ ,)]+]])
// CHECK:   context (%[[C:[^ ,)]+]]) attributes {device = #M.device_ref<"cpu", 0>, fusionInfo = {{.*}}} {
// CHECK:     mogg.call @[[STUB0:__[A-Za-z0-9_]+]] %[[IP]], %[[OP]], %[[C]] : !mogg.pointer, !mogg.pointer, !mogg.context
// CHECK-NOT: mogg.call.opaque
// CHECK: mogg.func @[[STUB0]] private (%{{.*}}: !mogg.pointer, %{{.*}}: !mogg.pointer, %{{.*}}: !mogg.context) attributes {abi = #mogg.abi<MOGG>,
// CHECK:   %[[ISH:.*]] = mogg.shape.create() {{.*}}{{$}}
// CHECK:   %[[IT:.*]] = mogg.tensor.create(%{{.*}}, %[[ISH]]) {{.*}}mut = false{{.*}} : !mogg.tensor
// CHECK:   %[[OT:.*]] = mogg.tensor.create(%{{.*}}, %{{.*}}) {{.*}}mut = true{{.*}} : !mogg.tensor
// CHECK:   mogg.call.opaque @{{.*}}MapLowerOnePlainOutput inputs(%[[IT]] : !mogg.tensor) outputs(%[[OT]] : !mogg.tensor) context(%{{.*}} : !mogg.context)
// CHECK:   mogg.output
mo.graph @one_plain(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom [symbol = "map_lower_one_plain_output", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// Fused output: the epilogue lambda + `mogg.tensor.create.epilogue` ride inside
// the stub body alongside the call.
// CHECK-LABEL: mo.graph @one_fused
// CHECK:   inputs (%[[IP:[^ ,)]+]])
// CHECK:   outputs (%[[OP:[^ ,)]+]])
// CHECK:   context (%[[C:[^ ,)]+]]) attributes {device = #M.device_ref<"cpu", 0>, fusionInfo = {{.*}}} {
// CHECK:     mogg.call @[[STUB1:__[A-Za-z0-9_]+]] %[[IP]], %[[OP]], %[[C]] : !mogg.pointer, !mogg.pointer, !mogg.context
// CHECK: mogg.func @[[STUB1]] private {{.*}} attributes {abi = #mogg.abi<MOGG>,
// CHECK:   %{{.*}} = mogg.tensor.create.epilogue %{{.*}} (%{{.*}} : !mogg.simd<f32>, %{{.*}} : !map.index<2>) capturing {%[[CAP:[^ ,)]+]] = %{{.*}} : !mogg.tensor} ptr %{{.*}} {{.*}} {
// CHECK:     mogg.tensor.store %[[CAP]][%{{.*}}] = %{{.*}}
// CHECK:   }
// CHECK:   mogg.call.opaque @{{.*}}MapLowerOneFusedOutput
// CHECK:   mogg.output
mo.graph @one_fused(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// Dynamic dim: the `!mogg.dim` block arg is forwarded through the call.stub into
// the stub signature, where `mogg.shape.create` consumes it.
// CHECK-LABEL: mo.graph @dyn
// CHECK:   inputs (%[[IP:[^ ,)]+]])
// CHECK:   outputs (%[[OP:[^ ,)]+]])
// CHECK:   context (%[[C:[^ ,)]+]])
// CHECK:   dims (%[[D:.*]]: !mogg.dim<batch>) attributes {device = #M.device_ref<"cpu", 0>, fusionInfo = {{.*}}} {
// CHECK:     mogg.call @[[STUB2:__[A-Za-z0-9_]+]] %[[IP]], %[[OP]], %[[D]], %[[C]] : !mogg.pointer, !mogg.pointer, !mogg.dim<batch>, !mogg.context
// CHECK: mogg.func @[[STUB2]] private (%{{.*}}: !mogg.pointer, %{{.*}}: !mogg.pointer, %[[SD:.*]]: !mogg.dim<batch>, %{{.*}}: !mogg.context) attributes {abi = #mogg.abi<MOGG>,
// CHECK:   mogg.shape.create(%[[SD]]) {{.*}} : !mogg.dim<batch>
mo.graph @dyn(%a: !mo.tensor<[batch, 4], f32>) -> !mo.tensor<[batch, 4], f32> {
  %c = mo.custom [symbol = "map_lower_one_plain_output", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[batch, 4], f32>) -> !mo.tensor<[batch, 4], f32>
  mo.output %c : !mo.tensor<[batch, 4], f32>
}
