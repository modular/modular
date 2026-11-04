// RUN: rm -rf %t && mkdir -p %t
// RUN: env MODULAR_DEBUG=ir-output-dir=%t mo-opt %s \
// RUN:   --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-to-map --map-to-mogg --mogg-conform-call-operands \
// RUN:   --mogg-outline-routines --mogg-emit-mojo -verify-each=false
// RUN: FileCheck %s < %t/graph.mojo

// End-to-end Mojo emission for the MOGG-ABI `mogg.func`
// (FuncOp::emitMojoMOGGABI plus the shape.create / tensor.create /
// call.opaque / scalar.create / tuple.create emitters). The MOGG stub takes raw
// `!mogg.pointer` / `!mogg.dim` / `!mogg.context` handles and reconstructs the
// codegen tensors internally before invoking the opaque kernel's execute
// function.
//
// The pipeline is stopped right after emission: the surrounding `mogg.routine`
// container is only consumed later (MOGGToMGP + region outlining), so its
// `mogg.call` dangles once emit erases the stub -- `-verify-each=false`
// tolerates that. The emitted Mojo is read from the `ir-output-dir` dump rather
// than `--mogg-dump-mojo`, which needs a fully lowered `mgp.model`.

// Plain output: static shape -> compile-time `Idx[N]` Coord, a raw-pointer
// `mogg_tensor_init` per tensor, and a DPS execute call (output, input, ctx).
// CHECK: def {{__[A-Za-z0-9_]+}}({{.*}}: OpaquePointer[MutAnyOrigin], {{.*}}: OpaquePointer[MutAnyOrigin], [[PC:.*]]: {{.*}}DeviceContext) raises:
// CHECK:   var [[PSHAPE:.*]] = layout.Coord(layout.Idx[2], layout.Idx[2])
// CHECK:   var [[PLAYOUT:.*]] = layout.row_major([[PSHAPE]])
// CHECK:   var [[PIN:.*]] = {{.*}}mogg_tensor_init[DType.float32, 2, False, extensibility.IO.Input, {{[0-9]+}}]({{.*}}.unsafe_bitcast[NoneType](), [[PLAYOUT]])
// CHECK:   var [[POUT:.*]] = {{.*}}mogg_tensor_init[DType.float32, 2, True, extensibility.IO.Output, {{[0-9]+}}]({{.*}}.unsafe_bitcast[NoneType](), {{.*}})
// CHECK:   {{.*}}.MapLowerOnePlainOutput.execute[target = "cpu"]([[POUT]], [[PIN]], [[PC]])

mo.graph @plain(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom [symbol = "map_lower_one_plain_output", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// Dynamic dim: the `!mogg.dim` handle arrives as a trailing `Int` argument and
// feeds a runtime `Scalar[DType.int]` Coord component.
// CHECK: def {{__[A-Za-z0-9_]+}}({{.*}}: OpaquePointer[MutAnyOrigin], {{.*}}: OpaquePointer[MutAnyOrigin], [[DIM:.*]]: Int, {{.*}}: {{.*}}DeviceContext) raises:
// CHECK:   var {{.*}} = layout.Coord(Scalar[DType.int]([[DIM]]), layout.Idx[4])

mo.graph @dyn(%a: !mo.tensor<[batch, 4], f32>) -> !mo.tensor<[batch, 4], f32> {
  %c = mo.custom [symbol = "map_lower_one_plain_output", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[batch, 4], f32>) -> !mo.tensor<[batch, 4], f32>
  mo.output %c : !mo.tensor<[batch, 4], f32>
}

// Scalar operand: `scalar.create` packs the single-element tensor into a scalar
// via `mogg_as_scalar`, passed after the input in the DPS execute call.
// CHECK: def {{__[A-Za-z0-9_]+}}(
// CHECK:   var [[SC:.*]] = {{.*}}mogg_as_scalar({{.*}})
// CHECK:   {{.*}}.MapLowerScalar.execute[target = "cpu"]({{.*}}, {{.*}}, [[SC]], {{.*}})

mo.graph @scalar(%a: !mo.tensor<[4], f32>, %s: !mo.tensor<[], f32>) -> !mo.tensor<[4], f32> {
  %c = mo.custom [symbol = "map_lower_scalar", device = #M.device_ref<"cpu", 0>](%a, %s) : (!mo.tensor<[4], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[4], f32>
  mo.output %c : !mo.tensor<[4], f32>
}

// Variadic operands: `tuple.create` packs the run of tensors into a
// `VariadicTensors` (pointer + shape tuples plus a comptime static-spec list).
// CHECK: def {{__[A-Za-z0-9_]+}}(
// CHECK:   var {{.*}} = std.utils.StaticTuple[size = 2]({{.*}}._ptr, {{.*}}._ptr)
// CHECK:   var {{.*}} = std.utils.StaticTuple[size = 2]({{.*}}.shape(), {{.*}}.shape())
// CHECK:   var [[VT:.*]] = extensibility.VariadicTensors[{{.*}}static_specs = {{.*}}]({{.*}}, {{.*}})
// CHECK:   {{.*}}.MapLowerVariadic.execute[target = "cpu"]({{.*}}, [[VT]], {{.*}})

mo.graph @variadic(%a: !mo.tensor<[4], f32>, %b: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %c = mo.custom [symbol = "map_lower_variadic", device = #M.device_ref<"cpu", 0>](%a, %b) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  mo.output %c : !mo.tensor<[4], f32>
}
