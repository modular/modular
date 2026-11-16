// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-to-map --map-to-mogg --mogg-conform-call-operands | FileCheck %s

// ConformCallOperands re-runs operand inference against the callee
// mogg.kernel.decl and materializes the non-tensor operands the kernel expects:
// a scalar operand is wrapped in mogg.scalar.create, and a run of variadic
// tensor operands is packed into one mogg.tuple.create. Trait arguments take
// their device from the enclosing routine, so the calls start from `mo.custom`
// rather than bare codegen ops. conform-trait-tensor-tuple-fused-store.mlir
// checks that repacking keeps the call's epilogue flag.

// The second input is a scalar kernel argument.
// CHECK-LABEL: mo.graph @scalar
// CHECK: %[[SC:.*]] = mogg.scalar.create(%{{.*}})
// CHECK: mogg.call.opaque @{{.*}}MapLowerScalar inputs(%{{.*}}, %[[SC]] : !mogg.tensor, !mogg.scalar) outputs(%{{.*}} : !mogg.tensor) context(%{{.*}} : !mogg.context)
mo.graph @scalar(%a: !mo.tensor<[4], f32>, %s: !mo.tensor<[], f32>) -> !mo.tensor<[4], f32> {
  %c = mo.custom [symbol = "map_lower_scalar", device = #M.device_ref<"cpu", 0>](%a, %s) : (!mo.tensor<[4], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[4], f32>
  mo.output %c : !mo.tensor<[4], f32>
}

// The two inputs satisfy one variadic kernel argument.
// CHECK-LABEL: mo.graph @variadic
// CHECK: %[[TUP:.*]] = mogg.tuple.create(%{{.*}}, %{{.*}}) variadic_abi = <tensor_tuple> : !mogg.tensor, !mogg.tensor
// CHECK: mogg.call.opaque @{{.*}}MapLowerVariadic inputs(%[[TUP]] : !mogg.tuple) outputs(%{{.*}} : !mogg.tensor) context(%{{.*}} : !mogg.context)
mo.graph @variadic(%a: !mo.tensor<[4], f32>, %b: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %c = mo.custom [symbol = "map_lower_variadic", device = #M.device_ref<"cpu", 0>](%a, %b) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  mo.output %c : !mo.tensor<[4], f32>
}
