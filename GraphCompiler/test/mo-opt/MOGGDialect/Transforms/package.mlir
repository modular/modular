// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mogg-conform-call-operands | FileCheck %s

// ConformCallOperands runs directly on the mogg codegen ops (the
// post-MAPToMOGG form, with plain !mogg.tensor operands). It re-runs operand
// inference against the callee mogg.kernel.decl and materializes the non-tensor
// operands the kernel expects: a scalar operand is wrapped in
// mogg.scalar.create, and a run of variadic tensor operands is packed into one
// mogg.tuple.create.

// The second input is a scalar kernel argument.
// CHECK-LABEL: func.func @scalar
// CHECK: %[[SC:.*]] = mogg.scalar.create(%{{.*}})
// CHECK: mogg.call.opaque @{{.*}}MapLowerScalar inputs(%{{.*}}, %[[SC]] : !mogg.tensor, !mogg.scalar) outputs(%{{.*}} : !mogg.tensor) context(%{{.*}} : !mogg.context)
func.func @scalar(%in: !mogg.tensor, %scalar: !mogg.tensor, %out: !mogg.tensor, %ctx: !mogg.context) {
  mogg.call.opaque @testing_kernels.testing_kernels.MapLowerScalar inputs(%in, %scalar : !mogg.tensor, !mogg.tensor) outputs(%out : !mogg.tensor) context(%ctx : !mogg.context)
  return
}

// The two inputs satisfy one variadic kernel argument.
// CHECK-LABEL: func.func @variadic
// CHECK: %[[TUP:.*]] = mogg.tuple.create(%{{.*}}, %{{.*}}) : !mogg.tensor, !mogg.tensor
// CHECK: mogg.call.opaque @{{.*}}MapLowerVariadic inputs(%[[TUP]] : !mogg.tuple) outputs(%{{.*}} : !mogg.tensor) context(%{{.*}} : !mogg.context)
func.func @variadic(%a: !mogg.tensor, %b: !mogg.tensor, %out: !mogg.tensor, %ctx: !mogg.context) {
  mogg.call.opaque @testing_kernels.testing_kernels.MapLowerVariadic inputs(%a, %b : !mogg.tensor, !mogg.tensor) outputs(%out : !mogg.tensor) context(%ctx : !mogg.context)
  return
}

// Repacking rebuilds the call, so it has to carry the epilogue flag over. Emit
// reads it instead of inspecting an output's producer, and packing is what
// replaces that producer -- dropping it silently disables the epilogue.
// CHECK-LABEL: func.func @variadicKeepsEpilogueFlag
// CHECK: mogg.call.opaque @{{.*}}MapLowerVariadic inputs(%{{.*}} : !mogg.tuple) outputs(%{{.*}} : !mogg.tensor) context(%{{.*}} : !mogg.context) hasEpilogueFusion = true
func.func @variadicKeepsEpilogueFlag(%a: !mogg.tensor, %b: !mogg.tensor, %out: !mogg.tensor, %ctx: !mogg.context) {
  mogg.call.opaque @testing_kernels.testing_kernels.MapLowerVariadic inputs(%a, %b : !mogg.tensor, !mogg.tensor) outputs(%out : !mogg.tensor) context(%ctx : !mogg.context) hasEpilogueFusion = true
  return
}
