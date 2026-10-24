// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_pkg" | FileCheck %s

// Verify that mogg-import-packages populates the module with mogg.kernel.decl
// ops for each @register-decorated struct in the loaded packages, and
// mogg.primitive.decl ops for singleton primitive functions.
// For kernels, sym_name is the fully-qualified Mojo source name (struct name).
// For primitives, sym_name equals the registration string (for MLIR symbol
// table uniqueness), and the Mojo source name is stored in mojo_name.

// Kernel with execute and shape (struct Foo).
// CHECK-DAG: mogg.kernel.decl @testing_kernels.testing_kernels.Foo registration = "imposter_add"

// Kernel with execute only (struct AddFuseLHS).
// CHECK-DAG: mogg.kernel.decl @testing_kernels.testing_kernels.AddFuseLHS registration = "imposter_add_lhs"

// Kernel with elementwise only (struct AddElementwise).
// CHECK-DAG: mogg.kernel.decl @testing_kernels.testing_kernels.AddElementwise registration = "imposter_add_elementwise"

// Primitive singleton function; sym_name is the fully-qualified Mojo source
// function name, registration is the @register_internal string.
// CHECK-DAG: mogg.primitive.decl @builtin_primitives.primitives._foreach registration = "mogg.call.foreach"

module {}
