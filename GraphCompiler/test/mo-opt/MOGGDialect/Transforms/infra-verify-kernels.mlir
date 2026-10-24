// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_pkg" --mogg-verify-kernels | FileCheck %s

// Verify that mogg-verify-kernels passes without error when processing
// mogg.kernel.decl ops produced by mogg-import-packages (KernelDeclAdaptor
// path).

// CHECK-DAG: mogg.kernel.decl @testing_kernels.testing_kernels.Foo registration = "imposter_add"
// CHECK-DAG: mogg.kernel.decl @testing_kernels.testing_kernels.AddFuseLHS registration = "imposter_add_lhs"
// CHECK-DAG: mogg.kernel.decl @testing_kernels.testing_kernels.AddElementwise registration = "imposter_add_elementwise"

// A `build_graph` may take the slot-backed form of an `execute` argument
// (`StableInputVariadicTensors` for `InputVariadicTensors`, `StableInputTensor`
// for `InputTensor`); the two signatures still align.
// CHECK-DAG: mogg.kernel.decl @testing_kernels.testing_kernels.StableBuildGraph registration = "imposter_stable_build_graph"
// CHECK-DAG: mogg.kernel.decl @testing_kernels.testing_kernels.StableSingleBuildGraph registration = "imposter_stable_single_build_graph"

module {}
